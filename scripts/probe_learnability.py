"""Can any light model beat cosine at telling gold nodes from the rest?

The oracle sweep showed QAFD's edge-weight site responds: hand it the answer and
recall@10 moves +0.114. That establishes "if you knew, it would help". It says
nothing about whether you *can* know, and that is the half a trained scorer lives
or dies on.

Every scorer available at that site is a function of `(h_u, h_v, h_q)` over frozen
mpnet vectors. The graph carries no relation types — `kg_builder` adds edges with a
`weight` attribute and nothing else — so there is no relation vocabulary for a
learned matcher to exploit the way GNN-RAG's `omega(q, r)` does. Whether those
vectors carry signal cosine is leaving on the table is answerable offline, on CPU,
with no retrieval run and no LLM.

Task: rank entity nodes for a question; positives are entities extracted from that
question's gold passages. Models, all functions of the same arguments:

    cosine        h_e . h_q                      0 params, what the pipeline uses
    diagonal      sum_k w_k h_e[k] h_q[k]        d params, a learned per-dim metric
    lowrank       (A h_e) . (B h_q)              2dr params, a two-tower metric
    mlp           [h_e*h_q, |h_e-h_q|, cos, deg] -> hidden -> 1

`mlp` is the elementwise-product form GNN-RAG builds `omega` from, adapted because
there is no relation vector here to take the place of `r`.

Split is by QUESTION (default 700/300) and every number is reported on held-out
questions. Implemented in numpy so the probe has no dependency beyond what the
analysis env already has.

**Negatives are drawn from other questions' gold entities, matched on gold
frequency**, using one distribution for training, early stopping and evaluation.
Three earlier choices were wrong and the shuffle control caught every one:

    uniform random over all entities     control 0.60-0.63
    uniform over the gold support        control 0.5687
    frequency-weighted over the support  still leaks

A gold entity is a real, well-connected extraction; a random draw from 85k is
mostly OpenIE debris, so "looks gold" predicts the label with no query at all.
Restricting to the gold support removes that but leaves raw frequency. Weighting
by frequency still leaves the deepest one: a question's own positives can never
be its negatives, so if positives skew frequent, negatives drawn from what
remains are systematically less frequent. Exclusion itself creates signal.

Stratified matching removes what can be removed. What cannot be removed is
reported: `frequency` is a query-independent scorer included in every table, and
on permuted labels it is the floor a learned model reaches by reading only the
label prior. The control compares against THAT, not against 0.50 — an entity that
is gold for most questions is a positive more often than a negative no matter how
the negatives are drawn, and no sampler fixes that.

Always run the control first; it is a gate, not a formality:

    python scripts/probe_learnability.py musique --shuffle-control
    python scripts/probe_learnability.py musique

In the real run, read any learned gain over cosine against the `frequency` row:
a gain no larger than what frequency alone buys is not evidence of query-dependent
signal.

Note on `sim_mode`: "normalized", "relu" and "relu_sq" are all monotone in cosine,
so they produce *identical* rankings and identical AUC/recall. They matter inside
the pipeline only because the routing distribution normalises the weights, which
rescales differences the ranking cannot see. There is therefore one cosine baseline
here, not three.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if sys.version_info < (3, 10):
    sys.exit(
        f"\nFATAL: this script needs the mbuzai env (Python 3.10+), got "
        f"{sys.version.split()[0]}\n  interpreter: {sys.executable}"
    )

from mbuzai import dataio, metrics  # noqa: E402

OUT = ROOT / "out"
EMB_MODEL = "sentence-transformers/all-mpnet-base-v2"


# ---------------------------------------------------------------------------
# Models. Each exposes fit(X..., y) and score(rows) over the SAME arguments.
# ---------------------------------------------------------------------------

def _adam(params, grads, state, lr):
    """Minimal Adam. Keeps the models dependency-free and the code auditable."""
    for i, (p, g) in enumerate(zip(params, grads)):
        m, v, t = state.setdefault(i, (np.zeros_like(p), np.zeros_like(p), 0))
        t += 1
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * (g * g)
        state[i] = (m, v, t)
        p -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)


# Objective, set once per process before training. A module global rather than a
# parameter because the three models call the gradient helper directly and the
# workers inherit this by fork.
_OBJ = {"loss": "pointwise", "starts": None, "k": 10}


def _group_starts(qi):
    """Boundaries of contiguous per-question row blocks.

    Rows are built one question at a time and the validation split removes whole
    questions, so each question's rows stay contiguous and a single boundary
    array describes them.
    """
    if len(qi) == 0:
        return np.array([0], dtype=np.int64)
    cuts = np.flatnonzero(np.diff(qi) != 0) + 1
    return np.concatenate([[0], cuts, [len(qi)]]).astype(np.int64)


def _listwise_grad(s, y, starts):
    """Per-question softmax cross-entropy, and dL/ds.

    Pointwise BCE spends the same gradient on every row whatever its rank, which
    is why it lifts AUC -- a global-order statistic over ~2000 candidates -- while
    leaving the top of each question's list untouched. A per-question softmax puts
    the gradient on whichever negatives currently outscore the positives, i.e. on
    exactly the rows that decide recall@k, and it does so WITHOUT changing the
    negative distribution -- which the pool comment below explains has to stay
    identical in training and evaluation or the model inverts.
    """
    ds = np.zeros_like(s)
    for a, b in zip(starts[:-1], starts[1:]):
        yg = y[a:b]
        tot = yg.sum()
        if tot <= 0:
            continue
        e = np.exp(s[a:b] - s[a:b].max())
        ds[a:b] = e / e.sum() - yg / tot
    return 0.0, ds / max(len(starts) - 1, 1)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _pairwise_grad(s, y, starts, k=None):
    """RankNet over (positive, negative) pairs, optionally LambdaRank-weighted.

    The listwise softmax above is rank-aware but position-blind: a positive at
    rank 3 and one at rank 300 contribute the same term, so nothing in the
    objective knows where the recall@k cutoff is.

    With k set, each pair is weighted by |delta recall@k| -- the change swapping it
    would produce. For recall@k that is 1/|positives| when exactly one of the two
    sits inside the top k, and ZERO otherwise. So the gradient lands only on pairs
    that straddle the cutoff: negatives currently occupying a top-k slot, and the
    positives they are keeping out. That is the metric written into the loss
    rather than hoped for.

    The weights are computed from the current ranking and held constant for the
    step (the standard LambdaRank detachment -- the weights are piecewise constant
    in s, so they carry no gradient of their own). The finite-difference check in
    --selftest verifies the gradient of exactly that surrogate.
    """
    ds = np.zeros_like(s)
    groups = 0
    for a, b in zip(starts[:-1], starts[1:]):
        sg, yg = s[a:b], y[a:b]
        pi = np.flatnonzero(yg > 0)
        ni = np.flatnonzero(yg <= 0)
        if len(pi) == 0 or len(ni) == 0:
            continue
        d = sg[pi][:, None] - sg[ni][None, :]
        if k is None:
            w = np.full(d.shape, 1.0 / d.size)
        else:
            order = np.argsort(-sg)
            rank = np.empty(len(sg), dtype=np.int64)
            rank[order] = np.arange(len(sg))
            pos_out = (rank[pi] >= k)[:, None]      # positive missing the cutoff
            neg_in = (rank[ni] < k)[None, :]        # negative holding a slot
            w = (pos_out & neg_in).astype(np.float64) / len(pi)
        g = -_sigmoid(-d) * w                        # dL/d(s_pos - s_neg)
        view = ds[a:b]
        view[pi] += g.sum(1)
        view[ni] -= g.sum(0)
        groups += 1
    return 0.0, ds / max(groups, 1)


def _recall_at_k(scores, y, starts, k=10):
    """Mean per-question recall@k -- the statistic the pipeline actually spends."""
    tot, n = 0.0, 0
    for a, b in zip(starts[:-1], starts[1:]):
        yg = y[a:b]
        g = yg.sum()
        if g <= 0:
            continue
        top = np.argsort(-scores[a:b])[:k]
        tot += yg[top].sum() / g
        n += 1
    return tot / max(n, 1)


def _loss_grad(s, y):
    """Mean BCE-with-logits and dL/ds, or the listwise objective when selected."""
    if _OBJ["starts"] is not None:
        if _OBJ["loss"] == "listwise":
            return _listwise_grad(s, y, _OBJ["starts"])
        if _OBJ["loss"] == "pairwise":
            return _pairwise_grad(s, y, _OBJ["starts"], k=None)
        if _OBJ["loss"] == "lambdarank":
            return _pairwise_grad(s, y, _OBJ["starts"], k=_OBJ["k"])
    p = 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))
    loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
    return loss, (p - y) / len(y)


def fit_early_stop(model, he, hq, y, deg, val, epochs=200, lr=0.01, eval_every=5,
                   select="auc", qi=None, loss="pointwise", lambda_k=10):
    """Train, keeping the parameters that scored best on the SELECTION metric.

    select="auc" checkpoints on validation ROC-AUC, which ranks all ~2000
    candidates and is therefore dominated by the easy tail. select="recall"
    checkpoints on mean per-question recall@10 instead. The distinction is not
    academic: on NV-Embed features the models gained +0.057 AUC while losing
    recall@10, i.e. the run selected checkpoints for the metric nothing in the
    pipeline spends.

    This is what stops the probe from confusing "no signal" with "the optimiser
    wandered off". Every learned model here starts at or near cosine, so without
    early stopping a model can end BELOW the baseline simply because Adam takes
    ~lr-sized steps however small the gradient is — and a model that loses to
    cosine on cosine-shaped data would make a genuine null unreadable.
    """
    if not hasattr(model, "_step"):
        return model
    tr = ~val
    # Build each model's inputs ONCE. The MLP's feature block is
    # |rows| x (2d+2) floats — at d=768 that is hundreds of MB, and rebuilding it
    # per epoch made training cost more than everything else in this script put
    # together. Nothing in it depends on the parameters.
    ptr = model.prep(he[tr], hq[tr], deg[tr])
    pva = model.prep(he[val], hq[val], deg[val])
    ytr, yva = y[tr], y[val]
    tr_starts = _group_starts(qi[tr]) if qi is not None else None
    va_starts = _group_starts(qi[val]) if qi is not None else None
    if loss != "pointwise" and tr_starts is None:
        raise ValueError("listwise needs per-question row groups (qi)")
    _OBJ["loss"], _OBJ["starts"] = loss, tr_starts
    _OBJ["k"] = lambda_k
    st = {}
    best = (-np.inf, [p.copy() for p in model._params()])
    for ep in range(1, epochs + 1):
        model._step(ptr, ytr, st, lr)
        if ep % eval_every == 0 or ep == epochs:
            sv = model.score_prepped(pva)
            score = (roc_auc(sv, yva) if select == "auc"
                     else _recall_at_k(sv, yva, va_starts, k=10))
            if score > best[0]:
                best = (score, [p.copy() for p in model._params()])
    for p, b in zip(model._params(), best[1]):
        p[...] = b
    model.val_auc = best[0]      # the SELECTION score; see `select`
    model.val_metric = select
    return model


class Cosine:
    """The function the pipeline actually uses. No parameters, nothing to fit."""
    name = "cosine"

    def score(self, he, hq, deg):
        return he @ hq


class Frequency:
    """Scores by how many questions an entity is gold for. Ignores the query.

    A diagnostic, not a candidate. Some residual frequency signal is unavoidable:
    a question's own positives can never be its negatives, so an entity that is
    gold very often is a positive far more than a negative no matter how the
    negatives are drawn, and no sampler can balance an entity that is gold for
    most questions.

    Reporting it makes that floor visible instead of assumed. On permuted labels
    this is the level a learned model reaches by reading nothing but the label
    prior, so it — not 0.5 — is what the control must compare against.
    """
    name = "frequency"

    def __init__(self, freq_of):
        self.freq_of = freq_of      # entity id -> gold frequency

    def score(self, he, hq, deg, rows=None):
        return self.freq_of[rows] if rows is not None else np.zeros(len(he))


class Diagonal:
    """h_e^T diag(w) h_q — the smallest strict generalisation of cosine."""
    name = "diagonal"

    def __init__(self, d, seed=0):
        self.w = np.ones(d, dtype=np.float32)     # w = 1 is exactly cosine
        self.b = np.zeros(1, dtype=np.float32)

    def _params(self):
        return [self.w, self.b]

    def prep(self, he, hq, deg):
        return he * hq          # the only thing the gradient needs

    def _step(self, P, y, st, lr):
        _, ds = _loss_grad(P @ self.w + self.b, y)
        _adam(self._params(), [P.T @ ds, np.array([ds.sum()])], st, lr)

    def score_prepped(self, P):
        return P @ self.w + self.b

    def score(self, he, hq, deg):
        return (he * self.w) @ hq + self.b


class LowRank:
    """cos + (A h_e) . (B h_q) — a low-rank *correction* to cosine.

    Written as a residual rather than a free two-tower metric on purpose. A
    randomly initialised rank-r projection is a noisy approximation of cosine, so
    the model would start well below the baseline and any shortfall after training
    would be unreadable — is there no signal, or did it just fail to climb back?
    With A, B initialised tiny the model starts *at* cosine (A=B=0 is exactly the
    baseline) and every point it moves is signal cosine does not already have.
    """
    name = "lowrank"

    def __init__(self, d, r=64, seed=0):
        rng = np.random.default_rng(seed)
        s = 0.01 / np.sqrt(d)
        self.A = rng.normal(0, s, (d, r)).astype(np.float32)
        self.B = rng.normal(0, s, (d, r)).astype(np.float32)
        self.b = np.zeros(1, dtype=np.float32)

    def _params(self):
        return [self.A, self.B, self.b]

    def prep(self, he, hq, deg):
        return he, hq, np.sum(he * hq, axis=1)   # cos is fixed; A, B are not

    def _step(self, P, y, st, lr):
        he, hq, cos = P
        ea, qb = he @ self.A, hq @ self.B
        _, ds = _loss_grad(cos + np.sum(ea * qb, axis=1) + self.b, y)
        _adam(self._params(),
              [he.T @ (ds[:, None] * qb), hq.T @ (ds[:, None] * ea),
               np.array([ds.sum()])], st, lr)

    def score_prepped(self, P):
        he, hq, cos = P
        return cos + np.sum((he @ self.A) * (hq @ self.B), axis=1) + self.b

    def score(self, he, hq, deg):
        return he @ hq + (he @ self.A) @ (self.B.T @ hq) + self.b


class MLP:
    """One hidden layer over [h_e*h_q, |h_e-h_q|, cos, log deg].

    The elementwise product is the form GNN-RAG's omega(q, r) = phi(q (*) r) uses;
    the absolute difference and the degree are the cheap extras a heuristic cannot
    express at all.
    """
    name = "mlp"

    def __init__(self, d, hidden=128, seed=0):
        rng = np.random.default_rng(seed)
        f = 2 * d + 2
        self.W1 = rng.normal(0, np.sqrt(2.0 / f), (f, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, 1)).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)

    @staticmethod
    def _feat(he, hq, deg):
        cos = np.sum(he * hq, axis=1, keepdims=True)
        return np.hstack([he * hq, np.abs(he - hq), cos,
                          np.log1p(deg).reshape(-1, 1)])

    def _params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def prep(self, he, hq, deg):
        return self._feat(he, hq, deg)

    def _step(self, X, y, st, lr):
        h = np.maximum(0, X @ self.W1 + self.b1)
        _, ds = _loss_grad((h @ self.W2 + self.b2).ravel(), y)
        dh = (ds[:, None] @ self.W2.T) * (h > 0)
        _adam(self._params(),
              [X.T @ dh, dh.sum(0), h.T @ ds[:, None], np.array([ds.sum()])], st, lr)

    def score_prepped(self, X):
        h = np.maximum(0, X @ self.W1 + self.b1)
        return (h @ self.W2 + self.b2).ravel()

    def score(self, he, hq, deg):
        return self.score_prepped(self._feat(he, np.broadcast_to(hq, he.shape), deg))


# ---------------------------------------------------------------------------

# Training inputs, published to forked workers. Populated before the pool is
# created so children inherit them copy-on-write: at d=768 the row blocks are
# hundreds of MB and pickling them to each worker would cost more than the
# training it parallelises.
_SHARED = {}


def _train_one(model):
    """Fit one model in a worker. Only the parameters travel back."""
    return fit_early_stop(model, _SHARED["He"], _SHARED["Hq"], _SHARED["Y"],
                          _SHARED["Dg"], _SHARED["val"], lr=_SHARED["lr"],
                          select=_SHARED.get("select", "auc"),
                          qi=_SHARED.get("Qi"), loss=_SHARED.get("loss", "pointwise"),
                          lambda_k=_SHARED.get("lambda_k", 10))


def build_pool(positives):
    """Frequency-stratified negative pool.

    Returns (support, bins, by_bin) where `bins[j]` is a log2 frequency stratum
    for support[j] and `by_bin` maps stratum -> support positions in it.
    """
    sup, cnt = np.unique(np.concatenate(positives).astype(np.int64),
                         return_counts=True)
    bins = np.floor(np.log2(cnt)).astype(np.int64)
    by_bin = {int(b): np.flatnonzero(bins == b) for b in np.unique(bins)}
    return sup, cnt, bins, by_bin


def sample_from_pool(pool, pos, k, rng):
    """k negatives whose gold-frequency distribution MATCHES the positives'.

    Getting this right took three attempts, each caught by the shuffle control,
    and the reason is worth stating because it is not obvious:

      uniform random over all entities (control 0.60-0.63) — a gold entity is a
        real, well-connected extraction; a random draw from 85k is mostly OpenIE
        debris, so "looks gold" predicts the label with no query at all.
      uniform over the gold support (control 0.5687) — entities are gold for
        wildly different numbers of questions, and frequency alone still predicts.
      frequency-weighted over the support — still leaks, and this is the
        fundamental one: a question's own positives can never be its negatives,
        so if positives skew frequent, the negatives drawn from what remains are
        systematically LESS frequent. Exclusion itself creates the signal.

    The last cannot be fixed by reweighting, only by matching: draw each negative
    from the same log2-frequency stratum as a positive. Then the two classes have
    the same frequency profile by construction, exclusion or not, and nothing
    question-independent is left for a model to read.

    Module level so the regression test can drive it directly — the negative
    distribution is where this probe got its first three answers wrong.
    """
    sup, cnt, bins, by_bin = pool
    posmask = np.isin(sup, pos)
    pos_bins = bins[posmask]
    if len(pos_bins) == 0:
        return np.empty(0, dtype=np.int64)

    strata, counts = np.unique(pos_bins, return_counts=True)
    out = []
    for b, c in zip(strata, counts):
        want = max(1, int(round(k * c / len(pos_bins))))
        cand = by_bin[int(b)]
        cand = cand[~posmask[cand]]
        if len(cand) == 0:
            continue
        take = min(want, len(cand))
        out.append(sup[rng.choice(cand, size=take, replace=False)])
    if not out:
        return np.empty(0, dtype=np.int64)
    neg = np.concatenate(out)
    return neg[:k].astype(np.int64)


def roc_auc(scores, labels):
    """Rank-based AUC. Ties get average rank, so a constant scorer gives 0.5."""
    pos, neg = labels.sum(), len(labels) - labels.sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    # Average rank within tied groups, vectorised: early stopping calls this once
    # per eval per model, so a Python loop over every candidate dominates runtime.
    _, inv, counts = np.unique(s_sorted, return_inverse=True, return_counts=True)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = (starts + (counts - 1) / 2.0 + 1)[inv]
    return (ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)


def recall_at(scores, labels, k):
    top = np.argsort(-scores, kind="mergesort")[:k]
    return labels[top].sum() / labels.sum() if labels.sum() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--nodes", type=Path, default=None,
                    help="out/qafd_nodes_<dataset>.npz from export_qafd_nodes.sh")
    ap.add_argument("--train-questions", type=int, default=700)
    ap.add_argument("--pool", type=int, default=2000,
                    help="negatives per held-out question at eval")
    ap.add_argument("--neg-per-q", type=int, default=40, help="training negatives")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--jobs", type=int, default=1,
                    help="train the 3 models concurrently. Set BLAS threads per\nworker with OMP_NUM_THREADS before launching; scripts/run_probe.sh does both")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle-control", action="store_true",
                    help="pair each question with another question's gold. Every "
                         "model must collapse to AUC~0.5; if it does not, the "
                         "labels leak and no other number here means anything.")
    ap.add_argument("--device", default="cpu", help="passed to sentence-transformers")
    # An export from a 7B-encoder index carries 4096-d entity vectors, and encoding
    # the questions with mpnet would score them in a different space -- silently,
    # since a dot product between mismatched spaces is still a number. Read the
    # cached query vectors instead, the same ones the retrieval runs used.
    ap.add_argument("--query_emb", type=Path, default=None,
                    help="npz from --cache_query_emb; required when --nodes came "
                         "from an index built with something other than mpnet")
    ap.add_argument("--query_emb_kind", default="passage", choices=["passage", "fact"])
    # The three knobs that decide WHICH ranking the training targets. Defaults
    # reproduce every run so far; none of them changes the evaluation, so results
    # stay comparable across settings.
    ap.add_argument("--loss", default="pointwise",
                    choices=["pointwise", "listwise", "pairwise", "lambdarank"],
                    help="pointwise BCE weights every row equally and optimises a "
                         "global order; listwise is a per-question softmax and puts "
                         "the gradient on negatives that currently outrank positives; "
                         "pairwise is RankNet over pos/neg pairs; lambdarank weights "
                         "each pair by the recall@k change swapping it would cause, "
                         "which puts all gradient on pairs straddling the cutoff")
    ap.add_argument("--lambda-k", type=int, default=10,
                    help="the k in lambdarank's recall@k weighting")
    ap.add_argument("--select", default="auc", choices=["auc", "recall"],
                    help="early-stopping metric: validation ROC-AUC, or mean "
                         "per-question recall@10")
    # entity_degree is a popularity feature, and the shuffle control shows
    # popularity is precisely the signal that survives permuting the labels. Zeroing
    # it asks whether the learned AUC gain was that nuisance all along.
    ap.add_argument("--no-degree", action="store_true",
                    help="zero the degree feature (MLP is the only model using it)")
    args = ap.parse_args()

    npz_path = args.nodes or OUT / f"qafd_nodes_{args.dataset}.npz"
    if not npz_path.exists():
        sys.exit(f"missing {npz_path}\n  build it first (QAFD env):\n"
                 "    bash scripts/export_qafd_nodes.sh")
    z = np.load(npz_path, allow_pickle=False)

    # float32 throughout: the feature blocks are hundreds of MB at d=768 and
    # nothing here needs float64 precision.
    ent_emb = np.asarray(z["entity_emb"], dtype=np.float32)
    ent_emb /= np.linalg.norm(ent_emb, axis=1, keepdims=True) + 1e-12
    ent_deg = z["entity_degree"].astype(np.float32)
    if args.no_degree:
        ent_deg = np.zeros_like(ent_deg)
        print("degree feature zeroed (--no-degree)")
    n_ent, d = ent_emb.shape

    # entity -> gold pids, as sets keyed by pid for a cheap per-question lookup
    pid_to_entities = {}
    for e, p in zip(z["ent_gold_entity"], z["ent_gold_pid"]):
        pid_to_entities.setdefault(int(p), []).append(int(e))

    ds = dataio.load(args.dataset)
    rng = np.random.default_rng(args.seed)

    print(f"index: {n_ent} entities, dim {d}, "
          f"{len(z['edges'])} edges, {len(z['ent_gold_entity'])} entity-gold pairs")

    # positives per question
    queries, positives = [], []
    for q in ds.queries:
        pos = sorted({e for p in q.gold_pids for e in pid_to_entities.get(int(p), ())})
        if pos:
            queries.append(q)
            positives.append(np.array(pos, dtype=np.int64))
    print(f"questions with >=1 gold entity: {len(queries)}/{len(ds.queries)} "
          f"(mean {np.mean([len(p) for p in positives]):.1f} positives)")
    if not queries:
        sys.exit("no question has a gold entity — the entity/gold mapping is broken, "
                 "not the task. Check export_qafd_nodes.sh ran against this corpus.")

    if args.shuffle_control:
        perm = rng.permutation(len(positives))
        positives = [positives[i] for i in perm]
        print("SHUFFLE CONTROL: gold sets permuted across questions")

    # Encode questions with the same checkpoint the index used. STEncoder ignores
    # instruction kwargs for mpnet, so QAFD's query_to_fact instruction never
    # reached the encoder either — plain encoding is what the pipeline compares.
    if args.query_emb:
        _qz = np.load(args.query_emb, allow_pickle=False)
        _qmap = {q: i for i, q in enumerate(_qz["questions"])}
        _missing = [q for q in queries if q.question not in _qmap]
        if _missing:
            sys.exit(f"\nFATAL: {len(_missing)} questions absent from {args.query_emb}, "
                     f"e.g. {_missing[0].question[:60]!r}")
        _Q = np.asarray(_qz[args.query_emb_kind], dtype=np.float32)
        _Q /= np.maximum(np.linalg.norm(_Q, axis=1, keepdims=True), 1e-12)
        q_emb = np.stack([_Q[_qmap[q.question]] for q in queries])
        print(f"queries: {len(q_emb)} cached vectors from {args.query_emb.name}, "
              f"dim {q_emb.shape[1]}")
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMB_MODEL, device=args.device)
        q_emb = np.asarray(model.encode([q.question for q in queries],
                                        normalize_embeddings=True, convert_to_numpy=True,
                                        show_progress_bar=False), dtype=np.float32)
    if q_emb.shape[1] != ent_emb.shape[1]:
        sys.exit(f"\nFATAL: query vectors are {q_emb.shape[1]}-d but the node export is "
                 f"{ent_emb.shape[1]}-d.\n  Every model here scores h_e against h_q, so "
                 "these must come from ONE encoder.\n  Pass --query_emb for a non-mpnet "
                 "index.")

    order = rng.permutation(len(queries))
    tr, te = order[:args.train_questions], order[args.train_questions:]
    print(f"split: {len(tr)} train questions, {len(te)} held out\n")

    # ---- negative pool ------------------------------------------------------
    # Negatives are drawn from OTHER questions' gold entities, and the same pool
    # is used for training, early stopping and evaluation. Both earlier choices
    # were wrong, and the shuffle control caught both:
    #
    #   Uniform-random negatives are separable without the query at all. A gold
    #   entity is a real, well-formed, well-connected extraction; a uniform draw
    #   from 85k entities is mostly OpenIE debris. A model can score by learning
    #   "this looks like a gold entity", which survives permuting the labels --
    #   the control reported AUC 0.60-0.63 where it must report 0.50. Drawing
    #   negatives from the gold pool matches that nuisance distribution exactly:
    #   a negative here IS a positive, for a different question, so the only
    #   thing separating them is the query.
    #
    #   Mining hard negatives for training but evaluating against random ones
    #   also gave train and test different negative distributions. A model
    #   trained where high cosine usually means a mined negative learns to
    #   distrust high cosine, and then inverts on a random pool -- which is how
    #   `diagonal` reached 0.4258, below chance. One distribution everywhere.
    pool = build_pool(positives)
    _sup, _cnt, _bins, _by = pool
    uniq = len(_sup)
    print(f"negative pool: {int(_cnt.sum())} positive occurrences over {uniq} "
          f"distinct entities (max {int(_cnt.max())} questions for one entity), "
          f"{len(_by)} frequency strata")
    if uniq < 10 * args.neg_per_q:
        sys.exit("negative pool too small to sample from without heavy repetition")

    def sample_negatives(pos, k):
        return sample_from_pool(pool, pos, k, rng)

    He, Hq, Y, Dg, Qi = [], [], [], [], []
    for i in tr:
        pos = positives[i]
        neg = sample_negatives(pos, 2 * args.neg_per_q)
        rows = np.concatenate([pos, neg])
        lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        He.append(ent_emb[rows]); Hq.append(np.repeat(q_emb[i][None], len(rows), 0))
        Y.append(lab); Dg.append(ent_deg[rows])
        Qi.append(np.full(len(rows), i))
    He, Hq, Y, Dg, Qi = (np.concatenate(x) for x in (He, Hq, Y, Dg, Qi))

    # Validation split is by QUESTION, not by row: rows from one question share a
    # query vector, so splitting rows would leak the query across the boundary and
    # early stopping would select on data it had effectively trained on.
    val_q = set(rng.choice(tr, size=max(1, len(tr) // 5), replace=False).tolist())
    val = np.isin(Qi, list(val_q))
    print(f"training rows: {len(Y)} ({int(Y.sum())} positive), "
          f"{val.sum()} held out for early stopping ({len(val_q)} questions)")

    _SHARED.update(He=He, Hq=Hq, Y=Y, Dg=Dg, val=val, lr=args.lr, Qi=Qi,
                   select=args.select, loss=args.loss, lambda_k=args.lambda_k)
    todo = [Diagonal(d, seed=args.seed),
            LowRank(d, r=args.rank, seed=args.seed),
            MLP(d, seed=args.seed)]

    t0 = time.time()
    if args.jobs > 1:
        # fork, not spawn: the workers must inherit _SHARED rather than receive
        # it. BLAS threading is inherited too, so the driver sets
        # OMP_NUM_THREADS before python starts -- changing it here would be too
        # late, the thread pool is sized at import.
        import concurrent.futures as cf
        import multiprocessing as mp
        with cf.ProcessPoolExecutor(max_workers=min(args.jobs, len(todo)),
                                    mp_context=mp.get_context("fork")) as ex:
            trained = list(ex.map(_train_one, todo))
    else:
        trained = [_train_one(m) for m in todo]

    # Frequency is a diagnostic floor, not a candidate: it reads only the label
    # prior. Learned models are judged against IT on permuted labels.
    freq_of = np.zeros(n_ent, dtype=np.float64)
    freq_of[pool[0]] = pool[1]
    models = [Cosine(), Frequency(freq_of)] + trained
    for m in trained:
        print(f"  {m.name:<10} best val "
              f"{'AUC' if m.val_metric == 'auc' else 'recall@10'} {m.val_auc:.4f}")
    print(f"  trained in {time.time() - t0:.0f}s "
          f"({'parallel, %d workers' % min(args.jobs, len(todo)) if args.jobs > 1 else 'sequential'}, "
          f"{os.environ.get('OMP_NUM_THREADS', 'default')} BLAS threads each)")

    # ---- eval on held-out questions, identical candidate pool per model ----
    per_q = {m.name: {"auc": [], "r10": [], "r50": []} for m in models}
    for i in te:
        pos = positives[i]
        neg = sample_negatives(pos, args.pool)   # same distribution as training
        rows = np.concatenate([pos, neg])
        lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        he, dg = ent_emb[rows], ent_deg[rows]
        for m in models:
            s = (m.score(he, q_emb[i], dg, rows=rows)
                 if isinstance(m, Frequency) else m.score(he, q_emb[i], dg))
            per_q[m.name]["auc"].append(roc_auc(s, lab))
            per_q[m.name]["r10"].append(recall_at(s, lab, 10))
            per_q[m.name]["r50"].append(recall_at(s, lab, 50))

    print(f"\nheld-out questions: {len(te)}, pool {args.pool} negatives + positives")
    print(f"{'model':<12}{'ROC-AUC':>22}{'recall@10':>22}{'recall@50':>22}")
    base = per_q["cosine"]
    for m in models:
        cells = []
        for key in ("auc", "r10", "r50"):
            v = np.array(per_q[m.name][key], dtype=np.float64)
            v = v[~np.isnan(v)]
            lo, hi = metrics.bootstrap_ci(list(v))
            cells.append(f"{v.mean():.4f} [{lo:.3f},{hi:.3f}]")
        print(f"{m.name:<12}" + "".join(f"{c:>22}" for c in cells))

    print(f"\ndelta vs cosine (paired bootstrap 95% CI)")
    for m in models[1:]:
        cells = []
        for key in ("auc", "r10", "r50"):
            dvals = [a - b for a, b in zip(per_q[m.name][key], base[key])
                     if a == a and b == b]
            lo, hi = metrics.bootstrap_ci(dvals)
            star = "*" if lo > 0 or hi < 0 else " "
            cells.append(f"{np.mean(dvals):+.4f} [{lo:+.3f},{hi:+.3f}]{star}")
        print(f"{m.name:<12}" + "".join(f"{c:>22}" for c in cells))
    print("  * = CI excludes zero")

    # The control is a gate, not a suggestion. With the gold sets permuted there is
    # no question-entity relationship left, so anything scoring above chance is
    # reading a nuisance correlate of "is a gold entity" and every number in the
    # real run is contaminated by the same thing. Say so loudly.
    if args.shuffle_control:
        floor = float(np.nanmean(per_q["frequency"]["auc"]))
        learned = {m.name: float(np.nanmean(per_q[m.name]["auc"])) for m in trained}
        worst, who = max((v, k) for k, v in learned.items())
        # The bar is the frequency floor, not 0.50. Some frequency signal cannot be
        # sampled away -- an entity gold for most questions is a positive far more
        # often than a negative whatever the sampler does. What must NOT happen is a
        # model reading more than that floor while the query carries no information.
        bar = max(0.55, floor + 0.02)
        print(f"\ncontrol: frequency floor {floor:.4f} (query-independent, "
              f"unavoidable), worst learned {worst:.4f} ({who}), bar {bar:.4f}")
        if worst > bar:
            print(f"*** CONTROL FAILED: {who} reads {worst - floor:+.4f} beyond the "
                  "frequency floor on permuted labels.\n"
                  "    Do NOT read the main run: something query-independent is "
                  "still separable.")
        else:
            print("control passed: no learned model exceeds the frequency floor. "
                  "The main run is readable,\n"
                  "    bearing in mind any learned gain there must beat cosine by "
                  "more than this floor to mean anything.")

    dest = OUT / f"probe_learnability_{args.dataset}"
    dest = dest.with_name(dest.name + ("_shuffled" if args.shuffle_control else "") + ".json")
    dest.write_text(json.dumps(
        {m.name: {k: float(np.nanmean(v)) for k, v in per_q[m.name].items()}
         for m in models}, indent=1))
    print(f"\nwrote {dest}")

    print("""
how to read this:
  every learned model ~= cosine   -> the arguments carry no signal cosine misses.
                                     The oracle's +0.114 is not reachable from
                                     (h_u, h_v, h_q), and the in-pipeline scorer
                                     should not be built. This is the strong form
                                     of the README's claim.
  learned >> cosine               -> there is headroom in the same arguments, and
                                     the trained edge scorer has a target.
  run --shuffle-control first: it must collapse every model to AUC ~0.5.""")


if __name__ == "__main__":
    main()
