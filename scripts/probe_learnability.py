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

**Negatives are drawn from other questions' gold entities**, using one
distribution for training, early stopping and evaluation. Uniform-random
negatives do not work: a gold entity is a real, well-connected extraction while a
random draw from 85k is mostly OpenIE debris, so a model can score by learning
"this looks like a gold entity" without consulting the query at all. That is not
a hypothetical — it is what the first version of this script did, and the shuffle
control caught it at AUC 0.60-0.63 where chance is 0.50. Sampling negatives from
the gold pool matches the nuisance distribution exactly, because a negative here
IS a positive for some other question.

Always run the control first; it is a gate, not a formality:

    python scripts/probe_learnability.py musique --shuffle-control   # must be ~0.50
    python scripts/probe_learnability.py musique

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


def _logistic_loss_grad(s, y):
    """Mean BCE-with-logits and dL/ds."""
    p = 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))
    loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
    return loss, (p - y) / len(y)


def fit_early_stop(model, he, hq, y, deg, val, epochs=200, lr=0.01, eval_every=5):
    """Train, keeping the parameters with the best validation AUC.

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
    st = {}
    best = (-np.inf, [p.copy() for p in model._params()])
    for ep in range(1, epochs + 1):
        model._step(ptr, ytr, st, lr)
        if ep % eval_every == 0 or ep == epochs:
            auc = roc_auc(model.score_prepped(pva), yva)
            if auc > best[0]:
                best = (auc, [p.copy() for p in model._params()])
    for p, b in zip(model._params(), best[1]):
        p[...] = b
    model.val_auc = best[0]
    return model


class Cosine:
    """The function the pipeline actually uses. No parameters, nothing to fit."""
    name = "cosine"

    def score(self, he, hq, deg):
        return he @ hq


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
        _, ds = _logistic_loss_grad(P @ self.w + self.b, y)
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
        _, ds = _logistic_loss_grad(cos + np.sum(ea * qb, axis=1) + self.b, y)
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
        _, ds = _logistic_loss_grad((h @ self.W2 + self.b2).ravel(), y)
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
                          _SHARED["Dg"], _SHARED["val"], lr=_SHARED["lr"])


def sample_from_pool(pool, pos, k, rng):
    """k entities from `pool`, excluding this question's positives.

    Module level so the regression test can drive it directly: the negative
    distribution is where this probe got its first answer wrong, and it is the
    part most worth pinning down.
    """
    take = min(k + len(pos) + 16, len(pool))
    block = rng.choice(pool, size=take, replace=False)
    return block[~np.isin(block, pos)][:k]


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
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMB_MODEL, device=args.device)
    q_emb = np.asarray(model.encode([q.question for q in queries],
                                    normalize_embeddings=True, convert_to_numpy=True,
                                    show_progress_bar=False), dtype=np.float32)

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
    pool = np.array(sorted({int(e) for p in positives for e in p}), dtype=np.int64)
    print(f"negative pool: {len(pool)} entities that are gold for some question")
    if len(pool) < 10 * args.neg_per_q:
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

    _SHARED.update(He=He, Hq=Hq, Y=Y, Dg=Dg, val=val, lr=args.lr)
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

    models = [Cosine()] + trained
    for m in trained:
        print(f"  {m.name:<10} best val AUC {m.val_auc:.4f}")
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
            s = m.score(he, q_emb[i], dg)
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
        worst = max(float(np.nanmean(per_q[m.name]["auc"])) for m in models)
        if worst > 0.55:
            print(f"\n*** CONTROL FAILED: best AUC {worst:.4f} on permuted labels "
                  "(expected ~0.50).\n"
                  "    Some model is scoring without using the question. Do NOT read\n"
                  "    the main run until the negative sampling accounts for it.")
        else:
            print(f"\ncontrol passed: best AUC on permuted labels {worst:.4f} (~0.50). "
                  "The main run is readable.")

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
