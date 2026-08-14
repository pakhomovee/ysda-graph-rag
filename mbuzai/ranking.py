"""Ranking objectives, shared by every learned scorer in this repo.

These live here rather than in one script because two trainers now use them and a
second copy is how a train/inference disagreement gets introduced — the same reason
mbuzai/edgemodel.py keeps feature construction in exactly one place.

The three gradients are checked against finite differences in selftest(); a loss that
is merely plausible is the standard way an experiment like this produces a
meaningless null.

There is deliberately NO module-level objective state. An earlier version kept the
selected loss in a global that the trainer set and the gradient read, which works only
as long as both live in the same module — moving the functions out would have left the
trainer setting a global nothing read, silently reverting every run to pointwise.
make_grad_fn() closes over the choice instead, so the objective travels with the
callable and forked workers inherit it correctly.
"""

import numpy as np


def group_starts(qi):
    """Boundaries of contiguous per-question row blocks.

    Rows are built one question at a time and the validation split removes whole
    questions, so each question's rows stay contiguous and a single boundary
    array describes them.
    """
    if len(qi) == 0:
        return np.array([0], dtype=np.int64)
    cuts = np.flatnonzero(np.diff(qi) != 0) + 1
    return np.concatenate([[0], cuts, [len(qi)]]).astype(np.int64)


def listwise_grad(s, y, starts):
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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def pairwise_grad(s, y, starts, k=None):
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
        g = -sigmoid(-d) * w                        # dL/d(s_pos - s_neg)
        view = ds[a:b]
        view[pi] += g.sum(1)
        view[ni] -= g.sum(0)
        groups += 1
    return 0.0, ds / max(groups, 1)


def recall_at_k(scores, y, starts, k=10):
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

def pointwise_grad(s, y):
    """Mean BCE-with-logits and dL/ds.

    Weights every row equally regardless of rank, which optimises a global order —
    good for ROC-AUC over thousands of candidates, indifferent to the top of any one
    question's list.
    """
    p = 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))
    loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
    return loss, (p - y) / len(y)


def make_grad_fn(loss, starts=None, k=10):
    """(loss name, row groups) -> a grad_fn(s, y) -> (loss, ds).

    Explicit rather than global: the returned callable carries the objective, so it
    cannot be set in one place and read in another.
    """
    if loss == "pointwise":
        return pointwise_grad
    if starts is None:
        raise ValueError(f"{loss} needs per-question row groups (starts)")
    if loss == "listwise":
        return lambda s, y: listwise_grad(s, y, starts)
    if loss == "pairwise":
        return lambda s, y: pairwise_grad(s, y, starts, k=None)
    if loss == "lambdarank":
        return lambda s, y: pairwise_grad(s, y, starts, k=k)
    raise ValueError(f"unknown loss: {loss}")


def roc_auc(scores, labels):
    """Rank-based ROC-AUC. Ties get average ranks, as the Mann-Whitney form requires."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks within ties
    uniq, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    tie_sum = np.zeros(len(uniq))
    np.add.at(tie_sum, inv, ranks)
    ranks = (tie_sum / cnt)[inv]
    return float((ranks[y > 0].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def selftest():
    """Finite-difference every gradient, plus the properties they are chosen for."""
    rng = np.random.default_rng(0)
    qi = np.repeat([0, 1, 2], [12, 9, 15])
    starts = group_starts(qi)
    assert list(starts) == [0, 12, 21, 36], starts
    y = np.zeros(len(qi)); y[[0, 1, 2, 12, 13, 21, 22, 23, 24]] = 1.0
    s0 = rng.normal(size=len(qi))

    def listwise_loss(s):
        tot = 0.0
        for a, b in zip(starts[:-1], starts[1:]):
            yg = y[a:b]; g = yg.sum()
            if g <= 0:
                continue
            e = np.exp(s[a:b] - s[a:b].max())
            tot += -np.sum((yg / g) * np.log(e / e.sum() + 1e-12))
        return tot / (len(starts) - 1)

    def ranknet_loss(s, k):
        tot, groups = 0.0, 0
        for a, b in zip(starts[:-1], starts[1:]):
            sg, yg, s0g = s[a:b], y[a:b], s0[a:b]
            pi = np.flatnonzero(yg > 0); ni = np.flatnonzero(yg <= 0)
            if not len(pi) or not len(ni):
                continue
            d = sg[pi][:, None] - sg[ni][None, :]
            if k is None:
                w = np.full(d.shape, 1.0 / d.size)
            else:
                order = np.argsort(-s0g)
                rank = np.empty(len(s0g), int); rank[order] = np.arange(len(s0g))
                w = ((rank[pi] >= k)[:, None] & (rank[ni] < k)[None, :]).astype(float) / len(pi)
            tot += np.sum(w * np.log1p(np.exp(-np.clip(d, -30, 30))))
            groups += 1
        return tot / max(groups, 1)

    def check(name, analytic, loss_fn, tol=1e-8):
        num = np.zeros_like(s0); eps = 1e-6
        for i in range(len(s0)):
            a = s0.copy(); a[i] += eps
            b = s0.copy(); b[i] -= eps
            num[i] = (loss_fn(a) - loss_fn(b)) / (2 * eps)
        err = np.abs(analytic - num).max()
        assert err < tol, (name, err)
        print(f"  {name:<22} max |analytic - numeric| = {err:.3e}")

    check("listwise", listwise_grad(s0, y, starts)[1], listwise_loss)
    check("pairwise (RankNet)", pairwise_grad(s0, y, starts)[1], lambda s: ranknet_loss(s, None))
    check("lambdarank recall@3", pairwise_grad(s0, y, starts, k=3)[1], lambda s: ranknet_loss(s, 3))

    p = 1 / (1 + np.exp(-s0))
    assert np.allclose(pointwise_grad(s0, y)[1], (p - y) / len(y))

    # lambdarank's defining property: no push on positives already inside the cutoff
    ds = pairwise_grad(s0, y, starts, k=3)[1]
    sg, yg = s0[0:12], y[0:12]
    order = np.argsort(-sg); rank = np.empty(12, int); rank[order] = np.arange(12)
    inside = [i for i in range(12) if yg[i] > 0 and rank[i] < 3]
    assert all(abs(ds[i]) < 1e-12 for i in inside), "positive inside top-k must get no push"
    print(f"  positives already inside top-3 get zero gradient ({len(inside)} of "
          f"{int(yg.sum())} in group 0)")

    # groups with no positives contribute nothing
    y2 = y.copy(); y2[0:12] = 0
    assert np.allclose(listwise_grad(np.zeros(len(y)), y2, starts)[1][0:12], 0.0)
    assert np.allclose(pairwise_grad(s0, y2, starts)[1][0:12], 0.0)

    # recall@k at both bounds
    perfect = np.where(y > 0, 10.0, -10.0)
    assert abs(recall_at_k(perfect, y, starts, k=5) - 1.0) < 1e-12
    assert recall_at_k(-perfect, y, starts, k=1) < 1.0

    # roc_auc against a brute-force pair count, ties included
    for _ in range(3):
        sc = rng.integers(0, 4, 40).astype(float)   # deliberate ties
        lb = (rng.random(40) < 0.4).astype(float)
        pos, neg = sc[lb > 0], sc[lb <= 0]
        brute = np.mean([(1.0 if a > b else 0.5 if a == b else 0.0)
                         for a in pos for b in neg])
        assert abs(roc_auc(sc, lb) - brute) < 1e-12, (roc_auc(sc, lb), brute)
    print("  pointwise BCE, empty groups, recall@k bounds, roc_auc ties: ok")
    print("selftest ok")


if __name__ == "__main__":
    selftest()
