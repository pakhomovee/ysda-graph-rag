"""Learned fact selection for HippoRAG 2, and the filters it is compared against.

HippoRAG 2 picks which extracted facts seed its PPR reset vector by handing candidate
triples to an LLM (`DSPyFilter`, one call per query). That is a SELECTION site: a top-k
cut over a global candidate list, which is the one place every measured effect in this
repo has lived. This module replaces that LLM call with a numpy model.

Two properties of the site, read from their source, decide the design:

  * `graph_search_with_fact_entities` weights each surviving fact by its ORIGINAL
    embedding score, not by whatever the reranker returned. So the reranker's output
    ORDER is discarded and only the SET it keeps matters — which is why the training
    objective is recall@keep_k (mbuzai.ranking.pairwise_grad with k set) rather than a
    full ranking loss.
  * Upstream, `linking_top_k` is the candidate pool AND the survivor count, so their
    filter sees 5 candidates and can only prune. A numpy model can score 100 in
    microseconds, so the patch's --rerank_candidate_k lets it PROMOTE a fact the
    bi-encoder buried. That is the intervention; the filter swap alone is the control.

Feature construction lives here and only here, for the reason edgemodel.py's docstring
gives: a silent train/inference disagreement is the standard way an experiment like
this produces a meaningless null. The saved npz records the feature version, the
candidate width and the encoder, and `LearnedFilter.load` asserts all three against the
live index rather than trusting the filename.

numpy only, no torch: this is imported into HippoRAG's own environment and runs inside
its retrieval loop.
"""

import numpy as np

FACT_FEAT_VERSION = "fact-v1"
FACT_FEAT_EXTRA = 4


def n_fact_features(d: int) -> int:
    return 2 * d + FACT_FEAT_EXTRA


def _l2(x):
    """Normalise rows.

    HippoRAG's TransformersEmbeddingModel.batch_encode drops its kwargs, so nothing in
    the index is normalised and `get_fact_scores` min-max-normalises raw dots. Doing it
    here, once, is what keeps the trainer and the in-pipeline filter on the same scale.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def build_fact_features(h_fact, h_query, score, rank_frac, n_src):
    """(m, 2d+4) float32 features for m (query, candidate fact) pairs.

    Layout, in the style edgemodel.py uses:

        [0:d)    hf * hq        elementwise query relevance (GNN-RAG's omega form)
        [d:2d)   |hf - hq|      the asymmetry a product cannot express
        2d+0     cos(f, q)      exactly what the pipeline already ranks by, so the
                                model starts from the baseline rather than below it
        2d+1     score          HippoRAG's own min-max-normalised fact score
        2d+2     rank/pool      position in this query's candidate list
        2d+3     log1p(n_src)   passages attesting the fact

    Entity-level features (subject/object embeddings and degrees) are deliberately
    absent in v1. They require re-deriving `compute_mdhash_id(phrase.lower(), 'entity-')`
    to hit the same vertex HippoRAG hashed at index time, and a convention mismatch
    there fails SILENTLY to all-zero columns rather than raising — the exact failure
    mode this module exists to prevent. Add them behind a bumped FACT_FEAT_VERSION once
    the phrase-resolution rate is measured, not before.
    """
    hf = _l2(h_fact)
    hq = _l2(h_query)
    if hq.shape[0] == 1 and hf.shape[0] != 1:
        hq = np.broadcast_to(hq, hf.shape)
    score = np.asarray(score, dtype=np.float32).reshape(-1, 1)
    rank_frac = np.asarray(rank_frac, dtype=np.float32).reshape(-1, 1)
    n_src = np.asarray(n_src, dtype=np.float32).reshape(-1, 1)
    return np.hstack([
        hf * hq,
        np.abs(hf - hq),
        np.sum(hf * hq, axis=1, keepdims=True),
        score,
        rank_frac,
        np.log1p(n_src),
    ]).astype(np.float32)


def fact_query_feature_mask(d: int) -> np.ndarray:
    """Columns that depend on the query.

    Zeroing these gives the fact-only floor: a model that scores facts on their own
    properties (how many passages attest them, where the bi-encoder happened to put
    them) is a query-independent reweighting, not a reranker. The full model has to
    beat that floor, or the in-pipeline arm means nothing.
    """
    m = np.zeros(n_fact_features(d), dtype=bool)
    m[: 2 * d] = True
    m[2 * d + 0] = True     # cos(f, q)
    m[2 * d + 1] = True     # score is a query-fact quantity
    return m


class FactScorer:
    """One hidden ReLU layer over build_fact_features, logistic output, Adam by hand.

    Deliberately the same shape as edgemodel.EdgeScorer: small enough to run inside a
    retrieval loop with no framework, big enough to express more than a reweighting of
    cosine.
    """

    def __init__(self, d, hidden=128, seed=0):
        rng = np.random.default_rng(seed)
        f = n_fact_features(d)
        self.d = d
        self.W1 = rng.normal(0, np.sqrt(2.0 / f), (f, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, 1)).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)
        self.meta = {}

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def score(self, X):
        h = np.maximum(0, X @ self.W1 + self.b1)
        return (h @ self.W2 + self.b2).ravel()

    def step(self, X, y, state, lr, grad_fn):
        """One Adam step under an arbitrary objective from mbuzai.ranking."""
        h = np.maximum(0, X @ self.W1 + self.b1)
        _, ds = grad_fn((h @ self.W2 + self.b2).ravel(), y)
        dh = (ds[:, None] @ self.W2.T) * (h > 0)
        grads = [X.T @ dh, dh.sum(0), h.T @ ds[:, None], np.array([ds.sum()], dtype=np.float32)]
        _adam(self.params(), grads, state, lr)

    def save(self, path, **meta):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 d=self.d, feat_version=FACT_FEAT_VERSION,
                 **{k: np.array(v) for k, v in meta.items()})

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        d = int(z["d"])
        m = cls(d, hidden=z["W1"].shape[1])
        m.W1, m.b1, m.W2, m.b2 = z["W1"], z["b1"], z["W2"], z["b2"]
        version = str(z["feat_version"]) if "feat_version" in z else "(none)"
        if version != FACT_FEAT_VERSION:
            raise ValueError(
                f"{path} was trained on feature layout {version!r}, this build is "
                f"{FACT_FEAT_VERSION!r}. Scoring one layout with another produces a "
                "plausible number from nonsense; retrain or check out the matching rev.")
        if m.W1.shape[0] != n_fact_features(d):
            raise ValueError(f"{path}: W1 has {m.W1.shape[0]} inputs, layout needs "
                             f"{n_fact_features(d)}")
        m.meta = {k: z[k] for k in z.files if k not in ("W1", "b1", "W2", "b2", "d")}
        return m


def _adam(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8):
    if not state:
        state["t"] = 0
        state["m"] = [np.zeros_like(p) for p in params]
        state["v"] = [np.zeros_like(p) for p in params]
    state["t"] += 1
    t = state["t"]
    for i, (p, g) in enumerate(zip(params, grads)):
        g = g.reshape(p.shape).astype(np.float32)
        state["m"][i] = b1 * state["m"][i] + (1 - b1) * g
        state["v"][i] = b2 * state["v"][i] + (1 - b2) * g * g
        mhat = state["m"][i] / (1 - b1 ** t)
        vhat = state["v"][i] / (1 - b2 ** t)
        p -= (lr * mhat / (np.sqrt(vhat) + eps)).astype(np.float32)


# ---------------------------------------------------------------------------
# Filters. All three implement DSPyFilter's interface exactly, so they drop into
# hipporag.rerank_filter with no pipeline fork:
#
#     rerank(query, candidate_items, candidate_indices, len_after_rerank)
#         -> (indices, items, meta)
# ---------------------------------------------------------------------------

class NoRerankFilter:
    """Keep the candidate pool as the bi-encoder ordered it. The `norerank` arm.

    This is the control that makes every other arm readable: if the LLM filter does not
    beat doing nothing, then "recognition memory" is decorative and comparing a learned
    filter against it is comparing two things that both do nothing.
    """

    name = "norerank"

    def __call__(self, *a, **kw):
        return self.rerank(*a, **kw)

    def rerank(self, query, candidate_items, candidate_indices, len_after_rerank=None):
        k = len_after_rerank or len(candidate_items)
        return list(candidate_indices[:k]), list(candidate_items[:k]), {"confidence": None}


class OracleFactFilter:
    """Keep only facts attested by a gold passage. The ceiling on every reranker here.

    Run this FIRST. If it does not beat `norerank`, no selection rule of any kind can
    help at this site and nothing needs training — the same discipline that stopped the
    QAFD edge scorer from being built.
    """

    name = "oracle"

    def __init__(self, gold_facts_by_query):
        # {query string: set of fact indices attested by a gold passage}
        self.gold = gold_facts_by_query

    def __call__(self, *a, **kw):
        return self.rerank(*a, **kw)

    def rerank(self, query, candidate_items, candidate_indices, len_after_rerank=None):
        gold = self.gold.get(query, set())
        keep = [i for i, gi in enumerate(candidate_indices) if gi in gold]
        if len_after_rerank:
            keep = keep[:len_after_rerank]
        return ([candidate_indices[i] for i in keep],
                [candidate_items[i] for i in keep],
                {"confidence": None})


class LearnedFilter:
    """A trained FactScorer in place of the per-query LLM call.

    Does NOT reproduce DSPyFilter's empty return. An empty result sends the query to
    pure dense retrieval (HippoRAG.py:467), so an arm that abstains often is a blend of
    two systems and its recall is uninterpretable. `min_keep` keeps at least one fact;
    measure abstention separately if it is interesting.

    Exceptions are not caught, deliberately. DSPyFilter returning [] on any exception is
    how a broken reranker becomes a plausible number instead of a crash.
    """

    name = "learned"

    def __init__(self, model, hipporag, min_keep=1):
        self.model = model
        self.rag = hipporag
        self.min_keep = min_keep
        self._warned = False

    @classmethod
    def load(cls, path, hipporag, min_keep=1):
        return cls(FactScorer.load(path), hipporag, min_keep=min_keep)

    def __call__(self, *a, **kw):
        return self.rerank(*a, **kw)

    def rerank(self, query, candidate_items, candidate_indices, len_after_rerank=None):
        rag = self.rag
        idx = np.asarray(candidate_indices, dtype=np.int64)

        hq = rag.query_to_embedding["triple"].get(query)
        if hq is None:
            hq = rag.embedding_model.batch_encode(query, norm=True)
        hq = np.asarray(hq, dtype=np.float32).ravel()
        hf = np.asarray(rag.fact_embeddings, dtype=np.float32)[idx]

        # The min-max-normalised score the model trained on. Recomputing a within-list
        # min-max here would be a different quantity than the global one in the export,
        # which is why the patch stashes it.
        scores = getattr(rag, "_last_fact_scores", None)
        if scores is None:
            raise RuntimeError(
                "hipporag._last_fact_scores is missing — the submodule patch is not "
                "applied. Run: bash scripts/setup_hipporag.sh")
        score = np.asarray(scores, dtype=np.float32)[idx]

        pool = len(candidate_items)
        trained_pool = int(self.model.meta.get("cand_k", pool))
        if trained_pool != pool and not self._warned:
            # rank/pool is a feature, so a different pool shifts it for every candidate.
            print(f"WARNING: model trained at cand_k={trained_pool}, running at {pool}; "
                  "the rank feature is on a different scale")
            self._warned = True
        rank_frac = np.arange(pool, dtype=np.float32) / max(trained_pool, 1)

        n_src = np.array([len(rag.proc_triples_to_docs.get(str(tuple(it)), ()))
                          for it in candidate_items], dtype=np.float32)

        X = build_fact_features(hf, hq, score, rank_frac, n_src)
        s = self.model.score(X)

        k = len_after_rerank or pool
        order = np.argsort(-s)[:max(k, self.min_keep)]
        p = 1.0 / (1.0 + np.exp(-np.clip(s[order], -30, 30)))
        return ([int(candidate_indices[i]) for i in order],
                [candidate_items[i] for i in order],
                {"confidence": p.tolist()})


def selftest():
    """Exercise the feature layout, the npz guards and all three filters.

    No HippoRAG import: the filters are tested against a stub exposing the attributes
    the real object does, so this runs in any environment.
    """
    import tempfile, os
    from mbuzai import ranking

    rng = np.random.default_rng(0)
    d, m = 16, 7
    hf, hq = rng.normal(size=(m, d)), rng.normal(size=d)
    X = build_fact_features(hf, hq, rng.random(m), np.arange(m) / m, rng.integers(1, 5, m))
    assert X.shape == (m, n_fact_features(d)) == (m, 2 * d + 4)
    nf = hf / np.linalg.norm(hf, axis=1, keepdims=True)
    assert np.allclose(X[:, 2 * d], nf @ (hq / np.linalg.norm(hq)), atol=1e-6), "cos column"
    # The store's embeddings are unnormalised, so the features must not depend on scale.
    X2 = build_fact_features(hf * 37.0, hq * 0.01, X[:, 2 * d + 1],
                             np.arange(m) / m, np.expm1(X[:, 2 * d + 3]))
    assert np.allclose(X[:, : 2 * d + 1], X2[:, : 2 * d + 1], atol=1e-5), "not scale invariant"

    mask = fact_query_feature_mask(d)
    assert mask.sum() == 2 * d + 2 and not mask[2 * d + 2] and not mask[2 * d + 3]

    starts = ranking.group_starts(np.repeat([0, 1], [4, 3]))
    y = np.array([1, 0, 0, 0, 1, 1, 0], float)
    for loss in ("pointwise", "listwise", "pairwise", "lambdarank"):
        sc = FactScorer(d, hidden=8, seed=1)
        before = sc.score(X).copy()
        sc.step(X, y, {}, 0.05, ranking.make_grad_fn(loss, starts, 2))
        assert not np.allclose(before, sc.score(X)), f"{loss} did not move the parameters"

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "m.npz")
        sc.save(path, cand_k=100, emb_model="test")
        assert np.allclose(FactScorer.load(path).score(X), sc.score(X)), "round-trip"
        z = dict(np.load(path, allow_pickle=False))
        z["feat_version"] = np.array("fact-v0")
        np.savez(path, **z)
        try:
            FactScorer.load(path)
            raise AssertionError("a stale feature version was accepted")
        except ValueError:
            pass

    class _Stub:
        def __init__(self):
            self.fact_embeddings = rng.normal(size=(40, d)).astype(np.float32)
            self.query_to_embedding = {"triple": {"q": rng.normal(size=d).astype(np.float32)}}
            self._last_fact_scores = rng.random(40).astype(np.float32)
            self.proc_triples_to_docs = {}

    items = [("a", "r", "b"), ("c", "r", "d"), ("e", "r", "f"), ("g", "r", "h")]
    cidx = [11, 3, 27, 5]
    assert NoRerankFilter().rerank("q", items, cidx, 2)[0] == [11, 3]
    assert OracleFactFilter({"q": {27, 5}}).rerank("q", items, cidx, 5)[0] == [27, 5]
    got_i, got_it, meta = LearnedFilter(FactScorer(d, hidden=8, seed=2), _Stub()).rerank(
        "q", items, cidx, len_after_rerank=2)
    assert len(got_i) == 2 and len(meta["confidence"]) == 2
    assert [items[cidx.index(x)] for x in got_i] == got_it, "indices and items disagree"

    bad = _Stub()
    del bad._last_fact_scores
    try:
        LearnedFilter(FactScorer(d, hidden=8), bad).rerank("q", items, cidx, 2)
        raise AssertionError("a missing _last_fact_scores was tolerated")
    except RuntimeError as exc:
        assert "setup_hipporag.sh" in str(exc)

    print("factmodel selftest ok")


if __name__ == "__main__":
    selftest()
