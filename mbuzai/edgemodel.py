"""A light learned edge weight for flow diffusion: w(u, v, q).

Replaces QAFD's heuristic `H_sim(h_u,h_v) * (a + b(H_sim(h_u,h_q)+H_sim(h_v,h_q)))`
with a small MLP over the same arguments plus cheap structural features.

Both the trainer and the patched `graph_adapter` import this module, so the
feature construction and the forward pass have exactly one definition. A silent
disagreement between how features are built at training time and at inference is
the standard way an experiment like this produces a meaningless null.

numpy only: the model is ~200k parameters and inference runs inside the diffusion
loop, where a torch import per worker would cost more than the arithmetic.
"""

from __future__ import annotations

import numpy as np

# Feature layout, in one place. `d` is the embedding dimension.
#   h_u * h_q          d    query relevance of the source
#   h_v * h_q          d    query relevance of the target
#   h_u * h_v          d    structural affinity, what the heuristic already has
#   |h_u - h_v|        d    asymmetry the product form cannot express
#   cos(u,q), cos(v,q), cos(u,v)          3
#   log1p(deg_u), log1p(deg_v), w_struct  3
FEAT_EXTRA = 6


def n_features(d: int) -> int:
    return 4 * d + FEAT_EXTRA


def build_features(hu, hv, hq, deg_u, deg_v, w_struct):
    """(m, 4d+6) float32. All embeddings must already be L2-normalised."""
    hu = np.asarray(hu, dtype=np.float32)
    hv = np.asarray(hv, dtype=np.float32)
    hq = np.asarray(hq, dtype=np.float32)
    if hq.ndim == 1:
        hq = np.broadcast_to(hq, hu.shape)
    return np.hstack([
        hu * hq,
        hv * hq,
        hu * hv,
        np.abs(hu - hv),
        np.sum(hu * hq, axis=1, keepdims=True),
        np.sum(hv * hq, axis=1, keepdims=True),
        np.sum(hu * hv, axis=1, keepdims=True),
        np.log1p(np.asarray(deg_u, dtype=np.float32)).reshape(-1, 1),
        np.log1p(np.asarray(deg_v, dtype=np.float32)).reshape(-1, 1),
        np.asarray(w_struct, dtype=np.float32).reshape(-1, 1),
    ]).astype(np.float32)


class EdgeScorer:
    """One hidden layer, logistic output. score() returns a raw logit."""

    def __init__(self, d: int, hidden: int = 128, seed: int = 0):
        rng = np.random.default_rng(seed)
        f = n_features(d)
        self.d = d
        self.W1 = rng.normal(0, np.sqrt(2.0 / f), (f, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, 1)).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)

    # -- training ------------------------------------------------------
    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, X):
        h = np.maximum(0, X @ self.W1 + self.b1)
        return h, (h @ self.W2 + self.b2).ravel()

    def step(self, X, y, state, lr):
        h, s = self.forward(X)
        p = 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))
        ds = ((p - y) / len(y)).astype(np.float32)
        dh = (ds[:, None] @ self.W2.T) * (h > 0)
        grads = [X.T @ dh, dh.sum(0), h.T @ ds[:, None], np.array([ds.sum()], np.float32)]
        for i, (prm, g) in enumerate(zip(self.params(), grads)):
            m, v, t = state.setdefault(i, (np.zeros_like(prm), np.zeros_like(prm), 0))
            t += 1
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * (g * g)
            state[i] = (m, v, t)
            prm -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
        return float(loss)

    # -- inference -----------------------------------------------------
    def score(self, X):
        return self.forward(X)[1]

    def weight(self, X, beta: float, w_struct):
        """Edge weight handed to the diffusion.

        w_struct * exp(beta * sigmoid(logit)). Multiplicative on the structural
        weight, exactly like QAFD's Hybrid form, but the query-dependent factor
        spans exp(0)..exp(beta) instead of the bounded [1, 1.5]. Routing
        normalises within a neighbourhood, so this range is what decides whether
        the model can steer at all.
        """
        p = 1.0 / (1.0 + np.exp(-np.clip(self.score(X), -30, 30)))
        return np.asarray(w_struct, dtype=np.float64) * np.exp(beta * p)

    # -- persistence ---------------------------------------------------
    def save(self, path, **meta):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 d=np.int64(self.d), **meta)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        m = cls(int(z["d"]), hidden=z["W1"].shape[1])
        m.W1, m.b1, m.W2, m.b2 = z["W1"], z["b1"], z["W2"], z["b2"]
        m.meta = {k: z[k] for k in z.files if k not in ("W1", "b1", "W2", "b2", "d")}
        return m
