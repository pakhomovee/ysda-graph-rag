"""The intervention: sigma_max in place of sigma_q.

LinearRAG propagates activation through the sentence-entity bipartite graph as

    a^t = MAX( M^T ( sigma_q  *  (M a^{t-1}) ), a^{t-1} )

where sigma_q[i] = sim(query, sentence_i) gates which sentences conduct signal.
A single pooled query vector carries no signal for hops past the first, because
neither endpoint of a (bridge -> answer) edge resembles the question. Replacing
the gate with a max over self-contained sub-questions restores it.

    sigma_max[i] = max_j sim(q_j, sentence_i)

The original question is always included in the set, so sigma_max >= sigma_q
elementwise and the method cannot score below vanilla by construction.
"""

# LinearRAG runs on Python 3.9, where `list[str] | None` in a signature is
# evaluated at definition time and raises TypeError. Deferring annotations keeps
# this module importable there. `mbuzai/__init__.py` is empty by design, so
# importing mbuzai.gate pulls in nothing else — dataio and metrics have the same
# 3.9 problem and must never become a dependency of the gate.
from __future__ import annotations

import numpy as np


def sigma_pooled(sent_emb: np.ndarray, q_emb: np.ndarray) -> np.ndarray:
    """Vanilla LinearRAG gate. sent_emb (|S|, d) and q_emb (d,) must be L2-normalised."""
    return sent_emb @ q_emb


def sigma_max(sent_emb: np.ndarray, q_embs: np.ndarray) -> np.ndarray:
    """Sub-query gate. `q_embs` is (m, d), row 0 conventionally the original question."""
    if q_embs.ndim == 1:
        q_embs = q_embs[None, :]
    return (sent_emb @ q_embs.T).max(axis=1)


def encode_query_set(model, question: str, subqs: list[str] | None) -> np.ndarray:
    """Always prepend the original question — it is the floor the max can only beat."""
    texts = [question] + list(subqs or [])
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def build_gate(model, sent_emb: np.ndarray, question: str, subqs=None) -> np.ndarray:
    """Drop-in replacement for the sigma_q computation inside LinearRAG.

    Passing subqs=None reproduces vanilla exactly, so one binary serves both arms.
    """
    return sigma_max(sent_emb, encode_query_set(model, question, subqs))
