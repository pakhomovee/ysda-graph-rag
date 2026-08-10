"""Recall@k with the breakdowns the experiment actually turns on.

Overall recall is the headline number, but the claim is about *where* the gain
lands, so every run also reports recall split by hop count, by composition shape,
and — on MuSiQue — per individual hop.
"""

from collections import defaultdict
from statistics import mean

import numpy as np

from .dataio import Dataset, Query


def _recall(ranked: list[int], gold: set[int], k: int) -> float:
    if not gold:
        return float("nan")
    return len(gold & set(ranked[:k])) / len(gold)


def bootstrap_ci(values: list[float], n: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile CI. The 4-hop buckets are small (n=166, n=27) — never report a
    bare delta on those."""
    vals = np.asarray([v for v in values if v == v])
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = vals[rng.integers(0, len(vals), size=(n, len(vals)))].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [alpha / 2, 1 - alpha / 2]))


def evaluate(ds: Dataset, ranked: list[list[int]], ks=(2, 5, 10)) -> dict:
    """`ranked[i]` is the retrieved pid list for `ds.queries[i]`, best first."""
    assert len(ranked) == len(ds.queries), "one ranking per query"
    per_q = {k: [_recall(r, q.gold_pids, k) for r, q in zip(ranked, ds.queries)] for k in ks}

    out = {
        "n": len(ds.queries),
        "overall": {},
        "by_hops": defaultdict(dict),
        "by_shape": defaultdict(dict),
        "per_hop": {},
    }

    for k in ks:
        vals = [v for v in per_q[k] if v == v]
        lo, hi = bootstrap_ci(vals)
        out["overall"][f"recall@{k}"] = {"mean": mean(vals), "ci95": [lo, hi], "n": len(vals)}

    def _group(attr):
        buckets = defaultdict(list)
        for i, q in enumerate(ds.queries):
            buckets[getattr(q, attr)].append(i)
        return buckets

    for label, buckets in (("by_hops", _group("n_hops")), ("by_shape", _group("shape"))):
        for key, idxs in sorted(buckets.items(), key=lambda kv: str(kv[0])):
            for k in ks:
                vals = [per_q[k][i] for i in idxs if per_q[k][i] == per_q[k][i]]
                if not vals:
                    continue
                lo, hi = bootstrap_ci(vals)
                out[label][str(key)][f"recall@{k}"] = {
                    "mean": mean(vals), "ci95": [lo, hi], "n": len(vals),
                }

    out["per_hop"] = per_hop_recall(ds, ranked, ks)
    out["by_hops"] = dict(out["by_hops"])
    out["by_shape"] = dict(out["by_shape"])
    return out


def per_hop_recall(ds: Dataset, ranked: list[list[int]], ks=(2, 5, 10)) -> dict:
    """Was hop i's own gold passage retrieved?

    This is the diagnostic the whole project turns on: the prediction is that
    query-aware weighting helps hop 1 and goes flat or negative beyond it.
    Requires `paragraph_support_idx`, so MuSiQue only.
    """
    hits: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r, q in zip(ranked, ds.queries):
        for i, hop in enumerate(q.hops, start=1):
            if hop.gold_pid is None:
                continue
            for k in ks:
                hits[i][k].append(float(hop.gold_pid in set(r[:k])))

    out = {}
    for hop_i in sorted(hits):
        out[f"hop{hop_i}"] = {}
        for k in ks:
            vals = hits[hop_i][k]
            if not vals:
                continue
            lo, hi = bootstrap_ci(vals)
            out[f"hop{hop_i}"][f"recall@{k}"] = {
                "mean": mean(vals), "ci95": [lo, hi], "n": len(vals),
            }
    return out


def render(report: dict, title: str = "") -> str:
    lines = []
    if title:
        lines.append(f"\n{title}\n{'=' * len(title)}")
    lines.append(f"n = {report['n']}")

    def row(label, d):
        cells = "  ".join(
            f"{k}={v['mean']:.4f} [{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}]"
            for k, v in d.items()
        )
        return f"  {label:<14} {cells}"

    lines.append("\noverall")
    lines.append(row("", report["overall"]))
    for section, header in (("per_hop", "per hop (gold passage for that step)"),
                            ("by_hops", "by hop count"),
                            ("by_shape", "by shape")):
        if report.get(section):
            lines.append(f"\n{header}")
            for key, d in report[section].items():
                n = next(iter(d.values()))["n"]
                lines.append(row(f"{key} (n={n})", d))
    return "\n".join(lines)
