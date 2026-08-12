"""Score QAFD retrieval runs: recall@k per arm, plus the paired delta.

Reads the sidecar dumps written by the patched ``benchmark_runner.py``
(``qafd_<dataset>_<arm>.json``) and scores them against ``q.gold_pids``.

Simpler than ``score_linearrag.py`` in one respect: the QAFD dump values are
already indices into OUR corpus — ``benchmark_runner`` maps each retrieved doc
string back through ``docs`` before writing — so they *are* our pids and go
straight to the same recall the rest of the repo uses. No chunk expansion, and
the absolute numbers are directly comparable to ``baselines.py`` and
``eval_subq.py``.

    python scripts/score_qafd.py musique --runs out/qafd_musique_*.json

The delta table is the number that matters; the arms share an index and a seed,
so run-to-run noise (~0.002-0.005) is the floor on what to believe.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if sys.version_info < (3, 10):  # dataio/metrics use PEP 604 annotations
    sys.exit(
        f"\nFATAL: this script needs the mbuzai env (Python 3.10+), got "
        f"{sys.version.split()[0]}\n  interpreter: {sys.executable}\n"
        "  QAFD's own venv is separate and only benchmark_runner.py runs there.\n"
        "  Fix:  deactivate   (or point at the mbuzai interpreter directly)"
    )

from mbuzai import dataio, metrics  # noqa: E402
from mbuzai.subq_io import match_qid  # noqa: E402

OUT = ROOT / "out"


def recall_at(retrieved, gold, k):
    """Same estimator as mbuzai.metrics: fraction of gold passages in the top k."""
    if not gold:
        return float("nan")
    return len(set(gold) & set(retrieved[:k])) / len(gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--ks", type=int, nargs="*", default=[2, 10, 50])
    ap.add_argument("--baseline", default="vanilla", help="arm the deltas are measured against")
    args = ap.parse_args()

    # A glob over out/qafd_<ds>_*.json also matches this script's own summary,
    # which holds no question ids and would intersect every arm down to nothing.
    arms = {}
    for p in args.runs:
        data = json.loads(p.read_text())
        if not (isinstance(data, dict) and data
                and all(isinstance(v, list) for v in data.values())):
            print(f"  skipping {p.name}: not a retrieval run dump")
            continue
        arms[p.stem.split("_")[-1]] = data
    if args.baseline not in arms:
        sys.exit(f"baseline arm {args.baseline!r} not among {sorted(arms)}")

    ds = dataio.load(args.dataset)
    by_qid = {q.qid: q for q in ds.queries}
    qids = [t for t in arms[args.baseline] if match_qid(t, by_qid)
            and all(t in a for a in arms.values())]
    if not qids:
        sys.exit("no scorable questions: the run dumps share no question ids with "
                 f"{args.dataset}. Check the runs are for this dataset.")
    print(f"scoring {len(qids)} questions across arms: {', '.join(sorted(arms))}")

    gold = {t: by_qid[match_qid(t, by_qid)].gold_pids for t in qids}

    # An unmapped doc is written as -1. A run full of them would score ~0 and read
    # as a catastrophic regression rather than a broken corpus mapping.
    for name, run in sorted(arms.items()):
        bad = sum(1 for t in qids for i in run[t] if i < 0)
        if bad:
            print(f"  WARNING {name}: {bad} unmapped docs (-1) — corpus mismatch?")
        depth = max((len(run[t]) for t in qids), default=0)
        short = [k for k in args.ks if k > depth]
        if short:
            print(f"  WARNING {name}: depth {depth}; recall@{short} is really recall@{depth}.")

    print("\nabsolute recall (over OUR passage corpus — comparable to baselines.py)")
    per_q = {}
    for name, run in sorted(arms.items()):
        cells = []
        for k in args.ks:
            vals = [recall_at(run[t], gold[t], k) for t in qids]
            per_q[(name, k)] = vals
            cells.append(f"@{k}={sum(vals)/len(vals):.4f}")
        print(f"  {name:<22} {'  '.join(cells)}")

    # Depth and shape kept in separate sections, as in score_linearrag: on MuSiQue
    # the two label sets overlap and interleaving them reads as duplicate rows.
    depth, shape = {}, {}
    for i, t in enumerate(qids):
        q = by_qid[match_qid(t, by_qid)]
        depth.setdefault(f"{q.n_hops}hop-{'join' if q.is_join else 'chain'}", []).append(i)
        shape.setdefault(q.shape or "?", []).append(i)

    others = [n for n in sorted(arms) if n != args.baseline]
    for k in args.ks:
        base = per_q[(args.baseline, k)]
        print(f"\ndelta vs {args.baseline}, recall@{k}  (paired bootstrap 95% CI)")
        print("  " + " " * 25 + "".join(f"{n:>25}" for n in others))

        def row(label, idxs):
            cells = []
            for n in others:
                d = [per_q[(n, k)][i] - base[i] for i in idxs
                     if per_q[(n, k)][i] == per_q[(n, k)][i]]
                if not d:
                    cells.append(f"{'-':>25}")
                    continue
                lo, hi = metrics.bootstrap_ci(d)
                if lo != lo:
                    cells.append(f"{sum(d)/len(d):>+8.4f} {'(n<2)':>16}")
                    continue
                star = "*" if lo > 0 or hi < 0 else " "
                cells.append(f"{sum(d)/len(d):>+8.4f} [{lo:+.3f},{hi:+.3f}]{star}")
            print(f"  {label:<18} n={len(idxs):<4}" + "".join(cells))

        row("all", list(range(len(qids))))
        print("  -- depth x shape")
        for label in sorted(depth):
            row(label, depth[label])
        print("  -- question type")
        for label in sorted(shape):
            row(label, shape[label])
        print("  * = CI excludes zero")

    dest = OUT / f"scored_qafd_{args.dataset}.json"  # outside the runs glob
    dest.write_text(json.dumps(
        {f"{n}@{k}": sum(v) / len(v) for (n, k), v in per_q.items()}, indent=1))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
