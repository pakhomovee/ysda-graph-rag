"""Score LinearRAG retrieval runs: recall@k per arm, plus the paired delta.

Reads the chunk-id dumps from `run_linearrag_retrieval.py` and the gold mapping
from `prepare_linearrag_gold.py`, and reuses `mbuzai.metrics.bootstrap_ci` so the
CIs are the same estimator as everywhere else in the repo.

The paired delta is the number that matters. Absolute recall here is over 1,354
chunks of ~820 words, not 11,656 passages of ~80, so it is not comparable to
`baselines.py` or `eval_subq.py` — but both arms see the identical corpus and the
identical index, so vanilla vs sigma_max is clean.

    python scripts/score_linearrag.py musique --runs out/linearrag_musique_*.json
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
        "  LinearRAG's .venv-linear is 3.9 and only run_linearrag_retrieval.py runs there.\n"
        "  Fix:  deactivate   (or point at the mbuzai interpreter directly)"
    )

from mbuzai import dataio, metrics  # noqa: E402
from mbuzai.subq_io import match_qid  # noqa: E402

OUT = ROOT / "out"


def recall_at(retrieved, gold, k):
    """Fraction of gold PASSAGES covered by the top-k retrieved chunks.

    `gold` is one entry per gold passage, each listing the chunks that would
    satisfy it — a passage straddling an overlap has two. Counting distinct
    chunks instead would let one passage score twice.
    """
    if not gold:
        return float("nan")
    top = set(retrieved[:k])
    if not isinstance(gold[0], list):        # legacy flat format
        return len(top & set(gold)) / len(gold)
    return sum(1 for opts in gold if top & set(opts)) / len(gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="our dataset name, for the hop/shape breakdown")
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--gold", type=Path, default=None)
    ap.add_argument("--ks", type=int, nargs="*", default=[2, 5, 10])
    ap.add_argument("--baseline", default="vanilla", help="arm the deltas are measured against")
    args = ap.parse_args()

    gold_path = args.gold or OUT / f"linearrag_gold_{args.dataset}.json"
    gold = json.loads(gold_path.read_text())
    # A glob over out/linearrag_<ds>_*.json also matches this script's own
    # summary, which then becomes an "arm" holding no question ids and
    # intersects every arm down to nothing. Take only real run dumps.
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

    # their ids are keyed inconsistently across datasets; match_qid absorbs that.
    # Keys stay in their form so the run dumps line up, and resolve to ours for
    # the hop/shape buckets.
    ds = dataio.load(args.dataset)
    by_qid = {q.qid: q for q in ds.queries}
    qids = [t for t in gold if match_qid(t, by_qid)
            and all(t in a for a in arms.values())]
    if not qids:
        sys.exit("no scorable questions: the run dumps and the gold file share no "
                 "question ids. Regenerate gold with prepare_linearrag_gold.py, or "
                 "check the runs are for this dataset.")
    print(f"scoring {len(qids)} questions across arms: {', '.join(sorted(arms))}")

    # recall@k over a list shorter than k is recall@len(list). Reranked arms are
    # truncated to their topk, so scoring them at a larger k silently compares a
    # 10-item list against 50-item ones and reports a huge spurious deficit.
    for name, run in sorted(arms.items()):
        depth = max((len(v) for v in run.values()), default=0)
        bad = [k for k in args.ks if k > depth]
        if bad:
            print(f"  WARNING {name}: depth {depth}; recall@{bad} is really recall@{depth}. "
                  "Do not compare it against deeper arms.")
    if len(qids) < len(gold):
        print(f"  ({len(gold) - len(qids)} skipped: no gold chunks, or absent from a run)")

    # Their chunk ids top out around 1.3k; ours run to len(corpus). If the runs
    # index our own passages, the numbers ARE comparable and saying otherwise
    # would be the more damaging error.
    max_id = max((max(v) for v in arms[args.baseline].values() if v), default=0)
    fine = max_id >= len(ds.corpus) * 0.5
    print("\nabsolute recall  " + ("(over OUR passage corpus — directly comparable to "
          "baselines.py and eval_subq.py)" if fine else
          "(over LinearRAG's chunk corpus — NOT comparable to baselines.py)"))
    per_q = {}
    for name, run in sorted(arms.items()):
        cells = []
        for k in args.ks:
            vals = [recall_at(run[t], gold[t], k) for t in qids]
            per_q[(name, k)] = vals
            cells.append(f"@{k}={sum(vals)/len(vals):.4f}")
        print(f"  {name:<12} {'  '.join(cells)}")

    # Kept in separate sections, as in eval_subq: on MuSiQue the shape labels and
    # the depth labels overlap ("2hop" vs "2hop-chain"), and interleaving them
    # alphabetically reads as duplicate rows rather than two different cuts.
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

    dest = OUT / f"scored_linearrag_{args.dataset}.json"  # outside the runs glob
    dest.write_text(json.dumps(
        {f"{n}@{k}": sum(v) / len(v) for (n, k), v in per_q.items()}, indent=1))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
