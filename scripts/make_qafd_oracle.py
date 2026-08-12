"""Build the oracle gold map for the QAFD edge-weight probe.

Writes ``out/qafd_oracle_<dataset>.json`` = ``{normalised question: [corpus_idx, ...]}``,
which ``benchmark_runner --oracle_gold_file`` resolves to vertex indices and hands to
the diffusion as ``oracle_nodes``.

Why an oracle at all: RESULTS.md records that sigma_max on QAFD edge weights is null,
and there are two incompatible readings — the site is inert, or the heuristic is too
blunt. An oracle edge weight upper-bounds *every* scorer at that site, learned
included: no function of (h_u, h_v, h_q) can separate relevant edges better than being
handed the answer. If recall does not move under this, there is nothing to train.

The oracle is deliberately **generous** — every entity in any gold passage, not just
the ones on a shortest path between them. A generous oracle that does nothing settles
the question; a restrictive one that does nothing leaves "you picked the wrong edges"
open.

    python scripts/make_qafd_oracle.py musique
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mbuzai import dataio  # noqa: E402
from mbuzai.subq_io import normalize_question  # noqa: E402

OUT = ROOT / "out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ds = dataio.load(args.dataset)
    dest = args.out or OUT / f"qafd_oracle_{args.dataset}.json"

    gold = {}
    collisions = 0
    for q in ds.queries:
        key = normalize_question(q.question)
        if key in gold:
            # Two questions normalising to one key would silently give one of them
            # the other's gold. Union them: over-generous is the right failure
            # direction for an upper bound.
            collisions += 1
            gold[key] = sorted(set(gold[key]) | set(q.gold_pids))
        else:
            gold[key] = sorted(q.gold_pids)

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(gold))

    sizes = [len(v) for v in gold.values()]
    print(f"wrote {dest}")
    print(f"  {len(gold)} keys from {len(ds.queries)} questions "
          f"({collisions} normalisation collisions, unioned)")
    print(f"  gold passages per question: min {min(sizes)}, "
          f"mean {sum(sizes) / len(sizes):.2f}, max {max(sizes)}")
    empty = sum(1 for s in sizes if s == 0)
    if empty:
        print(f"  WARNING: {empty} questions have no gold passages — "
              "they will be skipped by the oracle and drag the coverage check")


if __name__ == "__main__":
    main()
