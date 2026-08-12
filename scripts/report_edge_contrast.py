"""Did the edge weights actually steer anything?

Reads the ``--edge_stats_file`` dumps and tabulates the mean dispersion of the
routing distributions the diffusion actually used.

Why this exists: QAFD routes mass as ``mass[j] += excess * w_ij / sum_k w_ik``
(``graph_adapter._push``). The shares are **normalised**, so the absolute level of
an edge weight is divided out and only its spread within a neighbourhood can
influence where mass goes. A weight function with near-zero spread routes mass
exactly as uniform weights would, however it was computed.

That makes ``routing_cv`` the load-bearing diagnostic for the whole probe. A null
recall delta means one of two very different things, and only this number
separates them:

    routing_cv ~ baseline  ->  the weight never steered anything.
                               A plumbing fact. The arm proves nothing.
    routing_cv >> baseline ->  the weight steered mass and recall still did not
                               move. THAT is evidence the site is inert.

``boosted_edges`` is the same check for the oracle arms: it counts distinct
ordered (i, j) pairs whose multiplier fired, so zero means the gold map never
resolved and the arm is vacuous.

    python scripts/report_edge_contrast.py out/edgestats_musique_*.json
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stats", type=Path, nargs="+")
    ap.add_argument("--baseline", default="vanilla")
    args = ap.parse_args()

    rows = []
    for p in args.stats:
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  skipping {p.name}: {exc}")
            continue
        rows.append((d.get("arm") or p.stem, d))
    if not rows:
        sys.exit("no readable stats files")

    base_cv = next((d.get("routing_cv") for n, d in rows if n == args.baseline), None)
    if base_cv is None:
        print(f"note: no {args.baseline!r} arm here — ratios omitted\n")

    print(f"{'arm':<24}{'routing_cv':>12}{'vs base':>10}{'pushes':>12}"
          f"{'queries':>9}{'boosted_edges':>15}")
    for name, d in sorted(rows):
        cv = d.get("routing_cv")
        ratio = (f"{cv / base_cv:>9.2f}x" if cv and base_cv else f"{'-':>10}")
        print(f"{name:<24}{(f'{cv:.4f}' if cv is not None else '-'):>12}{ratio}"
              f"{d.get('pushes', 0):>12}{d.get('queries', 0):>9}"
              f"{d.get('boosted_edges', 0):>15}")

    print("\nread this before reading any recall delta:")
    print("  routing_cv flat vs baseline  -> the arm never steered mass; it is vacuous")
    print("  boosted_edges == 0 on an oracle arm -> the gold map never resolved")


if __name__ == "__main__":
    main()
