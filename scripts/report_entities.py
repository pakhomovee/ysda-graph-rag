"""Tier histograms for the entity activation — does the graph search go anywhere?

The claim is about reaching *distant* entities, and recall@k cannot see that. This
reads the `entities_*.json` sidecars from `run_linearrag_retrieval.py` and reports,
per arm, how far activation actually travelled.

Tier 1 is a seed, tier 2 is one hop, tier 3 is two hops. Propagation is
multiplicative and pruned below `iteration_threshold`, so with mpnet cosines
(~0.3-0.5) against a 0.4 threshold, hop 2 needs sigma ~0.75 and should be close to
absent at the default config. If tier 1 dominates, sigma_max has nothing to act on
and the search is one hop wearing a multi-hop name.

    python scripts/report_entities.py musique --traces out/entities_linearrag_musique_*.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mbuzai import dataio  # noqa: E402
from mbuzai.subq_io import match_qid  # noqa: E402

OUT = ROOT / "out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--traces", type=Path, nargs="+", required=True)
    ap.add_argument("--gold", type=Path, default=None,
                    help="linearrag_gold_<ds>.json, for the gold-overlap column")
    args = ap.parse_args()

    ds = dataio.load(args.dataset)
    by_qid = {q.qid: q for q in ds.queries}
    corpus = ds.corpus

    gold_path = args.gold or OUT / f"linearrag_gold_{args.dataset}.json"
    gold_text = {}
    if gold_path.exists():
        gold = json.loads(gold_path.read_text())
        for t in gold:
            q = by_qid.get(match_qid(t, by_qid))
            if q:
                gold_text[t] = " ".join(
                    (corpus[p]["title"] + " " + corpus[p]["text"]).lower()
                    for p in q.gold_pids
                )

    for path in args.traces:
        trace = json.loads(path.read_text())
        arm = path.stem.split("_")[-1]
        n = len(trace)
        tiers = Counter()
        per_q_total, bypassed, in_gold, in_gold_deep = [], 0, 0, 0
        for t in trace:
            if not t["seeded"]:
                bypassed += 1
            ents = t["entities"]
            per_q_total.append(len(ents))
            g = gold_text.get(t["id"], "")
            for text, tier, _score in ents:
                tiers[tier] += 1
                if g and text.lower() in g:
                    in_gold += 1
                    if tier > 1:
                        in_gold_deep += 1

        total = sum(tiers.values())
        print(f"\n=== {arm}  ({path.name}) ===")
        print(f"  questions                : {n}")
        print(f"  bypassed graph search    : {bypassed} ({bypassed/max(n,1):.1%})  "
              "<- NER found no entity; pure DPR, both arms identical")
        print(f"  activated entities/question: mean "
              f"{sum(per_q_total)/max(n,1):.1f}  max {max(per_q_total, default=0)}")
        print("  tier distribution        :")
        for tier in sorted(tiers):
            label = {1: "seed", 2: "1 hop", 3: "2 hops"}.get(tier, f"{tier-1} hops")
            print(f"      tier {tier} ({label:<6}) {tiers[tier]:>7}  {tiers[tier]/max(total,1):>6.1%}")
        if gold_text:
            print(f"  activations hitting a gold passage : {in_gold} "
                  f"({in_gold/max(total,1):.1%} of activations)")
            print(f"      of which beyond the seed        : {in_gold_deep}"
                  "   <- what the intervention is for")

    print("\nIf tier 1 dominates, the search never leaves the seed neighbourhood and")
    print("no query-set change can matter. Lower --iteration_threshold to open it up.")


if __name__ == "__main__":
    main()
