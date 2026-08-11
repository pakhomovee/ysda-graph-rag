"""Corpus and composition statistics that size the build and set the predictions.

Reports the difficulty axis (hop count x chain/join shape), how many gold
sub-questions carry an unresolved #N placeholder, and the sentence count that
determines the encode budget for sigma_q.

    python scripts/analyze_musique.py
"""

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if sys.version_info < (3, 10):  # dataio/metrics use PEP 604 annotations
    sys.exit(
        f"\nFATAL: this script needs the mbuzai env (Python 3.10+), got "
        f"{sys.version.split()[0]}\n  interpreter: {sys.executable}\n"
        "  LinearRAG's .venv-linear is 3.9 and only run_linearrag_retrieval.py runs there.\n"
        "  Fix:  deactivate   (or point at the mbuzai interpreter directly)"
    )

from mbuzai import dataio  # noqa: E402


def main():
    ds = dataio.load("musique")
    q = ds.queries

    print(f"corpus     : {len(ds.corpus)} passages")
    words = [len(c["text"].split()) for c in ds.corpus]
    sents = sum(len(re.findall(r"[.!?]+\s", c["text"])) + 1 for c in ds.corpus)
    print(f"words      : {sum(words):,} total, {sum(words) / len(words):.0f}/passage")
    print(f"sentences  : ~{sents:,}   <- |V_s|, the sigma_q encode budget")
    print(f"queries    : {len(q)}")

    print("\nshape x hops")
    print(f"  {'shape':<10}{'n':>6}{'joins':>8}{'gold psg':>10}")
    for shape, n in sorted(Counter(x.shape for x in q).items()):
        sub = [x for x in q if x.shape == shape]
        joins = sum(x.is_join for x in sub)
        gold = sum(len(x.gold_pids) for x in sub) / len(sub)
        print(f"  {shape:<10}{n:>6}{joins:>8}{gold:>10.2f}")
    print(f"  {'TOTAL':<10}{len(q):>6}{sum(x.is_join for x in q):>8}")

    ph = [
        bool(re.search(r"#\d", h.question))
        for x in q for h in x.hops
    ]
    print(f"\nsub-questions with a #N placeholder: {sum(ph)}/{len(ph)} = {sum(ph)/len(ph):.1%}")
    has_ph = sum(any(re.search(r"#\d", h.question) for h in x.hops) for x in q)
    print(f"questions with >=1 such sub-question: {has_ph}/{len(q)}")
    print("  -> gold decompositions are unusable as query vectors; generate instead")

    unresolved = sum(1 for x in q for h in x.hops if h.gold_pid is None)
    print(f"\nhops whose gold passage failed to resolve: {unresolved} (should be 0)")

    print("\nper-hop gold passages available (the diagnostic's denominator)")
    by_hop = Counter(i for x in q for i, _ in enumerate(x.hops, 1))
    for i, n in sorted(by_hop.items()):
        print(f"  hop{i}: {n}")


if __name__ == "__main__":
    main()
