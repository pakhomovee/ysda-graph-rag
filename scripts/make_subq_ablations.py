"""Build the two gold-derived sub-question sets. No API calls.

These are matrix rows 4 and 5:

  raw       gold sub-questions verbatim, #N placeholders intact.
            Expected to underperform the generated set — that is the point.
            It quantifies the placeholder problem instead of asserting it.

  resolved  #N substituted with the gold answer of step N. The oracle: the
            ceiling a perfect sequential decomposer could reach. If this does
            not beat pooled sigma_q, no generator will, and the direction is dead.

    python scripts/make_subq_ablations.py musique
"""

import argparse
import json
import re
import sys
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

OUT = Path(__file__).resolve().parent.parent / "out"


def resolve(question: str, answers: list[str]) -> str:
    """Replace #N with the answer of step N (1-indexed)."""
    def sub(m):
        i = int(m.group(1)) - 1
        return answers[i] if 0 <= i < len(answers) else m.group(0)

    return re.sub(r"#(\d)", sub, question)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", default="musique", nargs="?")
    args = ap.parse_args()

    ds = dataio.load(args.dataset)
    raw, resolved = {}, {}
    for q in ds.queries:
        answers = [h.answer for h in q.hops]
        raw[q.qid] = [h.question for h in q.hops]
        resolved[q.qid] = [resolve(h.question, answers) for h in q.hops]

    OUT.mkdir(exist_ok=True)
    for tag, obj in (("raw", raw), ("resolved", resolved)):
        dest = OUT / f"subq_{ds.name}_{tag}.json"
        dest.write_text(json.dumps(obj, indent=1))
        print(f"wrote {dest}  ({len(obj)} questions)")

    still = sum(1 for v in resolved.values() for s in v if re.search(r"#\d", s))
    print(f"\nunresolved placeholders remaining: {still} (should be 0)")
    sample = ds.queries[0]
    print(f"\nexample  {sample.qid}")
    for a, b in zip(raw[sample.qid], resolved[sample.qid]):
        print(f"  raw      {a}")
        print(f"  resolved {b}")


if __name__ == "__main__":
    main()
