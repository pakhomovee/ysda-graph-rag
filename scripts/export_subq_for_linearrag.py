"""Re-key a sub-question set by question text, for consumption inside LinearRAG.

LinearRAG runs on Python 3.9 against its own dataset bundle (`Zly0523/linear-rag`,
where MuSiQue's sibling is named `2wikimultihop`, not `2wikimultihopqa`). It has
no access to our qids and cannot import `mbuzai.dataio`. So the qid -> question
join happens here, in our environment, and the patched retriever only ever does a
dictionary lookup on a normalised string.

Collisions are reported rather than silently overwritten: two questions that
normalise to the same key would otherwise share a sub-question set, and on a
paired comparison that is indistinguishable from a real effect.

    python scripts/export_subq_for_linearrag.py musique \
        --subq out/subq_musique_generated.json
"""

import argparse
import json
import sys
from collections import defaultdict
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

from mbuzai import dataio  # noqa: E402
from mbuzai.subq_io import normalize_question  # noqa: E402

OUT = ROOT / "out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--subq", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ds = dataio.load(args.dataset)
    subq = json.loads(args.subq.read_text())

    by_key = {}
    collisions = defaultdict(list)
    missing = 0
    for q in ds.queries:
        sets = subq.get(q.qid)
        if not sets:
            missing += 1
            continue
        key = normalize_question(q.question)
        if key in by_key and by_key[key] != sets:
            collisions[key].append(q.qid)
        by_key[key] = sets

    dest = args.out or OUT / f"{args.subq.stem}_bytext.json"
    dest.write_text(json.dumps(by_key, indent=1, ensure_ascii=False))

    print(f"{args.subq.name}: {len(ds.queries)} questions")
    print(f"  exported          : {len(by_key)} keys")
    print(f"  no sub-questions  : {missing} (these fall back to vanilla, by design)")
    if collisions:
        print(f"  KEY COLLISIONS    : {len(collisions)} — two questions share a "
              "normalised key with different sub-questions")
        for key, qids in list(collisions.items())[:3]:
            print(f"      {key[:60]!r} <- {qids}")
    print(f"\nwrote {dest}")
    print("point LinearRAG at this with --subq_file")


if __name__ == "__main__":
    main()
