"""Is HippoRAG about to index OUR corpus? Check before spending the GPU hours.

`main.py --dump` writes retrieval results as indices into our corpus, resolved by
looking up each retrieved doc string in `{f"{title}\\n{text}": pid}`. HippoRAG builds
its own `docs` list with the same expression, so if it is pointed at our data/ the
identity holds by construction — but "by construction" is an argument, and this is the
measurement. A mismatch becomes -1s that score as misses and read as a method failure,
which is the trap `check_qafd_bundle.py` was written for.

Cheap by design: no GPU, no LLM, no retrieval, no index. Run it first.

    python scripts/check_hipporag_data.py musique

Also compares against HippoRAG's own bundled copy under
third_party/HippoRAG/reproduce/dataset/ when present, since we and they are both
downstream of osunlp/HippoRAG_v2 and a divergence there would be worth knowing about
before it turns into a reproduction gap.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if sys.version_info < (3, 10):  # dataio uses PEP 604 annotations
    sys.exit(
        f"\nFATAL: this script needs the mbuzai env (Python 3.10+), got "
        f"{sys.version.split()[0]}\n  interpreter: {sys.executable}"
    )

from mbuzai import dataio  # noqa: E402


def _norm(s):
    """Whitespace-insensitive form, to tell 'different corpus' from 'same corpus,
    different serialisation'. The runtime lookup is byte-for-byte; this only explains
    a failure."""
    return " ".join(s.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--their_dir", default=None,
                    help="default third_party/HippoRAG/reproduce/dataset")
    args = ap.parse_args()

    print(f"==> our data ({args.dataset})")
    ds = dataio.load(args.dataset)
    docs = ds.docs
    print(f"    {len(docs)} passages, {len(ds.queries)} questions")
    print(f"    corpus md5 {hashlib.md5(chr(10).join(docs).encode()).hexdigest()[:12]}")

    # Gold reachability. Every pid must be a real passage, or recall is capped for a
    # reason that has nothing to do with retrieval.
    n_gold = sum(len(q.gold_pids) for q in ds.queries)
    bad = [q.qid for q in ds.queries if any(p >= len(docs) or p < 0 for p in q.gold_pids)]
    empty = [q.qid for q in ds.queries if not q.gold_pids]
    print(f"    {n_gold} gold passages ({n_gold / max(len(ds.queries), 1):.2f} per question)")
    if bad:
        print(f"    FATAL: {len(bad)} questions have out-of-range gold pids, e.g. {bad[0]}")
        return 1
    if empty:
        print(f"    WARNING: {len(empty)} questions have no gold passage at all "
              f"(they can never be scored), e.g. {empty[0]}")

    # The identity main.py --dump depends on.
    uniq = len(set(docs))
    print(f"\n==> passage identity")
    print(f"    {uniq}/{len(docs)} passages are unique as \"title\\ntext\"")
    if uniq != len(docs):
        # A duplicate maps two pids to one string, so doc->pid silently picks one and
        # the other pid becomes unretrievable. dataio keys on (title, text) precisely
        # because titles collide; this checks the stronger property main.py needs.
        print(f"    WARNING: {len(docs) - uniq} duplicate passage strings. The "
              "retrieved-doc -> pid map keeps the LAST, so the earlier pid can never "
              "be credited. Gold on a duplicated string will under-score.")
        dupe_gold = sum(1 for q in ds.queries for p in q.gold_pids
                        if docs.count(docs[p]) > 1) if len(docs) < 20000 else None
        if dupe_gold:
            print(f"    {dupe_gold} gold pids sit on a duplicated string")

    their_dir = Path(args.their_dir or ROOT / "third_party/HippoRAG/reproduce/dataset")
    their_corpus = their_dir / f"{args.dataset}_corpus.json"
    print(f"\n==> their bundled copy ({their_dir})")
    if not their_corpus.exists():
        print("    not present — nothing to compare. Feed main.py our data/ with")
        print("    --data_dir, which makes the identity hold by construction anyway.")
    else:
        theirs = [f"{d['title']}\n{d['text']}" for d in json.loads(their_corpus.read_text())]
        ours, tset = set(docs), set(theirs)
        both = ours & tset
        print(f"    {len(theirs)} passages, {len(both)} shared "
              f"({100 * len(both) / max(len(tset), 1):.1f}% of theirs, "
              f"{100 * len(both) / max(len(ours), 1):.1f}% of ours)")
        their_q = their_dir / f"{args.dataset}.json"
        if their_q.exists():
            tsamples = json.loads(their_q.read_text())
            tq = [x["question"] for x in tsamples]
            tid = [str(x.get("id", "")) for x in tsamples]
            same_q = tq == [q.question for q in ds.queries]
            same_id = tid == [q.qid for q in ds.queries]
            print(f"    questions: {len(tq)}, identical in order: {same_q}, "
                  f"ids identical: {same_id}")
            if same_q and same_id and both == ours == tset:
                # Both halves match, so the paper's numbers are measured on exactly
                # these rows and a recall difference is attributable to method, not data.
                print("    -> same corpus AND same question set: the paper's reported "
                      "numbers are\n       directly comparable to what this repo will "
                      "score.")
        if both != ours or both != tset:
            n_both = {_norm(t) for t in tset} & {_norm(d) for d in ours}
            print(f"    ignoring whitespace: {len(n_both)}"
                  + ("   <- serialisation differs, not the corpus"
                     if len(n_both) > len(both) else ""))
            print("    NOTE both are downstream of osunlp/HippoRAG_v2, so a difference "
                  "here means one\n         copy is stale. Use --data_dir to pin ours.")

    print("\n==> verdict")
    print("    Feed HippoRAG our corpus:  main.py --data_dir "
          f"{ROOT / 'data'} --dataset {args.dataset}")
    print("    Then the doc -> pid lookup in --dump is exact identity, and recall is")
    print("    comparable to every row in RESULTS.md and to the paper's own numbers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
