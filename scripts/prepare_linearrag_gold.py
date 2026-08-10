"""Map our gold passages onto LinearRAG's chunks, so retrieval can be scored.

Their bundle carries no gold labels — `questions.json` has an `evidence` field and
it is the empty string on every row — so recall has to come from our copy of the
dataset. The two do not share a corpus:

    ours   11,656 passages   ~80 words each
    theirs  1,354 chunks    ~820 words each

They concatenated the source passages into ~1000-token chunks, so one of their
chunks swallows roughly ten of ours. **Recall@k on their corpus is therefore not
comparable to recall@k on ours** — retrieving 5 chunks is retrieving ~50
passages. Vanilla vs sigma_max inside LinearRAG stays a valid paired comparison,
because both arms see the identical corpus; putting either number in the same
table as `baselines.py` does not.

Their question ids need no text join, but the bundle is inconsistent about them:
MuSiQue rows carry a source prefix (`musique_2hop__13548_13529`) while 2Wiki rows
are the bare hex id we already use. `mbuzai.subq_io.match_qid` handles both;
splitting on "_" unconditionally raises IndexError on 2Wiki.

2Wiki's rows *do* populate `evidence` (as triples) where MuSiQue's are empty, but
gold still has to come from our copy either way — triples are not passage ids.

    python scripts/prepare_linearrag_gold.py musique \
        --bundle third_party/LinearRAG/dataset/musique
"""

import argparse
import bisect
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mbuzai import dataio  # noqa: E402
from mbuzai.subq_io import match_qid  # noqa: E402

OUT = ROOT / "out"
_NORM = re.compile(r"[^a-z0-9]+")
SEP = " \x00 "


def norm(text: str) -> str:
    return _NORM.sub(" ", text.lower()).strip()


def build_index(chunks):
    """One big normalised string plus chunk-start offsets, so locating a passage
    is a single find() rather than a scan over 1,354 chunks."""
    parts, starts, pos = [], [], 0
    for c in chunks:
        n = norm(c)
        starts.append(pos)
        parts.append(n)
        pos += len(n) + len(SEP)
    return SEP.join(parts), starts


def locate(joined, starts, text):
    """Chunk index containing `text`, or None. Shrinks the probe before giving
    up: their text went through a tokeniser (`( rfef )`), so a long probe can
    miss on spacing that a shorter one survives."""
    n = norm(text)
    for width in (90, 60, 40):
        if len(n) < width:
            continue
        at = joined.find(n[:width])
        if at != -1:
            return bisect.bisect_right(starts, at) - 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="our dataset name, e.g. musique")
    ap.add_argument("--bundle", type=Path, required=True,
                    help="their dataset dir holding chunks.json / questions.json")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    chunks = json.loads((args.bundle / "chunks.json").read_text())
    theirq = json.loads((args.bundle / "questions.json").read_text())
    ds = dataio.load(args.dataset)

    joined, starts = build_index(chunks)
    pid_to_chunk, unmatched = {}, 0
    for pid in sorted({p for q in ds.queries for p in q.gold_pids}):
        c = locate(joined, starts, ds.corpus[pid]["text"])
        if c is None:
            unmatched += 1
        else:
            pid_to_chunk[pid] = c

    by_qid = {q.qid: q for q in ds.queries}
    gold, missing_q = {}, 0
    for e in theirq:
        qid = match_qid(e["id"], by_qid)
        q = by_qid.get(qid) if qid else None
        if q is None:
            missing_q += 1
            continue
        cids = sorted({pid_to_chunk[p] for p in q.gold_pids if p in pid_to_chunk})
        if cids:
            gold[e["id"]] = cids

    dest = args.out or OUT / f"linearrag_gold_{args.dataset}.json"
    dest.write_text(json.dumps(gold, indent=1))

    sizes = [len(v) for v in gold.values()]
    npids = [len(by_qid[match_qid(k, by_qid)].gold_pids) for k in gold]
    print(f"chunks {len(chunks)} | their questions {len(theirq)} | ours {len(ds.queries)}")
    print(f"  gold passages located in a chunk : "
          f"{len(pid_to_chunk)}/{len(pid_to_chunk) + unmatched}")
    print(f"  questions with gold chunks       : {len(gold)}/{len(theirq)}"
          + (f"  ({missing_q} qid misses)" if missing_q else ""))
    print(f"  gold chunks per question         : mean {sum(sizes)/max(len(sizes),1):.2f}"
          f"  (from mean {sum(npids)/max(len(npids),1):.2f} gold passages)")
    print("\nNOTE: several gold passages collapse into one chunk, so recall here is")
    print("      mechanically easier than on our 11,656-passage corpus. Compare")
    print("      vanilla vs sigma_max, never these numbers against baselines.py.")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
