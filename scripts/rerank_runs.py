"""Rerank an existing retrieval run with a cross-encoder.

The cross-encoder is the strongest cheap non-graph baseline in this repo — at @2
it beats patched LinearRAG outright (0.4279 vs 0.3536) — so the obvious question
is whether the two compose. This reranks whatever a run retrieved instead of
retrieving again, so it applies to any arm: vanilla, sigma_max, or an oracle.

**Reranking cannot raise recall@k above the depth it is given.** Reordering a
top-10 list leaves recall@10 identical by construction; only recall@2 and @5 can
move. Run the retrieval with --retrieval_top_k 50 first, then rerank to 10.

Output arms get an `rr` suffix (vanilla -> vanillarr), so both the original and
reranked arms can be scored in one table without their names colliding.

    python scripts/rerank_runs.py musique --runs out/linearrag_musique_fine_*.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

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
from mbuzai.subq_io import match_qid  # noqa: E402

OUT = ROOT / "out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="our dataset name; supplies the passage text")
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--chunks", type=Path, default=None,
                    help="their chunks.json, if the runs index THEIR corpus rather "
                         "than ours. Omit for musique_fine, where id == our pid.")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ds = dataio.load(args.dataset)
    docs = json.loads(args.chunks.read_text()) if args.chunks else ds.docs
    by_qid = {q.qid: q for q in ds.queries}
    question_of = {}
    for q in ds.queries:
        question_of[q.qid] = q.question

    from sentence_transformers import CrossEncoder

    ce = CrossEncoder(args.model, device=args.device, max_length=384)

    for path in args.runs:
        run = json.loads(path.read_text())
        if not (isinstance(run, dict) and run
                and all(isinstance(v, list) for v in run.values())):
            print(f"skipping {path.name}: not a retrieval run dump")
            continue

        depths = [len(v) for v in run.values()]
        if max(depths, default=0) <= args.topk:
            print(f"WARNING {path.name}: depth {max(depths, default=0)} <= topk "
                  f"{args.topk} — reranking can only reorder, recall@{args.topk} "
                  "will be unchanged. Re-run retrieval with --retrieval_top_k 50.")

        out = {}
        for tid, ids in run.items():
            qid = match_qid(tid, by_qid)
            question = question_of.get(qid)
            if question is None or not ids:
                out[tid] = ids[: args.topk]
                continue
            scores = ce.predict([(question, docs[i]) for i in ids],
                                batch_size=args.batch_size, show_progress_bar=False)
            order = np.argsort(-np.asarray(scores))[: args.topk]
            out[tid] = [int(ids[i]) for i in order]

        stem = path.stem
        dest = path.with_name(f"{stem}rr.json")
        dest.write_text(json.dumps(out, indent=1))
        print(f"wrote {dest.name}  (from {path.name}, depth {max(depths, default=0)} "
              f"-> {args.topk})")

    print("\nScore originals and reranked together — the rr suffix keeps the arm "
          "names distinct:\n  python scripts/score_linearrag.py "
          f"{args.dataset} --gold <gold> --runs <prefix>_*.json")


if __name__ == "__main__":
    main()
