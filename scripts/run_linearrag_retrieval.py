"""Run patched LinearRAG retrieval for both arms, and dump what was retrieved.

Runs inside LinearRAG's Python 3.9 environment, from the LinearRAG checkout (its
code uses relative paths for `dataset/` and `./import`). Retrieval only — the LLM
answer stage is skipped entirely, so no API key is needed and nothing sits between
the intervention and the metric.

Both arms run in one process against one index. Indexing is the expensive part and
sharing it means the arms cannot differ by anything except the query set — no
re-chunking, no re-embedding, no NER rerun. The sigma_max arm is switched on by
assigning `rag_model.query_sets`, which is exactly what the patched `retrieve()`
reads.

Scoring deliberately happens elsewhere: `mbuzai.metrics` imports `dataio`, whose
PEP 604 annotations do not survive 3.9. This writes raw retrieved chunk ids and
`scripts/score_linearrag.py` turns them into recall with paired CIs.

    cd third_party/LinearRAG
    export PYTHONPATH=/path/to/mbuzai-rag:$PYTHONPATH
    python /path/to/mbuzai-rag/scripts/run_linearrag_retrieval.py \
        --dataset_name musique \
        --subq_file /path/to/mbuzai-rag/out/subq_musique_generated_bytext.json \
        --out /path/to/mbuzai-rag/out/linearrag_musique
"""

import argparse
import json
import os
import sys


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", required=True, help="their name, e.g. musique / 2wikimultihop")
    ap.add_argument("--subq_file", default=None, help="exported *_bytext.json for the sigma_max arm")
    ap.add_argument("--embedding_model", default="model/all-mpnet-base-v2")
    ap.add_argument("--spacy_model", default="en_core_web_trf")
    ap.add_argument("--retrieval_top_k", type=int, default=10,
                    help="must be >= the largest k you intend to report")
    ap.add_argument("--top_k_sentence", type=int, default=3)
    ap.add_argument("--max_iterations", type=int, default=3)
    ap.add_argument("--iteration_threshold", type=float, default=0.4)
    ap.add_argument("--passage_ratio", type=float, default=2)
    ap.add_argument("--use_vectorized_retrieval", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None, help="sets CUDA_VISIBLE_DEVICES")
    ap.add_argument("--out", required=True, help="output prefix; _vanilla.json / _subq.json")
    return ap.parse_args()


def chunk_id(passage_text):
    """Recover the chunk index from a retrieved passage.

    run.py builds passages as f'{idx}:{chunk}', and the bundled chunks already
    carry their own '0:' prefix, so the stored text reads '0:0:...'. We replicate
    run.py exactly rather than 'fixing' it — their published numbers come from
    that pipeline — and the leading integer is the enumerate index either way.
    """
    head = passage_text.split(":", 1)[0]
    return int(head) if head.isdigit() else None


def main():
    args = parse_args()
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # Their run.py hardcodes CUDA_VISIBLE_DEVICES=4 at import time; we set ours
    # first so importing their modules cannot silently reassign the GPU.
    from sentence_transformers import SentenceTransformer

    sys.path.insert(0, os.getcwd())
    from src.config import LinearRAGConfig
    from src.LinearRAG import LinearRAG
    from mbuzai.subq_io import load_query_sets, lookup

    with open(f"dataset/{args.dataset_name}/questions.json", encoding="utf-8") as fh:
        questions = json.load(fh)
    with open(f"dataset/{args.dataset_name}/chunks.json", encoding="utf-8") as fh:
        chunks = json.load(fh)
    if args.limit:
        questions = questions[: args.limit]
    passages = ["%d:%s" % (idx, chunk) for idx, chunk in enumerate(chunks)]

    print("%s: %d chunks, %d questions" % (args.dataset_name, len(chunks), len(questions)))

    config = LinearRAGConfig(
        dataset_name=args.dataset_name,
        embedding_model=SentenceTransformer(args.embedding_model, device="cuda"),
        llm_model=None,                     # retrieval only; retrieve() never touches it
        spacy_model=args.spacy_model,
        retrieval_top_k=args.retrieval_top_k,
        top_k_sentence=args.top_k_sentence,
        max_iterations=args.max_iterations,
        iteration_threshold=args.iteration_threshold,
        passage_ratio=args.passage_ratio,
        use_vectorized_retrieval=args.use_vectorized_retrieval,
        subq_file=None,                     # arm 1; switched below for arm 2
    )
    rag = LinearRAG(global_config=config)
    rag.index(passages)

    arms = [("vanilla", None)]
    if args.subq_file:
        arms.append(("subq", load_query_sets(args.subq_file)))

    for name, query_sets in arms:
        rag.query_sets = query_sets
        print("\n=== arm: %s ===" % name)
        if query_sets is not None:
            # A silent join failure and a real null result look identical in the
            # final table, so refuse to produce the ambiguous one.
            hits = sum(1 for q in questions if lookup(query_sets, q["question"]))
            print("query sets matched: %d/%d questions" % (hits, len(questions)))
            if hits == 0:
                sys.exit("FATAL: no question matched a query set — the text join is "
                         "broken, and this arm would be vanilla wearing a different name")
            if hits < len(questions) // 2:
                print("WARNING: under half matched; the delta is diluted toward zero")
        results = rag.retrieve(questions)
        out = {}
        for q, r in zip(questions, results):
            ids = [chunk_id(p) for p in r["sorted_passage"]]
            out[q["id"]] = [i for i in ids if i is not None]
        dest = "%s_%s.json" % (args.out, name)
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        empty = sum(1 for v in out.values() if not v)
        print("wrote %s  (%d questions, %d with nothing retrieved)" % (dest, len(out), empty))

    print("\nnow score both arms in the mbuzai env:")
    print("  python scripts/score_linearrag.py <our-dataset-name> --runs %s_*.json" % args.out)


if __name__ == "__main__":
    main()
