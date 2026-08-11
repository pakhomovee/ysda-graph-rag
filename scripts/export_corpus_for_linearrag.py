"""Give LinearRAG our corpus, at passage granularity, instead of their chunks.

Their bundle re-chunks the source into ~1000-token blocks: 1,354 chunks of ~820
words where the benchmark's own unit is the ~80-word paragraph. Three consequences
make it a poor instrument for a retrieval-metric study:

  * 18.4% of MuSiQue questions have all their gold inside ONE chunk (31% of
    2-hop), so one retrieval returns everything and no multi-hop problem remains.
  * an activated entity occurs in many 820-word chunks, so the entity-occurrence
    bonus that carries a_e into passage scores barely discriminates.
  * recall@k is not comparable to baselines.py or eval_subq.py, which score over
    our 11,656 passages.

`LinearRAG.index()` takes a list of strings, so it can index ours directly. Chunk
index then *is* our pid, which also makes gold exact — no substring probing, no
overlap ambiguity, no unlocatable passages.

Passage text is `title\\ntext`, matching `dataio.Dataset.docs`, so the units are
identical to every other measurement in this repo. Titles are entity-dense and
their pipeline runs NER over whatever it is given.

    python scripts/export_corpus_for_linearrag.py musique
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mbuzai import dataio  # noqa: E402

OUT = ROOT / "out"
SUB = ROOT / "third_party" / "LinearRAG"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="our dataset name, e.g. musique")
    ap.add_argument("--name", default=None,
                    help="name for the exported set (default: <dataset>_fine)")
    ap.add_argument("--dest", type=Path, default=None,
                    help="LinearRAG dataset dir (default: third_party/LinearRAG/dataset)")
    args = ap.parse_args()

    ds = dataio.load(args.dataset)
    name = args.name or f"{args.dataset}_fine"
    dest = (args.dest or SUB / "dataset") / name
    dest.mkdir(parents=True, exist_ok=True)

    # Order is load order, so chunk index == pid. Everything downstream depends
    # on that, which is the whole point of exporting rather than mapping.
    (dest / "chunks.json").write_text(json.dumps(ds.docs, ensure_ascii=False))

    questions = [
        {"id": q.qid, "question": q.question, "answer": q.answer,
         "source": args.dataset, "question_type": q.shape, "evidence": ""}
        for q in ds.queries
    ]
    (dest / "questions.json").write_text(json.dumps(questions, ensure_ascii=False))

    # Exact gold: one entry per gold passage, each a single-element option list
    # to match the format prepare_linearrag_gold.py emits for overlapping chunks.
    gold = {q.qid: [[p] for p in sorted(q.gold_pids)] for q in ds.queries if q.gold_pids}
    gold_dest = OUT / f"linearrag_gold_{name}.json"
    gold_dest.write_text(json.dumps(gold, indent=1))

    words = sum(len(d.split()) for d in ds.docs)
    print(f"{name}: {len(ds.docs)} passages, ~{words // len(ds.docs)} words each")
    print(f"  wrote {dest}/chunks.json")
    print(f"  wrote {dest}/questions.json  ({len(questions)} questions)")
    print(f"  wrote {gold_dest}  ({len(gold)} with gold, "
          f"mean {sum(len(v) for v in gold.values())/max(len(gold),1):.2f} passages each)")
    print("\nGold is exact here — chunk index IS our pid. No substring probing, no")
    print("overlap, nothing unlocatable, and recall@k is finally in the same units")
    print("as baselines.py and eval_subq.py.")
    print(f"""
Next (the index is built from scratch for this granularity — NER over
{len(ds.docs)} passages, so expect it to take a while on the first run):

  cd third_party/LinearRAG
  PYTHONPATH={ROOT}:$PYTHONPATH .venv-linear/bin/python \\
      {ROOT}/scripts/run_linearrag_retrieval.py \\
      --dataset_name {name} --device 3 --retrieval_top_k 10 \\
      --embedding_model sentence-transformers/all-mpnet-base-v2 \\
      --subq_file {OUT}/subq_{args.dataset}_generated_bytext.json \\
      --out {OUT}/linearrag_{name}

  cd - && python3 scripts/score_linearrag.py {args.dataset} \\
      --gold {gold_dest} --runs {OUT}/linearrag_{name}_*.json""")


if __name__ == "__main__":
    main()
