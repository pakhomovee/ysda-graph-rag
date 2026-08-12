"""Why is QAFD's own dense retrieval 0.09 worse than ours on the same corpus?

QAFD's internal dense passage retrieval scores 0.4833 recall@10 where
scripts/baselines.py --method dense scores 0.5733 -- same passages, same encoder,
same questions, same gold. That gap is larger than anything the flow diffusion
does, and it sits underneath every QAFD number we have, so it has to be explained
before any of them mean anything.

Their retrieval is textbook (`retriever._dense_passage_retrieval`): normalised
dot product against the stored chunk embeddings, full ranking. So the difference
has to be in what got stored. This checks, in order:

  1. do the indexed chunk texts match our corpus strings exactly?
  2. do the stored embeddings match what the encoder produces for those strings?
  3. what recall do the stored embeddings give, versus freshly encoded ones?

Whichever of those breaks localises the problem to indexing, to the encoder
configuration, or to neither -- in which case it is the query side.

    python scripts/diagnose_dpr.py musique \\
        --index third_party/QAFD-RAG/outputs/musique/openai_gpt-oss-20b_mpnet
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if sys.version_info < (3, 10):
    sys.exit(f"needs the mbuzai env (3.10+), got {sys.version.split()[0]}")

from mbuzai import dataio, metrics  # noqa: E402

EMB_MODEL = "sentence-transformers/all-mpnet-base-v2"


def read_parquet(path):
    """pandas if present, else pyarrow — neither is a declared dependency here."""
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        return list(df["hash_id"]), list(df["content"]), list(df["embedding"])
    except ImportError:
        import pyarrow.parquet as pq
        t = pq.read_table(path)
        return (t["hash_id"].to_pylist(), t["content"].to_pylist(),
                t["embedding"].to_pylist())


def recall_at(ranked, gold, k):
    return len(set(gold) & set(ranked[:k])) / len(gold) if gold else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--index", type=Path, required=True,
                    help="QAFD working_dir holding chunk_embeddings/vdb_chunk.parquet")
    ap.add_argument("--sample-vectors", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ks", type=int, nargs="*", default=[2, 10, 50])
    args = ap.parse_args()

    pq_path = args.index / "chunk_embeddings" / "vdb_chunk.parquet"
    if not pq_path.exists():
        sys.exit(f"missing {pq_path}")
    ids, contents, embs = read_parquet(pq_path)
    stored = np.asarray([np.asarray(e, dtype=np.float32) for e in embs])
    print(f"index: {len(ids)} chunks, dim {stored.shape[1]}")

    ds = dataio.load(args.dataset)
    docs = ds.docs
    print(f"corpus: {len(docs)} passages")

    # ---- 1. do the indexed texts match our corpus strings? ----------------
    doc_to_pid = {d: i for i, d in enumerate(docs)}
    pid_of = [doc_to_pid.get(c, -1) for c in contents]
    matched = sum(1 for p in pid_of if p >= 0)
    print(f"\n1. text match: {matched}/{len(contents)} indexed chunks are byte-identical "
          f"to a corpus passage ({matched/len(contents):.1%})")
    if matched < len(contents):
        bad = next(c for c, p in zip(contents, pid_of) if p < 0)
        print(f"   first non-matching chunk starts: {bad[:120]!r}")
        print("   -> the index was built over different text; nothing below is comparable")

    # ---- 2. do the stored vectors match what the encoder produces? --------
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMB_MODEL, device=args.device)

    norms = np.linalg.norm(stored, axis=1)
    print(f"\n2. stored vectors: norm mean {norms.mean():.4f}, "
          f"min {norms.min():.4f}, max {norms.max():.4f}  "
          f"({'normalised' if abs(norms.mean() - 1) < 0.01 else 'NOT normalised'})")

    rng = np.random.default_rng(0)
    idx = rng.choice(len(contents), min(args.sample_vectors, len(contents)), replace=False)
    fresh = np.asarray(model.encode([contents[i] for i in idx], normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False))
    ref = stored[idx] / (np.linalg.norm(stored[idx], axis=1, keepdims=True) + 1e-12)
    agree = np.sum(fresh * ref, axis=1)
    print(f"   cosine(stored, re-encoded) over {len(idx)} sampled chunks: "
          f"mean {agree.mean():.4f}, min {agree.min():.4f}")
    if agree.mean() < 0.99:
        print("   -> the stored vectors are NOT what this encoder produces for this text.\n"
              "      The index was built with a different model, or different pooling.")

    # ---- 3. recall from stored vs freshly encoded passages ----------------
    q_emb = np.asarray(model.encode([q.question for q in ds.queries],
                                    normalize_embeddings=True, convert_to_numpy=True,
                                    show_progress_bar=False))
    order = np.argsort([p if p >= 0 else 10**9 for p in pid_of])
    keep = [i for i in order if pid_of[i] >= 0]
    A = stored[keep] / (np.linalg.norm(stored[keep], axis=1, keepdims=True) + 1e-12)
    pids = np.array([pid_of[i] for i in keep])

    print("\n3. recall using QAFD's stored passage vectors (query encoded here)")
    res = {}
    for k in args.ks:
        vals = []
        for qi, q in enumerate(ds.queries):
            top = pids[np.argsort(-(A @ q_emb[qi]))[:k]]
            vals.append(recall_at(list(top), q.gold_pids, k))
        res[k] = float(np.mean(vals))
        lo, hi = metrics.bootstrap_ci(vals)
        print(f"   recall@{k:<3} {res[k]:.4f}  [{lo:.3f},{hi:.3f}]")

    print("""
how to read this:
  text match < 100%            -> the index covers different passages than we score
  cosine(stored, re-encoded) low -> stored with a different encoder or pooling
  both fine, recall here ~0.57 -> the vectors are good and the loss is on the
      query side or in their scoring path
  both fine, recall here ~0.48 -> the stored vectors themselves retrieve worse,
      despite matching text and encoder. Compare against baselines.py directly.""")


if __name__ == "__main__":
    main()
