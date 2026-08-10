"""Non-graph retrieval floor: BM25, dense, and reciprocal-rank fusion.

This is matrix row 1. A graph method that does not clear a tuned hybrid baseline
has not shown anything, and this is the comparison reviewers reach for first.

    python scripts/baselines.py musique --method bm25
    python scripts/baselines.py musique --method dense --batch-size 32
    python scripts/baselines.py musique --method hybrid
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mbuzai import dataio, metrics  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out"

# import name -> pip name, per method. Checked up front; see require().
DEPS = {
    "bm25": [("bm25s", "bm25s"), ("Stemmer", "PyStemmer")],
    "dense": [("faiss", "faiss-cpu"), ("sentence_transformers", "sentence-transformers")],
}
DEPS["hybrid"] = DEPS["bm25"] + DEPS["dense"]


def require(method: str) -> None:
    """Check every import a method needs before touching the data.

    Missing packages here almost always mean the wrong environment rather than a
    broken install: the generation venv carries numpy and a CUDA torch, so the
    header prints `device=cuda` quite happily and then dies on the first
    retrieval import — one module per run, three runs to learn the same thing.
    """
    import importlib.util

    missing = [pkg for mod, pkg in DEPS[method] if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    sys.exit(
        f"\nFATAL: --method {method} needs {', '.join(missing)}, not importable here.\n"
        f"  interpreter: {sys.executable}\n"
        "  This is the analysis env's job, not .venv-gen (vllm) or LinearRAG's 3.9.\n"
        "  Fix:  . .venv/bin/activate && pip install -r requirements.txt\n"
        f"  Or:   pip install {' '.join(missing)}"
    )


def rank_bm25(docs, questions, topk):
    import bm25s
    import Stemmer

    stemmer = Stemmer.Stemmer("english")
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(docs, stopwords="en", stemmer=stemmer, show_progress=False))
    idx, _ = retriever.retrieve(
        bm25s.tokenize(questions, stopwords="en", stemmer=stemmer, show_progress=False),
        k=topk,
        show_progress=False,
    )
    return idx


def resolve_device(spec: str) -> str:
    if spec != "auto":
        return spec
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def cache_path(name: str, model_name: str, kind: str) -> Path:
    """Cache key includes the model — otherwise switching encoders silently
    reuses the previous model's vectors."""
    slug = model_name.rstrip("/").split("/")[-1]
    return OUT / f"emb_{name}_{slug}_{kind}.npy"


def embed(model_name, texts, batch_size, cache: Path | None, device: str):
    if cache and cache.exists():
        print(f"  cached {cache.name}")
        return np.load(cache)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, emb)
    return emb


def rank_dense(docs, questions, topk, model_name, batch_size, name, device):
    import faiss

    doc_emb = embed(model_name, docs, batch_size, cache_path(name, model_name, "docs"), device)
    q_emb = embed(model_name, questions, batch_size, cache_path(name, model_name, "queries"), device)
    index = faiss.IndexFlatIP(doc_emb.shape[1])
    index.add(doc_emb)
    _, idx = index.search(q_emb, topk)
    return idx


def rrf(rankings: list[np.ndarray], topk: int, k: int = 60):
    """Reciprocal rank fusion — no score calibration needed between retrievers."""
    fused = []
    for row_set in zip(*rankings):
        scores: dict[int, float] = {}
        for row in row_set:
            for rank, pid in enumerate(row):
                scores[int(pid)] = scores.get(int(pid), 0.0) + 1.0 / (k + rank + 1)
        fused.append([p for p, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:topk]])
    return np.array(fused)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--method", choices=["bm25", "dense", "hybrid"], default="bm25")
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto",
                    help="cuda / cpu / auto. Dense encoding only; BM25 is CPU either way.")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    require(args.method)
    device = resolve_device(args.device)
    ds = dataio.load(args.dataset)
    if args.limit:
        ds.queries = ds.queries[: args.limit]
    docs = ds.docs
    questions = [q.question for q in ds.queries]
    print(f"{ds.name}: {len(docs)} passages, {len(questions)} questions "
          f"| method={args.method} device={device}", flush=True)

    t0 = time.time()
    if args.method == "bm25":
        ranked = rank_bm25(docs, questions, args.topk)
    elif args.method == "dense":
        ranked = rank_dense(docs, questions, args.topk, args.model,
                            args.batch_size, ds.name, device)
    else:
        ranked = rrf(
            [
                rank_bm25(docs, questions, args.topk * 3),
                rank_dense(docs, questions, args.topk * 3, args.model,
                           args.batch_size, ds.name, device),
            ],
            args.topk,
        )
    elapsed = time.time() - t0

    ranked = [[int(p) for p in row] for row in ranked]
    report = metrics.evaluate(ds, ranked)
    report["method"] = args.method
    report["seconds"] = round(elapsed, 1)
    print(metrics.render(report, f"{ds.name} / {args.method} ({elapsed:.1f}s)"))

    OUT.mkdir(exist_ok=True)
    dest = OUT / f"{ds.name}_{args.method}.json"
    dest.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
