"""Score sub-question sets against each other — Gate B, without LinearRAG.

This answers "does sigma_max beat sigma_q, and does the gain grow with depth?"
using the same max-over-sub-questions rule the real gate uses, but at passage
granularity and with no graph propagation:

    score[p] = max_j sim(q_j, passage_p)        vs      sim(question, passage_p)

**It is a proxy, not the method.** LinearRAG gates ~43.7k sentences and then
propagates activation through the bipartite graph; this ranks 11,656 passages
directly. A win here is necessary but not sufficient evidence for the real gate.
What it does give you, today and on a laptop, is the comparison that decides
whether to keep going: if `resolved` — the oracle — cannot beat the pooled query
on passage recall, it will not rescue the sentence-level gate either.

Every arm shares the passage embedding cache with baselines.py --method dense,
so if that has run, this pays no encode cost.

    python scripts/eval_subq.py musique
    python scripts/eval_subq.py musique --subq out/subq_musique_generated.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mbuzai import dataio, gate, metrics  # noqa: E402
from mbuzai.metrics import _recall  # noqa: E402  (shares the empty-gold convention)

# Imported rather than reimplemented: the embedding cache is keyed by dataset and
# model there, and a second convention would silently re-encode the corpus.
import baselines  # noqa: E402

OUT = ROOT / "out"


def load_subq(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        sys.exit(f"{path}: expected {{qid: [question, ...]}}")
    return data


def discover(dataset: str) -> list[Path]:
    """Every sub-question set on disk for this dataset, oracle-ish ones first."""
    order = {"raw": 0, "resolved": 1, "generated": 2}
    found = sorted(OUT.glob(f"subq_{dataset}_*.json"))
    found = [p for p in found if not p.name.endswith("_raw.jsonl")]
    return sorted(found, key=lambda p: order.get(p.stem.split("_")[-1], 99))


def rank_arm(ds, doc_emb, subq, model_name, batch_size, device, topk):
    """Rank passages per query by max similarity over {question} u sub-questions."""
    texts, spans = [], []
    for q in ds.queries:
        extra = list(subq.get(q.qid, [])) if subq else []
        start = len(texts)
        texts.extend([q.question] + extra)   # row 0 is always the original question
        spans.append((start, len(texts)))

    q_emb = baselines.embed(model_name, texts, batch_size, None, device)

    ranked = []
    kth = min(topk, len(doc_emb) - 1)
    for start, end in spans:
        scores = gate.sigma_max(doc_emb, q_emb[start:end])
        top = np.argpartition(-scores, kth)[:topk]
        ranked.append([int(p) for p in top[np.argsort(-scores[top])]])
    return ranked


def per_query_recall(ds, ranked, k):
    return [_recall(r, q.gold_pids, k) for r, q in zip(ranked, ds.queries)]


def delta_table(ds, base, arms, k):
    """Paired deltas vs the pooled-query baseline, overall and by hop count.

    Paired because the arms see identical questions: the per-question difference
    has far less variance than the difference of two independent means, and the
    4-hop buckets are too small (n=166, n=27) to see anything otherwise.
    """
    base_r = per_query_recall(ds, base, k)
    lines = [f"\ndelta vs pooled sigma_q, recall@{k}  (paired bootstrap 95% CI)"]

    buckets: dict[str, list[int]] = {"all": list(range(len(ds.queries)))}
    for i, q in enumerate(ds.queries):
        buckets.setdefault(f"{q.n_hops}hop", []).append(i)
        if q.is_join:
            buckets.setdefault("join", []).append(i)

    header = "  " + " " * 16 + "".join(f"{name:>25}" for name in arms)
    lines.append(header)
    for label in ["all"] + sorted(k for k in buckets if k != "all"):
        idxs = buckets[label]
        cells = []
        for name, ranked in arms.items():
            arm_r = per_query_recall(ds, ranked, k)
            d = [arm_r[i] - base_r[i] for i in idxs if arm_r[i] == arm_r[i]]
            if not d:
                cells.append(f"{'-':>22}")
                continue
            lo, hi = metrics.bootstrap_ci(d)
            star = "*" if lo > 0 or hi < 0 else " "
            cells.append(f"{np.mean(d):>+8.4f} [{lo:+.3f},{hi:+.3f}]{star}")
        lines.append(f"  {label:<10} n={len(idxs):<4}" + "".join(cells))
    lines.append("  * = CI excludes zero")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--subq", type=Path, nargs="*", default=None,
                    help="sub-question files. Default: every set found in out/")
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--k-report", type=int, nargs="*", default=[2, 5, 10])
    args = ap.parse_args()

    baselines.require("dense")
    device = baselines.resolve_device(args.device)
    ds = dataio.load(args.dataset)

    files = args.subq if args.subq is not None else discover(args.dataset)
    if not files:
        sys.exit(f"no sub-question files for {args.dataset} in {OUT}")

    print(f"{ds.name}: {len(ds.corpus)} passages, {len(ds.queries)} questions | device={device}")

    doc_emb = baselines.embed(
        args.model, ds.docs, args.batch_size,
        baselines.cache_path(ds.name, args.model, "docs"), device,
    )

    arms, reports = {}, {}
    base = rank_arm(ds, doc_emb, None, args.model, args.batch_size, device, args.topk)
    reports["pooled"] = metrics.evaluate(ds, base, ks=tuple(args.k_report))

    for path in files:
        subq = load_subq(path)
        name = path.stem.replace(f"subq_{args.dataset}_", "")
        covered = sum(1 for q in ds.queries if subq.get(q.qid))
        sizes = [len(subq.get(q.qid, [])) for q in ds.queries]
        print(f"\n{name:<10} {covered}/{len(ds.queries)} questions have sub-questions, "
              f"mean {np.mean(sizes):.2f} each")
        if covered < len(ds.queries):
            print(f"           {len(ds.queries) - covered} fall back to the question alone "
                  "— they dilute the delta toward zero, they do not bias it")
        arms[name] = rank_arm(ds, doc_emb, subq, args.model, args.batch_size,
                              device, args.topk)
        reports[name] = metrics.evaluate(ds, arms[name], ks=tuple(args.k_report))

    print(f"\n{'=' * 72}\nabsolute recall")
    for name, rep in reports.items():
        cells = "  ".join(f"@{k.split('@')[1]}={v['mean']:.4f}"
                          for k, v in rep["overall"].items())
        print(f"  {name:<12} {cells}")

    for k in args.k_report:
        print(delta_table(ds, base, arms, k))

    dest = OUT / f"{ds.name}_subq_eval.json"
    dest.write_text(json.dumps(reports, indent=1))
    print(f"\nwrote {dest}")
    print("passage-level proxy for the sentence-level gate — see the module docstring")


if __name__ == "__main__":
    main()
