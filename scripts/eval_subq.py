"""Score sub-question sets against each other — Gate B, without LinearRAG.

This answers "does sigma_max beat sigma_q, and does the gain grow with depth?"
using the same max-over-sub-questions rule the real gate uses, but at passage
granularity and with no graph propagation:

    score[p] = max_j sim(q_j, passage_p)        vs      sim(question, passage_p)

**It is a proxy, not the method.** LinearRAG gates sentences, iterates a top-k
selection per entity, and ranks passages by personalised PageRank over the
result; this ranks passages directly by the gate. A win here is necessary but not
sufficient evidence for the real gate, and the missing propagation is exactly
what would carry signal past hop 1 — so this understates the method where the
method is supposed to work.
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


def _topk(scores, topk):
    kth = min(topk, len(scores) - 1)
    top = np.argpartition(-scores, kth)[:topk]
    return [int(p) for p in top[np.argsort(-scores[top])]]


def rank_arm(ds, doc_emb, subq, model_name, batch_size, device, topk, fusion="max"):
    """Rank passages per query by combining {question} u sub-questions.

    `max` is the intervention: sigma_max, the same rule the LinearRAG gate uses.
    The others exist because "does the gate beat simply decomposing and fusing?"
    is the first thing anyone will ask, and without them sigma_max is only ever
    compared against the pooled question — a control it beats trivially.

      max   score[p] = max_j sim(q_j, p)
      mean  score[p] = mean_j sim(q_j, p)
      rrf   rank per sub-question independently, fuse by reciprocal rank

    `rrf` is the standard decompose-and-fuse baseline and needs no gate, no graph
    and no scores on a common scale.
    """
    texts, spans = [], []
    for q in ds.queries:
        extra = list(subq.get(q.qid, [])) if subq else []
        start = len(texts)
        texts.extend([q.question] + extra)   # row 0 is always the original question
        spans.append((start, len(texts)))

    q_emb = baselines.embed(model_name, texts, batch_size, None, device)

    ranked = []
    for start, end in spans:
        Q = q_emb[start:end]
        if fusion == "max":
            ranked.append(_topk(gate.sigma_max(doc_emb, Q), topk))
        elif fusion == "mean":
            ranked.append(_topk((doc_emb @ Q.T).mean(axis=1), topk))
        elif fusion == "rrf":
            # baselines.rrf so the fusion constant and tie-breaking match the
            # hybrid baseline exactly rather than being reimplemented here.
            rows = [np.array(_topk(doc_emb @ Q[j], topk * 3)).reshape(1, -1)
                    for j in range(Q.shape[0])]
            ranked.append([int(p) for p in baselines.rrf(rows, topk)[0]])
        else:
            sys.exit(f"unknown fusion {fusion!r}")
    return ranked


def per_query_recall(ds, ranked, k):
    return [_recall(r, q.gold_pids, k) for r, q in zip(ranked, ds.queries)]


def delta_table(ds, base, arms, covered, k):
    """Paired deltas vs the pooled-query baseline, split three ways.

    Paired because the arms see identical questions: the per-question difference
    has far less variance than the difference of two independent means, and the
    small buckets (MuSiQue 4hop2 n=27) show nothing otherwise.

    Depth and join-ness are crossed rather than reported separately. A join
    question belongs to both, so a flat `join` row is compared against hop rows
    that contain those very questions — which cannot test "joins beat chains of
    equal depth". `{n}hop-chain` vs `{n}hop-join` can.

    The shape rows are the axis 2Wiki turns on (comparison / bridge_comparison /
    compositional / inference); on MuSiQue they recover the hop-shape labels.
    """
    base_r = per_query_recall(ds, base, k)
    arm_r = {name: per_query_recall(ds, ranked, k) for name, ranked in arms.items()}

    def cell(name, idxs):
        d = [arm_r[name][i] - base_r[i] for i in idxs if arm_r[name][i] == arm_r[name][i]]
        if not d:
            return f"{'-':>25}"
        lo, hi = metrics.bootstrap_ci(d)
        if lo != lo:  # bucket too small to bootstrap — say so rather than print nan
            return f"{np.mean(d):>+8.4f} {'(n<2)':>16}"
        star = "*" if lo > 0 or hi < 0 else " "
        return f"{np.mean(d):>+8.4f} [{lo:+.3f},{hi:+.3f}]{star}"

    depth: dict[str, list[int]] = {}
    shape: dict[str, list[int]] = {}
    for i, q in enumerate(ds.queries):
        depth.setdefault(f"{q.n_hops}hop-{'join' if q.is_join else 'chain'}", []).append(i)
        shape.setdefault(q.shape or "?", []).append(i)

    lines = [f"\ndelta vs pooled sigma_q, recall@{k}  (paired bootstrap 95% CI)"]
    lines.append("  " + " " * 25 + "".join(f"{name:>25}" for name in arms))

    def section(buckets, title=None):
        if title:
            lines.append(f"  -- {title}")
        for label in sorted(buckets):
            idxs = buckets[label]
            lines.append(f"  {label:<18} n={len(idxs):<4}"
                         + "".join(cell(name, idxs) for name in arms))

    section({"all": list(range(len(ds.queries)))})
    section(depth, "depth x shape")
    section(shape, "question type")

    # Coverage differs per arm, so this cannot be a row in the table above.
    lines.append("  -- coverage-corrected (only questions where the arm has sub-questions)")
    for name in arms:
        idxs = covered[name]
        lines.append(f"  {name:<18} n={len(idxs):<4}" + cell(name, idxs))
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
    ap.add_argument("--fusion", nargs="*", default=["max"], choices=["max", "mean", "rrf"],
                    help="how to combine the query set. max = sigma_max, the method")
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

    arms, reports, covered = {}, {}, {}
    base = rank_arm(ds, doc_emb, None, args.model, args.batch_size, device, args.topk)
    reports["pooled"] = metrics.evaluate(ds, base, ks=tuple(args.k_report))

    for path in files:
        subq = load_subq(path)
        name = path.stem.replace(f"subq_{args.dataset}_", "")
        covered[name] = [i for i, q in enumerate(ds.queries) if subq.get(q.qid)]
        sizes = [len(subq.get(q.qid, [])) for q in ds.queries]
        print(f"\n{name:<10} {len(covered[name])}/{len(ds.queries)} questions have "
              f"sub-questions, mean {np.mean(sizes):.2f} each")
        if len(covered[name]) < len(ds.queries):
            print(f"           {len(ds.queries) - len(covered[name])} fall back to the "
                  "question alone — their delta is exactly zero, so they dilute the "
                  "mean without biasing it")
        for fusion in args.fusion:
            arm = name if fusion == "max" and len(args.fusion) == 1 else f"{name}/{fusion}"
            covered[arm] = covered[name] if arm != name else covered[name]
            arms[arm] = rank_arm(ds, doc_emb, subq, args.model, args.batch_size,
                                 device, args.topk, fusion)
            reports[arm] = metrics.evaluate(ds, arms[arm], ks=tuple(args.k_report))
        if len(args.fusion) > 1 or args.fusion[0] != "max":
            covered.pop(name, None)

    print(f"\n{'=' * 72}\nabsolute recall")
    for name, rep in reports.items():
        cells = "  ".join(f"@{k.split('@')[1]}={v['mean']:.4f}"
                          for k, v in rep["overall"].items())
        print(f"  {name:<12} {cells}")

    for k in args.k_report:
        print(delta_table(ds, base, arms, covered, k))

    dest = OUT / f"{ds.name}_subq_eval.json"
    dest.write_text(json.dumps(reports, indent=1))
    print(f"\nwrote {dest}")
    print("passage-level proxy for the sentence-level gate — see the module docstring")


if __name__ == "__main__":
    main()
