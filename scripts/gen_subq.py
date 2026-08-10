"""Generate self-contained sub-questions with a locally hosted gpt-oss-20b.

Matrix row 3, the method arm. Runs offline on one A100 40GB; ~12-13 GB of MXFP4
weights leaves ample room for KV cache. Ampere has no native MXFP4, so vLLM
upcasts to bf16 for the dot product — slower than Hopper, immaterial at 1k prompts.

Two gpt-oss quirks drive the design:

  * Structured output does not work offline. vLLM issue #37359: harmony channel
    tokens appear only in the model's output, never the prompt, so the guidance
    FSM never activates under LLM.generate/chat and the schema is silently
    ignored. We therefore ask for one question per line and parse defensively
    rather than passing guided_json.

  * Harmony emits an analysis channel whether or not it was requested, so the
    raw text may carry reasoning before the answer. `extract_final` strips it.

Output format is identical to the gold-derived ablations, so every downstream
arm reads the same shape regardless of where its sub-questions came from.

    python scripts/gen_subq.py musique
    python scripts/gen_subq.py musique --limit 20 --dry-run   # inspect first
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mbuzai import dataio  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out"

SYSTEM = "Reasoning: low"

# MXFP4 weights of gpt-oss-20b, with a little slack. The floor below which a
# card cannot hold the model at all, never mind a KV cache.
WEIGHTS_MIB = 14000

PROMPT = """Break the question below into 3-5 sub-questions, each targeting one \
fact needed to answer it.

Every sub-question must be SELF-CONTAINED and independently readable. Never write \
a placeholder such as "#1", "it", or "the previous answer". Where a sub-question \
depends on an intermediate fact you do not know, refer to that fact by definite \
description instead.

Good: "Where was the performer of III born?"
Bad:  "Where was #1 born?"

Output one sub-question per line. No numbering, no bullets, no commentary, no \
preamble. Do not answer the question.

Question: {q}"""

# harmony: <|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>...
FINAL = re.compile(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)", re.S)
STRIP_TAGS = re.compile(r"<\|[^|]*\|>")
LEADER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
PLACEHOLDER = re.compile(r"#\d")


def extract_final(text: str) -> str:
    """Return the final channel if harmony markup survived, else the whole text."""
    m = FINAL.search(text)
    if m:
        return m.group(1)
    # A reasoning parser may have stripped the markers but left the analysis prose;
    # if a bare 'final' marker is present, cut at it.
    if "final<|message|>" in text:
        text = text.split("final<|message|>", 1)[1]
    return STRIP_TAGS.sub("", text)


def parse_questions(text: str, max_n: int = 6) -> list[str]:
    out = []
    for line in extract_final(text).splitlines():
        line = LEADER.sub("", line.strip()).strip()
        if len(line) < 8 or not line.endswith("?"):
            continue
        if line.lower().startswith(("here", "sure", "sub-question", "output")):
            continue
        if line not in out:
            out.append(line)
    return out[:max_n]


def gpu_table() -> list[tuple[int, int, int]]:
    """(index, free_MiB, total_MiB) via nvidia-smi.

    Deliberately a subprocess rather than torch: querying through torch would
    initialise a CUDA context in this process, after which setting
    CUDA_VISIBLE_DEVICES has no effect.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.strip().splitlines():
        i, free, total = (x.strip() for x in line.split(","))
        rows.append((int(i), int(free), int(total)))
    return rows


def pick_device(requested: str | None, need_mib: int) -> tuple[int, int] | None:
    """Set CUDA_VISIBLE_DEVICES before vLLM is imported, and fail loudly if the
    chosen GPU is already occupied.

    gpt-oss-20b needs roughly 13 GB of weights plus KV cache; a card with a few
    hundred MiB free cannot even create a CUDA context, which surfaces as an
    opaque OOM inside MemorySnapshot rather than as 'this GPU is busy'.

    Returns (free_MiB, total_MiB) for the selected card, or None if nvidia-smi
    told us nothing. Under tensor parallelism the tightest card is the binding
    constraint, so the minimum across the selection is what comes back.
    """
    if requested is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = requested
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    rows = gpu_table()
    if not rows:
        print("nvidia-smi unavailable — skipping GPU preflight")
        return None

    print("\ngpu   free / total (MiB)")
    for i, free, total in rows:
        print(f"  {i}   {free:>6} / {total:>6}{'   <- busy' if free < need_mib else ''}")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        wanted = {int(x) for x in visible.split(",") if x.strip().isdigit()}
        rows = [r for r in rows if r[0] in wanted] or rows
    else:
        best = max(rows, key=lambda r: r[1])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best[0])
        print(f"\nauto-selected GPU {best[0]} ({best[1]} MiB free)")
        rows = [best]

    if max(r[1] for r in rows) < need_mib:
        sys.exit(
            f"\nFATAL: no selected GPU has {need_mib} MiB free.\n"
            "  Another process owns it. Check:  nvidia-smi --query-compute-apps="
            "pid,used_memory --format=csv\n"
            "  Then pick a free card:           --device 1\n"
            "  Or clear your own stale run:     pkill -f EngineCore"
        )

    return min(r[1] for r in rows), min(r[2] for r in rows)


def fit_gpu_util(requested: float, free_mib: int, total_mib: int,
                 headroom_mib: int = 1024) -> float:
    """Cap gpu_memory_utilization at what the card actually has left.

    vLLM measures the fraction against TOTAL memory, not free memory, so on a
    shared box the default is a crash waiting to happen: 0.85 of a 40 GB A100 is
    34.8 GB, and vLLM refuses to start when a neighbouring process already holds
    enough that only 25 GB remains. The absolute --need-mib floor above does not
    catch this — it never looks at the fraction.
    """
    usable = max(free_mib - headroom_mib, 0) / total_mib
    return min(requested, round(usable, 3))


def hf_cache_root() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def cached_snapshot(model: str) -> Path | None:
    """Newest complete snapshot of `model` in the local HF cache, else None.

    Walked by hand rather than through huggingface_hub, because the decision this
    feeds — HF_HUB_OFFLINE — is read into module constants at import time. Import
    the library to answer the question and you have already frozen the old value.

    Snapshot entries are symlinks into blobs/, and a half-downloaded blob has no
    symlink yet, so existence is a genuine completeness check. Where a shard
    index is present every shard it names is verified.
    """
    if Path(model).expanduser().is_dir():  # already a local path
        return Path(model).expanduser()

    root = hf_cache_root() / f"models--{model.replace('/', '--')}" / "snapshots"
    if not root.is_dir():
        return None

    for snap in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not (snap / "config.json").exists():
            continue
        index = snap / "model.safetensors.index.json"
        if index.exists():
            shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
            if shards and all((snap / s).exists() for s in shards):
                return snap
        elif any(snap.glob("*.safetensors")):
            return snap
    return None


def preflight_hub(model: str, mode: str) -> None:
    """Decide online vs offline before vLLM (and huggingface_hub) is imported.

    vLLM re-resolves config and tokenizer files through the Hub even when every
    byte is already cached. On a box with slow or filtered egress those HEAD
    requests retry with backoff and the engine looks hung — several minutes of
    silence after 'Parse safetensors files' has already sailed past at local-disk
    speed. Cached weights mean there is nothing to ask the network for.
    """
    snap = cached_snapshot(model)
    offline = mode == "on" or (mode == "auto" and snap is not None)

    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        where = snap if snap else "not found in cache — --offline on was forced"
        print(f"HF_HUB_OFFLINE=1  ({where})")
        return

    # Online: keep a stalled hub from masquerading as a hang.
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "15")
    print(f"{model} not fully cached under {hf_cache_root()} — the engine will fetch it")
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        print("  no HF_TOKEN set: unauthenticated downloads are rate-limited and slower")
    print("  a resumable pre-fetch is kinder than doing it inside the engine:")
    print(f"    hf download {model}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    ap.add_argument("--device", default=None,
                    help="GPU index, e.g. 1 or 0,1. Default: auto-pick the freest.")
    ap.add_argument("--need-mib", type=int, default=20000,
                    help="minimum free VRAM to consider a GPU usable")
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=1024, help="room for the analysis channel")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print samples, do not write")
    ap.add_argument("--offline", choices=["auto", "on", "off"], default="auto",
                    help="skip Hub round-trips. auto: offline when the model is cached.")
    args = ap.parse_args()

    preflight_hub(args.model, args.offline)
    selected = pick_device(args.device, args.need_mib)

    gpu_util = args.gpu_util
    if selected:
        free, total = selected
        gpu_util = fit_gpu_util(args.gpu_util, free, total)
        if gpu_util < args.gpu_util:
            print(f"\ngpu-util {args.gpu_util} -> {gpu_util}: the fraction is of TOTAL "
                  f"({total} MiB) and only {free} MiB is free")
        if gpu_util * total < WEIGHTS_MIB:
            sys.exit(
                f"\nFATAL: {gpu_util * total:.0f} MiB budget on this card, but "
                f"{args.model} needs ~{WEIGHTS_MIB} MiB of weights before any KV cache.\n"
                "  Wait for the card, or pick another:  --device 1"
            )

    ds = dataio.load(args.dataset)
    queries = ds.queries[: args.limit] if args.limit else ds.queries
    print(f"\n{ds.name}: generating for {len(queries)} questions with {args.model}", flush=True)
    print("loading vllm + torch (first import is slow, ~30-60s) ...", flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=gpu_util,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )
    # NOTE: deliberately no guided_decoding — see module docstring / vLLM #37359.
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=0.9,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    convos = [
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT.format(q=q.question)},
        ]
        for q in queries
    ]
    outputs = llm.chat(convos, sampling)

    result, empty, leaked = {}, [], []
    for q, o in zip(queries, outputs):
        subqs = parse_questions(o.outputs[0].text)
        if not subqs:
            empty.append(q.qid)
        if any(PLACEHOLDER.search(s) for s in subqs):
            leaked.append(q.qid)
        result[q.qid] = subqs

    n = len(result)
    lens = [len(v) for v in result.values()]
    print(f"\nparsed        : {n - len(empty)}/{n} non-empty")
    print(f"sub-qs each   : mean {sum(lens)/max(n,1):.2f}  min {min(lens, default=0)}  max {max(lens, default=0)}")
    print(f"placeholder leak: {len(leaked)} questions  <- must be ~0; the whole point is self-containment")
    if empty:
        print(f"empty parses  : {len(empty)} (first few: {empty[:5]})")

    qid = next((k for k, v in result.items() if v), None)
    if qid:
        print(f"\nexample  {qid}")
        print(f"  ORIGINAL  {next(q.question for q in queries if q.qid == qid)}")
        for s in result[qid]:
            print(f"  -         {s}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    OUT.mkdir(exist_ok=True)
    dest = OUT / f"subq_{ds.name}_generated.json"
    dest.write_text(json.dumps(result, indent=1))
    print(f"\nwrote {dest}")
    print("record the generator model in your results — it is part of the method")


if __name__ == "__main__":
    main()
