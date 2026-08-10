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
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mbuzai import dataio  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out"

SYSTEM = "Reasoning: low"

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=1024, help="room for the analysis channel")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print samples, do not write")
    args = ap.parse_args()

    ds = dataio.load(args.dataset)
    queries = ds.queries[: args.limit] if args.limit else ds.queries
    print(f"{ds.name}: generating for {len(queries)} questions with {args.model}", flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_util,
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
