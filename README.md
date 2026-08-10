# mbuzai-rag

Harness for testing **sub-query gating** in graph RAG: replace the pooled query
vector in LinearRAG's activation gate with a max over self-contained
sub-questions, and measure whether the gain scales with chain depth.

Runbook (background, gates, reporting plan):
<https://claude.ai/code/artifact/7cb68ec0-bc70-4de0-aaeb-c31404d44496>

## The claim

A single pooled query embedding carries no signal past hop 1. On a
`(bridge → answer)` edge neither endpoint resembles the question — the bridge is
what you don't know, the answer is what you're looking for — so no function of
`(h_u, h_v, h_q)` can score it. The fix changes the *arguments*, not the function:

```
σ_q[i]     = sim(question, sentence_i)              # vanilla
σ_max[i]   = max_j sim(q_j, sentence_i)             # q_j = self-contained sub-questions
```

The original question is always element 0 of the query set, so `σ_max ≥ σ_q`
elementwise and the method cannot score below vanilla by construction.

**Prediction:** gain grows with chain depth, is larger on join-shaped questions
than chains of equal depth, and is neutral-to-negative on single-hop controls.

## Layout

```
mbuzai/dataio.py    per-dataset adapters -> one common shape
mbuzai/metrics.py   recall@k with per-hop / per-shape breakdowns + bootstrap CIs
mbuzai/gate.py      σ_max — the intervention, as a drop-in for LinearRAG
scripts/            runnable steps (below)
data/               HippoRAG-format datasets
out/                embeddings cache, sub-question sets, result JSON
```

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU box
pip install -r requirements.txt
```

`vllm` is only needed on the GPU box, for `gen_subq.py`.

## Steps

```bash
python scripts/download_data.py                    # 5 datasets, ~65 MB
python scripts/analyze_musique.py                  # difficulty axis + encode budget
python scripts/baselines.py musique --method bm25  # matrix row 1: the floor
python scripts/baselines.py musique --method dense
python scripts/baselines.py musique --method hybrid
python scripts/make_subq_ablations.py musique      # rows 4 and 5, no API
python scripts/gen_subq.py musique                 # row 3, needs a GPU
```

Everything writes to `out/`. Dense embeddings are cached per dataset, so only the
first dense run pays the encode cost.

### Sub-question sets

Three files, one shape (`{qid: [question, ...]}`), interchangeable downstream:

| file | source | role |
|---|---|---|
| `subq_musique_raw.json` | gold decomposition verbatim | ablation — quantifies the `#N` placeholder problem |
| `subq_musique_resolved.json` | gold, `#N` substituted | **oracle** — the ceiling a perfect decomposer reaches |
| `subq_musique_generated.json` | gpt-oss-20b | the method |

`resolved` is Gate B. If it does not beat pooled `σ_q`, no generator will and the
direction is dead — stop there and write the negative result.

## Patching LinearRAG

Clone [DEEP-PolyU/LinearRAG](https://github.com/DEEP-PolyU/LinearRAG) separately
(Python 3.9, its own dataset bundle at `Zly0523/linear-rag` — a different schema
from the copies here, and the dataset is named `2wikimultihop`, not
`2wikimultihopqa`).

Find where the sentence-similarity vector is built:

```bash
grep -rn "sigma\|sim(q\|query_emb\|encode(.*question" src/
```

It is the `|S| × 1` gate in `a^t = MAX(Mᵀ(σ_q ⊙ (M a^{t-1})), a^{t-1})`. Replace with
`mbuzai.gate.build_gate(model, sent_emb, question, subqs)`. Passing `subqs=None`
reproduces vanilla exactly, so gate it behind a `--subq_file` flag and keep one
binary for both arms.

## Generation on gpt-oss-20b

Single A100 40GB is enough — ~12–13 GB of MXFP4 weights. Ampere has no native
MXFP4 so vLLM upcasts to bf16 for the dot product; slower than Hopper, immaterial
at 1k prompts.

**Do not use `guided_json`.** vLLM
[#37359](https://github.com/vllm-project/vllm/issues/37359): harmony channel
tokens appear only in output, never the prompt, so the guidance FSM never
activates offline and the schema is silently ignored. `gen_subq.py` asks for one
question per line and parses the harmony `final` channel defensively instead.

Always `--dry-run --limit 20` first and read the samples. Watch the
`placeholder leak` counter — self-containment is the entire point, and a model
that reverts to `#1` has produced the raw-ablation arm by accident.

## Known traps

- **Passage identity.** 2,465/11,656 MuSiQue passages share a title with another;
  47/1000 questions have gold paragraphs that collide on it. `dataio` keys on
  `(title, text)`. Keying on title inflates recall.
- **Schema drift.** MuSiQue uses `paragraphs`/`is_supporting`; 2Wiki uses
  `context`/`supporting_facts`. Go through `dataio.load`, never the raw JSON.
- **Sentence embeddings, not passage.** `σ_q` runs against ~43.7k sentences,
  roughly 4× the passage count.
- **Joins break linear paths.** 131/1000 MuSiQue questions are trees with a join
  node. A sequential path cannot express them; handle or report.
- **Small buckets.** n=166 at 4-hop, n=27 at 4hop2. `metrics` bootstraps CIs —
  never report a bare delta on those.
- **Freeze before transfer.** Tune on MuSiQue only. Touching the method after
  seeing 2Wiki or HotpotQA numbers turns transfer results into tuning.

## Measured on this corpus

MuSiQue: 11,656 passages, 930k words, ~43.7k sentences, 1,000 questions.
869 pure chains / 131 joins. 57.3% of gold sub-questions carry a `#N` placeholder
and every question has at least one — which is why the method generates rather
than reuses them.
