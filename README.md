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

**Use three environments.** They have incompatible Python requirements and there
is no reason to reconcile them — the handoff between generation and LinearRAG is
a JSON file on disk, not an import.

| env | Python | contents |
|---|---|---|
| `mbuzai` (this repo) | 3.10–3.14 | analysis, metrics, retrieval baselines |
| generation | **3.12** | `vllm` + gpt-oss-20b, GPU box only |
| LinearRAG | 3.9 | its own clone, its own `requirements.txt` |

Python 3.14 will fail the generation install. vLLM itself supports 3.10–3.14
since 0.20.0, but parts of its dependency tree still guard `<3.14` and the build
dies in `_guard_py_ver`. 3.12 is vLLM's recommended version; take it.

```bash
# this repo — any modern Python
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU box only
pip install -r requirements.txt

# generation — uv fetches the interpreter, no sudo, no conda
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 .venv-gen
. .venv-gen/bin/activate
uv pip install "vllm>=0.26" --torch-backend=auto
python -c "import vllm; print(vllm.__version__)"   # sanity: must be >= 0.10
```

Both arguments matter. **`--torch-backend=auto`** lets uv select the right CUDA
torch wheel; without it torch resolution fails, uv backtracks looking for a vLLM
whose deps it *can* satisfy, and lands on a 2023 release — which then dies with
`KeyError: 'type'` in `_get_and_verify_max_len`, because configs of that era used
`rope_scaling["type"]` rather than `rope_type`. **The `>=0.26` floor** turns that
silent backtrack into an honest resolution error. gpt-oss needs ≥ 0.10.0 at
minimum; take current.

Let vLLM pull its own `torch`; do not pre-install a CPU wheel into `.venv-gen`.
If an ancient vLLM already landed there, delete the venv and start over rather
than upgrading in place — it dragged old `transformers` and friends in with it.

## Steps

```bash
python scripts/download_data.py                    # 5 datasets, ~65 MB
python scripts/analyze_musique.py                  # difficulty axis + encode budget
python scripts/baselines.py musique --method bm25  # matrix row 1: the floor
python scripts/baselines.py musique --method dense
python scripts/baselines.py musique --method hybrid
python scripts/make_subq_ablations.py musique      # rows 4 and 5, no API
python scripts/gen_subq.py musique                 # row 3, needs a GPU
python scripts/eval_subq.py musique                # compare the sets — Gate B

bash scripts/run_qafd_probe.sh                     # edge-weight probe
```

The probe reads `LOCAL_LLM_BASE_URL` / `LOCAL_LLM_MODEL` / `LOCAL_LLM_API_KEY` if
they are exported, else probes `:5679`, `:5678`, `:8000` and reads the model id
from `/models`. That id also selects the index directory (`<save_dir>/<llm>_<emb>`),
so if the server is now serving something other than what indexed the corpus, set
`LOCAL_LLM_MODEL` explicitly — the preflight prints the directories it can see.

Arms run concurrently (`JOBS`, default 4). Retrieval is sequential *inside* one
arm — one query at a time, one reranker call per query — so a single run leaves
vLLM nearly idle however much it could serve; concurrency across arms is what
fills it, with no change to their retrieval loop. Pick `JOBS` from the per-run log
line `Retrieval done. total=Xs, rerank=Ys, qafd=Zs`: if `rerank ≈ total` the LLM
binds and `JOBS` can go high, if `qafd ≈ total` the pure-Python push-relabel binds
and `JOBS` should stay near the core count. Each process holds its own copy of the
index, so watch RAM. Query encoding is pinned to CPU (`ST_DEVICE`) so N copies of
mpnet don't compete with vLLM for VRAM.

### The edge-weight probe

`σ_max` on QAFD's diffusion edge weights is null. That has two incompatible
readings — the site is inert, or the heuristic is too blunt — and only one of them
says a trained edge scorer is worth building. `run_qafd_probe.sh` separates them
without training anything:

* an **oracle** edge weight, swept 10×/100×/1000×. It upper-bounds every possible
  scorer at that site, so if recall is flat the ceiling is zero and there is
  nothing to train. The sweep also measures how much dynamic range the site needs.
* an **exponentiated** weight, `w·exp(β(s_u+s_v))`. Every published variant is
  bounded — Hybrid spans `[1, 1.5]`, Product `[0, 1]` — and routing normalises
  (`mass[j] += excess·w_ij/Σw`), so only the *spread* within a neighbourhood can
  steer mass. One knob tests whether range was the blocker.
* `qa_sink_gamma` / `qa_accum_gamma`, the two propagation sites where query
  similarity enters unbounded, and which nothing in RESULTS.md has swept.

Read `report_edge_contrast.py` before the recall table. It reports the dispersion
of the routing distributions actually used, which is the only channel an edge
weight has: **a null arm whose routing CV matches vanilla's never steered mass and
proves nothing.**

The oracle came back positive (+0.114 @10 at 1000×, monotone), so the site is
range-limited rather than inert — see RESULTS.md §5. Two questions follow, and both
are cheaper than training a scorer:

```bash
# does it still hold without the leak? the oracle boosts gold PASSAGE nodes,
# which are also the ranking targets. --oracle_nodes entities drops them.
ARMS="oracle1000ent oracle100ent oracle1000seed" JOBS=3 bash scripts/run_qafd_probe.sh

# the oracle shows "if you knew, it would help". can you know?
bash scripts/export_qafd_nodes.sh    # QAFD env, no LLM
bash scripts/run_probe.sh            # mbuzai env, CPU; control + real, in parallel
```

`run_probe.sh` runs all six trainings at once — 3 models × {control, real} — and
sizes BLAS threads per worker from `nproc`. That has to happen before Python
starts, since OpenBLAS fixes its pool size when numpy is imported, and BLAS
scaling on these shapes flattens long before 96 threads: 6 workers × 16 threads
beats 1 × 96 comfortably. It refuses to let you read the real run if the control
failed.

`probe_learnability.py` asks whether any light model — a learned diagonal metric, a
low-rank correction to cosine, or an MLP over `[h_e⊙h_q, |h_e−h_q|, cos, deg]` — beats
plain cosine at ranking a question's gold entities. All are functions of the same
arguments the pipeline already has, split by question, reported on held-out questions.
If they tie with cosine, the oracle's headroom is not reachable from these vectors and
the in-pipeline scorer should not be built.

Everything writes to `out/`. Dense embeddings are cached per dataset, so only the
first dense run pays the encode cost.

Every step except `gen_subq.py` runs in the `mbuzai` env. `.venv-gen` has numpy
and a CUDA torch but none of the retrieval stack, so `baselines.py` gets far
enough to print `device=cuda` there before failing — it now checks its imports up
front and names the env instead.

### Sub-question sets

Three files, one shape (`{qid: [question, ...]}`), interchangeable downstream:

| file | source | role |
|---|---|---|
| `subq_musique_raw.json` | gold decomposition verbatim | ablation — quantifies the `#N` placeholder problem |
| `subq_musique_resolved.json` | gold, `#N` substituted | **oracle** — the ceiling a perfect decomposer reaches |
| `subq_musique_generated.json` | gpt-oss-20b | the method |

`resolved` is Gate B. If it does not beat pooled `σ_q`, no generator will and the
direction is dead — stop there and write the negative result.

`eval_subq.py` runs that comparison without LinearRAG: it applies the same
max-over-sub-questions rule at *passage* granularity and reports paired deltas
against the pooled query, bucketed by hop count and by join shape. A proxy, not
the method — no sentence gate, no graph propagation — but it shares the dense
baseline's embedding cache, so it is nearly free, and a `resolved` arm that
cannot win here will not win at sentence level either.

`gen_subq.py` also writes `subq_<ds>_generated_raw.jsonl`, the untouched
completions. Parsing sits downstream of a 20-minute GPU run, so
`--reparse out/subq_musique_generated_raw.jsonl` rebuilds the set after a parser
fix for free. This is not hypothetical; see below.

## Patching LinearRAG

Clone [DEEP-PolyU/LinearRAG](https://github.com/DEEP-PolyU/LinearRAG) separately
(Python 3.9, its own dataset bundle at `Zly0523/linear-rag` — a different schema
from the copies here, and the dataset is named `2wikimultihop`, not
`2wikimultihopqa`).

LinearRAG is a **pinned submodule** at `third_party/LinearRAG`, so the patch can
never apply to lines that upstream has since moved. It is GPL-3.0 and referenced,
not vendored — the repo stores one gitlink, no source of theirs. `ignore = dirty`
is set because we patch its working tree on purpose and that would otherwise show
up as a permanent modification.

```bash
bash scripts/setup_linearrag.sh    # init submodule, apply patch, verify. Idempotent.
```

The script refuses to guess: if the patch neither applies nor is already applied,
the submodule has drifted or has local edits, and it says so rather than
half-patching.

**What the retrieval actually does**, having read it rather than assumed: the gate
is `sentence_similarities`, computed twice — in `calculate_entity_scores` (BFS)
and `calculate_entity_scores_vectorized`, selected by `use_vectorized_retrieval`.
Both feed a per-entity top-k sentence selection that decides which entities
activate next; passages are then ranked by personalised PageRank in `run_ppr`.
There is no `a^t = MAX(Mᵀ(σ_q ⊙ (M a^{t-1})), a^{t-1})` — that formula was our
shorthand, and the real lever is *which sentences survive the per-entity top-k*.
Raising σ still moves it, but say what it is.

`question_embedding` reaches **four** call sites, not two. The patch changes only
the two gate sites and threads a separate `(m, d)` query-set matrix to them.
`dense_passage_retrieval` keeps the pooled vector, deliberately: it does
`question_embedding.reshape(1, -1)`, so a matrix would silently flatten to
`(1, m·d)` and corrupt passage scores — and more importantly, the claim is about
the *sentence gate*, so leaving DPR pooled isolates the intervention. It also
makes the measured effect conservative. Record it as a choice, not an oversight.

Vanilla parity is by construction: with no sub-questions the query set is `(1, d)`
and `sigma_max` reduces to the original dot product. Verified bitwise identical at
43.7k × 768 in float32, not merely close. One binary serves both arms; the arm is
selected by `--subq_file`.

Sub-question files are keyed by our qids, which LinearRAG never sees — it gets
`question_info["question"]`. Bridge that before running:

```bash
python scripts/export_subq_for_linearrag.py musique --subq out/subq_musique_generated.json
```

That re-keys by normalised question text and reports collisions (MuSiQue has one:
two questions share text but carry different gold decompositions).

Two things in their `run.py` that will bite: it hardcodes
`os.environ["CUDA_VISIBLE_DEVICES"] = "4"`, and it constructs `LLM_Model` before
any retrieval happens. `run_linearrag_retrieval.py` sidesteps both — it builds the
config with `llm_model=None` and calls `index()` then `retrieve()`, which never
touches the LLM.

### Their corpus is not our corpus

This is the thing to internalise before reading any LinearRAG number:

|  | passages | mean length |
|---|---|---|
| ours (`data/musique_corpus.json`) | 11,656 | ~80 words |
| theirs (`dataset/musique/chunks.json`) | 1,354 | ~820 words |

They concatenate the source passages into ~1000-token chunks, so one chunk holds
roughly ten of ours and 0/1354 match ours as text. **Recall@k over their chunks is
mechanically easier and is not comparable to `baselines.py` or `eval_subq.py`.**
Vanilla vs σ_max stays clean — same corpus, same index, same everything but the
query set — so report the paired delta and never a cross-corpus absolute.

Their `questions.json` also carries no gold labels (`evidence` is `""` on every
row), so gold comes from our copy: `prepare_linearrag_gold.py` locates each of our
gold passages inside a chunk by normalised substring (1,399/1,456 found; 998/1000
questions scorable, mean 2.19 gold chunks from 2.65 gold passages). Their question
ids are ours with a source prefix (`musique_2hop__13548_13529`), matching
1000/1000 — no text join needed.

### Running it

One script per dataset does the whole thing — preflight, gold, arms, score:

```bash
bash scripts/run_linear_musique.sh
bash scripts/run_linear_wiki.sh
bash scripts/run_linear_musique.sh out/subq_musique_resolved.json   # oracle arm
```

Retrieval needs LinearRAG's 3.9 interpreter and the rest needs the mbuzai env, so
point `PY39` at the former: `PY39=~/.venv39/bin/python bash scripts/run_linear_musique.sh`.
`DEVICE` (default 3), `TOPK` (10) and `EMB` are overridable the same way.

The steps, if you would rather drive them by hand:

```bash
python scripts/prepare_linearrag_gold.py musique --bundle third_party/LinearRAG/dataset/musique
python scripts/export_subq_for_linearrag.py musique --subq out/subq_musique_generated.json

cd third_party/LinearRAG                       # their relative paths need this cwd
export PYTHONPATH=/path/to/mbuzai-rag:$PYTHONPATH
python /path/to/mbuzai-rag/scripts/run_linearrag_retrieval.py \
    --dataset_name musique --device 3 --retrieval_top_k 10 \
    --subq_file /path/to/mbuzai-rag/out/subq_musique_generated_bytext.json \
    --out /path/to/mbuzai-rag/out/linearrag_musique

cd -                                           # back in the mbuzai env
python scripts/score_linearrag.py musique --runs out/linearrag_musique_*.json
```

Both arms run in one process against one index, so they cannot differ by anything
except the query set — no re-chunking, no re-embedding, no second NER pass.

## Generation on gpt-oss-20b

Single A100 40GB is enough — ~12–13 GB of MXFP4 weights. Ampere has no native
MXFP4 so vLLM upcasts to bf16 for the dot product; slower than Hopper, immaterial
at 1k prompts.

**A card with room to spare is not the same as an idle card.** `gpu_memory_utilization`
is a fraction of *total* VRAM, not free VRAM, so the 0.85 default asks for 34.8 GB
of a 40 GB A100 and vLLM aborts at startup if a neighbour already holds 15 GB.
`gen_subq.py` caps the fraction at what the selected card actually has free and
prints the adjustment; `--need-mib` is a separate, absolute floor and does not
catch this on its own.

**Do not use `guided_json`.** vLLM
[#37359](https://github.com/vllm-project/vllm/issues/37359): harmony channel
tokens appear only in output, never the prompt, so the guidance FSM never
activates offline and the schema is silently ignored. `gen_subq.py` asks for one
question per line and parses the harmony `final` channel defensively instead.

**Cached weights still hit the network.** vLLM re-resolves config and tokenizer
files through the Hub even when every byte is on disk, so on a box with slow or
filtered egress the engine stalls in an SSL read — minutes of silence *after*
`Parse safetensors files` has flown past at local-disk speed. `gen_subq.py`
detects a complete snapshot in the HF cache and sets `HF_HUB_OFFLINE=1` itself
(`--offline auto`, the default; `on`/`off` to force). A partial cache stays
online, since offline mode would only turn a slow download into a hard failure.
Pre-fetch with `hf download openai/gpt-oss-20b` — resumable, and it shows
progress — rather than discovering the download inside the engine.

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
- **Harmony channel names survive as prose.** Decoding with `skip_special_tokens`
  deletes the `<|...|>` delimiters but keeps the channel *names*, so the output
  arrives as `analysis<reasoning>assistantfinal<answer>` with no markup left to
  match on. Matching only the marked-up forms sends the whole text — reasoning
  included — to the line parser, and analysis prose ends in a question mark often
  enough to pass for a sub-question: 733 of 910 on the first real run, undetected
  by the `placeholder leak` counter because reasoning does not contain `#N`.
  `extract_final` handles all three renderings and treats analysis-with-no-final
  as empty rather than as content. Sanity check any new generator with
  `--dry-run --limit 20` **and** grep the output for `assistantfinal`.
- **Freeze before transfer.** Tune on MuSiQue only. Touching the method after
  seeing 2Wiki or HotpotQA numbers turns transfer results into tuning.

## Measured on this corpus

MuSiQue: 11,656 passages, 930k words, ~43.7k sentences, 1,000 questions.
869 pure chains / 131 joins. 57.3% of gold sub-questions carry a `#N` placeholder
and every question has at least one — which is why the method generates rather
than reuses them.

2Wiki: 6,119 passages, 1,000 questions. 413 compositional / 244 comparison /
235 bridge_comparison / 108 inference; 479 joins.

## The oracle leaks — read `resolved` accordingly

`resolved` substitutes each `#N` with the gold answer of step N. On MuSiQue,
1,648 of 2,648 hops then contain their answer string inside the sub-question, and
**every one of those strings also appears in the gold passage.** "When was Diego
Maradona signed by Barcelona?" retrieves the Maradona passage because it was
handed the word "Maradona" — but Maradona *is* the bridge, the thing you do not
know.

So `resolved` is a valid necessary gate and **not a reachable target.** The
generated→resolved gap is mostly information unavailable at inference, not
generator headroom, and closing it by scaling the generator is not possible: no
model can know the bridge entity from the question alone. Measured on the
passage proxy: pooled 0.573 recall@10, resolved 0.842, generated 0.616.

## Pre-registered: the 2Wiki transfer

Written before running `eval_subq.py` on 2Wiki. The mechanism says sub-query
gating pays off where decomposition is possible *without* oracle knowledge, which
is what comparison questions give you — both entities are named in the question,
so the sub-questions are self-contained with nothing to look up first. A single
pooled vector also cannot rank two targets at once; a max over two sub-questions
can. Hence:

1. `comparison` ≫ `compositional`. No unknown bridge, and two retrieval targets a
   single query vector must serve simultaneously.
2. `bridge_comparison` shows the largest gain — join-shaped *and* 4 hops, so both
   predicted effects compound.
3. `compositional` lands near MuSiQue's +0.05, being the same unknown-bridge
   shape as a MuSiQue chain.
4. Depth scaling holds within 2Wiki as it does on MuSiQue.

Known caps before looking: `bridge_comparison` has 4 gold passages, so recall@2
cannot exceed 0.5 there — read @5/@10. 2Wiki's sub-questions are templated from
`evidences` triples, so its oracle is comparable *within* 2Wiki only, never
against MuSiQue's human-written decompositions.

MuSiQue stays the primary result whatever 2Wiki says. This is a scope condition
being tested, not a headline being shopped for.
