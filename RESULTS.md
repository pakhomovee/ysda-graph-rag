# Sub-query gating: where query decomposition helps in graph RAG

Status of the experiment as of 2026-08-11. Every number below is recall@k over
**our** MuSiQue corpus (11,656 passages, ~80 words each, 1,000 questions), scored
against `q.gold_pids` with paired bootstrap 95% CIs. `*` marks a CI excluding
zero. Run-to-run noise, measured by running one configuration twice, is
**~0.002–0.005**; treat smaller differences as nothing.

Every arm uses `all-mpnet-base-v2`. Sub-questions come from `gpt-oss-20b`
(960/1000 questions covered). Where a number is not comparable to the others it
is marked.

---

## The claim we started with

Replace the pooled query vector in a graph retriever's activation gate with a max
over self-contained sub-questions:

```
σ_q[i]   = sim(question, node_i)              # vanilla
σ_max[i] = max_j sim(q_j, node_i)             # q_j = sub-questions, q_0 = the question
```

Prediction: the gain grows with chain depth, because a single pooled vector
carries no signal on a `(bridge → answer)` edge — neither endpoint resembles the
question.

## The claim we can actually defend

**Query decomposition pays where candidates are selected and scored, not where
activation is propagated.** Four injection points across two unrelated graph
retrievers, plus a content control on the largest effect.

| query set applied at | system | effect @10 |
|---|---|---|
| sentence gate — *propagation* | LinearRAG | +0.011\*, falls to +0.004 under a good scorer |
| edge weights — *propagation* | QAFD-RAG | +0.002, null — but oracle +0.114\*, so range-limited, not inert |
| **fact scores — *selection*** | **QAFD-RAG** | **+0.073\*  (+16.0% relative)** |
| **cross-encoder — *scoring*** | any | **+0.019\***, on any retriever's candidates |

---

## Results

### 1. Passage-level proxy — no graph, `eval_subq.py`

Ranks passages directly by `max_j sim(q_j, passage)`. Isolates the query-set idea
from any graph machinery.

| arm | @2 | @10 | Δ@10 vs pooled |
|---|---|---|---|
| pooled question | 0.3446 | 0.5733 | — |
| generated / **max** | 0.4019 | 0.6164 | **+0.0432\*** |
| generated / mean | 0.2891 | 0.5160 | −0.0573\* |
| generated / rrf | 0.2195 | 0.5148 | −0.0584\* |
| resolved / max | 0.5221 | 0.8424 | +0.2692\* — **leaky, see caveats** |

**The `max` formulation is doing the work, not decomposition per se.** Paired,
`generated/max − generated/rrf` = **+0.1016\* @10** (+0.1824\* @2). Mean and RRF
are *worse than not decomposing at all*: RRF treats each sub-question as an
independent retriever, so it only works when each retrieves well on its own —
true for the leaky oracle (`resolved/rrf` +0.0912\*), false for sub-questions that
refer to the bridge by description. `max` needs only one member to match, and
`σ_max ≥ σ_q` keeps the original question as a floor.

### 2. 2Wiki — pre-registered scope condition

Predictions were written into the README and committed **before** the run.

| type | n | Δ@10 | predicted |
|---|---|---|---|
| `bridge_comparison` | 235 | +0.2074\* | largest — join-shaped *and* 4 hops |
| `comparison` | 244 | +0.1127\* | ≫ compositional — both entities named |
| `compositional` | 413 | +0.0327\* | ≈ MuSiQue's +0.05 — same unknown bridge |
| overall | 1000 | +0.0943\* | |

All four held. On `comparison` — the one bucket where the oracle cannot leak,
since there is no bridge to substitute — the **generated** set beats the **gold**
decomposition (+0.113 vs +0.080).

### 3. Non-graph baselines, same corpus and gold

| method | @2 | @10 |
|---|---|---|
| dense (mpnet) | 0.3446 | 0.5733 |
| hybrid (BM25 + dense, RRF) | 0.3740 | 0.5982 |
| cross-encoder rerank (dense top-50) | 0.4279 | 0.6394 |

A 60-second cross-encoder gets within 0.5 points of a full graph pipeline at @10
and beats every graph arm at @2. That belongs in the paper; the GraphRAG
literature routinely omits it.

### 4. LinearRAG — the gate

Indexed over **our** passages (`musique_fine`), not their 1,354×820-word chunks,
so these are comparable to everything above.

| arm | @2 | @10 | @50 |
|---|---|---|---|
| vanilla | 0.3444 | 0.6441 | 0.8662 |
| + σ_max | 0.3521 | 0.6551 (+0.0110\*) | 0.8806 (+0.0143\*) |
| + oracle | 0.3849 | 0.6916 | 0.8932 |
| + pooled rerank | 0.4427 | 0.6792 | — |
| + σ_max + pooled rerank | 0.4424 | 0.6813 | — |
| + query-set rerank | 0.4536 | 0.6979 | — |
| **+ σ_max + query-set rerank** | **0.4546** | **0.7015** | — |

**The gate works at the entity level and the pipeline discards it.** From
identical seeds (1,681 in both arms), σ_max produces **8× more two-hop entity
activations** (3,200 vs 402), 39% more activations per question, and 32% more
gold-passage hits beyond the seed. End-to-end that is worth +0.011.

**Reranking subsumes it.** Paired, `generatedrr − vanillarr` = +0.0022 (null) and
`generatedrrx − vanillarrx` = +0.0036 (null). Query-set reranking is worth
+0.019\* — but on *anyone's* candidates, including vanilla's.

Three ablations came back negative and are worth recording so nobody repeats
them: lowering `iteration_threshold` (the advantage *shrinks*), removing the
`/tier` hop-distance penalty (no change in either arm), and cutting
`passage_ratio` 2→0.5 (both arms worse).

### 5. QAFD-RAG — the largest effect, and the control

| arm | @2 | @10 | @50 |
|---|---|---|---|
| vanilla | 0.1914 | 0.4597 | 0.7123 |
| σ_max on **edge weights** | 0.1947 | 0.4622 (+0.0025, null) | 0.7120 |
| σ_max on **fact scores** (seeding) | 0.2492 | **0.5332 (+0.0734\*)** | 0.7452 (+0.0329\*) |
| **shuffled control**, seeding | 0.0973 | 0.3844 (−0.0753\*) | 0.6895 |

Seeding gives +16.0% relative, significant in every bucket at every k, growing with
depth (2hop +0.061 → 4hop-join +0.116 → 4hop3 +0.145).

Edge weighting — the site the flow-diffusion paper makes query-aware, and our first
guess — is null in every bucket at every k. **That null is a property of the
parameterisation, not of the site.** An oracle edge weight, run against the same
index, moves recall a long way:

| arm | routing_cv | Δ@2 | Δ@10 | Δ@50 |
|---|---|---|---|---|
| vanilla | 0.122 | — | — | — |
| σ_max on edge weights | — | −0.002 | +0.000 | +0.001 |
| oracle 10× | 0.256 | +0.047\* | +0.035\* | +0.009\* |
| oracle 100× | 0.632 | +0.117\* | +0.085\* | +0.026\* |
| oracle 1000× | 1.069 | +0.159\* | **+0.114\*** | +0.042\* |

Monotone in magnitude, significant in every bucket at every k. The mechanism is
visible directly. QAFD routes mass as `mass[j] += excess · w_ij / Σ_k w_ik`, so the
shares are **normalised**: only the *spread* of weights within a neighbourhood can
steer anything, and the absolute level divides out. `routing_cv` measures that spread
over the neighbourhoods actually pushed. Vanilla routes at 0.122 and the oracle needs
~8.75× that to deliver its gain — but Product is bounded in `[0,1]` and Hybrid in
`[1,1.5]`, so neither can produce it. σ_max makes this worse rather than better: `max`
lifts every node's similarity, irrelevant ones included, and under a normalised
distribution lifting everything toward the ceiling *reduces* contrast.

So the sharper statement of the placement result is not "propagation is dead":

> `max` helps where **global rank** decides — seeding takes a top-N cut over all
> nodes — and washes out where only **local relative contrast** decides, because
> routing normalises per neighbourhood.

**The oracle leaks, and its size should be read with that in mind.** It boosts edges
among gold-*passage* nodes, and passage nodes are the ranking targets, so it partly
pumps mass straight into the scored node rather than steering traversal toward it. A
learned scorer cannot identify the gold passage — that is the retrieval task. An
entity-only oracle, dropping the passage node and keeping only its extracted entities,
is the arm that separates steering from leak; until it runs, +0.114 is an upper bound
on an upper bound.

**Session-to-session drift is measured, not assumed.** Re-running vanilla against the
same index gives −0.0022 @10, CI [−0.007, +0.002] — null, and consistent with the
~0.002–0.005 run-to-run figure quoted at the top. Every delta in this section is
paired against a same-session baseline.

**The shuffled control is the strongest single piece of evidence here.** Same
number of sub-questions, same `max` over the same number of vectors, content
attached to the wrong question: **−0.0753\***. Wrong content is worse than no
content, so the gain is not the max operator inflating scores before
normalisation. Content effect = seeds − shuffled = **+0.149 @10**.

---

## Caveats that belong in the paper

**The `resolved` oracle leaks.** It substitutes `#N` with the *gold answer* of
step N. On MuSiQue, 1,648 of 2,648 hops then contain their answer string inside
the sub-question, and **all 1,648 of those strings also appear in the gold
passage**. "When was *Diego Maradona* signed by Barcelona?" retrieves the
Maradona passage because it was handed the word "Maradona" — the bridge entity,
which is by definition unknown at inference. `resolved` is a valid *necessary*
gate and **not a reachable target**; the generated→resolved gap is mostly
unavailable information, not generator headroom.

**QAFD's absolute recall is weak here.** 0.5332 after the gain still trails plain
dense retrieval (0.5733). Its defaults are presumably tuned for NV-Embed-v2 (7B);
we ran mpnet so the comparison is about method rather than encoder, and added a
sentence-transformers wrapper to their registry to make that possible. The
relative gain is large and the absolute position is not competitive — report
both.

**Retrieval unit matters more than expected.** LinearRAG's bundle re-chunks
MuSiQue into 1,354 blocks of ~820 words, which collapses all the gold of 18.4% of
questions (31% of 2-hop) into a *single* chunk — no multi-hop problem left to
solve. We re-indexed at passage granularity; the depth signature only became
visible afterwards.

**Three of our own mechanistic hypotheses were falsified by instrumentation**:
that LinearRAG's search is effectively one hop (it isn't — tiers 2/3 are
abundant), that σ_max starves the frontier via sentence dedup (it expands it),
and that the `/tier` penalty was blocking the signal (removing it changes
nothing). The traces are in `report_entities.py`.

**Not yet controlled**: the shuffled control has only been run on QAFD seeding.
The same max-operator exposure exists in `eval_subq --fusion max` and in the
query-set reranking, and should be checked the same way before those numbers are
quoted as content effects.

---

## What the paper can be about

### Recommended: a placement study

> **"Where does query decomposition help in graph retrieval?"**

The contribution is not the operator — `max` over sub-question embeddings is not
novel. It is the **localisation**: the same intervention, applied at four points
across two independent graph retrievers, helps by +16% at candidate *selection*,
+2% at *propagation* in one system and not at all in the other, and +19% relative
at final *scoring*. With a content control showing the effect is the
sub-questions and not the operator.

This is a claim about architecture, not about a method, and it transfers: any
system with a query-aware component now has a prior about where to put a
decomposition.

Supporting results that make it publishable:
- pre-registered scope condition confirmed on a second dataset (2Wiki comparison
  questions), including generated sub-questions beating the gold decomposition
- `max` ≫ RRF/mean fusion by +0.10 paired — the formulation matters, not just
  decomposition
- a strong cheap baseline (cross-encoder) included, which the GraphRAG literature
  usually omits — and which nearly matches a graph pipeline

### Weaker alternatives

**"σ_max improves graph RAG retrieval"** — the honest numbers are +0.011 in
LinearRAG and +0.073 in QAFD off a weak baseline. Neither is a headline, and a
reviewer will point at the cross-encoder.

**"A negative result on activation gating"** — defensible and well-evidenced, but
it throws away the QAFD seeding result, which is the largest and best-controlled
effect in the set.

### What would strengthen it most, in order

1. **The entity-only oracle on QAFD edge weights.** The +0.114 above boosts edges
   among gold *passage* nodes, which are also the ranking targets, so part of it is
   mass landing on the answer rather than steering toward it. Dropping the passage
   node from the oracle set separates the two. This gates everything else about the
   propagation site: if it goes flat, there is no reachable headroom there.
2. **The oracle arm on QAFD seeding** — the ceiling for the largest effect, and the
   only way to compare sites honestly. Right now a *ceiling* at propagation (+0.114)
   is being read against a *realized* gain at selection (+0.073).
3. **The shuffled control on the proxy and on the reranker.** Cheap, and it
   closes the last operator-versus-content question.
4. **A second dataset through QAFD** (2Wiki), since the placement claim currently
   rests on one dataset for the largest number.
5. **Answer-level metrics (EM/F1)**, since everything here is retrieval recall.

## Reproducing

```bash
python scripts/download_data.py
python scripts/baselines.py musique --method dense --device cuda:3 --batch-size 256
python scripts/make_subq_ablations.py musique
python scripts/gen_subq.py musique --device 3 --max-tokens 2048      # .venv-gen
python scripts/eval_subq.py musique --device cuda:3 --fusion max mean rrf

bash scripts/setup_linearrag.sh   && bash scripts/run_linear_musique.sh
bash scripts/setup_qafd.sh        # then src/passage_entity/benchmark_runner.py

# edge-weight probe: oracle sweep, exp sharpening, the unswept propagation knobs
JOBS=4 bash scripts/run_qafd_probe.sh
ARMS="oracle1000ent oracle100ent oracle1000seed" JOBS=3 bash scripts/run_qafd_probe.sh

# offline: can any light model beat cosine at finding gold nodes? CPU, no LLM
bash scripts/export_qafd_nodes.sh                       # QAFD env
python scripts/probe_learnability.py musique --shuffle-control   # must give AUC~0.5
python scripts/probe_learnability.py musique
```

Both third-party systems are pinned submodules patched from `patches/*.patch`;
`setup_*.sh` applies them and is idempotent. σ_max is bitwise inert without
`--subq_file` in both — verified at 43.7k×768 float32 for LinearRAG and across all
three similarity modes for QAFD.
