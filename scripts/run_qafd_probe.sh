#!/usr/bin/env bash
# Edge-weight probe on QAFD: does the propagation site have any headroom at all?
#
#     bash scripts/run_qafd_probe.sh                      # picks up LOCAL_LLM_* exports
#     ARMS="vanilla oracle1000" bash scripts/run_qafd_probe.sh          # subset
#
# The server is found from LOCAL_LLM_BASE_URL (or LLM_BASE_URL), else by probing
# :5679 and :5678; the model id is read from /models when not set. Both names
# work, so an already-exported LOCAL_LLM_* environment needs no arguments:
#
#     export LOCAL_LLM_BASE_URL="http://localhost:5679/v1"
#     export LOCAL_LLM_MODEL="$(curl -s $LOCAL_LLM_BASE_URL/models \
#         | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')"
#     export LOCAL_LLM_API_KEY="dummy"
#
# RESULTS.md records sigma_max on QAFD edge weights as null (+0.0025 @10). Two
# incompatible readings: the site is inert, or the heuristic is too blunt. This
# separates them WITHOUT training anything.
#
#   oracle10/100/1000  hand the edge the answer. Upper-bounds every scorer at
#                      this site, learned included. Flat at 1000x => nothing to
#                      train. Swept by magnitude because "how much dynamic range
#                      does this site need" is exactly what a trained model
#                      would have to hit.
#   expb2/8/32         w * exp(beta*(s_u+s_v)). Every published variant is
#                      bounded (Hybrid spans [1,1.5], Product [0,1]) and routing
#                      normalises, so only spread steers mass. One knob, no
#                      training. If this captures the oracle's gain, no model
#                      is needed.
#   sink4/accum4       the OTHER two propagation sites, where query similarity
#                      enters unbounded (config.py qa_sink_gamma /
#                      qa_accum_gamma, both default 0.0). Neither appears in
#                      RESULTS.md, so "propagation is dead" does not cover them.
#
# Retrieval-only against the cached index: no OpenIE, no indexing GPU-hours.
# benchmark_runner runs in QAFD's env; scoring runs in the mbuzai env (3.10+).
set -euo pipefail

OURS=musique
PY=${PY:-python3}                 # QAFD env
PY_MBUZAI=${PY_MBUZAI:-python3}   # mbuzai env (3.10+)
EMB=${EMB:-mpnet}
TOPK=${TOPK:-200}
SAVE_DIR=${SAVE_DIR:-outputs/$OURS}
# LOCAL_LLM_* is the convention already exported on the GPU box; LLM_* overrides it.
LLM_MODEL=${LLM_MODEL:-${LOCAL_LLM_MODEL:-}}
LLM_BASE_URL=${LLM_BASE_URL:-${LOCAL_LLM_BASE_URL:-}}
LLM_API_KEY=${LLM_API_KEY:-${LOCAL_LLM_API_KEY:-dummy}}   # vLLM ignores it, the client requires it
PORTS=${PORTS:-"5679 5678 8000"}  # probed in order when no base URL is set

# Arms run as concurrent processes. Retrieval is sequential inside one arm — one
# query at a time, one reranker call per query — so a single run leaves vLLM
# almost idle no matter how much it could serve. Concurrency across arms is what
# fills it, and needs no changes to their retrieval loop.
#
# Pick JOBS from whichever resource binds. The per-run log line
#   "Retrieval done. total=Xs, rerank=Ys, qafd=Zs"
# says which: rerank ~ total means the LLM is the bottleneck and JOBS can go high
# (vLLM batches happily); qafd ~ total means the pure-Python push-relabel is, and
# JOBS should stay near the core count. Each process also holds its own copy of
# the index, so watch RAM before raising it.
JOBS=${JOBS:-4}
ST_DEVICE=${ST_DEVICE:-cpu}   # query encoder; cpu keeps the GPU entirely for vLLM

# The diffusion regime. Three settings jointly decide whether edge weights can
# influence anything, and this harness ships all three conservative:
#
#   ALPHA      injected mass is alpha x total_sink, and total_sink is normalised
#              to 10 while source weights sum to 1 -- so the excess that actually
#              propagates is 10(alpha-1). At the shipped 1.5 that is 5 units and
#              the measured spread is ~87 pushes per question on a 96,920-node
#              graph: a local expansion, not a diffusion. The paper specifies 50
#              (490 units), and their own src/retrievers/flow_diffusion.py
#              defaults to 50 -- only this passage-entity pipeline uses 1.5.
#   MAXITER    with BATCH_PUSH off this caps pushes per question outright. 500 is
#              fine for alpha=1.5 (87 used) and truncates alpha=50 (~8700 needed)
#              into a cut-off diffusion that is not the paper's either.
#   BATCH_PUSH graph_adapter's own comment: "makes edge weights effective because
#              each iteration touches all excess nodes' edges, not just one
#              random node's". Off by default, so routing is sampled one random
#              excess node at a time.
#
# Any null about edge weighting is conditional on these. Defaults reproduce every
# run so far; the paper's regime is ALPHA=50 MAXITER=20000 BATCH_PUSH=1.
# PROFILE=paper uses the config QAFD's own entry point ships for multihop --
# benchmarks/run.py GRAPH_TYPE_DEFAULTS["passage-entity"], commented there as
# "from proven successful runs". It differs from benchmark_runner.py's argparse
# defaults in three places, and calling benchmark_runner directly (as this
# runbook did) silently gets the argparse ones:
#
#                    benchmarks/run.py     argparse default
#   weight_scheme    original (Eq. 5c)     multiply (Eq. 5b)
#   linking_top_k    5                     10
#   alpha            2.0                   1.5
#
# weight_scheme is the one that matters here: every edge-weight result so far was
# measured against Product, not the Hybrid form they ship. linking_top_k feeds
# seeding, so it is upstream of the fact-score arms too.
# PROFILE=bundle is the config the AUTHORS' OWN run used, read out of the
# results_<ds>.json their HuggingFace KG ships (their benchmark_runner writes it
# into working_dir, recording config.* as the run actually used them):
#
#   qafd_alpha 3.0   qafd_epsilon 0.005   qafd_weight_scheme "none"
#   linking_top_k 5  retrieval_top_k 200  qafd_max_iterations 500
#
# Three of those differ from what any entry point in the repo defaults to, and
# weight_scheme "none" means that run had the query-aware edge weighting -- the
# paper's contribution -- switched off. Reproducing their setting is therefore a
# different thing from reproducing benchmarks/run.py, and both differ from the
# paper text (alpha 50). Keep all three reachable by name rather than by memory.
PROFILE=${PROFILE:-}
if [ "$PROFILE" = paper ]; then
    ALPHA=${ALPHA:-2.0}
    WEIGHT_SCHEME=${WEIGHT_SCHEME:-original}
    LINKING_TOP_K=${LINKING_TOP_K:-5}
elif [ "$PROFILE" = bundle ]; then
    ALPHA=${ALPHA:-3.0}
    EPSILON=${EPSILON:-0.005}
    WEIGHT_SCHEME=${WEIGHT_SCHEME:-none}
    LINKING_TOP_K=${LINKING_TOP_K:-5}
else
    ALPHA=${ALPHA:-1.5}
    WEIGHT_SCHEME=${WEIGHT_SCHEME:-multiply}
    LINKING_TOP_K=${LINKING_TOP_K:-10}
fi
EPSILON=${EPSILON:-0.01}          # benchmark_runner's argparse default
# Eq. 2c is w * (a + b * avg_query_sim). The PAPER fixes a=1, b=1/4; the code
# defaults b to 0.5 and never overrode it, so every "original" arm so far ran at
# twice the published spread -- generous to the mechanism, but not the paper.
HYBRID_A=${HYBRID_A:-1.0}
HYBRID_B=${HYBRID_B:-0.5}
MAXITER=${MAXITER:-500}
BATCH_PUSH=${BATCH_PUSH:-}

# SHARDS splits one arm's question set across independent processes. Retrieval is
# a sequential loop over independent questions, so at the paper's alpha the wall
# time is per-question diffusion in pure Python and this is the only way to use
# more than one core within an arm. Shards are strided, so each covers the same
# mix of hop depths, and the dumps are merged back into the canonical arm name.
SHARDS=${SHARDS:-1}
# MERGE_ONLY=1 re-merges existing shards for ARMS and exits, for when a run
# completed but the merge did not.
ARMS=${ARMS:-"vanilla oracle10 oracle100 oracle1000 expb2 expb8 expb32 sink4 accum4"}

# Thread budget. Every worker is a separate python process, and numpy/torch each
# size their pool from the core count at import, so N workers on a 64-core box
# claim 64 threads EACH unless told otherwise -- JOBS=2 SHARDS=16 is 32 workers
# asking for 2048 threads. Divide the box instead. Must be exported before python
# starts; setting it inside would be too late.
#
# Process accounting, since the count is surprising: each shard is three
# processes -- the backgrounded run_shard shell, the ( cd ... ) subshell, and
# python -- plus one shell per arm. JOBS x SHARDS is the number that matters.
_WORKERS=$(( JOBS * (SHARDS > 1 ? SHARDS : 1) ))
_CORES=${CORES:-$(nproc)}
_THREADS=$(( _CORES / (_WORKERS > 0 ? _WORKERS : 1) ))
[ "$_THREADS" -lt 1 ] && _THREADS=1
export OMP_NUM_THREADS=$_THREADS OPENBLAS_NUM_THREADS=$_THREADS \
       MKL_NUM_THREADS=$_THREADS NUMEXPR_NUM_THREADS=$_THREADS

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB="$ROOT/third_party/QAFD-RAG"
ORACLE="$ROOT/out/qafd_oracle_${OURS}.json"
SUBQ=${SUBQ:-$ROOT/out/subq_${OURS}_generated_bytext.json}
EDGE_MODEL=${EDGE_MODEL:-$ROOT/out/edge_scorer_${OURS}.npz}
# The same model trained on permuted gold sets. Running it in the pipeline
# separates "the learned signal helps" from "perturbing the weights helps".
EDGE_MODEL_SHUF=${EDGE_MODEL_SHUF:-$ROOT/out/edge_scorer_${OURS}_shuffled.npz}
# Cached query vectors. With the authors' prebuilt KG the encoder is NV-Embed-v2
# (7B), and every shard worker would otherwise load its own copy. Encode once:
#   python src/passage_entity/benchmark_runner.py ... --cache_query_emb <path>
QUERY_EMB=${QUERY_EMB:-}

echo "==> preflight"
[ -d "$SUB/src" ] || { echo "submodule missing — run: bash scripts/setup_qafd.sh" >&2; exit 1; }
require_flags () {
    # Ask the runner what it actually accepts, rather than grepping for a marker.
    # A marker proves *some* version of the patch is applied; `git pull` updates
    # patches/qafd_sigma_max.patch but NOT the submodule working tree it applies
    # to, so a stale tree passes any fixed marker check and then dies on argparse
    # tens of minutes in — or, in a parallel sweep, on every arm at once.
    local help_text
    help_text=$( cd "$SUB" && $PY src/passage_entity/benchmark_runner.py --help 2>&1 ) || {
        echo "benchmark_runner.py --help failed:" >&2; echo "$help_text" | tail -5 >&2; exit 1; }
    for f in "$@"; do
        case "$help_text" in
            *"$f"*) ;;
            *) cat >&2 <<EOF
benchmark_runner.py does not accept $f — the submodule has a stale patch.
  git pull updates patches/qafd_sigma_max.patch; it does not reapply it.
  Fix:  bash scripts/setup_qafd.sh
EOF
               exit 1 ;;
        esac
    done
}
require_flags --oracle_gold_file --oracle_edge_mult --oracle_nodes --oracle_site \
              --exp_beta --edge_stats_file
# The fact reranker runs inside the retrieval loop and swallows every exception
# (reranker.py: "except Exception ... generated_facts = []"). An unreachable
# server therefore does not fail the run — it silently drops reranking from all
# 1000 questions and quietly produces a different pipeline than RESULTS.md
# measured. Resolve and check the server once, here, not in the deltas.
if [ -z "$LLM_BASE_URL" ]; then
    for _p in $PORTS; do
        if curl -sf --max-time 5 "http://localhost:$_p/v1/models" >/dev/null 2>&1; then
            LLM_BASE_URL="http://localhost:$_p/v1"
            echo "    found a server on :$_p"
            break
        fi
    done
fi
[ -n "$LLM_BASE_URL" ] || { echo "no server on any of: $PORTS — set LOCAL_LLM_BASE_URL" >&2; exit 1; }
curl -sf --max-time 10 "$LLM_BASE_URL/models" >/dev/null \
    || { echo "no LLM at $LLM_BASE_URL — the reranker would silently no-op on every query" >&2; exit 1; }

# The model id selects the INDEX DIRECTORY as well as the reranker, so it has to
# be the id the server actually reports — a checkpoint path where vLLM was
# started with --served-model-name points at a different (or absent) KG.
if [ -z "$LLM_MODEL" ]; then
    LLM_MODEL=$(curl -s --max-time 10 "$LLM_BASE_URL/models" \
        | $PY_MBUZAI -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')
    echo "    model id from /models: $LLM_MODEL"
fi
[ -n "$LLM_MODEL" ] || { echo "could not read a model id from $LLM_BASE_URL/models" >&2; exit 1; }

# working_dir is a derived property (save_dir/<llm>_<emb>), not a flag. Ask their
# own config for it rather than reimplementing the derivation in bash.
WORKDIR=$( cd "$SUB" && $PY_MBUZAI - "$OURS" "$SAVE_DIR" "$LLM_MODEL" "$EMB" <<'PYEOF'
import importlib.util, sys
# Load config.py by path rather than as src.passage_entity.config: importing the
# package runs src/__init__.py, which pulls in the full QAFD dependency tree for
# what is a pure string derivation. config.py has no imports of its own.
spec = importlib.util.spec_from_file_location("_cfg", "src/passage_entity/config.py")
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)
ds, save_dir, llm, emb = sys.argv[1:5]
print(cfg.PassageEntityConfig(dataset=ds, save_dir=save_dir,
                              llm_model=llm, embedding_model_key=emb).working_dir)
PYEOF
)
WORKDIR="$SUB/$WORKDIR"
[ -d "$WORKDIR" ] || { cat >&2 <<EOF
no index at $WORKDIR
  This probe reuses the index that produced the RESULTS.md QAFD rows; it does
  not build one. The path is derived as <save_dir>/<llm>_<emb>, so all three of
  these must match the run that built it:
    SAVE_DIR=$SAVE_DIR
    LLM_MODEL=$LLM_MODEL
    EMB=$EMB
  If the id was auto-detected and the server is now serving a different model
  than it was at indexing time, set LOCAL_LLM_MODEL (or LLM_MODEL) explicitly.
  Available:
$(ls -d "$SUB/$SAVE_DIR"/*/ 2>/dev/null | sed 's/^/    /' || echo "    (none under $SUB/$SAVE_DIR)")
EOF
exit 1; }
echo "    ok: patched submodule, LLM reachable, index at $WORKDIR"

# Report every missing dependency at once. These surface one import at a time,
# tens of minutes apart when arms run in parallel, and QAFD's requirements.txt
# does not list pandas at all — it arrives transitively via `datasets`, so a
# partial install leaves an env that gets a long way in before failing.
# find_spec locates without importing, so this is fast and side-effect free.
_missing=$($PY - <<'PYEOF'
import importlib.util as u
need = {"numpy": "numpy", "pandas": "pandas", "igraph": "python-igraph",
        "networkx": "networkx", "torch": "torch", "tqdm": "tqdm",
        "openai": "openai", "sentence_transformers": "sentence-transformers"}
print(" ".join(pip for mod, pip in need.items() if u.find_spec(mod) is None))
PYEOF
)
[ -z "$_missing" ] || { cat >&2 <<EOF
missing from the QAFD env ($($PY -c 'import sys; print(sys.executable)')):
    $_missing
  Install, then rerun:
    $PY -m pip install $_missing
EOF
exit 1; }
echo "    ok: QAFD env has the retrieval-path imports"

echo "==> oracle gold map"
$PY_MBUZAI "$ROOT/scripts/make_qafd_oracle.py" $OURS

# Mirror benchmark_runner's regime suffix so the skip check and the stats files
# agree with the dump it will actually write. printf %g on both sides so 50 and
# 50.0 do not produce two different names for one regime.
SUFFIX=""
[ "$(printf '%g' "$ALPHA")" != "1.5" ] && SUFFIX="${SUFFIX}-a$(printf '%g' "$ALPHA")"
[ "$(printf '%g' "$EPSILON")" != "0.01" ] && SUFFIX="${SUFFIX}-e$(printf '%g' "$EPSILON")"
[ -n "$BATCH_PUSH" ] && SUFFIX="${SUFFIX}-bp"
# Non-destructive: runs at the shipped config get their own names rather than
# overwriting the dumps behind the existing RESULTS.md table.
[ "$WEIGHT_SCHEME" != multiply ] && SUFFIX="${SUFFIX}-ws${WEIGHT_SCHEME}"
[ "$WEIGHT_SCHEME" = original ] && [ "$(printf '%g' "$HYBRID_B")" != "0.5" ] \
    && SUFFIX="${SUFFIX}-hb$(printf '%g' "$HYBRID_B")"
[ "$LINKING_TOP_K" != 10 ] && SUFFIX="${SUFFIX}-ltk${LINKING_TOP_K}"
[ -n "${NUM_QUERIES:-}" ] && SUFFIX="${SUFFIX}-n${NUM_QUERIES}"

# One shard of one arm.
# benchmark_runner composes the dump name itself, starting from "vanilla" and
# appending one suffix per non-default setting, so an arm called oracle1000ent
# lands in qafd_<ds>_vanilla-oracle1000ent<regime>.json. Reproducing that rule
# here is what broke the merge; keep ONE definition of it.
dump_stem () {
    local name=$1
    case $name in
        # With --subq_file the runner names the arm "<scope>-<queryset>" instead
        # of starting from "vanilla".
        subqedges) echo "edges-generated${SUFFIX}" ;;
        subqseeds) echo "seeds-generated${SUFFIX}" ;;
        # benchmark_runner appends retrieval_mode and the edge-model tag AFTER
        # the regime suffix, not before, so these two do not follow the pattern
        # the oracle / exp / sink arms use.
        dpr)       echo "vanilla${SUFFIX}-dpr" ;;
        edgemodel*) echo "vanilla${SUFFIX}-em${name#edgemodel}" ;;
        emshuf*)   echo "vanilla${SUFFIX}-emshuf${name#emshuf}" ;;
        em*)       echo "vanilla${SUFFIX}-em${name#em}" ;;
        vanilla)   echo "vanilla${SUFFIX}" ;;
        *)         echo "vanilla-${name}${SUFFIX}" ;;
    esac
}

run_shard () {
    local name=$1 shard=$2; shift 2
    local tag; tag=$(dump_stem "$name")
    local log="$ROOT/out/probe_${OURS}_${tag}${shard:+.shard${shard/\//of}}.log"
    # By path, NOT `python -m src.passage_entity.benchmark_runner`. The runner
    # sidesteps src/__init__.py — which imports aioboto3 and the rest of the AWS
    # stack — by registering src/* as plain namespace packages in sys.modules
    # before importing anything from them. With -m, Python imports the real `src`
    # package to locate the submodule, so __init__.py runs first and the bypass
    # never gets the chance: ModuleNotFoundError: aioboto3.
    ( cd "$SUB" && PYTHONPATH="$ROOT:${PYTHONPATH:-}" MBUZAI_ST_DEVICE="$ST_DEVICE" \
      $PY src/passage_entity/benchmark_runner.py \
        --dataset $OURS \
        --save_dir "$SAVE_DIR" \
        --llm_model "$LLM_MODEL" \
        --llm_base_url "$LLM_BASE_URL" \
        --llm_api_key "$LLM_API_KEY" \
        --embedding_model "$EMB" \
        --retrieval_top_k "$TOPK" \
        ${QUERY_EMB:+--query_emb_file "$QUERY_EMB"} \
        --qafd_alpha "$ALPHA" \
        --qafd_epsilon "$EPSILON" \
        --qafd_weight_scheme "$WEIGHT_SCHEME" \
        --hybrid_a "$HYBRID_A" --hybrid_b "$HYBRID_B" \
        --linking_top_k "$LINKING_TOP_K" \
        --qafd_max_iterations "$MAXITER" \
        ${BATCH_PUSH:+--batch_push} \
        --skip_qa \
        ${NUM_QUERIES:+--num_queries "$NUM_QUERIES"} \
        ${shard:+--query_shard "$shard"} \
        --edge_stats_file "$ROOT/out/edgestats_${OURS}_${tag}${shard:+.shard${shard/\//of}}.json" \
        "$@" ) >"$log" 2>&1 \
        && echo "    done: $name${shard:+ shard $shard}" \
        || { echo "    FAILED: $name${shard:+ shard $shard} — see $log" >&2; return 1; }
}

# All shards of one arm, then merge their dumps into the canonical arm name so
# score_qafd.py sees one run. The dumps are {qid: [pids]} and the shards are
# disjoint by construction, so the merge is a plain dict union.
run_arm () {
    local name=$1; shift
    if [ "$SHARDS" -le 1 ]; then
        run_shard "$name" "" "$@"
        return
    fi
    local rc=0 i
    for i in $(seq 0 $((SHARDS - 1))); do
        run_shard "$name" "$i/$SHARDS" "$@" &
    done
    while [ "$(jobs -rp | wc -l)" -gt 0 ]; do wait -n || rc=1; done
    [ "$rc" -eq 0 ] || return 1
    # Merge the paths the runner REPORTED, not names rebuilt from its rules.
    # Reconstructing them has now failed twice -- once on the "vanilla-" prefix,
    # once on the config suffix -- because the rule lives in benchmark_runner and
    # every copy of it here is a copy that can drift.
    local paths
    paths=$(grep -h "^wrote .*qafd_.*\.json" \
            "$ROOT"/out/probe_${OURS}_$(dump_stem "$name").shard*of${SHARDS}.log \
            | awk -v base="$SUB" '{ print ($2 ~ /^\//) ? $2 : base "/" $2 }')
    [ -n "$paths" ] || { echo "    FATAL: no shard reported a dump path" >&2; return 1; }
    merge_shards $paths
}

# Merge one arm's shards. The destination is derived from the shard filenames by
# removing the .shardIofN segment, so it cannot disagree with what was written.
merge_shards () {
    $PY_MBUZAI - "$@" <<'PYEOF'
import json, sys, os, re
parts = sys.argv[1:]
dests = {re.sub(r"\.shard\d+of\d+\.json$", ".json", p) for p in parts}
if len(dests) != 1:
    raise SystemExit(f"FATAL: shards disagree on destination: {sorted(dests)}")
dest = dests.pop()
merged, seen = {}, 0
for part in parts:
    with open(part) as f:
        d = json.load(f)
    seen += len(d)
    merged.update(d)
if len(merged) != seen:
    raise SystemExit(f"FATAL: shards overlap — {seen} rows collapsed to {len(merged)}")
with open(dest, "w") as f:
    json.dump(merged, f)
print(f"    merged {len(parts)} shards -> {os.path.basename(dest)} ({len(merged)} questions)")
PYEOF
}

arm_flags () {   # arm name -> benchmark_runner flags
    local a=$1 mult
    case $a in
        vanilla)   ;;
        dpr)       printf '%s\n' --retrieval_mode dpr ;;
        # trained w(u,v,q). edgemodel<beta> sets its dynamic range.
        # em<beta> and edgemodel<beta> are the same arm; "edgemodel" reads as
              # "edgemode1" often enough to be worth an alias.
        edgemodel*) printf '%s\n' --edge_model_file "$EDGE_MODEL" \
                        --edge_model_beta "${a#edgemodel}" ;;
        emshuf*)   printf '%s\n' --edge_model_file "$EDGE_MODEL_SHUF" \
                        --edge_model_beta "${a#emshuf}" --edge_model_tag emshuf ;;
        em*)       printf '%s\n' --edge_model_file "$EDGE_MODEL" \
                        --edge_model_beta "${a#em}" ;;
        # oracle<mult>[ent][seed] — ent drops the gold PASSAGE node (which is also a
        # ranking target) so the arm measures steering rather than mass landing on
        # the answer; seed spends the multiplier on source mass instead of edges.
        oracle*)
            mult=${a#oracle}; mult=${mult%seed}; mult=${mult%ent}
            printf '%s\n' --oracle_gold_file "$ORACLE" --oracle_edge_mult "$mult"
            case $a in *ent*)  printf '%s\n' --oracle_nodes entities ;; esac
            case $a in *seed*) printf '%s\n' --oracle_site  seeds    ;; esac ;;
        # The sigma_max arms: the query set applied at propagation vs selection.
        subqedges) printf '%s\n' --subq_file "$SUBQ" --subq_scope edges ;;
        subqseeds) printf '%s\n' --subq_file "$SUBQ" --subq_scope seeds ;;
        expb*)     printf '%s\n' --qafd_weight_scheme exp --exp_beta "${a#expb}" ;;
        sink*)     printf '%s\n' --qa_sink_gamma "${a#sink}" ;;
        accum*)    printf '%s\n' --qa_accum_gamma "${a#accum}" ;;
        *)         echo "unknown arm: $a" >&2; return 1 ;;
    esac
}

# Recover shards whose merge failed: the retrieval is already paid for, so
# re-running it to fix a filename would be pure waste.
if [ -n "${MERGE_ONLY:-}" ]; then
    for arm in $ARMS; do
        echo "    $arm"
        paths=$(grep -h "^wrote .*qafd_.*\.json" \
                "$ROOT"/out/probe_${OURS}_$(dump_stem "$arm").shard*of${SHARDS}.log \
                | awk -v base="$SUB" '{ print ($2 ~ /^\//) ? $2 : base "/" $2 }')
        [ -n "$paths" ] || { echo "    no shard logs for $arm at SHARDS=$SHARDS" >&2; continue; }
        merge_shards $paths
    done
    exit 0
fi

echo "==> arms ($_WORKERS workers x $_THREADS threads of $_CORES cores, JOBS=$JOBS SHARDS=$SHARDS, ${PROFILE:+profile=$PROFILE }alpha=$ALPHA eps=$EPSILON scheme=$WEIGHT_SCHEME b=$HYBRID_B ltk=$LINKING_TOP_K maxiter=$MAXITER batch_push=${BATCH_PUSH:-off})"
PENDING=()
for arm in $ARMS; do
    arm_flags "$arm" >/dev/null || exit 1     # validate every name before starting
    dump="$WORKDIR/qafd_${OURS}_$(dump_stem "$arm").json"
    if [ -f "$dump" ] && [ -z "${FORCE:-}" ]; then
        echo "    have $(basename "$dump"), skipping (FORCE=1 to rerun)"
    else
        PENDING+=("$arm")
    fi
done

if [ ${#PENDING[@]} -gt 0 ]; then
    echo "    running: ${PENDING[*]}"
    started=0
    for arm in "${PENDING[@]}"; do
        # `|| true`: wait -n reports the finished job's status, and under set -e a
        # bare one would abort the whole sweep the moment any single arm failed.
        # Failures are counted in the drain loop below and reported together.
        while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n || true; done
        mapfile -t flags < <(arm_flags "$arm")
        run_arm "$arm" "${flags[@]}" &
        started=$((started + 1))
        # Each process loads the whole index (graph + entity/passage/fact vectors)
        # before it does any work. Starting them in lockstep means every copy of
        # that read lands at once; a short stagger keeps the box responsive.
        [ "$started" -lt "${#PENDING[@]}" ] && sleep "${STAGGER:-20}"
    done
    fails=0
    while [ "$(jobs -rp | wc -l)" -gt 0 ]; do wait -n || fails=$((fails + 1)); done
    [ "$fails" -eq 0 ] || { echo "$fails arm(s) failed — see out/probe_${OURS}_*.log" >&2; exit 1; }
fi

echo "==> did the weights steer anything? (read this FIRST)"
$PY_MBUZAI "$ROOT/scripts/report_edge_contrast.py" "$ROOT"/out/edgestats_${OURS}_*.json

echo "==> recall"
# Score exactly the arms this invocation covers, named through dump_stem.
# Globbing the directory pulls in per-shard slices and runs from other configs
# and question counts, and score_qafd intersects question ids across everything
# it is given -- one 100-question dump in the list silently collapses every arm
# to 100 questions.
_score=("$WORKDIR/qafd_${OURS}_$(dump_stem vanilla).json")
for arm in $ARMS; do
    _f="$WORKDIR/qafd_${OURS}_$(dump_stem "$arm").json"
    [ -f "$_f" ] && [ "$_f" != "${_score[0]}" ] && _score+=("$_f")
done
if [ -f "${_score[0]}" ] && [ ${#_score[@]} -gt 1 ]; then
    $PY_MBUZAI "$ROOT/scripts/score_qafd.py" $OURS \
        --baseline "$(dump_stem vanilla)" --runs "${_score[@]}"
else
    echo "    (need $(dump_stem vanilla) plus one more arm of the same config)"
fi

cat <<'EOF'

How to read the result:

  oracle flat at every magnitude, routing_cv confirms it steered mass
      -> the site is inert. A trained edge scorer cannot help, because the
         oracle is its ceiling. Report the null; do not build the model.

  oracle responds, exp captures the same gain
      -> dynamic range was the blocker and one hyperparameter fixes it.
         No model needed.

  oracle responds, exp misses it
      -> the only case where training is worth doing, and the oracle sweep has
         already told you the output range to train toward.

A null arm whose routing_cv matches vanilla's proves nothing — the weight never
reached the routing distribution. Read the contrast table before the recall one.
EOF
