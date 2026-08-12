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
ARMS=${ARMS:-"vanilla oracle10 oracle100 oracle1000 expb2 expb8 expb32 sink4 accum4"}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB="$ROOT/third_party/QAFD-RAG"
ORACLE="$ROOT/out/qafd_oracle_${OURS}.json"

echo "==> preflight"
[ -d "$SUB/src" ] || { echo "submodule missing — run: bash scripts/setup_qafd.sh" >&2; exit 1; }
grep -q "_oracle" "$SUB/src/passage_entity/graph_adapter.py" \
    || { echo "QAFD is not patched for the probe — run: bash scripts/setup_qafd.sh" >&2; exit 1; }
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

echo "==> oracle gold map"
$PY_MBUZAI "$ROOT/scripts/make_qafd_oracle.py" $OURS

run_arm () {
    local name=$1; shift
    local dump="$WORKDIR/qafd_${OURS}_${name}.json"
    if [ -f "$dump" ] && [ -z "${FORCE:-}" ]; then
        echo "    $name: have $(basename "$dump"), skipping (FORCE=1 to rerun)"
        return
    fi
    echo "    $name"
    ( cd "$SUB" && PYTHONPATH="$ROOT:${PYTHONPATH:-}" $PY -m src.passage_entity.benchmark_runner \
        --dataset $OURS \
        --save_dir "$SAVE_DIR" \
        --llm_model "$LLM_MODEL" \
        --llm_base_url "$LLM_BASE_URL" \
        --llm_api_key "$LLM_API_KEY" \
        --embedding_model "$EMB" \
        --retrieval_top_k "$TOPK" \
        --skip_qa \
        --edge_stats_file "$ROOT/out/edgestats_${OURS}_${name}.json" \
        "$@" )
}

echo "==> arms"
for arm in $ARMS; do
    case $arm in
        vanilla)   run_arm vanilla ;;
        oracle*)   run_arm "$arm" --oracle_gold_file "$ORACLE" \
                       --oracle_edge_mult "${arm#oracle}" ;;
        expb*)     run_arm "$arm" --qafd_weight_scheme exp --exp_beta "${arm#expb}" ;;
        sink*)     run_arm "$arm" --qa_sink_gamma "${arm#sink}" ;;
        accum*)    run_arm "$arm" --qa_accum_gamma "${arm#accum}" ;;
        *)         echo "unknown arm: $arm" >&2; exit 1 ;;
    esac
done

echo "==> did the weights steer anything? (read this FIRST)"
$PY_MBUZAI "$ROOT/scripts/report_edge_contrast.py" "$ROOT"/out/edgestats_${OURS}_*.json

echo "==> recall"
$PY_MBUZAI "$ROOT/scripts/score_qafd.py" $OURS --runs "$WORKDIR"/qafd_${OURS}_*.json

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
