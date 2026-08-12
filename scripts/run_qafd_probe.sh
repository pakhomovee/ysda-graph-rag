#!/usr/bin/env bash
# Edge-weight probe on QAFD: does the propagation site have any headroom at all?
#
#     LLM_MODEL=<served-model-id> LLM_BASE_URL=http://localhost:8000/v1 \
#         bash scripts/run_qafd_probe.sh
#     ARMS="vanilla oracle1000" LLM_MODEL=... bash scripts/run_qafd_probe.sh   # subset
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
LLM_MODEL=${LLM_MODEL:-}
LLM_BASE_URL=${LLM_BASE_URL:-http://localhost:8000/v1}
LLM_API_KEY=${LLM_API_KEY:-x}     # vLLM ignores it, the client requires it
ARMS=${ARMS:-"vanilla oracle10 oracle100 oracle1000 expb2 expb8 expb32 sink4 accum4"}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB="$ROOT/third_party/QAFD-RAG"
ORACLE="$ROOT/out/qafd_oracle_${OURS}.json"

echo "==> preflight"
[ -d "$SUB/src" ] || { echo "submodule missing — run: bash scripts/setup_qafd.sh" >&2; exit 1; }
grep -q "_oracle" "$SUB/src/passage_entity/graph_adapter.py" \
    || { echo "QAFD is not patched for the probe — run: bash scripts/setup_qafd.sh" >&2; exit 1; }
[ -n "$LLM_MODEL" ] || { cat >&2 <<'EOF'
set LLM_MODEL to the id your server reports — it selects the index directory as
well as the reranker model, so a different value here silently points at a
different (or absent) KG:
    curl -s "$LLM_BASE_URL/models" | python3 -m json.tool
EOF
exit 1; }

# The fact reranker runs inside the retrieval loop and swallows every exception
# (reranker.py: "except Exception ... generated_facts = []"). An unreachable
# server therefore does not fail the run — it silently drops reranking from all
# 1000 questions and quietly produces a different pipeline than RESULTS.md
# measured. Check once, here, rather than discover it in the deltas.
curl -sf --max-time 10 "$LLM_BASE_URL/models" >/dev/null \
    || { echo "no LLM at $LLM_BASE_URL — the reranker would silently no-op on every query" >&2; exit 1; }

# working_dir is a derived property (save_dir/<llm>_<emb>), not a flag. Ask their
# own config for it rather than reimplementing the derivation in bash.
WORKDIR=$( cd "$SUB" && $PY - "$OURS" "$SAVE_DIR" "$LLM_MODEL" "$EMB" <<'PYEOF'
import sys
sys.path.insert(0, ".")
from src.passage_entity.config import PassageEntityConfig
ds, save_dir, llm, emb = sys.argv[1:5]
print(PassageEntityConfig(dataset=ds, save_dir=save_dir,
                          llm_model=llm, embedding_model_key=emb).working_dir)
PYEOF
)
WORKDIR="$SUB/$WORKDIR"
[ -d "$WORKDIR" ] || { cat >&2 <<EOF
no index at $WORKDIR
  This probe reuses the index that produced the RESULTS.md QAFD rows; it does
  not build one. Check SAVE_DIR / LLM_MODEL / EMB match that run.
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
