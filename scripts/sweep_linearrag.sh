#!/usr/bin/env bash
# Sweep LinearRAG's propagation regime, both arms at every cell.
#
#     PY39=$PWD/.venv-linear/bin/python bash scripts/sweep_linearrag.sh
#     LIMIT=50 THRS="0.4 0.2" ITERS=3 PY39=... bash scripts/sweep_linearrag.sh
#
# Why: propagation is multiplicative (next = entity_score * sentence_score) and
# pruned below --iteration_threshold. Seeds enter at ~0.6-0.9 and mpnet
# question-sentence cosines run ~0.3-0.5, so at their default 0.4 a second hop
# needs sigma ~0.75 and effectively never happens. The search is one hop, and a
# gate meant to reach distant entities has nothing to reach.
#
# The prediction under test is a TREND, not a cell: sigma_max's advantage over
# vanilla should grow as the threshold falls. One good cell out of eight is
# multiple comparisons. Both arms move when the regime changes, so always read
# the paired delta, never sigma_max at a new setting against vanilla at the old.
set -euo pipefail

OURS=musique
THEIRS=musique
DEVICE=${DEVICE:-3}
TOPK=${TOPK:-10}
LIMIT=${LIMIT:-}
THRS=${THRS:-"0.4 0.3 0.2 0.1"}
ITERS=${ITERS:-"3 5"}
PY=${PY:-python3}
PY39=${PY39:-python3}
EMB=${EMB:-sentence-transformers/all-mpnet-base-v2}
MAXSEC=${MAXSEC:-900}   # warn past this; the activated set grows fast when thr drops

case "$PY39" in */*) PY39="$(cd "$(dirname "$PY39")" && pwd)/$(basename "$PY39")" ;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB="$ROOT/third_party/LinearRAG"
SUBQ=${1:-$ROOT/out/subq_${OURS}_generated.json}
BYTEXT="$ROOT/out/$(basename "${SUBQ%.json}")_bytext.json"

echo "==> preflight"
grep -q "sigma_max(sentence_embeddings" "$SUB/src/LinearRAG.py" \
    || { echo "LinearRAG is not patched — run: bash scripts/setup_linearrag.sh" >&2; exit 1; }
grep -q "entity_trace" "$SUB/src/LinearRAG.py" \
    || { echo "submodule has an older patch (no entity instrumentation) — run: bash scripts/setup_linearrag.sh" >&2; exit 1; }
[ -f "$SUB/dataset/$THEIRS/chunks.json" ] || { echo "bundle missing" >&2; exit 1; }
[ -f "$SUBQ" ] || { echo "missing $SUBQ" >&2; exit 1; }

echo "==> gold + re-key (once)"
$PY "$ROOT/scripts/prepare_linearrag_gold.py" $OURS --bundle "$SUB/dataset/$THEIRS" >/dev/null
$PY "$ROOT/scripts/export_subq_for_linearrag.py" $OURS --subq "$SUBQ" >/dev/null
echo "    ok"

for IT in $ITERS; do
  for T in $THRS; do
    TAG="thr${T//./}_it${IT}"
    STEM="$ROOT/out/linearrag_${OURS}_${TAG}"
    echo
    echo "############ iteration_threshold=$T  max_iterations=$IT ############"
    START=$(date +%s)
    ( cd "$SUB" && PYTHONPATH="$ROOT:${PYTHONPATH:-}" $PY39 \
        "$ROOT/scripts/run_linearrag_retrieval.py" \
        --dataset_name $THEIRS --device "$DEVICE" --retrieval_top_k "$TOPK" \
        --embedding_model "$EMB" --subq_file "$BYTEXT" \
        --iteration_threshold "$T" --max_iterations "$IT" \
        ${LIMIT:+--limit "$LIMIT"} --out "$STEM" )
    ELAPSED=$(( $(date +%s) - START ))
    echo "    cell took ${ELAPSED}s"
    [ "$ELAPSED" -gt "$MAXSEC" ] && echo "    WARNING: past ${MAXSEC}s; lower thresholds will be slower still"

    compgen -G "${STEM}_*.json" >/dev/null || { echo "no runs for this cell" >&2; continue; }
    $PY "$ROOT/scripts/score_linearrag.py" $OURS --runs "${STEM}"_*.json --ks 2 5 10 \
        | sed -n '/^absolute recall/,/^  \* =/p' | head -20
    $PY "$ROOT/scripts/report_entities.py" $OURS \
        --traces "$ROOT/out/entities_linearrag_${OURS}_${TAG}"_*.json \
        | grep -E "tier |bypassed|beyond the seed" || true
  done
done

cat <<'EOF'

Read the sweep as a trend across thresholds, not as a best cell. If the paired
delta does not grow as iteration_threshold falls, opening propagation is not what
was holding the gate back, and the negative result stands.
EOF
