#!/usr/bin/env bash
# End-to-end patched-LinearRAG run on 2WikiMultiHopQA: gold, arms, score.
#
#     bash scripts/run_linear_wiki.sh
#     bash scripts/run_linear_wiki.sh out/subq_2wikimultihopqa_resolved.json   # oracle arm
#
# Two interpreters are involved and they are not interchangeable. Retrieval runs
# in LinearRAG's Python 3.9 env; everything else runs in the mbuzai env, because
# mbuzai.dataio's annotations do not survive 3.9. Point PY39 at the right one:
#
#     PY39=~/.venv39/bin/python bash scripts/run_linear_wiki.sh
set -euo pipefail

OURS=2wikimultihopqa   # our dataset name (data/, out/)
THEIRS=2wikimultihop  # their bundle directory name — note: no "qa" suffix
DEVICE=${DEVICE:-3}
TOPK=${TOPK:-10}
LIMIT=${LIMIT:-}       # set for a smoke run, e.g. LIMIT=20
PY=${PY:-python3}     # mbuzai env
PY39=${PY39:-python3} # LinearRAG env
EMB=${EMB:-sentence-transformers/all-mpnet-base-v2}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB="$ROOT/third_party/LinearRAG"
BUNDLE="$SUB/dataset/$THEIRS"
SUBQ=${1:-$ROOT/out/subq_${OURS}_generated.json}
STEM="$ROOT/out/linearrag_${OURS}"

echo "==> preflight"
[ -d "$SUB/src" ] || { echo "submodule missing — run: bash scripts/setup_linearrag.sh" >&2; exit 1; }
grep -q "sigma_max(sentence_embeddings" "$SUB/src/LinearRAG.py" \
    || { echo "LinearRAG is not patched — run: bash scripts/setup_linearrag.sh" >&2; exit 1; }
[ -f "$BUNDLE/chunks.json" ] || { cat >&2 <<EOF
missing $BUNDLE/chunks.json
  Fetch it directly — a git clone without git-lfs installed yields pointer
  stubs, not JSON, and the failure surfaces much later as a parse error:
    B=https://huggingface.co/datasets/Zly0523/linear-rag/resolve/main
    mkdir -p $SUB/dataset/$THEIRS
    curl -L -o $SUB/dataset/$THEIRS/chunks.json    \$B/$THEIRS/chunks.json
    curl -L -o $SUB/dataset/$THEIRS/questions.json \$B/$THEIRS/questions.json
EOF
exit 1; }
[ -f "$SUBQ" ] || { echo "missing $SUBQ — run scripts/gen_subq.py $OURS first" >&2; exit 1; }
echo "    ok: patched submodule, bundle, $(basename "$SUBQ")"

echo "==> gold (our labels onto their chunks; their evidence field is empty)"
$PY "$ROOT/scripts/prepare_linearrag_gold.py" $OURS --bundle "$BUNDLE"

echo "==> re-key sub-questions by question text"
$PY "$ROOT/scripts/export_subq_for_linearrag.py" $OURS --subq "$SUBQ"
BYTEXT="$ROOT/out/$(basename "${SUBQ%.json}")_bytext.json"

echo "==> retrieval, both arms off one index (LinearRAG env, from its checkout)"
( cd "$SUB" && PYTHONPATH="$ROOT:${PYTHONPATH:-}" $PY39 \
    "$ROOT/scripts/run_linearrag_retrieval.py" \
    --dataset_name $THEIRS \
    --device "$DEVICE" \
    --retrieval_top_k "$TOPK" \
    --embedding_model "$EMB" \
    --subq_file "$BYTEXT" \
    ${LIMIT:+--limit "$LIMIT"} \
    --out "$STEM" )

echo "==> score"
$PY "$ROOT/scripts/score_linearrag.py" $OURS --runs "${STEM}"_*.json

cat <<'EOF'

Read the paired delta, not the absolute recall: their corpus is 1,354 chunks of
~820 words against our 11,656 of ~80, so absolutes here are not comparable to
baselines.py or eval_subq.py. Both arms share one index, so the delta is clean.
EOF
