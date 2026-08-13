#!/usr/bin/env bash
# Serve gpt-oss-20b locally for OpenIE, sized to the card's ACTUAL free memory.
#
#     bash scripts/serve_gptoss.sh                 # device 3, port 5679
#     DEVICE=1 PORT=8000 bash scripts/serve_gptoss.sh
#     RESERVE_MIB=4000 bash scripts/serve_gptoss.sh   # leave room for an encoder
#
# vLLM's --gpu-memory-utilization is a fraction of TOTAL VRAM, not free VRAM, so
# the 0.9 default asks for 36 GB of a 40 GB card and refuses to start whenever a
# neighbour holds anything. This computes the fraction from what is actually
# free, minus RESERVE_MIB for whatever else you intend to run on the same card —
# QAFD's mpnet encoder, for instance.
set -euo pipefail

DEVICE=${DEVICE:-3}
PORT=${PORT:-5679}
MODEL=${MODEL:-openai/gpt-oss-20b}
# The id the server answers to. QAFD derives working_dir from llm_model, so to
# run against the authors' prebuilt KG (kg/multihop/<llm>_<emb>_<dataset>) the
# served id has to equal the <llm> part of that directory name -- otherwise
# llm_model either finds the KG or works for the reranker, never both.
SERVED_NAME=${SERVED_NAME:-$MODEL}
RESERVE_MIB=${RESERVE_MIB:-3000}   # headroom for a co-resident encoder
WEIGHTS_MIB=${WEIGHTS_MIB:-14000}  # gpt-oss-20b MXFP4 weights, with slack
MAXLEN=${MAXLEN:-8192}

read -r FREE TOTAL < <(nvidia-smi --query-gpu=memory.free,memory.total \
    --format=csv,noheader,nounits -i "$DEVICE" | tr -d ',')

USABLE=$(( FREE - RESERVE_MIB ))
UTIL=$(awk -v u="$USABLE" -v t="$TOTAL" 'BEGIN{ printf "%.3f", (u>0? u/t : 0) }')
BUDGET=$(awk -v uu="$UTIL" -v t="$TOTAL" 'BEGIN{ printf "%d", uu*t }')

echo "GPU $DEVICE: ${FREE} MiB free of ${TOTAL}"
echo "  reserving ${RESERVE_MIB} MiB for a co-resident encoder"
echo "  --gpu-memory-utilization ${UTIL}  -> ${BUDGET} MiB budget"

if [ "$BUDGET" -lt "$WEIGHTS_MIB" ]; then
    echo "FATAL: ${BUDGET} MiB cannot hold ${MODEL} (~${WEIGHTS_MIB} MiB of weights)." >&2
    echo "  Free the card, lower RESERVE_MIB, or pick another with DEVICE=n." >&2
    nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader >&2
    exit 1
fi

cat <<EOF

Point QAFD at it once it reports "Application startup complete":
  export LOCAL_LLM_BASE_URL="http://localhost:${PORT}/v1"
  export LOCAL_LLM_MODEL="${SERVED_NAME}"
  export LOCAL_LLM_API_KEY="dummy"

Verify the served id matches LOCAL_LLM_MODEL — a mismatch fails per request, so
indexing would discover it ~11.5k times rather than once:
  curl -s http://localhost:${PORT}/v1/models | python3 -m json.tool

EOF

set -x
CUDA_VISIBLE_DEVICES="$DEVICE" vllm serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --port "$PORT" \
    --gpu-memory-utilization "$UTIL" \
    --max-model-len "$MAXLEN"
