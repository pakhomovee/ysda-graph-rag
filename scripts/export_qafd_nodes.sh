#!/usr/bin/env bash
# Dump QAFD's graph and frozen vectors for the offline learnability probe.
#
#     bash scripts/export_qafd_nodes.sh
#
# The oracle sweep showed the edge-weight site responds (+0.114 @10 at 1000x), i.e.
# "if you knew which edges mattered, it would help". It cannot show whether you CAN
# know. Every scorer available here is a function of (h_u, h_v, h_q) over frozen
# mpnet vectors, and the graph carries no relation types -- kg_builder adds edges
# with a `weight` attribute and nothing else -- so there is no relation vocabulary
# for a learned matcher to use the way GNN-RAG's omega(q, r) does.
#
# That question is answerable offline: no retrieval, no LLM, CPU, minutes. This
# script produces the input; scripts/probe_learnability.py answers it.
#
# Runs benchmark_runner with --export_nodes, which prepares the retriever, writes
# the dump and exits before retrieval. No server needed.
set -euo pipefail

OURS=musique
PY=${PY:-python3}                 # QAFD env
EMB=${EMB:-mpnet}
SAVE_DIR=${SAVE_DIR:-outputs/$OURS}
LLM_MODEL=${LLM_MODEL:-${LOCAL_LLM_MODEL:-}}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB="$ROOT/third_party/QAFD-RAG"
DEST="$ROOT/out/qafd_nodes_${OURS}.npz"

[ -d "$SUB/src" ] || { echo "submodule missing — run: bash scripts/setup_qafd.sh" >&2; exit 1; }

# Check the flag exists before doing any work. `git pull` updates the patch file
# but does not reapply it to the submodule working tree, so a stale tree fails
# here on argparse — after the index has already been loaded.
_help=$( cd "$SUB" && $PY src/passage_entity/benchmark_runner.py --help 2>&1 ) || {
    echo "benchmark_runner.py --help failed:" >&2; echo "$_help" | tail -5 >&2; exit 1; }
case "$_help" in
    *--export_nodes*) ;;
    *) cat >&2 <<'EOF'
benchmark_runner.py does not accept --export_nodes — the submodule has a stale patch.
  git pull updates patches/qafd_sigma_max.patch; it does not reapply it.
  Fix:  bash scripts/setup_qafd.sh
EOF
       exit 1 ;;
esac
[ -n "$LLM_MODEL" ] || { cat >&2 <<'EOF'
set LOCAL_LLM_MODEL (or LLM_MODEL) to the id the index was built with. No server is
contacted here, but working_dir is derived as <save_dir>/<llm>_<emb>, so the value
still selects which KG gets read.
EOF
exit 1; }

# Same derivation as run_qafd_probe.sh: ask their config, do not reimplement it.
# config.py is loaded by path because importing src.passage_entity.config runs
# src/__init__.py, which pulls in the whole QAFD dependency tree.
WORKDIR=$( cd "$SUB" && $PY - "$OURS" "$SAVE_DIR" "$LLM_MODEL" "$EMB" <<'PYEOF'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("_cfg", "src/passage_entity/config.py")
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)
ds, save_dir, llm, emb = sys.argv[1:5]
print(cfg.PassageEntityConfig(dataset=ds, save_dir=save_dir,
                              llm_model=llm, embedding_model_key=emb).working_dir)
PYEOF
)
[ -d "$SUB/$WORKDIR" ] || { echo "no index at $SUB/$WORKDIR — check SAVE_DIR / LLM_MODEL / EMB" >&2; exit 1; }
echo "==> exporting from $WORKDIR"

# By path, not -m: the runner's namespace-package bypass of src/__init__.py only
# gets to run if the package is never imported. See run_qafd_probe.sh.
( cd "$SUB" && PYTHONPATH="$ROOT:${PYTHONPATH:-}" MBUZAI_ST_DEVICE="${ST_DEVICE:-cpu}" \
  $PY src/passage_entity/benchmark_runner.py \
    --dataset $OURS \
    --save_dir "$SAVE_DIR" \
    --llm_model "$LLM_MODEL" \
    --embedding_model "$EMB" \
    --skip_qa \
    --export_nodes "$DEST" )

echo
echo "next (mbuzai env, CPU):"
echo "    python scripts/probe_learnability.py $OURS"
