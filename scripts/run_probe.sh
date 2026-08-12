#!/usr/bin/env bash
# Offline learnability probe: control and real run, in parallel, using the box.
#
#     bash scripts/run_probe.sh
#     CORES=32 bash scripts/run_probe.sh          # cap the footprint
#
# Six trainings total -- 3 models x {shuffle control, real run} -- and they are
# all independent, so they all run at once.
#
# Why CPU at all: the models are tiny (diagonal 768 params, low-rank 2x768x64,
# MLP ~200k) over ~77k rows. That is seconds to minutes of BLAS, the GPU on this
# box is committed to vLLM, and numpy-only keeps the probe dependency-free and
# testable. The work is already threaded inside each matmul; what was missing is
# that the models trained one after another and the two runs did too.
#
# Thread allocation matters more than it looks. OpenBLAS sizes its pool when
# numpy is imported, so OMP_NUM_THREADS has to be set BEFORE python starts --
# setting it inside the script would be too late. And BLAS scaling on these
# shapes flattens well before 96 threads, so 6 workers x 16 threads beats
# 1 worker x 96 by a wide margin. Left unset, each of the 6 would try to grab all
# 96 and thrash.
set -euo pipefail

OURS=${1:-musique}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-python3}
CORES=${CORES:-$(nproc)}
WORKERS=6                       # 3 models x 2 runs
THREADS=$(( CORES / WORKERS ))
[ "$THREADS" -lt 1 ] && THREADS=1

export OMP_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export NUMEXPR_NUM_THREADS=$THREADS
# Question encoding is a torch forward pass over 1000 short texts. Torch reads
# OMP_NUM_THREADS for intra-op parallelism, so it is already covered above and
# the two copies of mpnet will not fight over the whole box.

echo "==> $CORES cores -> $WORKERS workers x $THREADS threads"
[ -f "$ROOT/out/qafd_nodes_${OURS}.npz" ] || {
    echo "missing out/qafd_nodes_${OURS}.npz — run: bash scripts/export_qafd_nodes.sh" >&2
    exit 1; }

cd "$ROOT"
$PY scripts/probe_learnability.py "$OURS" --shuffle-control --jobs 3 \
    >out/probe_${OURS}_control.log 2>&1 &
ctl=$!
$PY scripts/probe_learnability.py "$OURS" --jobs 3 \
    >out/probe_${OURS}_real.log 2>&1 &
real=$!

fail=0
wait $ctl || fail=1
wait $real || fail=1

echo
echo "======================= CONTROL (must be ~0.50) ======================="
sed -n '/held-out questions/,$p' out/probe_${OURS}_control.log
echo
echo "========================== REAL RUN =================================="
sed -n '/held-out questions/,$p' out/probe_${OURS}_real.log

[ "$fail" -eq 0 ] || { echo "a run failed — see out/probe_${OURS}_*.log" >&2; exit 1; }

# The control gates the real run. Refuse to let a contaminated result be read as
# a finding: this exact failure already happened once, with random negatives
# giving 0.60-0.63 on permuted labels.
if grep -q "CONTROL FAILED" out/probe_${OURS}_control.log; then
    echo
    echo "*** The control failed. The real run above is NOT interpretable." >&2
    exit 1
fi
