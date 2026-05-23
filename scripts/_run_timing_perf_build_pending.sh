#!/usr/bin/env bash
# Cluster-local wrapper for the build_pending perf-branch timing comparison.
# Run from a checkout of the `lenvm-perf-build-pending` branch on liquid-gpu-055.
#
# Stages:
#   1) smoke (3Q×n=4, ~minute) to sanity-check the patched server boots and the
#      LenVM path still produces non-empty responses.
#   2) full (50Q×n=16, paper config) so the timing.jsonl population is large
#      enough to beat the noise band on t_lvm_build_pending_ms.
#
# Outputs land under results/timing/perf_build_pending_{smoke,full}/. Compare
# against the matching dirs produced by `_run_timing_smoke.sh` /
# `_run_timing_full.sh` on the base branch (lenvm-timing-analysis).

set -euo pipefail
cd /home/changyi/Length-Value-Model

export CUDA_HOME="/home/changyi/miniconda3/envs/bump-python-312"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

set -a
source .env
set +a

export CACHE_ROOT="/home/changyi/Length-Value-Model/cache"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export XDG_CACHE_HOME="$CACHE_ROOT"

STAGE="${STAGE:-both}"   # smoke | full | both

run_smoke() {
  echo "==> stage: smoke"
  MAX_QUESTIONS=3 MAX_CONCURRENCY=10 N_SAMPLES=4 MAX_TOKENS=1500 \
    RESULTS_DIR=./results/timing/perf_build_pending_smoke \
    bash scripts/inference/lenvm_timing.sh
}

run_full() {
  echo "==> stage: full"
  RESULTS_DIR=./results/timing/perf_build_pending_full \
    bash scripts/inference/lenvm_timing.sh
}

case "$STAGE" in
  smoke) run_smoke ;;
  full)  run_full ;;
  both)  run_smoke && run_full ;;
  *)     echo "Unknown STAGE=$STAGE (smoke | full | both)" >&2; exit 1 ;;
esac

echo "Done. Results in results/timing/perf_build_pending_*"
echo
echo "Headline rows (t_lvm_build_pending_ms_mean):"
for d in results/timing/perf_build_pending_*; do
  test -f "$d/summary.csv" || continue
  printf '  %s : ' "$d"
  python -c "
import csv, sys
with open('$d/summary.csv') as f:
    r = csv.DictReader(f)
    rows = list(r)
for row in rows:
    tag = row['tag']
    v = row.get('t_lvm_build_pending_ms_mean') or ''
    if tag == 'lenvm' and v:
        print(f'{float(v):.3f} ms')
        break
"
done
