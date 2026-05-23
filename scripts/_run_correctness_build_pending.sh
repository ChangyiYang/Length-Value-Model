#!/usr/bin/env bash
# Cluster-local correctness gate for the build_pending perf branch.
# Run AFTER `scripts/_run_timing_perf_build_pending.sh` so the responses.jsonl
# files already exist on disk.
#
# What it does:
#   - Re-runs sample_eval with --stage eval against the timing run's
#     responses.jsonl to compute pass@1 / pass@16 / correct_first / correct_any.
#   - Writes one summary JSON per timing run under results/correctness/.
#   - If RESULTS_BASELINE is set to a baseline summary.json, prints a delta.
#
# Caveat (plan §1 bitwise gate): inference/tradeoff/sample_eval.py does not
# pipe a seed through to the SGLang server, so we cannot do strict bitwise
# diff across two runs without first extending that. This script is the
# pass@k half of the §1 correctness gate only.

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

DATASET="${DATASET:-gsm8k}"
TIMING_DIR="${TIMING_DIR:-./results/timing/perf_build_pending_full}"
TAG="${TAG:-lenvm_q50_n16_p1.0_topk5_minp0.01}"
OUT_DIR="${OUT_DIR:-./results/correctness/perf_build_pending}"
RESULTS_BASELINE="${RESULTS_BASELINE:-}"

if [[ ! -f "$TIMING_DIR/${DATASET}.${TAG}.responses.jsonl" ]]; then
  echo "ERROR: expected responses jsonl not found: $TIMING_DIR/${DATASET}.${TAG}.responses.jsonl" >&2
  echo "Run scripts/_run_timing_perf_build_pending.sh first." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

source .venv-eval/bin/activate

# stage=eval re-uses the existing responses.jsonl; no server needed.
python -m inference.tradeoff.sample_eval \
  --dataset-name "$DATASET" \
  --server-url http://127.0.0.1:0 \
  --output-dir "$TIMING_DIR" \
  --tag "$TAG" \
  --stage eval \
  --max-questions 50 \
  --n 16 \
  --top-p 1.0 \
  --top-k 5 \
  --min-p 0.01 \
  --max-tokens 6000 \
  --temperature 1.0

# sample_eval writes <output-dir>/<dataset>.<tag>.summary.json
SUMMARY_SRC="$TIMING_DIR/${DATASET}.${TAG}.summary.json"
SUMMARY_DST="$OUT_DIR/${DATASET}.${TAG}.summary.json"
cp "$SUMMARY_SRC" "$SUMMARY_DST"
echo "Wrote $SUMMARY_DST"

echo
echo "== Pass@k headline =="
python -c "
import json, sys
s = json.load(open('$SUMMARY_DST'))
keys = ('n_questions','n_correct_first','n_correct_any','sum_pass_k_expected','sum_pass_k_firstk','sum_completion_tokens','sum_total_tokens')
for k in keys:
    if k in s:
        print(f'{k}: {s[k]}')
"

if [[ -n "$RESULTS_BASELINE" && -f "$RESULTS_BASELINE" ]]; then
  echo
  echo "== Delta vs baseline ($RESULTS_BASELINE) =="
  python -c "
import json
b = json.load(open('$RESULTS_BASELINE'))
p = json.load(open('$SUMMARY_DST'))
def get(d, k): return d.get(k, 'N/A')
for k in ('n_questions','n_correct_first','n_correct_any'):
    print(f'{k:20s}: baseline={get(b,k):>6} patched={get(p,k):>6}')
def fmt(d):
    s = d.get('sum_pass_k_expected') or {}
    return {int(k): float(v) for k,v in s.items()}
b_pk = fmt(b); p_pk = fmt(p)
for k in sorted(set(b_pk) | set(p_pk)):
    bv = b_pk.get(k,0); pv = p_pk.get(k,0)
    nq = max(get(b,'n_questions') or 1, 1)
    bv_n = bv/nq; pv_n = pv/nq
    print(f'pass@{k:<3}_mean : baseline={bv_n:.4f}  patched={pv_n:.4f}  delta={pv_n-bv_n:+.4f}')
"
fi
