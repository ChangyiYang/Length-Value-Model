#!/usr/bin/env bash
# ABLATION (plan §0 #1, NOT for production): measure how invoking LVM
# only every N decode steps trades speed vs correctness.
#
# For each stride N in {1, 2, 5, 10}:
#   1) launch SGLang+LenVM server with LENVM_STRIDE=N env
#   2) run sample_eval on GSM8K 50Q x n=16 with centered_exp scale=0
#      (same flags as scripts/inference/demo_tradeoff.sh's centered_exp_0 row)
#   3) record wall-clock + summary.json
#
# Output: results/ablation/stride/stride_<N>/
#
# After all strides finish, run scripts/_diff_ablation_stride.py to
# aggregate into one table.

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

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-10020}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
LENVM_MODEL="${LENVM_MODEL:-./models/namezz/lvm-math-0402-a-qwen2.5-7b-instruct-b-qwen2.5-1.5b-instruct}"
STRIDES=(${STRIDES:-1 2 5 10})

OUT_ROOT="${OUT_ROOT:-./results/ablation/stride}"
mkdir -p "$OUT_ROOT"

run_for_stride() {
  local STRIDE="$1"
  local OUT_DIR="$OUT_ROOT/stride_$STRIDE"
  mkdir -p "$OUT_DIR"
  echo "==================== stride=$STRIDE ===================="

  # ---- launch server with stride env -------------------------------------
  source .venv-infer/bin/activate
  LENVM_STRIDE="$STRIDE" \
  python -m sglang.launch_server \
    --model-path "$BASE_MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --tp-size 1 --dp-size 1 \
    --context-length 30000 \
    --enable-lvm-guided-sampling \
    --lvm-guided-inproc \
    --lvm-guided-inproc-model-path "$LENVM_MODEL" \
    --lvm-guided-inproc-json-model-override-args '{"architectures":["Qwen2ForLengthValueModel"]}' \
    --disable-overlap-schedule \
    --mem-fraction-static 0.4 \
    --lvm-guided-inproc-mem-fraction-static 0.4 \
    --lvm-guided-fn sglang.srt.lvm.lvm_guided_sampling:lvm_combined_guidance \
    >"$OUT_DIR/server.log" 2>&1 &
  local SERVER_PID=$!
  trap "kill $SERVER_PID 2>/dev/null || true; wait 2>/dev/null || true" EXIT

  for _ in $(seq 1 600); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
    sleep 1
  done
  echo "server ready (stride=$STRIDE)"

  # Verify the warning fired (confirms env var was picked up)
  if [[ "$STRIDE" != "1" ]]; then
    if ! grep -q "LENVM_STRIDE=$STRIDE" "$OUT_DIR/server.log"; then
      echo "WARNING: server.log did not show 'LENVM_STRIDE=$STRIDE'; sampler may not have picked up env" >&2
    fi
  fi

  # ---- run sample_eval (centered_exp scale=0, the paper config) ----------
  source .venv-eval/bin/activate
  local START_T=$(date +%s.%N)
  local TAG="stride_${STRIDE}_lenvm_q50_n16_centered_exp_0"
  python -m inference.tradeoff.sample_eval \
    --dataset-name gsm8k \
    --server-url "http://127.0.0.1:$PORT" \
    --output-dir "$OUT_DIR" \
    --tag "$TAG" \
    --stage all \
    --max-questions 50 --max-concurrency 50 --request-timeout 600000 \
    --max-tokens 6000 --temperature 1.0 --top-p 1.0 --n 16 \
    --http-backend aiohttp --top-k 5 --min-p 0.01 \
    --value-scale 0 --value-mode centered_exp --value-gamma 0.997 \
    >"$OUT_DIR/sample_eval.log" 2>&1
  local END_T=$(date +%s.%N)
  echo "$(echo "$END_T - $START_T" | bc) wall_clock_s" >"$OUT_DIR/wall_clock.txt"
  echo "stride=$STRIDE done in $(cat $OUT_DIR/wall_clock.txt)"

  # ---- teardown -----------------------------------------------------------
  kill "$SERVER_PID" 2>/dev/null || true
  wait 2>/dev/null || true
  trap - EXIT
  # Brief settle so GPU mem is fully released before next launch
  sleep 5
}

for s in "${STRIDES[@]}"; do
  run_for_stride "$s"
done

echo
echo "All strides done. Aggregating..."
python3 scripts/_diff_ablation_stride.py
