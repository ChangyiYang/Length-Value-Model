#!/usr/bin/env bash
# Cartesian sweep: VALUE_SCALE x LENVM_STRIDE, to quantify how stride breaks
# LVM length control at scales where LVM actually applies pressure.
#
# At scale=0 (paper neutral) LVM tilt is mild and stride does little visible
# harm. At scale=-100 (strong "make it shorter") LVM compresses output by
# ~50 tokens; if stride breaks length control we should see avg_length
# creep back toward the vanilla baseline as stride grows.
#
# Output: results/ablation/stride_length/scale_<S>_stride_<N>/

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
SCALES=(${SCALES:-0 -10 -100})
STRIDES=(${STRIDES:-1 2 5 10})

OUT_ROOT="${OUT_ROOT:-./results/ablation/stride_length}"
mkdir -p "$OUT_ROOT"

# ---------------------------------------------- vanilla (LVM model not loaded)
run_vanilla() {
  local OUT_DIR="$OUT_ROOT/vanilla_topk5"
  mkdir -p "$OUT_DIR"
  echo "==================== vanilla (no LVM) ===================="
  source .venv-infer/bin/activate
  python -m sglang.launch_server \
    --model-path "$BASE_MODEL" \
    --host "$HOST" --port "$PORT" \
    --tp-size 1 --dp-size 1 \
    --context-length 30000 \
    --mem-fraction-static 0.4 \
    >"$OUT_DIR/server.log" 2>&1 &
  local SERVER_PID=$!
  trap "kill $SERVER_PID 2>/dev/null || true; wait 2>/dev/null || true" EXIT
  for _ in $(seq 1 600); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
    sleep 1
  done
  echo "server ready (vanilla)"
  source .venv-eval/bin/activate
  local START_T=$(date +%s.%N)
  python -m inference.tradeoff.sample_eval \
    --dataset-name gsm8k \
    --server-url "http://127.0.0.1:$PORT" \
    --output-dir "$OUT_DIR" \
    --tag "vanilla_topk5_q50_n16" \
    --stage all \
    --max-questions 50 --max-concurrency 50 --request-timeout 600000 \
    --max-tokens 6000 --temperature 1.0 --top-p 1.0 --n 16 \
    --http-backend aiohttp --top-k 5 --min-p 0.01 \
    >"$OUT_DIR/sample_eval.log" 2>&1
  local END_T=$(date +%s.%N)
  echo "$(echo "$END_T - $START_T" | bc) wall_clock_s" >"$OUT_DIR/wall_clock.txt"
  echo "vanilla done in $(cat $OUT_DIR/wall_clock.txt)"
  kill "$SERVER_PID" 2>/dev/null || true
  wait 2>/dev/null || true
  trap - EXIT
  sleep 5
}

# ---------------------------------------------- LVM at (scale, stride)
run_cell() {
  local SCALE="$1"
  local STRIDE="$2"
  local OUT_DIR="$OUT_ROOT/scale_${SCALE}_stride_${STRIDE}"
  mkdir -p "$OUT_DIR"
  echo "==================== scale=$SCALE stride=$STRIDE ===================="

  source .venv-infer/bin/activate
  LENVM_STRIDE="$STRIDE" \
  python -m sglang.launch_server \
    --model-path "$BASE_MODEL" \
    --host "$HOST" --port "$PORT" \
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

  source .venv-eval/bin/activate
  local START_T=$(date +%s.%N)
  python -m inference.tradeoff.sample_eval \
    --dataset-name gsm8k \
    --server-url "http://127.0.0.1:$PORT" \
    --output-dir "$OUT_DIR" \
    --tag "scale_${SCALE}_stride_${STRIDE}" \
    --stage all \
    --max-questions 50 --max-concurrency 50 --request-timeout 600000 \
    --max-tokens 6000 --temperature 1.0 --top-p 1.0 --n 16 \
    --http-backend aiohttp --top-k 5 --min-p 0.01 \
    --value-scale "$SCALE" --value-mode centered_exp --value-gamma 0.997 \
    >"$OUT_DIR/sample_eval.log" 2>&1
  local END_T=$(date +%s.%N)
  echo "$(echo "$END_T - $START_T" | bc) wall_clock_s" >"$OUT_DIR/wall_clock.txt"
  echo "scale=$SCALE stride=$STRIDE done in $(cat $OUT_DIR/wall_clock.txt)"

  kill "$SERVER_PID" 2>/dev/null || true
  wait 2>/dev/null || true
  trap - EXIT
  sleep 5
}

run_vanilla
for SCALE in "${SCALES[@]}"; do
  for STRIDE in "${STRIDES[@]}"; do
    run_cell "$SCALE" "$STRIDE"
  done
done

echo
echo "All cells done. Aggregating..."
python3 scripts/_diff_ablation_stride_length.py
