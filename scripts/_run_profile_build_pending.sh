#!/usr/bin/env bash
# Drive a torch-profiler capture against the LenVM-enabled server with a
# *real* LenVM-active workload (sample_eval + value_scale + value_mode in
# custom_params). Without those custom_params LenVM is bypassed and the
# trace would not exercise LvmGuidedSampler.apply at all.
#
# Sequence:
#   1) launch SGLang+LenVM server (same flags as lenvm_timing.sh stage-2)
#   2) launch sample_eval against it in the background (10Q × n=8) so the
#      server has ~80 concurrent LVM-guided streams in steady state
#   3) wait for warmup, POST /start_profile with num_steps=5
#   4) wait for trace flush, kill workload + server
#   5) print the trace dir for the analyze script

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

PORT="${PORT:-10020}"
HOST="${HOST:-0.0.0.0}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
LENVM_MODEL="${LENVM_MODEL:-./models/namezz/lvm-math-0402-a-qwen2.5-7b-instruct-b-qwen2.5-1.5b-instruct}"
TRACE_DIR="${TRACE_DIR:-$CACHE_ROOT/profile_lenvm_$(git rev-parse --short HEAD)}"

mkdir -p "$TRACE_DIR"

# --------------------------------------------------------------------------- #
# 1) launch server in background
# --------------------------------------------------------------------------- #

source .venv-infer/bin/activate

python -m sglang.launch_server \
  --model-path "$BASE_MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --tp-size 1 \
  --dp-size 1 \
  --context-length 30000 \
  --enable-lvm-guided-sampling \
  --lvm-guided-inproc \
  --lvm-guided-inproc-model-path "$LENVM_MODEL" \
  --lvm-guided-inproc-json-model-override-args '{"architectures":["Qwen2ForLengthValueModel"]}' \
  --disable-overlap-schedule \
  --mem-fraction-static 0.4 \
  --lvm-guided-inproc-mem-fraction-static 0.4 \
  --lvm-guided-fn sglang.srt.lvm.lvm_guided_sampling:lvm_combined_guidance \
  >"$TRACE_DIR/server.log" 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null || true; wait 2>/dev/null || true" EXIT

# wait for ready
for _ in $(seq 1 600); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; then break; fi
  sleep 1
done
echo "server ready"

# --------------------------------------------------------------------------- #
# 2) launch LenVM-active sample_eval in background
# --------------------------------------------------------------------------- #

source .venv-eval/bin/activate

python -m inference.tradeoff.sample_eval \
  --dataset-name gsm8k \
  --server-url "http://127.0.0.1:$PORT" \
  --output-dir "$TRACE_DIR/workload" \
  --tag profile_drive_lenvm \
  --stage run \
  --max-questions 10 \
  --max-concurrency 50 \
  --request-timeout 600000 \
  --max-tokens 2000 \
  --temperature 1.0 \
  --top-p 1.0 \
  --top-k 5 \
  --min-p 0.01 \
  --n 8 \
  --http-backend aiohttp \
  --value-scale 0 \
  --value-mode centered_exp \
  --value-gamma 0.997 \
  >"$TRACE_DIR/workload.log" 2>&1 &
WORKLOAD_PID=$!

# --------------------------------------------------------------------------- #
# 3) wait for steady-state batch, then arm profile
# --------------------------------------------------------------------------- #

echo "waiting 10s for workload to reach steady state..."
sleep 10

echo "POST /start_profile (num_steps=5)..."
curl -sf -X POST "http://127.0.0.1:$PORT/start_profile" \
  -H 'Content-Type: application/json' \
  -d "$(python -c "import json; print(json.dumps({'output_dir': '$TRACE_DIR', 'num_steps': 5, 'with_stack': True, 'record_shapes': True}))")"
echo

# --------------------------------------------------------------------------- #
# 4) wait for profile capture + flush
# --------------------------------------------------------------------------- #

echo "waiting for trace flush (server auto-stops after num_steps)..."
# 5 steps × ~50ms = 0.25s capture; flush typically ~10-20s on H100
sleep 40

ls -la "$TRACE_DIR"/*.trace.json* 2>/dev/null || ls -la "$TRACE_DIR" || true

echo
echo "TRACE_DIR=$TRACE_DIR"
