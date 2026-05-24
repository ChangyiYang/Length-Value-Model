#!/usr/bin/env bash
# Vanilla SGLang baseline (NO LenVM model loaded, NO LenVM custom_params)
# with the *same* sampling params as the stride ablation, so we can compare
# "LVM off" vs "LVM on at stride N" apples-to-apples.
#
# Differs from the existing baseline_full_redo run by:
#   - top-k=5 (matching the LVM stride sweep), not top-k=-1
#   - LVM model not loaded at server-launch time (max speed)

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
OUT_DIR="${OUT_DIR:-./results/ablation/stride/vanilla_topk5}"
mkdir -p "$OUT_DIR"

source .venv-infer/bin/activate
python -m sglang.launch_server \
  --model-path "$BASE_MODEL" \
  --host "$HOST" --port "$PORT" \
  --tp-size 1 --dp-size 1 \
  --context-length 30000 \
  --mem-fraction-static 0.4 \
  >"$OUT_DIR/server.log" 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null || true; wait 2>/dev/null || true" EXIT

for _ in $(seq 1 600); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
  sleep 1
done
echo "server ready (vanilla, no LVM)"

source .venv-eval/bin/activate
START_T=$(date +%s.%N)
TAG="vanilla_topk5_q50_n16"
python -m inference.tradeoff.sample_eval \
  --dataset-name gsm8k \
  --server-url "http://127.0.0.1:$PORT" \
  --output-dir "$OUT_DIR" \
  --tag "$TAG" \
  --stage all \
  --max-questions 50 --max-concurrency 50 --request-timeout 600000 \
  --max-tokens 6000 --temperature 1.0 --top-p 1.0 --n 16 \
  --http-backend aiohttp --top-k 5 --min-p 0.01 \
  >"$OUT_DIR/sample_eval.log" 2>&1
END_T=$(date +%s.%N)
echo "$(echo "$END_T - $START_T" | bc) wall_clock_s" >"$OUT_DIR/wall_clock.txt"
echo "vanilla done in $(cat $OUT_DIR/wall_clock.txt)"
