#!/usr/bin/env bash
# Re-run +C+A at scale=-100 with SGLANG_LVM_TIMING_LOG enabled, so we can
# count how often the fused extend+candidate path actually fires.
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

OUT_DIR=./results/ablation/phase_ac_timing
mkdir -p "$OUT_DIR"

source .venv-infer/bin/activate

SGLANG_LVM_TIMING_LOG="$OUT_DIR/lvm.jsonl" \
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 10020 \
  --tp-size 1 --dp-size 1 --context-length 30000 \
  --enable-lvm-guided-sampling --lvm-guided-inproc \
  --lvm-guided-inproc-model-path ./models/namezz/lvm-math-0402-a-qwen2.5-7b-instruct-b-qwen2.5-1.5b-instruct \
  --lvm-guided-inproc-json-model-override-args '{"architectures":["Qwen2ForLengthValueModel"]}' \
  --disable-overlap-schedule --mem-fraction-static 0.4 --lvm-guided-inproc-mem-fraction-static 0.4 \
  --lvm-guided-fn sglang.srt.lvm.lvm_guided_sampling:lvm_combined_guidance \
  >"$OUT_DIR/server.log" 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null || true; wait 2>/dev/null || true" EXIT
for _ in $(seq 1 600); do curl -sf http://127.0.0.1:10020/v1/models >/dev/null && break; sleep 1; done
echo "server ready"

source .venv-eval/bin/activate
START=$(date +%s.%N)
python -m inference.tradeoff.sample_eval \
  --dataset-name gsm8k --server-url http://127.0.0.1:10020 \
  --output-dir "$OUT_DIR" --tag scale_neg100 --stage all \
  --max-questions 50 --max-concurrency 50 --request-timeout 600000 \
  --max-tokens 6000 --temperature 1.0 --top-p 1.0 --n 16 \
  --http-backend aiohttp --top-k 5 --min-p 0.01 \
  --value-scale -100 --value-mode centered_exp --value-gamma 0.997 \
  >"$OUT_DIR/eval.log" 2>&1
END=$(date +%s.%N)
echo "wall_clock_s=$(echo "$END - $START" | bc)"

kill $SERVER_PID 2>/dev/null || true; wait 2>/dev/null || true
echo
python3 -c "
import json
total = fused = 0
for line in open('$OUT_DIR/lvm.jsonl'):
    try:
        r = json.loads(line)
    except Exception:
        continue
    if r.get('lvm_active'):
        total += 1
        if r.get('lvm_fused_path') == 1:
            fused += 1
print(f'LVM-active steps: {total}')
print(f'Fused path used:  {fused}')
if total > 0:
    print(f'Fusion rate:      {100.0*fused/total:.1f}%')
"
