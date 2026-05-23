#!/usr/bin/env bash
# Reproduce scripts/inference/demo_tradeoff.sh against the 7B base +
# 1.5B math LenVM (the only LenVM checkpoint that lives on this box),
# so we can sanity-check the patched build_pending against the
# user-supplied reference table:
#
#   baseline budget sweep (avg_length, pass@1) at budgets 200..10000
#   centered_exp scales -100, -10, -5, -2, 0 (avg_length, pass@1)
#
# Run from a checkout of the perf branch on liquid-gpu-055.

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
OUT_DIR="${OUT_DIR:-./results/tradeoff/Qwen2.5-7B-Instruct_vs_1.5b-math/gsm8k}"

mkdir -p "$OUT_DIR"

# Reference table coverage:
#   baseline      (top_k=-1)
#   centered_exp  (top_k=5, scales below)
SCALES=(-100 -10 -5 -2 0)

# -------------------------------------------------- server up ----------------
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
  >"$OUT_DIR/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true; wait 2>/dev/null || true' EXIT

for _ in $(seq 1 600); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
  sleep 1
done
echo "server ready"

# -------------------------------------------------- sampling -----------------
source .venv-eval/bin/activate

# baseline (no top_k cap, no LenVM custom_params -> LenVM is bypassed)
BASELINE_TAG="baseline_q50_n16_p1.0_topk-1_minp0.01"
python -m inference.tradeoff.sample_eval \
  --dataset-name gsm8k \
  --server-url "http://127.0.0.1:$PORT" \
  --output-dir "$OUT_DIR" \
  --tag "$BASELINE_TAG" \
  --stage all \
  --max-questions 50 --max-concurrency 50 --request-timeout 600000 \
  --max-tokens 6000 --temperature 1.0 --top-p 1.0 --n 16 \
  --http-backend aiohttp --top-k -1 --min-p 0.01 \
  >"$OUT_DIR/baseline.run.log" 2>&1
echo "baseline done"

for SCALE in "${SCALES[@]}"; do
  TAG="lenvm_q50_n16_p1.0_topk5_minp0.01_gamma0.997_centered_exp_${SCALE}"
  python -m inference.tradeoff.sample_eval \
    --dataset-name gsm8k \
    --server-url "http://127.0.0.1:$PORT" \
    --output-dir "$OUT_DIR" \
    --tag "$TAG" \
    --stage all \
    --max-questions 50 --max-concurrency 50 --request-timeout 600000 \
    --max-tokens 6000 --temperature 1.0 --top-p 1.0 --n 16 \
    --http-backend aiohttp --top-k 5 --min-p 0.01 \
    --value-scale "$SCALE" --value-mode centered_exp --value-gamma 0.997 \
    >"$OUT_DIR/lenvm_scale_${SCALE}.run.log" 2>&1
  echo "lenvm scale=$SCALE done"
done

# -------------------------------------------------- budget sweep --------------
python -m inference.tradeoff.budget_eval \
  --responses "$OUT_DIR/gsm8k.${BASELINE_TAG}.responses.jsonl" \
  --tokenizer "$BASE_MODEL" \
  --output-dir "$OUT_DIR/budget_baseline" \
  --budgets 200 250 300 350 400 500 600 700 800 900 1000 1500 2000 5000 10000 \
  >"$OUT_DIR/budget_eval.log" 2>&1
echo "budget eval done"

echo
echo "=================== HEADLINE ==================="
python3 - <<PY
import json, csv, sys, pathlib
out = pathlib.Path("$OUT_DIR")
b_csv = out / "budget_baseline" / "length_vs_passk.csv"
print("-- baseline budget sweep (this run) --")
with open(b_csv) as f:
    rdr = csv.DictReader(f)
    for r in rdr:
        print(f"  budget={r['budget']:>5s}  avg_length={float(r['avg_capped_tokens']):>9.4f}  pass@1={float(r['pass@1']):.6f}")

print()
print("-- centered_exp scales (this run) --")
for s in ["-100","-10","-5","-2","0"]:
    tag = f"lenvm_q50_n16_p1.0_topk5_minp0.01_gamma0.997_centered_exp_{s}"
    p = out / f"gsm8k.{tag}.summary.json"
    if not p.exists():
        print(f"  scale={s:>6}  MISSING summary {p}"); continue
    d = json.load(open(p))
    avg_len = d.get("token_usage", {}).get("avg_completion_tokens_per_choice_assuming_total_over_n")
    pass_at_1_exp = (d.get("pass_at_k_expected") or {}).get("1")
    print(f"  scale={s:>6}  avg_completion_per_choice={avg_len:.4f}  pass@1_expected={pass_at_1_exp:.6f}  acc_first={d.get('acc_first')}")
PY
