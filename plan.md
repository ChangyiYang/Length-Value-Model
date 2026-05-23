# LenVM Inference Optimization Plan (anti-hack guarded)

Target: 降低 LenVM-guided decoding 的 wall-clock 开销，同时**不动 LVM 算法语义**。
当前基线（来自 `inference/timing/README.md` 与 `results/timing/full_q50_n16_7b_math1.5b/summary.csv`，1× H100，GSM8K 50Q × n=16，max_tokens=6000）：

- e2e wall clock 4.55× slowdown vs baseline
- `Sampler.forward` 0.19 ms → 50.31 ms（268×）
- 99% delta 来自 `LvmGuidedSampler.apply`，其中 LenVM forward 36.6 ms / build_pending 12.3 ms / apply_guidance 5.3 ms
- 83% 额外 FLOPs 来自 top_k=5 候选 forwards 经 1.5B value model
- H100 bf16 利用率 ~9%（baseline ~18%）→ 一半 slowdown 是 GPU 欠利用

## 0. Anti-hack 硬约束（不可越过）

agent 在任何 PR / 迭代中**只允许**改 LVM 的**实现/调度/kernel/overlap**；以下属于"改变 LVM 逻辑"，**严禁触碰**：

1. 每个 decoding step 必须仍然为 `LENVM_TOP_K` 个候选各走一次 value-model forward（**不得改成"每 N 个 token 预测一次"、不得改成 stride/skip、不得跨 step 复用上一步的 value**）。
2. `lvm_combined_guidance`（`sglang-LenVM/python/sglang/srt/lvm/lvm_guided_sampling.py:410`）的数学形式不可改 —— `centered_exp` / `value_scale` / `value_gamma` / `min_p` / `top_k` 行为必须保持。
3. Sampler 路径中 pre-LVM softmax → guidance overlay → sample kernel 的顺序与语义不变。
4. `tree_value_extend` 后每个 candidate 都是独立的单 token forward 这一**对外行为**保持；允许在底层把 k 个 candidate **batch 成一次 forward**，前提是输出 logits 与逐个调用**逐元素一致**（bf16 reduction order 偏差不得改变 top-1 选择，见 §4 验证）。
5. 不允许引入 value head 的近似（FP8 量化、低秩、蒸馏小头、heuristic skip）作为"优化"。
6. 不允许通过缩短/裁剪 prompt、改 prefix cache 行为来"加速"。
7. 不允许在 timing 路径打开任何 `--disable-*` 或 fast-path flag 后**只**在 timing 跑、correctness 不跑。每个候选优化都要双跑（timing + correctness）。

如果一个想法看起来需要打破上面任一条 → 停下、写到 plan 末尾的 "out-of-scope" 区里、等人 review，**不要自己绕过**。

## 1. Correctness verification（每个 PR 跑通才能合）

仓库里已有的"正确性"入口（不要新造）：

| 入口 | 跑什么 | 用途 |
| --- | --- | --- |
| `inference/tradeoff/sample_eval.py`（stage `all`，调用 `RuleBasedMathEvaluator.judge`）| GSM8K / MATH 等 dataset 的 pass@k、correct_first、correct_any | LVM 是否还在给出对的答案 |
| `inference/length_prediction/eval_first_token_prediction.py` | 第一 token 预测的长度准确率 | LVM 头本身没被破坏 |
| `scripts/inference/demo_tradeoff.sh` | 上面那个的端到端 wrapper（含 server lifecycle） | 一键回归 |
| `scripts/inference/demo_lifebench.sh` | LIFEBench baseline + LenVM-guided | 长上下文场景守门 |
| `inference/timing/analyze.py` | 读 `*.timing.jsonl` + `*.meta.json` 算 per-step + FLOPs | 性能回归，不是 correctness，但顺手跑 |

**每个性能 PR 强制：**

1. **Bitwise / near-bitwise 守门（必跑）**：固定 `seed`、固定 prompt set（GSM8K 20Q × n=4），对比 patched server vs baseline server 的逐样本 `responses.jsonl`。允许 bf16 reduction order 抖动；不允许任何 sample 的 token 序列出现**不同的第一个 divergent token**超过 1‱（参考阈值，可在第一轮固定后写死）。
2. **pass@k 守门（必跑）**：`sample_eval.py --stage all --dataset gsm8k --n 16 --max-questions 50`，pass@1 / pass@16 与 baseline 偏差 ≤ noise band（先用 main 上的 LenVM 跑 3 次 estimate noise，写进 `results/correctness/baseline.json`，之后 PR 不得低于 baseline − 1σ）。
3. **First-token length prediction（必跑一次/周）**：`demo_length_prediction.sh`，pass 标准 = 与上一周相同。
4. **LIFEBench（PR 影响长上下文路径时必跑）**：`demo_lifebench.sh`。

任何一项失败 → 优化无效，回滚，不要"调参数把它压回去"。

## 2. 在现有 profile 脚本基础上推进（不另起炉灶）

已经存在的 timing infra，直接复用：

- `scripts/inference/lenvm_timing.sh` — 双 server lifecycle，baseline ↔ lenvm，发 `SGLANG_LVM_TIMING_LOG` 到独立 JSONL。
- `scripts/_run_timing_smoke.sh` — 3Q × n=4，~分钟级 smoke。
- `scripts/_run_timing_full.sh` — 50Q × n=16，paper config。
- `scripts/_run_timing_topk_sweep.sh` / `_topk_sweep_0.5b.sh` — k ∈ {1..5} sweep + 聚合。
- `inference/timing/analyze.py` — per-step 分解 + 理论 FLOPs + plots（`per_step_breakdown.png` / `lvm_apply_breakdown.png` / `flops_breakdown.png`）。
- `inference/timing/sweep_analyze.py` — k-vs-metric 聚合。
- Server-side timer：`sglang-LenVM/python/sglang/srt/lvm/timing.py`（已 hook 进 `Sampler.forward` + `LvmGuidedSampler.apply`）。

**约束**：

- 新指标要加 → 加新 section name 进 `timing.py`，**不要**替换/重命名已有的 7 个字段，下游 `analyze.py` 依赖它们。
- 新实验要跑 → 加新的 `scripts/_run_timing_<name>.sh`，**不要**改既有 wrapper 的语义（其他人的 results dir 都对着）。
- 每次优化前先 `_run_timing_smoke.sh` 跑通；正式比对必须用 `_run_timing_full.sh`，否则 noise 盖过 signal。

## 3. 实验环境（SSH）

- 所有真正跑数的实验在 **`changyi@liquid-gpu-055`** 上做；本地只做 plan / 改代码 / git push。
- 远端工作目录：`/home/changyi/Length-Value-Model`（既有 wrapper 已经 hardcode 这个路径，CUDA / venv / cache 路径都对）。
- venvs：`.venv-infer`（server），`.venv-eval`（client / analyze）。
- 节奏：本地 commit → push 到 `lenvm-timing-analysis` 派生的 perf 分支 → ssh 远端 pull → 跑 smoke → 看 summary → 跑 full → 跑 correctness。

```bash
# 远端模板（在 plan agent 里只贴命令，不在本地直接 ssh 执行除非用户授权）
ssh changyi@liquid-gpu-055 'cd Length-Value-Model && git fetch && git checkout <perf-branch> && \
  bash scripts/_run_timing_smoke.sh'
```

## 4. 优化候选优先级（按 ROI × 风险）

来源：timing 报告显示 `LvmGuidedSampler.apply` ≈ 99% 的 per-step delta，且 GPU 利用率只有 ~9%。

**P0（先做，预期增益最大、最不碰逻辑）**

1. **k 个 candidate forwards 合 batch**：当前是 k 次独立单 token forward（README §"Caveats"已明示）。合成一次 batched forward（batch=k，attend 同一 prefix），输出 logits **逐元素等价**到 reduction-order 抖动。验证：bitwise 守门 + pass@k 守门 + `t_lvm_forward_ms` 应显著下降。
2. **LenVM forward 与 base-model forward 的 stream overlap**：当前 `--disable-overlap-schedule` 是为了 timing 干净，但生产路径里这是合法 overlap 机会。先在 `timing.py` 加一段记录 base/lvm stream sync 等待时间，再决定具体改法。
3. **CPU build_pending 优化**：12.3 ms / step 是纯 Python/Host 工作。先 profiler 看哪几行最贵（`llm-torch-profiler-analysis` skill 一把过），再做局部 vectorize / cache。

**P1**

4. **Value head 输出收集合 kernel 化**（`apply_guidance` 5.3 ms 内的 host↔device 拷贝）。
5. **k=5 时的 KV cache 复用**：candidate 共享 prefix，确认底层确实命中而不是重新算 attention。

**P2（架构改动，需要单独 plan 审）**

6. 把 value model 改成与 base model 共享某些 weight tile（仅当**完全不改 LVM 算法语义**）。

每条优化进展记录在 `plan.md` 末尾的 "Tried" 表里（patch 链接、smoke / full / correctness 结果、是否合）。

## 5. Skills 使用约定

- **`llm-torch-profiler-analysis`** — 任何一个优化开工前，先拿 torch profiler 在 LenVM server 跑一次 trace（profile 工具会驱动 server，按 skill 指示），出 kernel + overlap + fuse 三表。结论引到 §4 候选里。
- **`sglang-prod-incident-triage`** — 如果某次优化后 server 起不来 / hang / OOM，用它做 replay-first 复现，不要靠 print 调。
- **`model-pr-history-knowledge`** — 在动 Qwen2/Qwen2.5 相关路径（包括 LenVM 用的 value model wrapper `qwen2_lvm.py`）前查一遍 SGLang 历史 PR，避免重做或踩已 revert 的坑。
- **`sglang-humanize-review`** — 每个优化 PR 提交前对 diff 跑一遍，按人类 maintainer 风格挑刺。
- **`sglang-sota-humanize-loop`** — 留到最后一步：在所有手工优化都跑完、bitwise 守门都过之后，再用它做收敛闭环 / 对齐 vLLM·TRT-LLM 基线。**不要**一开始就丢给它，会拿走方向控制。

## 6. 交付物

- `results/timing/optim_<short-name>/summary.csv` — 每个优化一份。
- `results/correctness/<short-name>.json` — pass@k + bitwise diff 报告。
- `plan.md`（本文件）末尾的 "Tried" 表持续追加。
- 合并的 PR 必须在描述里贴：smoke 通过截图、full summary diff、correctness 守门结果。

## 7. Out-of-scope（碰到这些先停）

- 改 `lvm_combined_guidance` 数学形式。
- 改 top_k / stride / skip 语义。
- 用近似/量化/蒸馏替换 value head。
- 通过改 dataset / prompt 来"加速"。
- 跳过 §1 的 correctness 守门。

## Tried

| 日期 | 改动 | smoke | full e2e ratio | pass@k Δ | 状态 |
| --- | --- | --- | --- | --- | --- |
| 2026-05-23 | **P0-1 verification (no code change)**: 确认 `eval_candidates_batch_gpu` 已经把 `k * B` candidate tokens 合到 **一个** `ForwardBatch`（tree-attention custom mask），README §"Caveats" 说"各候选独立 forward" 是 stale。同步修正 README。 | n/a | n/a | n/a | doc-only ✅ |
| 2026-05-23 | **P0-3: build_pending CPU/sync trim**（commit `9756021`, branch `lenvm-perf-build-pending`）。合 6 个 `.any().item()` → 1 个 bulk gate；干掉 `_gpu_*_chunks` 累加，直接复用 `vals[send_mask]` / `topk_idx[send_mask]`；deterministic-row scatter 改成 `index_fill_` + advanced indexing；`_req_wants_value_guidance` / `_req_has_hard_target` 在 Req 上缓存；`top_p` / `min_p` / `counts==0` 改成 always-run 的 `torch.where`（min_p=0/top_p=1 时为 no-op，省 sync）。算法语义不变。 | smoke clean (server boots, n_steps=343, no crash) | **同一时段 baseline_full_redo vs patched**: e2e 77.26s → **71.25s (-7.8%)**, throughput 3089 → 3340 tok/s (+8.1%), `t_lvm_build_pending_ms` 11.50 → **8.35 ms (-27.4% mean, -41.8% p95)** ← 直接被优化的指标，旧 baseline 对比也是 -27.9%/-40.3%，跨时段一致。`t_lvm_apply_outer_ms` 49.06 → 44.82 (-8.6%). batch_size/output_tokens 变化 <0.5%（算法没漂移）。注：vs 无 LVM 的 ratio 因每次 no-LVM 分母不同而失真，看绝对值更可靠。 | pass@1 expected 0.9025 → 0.9000 (Δ-0.0025, ≪1σ), pass@8 expected 0.9751 → 0.9845, pass@16 expected 1.0 → 1.0, `sum_completion_tokens` 237,724 → 237,926 (+0.09%) | ✅ merged-ready |

### How to validate on `liquid-gpu-055`

```bash
ssh changyi@liquid-gpu-055 'cd Length-Value-Model && \
  git fetch && git checkout lenvm-perf-build-pending && \
  bash scripts/_run_timing_perf_build_pending.sh                   # smoke + full
ssh changyi@liquid-gpu-055 'cd Length-Value-Model && \
  bash scripts/_run_correctness_build_pending.sh                    # pass@k on the full responses
```

To compare against the base branch (`lenvm-timing-analysis`):

```bash
# On the base branch, snapshot baseline timing+responses to its own dir:
ssh changyi@liquid-gpu-055 'cd Length-Value-Model && \
  git checkout lenvm-timing-analysis && \
  RESULTS_DIR=./results/timing/baseline_lenvm_full bash scripts/inference/lenvm_timing.sh
# then on the perf branch, point the correctness diff at the baseline summary:
ssh changyi@liquid-gpu-055 'cd Length-Value-Model && \
  git checkout lenvm-perf-build-pending && \
  RESULTS_BASELINE=./results/correctness/baseline_lenvm_full/gsm8k.lenvm_q50_n16_p1.0_topk5_minp0.01.summary.json \
  bash scripts/_run_correctness_build_pending.sh
```

### Known correctness-gate gap

`inference/tradeoff/sample_eval.py` does not forward `seed` to the SGLang server, so the §1 **strict bitwise** gate cannot run yet — pass@k stability is the only available gate for this PR. Threading `seed` through `sample_eval` (and through `run_timing`) is a separate small PR; track it in `out-of-scope` below if it's not started before the next perf change.

### Post-patch profiler verdict (2026-05-23, single-trace)

Drove `llm-torch-profiler-analysis` against the patched server (commit `b6198eb`) with a **LenVM-active workload** (sample_eval `value_scale=0` / `value_mode=centered_exp`, 10Q × n=8 GSM8K). Trace at `cache/profile_lenvm_b6198eb/1779567996.9350457-TP-0.trace.json.gz`. Driver script: `scripts/_run_profile_build_pending.sh`.

**Kernel-table headline**: nvjet GEMM family ~65% (mostly `_forward_raw`), FlashAttention SM90 ~10%, FusedAddRMSNorm 3.1%, SwiGLU `act_and_mul` 2.2%. The only `_build_pending`-source kernel above 1% is `radixFindKthValues` (top-k) at 0.52 ms / 5 steps ≈ 0.1 ms / step. **Conclusion: post-patch `_build_pending` has no remaining GPU hotspot**; the 8.35 ms/step we still measure is CPU + sync work (per-request Python loop gathering `prefix_ids_send`, plus a few unavoidable `.tolist()`s). Further P0-3 trimming is diminishing returns.

**Overlap-table headline**: single-trace mode couldn't attribute overlap (skill noted "Use mapping/formal mode"). A two-trace run (graph-off mapping vs graph-on formal) is the right shape for **P0-2 stream overlap** evidence; this is the next experiment.

**Largest remaining wall-clock opportunity (= P0-2)**: `t_lvm_forward_ms` is 32 ms/step wall-clock, but no individually-LenVM-tagged kernel crosses 2% of GPU time. Allowing for shared `_forward_raw` site share, actual qwen2_lvm GPU compute is on the order of 1-2 ms/step. The ~30× gap between wall-clock and GPU time is stream-sync stall — exactly what `--disable-overlap-schedule` blocks. P0-2 should target this.

**Out-of-band finding (out-of-scope here)**: fuse-table top row is PR #22392 (CUTLASS FP8 scaled MM replacing nvjet) at 64.8% related GPU time. Applying to the **base** model would shrink the dominant kernel band. Applying to the **value head** is forbidden by §0 #5 (no FP8 on value head). Tracked as a base-model-side perf option, not part of this perf plan.

## Out-of-scope (queue, not in this PR)

- Plumb `seed` from `sample_eval.py` → OpenAI-compatible chat completion `seed` param so the §1 bitwise/near-bitwise diff is actually runnable. Required before any PR that touches GPU reduction order.
- Add `t_lvm_stream_wait_ms` (or similar) to `timing.py` so we can quantify how much wall-clock the main stream wastes waiting on `embed_ready` / `extend_ready`. Prerequisite for P0-2 (stream overlap re-enable). Additive — don't rename the existing 7 keys.
