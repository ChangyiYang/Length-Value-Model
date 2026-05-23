"""Aggregate stride-ablation runs into one table.

Reads results/ablation/stride/stride_<N>/ for N in {1,2,5,10} and prints
speed (e2e wall clock, throughput, t_lvm_apply_outer_ms_mean) plus
correctness (pass@1 expected, pass@16 expected, avg completion tokens).
"""
import json
import pathlib

ROOT = pathlib.Path("results/ablation/stride")
STRIDES = [1, 2, 5, 10]


def load_one(stride):
    d = ROOT / f"stride_{stride}"
    if not d.exists():
        return None
    tag = f"stride_{stride}_lenvm_q50_n16_centered_exp_0"
    summary_path = d / f"gsm8k.{tag}.summary.json"
    wall_path = d / "wall_clock.txt"
    if not summary_path.exists():
        return None
    s = json.load(open(summary_path))
    wall = None
    if wall_path.exists():
        try:
            wall = float(wall_path.read_text().split()[0])
        except Exception:
            pass
    return {
        "stride": stride,
        "wall_clock_s": wall,
        "n_questions_used": s.get("n_questions_used"),
        "pass@1_exp": (s.get("pass_at_k_expected") or {}).get("1"),
        "pass@2_exp": (s.get("pass_at_k_expected") or {}).get("2"),
        "pass@4_exp": (s.get("pass_at_k_expected") or {}).get("4"),
        "pass@8_exp": (s.get("pass_at_k_expected") or {}).get("8"),
        "pass@16_exp": (s.get("pass_at_k_expected") or {}).get("16"),
        "acc_first": s.get("acc_first"),
        "avg_completion_per_choice": (s.get("token_usage") or {}).get(
            "avg_completion_tokens_per_choice_assuming_total_over_n"
        ),
        "sum_completion_tokens": (s.get("token_usage") or {}).get("sum_completion_tokens"),
    }


rows = [r for r in (load_one(s) for s in STRIDES) if r is not None]
if not rows:
    print(f"No data under {ROOT}")
    raise SystemExit(1)

# Use stride=1 as the reference for correctness deltas
ref = next((r for r in rows if r["stride"] == 1), rows[0])

print("=" * 100)
print("Stride ablation (centered_exp scale=0, GSM8K 50Q x n=16, max_tokens=6000)")
print("=" * 100)

print()
print("Speed:")
print(f"  {'stride':>6}  {'wall_clock_s':>12}  {'speedup vs 1':>14}  {'avg_completion':>16}")
ref_wall = ref.get("wall_clock_s") or 0
for r in rows:
    w = r.get("wall_clock_s")
    sp = f"{ref_wall/w:.3f}x" if w and ref_wall else "n/a"
    avg = r.get("avg_completion_per_choice") or 0
    print(f"  {r['stride']:>6d}  {w if w is not None else 'n/a':>12}  {sp:>14}  {avg:>16.3f}")

print()
print("Correctness (pass@k expected):")
header = "  " + " ".join(f"{n:>10s}" for n in
                          ["stride", "pass@1", "pass@2", "pass@4", "pass@8", "pass@16", "acc_first"])
print(header)
for r in rows:
    line = "  " + " ".join([f"{r['stride']:>10d}"] +
                            [f"{r.get(k, 0):>10.4f}" if r.get(k) is not None else f"{'n/a':>10}"
                             for k in ["pass@1_exp", "pass@2_exp", "pass@4_exp",
                                       "pass@8_exp", "pass@16_exp", "acc_first"]])
    print(line)

print()
print("Correctness delta vs stride=1:")
print(f"  {'stride':>6}  {'Δ pass@1':>9}  {'Δ pass@16':>10}  {'Δ avg_len':>10}  {'Δ avg_len %':>12}")
ref_avg = ref.get("avg_completion_per_choice") or 0
for r in rows:
    if r["stride"] == 1:
        continue
    dp1 = r["pass@1_exp"] - ref["pass@1_exp"] if r["pass@1_exp"] is not None else 0
    dp16 = r["pass@16_exp"] - ref["pass@16_exp"] if r["pass@16_exp"] is not None else 0
    avg = r.get("avg_completion_per_choice") or 0
    dl = avg - ref_avg
    dl_pct = (dl / ref_avg * 100) if ref_avg else 0
    print(f"  {r['stride']:>6d}  {dp1:>+9.4f}  {dp16:>+10.4f}  {dl:>+10.3f}  {dl_pct:>+11.2f}%")
