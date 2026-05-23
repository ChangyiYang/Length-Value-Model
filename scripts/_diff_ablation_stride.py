"""Aggregate stride-ablation runs into one table.

Reads results/ablation/stride/stride_<N>/ for N in {1,2,5,10} and prints
speed (e2e wall clock, throughput, t_lvm_apply_outer_ms_mean) plus
correctness (pass@1 expected, pass@16 expected, avg completion tokens).
"""
import json
import pathlib

ROOT = pathlib.Path("results/ablation/stride")
STRIDES = [1, 2, 5, 10]


def load_vanilla():
    d = ROOT / "vanilla_topk5"
    if not d.exists():
        return None
    tag = "vanilla_topk5_q50_n16"
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
        "label": "vanilla (no LVM)",
        "stride": "—",
        "wall_clock_s": wall,
        "pass@1_exp": (s.get("pass_at_k_expected") or {}).get("1"),
        "pass@2_exp": (s.get("pass_at_k_expected") or {}).get("2"),
        "pass@4_exp": (s.get("pass_at_k_expected") or {}).get("4"),
        "pass@8_exp": (s.get("pass_at_k_expected") or {}).get("8"),
        "pass@16_exp": (s.get("pass_at_k_expected") or {}).get("16"),
        "acc_first": s.get("acc_first"),
        "avg_completion_per_choice": (s.get("token_usage") or {}).get(
            "avg_completion_tokens_per_choice_assuming_total_over_n"
        ),
    }


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
        "label": f"LVM stride={stride}",
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


vanilla = load_vanilla()
lvm_rows = [r for r in (load_one(s) for s in STRIDES) if r is not None]
rows = ([vanilla] if vanilla is not None else []) + lvm_rows
if not rows:
    print(f"No data under {ROOT}")
    raise SystemExit(1)

# Two refs: vanilla (LVM off) for slowdown, stride=1 for correctness delta
stride1 = next((r for r in lvm_rows if r["stride"] == 1), lvm_rows[0] if lvm_rows else None)
vanilla_wall = (vanilla or {}).get("wall_clock_s") or 0
stride1_wall = (stride1 or {}).get("wall_clock_s") or 0
stride1_pass1 = (stride1 or {}).get("pass@1_exp")
stride1_avg = (stride1 or {}).get("avg_completion_per_choice")

print("=" * 110)
print("LVM stride ablation vs vanilla baseline (GSM8K 50Q x n=16, max_tokens=6000, centered_exp scale=0 for LVM rows)")
print("All runs: top-k=5, min-p=0.01, temp=1.0, top-p=1.0; server restart between rows; same session")
print("=" * 110)

print()
print("Speed:")
hdr = f"  {'config':<22}  {'wall_clock_s':>13}  {'slow vs vanilla':>16}  {'speedup vs s=1':>14}  {'tok/s (avg)':>12}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for r in rows:
    w = r.get("wall_clock_s") or 0
    slow = f"{w/vanilla_wall:.2f}x" if vanilla_wall and w else "n/a"
    speedup = f"{stride1_wall/w:.2f}x" if stride1_wall and w else "n/a"
    avg = r.get("avg_completion_per_choice") or 0
    # tok/s = avg_completion * 50 questions * 16 samples / wall_clock
    tok_s = (avg * 50 * 16 / w) if w else 0
    print(f"  {r['label']:<22}  {w:>13.2f}  {slow:>16}  {speedup:>14}  {tok_s:>12.1f}")

print()
print("Correctness:")
hdr = f"  {'config':<22}  {'pass@1':>8}  {'pass@2':>8}  {'pass@4':>8}  {'pass@8':>8}  {'pass@16':>8}  {'acc_first':>10}  {'avg_len':>9}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for r in rows:
    print(f"  {r['label']:<22}  "
          f"{r.get('pass@1_exp', 0):>8.4f}  "
          f"{r.get('pass@2_exp', 0):>8.4f}  "
          f"{r.get('pass@4_exp', 0):>8.4f}  "
          f"{r.get('pass@8_exp', 0):>8.4f}  "
          f"{r.get('pass@16_exp', 0):>8.4f}  "
          f"{r.get('acc_first', 0):>10.4f}  "
          f"{r.get('avg_completion_per_choice', 0):>9.3f}")

print()
print("Δ vs LVM stride=1 (the unmodified LVM run):")
hdr = f"  {'config':<22}  {'Δ pass@1':>9}  {'Δ pass@16':>10}  {'Δ avg_len':>10}  {'Δ avg_len %':>12}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for r in rows:
    if stride1 is None or r is stride1:
        continue
    p1 = r.get("pass@1_exp")
    p16 = r.get("pass@16_exp")
    avg = r.get("avg_completion_per_choice")
    if p1 is None or p16 is None or avg is None:
        continue
    dp1 = p1 - stride1_pass1
    dp16 = p16 - (stride1.get("pass@16_exp") or 0)
    dl = avg - (stride1_avg or 0)
    dl_pct = (dl / stride1_avg * 100) if stride1_avg else 0
    print(f"  {r['label']:<22}  {dp1:>+9.4f}  {dp16:>+10.4f}  {dl:>+10.3f}  {dl_pct:>+11.2f}%")
