"""Phase A+C perf+correctness summary across the cell directories."""
import json
import glob
import pathlib

ROOTS = [
    ("vanilla", pathlib.Path("results/ablation/stride_length/vanilla_topk5"),
     "gsm8k.vanilla_topk5_q50_n16.summary.json"),
    ("pre-PR3 (b6198eb) scale=0",
     pathlib.Path("results/ablation/entropy_threshold/scale_0_thr_0"),
     "gsm8k.scale_0_thr_0.summary.json"),
    ("pre-PR3 (b6198eb) scale=-100",
     pathlib.Path("results/ablation/entropy_threshold/scale_-100_thr_0"),
     "gsm8k.scale_-100_thr_0.summary.json"),
    ("+C scale=0",
     pathlib.Path("results/ablation/phase_c_smoke/scale_0_thr_0"),
     "gsm8k.scale_0_thr_0.summary.json"),
    ("+C scale=-100",
     pathlib.Path("results/ablation/phase_c_smoke/scale_-100_thr_0"),
     "gsm8k.scale_-100_thr_0.summary.json"),
    ("+C+A scale=0",
     pathlib.Path("results/ablation/phase_ac_smoke/scale_0_thr_0"),
     "gsm8k.scale_0_thr_0.summary.json"),
    ("+C+A scale=-100",
     pathlib.Path("results/ablation/phase_ac_smoke/scale_-100_thr_0"),
     "gsm8k.scale_-100_thr_0.summary.json"),
]

rows = []
for label, d, summary_name in ROOTS:
    summary_path = d / summary_name
    wall_path = d / "wall_clock.txt"
    if not summary_path.exists():
        rows.append((label, None, None, None, None))
        continue
    s = json.load(open(summary_path))
    wall = None
    if wall_path.exists():
        try:
            wall = float(wall_path.read_text().split()[0])
        except Exception:
            pass
    tu = s.get("token_usage", {})
    pk = s.get("pass_at_k_expected", {})
    rows.append((
        label,
        wall,
        tu.get("avg_completion_tokens_per_choice_assuming_total_over_n"),
        pk.get("1"),
        pk.get("16"),
    ))

print(f"{'config':<40} {'wall_s':>8} {'avg_len':>8} {'pass@1':>8} {'pass@16':>9}")
print("-" * 80)
for label, w, avg, p1, p16 in rows:
    def fmt(v, w):
        return f"{v:>{w}.4f}" if isinstance(v, float) else f"{'n/a':>{w}}"
    def fmt2(v, w):
        return f"{v:>{w}.2f}" if isinstance(v, float) else f"{'n/a':>{w}}"
    print(f"{label:<40} {fmt2(w,8)} {fmt2(avg,8)} {fmt(p1,8)} {fmt(p16,9)}")
