"""Three-way compare: user-supplied reference table vs same-session baseline-code
run vs same-session patched run, for the GSM8K tradeoff sweep.

Run from repo root, after both timing dirs exist.
"""
import json
import csv
import pathlib

P = pathlib.Path("results/tradeoff/Qwen2.5-7B-Instruct_vs_1.5b-math/gsm8k")
B = pathlib.Path("results/tradeoff/Qwen2.5-7B-Instruct_vs_1.5b-math_baseline_code/gsm8k")

REF_BASELINE = [
    (200, 199.6463, 0.0225),
    (250, 245.1813, 0.15875),
    (300, 282.2675, 0.3225),
    (350, 310.530, 0.49125),
    (400, 329.4937, 0.6125),
    (500, 350.2025, 0.72375),
    (600, 358.6875, 0.775),
    (700, 361.470, 0.795),
    (800, 362.0225, 0.79875),
    (900, 362.0225, 0.79875),
    (1000, 362.0225, 0.79875),
    (1500, 362.0225, 0.79875),
    (2000, 362.0225, 0.79875),
    (5000, 362.0225, 0.79875),
    (10000, 362.0225, 0.79875),
]
REF_CENT = [
    ("-100", 278.905, 0.76125),
    ("-10", 343.32375, 0.79875),
    ("-5", 351.9775, 0.80075),
    ("-2", 350.62375, 0.79875),
    ("0", 353.2875, 0.7925),
]


def read_csv(p):
    rows = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            rows[int(r["budget"])] = (float(r["avg_capped_tokens"]), float(r["pass@1"]))
    return rows


HDR1 = ["budget", "ref len", "bcode len", "patch len", "P-B delta",
        "ref p@1", "bcode p@1", "patch p@1", "P-B delta"]
HDR2 = ["scale", "ref len", "bcode len", "patch len", "P-B delta",
        "ref p@1", "bcode p@1", "patch p@1", "P-B delta"]


def print_header(hdr):
    parts = [f"{hdr[0]:>7s} |"]
    for h in hdr[1:5]:
        parts.append(f" {h:>9s}")
    parts.append(" |")
    for h in hdr[5:]:
        parts.append(f" {h:>9s}")
    print("".join(parts))
    print("-" * 110)


p_b = read_csv(P / "budget_baseline" / "length_vs_passk.csv")
b_b = read_csv(B / "budget_baseline" / "length_vs_passk.csv")

print("=== Baseline (top_k=-1, no LenVM custom_params -> _build_pending bypassed) ===")
print_header(HDR1)
for budget, ref_len, ref_p in REF_BASELINE:
    if budget not in p_b:
        continue
    pl, pp = p_b[budget]
    bl, bp = b_b[budget]
    print(f"{budget:>7d} | {ref_len:>9.3f} {bl:>9.3f} {pl:>9.3f} {pl-bl:>+9.3f} |"
          f" {ref_p:>9.4f} {bp:>9.4f} {pp:>9.4f} {pp-bp:>+9.4f}")

print()
print("=== centered_exp (top_k=5, LenVM custom_params -> _build_pending exercised) ===")
print_header(HDR2)
for s, ref_len, ref_p in REF_CENT:
    tag = f"lenvm_q50_n16_p1.0_topk5_minp0.01_gamma0.997_centered_exp_{s}"
    p_path = P / f"gsm8k.{tag}.summary.json"
    b_path = B / f"gsm8k.{tag}.summary.json"
    if not p_path.exists() or not b_path.exists():
        print(f"{s:>7s}  MISSING")
        continue
    pd = json.load(open(p_path))
    bd = json.load(open(b_path))
    pl = pd["token_usage"]["avg_completion_tokens_per_choice_assuming_total_over_n"]
    bl = bd["token_usage"]["avg_completion_tokens_per_choice_assuming_total_over_n"]
    pp = pd["pass_at_k_expected"]["1"]
    bp = bd["pass_at_k_expected"]["1"]
    print(f"{s:>7s} | {ref_len:>9.3f} {bl:>9.3f} {pl:>9.3f} {pl-bl:>+9.3f} |"
          f" {ref_p:>9.4f} {bp:>9.4f} {pp:>9.4f} {pp-bp:>+9.4f}")
