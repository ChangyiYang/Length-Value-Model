"""2D scale x entropy-threshold aggregator.

Reads results/ablation/entropy_threshold/scale_<S>_thr_<T>/ for all
(scale, threshold) cells and prints heatmap-like tables for:
  - wall_clock_s (speed)
  - avg_completion_per_choice (length control signal)
  - pass@1 expected (correctness)

Also pulls in the vanilla baseline from results/ablation/stride_length/vanilla_topk5/
if present, for the "no LVM at all" reference.
"""
import json
import pathlib

ROOT = pathlib.Path("results/ablation/entropy_threshold")
VANILLA_ROOT = pathlib.Path("results/ablation/stride_length/vanilla_topk5")
SCALES = ["0", "-100"]
THRESHOLDS = ["0", "0.25", "0.5", "1.0", "1.5"]


def load_cell(scale, thr):
    d = ROOT / f"scale_{scale}_thr_{thr}"
    summary_path = d / f"gsm8k.scale_{scale}_thr_{thr}.summary.json"
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
        "wall_clock_s": wall,
        "avg_completion": (s.get("token_usage") or {}).get(
            "avg_completion_tokens_per_choice_assuming_total_over_n"),
        "pass@1": (s.get("pass_at_k_expected") or {}).get("1"),
        "pass@16": (s.get("pass_at_k_expected") or {}).get("16"),
    }


def load_vanilla():
    d = VANILLA_ROOT
    summary_path = d / "gsm8k.vanilla_topk5_q50_n16.summary.json"
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
        "wall_clock_s": wall,
        "avg_completion": (s.get("token_usage") or {}).get(
            "avg_completion_tokens_per_choice_assuming_total_over_n"),
        "pass@1": (s.get("pass_at_k_expected") or {}).get("1"),
        "pass@16": (s.get("pass_at_k_expected") or {}).get("16"),
    }


vanilla = load_vanilla()
print("=" * 110)
print("LVM entropy-threshold ablation (GSM8K 50Q × n=16, top-k=5, max_tokens=6000, single server lifecycle)")
print("Skip LVM apply when row entropy <= threshold (nats). thr=0 effectively never skips.")
print("=" * 110)
print()
if vanilla is not None:
    print("Vanilla baseline (no LVM, top-k=5, different session):")
    print(f"  wall_clock_s={vanilla['wall_clock_s']:.2f}  "
          f"avg_completion={vanilla['avg_completion']:.2f}  "
          f"pass@1={vanilla['pass@1']:.4f}  pass@16={vanilla['pass@16']:.4f}")
    print()


def print_grid(metric, fmt, title):
    print(f"--- {title} ---")
    label = "scale|threshold"
    hdr = f"  {label:<16}" + "".join(f"{str(t):>10s}" for t in THRESHOLDS)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for scale in SCALES:
        row = [f"  scale={scale:<10s}"]
        for thr in THRESHOLDS:
            cell = load_cell(scale, thr)
            v = (cell or {}).get(metric)
            row.append(f"{(fmt % v if v is not None else 'n/a'):>10s}")
        print("".join(row))
    print()


print_grid("wall_clock_s", "%.2f", "wall_clock_s (perf)")
print_grid("avg_completion", "%.2f", "avg_completion_per_choice (length-control signal)")
print_grid("pass@1", "%.4f", "pass@1_expected (correctness)")

# Delta tables vs thr=0 (no-skip baseline within this session)
print("--- Δ wall_clock_s vs thr=0 (speedup from skipping) ---")
label = "scale|threshold"
hdr = f"  {label:<16}" + "".join(f"{str(t):>10s}" for t in THRESHOLDS)
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for scale in SCALES:
    base = load_cell(scale, "0")
    base_wall = (base or {}).get("wall_clock_s")
    row = [f"  scale={scale:<10s}"]
    for thr in THRESHOLDS:
        cell = load_cell(scale, thr)
        v = (cell or {}).get("wall_clock_s")
        if v is None or base_wall is None:
            row.append(f"{'n/a':>10s}")
        else:
            speedup = base_wall / v
            row.append(f"{speedup:>9.2f}x")
    print("".join(row))
print()

print("--- Δ pass@1 vs thr=0 (correctness drift from skipping) ---")
hdr = f"  {label:<16}" + "".join(f"{str(t):>10s}" for t in THRESHOLDS)
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for scale in SCALES:
    base = load_cell(scale, "0")
    base_p1 = (base or {}).get("pass@1")
    row = [f"  scale={scale:<10s}"]
    for thr in THRESHOLDS:
        cell = load_cell(scale, thr)
        v = (cell or {}).get("pass@1")
        if v is None or base_p1 is None:
            row.append(f"{'n/a':>10s}")
        else:
            row.append(f"{v - base_p1:>+9.4f}")
    print("".join(row))
print()

# Length-control loss: at scale=-100 only (where LVM has real control to lose)
print("--- Length-control loss at scale=-100 (drift toward vanilla) ---")
base = load_cell("-100", "0")
v_len = (vanilla or {}).get("avg_completion")
base_len = (base or {}).get("avg_completion")
if v_len is not None and base_len is not None:
    gap = v_len - base_len
    print(f"  vanilla={v_len:.2f}, thr=0 baseline={base_len:.2f}, gap={gap:+.2f}")
    print(f"  0% = LVM in full control, 100% = degenerated to vanilla")
    print()
    hdr = f"  {'':<16}" + "".join(f"{str(t):>10s}" for t in THRESHOLDS)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    row = [f"  scale=-100      "]
    for thr in THRESHOLDS:
        cell = load_cell("-100", thr)
        v = (cell or {}).get("avg_completion")
        if v is None or abs(gap) < 1e-6:
            row.append(f"{'n/a':>10s}")
        else:
            drift = (v - base_len) / gap * 100.0
            row.append(f"{drift:>+9.1f}%")
    print("".join(row))
else:
    print("  vanilla data or thr=0 baseline missing; cannot compute drift")
print()
