"""2D scale × stride aggregator: how much does stride break LVM length control?

Reads results/ablation/stride_length/{vanilla_topk5, scale_<S>_stride_<N>}/
and prints two heatmap-like tables (rows = scale, cols = stride):
  * avg_completion_per_choice  (the length-control signal)
  * pass@1 expected            (the correctness signal)

Vanilla baseline (LVM off) is printed once at the top as reference.
"""
import json
import pathlib

ROOT = pathlib.Path("results/ablation/stride_length")
SCALES = ["0", "-10", "-100"]
STRIDES = [1, 2, 5, 10]


def load_vanilla():
    d = ROOT / "vanilla_topk5"
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


def load_cell(scale, stride):
    d = ROOT / f"scale_{scale}_stride_{stride}"
    summary_path = d / f"gsm8k.scale_{scale}_stride_{stride}.summary.json"
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
print("=" * 100)
print("LVM stride × scale ablation (GSM8K 50Q × n=16, top-k=5, max_tokens=6000, same session)")
print("=" * 100)
print()
if vanilla is not None:
    print(f"Vanilla baseline (no LVM, top-k=5):")
    print(f"  wall_clock_s={vanilla['wall_clock_s']:.2f}  "
          f"avg_completion={vanilla['avg_completion']:.2f}  "
          f"pass@1={vanilla['pass@1']:.4f}  pass@16={vanilla['pass@16']:.4f}")
    print()


def print_grid(metric, fmt, title):
    print(f"--- {title} ---")
    label = "scale|stride"
    hdr = f"  {label:<16}" + "".join(f"{str(s):>10s}" for s in STRIDES)
    if vanilla is not None:
        hdr += f"{'vanilla':>12s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for scale in SCALES:
        row = [f"  scale={scale:<10s}"]
        for stride in STRIDES:
            cell = load_cell(scale, stride)
            v = (cell or {}).get(metric)
            row.append(f"{(fmt % v if v is not None else 'n/a'):>10s}")
        if vanilla is not None:
            v = vanilla.get(metric)
            row.append(f"{(fmt % v if v is not None else 'n/a'):>12s}")
        print("".join(row))
    print()


print_grid("avg_completion", "%.2f", "avg_completion_per_choice (length-control signal)")
print_grid("pass@1", "%.4f", "pass@1_expected")
print_grid("wall_clock_s", "%.2f", "wall_clock_s")

# Length-control "loss" table: how far each (scale, stride) drifts back toward vanilla
if vanilla is not None and vanilla.get("avg_completion"):
    v_len = vanilla["avg_completion"]
    print(f"--- Length-control loss: avg_len drift toward vanilla ({v_len:.2f}) ---")
    print(f"  positive = LVM is losing length control (output growing back toward vanilla)")
    print(f"  Each cell = ((this_avg_len - stride1_avg_len) / (vanilla - stride1_avg_len)) * 100%")
    print(f"  0% = LVM fully in control (matches stride=1), 100% = fully degenerated to vanilla")
    print()
    label = "scale|stride"
    hdr = f"  {label:<16}" + "".join(f"{str(s):>10s}" for s in STRIDES)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for scale in SCALES:
        s1 = load_cell(scale, 1)
        if s1 is None or s1.get("avg_completion") is None:
            print(f"  scale={scale:<10s}  no stride=1 data")
            continue
        s1_len = s1["avg_completion"]
        gap = v_len - s1_len
        row = [f"  scale={scale:<10s}"]
        for stride in STRIDES:
            cell = load_cell(scale, stride)
            v = (cell or {}).get("avg_completion")
            if v is None:
                row.append(f"{'n/a':>10s}")
            elif abs(gap) < 1e-6:
                # No length control to begin with at this scale
                row.append(f"{'~':>10s}")
            else:
                drift = (v - s1_len) / gap * 100.0
                row.append(f"{drift:>+9.1f}%")
        print("".join(row))
    print()
