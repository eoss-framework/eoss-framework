#!/usr/bin/env python3
"""Global-atomic (L2) completion latency across GPU architectures.

Measured with a dedicated single-thread dependent-atomic microbench (no OSIRIS
slot array), which is the only way to reach global atomics on pre-Hopper parts:
their atomic base is a GPR that the slot array clobbers, whereas Hopper+ use a
slot-stable uniform-register descriptor. Values are cycles/atomic for a
serialized chain of atom.global.add.u64 to one aligned L2-resident line.
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from plots.violin import ARCH_COLORS, GENERATION_LABELS

# arch -> (part label, latency cy, method). Ordered old->new.
DATA = [
    ("SM70",  "V100",  230, "microbench"),
    ("SM80",  "A30",   375, "microbench"),      # datacenter Ampere (GA100, HBM2)
    ("SM86",  "A6000", 268, "microbench"),      # consumer/pro Ampere (GA102, GDDR6)
    ("SM89",  "RTX6000 Ada", 311, "microbench"),  # Ada (AD102, GDDR6)
    ("SM90",  "H100",  276, "OSIRIS"),
    ("SM100", "B200",  302, "microbench"),   # server Blackwell (GB100, HBM3e)
    ("SM120", "RTX 5090", 383, "microbench"),
]
# all 7 measured (Volta/Ampere-DC/Ampere-consumer/Ada/Hopper/Blackwell-DC/Blackwell-consumer)
PENDING = []

def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 12,
                         "axes.labelsize": 13, "figure.dpi": 300})
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    xs = range(len(DATA))
    for i, (arch, part, val, method) in enumerate(DATA):
        color = ARCH_COLORS.get(arch, "#999999")
        # OSIRIS points hatched so the method difference is explicit
        ax.bar(i, val, width=0.62, color=color, alpha=0.85,
               edgecolor="#222222", linewidth=0.8,
               hatch="//" if method == "OSIRIS" else None)
        ax.text(i, val + 6, f"{val}", ha="center", va="bottom", fontsize=11)

    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{p}\n{a}\n({GENERATION_LABELS.get(a,'')})"
                        for a, p, _, _ in DATA], fontsize=9)
    ax.set_ylabel("Global-atomic (L2) latency (cycles)")
    ax.set_ylim(0, max(v for _, _, v, _ in DATA) * 1.18)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    # method legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#bbbbbb", label="dependent-atomic microbench"),
                       Patch(facecolor="#bbbbbb", hatch="//", label="OSIRIS (H100 offline)")],
              loc="upper left", fontsize=9, frameon=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"figure_atomic_l2.{ext}", bbox_inches="tight")
    print("saved figure_atomic_l2.pdf / .png  |  pending:", ", ".join(PENDING))

if __name__ == "__main__":
    main()
