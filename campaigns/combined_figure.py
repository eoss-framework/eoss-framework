#!/usr/bin/env python3
"""Figure 7 (replacement): completion latency across GPU architectures for the
three major pipelines, as one wide 3-panel figure.

  (a) Compute (ALU/FP/SFU)   (b) Uniform datapath   (c) Memory (mio, super harness)

Reuses the per-arch sliced campaign CSVs written by build_matrix.py, and the
generation color scheme / paper style from plots/violin.py.
"""
import csv, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from plots.violin import ARCH_COLORS, GENERATION_LABELS, ARCH_ORDER

SLICES = HERE / "slices"
COMPUTE_ARCHS = ["SM70", "SM80", "SM86", "SM89", "SM90", "SM100", "SM120"]
UNIFORM_ARCHS = ["SM80", "SM86", "SM89", "SM90", "SM100", "SM120"]  # no uniform datapath pre-Turing
MEM_ARCHS     = ["SM70", "SM80", "SM86", "SM89", "SM90", "SM100", "SM120"]

def _mem_csv(arch):
    sm = arch[2:].lower()
    sup = HERE / f"campaign_sm{sm}_mem_super.csv"
    return sup if sup.exists() else HERE / f"campaign_sm{sm}_mem.csv"

def load_atomic():
    """L2-atomic latency per arch from the dependent-atomic microbench."""
    p = HERE / "atomic_l2_data.csv"
    out = {}
    if p.exists():
        with open(p) as f:
            for r in csv.DictReader(f):
                out["SM" + r["sm"]] = int(r["latency_cy"])
    return out

def load(path, max_latency):
    out = []
    if not Path(path).exists():
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("error", "") not in ("", "OK", "None"):
                continue
            try:
                v = int(r["hot_delta"])
                if 1 <= v <= max_latency:
                    out.append(v)
            except (ValueError, KeyError):
                continue
    return out

PANELS = [
    ("(a) Compute  (ALU / FP / SFU)", {a: SLICES / f"compute_{a}.csv" for a in COMPUTE_ARCHS}, COMPUTE_ARCHS, 55),
    ("(b) Uniform datapath",          {a: SLICES / f"uniform_{a}.csv" for a in UNIFORM_ARCHS}, UNIFORM_ARCHS, 55),
    # Slices are per-op-min (written by build_matrix.py) and include the
    # microbench L2-atomic tier folded in as the top lobe. Cap ~410 to show it.
    ("(c) Memory  (mio + L2-atomic)",  {a: SLICES / f"memory_{a}.csv" for a in MEM_ARCHS}, MEM_ARCHS, 410),
]

def main():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 16,
        "axes.labelsize": 20, "axes.titlesize": 20,
        "xtick.labelsize": 16, "ytick.labelsize": 16, "figure.dpi": 300,
    })
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    mem_ax = mem_present = None
    for ax, (title, csvs, archs, ymax) in zip(axes, PANELS):
        present = [a for a in ARCH_ORDER if a in archs and Path(csvs[a]).exists()
                   and load(csvs[a], ymax)]
        data = [load(csvs[a], ymax) for a in present]
        pos = list(range(1, len(present) + 1))
        parts = ax.violinplot(data, positions=pos, showmedians=False,
                              showextrema=False, widths=0.7)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(ARCH_COLORS[present[i]])
            pc.set_edgecolor(ARCH_COLORS[present[i]])
            pc.set_alpha(0.8)
        ax.scatter(pos, [np.median(d) for d in data], s=24, color="white",
                   edgecolors="#222222", linewidths=0.7, zorder=6)
        ax.set_xticks(pos)
        ax.set_xticklabels(present, rotation=35, ha="right", fontsize=13)
        ax.set_xlabel("Architecture")
        ax.set_title(title)
        ax.set_ylim(0, None)
        ax.set_ylabel("Hot Latency (cycles)")
    # (the L2-atomic tier is now a violin lobe in the memory panel, folded in by
    # build_matrix.py — no separate marker needed)

    # one generation legend for the whole figure
    seen = {}
    for a in ARCH_ORDER:
        g = GENERATION_LABELS.get(a)
        if g and g not in seen and a in ARCH_COLORS:
            seen[g] = ARCH_COLORS[a]
    handles = [Patch(facecolor=c, alpha=0.8, label=g) for g, c in seen.items()]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=True, bbox_to_anchor=(0.5, 1.08), fontsize=16)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"figure7_pipelines.{ext}", bbox_inches="tight")
    print("saved figure7_pipelines.pdf / .png")

if __name__ == "__main__":
    main()
