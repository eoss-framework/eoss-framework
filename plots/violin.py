#!/usr/bin/env python3
"""
Figure 7: Completion latency violin plots across GPU architectures.

Reads one or more OSIRIS campaign CSVs and plots the distribution of
completion latency (scoreboard-enforced hot path) as violin plots.

Usage (single architecture):
    python3 plot_violin.py campaign_fenced.csv --label "RTX 5090 (SM120)"

Usage (multiple architectures — reproducing Figure 7):
    python3 plot_violin.py \
        --csv SM70=campaign_sm70.csv \
        --csv SM80=campaign_sm80.csv \
        --csv SM86=campaign_sm86.csv \
        --csv SM89=campaign_sm89.csv \
        --csv SM90=campaign_sm90.csv \
        --csv SM100=campaign_sm100.csv \
        --csv SM120=campaign_sm120.csv

A single-architecture run produces a single violin; multi-architecture
produces the side-by-side comparison from the paper.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ARCH_ORDER = ['SM70', 'SM80', 'SM86', 'SM89', 'SM90', 'SM100', 'SM120', 'SM121']

# Colored by generation using the matplotlib default (tab10) palette, matching
# Figure 7 of the paper: Volta=blue, Ampere=orange, Ada=green, Hopper=red,
# Blackwell=purple.
ARCH_COLORS = {
    'SM70':  '#1f77b4',   # Volta
    'SM80':  '#ff7f0e',   # Ampere
    'SM86':  '#ff7f0e',   # Ampere
    'SM89':  '#2ca02c',   # Ada Lovelace
    'SM90':  '#d62728',   # Hopper
    'SM100': '#9467bd',   # Blackwell
    'SM120': '#9467bd',   # Blackwell
    'SM121': '#9467bd',   # Blackwell
}

GENERATION_LABELS = {
    'SM70':  'Volta',
    'SM80':  'Ampere',
    'SM86':  'Ampere',
    'SM89':  'Ada Lovelace',
    'SM90':  'Hopper',
    'SM100': 'Blackwell',
    'SM120': 'Blackwell',
    'SM121': 'Blackwell',
}


def load_latencies(csv_path, column='hot_delta', max_latency=300):
    """Load latency values from campaign CSV, filtering errors and outliers."""
    latencies = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('error', '') not in ('', 'OK', 'None'):
                continue
            try:
                val = int(row[column])
                if 1 <= val <= max_latency:
                    latencies.append(val)
            except (ValueError, KeyError):
                continue
    return latencies


def plot_single(csv_path, label, output, max_latency=200):
    """Plot a single-architecture violin."""
    latencies = load_latencies(csv_path, max_latency=max_latency)
    if not latencies:
        print(f"Error: no valid latencies in {csv_path}", file=sys.stderr)
        sys.exit(1)

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 10,
        'axes.labelsize': 12, 'figure.dpi': 300,
    })

    fig, ax = plt.subplots(figsize=(4, 5))
    parts = ax.violinplot([latencies], positions=[1], showmedians=True,
                          showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('#66CCEE')
        pc.set_alpha(0.7)

    median = np.median(latencies)
    ax.set_title(f'{label}\nN={len(latencies):,}, median={median:.0f} cy')
    ax.set_ylabel('Hot Latency (cycles)')
    ax.set_xticks([1])
    ax.set_xticklabels([label])
    ax.set_ylim(0, None)

    fig.tight_layout()
    fig.savefig(output, bbox_inches='tight')
    print(f"Saved {output} ({len(latencies):,} data points)")


def plot_multi(arch_csvs, output, max_latency=200):
    """Plot multi-architecture violin comparison (Figure 7)."""
    ordered = sorted(arch_csvs.keys(), key=lambda a: ARCH_ORDER.index(a)
                     if a in ARCH_ORDER else 999)

    data = []
    labels = []
    colors = []
    for arch in ordered:
        latencies = load_latencies(arch_csvs[arch], max_latency=max_latency)
        if latencies:
            data.append(latencies)
            labels.append(arch)
            colors.append(ARCH_COLORS.get(arch, '#999999'))

    if not data:
        print("Error: no valid data from any CSV", file=sys.stderr)
        sys.exit(1)

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 9,
        'axes.labelsize': 11, 'axes.titlesize': 11,
        'xtick.labelsize': 8, 'ytick.labelsize': 9,
        'figure.dpi': 300,
    })

    fig, ax = plt.subplots(figsize=(max(6, len(data) * 1.2), 4.5))
    positions = range(1, len(data) + 1)

    positions = list(positions)
    parts = ax.violinplot(data, positions=positions, showmedians=False,
                          showextrema=False, widths=0.7)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor(colors[i])
        pc.set_alpha(0.8)

    # Median marker per architecture: a white dot with a thin dark edge. Reads
    # cleanly on the colored violins and keeps zero-variance columns legible
    # (e.g. an ISA whose uniform ops all sit at the latency floor).
    ax.scatter(positions, [np.median(d) for d in data], s=26, color='white',
               edgecolors='#222222', linewidths=0.7, zorder=6)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Architecture')
    ax.set_ylabel('Hot Latency (cycles)')
    ax.set_ylim(0, None)

    # Add generation legend
    seen = {}
    for arch in labels:
        gen = GENERATION_LABELS.get(arch, '')
        if gen and gen not in seen:
            seen[gen] = ARCH_COLORS.get(arch, '#999999')
    if seen:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, alpha=0.7, label=g)
                           for g, c in seen.items()]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    fig.tight_layout()
    fig.savefig(output, bbox_inches='tight')
    print(f"Saved {output} ({len(data)} architectures, "
          f"{sum(len(d) for d in data):,} total data points)")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Figure 7: Completion latency violin plots')
    parser.add_argument('csv_file', nargs='?',
                        help='Single campaign CSV (for single-arch mode)')
    parser.add_argument('--label', default='SM120',
                        help='Architecture label for single-CSV mode')
    parser.add_argument('--csv', action='append', metavar='ARCH=FILE',
                        help='Architecture=CSV pairs for multi-arch mode')
    parser.add_argument('-o', '--output', default='figure7_violin.pdf',
                        help='Output file (default: figure7_violin.pdf)')
    parser.add_argument('--max-latency', type=int, default=200,
                        help='Maximum latency to include (default: 200)')
    args = parser.parse_args(argv)

    if args.csv:
        arch_csvs = {}
        for entry in args.csv:
            arch, path = entry.split('=', 1)
            arch_csvs[arch] = path
        plot_multi(arch_csvs, args.output, args.max_latency)
    elif args.csv_file:
        plot_single(args.csv_file, args.label, args.output, args.max_latency)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
