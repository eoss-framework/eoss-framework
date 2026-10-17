#!/usr/bin/env python3
"""
Figure 8: Cumulative issue-latency distribution for hot and cold paths.

Plots overlapping CDFs of hot_delta (trigger present) and cold_delta
(trigger NOP'd) from OSIRIS campaign data. Confirms that no instruction
produces a measurable issue-mode side channel (hot and cold CDFs overlap).

Usage (single architecture):
    python3 plot_hotcold_cdf.py campaign.csv --label "RTX 5090 (SM120)"

Usage (multiple architectures — reproducing Figure 8):
    python3 plot_hotcold_cdf.py \
        --csv "V100 (SM70)"=campaign_sm70.csv \
        --csv "A30 (SM80)"=campaign_sm80.csv \
        --csv "L40S (SM86)"=campaign_sm86.csv \
        --csv "RTX A6000 Ada (SM89)"=campaign_sm89.csv \
        --csv "H100 (SM90)"=campaign_sm90.csv \
        --csv "B200 (SM100)"=campaign_sm100.csv \
        --csv "RTX5090 (SM120)"=campaign_sm120.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_hotcold(csv_path, max_latency=300):
    """Load hot and cold latency arrays from campaign CSV."""
    hot, cold = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('error', '') not in ('', 'OK', 'None'):
                continue
            try:
                h = int(row['hot_delta'])
                c = int(row['cold_delta'])
                if 1 <= h <= max_latency:
                    hot.append(h)
                if 1 <= c <= max_latency:
                    cold.append(c)
            except (ValueError, KeyError):
                continue
    return np.array(hot), np.array(cold)


def plot_cdf_panel(ax, hot, cold, title):
    """Plot hot/cold CDF on a single axes."""
    if len(hot) == 0 and len(cold) == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)
        return

    if len(hot) > 0:
        sorted_hot = np.sort(hot)
        cdf_hot = np.arange(1, len(sorted_hot) + 1) / len(sorted_hot)
        ax.plot(sorted_hot, cdf_hot, '-', color='#4477AA', linewidth=1.2,
                label='Hot')

    if len(cold) > 0:
        sorted_cold = np.sort(cold)
        cdf_cold = np.arange(1, len(sorted_cold) + 1) / len(sorted_cold)
        ax.plot(sorted_cold, cdf_cold, '--', color='#EE6677', linewidth=1.2,
                label='Cold')

    ax.set_title(title, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)


def plot_single(csv_path, label, output, max_latency=300):
    """Single-architecture hot/cold CDF."""
    hot, cold = load_hotcold(csv_path, max_latency)
    if len(hot) == 0:
        print(f"Error: no valid data in {csv_path}", file=sys.stderr)
        sys.exit(1)

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 10,
        'axes.labelsize': 11, 'figure.dpi': 300,
    })

    fig, ax = plt.subplots(figsize=(5, 3.5))
    plot_cdf_panel(ax, hot, cold, label)
    ax.set_xlabel('Latency (cycles)')
    ax.set_ylabel('CDF')
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output, bbox_inches='tight')
    print(f"Saved {output} (hot: {len(hot):,}, cold: {len(cold):,} points)")


def plot_multi(label_csvs, output, max_latency=300):
    """Multi-architecture hot/cold CDF grid (Figure 8)."""
    n = len(label_csvs)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 8,
        'axes.labelsize': 9, 'axes.titlesize': 9,
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'figure.dpi': 300,
    })

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.2, nrows * 2.5),
                             squeeze=False)

    labels = list(label_csvs.keys())
    for i, label in enumerate(labels):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        hot, cold = load_hotcold(label_csvs[label], max_latency)
        plot_cdf_panel(ax, hot, cold, label)

        if r == nrows - 1:
            ax.set_xlabel('Latency (cycles)')
        if c == 0:
            ax.set_ylabel('CDF')

    # Hide unused axes
    for i in range(n, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r][c].set_visible(False)

    # Shared legend
    handles, lbls = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, lbls, loc='upper center', ncol=2,
                   fontsize=8, bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output, bbox_inches='tight')
    total = sum(len(load_hotcold(p, max_latency)[0]) for p in label_csvs.values())
    print(f"Saved {output} ({n} architectures, {total:,} hot data points)")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Figure 8: Hot/cold issue-latency CDF')
    parser.add_argument('csv_file', nargs='?',
                        help='Single campaign CSV')
    parser.add_argument('--label', default='RTX 5090 (SM120)',
                        help='Panel title for single-CSV mode')
    parser.add_argument('--csv', action='append', metavar='LABEL=FILE',
                        help='Label=CSV pairs for multi-arch mode')
    parser.add_argument('-o', '--output', default='figure8_hotcold_cdf.pdf',
                        help='Output file (default: figure8_hotcold_cdf.pdf)')
    parser.add_argument('--max-latency', type=int, default=300,
                        help='Maximum latency to include (default: 300)')
    args = parser.parse_args(argv)

    if args.csv:
        label_csvs = {}
        for entry in args.csv:
            label, path = entry.split('=', 1)
            label_csvs[label] = path
        plot_multi(label_csvs, args.output, args.max_latency)
    elif args.csv_file:
        plot_single(args.csv_file, args.label, args.output, args.max_latency)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
