#!/usr/bin/env python3
"""
Generate LaTeX tables from OSIRIS campaign results.

This script parses campaign CSV files and generates publication-ready
LaTeX tables summarizing:
- Instruction coverage statistics
- Contention pair analysis
- Latency distributions
- Top contention-inducing instruction pairs
"""

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class ContentionPair:
    """Statistics for a (reset, trigger) instruction pair."""
    reset: str
    trigger: str
    count: int = 0
    total_diff: int = 0
    max_diff: int = 0
    min_diff: int = 0
    positive_count: int = 0  # Count of measurements with diff > 0
    negative_count: int = 0  # Count of measurements with diff < 0

    @property
    def avg_diff(self) -> float:
        return self.total_diff / self.count if self.count > 0 else 0.0

    @property
    def contention_rate(self) -> float:
        """Percentage of measurements showing contention (diff != 0)."""
        return (self.positive_count + self.negative_count) / self.count * 100 if self.count > 0 else 0.0


@dataclass
class InstructionStats:
    """Statistics for a single instruction."""
    name: str
    as_reset_count: int = 0
    as_trigger_count: int = 0
    as_measure_count: int = 0
    caused_contention: int = 0  # Times this instruction as trigger caused diff != 0
    total_latency: int = 0
    latency_count: int = 0

    @property
    def avg_latency(self) -> float:
        return self.total_latency / self.latency_count if self.latency_count > 0 else 0.0


@dataclass
class CampaignResults:
    """Aggregated results from campaign CSV files."""
    total_measurements: int = 0
    error_count: int = 0
    unique_resets: Set[str] = field(default_factory=set)
    unique_triggers: Set[str] = field(default_factory=set)
    unique_measures: Set[str] = field(default_factory=set)
    instruction_stats: Dict[str, InstructionStats] = field(default_factory=dict)
    contention_pairs: Dict[Tuple[str, str], ContentionPair] = field(default_factory=dict)
    diff_distribution: Dict[int, int] = field(default_factory=lambda: defaultdict(int))

    # Summary stats
    contention_measurements: int = 0  # Measurements with diff != 0
    max_positive_diff: int = 0
    max_negative_diff: int = 0


def parse_campaign_csv(filepath: Path, results: CampaignResults, verbose: bool = False) -> None:
    """Parse a single campaign CSV file and update results."""
    if verbose:
        print(f"Parsing {filepath}...")

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        row_count = 0

        for row in reader:
            row_count += 1
            results.total_measurements += 1

            # Check for errors
            error = row.get('error', '').strip()
            if error:
                results.error_count += 1
                continue

            reset = row.get('reset', '').strip()
            trigger = row.get('trigger', '').strip()
            measure = row.get('measure', '').strip()

            # Skip empty rows
            if not reset or not trigger:
                continue

            # Track unique instructions
            results.unique_resets.add(reset)
            results.unique_triggers.add(trigger)
            if measure:
                results.unique_measures.add(measure)

            # Update instruction stats
            for instr in [reset, trigger, measure]:
                if instr and instr not in results.instruction_stats:
                    results.instruction_stats[instr] = InstructionStats(name=instr)

            if reset:
                results.instruction_stats[reset].as_reset_count += 1
            if trigger:
                results.instruction_stats[trigger].as_trigger_count += 1
            if measure:
                results.instruction_stats[measure].as_measure_count += 1

            # Parse timing values
            try:
                hot_delta = int(row.get('hot_delta', 0))
                cold_delta = int(row.get('cold_delta', 0))
                diff = int(row.get('diff', 0))
            except (ValueError, TypeError):
                continue

            # Track diff distribution
            results.diff_distribution[diff] += 1

            # Track contention
            if diff != 0:
                results.contention_measurements += 1
                results.instruction_stats[trigger].caused_contention += 1

                if diff > results.max_positive_diff:
                    results.max_positive_diff = diff
                if diff < results.max_negative_diff:
                    results.max_negative_diff = diff

            # Track contention pairs
            pair_key = (reset, trigger)
            if pair_key not in results.contention_pairs:
                results.contention_pairs[pair_key] = ContentionPair(reset=reset, trigger=trigger)

            pair = results.contention_pairs[pair_key]
            pair.count += 1
            pair.total_diff += diff
            if diff > pair.max_diff:
                pair.max_diff = diff
            if diff < pair.min_diff:
                pair.min_diff = diff
            if diff > 0:
                pair.positive_count += 1
            elif diff < 0:
                pair.negative_count += 1

            # Track latency (use hot_delta as base latency)
            if measure:
                results.instruction_stats[measure].total_latency += hot_delta
                results.instruction_stats[measure].latency_count += 1

    if verbose:
        print(f"  Processed {row_count:,} rows")


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    replacements = {
        '_': r'\_',
        '%': r'\%',
        '&': r'\&',
        '#': r'\#',
        '$': r'\$',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


def generate_summary_table(results: CampaignResults) -> str:
    """Generate a summary statistics table."""
    valid_measurements = results.total_measurements - results.error_count
    contention_rate = (results.contention_measurements / valid_measurements * 100) if valid_measurements > 0 else 0

    latex = r"""
\begin{table}[htbp]
\centering
\caption{OSIRIS Campaign Summary Statistics}
\label{tab:campaign-summary}
\begin{tabular}{lr}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Total measurements & """ + f"{results.total_measurements:,}" + r""" \\
Valid measurements & """ + f"{valid_measurements:,}" + r""" \\
Error rate & """ + f"{results.error_count / results.total_measurements * 100:.2f}" + r"""\% \\
\midrule
Unique reset instructions & """ + f"{len(results.unique_resets):,}" + r""" \\
Unique trigger instructions & """ + f"{len(results.unique_triggers):,}" + r""" \\
Unique measure instructions & """ + f"{len(results.unique_measures):,}" + r""" \\
Total unique instructions & """ + f"{len(results.instruction_stats):,}" + r""" \\
\midrule
Measurements with contention & """ + f"{results.contention_measurements:,}" + r""" \\
Contention rate & """ + f"{contention_rate:.2f}" + r"""\% \\
Max positive diff (cycles) & """ + f"{results.max_positive_diff:,}" + r""" \\
Max negative diff (cycles) & """ + f"{results.max_negative_diff:,}" + r""" \\
Unique (reset, trigger) pairs & """ + f"{len(results.contention_pairs):,}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_top_contention_table(results: CampaignResults, top_n: int = 20) -> str:
    """Generate table of top contention-inducing instruction pairs."""
    # Sort by absolute average diff
    sorted_pairs = sorted(
        results.contention_pairs.values(),
        key=lambda p: abs(p.avg_diff),
        reverse=True
    )[:top_n]

    latex = r"""
\begin{table}[htbp]
\centering
\caption{Top """ + str(top_n) + r""" Contention-Inducing Instruction Pairs}
\label{tab:top-contention}
\begin{tabular}{llrrr}
\toprule
\textbf{Reset} & \textbf{Trigger} & \textbf{Avg Diff} & \textbf{Max Diff} & \textbf{Count} \\
\midrule
"""

    for pair in sorted_pairs:
        reset_escaped = escape_latex(pair.reset[:25] + "..." if len(pair.reset) > 25 else pair.reset)
        trigger_escaped = escape_latex(pair.trigger[:25] + "..." if len(pair.trigger) > 25 else pair.trigger)
        latex += f"{reset_escaped} & {trigger_escaped} & {pair.avg_diff:.1f} & {pair.max_diff} & {pair.count:,} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_instruction_coverage_table(results: CampaignResults, top_n: int = 30) -> str:
    """Generate table showing instruction coverage statistics."""
    # Sort by total appearances
    sorted_instrs = sorted(
        results.instruction_stats.values(),
        key=lambda s: s.as_trigger_count + s.as_reset_count,
        reverse=True
    )[:top_n]

    latex = r"""
\begin{table}[htbp]
\centering
\caption{Instruction Coverage Statistics (Top """ + str(top_n) + r""")}
\label{tab:instruction-coverage}
\begin{tabular}{lrrrr}
\toprule
\textbf{Instruction} & \textbf{As Reset} & \textbf{As Trigger} & \textbf{Contention} & \textbf{Rate} \\
\midrule
"""

    for instr in sorted_instrs:
        name_escaped = escape_latex(instr.name[:30] + "..." if len(instr.name) > 30 else instr.name)
        rate = (instr.caused_contention / instr.as_trigger_count * 100) if instr.as_trigger_count > 0 else 0
        latex += f"{name_escaped} & {instr.as_reset_count:,} & {instr.as_trigger_count:,} & {instr.caused_contention:,} & {rate:.1f}\\% \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_latency_distribution_table(results: CampaignResults) -> str:
    """Generate table showing diff distribution buckets."""
    # Create buckets
    buckets = {
        "0 (no contention)": 0,
        "1-5 cycles": 0,
        "6-10 cycles": 0,
        "11-20 cycles": 0,
        "21-50 cycles": 0,
        "51-100 cycles": 0,
        ">100 cycles": 0,
        "Negative (1-5)": 0,
        "Negative (6-10)": 0,
        "Negative (>10)": 0,
    }

    for diff, count in results.diff_distribution.items():
        if diff == 0:
            buckets["0 (no contention)"] += count
        elif 1 <= diff <= 5:
            buckets["1-5 cycles"] += count
        elif 6 <= diff <= 10:
            buckets["6-10 cycles"] += count
        elif 11 <= diff <= 20:
            buckets["11-20 cycles"] += count
        elif 21 <= diff <= 50:
            buckets["21-50 cycles"] += count
        elif 51 <= diff <= 100:
            buckets["51-100 cycles"] += count
        elif diff > 100:
            buckets[">100 cycles"] += count
        elif -5 <= diff <= -1:
            buckets["Negative (1-5)"] += count
        elif -10 <= diff <= -6:
            buckets["Negative (6-10)"] += count
        elif diff < -10:
            buckets["Negative (>10)"] += count

    total = sum(buckets.values())

    latex = r"""
\begin{table}[htbp]
\centering
\caption{Timing Difference Distribution}
\label{tab:diff-distribution}
\begin{tabular}{lrr}
\toprule
\textbf{Diff Range} & \textbf{Count} & \textbf{Percentage} \\
\midrule
"""

    for bucket_name, count in buckets.items():
        pct = (count / total * 100) if total > 0 else 0
        latex += f"{bucket_name} & {count:,} & {pct:.2f}\\% \\\\\n"

    latex += r"""\midrule
Total & """ + f"{total:,}" + r""" & 100.00\% \\
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_contention_heatmap_data(results: CampaignResults, output_path: Path) -> str:
    """Generate data file for contention heatmap visualization."""
    # Get unique instructions sorted by frequency
    reset_instrs = sorted(results.unique_resets,
                          key=lambda x: results.instruction_stats.get(x, InstructionStats(x)).as_reset_count,
                          reverse=True)[:20]
    trigger_instrs = sorted(results.unique_triggers,
                            key=lambda x: results.instruction_stats.get(x, InstructionStats(x)).as_trigger_count,
                            reverse=True)[:20]

    # Write CSV for heatmap
    csv_path = output_path.with_suffix('.heatmap.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + [t[:20] for t in trigger_instrs])

        for reset in reset_instrs:
            row = [reset[:20]]
            for trigger in trigger_instrs:
                pair = results.contention_pairs.get((reset, trigger))
                row.append(f"{pair.avg_diff:.1f}" if pair else "0")
            writer.writerow(row)

    return str(csv_path)


def generate_all_tables(results: CampaignResults, output_path: Path) -> None:
    """Generate all LaTeX tables and write to file."""
    latex_content = r"""%% Auto-generated LaTeX tables from OSIRIS campaign results
%% Generated by generate_latex_tables.py

\documentclass{article}
\usepackage{booktabs}
\usepackage{graphicx}

\begin{document}

"""

    latex_content += generate_summary_table(results)
    latex_content += "\n\\clearpage\n"
    latex_content += generate_top_contention_table(results)
    latex_content += "\n\\clearpage\n"
    latex_content += generate_instruction_coverage_table(results)
    latex_content += "\n\\clearpage\n"
    latex_content += generate_latency_distribution_table(results)

    latex_content += r"""
\end{document}
"""

    with open(output_path, 'w') as f:
        f.write(latex_content)

    print(f"LaTeX tables written to {output_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from OSIRIS campaign results"
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        type=Path,
        help="Campaign result CSV files to process"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("campaign_tables.tex"),
        help="Output LaTeX file (default: campaign_tables.tex)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top entries to show in tables (default: 20)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only output summary table"
    )
    parser.add_argument(
        "--heatmap",
        action="store_true",
        help="Also generate heatmap data CSV"
    )

    args = parser.parse_args(argv)

    # Parse all CSV files
    results = CampaignResults()

    for csv_file in args.csv_files:
        if not csv_file.exists():
            print(f"Warning: {csv_file} not found, skipping", file=sys.stderr)
            continue
        parse_campaign_csv(csv_file, results, verbose=args.verbose)

    if results.total_measurements == 0:
        print("Error: No measurements found in input files", file=sys.stderr)
        sys.exit(1)

    # Print summary to console
    valid = results.total_measurements - results.error_count
    print(f"\n{'='*60}")
    print(f"Campaign Results Summary")
    print(f"{'='*60}")
    print(f"Total measurements:     {results.total_measurements:,}")
    print(f"Valid measurements:     {valid:,}")
    print(f"Error rate:             {results.error_count / results.total_measurements * 100:.2f}%")
    print(f"Unique instructions:    {len(results.instruction_stats):,}")
    print(f"Contention measurements:{results.contention_measurements:,}")
    print(f"Contention rate:        {results.contention_measurements / valid * 100:.2f}%")
    print(f"Max positive diff:      {results.max_positive_diff} cycles")
    print(f"Max negative diff:      {results.max_negative_diff} cycles")
    print(f"{'='*60}\n")

    # Generate output
    if args.summary_only:
        print(generate_summary_table(results))
    else:
        generate_all_tables(results, args.output)

        if args.heatmap:
            heatmap_path = generate_contention_heatmap_data(results, args.output)
            print(f"Heatmap data written to {heatmap_path}")


if __name__ == "__main__":
    main()
