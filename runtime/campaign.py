"""Campaign runner: orchestrates timing-differential campaigns over
instruction sets (OSIRIS-style fuzzing, Section 8).

Modes:

* single-sequence — one triple per kernel launch; every slot gets the same
  (RESET, TRIGGER, MEASURE) triple. Slow (~3 tests/s) but tests all
  instructions, including unsafe ones.
* multi-sequence — pack up to 4,095 different triples per launch with
  shuffled slot assignments (~4095x faster); use ``filter_safe`` to exclude
  memory/control-flow/barrier/sync instructions.

Patterns:

* ``diagonal`` — RESET == TRIGGER == MEASURE (baseline latency, N triples)
* ``osiris``   — TRIGGER == MEASURE, all (RESET, X) pairs (N^2 triples)
* ``sweep``    — fixed RESET, vary TRIGGER == MEASURE

Absorbed from OSIRIS/run_campaign.py onto SlotHarness / schemas v2.
"""

import argparse
import csv
import json
import os
import random
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ..gpu import detect_sm
from ..schemas import load_encodings
from .measure import SlotHarness

# ---------------------------------------------------------------------------
# Safety filtering (experiment policy: which instructions may be mixed
# across slots in a multi-sequence launch without crashing the GPU)
# ---------------------------------------------------------------------------

SAFE_PREFIXES = (
    'fadd', 'fmul', 'ffma', 'fmnmx', 'fset', 'fsetp', 'fcmp', 'fchk',
    'fsel', 'fswzadd', 'f2f', 'f2i', 'i2f', 'i2i', 'i2ip', 'f2fp', 'frnd',
    'dadd', 'dfma', 'dmul', 'dmnmx', 'dset', 'dsetp',
    'hadd', 'hfma', 'hmul', 'hmnmx', 'hset', 'hsetp',
    'iadd', 'imad', 'imul', 'imnmx', 'isetp', 'iset', 'icmp',
    'lop3', 'lop', 'shf', 'shl', 'shr', 'bfe', 'bfi', 'flo', 'popc',
    'brev', 'bmsk', 'sgxt',
    'mufu',
    'mov', 'sel', 'shfl', 'prmt',
    'p2r', 'r2p',
    's2r', 'cs2r',
    'lea',
    'vote', 'voteu', 'redux',
)

UNSAFE_PREFIXES = (
    'ldg', 'stg', 'atom', 'red',
    'ldl', 'stl',
    'lds', 'sts',
    'ldc',
    'tex', 'tld', 'txd', 'txq',
    'samp', 'surf', 'suatom', 'sured',
    'bar', 'bsync', 'bssy', 'yield',
    'brk', 'bra', 'jmp', 'jmx',
    'ret', 'exit',
    'call',
    'ttu',
    'depbar',
    'warpsync', 'nanosleep',
    # uniform datapath (udp_pipe) is compute and now safe, but a family of
    # uniform ops issue through it that are NOT straight-line compute and crash
    # or hang a packed multi-launch (they access memory / TMA descriptors /
    # tensor memory / hardware state). Especially prominent on Blackwell:
    #   ucgabar  cluster barrier (needs a cluster launch context)
    #   ublk     block prefetch/reduce (memory)
    #   utma*    TMA descriptor load/store/reduce/prefetch/ctl/cmdflush
    #   utc*     tensor-memory bar/copy/load-store-workspace/shift (sm_100)
    #   umemsets uniform memset (memory), usetshmsz set-shared-mem-size,
    #   uvirtcount virtual counter
    'ucgabar', 'ublk', 'utma', 'utc', 'umemsets', 'usetshmsz', 'uvirtcount',
)

SAFE_PIPELINES = {
    'fmalighter_pipe', 'fma64lite_pipe', 'sfu_pipe', 'fp16_pipe',
    'int_pipe', 'alu_pipe', 'unifalu_pipe', 'DUALALU_OPS',
    'udp_pipe',
}

MAX_TRIPLES_PER_LAUNCH = 4095

# CSV schemas (v2, consistent between writers and readers)
RAW_HEADER_MULTI = ['sm_version', 'mode', 'pattern', 'reset', 'trigger',
                    'measure', 'iteration', 'slot', 'hot_delta', 'cold_delta',
                    'diff']
RAW_HEADER_SINGLE = ['sm_version', 'mode', 'pattern', 'reset', 'trigger',
                     'measure', 'trial', 'slot', 'hot_delta', 'cold_delta',
                     'diff']
SUMMARY_HEADER = ['sm_version', 'mode', 'pattern', 'reset', 'trigger',
                  'measure', 'n_samples', 'n_unique_slots',
                  'diff_median', 'diff_mean', 'diff_std', 'diff_min', 'diff_max',
                  'hot_median', 'hot_mean', 'cold_median', 'cold_mean']


def _record_name(rec) -> str:
    if isinstance(rec, dict):
        return rec.get('instruction', rec.get('opcode_key', ''))
    return rec.instruction


def _record_pipeline(rec) -> str:
    if isinstance(rec, dict):
        return rec.get('pipeline', '') or ''
    return rec.pipeline or ''


def is_instruction_safe(rec) -> bool:
    name = _record_name(rec).lower()
    pipeline = _record_pipeline(rec)
    for prefix in UNSAFE_PREFIXES:
        if name.startswith(prefix):
            return False
    for prefix in SAFE_PREFIXES:
        if name.startswith(prefix):
            return True
    return pipeline in SAFE_PIPELINES


def filter_safe_instructions(instructions: list) -> list:
    return [i for i in instructions if is_instruction_safe(i)]


# ---------------------------------------------------------------------------
# Triple generation
# ---------------------------------------------------------------------------

def generate_triples(instructions: list, pattern: str,
                     reset_name: str = None,
                     max_triples: int = None,
                     rng: Optional[random.Random] = None) -> List[Tuple[str, str, str]]:
    names = [_record_name(i) for i in instructions]
    if pattern == "diagonal":
        triples = [(n, n, n) for n in names]
    elif pattern == "osiris":
        triples = [(r, x, x) for r in names for x in names]
        (rng or random).shuffle(triples)
    elif pattern == "sweep":
        r = reset_name or names[0]
        triples = [(r, n, n) for n in names]
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    if max_triples:
        triples = triples[:max_triples]
    return triples


def batch_triples(triples, batch_size=MAX_TRIPLES_PER_LAUNCH):
    return [triples[i:i + batch_size] for i in range(0, len(triples), batch_size)]


# ---------------------------------------------------------------------------
# Multi-sequence core
# ---------------------------------------------------------------------------

def run_multi_iteration(harness: SlotHarness, triples, mode, rng, verbose=False,
                        keep_binaries=None, iteration_label="", tmp_dir=None):
    n = len(triples)
    if n > MAX_TRIPLES_PER_LAUNCH:
        raise ValueError(f"Too many triples ({n}); max {MAX_TRIPLES_PER_LAUNCH}")

    slot_indices = list(range(1, n + 1))
    rng.shuffle(slot_indices)
    triple_to_slot = {i: slot_indices[i] for i in range(n)}
    slot_to_triple = {slot_indices[i]: i for i in range(n)}

    slot_assignments = [
        (triple_to_slot[i],
         harness.get_instruction_bytes(r),
         harness.get_instruction_bytes(t),
         harness.get_instruction_bytes(m))
        for i, (r, t, m) in enumerate(triples)
    ]

    tmp_dir = Path(tmp_dir) if tmp_dir else Path(tempfile.gettempdir())
    hot_path = tmp_dir / f".tmp_multi_hot_{os.getpid()}"
    cold_path = tmp_dir / f".tmp_multi_cold_{os.getpid()}"
    harness.create_multi_patched_binary(hot_path, mode, slot_assignments, cold=False)
    harness.create_multi_patched_binary(cold_path, mode, slot_assignments, cold=True)

    try:
        hot_timings = harness.run_binary(hot_path)
        cold_timings = harness.run_binary(cold_path)
    finally:
        if keep_binaries:
            keep_binaries.mkdir(parents=True, exist_ok=True)
            label = iteration_label or "batch"
            shutil.copy2(hot_path, keep_binaries / f"{label}_hot")
            shutil.copy2(cold_path, keep_binaries / f"{label}_cold")
        hot_path.unlink(missing_ok=True)
        cold_path.unlink(missing_ok=True)

    hot_dict = dict(hot_timings)
    cold_dict = dict(cold_timings)

    results = []
    for slot_idx, triple_idx in slot_to_triple.items():
        r_name, t_name, m_name = triples[triple_idx]
        hot_d = hot_dict.get(slot_idx)
        cold_d = cold_dict.get(slot_idx)
        if hot_d is not None and cold_d is not None:
            results.append({
                'reset': r_name, 'trigger': t_name, 'measure': m_name,
                'slot': slot_idx, 'hot_delta': hot_d, 'cold_delta': cold_d,
                'diff': hot_d - cold_d,
            })

    if verbose:
        diffs = [r['diff'] for r in results]
        if diffs:
            print(f"    {len(results)} results, "
                  f"diff: mean={statistics.mean(diffs):.1f} "
                  f"median={statistics.median(diffs):.0f} "
                  f"range=[{min(diffs)}, {max(diffs)}]")
    return results


# ---------------------------------------------------------------------------
# Crash recovery (binary-search bisection of crashing batches)
# ---------------------------------------------------------------------------

def find_crashing_triples(harness, triples, mode, verbose=False):
    if not triples:
        return set()
    if len(triples) == 1:
        rng = random.Random(0)
        try:
            run_multi_iteration(harness, triples, mode, rng)
            return set()
        except Exception:
            if verbose:
                print(f"    Isolated crashing triple: {triples[0]}")
            return {triples[0]}

    rng = random.Random(0)
    try:
        run_multi_iteration(harness, triples, mode, rng)
        return set()
    except Exception:
        pass

    mid = len(triples) // 2
    return (find_crashing_triples(harness, triples[:mid], mode, verbose) |
            find_crashing_triples(harness, triples[mid:], mode, verbose))


def run_batch_with_recovery(harness, triples, mode, rng, known_bad,
                            verbose=False, max_rounds=20):
    clean = [t for t in triples if t not in known_bad]
    if not clean:
        return [], known_bad

    for round_num in range(max_rounds + 1):
        try:
            results = run_multi_iteration(harness, clean, mode, rng, verbose)
            return results, known_bad
        except Exception as e:
            if round_num == max_rounds:
                if verbose:
                    print(f"\n    Recovery budget exhausted. Skipping.")
                return [], known_bad
            if verbose:
                print(f"\n    Crash (round {round_num + 1}): {e}")
                print(f"    Binary search ({len(clean)} triples)...")
            newly_bad = find_crashing_triples(harness, clean, mode, verbose)
            if not newly_bad:
                if verbose:
                    print(f"    Could not isolate crash cause.")
                return [], known_bad
            known_bad = known_bad | newly_bad
            clean = [t for t in clean if t not in known_bad]
            if verbose:
                print(f"    Excluded {len(newly_bad)}. "
                      f"Total: {len(known_bad)}. Remaining: {len(clean)}")
            if not clean:
                return [], known_bad

    return [], known_bad


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(all_results):
    by_triple: Dict[Tuple[str, str, str], list] = {}
    for r in all_results:
        key = (r['reset'], r['trigger'], r['measure'])
        by_triple.setdefault(key, []).append(r)

    summaries = {}
    for key, measurements in by_triple.items():
        diffs = [m['diff'] for m in measurements]
        hots = [m['hot_delta'] for m in measurements]
        colds = [m['cold_delta'] for m in measurements]
        slots = {m['slot'] for m in measurements}
        summaries[key] = {
            'n_samples': len(diffs),
            'diff_median': statistics.median(diffs),
            'diff_mean': statistics.mean(diffs),
            'diff_std': statistics.pstdev(diffs) if len(diffs) > 1 else 0.0,
            'diff_min': min(diffs), 'diff_max': max(diffs),
            'hot_median': statistics.median(hots), 'hot_mean': statistics.mean(hots),
            'cold_median': statistics.median(colds), 'cold_mean': statistics.mean(colds),
            'n_unique_slots': len(slots),
        }
    return summaries


# ---------------------------------------------------------------------------
# Campaign configuration and drivers
# ---------------------------------------------------------------------------

@dataclass
class CampaignConfig:
    mode: str = "issue"                # "issue" | "completion"
    pattern: str = "diagonal"          # "diagonal" | "osiris" | "sweep"
    multi: bool = False
    iterations: int = 5
    trials: int = 3
    seed: Optional[int] = None
    filter_safe: bool = False
    crash_recovery: bool = False
    reset_instruction: Optional[str] = None
    max_triples: Optional[int] = None
    keep_binaries: Optional[str] = None
    verbose: bool = False
    output: Optional[str] = None
    summary: Optional[str] = None
    isa_json: Optional[str] = None
    all_instructions: Optional[str] = None
    all_validated: Optional[str] = None


def print_isa_summary(cfg: CampaignConfig, instructions, known_bad=None):
    print()
    print("=" * 60)
    print("ISA PIPELINE SUMMARY")
    print("=" * 60)

    if cfg.isa_json and Path(cfg.isa_json).exists():
        with open(cfg.isa_json) as f:
            isa_data = json.load(f)
        print(f"  Instruction classes (ISA):   {len(isa_data.get('INSTRUCTIONS', {}))}")

    if cfg.all_instructions and Path(cfg.all_instructions).exists():
        with open(cfg.all_instructions) as f:
            gen_data = json.load(f)
        print(f"  Generated encodings:         {len(gen_data)}")

    if cfg.all_validated and Path(cfg.all_validated).exists():
        with open(cfg.all_validated) as f:
            val_data = json.load(f)
        print(f"  GPU-validated:               {len(val_data)}")

    campaign_names = sorted(_record_name(i) for i in instructions)
    print(f"  Campaign instructions:       {len(instructions)}")
    print(f"  Campaign mode:               {cfg.mode}")
    print(f"  Campaign pattern:            {cfg.pattern}")

    if cfg.filter_safe:
        print(f"  Safety filter:               ENABLED")
    if known_bad:
        print(f"  Crash-excluded:              {len(known_bad)}")

    if cfg.all_validated and Path(cfg.all_validated).exists():
        with open(cfg.all_validated) as f:
            val_data = json.load(f)
        val_names = {_record_name(i) for i in val_data}
        excluded = sorted(val_names - set(campaign_names))
        if excluded:
            print(f"\n  Excluded from campaign ({len(excluded)}):")
            for name in excluded:
                print(f"    {name}")

    print(f"\n  Campaign instructions ({len(campaign_names)}):")
    for name in campaign_names:
        print(f"    {name}")
    print("=" * 60)


def run_single_campaign(cfg: CampaignConfig, harness: SlotHarness,
                        instructions, triples, sm_version):
    print()
    print("=" * 60)
    print("SINGLE-SEQUENCE CAMPAIGN")
    print("=" * 60)
    print(f"  SM version:   {sm_version}")
    print(f"  Pattern:      {cfg.pattern}")
    print(f"  Mode:         {cfg.mode}")
    print(f"  Triples:      {len(triples)}")
    print(f"  Trials:       {cfg.trials}")
    print(f"  Output:       {cfg.output}")
    print()

    with open(cfg.output, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(RAW_HEADER_SINGLE)

        for idx, (r_name, t_name, m_name) in enumerate(triples):
            if idx % 10 == 0:
                print(f"  [{idx + 1}/{len(triples)}] ({r_name}, {t_name}, {m_name})")
            try:
                keep_dir = Path(cfg.keep_binaries) if cfg.keep_binaries else None
                results = harness.measure_triple(
                    mode=cfg.mode,
                    reset_name=r_name, trigger_name=t_name, measure_name=m_name,
                    trials=cfg.trials, keep_binaries=keep_dir,
                )
            except Exception as e:
                print(f"  ERROR: {e}", file=sys.stderr)
                continue

            for trial in results['trials']:
                for d in trial['slot_data']:
                    writer.writerow([
                        sm_version, cfg.mode, cfg.pattern,
                        r_name, t_name, m_name,
                        trial['trial_index'], d['slot'],
                        d['hot_delta'], d['cold_delta'], d['diff'],
                    ])

            if 'summary' in results and cfg.verbose:
                s = results['summary']
                print(f"    hot={s['hot_mean']:.1f} cold={s['cold_mean']:.1f} "
                      f"diff={s['diff_mean']:.1f}")

    print_isa_summary(cfg, instructions)
    print(f"\nCampaign complete. Results: {cfg.output}")


def run_multi_campaign(cfg: CampaignConfig, harness: SlotHarness,
                       instructions, triples, sm_version):
    rng = random.Random(cfg.seed)
    # One slot per triple per launch, so a batch can be no larger than the
    # harness's usable slot count. Slots are 1-indexed in run_multi_iteration
    # (``range(1, n+1)``; slot 0 is reserved), so n triples occupy slots 1..n
    # and the cap is len(slots)-1. Harnesses are commonly built with fewer than
    # MAX_TRIPLES_PER_LAUNCH slots (e.g. 1024 for the union build).
    batch_size = min(MAX_TRIPLES_PER_LAUNCH, len(harness.slots) - 1)
    batches = batch_triples(triples, batch_size)
    n_batches = len(batches)

    print()
    print("=" * 60)
    print("MULTI-SEQUENCE CAMPAIGN")
    print("=" * 60)
    print(f"  SM version:     {sm_version}")
    print(f"  Pattern:        {cfg.pattern}")
    print(f"  Mode:           {cfg.mode}")
    print(f"  Triples:        {len(triples)}")
    print(f"  Batches:        {n_batches} (max {batch_size}/launch)")
    print(f"  Iterations:     {cfg.iterations}")
    print(f"  Total launches: {n_batches * cfg.iterations * 2}")
    print(f"  Seed:           {cfg.seed if cfg.seed is not None else 'random'}")
    print(f"  Filter safe:    {cfg.filter_safe}")
    print(f"  Crash recovery: {cfg.crash_recovery}")
    print(f"  Output:         {cfg.output}")
    if cfg.summary:
        print(f"  Summary:        {cfg.summary}")
    print()

    # Verify all instruction bytes are loadable
    print("Validating instruction bytes... ", end="", flush=True)
    bad = [t for t in triples
           if any(name not in harness.instr_by_name for name in t)]
    if bad:
        print(f"FAILED: {len(bad)} triples reference unknown instructions")
        for bt in bad[:5]:
            print(f"    {bt}")
        sys.exit(1)
    print(f"OK ({len(triples)} triples)")

    all_results = []
    known_bad: Set[Tuple[str, str, str]] = set()
    t_start = time.time()

    with open(cfg.output, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(RAW_HEADER_MULTI)

        for batch_idx, batch in enumerate(batches):
            batch_label = f"Batch {batch_idx + 1}/{n_batches} ({len(batch)} triples)"
            for iteration in range(cfg.iterations):
                print(f"{batch_label}, Iter {iteration + 1}/{cfg.iterations}...",
                      end=" ", flush=True)

                if cfg.crash_recovery:
                    results, known_bad = run_batch_with_recovery(
                        harness, batch, cfg.mode, rng, known_bad, cfg.verbose)
                else:
                    try:
                        keep_dir = (Path(cfg.keep_binaries)
                                    if cfg.keep_binaries else None)
                        results = run_multi_iteration(
                            harness, batch, cfg.mode, rng, cfg.verbose,
                            keep_dir, f"iter{iteration:04d}_batch{batch_idx:02d}")
                    except RuntimeError as e:
                        print(f"ERROR: {e}")
                        continue
                    except Exception as e:
                        print(f"UNEXPECTED: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

                for r in results:
                    r['iteration'] = iteration
                    writer.writerow([
                        sm_version, cfg.mode, cfg.pattern,
                        r['reset'], r['trigger'], r['measure'],
                        iteration, r['slot'],
                        r['hot_delta'], r['cold_delta'], r['diff'],
                    ])
                csvfile.flush()
                all_results.extend(results)

                diffs = [r['diff'] for r in results]
                if diffs:
                    excl = f" ({len(known_bad)} excluded)" if known_bad else ""
                    print(f"{len(results)} results, "
                          f"diff median={statistics.median(diffs):.0f}{excl}")
                else:
                    print("no results")

    elapsed = time.time() - t_start

    print(f"\n{'=' * 60}")
    print(f"Campaign complete in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Total measurements: {len(all_results)}")
    if known_bad:
        print(f"Excluded (crash): {len(known_bad)} triples")
        for bt in sorted(known_bad):
            if bt[0] == bt[1] == bt[2]:
                print(f"    {bt[0]}")
            else:
                print(f"    ({bt[0]}, {bt[1]}, {bt[2]})")
    print(f"Raw data: {cfg.output}")

    summaries = aggregate_results(all_results)

    if cfg.summary:
        with open(cfg.summary, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(SUMMARY_HEADER)
            for (r_name, t_name, m_name), s in sorted(summaries.items()):
                writer.writerow([
                    sm_version, cfg.mode, cfg.pattern,
                    r_name, t_name, m_name,
                    s['n_samples'], s['n_unique_slots'],
                    f"{s['diff_median']:.1f}", f"{s['diff_mean']:.2f}",
                    f"{s['diff_std']:.2f}",
                    s['diff_min'], s['diff_max'],
                    f"{s['hot_median']:.1f}", f"{s['hot_mean']:.2f}",
                    f"{s['cold_median']:.1f}", f"{s['cold_mean']:.2f}",
                ])
        print(f"Summary:  {cfg.summary}")

    # Per-triple summary table
    print(f"\n{'=' * 60}")
    print("Per-triple summary (sorted by diff_median):")
    print(f"{'Instruction':<40s} {'N':>5s} {'Slots':>5s} "
          f"{'Med':>6s} {'Mean':>7s} {'Std':>6s} {'Min':>5s} {'Max':>5s}")
    print('-' * 80)

    sorted_summaries = sorted(summaries.items(),
                              key=lambda x: x[1]['diff_median'], reverse=True)
    if len(sorted_summaries) <= 25:
        show = sorted_summaries
    else:
        show = sorted_summaries[:10] + [None] + sorted_summaries[-10:]

    for item in show:
        if item is None:
            print(f"  {'... (truncated) ...':^78s}")
            continue
        (r, t, m), s = item
        label = t if r == t == m else f"({r},{t},{m})"
        print(f"  {label:<38s} {s['n_samples']:>5d} {s['n_unique_slots']:>5d} "
              f"{s['diff_median']:>6.0f} {s['diff_mean']:>7.1f} "
              f"{s['diff_std']:>6.1f} {s['diff_min']:>5d} {s['diff_max']:>5d}")

    # Sanity checks
    print(f"\n{'=' * 60}")
    print("Sanity checks:")
    fadd_key = ("fadd__RRR_RR", "fadd__RRR_RR", "fadd__RRR_RR")
    if fadd_key in summaries:
        s = summaries[fadd_key]
        print(f"  FADD diagonal: diff_median={s['diff_median']:.0f} (expected ~0)")
        if abs(s['diff_median']) > 5:
            print(f"    WARNING: FADD differential unexpectedly large")
    else:
        print(f"  FADD diagonal: not in test set")

    min_slots = min(cfg.iterations, 3)
    low_cov = [(k, s) for k, s in summaries.items()
               if s['n_unique_slots'] < min_slots]
    if low_cov:
        print(f"  WARNING: {len(low_cov)} triples in < {min_slots} unique slots")
    else:
        print(f"  Slot coverage: OK")

    print_isa_summary(cfg, instructions, known_bad)
    print(f"\nDone.")


def run_campaign(cfg: CampaignConfig, binary, offsets, instructions_path,
                 sm_version: Optional[str] = None):
    """Programmatic campaign entry point."""
    harness = SlotHarness(binary, offsets, instructions_path,
                          verbose=cfg.verbose)
    instructions = json.loads(Path(instructions_path).read_text()) \
        if isinstance(instructions_path, (str, Path)) else instructions_path

    if cfg.filter_safe:
        n_before = len(instructions)
        instructions = filter_safe_instructions(instructions)
        print(f"Safety filter: {n_before} validated -> {len(instructions)} safe "
              f"({n_before - len(instructions)} excluded: memory, control-flow, "
              f"barrier, sync)")

    rng = random.Random(cfg.seed)
    triples = generate_triples(instructions, cfg.pattern, cfg.reset_instruction,
                               cfg.max_triples, rng)

    sm_version = sm_version or detect_sm()
    if cfg.multi:
        run_multi_campaign(cfg, harness, instructions, triples, sm_version)
    else:
        run_single_campaign(cfg, harness, instructions, triples, sm_version)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EoSS campaign runner (single + multi-sequence)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,  # reusable as a parents= parser for `eoss campaign`
    )
    parser.add_argument("--binary", required=True)
    parser.add_argument("--offsets", required=True)
    parser.add_argument("--instructions", required=True)
    parser.add_argument("--mode", choices=["issue", "completion"], default="issue")
    parser.add_argument("--pattern", choices=["diagonal", "osiris", "sweep"],
                        default="diagonal")
    parser.add_argument("--reset-instruction", default=None,
                        help="Fixed RESET for sweep pattern")
    parser.add_argument("--output", default=None)
    parser.add_argument("--keep-binaries", metavar="DIR", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--isa-json", default=None)
    parser.add_argument("--all-instructions", default=None)
    parser.add_argument("--all-validated", default=None)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-triples", type=int, default=None)
    parser.add_argument("--multi", action="store_true",
                        help="Multi-sequence mode (4095 triples/launch)")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--filter-safe", action="store_true",
                        help="Exclude memory, control-flow, barrier, and sync "
                             "instructions (recommended for --multi)")
    parser.add_argument("--crash-recovery", action="store_true")
    parser.add_argument("--summary", default=None)
    return parser


def config_from_args(args) -> CampaignConfig:
    """Build a CampaignConfig from a parsed argparse namespace."""
    sm_version = detect_sm()
    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = args.output
    if not output:
        prefix = "multi" if args.multi else "campaign"
        output = f"{prefix}_{args.pattern}_{args.mode}_sm{sm_version}_{ts}.csv"
    summary = args.summary
    if args.multi and not summary:
        summary = output.replace('.csv', '_summary.csv')
    return CampaignConfig(
        mode=args.mode, pattern=args.pattern, multi=args.multi,
        iterations=args.iterations, trials=args.trials, seed=args.seed,
        filter_safe=args.filter_safe, crash_recovery=args.crash_recovery,
        reset_instruction=args.reset_instruction, max_triples=args.max_triples,
        keep_binaries=args.keep_binaries, verbose=args.verbose,
        output=output, summary=summary,
        isa_json=args.isa_json, all_instructions=args.all_instructions,
        all_validated=args.all_validated,
    )


def main(argv=None):
    args = build_argparser().parse_args(argv)
    cfg = config_from_args(args)
    run_campaign(cfg, args.binary, args.offsets, args.instructions,
                 detect_sm())


if __name__ == "__main__":
    main()
