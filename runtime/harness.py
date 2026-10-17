"""Harness building, offset extraction, and run/output protocols.

A *harness* is a CUDA kernel with patchable instruction slots and a fixed
measurement protocol. Two harness families are supported:

* ``osiris_slots`` — 4,096-slot timing harness (RESET/TRIGGER/MEASURE
  triples bracketed by CS2R clock reads; packaged template
  ``eoss/harnesses/osiris_slots.cu``).
* ``contention_chain`` — FADD-chain harness used by the contention
  experiment (source lives in ``Contention/``).

Offset extraction is unified here: cuobjdump disassembly + per-harness
anchor grammar + anchor-based code-base location, emitting the
``eoss.offsets/v2`` schema (:class:`eoss.schemas.OffsetsDoc`).
"""

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .. import paths
from ..binary.anchors import find_code_base
from ..binary.cuobjdump import find_kernel, get_sass_dump, parse_sass_instructions
from ..schemas import OffsetsDoc, Slot

DEFAULT_NVCC = "/usr/local/cuda/bin/nvcc"


def nvcc_path() -> str:
    env = os.environ.get("NVCC")
    if env:
        return env
    found = shutil.which("nvcc")
    if found:
        return found
    if os.path.isfile(DEFAULT_NVCC):
        return DEFAULT_NVCC
    raise RuntimeError("nvcc not found (set $NVCC or add CUDA bin/ to PATH)")


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

def build_harness(sm, harness: str = "osiris_slots", num_slots: int = 4096,
                  out_dir=".", source=None, output_name: Optional[str] = None,
                  extra_flags: Optional[List[str]] = None,
                  verbose: bool = True) -> Path:
    """Compile a harness .cu into an executable binary.

    Args:
        sm: SM version (e.g. 120 or "120").
        harness: packaged harness name ("osiris_slots") or ignored when
            ``source`` is given.
        num_slots: number of measurement slots (-DNUM_SLOTS).
        out_dir: output directory for the binary.
        source: explicit .cu path (overrides packaged template).
        output_name: binary file name (default ``harness_sm<SM>``).
        extra_flags: additional nvcc flags.
        verbose: print progress.

    Returns:
        Path to the compiled binary.
    """
    src = Path(source) if source else paths.harness_source(harness)
    if not src.is_file():
        raise FileNotFoundError(f"harness source not found: {src}")

    sm = str(sm)
    out = Path(out_dir) / (output_name or f"harness_sm{sm}")
    flags = [
        f"-gencode", f"arch=compute_{sm},code=sm_{sm}",
        "-O2", "--no-compress",
        f"-DNUM_SLOTS={num_slots}",
    ]
    if extra_flags:
        flags.extend(extra_flags)

    if verbose:
        print(f"[BUILD] nvcc {src.name} -> {out} (sm_{sm}, {num_slots} slots)")
    result = subprocess.run(
        [nvcc_path(), *flags, "-o", str(out), str(src)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvcc failed:\n{result.stderr}")
    return out


def build_harness_flavor(sm, flavor: str, num_slots: int = 4096,
                         out_dir=".", verbose: bool = True) -> Path:
    """Assemble and compile one harness *flavor* from the parametrized template
    (``harnesses/osiris_harness.cu``), enabling only the feature blocks the
    flavor needs. TMA pulls in the driver library. Returns the binary path.

    Raises KeyError if the flavor is unknown, and RuntimeError (from nvcc) if
    the generation cannot compile a feature — caller skips that flavor.
    """
    from ..isa.registry import HARNESS_FLAVORS
    if flavor not in HARNESS_FLAVORS:
        raise KeyError(f"unknown harness flavor {flavor!r}; "
                       f"have {sorted(HARNESS_FLAVORS)}")
    features = HARNESS_FLAVORS[flavor]
    flags = [f"-DFEAT_{feat}=1" for feat in features]
    if "TMA" in features:
        flags.append("-lcuda")
    return build_harness(sm, num_slots=num_slots, out_dir=out_dir,
                         source=paths.harness_source("osiris_harness"),
                         output_name=f"harness_{flavor}_sm{sm}",
                         extra_flags=flags, verbose=verbose)


# ---------------------------------------------------------------------------
# Offset extraction — per-harness anchor grammars
# ---------------------------------------------------------------------------

def _extract_osiris_slots(sass: str, kernel_name: str) -> List[Slot]:
    """OSIRIS grammar: FMUL = TRIGGER marker; adjacent FADDs = RESET/MEASURE;
    CS2R between TRIGGER and MEASURE = start clock; CS2R after = end clock."""
    instrs = parse_sass_instructions(sass, kernel_name)
    fmul = sorted(i.offset for i in instrs if i.has_mnemonic("FMUL"))
    fadd = sorted(i.offset for i in instrs if i.has_mnemonic("FADD"))
    cs2r = sorted(i.offset for i in instrs if i.has_mnemonic("CS2R"))

    slots = []
    remaining = list(fadd)
    for idx, trigger in enumerate(fmul):
        before = [f for f in remaining if f < trigger]
        after = [f for f in remaining if f > trigger]
        if not before or not after:
            continue
        reset = max(before)
        measure = min(after)
        anchors = {"reset": reset, "trigger": trigger, "measure": measure}
        start = [c for c in cs2r if trigger < c < measure]
        if start:
            anchors["cs2r_start"] = min(start)
        end = [c for c in cs2r if c > measure]
        if end:
            anchors["cs2r_end"] = min(end)
        if reset in remaining:
            remaining.remove(reset)
        if measure in remaining:
            remaining.remove(measure)
        slots.append(Slot(index=idx, anchors=anchors))
    return slots


def _extract_contention_chain(sass: str, kernel_name: str) -> Tuple[List[Slot], Dict]:
    """Contention grammar: CS2R CLOCKLO (start) ... FADD chain ... CS2R (end).

    Each chain FADD becomes a slot with a single ``measure`` anchor;
    the clock anchors are recorded in the returned extra dict.
    """
    instrs = parse_sass_instructions(sass, kernel_name)
    clocks = [i for i in instrs if i.has_mnemonic("CS2R") and "CLOCKLO" in i.text]
    if len(clocks) < 2:
        clocks = [i for i in instrs if i.has_mnemonic("CS2R")]
    if len(clocks) < 2:
        raise RuntimeError("could not find two CS2R clock reads in kernel")

    start_clock, end_clock = clocks[0], clocks[1]
    chain = [i for i in instrs
             if i.has_mnemonic("FADD") and start_clock.offset < i.offset < end_clock.offset]

    slots = [Slot(index=idx, anchors={"measure": ins.offset})
             for idx, ins in enumerate(chain)]
    extra = {
        "cs2r_start": {"offset": start_clock.offset, "instruction": start_clock.text},
        "cs2r_end": {"offset": end_clock.offset, "instruction": end_clock.text},
    }
    return slots, extra


def extract_offsets(binary_path, harness: str = "osiris_slots",
                    kernel_hint: Optional[str] = None,
                    output=None, verbose: bool = True) -> OffsetsDoc:
    """Extract patchable slot offsets from a compiled harness binary.

    Args:
        binary_path: compiled harness binary.
        harness: "osiris_slots" or "contention_chain".
        kernel_hint: substring to select the kernel function
            (default: "osiris_portable" / "contention_patchable").
        output: optional path to write the offsets JSON.
        verbose: print progress.
    """
    binary_path = Path(binary_path)
    sass = get_sass_dump(str(binary_path))

    if kernel_hint is None:
        kernel_hint = ("osiris_portable" if harness == "osiris_slots"
                       else "contention_patchable")
    kernel_name = find_kernel(sass, prefer=kernel_hint)
    if kernel_name is None:
        raise RuntimeError(f"no kernel found in {binary_path}")

    extra: Dict = {}
    if harness == "osiris_slots":
        slots = _extract_osiris_slots(sass, kernel_name)
        # Record the resource register map (global descriptor, texture/surface
        # handles, coordinate/address regs) so the generator's operand policy
        # can target the registers this harness keeps live at the measure slot.
        from .regmap import extract_register_map
        extra["register_map"] = extract_register_map(
            binary_path, kernel_hint=kernel_name, sass=sass).to_json()
    elif harness == "contention_chain":
        slots, extra = _extract_contention_chain(sass, kernel_name)
    else:
        raise ValueError(f"unknown harness grammar: {harness}")

    with open(binary_path, "rb") as f:
        binary_data = f.read()
    code_base = find_code_base(binary_data, sass, kernel_name)
    if code_base is None:
        raise RuntimeError("could not locate code base in binary")

    doc = OffsetsDoc(
        binary=str(binary_path),
        kernel_name=kernel_name,
        code_base_offset=code_base,
        harness=harness,
        slots=slots,
        extra=extra,
    )
    if output:
        import json
        with open(output, "w") as f:
            json.dump(doc.to_json(), f, indent=2)

    if verbose:
        n_inbounds = sum(
            1 for s in slots
            if 0 <= code_base + next(iter(s.anchors.values())) < len(binary_data) - 16)
        print(f"[EXTRACT] kernel={kernel_name} slots={len(slots)} "
              f"code_base=0x{code_base:x} in-bounds={n_inbounds}/{len(slots)}")
    return doc


# ---------------------------------------------------------------------------
# Running and output protocols
# ---------------------------------------------------------------------------

#: GNU coreutils ``timeout`` for OS-level hard kill of hung kernels.
_TIMEOUT_BIN = shutil.which("timeout")


def run_harness(binary_path, args: Optional[List[str]] = None,
                timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a (possibly patched) harness binary. Raises RuntimeError on
    non-zero exit.

    A patched instruction can hang the kernel forever on GPUs without a
    display watchdog (datacenter parts like the A30/A100/H100) — and
    ``subprocess.run``'s own SIGKILL does not reliably reap a process stuck
    in the CUDA driver. When available, wrap the run in coreutils ``timeout``
    (SIGTERM at the limit, SIGKILL 2s later), which terminates the hang and
    returns 124; a slightly longer ``subprocess`` timeout is the backstop.
    """
    abs_path = str(Path(binary_path).resolve())
    cmd = [abs_path, *(args or [])]
    if _TIMEOUT_BIN:
        cmd = [_TIMEOUT_BIN, "-k", "2", str(timeout), *cmd]
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=timeout + (10 if _TIMEOUT_BIN else 0),
    )
    if result.returncode != 0:
        err = result.stderr.strip()[:300] if result.stderr else "unknown"
        raise RuntimeError(f"Binary failed (rc={result.returncode}): {err}")
    return result


def parse_perslot_timing(stdout: str) -> List[Tuple[int, int]]:
    """Parse the PERSLOT_TIMING protocol: (slot_index, delta) pairs."""
    timings = []
    in_data = False
    for line in stdout.split("\n"):
        line = line.strip()
        if line == "PERSLOT_TIMING_START":
            in_data = True
            continue
        if line == "PERSLOT_TIMING_END":
            in_data = False
            continue
        if in_data and "," in line:
            parts = line.split(",")
            timings.append((int(parts[0]), int(parts[1])))
    return timings


def parse_contention_summary(stdout: str) -> Optional[Dict[str, str]]:
    """Parse the CONTENTION_RESULTS_START/END summary block (key=value pairs)."""
    m = re.search(r"CONTENTION_RESULTS_START\n(.+)\nCONTENTION_RESULTS_END",
                  stdout)
    if not m:
        return None
    out = {}
    for part in m.group(1).split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def parse_contention_csv(stdout: str) -> Dict:
    """Parse the contention harness's 12-column per-warp CSV output.

    Columns: run,block,warp,smid,ctaid_x,ctaid_y,ctaid_z,num_blocks,
    num_slots,start,end,delta

    Returns aggregate statistics plus the raw per-warp rows (replaces five
    copy-pasted parsers in the Contention plot scripts).
    """
    lines = stdout.strip().split("\n")
    if len(lines) < 2:
        return {"error": "No CSV data"}

    starts, ends, deltas = [], [], []
    warp_data = []
    for line in lines[1:]:  # skip header
        parts = line.split(",")
        if len(parts) < 12:
            continue
        try:
            row = {
                "run": int(parts[0]),
                "block": int(parts[1]),
                "warp": int(parts[2]),
                "smid": int(parts[3]),
                "ctaid_x": int(parts[4]),
                "ctaid_y": int(parts[5]),
                "ctaid_z": int(parts[6]),
                "start": int(parts[9]),
                "end": int(parts[10]),
                "delta": int(parts[11]),
            }
        except (ValueError, IndexError):
            continue
        starts.append(row["start"])
        ends.append(row["end"])
        deltas.append(row["delta"])
        warp_data.append(row)

    if not deltas:
        return {"error": "Could not parse CSV"}
    return {
        "avg_delta": sum(deltas) / len(deltas),
        "min_delta": min(deltas), "max_delta": max(deltas),
        "avg_start": sum(starts) / len(starts),
        "min_start": min(starts), "max_start": max(starts),
        "avg_end": sum(ends) / len(ends),
        "min_end": min(ends), "max_end": max(ends),
        "num_samples": len(deltas),
        "warp_data": warp_data,
        "error": None,
    }


def run_contention_harness(binary_path, num_blocks: int, runs: int = 10,
                           detailed: bool = False) -> Dict:
    """Run a contention harness binary and parse its output.

    ``detailed=True`` uses --csv and returns per-warp aggregates
    (:func:`parse_contention_csv`); otherwise parses the summary markers.
    """
    cmd = [str(Path(binary_path).resolve()),
           "--blocks", str(num_blocks), "--runs", str(runs)]
    if detailed:
        cmd.append("--csv")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "avg": -1, "min": -1, "max": -1}
    if result.returncode != 0:
        return {"error": result.stderr, "avg": -1, "min": -1, "max": -1}

    if detailed:
        return parse_contention_csv(result.stdout)

    m = re.search(
        r"CONTENTION_RESULTS_START\n.*avg=([0-9.]+),min=([0-9.]+),max=([0-9.]+)",
        result.stdout, re.DOTALL)
    if m:
        return {"avg": float(m.group(1)), "min": float(m.group(2)),
                "max": float(m.group(3)), "error": None}
    return {"error": "Could not parse output", "avg": -1, "min": -1, "max": -1}
