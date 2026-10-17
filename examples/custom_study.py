#!/usr/bin/env python3
"""Example custom study built from the EoSS programmatic API.

This is a template for writing your own experiment on top of the package rather
than through the bundled case studies. It measures the completion latency of one
representative encoding per instruction *family*, across several functional-unit
pipelines, on the detected GPU.

The pattern is the whole point — a custom study is three steps:

    1. query the ISA           (eoss.ISASpec)
    2. build the harness once   (eoss.runtime.build_harness / extract_offsets)
    3. measure whatever you want (SlotHarness.measure_triple)

Usage:
    python3 examples/custom_study.py                  # default family set
    python3 examples/custom_study.py mufu dadd fadd   # your own families
"""
import json
import sys

from eoss import ISASpec, paths
from eoss.gpu import detect_sm
from eoss.runtime import build_harness, extract_offsets, SlotHarness

# One family per functional-unit pipeline (FP add, FMA, int MAD, SFU
# transcendental, bit-count, find-leading-one, FP64 add).
DEFAULT_FAMILIES = ["fadd", "ffma", "imad", "mufu", "popc", "flo", "dadd"]


def family_of(name: str) -> str:
    return name.split("__")[0].split("_")[0]


def representative(validated, family: str):
    """First validated class in `family`, preferring a register-only variant."""
    cands = sorted(n for n in validated if family_of(n) == family)
    cands.sort(key=lambda n: ("RR" not in n.split("__")[-1], n))
    return cands[0] if cands else None


def main(argv):
    families = argv or DEFAULT_FAMILIES
    sm = detect_sm()
    spec = ISASpec.load(sm=sm)                       # 1. query the ISA
    gd = paths.generated_dir(sm)
    val_path = gd / "validated_gpr.json"
    if not val_path.is_file():
        sys.exit(f"no validated set at {val_path} — run `eoss full --sm {sm}` first")
    validated = [r["instruction"] for r in json.load(open(val_path))]

    # 2. build the harness once (reuse a prebuilt one if present)
    binary = gd / f"harness_sm{sm}"
    offsets = gd / f"offsets_sm{sm}.json"
    if not (binary.is_file() and offsets.is_file()):
        binary = build_harness(sm, out_dir=str(gd))
        offsets = extract_offsets(binary, output=gd / f"offsets_sm{sm}.json")
    h = SlotHarness(str(binary), offsets, str(val_path))

    # a benign reference instruction for the reset/trigger slots
    benign = "fadd__RRR_RR" if "fadd__RRR_RR" in h.instr_by_name else validated[0]

    print(f"# {spec.sm}: completion latency (median cycles), reference={benign}")
    rows = []
    for fam in families:
        name = representative(validated, fam)
        if name is None:
            print(f"  {fam:8} (no validated class)")
            continue
        res = h.measure_triple("completion", benign, benign, name, trials=3)  # 3. measure
        s = res.get("summary")
        if s:
            rows.append((fam, name, s["hot_median"]))

    print(f"{'family':8} {'class':26} {'cycles':>7}")
    for fam, name, cyc in sorted(rows, key=lambda r: r[2]):
        print(f"{fam:8} {name:26} {cyc:7}")


if __name__ == "__main__":
    main(sys.argv[1:])
