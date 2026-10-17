#!/usr/bin/env python3
"""Slice per-arch campaigns into pipeline buckets and render one multi-arch
violin per major pipeline (compute / uniform datapath / memory).

Per-op MIN aggregation: each instruction contributes its *minimum* hot_delta
over all iterations/slots. In the multi-mode campaign an op's latency can be
inflated when it is packed next to a neighbor whose scoreboard stall bleeds into
its slot; a stall can only ADD cycles, so the min over placements is the
uncontaminated latency. This removes packing artifacts without suppressing any
real high-latency op (whose samples are identical across placements, so
min == median for them). Verified against single-sequence re-measurement.
"""
import csv, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VIOLIN = HERE.parent / "plots" / "violin.py"
SLICES = HERE / "slices"
SLICES.mkdir(exist_ok=True)

# arch -> (compute_csv, memory_csv). Memory prefers the super-harness run.
ARCHES = {
    "SM70":  ("campaign_sm70_compute.csv",       "campaign_sm70_mem.csv"),
    "SM80":  ("campaign_sm80_compute.csv",       "campaign_sm80_mem.csv"),
    "SM86":  ("campaign_sm86_compute.csv",       "campaign_sm86_mem.csv"),
    "SM89":  ("campaign_sm89_compute.csv",       "campaign_sm89_mem.csv"),
    "SM90":  ("campaign_sm90_compute.csv",       "campaign_sm90_mem.csv"),
    "SM100": ("campaign_sm100_compute.csv",      "campaign_sm100_mem.csv"),
    "SM120": ("campaign_sm120_compute_full.csv", "campaign_sm120_mem.csv"),
}

def _mem_csv(mem_name):
    sup = HERE / mem_name.replace(".csv", "_super.csv")
    return sup if sup.exists() else (HERE / mem_name)

def _fam(n):
    return re.split(r"__|_", n)[0]

def load_atomic_family():
    """{SMxx: [L2-atomic latencies]} from the dependent-atomic microbench.
    Real op-family (add/min/max/.../cas) where measured; a single representative
    value for archs whose boxes were unavailable (the family is op-invariant)."""
    p = HERE / "atomic_family_data.csv"
    out = {}
    if p.exists():
        with open(p) as f:
            for r in csv.DictReader(f):
                out.setdefault("SM" + r["sm"], []).append(int(r["latency_cy"]))
    return out

def min_per_op(csv_path, keep):
    """{measure: min hot_delta} over rows where keep(measure) is True."""
    perop = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r.get("error", "") not in ("", "OK", "None"):
                continue
            n = r["measure"]
            if not keep(n):
                continue
            try:
                v = int(r["hot_delta"])
            except (ValueError, KeyError):
                continue
            if n not in perop or v < perop[n]:
                perop[n] = v
    return perop

def write_slice(path, perop):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["measure", "hot_delta"])
        for n, v in perop.items():
            w.writerow([n, v])
    return path

def render(bucket, arch_csvs, ymax):
    if not arch_csvs:
        print(f"[skip] {bucket}: no data"); return
    base = [sys.executable, str(VIOLIN)]
    for arch, path in arch_csvs.items():
        base += ["--csv", f"{arch}={path}"]
    for ext in ("pdf", "png"):
        subprocess.run(base + ["-o", str(HERE / f"matrix_{bucket}.{ext}"),
                               "--max-latency", str(ymax)], check=True)
    print(f"[ok] {bucket}: {', '.join(arch_csvs)} -> matrix_{bucket}.pdf/.png")

def main():
    compute_csvs, uniform_csvs, memory_csvs = {}, {}, {}
    atomic = load_atomic_family()
    isu = lambda n: n.lower().startswith("u")
    for arch, (cc, mc) in ARCHES.items():
        cc_p, mc_p = HERE / cc, _mem_csv(mc)
        if cc_p.exists():
            comp = min_per_op(cc_p, lambda n: not isu(n))
            unif = min_per_op(cc_p, isu)
            if comp:
                compute_csvs[arch] = write_slice(SLICES / f"compute_{arch}.csv", comp)
            if unif:
                uniform_csvs[arch] = write_slice(SLICES / f"uniform_{arch}.csv", unif)
        if mc_p.exists():
            # Drop OSIRIS atomg/redg (only 2 archs have them) and fold in the
            # uniform microbench L2-atomic family as the memory hierarchy's top
            # tier, so it's a violin lobe on every arch rather than a marker.
            mem = min_per_op(mc_p, lambda n: _fam(n) not in ("atomg", "redg"))
            for i, lat in enumerate(atomic.get(arch, [])):
                mem[f"atomL2_{i:02d}"] = lat
            memory_csvs[arch] = write_slice(SLICES / f"memory_{arch}.csv", mem)
            print(f"  memory {arch}: {mc_p.name} -> {len(mem)} ops "
                  f"(+{len(atomic.get(arch, []))} atomic)")
    print("compute:", list(compute_csvs), "| uniform:", list(uniform_csvs),
          "| memory:", list(memory_csvs))
    render("compute", compute_csvs, 60)
    render("uniform", uniform_csvs, 60)
    render("memory",  memory_csvs,  500)

if __name__ == "__main__":
    main()
