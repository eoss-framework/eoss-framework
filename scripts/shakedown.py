#!/usr/bin/env python3
"""One-shot cross-generation shakedown.

Builds the harness flavors the detected GPU supports (from the parametrized
template, per the capability table), runs the union instruction DB, and
reports runtime coverage + a per-family breakdown. Handles headless /
datacenter GPUs automatically: if persistence mode is off it holds a CUDA
keep-alive context so per-subprocess driver init doesn't dominate wall-clock.

Two ways to run:

  # on the GPU box (after syncing the repo + the target JSON/SM<nn> spec):
  PYTHONPATH=<parent-of-eoss> python3 eoss/scripts/shakedown.py

  # from a laptop — rsync the repo to the box and run it there in one shot:
  python3 scripts/shakedown.py --remote user@host --key ~/.ssh/id --sm 70

The union DB is written to JSON/SM<nn>/validated_gpr.json on the box.
"""
from __future__ import annotations

import argparse
import collections
import glob
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

FAMILIES = ["fadd", "imad", "mov", "lop3", "tex", "tld", "suld", "sust",
            "lds", "sts", "ldl", "stl", "atoms", "ldg", "stg", "ld", "st",
            "utmaldg", "utmastg", "ucgabar"]
_ALL_SMS = ("70", "72", "75", "80", "86", "87", "89", "90", "100", "120", "121")


# ---------------------------------------------------------------------------
# on-box shakedown
# ---------------------------------------------------------------------------

def _ensure_cuda_on_path() -> None:
    """Make nvcc/cuobjdump findable even if CUDA isn't on PATH."""
    from eoss.runtime.harness import nvcc_path
    try:
        nvcc_path()
        return
    except Exception:
        pass
    for binp in sorted(glob.glob("/usr/local/cuda*/bin"), reverse=True):
        if Path(binp, "nvcc").is_file():
            os.environ["PATH"] = binp + os.pathsep + os.environ.get("PATH", "")
            os.environ.setdefault("NVCC", str(Path(binp, "nvcc")))
            os.environ.setdefault("CUOBJDUMP", str(Path(binp, "cuobjdump")))
            return


def _persistence_off() -> bool:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=persistence_mode", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        return "disabled" in r.stdout.lower()
    except Exception:
        return False


def _start_keepalive():
    """Hold a CUDA context so the driver stays warm (userspace persistence
    mode). Returns (proc, files) or None."""
    from eoss.runtime.harness import nvcc_path
    src = Path(tempfile.gettempdir(), "eoss_keepalive.cu")
    exe = Path(tempfile.gettempdir(), "eoss_keepalive")
    src.write_text("#include <cuda_runtime.h>\n#include <unistd.h>\n"
                   "int main(){cudaSetDevice(0);cudaFree(0);sleep(86400);return 0;}\n")
    try:
        subprocess.run([nvcc_path(), "-o", str(exe), str(src)],
                       check=True, capture_output=True, timeout=180)
        proc = subprocess.Popen([str(exe)])
        time.sleep(2)
        return proc, (src, exe)
    except Exception as e:
        print(f"[keepalive] unavailable ({str(e)[:60]}) — running cold", flush=True)
        return None


def _build_stages(sm: str, slots: int):
    """Build classic + every supported flavor from the template; return the
    union stage list and the names actually built."""
    from eoss import paths
    from eoss.isa.registry import harness_flavors
    from eoss.runtime.harness import (build_harness, build_harness_flavor,
                                      extract_offsets)
    from eoss.schemas import OffsetsDoc
    d = paths.generated_dir(sm)
    d.mkdir(parents=True, exist_ok=True)

    cb = build_harness(sm, num_slots=slots, out_dir=d, verbose=False)
    extract_offsets(cb, output=d / f"offsets_sm{sm}.json", verbose=False)
    built = []
    for name in harness_flavors(sm):
        try:
            fb = build_harness_flavor(sm, name, num_slots=slots, out_dir=d,
                                      verbose=False)
            extract_offsets(fb, output=d / f"offsets_{name}_sm{sm}.json",
                            kernel_hint="osiris_kernel", verbose=False)
            built.append(name)
        except Exception as e:
            print(f"  flavor {name}: skipped ({str(e).splitlines()[-1][:70]})",
                  flush=True)

    stages = [{"label": "classic", "mode": "generate", "register_map": None,
               "binary": d / f"harness_sm{sm}",
               "offsets": d / f"offsets_sm{sm}.json"}]
    for name in built:
        op = d / f"offsets_{name}_sm{sm}.json"
        rmap = None if name == "cga" else OffsetsDoc.load(op).extra.get("register_map")
        stages.append({"label": name, "mode": "generate", "register_map": rmap,
                       "binary": d / f"harness_{name}_sm{sm}", "offsets": op})
    stages.append({"label": "harvest", "mode": "harvest", "kernel_hint": None,
                   "binary": d / f"harness_sm{sm}",
                   "offsets": d / f"offsets_sm{sm}.json"})
    if "super" in built:
        stages.append({"label": "harvest+", "mode": "harvest",
                       "kernel_hint": "osiris_kernel",
                       "binary": d / f"harness_super_sm{sm}",
                       "offsets": d / f"offsets_super_sm{sm}.json"})
    return stages, built


def shakedown(sm=None, slots=1024, run_timeout=4, out=None, limit=0,
              keepalive=True) -> "tuple[int, int]":
    _ensure_cuda_on_path()
    from eoss import paths
    from eoss.gpu import detect_sm, gpu_name
    from eoss.isa.model import ISASpec
    from eoss.isa.registry import arch_info, harness_flavors, normalize_sm
    from eoss.runtime.union import build_union_db

    sm = normalize_sm(sm) if sm else detect_sm()
    try:
        name = gpu_name()
    except Exception:
        name = "GPU"
    ai = arch_info(sm)
    gen = f", {ai.generation}" if ai else ""
    print(f"=== EoSS shakedown: {name} (sm_{sm}{gen}) ===", flush=True)
    if not paths.isa_json_path(sm).is_file():
        sys.exit(f"! no ISA JSON for sm_{sm} — sync JSON/SM{sm}/ to this box")
    spec = ISASpec.load(sm=sm)
    print(f"flavors this gen supports: {', '.join(harness_flavors(sm)) or '(none)'}",
          flush=True)

    ka = None
    if keepalive and _persistence_off():
        print("[keepalive] persistence mode off -> holding a CUDA context "
              "(warm driver)", flush=True)
        ka = _start_keepalive()
    try:
        print("building harnesses…", flush=True)
        stages, built = _build_stages(sm, slots)
        print(f"built: classic + {', '.join(built) or '(none)'} "
              f"({slots} slots)", flush=True)

        classes = set(list(spec.instructions)[::max(1, len(spec.instructions) // limit)]) \
            if limit else None
        if classes:
            print(f"[sample] restricting to {len(classes)} of {len(spec)} classes",
                  flush=True)

        outp = Path(out) if out else paths.generated_dir(sm) / "validated_gpr.json"
        t0 = time.time()
        recs, summ = build_union_db(
            spec, stages, output_path=outp, run_timeout=run_timeout,
            classes=classes,
            on_progress=lambda l, n, r: print(f"  {l:9} +{n}", flush=True))
        dt = time.time() - t0
    finally:
        if ka:
            ka[0].kill()
            for f in ka[1]:
                try: f.unlink()
                except OSError: pass

    denom = len(classes) if classes else len(spec)
    print(f"\nCOVERAGE: {len(recs)}/{denom} = {len(recs) / denom * 100:.1f}%"
          f"  ({dt:.0f}s{' sampled' if classes else ''})", flush=True)
    if not classes:  # per-family denominators are full-ISA — only meaningful on a full run
        by = collections.Counter(re.split(r"_", r.instruction)[0] for r in recs)
        present = [f for f in FAMILIES
                   if any(re.split(r"_", n)[0] == f for n in spec.instructions)]
        fam = "  ".join(
            f"{f} {sum(v for k, v in by.items() if k == f)}/"
            f"{sum(1 for n in spec.instructions if re.split(r'_', n)[0] == f)}"
            for f in present)
        print("by family:", fam, flush=True)
    print(f"wrote {outp}", flush=True)
    return len(recs), denom


# ---------------------------------------------------------------------------
# --remote: rsync the repo to a box and run the shakedown there
# ---------------------------------------------------------------------------

_SSH_OPTS = ["-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20"]


def _ssh_base(key, port=None):
    return (["ssh", *_SSH_OPTS]
            + (["-p", str(port)] if port else [])
            + (["-i", key] if key else []))


def _ssh_e(key, port=None):
    """The rsync -e transport string (must match _ssh_base's port/key)."""
    return ("ssh -o StrictHostKeyChecking=accept-new"
            + (f" -p {port}" if port else "")
            + (f" -i {key}" if key else ""))


def _detect_remote_sm(host, key, port=None):
    r = subprocess.run(
        _ssh_base(key, port) + [host, "nvidia-smi --query-gpu=compute_cap "
                                "--format=csv,noheader | head -1"],
        capture_output=True, text=True, timeout=30)
    cc = r.stdout.strip().splitlines()[0].strip() if r.stdout.strip() else ""
    return cc.replace(".", "") or None


def run_remote(args) -> int:
    repo = Path(__file__).resolve().parent.parent           # the eoss package dir
    key = args.key
    port = args.port
    sm = args.sm
    if not sm:
        print("detecting remote GPU…", flush=True)
        sm = _detect_remote_sm(args.remote, key, port)
        if not sm:
            sys.exit("! could not detect remote SM; pass --sm")
        print(f"remote GPU is sm_{sm}", flush=True)

    rdir = args.remote_dir.rstrip("/")
    subprocess.run(_ssh_base(key, port) + [args.remote, f"mkdir -p {rdir}/eoss"],
                   check=True)
    excludes = ["--exclude=.git", "--exclude=__pycache__", "--exclude=*.egg-info",
                "--exclude=harness_*sm*", "--exclude=offsets_*sm*.json",
                "--exclude=validated_gpr.json", "--exclude=validated_rebuilt.json",
                "--exclude=all_instructions.json"]
    excludes += [f"--exclude=JSON/SM{o}" for o in _ALL_SMS if o != str(sm)]
    ssh_e = _ssh_e(key, port)
    print(f"syncing repo -> {args.remote}:{rdir}/eoss (only JSON/SM{sm})…", flush=True)
    subprocess.run(["rsync", "-az", "-e", ssh_e, *excludes,
                    f"{repo}/", f"{args.remote}:{rdir}/eoss/"], check=True)

    py = args.python or ("$(command -v python3.12 python3.11 python3.10 python3 "
                         "2>/dev/null | head -1)")
    remote_cmd = (
        f"cd {rdir}/eoss && export PATH=/usr/local/cuda/bin:$PATH && "
        f"find . -name __pycache__ -exec rm -rf {{}} + 2>/dev/null; "
        f"PYTHONPATH={rdir} {py} scripts/shakedown.py --sm {sm} "
        f"--slots {args.slots} --run-timeout {args.run_timeout}"
        + (f" --limit {args.limit}" if args.limit else "")
        + (" --no-keepalive" if args.no_keepalive else ""))
    print("running shakedown on the box…\n", flush=True)
    rc = subprocess.run(_ssh_base(key, port) + [args.remote, remote_cmd]).returncode

    if rc == 0 and not args.no_fetch:
        localdir = repo / "JSON" / f"SM{sm}"
        localdir.mkdir(parents=True, exist_ok=True)
        print(f"\nfetching results -> {localdir}/ …", flush=True)
        subprocess.run(["rsync", "-az", "-e", ssh_e,
                        "--include=validated_gpr.json", "--include=offsets_*.json",
                        "--exclude=*",
                        f"{args.remote}:{rdir}/eoss/JSON/SM{sm}/", f"{localdir}/"])
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description="EoSS cross-generation shakedown")
    ap.add_argument("--sm", help="SM version (default: detect on the GPU)")
    ap.add_argument("--slots", type=int, default=1024,
                    help="harness slots (1024/2048/4096)")
    ap.add_argument("--run-timeout", type=int, default=4,
                    help="per-instruction validation timeout (s)")
    ap.add_argument("--limit", type=int, default=0,
                    help="validate only ~N stratified classes (quick estimate)")
    ap.add_argument("--out", help="write the validated DB here")
    ap.add_argument("--no-keepalive", action="store_true",
                    help="don't hold a CUDA keep-alive context")
    ap.add_argument("--remote", help="user@host — rsync the repo and run there")
    ap.add_argument("--key", help="ssh key for --remote")
    ap.add_argument("--port", type=int, help="ssh port for --remote (default 22)")
    ap.add_argument("--python", help="remote python (default: autodetect 3.1x)")
    ap.add_argument("--remote-dir", default="~/eoss-work",
                    help="remote working dir for --remote")
    ap.add_argument("--no-fetch", action="store_true",
                    help="don't copy the validated DB back to local JSON/SM<nn>/")
    args = ap.parse_args()
    if args.remote:
        return run_remote(args)
    shakedown(sm=args.sm, slots=args.slots, run_timeout=args.run_timeout,
              out=args.out, limit=args.limit, keepalive=not args.no_keepalive)
    return 0


if __name__ == "__main__":
    sys.exit(main())
