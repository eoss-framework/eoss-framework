"""EoSS unified command-line interface.

Commands:

    eoss gpu info                          GPU / SM detection
    eoss spec build --sm 120               Rosetta String -> ISA JSON
    eoss isa stats --sm 120                ISA JSON statistics
    eoss sr list --sm 120                  special-register table
    eoss generate --sm 120                 bulk candidate encodings
    eoss encode --sm 120 --class fadd__RRR_RR --set Rd=R5 ...
    eoss build --sm 120                    compile harness + extract offsets
    eoss validate --sm 120                 runtime validation on the GPU
    eoss measure --triple a,b,c ...        one triple, hot vs cold
    eoss campaign ...                      timing-differential campaign
    eoss patch binary --json ... --target FADD --opcode-only FFMA
    eoss tui binary.cubin                  curses binary modifier
    eoss full --sm 120                     build + generate + validate
"""

import argparse
import sys
from pathlib import Path

from . import paths
from .gpu import detect_sm, gpu_name, sm_count
from .isa.registry import arch_info


def _resolve_sm(args) -> str:
    sm = getattr(args, "sm", None)
    if sm:
        return str(sm)
    return detect_sm()


def _isa_json(sm: str):
    return paths.isa_json_path(sm)


def _generated(sm: str):
    return paths.generated_dir(sm) / "all_instructions.json"


def _validated(sm: str):
    return paths.generated_dir(sm) / "validated_gpr.json"


# ---------------------------------------------------------------------------
# gpu
# ---------------------------------------------------------------------------

def cmd_gpu_info(args) -> int:
    sm = detect_sm()
    print(f"GPU:      {gpu_name()}")
    print(f"SM:       {sm} (sm_{sm})")
    info = arch_info(sm)
    if info:
        print(f"Arch:     {info.generation}")
    print(f"SM count: {sm_count()}")
    print(f"ISA JSON: {_isa_json(sm)} "
          f"({'present' if _isa_json(sm).is_file() else 'MISSING'})")
    return 0


# ---------------------------------------------------------------------------
# spec
# ---------------------------------------------------------------------------

def cmd_spec_build(args) -> int:
    from .spec.rosetta import main as rosetta_main
    argv = ["--sm", str(args.sm)]
    if args.sass_dir:
        argv += ["--sass-dir", args.sass_dir]
    argv += ["--output", args.output or str(_isa_json(args.sm))]
    rosetta_main(argv)
    return 0


# ---------------------------------------------------------------------------
# isa / sr
# ---------------------------------------------------------------------------

def cmd_isa_stats(args) -> int:
    from .isa.model import ISASpec
    spec = ISASpec.load(sm=args.sm)
    n_classes = len(spec)
    n_opcodes = len(spec.defined_opcodes())
    print(f"SM{spec.sm}: {n_classes} instruction classes, "
          f"{n_opcodes} unique opcodes")
    print(f"  source: {spec.source}")
    gen = _generated(args.sm)
    val = _validated(args.sm)
    if gen.is_file():
        import json
        print(f"  generated: {len(json.load(open(gen)))} ({gen})")
    if val.is_file():
        import json
        print(f"  validated: {len(json.load(open(val)))} ({val})")
    return 0


def cmd_isa_list(args) -> int:
    from .isa.model import ISASpec
    spec = ISASpec.load(sm=args.sm)
    names = sorted(spec.instructions.keys())
    if args.filter:
        f = args.filter.lower()
        names = [n for n in names if f in n.lower()]
    for n in names:
        print(n)
    print(f"# {len(names)} classes", file=sys.stderr)
    return 0


def cmd_sr_list(args) -> int:
    from .isa.model import ISASpec
    spec = ISASpec.load(sm=args.sm)
    aliases = {}
    for name, idx in spec.special_register_aliases.items():
        if name != f"SR{idx}":
            aliases.setdefault(idx, []).append(name)
    for idx, name in sorted(spec.special_registers.items()):
        suffix = f" (aliases: {', '.join(aliases[idx])})" if idx in aliases else ""
        print(f"  SR{idx:<4d} (0x{idx:02x})  {name}{suffix}")
    return 0


# ---------------------------------------------------------------------------
# generate / encode
# ---------------------------------------------------------------------------

def cmd_generate(args) -> int:
    from .enc.generator import main as gen_main
    output = args.output or _generated(args.sm)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    argv = ["--json", str(_isa_json(args.sm)), "--output", str(output)]
    if args.all_rz:
        argv.append("--all-rz")
    if args.zero_immediates:
        argv.append("--zero-immediates")
    if args.all_variants:
        argv.append("--all-variants")
    k = getattr(args, "encodings_per_class", 1)
    if k and k > 1:
        argv += ["--encodings-per-class", str(k)]
    if args.max_instructions:
        argv += ["--max-instructions", str(args.max_instructions)]
    if args.seed is not None:
        argv += ["--seed", str(args.seed)]
    if args.cuobjdump_validate:
        argv += ["--cuobjdump-validate", "--harness", str(args.binary),
                 "--offsets", str(args.offsets)]
    if args.verbose:
        argv.append("--verbose")
    gen_main(argv)
    return 0


def cmd_encode(args) -> int:
    from .enc.generator import Encoder
    enc = Encoder(sm=args.sm, seed=args.seed)
    fields = {}
    for kv in args.set or []:
        if "=" not in kv:
            print(f"Error: --set expects NAME=VALUE, got {kv!r}", file=sys.stderr)
            return 2
        k, v = kv.split("=", 1)
        fields[k] = v
    ins = enc.build(args.class_name, fields,
                    stall=args.stall, opex=args.opex)
    print(f"class:  {args.class_name}")
    print(f"bytes:  {ins.hex}")
    print(f"valid:  {ins.valid}")
    if ins.failed_conditions:
        print(f"failed: {', '.join(ins.failed_conditions)}")
    c = ins.control
    print(f"control: stall={c['stall']} yield={c['yield_']} "
          f"wr={c['wr_bar']} rd={c['rd_bar']} wt={c['wt_bar']}")
    if args.values:
        for k, v in sorted(ins.argument_values.items()):
            print(f"  {k} = {v}")
    return 0 if ins.valid else 1


# ---------------------------------------------------------------------------
# build / validate / measure / campaign / full
# ---------------------------------------------------------------------------

def cmd_build(args) -> int:
    from .runtime.harness import build_harness, extract_offsets
    sm = _resolve_sm(args)
    binary = build_harness(sm, harness=args.harness, num_slots=args.slots,
                           out_dir=args.out_dir, source=args.source)
    out = Path(args.out_dir) / f"offsets_sm{sm}.json"
    extract_offsets(binary, harness=args.harness, output=out)
    print(f"[BUILD] binary:  {binary}")
    print(f"[BUILD] offsets: {out}")
    return 0


def cmd_validate(args) -> int:
    from .runtime.validate import validate_instructions
    sm = _resolve_sm(args)
    instructions = args.instructions or _generated(sm)
    output = args.output or _validated(sm)
    keep = args.keep_binaries or str(Path(args.out_dir) / "validated_binaries")
    validate_instructions(args.binary, args.offsets, instructions, output,
                          verbose=args.verbose, keep_binaries=keep)
    return 0


def cmd_measure(args) -> int:
    from .runtime.measure import SlotHarness
    h = SlotHarness(args.binary, args.offsets, args.instructions,
                    verbose=args.verbose)
    if args.sanity_check:
        return 0 if h.run_sanity_check(mode=args.mode) else 1
    if not args.instructions:
        print("Error: --instructions required for measurement", file=sys.stderr)
        return 2
    parts = (args.triple or "").split(",")
    if len(parts) != 3:
        print("Error: --triple must be reset,trigger,measure", file=sys.stderr)
        return 2
    res = h.measure_triple(args.mode, parts[0], parts[1], parts[2],
                           trials=args.trials)
    s = res.get("summary")
    if s:
        print(f"hot:  mean={s['hot_mean']:.2f} median={s['hot_median']}")
        print(f"cold: mean={s['cold_mean']:.2f} median={s['cold_median']}")
        print(f"diff: mean={s['diff_mean']:.2f} median={s['diff_median']} "
              f"range=[{s['diff_min']}, {s['diff_max']}] (n={s['n_samples']})")
    return 0


def cmd_campaign(args) -> int:
    from .gpu import detect_sm as _detect
    from .runtime.campaign import config_from_args, run_campaign
    cfg = config_from_args(args)
    run_campaign(cfg, args.binary, args.offsets, args.instructions, _detect())
    return 0


def cmd_full(args) -> int:
    """build + generate + validate."""
    sm = str(args.sm)
    cmd_build(args)
    binary = str(Path(args.out_dir) / f"harness_sm{sm}")
    offsets = str(Path(args.out_dir) / f"offsets_sm{sm}.json")

    gen_args = argparse.Namespace(
        sm=sm, output=str(_generated(sm)), all_rz=True, zero_immediates=True,
        all_variants=False, max_instructions=None, seed=None,
        cuobjdump_validate=True, binary=binary, offsets=offsets,
        verbose=args.verbose)
    _generated(sm).parent.mkdir(parents=True, exist_ok=True)
    cmd_generate(gen_args)

    val_args = argparse.Namespace(
        sm=sm, binary=binary, offsets=offsets,
        instructions=str(_generated(sm)), output=None,
        keep_binaries=None, out_dir=args.out_dir, verbose=args.verbose)
    cmd_validate(val_args)

    print(f"\nPipeline complete for SM{sm}")
    print(f"  Generated: {_generated(sm)}")
    print(f"  Validated: {_validated(sm)}")
    return 0


# ---------------------------------------------------------------------------
# patch (absorbs VisualModifier/patch_sass.py)
# ---------------------------------------------------------------------------

def cmd_patch(args) -> int:
    from .binary.patcher import BinaryPatcher
    from .isa.model import ISASpec

    spec = ISASpec.load(path=args.json)
    target_opcodes = spec.opcodes_for_mnemonic(args.target)
    if not target_opcodes:
        print(f"Error: no opcodes found for target {args.target!r}",
              file=sys.stderr)
        return 1

    if args.replace_hex:
        mode, replacement = "full", bytes.fromhex(args.replace_hex.replace("_", ""))
        if len(replacement) != 16:
            print("Error: --replace-hex must be 16 bytes (32 hex chars)",
                  file=sys.stderr)
            return 2
    elif args.instr_hex:
        mode, replacement = "instr", bytes.fromhex(args.instr_hex.replace("_", ""))
        if len(replacement) != 8:
            print("Error: --instr-hex must be 8 bytes (16 hex chars)",
                  file=sys.stderr)
            return 2
    else:
        mode = "opcode"
        opcodes = spec.opcodes_for_mnemonic(args.opcode_only)
        if not opcodes:
            print(f"Error: unknown replacement mnemonic {args.opcode_only!r}",
                  file=sys.stderr)
            return 1
        if len(opcodes) > 1:
            print(f"Warning: {args.opcode_only} maps to {len(opcodes)} opcodes, "
                  f"using {min(opcodes):#06x}", file=sys.stderr)
        replacement = min(opcodes)

    ctrl_overrides = {}
    if args.stall is not None:
        ctrl_overrides["stall"] = args.stall
    if args.yield_flag is not None:
        ctrl_overrides["yield_"] = 1 if args.yield_flag else 0
    if args.wr_bar is not None:
        ctrl_overrides["wr_bar"] = args.wr_bar
    if args.rd_bar is not None:
        ctrl_overrides["rd_bar"] = args.rd_bar
    if args.wt_bar is not None:
        ctrl_overrides["wt_bar"] = args.wt_bar
    if mode == "full" and ctrl_overrides:
        print("Error: control overrides require --instr-hex or --opcode-only",
              file=sys.stderr)
        return 2

    patcher = BinaryPatcher(args.binary)
    if not patcher.sections:
        print("Error: no CUDA .text sections found", file=sys.stderr)
        return 1

    if args.dry_run:
        probe = BinaryPatcher.from_bytes(bytes(patcher.data), str(args.binary))
        count = probe.patch_by_opcode(target_opcodes, replacement, mode,
                                      ctrl_overrides, args.after_n, args.verbose)
        print(f"DRY RUN: {count} instruction(s) would be patched")
        return 0

    count = patcher.patch_by_opcode(target_opcodes, replacement, mode,
                                    ctrl_overrides, args.after_n, args.verbose)
    if count == 0:
        print("No matching instructions found")
        return 0
    out = args.output or (args.binary if args.in_place else str(args.binary) + ".patched")
    patcher.save(out)
    print(f"Patched {count} instruction(s), written to {out}")
    return 0


def cmd_tui(args) -> int:
    """Launch the curses binary modifier (TUI port of VisualModifier)."""
    from .isa.model import ISASpec
    from .tui.model import EditorModel, infer_sm_from_file

    binary = Path(args.binary)
    if not binary.is_file():
        print(f"Error: no such file: {binary}", file=sys.stderr)
        return 1

    if args.json:
        spec = ISASpec.load(path=args.json)
    else:
        sm = args.sm or infer_sm_from_file(str(binary))
        if not sm:
            print("Error: could not infer SM version from the binary; "
                  "pass --sm or --json", file=sys.stderr)
            return 2
        json_path = _isa_json(str(sm))
        if not json_path.is_file():
            print(f"Error: no ISA JSON for SM{sm} ({json_path}); pass --json",
                  file=sys.stderr)
            return 2
        spec = ISASpec.load(sm=sm)

    model = EditorModel(str(binary), spec)
    if not model.sections:
        print("Warning: no CUDA .text sections found; "
              "instruction view will be empty", file=sys.stderr)

    from .tui.app import run
    run(model, theme=args.theme)
    if model.n_modified:
        plural = "s" if model.n_modified != 1 else ""
        print(f"{model.n_modified} modified byte{plural} not saved "
              f"(original untouched: {binary})")
    return 0


def cmd_plot(args) -> int:
    from .plots import hotcold, latex_tables, violin
    targets = {"violin": violin, "hotcold": hotcold, "latex": latex_tables}
    mod = targets[args.plot_kind]
    argv = list(args.argv or [])
    if argv and argv[0] == "--":
        argv = argv[1:]
    mod.main(argv or None)
    return 0


# ---------------------------------------------------------------------------
# argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eoss",
        description="EoSS: systematic exploration of NVIDIA's native SASS ISA")
    p.add_argument("--version", action="store_true", help="print version and exit")
    sub = p.add_subparsers(dest="command")

    # gpu info
    pg = sub.add_parser("gpu", help="GPU queries")
    pg.add_argument("info", nargs="?", default="info")
    pg.set_defaults(func=cmd_gpu_info)

    # spec build
    ps = sub.add_parser("spec", help="Rosetta String -> ISA JSON")
    ssub = ps.add_subparsers(dest="subcommand")
    psb = ssub.add_parser("build", help="build ISA JSON from Rosetta Strings")
    psb.add_argument("--sm", required=True)
    psb.add_argument("--sass-dir", default=None)
    psb.add_argument("--output", default=None)
    psb.set_defaults(func=cmd_spec_build)

    # isa
    pi = sub.add_parser("isa", help="ISA JSON queries")
    isub = pi.add_subparsers(dest="subcommand")
    pis = isub.add_parser("stats", help="ISA statistics")
    pis.add_argument("--sm", required=True)
    pis.set_defaults(func=cmd_isa_stats)
    pil = isub.add_parser("list", help="list instruction classes")
    pil.add_argument("--sm", required=True)
    pil.add_argument("--filter", default=None)
    pil.set_defaults(func=cmd_isa_list)

    # sr
    psr = sub.add_parser("sr", help="special registers")
    srsub = psr.add_subparsers(dest="subcommand")
    psrl = srsub.add_parser("list", help="list special registers")
    psrl.add_argument("--sm", required=True)
    psrl.set_defaults(func=cmd_sr_list)

    # generate
    pgen = sub.add_parser("generate", help="bulk candidate encoding generation")
    pgen.add_argument("--sm", required=True)
    pgen.add_argument("--output", default=None)
    pgen.add_argument("--all-rz", action="store_true", default=True)
    pgen.add_argument("--no-all-rz", dest="all_rz", action="store_false")
    pgen.add_argument("--zero-immediates", action="store_true", default=True)
    pgen.add_argument("--all-variants", action="store_true")
    pgen.add_argument("--encodings-per-class", type=int, default=1,
                      help="emit up to K distinct constraint-satisfying "
                           "encodings per class (needs --no-all-rz to vary "
                           "operands; default 1)")
    pgen.add_argument("--max-instructions", type=int, default=None)
    pgen.add_argument("--seed", type=int, default=None)
    pgen.add_argument("--cuobjdump-validate", action="store_true")
    pgen.add_argument("--binary", default=None)
    pgen.add_argument("--offsets", default=None)
    pgen.add_argument("--verbose", action="store_true")
    pgen.set_defaults(func=cmd_generate)

    # encode
    pe = sub.add_parser("encode", help="encode one instruction with field overrides")
    pe.add_argument("--sm", required=True)
    pe.add_argument("--class", dest="class_name", required=True,
                    help="instruction class name (e.g. fadd__RRR_RR)")
    pe.add_argument("--set", action="append", metavar="NAME=VALUE",
                    help="field override (registers like R5/RZ/PT, ints, hex)")
    pe.add_argument("--stall", type=int, default=None)
    pe.add_argument("--opex", type=int, default=None)
    pe.add_argument("--seed", type=int, default=0)
    pe.add_argument("--values", action="store_true",
                    help="print all resolved field values")
    pe.set_defaults(func=cmd_encode)

    # build
    pb = sub.add_parser("build", help="compile harness + extract offsets")
    pb.add_argument("--sm", default=None, help="default: auto-detect from GPU")
    pb.add_argument("--harness", default="osiris_slots")
    pb.add_argument("--slots", type=int, default=4096)
    pb.add_argument("--source", default=None, help="custom .cu source")
    pb.add_argument("--out-dir", default=".")
    pb.set_defaults(func=cmd_build)

    # validate
    pv = sub.add_parser("validate", help="runtime validation on the GPU")
    pv.add_argument("--sm", default=None)
    pv.add_argument("--binary", required=True)
    pv.add_argument("--offsets", required=True)
    pv.add_argument("--instructions", default=None,
                    help="default: <isa-dir>/SMxx/all_instructions.json")
    pv.add_argument("--output", default=None,
                    help="default: <isa-dir>/SMxx/validated_gpr.json")
    pv.add_argument("--keep-binaries", default=None, metavar="DIR")
    pv.add_argument("--out-dir", default=".")
    pv.add_argument("--verbose", action="store_true")
    pv.set_defaults(func=cmd_validate)

    # measure
    pm = sub.add_parser("measure", help="measure one instruction triple")
    pm.add_argument("--binary", required=True)
    pm.add_argument("--offsets", required=True)
    pm.add_argument("--instructions", default=None,
                    help="instruction encodings JSON (not needed for --sanity-check)")
    pm.add_argument("--mode", choices=["issue", "completion"], default="issue")
    pm.add_argument("--triple", default=None, help="reset,trigger,measure")
    pm.add_argument("--trials", type=int, default=3)
    pm.add_argument("--sanity-check", action="store_true")
    pm.add_argument("--verbose", action="store_true")
    pm.set_defaults(func=cmd_measure)

    # campaign
    from .runtime.campaign import build_argparser as _campaign_parser
    pc = sub.add_parser("campaign", help="timing-differential campaign",
                        parents=[_campaign_parser()])
    pc.set_defaults(func=cmd_campaign)

    # full
    pf = sub.add_parser("full", help="build + generate + validate")
    pf.add_argument("--sm", required=True)
    pf.add_argument("--harness", default="osiris_slots")
    pf.add_argument("--slots", type=int, default=4096)
    pf.add_argument("--source", default=None)
    pf.add_argument("--out-dir", default=".")
    pf.add_argument("--binary", default=None)
    pf.add_argument("--offsets", default=None)
    pf.add_argument("--verbose", action="store_true")
    pf.set_defaults(func=cmd_full)

    # patch
    pp = sub.add_parser("patch", help="patch SASS instructions in a CUDA binary")
    pp.add_argument("binary")
    pp.add_argument("--json", required=True, dest="json")
    pp.add_argument("--target", required=True)
    repl = pp.add_mutually_exclusive_group(required=True)
    repl.add_argument("--replace-hex")
    repl.add_argument("--instr-hex")
    repl.add_argument("--opcode-only")
    pp.add_argument("--stall", type=int, default=None)
    pp.add_argument("--yield", dest="yield_flag", action="store_true", default=None)
    pp.add_argument("--no-yield", dest="yield_flag", action="store_false")
    pp.add_argument("--wr-bar", type=int, default=None)
    pp.add_argument("--rd-bar", type=int, default=None)
    pp.add_argument("--wt-bar", type=int, default=None)
    pp.add_argument("--output", default=None)
    pp.add_argument("--in-place", action="store_true")
    pp.add_argument("--dry-run", action="store_true")
    pp.add_argument("--after-n", type=int, default=0)
    pp.add_argument("-v", "--verbose", action="store_true")
    pp.set_defaults(func=cmd_patch)

    # tui
    pt = sub.add_parser("tui", help="curses binary modifier (VisualModifier port)")
    pt.add_argument("binary", help="CUDA binary or cubin to open")
    pt.add_argument("--sm", default=None,
                    help="SM version (default: inferred from the binary)")
    pt.add_argument("--json", default=None,
                    help="explicit ISA JSON path (overrides --sm)")
    pt.add_argument("--theme", default="dark",
                    choices=["dark", "phosphor", "light"],
                    help="color scheme (cycle at runtime with 'c')")
    pt.set_defaults(func=cmd_tui)

    # plot
    ppl = sub.add_parser("plot", help="figures: violin (Fig 7), hotcold (Fig 8), latex tables")
    ppl.add_argument("plot_kind", choices=["violin", "hotcold", "latex"])
    ppl.add_argument("argv", nargs=argparse.REMAINDER,
                     help="arguments for the underlying plot script (use -- to separate)")
    ppl.set_defaults(func=cmd_plot)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "version", False):
        from . import __version__
        print(f"eoss {__version__}")
        return 0
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 0
    return func(args)


if __name__ == "__main__":
    sys.exit(main())
