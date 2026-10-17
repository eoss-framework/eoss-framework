# EoSS cross-generation validation results

Runtime coverage of the SASS ISA — the fraction of instruction classes for
which the framework can generate/lift an encoding that **executes cleanly** on
real hardware (patched into the OSIRIS slot harness, no CUDA error, sane
timings). Measured by building the union instruction DB across the harness
flavors each generation supports (`super` = texture/surface/shared, `tma`,
`cga` = cluster), plus a byte-templating harvest.

_Measured 2026-08-28 (B200 `sm_100` added 2026-09-03) · framework @ `847ba6b`._

## Coverage by generation

| GPU | Arch | SM | Coverage | Method | CUDA |
|-----|------|----|---------:|--------|------|
| Tesla V100 | Volta | `sm_70` | **539 / 646 = 83.4%** | full union | 12.x |
| A30 | Ampere | `sm_80` | **1066 / 1215 = 87.7%** | full union | 12.x |
| A100 80GB | Ampere | `sm_80` | **1066 / 1215 = 87.7%** | full union (cross-check) | 12.8 |
| RTX A6000 | Ampere (consumer) | `sm_86` | **1123 / 1269 = 88.5%** | full union | 12.x |
| RTX 6000 Ada | Ada | `sm_89` | **1159 / 1337 = 86.7%** | full union | 12.8 |
| H100 80GB HBM3 | Hopper | `sm_90` | **1338 / 1584 = 84.5%** | full union | 12.8 |
| NVIDIA B200 | Blackwell (datacenter) | `sm_100` | **1140 / 1377 = 82.8%** | full union | 12.8 |
| RTX 5090 | Blackwell (consumer) | `sm_120` | **1221 / 1411 = 86.5%** | full union | 13.0 |

Notes:
- Eight GPUs across five architectures (Volta → Blackwell) all land within
  ~6 points despite different ISAs and addressing models — Volta has no
  uniform-register datapath at all; Blackwell is fully descriptor/UR-based.
- **Determinism cross-check:** the A100 and A30 (both `sm_80`, different
  datacenter parts) produce *identical* results — 1066/1215 and the same
  per-family breakdown — confirming coverage is a function of the SM/ISA, not
  the specific silicon.
- The **consumer** Ampere (A6000, `sm_86`) scores highest at 88.5% — smallest
  advanced-op footprint, and texture fully recovers (incl. `tld` 10/10).
- **Datacenter vs consumer, same architecture — a consistent split.** The
  consumer part scores higher on *both* architectures where we have the pair:
  Ampere A6000 `sm_86` 88.5% > A100 `sm_80` 87.7%, and Blackwell RTX 5090
  `sm_120` 86.5% > B200 `sm_100` 82.8%. The datacenter parts carry a larger
  advanced-op footprint in the denominator, and the B200 additionally zeroes
  out generic/local addressing (`ld`/`st` 0/6, `ldl`/`stl` 0/4,0/3) while
  *recovering more* TMA/cluster (`tma` +21, `cga` +14 vs the 5090's +16/+8) —
  it fields more of the hard ops but over a harder ISA. B200 `sm_100` (82.8%)
  is the lowest of the set, just under Volta.
- **Texture/surface port across all four via the GPR-handle pattern**: Volta
  and Ampere put the bindless handle in a GPR, Hopper in a hybrid GPR+UR,
  Blackwell in a pure UR — all extracted post-compile. Volta: tex 8/8,
  suld/sust 12/12.
- **Ampere scores highest** not because it's better-supported but because its
  ISA is smaller (1215 classes) and lacks the hard advanced ops (TMA, cluster,
  tensor variants) that drag down the newer generations' percentages — those
  ops are in the denominator for `sm_90`/`sm_120` but absent on `sm_80`.
- **Blackwell** also has a single **combinable campaign harness** (`super`
  alone, all mutually co-executable in one kernel): **1197 / 1411 = 84.8%**.
  The union figure above is coverage across *all* flavors (not all combinable
  in one launch — TMA/cluster need their own launch context).
- The Ampere full run (texture-fixed) recovers **tex 10/10, suld/sust 8/8**,
  replacing an earlier 82.8% stratified sample taken before the pre-Hopper
  texture fix. It took ~14 min with a warm driver (see "operational notes").

## Union stage breakdown

| stage | sm_120 | sm_100 | sm_90 |
|-------|-------:|-------:|------:|
| classic (global + ALU) | 1132 | 1063 | 1245 |
| `super` (tex/surf/shared, reg-map policy) | +63 | +40 | +68 |
| `tma` (descriptor policy) | +16 | +21 | +17 |
| `cga` (cluster barrier) | +8 | +14 | +8 |
| harvest (byte-templating) | +2 | +2 | +0 → +N* |
| **total** | **1221** | **1140** | **1338** |

The two Blackwell parts diverge in a telling way: the datacenter B200
(`sm_100`) recovers *more* of the advanced families (`tma` +21, `cga` +14) but
a smaller `classic` base (1063 vs the 5090's 1132) — the descriptor/cluster
datapath is exercised harder while a chunk of generic/local addressing
(`ld`/`st`/`ldl`/`stl`) doesn't validate at all on the B200.

\* Hopper TMA came back 0/6 from the field policy; the harness's own `UTMALDG`
validates, so byte-templating now lifts it (`847ba6b`) — TMA harvest was added
after the 1338 run.

## Per-family (Hopper sm_90, representative)

| family | recovered | family | recovered |
|--------|----------:|--------|----------:|
| `imad` | 100/100 | `suld`/`sust` | 8/8, 7/8 |
| `fadd` | 5/5 | `lds`/`sts` | 3/3, 3/3 |
| `tex` | 10/10 | `atoms` | 9/9 |
| `tld` | 8/10 | `ldg`/`stg` | 2/6, 2/6 |
| `ucgabar` | 2/4 | `utmaldg`/`utmastg` | 0/6 → harvest |

## Addressing models — what is generation-specific

The register-map extractor reads these forms straight from each harness's
SASS post-compile; the generator's operand policy targets whatever it finds.

| resource | Ampere `sm_80` | Hopper `sm_90` | Blackwell `sm_120` |
|----------|----------------|----------------|--------------------|
| global memory | direct `[R6.64]` | `desc[UR8][R6.64]` | `desc[UR8][R2.64]` |
| texture handle | GPR + `.SCR` | hybrid GPR + UR | UR (`UR4`) |
| surface handle | constant bank | UR | UR |
| TMA descriptor | — (absent) | UR | UR |
| cluster barrier | — (absent) | present | present |

- What ports for free: common ALU (universal), global memory (byte-templating
  reads whatever `LDG` form the compiler emits), shared, surface store.
- Generation-specific work is bounded to **extractor SASS patterns** +
  **operand-policy branches** — e.g. the pre-Hopper GPR texture handle
  (`_TEX_GPR`), which also caught Hopper's hybrid form.

## Bugs found only by running on real non-consumer hardware

1. **Datacenter GPUs have no display watchdog** (A30/H100): a hung patched
   kernel runs forever and `subprocess.run`'s SIGKILL can't reap it — it
   wedged the whole union build. Fixed with an OS-level `timeout -k` wrapper
   (`85ce43a`). The 5090's watchdog had masked this entirely.
2. **Pre-Hopper bindless texture handle is a GPR**, not a uniform register.
   The register-map policy targeted `UR4`, so every texture class faulted on
   Ampere (0/8). Added a GPR-handle extractor pattern + policy branch
   (`a4a2894`) → Ampere 0/8 → 8/8, and Hopper 10/10.
3. **TMA field policy is not reproducible across generations** (an
   ISA-JSON-unmodeled bit, seen on sm_90): the harness's own `UTMALDG` runs
   clean but the generator's encoding does not. Fixed by adding the TMA family
   to byte-templating harvest (`847ba6b`).

## Methodology

- **One parametrized harness template** (`harnesses/osiris_harness.cu`)
  assembled per generation via `-DFEAT_*` flags; the capability table in
  `eoss.isa.registry` (`gen_features`, `harness_flavors`) gates which flavors
  a generation can build (TMA/cluster are `sm_90+`).
- **Union DB build** (`eoss.runtime.union.build_union_db`): each ISA class is
  validated across the flavors in order; it enters the DB on the first harness
  where it runs clean. Two recovery paths: generator field policy (with the
  extracted register map) and byte-templating harvest (lift the harness's own
  working bytes for ops the policy can't encode).
- **Register map** (`eoss.runtime.regmap`) is extracted from each compiled
  harness's SASS, so per-generation register assignments are automatic.
- Reproduce with one command: `scripts/shakedown.py --remote user@host --key
  <key> --sm <nn>` rsyncs the repo (only the target spec) and runs the whole
  build → union → report on the box (auto keep-alive, auto remote-python
  detect). On the box directly: `PYTHONPATH=<parent> python3 scripts/shakedown.py`.
  `--limit N` gives a quick stratified estimate.

## Operational notes (headless / datacenter GPUs)

Validation spawns a fresh subprocess per instruction, so per-process GPU
startup dominates the wall-clock on boxes tuned for throughput rather than
latency:

- **Persistence mode.** With `nvidia-smi -pm 1` *disabled* (default on the
  A30), every subprocess pays a cold driver init — measured **~2.4 s per
  harness run**. Enabling persistence mode, or — without root — holding a
  single CUDA context open in a background process (`cudaFree(0); sleep(...)`)
  so the driver never unloads, drops this to **~0.2 s** (≈12×). The full A30
  union went from a ~60 min projection to ~14 min this way. Consumer parts
  (RTX 5090) and most cloud instances (the H100 here) keep the driver warm, so
  this only bites on bare headless datacenter cards.
- **No display watchdog.** Datacenter cards don't auto-kill a runaway kernel,
  so an invalid instruction *hangs* (until the OS `timeout -k` wrapper reaps
  it) instead of fast-faulting. Use a short `run_timeout` (2–4 s) — the clean
  harness runs in <1 s warm, so anything longer just wastes time on the hangs.

## Caveats

- Point-in-time, per-GPU measurements — coverage depends on the exact GPU,
  driver, and CUDA toolkit. Datacenter vs consumer parts of the same
  architecture *do* differ measurably (B200 `sm_100` 82.8% vs RTX 5090
  `sm_120` 86.5%; A100 `sm_80` 87.7% vs A6000 `sm_86` 88.5%) — see the
  datacenter/consumer note above.
- The ~15% not covered is dominated by: graphics-only ops (need a 3D context,
  handled on a separate branch), sequence/state-dependent ops (TTU sessions,
  LDGSTS depbar/arrive, named cluster barriers), and legacy/dead addressing
  forms superseded by a validating sibling encoding.
