# EoSS — A Systematic Framework to Explore NVIDIA's SASS ISA

EoSS is an end-to-end framework for systematic native-ISA exploration on
NVIDIA GPUs. It extracts machine-readable instruction specifications from the
Rosetta String embedded in NVIDIA's tooling, generates valid instruction
encodings, and executes them on real hardware through binary patching.

It spans eight architectures from Volta through Blackwell (SM70–SM121) with
443–1,239 validated instruction classes per generation.

This repository contains the framework itself. The case studies from the
ASPLOS paper (TTU ray-tracing instructions, contention effects, OSIRIS-style
timing fuzzing) are included as thin experiment shells under `TTU/`,
`Contention/`, `OSIRIS/`, plus the supplementary `Unintentional/` explorer and
the `VisualModifier/` binary editor.

## Installation

```bash
# Option A: editable install (adds the `eoss` command)
python3 -m venv .venv --system-site-packages
.venv/bin/pip install -e .
export PATH="$PWD/.venv/bin:$PATH"

# Option B: no install — run from the repository's parent directory
cd .. && python3 -m eoss.cli --help
```

(Note: this repository's root *is* the `eoss` package directory, so for
Option B the parent directory must be the working directory.)

The framework core is stdlib-only. `numpy`/`matplotlib` are needed only for
plotting (`pip install -e .[plot]`). Python 3.10+ required.

## The `eoss` CLI

https://github.com/user-attachments/assets/0a4b3525-18af-4fc7-97b7-509b1e945889

```
eoss gpu info                            GPU / SM detection
eoss spec build --sm 120                 Rosetta String -> ISA JSON
eoss isa stats --sm 120                  ISA JSON statistics
eoss isa list --sm 120 --filter fadd     list instruction classes
eoss sr list --sm 120                    special-register table
eoss generate --sm 120                   bulk candidate encodings (one per class)
eoss encode --sm 120 --class fadd__RRR_RR --set Rd=R5 --stall 4
eoss build --sm 120                      compile harness + extract offsets
eoss validate --sm 120 --binary H --offsets O   runtime validation on the GPU
eoss measure --binary H --offsets O --instructions I --triple a,b,c
eoss campaign ...                        timing-differential campaigns
eoss patch BINARY --json ISA --target FADD --opcode-only FFMA
eoss tui BINARY                          curses SASS editor (VisualModifier port)
eoss plot violin campaign.csv -o fig7.pdf     Figure 7
eoss plot hotcold campaign.csv -o fig8.pdf    Figure 8
eoss full --sm 120                       build + generate + validate
```

## Programmatic API

```python
from eoss import ISASpec
from eoss.enc.generator import Encoder

spec = ISASpec.load(sm=120)
enc = Encoder(spec)

# Any instruction, any modifier, with constraint diagnostics:
ins = enc.build("fadd__RRR_RR", Rd="R5", Ra="R2", Rb="R3", stall=4)
ins.bytes              # 16-byte SASS encoding
ins.valid              # constraint check
ins.failed_conditions  # precise violated constraints (e.g. "OOR_REG_ERROR")

# Binary patching:
from eoss.binary import BinaryPatcher
p = BinaryPatcher("my_kernel")
p.patch_by_opcode(spec.opcodes_for_mnemonic("FADD"), replacement, mode="opcode")
p.save("my_kernel_patched")

# Runtime measurement:
from eoss.runtime import build_harness, extract_offsets, SlotHarness
binary = build_harness(120)
offsets = extract_offsets(binary)
h = SlotHarness(binary, offsets, "JSON/SM120/validated_gpr.json")
h.measure_triple("issue", "fadd__RRR_RR", "mufu__RRR_RR", "fadd__RRR_RR")
```

## Package layout

```
. (repo root = the eoss package)
├── spec/          # offline: Rosetta String -> ISA JSON
│   ├── conditions.py   # constraint-expression AST parser
│   ├── rosetta.py      # ISA JSON builder (regenerates shipped JSON byte-identically)
│   └── data/*.json     # scheduling enums, shader types, special registers
├── isa/           # ISA spec model + architecture registry
├── enc/           # encoding layer
│   ├── bits.py         # bit-range read/write (canonical)
│   ├── opcode.py       # 13-bit opcode extract/insert (canonical)
│   ├── control.py      # bytes 13-15 control fields + patch policies (canonical)
│   └── generator.py    # constraint-aware generator + programmable Encoder
├── binary/        # cuobjdump wrapper, pure-Python CUDA ELF parser,
│                  # anchor-based code-base search, unified BinaryPatcher
├── runtime/       # harness build/run, offset extraction (osiris_slots +
│                  # contention_chain grammars), validation, measurement,
│                  # campaign engine (batching, shuffling, crash recovery)
├── tui/           # curses SASS editor (`eoss tui`) — TUI port of VisualModifier
├── plots/         # Figures 7/8 + LaTeX tables
├── harnesses/     # packaged CUDA harness templates
├── data/          # packaged encodings (e.g. ttu_sm120.json)
├── tests/         # pytest suite (unit + golden + GPU-gated)
│
├── OSIRIS/        # Case Study 3: OSIRIS-style fuzzing — `make full/campaign`
├── TTU/           # Case Study 1: TTU hidden instruction demo — `make demo`
├── Contention/    # Case Study 2: contention effects — `make demo`
├── VisualModifier/# Browser SASS editor (modifier.html) + patch CLI wrapper
├── Unintentional/ # Supplementary: undefined-opcode exploration
├── JSON/SM*/      # pre-built per-architecture ISA specification JSON
└── SASS/          # optional Rosetta String inputs (gitignored, not distributed)
```

## Case studies

```bash
cd OSIRIS       && make full SM=120 && make campaign SM=120   # Sections 3-5, 8
cd TTU          && make demo        # Section 6 (SM120 only)
cd Contention   && make demo        # Section 7 (Figure 6)
cd Unintentional && make build SM=120 && make campaign-quick MAX_TESTS=100
```

The per-directory Makefiles are thin shims over the `eoss` CLI (they
auto-detect the repo `.venv`, else expect `eoss` on PATH).

## Prerequisites

- NVIDIA GPU (SM70+)
- CUDA Toolkit 13.0+ (nvcc, cuobjdump, compute-sanitizer)
- Nsight Compute (`ncu`) with metric read permissions (for TTU `demo-rays`)
- Python 3.10+
- (Optional) The Rosetta String, extracted using
  [DocumentSASS](https://github.com/0xD0GF00D/DocumentSASS). Only needed to
  regenerate the ISA JSON; pre-built ISA JSON is provided in `JSON/`. The
  shipped ISA JSONs regenerate **byte-identically** via `eoss spec build`
  (covered by the test suite).

Each ISA JSON's `SPECIAL_REGISTERS` maps all 256 numeric SR encodings to
display names parsed from that architecture's Rosetta String. Unnamed
encodings retain their `SR<number>` spelling; this does not imply hardware
support. `SPECIAL_REGISTER_ALIASES` maps alternate names to indices, including
dotted coordinate names and names that share an encoding. Existing underscore
spellings and the framework's `SR_TTU_SESSION_ACTIVE` name are preserved.
`eoss sr list --sm <SM>` shows the table and aliases; the TUI accepts either
spelling when editing an SR operand.

## Tests

```bash
python3 -m pytest tests/          # unit + golden tests (GPU-free)
python3 -m pytest tests/ -m gpu   # + GPU integration tests (needs a GPU)
```

The suite includes golden tests: ISA-JSON regeneration is byte-identical to
the shipped specs, and the generator reproduces the shipped SM120 encodings
byte-identically.

## Supported Architectures

| Name      | SM    | GPU Examples          |
|-----------|-------|-----------------------|
| Volta     | SM70  | V100                  |
| Ampere    | SM80  | A100                  |
| Ampere    | SM86  | RTX 3090, RTX A6000   |
| Ada       | SM89  | RTX 4090, RTX L40S    |
| Hopper    | SM90  | H100                  |
| Blackwell | SM100 | B200                  |
| Blackwell | SM120 | RTX 5090              |
| Blackwell | SM121 | DGX Spark (GB10)      |
