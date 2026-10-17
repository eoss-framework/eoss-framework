"""Repository/data path resolution.

The framework needs per-architecture ISA JSON files. Resolution order:

1. ``$EOSS_ISA_DIR`` environment variable (directory containing SMxx subdirs
   or ISA JSON files directly).
2. ``<repo>/JSON`` discovered by walking up from this file.
3. ``<cwd>/JSON`` as a last resort.

The Rosetta String inputs (optional, only needed to regenerate ISA JSON)
are resolved analogously with ``$EOSS_SASS_DIR`` and ``<repo>/SASS``.
"""

import os
from pathlib import Path
from typing import List, Optional


def repo_root() -> Path:
    """Best-effort repository root: nearest ancestor containing JSON/ or SASS/."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "JSON").is_dir() or (parent / "SASS").is_dir():
            return parent
    return here.parent.parent


def isa_dir() -> Path:
    """Directory holding per-SM ISA specification subdirectories (SM70, ...)."""
    env = os.environ.get("EOSS_ISA_DIR")
    if env:
        return Path(env)
    candidate = repo_root() / "JSON"
    if candidate.is_dir():
        return candidate
    return Path("JSON")


def isa_json_path(sm: int | str) -> Path:
    """Path to the ISA JSON for an SM version, e.g. JSON/SM120/json_120_with_conditions.json."""
    sm = str(sm)
    return isa_dir() / f"SM{sm}" / f"json_{sm}_with_conditions.json"


def generated_dir(sm: int | str) -> Path:
    """Directory where generated artifacts for an SM are stored."""
    return isa_dir() / f"SM{sm}"


def sass_dir() -> Path:
    """Directory holding Rosetta String text files (optional input)."""
    env = os.environ.get("EOSS_SASS_DIR")
    if env:
        return Path(env)
    candidate = repo_root() / "SASS"
    if candidate.is_dir():
        return candidate
    return Path("SASS")


def sass_instructions_path(sm: int | str) -> Path:
    return sass_dir() / f"sm_{sm}_instructions.txt"


def sass_latencies_path(sm: int | str) -> Path:
    return sass_dir() / f"sm_{sm}_latencies.txt"


def available_isa_sms() -> List[str]:
    """SM versions with a pre-built ISA JSON present, sorted ascending."""
    out = []
    base = isa_dir()
    if not base.is_dir():
        return out
    for child in base.iterdir():
        if child.is_dir() and child.name.startswith("SM"):
            sm = child.name[2:]
            if (child / f"json_{sm}_with_conditions.json").is_file():
                out.append(sm)
    return sorted(out, key=int)


def package_data(*parts: str) -> Path:
    """Path to a file shipped inside the eoss package (harnesses, data)."""
    return Path(__file__).resolve().parent.joinpath(*parts)


def harness_source(name: str = "osiris_slots") -> Path:
    """Path to a packaged CUDA harness template (.cu)."""
    return package_data("harnesses", f"{name}.cu")


def encoding_data(name: str) -> Path:
    """Path to a packaged encoding data file (e.g. 'ttu_sm120.json')."""
    return package_data("data", "encodings", name)


def resolve_existing(path: Optional[str], fallback: Path) -> Path:
    """Return ``path`` if given and it exists, else ``fallback``."""
    if path:
        p = Path(path)
        if p.exists():
            return p
        raise FileNotFoundError(f"path does not exist: {p}")
    return fallback
