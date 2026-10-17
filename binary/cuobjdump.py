"""Canonical cuobjdump/nvdisasm subprocess wrappers and SASS dump parsing.

Resolution order for the cuobjdump binary: ``$CUOBJDUMP`` environment
variable, then ``shutil.which``, then ``/usr/local/cuda/bin/cuobjdump``.
"""

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

DEFAULT_CUOBJDUMP = "/usr/local/cuda/bin/cuobjdump"


def cuobjdump_path() -> str:
    """Resolve the cuobjdump executable path."""
    env = os.environ.get("CUOBJDUMP")
    if env:
        return env
    found = shutil.which("cuobjdump")
    if found:
        return found
    if os.path.isfile(DEFAULT_CUOBJDUMP):
        return DEFAULT_CUOBJDUMP
    raise RuntimeError(
        "cuobjdump not found (set $CUOBJDUMP or add CUDA bin/ to PATH)")


def get_sass_dump(binary_path: str, timeout: int = 120) -> str:
    """Run ``cuobjdump -sass`` and return stdout. Raises RuntimeError on failure."""
    result = subprocess.run(
        [cuobjdump_path(), "-sass", str(binary_path)],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"cuobjdump -sass failed for {binary_path}: {result.stderr.strip()[:300]}")
    return result.stdout


# ---------------------------------------------------------------------------
# SASS dump parsing
# ---------------------------------------------------------------------------

FUNC_RE = re.compile(r"Function\s*:\s*(\S+)")
OFFSET_RE = re.compile(r"/\*([0-9a-fA-F]+)\*/")
INSTR_RE = re.compile(
    r"/\*([0-9a-fA-F]+)\*/\s+(.*?)\s*/\*\s*0x([0-9a-fA-F]+)\s*\*/")


@dataclass
class SassInstruction:
    """One instruction line from a cuobjdump SASS dump."""

    offset: int          # SASS offset (bytes) within the function's section
    text: str            # disassembly text, e.g. "FADD R2, R5, R2"
    encoding: int        # low 64 bits of the encoding as printed (big-endian hex)
    raw_line: str

    @property
    def encoding_bytes_le(self) -> bytes:
        """The printed encoding converted to the file's little-endian layout."""
        be_hex = f"{self.encoding:x}".zfill(16)
        le_hex = "".join(reversed([be_hex[i:i + 2] for i in range(0, 16, 2)]))
        return bytes.fromhex(le_hex)

    @property
    def mnemonic(self) -> str:
        return self.text.split()[0] if self.text else ""

    def has_mnemonic(self, name: str) -> bool:
        """Case-sensitive substring match used for marker discovery
        (e.g. 'FADD' also matches 'FADD.FTZ')."""
        return name in self.text


def parse_sass_instructions(sass: str, kernel_name: Optional[str] = None) -> List[SassInstruction]:
    """Parse instruction lines from a SASS dump.

    If ``kernel_name`` is given, only instructions belonging to that
    ``Function :`` block are returned.
    """
    out: List[SassInstruction] = []
    in_target = kernel_name is None
    for line in sass.split("\n"):
        fm = FUNC_RE.search(line)
        if fm:
            in_target = (kernel_name is None) or (fm.group(1) == kernel_name)
            continue
        if not in_target:
            continue
        m = INSTR_RE.search(line)
        if not m:
            continue
        offset = int(m.group(1), 16)
        encoding = int(m.group(3), 16)
        # Disassembly text sits between the offset comment and encoding comment
        text = m.group(2).strip().rstrip(";").strip()
        out.append(SassInstruction(offset=offset, text=text,
                                   encoding=encoding, raw_line=line))
    return out


def list_functions(sass: str) -> List[str]:
    """All function names in a SASS dump."""
    return [m.group(1) for m in FUNC_RE.finditer(sass)]


def find_kernel(sass: str, prefer: Optional[str] = None) -> Optional[str]:
    """Find a kernel function name in a SASS dump.

    Prefers names containing ``prefer`` (substring, case-insensitive), then
    any function containing "kernel" that is not a CUDA internal.
    """
    names = list_functions(sass)
    if prefer:
        prefer_l = prefer.lower()
        for name in names:
            if prefer_l in name.lower():
                return name
    for name in names:
        if "kernel" in name.lower() and not name.startswith("_"):
            return name
    return names[0] if names else None


@lru_cache(maxsize=32)
def disassemble(binary_path: str, timeout: int = 120) -> str:
    """Cached ``get_sass_dump`` for repeated queries on the same binary."""
    return get_sass_dump(binary_path, timeout=timeout)
