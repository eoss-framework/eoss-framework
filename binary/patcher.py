"""Unified binary patcher for CUDA binaries.

One implementation replacing:

* the opcode-scan patcher from VisualModifier/patch_sass.py
  (``patch_by_opcode`` with full / lower-half / opcode-only modes and
  control-field overrides),
* the generic patch-at-offset primitives from OSIRIS/patch_and_measure.py
  (``patch_at`` with an explicit :class:`PatchPolicy`, ``nop_at``).

Experiment-specific slot logic (RESET/TRIGGER/MEASURE triples) lives in
``eoss.runtime.measure``; this module is pure binary mechanics.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..enc import control as ctrl
from ..enc.opcode import extract_opcode, write_opcode
from . import elf


class BinaryPatcher:
    """Load a CUDA binary, patch instructions, write it back."""

    def __init__(self, binary_path):
        self.binary_path = Path(binary_path).resolve()
        with open(self.binary_path, "rb") as f:
            self.data = bytearray(f.read())
        self._sections: Optional[List[elf.TextSection]] = None

    # -- introspection -----------------------------------------------------

    @property
    def sections(self) -> List[elf.TextSection]:
        """All CUDA .text sections (cached)."""
        if self._sections is None:
            self._sections = elf.find_cuda_text_sections(self.data)
        return self._sections

    def instruction_at(self, file_off: int) -> bytes:
        return bytes(self.data[file_off:file_off + ctrl.INSTRUCTION_SIZE])

    # -- point patching ----------------------------------------------------

    def patch_at(self, file_off: int, new: bytes,
                 policy: ctrl.PatchPolicy = ctrl.PatchPolicy.MERGE_MEASURE) -> None:
        """Patch the 16-byte instruction at ``file_off`` per ``policy``."""
        ctrl.patch_at(self.data, file_off, new, policy)

    def nop_at(self, file_off: int) -> None:
        """Replace the instruction at ``file_off`` with a NOP."""
        self.data[file_off:file_off + ctrl.INSTRUCTION_SIZE] = ctrl.NOP_BYTES

    def replace_opcode_at(self, file_off: int, opcode: int) -> None:
        """Surgical 13-bit opcode swap, all other bits preserved."""
        patched = write_opcode(self.instruction_at(file_off), opcode)
        self.data[file_off:file_off + ctrl.INSTRUCTION_SIZE] = patched

    # -- opcode-scan patching (from patch_sass.py) --------------------------

    def patch_by_opcode(
        self,
        target_opcodes: Set[int],
        replacement,
        mode: str = "full",
        ctrl_overrides: Optional[Dict] = None,
        after_n: int = 0,
        verbose: bool = False,
    ) -> int:
        """Patch every instruction whose opcode is in ``target_opcodes``.

        Args:
            target_opcodes: set of 13-bit opcode ints to match.
            replacement: 16 bytes (mode="full"), 8 bytes (mode="instr")
                or an opcode int (mode="opcode").
            mode: "full" | "instr" | "opcode".
            ctrl_overrides: optional dict for :func:`control.write_control`
                (stall, yield_, wr_bar, rd_bar, wt_bar) — "instr"/"opcode" only.
            after_n: skip the first N matches.
            verbose: print per-instruction details.

        Returns:
            Number of patched instructions.
        """
        ctrl_overrides = ctrl_overrides or {}
        total = 0
        match_index = 0

        for sec_off, sec_size, sec_name in self.sections:
            if verbose:
                print(f"Section: {sec_name} at 0x{sec_off:x}, size {sec_size} "
                      f"({sec_size // 16} instructions)")
            for i in range(0, sec_size, 16):
                abs_pos = sec_off + i
                if abs_pos + 16 > len(self.data):
                    break
                instr = bytes(self.data[abs_pos:abs_pos + 16])
                if extract_opcode(instr) not in target_opcodes:
                    continue

                match_index += 1
                if match_index <= after_n:
                    continue

                if mode == "full":
                    new_instr = bytes(replacement[:16])
                elif mode == "opcode":
                    new_instr = write_opcode(instr, int(replacement))
                    if ctrl_overrides:
                        new_instr = ctrl.write_control(new_instr, **ctrl_overrides)
                elif mode == "instr":
                    new_instr = ctrl.replace_lower(bytes(replacement[:8]), instr)
                    if ctrl_overrides:
                        new_instr = ctrl.write_control(new_instr, **ctrl_overrides)
                else:
                    raise ValueError(f"unknown patch mode: {mode}")

                self.data[abs_pos:abs_pos + 16] = new_instr
                total += 1
                if verbose:
                    c = ctrl.read_control(new_instr)
                    print(f"  MATCH #{match_index} 0x{abs_pos:08x} -> "
                          f"opcode={extract_opcode(new_instr):#06x} "
                          f"stall={c['stall']} wr={c['wr_bar']} "
                          f"rd={c['rd_bar']} wt={c['wt_bar']}")

        return total

    # -- output --------------------------------------------------------------

    def save(self, output_path=None, make_executable: bool = True) -> Path:
        """Write the patched binary. Returns the output path."""
        out = Path(output_path) if output_path else self.binary_path.with_suffix(
            self.binary_path.suffix + ".patched")
        with open(out, "wb") as f:
            f.write(self.data)
        if make_executable:
            os.chmod(out, 0o755)
        return out

    def copy_to(self, output_path, make_executable: bool = True) -> Path:
        """Copy the *original* binary to output_path (start of a patch pipeline)."""
        out = Path(output_path)
        shutil.copy2(self.binary_path, out)
        if make_executable:
            os.chmod(out, 0o755)
        return out

    @classmethod
    def from_bytes(cls, data: bytes, source: str = "<bytes>") -> "BinaryPatcher":
        """Build a patcher from raw bytes (no backing file)."""
        obj = cls.__new__(cls)
        obj.binary_path = Path(source)
        obj.data = bytearray(data)
        obj._sections = None
        return obj


def load_binary_bytes(path) -> bytes:
    with open(path, "rb") as f:
        return f.read()
