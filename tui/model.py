"""Editor state for the TUI: the open binary, pending edits, undo, save.

Wraps the framework primitives (:mod:`eoss.binary.elf`, :mod:`eoss.enc.bits`,
:mod:`eoss.enc.opcode`, :mod:`eoss.enc.control`) with byte-level undo and
modified-byte tracking. No curses in this module.
"""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..binary import cuobjdump
from ..binary.elf import find_cuda_text_sections
from ..enc.bits import parse_int_literal, write_bit_segments
from ..enc.control import (INSTRUCTION_SIZE, NO_BARRIER, read_control,
                           write_control)
from ..enc.generator import Encoder
from ..enc.opcode import MAX_OPCODE, write_opcode
from ..isa.model import ISASpec
from . import elfview
from .decode import DecodedInstruction, Decoder


class EditError(ValueError):
    """A user edit that cannot be applied (bad value, out of range...)."""


@dataclass
class TextSection:
    abs_offset: int
    size: int
    name: str

    @property
    def n_instructions(self) -> int:
        return self.size // INSTRUCTION_SIZE


class EditorModel:
    """All mutable state of one editing session."""

    def __init__(self, path: str, spec: ISASpec):
        self.path = Path(path)
        self.spec = spec
        self.decoder = Decoder(spec)
        self.original = self.path.read_bytes()
        self.data = bytearray(self.original)

        self.sections: List[TextSection] = [
            TextSection(off, size, name)
            for off, size, name in find_cuda_text_sections(self.original)]
        self.images: List[elfview.ElfImage] = elfview.scan_images(self.original)

        #: byte offsets that currently differ from the original file
        self.modified: set = set()
        #: undo stack of (offset, old bytes, new bytes, label)
        self.undo_stack: List[Tuple[int, bytes, bytes, str]] = []
        #: per-offset instruction variant override (class name)
        self.variant_overrides: Dict[int, str] = {}
        self._decode_cache: Dict[Tuple[int, str], DecodedInstruction] = {}
        self._encoder: Optional[Encoder] = None

    # -- images / sections ---------------------------------------------------

    @property
    def cuda_images(self) -> List[elfview.ElfImage]:
        return [im for im in self.images if im.is_cuda] or self.images

    def image_for_section(self, section: TextSection) -> Optional[elfview.ElfImage]:
        best = None
        for image in self.images:
            if image.base <= section.abs_offset and (
                    best is None or image.base > best.base):
                best = image
        return best

    def nvinfo_records(self) -> List[elfview.NvInfoRecord]:
        out = []
        for image in self.cuda_images:
            out.extend(elfview.parse_nvinfo_records(bytes(self.data), image))
        return out

    def notes(self) -> List[elfview.NoteEntry]:
        out = []
        for image in self.cuda_images:
            out.extend(elfview.parse_notes(bytes(self.data), image))
        return out

    # -- decoding ------------------------------------------------------------

    def instruction_bytes(self, offset: int) -> bytes:
        return bytes(self.data[offset:offset + INSTRUCTION_SIZE])

    def decode_at(self, offset: int) -> DecodedInstruction:
        override = self.variant_overrides.get(offset, "")
        key = (offset, override)
        cached = self._decode_cache.get(key)
        raw = self.instruction_bytes(offset)
        if cached is not None and cached.raw == raw:
            return cached
        decoded = self.decoder.decode(offset, raw, override or None)
        self._decode_cache[key] = decoded
        return decoded

    def cycle_variant(self, offset: int) -> Optional[str]:
        """Select the next candidate class for a multi-variant opcode."""
        decoded = self.decode_at(offset)
        if len(decoded.candidates) < 2 or decoded.class_name is None:
            return None
        names = decoded.candidates
        idx = names.index(decoded.class_name) if decoded.class_name in names else 0
        self.variant_overrides[offset] = names[(idx + 1) % len(names)]
        return self.variant_overrides[offset]

    # -- writes --------------------------------------------------------------

    @property
    def n_modified(self) -> int:
        return len(self.modified)

    def _write(self, offset: int, new: bytes, label: str) -> None:
        old = bytes(self.data[offset:offset + len(new)])
        if old == new:
            return
        self.data[offset:offset + len(new)] = new
        for i in range(len(new)):
            pos = offset + i
            if self.data[pos] != self.original[pos]:
                self.modified.add(pos)
            else:
                self.modified.discard(pos)
        self.undo_stack.append((offset, old, new, label))

    def undo(self) -> Optional[str]:
        """Undo the most recent edit; returns its label (None if empty)."""
        if not self.undo_stack:
            return None
        offset, old, _new, label = self.undo_stack.pop()
        self.data[offset:offset + len(old)] = old
        for i in range(len(old)):
            pos = offset + i
            if self.data[pos] != self.original[pos]:
                self.modified.add(pos)
            else:
                self.modified.discard(pos)
        return label

    # -- instruction edits ---------------------------------------------------

    def parse_field_value(self, offset: int, field_name: str, text: str) -> int:
        """Parse user input for a field: enum member name, Bucket@Name or int."""
        decoded = self.decode_at(offset)
        view = next((f for f in decoded.fields if f.name == field_name), None)
        if view is None:
            raise EditError(f"unknown field {field_name!r}")
        text = text.strip()
        if not text:
            raise EditError("empty value")

        if field_name == "opcode":
            return self.parse_opcode_value(text)

        candidates = []
        if view.bucket:
            candidates.append(view.bucket)
        if "@" in text:
            base, _, member = text.partition("@")
            bucket = self.decoder.bucket(base)
            if bucket:
                for k in (member, member.upper(), member.lower()):
                    if k in bucket:
                        return int(bucket[k])
        for bucket_name in candidates:
            bucket = self.decoder.bucket(bucket_name) or {}
            for k in (text, text.upper(), text.lower()):
                if k in bucket:
                    return int(bucket[k])
        rev_sr = {name.upper(): idx
                  for idx, name in self.spec.special_registers.items()}
        rev_sr.update({name.upper(): idx
                       for name, idx in self.spec.special_register_aliases.items()})
        if text.upper() in rev_sr:
            return rev_sr[text.upper()]
        try:
            value = parse_int_literal(text)
        except ValueError:
            raise EditError(f"not a {view.bucket or 'numeric'} value: {text!r}")
        return value

    def set_field(self, offset: int, field_name: str, text: str) -> int:
        """Set one argument field from user text. Returns the new value."""
        decoded = self.decode_at(offset)
        view = next((f for f in decoded.fields if f.name == field_name), None)
        if view is None:
            raise EditError(f"unknown field {field_name!r}")
        value = self.parse_field_value(offset, field_name, text)
        if field_name == "opcode":
            self.set_opcode(offset, value)
            return value
        if value < 0 or (view.width and value >= (1 << view.width)):
            raise EditError(
                f"value 0x{value:X} does not fit in {view.width} bits")
        instr = bytearray(self.instruction_bytes(offset))
        write_bit_segments(instr, view.bits, value)
        self._write(offset, bytes(instr), f"set {field_name}={text.strip()}")
        return value

    def parse_opcode_value(self, text: str) -> int:
        """Opcode from mnemonic ('FFMA'), class name or numeric literal."""
        text = text.strip()
        try:
            value = parse_int_literal(text)
            if not 0 <= value <= MAX_OPCODE:
                raise EditError(f"opcode out of range: {value:#x}")
            return value
        except ValueError:
            pass
        defn = self.spec.get(text)
        if defn is not None:
            return defn.opcode
        opcodes = self.spec.opcodes_for_mnemonic(text)
        if opcodes:
            return min(opcodes)
        raise EditError(f"unknown mnemonic/class: {text!r}")

    def set_opcode(self, offset: int, opcode: int) -> None:
        instr = write_opcode(self.instruction_bytes(offset), opcode)
        self.variant_overrides.pop(offset, None)
        self._write(offset, instr, f"opcode=0x{opcode:04X}")

    def set_control(self, offset: int, name: str, value: int) -> None:
        """Set one control field: stall, yield_, wr_bar, rd_bar, wt_bar."""
        limits = {"stall": 15, "yield_": 1, "wr_bar": 7, "rd_bar": 7, "wt_bar": 63}
        if name not in limits:
            raise EditError(f"unknown control field {name!r}")
        if not 0 <= value <= limits[name]:
            raise EditError(f"{name} out of range 0..{limits[name]}")
        instr = write_control(self.instruction_bytes(offset), **{name: value})
        self._write(offset, instr, f"{name}={value}")

    def set_bytes(self, offset: int, new: bytes, label: str = "") -> None:
        if offset < 0 or offset + len(new) > len(self.data):
            raise EditError("write out of range")
        self._write(offset, new, label or f"bytes @0x{offset:X}")

    @property
    def encoder(self) -> Encoder:
        if self._encoder is None:
            self._encoder = Encoder(self.spec)
        return self._encoder

    def _generic_bytes(self, class_name: str) -> bytes:
        """Generator-built generic encoding of one class: RZ/PT operands,
        zero immediates, no barriers."""
        try:
            return self.encoder.build(class_name).bytes
        except Exception as exc:
            raise EditError(f"generator failed for {class_name}: {exc}")

    def generate_generic(self, offset: int) -> str:
        """Replace the instruction at ``offset`` with a generic encoding of
        its current class. Returns the class name (one undo entry)."""
        decoded = self.decode_at(offset)
        defn = decoded.defn
        if defn is None:
            raise EditError(
                f"unknown opcode 0x{decoded.opcode:04X} — pick one first")
        # keep the opcode value currently in the bits (aliases of one class
        # can differ from the class's primary OPCODE)
        instr = write_opcode(self._generic_bytes(defn.name), decoded.opcode)
        self._write(offset, instr, f"generate {defn.name}")
        self.variant_overrides[offset] = defn.name
        return defn.name

    def write_generic(self, offset: int, class_name: str) -> str:
        """Stamp a generic encoding of any instruction class at ``offset``
        (the ISA-browser action). Returns the class name (one undo entry)."""
        if self.spec.get(class_name) is None:
            raise EditError(f"unknown instruction class {class_name!r}")
        if offset < 0 or offset + INSTRUCTION_SIZE > len(self.data):
            raise EditError("write out of range")
        self._write(offset, self._generic_bytes(class_name),
                    f"generate {class_name}")
        self.variant_overrides[offset] = class_name
        return class_name

    # -- external disassembler ----------------------------------------------

    def disassemble(self) -> List[Tuple[str, List[cuobjdump.SassInstruction]]]:
        """``cuobjdump -sass`` of the current (edited) buffer, grouped as
        (function name, instructions). Raises RuntimeError when cuobjdump
        is unavailable or rejects the binary."""
        with tempfile.NamedTemporaryFile(
                suffix=self.path.suffix or ".cubin", delete=False) as f:
            f.write(bytes(self.data))
            tmp = f.name
        try:
            sass = cuobjdump.get_sass_dump(tmp, timeout=60)
        finally:
            os.unlink(tmp)
        return [(name, cuobjdump.parse_sass_instructions(sass, name))
                for name in cuobjdump.list_functions(sass)]

    def section_for_function(self, func: str) -> Optional[TextSection]:
        """The .text section holding ``func`` (cuobjdump SASS offsets are
        relative to its start)."""
        for sec in self.sections:
            if sec.name == f".text.{func}" or sec.name.endswith("." + func):
                return sec
        return None

    def swap_instructions(self, off_a: int, off_b: int) -> None:
        """Swap two adjacent 16-byte instructions (single undo entry)."""
        lo, hi = sorted((off_a, off_b))
        if hi - lo != INSTRUCTION_SIZE:
            raise EditError("can only swap adjacent instructions")
        a = self.instruction_bytes(lo)
        b = self.instruction_bytes(hi)
        self._write(lo, b + a, f"swap 0x{lo:X}<->0x{hi:X}")
        ov_lo = self.variant_overrides.pop(lo, None)
        ov_hi = self.variant_overrides.pop(hi, None)
        if ov_lo is not None:
            self.variant_overrides[hi] = ov_lo
        if ov_hi is not None:
            self.variant_overrides[lo] = ov_hi

    # -- barrier tracking ----------------------------------------------------

    def barrier_events(self, section: TextSection) -> List[Tuple[int, int, int]]:
        """(wr_bar, rd_bar, wt_mask) per instruction of ``section``.

        7 means "no barrier" for wr/rd; wt_mask bit N waits on barrier N.
        """
        events = []
        for i in range(section.n_instructions):
            c = read_control(
                self.instruction_bytes(section.abs_offset
                                       + i * INSTRUCTION_SIZE))
            events.append((c["wr_bar"], c["rd_bar"], c["wt_bar"]))
        return events

    def barrier_spans(self, section: TextSection
                      ) -> Dict[int, List[Tuple[int, Optional[int], str]]]:
        """Producer->consumer spans per barrier index (0-5).

        Returns {barrier: [(set_index, wait_index | None, kind)]} where kind
        is "wr" (dst_wr_sb) or "rd" (src_rel_sb) and wait_index is the first
        later instruction whose wait mask includes the barrier (None if the
        barrier is set but never awaited in this section).
        """
        spans: Dict[int, List[Tuple[int, Optional[int], str]]] = {
            b: [] for b in range(6)}
        active: Dict[int, Tuple[int, str]] = {}
        for i, (wr, rd, wt) in enumerate(self.barrier_events(section)):
            for b in range(6):
                if wt & (1 << b) and b in active:
                    start, kind = active.pop(b)
                    spans[b].append((start, i, kind))
            if wr != NO_BARRIER and 0 <= wr < 6:
                active[wr] = (i, "wr")
            if rd != NO_BARRIER and 0 <= rd < 6:
                # rd_bar wins if both name the same barrier (unusual)
                active.setdefault(rd, (i, "rd"))
        for b, (start, kind) in active.items():
            spans[b].append((start, None, kind))
        return spans

    # -- search --------------------------------------------------------------

    def search_instruction(self, section: TextSection, query: str,
                           start_index: int = 0) -> Optional[int]:
        """Next instruction index in ``section`` whose text matches ``query``."""
        q = query.strip().lower()
        if not q:
            return None
        n = section.n_instructions
        for step in range(n):
            idx = (start_index + step) % n
            offset = section.abs_offset + idx * INSTRUCTION_SIZE
            decoded = self.decode_at(offset)
            hay = f"{decoded.text} {decoded.class_name or ''}".lower()
            if q in hay:
                return idx
        return None

    # -- save ----------------------------------------------------------------

    def default_save_path(self) -> Path:
        return self.path.with_name(self.path.name + ".patched")

    def save(self, path: Optional[str] = None) -> Path:
        out = Path(path) if path else self.default_save_path()
        out.write_bytes(bytes(self.data))
        try:
            out.chmod(out.stat().st_mode | 0o111)  # keep it executable
        except OSError:
            pass
        return out

    # -- misc ----------------------------------------------------------------

    def control_of(self, offset: int) -> Dict[str, int]:
        return read_control(self.instruction_bytes(offset))

    def guess_sm(self) -> Optional[str]:
        """SM version inferred from the first CUDA image's e_flags."""
        for image in self.images:
            if image.is_cuda:
                name = elfview.CUDA_SM_NAMES.get(image.flags & 0xFF)
                if name:
                    digits = "".join(ch for ch in name.split("/")[0]
                                     if ch.isdigit())
                    return digits or None
        return None


def infer_sm_from_file(path: str) -> Optional[str]:
    """SM version from a binary's first CUDA ELF image (e.g. '90')."""
    data = Path(path).read_bytes()
    for image in elfview.scan_images(data):
        if image.is_cuda:
            name = elfview.CUDA_SM_NAMES.get(image.flags & 0xFF)
            if name:
                return "".join(ch for ch in name.split("/")[0] if ch.isdigit())
    return None
