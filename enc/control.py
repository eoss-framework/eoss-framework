"""SASS control-field (scheduling) layout and patching policies.

128-bit SASS instruction control layout (bytes 13-15), as used across the
framework::

    Byte  13:    yield(b0) | delay/stall(b1-b4) | opex_lo(b5) | dst_wr_sb_lo(b6-b7)
    Byte  14:    dst_wr_sb_hi(b0) | src_rel_sb(b1-b3) | req_bit_set_lo(b4-b7)
    Byte  15:    req_bit_set_hi(b0-b1) | opex_hi(b2-b4) | reserved(b5-b7)

Equivalently, viewed relative to the upper 64-bit word (bytes 8-15)::

    stall:      bits 41-44  (4 bits, == delay)      wr_bar (dst_wr_sb): bits 46-48
    yield:      bit  45     (== opex_lo)            rd_bar (src_rel_sb): bits 49-51
    wt_bar (req_bit_set): bits 52-57 (6 bits)       batch_t/opex_hi: bits 58-60

The ``stall``/``delay`` field plus the ``yield``/``opex_lo`` bit together form
the Rosetta String's 5-bit ``usched_info`` (opex low bits).

This module is the single source of truth for this layout. It replaces three
previously divergent patching policies:

* measurement normalization (delay forced to 4)      -> :func:`merge_measure`
* runtime-validation merging (stall forced to 1)     -> :func:`merge_validate`
* full / lower-half replacement (binary editor)      -> :func:`replace_full` / :func:`replace_lower`

Note: ``modifier.html`` (VisualModifier) mirrors the upper-word view in
JavaScript; this file is the canonical spec.
"""

from enum import Enum

INSTRUCTION_SIZE = 16

# NOP encoding: opcode 0x918, PT predicate, 4-cycle stall.
# A 128-bit SM70+ NOP that the GPU executes as a no-op on all architectures.
NOP_BYTES = bytes.fromhex("18790000000000000000000000080000")

# Delay (stall) field: byte 13, bits 1-4.
DELAY_MASK = 0x1E
#: Normalized delay used for timing measurements (matches NOP delay;
#: must be >= 2 for pipeline flush).
DELAY_NORMALIZED = 4

# opex_lo bit within byte 13 (usched_info bit 4 / classic "yield").
OPEX_LO_MASK = 0x20

# Barrier sentinel: 7 means "no scoreboard barrier" for 3-bit fields.
NO_BARRIER = 7


class PatchPolicy(Enum):
    """How the control bytes (13-15) are handled when patching."""

    #: Keep the harness instruction's control bytes entirely; only the
    #: delay field is normalized to DELAY_NORMALIZED (timing measurements).
    MERGE_MEASURE = "merge_measure"

    #: Like MERGE_MEASURE but the delay bits are taken from the patched-in
    #: instruction instead of being normalized.
    MERGE_DELAY = "merge_delay"

    #: Runtime-validation merge: preserve harness yield/dst_wr_sb bits
    #: (mask 0xC1), take the instruction's opex_lo bit, force stall=1.
    #: (Stall=0 means yield/pause on NVIDIA hardware and would inflate
    #: timing reads to ~61 cycles.)
    MERGE_VALIDATE = "merge_validate"

    #: Preserve the harness control bytes 13-15 completely (bytes 0-12 only).
    PRESERVE_CONTROL = "preserve_control"

    #: Replace only the low 10 bytes (opcode, operands, and the SR-selector
    #: byte at byte 9); preserve bytes 10-15 (scheduling/control) — used by
    #: the contention SR-fuzzing experiment.
    MERGE_LOW10 = "merge_low10"

    #: Replace the entire 16-byte instruction, control included.
    REPLACE_FULL = "replace_full"


# ---------------------------------------------------------------------------
# Field-level access (upper-word view; works on a 16-byte instruction)
# ---------------------------------------------------------------------------

def read_control(instr: bytes) -> dict:
    """Read control fields from a 16-byte instruction.

    Returns dict with keys: stall, yield_, wr_bar, rd_bar, wt_bar.
    """
    enc = int.from_bytes(instr[8:16], "little")
    return {
        "stall":  (enc >> 41) & 0xF,
        "yield_": (enc >> 45) & 0x1,
        "wr_bar": (enc >> 46) & 0x7,
        "rd_bar": (enc >> 49) & 0x7,
        "wt_bar": (enc >> 52) & 0x3F,
    }


def write_control(instr: bytes, *, stall=None, yield_=None, wr_bar=None,
                  rd_bar=None, wt_bar=None) -> bytes:
    """Return a copy of ``instr`` with selected control fields overwritten.

    Unspecified fields (``None``) are preserved from the original.
    """
    enc = int.from_bytes(instr[8:16], "little")

    stall = (enc >> 41) & 0xF if stall is None else stall
    yield_ = (enc >> 45) & 0x1 if yield_ is None else yield_
    wr_bar = (enc >> 46) & 0x7 if wr_bar is None else wr_bar
    rd_bar = (enc >> 49) & 0x7 if rd_bar is None else rd_bar
    wt_bar = (enc >> 52) & 0x3F if wt_bar is None else wt_bar

    enc &= ~(0x1FFFF << 41)  # clear bits 41-57
    enc |= (stall & 0xF) << 41
    enc |= (yield_ & 0x1) << 45
    enc |= (wr_bar & 0x7) << 46
    enc |= (rd_bar & 0x7) << 49
    enc |= (wt_bar & 0x3F) << 52

    return instr[0:8] + enc.to_bytes(8, "little")


# ---------------------------------------------------------------------------
# Byte-13/14/15 primitives (bytearray-at-offset view used by binary patchers)
# ---------------------------------------------------------------------------

def normalize_delay(data: bytearray, off: int, delay: int = DELAY_NORMALIZED) -> None:
    """Set the delay (stall) field of the instruction at ``off`` to ``delay``."""
    data[off + 13] = (data[off + 13] & ~DELAY_MASK) | ((delay << 1) & DELAY_MASK)


def set_scoreboard_write(data: bytearray, off: int, sb_index: int) -> None:
    """Set the write-barrier (dst_wr_sb) field to ``sb_index`` (0-5; 7=none).

    dst_wr_sb: byte 13 bits 6-7 + byte 14 bit 0 (3 bits).
    """
    if not 0 <= sb_index <= 7:
        raise ValueError(f"scoreboard index out of range: {sb_index}")
    data[off + 13] = (data[off + 13] & ~0xC0) | ((sb_index & 0x3) << 6)
    data[off + 14] = (data[off + 14] & ~0x01) | ((sb_index >> 2) & 0x1)


def set_scoreboard_wait(data: bytearray, off: int, sb_index: int) -> None:
    """OR a wait for ``sb_index`` into the req_bit_set wait mask (6 bits).

    req_bit_set: byte 14 bits 4-7 + byte 15 bits 0-1. Bit N waits for SB_N;
    multiple waits can be combined by repeated calls.
    """
    if not 0 <= sb_index <= 5:
        raise ValueError(f"scoreboard wait index out of range: {sb_index}")
    if sb_index < 4:
        data[off + 14] |= (1 << (4 + sb_index))
    else:
        data[off + 15] |= (1 << (sb_index - 4))


def clear_scoreboard_bits(data: bytearray, off: int) -> None:
    """Reset all scoreboard fields to defaults: dst_wr_sb=7, src_rel_sb=7,
    req_bit_set=0 (no dependencies)."""
    # dst_wr_sb = 7: byte13 bits 6-7, byte14 bit 0
    data[off + 13] |= 0xC0
    data[off + 14] |= 0x01
    # src_rel_sb = 7: byte14 bits 1-3
    data[off + 14] |= 0x0E
    # req_bit_set = 0: byte14 bits 4-7, byte15 bits 0-1
    data[off + 14] &= ~0xF0
    data[off + 15] &= ~0x03


# ---------------------------------------------------------------------------
# Whole-instruction merge policies
# ---------------------------------------------------------------------------

def merge_measure(new: bytes, orig: bytes,
                  delay: int = DELAY_NORMALIZED) -> bytes:
    """Measurement merge (issue-mode default).

    Bytes 0-12 from ``new``; byte 13 from ``orig`` with delay normalized;
    bytes 14-15 from ``orig``.
    """
    out = bytearray(orig[:INSTRUCTION_SIZE])
    out[0:13] = new[:13]
    out[13] = (orig[13] & ~DELAY_MASK) | ((delay << 1) & DELAY_MASK)
    return bytes(out)


def merge_delay(new: bytes, orig: bytes) -> bytes:
    """Like :func:`merge_measure`, but delay bits come from ``new``."""
    out = bytearray(orig[:INSTRUCTION_SIZE])
    out[0:13] = new[:13]
    out[13] = (orig[13] & ~DELAY_MASK) | (new[13] & DELAY_MASK)
    return bytes(out)


def merge_validate(new: bytes, orig: bytes) -> bytes:
    """Runtime-validation merge.

    Preserves harness yield/dst_wr_sb bits (mask 0xC1), takes the
    instruction's opex_lo bit, and forces stall=1 (stall=0 means
    yield/pause on NVIDIA hardware, inflating timings to ~61 cycles).
    """
    out = bytearray(orig[:INSTRUCTION_SIZE])
    out[0:13] = new[:13]
    out[13] = (orig[13] & 0xC1) | (new[13] & OPEX_LO_MASK) | 0x02
    return bytes(out)


def preserve_control(new: bytes, orig: bytes) -> bytes:
    """Bytes 0-12 from ``new``; control bytes 13-15 entirely from ``orig``."""
    out = bytearray(orig[:INSTRUCTION_SIZE])
    out[0:13] = new[:13]
    return bytes(out)


def merge_low10(new: bytes, orig: bytes) -> bytes:
    """Bytes 0-9 from ``new`` (opcode, operands, SR-selector byte); bytes
    10-15 from ``orig`` (scheduling/control preserved)."""
    out = bytearray(orig[:INSTRUCTION_SIZE])
    out[0:10] = new[:10]
    return bytes(out)


def replace_full(new: bytes, orig: bytes) -> bytes:  # noqa: ARG001 - uniform API
    """Full 16-byte replacement (control included)."""
    return bytes(new[:INSTRUCTION_SIZE])


def replace_lower(new: bytes, orig: bytes) -> bytes:
    """Replace the lower 8 bytes; preserve the upper 8 except bit 91
    (upper-word bit 27), which is cleared (used with ``write_opcode``
    semantics from the binary editor)."""
    orig_hi64 = int.from_bytes(orig[8:16], "little")
    orig_hi64 &= ~(1 << 27)  # clear bit 91 of the full instruction
    return bytes(new[0:8]) + orig_hi64.to_bytes(8, "little")


_POLICY_FUNCS = {
    PatchPolicy.MERGE_MEASURE: merge_measure,
    PatchPolicy.MERGE_DELAY: merge_delay,
    PatchPolicy.MERGE_VALIDATE: merge_validate,
    PatchPolicy.PRESERVE_CONTROL: preserve_control,
    PatchPolicy.MERGE_LOW10: merge_low10,
    PatchPolicy.REPLACE_FULL: replace_full,
}


def apply_policy(policy: PatchPolicy, new: bytes, orig: bytes) -> bytes:
    """Apply a :class:`PatchPolicy` to produce patched 16-byte instruction."""
    return _POLICY_FUNCS[policy](new, orig)


def patch_at(data: bytearray, off: int, new: bytes,
             policy: PatchPolicy = PatchPolicy.MERGE_MEASURE) -> None:
    """Patch ``data`` in place at byte offset ``off`` using ``policy``.

    ``new`` is the full 16-byte instruction to patch in; ``data[off:off+16]``
    holds the original instruction whose control bytes may be preserved.
    """
    orig = bytes(data[off:off + INSTRUCTION_SIZE])
    data[off:off + INSTRUCTION_SIZE] = apply_policy(policy, new, orig)
