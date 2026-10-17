"""Bit-range specification parsing and bit-level read/write for SASS encodings.

Bit specifications appear throughout the ISA JSON as:

* a string ``"91"``                  -> single bit
* a string ``"14_12"``               -> contiguous range (high_low, either order)
* a string ``"124_122_109_105"``     -> several ranges, MSB-first pairs
* a list of the above                -> concatenation of ranges

Ranges are returned normalized as ``(low, high)`` tuples, listed MSB-first.
When writing a value, the value's LSBs map to the *lowest* bit range first
(i.e. ranges are consumed in reverse order), matching every historical
implementation in this repository.
"""

import re
from typing import Any, List, Optional, Tuple

_BIT_RANGE_RE = re.compile(r"^(\d+)(?:_(\d+))?$")


def parse_bit_spec(bit_spec: Any) -> List[Tuple[int, int]]:
    """Parse a bit specification into a list of ``(low, high)`` ranges.

    Accepts single strings (``"91"``, ``"14_12"``, ``"124_122_109_105"``)
    or lists of such strings (``["91", "11_0"]``). Returns ranges
    MSB-first as ``(low, high)`` tuples. Unparseable input yields ``[]``.
    """
    segments: List[Tuple[int, int]] = []
    if isinstance(bit_spec, (list, tuple)):
        for part in bit_spec:
            segments.extend(parse_bit_spec(part))
        return segments
    if not isinstance(bit_spec, str):
        if isinstance(bit_spec, int):
            return [(bit_spec, bit_spec)]
        return segments
    token = bit_spec.strip()
    if not token:
        return segments

    parts = token.split("_")
    if len(parts) > 2:
        # Multi-range form like "124_122_109_105": consume MSB-first pairs.
        if len(parts) % 2 != 0:
            return []
        try:
            nums = [int(p) for p in parts]
        except ValueError:
            return []
        for i in range(0, len(nums), 2):
            hi, lo = nums[i], nums[i + 1]
            segments.append((min(hi, lo), max(hi, lo)))
        return segments

    match = _BIT_RANGE_RE.match(token)
    if not match:
        return []
    high = int(match.group(1))
    low = match.group(2)
    if low is None:
        return [(high, high)]
    low_v = int(low)
    return [(min(high, low_v), max(high, low_v))]


def spec_width(bit_spec: Any) -> int:
    """Total number of bits covered by a bit specification."""
    return sum(hi - lo + 1 for lo, hi in parse_bit_spec(bit_spec))


def set_bit(buffer: bytearray, bit_index: int, bit_value: int) -> None:
    """Set a single bit in a byte buffer (bit 0 = LSB of byte 0)."""
    byte_index = bit_index // 8
    within_byte = bit_index % 8
    mask = 1 << within_byte
    if bit_value & 1:
        buffer[byte_index] |= mask
    else:
        buffer[byte_index] &= ~mask


def write_bit_range(buffer: bytearray, start_bit: int, end_bit: int, value: int) -> None:
    """Write a value into a contiguous bit range (LSB of value -> start_bit)."""
    if start_bit > end_bit:
        start_bit, end_bit = end_bit, start_bit
    for offset, bit in enumerate(range(start_bit, end_bit + 1)):
        set_bit(buffer, bit, (value >> offset) & 1)


def write_bit_segments(buffer: bytearray, bit_spec: Any, value: int,
                       width: Optional[int] = None) -> None:
    """Write a value into possibly discontiguous bit segments.

    Value LSBs map to the lowest bit range first. ``width`` is accepted for
    backward compatibility and currently ignored (the total width is derived
    from ``bit_spec``).
    """
    segments = parse_bit_spec(bit_spec)
    remaining = int(value)
    shift = 0
    for start, end in reversed(segments):
        seg_width = end - start + 1
        mask = (1 << seg_width) - 1
        write_bit_range(buffer, start, end, (remaining >> shift) & mask)
        shift += seg_width


def read_bit_range(buffer: bytes, start_bit: int, end_bit: int) -> int:
    """Read a value from a contiguous bit range."""
    if start_bit > end_bit:
        start_bit, end_bit = end_bit, start_bit
    value = 0
    for offset, bit in enumerate(range(start_bit, end_bit + 1)):
        byte_index = bit // 8
        if byte_index >= len(buffer):
            break
        value |= ((buffer[byte_index] >> (bit % 8)) & 1) << offset
    return value


def read_bit_segments(buffer: bytes, bit_spec: Any) -> int:
    """Read a value from possibly discontiguous bit segments.

    Inverse of :func:`write_bit_segments`.
    """
    segments = parse_bit_spec(bit_spec)
    value = 0
    shift = 0
    for start, end in reversed(segments):
        seg_width = end - start + 1
        value |= read_bit_range(buffer, start, end) << shift
        shift += seg_width
    return value


def parse_int_literal(text: str) -> int:
    """Parse an integer from hex (0x), binary (0b) or decimal string."""
    text = text.strip()
    if text.lower().startswith("0x"):
        return int(text, 16)
    if text.lower().startswith("0b"):
        return int(text, 2)
    return int(text, 10)
