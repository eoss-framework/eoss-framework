"""13-bit SASS opcode extraction and insertion.

SASS opcodes are 13 bits, stored non-contiguously in the 128-bit
instruction word: bit 91 (MSB) concatenated with bits [11:0].

This is the canonical implementation; it replaces copies previously
embedded in patch_sass.py, modifier.html, assemble_instructions.py and
generate_undefined_opcodes.py.
"""

INSTRUCTION_SIZE = 16  # bytes per SASS instruction (128 bits)

MAX_OPCODE = (1 << 13) - 1  # 8191

#: Default bit-range spec for the opcode field (bit 91 + bits 11:0),
#: expressed in the BITS_ range-list convention used by ISA JSON ARGUMENTS.
OPCODE_BITS = ["91", "11_0"]


def extract_opcode(instr: bytes) -> int:
    """Extract the 13-bit opcode from a 16-byte instruction.

    Opcode = bit91 (1 bit) concatenated with bits[11:0] (12 bits).
    """
    full_128 = int.from_bytes(instr[0:16], "little")
    bit91 = (full_128 >> 91) & 1
    bits_11_0 = full_128 & 0xFFF
    return (bit91 << 12) | bits_11_0


def write_opcode(instr: bytes, new_opcode: int) -> bytes:
    """Write a 13-bit opcode into a 16-byte instruction.

    All other bits are preserved. Returns the new 16-byte instruction.
    """
    if not 0 <= new_opcode <= MAX_OPCODE:
        raise ValueError(f"opcode out of range (0..{MAX_OPCODE}): {new_opcode}")
    full_128 = int.from_bytes(instr[0:16], "little")
    # Clear bits 0-11 and bit 91
    full_128 &= ~0xFFF
    full_128 &= ~(1 << 91)
    # Set new bits 0-11 and new bit 91
    full_128 |= (new_opcode & 0xFFF)
    full_128 |= ((new_opcode >> 12) & 1) << 91
    return full_128.to_bytes(16, "little")


def parse_opcode_string(s: str) -> int:
    """Parse an opcode given as '0b...' binary string or int-like."""
    if isinstance(s, int):
        return s
    s = s.strip()
    if s.startswith("0b"):
        return int(s, 2)
    if s.startswith("0x"):
        return int(s, 16)
    return int(s)
