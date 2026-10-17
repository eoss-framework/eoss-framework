"""Encoding layer: bit packing, opcodes, control fields, instruction generation."""

from .bits import (  # noqa: F401
    parse_bit_spec,
    parse_int_literal,
    read_bit_segments,
    set_bit,
    spec_width,
    write_bit_range,
    write_bit_segments,
)
from .control import (  # noqa: F401
    DELAY_NORMALIZED,
    NOP_BYTES,
    PatchPolicy,
    apply_policy,
    clear_scoreboard_bits,
    merge_measure,
    merge_validate,
    normalize_delay,
    patch_at,
    read_control,
    set_scoreboard_wait,
    set_scoreboard_write,
    write_control,
)
from .opcode import (  # noqa: F401
    INSTRUCTION_SIZE,
    MAX_OPCODE,
    extract_opcode,
    parse_opcode_string,
    write_opcode,
)
