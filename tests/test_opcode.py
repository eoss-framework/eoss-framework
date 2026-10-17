"""Unit tests for eoss.enc.opcode."""

import pytest

from eoss.enc import opcode
from eoss.enc.control import NOP_BYTES


class TestOpcode:
    def test_nop_opcode(self):
        # NOP encoding: bits 11:0 = 0x918, bit 91 = 0 -> opcode 0x918
        assert opcode.extract_opcode(NOP_BYTES) == 0x918

    def test_fadd_style(self):
        # FADD opcode 0x221: byte0 = 0x21, byte1 low bits = 0x02
        instr = bytearray(16)
        instr[0] = 0x21
        instr[1] = 0x72  # upper nibble belongs to other fields
        assert opcode.extract_opcode(bytes(instr)) == 0x221

    def test_write_preserves_others(self):
        instr = bytearray(NOP_BYTES)
        new = opcode.write_opcode(bytes(instr), 0x221)
        assert opcode.extract_opcode(new) == 0x221
        # bit 91 cleared (FADD has bit91=0)
        assert new[11] & 0x08 == 0
        # control bytes 13-15 untouched
        assert new[13:16] == NOP_BYTES[13:16]

    def test_write_sets_bit91(self):
        instr = bytes(16)
        new = opcode.write_opcode(instr, 0x1000)
        assert new[11] & 0x08
        assert opcode.extract_opcode(new) == 0x1000

    def test_roundtrip_all(self):
        for op in (0x0, 0x1, 0x221, 0x918, 0xFFF, 0x1000, 0x1918, 0x1FFF):
            new = opcode.write_opcode(bytes(16), op)
            assert opcode.extract_opcode(new) == op

    def test_out_of_range(self):
        with pytest.raises(ValueError):
            opcode.write_opcode(bytes(16), 0x2000)

    def test_parse_opcode_string(self):
        assert opcode.parse_opcode_string("0b1000100001") == 0x221
        assert opcode.parse_opcode_string("0x221") == 0x221
        assert opcode.parse_opcode_string("545") == 545
        assert opcode.parse_opcode_string(545) == 545
