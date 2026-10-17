"""Unit tests for eoss.enc.bits."""

from eoss.enc import bits


class TestParseBitSpec:
    def test_single_bit_string(self):
        assert bits.parse_bit_spec("91") == [(91, 91)]

    def test_range_string(self):
        assert bits.parse_bit_spec("14_12") == [(12, 14)]

    def test_range_reversed_normalized(self):
        assert bits.parse_bit_spec("12_14") == [(12, 14)]

    def test_list_form(self):
        assert bits.parse_bit_spec(["91", "11_0"]) == [(91, 91), (0, 11)]

    def test_multi_part_string(self):
        assert bits.parse_bit_spec("124_122_109_105") == [(122, 124), (105, 109)]

    def test_list_of_multi(self):
        assert bits.parse_bit_spec(["124_122", "109_105"]) == [(122, 124), (105, 109)]

    def test_int_input(self):
        assert bits.parse_bit_spec(7) == [(7, 7)]

    def test_invalid(self):
        assert bits.parse_bit_spec("abc") == []
        assert bits.parse_bit_spec(None) == []

    def test_width(self):
        assert bits.spec_width(["91", "11_0"]) == 13
        assert bits.spec_width("124_122_109_105") == 8


class TestWriteReadRoundtrip:
    def test_single_range(self):
        buf = bytearray(16)
        bits.write_bit_segments(buf, "14_12", 0b101)
        assert bits.read_bit_segments(buf, "14_12") == 0b101

    def test_contiguous_value(self):
        buf = bytearray(16)
        bits.write_bit_segments(buf, "11_0", 0x918)
        assert buf[0] == 0x18
        assert buf[1] == 0x09
        assert bits.read_bit_segments(buf, "11_0") == 0x918

    def test_opcode_layout(self):
        """13-bit opcode: bit 91 + bits 11:0 (the SASS opcode layout)."""
        buf = bytearray(16)
        opcode = 0x1918  # bit 12 set -> maps to bit 91
        bits.write_bit_segments(buf, ["91", "11_0"], opcode)
        assert bits.read_bit_segments(buf, ["91", "11_0"]) == opcode
        # bit 91 must be byte 11 bit 3
        assert buf[11] & 0x08

    def test_opex_layout(self):
        """Discontiguous opex field: bits 124-122 + 109-105 (8 bits).

        Value LSBs map to the lowest range first: value bits 0-4 -> 105-109,
        value bits 5-7 -> 122-124.
        """
        buf = bytearray(16)
        value = 0b101_10010  # batch=101, usched=10010
        bits.write_bit_segments(buf, ["124_122", "109_105"], value)
        assert bits.read_bit_segments(buf, ["124_122", "109_105"]) == value
        # low 5 bits (0b10010) at full bits 105-109 -> byte 13 bits 1-5
        assert (buf[13] >> 1) & 0x1F == 0b10010

    def test_multi_part_string_roundtrip(self):
        buf = bytearray(16)
        bits.write_bit_segments(buf, "124_122_109_105", 0xA5)
        assert bits.read_bit_segments(buf, "124_122_109_105") == 0xA5

    def test_no_overflow_into_neighbors(self):
        buf = bytearray(16)
        bits.write_bit_segments(buf, "14_12", 0b111)
        # bits 12-14 = byte 1 bits 4-6
        assert (buf[1] >> 4) & 0b111 == 0b111
        assert buf[1] & ~0x70 == 0
        assert buf[0] == 0

    def test_parse_int_literal(self):
        assert bits.parse_int_literal("0x1F") == 31
        assert bits.parse_int_literal("0b101") == 5
        assert bits.parse_int_literal("42") == 42
