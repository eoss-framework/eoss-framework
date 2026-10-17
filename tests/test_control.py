"""Unit tests for eoss.enc.control (control fields + patch policies)."""

from eoss.enc import control as ctrl


def make_instr(byte13=0x00, byte14=0x00, byte15=0x00, lower=b"\xAA" * 13):
    instr = bytearray(16)
    instr[0:13] = lower
    instr[13] = byte13
    instr[14] = byte14
    instr[15] = byte15
    return bytes(instr)


class TestReadWriteControl:
    def test_roundtrip(self):
        instr = make_instr()
        new = ctrl.write_control(instr, stall=5, yield_=1, wr_bar=2, rd_bar=3, wt_bar=0x11)
        c = ctrl.read_control(new)
        assert c["stall"] == 5
        assert c["yield_"] == 1
        assert c["wr_bar"] == 2
        assert c["rd_bar"] == 3
        assert c["wt_bar"] == 0x11

    def test_partial_preserves(self):
        instr = ctrl.write_control(make_instr(), stall=7, wr_bar=1)
        new = ctrl.write_control(instr, stall=2)
        c = ctrl.read_control(new)
        assert c["stall"] == 2
        assert c["wr_bar"] == 1  # preserved

    def test_lower_bytes_untouched(self):
        instr = make_instr(lower=bytes(range(13)))
        new = ctrl.write_control(instr, stall=15)
        assert new[0:13] == bytes(range(13))


class TestScoreboardFields:
    def test_set_write(self):
        data = bytearray(16)
        ctrl.set_scoreboard_write(data, 0, 5)  # 0b101
        # byte13 bits 6-7 = 0b01, byte14 bit 0 = 1
        assert data[13] >> 6 == 0b01
        assert data[14] & 1 == 1

    def test_set_wait_or(self):
        data = bytearray(16)
        ctrl.set_scoreboard_wait(data, 0, 2)
        ctrl.set_scoreboard_wait(data, 0, 5)
        # bit 2 -> byte14 bit 6; bit 5 -> byte15 bit 1
        assert data[14] & (1 << 6)
        assert data[15] & 0b10

    def test_clear(self):
        data = bytearray([0xFF] * 16)
        ctrl.clear_scoreboard_bits(data, 0)
        # dst_wr_sb = 7, src_rel_sb = 7
        assert data[13] & 0xC0 == 0xC0
        assert data[14] & 0x0F == 0x0F
        # req_bit_set = 0
        assert data[14] & 0xF0 == 0
        assert data[15] & 0x03 == 0


class TestPolicies:
    def test_merge_measure_normalizes_delay(self):
        new = make_instr(byte13=0xFF, byte14=0xEE, byte15=0xDD)
        orig = make_instr(byte13=0x00, byte14=0x11, byte15=0x22, lower=b"\x00" * 13)
        out = ctrl.merge_measure(new, orig)
        # bytes 0-12 from new
        assert out[0:13] == new[0:13]
        # delay normalized to 4 -> byte13 bits 1-4 = 4
        assert (out[13] >> 1) & 0xF == 4
        # other byte13 bits from orig (0)
        assert out[13] & ~0x1E == 0
        # bytes 14-15 from orig
        assert out[14] == 0x11 and out[15] == 0x22

    def test_merge_delay_takes_new_delay(self):
        new = make_instr(byte13=0x1E)  # delay = 15
        orig = make_instr(byte13=0x00)
        out = ctrl.merge_delay(new, orig)
        assert (out[13] >> 1) & 0xF == 15

    def test_merge_validate(self):
        new = make_instr(byte13=0x20)  # opex_lo set
        orig = make_instr(byte13=0xC1)
        out = ctrl.merge_validate(new, orig)
        # (orig & 0xC1) | (new & 0x20) | 0x02
        assert out[13] == 0xC1 | 0x20 | 0x02

    def test_preserve_control(self):
        new = make_instr(byte13=0xFF, byte14=0xFF, byte15=0xFF)
        orig = make_instr(byte13=0x01, byte14=0x02, byte15=0x03)
        out = ctrl.preserve_control(new, orig)
        assert out[0:13] == new[0:13]
        assert out[13:] == bytes([0x01, 0x02, 0x03])

    def test_replace_full(self):
        new = make_instr(byte13=1, byte14=2, byte15=3)
        out = ctrl.replace_full(new, make_instr())
        assert out == new

    def test_replace_lower_clears_bit91(self):
        new = b"\x21\x72" + b"\x00" * 6  # lower 8 bytes
        orig = bytearray(16)
        orig[11] = 0x08  # bit 91 set
        orig[12] = 0xFF
        out = ctrl.replace_lower(new, bytes(orig))
        assert out[0:8] == new
        assert out[11] & 0x08 == 0  # bit 91 cleared
        assert out[12] == 0xFF  # rest preserved

    def test_patch_at(self):
        data = bytearray(32)
        new = make_instr(byte13=0x1E)
        ctrl.patch_at(data, 16, new, ctrl.PatchPolicy.MERGE_MEASURE)
        assert data[16:29] == new[0:13]
        assert (data[29] >> 1) & 0xF == 4
