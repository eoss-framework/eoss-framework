"""Unit tests for eoss.schemas and eoss.binary.patcher."""

import pytest

from eoss import schemas
from eoss.binary.patcher import BinaryPatcher
from eoss.enc import control as ctrl
from eoss.enc.opcode import extract_opcode

from test_elf import build_synthetic_cubin


class TestOffsetsDoc:
    def doc(self):
        return schemas.OffsetsDoc(
            binary="harness", kernel_name="k", code_base_offset=0x1000,
            harness="osiris_slots",
            slots=[
                schemas.Slot(index=0, anchors={"reset": 0x100, "trigger": 0x110,
                                               "measure": 0x130}),
                schemas.Slot(index=1, anchors={"reset": 0x200, "trigger": 0x210,
                                               "measure": 0x230}),
            ])

    def test_file_offset(self):
        doc = self.doc()
        assert doc.file_offset(doc.slots[1], "trigger") == 0x1000 + 0x210

    def test_roundtrip(self, tmp_path):
        doc = self.doc()
        p = tmp_path / "offsets.json"
        doc.save(p)
        loaded = schemas.OffsetsDoc.load(p)
        assert loaded.num_slots == 2
        assert loaded.slots[0].anchors["reset"] == 0x100
        assert loaded.code_base_offset == 0x1000
        assert loaded.has_anchor("trigger")

    def test_legacy_rejected(self, tmp_path):
        p = tmp_path / "legacy.json"
        p.write_text('{"slots": [{"slot_index": 0, "reset_offset": 1}]}')
        with pytest.raises(ValueError, match="legacy"):
            schemas.OffsetsDoc.load(p)


class TestEncodingRecord:
    def test_roundtrip(self):
        rec = schemas.EncodingRecord(
            instruction="fadd__RRR_RR", bytes_le=bytes(range(16)),
            opcode_key="fadd__RRR_RR", pipeline="fmalighter_pipe", delay=1)
        p = rec.to_json()
        assert p["bytes_le"] == bytes(range(16)).hex()
        rec2 = schemas.EncodingRecord.from_json(p)
        assert rec2.instruction == "fadd__RRR_RR"
        assert rec2.bytes_le == bytes(range(16))
        assert rec2.pipeline == "fmalighter_pipe"

    def test_load_save(self, tmp_path):
        recs = [schemas.EncodingRecord(instruction="a", bytes_le=bytes(16)),
                schemas.EncodingRecord(instruction="b", bytes_le=bytes(16))]
        p = tmp_path / "enc.json"
        schemas.save_encodings(recs, p)
        loaded = schemas.load_encodings(p)
        assert [r.instruction for r in loaded] == ["a", "b"]
        assert set(schemas.encodings_by_name(loaded)) == {"a", "b"}


class TestBinaryPatcher:
    def make_binary(self, tmp_path, text):
        data = build_synthetic_cubin(text)
        p = tmp_path / "bin.cubin"
        p.write_bytes(data)
        return p

    def test_sections_and_patch_at(self, tmp_path):
        p = self.make_binary(tmp_path, bytes(64))
        patcher = BinaryPatcher(p)
        assert len(patcher.sections) == 1
        off = patcher.sections[0][0]
        new = b"\x11" * 13 + b"\x22\x33\x44"
        patcher.patch_at(off, new, ctrl.PatchPolicy.PRESERVE_CONTROL)
        assert patcher.instruction_at(off)[0:13] == b"\x11" * 13
        assert patcher.instruction_at(off)[13:16] == b"\x00" * 3

    def test_nop_at(self, tmp_path):
        p = self.make_binary(tmp_path, bytes(64))
        patcher = BinaryPatcher(p)
        off = patcher.sections[0][0]
        patcher.nop_at(off)
        assert extract_opcode(patcher.instruction_at(off)) == 0x918

    def test_patch_by_opcode(self, tmp_path):
        # two instructions: one FADD-like (opcode 0x221), one other
        fadd = bytearray(16)
        fadd[0] = 0x21
        fadd[1] = 0x02
        other = bytearray(16)
        text = bytes(fadd) + bytes(other) + bytes(fadd) + bytes(16)
        p = self.make_binary(tmp_path, text)
        patcher = BinaryPatcher(p)
        count = patcher.patch_by_opcode({0x221}, 0x229, mode="opcode")
        assert count == 2
        off = patcher.sections[0][0]
        assert extract_opcode(patcher.instruction_at(off)) == 0x229
        assert extract_opcode(patcher.instruction_at(off + 32)) == 0x229
        assert extract_opcode(patcher.instruction_at(off + 16)) == 0

    def test_patch_by_opcode_after_n(self, tmp_path):
        # 4 zeroed instructions (all opcode 0x0): after_n=2 skips the first 2
        text = bytes(16) * 4
        p = self.make_binary(tmp_path, text)
        patcher = BinaryPatcher(p)
        count = patcher.patch_by_opcode({0x0}, bytes([0xFF] * 16),
                                        mode="full", after_n=2)
        assert count == 2

    def test_save(self, tmp_path):
        p = self.make_binary(tmp_path, bytes(64))
        patcher = BinaryPatcher(p)
        out = patcher.save(tmp_path / "out.bin")
        assert out.exists()
        assert out.read_bytes() == p.read_bytes()
