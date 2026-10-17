"""Tests for the encoding generator and the programmable Encoder API.

Golden encodings are pinned against the shipped SM120 validated set
(Contention/JSON/validated_gpr.json), which the refactored generator
reproduces byte-for-byte under the default operand policy.
"""

import collections
import json

import pytest

from eoss import paths
from eoss.enc.generator import Encoder, GeneratorConfig, generate_all_instructions
from eoss.isa.model import ISASpec

GOLDEN = {
    # class name -> expected 16-byte encoding hex (SM120, default policy)
    "fadd__RRR_RR": "2172ffffff0000000000000000c00f00",
    "cs2r_": "0578ff00000000000000010000c00f00",
    "s2r_": "1979ff00000000000000000000c00f00",
    "mufu__RRR_RR": "0873ff00ff0000000000000000c00f00",
    "ffma__RRR_RRR": "2372ffffff000000ff00000000c00f00",
    "imad__RRR_RRR": "2472ffffff000000ff008e0700c00f00",
    "mov_64__RR": "0272ff00ff000000000f010000c00f00",
}


@pytest.fixture(scope="module")
def spec():
    if not paths.isa_json_path(120).is_file():
        pytest.skip("SM120 ISA JSON not available")
    return ISASpec.load(sm=120)


@pytest.fixture(scope="module")
def encoder(spec):
    return Encoder(spec, seed=42)


class TestGoldenEncodings:
    @pytest.mark.parametrize("name,expected", GOLDEN.items())
    def test_golden(self, encoder, name, expected):
        ins = encoder.build(name)
        assert ins.hex == expected
        assert ins.valid

    def test_generate_all_deterministic(self, spec):
        cfg = GeneratorConfig(json_path=spec.source, output_path=None,
                              all_rz=True, zero_immediates=True, seed=42)
        a = generate_all_instructions(spec.raw, cfg)
        cfg2 = GeneratorConfig(json_path=spec.source, output_path=None,
                               all_rz=True, zero_immediates=True, seed=42)
        b = generate_all_instructions(spec.raw, cfg2)
        assert [r.bytes_le for r in a] == [r.bytes_le for r in b]
        assert len(a) == 1411


class TestEncoderAPI:
    def test_register_overrides(self, encoder):
        ins = encoder.build("fadd__RRR_RR", Rd="R5", Ra="R2", Rb="R3")
        assert ins.valid
        assert ins.bytes[2] == 5   # Rd at bits 16-23
        assert ins.bytes[3] == 2   # Ra at bits 24-31
        assert ins.bytes[4] == 3   # Rb at bits 32-39

    def test_stall_override(self, encoder):
        ins = encoder.build("fadd__RRR_RR", stall=9)
        assert ins.control["stall"] == 9
        assert ins.valid

    def test_scoreboard_overrides(self, encoder):
        ins = encoder.build("fadd__RRR_RR", Rd="R5", dst_wr_sb=0, req_bit_set=2)
        assert ins.control["wr_bar"] == 0
        assert ins.control["wt_bar"] == 2
        assert ins.valid

    def test_invalid_constraint_diagnosis(self, encoder):
        # R254 is reserved; the generator must report the exact constraint
        bad = encoder.build("fadd__RRR_RR", Rd="R254")
        assert not bad.valid
        assert "OOR_REG_ERROR" in bad.failed_conditions

    def test_unknown_class(self, encoder):
        with pytest.raises(KeyError):
            encoder.build("not_an_instruction")

    def test_unknown_field(self, encoder):
        with pytest.raises(KeyError):
            encoder.build("fadd__RRR_RR", not_a_field=1)

    def test_stall_opex_conflict(self, encoder):
        with pytest.raises(ValueError):
            encoder.build("fadd__RRR_RR", stall=1, opex=2)


class TestEncodingsPerClass:
    """First-k operand enumerator: multiple constraint-satisfying encodings
    per class by not collapsing free operand fields (registers, immediates)."""

    def test_k1_backward_compatible(self, spec):
        cfg = GeneratorConfig(json_path=spec.source, output_path=None, seed=42)
        base = generate_all_instructions(spec.raw, cfg)
        assert len(base) == 1411  # one representative encoding per class

    def test_k_gt_1_distinct_and_valid(self, spec):
        cfg = GeneratorConfig(json_path=spec.source, output_path=None,
                              encodings_per_class=4, seed=42)
        res = generate_all_instructions(spec.raw, cfg)
        assert 1411 < len(res) <= 1411 * 4
        by_class = collections.defaultdict(list)
        for r in res:
            by_class[r.opcode_key].append(r)
        saw_multi = False
        for group in by_class.values():
            assert len(group) <= 4                                   # never exceed k
            assert len({r.bytes_le for r in group}) == len(group)    # distinct bytes
            if len(group) > 1:
                saw_multi = True
                assert all(r.is_valid for r in group)                # all constraint-valid
        assert saw_multi  # classes with free operand space expanded past one

    def test_k_gt_1_deterministic(self, spec):
        def run():
            return generate_all_instructions(spec.raw, GeneratorConfig(
                json_path=spec.source, output_path=None,
                encodings_per_class=3, seed=42))
        assert [r.bytes_le for r in run()] == [r.bytes_le for r in run()]
