"""Tests for the harness register-map extractor and the generator's
harness-aware texture/surface operand policy."""

import pytest

from eoss import paths
from eoss.runtime.regmap import RegisterMap, extract_register_map

# A minimal cuobjdump-style SASS dump exercising each resource op.
SAMPLE_SASS = """
	code for sm_120
		Function : _Z11super_kernel
	.headerflags	@"EF_CUDA_SM120"
        /*0000*/  LDG.E R8, desc[UR8][R2.64] ;        /* 0x0000000806097981 */
        /*0010*/  LDG.E R9, desc[UR8][R4.64+0x4] ;    /* 0x0000040806097981 */
        /*0020*/  TEX RZ, R12, R12, UR4, 2D, 0x1 ;    /* 0x20ff04ff0c0c7f60 */
        /*0030*/  SULD.D.BA.2D.STRONG.SM R0, [R16], UR10 ; /* 0x60ff0a0010007f99 */
        /*0040*/  STS [R6], R5 ;                      /* 0x0000000000067388 */
        /*0050*/  UTMALDG.1D [UR40], [UR38] ;         /* 0x00000028260075b4 */
"""


class TestExtractor:
    def test_parse_sample(self):
        m = extract_register_map(None, sass=SAMPLE_SASS)
        assert m.global_desc_ur == 8
        assert 2 in m.address_regs and 4 in m.address_regs
        assert m.tex_handle_ur == 4
        assert m.tex_coord_regs == [12]        # TEX RZ, R12, R12 -> one distinct
        assert m.surf_handle_ur == 10
        assert m.surf_coord_regs == [16]
        assert m.shared_regs == [6]
        assert m.tma_shared_ur == 40 and m.tma_desc_ur == 38

    def test_empty_dump(self):
        m = extract_register_map(None, sass="Function : _Z1kv\n")
        assert m.global_desc_ur is None
        assert m.tex_handle_ur is None
        assert m.address_regs == []

    def test_pre_hopper_gpr_texture(self):
        # Pre-Hopper bindless: the texture handle is a GPR, coord the preceding
        # GPR — captured only when the Hopper UR form is absent.
        amp = ("Function : k\n        /*0010*/  "
               "TEX.SCR.LL RZ, R12, R10, R13, 0x0, 0x64, 2D, 0x1 ;  /* 0x1 */\n")
        m = extract_register_map(None, sass=amp)
        assert m.tex_handle_reg == 13 and m.tex_coord_reg == 10
        assert m.tex_handle_ur is None
        # the Hopper UR form must NOT trip the GPR pattern
        bw = ("Function : k\n        /*0010*/  "
              "TEX RZ, R12, R12, UR4, 2D ;  /* 0x1 */\n")
        b = extract_register_map(None, sass=bw)
        assert b.tex_handle_ur == 4 and b.tex_handle_reg is None

    def test_roundtrip(self):
        m = RegisterMap(global_desc_ur=8, tex_handle_ur=4,
                        tex_coord_regs=[12, 16], surf_handle_ur=10,
                        shared_regs=[6])
        assert RegisterMap.from_json(m.to_json()) == m


class TestEncoderPolicy:
    @pytest.fixture(scope="class")
    def spec(self):
        for sm in (120, 90):
            if paths.isa_json_path(sm).is_file():
                from eoss.isa.model import ISASpec
                return ISASpec.load(sm=sm)
        pytest.skip("no ISA JSON available")

    def _tex_class(self, spec):
        for name, ic in spec.instructions.items():
            if name.startswith("tex_") and "URe" in ic.arguments:
                return name
        return None

    def test_map_targets_handle_register(self, spec):
        from eoss.enc.generator import Encoder
        name = self._tex_class(spec)
        if name is None:
            pytest.skip("no texture class with a URe handle in this spec")
        # default policy parks the handle in UR4
        default = Encoder(spec).build(name)
        assert default.argument_values.get("URe") == 4
        # register map redirects it to the harness's live handle register
        mapped = Encoder(spec, register_map={"tex_handle_ur": 7,
                                             "tex_coord_regs": [12]}).build(name)
        assert mapped.argument_values.get("URe") == 7
        assert mapped.argument_values.get("Ra") == 12

    def test_tma_policy_targets_descriptor(self, spec):
        from eoss.enc.generator import Encoder
        name = next((n for n in spec.instructions
                     if n.startswith("utmaldg")
                     and "Ra" in spec.get(n).arguments
                     and "Rb" in spec.get(n).arguments), None)
        if name is None:
            pytest.skip("no utmaldg class with Ra/Rb in this spec")
        ins = Encoder(spec, register_map={"tma_desc_ur": 38,
                                          "tma_shared_ur": 40}).build(name)
        assert ins.argument_values.get("Ra") == 38    # descriptor UR
        assert ins.argument_values.get("Rb") == 40    # shared-dest UR

    def test_pre_hopper_gpr_texture_policy(self):
        # a GPR-handle map targets the texture handle/coord GPRs and sets SCR
        # (pre-Hopper bindless) — exercised on the SM80 spec where tex classes
        # carry a GPR handle rather than a URe.
        if not paths.isa_json_path(80).is_file():
            pytest.skip("no SM80 ISA JSON")
        from eoss.isa.model import ISASpec
        from eoss.enc.generator import Encoder
        spec80 = ISASpec.load(sm=80)
        name = next((n for n in spec80.instructions
                     if n.startswith("tex_scr")
                     and {"Ra", "Rb", "scr"} <= set(spec80.get(n).arguments)), None)
        if name is None:
            pytest.skip("no tex_scr class with Ra/Rb/scr in SM80")
        ins = Encoder(spec80, register_map={"tex_handle_reg": 13,
                                            "tex_coord_reg": 10}).build(name)
        assert ins.argument_values.get("Rb") == 13   # handle in a GPR
        assert ins.argument_values.get("Ra") == 10   # coord in a GPR
        assert ins.argument_values.get("scr") == 1    # SCR bindless mode

    def test_no_map_is_unchanged(self, spec):
        # register_map=None must reproduce the historical hardcoded policy
        from eoss.enc.generator import Encoder
        name = self._tex_class(spec)
        if name is None:
            pytest.skip("no texture class")
        a = Encoder(spec).build(name).bytes
        b = Encoder(spec, register_map=None).build(name).bytes
        assert a == b
