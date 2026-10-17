"""Tests for per-generation harness feature gating and the parametrized
harness template (harnesses/osiris_harness.cu)."""

import pytest

from eoss import paths
from eoss.isa.registry import (HARNESS_FLAVORS, gen_features, harness_flavors)


class TestCapabilityProfile:
    def test_blackwell_has_everything(self):
        feats = gen_features(120)
        assert {"TEXTURE", "SURFACE", "SHARED", "TMA", "CLUSTER"} <= feats
        assert set(harness_flavors(120)) == set(HARNESS_FLAVORS)

    def test_hopper_has_tma_and_cluster(self):
        assert {"TMA", "CLUSTER"} <= gen_features(90)
        assert "tma" in harness_flavors(90) and "cga" in harness_flavors(90)

    def test_ampere_no_tma_or_cluster(self):
        feats = gen_features(86)
        assert "TEXTURE" in feats and "SHARED" in feats
        assert "TMA" not in feats and "CLUSTER" not in feats
        assert set(harness_flavors(86)) == {"super"}

    def test_volta_only_super(self):
        assert set(harness_flavors(70)) == {"super"}

    def test_flavors_gate_on_all_features(self):
        # a flavor is buildable only if every one of its features is supported
        for sm in (70, 80, 86, 89, 90, 100, 120):
            feats = gen_features(sm)
            for name, fs in harness_flavors(sm).items():
                assert set(fs) <= feats, (sm, name)

    def test_normalize_accepts_sm_prefix(self):
        assert gen_features("sm_120") == gen_features(120)


def _have_nvcc_gpu() -> bool:
    from eoss.runtime.harness import nvcc_path
    try:
        nvcc_path()
    except Exception:
        return False
    try:
        from eoss.gpu import detect_sm
        detect_sm()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _have_nvcc_gpu(), reason="nvcc + GPU required")
class TestTemplateBuild:
    def test_super_flavor_from_template(self, tmp_path):
        # the one template compiles a working 'super' flavor whose register map
        # exposes the texture/surface handles the generator policy targets.
        from eoss.gpu import detect_sm
        from eoss.runtime.harness import build_harness_flavor, extract_offsets
        from eoss.runtime.regmap import extract_register_map

        sm = detect_sm()
        if "super" not in harness_flavors(sm):
            pytest.skip(f"super flavor unsupported on sm_{sm}")
        binary = build_harness_flavor(sm, "super", num_slots=1024,
                                      out_dir=tmp_path, verbose=False)
        doc = extract_offsets(binary, output=tmp_path / "o.json",
                              kernel_hint="osiris_kernel", verbose=False)
        assert doc.num_slots == 1024
        assert doc.kernel_name and "osiris_kernel" in doc.kernel_name
        rmap = extract_register_map(binary, kernel_hint="osiris_kernel")
        assert rmap.tex_handle_ur is not None  # texture object is bound
