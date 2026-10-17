"""Tests for the union DB builder and byte-templating harvest."""

import os

import pytest

from eoss import paths


@pytest.fixture(scope="module")
def spec():
    for sm in (120, 90):
        if paths.isa_json_path(sm).is_file():
            from eoss.isa.model import ISASpec
            return ISASpec.load(sm=sm)
    pytest.skip("no ISA JSON available")


def _fake_stage(tmp_path, label):
    from eoss.schemas import OffsetsDoc, Slot
    binp = tmp_path / f"h_{label}"
    binp.write_bytes(b"\x00" * 64)
    doc = OffsetsDoc(binary=str(binp), kernel_name="k", code_base_offset=0,
                     harness="osiris_slots", slots=[Slot(0, {"measure": 0})])
    return {"label": label, "mode": "generate", "binary": str(binp), "offsets": doc}


class TestUnionBookkeeping:
    def test_seed_dedup_and_stage_counts(self, spec, tmp_path, monkeypatch):
        import eoss.runtime.union as U
        from eoss.schemas import EncodingRecord
        # every buildable class "validates" -> stage 1 sweeps them all
        monkeypatch.setattr(U, "_validate_bytes", lambda *a, **k: True)

        names = list(spec.instructions)
        seed = [EncodingRecord(instruction=names[0], bytes_le=b"\x00" * 16)]
        stages = [_fake_stage(tmp_path, "s1"), _fake_stage(tmp_path, "s2")]
        records, summary = U.build_union_db(spec, stages, output_path=None,
                                            seed_records=seed)
        got = [r.instruction for r in records]
        assert names[0] in got                      # seed retained
        assert len(got) == len(set(got))            # no duplicate class
        assert summary[0][1] > 0                     # stage 1 placed classes
        assert summary[1][1] == 0                    # stage 2: nothing left

    def test_cancel_writes_partial(self, spec, tmp_path, monkeypatch):
        import eoss.runtime.union as U
        monkeypatch.setattr(U, "_validate_bytes", lambda *a, **k: True)
        out = tmp_path / "db.json"
        stages = [_fake_stage(tmp_path, "s1")]
        records, summary = U.build_union_db(
            spec, stages, output_path=out, should_cancel=lambda: True)
        assert out.is_file()          # partial DB still written
        assert records == []          # cancelled before any placement


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
class TestHarvestGPU:
    def test_harvest_recovers_global_ldg(self, tmp_path):
        from eoss.gpu import detect_sm
        from eoss.isa.model import ISASpec
        from eoss.runtime.harness import build_harness, extract_offsets
        from eoss.runtime.union import harvest_exemplars

        sm = detect_sm()
        if not paths.isa_json_path(sm).is_file():
            pytest.skip(f"SM{sm} ISA JSON not available")
        binary = build_harness(sm, num_slots=1024, out_dir=tmp_path, verbose=False)
        doc = extract_offsets(binary, output=tmp_path / "o.json", verbose=False)
        spec = ISASpec.load(sm=sm)
        found = harvest_exemplars(binary, doc, spec)
        # the classic harness performs a real global load/store; byte-templating
        # must lift at least one validating LDG/STG exemplar.
        assert any(n.startswith(("ldg", "stg")) for n in found), sorted(found)
        for name, raw in found.items():
            assert len(raw) == 16
