"""Tests for the TUI decode/model layers (eoss.tui) — no curses required.

Uses a synthetic SM90 cubin whose .text holds encodings produced by the
framework's own Encoder, so decode must round-trip exactly.
"""

import os
import shutil
import subprocess

import pytest

from eoss import paths
from eoss.enc.control import INSTRUCTION_SIZE, read_control
from eoss.enc.generator import Encoder
from eoss.enc.opcode import extract_opcode
from eoss.isa.model import ISASpec
from eoss.tui import elfview
from eoss.tui.decode import Decoder, collect_alias_keys
from eoss.tui.model import EditError, EditorModel

from test_elf import build_synthetic_cubin


@pytest.fixture(scope="module")
def spec():
    if not paths.isa_json_path(90).is_file():
        pytest.skip("SM90 ISA JSON not available")
    return ISASpec.load(sm=90)


@pytest.fixture(scope="module")
def encoder(spec):
    return Encoder(spec, seed=7)


@pytest.fixture(scope="module")
def decoder(spec):
    return Decoder(spec)


CLASSES = ["fadd__RRR_RR", "s2r_", "mov__RR", "imad__RRR_RRR"]


def _encode(encoder, name, fields=None):
    ins = encoder.build(name, fields or {})
    return ins.bytes


@pytest.fixture(scope="module")
def cubin(tmp_path_factory, encoder):
    chunks = []
    for name in CLASSES:
        try:
            chunks.append(_encode(encoder, name))
        except Exception:
            pass
    assert len(chunks) >= 2, "need at least two encodable classes"
    text = b"".join(chunks)
    data = build_synthetic_cubin(text)
    path = tmp_path_factory.mktemp("tui") / "synthetic.cubin"
    path.write_bytes(data)
    return path, len(chunks)


class TestDecoder:
    def test_collect_alias_keys(self):
        keys = collect_alias_keys("Pg", "Pg@not")
        assert "Pg" in keys and "pg" in keys

    def test_decode_known_classes(self, decoder, encoder):
        for name in CLASSES:
            try:
                raw = _encode(encoder, name)
            except Exception:
                continue
            decoded = decoder.decode(0, raw)
            assert decoded.defn is not None, name
            # same opcode family: decoded class shares the opcode
            assert decoded.defn.opcode == extract_opcode(raw) or \
                extract_opcode(raw) in decoded.defn.opcodes.values()
            assert decoded.mnemonic and decoded.mnemonic != "?"
            assert decoded.fields, name
            # no failing conditions on encoder-produced instructions
            assert decoded.n_failed == 0, (
                name, [c.name for c in decoded.checks if c.status == "fail"])

    def test_unknown_opcode(self, decoder):
        raw = bytes(16)
        decoded = decoder.decode(0, raw)
        assert decoded.opcode == 0
        assert decoded.text  # renders something even when unknown

    def test_field_values_match_bits(self, decoder, encoder):
        raw = _encode(encoder, "fadd__RRR_RR")
        decoded = decoder.decode(0, raw)
        rd = next(f for f in decoded.fields if f.name == "Rd")
        assert rd.value == raw[2]  # Rd lives in bits 23:16
        assert rd.display.startswith("R")

    def test_control_fields(self, decoder, encoder):
        raw = _encode(encoder, "fadd__RRR_RR")
        decoded = decoder.decode(0, raw)
        assert decoded.control == read_control(raw)

    def test_alias_resolved_sources(self, decoder, encoder):
        # FORMAT FIELDS name their target by alias: cs2r_'s /QInteger sz
        # modifier lives in an argument NOT named "sz", and the special
        # register is aliased SRa. Both must still resolve.
        raw = _encode(encoder, "cs2r_")
        decoded = decoder.decode(0, raw, override="cs2r_")
        sz = next(f for f in decoded.fields if f.alias == "sz")
        assert sz.bucket == "QInteger"
        assert sz.display in ("32", "64", "_32", "_64")
        sra = next(f for f in decoded.fields if f.alias == "SRa")
        assert sra.source == "SpecialRegister"
        assert sra.display.startswith("SR")

    @pytest.mark.skipif(not paths.isa_json_path(120).is_file(),
                        reason="SM120 ISA JSON not available")
    def test_sm120_imm8_special_register(self):
        spec120 = ISASpec.load(sm=120)
        raw = Encoder(spec120, seed=1).build("s2r_", {}).bytes
        decoded = Decoder(spec120).decode(0, raw, override="s2r_")
        imm8 = next(f for f in decoded.fields if f.name == "imm8")
        assert imm8.source == "SpecialRegister"
        assert imm8.display.startswith("SR")


class TestEditorModel:
    @pytest.fixture()
    def model(self, cubin, spec):
        path, _n = cubin
        return EditorModel(str(path), spec)

    def test_sections_found(self, model, cubin):
        _path, n = cubin
        assert len(model.sections) == 1
        assert model.sections[0].n_instructions == n

    def test_images_parsed(self, model):
        assert model.cuda_images
        image = model.cuda_images[0]
        assert image.is_cuda
        assert image.section_named(".text") is not None

    def test_set_field_roundtrip_and_undo(self, model):
        sec = model.sections[0]
        off = sec.abs_offset
        decoded = model.decode_at(off)
        assert decoded.defn is not None
        reg_field = next(f for f in decoded.fields
                         if f.bucket == "Register" and f.name != "opcode")
        model.set_field(off, reg_field.name, "R7")
        after = model.decode_at(off)
        new_view = next(f for f in after.fields if f.name == reg_field.name)
        assert new_view.value == 7
        assert new_view.display == "R7"
        assert model.n_modified > 0

        label = model.undo()
        assert label and reg_field.name in label
        assert model.n_modified == 0
        assert bytes(model.data) == model.original

    @pytest.mark.parametrize("text, index, display", [
        ("SR_EQMASK", 56, "SR_EQMASK"),
        ("SR_SNAP_PM_HI7", 131, "SR_SNAP_PM_HI7"),
        ("SR_TTU_TICKET_INFO", 133, "SR_TTU_SESSION_ACTIVE"),
        ("SR_SW_SCRATCH", 24, "SR_MACHINE_ID_0"),
        ("SR_TID.X", 33, "SR_TID_X"),
        ("SR_TID_X", 33, "SR_TID_X"),
        ("SR255", 255, "SRZ"),
        ("SR4", 4, "SR4"),
    ])
    def test_special_register_names_and_aliases(self, model, text, index, display):
        off = model.sections[0].abs_offset + INSTRUCTION_SIZE  # s2r_
        field = next(f for f in model.decode_at(off).fields
                     if f.source == "SpecialRegister")
        model.set_field(off, field.name, text)
        updated = next(f for f in model.decode_at(off).fields if f.name == field.name)
        assert updated.value == index
        assert updated.display == display

    def test_set_field_rejects_bad_values(self, model):
        sec = model.sections[0]
        off = sec.abs_offset
        decoded = model.decode_at(off)
        name = decoded.fields[0].name
        with pytest.raises(EditError):
            model.set_field(off, name, "not-a-value!!")
        with pytest.raises(EditError):
            model.set_field(off, "no_such_field", "1")

    def test_set_opcode_by_mnemonic(self, model):
        sec = model.sections[0]
        off = sec.abs_offset
        opcode = model.parse_opcode_value("FFMA")
        model.set_opcode(off, opcode)
        decoded = model.decode_at(off)
        assert decoded.opcode == opcode
        assert "FFMA" in decoded.mnemonic
        model.undo()

    def test_set_control(self, model):
        sec = model.sections[0]
        off = sec.abs_offset
        model.set_control(off, "stall", 4)
        assert model.control_of(off)["stall"] == 4
        with pytest.raises(EditError):
            model.set_control(off, "stall", 99)
        model.undo()

    def test_search(self, model):
        sec = model.sections[0]
        decoded = model.decode_at(sec.abs_offset)
        idx = model.search_instruction(sec, decoded.mnemonic.split(".")[0])
        assert idx == 0

    def test_save(self, model, tmp_path):
        sec = model.sections[0]
        model.set_control(sec.abs_offset, "stall", 5)
        out = tmp_path / "patched.cubin"
        model.save(str(out))
        saved = out.read_bytes()
        assert saved == bytes(model.data)
        assert saved != model.original
        model.undo()

    def test_hex_edit_tracks_modified(self, model):
        model.set_bytes(0x40, b"\xAA\xBB")
        assert {0x40, 0x41} <= model.modified
        model.undo()
        assert not model.modified

    def test_swap_instructions(self, model):
        sec = model.sections[0]
        off = sec.abs_offset
        first = model.instruction_bytes(off)
        second = model.instruction_bytes(off + INSTRUCTION_SIZE)
        model.swap_instructions(off, off + INSTRUCTION_SIZE)
        assert model.instruction_bytes(off) == second
        assert model.instruction_bytes(off + INSTRUCTION_SIZE) == first
        # one undo entry restores both slots
        model.undo()
        assert model.instruction_bytes(off) == first
        assert not model.modified
        with pytest.raises(EditError):
            model.swap_instructions(off, off + 2 * INSTRUCTION_SIZE)

    def test_generate_generic(self, model):
        sec = model.sections[0]
        off = sec.abs_offset
        before = model.decode_at(off)
        assert before.defn is not None
        # scramble an operand register, then regenerate a clean encoding
        reg_field = next(f for f in before.fields
                         if f.bucket == "Register" and f.name != "opcode")
        model.set_field(off, reg_field.name, "R13")
        model.variant_overrides[off] = before.class_name  # as the app pins it
        name = model.generate_generic(off)
        assert name == before.class_name
        after = model.decode_at(off)
        assert after.opcode == before.opcode
        assert after.class_name == name
        assert after.n_failed == 0, [c.name for c in after.checks
                                     if c.status == "fail"]
        # generic operand policy: registers back to RZ
        view = next(f for f in after.fields if f.name == reg_field.name)
        assert view.display == "RZ"
        # generate is a single undo entry
        label = model.undo()
        assert label and name in label
        model.undo()  # the scramble
        assert bytes(model.data) == model.original

    def test_generate_generic_unknown_opcode(self, model):
        off = model.sections[0].abs_offset
        model.set_bytes(off, bytes(INSTRUCTION_SIZE))
        with pytest.raises(EditError):
            model.generate_generic(off)

    def test_write_generic(self, model):
        # stamp a different class over the second instruction (ISA browser)
        off = model.sections[0].abs_offset + INSTRUCTION_SIZE
        name = model.write_generic(off, "imad__RRR_RRR")
        assert name == "imad__RRR_RRR"
        decoded = model.decode_at(off)
        assert decoded.class_name == "imad__RRR_RRR"
        assert decoded.n_failed == 0, [c.name for c in decoded.checks
                                       if c.status == "fail"]
        model.undo()
        assert not model.modified
        with pytest.raises(EditError):
            model.write_generic(off, "no_such_class")
        with pytest.raises(EditError):
            model.write_generic(len(model.data), "imad__RRR_RRR")

    def test_barrier_events_and_spans(self, model):
        sec = model.sections[0]
        off = sec.abs_offset
        # producer at 0 sets B2 (wr); consumer at 2 waits on B2
        model.set_control(off, "wr_bar", 2)
        model.set_control(off + 2 * INSTRUCTION_SIZE, "wt_bar", 1 << 2)
        events = model.barrier_events(sec)
        assert events[0][0] == 2
        assert events[2][2] & (1 << 2)
        spans = model.barrier_spans(sec)
        assert (0, 2, "wr") in spans[2]
        # rd_bar producer with no consumer -> open span
        model.set_control(off + 3 * INSTRUCTION_SIZE, "rd_bar", 4)
        spans = model.barrier_spans(sec)
        assert (3, None, "rd") in spans[4]
        for _ in range(3):
            model.undo()
        assert not model.modified


NVCC = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"


def _have_cuobjdump() -> bool:
    from eoss.binary.cuobjdump import cuobjdump_path
    try:
        cuobjdump_path()
        return True
    except RuntimeError:
        return False


@pytest.fixture(scope="module")
def real_cubin(tmp_path_factory):
    """A real SM90 cubin compiled with nvcc (skips without the toolkit)."""
    if not os.path.isfile(NVCC):
        pytest.skip("nvcc not available")
    if not _have_cuobjdump():
        pytest.skip("cuobjdump not available")
    d = tmp_path_factory.mktemp("disasm")
    src = d / "k.cu"
    src.write_text(
        "__global__ void kernel_add(float *out, const float *a) {\n"
        "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    out[i] = a[i] + 1.0f;\n"
        "}\n")
    out = d / "k90.cubin"
    subprocess.run([NVCC, "-arch=sm_90", "-cubin", "-o", str(out), str(src)],
                   check=True, capture_output=True, timeout=120)
    return out


def test_parse_sass_text_clean():
    from eoss.binary.cuobjdump import parse_sass_instructions
    line = ("        /*0010*/                   FADD R2, R5, R2 ;"
            "        /* 0x0000000205027221 */")
    (ins,) = parse_sass_instructions("Function : k\n" + line + "\n")
    assert ins.offset == 0x10
    assert ins.text == "FADD R2, R5, R2"
    assert ins.mnemonic == "FADD"
    assert ins.encoding == 0x0000000205027221


class TestDisassemble:
    def test_dump_matches_buffer(self, real_cubin, spec):
        model = EditorModel(str(real_cubin), spec)
        dump = model.disassemble()
        assert dump
        func, instrs = dump[0]
        assert "kernel_add" in func
        assert instrs and instrs[0].offset == 0
        sec = model.section_for_function(func)
        assert sec is not None
        # cuobjdump's printed low-64-bit encodings match our buffer bytes
        for ins in instrs[:4]:
            off = sec.abs_offset + ins.offset
            assert model.instruction_bytes(off)[:8] == ins.encoding_bytes_le

    def test_edited_buffer_round_trip(self, real_cubin, spec):
        # generator -> editor -> cuobjdump agree on a stamped instruction
        model = EditorModel(str(real_cubin), spec)
        func, _instrs = model.disassemble()[0]
        sec = model.section_for_function(func)
        model.write_generic(sec.abs_offset, "fadd__RRR_RR")
        _func, instrs = model.disassemble()[0]
        assert "FADD" in instrs[0].text
        assert instrs[0].encoding_bytes_le == \
            model.instruction_bytes(sec.abs_offset)[:8]

    def test_bad_binary_raises(self, cubin, spec):
        if not _have_cuobjdump():
            pytest.skip("cuobjdump not available")
        path, _n = cubin
        with pytest.raises(RuntimeError):
            EditorModel(str(path), spec).disassemble()


class TestElfView:
    def test_scan_images(self, cubin):
        path, _n = cubin
        data = path.read_bytes()
        images = elfview.scan_images(data)
        assert len(images) == 1
        image = images[0]
        assert image.is_cuda
        names = [s.name for s in image.sections]
        assert ".text" in names and ".shstrtab" in names

    def test_header_fields(self, cubin):
        path, _n = cubin
        image = elfview.scan_images(path.read_bytes())[0]
        assert image.header_values["e_machine"] == 190
        assert image.header_values["e_shnum"] == 3


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


class TestMeasureSession:
    """Session logic that needs no GPU: config, progress pump, CSV export."""

    @pytest.fixture()
    def session(self, cubin, spec):
        from eoss.tui.measure import MeasureSession
        path, _n = cubin
        return MeasureSession(EditorModel(str(path), spec))

    def test_defaults(self, session):
        assert session.mode == "issue"
        assert session.source == "cursor"
        assert not session.busy()
        assert not session.ready  # nothing discovered before refresh()

    def test_pump_collects_progress_and_rows(self, session):
        from eoss.tui.measure import Job, MeasureRow
        job = Job(kind="measure", total=2)
        session.job = job
        job.q.put(("progress", 1, 2, "IMAD"))
        job.q.put(("row", MeasureRow(trigger="IMAD", reset="NOP",
                                     measure="FADD", mode="issue", n=100,
                                     diff_mean=3.0)))
        job.q.put(("done", None))
        session.pump()
        assert (job.done, job.total) == (1, 2)
        assert [r.trigger for r in session.rows] == ["IMAD"]
        assert job.finished

    def test_pump_reports_error(self, session):
        from eoss.tui.measure import Job
        job = Job(kind="measure")
        session.job = job
        job.q.put(("error", "boom"))
        session.pump()
        assert job.finished and job.error == "boom"

    def test_export_csv_skips_errors(self, session, tmp_path):
        from eoss.tui.measure import MeasureRow
        session.sm = "120"
        session.rows = [
            MeasureRow("IMAD", "NOP", "FADD", "issue", n=1000, hot_mean=10.0,
                       cold_mean=4.0, diff_mean=6.0, diff_median=6,
                       diff_min=2, diff_max=9),
            MeasureRow("BADOP", "NOP", "FADD", "issue", error="fault"),
        ]
        out = tmp_path / "r.csv"
        assert session.export_csv(str(out)) == 1  # error row dropped
        text = out.read_text()
        assert text.splitlines()[0].startswith("sm_version,mode,pattern")
        assert "IMAD" in text and "BADOP" not in text

    def test_measure_row_magnitude_sort_key(self):
        from eoss.tui.measure import MeasureRow
        rows = [MeasureRow("a", "", "", "issue", diff_mean=-9.0),
                MeasureRow("b", "", "", "issue", diff_mean=2.0)]
        rows.sort(key=lambda r: -r.magnitude)
        assert [r.trigger for r in rows] == ["a", "b"]


@pytest.mark.skipif(not _have_nvcc_gpu(), reason="nvcc + GPU required")
class TestMeasureGPU:
    def test_build_then_cursor_measure(self, tmp_path):
        # end to end: compile a slot harness, then measure the instruction
        # under the editor cursor as TRIGGER via the session worker thread.
        from eoss.gpu import detect_sm
        from eoss.runtime.harness import build_harness, extract_offsets
        from eoss.tui.measure import MeasureSession

        sm = detect_sm()
        if not paths.isa_json_path(sm).is_file():
            pytest.skip(f"SM{sm} ISA JSON not available")
        harness = build_harness(sm, num_slots=1024, out_dir=tmp_path,
                                verbose=False)
        offsets = extract_offsets(harness, output=tmp_path / "off.json",
                                  verbose=False)
        assert offsets.num_slots == 1024

        spec = ISASpec.load(sm=sm)
        model = EditorModel(str(harness), spec)
        session = MeasureSession(model)
        # wire discovered state to the freshly built (out-of-tree) artifacts
        session.sm, session.gpu_ok = sm, True
        session.harness_path = harness
        session.offsets_path = tmp_path / "off.json"
        assert session.ready

        off = model.sections[0].abs_offset
        cur = model.instruction_bytes(off)
        assert session.start_measure(cur, "cursor test") is None
        session.job.thread.join(timeout=120)
        session.pump()
        assert session.job.finished and not session.job.error
        assert len(session.rows) == 1
        row = session.rows[0]
        assert row.error is None
        assert row.n > 0  # got real per-slot timing samples
