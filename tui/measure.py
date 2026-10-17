"""Measurement-bench session for the TUI.

Bridges the editor to the OSIRIS slot-harness campaign engine
(:mod:`eoss.runtime.measure`): discover or build a compiled slot harness for
the live GPU, then run either a single measurement of the instruction under
the cursor (as TRIGGER) or a full sweep over the ISA, streaming per-triple
timing differentials back to the UI.

Everything GPU/nvcc-touching runs on a background thread and reports through
a queue, so the synchronous curses loop stays responsive; the UI thread only
snapshots config, drains progress, and appends result rows.
"""

import csv
import queue
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .. import paths
from ..enc.control import NOP_BYTES
from ..schemas import EncodingRecord

#: harness slot counts the packaged osiris_slots.cu accepts (see its #error)
SLOT_CHOICES = (1024, 2048, 4096)
#: default cap on triggers per sweep — surfaced in the UI, never silent
DEFAULT_SWEEP_LIMIT = 64
MODES = ("issue", "completion")
PATTERNS = ("diagonal", "sweep", "osiris")

# synthetic instr_by_name entries for the single-cursor measurement
CURSOR = "__CURSOR__"
RESET = "__RESET__"
PROBE = "__PROBE__"


@dataclass
class MeasureRow:
    """One completed (RESET, TRIGGER, MEASURE) measurement."""
    trigger: str
    reset: str
    measure: str
    mode: str
    n: int = 0
    hot_mean: float = 0.0
    cold_mean: float = 0.0
    diff_mean: float = 0.0
    diff_median: int = 0
    diff_min: int = 0
    diff_max: int = 0
    error: Optional[str] = None

    @property
    def magnitude(self) -> float:
        return abs(self.diff_mean)


@dataclass
class Job:
    """A running background task (build or measure)."""
    kind: str                       # "build" | "measure"
    total: int = 0
    done: int = 0
    label: str = ""
    error: Optional[str] = None
    finished: bool = False
    q: "queue.Queue" = field(default_factory=queue.Queue)
    cancel: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None


class MeasureSession:
    """Discovery, in-TUI build, and threaded measurement for one editor."""

    def __init__(self, model):
        self.model = model
        self.job: Optional[Job] = None
        self.rows: List[MeasureRow] = []

        # discovered readiness (filled by refresh)
        self.sm: Optional[str] = None
        self.gpu_label: str = ""
        self.gpu_ok = False
        self.harness_path: Optional[Path] = None
        self.offsets_path: Optional[Path] = None
        self.super_path: Optional[Path] = None
        self.super_offsets_path: Optional[Path] = None
        self.tma_path: Optional[Path] = None
        self.tma_offsets_path: Optional[Path] = None
        self.cga_path: Optional[Path] = None
        self.cga_offsets_path: Optional[Path] = None
        self.db_path: Optional[Path] = None
        self._refreshed = False

        # config
        self.mode = "issue"
        self.trials = 1
        self.slots = 1024
        self.source = "cursor"       # "cursor" | "sweep"
        self.pattern = "diagonal"
        self.sweep_limit = DEFAULT_SWEEP_LIMIT
        self.last_status = ""

    # -- readiness -----------------------------------------------------------

    def refresh(self) -> None:
        """(Re)discover GPU + on-disk harness/offsets/instruction DB."""
        self._refreshed = True
        try:
            from ..gpu import detect_sm, gpu_name
            self.sm = detect_sm()
            self.gpu_label = gpu_name()
            self.gpu_ok = True
        except Exception as exc:
            self.sm = self.sm or self.model.guess_sm()
            self.gpu_label = f"no GPU ({exc})"
            self.gpu_ok = False

        self.harness_path = self.offsets_path = self.db_path = None
        self.super_path = self.super_offsets_path = None
        if not self.sm:
            return
        d = paths.generated_dir(self.sm)
        harness = d / f"harness_sm{self.sm}"
        offsets = d / f"offsets_sm{self.sm}.json"
        if harness.is_file():
            self.harness_path = harness
        if offsets.is_file():
            self.offsets_path = offsets
        # optional richer harness (texture/surface/shared) for the union DB
        super_h = d / f"harness_super_sm{self.sm}"
        super_o = d / f"offsets_super_sm{self.sm}.json"
        if super_h.is_file():
            self.super_path = super_h
        if super_o.is_file():
            self.super_offsets_path = super_o
        # optional TMA harness (Tensor Memory Accelerator descriptor ops)
        tma_h = d / f"harness_tma_sm{self.sm}"
        tma_o = d / f"offsets_tma_sm{self.sm}.json"
        self.tma_path = tma_h if tma_h.is_file() else None
        self.tma_offsets_path = tma_o if tma_o.is_file() else None
        # optional CGA harness (thread-block cluster barrier ops)
        cga_h = d / f"harness_cga_sm{self.sm}"
        cga_o = d / f"offsets_cga_sm{self.sm}.json"
        self.cga_path = cga_h if cga_h.is_file() else None
        self.cga_offsets_path = cga_o if cga_o.is_file() else None
        for cand in (d / "validated_gpr.json", d / "all_instructions.json"):
            if cand.is_file():
                self.db_path = cand
                break

    @property
    def ready(self) -> bool:
        """Enough to run a measurement (GPU + harness + offsets)."""
        return bool(self.gpu_ok and self.harness_path and self.offsets_path)

    @property
    def can_build(self) -> bool:
        from ..runtime.harness import nvcc_path
        try:
            nvcc_path()
            return bool(self.sm)
        except Exception:
            return False

    def busy(self) -> bool:
        return self.job is not None and not self.job.finished

    # -- progress pump (UI thread) ------------------------------------------

    def pump(self) -> None:
        """Drain the active job's queue into session state. UI-thread only."""
        job = self.job
        if job is None:
            return
        try:
            while True:
                ev = job.q.get_nowait()
                kind = ev[0]
                if kind == "status":
                    job.label = ev[1]
                    self.last_status = ev[1]
                elif kind == "progress":
                    job.done, job.total, job.label = ev[1], ev[2], ev[3]
                elif kind == "row":
                    self.rows.append(ev[1])
                elif kind == "error":
                    job.error = ev[1]
                    job.finished = True
                elif kind == "done":
                    job.finished = True
        except queue.Empty:
            pass
        if job.finished and job.thread is not None and not job.thread.is_alive():
            if job.kind in ("build", "validate"):
                self.refresh()  # discover the freshly built harness / DB
            job.thread = None

    def cancel(self) -> None:
        if self.job and not self.job.finished:
            self.job.cancel.set()
            self.last_status = "cancelling after current measurement…"

    # -- build ---------------------------------------------------------------

    def start_build(self) -> Optional[str]:
        if self.busy():
            return "a job is already running"
        if not self.can_build:
            return "nvcc not available"
        job = Job(kind="build")  # total=0 → shown as a status line, not a bar
        self.job = job
        job.thread = threading.Thread(target=self._build_worker, args=(job,),
                                      daemon=True)
        job.thread.start()
        return None

    def _build_worker(self, job: Job) -> None:
        try:
            from ..runtime.harness import (build_harness, build_harness_flavor,
                                           extract_offsets)
            from ..isa.registry import harness_flavors
            sm = self.sm
            out = paths.generated_dir(sm)
            out.mkdir(parents=True, exist_ok=True)
            # classic global-only base harness (every generation)
            job.q.put(("status", f"nvcc harness_sm{sm} (classic, {self.slots} slots)…"))
            binary = build_harness(sm, num_slots=self.slots, out_dir=out,
                                   verbose=False)
            if job.cancel.is_set():
                job.q.put(("error", "cancelled")); return
            extract_offsets(binary, output=out / f"offsets_sm{sm}.json",
                            verbose=False)
            # feature flavors this generation supports, each assembled from the
            # one parametrized template (harnesses/osiris_harness.cu).
            flavors = harness_flavors(sm)
            built = []
            for name, feats in flavors.items():
                if job.cancel.is_set():
                    break
                job.q.put(("status",
                           f"nvcc harness_{name}_sm{sm} ({'+'.join(feats)})…"))
                try:
                    fb = build_harness_flavor(sm, name, num_slots=self.slots,
                                              out_dir=out, verbose=False)
                except Exception as exc:  # feature unsupported on this gen
                    job.q.put(("status", f"skip {name}: {str(exc)[:40]}"))
                    continue
                extract_offsets(fb, output=out / f"offsets_{name}_sm{sm}.json",
                                kernel_hint="osiris_kernel", verbose=False)
                built.append(name)
            job.q.put(("status", f"built classic + {len(built)} flavor(s): "
                                 f"{', '.join(built) or '(none)'}"))
            job.q.put(("done", None))
        except Exception as exc:
            job.q.put(("error", str(exc)))

    # -- measure -------------------------------------------------------------

    def start_measure(self, cursor_bytes: Optional[bytes],
                      cursor_label: str) -> Optional[str]:
        if self.busy():
            return "a job is already running"
        if not self.ready:
            return "not ready — build a harness first"
        if self.source == "cursor" and cursor_bytes is None:
            return "no instruction under the cursor (tab 1)"
        self.rows = []
        snap = dict(mode=self.mode, trials=self.trials, source=self.source,
                    pattern=self.pattern, limit=self.sweep_limit,
                    harness=self.harness_path, offsets=self.offsets_path,
                    db=self.db_path, sm=self.sm,
                    cursor_bytes=cursor_bytes, cursor_label=cursor_label)
        job = Job(kind="measure")
        self.job = job
        job.thread = threading.Thread(target=self._measure_worker,
                                      args=(job, snap), daemon=True)
        job.thread.start()
        return None

    def _generic_records(self, spec, encoder) -> List[EncodingRecord]:
        """A generic (all-RZ, zero-immediate) encoding of every ISA class."""
        records = []
        for name in spec.instructions:
            try:
                records.append(EncodingRecord(
                    instruction=name, bytes_le=encoder.build(name).bytes))
            except Exception:
                continue
        return records

    def _probe_records(self, snap: dict):
        """(records, encoder) for the sweep: a discovered DB if present,
        else a generic encoding of every ISA class."""
        from ..schemas import load_encodings
        from ..enc.generator import Encoder
        from ..isa.model import ISASpec
        spec = ISASpec.load(sm=snap["sm"])
        encoder = Encoder(spec)
        if snap["db"] is not None:
            return load_encodings(snap["db"]), encoder
        return self._generic_records(spec, encoder), encoder

    # -- validated instruction DB (generate -> runtime-filter) ---------------

    def start_build_db(self) -> Optional[str]:
        if self.busy():
            return "a job is already running"
        if not self.ready:
            return "build a harness first"
        job = Job(kind="validate")
        self.job = job
        job.thread = threading.Thread(target=self._validate_worker,
                                      args=(job,), daemon=True)
        job.thread.start()
        return None

    def _union_stages(self) -> List[dict]:
        """Validation stages for the union DB: the classic base harness, then
        each feature flavor this generation supports (template-built), then a
        byte-templating harvest. Flavors and their register maps come straight
        from the discovered per-generation harnesses."""
        from ..schemas import OffsetsDoc
        from ..isa.registry import harness_flavors
        d = paths.generated_dir(self.sm)
        stages = [{"label": "classic", "mode": "generate", "register_map": None,
                   "binary": self.harness_path, "offsets": self.offsets_path}]
        super_offsets = None
        for name in harness_flavors(self.sm):
            hp = d / f"harness_{name}_sm{self.sm}"
            op = d / f"offsets_{name}_sm{self.sm}.json"
            if not (hp.is_file() and op.is_file()):
                continue
            if name == "super":
                super_offsets = op
            # cluster-barrier ops are operand-free — no register map needed
            rmap = (None if name == "cga"
                    else OffsetsDoc.load(op).extra.get("register_map"))
            stages.append({"label": name, "mode": "generate",
                           "register_map": rmap, "binary": hp, "offsets": op})
        stages.append({"label": "harvest", "mode": "harvest",
                       "kernel_hint": None, "binary": self.harness_path,
                       "offsets": self.offsets_path})
        if super_offsets is not None:
            stages.append({"label": "harvest+", "mode": "harvest",
                           "kernel_hint": "osiris_kernel",
                           "binary": d / f"harness_super_sm{self.sm}",
                           "offsets": super_offsets})
        return stages

    def _validate_worker(self, job: Job) -> None:
        """Build the union validated DB across the classic + super harnesses
        (generator policy per harness) plus a byte-templating harvest, and
        persist it as validated_gpr.json. Per-instruction progress + cancel;
        a cancelled build still saves what was cleared so far."""
        try:
            from ..isa.model import ISASpec
            from ..runtime.union import build_union_db
            spec = ISASpec.load(sm=self.sm)
            out = paths.generated_dir(self.sm) / "validated_gpr.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            stages = self._union_stages()

            def on_item(label, name, done, total):
                job.q.put(("progress", done, total, f"{label}: {name[:18]}"))

            records, summary = build_union_db(
                spec, stages, output_path=out,
                on_item=on_item, should_cancel=job.cancel.is_set)

            n = len(spec)
            parts = " ".join(f"{lbl}+{cnt}" for lbl, cnt in summary)
            partial = " (partial)" if job.cancel.is_set() else ""
            job.q.put(("progress", n, n, "done"))
            job.q.put(("status", f"union DB{partial}: {len(records)}/{n} clean "
                                 f"[{parts}] → {out.name}"))
            job.q.put(("done", None))
        except Exception as exc:
            job.q.put(("error", str(exc)))

    def _measure_worker(self, job: Job, snap: dict) -> None:
        try:
            tmp = Path(tempfile.mkdtemp(prefix="eoss_measure_"))
            try:
                if snap["source"] == "cursor":
                    self._run_cursor(job, snap, tmp)
                else:
                    self._run_sweep(job, snap, tmp)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            if not job.error:
                job.q.put(("done", None))
        except Exception as exc:
            job.q.put(("error", str(exc)))

    def _make_harness(self, snap: dict):
        from ..runtime.measure import SlotHarness
        return SlotHarness(snap["harness"], snap["offsets"], verbose=False)

    def _generic_probe(self, encoder, spec) -> Tuple[bytes, str]:
        """A generic FADD (readable ALU probe); NOP if none encodes."""
        dec = self.model.decoder
        for name, ic in spec.instructions.items():
            if dec.mnemonic_for(ic, ic.opcode) == "FADD":
                try:
                    return encoder.build(name).bytes, name
                except Exception:
                    continue
        return NOP_BYTES, "NOP"

    def _run_cursor(self, job: Job, snap: dict, tmp: Path) -> None:
        from ..enc.generator import Encoder
        from ..isa.model import ISASpec
        spec = ISASpec.load(sm=snap["sm"])
        probe_bytes, probe_name = self._generic_probe(Encoder(spec), spec)
        harness = self._make_harness(snap)
        harness.instr_by_name[RESET] = NOP_BYTES
        harness.instr_by_name[PROBE] = probe_bytes
        harness.instr_by_name[CURSOR] = snap["cursor_bytes"]
        job.q.put(("progress", 0, 1, snap["cursor_label"]))
        row = self._measure_one(harness, snap["mode"], snap["trials"],
                                RESET, CURSOR, PROBE, tmp,
                                trigger_label=snap["cursor_label"],
                                measure_label=f"FADD ({probe_name})")
        job.q.put(("row", row))
        job.q.put(("progress", 1, 1, "done"))
        if row.error:
            job.q.put(("status", f"cursor: {row.error}"))
        else:
            job.q.put(("status",
                       f"cursor {row.trigger}: diff {row.diff_mean:+.1f} "
                       f"(n={row.n})"))

    def _run_sweep(self, job: Job, snap: dict, tmp: Path) -> None:
        from ..runtime.campaign import generate_triples
        job.q.put(("status", "preparing instruction database…"))
        records, _enc = self._probe_records(snap)
        if not records:
            job.q.put(("error", "no encodable instructions for the sweep"))
            return
        by_name = {r.instruction: r.bytes_le for r in records}
        triples = generate_triples(records, snap["pattern"])
        if snap["limit"]:
            triples = triples[:snap["limit"]]
        harness = self._make_harness(snap)
        harness.instr_by_name.update(by_name)
        total = len(triples)
        measured = ok = 0
        for i, (r, t, m) in enumerate(triples):
            if job.cancel.is_set():
                job.q.put(("status", f"cancelled after {i}/{total} "
                                     f"({ok} ran clean)"))
                return
            job.q.put(("progress", i, total, t))
            row = self._measure_one(harness, snap["mode"], snap["trials"],
                                    r, t, m, tmp)
            measured += 1
            ok += row.error is None
            job.q.put(("row", row))
        job.q.put(("progress", total, total, "done"))
        job.q.put(("status", f"sweep done — {measured} measured, "
                             f"{ok} ran clean"))

    def _measure_one(self, harness, mode: str, trials: int,
                     reset: str, trigger: str, measure: str, tmp: Path,
                     trigger_label: Optional[str] = None,
                     measure_label: Optional[str] = None) -> MeasureRow:
        row = MeasureRow(trigger=trigger_label or trigger, reset=reset,
                         measure=measure_label or measure, mode=mode)
        try:
            res = harness.measure_triple(mode, reset, trigger, measure,
                                         trials=trials, tmp_dir=tmp)
        except Exception as exc:
            row.error = str(exc)[:60]
            return row
        s = res.get("summary")
        if not s:
            row.error = "no timing samples"
            return row
        row.n = s["n_samples"]
        row.hot_mean = s["hot_mean"]
        row.cold_mean = s["cold_mean"]
        row.diff_mean = s["diff_mean"]
        row.diff_median = s["diff_median"]
        row.diff_min = s["diff_min"]
        row.diff_max = s["diff_max"]
        return row

    # -- export --------------------------------------------------------------

    def export_csv(self, path) -> int:
        """Write the current result rows to a run_campaign-compatible summary
        CSV. Returns the number of rows written."""
        rows = [r for r in self.rows if r.error is None]
        header = ["sm_version", "mode", "pattern", "reset", "trigger",
                  "measure", "n_samples", "diff_mean", "diff_median",
                  "diff_min", "diff_max", "hot_mean", "cold_mean"]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in rows:
                w.writerow([self.sm, r.mode, self.pattern, r.reset, r.trigger,
                            r.measure, r.n, f"{r.diff_mean:.3f}", r.diff_median,
                            r.diff_min, r.diff_max, f"{r.hot_mean:.3f}",
                            f"{r.cold_mean:.3f}"])
        return len(rows)
