"""Union instruction-DB builder across specialized harnesses.

No single harness keeps every resource live at the measure slot (GPR
pressure trades global memory for texture/surface/shared — see
the ``super`` flavor of ``harnesses/osiris_harness.cu``). Maximum coverage is
*union* of what validates on each harness, each with the operand policy it
needs (a register map for the super harness, default for the classic one).

Two kinds of stage feed the union:

* ``generate`` — build a generic encoding per class with the stage's operand
  policy (``eoss.enc.generator.Encoder`` + optional register map) and keep the
  ones that run clean.
* ``harvest`` (byte-templating) — some classes cannot be encoded field-by-field
  (an ISA-JSON-unmodeled bit in Blackwell descriptor loads). The harness itself
  contains *working* exemplars of those ops; harvest lifts their exact bytes,
  decodes them to a class, and validates them at the measure slot.

A class enters the DB on the first stage where it runs clean; later stages only
see the classes still missing.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..enc import control as ctrl
from ..enc.opcode import extract_opcode
from ..schemas import EncodingRecord, OffsetsDoc, save_encodings
from .harness import run_harness
from .validate import _check_run

# Data-movement mnemonics worth harvesting as byte-templates. Includes the
# TMA family: the generator's field policy can't reproduce a runnable UTMALDG
# on every generation (an ISA-JSON-unmodeled bit — e.g. sm_90), but the
# harness's own UTMALDG validates, so byte-templating lifts it directly.
HARVEST_MNEMONICS = ("LDG", "STG", "LDS", "STS", "LD", "ST", "LDL", "STL",
                     "LDC", "LDCU", "ATOM", "ATOMG", "ATOMS", "RED", "REDG",
                     "LDGSTS", "LDSM", "STSM",
                     "UTMALDG", "UTMASTG", "UTMAPF", "UTMAREDG", "UBLKCP")
_PATCH_ANCHORS = ("reset", "trigger", "measure")


def _validate_bytes(base: bytes, doc: OffsetsDoc, inst_bytes: bytes,
                    run_timeout: int = 15) -> bool:
    """Patch ``inst_bytes`` into every slot's reset/trigger/measure anchor and
    return True if the harness still runs clean (the ``eoss validate`` check)."""
    data = bytearray(base)
    for slot in doc.slots:
        for anchor in _PATCH_ANCHORS:
            if slot.has(anchor):
                off = doc.file_offset(slot, anchor)
                if off + 16 <= len(data):
                    ctrl.patch_at(data, off, inst_bytes, ctrl.PatchPolicy.MERGE_VALIDATE)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(bytes(data))
        tmp = f.name
    Path(tmp).chmod(0o755)
    try:
        ok, _ = _check_run(run_harness(tmp, timeout=run_timeout))
    except Exception:
        ok = False
    finally:
        Path(tmp).unlink(missing_ok=True)
    return ok


def harvest_exemplars(binary_path, offsets, spec, kernel_hint=None,
                      only: Optional[set] = None,
                      run_timeout: int = 15) -> Dict[str, bytes]:
    """Byte-templating: lift working data-movement instructions straight out of
    a compiled harness and return ``{class_name: bytes}`` for the ones that
    validate at the measure slot.

    Args:
        binary_path: compiled harness binary.
        offsets: its OffsetsDoc (or path).
        spec: ISASpec for decoding harvested bytes to a class name.
        kernel_hint: measurement-kernel substring.
        only: if given, keep only exemplars whose class is in this set.
    """
    from ..binary.cuobjdump import find_kernel, get_sass_dump, parse_sass_instructions
    from ..tui.decode import Decoder  # pure ISA decoder (no curses)

    doc = offsets if isinstance(offsets, OffsetsDoc) else OffsetsDoc.load(offsets)
    base = Path(binary_path).read_bytes()
    decoder = Decoder(spec)

    sass = get_sass_dump(str(binary_path))
    kernel = find_kernel(sass, kernel_hint)
    seen: Dict[str, bytes] = {}
    for ins in parse_sass_instructions(sass, kernel):
        if not any(ins.text.startswith(m) for m in HARVEST_MNEMONICS):
            continue
        file_off = doc.code_base_offset + ins.offset
        if file_off + 16 > len(base):
            continue
        raw = bytes(base[file_off:file_off + 16])
        if extract_opcode(raw) not in spec.defined_opcodes():
            continue
        decoded = decoder.decode(0, raw)
        name = decoded.class_name
        if name is None or name in seen:
            continue
        if only is not None and name not in only:
            continue
        if _validate_bytes(base, doc, raw, run_timeout):
            seen[name] = raw
    return seen


def build_union_db(spec, stages: List[dict], output_path,
                   seed_records: Optional[List[EncodingRecord]] = None,
                   run_timeout: int = 15,
                   on_progress: Optional[Callable[[str, int, int], None]] = None,
                   on_item: Optional[Callable[[str, str, int, int], None]] = None,
                   should_cancel: Optional[Callable[[], bool]] = None,
                   classes: Optional[set] = None
                   ) -> "tuple[List[EncodingRecord], List[tuple]]":
    """Validate each ISA class across ``stages`` and union the clean results.

    Each stage is a dict with ``label``, ``binary``, ``offsets``, and either
    ``mode='generate'`` (optional ``register_map``) or ``mode='harvest'``
    (optional ``kernel_hint``). A class enters the DB on the first stage where
    it runs clean; later stages only process the classes still missing.

    ``on_item(label, name, done, total)`` fires per generate-stage instruction;
    ``should_cancel()`` is polled per instruction — on cancel the DB built so
    far is still written. Returns ``(records, summary)``.

    Returns ``(records, summary)`` where summary is ``[(label, n_new), ...]``.
    """
    from ..enc.generator import Encoder

    records: List[EncodingRecord] = list(seed_records or [])
    placed = {r.instruction for r in records}
    remaining = [n for n in spec.instructions
                 if n not in placed and (classes is None or n in classes)]
    summary = []
    cancelled = False

    for stage in stages:
        if cancelled:
            break
        base = Path(stage["binary"]).read_bytes()
        doc = (stage["offsets"] if isinstance(stage["offsets"], OffsetsDoc)
               else OffsetsDoc.load(stage["offsets"]))
        newly: List[str] = []

        if stage.get("mode") == "harvest":
            found = harvest_exemplars(stage["binary"], doc, spec,
                                      kernel_hint=stage.get("kernel_hint"),
                                      only=set(remaining), run_timeout=run_timeout)
            for name, raw in found.items():
                records.append(EncodingRecord(instruction=name, bytes_le=raw,
                                              pipeline=_pipeline(spec, name)))
                newly.append(name)
        else:
            enc = Encoder(spec, register_map=stage.get("register_map"))
            total = len(remaining)
            for i, name in enumerate(remaining):
                if should_cancel is not None and should_cancel():
                    cancelled = True
                    break
                if on_item is not None:
                    on_item(stage["label"], name, i, total)
                try:
                    b = enc.build(name).bytes
                except Exception:
                    continue
                if _validate_bytes(base, doc, b, run_timeout):
                    records.append(EncodingRecord(instruction=name, bytes_le=b,
                                                  pipeline=_pipeline(spec, name)))
                    newly.append(name)

        newly_set = set(newly)
        remaining = [n for n in remaining if n not in newly_set]
        summary.append((stage["label"], len(newly)))
        if on_progress:
            on_progress(stage["label"], len(newly), len(remaining))

    if output_path:
        save_encodings(records, output_path)
    return records, summary


def _pipeline(spec, name: str) -> Optional[str]:
    ic = spec.get(name)
    return ic.pipeline if ic else None
