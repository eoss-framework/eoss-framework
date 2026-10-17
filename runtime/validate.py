"""Runtime validation: execute each candidate encoding in isolation on the
GPU and discard any that crash the CUDA context or produce insane timings.

Absorbed from OSIRIS/validate_instructions.py onto the shared patching
infrastructure (MERGE_VALIDATE control policy, unified offsets schema).
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ..enc import control as ctrl
from ..schemas import OffsetsDoc, load_encodings, save_encodings

TIMING_MIN = 1
TIMING_MAX = 300000
MIN_VALID_FRACTION = 0.5


def validate_instructions(binary_path, offsets, instructions_path,
                          output_path, verbose: bool = False,
                          keep_binaries=None, run_timeout: int = 30):
    """Validate each instruction by patching all harness slots with it.

    Returns:
        (working_records, failing_records) — lists of encoding dicts.
    """
    doc = offsets if isinstance(offsets, OffsetsDoc) else OffsetsDoc.load(offsets)
    records = load_encodings(instructions_path)

    keep_dir = Path(keep_binaries) if keep_binaries else None
    if keep_dir:
        keep_dir.mkdir(parents=True, exist_ok=True)

    print(f"Validating {len(records)} instructions...")
    print(f"Binary: {binary_path}")
    print(f"Code base: 0x{doc.code_base_offset:x}")
    print(f"Slots: {doc.num_slots}")
    if keep_dir:
        print(f"Keeping patched binaries in: {keep_dir}")
    print()

    patch_anchors = ["reset", "trigger", "measure"]

    with open(binary_path, "rb") as f:
        base_data = bytes(f.read())

    working, failing = [], []
    tmp_dir = Path(tempfile.mkdtemp(prefix="eoss_validate_"))

    try:
        for i, rec in enumerate(records):
            inst_name = rec.instruction or f"inst_{i}"
            inst_bytes = rec.bytes_le

            data = bytearray(base_data)
            for slot in doc.slots:
                for anchor in patch_anchors:
                    if not slot.has(anchor):
                        continue
                    offset = doc.file_offset(slot, anchor)
                    if offset + 16 <= len(data):
                        ctrl.patch_at(data, offset, inst_bytes,
                                      ctrl.PatchPolicy.MERGE_VALIDATE)

            temp_path = tmp_dir / f"validate_inst_{i}"
            with open(temp_path, "wb") as f:
                f.write(data)
            temp_path.chmod(0o755)

            try:
                result = subprocess.run(
                    [str(temp_path)], capture_output=True, text=True,
                    timeout=run_timeout,
                )
                is_valid, fail_reason = _check_run(result)
            except subprocess.TimeoutExpired:
                is_valid, fail_reason = False, "TIMEOUT"
            except Exception as e:  # noqa: BLE001
                is_valid, fail_reason = False, f"ERROR: {e}"

            rec_dict = rec.to_json()
            if is_valid:
                working.append(rec_dict)
                status = "OK"
            else:
                failing.append(rec_dict)
                status = f"FAIL ({fail_reason})"

            if keep_dir:
                safe_name = inst_name.replace("/", "_").replace(" ", "_")
                shutil.move(str(temp_path), str(keep_dir / safe_name))
            else:
                temp_path.unlink(missing_ok=True)

            if verbose:
                print(f"  [{i + 1}/{len(records)}] {inst_name[:40]}: {status}")
            elif (i + 1) % 50 == 0:
                print(f"  Tested {i + 1}/{len(records)}: "
                      f"{len(working)} working, {len(failing)} failing")
                sys.stdout.flush()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\nValidation complete:")
    print(f"  Working: {len(working)}")
    print(f"  Failing: {len(failing)}")

    with open(output_path, "w") as f:
        import json
        json.dump(working, f, indent=2)
    print(f"\nSaved {len(working)} validated instructions to {output_path}")

    if failing and verbose:
        print(f"\nFailing instructions (first 20):")
        for rec in failing[:20]:
            print(f"  {rec.get('instruction', '?')}")
        if len(failing) > 20:
            print(f"  ... and {len(failing) - 20} more")

    return working, failing


def _check_run(result: subprocess.CompletedProcess):
    """Interpret a validation run: (is_valid, fail_reason)."""
    if result.returncode != 0:
        return False, f"rc={result.returncode}"
    if "CUDA Error" in result.stdout or "CUDA Error" in result.stderr:
        return False, "CUDA error"
    if "PERSLOT_TIMING_START" not in result.stdout:
        return False, "no timing output"
    if "SUCCESS" not in result.stdout:
        return False, "no SUCCESS marker"

    valid_timings = 0
    bad_timings = 0
    for line in result.stdout.split("\n"):
        if "," in line and not line.startswith("PERSLOT"):
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    timing = int(parts[1])
                    if len(parts) >= 3:
                        timing = max(timing, int(parts[2]))
                    if TIMING_MIN <= timing <= TIMING_MAX:
                        valid_timings += 1
                    else:
                        bad_timings += 1
                except ValueError:
                    pass

    total = valid_timings + bad_timings
    if total > 0 and valid_timings / total >= MIN_VALID_FRACTION:
        return True, ""
    return False, f"bad timing ({valid_timings}/{total} valid)"
