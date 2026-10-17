"""Slot-harness patch & measurement engine (OSIRIS methodology).

Patches (RESET, TRIGGER, MEASURE) instruction triples into the slots of a
compiled harness and measures per-slot timing differentials (hot vs cold,
where the TRIGGER is physically replaced by a NOP).

Two measurement modes:

* ``issue`` — no scoreboard dependencies; the differential reflects
  microarchitectural side effects of TRIGGER on MEASURE's dispatch timing.
* ``completion`` — a scoreboard chain (TRIGGER ->SB4, CS2R_START waits SB4,
  MEASURE ->SB5, CS2R_END waits SB5) so the window captures full
  completion latency.

Absorbed from OSIRIS/patch_and_measure.py onto the unified
``eoss.offsets/v2`` schema, :class:`eoss.binary.BinaryPatcher` mechanics
and the canonical control policies in ``eoss.enc.control``.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..enc import control as ctrl
from ..schemas import EncodingRecord, OffsetsDoc, Slot, load_encodings
from .harness import parse_perslot_timing, run_harness

# Scoreboard indices for the completion-mode dependency chain
# (SB4/SB5 avoid the compiler's own SB0-3 usage).
FENCE_SB_TRIGGER = 4
FENCE_SB_MEASURE = 5


class SlotHarness:
    """A compiled slot harness plus an optional instruction database."""

    def __init__(self, binary_path, offsets, instructions=None,
                 verbose: bool = False):
        """
        Args:
            binary_path: compiled harness binary.
            offsets: OffsetsDoc or path to an offsets JSON (schema v2).
            instructions: optional path to an encoding JSON, or a list of
                EncodingRecord, for name -> bytes lookup.
            verbose: verbose output.
        """
        self.binary_path = Path(binary_path).resolve()
        self.offsets = offsets if isinstance(offsets, OffsetsDoc) else OffsetsDoc.load(offsets)
        self.verbose = verbose

        if self.offsets.harness != "osiris_slots":
            raise ValueError(
                f"SlotHarness requires the osiris_slots grammar, "
                f"got harness={self.offsets.harness!r}")

        self.instr_by_name: Dict[str, bytes] = {}
        if instructions is not None:
            if isinstance(instructions, (str, Path)):
                records = load_encodings(instructions)
            else:
                records = instructions
            for rec in records:
                if isinstance(rec, EncodingRecord):
                    self.instr_by_name[rec.instruction] = rec.bytes_le
                elif isinstance(rec, dict):
                    name = rec.get("instruction", rec.get("opcode_key", ""))
                    self.instr_by_name[name] = bytes.fromhex(rec["bytes_le"])

        if self.verbose:
            print(f"Loaded {self.offsets.num_slots} slots; "
                  f"kernel={self.offsets.kernel_name}; "
                  f"instructions={len(self.instr_by_name)}")

    @property
    def slots(self) -> List[Slot]:
        return self.offsets.slots

    def file_offset(self, slot: Slot, anchor: str) -> int:
        return self.offsets.file_offset(slot, anchor)

    def get_instruction_bytes(self, name: str) -> bytes:
        if name not in self.instr_by_name:
            raise KeyError(f"Unknown instruction: {name}")
        return self.instr_by_name[name]

    # -- slot patching -------------------------------------------------------

    def patch_slot_issue_mode(self, data: bytearray, slot: Slot,
                              reset_bytes: bytes, trigger_bytes: bytes,
                              measure_bytes: bytes, cold: bool = False) -> None:
        """Issue mode: no scoreboard dependencies; delay normalized to 4."""
        reset_off = self.file_offset(slot, "reset")
        trigger_off = self.file_offset(slot, "trigger")
        measure_off = self.file_offset(slot, "measure")

        ctrl.patch_at(data, reset_off, reset_bytes, ctrl.PatchPolicy.MERGE_MEASURE)
        ctrl.clear_scoreboard_bits(data, reset_off)

        if cold:
            data[trigger_off:trigger_off + 16] = ctrl.NOP_BYTES
        else:
            ctrl.patch_at(data, trigger_off, trigger_bytes,
                          ctrl.PatchPolicy.MERGE_MEASURE)
            ctrl.clear_scoreboard_bits(data, trigger_off)

        ctrl.patch_at(data, measure_off, measure_bytes, ctrl.PatchPolicy.MERGE_MEASURE)
        ctrl.clear_scoreboard_bits(data, measure_off)

        if slot.has("cs2r_start"):
            ctrl.clear_scoreboard_bits(data, self.file_offset(slot, "cs2r_start"))
        if slot.has("cs2r_end"):
            ctrl.clear_scoreboard_bits(data, self.file_offset(slot, "cs2r_end"))

    def patch_slot_completion_mode(self, data: bytearray, slot: Slot,
                                   reset_bytes: bytes, trigger_bytes: bytes,
                                   measure_bytes: bytes, cold: bool = False) -> None:
        """Completion mode: full dependency chain via scoreboard barriers."""
        reset_off = self.file_offset(slot, "reset")
        trigger_off = self.file_offset(slot, "trigger")
        measure_off = self.file_offset(slot, "measure")

        # RESET: patch instruction, keep original scheduling
        ctrl.patch_at(data, reset_off, reset_bytes, ctrl.PatchPolicy.MERGE_MEASURE)

        # TRIGGER: patch (or NOP) + write barrier SB4
        if cold:
            data[trigger_off:trigger_off + 16] = ctrl.NOP_BYTES
        else:
            ctrl.patch_at(data, trigger_off, trigger_bytes,
                          ctrl.PatchPolicy.MERGE_MEASURE)
        ctrl.set_scoreboard_write(data, trigger_off, FENCE_SB_TRIGGER)

        # CS2R_START: wait for TRIGGER
        if slot.has("cs2r_start"):
            ctrl.set_scoreboard_wait(data, self.file_offset(slot, "cs2r_start"),
                                     FENCE_SB_TRIGGER)

        # MEASURE: patch + write barrier SB5
        ctrl.patch_at(data, measure_off, measure_bytes, ctrl.PatchPolicy.MERGE_MEASURE)
        ctrl.set_scoreboard_write(data, measure_off, FENCE_SB_MEASURE)

        # CS2R_END: wait for MEASURE
        if slot.has("cs2r_end"):
            ctrl.set_scoreboard_wait(data, self.file_offset(slot, "cs2r_end"),
                                     FENCE_SB_MEASURE)

    # -- binary creation -----------------------------------------------------

    def _new_binary_data(self) -> bytearray:
        with open(self.binary_path, "rb") as f:
            return bytearray(f.read())

    @staticmethod
    def _write_binary(path: Path, data: bytearray) -> Path:
        with open(path, "wb") as f:
            f.write(data)
        os.chmod(path, 0o755)
        return path

    def create_patched_binary(self, output_path, mode: str,
                              reset_bytes: bytes, trigger_bytes: bytes,
                              measure_bytes: bytes, cold: bool = False,
                              start_slot: int = 1) -> Path:
        """Patch every slot (from ``start_slot``) with the same triple."""
        data = self._new_binary_data()
        patch_fn = (self.patch_slot_issue_mode if mode == "issue"
                    else self.patch_slot_completion_mode)
        for slot in self.slots[start_slot:]:
            patch_fn(data, slot, reset_bytes, trigger_bytes, measure_bytes, cold=cold)
        if self.verbose:
            label = "COLD" if cold else "HOT"
            print(f"  [{label}] Patched {len(self.slots) - start_slot} slots, mode={mode}")
        return self._write_binary(Path(output_path), data)

    def create_multi_patched_binary(self, output_path, mode: str,
                                    slot_assignments: List[Tuple[int, bytes, bytes, bytes]],
                                    cold: bool = False) -> Path:
        """Patch different (RESET, TRIGGER, MEASURE) triples per slot."""
        data = self._new_binary_data()
        patch_fn = (self.patch_slot_issue_mode if mode == "issue"
                    else self.patch_slot_completion_mode)
        for slot_idx, r, t, m in slot_assignments:
            if slot_idx < 0 or slot_idx >= len(self.slots):
                raise IndexError(
                    f"slot index {slot_idx} out of range [0, {len(self.slots)})")
            patch_fn(data, self.slots[slot_idx], r, t, m, cold=cold)
        if self.verbose:
            label = "COLD" if cold else "HOT"
            print(f"  [{label}] Multi-patched {len(slot_assignments)} slots, mode={mode}")
        return self._write_binary(Path(output_path), data)

    # -- running ---------------------------------------------------------------

    def run_binary(self, binary_path, timeout: int = 120) -> List[Tuple[int, int]]:
        """Run a patched binary, returning (slot_index, delta) pairs."""
        result = run_harness(binary_path, timeout=timeout)
        return parse_perslot_timing(result.stdout)

    def measure_triple(self, mode: str, reset_name: str, trigger_name: str,
                       measure_name: str, trials: int = 1, start_slot: int = 1,
                       keep_binaries: Optional[Path] = None,
                       tmp_dir: Optional[Path] = None) -> Dict:
        """Measure one (RESET, TRIGGER, MEASURE) triple, hot vs cold."""
        reset_bytes = self.get_instruction_bytes(reset_name)
        trigger_bytes = self.get_instruction_bytes(trigger_name)
        measure_bytes = self.get_instruction_bytes(measure_name)

        results = {
            "reset": reset_name, "trigger": trigger_name,
            "measure": measure_name, "mode": mode, "trials": [],
        }

        tmp_dir = Path(tmp_dir) if tmp_dir else Path(".")
        hot_path = tmp_dir / f".tmp_patched_hot_{os.getpid()}"
        cold_path = tmp_dir / f".tmp_patched_cold_{os.getpid()}"
        self.create_patched_binary(hot_path, mode, reset_bytes, trigger_bytes,
                                   measure_bytes, cold=False, start_slot=start_slot)
        self.create_patched_binary(cold_path, mode, reset_bytes, trigger_bytes,
                                   measure_bytes, cold=True, start_slot=start_slot)

        if keep_binaries:
            keep_binaries.mkdir(parents=True, exist_ok=True)
            shutil.copy2(hot_path, keep_binaries / f"{trigger_name}_hot")
            shutil.copy2(cold_path, keep_binaries / f"{trigger_name}_cold")

        try:
            for trial in range(trials):
                hot_timings = dict(self.run_binary(hot_path))
                cold_timings = dict(self.run_binary(cold_path))
                trial_data = []
                for slot in self.slots[start_slot:]:
                    hot_d = hot_timings.get(slot.index)
                    cold_d = cold_timings.get(slot.index)
                    if hot_d is not None and cold_d is not None:
                        trial_data.append({
                            "slot": slot.index, "hot_delta": hot_d,
                            "cold_delta": cold_d, "diff": hot_d - cold_d,
                        })
                results["trials"].append({"trial_index": trial, "slot_data": trial_data})
                if self.verbose and trial_data:
                    hot_vals = [d["hot_delta"] for d in trial_data]
                    cold_vals = [d["cold_delta"] for d in trial_data]
                    diffs = [d["diff"] for d in trial_data]
                    print(f"  Trial {trial}: "
                          f"hot={sum(hot_vals) / len(hot_vals):.1f} "
                          f"cold={sum(cold_vals) / len(cold_vals):.1f} "
                          f"diff={sum(diffs) / len(diffs):.1f} "
                          f"(n={len(trial_data)} slots)")
        finally:
            hot_path.unlink(missing_ok=True)
            cold_path.unlink(missing_ok=True)

        all_hot, all_cold, all_diff = [], [], []
        for trial in results["trials"]:
            for d in trial["slot_data"]:
                all_hot.append(d["hot_delta"])
                all_cold.append(d["cold_delta"])
                all_diff.append(d["diff"])

        if all_hot:
            results["summary"] = {
                "n_samples": len(all_hot),
                "hot_mean": sum(all_hot) / len(all_hot),
                "hot_median": sorted(all_hot)[len(all_hot) // 2],
                "hot_min": min(all_hot), "hot_max": max(all_hot),
                "cold_mean": sum(all_cold) / len(all_cold),
                "cold_median": sorted(all_cold)[len(all_cold) // 2],
                "cold_min": min(all_cold), "cold_max": max(all_cold),
                "diff_mean": sum(all_diff) / len(all_diff),
                "diff_median": sorted(all_diff)[len(all_diff) // 2],
                "diff_min": min(all_diff), "diff_max": max(all_diff),
            }
        return results

    def run_sanity_check(self, mode: str = "issue") -> bool:
        """Run the unpatched binary and verify per-slot timing output."""
        print(f"\n=== Sanity Check (mode={mode}) ===")
        print(f"Binary: {self.binary_path}")
        try:
            timings = self.run_binary(self.binary_path)
        except RuntimeError as e:
            print(f"FAIL: {e}")
            return False

        print(f"  Got {len(timings)} timing values")
        if not timings:
            print("FAIL: No timing data received")
            return False

        deltas = [d for _, d in timings]
        mean = sum(deltas) / len(deltas)
        median = sorted(deltas)[len(deltas) // 2]
        print(f"  Mean:   {mean:.1f} cycles")
        print(f"  Median: {median} cycles")
        print(f"  Min:    {min(deltas)} cycles")
        print(f"  Max:    {max(deltas)} cycles")

        if median < 1 or median > 1000:
            print(f"WARNING: Median timing {median} looks suspicious")
        else:
            print("  Timing looks reasonable")

        expected = self.offsets.num_slots
        if len(timings) < expected * 0.9:
            print(f"WARNING: Got {len(timings)} timings, expected ~{expected}")
        else:
            print(f"  Slot count OK ({len(timings)}/{expected})")
        print("PASS: Sanity check passed")
        return True
