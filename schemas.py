"""Shared data schemas (version 2) for the EoSS framework.

Offsets document
----------------
A unified schema covering both harness families (OSIRIS-style 5-instruction
slots and chain-style harnesses such as the contention experiment)::

    {
      "schema": "eoss.offsets/v2",
      "binary": "harness_sm120",
      "kernel_name": "osiris_portable_kernel",
      "harness": "osiris_slots",
      "code_base_offset": 594200,
      "slots": [
        {"index": 0,
         "anchors": {"reset": 288, "trigger": 304, "cs2r_start": 320,
                      "measure": 336, "cs2r_end": 352}}
      ]
    }

File offset of an anchor = ``code_base_offset + anchors[name]``.

Encoding records
----------------
The per-instruction JSON produced by the generator and consumed by
validators/campaigns. Field names match the historical
``all_instructions.json`` / ``validated_gpr.json`` formats.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

OFFSETS_SCHEMA = "eoss.offsets/v2"


# ---------------------------------------------------------------------------
# Offsets document
# ---------------------------------------------------------------------------

@dataclass
class Slot:
    index: int
    anchors: Dict[str, int]  # anchor name -> SASS offset (bytes)

    def has(self, name: str) -> bool:
        return name in self.anchors


@dataclass
class OffsetsDoc:
    binary: str
    kernel_name: str
    code_base_offset: int
    harness: str = "unknown"
    slots: List[Slot] = field(default_factory=list)
    extra: Dict = field(default_factory=dict)  # harness-specific anchors etc.
    schema: str = OFFSETS_SCHEMA

    @property
    def num_slots(self) -> int:
        return len(self.slots)

    def file_offset(self, slot: Slot, anchor: str) -> int:
        return self.code_base_offset + slot.anchors[anchor]

    def has_anchor(self, name: str) -> bool:
        return bool(self.slots) and name in self.slots[0].anchors

    def to_json(self) -> dict:
        return {
            "schema": self.schema,
            "binary": self.binary,
            "kernel_name": self.kernel_name,
            "harness": self.harness,
            "code_base_offset": self.code_base_offset,
            "num_slots": self.num_slots,
            "slots": [{"index": s.index, "anchors": dict(s.anchors)}
                      for s in self.slots],
            "extra": self.extra,
        }

    def save(self, path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)

    @classmethod
    def from_json(cls, data: dict) -> "OffsetsDoc":
        if "slots" in data and data["slots"] and "anchors" not in data["slots"][0]:
            raise ValueError(
                "legacy offsets schema detected; regenerate with `eoss build`")
        return cls(
            binary=data.get("binary", ""),
            kernel_name=data.get("kernel_name", "unknown"),
            code_base_offset=data.get("code_base_offset", 0),
            harness=data.get("harness", "unknown"),
            slots=[Slot(index=s.get("index", i), anchors=dict(s.get("anchors", {})))
                   for i, s in enumerate(data.get("slots", []))],
            extra=dict(data.get("extra", {})),
            schema=data.get("schema", OFFSETS_SCHEMA),
        )

    @classmethod
    def load(cls, path) -> "OffsetsDoc":
        with open(path) as f:
            return cls.from_json(json.load(f))


# ---------------------------------------------------------------------------
# Encoding records (instruction encodings)
# ---------------------------------------------------------------------------

@dataclass
class EncodingRecord:
    """One instruction encoding with metadata."""

    instruction: str                 # class/variant name, e.g. "fadd__RRR_RR"
    bytes_le: bytes                  # 16-byte encoding (little-endian)
    opcode_key: Optional[str] = None
    class_name: Optional[str] = None
    variant: Optional[str] = None
    argument_values: Dict[str, int] = field(default_factory=dict)
    pipeline: Optional[str] = None
    latency_cycles: Optional[int] = None
    delay: int = 0
    is_valid: Optional[bool] = None
    static_valid: Optional[bool] = None
    failed_conditions: List[str] = field(default_factory=list)
    disasm_text: Optional[str] = None

    def to_json(self) -> dict:
        out = {
            "opcode_key": self.opcode_key,
            "instruction": self.instruction,
            "class": self.class_name,
            "variant": self.variant,
            "argument_values": self.argument_values,
            "bytes_le": self.bytes_le.hex(),
            "is_valid": self.is_valid,
            "failed_conditions": self.failed_conditions,
            "pipeline": self.pipeline,
            "latency_cycles": self.latency_cycles,
            "delay": self.delay,
            "static_valid": self.static_valid,
            "disasm_text": self.disasm_text,
        }
        return out

    @classmethod
    def from_json(cls, data: dict) -> "EncodingRecord":
        return cls(
            instruction=data.get("instruction") or data.get("opcode_key", ""),
            bytes_le=bytes.fromhex(data["bytes_le"]),
            opcode_key=data.get("opcode_key"),
            class_name=data.get("class"),
            variant=data.get("variant"),
            argument_values=data.get("argument_values", {}),
            pipeline=data.get("pipeline"),
            latency_cycles=data.get("latency_cycles"),
            delay=data.get("delay", 0),
            is_valid=data.get("is_valid"),
            static_valid=data.get("static_valid"),
            failed_conditions=data.get("failed_conditions", []),
            disasm_text=data.get("disasm_text"),
        )


def load_encodings(path) -> List[EncodingRecord]:
    """Load an instruction-encoding JSON file (list of records)."""
    with open(path) as f:
        data = json.load(f)
    return [EncodingRecord.from_json(d) for d in data]


def save_encodings(records: List[EncodingRecord], path) -> None:
    with open(path, "w") as f:
        json.dump([r.to_json() for r in records], f, indent=2)


def encodings_by_name(records: List[EncodingRecord]) -> Dict[str, EncodingRecord]:
    """Index records by instruction name."""
    return {r.instruction: r for r in records}
