"""ISA specification model: a queryable view over the per-architecture
ISA JSON produced by the offline spec stage.

Top-level JSON keys: DEFAULTS, INSTRUCTIONS, PARAMETERS, CONSTANTS,
TABLES, SCHEDULING, SHADER_TYPES, SPECIAL_REGISTERS, SPECIAL_REGISTER_ALIASES.

Each INSTRUCTIONS entry has: INSTRUCTION, OPCODE, OPCCODES (alias map),
ARGUMENTS (field -> {bits, alias, width}), CONDITIONS (with parsed AST
``tree``), PROPERTIES, PREDICATES, FORMAT (LINES/TEXT/TOKENS/FIELDS/DEFAULTS),
LATENCY_INFO.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set

from ..enc.opcode import parse_opcode_string
from .. import paths as _paths


@dataclass
class InstructionClass:
    """One instruction class (opcode-modifier combination) from the ISA JSON."""

    name: str
    data: dict

    @property
    def opcode(self) -> int:
        return parse_opcode_string(self.data["OPCODE"])

    @property
    def opcodes(self) -> Dict[str, int]:
        """Alias -> opcode map (aliases are what disassemblers print)."""
        out = {}
        for alias, val in self.data.get("OPCODES", {}).items():
            try:
                out[alias] = parse_opcode_string(val)
            except (ValueError, TypeError):
                continue
        return out

    @property
    def arguments(self) -> Dict[str, dict]:
        return self.data.get("ARGUMENTS", {})

    @property
    def conditions(self) -> List[dict]:
        return self.data.get("CONDITIONS", [])

    @property
    def properties(self) -> dict:
        return self.data.get("PROPERTIES", {})

    @property
    def format_defaults(self) -> Dict[str, str]:
        return self.data.get("FORMAT", {}).get("DEFAULTS", {})

    @property
    def pipeline(self) -> Optional[str]:
        info = self.data.get("LATENCY_INFO") or {}
        return info.get("pipeline")

    @property
    def latency_info(self) -> dict:
        return self.data.get("LATENCY_INFO") or {}

    @property
    def valid_in_shaders_bitmask(self) -> Optional[int]:
        return self.properties.get("VALID_IN_SHADERS_BITMASK")


class ISASpec:
    """Machine-readable ISA specification for one architecture."""

    def __init__(self, data: dict, source: Optional[Path] = None, sm: Optional[str] = None):
        self._data = data
        self.source = source
        self.sm = sm
        self._instructions: Dict[str, InstructionClass] = {}
        self._opcode_index: Optional[Dict[int, List[InstructionClass]]] = None

    # -- loading ------------------------------------------------------------

    @classmethod
    def load(cls, sm=None, path=None) -> "ISASpec":
        """Load the ISA spec for an SM version (e.g. ``120``/``"120"``)
        or from an explicit JSON path."""
        if path is None:
            if sm is None:
                raise ValueError("ISASpec.load requires sm= or path=")
            path = _paths.isa_json_path(sm)
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        if sm is None:
            # infer from filename json_<sm>_with_conditions.json
            name = path.name
            if name.startswith("json_"):
                sm = name.split("_")[1]
        return cls(data, source=path, sm=str(sm) if sm else None)

    # -- top-level sections ---------------------------------------------------

    @property
    def defaults(self) -> dict:
        return self._data.get("DEFAULTS", {})

    @property
    def parameters(self) -> dict:
        return self._data.get("PARAMETERS", {})

    @property
    def constants(self) -> dict:
        return self._data.get("CONSTANTS", {})

    @property
    def tables(self) -> dict:
        return self._data.get("TABLES", {})

    @property
    def scheduling(self) -> dict:
        return self._data.get("SCHEDULING", {})

    @property
    def shader_types(self) -> dict:
        return self._data.get("SHADER_TYPES", {})

    @property
    def special_registers(self) -> Dict[int, str]:
        return {int(k): v for k, v in
                self._data.get("SPECIAL_REGISTERS", {}).items()}

    @property
    def special_register_aliases(self) -> Dict[str, int]:
        """Alternate source spellings -> index; absent in older ISA JSONs."""
        return {name: int(index) for name, index in
                self._data.get("SPECIAL_REGISTER_ALIASES", {}).items()}

    @property
    def raw(self) -> dict:
        return self._data

    # -- instructions -----------------------------------------------------------

    @property
    def instructions(self) -> Dict[str, InstructionClass]:
        if not self._instructions:
            self._instructions = {
                name: InstructionClass(name, entry)
                for name, entry in self._data.get("INSTRUCTIONS", {}).items()
            }
        return self._instructions

    def __len__(self) -> int:
        return len(self._data.get("INSTRUCTIONS", {}))

    def __iter__(self) -> Iterator[InstructionClass]:
        return iter(self.instructions.values())

    def __contains__(self, name: str) -> bool:
        return name in self.instructions

    def get(self, name: str) -> Optional[InstructionClass]:
        return self.instructions.get(name)

    def __getitem__(self, name: str) -> InstructionClass:
        return self.instructions[name]

    # -- opcode lookups ---------------------------------------------------------

    def opcode_index(self) -> Dict[int, List[InstructionClass]]:
        """Map from 13-bit opcode value to all classes using it."""
        if self._opcode_index is None:
            index: Dict[int, List[InstructionClass]] = {}
            for ic in self.instructions.values():
                for opcode in set(ic.opcodes.values()) | {ic.opcode}:
                    index.setdefault(opcode, []).append(ic)
            self._opcode_index = index
        return self._opcode_index

    def classes_for_opcode(self, opcode: int) -> List[InstructionClass]:
        return self.opcode_index().get(opcode, [])

    def opcodes_for_mnemonic(self, mnemonic: str) -> Set[int]:
        """All 13-bit opcode values whose alias matches ``mnemonic``
        (case-insensitive), e.g. "FADD"."""
        target = mnemonic.upper()
        out = set()
        for ic in self.instructions.values():
            for alias, val in ic.opcodes.items():
                if alias.upper() == target:
                    out.add(val)
        return out

    def defined_opcodes(self) -> Set[int]:
        """All opcode values defined by any class (primary + aliases)."""
        return set(self.opcode_index().keys())
