"""Loaders for the spec layer's reverse-engineered data tables.

These were previously hardcoded inside json_maker.py's main(); they are
shipped as data so experiments and encoders can reuse them.
"""

import json
from pathlib import Path
from functools import lru_cache

_DATA_DIR = Path(__file__).resolve().parent / "data"


@lru_cache(maxsize=None)
def _load(name: str) -> dict:
    with open(_DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def scheduling_enums() -> dict:
    """USCHED_INFO and BATCH_T enumerations (reverse-engineered)."""
    return _load("scheduling.json")


@lru_cache(maxsize=None)
def usched_info() -> dict:
    return scheduling_enums()["USCHED_INFO"]


@lru_cache(maxsize=None)
def batch_t() -> dict:
    return scheduling_enums()["BATCH_T"]


@lru_cache(maxsize=None)
def ishader_values() -> dict:
    """ISHADER_* bitmask constants (for evaluating VALID_IN_SHADERS)."""
    return _load("shader_types.json")["ISHADER"]


@lru_cache(maxsize=None)
def shader_types() -> dict:
    """ST_* shader type constants (hardware SR20 values)."""
    return _load("shader_types.json")["SHADER_TYPES"]


@lru_cache(maxsize=None)
def special_registers() -> dict:
    """Framework-discovered special register names (int keys).

    These override display names from the architecture's Rosetta String,
    whose names remain available as aliases in the generated ISA JSON.
    This is supplemental data, not the complete special register table.
    """
    return {int(k): v for k, v in _load("special_registers.json").items()}
