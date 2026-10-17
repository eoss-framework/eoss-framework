"""Golden test for the packaged TTU encoding data: the canned
canonical_traversal sequence must equal the parameterized builders
(the construction used by the original demo patcher)."""

import json

import pytest

from eoss import paths


def _ttumf(fuse_count: int):
    return ((fuse_count << 40) | 0x79D4, 0x000FC80000000000)


def _ttust(offset: int, rb: int = 0):
    return ((offset << 40) | (rb << 32) | 0x73D1, 0x000FE20000000000)


EXPECTED = [
    ("TTUOPEN", (0x00000000000073D0, 0x000FEC0000000000)),
    ("TTUMACROFUSE(8)", _ttumf(8)),
    ("TTUST [0x000]", _ttust(0x000, 0)),
    ("TTUST [0x040]", _ttust(0x040, 0)),
    ("TTUST [0x050]", _ttust(0x050, 0)),
    ("TTUST [0x080]", _ttust(0x080, 0)),
    ("TTUGO", (0x00000000000073D3, 0x000FE20000000000)),
    ("TTULD", (0x00000000000073D2, 0x000FE20000000000)),
    ("TTULD", (0x00000000000073D2, 0x000FE20000000000)),
    ("TTULD.CLOSE", (0x00000000000073D2, 0x000FE20000000100)),
]


def test_ttu_sequence_data():
    p = paths.encoding_data("ttu_sm120.json")
    if not p.is_file():
        pytest.skip("ttu_sm120.json not packaged")
    data = json.loads(p.read_text())
    seq = data["sequences"]["canonical_traversal"]["instructions"]
    assert len(seq) == len(EXPECTED)
    for entry, (name, (lo, hi)) in zip(seq, EXPECTED):
        assert entry["name"] == name
        assert int(entry["lo"], 16) == lo, f"{name} lo mismatch"
        assert int(entry["hi"], 16) == hi, f"{name} hi mismatch"


def test_ttu_opcodes():
    data = json.loads(paths.encoding_data("ttu_sm120.json").read_text())
    assert data["opcodes"] == {
        "TTUOPEN": "0x3d0", "TTUMACROFUSE": "0x9d4", "TTUST": "0x3d1",
        "TTUGO": "0x3d3", "TTULD": "0x3d2", "TTUCCTL": "0x3d5",
    }
