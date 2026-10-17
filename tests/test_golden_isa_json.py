"""Golden test: regenerating ISA JSON from the Rosetta Strings must
reproduce the shipped ISA JSON byte-for-byte.

Skipped when the Rosetta String files (not distributed with the artifact)
are absent.
"""

import json

import pytest

from eoss import paths
from eoss.spec.rosetta import build_isa_json

# (sm, whether shipped ISA JSON exists for comparison)
GOLDEN_SMS = ["70", "80", "86", "89", "90", "100", "120", "121"]


def _have_inputs(sm: str) -> bool:
    return (paths.sass_instructions_path(sm).is_file()
            and paths.sass_latencies_path(sm).is_file()
            and paths.isa_json_path(sm).is_file())


@pytest.mark.parametrize("sm", GOLDEN_SMS)
def test_regenerate_isa_json_byte_identical(sm):
    if not _have_inputs(sm):
        pytest.skip(f"Rosetta String inputs for SM{sm} not available")
    regen = build_isa_json(paths.sass_instructions_path(sm),
                           paths.sass_latencies_path(sm), verbose=False)
    with open(paths.isa_json_path(sm)) as f:
        shipped = json.load(f)
    # Compare as serialized JSON (byte-identity of the canonical dump)
    assert json.dumps(regen, indent=2, ensure_ascii=False) == \
        json.dumps(shipped, indent=2, ensure_ascii=False)
