"""Special-register parsing and shipped tables, without requiring a GPU."""

import pytest

from eoss import paths
from eoss.isa.model import ISASpec
from eoss.spec.rosetta import parse_special_registers


def test_ranges_implicit_values_and_aliases():
    registers, aliases = parse_special_registers('''
        SpecialRegister SR(0..255)=(0..255), SR_LANEID=0,
            SR_MACHINE_ID_(0..3)=(24..27), SR_AFFINITY,
            SR_OTHER_NAME=24, SR_AFTER_ALIAS,
            SR_TID=32, "SR_TID.X", "SR_TID.Y", "SR_TID.Z",
            SR_TTU_TICKET_INFO=133,
            SR_TMem=141, SR_TMemSz=141,
            SR_CONTEXT_SAVE_ADDR_LO, SR_CONTEXT_SAVE_ADDR_HI, SRZ=255;
        UnrelatedEnum SR_NOT_A_REGISTER=7;
    '''.splitlines())

    assert set(registers) == set(range(256))
    assert registers[0] == "SR_LANEID"
    assert registers[4] == "SR4"
    assert registers[24] == "SR_MACHINE_ID_0"
    assert registers[27] == "SR_MACHINE_ID_3"
    assert registers[28] == "SR_AFFINITY"
    assert aliases["SR_OTHER_NAME"] == 24
    assert aliases["SR_AFTER_ALIAS"] == 25
    assert registers[33] == "SR_TID_X"
    assert aliases["SR_TID.X"] == 33
    assert registers[35] == "SR_TID_Z"
    assert aliases["SR_TID.Z"] == 35
    assert registers[133] == "SR_TTU_SESSION_ACTIVE"
    assert aliases["SR_TTU_TICKET_INFO"] == 133
    assert registers[141] == "SR_TMem"
    assert aliases["SR_TMemSz"] == 141
    assert registers[142] == "SR_CONTEXT_SAVE_ADDR_LO"
    assert registers[143] == "SR_CONTEXT_SAVE_ADDR_HI"
    assert registers[255] == "SRZ"
    assert aliases["SR255"] == 255
    assert "SR_NOT_A_REGISTER" not in aliases


@pytest.mark.parametrize("source", [
    "UnrelatedEnum ENTRY=0;",
    "SpecialRegister SR_MACHINE_ID_(0..3)=(24..26);",
    "SpecialRegister SR_MACHINE_ID_(3..0)=(24..27);",
    "SpecialRegister SR_TOO_LARGE=256;",
    "SpecialRegister SR_SAME_NAME=24, SR_SAME_NAME=25;",
])
def test_invalid_enumeration_fails_loudly(source):
    with pytest.raises(ValueError):
        parse_special_registers([source])


def test_older_json_without_aliases():
    spec = ISASpec({"SPECIAL_REGISTERS": {"0": "SR_LANEID"}})
    assert spec.special_registers == {0: "SR_LANEID"}
    assert spec.special_register_aliases == {}


@pytest.mark.parametrize("sm", [70, 80, 86, 89, 90, 100, 120, 121])
def test_shipped_registers_and_architecture_differences(sm):
    if not paths.isa_json_path(sm).is_file():
        pytest.skip(f"SM{sm} ISA JSON not available")
    spec = ISASpec.load(sm=sm)
    registers = spec.special_registers
    aliases = spec.special_register_aliases
    assert set(registers) == set(range(256))
    assert registers[20] == "SM_SHADER_TYPE"
    assert registers[41] == "SR_CirQueueIncrMinusOne"
    assert registers[42] == "SR_NLATC"
    assert registers[48] == "SR_SWINLO"
    assert registers[56] == "SR_EQMASK"
    assert registers[60] == "SR_GEMASK"
    assert registers[62] == "SR_BARRIERALLOC"
    assert registers[84] == "SR_ESR_PC"
    assert registers[85] == "SR_ESR_PC_HI"
    assert registers[100] == "SR_PM0"
    assert registers[115] == "SR_PM_HI7"
    assert registers[116] == "SR_SNAP_PM0"
    assert registers[131] == "SR_SNAP_PM_HI7"
    assert registers[133] == "SR_TTU_SESSION_ACTIVE"
    assert registers[255] == "SRZ"
    assert aliases["SR_TID.X"] == 33
    assert aliases["SR_CTAID.Z"] == 39
    assert all(index in registers for index in aliases.values())
    for index in range(256):
        assert registers[index] == f"SR{index}" or aliases[f"SR{index}"] == index

    if sm == 70:
        assert registers[132] == "SR132"
        assert "SR_UGPU_ID" not in aliases
        assert "SR_TTU_TICKET_INFO" not in aliases
    else:
        assert registers[132] == "SR_VARIABLE_RATE"
        assert aliases["SR_UGPU_ID"] == aliases["SR_SW_SCRATCH"] == 24
        assert aliases["SR_TTU_TICKET_INFO"] == 133

    assert registers[65] == ("SR_CGAERRORSTATUS" if sm >= 90 else "SR65")
    assert registers[134] == ("SR_WARPGROUP_INFO" if sm == 90 else "SR134")
    assert registers[135] == ("SR_WARPGROUPID" if sm == 90 else "SR135")
    if sm >= 100:
        assert aliases["SR_CgaSize"] == 138
        assert aliases["SR_TMemSz"] == 141
    else:
        assert "SR_CgaSize" not in aliases
        assert "SR_TMemSz" not in aliases
    if sm in (120, 121):
        assert registers[142] == "SR_CONTEXT_SAVE_ADDR_LO"
        assert registers[143] == "SR_CONTEXT_SAVE_ADDR_HI"
    else:
        assert registers[142] == "SR142"
        assert registers[143] == "SR143"
