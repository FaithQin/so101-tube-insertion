"""tools/dump_registers.py — the Sep 2, 2026 register audit.

Born from the Sep 1–2 calibration debacle: "EEPROM verified equal" covered only
Homing_Offset and the two position limits, while raw register writes during the
day touched other EEPROM fields (Protection_Time had been left at 0 before). The
elbow then showed a constant +3.07° settle error at two poses — the signature of
a control-parameter register, not of calibration. This tool reads EVERY EEPROM
register on every servo and flags the ones that differ across servos.

Hardware-free: parses the tool's source and unit-tests its pure function.
"""

import ast
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "dump_registers.py"


def _load():
    sys.path.insert(0, str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("dump_registers", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_eeprom_list_is_every_register_below_torque_enable():
    """The list is derived from lerobot's own control table, so it cannot go stale."""
    from lerobot.motors.feetech.tables import STS_SMS_SERIES_CONTROL_TABLE as T

    mod = _load()
    regs = mod.eeprom_registers()
    ram_start = T["Torque_Enable"][0]
    expected = [name for name, (addr, _) in T.items() if addr < ram_start]
    assert regs == expected
    for must in ("Model_Number", "Firmware_Major_Version", "Homing_Offset",
                 "P_Coefficient", "D_Coefficient", "I_Coefficient",
                 "CW_Dead_Zone", "CCW_Dead_Zone", "Protection_Time",
                 "Max_Torque_Limit", "Minimum_Startup_Force"):
        assert must in regs


def test_tool_never_writes_to_a_servo():
    """A register AUDIT must be read-only. Any write path fails this test."""
    tree = ast.parse(TOOL.read_text())
    offenders = [
        n.func.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in ("write", "sync_write", "write_calibration",
                            "set_half_turn_homings", "reset_calibration")
    ]
    assert offenders == []


def test_tool_leaves_torque_alone_on_disconnect():
    """Default disconnect() disables torque and drops the arm. ping_motors sets
    disable_torque=False; this tool must too."""
    assert "disconnect(disable_torque=False)" in TOOL.read_text()


def test_flag_outliers_ignores_per_servo_registers_and_flags_real_ones():
    mod = _load()
    table = {
        "ID": {"a": 1, "b": 2, "c": 3},
        "Homing_Offset": {"a": 5, "b": -9, "c": 0},
        "Min_Position_Limit": {"a": 700, "b": 900, "c": 1000},
        "P_Coefficient": {"a": 32, "b": 32, "c": 32},
        "CW_Dead_Zone": {"a": 1, "b": 1, "c": 35},
        "Status": {"a": 0, "b": None, "c": 0},
    }
    out = mod.flag_outliers(table)
    assert "ID" not in out
    assert "Homing_Offset" not in out
    assert "Min_Position_Limit" not in out
    assert "P_Coefficient" not in out
    assert out["CW_Dead_Zone"] == {"a": 1, "b": 1, "c": 35}
    assert "Status" not in out  # a None is 'unreadable', not a difference


def test_calibration_cross_check_reports_exact_mismatches():
    """The instrument validates itself: the six registers whose true values the
    calibration file already holds must match the live read. A mismatch is
    reported per (joint, register); a clean table reports none."""
    mod = _load()
    live = {
        "Homing_Offset": {"a": -436, "b": 576},
        "Min_Position_Limit": {"a": 731, "b": 773},
        "Max_Position_Limit": {"a": 3264, "b": 3178},
    }
    cal = {"a": {"homing_offset": -436, "range_min": 731, "range_max": 3264},
           "b": {"homing_offset": 576, "range_min": 773, "range_max": 3178}}
    assert mod.calibration_mismatches(live, cal) == []
    cal["b"]["range_max"] = 3000
    assert mod.calibration_mismatches(live, cal) == [("b", "Max_Position_Limit", 3178, 3000)]


def test_calibration_cross_check_flags_unreadable_as_mismatch():
    mod = _load()
    live = {"Homing_Offset": {"a": None}, "Min_Position_Limit": {"a": 1}, "Max_Position_Limit": {"a": 2}}
    cal = {"a": {"homing_offset": 5, "range_min": 1, "range_max": 2}}
    assert mod.calibration_mismatches(live, cal) == [("a", "Homing_Offset", None, 5)]
