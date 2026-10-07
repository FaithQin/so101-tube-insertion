"""tools/replay_offset.py — replay a recorded episode with a per-joint command
offset, on the production path (sync_write at the dataset fps).

Born Sep 2, 2026: the elbow lands a constant +3.07 deg above its goal at two
poses (analysis/anchor_results.json). If commanding elbow-3.07 makes v3 ep0
seat again, the fault is the elbow's settle behaviour. Hardware-free tests:
pure functions + source guards (no EEPROM writes, torque left as found).
"""

import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "replay_offset.py"


def _load():
    sys.path.insert(0, str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("replay_offset", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_offsets_names_joints_and_rejects_unknown():
    mod = _load()
    assert mod.parse_offsets(["elbow_flex=-3.07"]) == {"elbow_flex": -3.07}
    assert mod.parse_offsets(["elbow_flex=-3.07", "shoulder_pan=+6"]) == {"elbow_flex": -3.07, "shoulder_pan": 6.0}
    assert mod.parse_offsets([]) == {}
    with pytest.raises(ValueError):
        mod.parse_offsets(["elbow=-3"])
    with pytest.raises(ValueError):
        mod.parse_offsets(["elbow_flex"])


def test_apply_offsets_touches_only_the_named_column_and_not_the_input():
    mod = _load()
    acts = np.arange(18, dtype=float).reshape(6, 3).T  # 3 ticks x 6 joints
    acts = np.ascontiguousarray(acts)
    before = acts.copy()
    out = mod.apply_offsets(acts, {"elbow_flex": -3.07})
    assert np.array_equal(acts, before), "input must not be mutated"
    col = mod.JOINTS.index("elbow_flex")
    assert np.allclose(out[:, col], before[:, col] - 3.07)
    other = [i for i in range(6) if i != col]
    assert np.array_equal(out[:, other], before[:, other])
    assert np.array_equal(mod.apply_offsets(acts, {}), before)


def test_replay_uses_the_production_sync_write_path_and_never_touches_eeprom():
    tree = ast.parse(TOOL.read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    goal_writes = [n for n in calls if n.func.attr == "sync_write"]
    assert goal_writes, "Goal_Position must go through sync_write (servo 4 ignores single writes)"
    single_goal = [n for n in calls if n.func.attr == "write"
                   and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "Goal_Position"]
    assert single_goal == [], "no per-joint Goal_Position writes"
    eeprom = {"Homing_Offset", "Min_Position_Limit", "Max_Position_Limit", "P_Coefficient", "D_Coefficient",
              "I_Coefficient", "CW_Dead_Zone", "CCW_Dead_Zone", "Max_Torque_Limit", "Protection_Time", "ID"}
    written = {n.args[0].value for n in calls if n.func.attr in ("write", "sync_write")
               and n.args and isinstance(n.args[0], ast.Constant)}
    assert not (written & eeprom), f"EEPROM registers must never be written: {written & eeprom}"
    assert not any(n.func.attr in ("write_calibration", "set_half_turn_homings", "reset_calibration") for n in calls)


def test_replay_leaves_torque_on_at_disconnect():
    assert "disconnect(disable_torque=False)" in TOOL.read_text()


V3 = Path("~/.cache/huggingface/lerobot/faithqin/so101-tube-insert-v3_20260827_110242").expanduser()


@pytest.mark.skipif(not V3.exists(), reason="v3 dataset cache not present")
def test_load_episode_returns_actions_in_joint_order_at_dataset_fps():
    mod = _load()
    acts, fps = mod.load_episode("faithqin/so101-tube-insert-v3_20260827_110242", 0)
    assert fps == 20
    assert acts.ndim == 2 and acts.shape[1] == 6
    assert 150 < acts.shape[0] < 400            # v3 episodes are ~209 frames
    assert acts.dtype == np.float64


def test_record_rows_pair_each_tick_with_its_reported_positions():
    """--record writes one row per tick: tick, commanded x6, reported x6."""
    mod = _load()
    cmd = [{"shoulder_pan": 1.0, "shoulder_lift": 2.0, "elbow_flex": 3.0, "wrist_flex": 4.0, "wrist_roll": 5.0, "gripper": 6.0}]
    rep = [{"shoulder_pan": 1.1, "shoulder_lift": 2.2, "elbow_flex": 3.3, "wrist_flex": 4.4, "wrist_roll": 5.5, "gripper": 6.6}]
    rows = mod.record_rows(cmd, rep)
    assert rows[0][0] == 0
    assert rows[0][1:7] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert rows[0][7:13] == [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
    assert mod.RECORD_HEADER[0] == "tick" and mod.RECORD_HEADER[1] == "cmd_shoulder_pan" and mod.RECORD_HEADER[7] == "rep_shoulder_pan"


def test_recording_reads_through_sync_read_not_single_reads():
    """Servo 4 ignores single-register traffic; the observation path is sync_read."""
    src = TOOL.read_text()
    assert 'sync_read("Present_Position"' in src
