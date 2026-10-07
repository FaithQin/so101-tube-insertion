"""Homing must refuse a pose the RANGE-mode bus cannot actually reach, and must still allow the
frame-0 poses every trial to date has used.

Sep 6 2026, found by the bench-morning review and confirmed on the arm. `tools/_arms.py:33`
normalizes RANGE_M100_100 while production is DEGREES; the error scales with distance from the
calibration midpoint. Frame-0 poses sit near the midpoint and are unaffected (that is why every
v2/v3/v4 trial has worked). The PRE-GRASPED `v4-carry` pose is extended, and its wrist_roll of
49.7 lands near 89.5 -- about 40 degrees away -- while the operator reaches in to place a tube.

These tests are the guard, not the fix. The fix is `tests/test_arms_norm_mode_matches_production.py`,
an xfail(strict) that flips `_arms` to the production mode and needs its own bench verification.
Nothing here touches hardware: the calibration is read from the saved file.
"""
import json
import re
import sys

import pytest

from conftest import PROJECT, TOOLS

sys.path.insert(0, str(TOOLS))

import norm_units as nu  # noqa: E402

CAL = PROJECT / "calibration-backup" / "robots" / "so_follower" / "follower.json"
JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


@pytest.fixture(scope="module")
def cal():
    if not CAL.exists():
        pytest.skip("calibration-backup/follower.json not on this machine")
    return nu.load_calibration(CAL)


def _table(tag: str) -> dict:
    src = (TOOLS / "home_arm.py").read_text()
    m = re.search(rf'"{tag}":\s*\[([^\]]+)\]', src)
    assert m, f"home_arm.py has no {tag} pose"
    return dict(zip(JOINT_ORDER, [float(x) for x in m.group(1).split(",")]))


# ---------------------------------------------------------------------------
# the conversion itself
# ---------------------------------------------------------------------------
def test_the_midpoint_is_the_shared_zero_so_it_has_no_unit_error(cal):
    """Both modes measure from the calibration midpoint; only the scale differs. A pose AT the
    midpoint is therefore exact, which is why near-zero frame-0 values have always been safe."""
    for j in JOINT_ORDER:
        if j == "gripper":
            continue
        assert nu.unit_error_deg(j, 0.0, cal) == pytest.approx(0.0, abs=1e-9)


def test_the_error_grows_with_distance_from_the_midpoint(cal):
    e10 = abs(nu.unit_error_deg("wrist_roll", 10.0, cal))
    e50 = abs(nu.unit_error_deg("wrist_roll", 50.0, cal))
    assert e50 > e10 > 0
    assert e50 == pytest.approx(5 * e10, rel=1e-6), "the error must be proportional to the value"


def test_wrist_roll_is_the_worst_joint_because_its_calibrated_range_is_narrowest(cal):
    errs = {j: abs(nu.unit_error_deg(j, 50.0, cal)) for j in JOINT_ORDER if j != "gripper"}
    assert max(errs, key=errs.get) == "wrist_roll", errs


def test_the_gripper_is_exempt_because_both_buses_normalize_it_the_same_way(cal):
    """Measured on the arm Sep 6: production is DEGREES on the five arm joints and RANGE_0_100 on
    the gripper; tools/_arms.py is RANGE_M100_100 and RANGE_0_100. Same mode, no unit error --
    and treating it like an arm joint invents a ~60 deg error that is not there."""
    from lerobot.motors import MotorNormMode
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    import _arms

    diag = {n: m.norm_mode for n, m in _arms.motors().items()}
    assert diag["gripper"] == MotorNormMode.RANGE_0_100
    assert SO101FollowerConfig.use_degrees is True
    assert "gripper" in nu.EXEMPT_JOINTS
    assert nu.offenders({"gripper": 13.2}, cal) == {}


# ---------------------------------------------------------------------------
# the two poses that matter today
# ---------------------------------------------------------------------------
def test_the_v4_frame0_pose_is_safe_to_command(cal):
    """Every trial to date homed this way. It must keep working, or the fix is worse than the bug."""
    bad = nu.offenders(_table("v4"), cal)
    assert bad == {}, f"the frame-0 pose would now be refused: {nu.describe(bad)}"


def test_the_v4_carry_pose_is_refused_and_wrist_roll_is_the_reason(cal):
    bad = nu.offenders(_table("v4-carry"), cal)
    assert "wrist_roll" in bad, f"the carry pose was not caught: {nu.pose_unit_errors(_table('v4-carry'), cal)}"
    assert abs(bad["wrist_roll"]) > 30, bad
    assert "wrist_roll" in nu.describe(bad)


def test_shoulder_lift_is_exempt_because_it_is_commanded_past_its_stop(cal):
    """Its training value (-105) is beyond the -100 goal clamp and it settles passively, so its
    computed error is not a pose error — home_gate.passive_joints says the same."""
    assert abs(nu.unit_error_deg("shoulder_lift", -105.3, cal)) > nu.MAX_UNIT_ERROR_DEG
    assert "shoulder_lift" not in nu.offenders(_table("v4"), cal)


def test_older_frame0_poses_stay_safe_too(cal):
    for tag in ("v2", "v3"):
        assert nu.offenders(_table(tag), cal) == {}, tag


# ---------------------------------------------------------------------------
# home_arm actually refuses, before it opens the bus
# ---------------------------------------------------------------------------
def test_home_arm_refuses_an_unreachable_pose_before_touching_the_bus():
    src = (TOOLS / "home_arm.py").read_text()
    assert "norm_units" in src, "home_arm does not check the unit error at all"
    guard = re.search(r"norm_units\.offenders\(", src)
    assert guard, "home_arm never calls norm_units.offenders"
    assert guard.start() < src.index("open_bus("), (
        "the unit-error guard runs AFTER the bus is opened — by then the arm can already be commanded"
    )


@pytest.mark.parametrize("argv", [["--pose", "v4-carry", "--close-on-tube"]])  # safe: the guard is argparse-time, so the bus is never opened
def test_home_arm_exits_nonzero_on_the_carry_pose_today(argv):
    """EXECUTED. Until tools/_arms.py runs in the production norm mode, this command must not
    drive the arm: its wrist_roll lands ~40 deg from the table while a hand is in the workspace."""
    import subprocess

    r = subprocess.run([sys.executable, str(TOOLS / "home_arm.py"), *argv],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode != 0, f"home_arm accepted the carry pose (rc {r.returncode})\n{r.stdout}\n{r.stderr}"
    out = r.stdout + r.stderr
    assert "wrist_roll" in out and re.search(r"(?i)unit|norm", out), out
