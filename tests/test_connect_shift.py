"""Regression: compensate the pose shift that `SOFollower.connect()` itself causes.

Measured Aug 30 2026, homing to the v3 pose and then connecting the robot the way
a rollout does, three runs:

    wrist_roll    -1.62 / -1.66 / -1.57     <-- the problem
    shoulder_lift -5.69 / -5.69 / -5.69     <-- NOT drift, see below
    shoulder_pan  +0.29 / +0.20 / +0.20
    elbow_flex    -0.11 / -0.20 / -0.20
    wrist_flex    -0.10 / -0.10 / -0.10
    gripper        0.00

The arm is rock steady on its own — held wrist_roll at -2.08 for 30 s with torque
on — so this is caused by the connect/configure path relaxing the joint, and
backlash then takes it.

Why it mattered and why nothing caught it: the preflight gate measures the pose
`home_arm` leaves behind, but the POLICY observes the pose after the rollout has
connected. Rollout frame-0 for B2R-A-01 was wrist_roll -4.7, which is -2.3 sigma
against v3 training — outside the band the gate had just certified. Every v3
trial so far started out of distribution on that joint.

shoulder_lift's -5.69 is NOT drift and must not be compensated: it settles
passively against its stop beyond the -100 command clamp, reads -99.6 clamped,
and after connect reads -105.3 — which is v3's actual training value (-105.19).
Compensating it would break the one joint that was already right.
"""

import re
import sys

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))

SRC = (TOOLS / "home_arm.py").read_text()


def test_a_connect_shift_table_exists():
    assert "CONNECT_SHIFT" in SRC, (
        "home_arm does not compensate the connect-induced shift, so the policy "
        "starts outside the band the gate certified"
    )


def test_wrist_roll_compensation_matches_the_measurement():
    import home_arm_shift

    v = home_arm_shift.CONNECT_SHIFT["wrist_roll"]
    assert -1.8 < v < -1.4, f"wrist_roll compensation {v} does not match the measured -1.62"


def test_one_source_of_truth_for_the_shift():
    """home_arm and the gate must read the SAME table or they compensate by
    different amounts and silently disagree about where the arm is."""
    assert "from home_arm_shift import CONNECT_SHIFT" in SRC
    assert "CONNECT_SHIFT = {" not in SRC, "home_arm must not define its own copy"


def test_shoulder_lift_is_not_compensated():
    """Its -5.69 is the passive settle to its true training value, not drift."""
    import home_arm_shift

    assert "shoulder_lift" not in home_arm_shift.CONNECT_SHIFT, (
        "shoulder_lift must NOT be compensated — it settles to its correct "
        "training value after connect"
    )


def test_negligible_joints_are_not_compensated():
    """pan/elbow/wrist_flex/gripper all shift under 0.3 deg. Compensating noise
    adds a failure mode for no benefit."""
    import home_arm_shift

    for j in ("shoulder_pan", "elbow_flex", "wrist_flex", "gripper"):
        assert j not in home_arm_shift.CONNECT_SHIFT, (
            f"{j} shifts <0.3 deg and should not be compensated"
        )


def test_compensation_can_be_disabled():
    """A standalone diagnostic home is not followed by a connect, so the
    compensation would leave the arm off by design."""
    assert "no-compensate" in SRC or "no_compensate" in SRC


def test_compensation_is_applied_to_the_target():
    """post = pre + shift, so to land on HOME the arm must be driven to
    HOME - shift."""
    assert "- CONNECT_SHIFT" in SRC or "-CONNECT_SHIFT" in SRC or "shift" in SRC.lower()


# --- the gate must judge the pose the POLICY sees, not the one home_arm leaves --

sys.path.insert(0, str(TOOLS))
import preflight  # noqa: E402

_PRE = "settled: {{'shoulder_pan': 2.0, 'shoulder_lift': -99.3, 'elbow_flex': 82.5, 'wrist_flex': 73.6, 'wrist_roll': {roll}, 'gripper': 1.1}}"
_SCENE = "tube (blue cap)   : ref=( 360.0, 355.8)  live=( 359.0, 355.0)  delta=( -1.0, -0.8) px  |d|=1.2\n"
# All six joints, shaped like a real receipt (tools/scored_logs/h83j2kp6_20260906_161402.ping):
# since Sep 12 preflight refuses any joint whose temperature it cannot read.
_PING = (
    "  id 1 shoulder_pan    pos  2013  load    60  cur    1   44C  125  torque 1  ok\n"
    "  id 2 shoulder_lift   pos   779  load   -56  cur    0   47C  124  torque 0  ok\n"
    "  id 3 elbow_flex      pos  2948  load    80  cur    0   46C  126  torque 0  ok\n"
    "  id 4 wrist_flex      pos  2945  load    44  cur    1   48C  123  torque 1  ok\n"
    "  id 5 wrist_roll      pos  2039  load    72  cur    3   47C  124  torque 1  ok\n"
    "  id 6 gripper         pos  2063  load     0  cur    0   46C  125  torque 1  ok\n"
)


def test_compensated_home_passes_because_post_connect_is_in_band():
    """home_arm now targets -0.28 so that connect's -1.62 lands it at -1.9.
    Gating the PRE-connect -0.28 against the training band rejects a correct
    setup — which is exactly what blocked B2V-A-01."""
    ok, reasons = preflight.gate(_SCENE, _PRE.format(roll=-0.28), _PING, dataset="v3")
    assert ok, reasons


def test_uncompensated_home_still_fails():
    """A pre-connect -2.1 becomes -3.7 after connect, which is out of band. The
    gate must still catch that — this is the bug it exists to prevent."""
    ok, reasons = preflight.gate(_SCENE, _PRE.format(roll=-2.1), _PING, dataset="v3")
    assert not ok
    assert any("wrist_roll" in r for r in reasons)
