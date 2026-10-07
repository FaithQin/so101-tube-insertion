"""How far a pose table lands from where it says, because the homing bus and the dataset disagree.

Sep 6 2026, bench morning. `tools/_arms.py:33` builds the follower's joints in
`MotorNormMode.RANGE_M100_100` (gripper `RANGE_0_100`), while production is DEGREES
(`SOFollowerConfig.use_degrees = True`). `tools/home_arm.py:21,87` drives the arm through that
helper, but every pose table it commands -- `home_arm.HOME_POSES`, `home_gate.FRAME0`,
`TRAIN_HOME_*` -- is derived from the DATASET, which is in DEGREES. The two modes share a zero
(the calibration midpoint) but not a scale: RANGE is 200 / range_width per raw tick, DEGREES is
360 / 4095. So the error is PROPORTIONAL TO THE VALUE, and near the midpoint it nearly vanishes.

That is why this has never mattered. Every frame-0 pose sits close to the midpoint:

    v4 frame 0    pan +0.2   lift -6.0 (passive, rests on its stop)   elbow -0.8
                  wrist_flex -0.1   wrist_roll -1.4

and why the PRE-GRASPED pose is a different matter entirely -- it is an extended pose, far from
the midpoint on the joints whose calibrated range is narrowest:

    v4-carry      pan +7.6   lift +2.6   elbow +0.3   wrist_flex -0.1   wrist_roll +39.8

Commanding `v4-carry` through the RANGE bus rotates the wrist roughly 40 degrees away from where
the table says, with the operator's hand reaching in to place a tube. `home_arm` therefore
REFUSES any pose whose worst gated joint exceeds `MAX_UNIT_ERROR_DEG`, and says why.

The real fix is `tools/_arms.py` in the production norm mode, which
`tests/test_arms_norm_mode_matches_production.py` already tracks as `xfail(strict)` with the
warning "Do not flip this before a bench session that depends on the current homing." Flipping it
changes v2/v3/v4 homing too and needs its own bench verification. This module is the guard that
makes the unsafe case impossible in the meantime; it is not the fix.
"""
from __future__ import annotations

import json
from pathlib import Path

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
RAW_TICKS = 4095.0
DEG_PER_TICK = 360.0 / RAW_TICKS

# Above this, a pose is not safe to command through the RANGE-mode bus. 2.5 deg is the homing
# tolerance `home_arm.TOL_DEFAULT` already treats as "check for obstruction", so a unit error
# larger than that is indistinguishable from a mechanical fault in the residual line.
MAX_UNIT_ERROR_DEG = 2.5

# shoulder_lift's training value (-105) is beyond the -100 goal clamp: it is commanded past its
# stop deliberately and settles there passively, so its computed "error" is not a pose error.
# home_gate.PASSIVE_SETTLE_JOINTS says the same thing for the gate.
#
# The GRIPPER shares its mode with production, so it has no unit error at all. Measured on the
# arm, Sep 6 2026 -- the five arm joints differ, the gripper does not:
#     production SOFollower : shoulder_pan/lift, elbow_flex, wrist_flex/roll DEGREES; gripper RANGE_0_100
#     diagnostics _arms     : the five arm joints RANGE_M100_100;             gripper RANGE_0_100
# Its raw-to-value map is anchored at range_min rather than the midpoint, so running it through
# the arm-joint arithmetic below invents a ~60 deg error that does not exist -- and did not
# exist on the bench, where home_arm commands 1.1 and the jaw settles at 1.1.
EXEMPT_JOINTS = frozenset({"shoulder_lift", "gripper"})


def load_calibration(path) -> dict:
    return json.load(open(Path(path)))


def range_to_raw(joint: str, value: float, cal: dict) -> float:
    """A RANGE_M100_100 command -> the raw tick the motor bus will drive to. Arm joints only:
    the gripper is RANGE_0_100 on BOTH buses and is exempt (see EXEMPT_JOINTS)."""
    f = cal[joint]
    lo, hi = float(f["range_min"]), float(f["range_max"])
    return (lo + hi) / 2.0 + (value / 100.0) * ((hi - lo) / 2.0)


def raw_to_deg(joint: str, raw: float, cal: dict) -> float:
    """A raw tick -> the DEGREES value the dataset and the production follower use."""
    f = cal[joint]
    return (raw - (float(f["range_min"]) + float(f["range_max"])) / 2.0) * DEG_PER_TICK


def unit_error_deg(joint: str, value: float, cal: dict) -> float:
    """Degrees between where a table says the joint goes and where the RANGE bus puts it."""
    return raw_to_deg(joint, range_to_raw(joint, value, cal), cal) - value


def pose_unit_errors(pose: dict, cal: dict) -> dict:
    return {j: unit_error_deg(j, v, cal) for j, v in pose.items() if j in cal}


def offenders(pose: dict, cal: dict, limit: float = MAX_UNIT_ERROR_DEG) -> dict:
    """Joints whose unit error would move the arm materially, excluding the passive ones."""
    return {j: e for j, e in pose_unit_errors(pose, cal).items()
            if j not in EXEMPT_JOINTS and abs(e) > limit}


def describe(bad: dict) -> str:
    return "; ".join(f"{j} lands {e:+.1f} deg away" for j, e in sorted(bad.items(), key=lambda kv: -abs(kv[1])))
