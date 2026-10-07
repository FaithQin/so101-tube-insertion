#!/usr/bin/env python
"""Drive the follower to the TRAINING home pose, verified. Bus-only, no cameras.

Why: lerobot-rollout captures "initial_position" from whatever pose the arm
holds at connect. Our procedure runs warmup replays first, which end at the
rack and drop torque — so the rollout's notion of "home" can be a slumped
arbitrary pose, and every between-episode reset then returns to THAT
(observed Aug 15: episode 2+ started from a hover, overreached into walls).
Fix: home the arm to the pose the dataset actually starts from, measured
across training episodes' frame-0 states:

    pan 2.4  lift -105.2  elbow 82.9  wrist_flex 73.4  wrist_roll -2.2  gripper 1.1

Run BETWEEN warmup replays and the rollout command:
    python tools/home_arm.py
"""

import argparse
import time

from _arms import FOLLOWER_CAL, FOLLOWER_PORT, JOINTS, open_bus
import v4_carry  # the PRE-GRASPED pose's scripted close + settle rule (bus passed in, never opened there)

# Frame-0 pose per training dataset, measured across every episode's first
# frame (v1: Aug 15; v2: Aug 16 — parquet-derived, see
# tests/test_home_pose_matches_training.py, which enforces this table).
HOME_POSES = {
    "v1": [2.4, -105.2, 82.9, 73.4, -2.2, 1.1],
    "v2": [2.5, -105.2, 79.7, 73.4, -1.5, 1.3],
    # v3 (Aug 27 dataset), measured across all 50 episodes' frame 0. NOT the
    # same start as v2: the elbow mean differs by 2.6 deg (82.3 vs 79.7), more
    # than one sd on the joint with the widest start spread, and the gripper
    # sits lower (1.00 vs 1.25). Homing a v3 policy to the v2 pose starts it
    # outside its own training mean — caught Aug 29 when a v2-based gate
    # rejected an elbow of 85.30 that is squarely inside v3's range.
    "v3": [1.9, -105.2, 82.3, 73.8, -1.9, 1.0],
    # v4 (Sep 4-5 dataset), frame-0 means over its 50 TRAINING episodes (Sep 6 2026).
    # Elbow 85.6 -- 3.3 deg above v3; gripper 1.1. See tests/test_v4_support.py.
    "v4": [1.6, -105.3, 85.6, 73.7, -1.7, 1.1],
    # v4-carry (Sep 6 2026) -- the PRE-GRASPED start, a MODIFIED TASK: the tube is placed in the
    # open jaws by hand and the jaws close on it before the policy runs (S2 stratum). NOT a frame-0
    # pose: it is the mean CARRY frame (grasp_index + k, the first frame after the grasp where
    # shoulder_lift has moved >= 2 deg toward lifted) over the 50 v4 TRAINING episodes --
    # tools/v4_carry.py derives it; tests/test_v4_carry.py re-derives it from the parquet.
    # The gripper (13.2) is the TUBE'S WIDTH, not a closed jaw; --close-on-tube is mandatory.
    "v4-carry": [67.1, 45.3, -35.8, 78.5, 49.7, 13.2],
}

parser = argparse.ArgumentParser()
parser.add_argument("--pose", choices=sorted(HOME_POSES), default="v2",
                    help="which dataset's frame-0 pose to home to (default: v2)")
parser.add_argument("--no-compensate", action="store_true",
                    help="skip the connect-shift compensation (for a standalone "
                         "diagnostic home that is NOT followed by a rollout)")
parser.add_argument("--close-on-tube", action="store_true",
                    help="v4-carry only: after homing, open the jaws, wait for the operator to "
                         "place the tube, close on it under the follower's bounded lead, and "
                         "REFUSE (exit 1, no 'settled:' line) unless position AND contact load "
                         "match the training carry frames (tools/v4_carry.py)")
args = parser.parse_args()
# The carry pose with an EMPTY jaw parks the gripper at 13.2 -- inside the carry gate's band
# and indistinguishable from a held tube by position alone -- so it is not a pose this script
# will hand to preflight. The close is the only way in, and the only pose it applies to.
if (args.pose == v4_carry.TAG) != args.close_on_tube:
    parser.error(f"--close-on-tube is required for, and only for, --pose {v4_carry.TAG} "
                 f"(the PRE-GRASPED modified task); got --pose {args.pose}")

# UNIT GUARD (Sep 6 2026, found by the bench-morning review and confirmed on the arm).
# This script drives through tools/_arms.py, which normalizes RANGE_M100_100, while every pose
# table here is derived from the DATASET, which is DEGREES. The two modes share the calibration
# midpoint as their zero but not their scale, so the error grows with distance from it: the
# frame-0 poses (near the midpoint) land within a degree, but the extended v4-carry pose puts
# wrist_roll about 40 deg from where the table says -- with the operator's hand reaching in to
# place a tube. Refuse here, during argument validation, so nothing can open the bus. The real
# fix is tools/_arms.py in the production norm mode
# (tests/test_arms_norm_mode_matches_production.py, xfail(strict)), which changes v2/v3/v4
# homing too and needs its own bench verification. See tools/norm_units.py.
import norm_units  # noqa: E402  (pure: arithmetic on the saved calibration, no bus)

_bad = norm_units.offenders(dict(zip(JOINTS, HOME_POSES[args.pose])),
                            norm_units.load_calibration(FOLLOWER_CAL))
if _bad:
    parser.error(
        f"--pose {args.pose} cannot be commanded through this bus: {norm_units.describe(_bad)}. "
        f"tools/_arms.py normalizes RANGE_M100_100 but the pose table is in DEGREES, and the "
        f"error grows with distance from the calibration midpoint. Fix _arms.py's norm mode "
        f"(tests/test_arms_norm_mode_matches_production.py) and re-verify homing on the bench "
        f"before using this pose."
    )
print(f"homing to {args.pose} training pose")

# `SOFollower.connect()` itself moves the arm — wrist_roll by about -1.62 deg,
# measured Aug 30 2026. The table lives in home_arm_shift so this script and the
# preflight gate cannot drift apart: home_arm drives to HOME - shift so the arm
# lands on HOME after connect, and the gate adds the shift back so it judges the
# pose the POLICY will observe. See home_arm_shift.py for the measurements and
# for why shoulder_lift is excluded.
from home_arm_shift import CONNECT_SHIFT  # noqa: E402

HOME = dict(zip(JOINTS, HOME_POSES[args.pose]))
if not args.no_compensate:
    # post = pre + shift, so drive to HOME - shift to land on HOME after connect.
    HOME = {j: v - CONNECT_SHIFT.get(j, 0.0) for j, v in HOME.items()}
# shoulder_lift's training home (-105.2) sits BEYOND the -100 goal clamp — the
# arm settles there passively against its stop, so its residual runs ~5.
TOL = {"shoulder_lift": 7.0}
TOL_DEFAULT = 2.5
DURATION_S, FPS = 3.0, 50

bus = open_bus(FOLLOWER_PORT, FOLLOWER_CAL)
try:
    bus.enable_torque()
except Exception:
    for j in JOINTS:
        bus.write("Torque_Enable", j, 1, normalize=False)

cur = {j: float(bus.read("Present_Position", j, normalize=True)) for j in JOINTS}
print("current:", {j: round(v, 1) for j, v in cur.items()})

steps = int(DURATION_S * FPS)
for s in range(1, steps + 1):
    t = s / steps
    for j in JOINTS:
        bus.write("Goal_Position", j, cur[j] * (1 - t) + HOME[j] * t, normalize=True)
    time.sleep(1 / FPS)

# ---------------------------------------------------------------------------
# Anti-backlash approach. Measured Aug 30 2026 on the follower at 45 C: the same
# elbow command lands 1.5 deg apart depending on approach direction —
# 79.7 -> 80.1 coming up from 74.0, but 79.7 -> 81.7 coming down from 86.7.
# That is backlash plus friction (published STS3215: 0.87 deg backlash + 0.88 deg
# dead zone), not a fault. Because this routine interpolates from wherever the
# arm is left by the previous episode, the landing point drifted with it: across
# Aug 29 the elbow homed to 74.1 / 77.1 / 78.3 / 82.1 / 83.2 / 84.9 / 85.3 / 86.3
# and was very nearly diagnosed as a failing servo.
#
# Fix: undershoot past the target, let it settle, then come back UP to it, so
# every home approaches from the same side. shoulder_lift is excluded — it
# settles against a mechanical stop beyond the goal clamp, so undershooting it
# only fights the stop.
BACKLASH_MARGIN = 3.0

# Undershooting is only half the technique — the return has to be driven until
# it ARRIVES. Writing the target once left wrist_flex 1.0 deg short (72.80 for a
# 73.80 target), which is inside this routine's own 2.5 residual tolerance, so it
# reported HOMED while the pose was outside the training band. Converge instead.
BACKLASH_SETTLE_TOL = 0.3
BACKLASH_RETURN_SETTLE_S = 0.6
BACKLASH_DAMP = 0.5

_backlash_joints = [j for j in JOINTS if j != "shoulder_lift"]

# WHICH SIDE to approach from is per-joint, and for the elbow it flipped on Sep 6 2026.
# The Aug 30 rule above (undershoot, then come back UP) assumes the joint can LIFT into its
# target. The elbow's gearbox has since worn -- moved by hand it steps in distinct notches, one
# per gear tooth, unlike every other joint -- and it can no longer lift into the v4 target
# against gravity. Measured today: six consecutive homes approaching from below landed 82.3-83.6
# for an 85.6 target, and the motor stalled there at load 550 / current 245 without moving, at
# both raised proportional and raised integral gain. A home that DESCENDED from 90.2 landed at
# 85.8 -- inside the training window, holding at load 44 -- and passed the preflight gate for the
# first time that day. So the elbow approaches from ABOVE, where gravity does the last degree of
# work, and every other joint keeps the Aug 30 from-below approach.
# This compensates for the fault; it does not repair it. The servo still droops ~1 deg more than
# it did when v4 was recorded (the lab notebook (private), Sep 6), which no homing change can fix.
# wrist_flex joins it for the same reason at a smaller scale: from below it lands 73.20 against a
# 73.23 window floor -- three consecutive homes, missing by 0.03 deg -- while a descent puts it
# mid-window. Its training sd is 0.17 deg, the tightest joint on the arm, so there is no slack.
APPROACH_FROM_ABOVE = {"elbow_flex", "wrist_flex"}


def _approach_from(j):
    """The pre-position the joint returns to its target from."""
    return HOME[j] + BACKLASH_MARGIN if j in APPROACH_FROM_ABOVE else HOME[j] - BACKLASH_MARGIN


# 1. Overshoot past the target so the approach always comes from one side.
for j in _backlash_joints:
    bus.write("Goal_Position", j, _approach_from(j), normalize=True)
time.sleep(0.6)

# 2. RETURN to the target and let it settle. This step is not optional: without
# it the correction below reads the arm at the undershoot position, computes a
# 3 deg error, and commands 3 deg PAST the target — then back — oscillating.
# Measured Aug 30 on wrist_roll, which tracks commands to +/-0.2 deg and still
# homed to -3.6 against a -1.9 target because of exactly that oscillation.
BACKLASH_RETURN = True
for j in _backlash_joints:
    bus.write("Goal_Position", j, HOME[j], normalize=True)
time.sleep(BACKLASH_RETURN_SETTLE_S)

# 3. Only now, nudge any joint still short — damped, and ONLY the short ones,
# because re-commanding a joint already on target can push it off.
for _ in range(4):
    actual = {j: float(bus.read("Present_Position", j, normalize=True)) for j in _backlash_joints}
    short = [j for j in _backlash_joints if abs(actual[j] - HOME[j]) > BACKLASH_SETTLE_TOL]
    if not short:
        break
    for j in short:
        err = HOME[j] - actual[j]
        bus.write("Goal_Position", j, HOME[j] + BACKLASH_DAMP * err, normalize=True)
    time.sleep(0.45)

time.sleep(0.4)  # settle
worst = 0.0
for attempt in range(2):
    residuals = {
        j: float(bus.read("Present_Position", j, normalize=True)) - HOME[j] for j in JOINTS
    }
    worst = max(abs(r) - (TOL.get(j, TOL_DEFAULT) - TOL_DEFAULT) for j, r in residuals.items())
    if worst <= TOL_DEFAULT:
        break
    # corrective pass on stragglers
    for j, r in residuals.items():
        if abs(r) > TOL.get(j, TOL_DEFAULT):
            bus.write("Goal_Position", j, HOME[j], normalize=True)
    time.sleep(0.6)

print("residuals:", {j: round(r, 1) for j, r in residuals.items()})
print(f"HOMED (worst adjusted residual {worst:.1f})" if worst <= TOL_DEFAULT
      else f"WARNING: adjusted residual {worst:.1f} exceeds {TOL_DEFAULT} — check for obstruction")

# PRE-GRASPED (v4-carry, a MODIFIED TASK): the tube goes into the open jaws by hand, the jaws
# close on it, and only a jaw that reads the tube's width AND contact load gets a 'settled:'
# line. run_scored_trial.sh ignores this script's exit code and preflight keys off 'settled:',
# so a refusal must leave that line unprinted -- exit 1 here, before it. The arm keeps holding.
if args.close_on_tube:
    import sys
    _pose_now = {j: float(bus.read("Present_Position", j, normalize=True)) for j in JOINTS}
    if not args.no_compensate:
        from home_arm_shift import predict_post_connect
        _pose_now = predict_post_connect(_pose_now)   # judge the pose the POLICY will see
    try:
        _grip = v4_carry.place_and_close(bus, _pose_now, prompt=v4_carry.operator_prompt)
    except v4_carry.CarryRefused as exc:
        print(f"CARRY REFUSED: {exc}")
        bus.disconnect(disable_torque=False)
        sys.exit(1)
    print(f"carry grip: gripper {_grip.position:.2f} |load| {_grip.load:.0f} "
          f"(training 12.13-15.42, |load| >= {v4_carry.CARRY_LOAD_MIN:.0f})")

# Settle ONLY shoulder_lift against its stop (the true training start,
# -105.2, sits beyond the -100 command clamp). Every other joint keeps torque
# and holds its target: training frame-0s were captured under teleop with
# torque ON, and a full-bus release lets the unpowered elbow sag out of
# training support (measured 88.4 vs training range 75.5–84.4, Aug 16).
# At the carry pose shoulder_lift (+45) has no stop under it: settle_joints_for empties this.
SETTLE_JOINTS = ("shoulder_lift",)
import contextlib
with contextlib.suppress(Exception):
    for j in v4_carry.settle_joints_for(args.pose, SETTLE_JOINTS):
        bus.write("Torque_Enable", j, 0, normalize=False)
    time.sleep(0.5)
    final = {j: float(bus.read("Present_Position", j, normalize=True)) for j in JOINTS}
    print("settled:", {j: round(v, 1) for j, v in final.items()})
# disable_torque defaults to True on disconnect — that full release is the
# OTHER half of the Aug 16 elbow-sag bug. Keep the held joints holding.
bus.disconnect(disable_torque=False)
