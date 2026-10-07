"""Pre-launch gate: is the homed arm inside the pose distribution the policy
being run was trained from?

Two lessons are baked in here, both paid for on Aug 29 2026.

**1. Gate on training support, not on a residual.** `home_arm.py` warns above a
flat 2.5 deg residual. That warning was seen and narrated three times and
launched through every time. At residual 5.6 the elbow sits at 74.1 deg, below
the v2 training floor of 75.47, and a *deterministic replay* of v2 ep0 then
sailed past without touching the tube. Meanwhile the flat tolerance is TIGHTER
than the training spread (v2 elbow sd = 2.20), so it also rejects perfectly
ordinary poses. Residual answers "did the servo arrive?"; support answers "is
this a state the policy has ever seen?" Only the second one predicts behaviour.

**2. Gate against the RIGHT dataset.** The first version of this file hardcoded
v2 and was then used on v3 policies. The two differ materially:

                    v2 frame-0            v3 frame-0
    elbow_flex   75.47 – 84.44        78.11 – 90.42
    gripper       1.14 –  1.64         0.79 –  1.57

So an arm at elbow 85.30 was rejected as out-of-support when it is squarely
inside v3's range, and anything in 75.47–78.11 would have passed while being
outside everything v3 ever saw. Callers running a v3 policy MUST pass
`dataset="v3"`.

Bounds are per-joint frame-0 min/max over all 50 episodes of each dataset;
`tests/test_home_gate.py` and `tests/test_v3_support.py` re-derive them from the
parquet, so they cannot quietly go stale.

    from home_gate import check_pose
    ok, offenders = check_pose(settled_pose, dataset="v3")
"""

# The gripper is gated on CLOSED, not on its frame-0 min/max, and that is a
# measurement, not a convenience. Calibration span is 1401 raw ticks; the jaw
# bottoms out on command at raw 2074, while v3's frame-0 range 0.79-1.57 is raw
# 2059-2070 — a gap of FOUR raw ticks, 0.29% of jaw travel. The arm cannot close
# those last ticks by commanding a position because during recording the jaw is
# driven by the leader's TRIGGER, which pushes past the point where the jaws
# touch. So the frame-0 gripper distribution encodes operator pressure, not a
# reachable target — the same reason shoulder_lift is exempt for settling
# against its stop.
#
# What must still be caught is a jaw left genuinely OPEN, which changes the whole
# grasp sequence. 3.0 clears the measured contact point (~1.86-2.00) with margin
# and sits far below grasp_lock.OPEN_T = 15.0, the project's own definition of
# "open". Blocked Block 2 for half an hour on Aug 29 before this was measured.
GRIPPER_CLOSED_MAX = 3.0

# Per-joint frame-0 mean / sd / observed min-max, measured across all 50 episodes
# of each dataset. Bounds are computed from these, not hardcoded, so the
# derivation stays visible; tests re-derive them from the parquet.
FRAME0 = {
    # faithqin/so101-tube-insert-v2_20260816_035314
    "v2": {
        #                mean      sd      min       max
        "shoulder_pan": (2.46, 2.04, -3.21, 6.81),
        "shoulder_lift": (-105.21, 0.10, -105.54, -104.92),
        "elbow_flex": (79.73, 2.20, 75.47, 84.44),
        "wrist_flex": (73.41, 0.30, 71.91, 73.85),
        "wrist_roll": (-1.46, 1.50, -5.49, 2.86),
        "gripper": (1.25, 0.09, 1.14, 1.64),
    },
    # faithqin/so101-tube-insert-v3_20260827_110242
    "v3": {
        "shoulder_pan": (1.86, 1.78, -1.10, 6.11),
        "shoulder_lift": (-105.19, 0.21, -105.71, -104.04),
        "elbow_flex": (82.34, 2.83, 78.11, 90.42),
        "wrist_flex": (73.80, 0.18, 72.97, 74.20),
        "wrist_roll": (-1.90, 1.20, -4.35, 1.71),
        "gripper": (1.00, 0.11, 0.79, 1.57),
    },
    # faithqin/so101-tube-insert-v4_20260904_220924 -- the 50 TRAINING episodes only
    # (tests/conftest.v4_training_episodes); measured Sep 6 2026. Elbow +3.3 deg vs v3.
    "v4": {
        "shoulder_pan": (1.65, 1.18, -0.75, 3.91),
        "shoulder_lift": (-105.34, 0.11, -105.54, -105.01),
        "elbow_flex": (85.61, 2.74, 79.60, 91.30),
        "wrist_flex": (73.65, 0.17, 73.23, 74.02),
        "wrist_roll": (-1.71, 1.88, -7.08, 2.77),
        "gripper": (1.11, 0.25, 0.93, 2.21),
    },
    # v4-carry (Sep 6 2026): NOT a frame-0 window. The PRE-GRASPED start of the S2 stratum -- a
    # MODIFIED TASK -- is the CARRY frame (grasp_index + k, first frame after the grasp where
    # shoulder_lift has moved >= 2 deg toward lifted) over the same 50 v4 TRAINING episodes;
    # tools/v4_carry.py derives it, tests/test_v4_carry.py re-derives it from the parquet. The
    # spread is wide on purpose (the pick site follows the tube's start zone; elbow sd 13.68).
    # The gripper window is the TUBE'S WIDTH (see GRIPPER_ON_TUBE_DATASETS below).
    "v4-carry": {
        "shoulder_pan": (67.09, 3.17, 61.14, 77.93),
        "shoulder_lift": (45.35, 7.20, 23.43, 57.89),
        "elbow_flex": (-35.84, 13.68, -59.91, 10.33),
        "wrist_flex": (78.49, 7.08, 59.78, 92.75),
        "wrist_roll": (49.66, 7.78, 25.71, 60.35),
        "gripper": (13.16, 0.58, 12.13, 15.42),
    },
}

# Gate on mean +/- 1 sigma rather than the full observed range. Min/max is enough
# for VALIDITY (is this a pose the policy ever saw?) but not for CONSISTENCY,
# which a paired comparison also needs. On Aug 29 three v3 trials started at
# 78.3, 83.2 and 86.3 — all legitimately inside v3's wide [78.11, 90.42] range —
# and the 86.3 one was observed approaching too high. A pair whose two halves
# start 8 degrees apart is not controlled for start pose.
SIGMAS = 1.0

# Several joints have a tiny frame-0 spread (v3 wrist_flex sd = 0.18). A literal
# +/-1 sigma there is a 0.36 degree window no arm hits repeatably, so the gate
# would abort forever. Floor the half-width, then clip to what was observed so
# the floor can never admit a pose outside the training data.
MIN_HALF_WIDTH = 1.0


def _bounds(mean: float, sd: float, lo_obs: float, hi_obs: float) -> tuple:
    half = max(SIGMAS * sd, MIN_HALF_WIDTH)
    return (max(mean - half, lo_obs), min(mean + half, hi_obs))


# Datasets whose gate pose has the jaw CLOSED ON THE TUBE, not on nothing (Sep 6 2026, the
# S2 PRE-GRASPED stratum -- a MODIFIED TASK). The GRIPPER_CLOSED_MAX rule above is a
# frame-0 fact: at frame 0 the jaw is shut on itself. At the v4 carry frame it holds the
# tube and reads the tube's width -- 12.13-15.42 over the 50 training episodes (mean 13.16,
# sd 0.58) against 0.93-2.21 at frame 0. A 0-3.0 bound would have refused every legitimate
# S2 start on the bench. These datasets are gated on the observed carry RANGE instead:
# it still refuses an empty jaw (~1.1) and an open one (v4_audit's closed threshold is 16).
# What position cannot see -- an empty jaw parked at 13.2 in free air -- is caught by the
# contact-load check in tools/v4_carry.verify_carry_grip, which home_arm runs on this pose.
GRIPPER_ON_TUBE_DATASETS = frozenset({"v4-carry"})


def _gripper_bounds(dataset: str, stats: tuple) -> tuple:
    if dataset in GRIPPER_ON_TUBE_DATASETS:
        _mean, _sd, lo_obs, hi_obs = stats
        return (lo_obs, hi_obs)
    return (0.0, GRIPPER_CLOSED_MAX)


SUPPORT_BY_DATASET = {
    ds: {
        j: _gripper_bounds(ds, stats) if j in ("gripper",) else _bounds(*stats)
        for j, stats in joints.items()
    }
    for ds, joints in FRAME0.items()
}

DEFAULT_DATASET = "v2"

# Backwards-compatible alias: the v2 table, which is what callers got before the
# gate became dataset-aware.
SUPPORT = SUPPORT_BY_DATASET[DEFAULT_DATASET]

# shoulder_lift's training home (~-105.2) sits BEYOND the -100 goal clamp, so it
# is never commanded there: it settles passively against its mechanical stop and
# reads ~5-6 deg off by design. Gating it on training support would abort every
# launch. home_arm.py makes the same exemption.
PASSIVE_SETTLE_JOINTS = frozenset({"shoulder_lift"})

# ...except at the carry pose, where shoulder_lift is +45.35 (sd 7.20): far inside the goal
# clamp, actively holding the extended arm, nothing to settle against. There it is the joint
# that DEFINES "lifted", so it is gated like any other, and tools/v4_carry.settle_joints_for
# keeps both homing paths from releasing it (the arm would fall, holding the tube).
LIFT_ACTIVE_DATASETS = frozenset({"v4-carry"})


def passive_joints(dataset: str = DEFAULT_DATASET) -> frozenset:
    return frozenset() if dataset in LIFT_ACTIVE_DATASETS else PASSIVE_SETTLE_JOINTS


# Joints gated on a CONTACT bound rather than on their frame-0 min/max, because
# their recorded start value reflects contact force rather than a commandable
# position (see the GRIPPER_CLOSED_MAX note above). Tests that verify bounds
# against the parquet must skip these, or they will demand a range the arm
# physically cannot command its way into.
CONTACT_SETTLE_JOINTS = frozenset({"gripper"})


def support_for(dataset: str = DEFAULT_DATASET) -> dict:
    if dataset not in SUPPORT_BY_DATASET:
        raise KeyError(f"unknown dataset {dataset!r}; expected one of {sorted(SUPPORT_BY_DATASET)}")
    return SUPPORT_BY_DATASET[dataset]


def in_training_support(joint: str, value: float, dataset: str = DEFAULT_DATASET) -> bool:
    """Is this joint's homed position one the policy saw at episode start?

    Raises KeyError on an unknown joint or dataset — a typo must abort the
    launch, not silently pass it.
    """
    table = support_for(dataset)
    if joint not in table:
        raise KeyError(f"unknown joint {joint!r}; expected one of {sorted(table)}")
    if joint in passive_joints(dataset):
        return True
    lo, hi = table[joint]
    return lo <= value <= hi


def check_pose(pose: dict, dataset: str = DEFAULT_DATASET) -> tuple[bool, dict]:
    """Gate a whole settled pose. Returns (ok, {joint: (value, lo, hi)})."""
    table = support_for(dataset)
    offenders = {}
    for joint, value in pose.items():
        if not in_training_support(joint, value, dataset=dataset):
            offenders[joint] = (value, *table[joint])
    return (not offenders), offenders


def describe(offenders: dict) -> str:
    return "; ".join(
        f"{j}={v:.2f} outside [{lo:.2f}, {hi:.2f}]" for j, (v, lo, hi) in sorted(offenders.items())
    )


if __name__ == "__main__":
    import json
    import sys

    ds = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DATASET
    ok, bad = check_pose(json.loads(sys.argv[1]), dataset=ds)
    print(f"HOME_GATE PASS ({ds})" if ok else f"HOME_GATE FAIL ({ds}) — {describe(bad)}")
    sys.exit(0 if ok else 1)
