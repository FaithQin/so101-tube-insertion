"""Regression: the pre-launch home gate must be derived from the dataset, and
must reject start poses outside the training distribution.

Born from the Aug 29 2026 session. `home_arm.py` warns when a joint's residual
exceeds a flat 2.5 deg, and that warning was repeatedly observed and then
ignored — three policy episodes and one priming replay were run on an arm whose
elbow sat 5.6 deg low, i.e. at 74.1 deg, BELOW the v2 training floor of 75.47.
The deterministic replay then sailed past without touching the tube at all,
which is what finally exposed it.

Two things this file pins down:

1. The gate is expressed in TRAINING SUPPORT, not in an arbitrary residual.
   A flat 2.5 deg tolerance is tighter than the training data's own spread
   (elbow frame-0 sd is 2.20), so it rejects perfectly ordinary start poses
   while saying nothing about whether the pose is one the policy ever saw.
2. The bounds in the tool match the dataset that is actually on disk. If the
   dataset is re-recorded, this test fails rather than the gate silently
   admitting out-of-distribution poses.
"""

import sys

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))

import home_gate  # noqa: E402

# Measured on the arm, Aug 29 2026: home_arm left the elbow here and the
# deterministic replay of v2 ep0 missed the tube entirely.
ELBOW_OBSERVED_FAILURE = 74.1
# An ordinary v2 start INSIDE one sigma of the mean (79.73 +/- 2.20). 77.1 was
# used here before the Aug 29 tightening; it sits 1.2 sigma low and is now
# correctly rejected, so the "ordinary pose must pass" case moved to the mean.
ELBOW_OBSERVED_PASS = 79.7


def test_elbow_failure_pose_is_rejected():
    assert not home_gate.in_training_support("elbow_flex", ELBOW_OBSERVED_FAILURE), (
        f"elbow {ELBOW_OBSERVED_FAILURE} is below the v2 training floor and produced a "
        f"total replay miss on Aug 29 — the gate must reject it"
    )


def test_elbow_ordinary_low_pose_is_accepted():
    assert home_gate.in_training_support("elbow_flex", ELBOW_OBSERVED_PASS), (
        f"elbow {ELBOW_OBSERVED_PASS} is the v2 frame-0 mean — rejecting it would be "
        f"the flat-2.5-deg mistake in a new costume"
    )


def test_bounds_match_the_dataset_on_disk(v2_frame0_stats):
    """The gate's numbers must come from the dataset, not from memory. Since the
    Aug 29 tightening the bound is mean +/- 1 sigma (floored, then clipped to the
    observed range), so the MEAN and SD are what must match the parquet."""
    for joint, mean in v2_frame0_stats["mean"].items():
        if joint in home_gate.PASSIVE_SETTLE_JOINTS | home_gate.CONTACT_SETTLE_JOINTS:
            continue
        m, sd, lo_obs, hi_obs = home_gate.FRAME0["v2"][joint]
        assert abs(m - mean) < 0.05, f"{joint}: gate mean {m} vs dataset {mean:.2f}"
        assert abs(sd - v2_frame0_stats["std"][joint]) < 0.05
        assert abs(lo_obs - v2_frame0_stats["min"][joint]) < 0.05
        assert abs(hi_obs - v2_frame0_stats["max"][joint]) < 0.05


def test_boundaries_are_inclusive_and_tight(v2_frame0_stats):
    lo, hi = home_gate.SUPPORT["elbow_flex"]
    assert home_gate.in_training_support("elbow_flex", lo)
    assert home_gate.in_training_support("elbow_flex", hi)
    assert not home_gate.in_training_support("elbow_flex", lo - 0.01)
    assert not home_gate.in_training_support("elbow_flex", hi + 0.01)


def test_passive_settle_joint_is_exempt():
    """shoulder_lift's training home (-105.2) sits beyond the -100 goal clamp, so
    it settles against its stop and always reads ~5-6 deg off. Gating it on
    training support would abort every launch."""
    assert "shoulder_lift" in home_gate.PASSIVE_SETTLE_JOINTS
    assert home_gate.in_training_support("shoulder_lift", -99.3)


def test_unknown_joint_is_not_silently_accepted():
    try:
        home_gate.in_training_support("elbow", 79.0)
    except KeyError:
        return
    raise AssertionError("a misspelled joint name must raise, not pass the gate")


def test_check_pose_reports_the_offending_joint():
    ok, bad = home_gate.check_pose(
        {"elbow_flex": ELBOW_OBSERVED_FAILURE, "wrist_flex": 73.4, "shoulder_lift": -99.3}
    )
    assert not ok
    assert "elbow_flex" in bad


def test_check_pose_passes_a_good_home():
    ok, bad = home_gate.check_pose(
        {
            "shoulder_pan": 2.5,
            "shoulder_lift": -99.3,
            "elbow_flex": ELBOW_OBSERVED_PASS,
            "wrist_flex": 73.4,
            "wrist_roll": -1.5,
            "gripper": 1.25,
        }
    )
    assert ok, f"a clean home was rejected: {bad}"
