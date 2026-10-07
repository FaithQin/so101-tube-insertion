"""Regression: gate the gripper on CLOSED, not on its frame-0 min/max.

Born from Aug 29, 23:45, after the gate blocked Block 2 for half an hour.

The measurement that settles it. Gripper calibration span is 1401 raw ticks
(range_min 2048, range_max 3449). The jaw bottoms out at raw 2074. v3's frame-0
range 0.79-1.57 is raw 2059-2070. **The gap is 4 raw ticks — 0.286% of jaw
travel.**

Why the arm cannot close those last 4 ticks on command: during recording the
follower's jaw mirrors the LEADER's trigger, and squeezing the trigger drives it
past the point where the jaws touch. A position command stops at contact. So the
frame-0 gripper distribution encodes operator trigger PRESSURE, not a position
any `home_arm` command can reproduce.

That makes the gripper structurally identical to shoulder_lift, which is already
exempt because it settles passively against its stop. Both record a contact
point rather than a commanded target, so min/max over training episodes is the
wrong statistic for both.

What still needs catching: a jaw left genuinely OPEN at episode start. That is a
real out-of-distribution state — the policy opens to grasp, so starting open
changes the whole grasp sequence. Hence a loose closed-bound rather than an
exemption.
"""

import sys

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))

import home_gate  # noqa: E402

MECHANICAL_FLOOR = 1.86  # measured: where the jaw stops on command, raw 2074


def test_the_measured_mechanical_floor_passes():
    """The exact value that blocked Block 2 for half an hour."""
    for ds in ("v2", "v3"):
        assert home_gate.in_training_support("gripper", MECHANICAL_FLOOR, dataset=ds), (
            f"{ds}: the jaw's own contact point must not be treated as out of distribution — "
            f"it is 4 raw ticks (0.29% of travel) from the frame-0 max"
        )
    assert home_gate.in_training_support("gripper", 2.00, dataset="v3")


def test_training_values_still_pass():
    for v in (0.79, 1.00, 1.25, 1.57, 1.64):
        assert home_gate.in_training_support("gripper", v, dataset="v3")


def test_an_open_jaw_still_fails():
    """The thing worth catching: the policy opens to grasp, so starting open
    changes the whole grasp sequence. grasp_lock.OPEN_T is 15.0."""
    for v in (8.0, 15.0, 20.0, 40.0):
        assert not home_gate.in_training_support("gripper", v, dataset="v3"), (
            f"gripper={v} is an open jaw and must still abort the launch"
        )


def test_threshold_sits_between_the_floor_and_open():
    lo, hi = home_gate.SUPPORT_BY_DATASET["v3"]["gripper"]
    assert hi > MECHANICAL_FLOOR, "bound must clear the measured contact point"
    assert hi < 15.0, "bound must stay well below grasp_lock's OPEN_T=15"


def test_gripper_is_not_gated_on_the_tight_frame0_range():
    """1.64 was the v2 frame-0 max and 1.57 the v3 max. Gating on those is what
    rejected a jaw sitting 4 ticks away."""
    _, hi = home_gate.SUPPORT_BY_DATASET["v3"]["gripper"]
    assert hi not in (1.57, 1.64)


def test_other_joints_keep_their_tight_dataset_bounds():
    """This must not become a general loosening. The elbow gate is the one that
    caught a real failure — 74.1 deg made a deterministic replay miss entirely."""
    assert not home_gate.in_training_support("elbow_flex", 74.1, dataset="v2")
    assert not home_gate.in_training_support("elbow_flex", 76.0, dataset="v3")
    # Since the Aug 29 tightening the elbow bound is mean +/- 1 sigma, not the
    # full observed range — but it is still a real, narrow, dataset-derived gate.
    lo, hi = home_gate.SUPPORT_BY_DATASET["v3"]["elbow_flex"]
    assert 79.0 < lo < 80.0 and 85.0 < hi < 85.5, (lo, hi)
