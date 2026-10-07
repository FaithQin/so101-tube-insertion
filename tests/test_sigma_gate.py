"""Regression: gate on +/-1 sigma of the frame-0 mean, not the full min/max.

Born from Aug 29, 23:50. The min/max gate was correct but too permissive to
support a PAIRED comparison. v3's own frame-0 elbow spread is [78.11, 90.42]
(sd 2.83), so the three v3 trials that ran started at 78.3, 83.2 and 86.3 — an
8 degree spread — and every one of them passed the gate legitimately. Faith:
"I thought checking home position was supposed to solve this."

It did solve validity. It does not solve CONSISTENCY, and a paired v3-A vs v3-B
comparison needs both: if A starts at 78.3 and B starts at 86.3, the pair is not
controlled for start pose and the contrast is contaminated by it. The 86.3 trial
is the one Faith observed approaching too high.

Design note — why not a flat +/-1 sigma everywhere: several joints have a tiny
frame-0 spread (v3 wrist_flex sd = 0.18), and +/-1 sigma there is a 0.36 degree
window no arm can hit repeatably, so the gate would abort forever. The half-width
is therefore max(1 sigma, MIN_HALF_WIDTH), and is additionally clipped to the
observed min/max so the gate can never admit a pose the policy never saw.

Limit worth stating plainly: a tighter gate produces MORE ABORTS on a drifting
joint, not better trials. Software can refuse a bad start; it cannot make a
wandering elbow land where it is told.
"""

import sys

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))

import home_gate  # noqa: E402

# The three v3 trials that ran on Aug 29, from their saved .home files.
TRIAL_LOW, TRIAL_MID, TRIAL_HIGH = 78.3, 83.2, 86.3


def test_the_high_trial_would_now_be_rejected():
    """B2-A-02 started at 86.3 and was observed approaching too high."""
    assert not home_gate.in_training_support("elbow_flex", TRIAL_HIGH, dataset="v3")


def test_the_low_trial_would_now_be_rejected():
    """B2-A-01 started at 78.3, at v3's floor — the other end of the 8 deg spread."""
    assert not home_gate.in_training_support("elbow_flex", TRIAL_LOW, dataset="v3")


def test_the_mid_trial_is_still_accepted():
    assert home_gate.in_training_support("elbow_flex", TRIAL_MID, dataset="v3")


def test_the_mean_is_accepted(v3_frame0_stats):
    assert home_gate.in_training_support(
        "elbow_flex", v3_frame0_stats["mean"]["elbow_flex"], dataset="v3"
    )


def test_bounds_are_one_sigma_around_the_mean(v3_frame0_stats):
    lo, hi = home_gate.SUPPORT_BY_DATASET["v3"]["elbow_flex"]
    mean = v3_frame0_stats["mean"]["elbow_flex"]
    sd = v3_frame0_stats["std"]["elbow_flex"]
    assert abs(lo - (mean - sd)) < 0.1, f"low bound {lo} is not mean-1sd ({mean - sd:.2f})"
    assert abs(hi - (mean + sd)) < 0.1, f"high bound {hi} is not mean+1sd ({mean + sd:.2f})"


def test_tight_joints_get_a_minimum_window(v3_frame0_stats):
    """v3 wrist_flex sd is 0.18. A literal +/-1 sigma window is 0.36 deg wide and
    would abort every launch; observed homes were 73.4-73.9."""
    lo, hi = home_gate.SUPPORT_BY_DATASET["v3"]["wrist_flex"]
    # The window is floored at MIN_HALF_WIDTH and THEN clipped to what was
    # observed, so it can end up narrower than 2*MIN_HALF_WIDTH. What matters is
    # that it is far wider than a literal +/-1 sigma (0.36 deg) and admits the
    # poses the arm actually homes to.
    assert hi - lo > 4 * v3_frame0_stats["std"]["wrist_flex"]
    for observed in (73.4, 73.8, 73.9):
        assert home_gate.in_training_support("wrist_flex", observed, dataset="v3")


def test_bounds_never_exceed_what_was_actually_observed(v3_frame0_stats):
    """The minimum-window floor must not let the gate admit a pose outside the
    training data entirely."""
    for joint in ("elbow_flex", "wrist_flex", "wrist_roll", "shoulder_pan"):
        lo, hi = home_gate.SUPPORT_BY_DATASET["v3"][joint]
        assert lo >= v3_frame0_stats["min"][joint] - 0.02
        assert hi <= v3_frame0_stats["max"][joint] + 0.02


def test_out_of_support_poses_are_still_rejected():
    """The original catch must survive the tightening: elbow 74.1 made a
    deterministic replay miss the tube entirely."""
    assert not home_gate.in_training_support("elbow_flex", 74.1, dataset="v2")
    assert not home_gate.in_training_support("elbow_flex", 84.9, dataset="v2")


def test_gripper_still_uses_the_contact_bound():
    """The gripper is gated on closed-jaw, not on sigma — its frame-0 values
    encode trigger pressure, not a commandable position."""
    assert home_gate.in_training_support("gripper", 1.86, dataset="v3")
    assert not home_gate.in_training_support("gripper", 15.0, dataset="v3")
