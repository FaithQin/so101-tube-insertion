"""Regression: home the arm from a CONSISTENT direction, or backlash decides
where it lands.

Measured on the follower, Aug 30 2026 ~00:05, elbow_flex at a steady 45 C:

    commanded   actual   error   approached from
       70.0      74.0    +4.0    (range limit)
       79.7      80.1    +0.4    BELOW  (came up from 74.0)
       95.0      86.7    -8.3    (range limit)
       79.7      81.7    +2.0    ABOVE  (came down from 86.7)
       88.0      85.4    -2.6    (range limit)
       79.7      81.6    +1.9    ABOVE  (came down from 85.4)

The same command lands **1.5 degrees apart** depending on which side it is
approached from. That is backlash plus friction, and it is not a fault — the
published STS3215 figure is 0.87 deg of backlash plus a 0.88 deg firmware dead
zone, so a ~1.5 deg spread is exactly what the hardware does.

What it explains: `home_arm` interpolates from wherever the arm happens to be
sitting. An episode that ends high approaches home from above and lands ~81.7;
one that ends low approaches from below and lands ~80.1. Across Aug 29 the elbow
"wandered" 74.1 / 77.1 / 78.3 / 82.1 / 83.2 / 84.9 / 85.3 / 86.3 and was
diagnosed as a failing servo. It was the homing routine all along.

It also explains why v3's frame-0 elbow spread (sd 2.83) is wider than v2's
(2.20): teleop reached the start pose from varying directions during recording.

The fix is the standard one: undershoot past the target, then come back up to it,
so every home approaches from the same side and lands in the same place.
"""

import re

from conftest import TOOLS

SRC = (TOOLS / "home_arm.py").read_text()


def test_home_arm_has_an_anti_backlash_approach():
    assert "APPROACH_FROM_BELOW" in SRC or "BACKLASH" in SRC, (
        "home_arm interpolates from the current pose, so the landing point depends "
        "on which side it comes from — measured 1.5 deg of spread on the elbow"
    )


def test_the_undershoot_margin_exceeds_the_measured_hysteresis():
    """Undershooting by less than the hysteresis does not clear the backlash."""
    m = re.search(r"BACKLASH_MARGIN\s*=\s*([\d.]+)", SRC)
    assert m, "no BACKLASH_MARGIN constant in home_arm.py"
    assert float(m.group(1)) >= 1.5, (
        f"margin {m.group(1)} does not clear the measured 1.5 deg elbow hysteresis"
    )


def test_the_approach_runs_before_the_residual_check():
    """Measuring the residual before the anti-backlash pass would report the
    pre-correction pose and re-introduce the bug the gate exists to catch."""
    approach = SRC.index("BACKLASH_MARGIN")
    residual = SRC.index("residuals:")
    assert approach < residual, "anti-backlash pass must precede the residual report"


def test_passive_settle_joint_is_excluded():
    """shoulder_lift settles against its mechanical stop beyond the goal clamp.
    Undershooting it just fights the stop."""
    block = SRC[SRC.index("BACKLASH_MARGIN"):]
    assert "SETTLE_JOINTS" in block or "shoulder_lift" in block, (
        "the anti-backlash pass must skip the joint that settles against a stop"
    )


def test_measured_evidence_is_recorded_in_the_source():
    """The next person to touch this needs the numbers, not the conclusion."""
    assert "backlash" in SRC.lower()
    assert "1.5" in SRC or "81.7" in SRC


def test_the_return_converges_rather_than_being_commanded_once():
    """The first version of the anti-backlash pass undershot 3 deg and then wrote
    the target once. On wrist_flex that landed 1.0 deg short (72.80 against a
    73.80 target) — inside home_arm's own 2.5 residual tolerance, so it reported
    HOMED, while the gate correctly refused the pose. Undershooting is only half
    the technique; the return has to be driven until it arrives."""
    assert "BACKLASH_SETTLE_TOL" in SRC, "no convergence tolerance for the return pass"
    m = re.search(r"BACKLASH_SETTLE_TOL\s*=\s*([\d.]+)", SRC)
    assert float(m.group(1)) <= 0.5, (
        f"return tolerance {m.group(1)} is looser than the gate's tightest band and "
        f"would let the same 1.0 deg shortfall through"
    )
    # and it must actually iterate
    block = SRC[SRC.index("BACKLASH_MARGIN"):SRC.index("residuals:")]
    assert "for" in block and "range" in block, "the return pass does not iterate"


def test_the_return_to_target_precedes_any_correction():
    """Measured Aug 30: wrist_roll tracks commands to +/-0.2 deg, yet homed to
    -3.6 against a -1.9 target. Cause: the correction loop read the arm right
    after the 3 deg undershoot, computed err=3.0, and commanded target+3.0 — a
    full overshoot the other way, then back, oscillating. The plain return to
    target must happen and settle BEFORE any error-proportional correction."""
    block = SRC[SRC.index("BACKLASH_MARGIN"):SRC.index("residuals:")]
    undershoot = block.index("HOME[j] - BACKLASH_MARGIN")
    plain_return = block.index("BACKLASH_RETURN = True")  # the operative line
    correction = block.index("BACKLASH_DAMP * err")  # the usage, not the constant
    assert undershoot < plain_return < correction, (
        "order must be: undershoot -> plain return -> damped correction"
    )


def test_the_correction_is_damped():
    """Gain 1.0 on the error means commanding target+err, which is a full
    reflection about the target and oscillates rather than converging."""
    m = re.search(r"BACKLASH_DAMP\s*=\s*([\d.]+)", SRC)
    assert m, "no damping factor on the correction"
    assert 0 < float(m.group(1)) < 1.0, f"damping {m.group(1)} does not converge"


def test_only_short_joints_are_corrected():
    """Re-commanding a joint that is already on target can push it off. The
    first version wrote a correction to every joint on every pass."""
    block = SRC[SRC.index("BACKLASH_DAMP"):SRC.index("residuals:")]
    assert "for j in short" in block, "correction must apply only to joints that are short"
