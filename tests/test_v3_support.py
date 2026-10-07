"""Regression: a policy must be homed and gated against ITS OWN dataset.

Born from Aug 29, 23:40. The preflight gate was built from v2 frame-0 statistics
and then used to gate v3 policy trials. Measured on the two datasets:

                    v2 frame-0            v3 frame-0
    elbow_flex   75.47 – 84.44        78.11 – 90.42   (mean 79.73 vs 82.34)
    gripper       1.14 –  1.64         0.79 –  1.57
    shoulder_pan -3.21 –  6.81        -1.10 –  6.11

Two failures follow from mixing them up, and one had already happened:

  * FALSE ABORT — the arm homed to elbow 85.30 and the v2-based gate rejected
    it. 85.30 is comfortably inside v3's range; a valid v3 start state was
    thrown away.
  * FALSE PASS — anything between 75.47 and 78.11 passes a v2 gate while being
    outside everything v3 ever saw.

And `home_arm --pose v2` starts a v3 policy ~2.6 deg off its own training mean
in the elbow — more than one standard deviation, on the joint whose start state
already has the widest spread.
"""

import re
import sys

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))

import home_gate  # noqa: E402

JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
PASSIVE = {"shoulder_lift"}
TOL = 2.5


def _home_arm_pose(tag: str) -> dict:
    src = (TOOLS / "home_arm.py").read_text()
    m = re.search(rf'"{tag}":\s*\[([^\]]+)\]', src)
    assert m, f"home_arm.py has no {tag} pose — a {tag} policy cannot be homed to its own start"
    return dict(zip(JOINT_ORDER, [float(x) for x in m.group(1).split(",")]))


def test_home_arm_has_a_v3_pose():
    assert _home_arm_pose("v3")


def test_v3_home_pose_matches_the_v3_dataset(v3_frame0_stats):
    pose = _home_arm_pose("v3")
    for joint, mean in v3_frame0_stats["mean"].items():
        if joint in PASSIVE:
            continue
        assert abs(pose[joint] - mean) <= TOL, (
            f"home_arm v3 {joint}={pose[joint]} vs v3 frame-0 mean {mean:.2f} "
            f"(sd {v3_frame0_stats['std'][joint]:.2f}) — homing outside v3 support"
        )


def test_v2_and_v3_poses_actually_differ(v2_frame0_stats, v3_frame0_stats):
    """If these were the same, the whole fix would be pointless. They are not:
    the elbow means differ by ~2.6 deg."""
    d = abs(v2_frame0_stats["mean"]["elbow_flex"] - v3_frame0_stats["mean"]["elbow_flex"])
    assert d > 1.0, f"v2 and v3 elbow means differ by only {d:.2f} — check the datasets"


def test_home_gate_exposes_v3_support(v3_frame0_stats):
    """The v3 statistics in the gate must come from the v3 parquet. Bounds are
    derived from mean/sd since the Aug 29 tightening, so those are what match."""
    assert "v3" in home_gate.SUPPORT_BY_DATASET
    for joint, mean in v3_frame0_stats["mean"].items():
        if joint in home_gate.PASSIVE_SETTLE_JOINTS | home_gate.CONTACT_SETTLE_JOINTS:
            continue
        m, sd, lo_obs, hi_obs = home_gate.FRAME0["v3"][joint]
        assert abs(m - mean) < 0.05, f"v3 {joint}: gate mean {m} vs dataset {mean:.2f}"
        assert abs(sd - v3_frame0_stats["std"][joint]) < 0.05
        assert abs(lo_obs - v3_frame0_stats["min"][joint]) < 0.05
        assert abs(hi_obs - v3_frame0_stats["max"][joint]) < 0.05


def test_the_false_abort_would_not_recur():
    """A high elbow is out of range for v2 and in range for v3. 84.5 sits inside
    v3's mean+1sd (85.17) and well outside v2's (81.93). Before the datasets were
    separated, a pose like this was rejected on v2's bounds while running a v3
    policy."""
    assert not home_gate.in_training_support("elbow_flex", 84.5, dataset="v2")
    assert home_gate.in_training_support("elbow_flex", 84.5, dataset="v3")


def test_the_false_pass_would_not_recur():
    """The dangerous direction: a pose a v2 gate waves through that v3 never saw.
    78.5 is inside v2's band and below v3's mean-1sd (79.51)."""
    assert home_gate.in_training_support("elbow_flex", 78.5, dataset="v2")
    assert not home_gate.in_training_support("elbow_flex", 78.5, dataset="v3")


def test_dataset_must_be_explicit_or_default_documented():
    """Silently defaulting to v2 is how this bug happened. The default must be
    v2 for backward compatibility, but callers running v3 must pass it."""
    assert home_gate.in_training_support("elbow_flex", 79.7) is True  # v2 default


def test_unknown_dataset_raises():
    try:
        home_gate.in_training_support("elbow_flex", 79.0, dataset="v9")
    except KeyError:
        return
    raise AssertionError("an unknown dataset tag must raise, not silently pass")


def test_trial_script_selects_support_from_the_policy_name():
    src = (TOOLS / "run_scored_trial.sh").read_text()
    assert "DATASET_TAG" in src, (
        "the trial script must pick v2/v3 support from the policy being run, not "
        "hardcode one of them"
    )
