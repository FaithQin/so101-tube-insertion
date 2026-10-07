"""A v4 policy is homed to, and gated on, v4's OWN training support — never v3's.

The v2->v3 bug (Aug 29) and its recurrence (Sep 3: every v3 trial re-homed to v2) both came
from a table that lacked the newest dataset. v4's frame-0 support, measured over the 50
TRAINING episodes of so101-tube-insert-v4 (Sep 6):

    elbow_flex   mean 85.61  sd 2.77  [79.60, 91.30]     v3: 82.34  sd 2.83  [78.11, 90.42]
    gripper      mean  1.11  sd 0.25  [ 0.93,  2.21]     v3:  1.00  sd 0.11  [ 0.79,  1.57]

Homing a v4 policy to the v3 pose would start its elbow 3.3 deg low. Every number below is
re-derived from the parquet by the `v4_frame0_stats` fixture, so a stale table fails here.
"""
import re
import sys

import pytest

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))

import home_gate  # noqa: E402

JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
PASSIVE = {"shoulder_lift"}   # sits beyond the goal clamp; homed, not gated
TOL = 2.5


def _pose_from_source(path, tag: str) -> dict:
    src = path.read_text()
    m = re.search(rf'"{tag}":\s*\[([^\]]+)\]', src)
    assert m, f"{path.name} has no {tag} pose"
    return dict(zip(JOINT_ORDER, [float(x) for x in m.group(1).split(",")]))


def _train_home_from_wrapper(tag: str) -> dict:
    src = (TOOLS / "rollout_30hz_stale_ok.py").read_text()
    m = re.search(rf"TRAIN_HOME_{tag.upper()}\s*=\s*\{{(.*?)\}}", src, re.S)
    assert m, f"rollout_30hz_stale_ok.py has no TRAIN_HOME_{tag.upper()} table"
    return {k: float(v) for k, v in re.findall(r'"(\w+)\.pos":\s*([-\d.]+)', m.group(1))}


def test_home_arm_has_a_v4_pose_that_matches_the_v4_dataset(v4_frame0_stats):
    pose = _pose_from_source(TOOLS / "home_arm.py", "v4")
    for joint, mean in v4_frame0_stats["mean"].items():
        if joint in PASSIVE:
            continue
        assert abs(pose[joint] - mean) <= TOL, f"home_arm v4 {joint}={pose[joint]} vs frame-0 mean {mean:.2f}"


def test_wrapper_train_home_v4_matches_the_v4_dataset_and_is_selectable(v4_frame0_stats):
    home = _train_home_from_wrapper("v4")
    for joint, mean in v4_frame0_stats["mean"].items():
        assert abs(home[joint] - mean) <= 0.11, f"TRAIN_HOME_V4 {joint}={home[joint]} vs {mean:.2f}"
    src = (TOOLS / "rollout_30hz_stale_ok.py").read_text()
    assert re.search(r'"v4":\s*TRAIN_HOME_V4', src), "TRAIN_HOME table does not offer v4 (CAPSTONE_HOME_POSE=v4)"


def test_home_gate_exposes_v4_support_from_the_parquet(v4_frame0_stats):
    for joint, mean in v4_frame0_stats["mean"].items():
        m, sd, lo, hi = home_gate.FRAME0["v4"][joint]
        assert abs(m - mean) < 0.05, f"v4 {joint}: gate mean {m} vs dataset {mean:.2f}"
        assert abs(sd - v4_frame0_stats["std"][joint]) < 0.05
        assert abs(lo - v4_frame0_stats["min"][joint]) < 0.05
        assert abs(hi - v4_frame0_stats["max"][joint]) < 0.05
    assert "v4" in home_gate.SUPPORT_BY_DATASET


def test_v3_and_v4_supports_actually_differ(v3_frame0_stats, v4_frame0_stats):
    d = abs(v3_frame0_stats["mean"]["elbow_flex"] - v4_frame0_stats["mean"]["elbow_flex"])
    assert d > 1.0, f"v3 and v4 elbow means differ by only {d:.2f} -- is the v4 table a copy?"
    assert home_gate.FRAME0["v4"]["elbow_flex"][0] != home_gate.FRAME0["v3"]["elbow_flex"][0]


def test_the_v3_pose_is_outside_the_v4_gate_on_the_elbow():
    """The bug this file exists to prevent: a v4 policy homed to v3's elbow (82.3) must be REFUSED
    by the v4 gate (mean 85.61, sd 2.77 -> band [82.84, 88.38])."""
    assert not home_gate.in_training_support("elbow_flex", 82.3, dataset="v4")
    assert home_gate.in_training_support("elbow_flex", 85.6, dataset="v4")


def test_act_runner_selects_v4_support_from_the_policy_name():
    src = (TOOLS / "run_scored_trial.sh").read_text()
    assert re.search(r"\*v4\*\)\s*DATASET_TAG=v4", src), "run_scored_trial.sh has no *v4*) branch"


def test_smolvla_runner_takes_the_dataset_tag_from_the_side_table_not_a_constant():
    src = (TOOLS / "run_smolvla_trial.sh").read_text()
    assert not re.search(r"^DATASET_TAG=v3", src, re.M), "SmolVLA runner still hardcodes DATASET_TAG=v3"
    assert "--print gen" in src or "--print dataset_tag" in src


def test_smolvla_side_table_reports_its_generation():
    import smolvla_dryrun as sd
    assert sd.print_field("gen", sd.active_sides({})["A"]) == "v4"
    assert sd.print_field("gen", sd.active_sides({"CAPSTONE_POLICY_GEN": "v3"})["B"]) == "v3"
