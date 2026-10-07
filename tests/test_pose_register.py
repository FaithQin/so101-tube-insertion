"""Regression: the dataset-frame pose registration must measure in the RIGHT frame.

Born Sep 2 2026, after a day of measuring the follower against the LEADER arm
and patching calibration on that basis. The leader comparison had no pre-fault
baseline, so its deltas were never evidence of change; the patches built on it
left the arm unable to touch the tube. Faith's question — "are we matching to a
different calibration than the episodes were recorded in?" — named the flaw.

The only frame that matters is the one the datasets were recorded in. This tool
registers against it directly: a recorded frame is blended over the live camera,
the operator poses the torque-off follower onto the recorded silhouette, and the
follower's REPORTED state minus the dataset's RECORDED state at that frame is the
per-joint offset. No leader, no Jacobians, no sign guessing.

Pinned here:
1. The offset is dataset-minus-reported per joint, so a POSITIVE value means the
   follower must report higher — the sign convention is fixed once, in code.
2. Consistency across poses is the discriminator: constant offsets mean a
   calibration constant (fixable); pose-dependent offsets mean a servo/gear
   problem that no calibration fixes. The verdict must say which.
3. The blend must keep both images visible (neither fully opaque) or the
   operator cannot see the recorded arm through the live one.
"""

import sys

import numpy as np
from conftest import TOOLS

sys.path.insert(0, str(TOOLS))

import pose_register as pr  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def test_offset_is_dataset_minus_reported():
    recorded = {"elbow_flex": 87.0, "wrist_flex": 74.0}
    reported = {"elbow_flex": 80.0, "wrist_flex": 76.0}
    off = pr.joint_offsets(recorded, reported)
    assert off["elbow_flex"] == 7.0   # follower must report 7 higher
    assert off["wrist_flex"] == -2.0  # follower must report 2 lower


def test_consistency_verdict_constant_offsets():
    per_pose = [
        {"elbow_flex": 7.1, "wrist_flex": -1.9},
        {"elbow_flex": 6.8, "wrist_flex": -2.2},
        {"elbow_flex": 7.3, "wrist_flex": -2.0},
    ]
    v = pr.consistency_verdict(per_pose, tol_deg=1.5)
    assert v["status"] == "CALIBRATION_CONSTANT"
    assert abs(v["median"]["elbow_flex"] - 7.1) < 0.2


def test_consistency_verdict_pose_dependent():
    per_pose = [
        {"elbow_flex": 1.0, "wrist_flex": 0.5},
        {"elbow_flex": 6.5, "wrist_flex": 0.4},
        {"elbow_flex": 11.0, "wrist_flex": 0.6},
    ]
    v = pr.consistency_verdict(per_pose, tol_deg=1.5)
    assert v["status"] == "POSE_DEPENDENT"
    assert "elbow_flex" in v["inconsistent_joints"]
    assert "wrist_flex" not in v["inconsistent_joints"]


def test_consistency_needs_at_least_two_poses():
    try:
        pr.consistency_verdict([{"elbow_flex": 1.0}], tol_deg=1.5)
    except ValueError:
        return
    raise AssertionError("one pose cannot establish consistency")


def test_blend_keeps_both_images_visible():
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    b = np.full((4, 4, 3), 200, dtype=np.uint8)
    out = pr.blend(a, b, alpha=0.5)
    assert out.dtype == np.uint8
    assert 90 <= int(out[0, 0, 0]) <= 110, "50/50 blend of 0 and 200 should be ~100"


def test_blend_rejects_degenerate_alpha():
    a = np.zeros((2, 2, 3), dtype=np.uint8)
    for bad in (0.0, 1.0, 1.5):
        try:
            pr.blend(a, a, alpha=bad)
        except ValueError:
            continue
        raise AssertionError(f"alpha={bad} hides one image entirely and must be rejected")
