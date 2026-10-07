"""Regression: the slip-timestamp forensics must measure posture, not lighting.

Born Sep 1 2026. A horn slip (~10 deg elbow, ~7 deg wrist_flex) was measured
physically after two days of degrading replays. Every trial video begins at the
same commanded home pose, so frame 0 is a physical posture record at identical
commands: the slip must appear as a STEP CHANGE in the gripper's frame-0
position somewhere in the chronology. Locating that step decides which trials
ran on a bent plant — including whether Aug 29-30's Block 2 probes and
Aug 31's two counted B trials survive.

Pinned here:

1. The gripper-tip proxy is the centroid of vivid-red jaw pixels in the UPPER
   part of the frame only. The rack is also red and lives in the lower half;
   without the crop the rack dominates the centroid and every frame looks
   identical (the rack never moves) — measuring exactly nothing.
2. Red detection must be chroma-based, not brightness-based, so day/night
   lighting shifts (which the v2 audit showed are real) do not read as motion.
3. Step detection must find a sustained level change and ignore single-frame
   outliers (one noisy detection is not a slip event).
4. Groups are analyzed separately: v2-homed and v3-homed trials have different
   commanded homes, so pooling them manufactures a fake step at the v2->v3
   boundary.
"""

import sys

import numpy as np
from conftest import TOOLS

sys.path.insert(0, str(TOOLS))

import slip_forensics as sf  # noqa: E402


def _frame(h=120, w=160):
    return np.full((h, w, 3), 200, dtype=np.uint8)  # light-grey bench


def test_red_centroid_finds_a_red_blob():
    f = _frame()
    f[20:30, 70:80] = (210, 30, 25)  # vivid red blob, upper region
    c = sf.red_centroid(f, top_frac=0.55)
    assert c is not None
    cx, cy, n = c
    assert 69 <= cx <= 80 and 19 <= cy <= 30 and n >= 50


def test_red_centroid_ignores_the_rack_in_the_lower_frame():
    """The rack is red and static; it must not anchor the centroid."""
    f = _frame()
    f[100:115, 60:120] = (210, 30, 25)  # big red rack, lower region
    f[20:26, 70:76] = (210, 30, 25)     # small red jaw, upper region
    cx, cy, n = sf.red_centroid(f, top_frac=0.55)
    assert cy < 30, "centroid was dragged into the rack region"


def test_red_centroid_is_chroma_not_brightness():
    """A bright grey highlight (day/night lighting) is not red."""
    f = _frame()
    f[20:30, 70:80] = (250, 250, 250)
    assert sf.red_centroid(f, top_frac=0.55) is None


def test_red_centroid_none_when_no_red():
    assert sf.red_centroid(_frame(), top_frac=0.55) is None


def test_step_detection_finds_a_sustained_level_change():
    ys = [100.0] * 8 + [140.0] * 8
    k = sf.find_step(ys, min_jump=15.0, min_run=3)
    assert k == 8


def test_step_detection_ignores_a_single_outlier():
    ys = [100.0] * 7 + [180.0] + [100.0] * 8
    assert sf.find_step(ys, min_jump=15.0, min_run=3) is None


def test_step_detection_none_when_flat():
    ys = list(np.random.default_rng(0).normal(100, 2.0, 20))
    assert sf.find_step(ys, min_jump=15.0, min_run=3) is None


def test_step_uses_medians_so_noise_does_not_fake_a_jump():
    ys = list(np.random.default_rng(1).normal(100, 4.0, 10)) + list(
        np.random.default_rng(2).normal(103, 4.0, 10)
    )
    assert sf.find_step(ys, min_jump=15.0, min_run=3) is None


def test_group_key_separates_v2_and_v3_homes():
    assert sf.group_key("rollout_trial_A_v2_20260825_1") == "v2"
    assert sf.group_key("rollout_probe_lockON_a25_v2_20260829_1") == "v2"
    assert sf.group_key("rollout_B2-A-01_20260829_230748") == "v3"
    assert sf.group_key("rollout_B2B-B50-01_20260831_215303") == "v3"
    assert sf.group_key("rollout_B2V-A-02_20260830_233508") == "v3"
