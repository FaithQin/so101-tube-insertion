"""Regression: calibration integrity and v2 dataset invariants.

Calibration: lerobot reads ONLY ~/.cache/huggingface/lerobot/calibration/.
calibration-backup/ in the project is a manual copy — these tests catch the
two historical failure modes: backup drifting out of sync after a
recalibration, and the Aug 7 over-swept gripper ranges (a sweep that
hand-forces past the safe stop writes endpoints the servo later stalls
trying to hold). Range WIDTHS are the frame-independent check — endpoints
are raw values relative to per-calibration homing offsets and cannot be
compared across calibrations.
"""

import filecmp
import json

import pytest

from conftest import CALIBRATION_BACKUP, LIVE_CALIBRATION, V2_DATASET_ROOT

# Known-good widths (tools/README.md). Anything near double = over-travelled.
GRIPPER_WIDTH_BANDS = {
    "follower": (1350, 1450),
    "leader": (1180, 1280),
}


def _cal_files(root):
    return sorted(p for p in root.rglob("*.json"))


def test_backup_matches_live_calibration():
    live = _cal_files(LIVE_CALIBRATION)
    assert live, f"no calibration files under {LIVE_CALIBRATION}"
    for lp in live:
        bp = CALIBRATION_BACKUP / lp.relative_to(LIVE_CALIBRATION)
        assert bp.exists(), (
            f"{lp.relative_to(LIVE_CALIBRATION)} missing from calibration-backup/ — "
            f"re-copy after recalibration (cp -R ~/.cache/huggingface/lerobot/calibration/. calibration-backup/)"
        )
        assert filecmp.cmp(lp, bp, shallow=False), (
            f"calibration-backup/{lp.relative_to(LIVE_CALIBRATION)} differs from live — "
            f"the backup is stale (or worse, live was changed without re-copying)"
        )


@pytest.mark.parametrize("arm", ["follower", "leader"])
def test_gripper_width_not_overswept(arm):
    matches = [p for p in _cal_files(LIVE_CALIBRATION) if arm in str(p)]
    assert matches, f"no live calibration file for {arm}"
    cal = json.loads(matches[0].read_text())
    g = cal["gripper"]
    width = g["range_max"] - g["range_min"]
    lo, hi = GRIPPER_WIDTH_BANDS[arm]
    assert lo <= width <= hi, (
        f"{arm} gripper width {width} outside known-good {lo}-{hi} — "
        f"~double means the sweep over-travelled (Aug 7 failure). "
        f"Fix with tools/gripper_measure.py + tools/patch_gripper.py, not a recal."
    )


def test_v2_dataset_invariants():
    info = json.loads((V2_DATASET_ROOT / "meta" / "info.json").read_text())
    assert info["fps"] == 20, f"v2 fps is {info['fps']} — rollouts pace --fps=20 against it"
    assert info["total_episodes"] == 50, (
        f"v2 has {info['total_episodes']} episodes, expected 50 "
        f"(ep33 is the overheat casualty — excluded at TRAIN time via include-list, "
        f"never deleted from the dataset)"
    )
    state = info["features"]["observation.state"]
    assert state["shape"] == [18], (
        f"v2 observation.state shape {state['shape']} != [18] — "
        f"the load/current channels are missing; B-policies cannot train on this"
    )
