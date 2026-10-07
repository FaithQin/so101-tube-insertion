"""tools/compare_tracking.py — live replay tracking vs the recording-time
baseline, per joint, over the motion window (rest poses excluded: at rest the
leader presses the follower past its limits, so end-pose errors are structure,
not tracking — measured Sep 2: elbow -9 deg, wrist_flex -5 deg at rest in every
v3 episode)."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import compare_tracking as ct  # noqa: E402


def _synthetic(n=100):
    t = np.arange(n)
    cmd = np.stack([np.sin(t / 10) * 20 + i for i in range(6)], axis=1)
    rec = cmd.copy(); rec[:, 2] += 1.0          # recording: elbow tracked 1 deg high
    live = cmd.copy(); live[:, 2] += 4.0        # live: elbow tracks 4 deg high -> +3 vs recording
    live[:, 3] -= 0.5
    return cmd, rec, live


def test_compare_reports_live_minus_recording_per_joint_over_the_window():
    cmd, rec, live = _synthetic()
    out = ct.compare(cmd, rec, live, window=(10, 90))
    assert abs(out["elbow_flex"]["live_minus_rec_mean"] - 3.0) < 1e-9
    assert abs(out["wrist_flex"]["live_minus_rec_mean"] + 0.5) < 1e-9
    assert abs(out["shoulder_pan"]["live_minus_rec_mean"]) < 1e-9
    assert abs(out["elbow_flex"]["rec_minus_cmd_mean"] - 1.0) < 1e-9
    assert abs(out["elbow_flex"]["live_minus_cmd_mean"] - 4.0) < 1e-9
    assert out["elbow_flex"]["n"] == 80


def test_compare_truncates_to_the_shorter_series_and_rejects_empty_window():
    cmd, rec, live = _synthetic()
    out = ct.compare(cmd, rec, live[:50], window=(0, 200))
    assert out["elbow_flex"]["n"] == 50
    import pytest
    with pytest.raises(ValueError):
        ct.compare(cmd, rec, live, window=(90, 10))


def test_load_record_reads_the_replay_csv_layout(tmp_path):
    csv = tmp_path / "t.csv"
    header = ["tick"] + [f"cmd_{j}" for j in ct.JOINTS] + [f"rep_{j}" for j in ct.JOINTS]
    rows = [[0] + list(range(1, 7)) + list(range(11, 17)), [1] + list(range(2, 8)) + list(range(12, 18))]
    csv.write_text(",".join(header) + "\n" + "\n".join(",".join(map(str, r)) for r in rows) + "\n")
    cmd, rep = ct.load_record(csv)
    assert cmd.shape == (2, 6) and rep.shape == (2, 6)
    assert cmd[0, 0] == 1 and rep[1, 5] == 17
