"""`tools/cert_summary.py` -- the numbers a certification replay is logged with, from its own CSV.

Sep 14 2026. The bench record and the the lab notebook (private) (Sep 14 12:37) carry three numbers per
certification pass -- loop rate, the end-effector height proxy over the last 100 frames, and the
minimum jaw opening through the carry window -- and until this file they were computed by an
ad-hoc snippet in the chat. A number in the record that no tool reproduces is a number nobody can
check. This pins the definitions to the ones already published:

  * loop_hz    = (rows - 1) / (t_mono[last] - t_mono[first])            (the receipts' definition)
  * ee_proxy   = mean over frames 103..202 of shoulder_lift + elbow_flex + wrist_flex position
                 (the lab notebook (private) Sep 14 12:37: "EE proxy = mean of (shoulder_lift + elbow_flex +
                 wrist_flex) position over frames 103-202")
  * jaw_min    = min gripper position over frames 96..136, the window where the recorded episode
                 holds the tube (the G-R' draft's carry check; JAW_HELD_MIN_DEG = 5.0 is the
                 published floor, tools/v4_contact_events.py)

and it refuses a CSV that does not hold all 203 frames of v4 episode 0, because the windows are
defined on that episode: a stalled replay is not a certification pass.

The regression that matters: on the three Sep 14 12:37 passes on disk the tool must print
29.79 / 29.46 / 30.34, the values the the lab notebook (private) already states.
"""
import csv
import sys
from pathlib import Path

import pytest

from conftest import PROJECT, TOOLS

sys.path.insert(0, str(TOOLS))

import cert_summary  # noqa: E402
import telemetry_sidecar as ts  # noqa: E402

N = 203


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(ts.HEADER)
        for r in rows:
            w.writerow([r.get(c, 0.0) for c in ts.HEADER])


def _synthetic(n=N, period=0.05):
    rows = []
    for i in range(n):
        r = {"frame_index": i, "t_mono": 100.0 + i * period}
        # heights: 10 everywhere except frames 103..202 where lift=1, elbow=2, wrist=3 -> proxy 6
        in_tail = 103 <= i <= 202
        r["shoulder_lift.pos"] = 1.0 if in_tail else 10.0
        r["elbow_flex.pos"] = 2.0 if in_tail else 10.0
        r["wrist_flex.pos"] = 3.0 if in_tail else 10.0
        # jaw: 20 open, 13 through the carry window, one dip to 7 at frame 120, 1.0 after 136
        if 96 <= i <= 136:
            r["gripper.pos"] = 7.0 if i == 120 else 13.0
        elif i > 136:
            r["gripper.pos"] = 1.0
        else:
            r["gripper.pos"] = 20.0
        rows.append(r)
    return rows


def test_definitions_on_a_synthetic_pass(tmp_path):
    p = tmp_path / "CERT-X_telemetry.csv"
    _write_csv(p, _synthetic())
    s = cert_summary.summarize(p)
    assert s["rows"] == N
    assert s["loop_hz"] == pytest.approx(20.0, abs=1e-6)
    assert s["ee_proxy"] == pytest.approx(6.0, abs=1e-9)      # only the 103..202 window counts
    assert s["jaw_min"] == pytest.approx(7.0, abs=1e-9)       # the dip inside 96..136, not the 1.0 after it
    assert s["jaw_held"] is True                              # 7.0 >= JAW_HELD_MIN_DEG (5.0)


def test_jaw_held_is_false_when_the_tube_was_dropped(tmp_path):
    rows = _synthetic()
    rows[110]["gripper.pos"] = 1.07                           # CHAMF-GATE-02's measured drop
    p = tmp_path / "CERT-DROP_telemetry.csv"
    _write_csv(p, rows)
    s = cert_summary.summarize(p)
    assert s["jaw_min"] == pytest.approx(1.07)
    assert s["jaw_held"] is False


def test_refuses_a_stalled_replay(tmp_path):
    p = tmp_path / "CERT-SHORT_telemetry.csv"
    _write_csv(p, _synthetic(n=150))
    with pytest.raises(cert_summary.CertSummaryError, match="150"):
        cert_summary.summarize(p)


def test_reproduces_the_three_published_sep14_passes():
    """the lab notebook (private) Sep 14 12:37: CERT-B2-01 29.79, CERT-B2-02 29.46, CERT-B2-03 30.34."""
    logs = PROJECT / "tools" / "scored_logs"
    want = {"CERT-B2-01": 29.79, "CERT-B2-02": 29.46, "CERT-B2-03": 30.34}
    for name, ee in want.items():
        p = logs / f"{name}_telemetry.csv"
        if not p.exists():
            pytest.skip(f"{p.name} not on this machine")
        s = cert_summary.summarize(p)
        assert round(s["ee_proxy"], 2) == ee, name
        assert 19.5 < s["loop_hz"] < 20.5, name                # "Loop rate 19.99 Hz on all three"


def test_cli_prints_one_line_per_file(tmp_path, capsys):
    p = tmp_path / "CERT-X_telemetry.csv"
    _write_csv(p, _synthetic())
    rc = cert_summary.main([str(p)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "CERT-X" in out and "loop 20.00 Hz" in out and "ee_proxy 6.00" in out and "jaw_min 7.00" in out
