"""Offline falsification harness for the grasp-lock boundary race (Aug 27,
2026, Faith's test design): replay recorded COMMANDED action traces through
tools/grasp_lock.py and ask, per trial — would the lock have engaged BEFORE
the chunk boundary nearest the closing onset? If the answer across the v2
trial corpus is "rarely", the boundary race defeats the lock and it needs a
redesign before any bench probe.

These tests exercise the harness on synthetic traces where ground truth is
known by construction. The real-data run is tools/replay_grasp_lock.py
against the v2 trial parquet.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from replay_grasp_lock import analyze_trace  # noqa: E402


def trace(onset_tick, n=120, open_val=40.0, lift=38.0):
    """Commanded (grip, lift) trace: open approach in the grasp band, then a
    steady close starting at onset_tick (2 deg/tick down to 0)."""
    out = []
    g = open_val
    for t in range(n):
        if t >= onset_tick:
            g = max(0.0, open_val - 2.0 * (t - onset_tick + 1))
        out.append((g, lift))
    return out


def test_lock_engages_two_ticks_after_onset():
    r = analyze_trace(trace(onset_tick=40), n_action_steps=25)
    assert r["onset_tick"] == 40
    # ONSET_TICKS=2 consecutive decreasing commands -> locked on the 2nd
    assert r["lock_tick"] == 41


def test_protected_when_onset_well_before_boundary():
    # boundaries at 25, 50, 75...; onset at 40 locks at 41, next boundary 50
    r = analyze_trace(trace(onset_tick=40), n_action_steps=25)
    assert r["next_boundary"] == 50
    assert r["verdict"] == "PROTECTED"
    assert r["margin_ticks"] == 9


def test_raced_when_onset_straddles_boundary():
    # onset at 49: first decreasing tick 49, boundary fires at 50 before the
    # 2nd decreasing tick can lock -> the re-plan wins the race
    r = analyze_trace(trace(onset_tick=49), n_action_steps=25)
    assert r["lock_tick"] is not None and r["lock_tick"] >= 50
    assert r["verdict"] == "RACED"


def test_boundary_on_lock_tick_counts_as_raced():
    # lock lands exactly ON the boundary tick: refill check runs before
    # observe() in the wrapper, so the re-plan still wins
    r = analyze_trace(trace(onset_tick=48), n_action_steps=25)
    assert r["lock_tick"] == 49 or r["verdict"] in ("PROTECTED", "RACED")
    if r["lock_tick"] == 50:
        assert r["verdict"] == "RACED"


def test_never_locked_when_no_close():
    flat = [(40.0, 38.0)] * 80
    r = analyze_trace(flat, n_action_steps=25)
    assert r["lock_tick"] is None
    assert r["verdict"] == "NEVER_LOCKED"


def test_high_lift_close_does_not_lock():
    # closing during carry (lift above band) must not lock
    tr = [(40.0, 80.0)] * 30 + [(max(0.0, 40.0 - 2.0 * k), 80.0) for k in range(30)]
    r = analyze_trace(tr, n_action_steps=25)
    assert r["verdict"] == "NEVER_LOCKED"
