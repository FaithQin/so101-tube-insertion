"""PROPOSAL (Sep 7 2026, unratified): co-primary endpoint 2 as a RESISTANCE-RECOVERY response.

Nothing here is wired into the frozen analysis path. `tools/v4_recovery_response.py` is a new
module; `tools/v4_bench_analysis.py` and `tools/v4_contact_events.py` are untouched. Faith
ratifies (or refuses) the method change; until then this is evidence, not the endpoint.

WHY a different discriminandum. `Start Schedule — v4.md:16` says every episode, CLEAN included,
"continues to the seat", so every rack window contains a genuine tube-meets-rack contact and
"did contact occur" cannot separate anything. The same line describes what RESIST actually is:
"brought down at a wrong position or angle so the tip meets resistance ... backs off, re-aligns,
inserts". That is a PRESS -> RETREAT -> RE-PRESS, and it is what these tests pin down.

THE HOLD-OUT. Episodes 11, 19, 25, 31, 35, 50, 51, 57 are the dataset's held-out split
(`tools/v4_contact_events.py:147`). Nothing in this file loads one; `test_holdout_is_refused`
makes that a checked property of the module rather than a promise.
"""
import sys

import numpy as np
import pytest

from conftest import TOOLS, V4_DATASET_ROOT, v4_training_episodes

sys.path.insert(0, str(TOOLS))
import v4_recovery_response as rr  # noqa: E402

FPS = 20


# --- helpers: synthetic depth traces ------------------------------------------------------

def _trace(depth_seq, jaw=None, pan=10.0, wrist_flex=0.0):
    """(n, 6) observation rows whose end-effector proxy (dims 1+2+3) equals `depth_seq`.

    The whole depth is carried on shoulder_lift so the proxy is exactly the sequence; pan and
    wrist_flex are parked inside `v4_audit.rack_slice`'s region so the window is the whole trace.
    """
    d = np.asarray(depth_seq, dtype=float)
    n = len(d)
    st = np.zeros((n, 6))
    st[:, 0] = pan
    st[:, 1] = d - wrist_flex
    st[:, 3] = wrist_flex
    st[:, 5] = 0.0 if jaw is None else np.asarray(jaw, dtype=float)
    return st


def _stretch(x, k):
    x = np.asarray(x, dtype=float)
    return np.interp(np.linspace(0, 1, k), np.linspace(0, 1, len(x)), x)


SINGLE = np.r_[np.linspace(20, 0, 12), np.linspace(0, 18, 10)]          # descend once, come up
DOUBLE = np.r_[np.linspace(20, 0, 12), np.linspace(0, 9, 6),
               np.linspace(9, 1, 6), np.linspace(1, 18, 8)]              # press, retreat, re-press


# --- the statistic -------------------------------------------------------------------------

def test_single_press_scores_exactly_zero():
    """A trial that seats cleanly is not 'undefined' and not 'small' — it is 0.0."""
    r = rr.recovery_response(_trace(SINGLE), None)
    assert r["reached_rack"] is True
    assert r["n_press"] == 1
    assert r["repress_amplitude_deg"] == 0.0


def test_double_press_scores_the_V_amplitude():
    r = rr.recovery_response(_trace(DOUBLE), None)
    assert r["n_press"] == 2
    # the interior maximum sits at 9; its shallower leg is min(9 - 0, 9 - 1) = 8
    assert r["repress_amplitude_deg"] == pytest.approx(8.0, abs=0.25)


def test_stretching_a_clean_press_cannot_manufacture_a_repress():
    """THE DURATION CONTROL, in code. Fact 4 of the diagnosis: on this dataset a longer window
    is what fools people. Time-scaling a single press by 4x must leave the statistic at 0."""
    for k in (30, 60, 90):
        r = rr.recovery_response(_trace(_stretch(SINGLE, k)), None)
        assert r["repress_amplitude_deg"] == 0.0, k
        assert r["n_press"] == 1, k


def test_compressing_a_resisted_press_preserves_the_amplitude():
    """The converse control: the statistic is a level in degrees, not a length in ticks."""
    full = rr.recovery_response(_trace(DOUBLE), None)["repress_amplitude_deg"]
    short = rr.recovery_response(_trace(_stretch(DOUBLE, 18)), None)["repress_amplitude_deg"]
    assert short == pytest.approx(full, rel=0.10)


# --- the window ----------------------------------------------------------------------------

def test_window_ends_at_the_release_and_the_withdrawal_is_excluded():
    """Running the window past the release makes every episode fire: the arm withdraws and
    descends again on the way out. Measured on the 50 v4 training episodes — cutting at the
    rack-slice end fires 45/46 non-RESIST. The jaw edge is what makes the statistic mean
    anything, so the module must cut there."""
    depth = np.r_[SINGLE, np.linspace(18, 2, 8), np.linspace(2, 25, 8)]   # + a post-release dip
    jaw = np.r_[np.full(len(SINGLE), 5.0), np.full(16, 30.0)]             # jaws open at the release
    r = rr.recovery_response(_trace(depth, jaw=jaw), None)
    assert r["window"][1] <= len(SINGLE) + 1
    assert r["repress_amplitude_deg"] == 0.0


def test_window_is_capped_by_the_horizon_when_the_trial_never_releases():
    """A JAM reaches T and never opens the jaws. Without a cap the window runs to the end of the
    episode and the statistic is meaningless; with it the trial is still scoreable."""
    depth = np.r_[SINGLE, np.tile(np.r_[np.linspace(18, 2, 6), np.linspace(2, 18, 6)], 20)]
    r = rr.recovery_response(_trace(depth, jaw=np.full(len(depth), 5.0)), None)
    a, b = r["window"]
    assert b - a == rr.HORIZON_TICKS
    assert r["window_truncated_by_horizon"] is True


# --- the latency ---------------------------------------------------------------------------

def test_a_relief_at_the_stall_tick_is_a_lead_not_a_zero_latency():
    """Defect 7 of the audit: a reversal already under way on the onset tick LEADS the event, it
    does not respond to it. It must be reported as a reason, never as 0.00 s."""
    depth = np.r_[np.linspace(20, 8, 10), np.full(16, 8.0)]              # arm arrests at tick 10
    st = _trace(depth, jaw=np.full(26, 5.0))
    act = st.copy()
    act[:10, 1] = np.linspace(14, -2, 10)            # command dives past the arm
    act[10:, 1] = 2.0 + 0.5 * np.arange(16)          # and is already 4 deg back up at the arrest
    r = rr.recovery_response(st, act)
    assert r["resistance_detected"] is True
    assert r["recovery_latency_s"] is None
    assert "lead" in (r["recovery_latency_reason"] or "").lower()


def test_the_latency_search_stops_at_the_window_end():
    """Defect 6: `first_opposing_command` searches to the end of the trace, so a reversal seconds
    after the window can be reported as this event's latency."""
    depth = np.r_[np.linspace(20, 8, 10), np.full(10, 8.0)]
    st = _trace(np.r_[depth, np.linspace(8, 40, 40)], jaw=np.r_[np.full(20, 5.0), np.full(40, 30.0)])
    act = st.copy()
    act[:, 1] = st[:, 1] - 6.0                        # commanded deeper throughout the window
    r = rr.recovery_response(st, act)
    assert r["resistance_detected"] is True
    assert r["recovery_latency_s"] is None
    assert "window" in (r["recovery_latency_reason"] or "").lower()


# --- the guard -----------------------------------------------------------------------------

def test_holdout_is_refused():
    for ep in (11, 19, 25, 31, 35, 50, 51, 57):
        with pytest.raises(rr.HoldoutEpisode):
            rr.check_not_holdout(ep)
    rr.check_not_holdout(2)          # a training RESIST is fine


# --- the real data -------------------------------------------------------------------------

@pytest.mark.skipif(not V4_DATASET_ROOT.exists(), reason="v4 dataset not on this machine")
def test_training_resist_represses_and_nothing_else_does():
    """The measured result, on TRAINING episodes only: repress_amplitude_deg > 0 for all four
    RESIST episodes and exactly 0.0 for all 46 others (30 CLEAN, 4 NUDGE, 12 ROT/LEFT-MIMIC)."""
    import v4_contact_events as ce
    import v4_manifest

    hold = set(ce.HOLDOUT_HANDOFF)
    m = v4_manifest.Manifest.load(TOOLS / "v4_manifest.json")
    states, actions, _ = ce.load_v4_dataset()
    keep = [e for e in v4_training_episodes() if e in states]
    assert not (set(keep) & hold)
    labels = {int(k): v for k, v in m.assigned.items()}

    pos, zero = [], []
    for e in keep:
        rr.check_not_holdout(e)
        amp = rr.recovery_response(states[e], actions[e])["repress_amplitude_deg"]
        (pos if labels[e] == "RESIST" else zero).append((e, amp))

    assert [e for e, _ in pos] == [2, 20, 27, 39]
    assert all(a > 0 for _, a in pos), pos
    assert all(a == 0.0 for _, a in zero), [x for x in zero if x[1] > 0]
    assert min(a for _, a in pos) > 6.0


def test_arriving_from_below_is_not_a_press():
    """A window whose first move is UPWARD (the arm enters the rack region from below, or the
    entry tick lands mid-recovery) must not have that leading rise scored as a press: the peak
    would otherwise be an 'interior maximum' flanked by the window edge and one real minimum,
    and a single descent would report a large re-press amplitude."""
    depth = np.r_[np.linspace(0, 14, 8), np.linspace(14, 0, 8), np.linspace(0, 14, 8)]
    r = rr.recovery_response(_trace(depth), None)
    assert r["n_press"] == 1
    assert r["repress_amplitude_deg"] == 0.0


def test_n_press_counts_descents_not_peaks():
    assert rr.repress(SINGLE)[0] == 1
    assert rr.repress(DOUBLE)[0] == 2
    assert rr.repress(np.linspace(20, 0, 20))[0] == 0     # descends and never comes back up


def test_the_commanded_release_edge_is_preferred_when_an_action_stream_exists():
    """The observed jaw opens ~4 ticks after the command that opens it (`v4_contact_events.
    rack_window` docstring, measured on v4 ep 0). Those extra ticks are the release itself, and
    on the 30 training CLEAN episodes they turn 1 spurious kinematic stall into 3. The window
    therefore cuts on the COMMAND when there is one, and falls back to the observed jaw — which
    gives the identical re-press amplitude on all 50 training episodes — when there is not."""
    depth = np.r_[SINGLE, np.linspace(18, 4, 6), np.linspace(4, 24, 6)]
    jaw = np.r_[np.full(len(SINGLE) + 4, 5.0), np.full(8, 30.0)]      # observed jaw opens late
    st = _trace(depth, jaw=jaw)
    act = st.copy()
    act[:, 5] = np.r_[np.full(len(SINGLE), 5.0), np.full(12, 30.0)]   # commanded 4 ticks earlier
    assert rr.rack_window(st, act)[1] < rr.rack_window(st, None)[1]
    assert rr.rack_window(st, act)[1] == len(SINGLE)
