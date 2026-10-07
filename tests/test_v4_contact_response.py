"""tools/v4_contact_response.py — the OPTION A contact-response endpoint (a PROPOSAL).

These tests pin the four engineering defects that were measured in the existing endpoint
(`tools/v4_contact_events.py`) on Sep 7, and the honest confound that the duration-matched
control exposed in the proposal itself. Nothing here ratifies the method change: the
pre-registration's §1 anchors co-primary 2 on "distal-load onset", and re-anchoring it on a
kinematic stall is Faith's to ratify (§7 amendment).

Hold-out episodes 11, 19, 25, 31, 35, 50, 51, 57 are never loaded by any test in this file.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from conftest import TOOLS, V4_DATASET_ROOT, v4_training_episodes

sys.path.insert(0, str(TOOLS))

import v4_contact_response as cr  # noqa: E402

HOLDOUT = [11, 19, 25, 31, 35, 50, 51, 57]
SCORED = TOOLS / "scored_logs"


# --- synthetic fixtures -------------------------------------------------------------

def _trace(n=60):
    """An 18-dim state and a 6-dim action with nothing interesting in them."""
    s = np.zeros((n, 18), dtype=np.float64)
    s[:, 6:18] = np.linspace(0.0, 1.0, n)[:, None]   # live (non-constant) load/current
    a = np.zeros((n, 6), dtype=np.float64)
    return s, a


# --- 1. THE CLOCK -------------------------------------------------------------------

def test_sample_times_uses_t_mono_when_the_sidecar_wrote_one():
    """DEFECT: times were computed at a hard-coded 20 fps while the loop ran 19.35-19.40 Hz,
    and the sidecar's own t_mono column was loaded and then discarded."""
    t_mono = 1000.0 + np.arange(40) / 19.352
    t, src = cr.sample_times(40, t_mono=t_mono)
    assert src == "t_mono"
    assert t[0] == 0.0
    assert t[-1] == pytest.approx(39 / 19.352, abs=1e-9)
    # the nominal clock would have under-reported this span by 3.35 %
    assert t[-1] > 39 / 20.0


def test_sample_times_falls_back_to_the_nominal_rate_and_says_so():
    t, src = cr.sample_times(40)
    assert "20" in src and "nominal" in src
    assert t[-1] == pytest.approx(39 / 20.0)


def test_latency_is_measured_on_the_real_clock_not_the_nominal_one():
    """The same reversal, 10 samples after the stall, is 0.500 s at 20 Hz and 0.517 s at 19.35 Hz."""
    s, a = _trace(60)
    s[:, 2] = np.r_[np.arange(20) * 1.0, np.full(40, 19.0)]   # elbow drives in, then jams
    a[:, 2] = s[:, 2] + 6.0                                   # command leads by 6 deg -> a stall
    a[30:, 2] = s[30:, 2] - 6.0                               # reversal at sample 30
    slow = 1000.0 + np.arange(60) / 19.352
    onsets = cr.stall_onsets(s, a, (0, 60), cr.sample_times(60, t_mono=slow)[0])
    assert onsets, "the synthetic stall must be detected"
    on = onsets[0]
    nominal = cr.first_opposing_command(s, a, on, cr.sample_times(60)[0], hard_end=60)
    real = cr.first_opposing_command(s, a, on, cr.sample_times(60, t_mono=slow)[0], hard_end=60)
    assert real["latency_s"] > nominal["latency_s"]
    assert real["latency_s"] / nominal["latency_s"] == pytest.approx(20.0 / 19.352, rel=1e-6)


# --- 2. A LEAD IS NOT A RESPONSE ----------------------------------------------------

def test_an_opposing_command_at_the_anchor_tick_is_not_a_zero_latency_response():
    """DEFECT: a 0.00 s latency was reportable. A command already opposing at the anchor LEADS
    the event; it cannot be the response to it."""
    s, a = _trace(40)
    s[:, 2] = np.r_[np.arange(10) * 2.0, np.full(30, 18.0)]
    a[:, 2] = s[:, 2] + 5.0
    a[12:, 2] = s[12:, 2] - 5.0          # opposing from sample 12 onward
    t = cr.sample_times(40)[0]
    r = cr.first_opposing_command(s, a, 12, t, hard_end=40)
    assert r["latency_s"] != 0.0
    assert r["frame"] == 13
    assert r["latency_s"] == pytest.approx(0.05)


# --- 3. THE SEARCH IS BOUNDED -------------------------------------------------------

def test_the_latency_search_stops_at_the_window_end():
    """DEFECT: `first_opposing_command` searched to the end of the trace, so a reversal seconds
    after the event could be reported as that event's latency."""
    s, a = _trace(200)
    s[:, 2] = np.r_[np.arange(10) * 2.0, np.full(190, 18.0)]
    a[:, 2] = s[:, 2] + 5.0
    a[50:, 2] = s[50:, 2] - 5.0          # the reversal is 2.0 s later, outside a 40-sample window
    t = cr.sample_times(200)[0]
    assert cr.RESP_MAX_S > (50 - 10) / 20.0          # the horizon is not what excludes it
    assert cr.first_opposing_command(s, a, 10, t, hard_end=40)["latency_s"] is None
    assert cr.first_opposing_command(s, a, 10, t, hard_end=200)["latency_s"] == pytest.approx(2.0)


def test_the_latency_search_stops_at_the_response_horizon():
    s, a = _trace(400)
    s[:, 2] = np.r_[np.arange(10) * 2.0, np.full(390, 18.0)]
    a[:, 2] = s[:, 2] + 5.0
    a[120:, 2] = s[120:, 2] - 5.0        # 5.5 s after the anchor, past RESP_MAX_S = 3.0 s
    t = cr.sample_times(400)[0]
    assert cr.RESP_MAX_S < (120 - 10) / 20.0
    assert cr.first_opposing_command(s, a, 10, t, hard_end=400)["latency_s"] is None


# --- 4. THE CENSORED CHANNEL --------------------------------------------------------

def test_gripper_load_is_not_in_the_rack_distal_set():
    """gripper.load is pinned at exactly -120.0 in 99.43 % of training rack-window frames and is
    the ONLY load channel hard-censored (at +500 = Max_Torque_Limit). It cannot carry a contact
    magnitude at the rack, and its censoring made two unrelated trials print an identical z_max."""
    assert 11 not in cr.DISTAL_RACK
    assert set(cr.DISTAL_RACK) == {8, 9, 10}


def test_a_saturated_gripper_cannot_change_the_impulse():
    s, a = _trace(60)
    s[:, 8] = 40.0                       # elbow_flex.load carries the whole signal
    clean, _ = cr.contact_impulse(s, 20, cr.sample_times(60)[0], hard_end=60)
    s[20:40, 11] = 500.0                 # gripper pegged at Max_Torque_Limit
    pegged, _ = cr.contact_impulse(s, 20, cr.sample_times(60)[0], hard_end=60)
    assert pegged == clean


def test_the_impulse_is_duration_free():
    """THE CONFOUND THIS DATASET USES TO FOOL PEOPLE. RESIST rack windows are 2.50-3.70 s and
    CLEAN ones 0.75-1.60 s, so every raw sum over an event's own ticks -- which is what the old
    `impulse_load_s` was -- separates the classes for a reason that has nothing to do with
    contact. Two traces with the SAME load level and DIFFERENT window lengths must give the same
    impulse."""
    s_short, _ = _trace(60)
    s_long, _ = _trace(200)
    for s in (s_short, s_long):
        s[:, 8:11] = 0.0                 # only the ramp in 6:18 differs between the two lengths
        s[20:, 8] = 40.0                 # identical excursion, identical start
    t_short = cr.sample_times(60)[0]
    t_long = cr.sample_times(200)[0]
    short, _ = cr.contact_impulse(s_short, 20, t_short, hard_end=60)
    long_, _ = cr.contact_impulse(s_long, 20, t_long, hard_end=200)
    assert short == pytest.approx(long_)
    # 40 raw units held across a fixed 0.70 s window, to within one 0.05 s sample
    assert abs(short - 40.0 * cr.IMPULSE_WIN_S) <= 40.0 * 0.05 + 1e-9


def test_the_impulse_is_undefined_rather_than_truncated_when_the_window_is_too_short():
    """Truncating the fixed window to whatever room is left would put duration straight back in."""
    s, _ = _trace(60)
    s[:, 8] = np.r_[np.zeros(20), np.full(40, 40.0)]
    t = cr.sample_times(60)[0]
    v, why = cr.contact_impulse(s, 20, t, hard_end=25)   # only 0.25 s of room, need 0.70 s
    assert v is None
    assert "does not fit" in why


def test_censored_channels_names_a_channel_pinned_at_its_own_extreme():
    s, _ = _trace(200)
    s[:, 11] = np.linspace(-120.0, 100.0, 200)
    assert cr.censored_channels(s, [11]) == []
    s[50:50 + cr.CENSOR_MIN_REPEATS, 11] = 500.0
    flagged = cr.censored_channels(s, [11])
    assert [d for d, _v, _n in flagged] == [11]
    assert flagged[0][1] == 500.0


def test_a_censored_retained_distal_channel_is_refused_not_silently_scored():
    s, a = _trace(60)
    s[10:10 + cr.CENSOR_MIN_REPEATS, 9] = 500.0     # wrist_flex.load pegged: a RETAINED channel
    with pytest.raises(cr.ContactResponseError, match="wrist_flex.load"):
        cr.trial_contact_response(s, a)


# --- 5. reached_T FROM TELEMETRY ALONE ----------------------------------------------

def test_reached_T_is_false_when_the_jaw_collapses_during_the_carry():
    """The bench sheet promises reached_T is computed offline; the symbol exists nowhere in
    `v4_contact_events.py` and `v4_bench_analysis.py:552-555` silently falls back to the
    hand-scored stage letter."""
    ok, why = cr.reached_T(_carry_trace(jaw=12.5))
    assert ok is True and why is None
    ok, why = cr.reached_T(_carry_trace(jaw=1.1))
    assert ok is False and "collapsed" in why


def _carry_trace(jaw: float):
    """pick site -> grasp -> carry -> rack area, with the jaw held at `jaw` through the carry."""
    import v4_audit
    n = 120
    s = np.zeros((n, 18), dtype=np.float64)
    s[:, 6:18] = np.linspace(0.0, 1.0, n)[:, None]
    s[:60, 0] = 68.0                                    # pan at the pick site
    s[60:, 0] = np.linspace(68.0, 15.0, n - 60)         # pan to the rack
    s[:, 3] = np.r_[np.full(70, 30.0), np.full(n - 70, 5.0)]   # wrist_flex drops into the rack pose
    s[:30, 5] = 30.0                                    # jaws open
    s[30:, 5] = jaw                                     # closed on the tube (or on nothing)
    assert v4_audit.grasp_index(s[:, 0], s[:, 5]) is not None
    return s


# --- 6. REAL DATA: the 7 telemetry CSVs ---------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("CERT-B1-01_telemetry.csv", True),
    ("CHAMF-GATE-02_telemetry.csv", False),      # "dropped it by 5 s, jaw collapsed 15.5 -> 1.1"
    ("T0-SMOKE_20260906_145245_telemetry.csv", True),   # scored SEAT at 11.9 s, held to end
])
def test_reached_T_on_the_real_sidecar_csvs(name, expected):
    import v4_contact_events as ce
    path = SCORED / name
    if not path.exists():
        pytest.skip(f"{name} not on this machine")
    state, _meta = ce.load_telemetry(path)
    assert cr.reached_T(state)[0] is expected


def test_the_two_policy_trials_have_a_real_clock_slower_than_20_hz():
    import v4_contact_events as ce
    for name in ("T0-SMOKE_20260906_145245_telemetry.csv", "h83j2kp6_20260906_161402_telemetry.csv"):
        path = SCORED / name
        if not path.exists():
            pytest.skip(f"{name} not on this machine")
        _state, meta = ce.load_telemetry(path)
        t, src = cr.sample_times(meta["n"], t_mono=meta["t_mono"])
        assert src == "t_mono"
        hz = (len(t) - 1) / t[-1]
        assert 19.0 < hz < 19.6, hz


# --- 7. REAL DATA: the training episodes, and the confound ---------------------------

@pytest.fixture(scope="module")
def v4_training():
    if not (V4_DATASET_ROOT / "meta" / "info.json").exists():
        pytest.skip("v4 dataset not on this machine")
    import v4_contact_events as ce
    states, actions, _names = ce.load_v4_dataset(V4_DATASET_ROOT)
    for e in HOLDOUT:                       # never loaded into any comparison
        states.pop(e, None)
        actions.pop(e, None)
    train = set(v4_training_episodes())
    states = {e: v for e, v in states.items() if e in train}
    actions = {e: v for e, v in actions.items() if e in train}
    assert not set(states) & set(HOLDOUT)
    import v4_manifest
    m = v4_manifest.Manifest.load(TOOLS / "v4_manifest.json")
    return states, actions, {e: m.type_of(e) for e in states}


def _stall_rate(states, actions, types, kind, cap_s=None):
    import v4_contact_events as ce
    n = fired = 0
    for e, s in states.items():
        if types[e] != kind:
            continue
        win = ce.rack_window(s, actions[e])
        if win is None:
            continue
        t = cr.sample_times(len(s))[0]
        if cap_s is not None:
            win = (win[0], min(win[1], win[0] + int(round(cap_s * 20))))
        n += 1
        fired += bool(cr.stall_onsets(s, actions[e], win, t))
    return fired, n


def test_the_stall_fires_on_every_training_resist_and_almost_no_clean(v4_training):
    states, actions, types = v4_training
    assert _stall_rate(states, actions, types, "RESIST") == (4, 4)
    assert _stall_rate(states, actions, types, "CLEAN") == (1, 30)


def test_the_stall_does_NOT_survive_its_own_duration_matched_control(v4_training):
    """THE HONEST WEAKNESS, pinned so it cannot be quietly dropped from the write-up.

    RESIST rack windows run 2.50-3.70 s; CLEAN never exceeds 1.60 s. Truncate every window to
    the longest CLEAN one and 3 of the 4 RESIST stalls vanish, because they occur at 1.70-2.45 s
    into the window -- after every CLEAN episode has already released. The 4/4-vs-1/30
    separation is therefore NOT evidence that the stall detects contact rather than duration.
    """
    states, actions, types = v4_training
    assert _stall_rate(states, actions, types, "RESIST", cap_s=1.60) == (1, 4)
    assert _stall_rate(states, actions, types, "CLEAN", cap_s=1.60) == (1, 30)


def test_the_seconds_based_stall_reproduces_the_existing_tick_based_rule_on_a_20fps_dataset(v4_training):
    """The clock change must not smuggle in a threshold change: on the dataset, whose timestamps
    are exactly 0.05 s apart, the two rules must agree episode for episode."""
    import v4_contact_events as ce
    states, actions, _types = v4_training
    for e, s in states.items():
        win = ce.rack_window(s, actions[e])
        old = ce.stall_reference(s, actions[e], win)
        new = cr.stall_onsets(s, actions[e], win, cr.sample_times(len(s))[0])
        assert (old is None) == (not new), e
        if old is not None:
            assert new[0] == old, e
