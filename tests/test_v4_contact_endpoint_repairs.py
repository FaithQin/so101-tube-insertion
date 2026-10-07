"""Co-primary endpoint 2's instrument: the PRE-REGISTERED rule on the default path, the Sep 7
amendment behind an explicit opt-in, and the correctness defects fixed on both.

THE GOVERNING RULE (Sep 14 repair, after four refuters REFUTED the Sep 7 work). `Eval Protocol —
v4 merged (Sep 6).md:56-59` pre-registers "contact response from `<trial>_telemetry.csv`
(per-trial median latency from distal-load onset to first opposing command; impulse), restricted
to trials reaching T". The DEFAULT path computes exactly that, as the pre-registration's own commit
(8958af5, Sep 6 09:03) implemented it: the search for the first opposing command runs from the
onset tick to the end of the trace, a command already opposing at the onset scores 0.00 s, the
approach direction looks back LOOKBACK_TICKS (10) ticks, and T is the ADJUDICATED stage letter.
Every Sep 7 departure — the response horizon (RESP_MAX_S), the event grace (RESP_GRACE_S), the lead
exclusion, the 0.5 s clock lookback, a telemetry reached_T in the scored column — is reachable only
through `rule=RULE_AMENDED` / `--rule amended-2026-09-07` / `--reached-T-from-telemetry`, and is a
ratification item.

WHAT IS A DEFECT, FIXED ON EVERY PATH (not a method choice):
  E1 CLOCK      times from the sidecar's own `t_mono`, never a hard-coded 20 fps (18.52-19.40 Hz
                measured on the seven CSVs on disk).
  E3' LEADS     a command already opposing at the onset is LABELLED a lead (`lead`, `lead_ticks`)
                on both rules and counted beside the median (`n_leads_in_median`); it is never
                presented as a response. Whether it SCORES 0.00 (pre-registered) or is excluded
                (amended) is the rule.
  E4 CENSORING  only `gripper.load` is written a Max_Torque_Limit (so_follower.py:180-185, inside
                `if motor == "gripper"`), so only that channel is ever flagged as censored.
  E5 SCALARS    `--scalars-out` writes the bare ledger label (the stamp is its own field), writes
                an undefined value as an EMPTY cell (never 0), and `--merge-scalars` is the one
                tested route into `scores.csv`.
  HOLD-OUT      the v4 loader filters the spent hold-out INSIDE `pd.read_parquet`; no DataFrame,
                dict or array built by this module holds a hold-out row (tested with a recording
                reader on a synthetic dataset).

READ-ONLY REAL DATA. Real-data tests skip LOUDLY when their files are absent, and each has a
fixture-free test of the same logic. No test here materialises a hold-out row: the only reads of
the v4 dataset go through `ce.load_v4_dataset`, whose read-time filter is tested below. The first
blinded pair trial's ROLLOUT dataset is never opened (a rollout's observation layout depends on the
policy); its telemetry CSV is, because the sidecar writes all 18 channels whatever the policy sees
(tools/run_scored_trial.sh:145).
"""
from __future__ import annotations

import ast
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))
import telemetry_sidecar as ts  # noqa: E402
import v4_contact_events as ce  # noqa: E402
import v4_input_ablation as ab  # noqa: E402

FPS = 20
SCORED = TOOLS / "scored_logs"
POLICY_TRIALS = ("T0-SMOKE_20260906_145245_telemetry.csv", "h83j2kp6_20260906_161402_telemetry.csv")
CERT_NAMES = [f"CERT-B1-0{k}_telemetry.csv" for k in (1, 2, 3)]
HF = Path.home() / ".cache/huggingface/lerobot/faithqin"


# --- helpers (same canvas idiom as tests/test_v4_contact_events.py) -----------------

def _baseline():
    return ce.Baseline(mean=np.zeros(18), std=np.ones(18), source="unit", n_rows=100)


def _live(s, seed=7, scale=0.01):
    """Jitter on the 12 load/current channels: `check_channels_live` refuses a frozen column, and
    0.01 against a unit sigma cannot manufacture an event."""
    s = np.asarray(s, dtype=float)
    s[:, 6:18] += np.random.default_rng(seed).normal(0.0, scale, size=(len(s), 12))
    return s


def _rack_canvas(n):
    """A state/action pair whose whole trace is the rack phase, jaws held, trigger closed."""
    s = _live(np.zeros((n, 18)))
    a = np.zeros((n, 6))
    s[:, 0] = 15.0      # pan < 40
    s[:, 3] = -5.0      # wrist_flex < 20  -> v4_audit.rack_slice covers the trace
    s[:, 5] = 13.3      # jaw held on the tube
    a[:, 5] = 1.0       # trigger closed: no release edge
    return s, a


def _press_then_reverse(n, onset, reverse_at, event_len=6):
    """The elbow descends into a contact at `onset` (a load excursion on wrist_flex) and the
    command reverses behind the position at `reverse_at`."""
    s, a = _rack_canvas(n)
    s[:, 2] = 60.0 - np.minimum(np.arange(n), onset) * 1.0        # descends, stalls at onset
    a[:, 2] = s[:, 2] - 4.0                                       # command presses on
    a[reverse_at:, 2] = s[reverse_at:, 2] + 3.0                   # ... then retreats
    s[onset:onset + event_len, 9] = 5.0                           # wrist_flex load, 5 sigma
    return s, a


def _three_event_trial(n=200, onsets=(40, 90, 140), values=(50.0, 100.0, 450.0), lags=(2, 6, 18),
                       length=4):
    """Three separate contacts. Before each onset the elbow descends 1 deg/tick for 12 ticks (so
    the approach is defined), the command presses 4 deg ahead, and it retreats for 3 ticks
    `lag` ticks after the onset. Impulse = value x length x 0.05 s on a unit baseline, so
    (50, 100, 450) -> (10, 20, 90); latency = lag / 20 -> (0.1, 0.3, 0.9) s."""
    s, a = _rack_canvas(n)
    vel = np.zeros(n)
    for o in onsets:
        vel[o - 11:o + 1] = -1.0
    s[:, 2] = 100.0 + np.cumsum(vel)
    a[:, 2] = s[:, 2] - 4.0
    for o, v, lag in zip(onsets, values, lags):
        s[o:o + length, 9] = v
        a[o + lag:o + lag + 3, 2] = s[o + lag:o + lag + 3, 2] + 3.0
    return s, a


def _skewed_clock(n, hz):
    """What the sidecar actually wrote: monotonic seconds at the loop's real rate, not 20 Hz."""
    return 1000.0 + np.arange(n, dtype=float) / hz


def _write_sidecar_csv(path, state, hz=20.0, t0=1000.0):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(ts.HEADER)
        for i, row in enumerate(np.asarray(state, dtype=float)):
            w.writerow([i, f"{t0 + i / hz:.4f}"] + [float(v) for v in row])
    return path


def _quiet_block(n=60, seed=3):
    """A live, contact-free block: unit-sigma noise on every channel."""
    return np.random.default_rng(seed).normal(0.0, 1.0, size=(n, 18))


def _write_actions_parquet(path, action):
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"episode_index": 0, "frame_index": i, "action": np.asarray(r, dtype=np.float32)}
            for i, r in enumerate(np.asarray(action, dtype=float))]
    pd.DataFrame(rows).to_parquet(path)
    return path


def _smoke_parquet():
    roots = sorted(glob.glob(str(HF / "rollout_T0-SMOKE_*")))
    if not roots:
        return None
    pq = sorted(glob.glob(str(Path(roots[0]) / "data" / "**" / "*.parquet"), recursive=True))
    return Path(pq[0]) if pq else None


def _smoke_inputs():
    """(telemetry, cert csvs, rollout parquet) or a LOUD skip naming what is missing."""
    tel, cert = SCORED / POLICY_TRIALS[0], [SCORED / c for c in CERT_NAMES]
    missing = [p.name for p in [tel, *cert] if not p.exists()]
    if missing:
        pytest.skip(f"LOUD SKIP: {missing} not on this machine — the T0-SMOKE pins are NOT checked "
                    f"by this run (the synthetic tests of the same logic still run)")
    pq = _smoke_parquet()
    if pq is None:
        pytest.skip("LOUD SKIP: the T0-SMOKE rollout dataset is not in ~/.cache/huggingface — the "
                    "real latency pins are NOT checked by this run")
    return tel, cert, pq


@pytest.fixture(scope="module")
def training():
    """The 50 v4 TRAINING episodes, read through the loader's hold-out filter."""
    ds = ce.load_v4_dataset()
    if ds is None:
        pytest.skip("LOUD SKIP: the v4 dataset is not in ~/.cache/huggingface on this machine, so "
                    "the TRAINING-episode evidence (JAW_HELD_CALIBRATION, RESP_HORIZON_PROVENANCE) "
                    "is NOT checked by this run; the fixture-free tests of the same logic still run")
    states, actions, _names = ds
    assert not set(states) & set(ce.HOLDOUT_HANDOFF)
    train = ce.training_episodes()
    assert not [e for e in train if e not in states]
    return {e: states[e] for e in train}, {e: actions[e] for e in train}


# ===================================================================================
# THE RULE — pre-registered by default, the Sep 7 amendment only when asked for
# ===================================================================================

def test_the_default_rule_is_the_preregistered_one_and_every_output_names_its_rule():
    """MUTATION TARGET: make the default rule the amendment. Every output carries `rule_version`."""
    assert ce.RULE_PREREGISTERED == "preregistered-2026-09-06"
    assert ce.RULE_AMENDED == "amended-2026-09-07"
    s, a = _press_then_reverse(80, onset=30, reverse_at=24)          # a LEAD: the rules disagree
    assert ce.analyze(s, a, _baseline())["rule_version"] == ce.RULE_PREREGISTERED
    sc = ce.trial_scalars(s, a, _baseline())
    assert sc["rule_version"] == ce.RULE_PREREGISTERED and sc["contact_latency_s"] == 0.0
    amended = ce.trial_scalars(s, a, _baseline(), rule=ce.RULE_AMENDED)
    assert amended["rule_version"] == ce.RULE_AMENDED and amended["contact_latency_s"] is None
    with pytest.raises(ValueError):
        ce.analyze(s, a, _baseline(), rule="amended")


def test_the_preregistered_rule_searches_to_the_end_of_the_trace():
    """THE T0-SMOKE SHAPE: a reversal 2.55 s after the event has ended. The pre-registered search
    (8958af5: `act[onset:]`, the whole remaining trace) takes it; the amendment does not."""
    rev = 36 + int(round(2.55 * FPS))                                  # event [30, 36) + 2.55 s
    s, a = _press_then_reverse(200, onset=30, reverse_at=rev)
    pre = ce.analyze(s, a, _baseline(), rule=ce.RULE_PREREGISTERED)["events"][0]
    amd = ce.analyze(s, a, _baseline(), rule=ce.RULE_AMENDED)["events"][0]
    assert pre["opposing_frame"] == rev and pre["latency_s"] == pytest.approx((rev - 30) / FPS)
    assert amd["latency_s"] is None and amd["opposing_frame"] is None and amd["latency_reason"]
    with pytest.raises(ValueError):          # the pre-registered rule has no horizon to pass
        ce.first_opposing_command(s, a, onset=30, rule=ce.RULE_PREREGISTERED, hard_end=40)


def test_a_lead_scores_zero_under_the_preregistered_rule_and_is_labelled_a_lead_on_both():
    """8958af5 searched `t >= onset`, so a command already opposing at the onset scored 0.00 s.
    That number is the pre-registered rule's; presenting it as a RESPONSE was the defect, so the
    lead is labelled on both rules and counted beside the median."""
    s, a = _press_then_reverse(80, onset=30, reverse_at=24)
    pre = ce.first_opposing_command(s, a, onset=30, rule=ce.RULE_PREREGISTERED)
    assert pre["latency_s"] == 0.0 and pre["frame"] == 30
    assert pre["lead"] is True and pre["lead_ticks"] == 7 and pre["lead_s"] == pytest.approx(6 / FPS)
    amd = ce.first_opposing_command(s, a, onset=30, rule=ce.RULE_AMENDED)
    assert amd["latency_s"] is None and amd["frame"] is None
    assert amd["lead"] is True and amd["lead_ticks"] == 7 and amd["reason"]
    sc = ce.trial_scalars(s, a, _baseline(), rule=ce.RULE_PREREGISTERED)
    assert sc["n_latencies"] == 1 and sc["n_leads_in_median"] == 1
    assert ce.trial_scalars(s, a, _baseline(), rule=ce.RULE_AMENDED)["n_leads_in_median"] == 0
    press, pa = _press_then_reverse(80, onset=30, reverse_at=45)        # an ordinary response
    r = ce.first_opposing_command(press, pa, onset=30, rule=ce.RULE_PREREGISTERED)
    assert r["lead"] is False and r["frame"] == 45


def test_the_lead_walk_back_counts_only_the_consecutive_run_ending_at_the_onset():
    pos, act = _press_then_reverse(80, onset=30, reverse_at=24)
    act[26, 2] = pos[26, 2] - 4.0                    # one tick of pressing breaks the run
    for rule in ce.RULES:
        assert ce.first_opposing_command(pos, act, onset=30, rule=rule)["lead_ticks"] == 4   # ticks 27..30


def test_the_preregistered_lookback_is_ten_ticks_the_amended_one_is_half_a_second_of_clock():
    """8958af5: `ref = max(onset - lookback, 0)`, in TICKS. The amendment converted it to 0.5 s of
    the real clock, which moves the reference by one tick at 19.35 Hz. Built so that one tick
    decides whether the arm was moving into the contact."""
    n, hz, onset = 80, 19.352, 30
    s, a = _rack_canvas(n)
    s[:, 2] = 60.0
    s[:21, 2] = 61.2                         # the elbow last moved between ticks 20 and 21
    a[:, 2] = s[:, 2] - 4.0
    a[40:, 2] = s[40:, 2] + 3.0
    tt = _skewed_clock(n, hz) - 1000.0
    assert int(np.searchsorted(tt, tt[onset] - ce.LOOKBACK_TICKS / FPS, side="left")) == 21
    pre = ce.first_opposing_command(s, a, onset, t=tt, rule=ce.RULE_PREREGISTERED)
    amd = ce.first_opposing_command(s, a, onset, t=tt, rule=ce.RULE_AMENDED)
    assert pre["approach_motion_deg"] == pytest.approx(1.2) and pre["frame"] == 40
    assert amd["approach_motion_deg"] == pytest.approx(0.0) and amd["latency_s"] is None


# ===================================================================================
# E1 — THE CLOCK (a defect on both rules)
# ===================================================================================

def test_sample_times_prefers_the_sidecars_own_clock_and_names_the_fallback():
    t, src = ce.sample_times(40, t_mono=_skewed_clock(40, 19.352))
    assert src == "t_mono"
    assert t[0] == 0.0 and t[-1] == pytest.approx(39 / 19.352)
    t2, src2 = ce.sample_times(40)
    assert t2[-1] == pytest.approx(39 / 20.0)
    assert src2 != "t_mono"


def test_sample_times_refuses_a_clock_that_is_not_monotonic():
    bad = _skewed_clock(20, 19.4)
    bad[10] = bad[9]
    with pytest.raises(ce.TelemetryError):
        ce.sample_times(20, t_mono=bad)


@pytest.mark.parametrize("rule", ["preregistered-2026-09-06", "amended-2026-09-07"])
def test_every_time_analyze_reports_comes_from_the_real_clock(rule):
    """MUTATION TARGET: revert `analyze` to `ev.onset / fps`. At 19.352 Hz a 20 fps clock
    understates a 3.0 s onset by 97 ms and every dwell by 3.3 %."""
    n, hz = 120, 19.352
    s, a = _rack_canvas(n)
    s[60:70, 9] = 5.0
    out = ce.analyze(s, a, _baseline(), t_mono=_skewed_clock(n, hz), rule=rule)
    assert out["clock"] == "t_mono"
    assert out["duration_s"] == pytest.approx((n - 1) / hz)
    assert out["loop_hz"] == pytest.approx(hz, abs=1e-6)
    e = out["events"][0]
    assert e["onset_s"] == pytest.approx(60 / hz) and e["onset_s"] != pytest.approx(60 / 20.0)
    assert e["dwell_s"] == pytest.approx(10 / hz)


def test_the_impulse_is_integrated_on_the_real_clock_not_divided_by_a_nominal_fps():
    """The impulse is load x TIME (mutation: `excess.sum() / fps`)."""
    n = 60
    s, a = _rack_canvas(n)
    s[20:30, 9] = 4.0
    nominal = ce.analyze(s, a, _baseline())["events"][0]["impulse_load_s"]
    real = ce.analyze(s, a, _baseline(), t_mono=_skewed_clock(n, 18.523))["events"][0]["impulse_load_s"]
    assert real == pytest.approx(nominal * 20.0 / 18.523, rel=1e-6)


@pytest.mark.parametrize("name,lo,hi", [
    ("CERT-B1-01_telemetry.csv", 18.4, 18.7),
    ("T0-SMOKE_20260906_145245_telemetry.csv", 19.3, 19.5),
    ("h83j2kp6_20260906_161402_telemetry.csv", 19.3, 19.5),
])
def test_the_real_csvs_are_timed_on_their_own_clock_which_is_not_20_hz(name, lo, hi):
    path = SCORED / name
    if not path.exists():
        pytest.skip(f"LOUD SKIP: {name} not on this machine — the real clock is NOT checked by this run")
    state, meta = ce.load_telemetry(path)
    bl = ce.Baseline(mean=state.mean(0), std=state.std(0) + 1.0, source="unit", n_rows=len(state))
    out = ce.analyze(state, None, bl, t_mono=meta["t_mono"])
    assert out["clock"] == "t_mono"
    assert lo < out["loop_hz"] < hi
    assert out["duration_s"] > (out["n_frames"] - 1) / 20.0


def test_the_first_tick_of_a_policy_trial_takes_half_a_second_and_the_nominal_clock_hides_it():
    """Measured Sep 7: the FIRST interval of each policy trial is 0.520 s (T0-SMOKE) and 0.582 s
    (the blinded trial) against a 0.050 s median, so nominal 0.75 s is really 1.22 s."""
    for name, first_dt, real_t15 in (("T0-SMOKE_20260906_145245_telemetry.csv", 0.520, 1.2198),
                                     ("h83j2kp6_20260906_161402_telemetry.csv", 0.582, 1.2808)):
        path = SCORED / name
        if not path.exists():
            pytest.skip(f"LOUD SKIP: {name} not on this machine")
        _s, meta = ce.load_telemetry(path)
        t, src = ce.sample_times(meta["n"], t_mono=meta["t_mono"])
        assert src == "t_mono"
        assert t[1] == pytest.approx(first_dt, abs=5e-3)
        assert np.median(np.diff(t)) == pytest.approx(0.05, abs=1e-3)
        assert t[15] == pytest.approx(real_t15, abs=5e-3) and t[15] > 1.5 * (15 / 20.0)


# ===================================================================================
# THE AMENDMENT'S OWN MECHANICS (opt-in) — each bound alone, and its declared band
# ===================================================================================

def test_the_amended_search_stops_at_the_hard_end():
    pos, act = _press_then_reverse(80, onset=30, reverse_at=45)
    assert ce.first_opposing_command(pos, act, onset=30, rule=ce.RULE_AMENDED)["frame"] == 45
    r = ce.first_opposing_command(pos, act, onset=30, hard_end=40, rule=ce.RULE_AMENDED)
    assert r["frame"] is None and r["latency_s"] is None and r["reason"]


def test_the_amended_search_stops_at_the_declared_response_horizon():
    """The horizon ALONE excludes this one: no hard_end is passed (mutation: drop the RESP_MAX_S
    bound)."""
    late = int(ce.RESP_MAX_S * FPS) + 40
    pos, act = _press_then_reverse(late + 20, onset=30, reverse_at=late)
    assert ce.first_opposing_command(pos, act, onset=30, rule=ce.RULE_AMENDED)["frame"] is None
    assert ce.first_opposing_command(pos, act, onset=30, rule=ce.RULE_PREREGISTERED)["frame"] == late


def test_a_reversal_after_the_event_has_ended_is_not_that_events_amended_latency():
    """The event's grace ALONE excludes the far one: it is 1.8 s after the onset, inside RESP_MAX_S
    (mutation: drop `hard_end` from `describe_event`)."""
    inside = int((6 + ce.RESP_GRACE_S * FPS) - 2)
    s, a = _press_then_reverse(200, onset=30, reverse_at=30 + inside)
    near = ce.analyze(s, a, _baseline(), rule=ce.RULE_AMENDED)["events"][0]
    assert near["latency_s"] == pytest.approx(inside / FPS)

    far_at = 30 + int(6 + ce.RESP_GRACE_S * FPS) + 10
    assert (far_at - 30) / FPS < ce.RESP_MAX_S
    s2, a2 = _press_then_reverse(200, onset=30, reverse_at=far_at)
    far = ce.analyze(s2, a2, _baseline(), rule=ce.RULE_AMENDED)["events"][0]
    assert far["latency_s"] is None and far["latency_reason"]


def test_on_the_real_smoke_trial_the_short_events_later_reversal_is_outside_BOTH_amended_bounds():
    """T0-SMOKE, certification baseline. The pre-registered search hands the event at tick 15 a
    reversal that is MORE than RESP_MAX_S after its onset AND more than RESP_GRACE_S after its end,
    so on this event either bound alone excludes it: this test pins those two gaps, and the
    per-bound mutations are killed by the two synthetic tests above, not here."""
    tel, cert, pq = _smoke_inputs()
    state, meta = ce.load_telemetry(tel)
    act = ce.load_actions_parquet(pq, None)
    bl = ce.Baseline.from_csvs(cert)
    t, _src = ce.sample_times(len(state), t_mono=meta["t_mono"])
    pre = ce.analyze(state, act, bl, t_mono=meta["t_mono"], rule=ce.RULE_PREREGISTERED)["events"][0]
    amd = ce.analyze(state, act, bl, t_mono=meta["t_mono"], rule=ce.RULE_AMENDED)["events"][0]
    assert pre["onset"] == amd["onset"] == 15 and amd["dwell_s"] < 0.7
    rev = pre["opposing_frame"]
    assert rev is not None and t[rev] - t[15] > ce.RESP_MAX_S
    assert t[rev] - amd["end_s"] > ce.RESP_GRACE_S
    assert amd["latency_s"] is None and amd["opposing_frame"] is None


def test_the_amended_rules_sensitivity_band_on_the_smoke_trial(monkeypatch):
    """The two declared constants are knobs on a latency; their band on the one real trial with a
    rollout is pinned so a change cannot pass silently. (median s, n latencies) per
    (RESP_MAX_S, RESP_GRACE_S)."""
    tel, cert, pq = _smoke_inputs()
    state, meta = ce.load_telemetry(tel)
    act = ce.load_actions_parquet(pq, None)
    bl = ce.Baseline.from_csvs(cert)
    got = {}
    for rmax in (2.0, 3.0, 4.0, 5.0):
        for grace in (0.25, 0.5, 1.0, 2.0, 3.0):
            monkeypatch.setattr(ce, "RESP_MAX_S", rmax)
            monkeypatch.setattr(ce, "RESP_GRACE_S", grace)
            sc = ce.trial_scalars(state, act, bl, t_mono=meta["t_mono"], rule=ce.RULE_AMENDED)
            got[(rmax, grace)] = (round(sc["contact_latency_s"], 2), sc["n_latencies"])
    for rmax in (2.0, 3.0, 4.0, 5.0):
        for grace in (0.25, 0.5, 1.0, 2.0):
            assert got[(rmax, grace)] == (0.25, 2), (rmax, grace)
    assert got[(2.0, 3.0)] == (0.25, 2)
    assert got[(3.0, 3.0)] == (0.40, 3)
    assert got[(4.0, 3.0)] == got[(5.0, 3.0)] == (1.54, 4)


def test_the_declared_constants_that_move_a_latency_are_pinned():
    """On T0-SMOKE the amended median is 0.25 s for every RESP_GRACE_S from 0.25 to 2.0 s, so no
    real-data pin can see a change to it; the declared values are pinned here instead, beside the
    pre-registered ones. Changing any of them is a ratification decision, not an edit."""
    assert (ce.LOOKBACK_TICKS, ce.MOTION_MIN_DEG, ce.DEADBAND_DEG, ce.ONSET_TICKS, ce.OFF_TICKS) == (10, 1.0, 1.0, 3, 3)
    assert (ce.RESP_MAX_S, ce.RESP_GRACE_S) == (3.0, 1.0)
    assert ce.DEFAULT_RULE == ce.RULE_PREREGISTERED and ce.RULES == (ce.RULE_PREREGISTERED, ce.RULE_AMENDED)


def test_the_response_horizon_rationale_is_derived_from_the_training_rack_windows(training):
    states, actions = training
    m = ce._manifest()
    ticks = {}
    for e in states:
        w = ce.rack_window(states[e], actions[e])
        if w is not None:
            ticks[e] = w[1] - w[0]
    longest = max(ticks, key=ticks.get)
    rec = ce.RESP_HORIZON_PROVENANCE
    assert (ticks[longest], longest, m.type_of(longest)) == (
        rec["longest_training_rack_window_ticks"], rec["episode"], rec["type"])
    assert tuple(sorted(v for e, v in ticks.items() if m.type_of(e) == "RESIST")) == rec["resist_window_ticks"]


def test_the_amended_horizon_does_not_span_every_training_contact_phase_and_the_record_says_so():
    """The Sep 7 comment said 3 s 'already spans an entire contact phase' beside a 3.70 s window.
    The record states the truth and this pins it; ratifying a different horizon means restating it."""
    rec = ce.RESP_HORIZON_PROVENANCE
    assert ce.RESP_MAX_S * ce.FPS < rec["longest_training_rack_window_ticks"]
    longer = sum(w > ce.RESP_MAX_S * ce.FPS for w in rec["resist_window_ticks"])
    assert longer == rec["resist_windows_longer_than_horizon"] == 3


# ===================================================================================
# THE PER-TRIAL SCALARS — the aggregator, the denominators, the pins
# ===================================================================================

@pytest.mark.parametrize("rule", ["preregistered-2026-09-06", "amended-2026-09-07"])
def test_the_latency_and_the_impulse_are_the_MEDIAN_over_the_trials_events(rule):
    """§1: the 'per-trial median'. Three events, so median, max and sum all differ
    (mutations: np.median -> np.max / np.sum on either scalar)."""
    s, a = _three_event_trial()
    ev = ce.analyze(s, a, _baseline(), rule=rule)["events"]
    assert [e["latency_s"] for e in ev] == pytest.approx([0.1, 0.3, 0.9])
    assert [e["impulse_load_s"] for e in ev] == pytest.approx([10.0, 20.0, 90.0], rel=1e-3)
    sc = ce.trial_scalars(s, a, _baseline(), rule=rule)
    assert sc["contact_latency_s"] == pytest.approx(0.3)
    assert sc["contact_impulse"] == pytest.approx(20.0, rel=1e-3)
    assert sc["contact_impulse_sum"] == pytest.approx(120.0, rel=1e-3)
    assert sc["n_events"] == 3 and sc["n_latencies"] == 3


def test_the_scalars_report_how_many_latencies_came_from_the_rack_window():
    """§1 does not restrict the median to the rack window, so the default does not either — but
    the denominator is reported, because on T0-SMOKE none of them did."""
    s, a = _three_event_trial()
    s[:130, 3] = 30.0                        # not in the rack area before tick 130: only the last event is
    sc = ce.trial_scalars(s, a, _baseline())
    assert sc["contact_latency_s"] == pytest.approx(0.3) and sc["n_latencies"] == 3
    assert sc["n_events_in_rack_window"] == 1 and sc["n_latencies_in_rack_window"] == 1
    assert sc["latency_from_rack_window"] is True
    s[:, 3] = 30.0                           # no rack phase at all
    sc2 = ce.trial_scalars(s, a, _baseline())
    assert sc2["n_latencies"] == 3 and sc2["n_latencies_in_rack_window"] == 0
    assert sc2["latency_from_rack_window"] is False


@pytest.mark.parametrize("rule,latency,n_lat,n_leads", [
    ("preregistered-2026-09-06", 0.39999999996, 5, 1),
    ("amended-2026-09-07", 0.24994999997, 2, 0),
])
def test_the_cli_pins_the_smoke_trial_scalars(tmp_path, rule, latency, n_lat, n_leads):
    """The published numbers, pinned. The pre-registered median is 8958af5's 0.40 s (5 of 6 events,
    one of them a LEAD scored 0.00), now on the t_mono clock; the amendment's is 0.25 s (2 of 6).
    None of the 6 events is in the rack window. Mutation: call trial_scalars without t_mono."""
    tel, cert, pq = _smoke_inputs()
    out, js = tmp_path / "s.csv", tmp_path / "s.json"
    args = ["--telemetry", str(tel), "--baseline", *[str(c) for c in cert], "--actions-parquet", str(pq),
            "--scalars-out", str(out), "--json", str(js)]
    if rule != ce.RULE_PREREGISTERED:
        args += ["--rule", rule]
    assert ce.main(args) == 0
    row = next(csv.DictReader(open(out, newline="")))
    sc = json.load(open(js))["scalars"]
    assert row["label"] == "T0-SMOKE" and sc["stamp"] == "20260906_145245"
    assert float(row["contact_latency_s"]) == pytest.approx(latency, abs=1e-6)
    assert float(row["contact_impulse"]) == pytest.approx(239.148083, rel=1e-6)
    assert row["reached_T"] == ""
    assert sc["rule_version"] == rule and sc["clock"] == "t_mono"
    assert sc["n_events"] == 6 and sc["n_latencies"] == n_lat and sc["n_leads_in_median"] == n_leads
    assert sc["impulse_is_lower_bound"] is True and sc["censored"] == {"gripper.load": 1}
    assert sc["n_latencies_in_rack_window"] == 0 and sc["latency_from_rack_window"] is False
    assert sc["reached_T_telemetry"] == 1


def test_under_either_rule_every_zero_latency_on_the_smoke_trial_is_a_labelled_lead():
    tel, cert, pq = _smoke_inputs()
    state, meta = ce.load_telemetry(tel)
    act = ce.load_actions_parquet(pq, None)
    bl = ce.Baseline.from_csvs(cert)
    for rule in ce.RULES:
        ev = ce.analyze(state, act, bl, t_mono=meta["t_mono"], rule=rule)["events"]
        zero = [e for e in ev if e["latency_s"] == 0.0]
        assert all(e["lead"] for e in zero)
        assert sum(bool(e["lead"]) for e in ev) == 1
    amd = ce.analyze(state, act, bl, t_mono=meta["t_mono"], rule=ce.RULE_AMENDED)["events"]
    assert not [e for e in amd if e["latency_s"] == 0.0]


# ===================================================================================
# E4 — THE CENSORED CHANNEL IS THE GRIPPER, AND ONLY THE GRIPPER
# ===================================================================================

def _torque_cap_writes(src: str) -> list[tuple[int, str | None]]:
    """(value, the motor name its enclosing `if motor == "<name>"` pins, or None) for every
    `*.write("Max_Torque_Limit", motor, <int>)` in `src`. Parsed with `ast`, never grepped."""
    tree = ast.parse(src)
    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "write"
                and len(node.args) == 3 and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "Max_Torque_Limit" and isinstance(node.args[2], ast.Constant)):
            continue
        guard, p = None, parent.get(node)
        while p is not None:
            if (isinstance(p, ast.If) and isinstance(p.test, ast.Compare) and isinstance(p.test.left, ast.Name)
                    and p.test.left.id == "motor" and len(p.test.ops) == 1 and isinstance(p.test.ops[0], ast.Eq)
                    and isinstance(p.test.comparators[0], ast.Constant)):
                guard = p.test.comparators[0].value
                break
            p = parent.get(p)
        out.append((node.args[2].value, guard))
    return out


def test_the_torque_cap_is_read_off_the_follower_patch_and_applies_to_the_gripper_only():
    """so_follower.py:180 `if motor == "gripper":` / :185 `self.bus.write("Max_Torque_Limit", motor,
    500)`. The cap's VALUE and the MOTOR it is written to are both pinned, and the check itself is
    shown to see the difference: on an in-memory copy of the source with the write MOVED out of the
    gripper branch (onto every motor) it reports no guard, and on a write guarded by another motor
    it names that motor. The installed file is only read."""
    import lerobot
    src = (Path(lerobot.__file__).resolve().parent / "robots" / "so_follower" / "so_follower.py").read_text()
    writes = _torque_cap_writes(src)
    assert writes == [(500, "gripper")], writes
    assert ce.MAX_TORQUE_LIMIT == 500.0
    assert [ce.state_columns()[d] for d in ce.CENSORED_LOAD_DIMS] == ["gripper.load"]
    line = '                    self.bus.write("Max_Torque_Limit", motor, 500)  # upstream: 500\n'
    moved = src.replace(line, "", 1).replace(
        '                if motor == "gripper":\n',
        '                self.bus.write("Max_Torque_Limit", motor, 500)\n                if motor == "gripper":\n', 1)
    assert moved != src and _torque_cap_writes(moved) == [(500, None)]
    other = 'for motor in m:\n    if motor == "elbow_flex":\n        self.bus.write("Max_Torque_Limit", motor, 500)\n'
    assert _torque_cap_writes(other) == [(500, "elbow_flex")]


def test_only_the_gripper_is_censored_at_its_torque_cap():
    """MUTATION TARGET: censored_ticks' default dims back to all six load channels. An arm joint
    has no 500 cap (T0-SMOKE shoulder_lift.load reaches -842), so passing through it is data."""
    s, a = _rack_canvas(60)
    s[20:23, 9] = [-495.0, -ce.MAX_TORQUE_LIMIT, -506.0]            # wrist_flex passes through -500
    assert ce.censored_ticks(s) == {}
    out = ce.analyze(s, a, _baseline())
    e = out["events"][0]
    assert e["joint"] == "wrist_flex" and e["peak_is_lower_bound"] is False
    assert e["censored_ticks_in_event"] == 0 and out["censored"] == {}

    p, pa = _rack_canvas(60)
    p[20:23, 9] = [-495.0, -ce.MAX_TORQUE_LIMIT, -499.0]             # ... and PEAKS exactly at -500
    op = ce.analyze(p, pa, _baseline())
    assert op["z_max_is_lower_bound"] is False and op["events"][0]["peak_is_lower_bound"] is False
    assert ce.trial_scalars(p, pa, _baseline())["impulse_is_lower_bound"] is False

    g, ga = _rack_canvas(60)
    g[10:14, 11] = ce.MAX_TORQUE_LIMIT                               # gripper.load saturated
    assert ce.censored_ticks(g) == {"gripper.load": 4}
    og = ce.analyze(g, ga, _baseline())
    assert og["z_max_is_lower_bound"] is True and og["events"][0]["peak_is_lower_bound"] is True
    assert og["events"][0]["censored_ticks_in_event"] == 4
    assert ce.trial_scalars(g, ga, _baseline())["impulse_is_lower_bound"] is True


def test_an_uncensored_trial_is_not_flagged():
    s, a = _rack_canvas(60)
    s[20:26, 9] = 400.0
    out = ce.analyze(s, a, _baseline())
    assert out["censored"] == {} and out["z_max_is_lower_bound"] is False
    assert out["events"][0]["peak_is_lower_bound"] is False


@pytest.mark.parametrize("name,expected", [
    ("T0-SMOKE_20260906_145245_telemetry.csv", {"gripper.load": 1}),
    ("CERT-B1-01_telemetry.csv", {"gripper.load": 1}),
])
def test_on_the_real_csvs_only_the_gripper_is_reported_censored(name, expected):
    """Before this repair: T0-SMOKE {'shoulder_lift.load': 1, 'gripper.load': 1} and CERT-B1-01
    {'shoulder_lift.load': 1, 'wrist_roll.load': 1, 'gripper.load': 1}."""
    path = SCORED / name
    if not path.exists():
        pytest.skip(f"LOUD SKIP: {name} not on this machine")
    state, _meta = ce.load_telemetry(path)
    assert ce.censored_ticks(state) == expected


@pytest.mark.parametrize("name", POLICY_TRIALS)
def test_both_real_policy_trials_are_flagged_censored_on_the_gripper(name):
    path = SCORED / name
    if not path.exists():
        pytest.skip(f"LOUD SKIP: {name} not on this machine")
    state, _meta = ce.load_telemetry(path)
    assert ce.censored_ticks(state).get("gripper.load", 0) >= 1
    cert = [SCORED / c for c in CERT_NAMES]
    if not all(c.exists() for c in cert):
        pytest.skip("LOUD SKIP: the certification replays are not on this machine")
    out = ce.analyze(state, None, ce.Baseline.from_csvs(cert))
    assert out["z_max"] == pytest.approx(4.352, abs=5e-3)
    assert out["z_max_channel"] == "gripper.load" and out["z_max_is_lower_bound"] is True


# ===================================================================================
# reached_T — the ADJUDICATED letter by default; the telemetry proxy beside it
# ===================================================================================

def _carry(jaw, rack_samples=None):
    """Grasp at the pick site, carry to the rack; `jaw` is the observed jaw width through it.
    `rack_samples` limits how many samples end in the rack area."""
    import v4_audit
    n = 120
    s = _live(np.zeros((n, 18)))
    s[:60, 0] = 68.0
    s[60:, 0] = np.linspace(68.0, 15.0, n - 60)
    s[:, 3] = np.r_[np.full(70, 30.0), np.full(n - 70, 5.0)]
    s[:30, 5] = 30.0
    s[30:, 5] = jaw
    if rack_samples is not None:
        rack = v4_audit.rack_slice(s[:, 0], s[:, 3])
        s[rack[rack_samples:], 3] = 30.0
    assert v4_audit.grasp_index(s[:, 0], s[:, 5]) is not None
    return s


def test_reached_T_is_computed_from_telemetry_and_a_collapsed_jaw_is_not_a_carry():
    ok, why = ce.reached_T(_carry(13.2))
    assert ok is True and why is None
    bad, why2 = ce.reached_T(_carry(1.1))
    assert bad is False and why2


def test_reached_T_needs_ten_rack_samples_not_one():
    """MUTATION TARGET: delete the `len(rack) < 10` guard. Five samples in the rack area is a
    pass through it, not an arrival."""
    assert ce.reached_T(_carry(13.2, rack_samples=5))[0] is False
    assert ce.reached_T(_carry(13.2, rack_samples=12))[0] is True


@pytest.mark.parametrize("jaw,expected", [
    (1.07, False), (1.21, False), (1.36, False), (1.86, False),   # the closed stop, on nothing
    (8.85, True), (11.42, True),                                   # the smallest real held carries
])
def test_reached_T_rejects_closed_jaw_carries_at_the_hard_stop(jaw, expected):
    """The rollouts on disk close on nothing at 1.07-1.36 (list below); held carries never go
    below 8.85. MUTATION TARGET: JAW_HELD_MIN_DEG = 1.2 turns 1.21 / 1.36 / 1.86 red."""
    assert ce.reached_T(_carry(jaw))[0] is expected


def test_carry_jaw_min_is_the_jaw_minimum_between_the_grasp_and_rack_entry():
    for jaw in (1.07, 8.85, 13.2):
        assert ce.carry_jaw_min(_carry(jaw)) == pytest.approx(jaw)
    never = _carry(13.2)
    never[:, 0] = 68.0
    assert ce.carry_jaw_min(never) is None


# The rack-reaching, grasp-found rollouts on disk whose carry sits at the jaw's closed stop
# (carry minimum 1.071-1.356), found Sep 14 by scanning every rollout_* dataset EXCEPT the blinded
# trial's: 45 grasped + rack-reaching episodes, 12 of them here, the other 33 at >= 8.851.
ROLLOUT_CLOSED_JAW = (
    "rollout_P05-02_20260903_202804",
    "rollout_trial_B_v2_20260825_181012_20260825_181020",
    "rollout_P05-01r_20260903_203541",
    "rollout_P05-02r_20260903_204106",
    "rollout_SV-A-01r_20260904_163352",
    "rollout_probe_b50_v2_20260823_150050_20260823_150058",
    "rollout_trial_A_v2_20260825_192231_20260825_192238",
    "rollout_trial_B_v2_20260825_201132_20260825_201139",
    "rollout_trial_B_v2_20260825_201410_20260825_201420",
    "rollout_trial_B_v2_20260825_201626_20260825_201633",
    "rollout_B2V-B-01_20260830_233916",
    "rollout_SV-B-01_20260904_155236",
)


def _rollout_positions(name):
    import pandas as pd
    pq = sorted(glob.glob(str(HF / name / "data" / "**" / "*.parquet"), recursive=True))
    df = pd.concat(pd.read_parquet(p, columns=["episode_index", "frame_index", "observation.state"]) for p in pq)
    g = df[df["episode_index"] == 0].sort_values("frame_index")
    return np.stack(g["observation.state"].to_numpy()).astype(np.float64)[:, :6]


@pytest.fixture(scope="module")
def closed_jaw_rollouts():
    import re
    token_shaped = [n for n in ROLLOUT_CLOSED_JAW if re.fullmatch(r"rollout_[a-z0-9]{8}_\d{8}_\d{6}", n)]
    assert not token_shaped, "a blinded-token-shaped rollout is in the negative control"
    missing = [n for n in ROLLOUT_CLOSED_JAW if not (HF / n).exists()]
    if missing:
        pytest.skip(f"LOUD SKIP: {len(missing)} of the closed-jaw rollouts are not in ~/.cache/huggingface "
                    f"(e.g. {missing[:2]}) — the real negative control is NOT checked by this run")
    return {n: _rollout_positions(n) for n in ROLLOUT_CLOSED_JAW}


def test_reached_T_negative_control_on_the_rack_reaching_closed_jaw_rollouts(closed_jaw_rollouts):
    """Twelve real carries that reached the rack area with nothing in the jaws. At 5.0 all twelve
    are refused; at 1.2, ten of them would pass (mutation check)."""
    verdicts = {n: ce.reached_T(p) for n, p in closed_jaw_rollouts.items()}
    assert not [n for n, (ok, _w) in verdicts.items() if ok]
    mins = {n: ce.carry_jaw_min(p) for n, p in closed_jaw_rollouts.items()}
    assert all(m is not None and 1.0 < m < 1.4 for m in mins.values()), mins


def test_the_demonstrations_carry_no_information_about_the_jaw_threshold(training, closed_jaw_rollouts):
    """Why the Sep 7 held-out check of reached_T (8/8) could not have failed: every v4
    demonstration continues to the seat (Start Schedule — v4.md:16), so the verdict is identical at
    ANY threshold from 0.0 to the smallest training carry. Only a set with negatives discriminates."""
    states, _actions = training
    at0 = [ce.reached_T(s, jaw_held_min=0.0)[0] for s in states.values()]
    at5 = [ce.reached_T(s, jaw_held_min=ce.JAW_HELD_MIN_DEG)[0] for s in states.values()]
    assert at0 == at5 and sum(at5) == len(states) == 50
    flips = [n for n, p in closed_jaw_rollouts.items()
             if ce.reached_T(p, jaw_held_min=0.0)[0] != ce.reached_T(p)[0]]
    assert len(flips) == 12


def test_the_jaw_threshold_calibration_record_matches_the_bench_csvs():
    """56 held carries, not 57: CHAMF-GATE-02 is the negative, not a fifth replay positive. The
    calibration used both policy trials, including the first blinded pair trial."""
    rec = ce.JAW_HELD_CALIBRATION
    replays = ("CERT-B1-01", "CERT-B1-02", "CERT-B1-03", "CHAMF-GATE-01", "CHAMF-GATE-02")
    policy = tuple(p.replace("_telemetry.csv", "") for p in POLICY_TRIALS)
    mins = {}
    for stem in replays + policy:
        p = SCORED / f"{stem}_telemetry.csv"
        if not p.exists():
            pytest.skip(f"LOUD SKIP: {p.name} not on this machine — the calibration counts are NOT checked")
        mins[stem] = ce.carry_jaw_min(ce.load_telemetry(p)[0])
    thr = ce.JAW_HELD_MIN_DEG
    assert tuple(sorted(k for k, v in mins.items() if v < thr)) == rec["negatives"] == ("CHAMF-GATE-02",)
    replay_pos = tuple(k for k in replays if mins[k] >= thr)
    assert replay_pos == rec["replay_positives"] and len(replay_pos) == rec["held_carries"]["replay CSVs"] == 4
    assert all(mins[k] >= thr for k in policy) and rec["held_carries"]["policy trials"] == len(policy) == 2
    assert min(mins[k] for k in replay_pos) == pytest.approx(rec["min_held"]["replay CSVs"], abs=0.005)
    assert min(mins[k] for k in policy) == pytest.approx(rec["min_held"]["policy trials"], abs=0.005)
    assert mins["CHAMF-GATE-02"] == pytest.approx(rec["max_negative_deg"], abs=0.005)
    assert sum(rec["held_carries"].values()) == rec["n_held_carries"] == 56


def test_the_jaw_threshold_is_the_midpoint_its_record_says_it_is():
    """Fixture-free half of the calibration: the constant, the record and the rule that joins
    them. MUTATION TARGET: move JAW_HELD_MIN_DEG by a degree."""
    rec = ce.JAW_HELD_CALIBRATION
    assert rec["threshold_deg"] == ce.JAW_HELD_MIN_DEG
    assert ce.reached_T.__defaults__[0] == ce.JAW_HELD_MIN_DEG
    gap_top = min(rec["min_held"].values())
    assert ce.JAW_HELD_MIN_DEG == pytest.approx((rec["max_negative_deg"] + gap_top) / 2, abs=0.05)
    assert sum(rec["held_carries"].values()) == rec["n_held_carries"]


def test_the_jaw_threshold_calibration_record_matches_the_training_demonstrations(training):
    states, _actions = training
    mins = {e: ce.carry_jaw_min(s) for e, s in states.items()}
    assert all(v is not None for v in mins.values())
    assert len(mins) == ce.JAW_HELD_CALIBRATION["held_carries"]["training demonstrations"] == 50
    assert min(mins.values()) == pytest.approx(
        ce.JAW_HELD_CALIBRATION["min_held"]["training demonstrations"], abs=0.005)


@pytest.mark.parametrize("name,expected", [
    ("CERT-B1-01_telemetry.csv", True),
    ("CHAMF-GATE-02_telemetry.csv", False),
    ("T0-SMOKE_20260906_145245_telemetry.csv", True),
])
def test_reached_T_on_the_real_sidecar_csvs(name, expected):
    path = SCORED / name
    if not path.exists():
        pytest.skip(f"LOUD SKIP: {name} not on this machine")
    state, _meta = ce.load_telemetry(path)
    assert ce.reached_T(state)[0] is expected


def test_trial_scalars_leaves_reached_T_to_the_adjudicated_letter_and_reports_the_proxy_beside_it():
    """§1 restricts to trials reaching T of the ADJUDICATED taxonomy. The proxy is a cross-check
    unless --reached-T-from-telemetry is passed (mutation: wire `ok_T` to True)."""
    dropped = ce.trial_scalars(_carry(1.1), None, _baseline(), label="X")
    assert dropped["reached_T"] is None and dropped["reached_T_source"] == ce.REACHED_T_ADJUDICATED
    assert dropped["reached_T_telemetry"] == 0 and dropped["reached_T_telemetry_reason"]
    held = ce.trial_scalars(_carry(13.2), None, _baseline(), label="X")
    assert held["reached_T"] is None and held["reached_T_telemetry"] == 1
    opt = ce.trial_scalars(_carry(1.1), None, _baseline(), reached_T_source=ce.REACHED_T_TELEMETRY)
    assert opt["reached_T"] == 0 and opt["reached_T_source"] == ce.REACHED_T_TELEMETRY
    assert ce.trial_scalars(_carry(13.2), None, _baseline(), reached_T_source=ce.REACHED_T_TELEMETRY)["reached_T"] == 1


def test_the_cli_writes_reached_T_blank_unless_the_telemetry_proxy_is_asked_for(tmp_path):
    tel, cert = SCORED / "CHAMF-GATE-02_telemetry.csv", [SCORED / c for c in CERT_NAMES]
    if not tel.exists() or not all(c.exists() for c in cert):
        pytest.skip("LOUD SKIP: the bench CSVs are not on this machine")
    base = ["--telemetry", str(tel), "--baseline", *[str(c) for c in cert]]
    assert ce.main(base + ["--scalars-out", str(tmp_path / "d.csv")]) == 0
    assert next(csv.DictReader(open(tmp_path / "d.csv", newline="")))["reached_T"] == ""
    assert ce.main(base + ["--scalars-out", str(tmp_path / "o.csv"), "--reached-T-from-telemetry"]) == 0
    assert next(csv.DictReader(open(tmp_path / "o.csv", newline="")))["reached_T"] == "0"


# ===================================================================================
# E5 — the fragment: an EMPTY cell for an undefined value, the ledger's label, the join
# ===================================================================================

def _disposition(labels, verdict="MISS"):
    return {"valid_pairs": ["P99"],
            "pairs": {"P99": {"stratum": "S0", "A": {"label": labels[0], "verdict": verdict},
                              "B": {"label": labels[1], "verdict": verdict}}}}


def _write_scores(path, labels, stage="T"):
    import v4_bench_analysis as ba
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ba.SCORE_COLUMNS)
        w.writeheader()
        for lab in labels:
            w.writerow({c: "" for c in ba.SCORE_COLUMNS} | {"label": lab, "stage": stage, "mechanism": "JAM"})
    return path


def test_trial_scalars_are_the_three_columns_the_bench_sheet_promises():
    import v4_bench_analysis as ba
    s, a = _press_then_reverse(120, onset=30, reverse_at=36)
    sc = ce.trial_scalars(s, a, _baseline(), t_mono=_skewed_clock(120, 19.4), label="X")
    assert set(ce.SCALAR_COLUMNS) <= set(ba.SCORE_COLUMNS)
    for k in ce.SCALAR_COLUMNS:
        assert k in sc
    assert sc["contact_latency_s"] == pytest.approx(6 / 19.4, rel=1e-6)
    assert sc["contact_impulse"] is not None
    assert sc["baseline"]["source"] == "unit"


def test_a_trial_with_no_usable_latency_reports_none_with_a_reason_never_a_zero():
    s, a = _rack_canvas(120)
    s[30:40, 9] = 5.0                                 # a load event, but the command never reverses
    for rule in ce.RULES:
        sc = ce.trial_scalars(s, a, _baseline(), label="X", rule=rule)
        assert sc["contact_latency_s"] is None
        assert sc["latency_reasons"] and all(isinstance(r, str) for r in sc["latency_reasons"])
        assert sc["n_events"] == 1 and sc["n_latencies"] == 0


def test_an_undefined_latency_is_an_empty_cell_and_the_analysis_reads_it_as_missing(tmp_path):
    """MUTATION TARGET (W01): write 0 for None. The analysis's `num()` turns any non-empty cell into
    a float, and a 0 would enter the paired Wilcoxon as 'responded instantly'."""
    import v4_bench_analysis as ba
    s, a = _rack_canvas(120)
    s[30:40, 9] = 5.0
    frag = tmp_path / "x_scalars.csv"
    ce.write_scalar_rows(frag, [ce.trial_scalars(s, a, _baseline(), label="zz9test0")])
    row = next(csv.DictReader(open(frag, newline="")))
    assert row["contact_latency_s"] == "" and row["contact_impulse"] != "" and row["reached_T"] == ""
    scores = _write_scores(tmp_path / "scores.csv", ["zz9test0", "zz9test1"])
    merged = tmp_path / "merged.csv"
    assert ce.main(["--merge-scalars", str(frag), "--into", str(scores), "--merged-out", str(merged)]) == 0
    rows = {r["label"]: r for r in ba._rows_from_files(_disposition(["zz9test0", "zz9test1"]), merged, False)}
    assert np.isnan(rows["zz9test0"]["contact_latency_s"])
    assert not np.isnan(rows["zz9test0"]["contact_impulse"])


def test_split_label_stamp_matches_the_runners_naming():
    """tools/run_scored_trial.sh:69-70 `RUN=.../${LABEL}_${STAMP}`, STAMP = %Y%m%d_%H%M%S, and :145
    `${RUN}_telemetry.csv`. The ledger keys on the bare label with the stamp in its own column."""
    assert ce.split_label_stamp("T0-SMOKE_20260906_145245_telemetry.csv") == ("T0-SMOKE", "20260906_145245")
    assert ce.split_label_stamp("/x/y/B2-A-01_20260829_230615_telemetry.csv") == ("B2-A-01", "20260829_230615")
    assert ce.split_label_stamp("CERT-B1-01_telemetry.csv") == ("CERT-B1-01", None)
    assert ce.split_label_stamp("trial_A_v2_20260825_205412_20260825_205419") == (
        "trial_A_v2_20260825_205412", "20260825_205419")


def test_the_scalar_fragment_joins_the_ledger_label_and_reaches_the_analysis(tmp_path):
    """The Sep 7 default label was `<label>_<stamp>`, which joins nothing (v4_bench_analysis keys
    scores by the bare label). End to end on a synthetic trial: CLI -> fragment -> --merge-scalars
    -> v4_bench_analysis._rows_from_files."""
    import v4_bench_analysis as ba
    s, a = _press_then_reverse(120, onset=30, reverse_at=36)
    tel = _write_sidecar_csv(tmp_path / "zz9test0_20260916_101500_telemetry.csv", s)
    base = _write_sidecar_csv(tmp_path / "base_telemetry.csv", _quiet_block())
    pq = _write_actions_parquet(tmp_path / "rollout" / "file-000.parquet", a)
    frag = tmp_path / "zz9test0_scalars.csv"
    assert ce.main(["--telemetry", str(tel), "--baseline", str(base), "--actions-parquet", str(pq),
                    "--scalars-out", str(frag)]) == 0
    row = next(csv.DictReader(open(frag, newline="")))
    assert list(row) == ["label"] + list(ce.SCALAR_COLUMNS) and set(row) <= set(ba.SCORE_COLUMNS)
    assert row["label"] == "zz9test0"
    scores = _write_scores(tmp_path / "scores.csv", ["zz9test0", "zz9test1"])
    merged = tmp_path / "merged.csv"
    assert ce.main(["--merge-scalars", str(frag), "--into", str(scores), "--merged-out", str(merged)]) == 0
    rows = {r["label"]: r for r in ba._rows_from_files(_disposition(["zz9test0", "zz9test1"]), merged, False)}
    assert rows["zz9test0"]["contact_latency_s"] == pytest.approx(0.30, abs=1e-3)
    assert np.isnan(rows["zz9test1"]["contact_latency_s"])


def test_the_merge_refuses_rather_than_guessing():
    score_rows = [{"label": "a", "contact_latency_s": "", "contact_impulse": "", "reached_T": ""},
                  {"label": "b", "contact_latency_s": "0.5", "contact_impulse": "", "reached_T": ""}]

    def frag(lab, lat):
        return {"label": lab, "contact_latency_s": lat, "contact_impulse": "3", "reached_T": ""}

    merged = ce.merge_scalar_rows(score_rows, [frag("a", "0.25")])
    assert merged[0]["contact_latency_s"] == "0.25" and merged[1] == score_rows[1]
    with pytest.raises(ce.TelemetryError):
        ce.merge_scalar_rows(score_rows, [frag("zzz", "0.1")])                 # not in scores.csv
    with pytest.raises(ce.TelemetryError):
        ce.merge_scalar_rows(score_rows, [frag("a", "0.1"), frag("a", "0.2")])  # two fragments, one label
    with pytest.raises(ce.TelemetryError):
        ce.merge_scalar_rows(score_rows, [frag("b", "0.9")])                   # a different hand-entered value


# ===================================================================================
# THE HOLD-OUT — filtered inside the reader, on both desk paths
# ===================================================================================

def _fake_v4_dataset(root, episodes=(0, 2, 11), n=30):
    import pandas as pd
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"features": {"observation.state": {"names": ce.state_columns()}}}))
    rng = np.random.default_rng(5)
    rows = [{"episode_index": e, "frame_index": i, "observation.state": rng.normal(size=18).astype(np.float32),
             "action": rng.normal(size=6).astype(np.float32)} for e in episodes for i in range(n)]
    pd.DataFrame(rows).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    return root


def test_the_holdout_is_never_materialised_on_either_desk_path(tmp_path, monkeypatch):
    """A SYNTHETIC dataset whose third episode carries a hold-out index (11). Every frame
    `pd.read_parquet` hands back is recorded: on --anchor-evaluability and --manipulation-check the
    index never comes back, because the filter runs inside the reader (MUTATION TARGET: drop it and
    pop afterwards, as the Sep 7 code did)."""
    import pandas as pd
    root = _fake_v4_dataset(tmp_path / "fake_v4")
    real_read, real_load = pd.read_parquet, ce.load_v4_dataset
    returned = []

    def recording_read(*args, **kwargs):
        df = real_read(*args, **kwargs)
        returned.extend(int(x) for x in df["episode_index"].unique())
        return df

    monkeypatch.setattr(pd, "read_parquet", recording_read)
    monkeypatch.setattr(ce, "load_v4_dataset", lambda *_a, **kw: real_load(root, **kw))
    seen = set()

    class _Stub:
        @staticmethod
        def anchor_evaluability(states, actions, types, dims=None):
            seen.update(states)
            return {"pooled_sigma": [], "baselines": {}}

    monkeypatch.setattr(ce, "_panel_module", lambda: _Stub)
    assert ce._print_anchor_evaluability(None) == 0
    assert sorted(set(returned)) == [0, 2] and seen == {0, 2}
    returned.clear()
    ce.main(["--manipulation-check"])            # stops on the missing training episodes, after the read
    assert sorted(set(returned)) == [0, 2]
    states, _actions, _names = real_load(root)
    assert sorted(states) == [0, 2]
    with pytest.raises(ValueError):
        real_load(root, exclude=[])              # the spent split cannot be opted back in


# ===================================================================================
# THE BASELINE CARRIES ITS LOOP RATE
# ===================================================================================

def test_the_baseline_carries_its_loop_rate_and_refuses_to_pool_different_rates(tmp_path):
    """Block B1's certification replays ran at 18.52-18.63 Hz; replays after the Sep 7 pacing fix
    run at ~20 Hz. Pooling the two into one mu/sigma is refused (gate, don't narrate)."""
    slow = _write_sidecar_csv(tmp_path / "slow_telemetry.csv", _quiet_block(seed=1), hz=18.5)
    fast = _write_sidecar_csv(tmp_path / "fast_telemetry.csv", _quiet_block(seed=2), hz=20.0)
    fast2 = _write_sidecar_csv(tmp_path / "fast2_telemetry.csv", _quiet_block(seed=4), hz=19.9)
    b_slow = ce.Baseline.from_csvs([slow])
    assert b_slow.loop_hz == pytest.approx(18.5, rel=2e-3)
    assert b_slow.summary()["loop_hz"] == b_slow.loop_hz
    assert ce.Baseline.from_csvs([fast, fast2]).loop_hz == pytest.approx(19.95, rel=3e-3)
    with pytest.raises(ce.TelemetryError):
        ce.Baseline.from_csvs([slow, fast])
    s, a = _press_then_reverse(80, onset=30, reverse_at=36)
    sc = ce.trial_scalars(s, a, b_slow)
    assert sc["baseline"]["loop_hz"] == b_slow.loop_hz
    assert ce.analyze(s, a, b_slow)["baseline"]["loop_hz"] == b_slow.loop_hz


def test_the_real_block_b1_baseline_reports_its_loop_rate():
    cert = [SCORED / c for c in CERT_NAMES]
    if not all(c.exists() for c in cert):
        pytest.skip("LOUD SKIP: the certification replays are not on this machine")
    assert 18.5 < ce.Baseline.from_csvs(cert).loop_hz < 18.7


# ===================================================================================
# THE DEFAULT PATH, AS BEHAVIOUR
# ===================================================================================

def test_the_default_cli_never_reaches_the_proposal_and_scores_against_the_given_baseline(tmp_path, monkeypatch):
    """Behaviour, not stdout: the proposal module is made to explode if touched, and the JSON says
    which rule and which baseline produced the numbers."""
    def _explode():
        raise AssertionError("the default path reached the unratified proposal module")

    monkeypatch.setattr(ce, "_panel_module", _explode)
    s, a = _press_then_reverse(120, onset=30, reverse_at=36)
    tel = _write_sidecar_csv(tmp_path / "zz9test0_20260916_101500_telemetry.csv", s)
    base = _write_sidecar_csv(tmp_path / "base_telemetry.csv", _quiet_block())
    pq = _write_actions_parquet(tmp_path / "rollout" / "file-000.parquet", a)
    js = tmp_path / "out.json"
    assert ce.main(["--telemetry", str(tel), "--baseline", str(base), "--actions-parquet", str(pq),
                    "--scalars-out", str(tmp_path / "f.csv"), "--json", str(js)]) == 0
    out = json.load(open(js))
    assert out["rule_version"] == ce.RULE_PREREGISTERED == out["scalars"]["rule_version"]
    assert out["baseline"]["source"] == "csv: base_telemetry.csv"
    assert out["scalars"]["reached_T_source"] == ce.REACHED_T_ADJUDICATED


def test_the_sidecar_header_still_names_the_clock_column_this_module_reads():
    assert ts.HEADER[1] == "t_mono"
    assert ab.CONTACT_Z == 2.0 and list(ab.DISTAL_LOAD) == [8, 9, 10, 11]   # METHOD: unchanged here
