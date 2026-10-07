"""The contact-event detector for the telemetry CSVs, and its manipulation check.

Sep 6 2026, eval redesign, handoff §3 item 6 / §6 co-primary endpoint 2: from a trial's
`<label>_telemetry.csv` (18 channels per recorded frame, `tools/telemetry_sidecar.py`) find
contact ONSET (distal load |z| > 2 for >= 3 ticks against a per-block baseline), PEAK, DWELL,
IMPULSE, and the LATENCY to the first OPPOSING command.

Everything here is synthesized at runtime. There are ZERO telemetry CSVs on disk tonight (the
sidecar has never run on hardware), and a committed fixture CSV could later be mistaken for a
real trial. The real-dataset test uses the v4 parquet through ONE adapter (dataset state ->
sidecar column layout) and is restricted to the TRAINING episodes -- the 8 hold-outs are asserted
against the handoff's list and never touched (ep 35 is a RESIST and a hold-out).
"""
import csv
import sys

import numpy as np
import pytest

from conftest import TOOLS, v4_training_episodes

sys.path.insert(0, str(TOOLS))
import telemetry_sidecar as ts  # noqa: E402
import v4_contact_events as ce  # noqa: E402
import v4_input_ablation as ab  # noqa: E402

FPS = 20
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


# --- helpers ------------------------------------------------------------------

def _write_csv(path, states, header=None):
    """A telemetry CSV in the sidecar's exact format, from an (n, 18) dataset-layout state."""
    header = ts.HEADER if header is None else header
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i, row in enumerate(np.asarray(states, dtype=float)):
            w.writerow([i, f"{i / FPS:.4f}"] + [float(v) for v in row])


def _quiet(n=200, seed=0):
    """Free-motion-like frames: small noise on every channel, no contact."""
    rng = np.random.default_rng(seed)
    s = rng.normal(0.0, 1.0, size=(n, 18))
    s[:, 0:6] += 10.0  # positions, arbitrary
    return s


def _baseline():
    return ce.Baseline(mean=np.zeros(18), std=np.ones(18), source="unit", n_rows=100)


def _live(s, seed=7, scale=0.01):
    """Put a live instrument under a synthetic canvas: jitter on all 12 load/current channels,
    small enough (0.01 vs the unit baseline sigma -> |z| ~ 0.03) that it cannot make an event.

    Needed because `check_channels_live` refuses a frozen load/current column everywhere a contact
    number is computed, and `np.zeros((n, 18))` is twelve frozen columns — a state the sidecar
    cannot produce. Do not "simplify" these canvases back to zeros or to a constant marker: that
    reintroduces exactly the input the gate exists to reject."""
    s = np.asarray(s, dtype=float)
    s[:, 6:18] += np.random.default_rng(seed).normal(0.0, scale, size=(len(s), 12))
    return s


def _marker(n, base):
    """A column marker that uniquely identifies a channel WITHOUT being constant — a constant
    marker is what the loader now refuses on the load/current channels."""
    return base + np.arange(n, dtype=float)


# --- the sidecar contract -----------------------------------------------------

def test_column_resolution_tracks_the_sidecar_header_at_call_time(tmp_path, monkeypatch):
    """If the sidecar renames a column, the detector must follow it. A retyped header copy would
    not (mutation: replace `ts.HEADER` in the loader with a literal list)."""
    renamed = [h.replace("elbow_flex.load", "elbow_flex.torque") for h in ts.HEADER]
    monkeypatch.setattr(ts, "HEADER", renamed)
    s = _quiet(20)
    s[:, 8] = _marker(20, 777.0)  # elbow load, dataset layout dim 8
    p = tmp_path / "t_telemetry.csv"
    _write_csv(p, s, header=renamed)
    state, meta = ce.load_telemetry(p)
    assert state.shape == (20, 18)
    assert np.all(state[:, 8] == _marker(20, 777.0))


def test_loader_resolves_columns_by_name_not_position(tmp_path):
    """A CSV whose columns are permuted must still land every channel in its dataset-layout dim."""
    s = _quiet(15)
    s[:, 8] = _marker(15, 111.0)   # elbow load
    s[:, 14] = _marker(15, 222.0)  # elbow current
    s[:, 2] = _marker(15, 333.0)   # elbow pos
    p = tmp_path / "perm_telemetry.csv"
    perm = list(ts.HEADER)
    perm[8], perm[14] = perm[14], perm[8]        # swap the elbow load/current columns
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(perm)
        for i, row in enumerate(s):
            vals = dict(zip(ts.HEADER[2:], row))
            w.writerow([i, i / FPS] + [vals[h] for h in perm[2:]])
    state, _ = ce.load_telemetry(p)
    assert np.all(state[:, 8] == _marker(15, 111.0)) and np.all(state[:, 14] == _marker(15, 222.0))
    assert np.all(state[:, 2] == _marker(15, 333.0))


def test_adapter_dataset_state_to_sidecar_layout_roundtrips(tmp_path):
    """ONE adapter: dataset observation.state (6 pos + 6 load + 6 current) -> sidecar rows -> state.
    Mutation: emit the blocks in load/pos/current order."""
    s = _quiet(12, seed=3)
    rows = ce.state_to_telemetry_rows(s)
    assert rows[0][:2] == [0, 0.0] and len(rows[0]) == len(ts.HEADER)
    assert rows[5][2 + 8] == s[5, 8]      # elbow load lands under "elbow_flex.load"
    assert rows[5][2 + 14] == s[5, 14]    # elbow current under "elbow_flex.current"
    p = tmp_path / "rt_telemetry.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(ts.HEADER); w.writerows(rows)
    back, _ = ce.load_telemetry(p)
    assert np.allclose(back, s)


def test_loader_refuses_a_csv_missing_a_channel_or_with_a_constant_load_column(tmp_path):
    """Named for the loader, and until Sep 6 its second half exercised `Baseline.from_csvs` — the
    BASELINE path — which is how "the loader refuses a constant load column" read as covered while
    a dead TRIAL went through. Both halves now go through the loader."""
    s = _quiet(30)
    p = tmp_path / "bad_telemetry.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(ts.HEADER[:-1])
        for i, row in enumerate(s):
            w.writerow([i, i / FPS] + list(row[:-1]))
    with pytest.raises(ce.TelemetryError, match="gripper.current"):
        ce.load_telemetry(p)
    flat = _quiet(30); flat[:, 9] = 5.0  # wrist_flex load never moves: the telemetry gate (handoff §4)
    q = tmp_path / "flat_telemetry.csv"
    _write_csv(q, flat)
    with pytest.raises(ce.TelemetryError, match="constant"):
        ce.load_telemetry(q)
    with pytest.raises(ce.TelemetryError, match="constant"):
        ce.Baseline.from_csvs([q])


# --- the detector ---------------------------------------------------------------

def test_onset_needs_three_consecutive_ticks_over_the_threshold():
    z = np.zeros(40)
    z[5:7] = 3.0            # 2 ticks: not an onset
    z[20:23] = 3.0          # 3 ticks: onset at 20
    ev = ce.find_events(z, thr=2.0, min_ticks=3, off_ticks=3)
    assert [(e.onset, e.end) for e in ev] == [(20, 23)]   # mutation: min_ticks - 1
    assert ce.find_events(z, thr=2.0, min_ticks=2, off_ticks=3)[0].onset == 5


def test_an_event_ends_after_off_ticks_below_threshold_and_dips_do_not_split_it():
    z = np.zeros(60)
    z[10:30] = 3.0
    z[15:16] = 0.0          # a 1-tick dip inside the event
    z[40:44] = 3.0
    ev = ce.find_events(z, thr=2.0, min_ticks=3, off_ticks=3)
    assert [(e.onset, e.end) for e in ev] == [(10, 30), (40, 44)]
    assert ev[0].dwell_ticks == 20


def test_peak_dwell_and_impulse_on_a_synthetic_contact():
    n = 100
    s = np.zeros((n, 18))
    s[40:60, 9] = 4.0       # wrist_flex load (dim 9) at 4 sigma for 20 ticks = 1.0 s
    s[50, 9] = 8.0          # the peak
    z = ab.load_z(s, np.zeros(18), np.ones(18))
    ev = ce.find_events(z, thr=2.0, min_ticks=3, off_ticks=3)
    assert len(ev) == 1
    d = ce.describe_event(ev[0], s, None, _baseline(), fps=FPS)
    assert d["joint"] == "wrist_flex" and d["onset_s"] == 2.0 and d["dwell_s"] == 1.0
    assert d["peak_z"] == pytest.approx(8.0, abs=1e-4) and d["peak_s"] == 2.5 and d["peak_load"] == 8.0
    # impulse = sum |load - mu| * dt on the onset joint: 19 * 4 + 8 = 84 units * 0.05 s
    assert d["impulse_load_s"] == pytest.approx(84 * 0.05)       # mutation: use z instead of load
    assert d["latency_s"] is None and d["latency_reason"] == "no action stream"


def _descent_then_backoff(reverse_at=None, stationary=False):
    """(pos, act): the elbow (dim 2) descends 1 deg/tick, stalls at tick 30, and the command
    keeps pressing 4 deg beyond it; from `reverse_at` the command retreats behind the position."""
    n = 80
    pos = np.zeros((n, 18)); act = np.zeros((n, 6))
    pos[:, 2] = 60.0
    for t in range(n):
        if not stationary:
            pos[t, 2] = 60.0 - min(t, 30) * 1.0
        act[t, 2] = pos[t, 2] - 4.0
        if reverse_at is not None and t >= reverse_at:
            act[t, 2] = pos[t, 2] + 3.0
    return pos, act


def test_first_opposing_command_is_a_reversal_along_the_pre_onset_motion():
    pos, act = _descent_then_backoff(reverse_at=45)
    r = ce.first_opposing_command(pos, act, onset=30, fps=FPS)
    assert r["frame"] == 45 and r["latency_s"] == pytest.approx(15 / FPS)
    assert r["reason"] is None
    # pressing on (command still ahead of the position along the approach) is NOT opposing
    pos2, act2 = _descent_then_backoff(reverse_at=None)
    assert ce.first_opposing_command(pos2, act2, onset=30, fps=FPS)["frame"] is None   # mutation: drop the sign


def test_latency_is_undefined_when_the_arm_was_not_moving_into_the_contact():
    pos, act = _descent_then_backoff(reverse_at=45, stationary=True)
    r = ce.first_opposing_command(pos, act, onset=30, fps=FPS)
    assert r["frame"] is None and r["latency_s"] is None and "stationary" in r["reason"]


def test_the_gripper_is_not_part_of_the_approach_direction():
    """Closing the jaws during a stall is a grasp, not a back-off (mutation: include dim 5)."""
    pos, act = _descent_then_backoff(reverse_at=None)
    pos[:, 5] = np.linspace(30, 0, 80); act[:, 5] = pos[:, 5] - 10   # jaws closing throughout
    act[45:, 5] = pos[45:, 5] + 20                                    # then commanded open
    assert ce.first_opposing_command(pos, act, onset=30, fps=FPS)["frame"] is None


def test_action_stream_must_align_one_to_one_with_the_telemetry_rows():
    s = np.zeros((50, 18)); act = np.zeros((49, 6))
    with pytest.raises(ce.TelemetryError, match="align"):
        ce.analyze(s, act, _baseline(), fps=FPS)


# --- the baseline ---------------------------------------------------------------

def test_baseline_from_csvs_is_the_per_channel_mean_and_std_over_all_rows(tmp_path):
    a = _quiet(50, seed=1); b = _quiet(70, seed=2)
    pa, pb = tmp_path / "rep1_telemetry.csv", tmp_path / "rep2_telemetry.csv"
    _write_csv(pa, a); _write_csv(pb, b)
    bl = ce.Baseline.from_csvs([pa, pb])
    both = np.concatenate([a, b])
    assert np.allclose(bl.mean, both.mean(0)) and np.allclose(bl.std, both.std(0))   # mutation: mean of means
    assert bl.n_rows == 120 and "rep1_telemetry.csv" in bl.source
    with pytest.raises(ce.TelemetryError, match="baseline"):
        ce.Baseline.from_csvs([])


def test_analyze_uses_load_z_from_the_ablation_module_over_the_distal_load_channels():
    """Reuse, not reimplementation: a shoulder_pan load spike (dim 6) or an elbow CURRENT spike
    (dim 14) must not be a contact; an elbow LOAD spike (dim 8) must."""
    s = _live(np.zeros((60, 18)))
    s[10:20, 6] = 9.0
    s[30:40, 14] = 9.0
    s[45:55, 8] = 9.0
    out = ce.analyze(s, None, _baseline(), fps=FPS)
    assert [e["onset_s"] for e in out["events"]] == [45 / FPS] and out["events"][0]["joint"] == "elbow_flex"
    assert out["rule"] == {"thr": ab.CONTACT_Z, "min_ticks": ce.ONSET_TICKS, "off_ticks": ce.OFF_TICKS}


# --- the manipulation check -----------------------------------------------------

HOLDOUT_HANDOFF = [11, 19, 25, 31, 35, 50, 51, 57]   # working notes (private) §1


def test_holdout_is_derived_and_equals_the_handoff_list():
    """Derived (manifest -> kept -> seeded split), then asserted against the handoff. If this fails
    something upstream changed and the check must stop, not proceed."""
    assert ce.holdout_episodes() == HOLDOUT_HANDOFF
    assert set(v4_training_episodes()).isdisjoint(HOLDOUT_HANDOFF)


def test_manipulation_check_uses_training_episodes_only_and_never_a_voided_one():
    """Fed a fake dataset containing hold-out and voided indices, the check must ignore them
    (mutation: iterate `states` instead of the training list)."""
    rng = np.random.default_rng(0)
    states = {e: rng.normal(size=(40, 18)) for e in [0, 2, 9, 35, 25, 4, 63]}
    actions = {e: rng.normal(size=(40, 6)) for e in states}
    res = ce.manipulation_check(states, actions, fps=FPS)
    assert set(res["episodes_used"]) == {0, 2}
    assert 35 not in res["episodes_used"] and 9 not in res["episodes_used"]
    assert res["holdout_excluded"] == HOLDOUT_HANDOFF


def test_manipulation_check_stops_if_the_holdout_list_changes(monkeypatch):
    monkeypatch.setattr(ce, "holdout_episodes", lambda: [11, 19, 25, 31, 36, 50, 51, 57])
    with pytest.raises(ce.SplitChanged):
        ce.manipulation_check({0: np.zeros((40, 18))}, {0: np.zeros((40, 6))}, fps=FPS)


def test_manipulation_check_reports_recall_with_a_clean_control_on_the_real_dataset():
    """Runs the SPECIFIED rule (|z| > 2, >= 3 ticks, training μ/σ) on the training RESIST/NUDGE
    episodes and reports recall in the labelled window beside the CLEAN false-positive rate in
    the same window. The numbers are reported, not gated (v4_audit.py:29-41). What IS gated:
    the dataset's state names must equal the sidecar's channel order, else every channel is
    mislabelled and the error is invisible in the numbers."""
    ds = ce.load_v4_dataset()
    if ds is None:
        pytest.skip("v4 dataset not on this machine")
    states, actions, names = ds
    assert names == list(ts.HEADER[2:])
    res = ce.manipulation_check(states, actions, fps=FPS)
    assert set(res["episodes_used"]) <= set(v4_training_episodes())
    for t in ("RESIST", "NUDGE"):
        r = res["by_type"][t]
        assert r["n"] >= 3 and 0.0 <= r["recall"] <= 1.0
        assert "CLEAN" in res["controls"][t] and res["controls"][t]["CLEAN"]["n"] >= 20
    assert res["baseline"]["source"].startswith("training episodes")
    assert res["baseline"]["n_rows"] == sum(len(states[e]) for e in res["episodes_used"])


def test_rack_window_ends_at_the_release_COMMAND_not_the_observed_jaw():
    """Measured on v4 ep 0 (Sep 6): the release command crosses 20 deg at 7.05 s, the gripper load
    jumps -120 -> 274 at 7.15 s, and the OBSERVED jaw only crosses 20 deg at 7.25 s. A window cut
    on the observed jaw keeps the release's load spike inside it, and the specified rule then
    'detects' the release in 30/30 CLEAN rack phases (joint = gripper). The window must end at the
    rising edge of the commanded jaw (mutation: cut on state[:, 5] > JAW_OPEN)."""
    n = 80
    s = np.zeros((n, 18)); a = np.zeros((n, 6))
    s[:, 0] = 15.0; s[:, 3] = -5.0                  # pan < 40, wrist_flex < 20: the rack phase throughout
    s[:, 5] = 13.3; a[:, 5] = 1.0                   # jaws held on the tube, trigger closed
    a[50:, 5] = np.minimum(1.0 + 7.0 * np.arange(n - 50), 34.0)   # release command from tick 50
    s[52:, 11] = 500.0                              # gripper load spike two ticks later
    s[53:, 5] = 30.0                                # observed jaw open three ticks later
    w = ce.rack_window(s, a)
    assert w is not None and w[0] == 0 and w[1] <= 51
    z = ab.load_z(s, np.zeros(18), np.full(18, 100.0))
    assert not [e for e in ce.find_events(z) if w[0] <= e.onset < w[1]]
    # without an action stream the observed jaw is the only signal available
    assert ce.rack_window(s, None)[1] == 53


def test_analyze_labels_each_event_with_the_windows_it_falls_in():
    n = 120
    s = _live(np.zeros((n, 18))); a = np.zeros((n, 6))
    s[:, 0] = 15.0; s[:, 3] = -5.0; s[:, 5] = 13.3; a[:, 5] = 1.0
    s[30:40, 9] = 5.0                                # wrist_flex load inside the rack phase
    out = ce.analyze(s, a, _baseline(), fps=FPS)
    assert out["windows"]["rack"] == [0.0, n / FPS] and out["windows"]["hover"] is None
    assert out["events"][0]["in_rack_window"] is True and out["events"][0]["in_hover_window"] is False


# --- gate G-T on the TRIAL, not only on the baseline ------------------------------

def _dead_instrument(n=200, seed=3, how="nan"):
    """A sidecar-shaped trial with REAL positions and a dead instrument on all 12 load/current
    channels. `how="nan"` is what the sidecar writes when the observation carries no `.load` /
    `.current` key (telemetry_sidecar.py:27, "a missing key is NaN in its column") — i.e. the
    so_follower load patch is not in place. `how="flat"` is a bus that stopped updating and left
    the last good value repeated. Neither is a measurement, and gate G-T names the second one
    ("load and current are non-constant across the episode")."""
    s = _quiet(n, seed=seed)
    s[:, 6:18] = np.nan if how == "nan" else 42.0
    return s


@pytest.mark.parametrize("how,pattern", [("nan", "NaN"), ("flat", "constant")])
def test_analyze_refuses_a_trial_whose_load_current_channels_are_dead(how, pattern):
    """The hole this closes: `load_z` on a dead trial is NaN, `z > CONTACT_Z` is False at every
    frame (every NaN comparison is False), and `analyze` returned "0 events" — which is exactly
    what a trial that never touched the rack looks like. The refusal existed for the BASELINE
    only (Baseline.from_states, v4_contact_events.py:197-209)."""
    s = _dead_instrument(how=how)
    with pytest.raises(ce.TelemetryError, match=pattern):
        ce.analyze(s, None, _baseline(), fps=FPS)


def test_the_refusal_names_the_offending_columns_not_just_the_fact():
    """One frozen channel out of twelve, and the message has to say WHICH — the operator's next
    move is to look at that servo (mutation: report a bare count)."""
    s = _quiet(200, seed=4)
    s[:, 10] = 7.0                                  # wrist_roll load frozen; the other eleven live
    with pytest.raises(ce.TelemetryError) as e:
        ce.analyze(s, None, _baseline(), fps=FPS)
    assert "wrist_roll.load" in str(e.value)
    assert "elbow_flex.load" not in str(e.value)    # the live channels are not accused


def test_load_telemetry_refuses_a_dead_trial_csv_and_names_the_file(tmp_path):
    p = tmp_path / "dead_telemetry.csv"
    _write_csv(p, _dead_instrument(how="flat"))
    with pytest.raises(ce.TelemetryError, match="constant") as e:
        ce.load_telemetry(p)
    assert "dead_telemetry.csv" in str(e.value)


def test_cli_exits_nonzero_on_the_dead_trial_that_used_to_report_zero_events(tmp_path, capsys):
    """The Sep 6 bench-morning reproduction, end to end: a 200-row sidecar-shaped CSV with real
    `.pos` and all 12 load/current columns NaN, against a VALID baseline, exited 0 and printed
    "z_max nan, 0 event(s)". That zero was eligible to enter co-primary endpoint 2."""
    trial = tmp_path / "trial_telemetry.csv"
    _write_csv(trial, _dead_instrument(how="nan"))
    base = tmp_path / "rep1_telemetry.csv"
    _write_csv(base, _quiet(200, seed=9))
    rc = ce.main(["--telemetry", str(trial), "--baseline", str(base)])
    err = capsys.readouterr().err
    assert rc == 1                                  # mutation: return 0 from the except branch
    assert "GATE FAILED" in err and "NaN" in err


def test_analyze_refuses_a_nan_action_stream():
    """Same silent shape on the other input: `(A[t] - P[t]) · û <= -deadband` is False at every
    tick when A is NaN, so every latency comes back None with the ordinary reason "no opposing
    command before the end of the trace" and the trial drops out of the median unremarked.
    Constant action COLUMNS are deliberately not gated — a joint that never moves is a real
    trajectory, and gate G-T's non-constant clause is about the instrument, not the motion."""
    s = _quiet(60, seed=5)
    a = np.zeros((60, 6)); a[:, 2] = np.nan
    with pytest.raises(ce.TelemetryError, match="NaN"):
        ce.analyze(s, a, _baseline(), fps=FPS)
    ce.analyze(s, np.zeros((60, 6)), _baseline(), fps=FPS)   # a still command is not a fault


def test_manipulation_check_refuses_an_episode_whose_load_channels_are_dead():
    """The per-episode hole: the check's baseline is the CONCATENATION of the training episodes,
    so one dead episode beside a live one leaves the concatenated std non-zero and
    `Baseline.from_states` passes. The refusal has to be per episode, where the number is made."""
    rng = np.random.default_rng(0)
    states = {e: rng.normal(size=(40, 18)) for e in (0, 2)}
    actions = {e: rng.normal(size=(40, 6)) for e in (0, 2)}
    states[0][:, 9] = 3.0                            # ep 0's wrist_flex load frozen
    ce.Baseline.from_states(np.concatenate([states[0], states[2]]), source="concat")  # still passes
    with pytest.raises(ce.TelemetryError, match="constant") as e:
        ce.manipulation_check(states, actions, fps=FPS)
    assert "wrist_flex.load" in str(e.value) and "0" in str(e.value)   # names the episode


def test_baseline_from_json_is_gated_too_the_third_input_to_the_same_number(tmp_path):
    """`--baseline-json` reaches `load_z` without passing through `Baseline.from_states`, so it is
    the one remaining way to put a dead instrument into the z-score. A NaN mu/sigma is the SILENT
    shape again (z is NaN -> no events -> exit 0); a zero sigma on a load channel is the loud one
    (z = |x| / 1e-6 -> events everywhere). Both are refused, at the only place they enter."""
    import json as _json
    good = {"mean": [0.0] * 18, "std": [1.0] * 18, "source": "unit", "n_rows": 100}
    p = tmp_path / "bl.json"; p.write_text(_json.dumps(good))
    assert ce.Baseline.from_json(p).n_rows == 100          # a real baseline still loads

    nan = dict(good, std=[1.0] * 9 + [float("nan")] + [1.0] * 8)   # dim 9 = wrist_flex.load
    q = tmp_path / "nan.json"; q.write_text(_json.dumps(nan))
    with pytest.raises(ce.TelemetryError, match="NaN") as e:
        ce.Baseline.from_json(q)
    assert "wrist_flex.load" in str(e.value) and "elbow_flex.load" not in str(e.value)

    # mu is as fatal as sigma: z = |x - mu| / sigma (mutation: check np.isnan(std) only)
    nan_mu = dict(good, mean=[0.0] * 8 + [float("nan")] + [0.0] * 9)   # dim 8 = elbow_flex.load
    qm = tmp_path / "nanmu.json"; qm.write_text(_json.dumps(nan_mu))
    with pytest.raises(ce.TelemetryError, match="NaN") as e:
        ce.Baseline.from_json(qm)
    assert "elbow_flex.load" in str(e.value)

    zero = dict(good, std=[1.0] * 8 + [0.0] + [1.0] * 9)
    r = tmp_path / "zero.json"; r.write_text(_json.dumps(zero))
    with pytest.raises(ce.TelemetryError, match="zero") as e:
        ce.Baseline.from_json(r)
    assert "elbow_flex.load" in str(e.value)

    pos_zero = dict(good, std=[0.0] + [1.0] * 17)          # a still POSITION channel is not a fault
    t = tmp_path / "pos.json"; t.write_text(_json.dumps(pos_zero))
    assert ce.Baseline.from_json(t).n_rows == 100
