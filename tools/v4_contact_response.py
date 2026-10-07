#!/usr/bin/env python
"""The DEMOTION panel for co-primary endpoint 2: the measurement that shows it is not evaluable,
and the DESCRIPTIVE SECONDARIES reported in its place.  ** A PROPOSAL. NOT RATIFIED. **

    python tools/v4_contact_events.py --telemetry <run>_telemetry.csv --baseline <cert csvs> \
        --secondary-panel                      # opt-in; the default path never reaches this file
    python tools/v4_contact_events.py --anchor-evaluability     # desk, training episodes only

WHAT CHANGED IN THIS FILE, AND WHY (Sep 7 2026). It was first written to PROMOTE a kinematic stall
to co-primary 2's anchor. That promotion is withdrawn: the stall does not survive its own
duration-matched control (RESIST rack windows run 2.50-3.70 s and CLEAN never exceeds 1.60 s;
truncate both to a common length and the 4/4-vs-1/30 headline collapses to 1/4-vs-1/30 —
`tests/test_v4_contact_response.py::test_the_stall_does_NOT_survive_its_own_duration_matched_control`
pins it). Substituting a replacement endpoint that works for one that did not is the exact failure
a pre-registration exists to prevent, so nothing here is promoted to anything. The plumbing stays;
the claim is gone.

WHAT IS PROPOSED INSTEAD. That co-primary 2, as pre-registered at
`the v4 protocol (text held until unblinding; SHA-256 in the README):56-59` ("per-trial median latency from distal-load ONSET to
first opposing command; impulse"), is reported NOT EVALUABLE on this instrument, with the
measurement that shows why (`anchor_evaluability`, training episodes only), and that the panel
below is reported as SECONDARIES, none of them promoted:
  S1 RETRIES, given a telemetry definition. §1:63 already pre-registers "retries" as a secondary
     and `tools/v4_bench_analysis.py:564-566`, `:669` already read it from scores.csv as a
     hand-scored column, so this needs no new endpoint — only a declared operational definition,
     dated and frozen BEFORE any more trials are scored. Printed beside it every time: that it
     reads no load or current channel, its duration dependence, and that it is 0.00 on both
     trials that exist.
  S2 STALL RATE, exploratory, with its conditional latency BENEATH it and flagged as conditioning
     on a post-randomisation variable.
  S3 FIXED-WINDOW IMPULSE, descriptive, labelled null.
  S4 `reached_T` from telemetry, printed beside the adjudicated stage letter; the video wins.

Demoting a co-primary after seeing the data is a change to the pre-registered METHOD and is
Faith's to ratify with a dated §7 amendment (§ "Rules of the freeze": §1 is never edited). Until
then this file is opt-in, computes numbers BESIDE the endpoint, and is deliberately NOT imported
by `tools/v4_bench_analysis.py` (`tests/test_v4_secondary_panel.py` pins that).

THE FIVE ENGINEERING REPAIRS THIS FILE PIONEERED HAVE LANDED IN THE ENDPOINT ITSELF
(`tools/v4_contact_events.py`, tranche 1: the clock, the bounded search, the lead, the censored
channel, the three scalars). They were defects, not method choices, so they did not wait for a
ratification. This module keeps its own copies of the helpers it was written around; the seconds-
based stall rule is verified equal to `v4_contact_events.stall_reference` episode for episode on
the 20 Hz dataset.

DEFINITIONS, precise enough to reimplement.
  WINDOW      the rack phase, `v4_contact_events.rack_window`: rack entry (v4_audit.rack_slice)
              to the release-command rising edge. Unchanged.
  STALL       the first sample of a maximal run, lasting >= STALL_MIN_S, in which the command
              leads the observed position by >= STALL_GAP_DEG (L2 over lift / elbow / wrist_flex)
              while the arm moves slower than STALL_SPEED_DEG_PER_S. Runs closer together than
              REARM_S are one stall. Thresholds are in DEGREES and DEGREES/SECOND, not per-tick,
              so they mean the same thing at 19.35 Hz as at 20.00 Hz.
  LATENCY     for each stall, the first sample STRICTLY after it, within RESP_MAX_S and before the
              window end, at which (A[t] - P[t]).u_hat <= -DEADBAND_DEG over the five arm joints.
              Undefined (with a reason) if the arm was not moving into the contact, or no such
              sample exists. A trial with no stall contributes NO latency, never a zero.
  IMPULSE     sum over the RETAINED distal load channels of |x - mu_pre| dt over a FIXED
              IMPULSE_WIN_S seconds from the anchor. DURATION-FREE BY CONSTRUCTION.
  reached_T   TELEMETRY-ONLY jaw-width proxy; `v4_contact_events.reached_T` is the one definition
              and this module defers to it.

reached_T's HONEST LIMITS. Telemetry cannot see the tube. Calibrated on 57 held carries (50
training demonstrations, min 12.13; the 5 replay CSVs, min 11.42; the two policy trials, min 8.85)
and exactly ONE measured drop (CHAMF-GATE-02, 1.07). A threshold with a single negative example is
a threshold with a single negative example; the video adjudication stays the arbiter.

Written Sep 7 2026 while Faith was in the air. Tests: tests/test_v4_contact_response.py (the
plumbing and its confound) and tests/test_v4_secondary_panel.py (the demotion panel).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4_audit  # noqa: E402
import v4_contact_events as ce  # noqa: E402

FPS_NOMINAL = 20.0

# --- channels ---------------------------------------------------------------------------
DISTAL_RACK = [8, 9, 10]        # elbow_flex.load, wrist_flex.load, wrist_roll.load
GRIPPER_LOAD = 11               # dropped at the rack: constant there, and the only censored one
STALL_DIMS = [1, 2, 3]          # shoulder_lift, elbow_flex, wrist_flex
ARM_DIMS = [0, 1, 2, 3, 4]      # the gripper is not a back-off

# --- the trigger ------------------------------------------------------------------------
STALL_GAP_DEG = 3.0
STALL_SPEED_DEG_PER_S = 10.0    # == 0.5 deg/tick at exactly 20 Hz
STALL_MIN_S = 0.10              # == a 3-sample run at exactly 20 Hz
REARM_S = 0.25

# --- the response -----------------------------------------------------------------------
LOOKBACK_S = 0.50
MOTION_MIN_DEG = 1.0
DEADBAND_DEG = 1.0
RESP_MAX_S = 3.0

# --- the magnitude ----------------------------------------------------------------------
IMPULSE_WIN_S = 0.70            # the largest round value inside the shortest TRAINING rack
                                # window (0.75 s), so no trial reaching T is truncated
IMPULSE_PRE_S = 0.50

# --- reached_T --------------------------------------------------------------------------
JAW_HELD_MIN_DEG = ce.JAW_HELD_MIN_DEG   # one constant, defined with its calibration in the
                                         # endpoint module (engineering repair E5)

# --- the demotion measurement (TRANCHE 2, opt-in) ---------------------------------------
PRE_RACK_BASELINE_S = 1.0   # "the trial's own pre-rack baseline": the second of trace immediately
                            # before rack entry, i.e. free transit, the last moment the arm is
                            # demonstrably not touching anything. 1.0 s = 20 samples at 20 Hz,
                            # the same MIN_BASELINE_ROWS the block baseline requires.
IN_RACK_BASELINE_S = 0.30   # ... and the third alternative: the first 0.30 s INSIDE the window,
                            # i.e. the approach to the rim. 6 samples at 20 Hz — short enough to
                            # end before the shortest CLEAN rack window (0.75 s) is half over.
CENSOR_MIN_REPEATS = 10


class ContactResponseError(RuntimeError):
    """The telemetry is not something a contact response can be computed from."""


# --- the clock ---------------------------------------------------------------------------

def sample_times(n: int, t_mono=None, fps: float = FPS_NOMINAL) -> tuple[np.ndarray, str]:
    """Seconds from the first sample, and where they came from. The sidecar's `t_mono` when it
    wrote one -- the loop does not run at exactly 20 Hz and every time here is a real duration."""
    if t_mono is not None:
        t = np.asarray(t_mono, dtype=np.float64)
        if t.ndim != 1 or len(t) != n:
            raise ContactResponseError(f"t_mono has {t.shape}, expected ({n},)")
        if np.any(np.diff(t) <= 0):
            raise ContactResponseError("t_mono is not strictly increasing — not a monotonic clock")
        return t - t[0], "t_mono"
    return np.arange(n, dtype=np.float64) / fps, f"nominal {fps:g} fps (no t_mono column)"


# --- censoring ---------------------------------------------------------------------------

def censored_channels(state, dims) -> list[tuple[int, float, int]]:
    """(dim, extreme, count) for each channel pinned at its own extreme value at least
    CENSOR_MIN_REPEATS times. A saturated servo reports its cap, not a measurement."""
    s = np.asarray(state, dtype=np.float64)
    out = []
    for d in dims:
        col = np.abs(s[:, d])
        m = float(col.max())
        if m == 0.0:
            continue
        hits = int((col == m).sum())
        if hits >= CENSOR_MIN_REPEATS:
            out.append((int(d), float(s[np.argmax(col), d]), hits))
    return out


# --- the trigger --------------------------------------------------------------------------

def stall_flag(state, action, t) -> np.ndarray:
    """Per-sample: the command leads the position by >= STALL_GAP_DEG while the arm crawls."""
    pos = np.asarray(state, dtype=np.float64)[:, STALL_DIMS]
    act = np.asarray(action, dtype=np.float64)[:, STALL_DIMS]
    gap = np.linalg.norm(act - pos, axis=1)
    dt = np.diff(np.asarray(t, dtype=np.float64))
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sp = np.where(dt > 0, step / np.where(dt > 0, dt, 1.0), np.inf)
    return (gap >= STALL_GAP_DEG) & (np.r_[np.inf, sp] < STALL_SPEED_DEG_PER_S)


def stall_onsets(state, action, window, t) -> list[int]:
    """Every stall ONSET inside `window`. A run qualifies on its own merit (>= STALL_MIN_S);
    qualifying runs closer together than REARM_S are merged so one physical stall counts once."""
    if window is None or action is None:
        return []
    a, b = window
    flag = stall_flag(state, action, t)
    b = min(b, len(flag))
    runs, i = [], a
    while i < b:
        if not flag[i]:
            i += 1
            continue
        j = i
        while j < b and flag[j]:
            j += 1
        runs.append((i, j))
        i = j
    qual = [(i, j) for i, j in runs if t[j - 1] - t[i] >= STALL_MIN_S - 1e-9]
    merged: list[list[int]] = []
    for i, j in qual:
        if merged and t[i] - t[merged[-1][1] - 1] < REARM_S:
            merged[-1][1] = j
        else:
            merged.append([i, j])
    return [i for i, _j in merged]


# --- the response ---------------------------------------------------------------------------

def first_opposing_command(state, action, onset: int, t, hard_end: int | None = None) -> dict:
    """The first sample STRICTLY after `onset`, within RESP_MAX_S and before `hard_end`, at which
    the command leads the observed position BACKWARD along the approach."""
    pos = np.asarray(state, dtype=np.float64)[:, ARM_DIMS]
    act = np.asarray(action, dtype=np.float64)[:, ARM_DIMS]
    t = np.asarray(t, dtype=np.float64)
    ref = int(np.searchsorted(t, t[onset] - LOOKBACK_S, side="left"))
    u = pos[onset] - pos[ref]
    norm = float(np.linalg.norm(u))
    out = {"frame": None, "latency_s": None, "reason": None, "approach_motion_deg": norm}
    if norm < MOTION_MIN_DEG:
        out["reason"] = (f"the arm was not moving into the contact "
                         f"({norm:.2f} deg over the {LOOKBACK_S:.2f} s before the stall)")
        return out
    uhat = u / norm
    lo = onset + 1                                    # strictly after: at the anchor it is a LEAD
    hi = int(np.searchsorted(t, t[onset] + RESP_MAX_S, side="right"))
    hi = min(hi, len(pos), len(pos) if hard_end is None else int(hard_end))
    if lo >= hi:
        out["reason"] = "no sample after the stall inside the horizon"
        return out
    proj = ((act[lo:hi] - pos[lo:hi]) * uhat).sum(axis=1)
    hit = np.where(proj <= -DEADBAND_DEG)[0]
    if not len(hit):
        out["reason"] = (f"no opposing command within {RESP_MAX_S:.1f} s of the stall "
                         f"and before the window end")
        return out
    k = lo + int(hit[0])
    out.update(frame=k, latency_s=float(t[k] - t[onset]))
    return out


# --- the magnitude ---------------------------------------------------------------------------

def contact_impulse(state, anchor: int, t, hard_end: int | None = None,
                    win_s: float = IMPULSE_WIN_S, pre_s: float = IMPULSE_PRE_S,
                    dims=None) -> tuple[float | None, str | None]:
    """Fixed-window load impulse against the trial's OWN pre-anchor level. Duration-free."""
    dims = list(DISTAL_RACK) if dims is None else list(dims)
    s = np.asarray(state, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    p0 = int(np.searchsorted(t, t[anchor] - pre_s, side="left"))
    if anchor - p0 < 2:
        return None, f"no {pre_s:.2f} s of trace before the anchor to take a baseline from"
    mu = s[p0:anchor, dims].mean(axis=0)
    end = int(np.searchsorted(t, t[anchor] + win_s, side="right"))
    limit = len(s) if hard_end is None else min(len(s), int(hard_end))
    if end > limit:
        return None, f"the fixed {win_s:.2f} s window does not fit before the window end"
    dt = np.r_[np.diff(t), float(np.median(np.diff(t)))]
    excess = np.abs(s[anchor:end, dims] - mu)
    return float((excess.sum(axis=1) * dt[anchor:end]).sum()), None


# --- reached_T ---------------------------------------------------------------------------------

def reached_T(state, jaw_held_min: float = JAW_HELD_MIN_DEG) -> tuple[bool, str | None]:
    """Stage T from telemetry alone. ONE definition, and it lives in the endpoint module: this
    landed there as engineering repair E5 (the bench sheet promises the tool computes it), so a
    second copy here would be two definitions of a scored column."""
    return ce.reached_T(state, jaw_held_min=jaw_held_min)


# --- one trial ---------------------------------------------------------------------------------

def trial_contact_response(state, action=None, t_mono=None) -> dict:
    """The three scalars the bench sheet promises, plus the denominators that make them readable."""
    s = np.asarray(state, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] != 18:
        raise ContactResponseError(f"expected an (n, 18) state, got {s.shape}")
    if action is not None:
        a = np.asarray(action, dtype=np.float64)
        if a.ndim != 2 or len(a) != len(s):
            raise ContactResponseError(
                f"action stream {a.shape} does not align 1:1 with the {len(s)} telemetry rows")
    ce.check_channels_live(s, "trial")
    cens = censored_channels(s, DISTAL_RACK)
    if cens:
        names = ce.state_columns()
        raise ContactResponseError(
            "censored retained distal channel(s) "
            + ", ".join(f"{names[d]} pinned at {v:g} in {n} frames" for d, v, n in cens)
            + " — a saturated servo reports its cap, not a contact magnitude")
    t, clock = sample_times(len(s), t_mono=t_mono)
    ok_T, why_T = reached_T(s)
    out = {
        "clock": clock, "n_frames": int(len(s)), "duration_s": float(t[-1]) if len(t) > 1 else 0.0,
        "loop_hz": float((len(t) - 1) / t[-1]) if len(t) > 1 and t[-1] > 0 else None,
        "reached_T": bool(ok_T), "reached_T_reason": why_T,
        "rack_window_s": None, "n_stalls": 0,
        "contact_latency_s": None, "latency_reasons": [], "stall_latencies_s": [],
        "contact_impulse": None, "impulse_reason": None,
        "impulse_stall": None,
        "distal_channels": [ce.state_columns()[d] for d in DISTAL_RACK],
        "gripper_load_excluded": True,
    }
    win = ce.rack_window(s, action)
    if win is None:
        out["latency_reasons"] = ["no rack window in this trial"]
        return out
    out["rack_window_s"] = [float(t[win[0]]), float(t[min(win[1], len(t) - 1)])]
    imp, why = contact_impulse(s, win[0], t, hard_end=win[1] + 1)
    out["contact_impulse"], out["impulse_reason"] = imp, why
    if action is None:
        out["latency_reasons"] = ["no action stream: the stall needs the commanded positions"]
        return out
    onsets = stall_onsets(s, action, win, t)
    out["n_stalls"] = len(onsets)
    lats, reasons, imps = [], [], []
    for on in onsets:
        r = first_opposing_command(s, action, on, t, hard_end=win[1] + 1)
        (lats if r["latency_s"] is not None else reasons).append(
            r["latency_s"] if r["latency_s"] is not None else r["reason"])
        v, _ = contact_impulse(s, on, t, hard_end=win[1] + 1)
        if v is not None:
            imps.append(v)
    out["stall_latencies_s"] = [float(x) for x in lats]
    out["contact_latency_s"] = float(np.median(lats)) if lats else None
    out["latency_reasons"] = reasons or (["no stall in the rack window"] if not onsets else [])
    out["impulse_stall"] = float(np.median(imps)) if imps else None
    return out


# ==========================================================================================
# TRANCHE 2 (a) — THE MEASUREMENT BEHIND THE DEMOTION.  ** A PROPOSAL. NOT RATIFIED. **
# ==========================================================================================

class HoldoutRefused(RuntimeError):
    """A held-out episode reached a routine that is still being designed against the data."""


def onset_under_baseline(state, window, mu, sigma, dims=None, thr: float = None,
                         min_ticks: int = None) -> int | None:
    """The PRE-REGISTERED anchor, with the baseline left as a free parameter: the first sample of
    a run of >= `min_ticks` inside `window` where max_d |x_d - mu_d| / sigma_d exceeds `thr`.

    The rule is `v4_contact_events`'s (CONTACT_Z for >= ONSET_TICKS over the distal load channels);
    only mu/sigma vary. That is the point of the measurement: §1 says "distal-load onset" and never
    says what the baseline is — the pooled certification-replay rule is an implementation choice at
    `tools/v4_contact_events.py:396` (`Baseline.from_states`), traceable to `working notes (private):131` and not to the freeze. So the honest question is not "does the implemented
    baseline work" but "is there ANY baseline under which this anchor is a contact landmark".
    """
    import v4_input_ablation as ab
    thr = ab.CONTACT_Z if thr is None else thr
    min_ticks = ce.ONSET_TICKS if min_ticks is None else min_ticks
    dims = list(ab.DISTAL_LOAD) if dims is None else list(dims)
    s = np.asarray(state, dtype=np.float64)
    a, b = int(window[0]), min(int(window[1]), len(s))
    sd = np.asarray(sigma, dtype=np.float64)
    if np.any(sd <= 0):
        return None                     # a dead channel is not an anchor; refuse, never divide
    z = np.abs((s[a:b, dims] - np.asarray(mu, dtype=np.float64)) / sd).max(axis=1)
    over = z > thr
    run = 0
    for k in range(len(over)):
        run = run + 1 if over[k] else 0
        if run >= min_ticks:
            return a + k - min_ticks + 1
    return None


def _spearman(x, y) -> float | None:
    """Rank correlation, numpy only (no scipy dependency in the analysis path)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 3 or len(x) != len(y):
        return None
    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=np.float64)
        r[order] = np.arange(len(v), dtype=np.float64)
        # average ties, so a flat column does not fake a correlation
        for val in np.unique(v):
            m = v == val
            r[m] = r[m].mean()
        return r
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def anchor_evaluability(states: dict, actions: dict, types: dict, dims=None) -> dict:
    """Is "distal-load onset" a contact landmark under ANY baseline? Training episodes only.

    Three baselines, one rule, per episode's RACK WINDOW:
      pooled    mu/sigma over EVERY row of the whole concatenated trajectory set — the rule the
                tool implements (`Baseline.from_states`).
      pre_rack  the trial's own PRE_RACK_BASELINE_S before rack entry (phase-local, free transit).
      in_rack   the first IN_RACK_BASELINE_S inside the window (phase-local, the approach).

    For each: how often it fires, per label; how often the onset it finds sits on the window's own
    EDGE (the first tick, or the first tick after the baseline ends — i.e. the anchor is the edge,
    not an event); and the rank correlation between the onset and the DEEPEST POINT OF THE PRESS
    (argmin of the `v4_recovery_response` depth proxy in the window), which is where a contact
    force must live if the anchor is measuring one.

    THE CLAIM THIS SUPPORTS is "not evaluable as pre-registered", NOT "load carries no contact
    signal" — the second is on §2's forbidden list in spirit and is not what this measures.
    """
    import v4_recovery_response as rr
    bad = sorted(set(states) & set(ce.HOLDOUT_HANDOFF))
    if bad:
        raise HoldoutRefused(
            f"episode(s) {bad} are in the held-out split {ce.HOLDOUT_HANDOFF}; this measurement is "
            f"still being designed against the data and must not see them")
    dims = list(dims) if dims is not None else None
    import v4_input_ablation as ab
    use = list(ab.DISTAL_LOAD) if dims is None else dims

    eps = [e for e in sorted(states) if e in actions]
    pooled = np.concatenate([np.asarray(states[e], dtype=np.float64) for e in eps])
    pooled_mu, pooled_sd = pooled[:, use].mean(axis=0), pooled[:, use].std(axis=0)

    out = {"dims": use, "channels": [ce.state_columns()[d] for d in use],
           "n_episodes": len(eps), "holdout_excluded": list(ce.HOLDOUT_HANDOFF),
           "pooled_sigma": [float(x) for x in pooled_sd], "baselines": {}}
    kinds = ("pooled", "pre_rack", "in_rack")
    acc = {k: {"fired": {}, "on_edge": 0, "onsets": {}, "deepest": {}, "no_window": 0} for k in kinds}

    for e in eps:
        s = np.asarray(states[e], dtype=np.float64)
        a = np.asarray(actions[e], dtype=np.float64)
        win = ce.rack_window(s, a)
        lab = types.get(e, "?")
        for k in kinds:
            acc[k]["fired"].setdefault(lab, [0, 0])
        if win is None:
            for k in kinds:
                acc[k]["no_window"] += 1
            continue
        lo, hi = win
        n_pre = int(round(PRE_RACK_BASELINE_S * FPS_NOMINAL))
        n_in = int(round(IN_RACK_BASELINE_S * FPS_NOMINAL))
        deepest = int(np.argmin(rr.depth(s)[lo:hi])) if hi > lo else None
        specs = {
            "pooled": (pooled_mu, pooled_sd, (lo, hi), lo),
            "pre_rack": ((s[max(lo - n_pre, 0):lo, use].mean(axis=0),
                          s[max(lo - n_pre, 0):lo, use].std(axis=0), (lo, hi), lo)
                         if lo >= 2 else None),
            "in_rack": ((s[lo:lo + n_in, use].mean(axis=0), s[lo:lo + n_in, use].std(axis=0),
                         (lo + n_in, hi), lo + n_in) if hi > lo + n_in else None),
        }
        for k in kinds:
            spec = specs[k]
            acc[k]["fired"][lab][1] += 1
            if spec is None:
                continue
            mu, sd, w, edge = spec
            k0 = onset_under_baseline(s, w, mu, sd, dims=use)
            if k0 is None:
                continue
            acc[k]["fired"][lab][0] += 1
            acc[k]["onsets"][e] = int(k0 - lo)
            if deepest is not None:
                acc[k]["deepest"][e] = deepest
            if k0 == edge:
                acc[k]["on_edge"] += 1

    described = {
        "pooled": "mu/sigma over every row of the whole concatenated trajectory set "
                  "(the baseline the tool implements)",
        "pre_rack": f"the trial's own {PRE_RACK_BASELINE_S:.2f} s before rack entry",
        "in_rack": f"the first {IN_RACK_BASELINE_S:.2f} s inside the rack window",
    }
    for k in kinds:
        keys = sorted(acc[k]["onsets"])
        out["baselines"][k] = {
            "description": described[k],
            "fired": {lab: tuple(v) for lab, v in sorted(acc[k]["fired"].items())},
            "on_edge": acc[k]["on_edge"],
            "n_fired": len(keys),
            "onsets": acc[k]["onsets"],
            "spearman_onset_vs_deepest": _spearman([acc[k]["onsets"][e] for e in keys],
                                                   [acc[k]["deepest"][e] for e in keys]),
            "no_window": acc[k]["no_window"],
        }
    return out


# ==========================================================================================
# TRANCHE 2 (b) — WHAT IS REPORTED INSTEAD: the SECONDARY panel.  ** NOT RATIFIED. **
# ==========================================================================================

PANEL_STATUS = ("NOT RATIFIED — a proposal. Co-primary 2 stands as pre-registered until Faith "
                "ratifies a dated §7 amendment; nothing below is promoted to a co-primary.")


def secondary_panel(state, action=None, t_mono=None) -> dict:
    """The four SECONDARIES reported in place of the demoted co-primary, each with the caveat that
    makes it readable. Every caveat is part of the number: printed with it, every time.

    S1 is not a new endpoint. "retries" is ALREADY a pre-registered secondary
    (`the v4 protocol (text held until unblinding; SHA-256 in the README):63`) that `tools/v4_bench_analysis.py:564-566` and
    `:669` read from scores.csv as a hand-scored column; what is proposed is a declared TELEMETRY
    DEFINITION for it, which must be dated and frozen before any more trials are scored, together
    with its constants — all of which were chosen on training episodes.
    """
    import v4_recovery_response as rr
    s = np.asarray(state, dtype=np.float64)
    t, clock = sample_times(len(s), t_mono=t_mono)
    rec = rr.recovery_response(s, action)
    n_press = rec["n_press"] or 0
    amp = rec["repress_amplitude_deg"]
    any_retry = bool(amp is not None and amp > 0.0)

    win = ce.rack_window(s, action)
    stalls = stall_onsets(s, action, win, t) if (win is not None and action is not None) else []
    lat = []
    for on in stalls:
        r = first_opposing_command(s, action, on, t, hard_end=win[1] + 1)
        if r["latency_s"] is not None:
            lat.append(r["latency_s"])
    imp, imp_why = (contact_impulse(s, win[0], t, hard_end=win[1] + 1) if win is not None
                    else (None, "no rack window"))
    okT, whyT = ce.reached_T(s)

    return {
        "status": PANEL_STATUS,
        "clock": clock,
        "rack_window_s": None if win is None else [float(t[win[0]]), float(t[min(win[1], len(t) - 1)])],
        "secondaries": {
            "S1_retries": {
                "role": "secondary",
                "value": float(max(n_press - 1, 0)) if any_retry else 0.0,
                "any_retry": any_retry,
                "n_press": n_press,
                "repress_amplitude_deg": amp,
                "reads_load_channels": False,
                "caveat": (
                    "already a pre-registered secondary (§1:63); this is a DECLARED telemetry "
                    "definition of it, not a substitution. It reads NO load or current channel — "
                    "it is a position statistic, so it is not evidence about load sensing. Its "
                    "duration dependence is real and MEASURED (Sep 7, 93 rollout datasets on "
                    "disk, 46 episodes with a rack window): window length predicts a non-zero "
                    "amplitude at AUC 0.885, Spearman +0.46, and every one of the 6 firing "
                    "episodes had a window >= 2.15 s against a median of 2.05 s, while the "
                    "shortest window in the set (0.55 s) does not fire. A longer window is more "
                    "likely to score a retry whatever the arm did. The column is also nearly all "
                    "zeros: 6 of 46 there, and 0.00 on both trials that exist. Analyse as a paired "
                    "sign / mid-p McNemar on 'any retry' with the MDE printed — an ~87 %-zero "
                    "column cannot carry the pre-declared Wilcoxon and have it mean what it was "
                    "declared to mean."),
            },
            "S2_stall": {
                "role": "secondary",
                "value": float(len(stalls) > 0),
                "stall_rate_numerator": int(len(stalls) > 0),
                "n_stalls": len(stalls),
                "latency_s": float(np.median(lat)) if lat else None,
                "caveat": (
                    "EXPLORATORY. The headline is the STALL RATE (fraction of T-reaching trials "
                    "with a rack stall); the latency sits beneath it and conditions on a "
                    "post-randomisation variable (only trials that stalled have one), so it is "
                    "not a randomised comparison. On training it fires 4/4 RESIST and 1/30 CLEAN, "
                    "but truncate both classes to a common window length and that collapses — the "
                    "separation is largely how long the window was. 0/2 on the trials run so far."),
            },
            "S3_impulse": {
                "role": "secondary",
                "value": imp,
                "reason": imp_why,
                "window_s": IMPULSE_WIN_S,
                "channels": [ce.state_columns()[d] for d in DISTAL_RACK],
                "caveat": (
                    "DESCRIPTIVE AND NULL. A fixed 0.70 s integral from rack entry against the "
                    "trial's own pre-anchor mean — duration-free by construction. On the training "
                    "episodes it separates nothing: RESIST vs CLEAN AUC 0.592, two-sided "
                    "permutation p 0.59 (n = 4 vs 30, 20000 permutations, measured Sep 7). "
                    "Reported as a NULL, and gripper.load is excluded from it because that "
                    "channel is censored at its torque cap."),
            },
            "S4_reached_T": {
                "role": "secondary",
                "value": int(bool(okT)),
                "reason": whyT,
                "caveat": (
                    "a jaw-width proxy with exactly ONE measured negative example "
                    "(CHAMF-GATE-02 at 1.07 against 57 held carries >= 8.85). It is printed BESIDE "
                    "the adjudicated stage letter and the disagreements are listed; the video "
                    "adjudication wins on disagreement."),
            },
        },
    }


# --- CLI ------------------------------------------------------------------------------------------

def _training_check() -> dict:
    """The stall rate and its DURATION-MATCHED control, on the 50 v4 TRAINING episodes."""
    import v4_manifest
    ds = ce.load_v4_dataset()
    if ds is None:
        raise ContactResponseError("no v4 dataset on this machine")
    states, actions, _ = ds
    hold = ce.holdout_episodes()
    if hold != ce.HOLDOUT_HANDOFF:
        raise ce.SplitChanged(f"derived hold-out {hold} != {ce.HOLDOUT_HANDOFF}")
    for e in hold:
        states.pop(e, None)
        actions.pop(e, None)
    train = set(ce.training_episodes())
    m = v4_manifest.Manifest.load(Path(__file__).resolve().parent / "v4_manifest.json")
    rows: dict[str, dict] = {}
    cap = 1.60   # the longest CLEAN rack window
    for e in sorted(train & set(states)):
        s, a = states[e], actions[e]
        win = ce.rack_window(s, a)
        if win is None:
            continue
        t = sample_times(len(s))[0]
        cut = (win[0], min(win[1], win[0] + int(round(cap * FPS_NOMINAL))))
        r = rows.setdefault(m.type_of(e), {"n": 0, "stall": 0, "stall_matched": 0, "eps": []})
        r["n"] += 1
        if stall_onsets(s, a, win, t):
            r["stall"] += 1
            r["eps"].append(e)
        r["stall_matched"] += bool(stall_onsets(s, a, cut, t))
    return {"cap_s": cap, "by_type": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--telemetry")
    ap.add_argument("--actions-parquet")
    ap.add_argument("--episode", type=int, default=None)
    ap.add_argument("--training-check", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    try:
        print("PROPOSAL — NOT RATIFIED. The pre-registration anchors co-primary 2 on distal-load "
              "onset (§1); the proposal is to report it NOT EVALUABLE and to report the panel "
              "below as SECONDARIES. Nothing here is promoted. See the module docstring.\n")
        if a.training_check:
            res = _training_check()
            print(f"TRAINING CHECK (hold-out never loaded). Stall in the rack window, and the same "
                  f"rule after truncating every window to {res['cap_s']:.2f} s "
                  f"(the longest CLEAN one):\n")
            print(f"  {'type':12s} {'stall':>8}   {'duration-matched':>17}")
            for k, v in sorted(res["by_type"].items()):
                print(f"  {k:12s} {v['stall']:>3}/{v['n']:<4}   {v['stall_matched']:>8}/{v['n']:<8}"
                      f"  eps {v['eps']}")
            print("\n  The duration-matched column is the honest one: the headline separation is "
                  "largely window length.")
        else:
            if not a.telemetry:
                ap.error("--telemetry or --training-check")
            state, meta = ce.load_telemetry(a.telemetry)
            act = ce.load_actions_parquet(a.actions_parquet, a.episode) if a.actions_parquet else None
            res = trial_contact_response(state, act, t_mono=meta["t_mono"])
            for k, v in res.items():
                print(f"  {k:22s} {v}")
        if a.json:
            Path(a.json).parent.mkdir(parents=True, exist_ok=True)
            json.dump(res, open(a.json, "w"), indent=1, default=float)
            print(f"\nwrote {a.json}")
        return 0
    except (ContactResponseError, ce.TelemetryError, ce.SplitChanged) as e:
        print(f"GATE FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
