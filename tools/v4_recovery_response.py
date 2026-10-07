#!/usr/bin/env python
"""PROPOSAL for co-primary endpoint 2 — the RESISTANCE-RECOVERY response at the rack.

STATUS: UNRATIFIED. This module is new and stands beside the frozen path; it is imported by
nothing. `tools/v4_contact_events.py` and `tools/v4_bench_analysis.py` are untouched. Adopting
it is a change to the pre-registered METHOD (`the v4 protocol (text held until unblinding; SHA-256 in the README)` §1
Endpoints) and is Faith's to ratify, with a dated §7 amendment. Until then: evidence, not
endpoint.

WHY THE DISCRIMINANDUM CHANGES. §1 defines co-primary 2 as "per-trial median latency from
distal-load onset to first opposing command; impulse". Two measured facts break that as written:

  1. There is no load ONSET to time from. Across all 34 training rack windows there is not one
     distal-load excursion of the specified shape (|z| > 2 for >= 3 ticks) against the trial's
     own pre-rack baseline: 0/4 RESIST and 0/30 CLEAN. The pooled-baseline sweep has no
     operating point with positive Youden J (thr 2.0: TPR 1/4, FPR 18/30).
  2. "Contact" is not the discriminandum. `Start Schedule — v4.md:16` says every episode, CLEAN
     included, "continues to the seat", so EVERY rack window contains a real tube-meets-rack
     contact. Asking a detector to fire on RESIST and stay silent on CLEAN asks it not to detect
     a contact that is really there.

The same line of the schedule says what RESIST actually is: the tube is "brought down at a wrong
position or angle so the tip meets resistance ... backs off, re-aligns, inserts". That is a
PRESS -> RETREAT -> RE-PRESS, and it is what this module measures.

THE SIGNAL. DEPTH is the project's own end-effector proxy, shoulder_lift + elbow_flex +
wrist_flex (`the lab notebook (private):2403`), summed from the observed positions; smaller is
deeper. Inside the rack window the depth trace is reduced to its alternating extrema with an
amplitude hysteresis of REPRESS_HYSTERESIS_DEG. A clean insertion descends once and comes back
up: one minimum, no interior maximum, and the statistic is exactly 0.0. A resisted insertion
descends, retreats, and descends again: an interior maximum flanked by two minima, and the
statistic is that V's shallower leg, in degrees.

  repress_amplitude_deg = max over interior maxima of min(rise from the previous minimum,
                                                          fall to the next minimum)
  n_press               = number of minima in the same reduction

WHY THIS IS NOT WINDOW LENGTH IN DISGUISE (the confound that governs this dataset: the rack
window is 2.50-3.70 s for RESIST and 0.75-1.60 s for CLEAN, so every integral and every count
separates for a reason that has nothing to do with contact). The amplitude is a level in degrees
and is invariant to time-scaling, and both directions were measured on the training episodes:
  * stretch all 46 non-RESIST rack windows to the median RESIST length (70 ticks, ~3.4x): the
    statistic stays 0.000 in every one. Length alone cannot manufacture a re-press.
  * compress all four RESIST windows to the median CLEAN length (21 ticks): 7.91/14.51/6.15/6.59
    -> 7.76/14.41/6.02/6.53. The statistic does not need length.
Truncating both classes to a common 15-40 ticks instead sets every RESIST to 0.0 — that control
deletes the event rather than the confound, and is reported as such.

THE WINDOW IS LOAD-BEARING, and it is NOT re-specified here. `v4_contact_events.rack_window` is
reused verbatim — rack entry to the release edge — so this proposal changes what is measured in
the window, not where the window is. It has to be that window: cut at the release, the statistic
is 0.0 in all 46 non-RESIST training episodes; run it to the end of `v4_audit.rack_slice` instead
and 45 of those 46 fire, because after the release the arm withdraws and dips again on the way
out. The one addition is a HORIZON_TICKS cap, so a trial that reaches T and never releases — a
JAM, the case a contact endpoint most wants to speak about — is scored instead of being
undefined. The cap binds on no training episode (longest window 74 ticks).

THE LATENCY, and its limits. Resistance onset is KINEMATIC, not load-based: `stall_reference`
(the command leads the observed position by >= 3 deg on lift/elbow/wrist_flex while the arm moves
< 0.5 deg/tick, for >= 3 ticks), which fires 4/4 RESIST, 1/30 CLEAN and 1/16 on the untouched
ROT/LEFT-MIMIC types. The response is the first tick STRICTLY AFTER the onset at which the
COMMANDED depth has risen REPRESS_HYSTERESIS_DEG above its running minimum, searched only inside
the window. A reversal already >= that far back up at the onset tick is a LEAD and is reported as
a reason, never as 0.00 s; a reversal after the window end is not this event's response. Measured
latencies on the four training RESIST episodes: 0.75 / 0.15 / 0.20 / 0.25 s (median 0.23 s);
the one CLEAN and the one ROT-20 that stall give 0.05 s. n = 4 and the tick is 50 ms: this
component is quantised and weak, and it is reported as a secondary of the endpoint, not as its
headline.

WHAT IT DOES ON A TRIAL THAT SIMPLY SEATS — which is most trials. It returns 0.0, not "undefined"
and not "no event". That is the point: the scalar is defined for every trial that reaches the
rack, so co-primary 2 needs no conditioning on a post-randomisation variable. On the 45 rollout
datasets on disk with a defined rack window the statistic is > 0 in 8 (n_press 2 in six, 4 in one,
11 in one); on the 48 without one the arm never reached the rack at all.

HOLD-OUT. Episodes 11, 19, 25, 31, 35, 50, 51, 57 are the dataset's held-out split
(`v4_contact_events.HOLDOUT_HANDOFF`). Every threshold here was set on the training episodes;
`check_not_holdout` exists so that a run over the dataset has to say out loud that it is not
touching them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4_audit  # noqa: E402
import v4_contact_events as ce  # noqa: E402

FPS = 20
DEPTH_DIMS = [1, 2, 3]            # shoulder_lift + elbow_flex + wrist_flex = the end-effector proxy
REPRESS_HYSTERESIS_DEG = 4.0      # a leg smaller than this is not a retreat; flat over 2.0-6.0
HORIZON_TICKS = 80                # 4.0 s: longer than every training rack window (max 74)
HOLDOUT = tuple(ce.HOLDOUT_HANDOFF)


class HoldoutEpisode(RuntimeError):
    """Raised when a held-out episode index reaches a tuning or scoring path."""


def check_not_holdout(episode: int) -> None:
    if int(episode) in HOLDOUT:
        raise HoldoutEpisode(
            f"episode {episode} is in the held-out split {list(HOLDOUT)}; it is the honest test "
            f"of this detector and must not be scored while the detector is being designed"
        )


def depth(x) -> np.ndarray:
    """The end-effector proxy of an (n, >=4) position or action array. Smaller = deeper."""
    return np.asarray(x, dtype=np.float64)[:, DEPTH_DIMS].sum(axis=1)


def turning_points(h, hysteresis: float = REPRESS_HYSTERESIS_DEG) -> list[tuple[int, str]]:
    """Alternating ('min'/'max') extrema of `h`, each leg at least `hysteresis`."""
    h = np.asarray(h, dtype=np.float64)
    if len(h) < 2:
        return []
    out: list[tuple[int, str]] = []
    i_min = i_max = 0
    mode = None
    for t in range(1, len(h)):
        if mode is None:
            if h[t] <= h[i_min]:
                i_min = t
            if h[t] >= h[i_max]:
                i_max = t
            if h[i_max] - h[t] >= hysteresis:
                out.append((i_max, "max"))
                mode, i_min = "down", t
            elif h[t] - h[i_min] >= hysteresis:
                out.append((i_min, "min"))
                mode, i_max = "up", t
        elif mode == "down":
            if h[t] <= h[i_min]:
                i_min = t
            if h[t] - h[i_min] >= hysteresis:
                out.append((i_min, "min"))
                mode, i_max = "up", t
        else:
            if h[t] >= h[i_max]:
                i_max = t
            if h[i_max] - h[t] >= hysteresis:
                out.append((i_max, "max"))
                mode, i_min = "down", t
    return out


def repress(h, hysteresis: float = REPRESS_HYSTERESIS_DEG) -> tuple[int, float]:
    """(n_press, repress_amplitude_deg). A single descent scores exactly (1, 0.0).

    An extremum sitting on tick 0 is the window EDGE, not a turning point, and is dropped: a
    window whose first move is upward (the arm arriving at the rack from below, or an entry tick
    that lands mid-recovery) would otherwise report its first peak as an interior maximum and
    score a large re-press for a single descent.
    """
    h = np.asarray(h, dtype=np.float64)
    tp = [x for x in turning_points(h, hysteresis) if x[0] != 0]
    n_press = sum(1 for _, kind in tp if kind == "min")
    amp = 0.0
    for j in range(1, len(tp) - 1):
        i, kind = tp[j]
        if kind != "max":
            continue
        amp = max(amp, min(h[i] - h[tp[j - 1][0]], h[i] - h[tp[j + 1][0]]))
    return n_press, float(amp)


def rack_window(state, action=None, horizon: int = HORIZON_TICKS):
    """(start, end, truncated_by_horizon) or None.

    The window itself is `v4_contact_events.rack_window` — rack entry (`v4_audit.rack_slice`:
    wrist_flex < 20 and pan < 40) to the RELEASE edge — reused rather than re-specified, so this
    proposal changes what is MEASURED in the window and not where the window is. That function
    cuts on the commanded jaw's rising edge when an action stream is given and on the observed
    jaw otherwise; the two give the identical re-press amplitude on all 50 v4 training episodes,
    while the looser observed cut turns 1 spurious kinematic stall into 3 across the 30 CLEAN
    episodes, so the command is preferred where it exists.

    The one thing added here is the HORIZON cap, so a trial that reaches T and never releases —
    a JAM, the case a contact endpoint most wants to speak about — is scored instead of being
    undefined. It binds on 1 of the 50 training episodes (ep 20, 80 ticks) and changes no
    amplitude there.
    """
    st = np.asarray(state, dtype=np.float64)
    win = ce.rack_window(st, None if action is None else np.asarray(action, dtype=np.float64))
    if win is None:
        rack = v4_audit.rack_slice(st[:, 0], st[:, 3])
        if len(rack) < 10:
            return None
        win = (int(rack[0]), int(rack[-1]) + 1)
    start, end = int(win[0]), int(win[1])
    truncated = False
    if end - start > horizon:
        end, truncated = start + horizon, True
    return (start, end, truncated) if end > start else None


def recovery_response(state, action=None, fps: int = FPS) -> dict:
    """The Option-B per-trial scalars for one trial. `action` may be None (kinematic resistance
    onset and the latency then have no command stream and are reported undefined)."""
    st = np.asarray(state, dtype=np.float64)
    out = {
        "reached_rack": False, "window": None, "window_s": None,
        "window_truncated_by_horizon": None,
        "n_press": None, "repress_amplitude_deg": None,
        "resistance_detected": None, "resistance_onset_s": None,
        "recovery_latency_s": None, "recovery_latency_reason": None,
        "hysteresis_deg": REPRESS_HYSTERESIS_DEG,
    }
    win = rack_window(st, action)
    if win is None:
        out["recovery_latency_reason"] = "the arm never reached the rack (no rack window)"
        return out
    a, b, truncated = win
    out.update(reached_rack=True, window=(a, b), window_s=(b - a) / fps,
               window_truncated_by_horizon=truncated)

    ho = depth(st)[a:b]
    n_press, amp = repress(ho)
    out.update(n_press=n_press, repress_amplitude_deg=amp)

    if action is None:
        out["resistance_detected"] = None
        out["recovery_latency_reason"] = "no action stream: the resistance onset is kinematic"
        return out

    act = np.asarray(action, dtype=np.float64)
    if len(act) != len(st):
        raise ce.TelemetryError(
            f"action stream has {len(act)} rows, telemetry has {len(st)}: they must align 1:1"
        )
    t0 = ce.stall_reference(st, act, (a, b))
    out["resistance_detected"] = t0 is not None
    if t0 is None:
        out["recovery_latency_reason"] = "no resistance: the arm never stalled inside the window"
        return out
    out["resistance_onset_s"] = (t0 - a) / fps

    hc = depth(act)[a:b]
    relief = hc - np.minimum.accumulate(hc)
    k0 = t0 - a
    if relief[k0] >= REPRESS_HYSTERESIS_DEG:
        out["recovery_latency_reason"] = (
            f"the command was already {relief[k0]:.1f} deg back up at the onset tick: a LEAD, "
            f"not a response"
        )
        return out
    hits = np.where(relief[k0 + 1:] >= REPRESS_HYSTERESIS_DEG)[0]
    if not len(hits):
        out["recovery_latency_reason"] = "no commanded retreat before the window end"
        return out
    out["recovery_latency_s"] = float(int(hits[0]) + 1) / fps
    return out
