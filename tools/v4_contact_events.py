#!/usr/bin/env python
"""Contact events from a trial's telemetry CSV: onset, peak, dwell, impulse, and the latency
from distal-load onset to the first OPPOSING command.

    # bench: one trial against its block's certification-replay baseline
    python tools/v4_contact_events.py --telemetry tools/scored_logs/<run>_telemetry.csv \\
        --baseline tools/scored_logs/<replay1>_telemetry.csv tools/scored_logs/<replay2>_telemetry.csv \\
        --actions-parquet <rollout dataset>/data/chunk-000/file-000.parquet --json out.json

    # bench: ... and the contact columns of scores.csv (label = the ledger's bare label)
    python tools/v4_contact_events.py --telemetry <label>_<stamp>_telemetry.csv --baseline <cert csvs> \\
        --actions-parquet <rollout>/data/chunk-000/file-000.parquet --scalars-out <label>_scalars.csv

    # after adjudication: copy the fragments' cells into scores.csv by label (a NEW file, refuses
    # a label it cannot place, a label in two fragments, or a different value already in a cell)
    python tools/v4_contact_events.py --merge-scalars <label>_scalars.csv ... \\
        --into analysis/bench_sep6/scores.csv --merged-out <scores_with_contact.csv>

    # desk: the manipulation check on the v4 TRAINING episodes (no hardware)
    python tools/v4_contact_events.py --manipulation-check [--json out.json]

    # OPT-IN, DEFAULT OFF — departures from the pre-registered method, each a ratification item:
    python tools/v4_contact_events.py ... --rule amended-2026-09-07     # the Sep 7 amendment (below)
    python tools/v4_contact_events.py ... --reached-T-from-telemetry    # proxy INTO the scored column
    python tools/v4_contact_events.py --telemetry ... --baseline ... --secondary-panel
    python tools/v4_contact_events.py --anchor-evaluability      # desk, training episodes only

WHY (Sep 6 2026, eval redesign). Co-primary endpoint 2 of the merged protocol (working notes (private) §6) is "contact response from `<trial>_telemetry.csv` (per-trial median latency from
distal-load onset to first opposing command; impulse)". That number is part of the
pre-registration, so its definition has to be unambiguous enough that two people compute the
same thing. This file is that definition, with tests (tests/test_v4_contact_events.py).

THE INPUT. `tools/telemetry_sidecar.py` writes one row per RECORDED frame with the header
`["frame_index", "t_mono"] + [f"{j}.{c}" for c in (pos, load, current) for j in JOINTS]` — all
six positions, then all six loads, then all six currents — which is exactly the 18-dim dataset
layout ([0:6] pos, [6:12] load, [12:18] current; the project notes "Money / accounts" (1)). The header
is read from `telemetry_sidecar.HEADER` AT CALL TIME, never retyped: the k-th sidecar channel is
dataset dim k, and CSV columns are located by NAME, so a permuted file still lands every channel
in its dim. `load_v4_dataset` gates that the dataset's own `observation.state` names equal that
header — getting the order wrong mislabels every channel and the error is invisible in the numbers.
When this file was first written (Sep 6 morning) there were zero telemetry CSVs on disk; since the
Sep 6 bench there are seven in tools/scored_logs/ (CERT-B1-01/02/03, CHAMF-GATE-01/02, T0-SMOKE and
the first blinded pair trial).

THE TELEMETRY GATE (G-T) IS APPLIED TO THE TRIAL, NOT ONLY TO THE BASELINE. §3's gates table of
the frozen pre-registration ("the v4 protocol (text held until unblinding; SHA-256 in the README)") specifies G-T as "<trial>_
telemetry.csv exists; row count == recorded frame count; load and current are non-constant across
the episode | Trial invalid -> whole pair invalid". Two of those three clauses are enforced at the
bench (tools/run_pair.sh:254-288 `trial_health`: VOIDS on zero rows, DEGRADES on a row/frame
mismatch); the non-constant clause is not checked there at all, and the Sep 6 16:10 amendment logs
that as a known gap covered offline "before any contact number is computed". Until that fix the
refusal guarded the BASELINE only, and a TRIAL whose 12 load/current columns were NaN or frozen ran
straight through: `v4_input_ablation.load_z` returns NaN, `z > CONTACT_Z` is False at every frame
because every NaN comparison is False, and this tool printed "0 event(s)" and exited 0 — which is
exactly what a trial where the arm never touched the rack looks like, so the zero was eligible to
enter co-primary endpoint 2. `check_channels_live` is now called on every trial, at load
(`load_telemetry`) and at the choke point every contact number flows through (`analyze`), and on
each episode of the manipulation check separately — the check's own baseline is the CONCATENATION
of the training episodes, so one dead episode beside a live one leaves the concatenated std
non-zero and `Baseline.from_states` passes it. Baseline CSVs are now gated per FILE as well, for
the same reason.

THE RULE (from the handoff, not from the data). Per frame, z = `v4_input_ablation.load_z`: the
max |(load - mu) / sigma| over the DISTAL LOAD channels only — `DISTAL_LOAD = [8, 9, 10, 11]` =
elbow_flex, wrist_flex, wrist_roll, gripper load (tools/v4_input_ablation.py:43); not the
currents, not shoulder_pan/shoulder_lift. ONSET = the first tick of a run of >= ONSET_TICKS (3)
consecutive ticks with z > CONTACT_Z (2.0). The event ENDS at the first tick of a run of
>= OFF_TICKS (3) consecutive ticks with z <= CONTACT_Z (a 1-2 tick dip does not split an event).
PEAK = the max z in the event and its tick; the event's JOINT is the distal channel that carries
that peak. DWELL = end - onset in seconds ON THE SIDECAR'S OWN CLOCK. IMPULSE = the integral over
the event of |load_joint - mu_joint| dt, in raw servo load units x seconds on the event's joint
(Feetech Present_Load, sign-decoded, ±~1000; so_follower patch) — dt from `t_mono`, not 1/fps.

mu / sigma are a `Baseline`: per channel, over every row of the block's certification-replay
telemetry CSVs (`Baseline.from_csvs`), gated non-constant on every load/current channel — the
same gate the handoff puts on the first trial. The certification replays that write these CSVs are
run by `tools/replay_telemetry.py` (block B1's three are CERT-B1-01/02/03). The check refuses to run
without a baseline rather than quietly falling back to dataset statistics. Each `Baseline` carries
the loop rate its replays ran at (`loop_hz`), because B1's ran at 18.52-18.63 Hz and replays after
the Sep 7 pacing fix run at ~20 Hz; CSVs of different rates are never pooled into one baseline.

OPPOSING COMMAND — the definition (sign-convention-free, so it does not depend on which way
Feetech signs Present_Load on a joint whose drive direction calibration may flip):
  * Arm joints only: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll (dims 0-4).
    The gripper is excluded: jaw motion during a stall is a grasp or a release, not a back-off.
  * Approach direction u = P[onset] - P[ref], where P is the OBSERVED position. If |u| <
    MOTION_MIN_DEG (1.0°) the arm was not moving into the contact and the latency is undefined
    (reported with a reason, never as 0 or as a number).
  * At a sample t the command A[t] (the dataset's `action` = the leader's position during
    recording; the policy's commanded position in a rollout) is OPPOSING when
        (A[t] - P[t]) · û  <=  -DEADBAND_DEG (1.0°)
    i.e. the command leads the observed position BACKWARD along the approach. While pressing on,
    the command leads forward (the servo lags its goal), so this is negative only on a reversal.
    The reference is P[t], not P[onset]: a back-off from a stall is a reversal relative to where
    the arm is stuck, and it can be well below where the arm was when the load rose (RESIST ep 2:
    elbow stalls at 43.6° commanded 39.4°, backs off to a command of 46.5° — below the 48.9° at
    onset, above the stalled position).
  * LATENCY = t[t_opposing] - t[onset]. Which sample is t_opposing is THE RULE, and there are two.
    The action stream must align 1:1 with the telemetry rows (`TelemetryError` otherwise).

THE RULE, VERSIONED. Every output carries `rule_version`.
  RULE_PREREGISTERED = "preregistered-2026-09-06" — THE DEFAULT. Exactly what the pre-registration's
     own commit implemented (8958af5; "the v4 protocol (text held until unblinding; SHA-256 in the README):9" names it as such):
     ref = onset - LOOKBACK_TICKS in TICKS (10); t_opposing = the first sample t >= onset that is
     opposing, searched to the END OF THE TRACE; a command already opposing at the onset therefore
     scores 0.00 s. That 0.00 is the rule's number and stays in the median, but it is never
     presented as a response: the event is labelled `lead` with `lead_ticks`, and the trial reports
     `n_leads_in_median`.
  RULE_AMENDED = "amended-2026-09-07" — OPT-IN (`--rule amended-2026-09-07`), UNRATIFIED. The Sep 7
     tranche-1 changes that alter the pre-registered number: the search is bounded to the event's
     end plus RESP_GRACE_S and to RESP_MAX_S from the onset (`response_horizon`); it starts strictly
     after the onset, so a lead is undefined rather than 0.00; and the lookback is LOOKBACK_TICKS /
     20 = 0.5 s of the real clock rather than 10 ticks. On T0-SMOKE the two rules give 0.40 s (5 of
     6 events, one a lead) and 0.25 s (2 of 6). Both constants were declared after looking at
     T0-SMOKE (RESP_GRACE_S) and at the rack windows (RESP_MAX_S; see RESP_HORIZON_PROVENANCE).

DEFECTS FIXED ON BOTH RULES (Sep 7 found them; Sep 14 separated them from the method changes above).
All measured on the seven CSVs on disk; tests/test_v4_contact_endpoint_repairs.py.
  E1 THE CLOCK. Every time was computed at a hard-coded 20 fps while the loop ran 18.523-18.626 Hz
     on the certification/gate replays and 19.396 / 19.352 Hz on the two policy trials, and
     `load_telemetry` parsed the sidecar's `t_mono` column and discarded it. Worse than the average
     rate: the FIRST interval of each policy trial is 0.520 / 0.582 s (the first inference), so a
     detection at nominal 0.75 s really happened at 1.22 s. `sample_times` reads `t_mono` and NAMES
     the fallback when there is none. (`tools/replay_telemetry.py`'s own pacing was fixed with it,
     so baselines recorded after Sep 7 run at ~20 Hz rather than B1's ~18.5 Hz — a disclosure, and
     the reason `Baseline.loop_hz` exists.)
  E3' A LEAD PRESENTED AS A RESPONSE. The 5.69 s event of T0-SMOKE (5.15 s on the nominal clock)
     printed "latency 0.00 s". Both rules now label it a lead (above); only the amendment drops it.
  E4 THE CENSORED CHANNEL. `gripper.load` is right-censored at exactly MAX_TORQUE_LIMIT (500): both
     policy trials printed z_max exactly 4.352, both from gripper.load at +500, at different frames
     (104 and 230). The cap is written to the GRIPPER ONLY (so_follower.py:180 `if motor ==
     "gripper":`, :185), so only `CENSORED_LOAD_DIMS` = gripper.load is ever counted or flagged; an
     arm joint passing through ±500 is data (T0-SMOKE shoulder_lift.load reaches -842). The channel
     is NOT dropped — narrowing `DISTAL_LOAD` is a method change and Faith's to ratify.
  E5 THE SCALARS. `the v4 bench sheet (held until unblinding):112-113` promises `contact_latency_s`,
     `contact_impulse` and `reached_T` are computed here, never hand-entered. `trial_scalars` and
     `write_scalar_rows` (`--scalars-out`) emit them keyed by the ledger's BARE label (the stamp is
     reported separately; `split_label_stamp`), an undefined value is an EMPTY cell, never 0, and
     `merge_scalar_rows` (`--merge-scalars`) is the tested route into scores.csv. `reached_T` is
     written BLANK by default, so `tools/v4_bench_analysis.py` restricts on the ADJUDICATED stage
     letter as §1 says; the telemetry proxy is reported beside it (`reached_T_telemetry`) and enters
     the scored column only with `--reached-T-from-telemetry`.
  HOLD-OUT. `load_v4_dataset` filters the spent hold-out INSIDE `pd.read_parquet`; nothing this
     module builds ever holds a hold-out row, and a caller cannot opt it back in.

THE SECONDARY PANEL (--secondary-panel) IS OFF BY DEFAULT and is not the endpoint. See
`tools/v4_contact_response.py`.

THE MANIPULATION CHECK (--manipulation-check) runs the rule, with mu/sigma from the TRAINING
episodes' own frames, on the training RESIST and NUDGE episodes of the 18-dim v4 dataset and
reports RECALL in the labelled window beside the CLEAN false-positive rate in the SAME window.
Labels: tools/v4_manifest.json "assigned"; the six "voided" episodes are excluded (ep 9's NUDGE
closed its jaws, ep 16 never reached the rack — a voided episode is a mislabelled example). The
8 hold-out episodes are derived (manifest -> kept -> seeded split, tools/v4_training_split.py) and
asserted equal to the handoff's list [11, 19, 25, 31, 35, 50, 51, 57] — `SplitChanged` stops the
check if they differ. Ep 35 is a RESIST and a hold-out: exactly the example one would most want,
and one this check never sees — the split was used once (Sep 7), is spent, and `load_v4_dataset`
filters it out inside the parquet reader.
  * RESIST (Start Schedule — v4.md:16): the ARM brings the tube down at a wrong position/angle
    against the unmodified rack, the operator feels the push-back through the leader, backs off,
    re-aligns, inserts — never a hand. Window: the rack phase (`v4_audit.rack_slice`) up to the
    release (the jaw opening at the rack loads the gripper channel and is not a contact).
  * NUDGE (Start Schedule A2): an open-jaw graze of the tube on the way in. Window: the jaws-open
    hover at the pick site, first open -> grasp (`v4_audit` approach signature).
  * There is NO hand-annotated contact time in the dataset. The timing reference for the RESIST
    window is kinematic and independent of the load channels: the STALL, the first >= 3 ticks in
    the window where the command leads the observed position by >= STALL_GAP_DEG (3.0°) on
    lift/elbow/wrist_flex while the arm moves < STALL_SPEED_DEG_PER_TICK (0.5°/tick). "Timing
    error" = onset - stall, in seconds. NUDGE has no kinematic reference (the tube moves, not the
    arm); its onset is reported as an offset into the hover window.
  * Recall and the control rate are REPORTED, never gated: a threshold without a control is a
    named bug class here (tools/v4_audit.py:29-41). The specified rule is implemented as
    specified; if the CLEAN control fires as often as RESIST in the same window, that is the
    finding, and retuning the threshold in this file is not the fix.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import telemetry_sidecar as ts  # noqa: E402
import v4_audit  # noqa: E402
import v4_input_ablation as ab  # noqa: E402
import v4_manifest  # noqa: E402
import v4_training_split as split  # noqa: E402

FPS = 20
ONSET_TICKS = 3            # handoff §3 item 6: |z| > 2 for >= 3 ticks
OFF_TICKS = 3              # symmetric hysteresis; a 1-2 tick dip does not split an event
LOOKBACK_TICKS = 10        # approach direction: 10 ticks before onset (0.5 s only at exactly 20 Hz)
MOTION_MIN_DEG = 1.0       # below this the arm was not moving into the contact -> latency undefined
DEADBAND_DEG = 1.0         # a reversal smaller than this is servo noise / lag, not a command
ARM_DIMS = [0, 1, 2, 3, 4]  # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll — not the gripper
STALL_DIMS = [1, 2, 3]     # lift / elbow / wrist_flex: where a rim landing shows as command-vs-position gap
STALL_GAP_DEG = 3.0
STALL_SPEED_DEG_PER_TICK = 0.5
RELEASE_CMD_RISE_DEG = 3.0   # the commanded jaw's rising edge at the rack = the release; v4 ep 20 holds at 17-21 and opens by ~3.6
MIN_BASELINE_ROWS = 20     # 1 s at 20 Hz; a certification replay is ~200 rows

# --- the rule, versioned (see the module docstring, "THE RULE, VERSIONED") -------------------
RULE_PREREGISTERED = "preregistered-2026-09-06"   # the default: 8958af5's search, exactly
RULE_AMENDED = "amended-2026-09-07"               # opt-in, unratified: horizon + grace + lead exclusion + clock lookback
RULES = (RULE_PREREGISTERED, RULE_AMENDED)
DEFAULT_RULE = RULE_PREREGISTERED

# --- the AMENDMENT's constants. They are read ONLY under RULE_AMENDED, at call time. ------------
# Declared on Sep 7 AFTER looking at T0-SMOKE and the rack windows; each moves every amended
# latency (sensitivity band pinned in tests/test_v4_contact_endpoint_repairs.py); both are
# ratification items and neither touches the default path.
RESP_MAX_S = 3.0      # the response horizon from the onset. NOT "longer than every contact phase":
                      # the longest TRAINING rack window is 74 ticks (ep 20, RESIST) = 3.70 s at the
                      # dataset's 20 fps, and 3 of the 4 training RESIST windows exceed 3.0 s
                      # (RESP_HORIZON_PROVENANCE, derived from the data in the tests). The unratified
                      # tools/v4_recovery_response.py:95 caps its window at HORIZON_TICKS = 80 (4.0 s)
                      # for the same reason, and tools/v4_contact_response.py:102 uses 3.0 s.
RESP_GRACE_S = 1.0    # ... and the search also stops this long after the EVENT ends. Chosen with
                      # T0-SMOKE in view: its event at tick 15 (dwell 0.55 s) was handed, by the
                      # pre-registered search, a reversal 2.55 s after it ended.
RESP_HORIZON_PROVENANCE = {           # training episodes only, hold-out filtered at read (tests)
    "longest_training_rack_window_ticks": 74, "episode": 20, "type": "RESIST",
    "resist_window_ticks": (50, 70, 71, 74),
    "resist_windows_longer_than_horizon": 3,
}

# --- defects fixed on both rules --------------------------------------------------------------
MAX_TORQUE_LIMIT = 500.0  # Feetech `Max_Torque_Limit`, written by the local so_follower patch to
                      # the GRIPPER ONLY: site-packages .../robots/so_follower/so_follower.py:180
                      # `if motor == "gripper":` then :185 `self.bus.write("Max_Torque_Limit",
                      # motor, 500)  # upstream: 500` (the project notes "Money / accounts" (2)). A saturated
                      # servo reports its cap, not the force. Measured: both policy trials print
                      # z_max exactly 4.352, both from gripper.load at +500 (T0-SMOKE frame 104, the
                      # blinded trial frame 230), and on T0-SMOKE gripper.load never exceeds 500
                      # while the uncapped arm channels do (shoulder_lift -842, wrist_flex -750,
                      # elbow_flex -716, wrist_roll -567).
CENSORED_LOAD_DIMS = (11,)  # gripper.load — the one channel the cap is written to (ast-pinned)
JAW_HELD_MIN_DEG = 5.0  # reached_T's jaw-width proxy: the midpoint of the gap between the one
                      # BENCH negative and the smallest held carry (JAW_HELD_CALIBRATION).
JAW_HELD_CALIBRATION = {
    "threshold_deg": 5.0,
    "rule": "midpoint of (max_negative_deg, the smallest held-carry minimum)",
    # "dropped it by 5 s, jaw collapsed 15.5 → 1.1" ("the v4 protocol (text held until unblinding; SHA-256 in the README)":362)
    "negatives": ("CHAMF-GATE-02",), "max_negative_deg": 1.07,
    # 56 held carries, not 57: CHAMF-GATE-02 is one of the five replay CSVs on disk and it is the
    # negative, so four replays are positives. The two policy trials are T0-SMOKE and the first
    # blinded pair trial: the threshold was set with both in view and is then applied to both.
    "held_carries": {"training demonstrations": 50, "replay CSVs": 4, "policy trials": 2},
    "n_held_carries": 56,
    "replay_positives": ("CERT-B1-01", "CERT-B1-02", "CERT-B1-03", "CHAMF-GATE-01"),
    "min_held": {"training demonstrations": 12.13, "replay CSVs": 11.42, "policy trials": 8.85},
    # WHAT CAN AND CANNOT VALIDATE IT. Every v4 demonstration continues to the seat (Start
    # Schedule — v4.md:16), so neither the 50 training nor the 8 held-out episodes contains a
    # negative by construction; on the 50 training episodes any threshold from 0.0 to 12.13 gives
    # the same verdict (measured, tests), and the Sep 7 held-out check (8/8) could not have failed. The discriminating set is CHAMF-GATE-02 plus the 12
    # rack-reaching closed-jaw rollouts named in the tests (carry minima 1.07-1.36; the other 33
    # grasped, rack-reaching rollouts on disk sit at >= 8.85). The video adjudication is the arbiter.
    "rollout_negatives": 12,
}

# The three telemetry columns `the v4 bench sheet (held until unblinding):112-113` promises this tool
# computes offline and that are "never hand-entered". They are a subset of
# `v4_bench_analysis.SCORE_COLUMNS` by construction (pinned in the tests).
SCALAR_COLUMNS = ("contact_latency_s", "contact_impulse", "reached_T")

# Pre-registered hold-out (working notes (private) §1). The check DERIVES the split and
# stops if the derivation no longer matches this list.
HOLDOUT_HANDOFF = [11, 19, 25, 31, 35, 50, 51, 57]
MANIFEST_PATH = Path(__file__).resolve().parent / "v4_manifest.json"
CHECK_TYPES = ("RESIST", "NUDGE")
CONTROL_TYPE = "CLEAN"


class TelemetryError(RuntimeError):
    """The telemetry CSV / action stream / baseline is not what the sidecar contract promises."""


class SplitChanged(RuntimeError):
    """The derived hold-out no longer equals the pre-registered list — stop, do not proceed."""


def check_rule(rule: str) -> str:
    """The rule name, or ValueError. A misspelt rule must not silently fall back to either one."""
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")
    return rule


# --- the sidecar contract -------------------------------------------------------

def state_columns() -> list[str]:
    """The 18 channel names in dataset-dim order, from the sidecar, at call time."""
    return list(ts.HEADER[2:])


def sample_times(n: int, t_mono=None, fps: int | float = FPS) -> tuple[np.ndarray, str]:
    """Seconds from the first sample, and WHERE THEY CAME FROM.

    The sidecar writes its own monotonic clock in column `t_mono` (`telemetry_sidecar.HEADER[1]`,
    :73) and the loop does not run at exactly `fps`: measured Sep 7 on all seven CSVs on disk, the
    certification replays ran 18.523 / 18.626 / 18.601 Hz, the chamferless passes 18.568 / 18.546,
    and the two policy trials 19.396 / 19.352. A hard-coded 20 fps understates every duration,
    dwell and latency by +3.1 % to +8.0 %, and understates the impulse (a load x TIME integral)
    by the same factor. `load_telemetry` already parsed this column and threw it away.

    The fallback is NAMED in the returned string and travels into the result dict, so a trial
    timed on a nominal clock can never be mistaken in the write-up for one that was measured.
    """
    if t_mono is not None:
        t = np.asarray(t_mono, dtype=np.float64)
        if t.ndim != 1 or len(t) != n:
            raise TelemetryError(f"t_mono has shape {t.shape}, expected ({n},) — one per telemetry row")
        if len(t) > 1 and np.any(np.diff(t) <= 0):
            bad = int(np.argmax(np.diff(t) <= 0))
            raise TelemetryError(f"t_mono is not strictly increasing at row {bad + 1} "
                                 f"({t[bad]:.4f} -> {t[bad + 1]:.4f}) — not a monotonic clock")
        return t - t[0], "t_mono"
    return np.arange(n, dtype=np.float64) / float(fps), f"nominal {float(fps):g} fps (no t_mono column)"


def _dt(t) -> np.ndarray:
    """Seconds each sample stands for: t[i+1] - t[i], the last one carried over. An integral over
    ticks divided by a nominal fps is the same number only when the loop hit that rate exactly."""
    t = np.asarray(t, dtype=np.float64)
    if len(t) < 2:
        return np.ones(len(t), dtype=np.float64) / FPS
    d = np.diff(t)
    return np.r_[d, float(np.median(d))]


def _edge_time(t, i: int) -> float:
    """The time an EXCLUSIVE index stands for: `t[i]` inside the trace, one sample past the last
    one at the end. `end / fps` used to say this implicitly; on a real clock it has to be said."""
    t = np.asarray(t, dtype=np.float64)
    i = int(i)
    return float(t[i]) if i < len(t) else float(t[-1] + _dt(t)[-1])


def censored_ticks(state, dims=None, cap: float = MAX_TORQUE_LIMIT) -> dict[str, int]:
    """{channel name: how many samples sit exactly at the torque cap}, on the CAPPED channels only.

    `Max_Torque_Limit` bounds the MAGNITUDE, so +cap and -cap are both the cap. A channel at its
    cap is right-censored: the true load is at least that, and the peak, the z and the impulse
    computed from it are LOWER BOUNDS. The cap is written to the gripper only (so_follower.py:180-
    185), so the default is `CENSORED_LOAD_DIMS`: an arm joint reading exactly 500 on its way to
    -842 is a measurement, not a censored one. Reported, never silently dropped — dropping a channel
    is a change to the pre-registered `DISTAL_LOAD` set and that is Faith's to ratify.
    """
    s = np.asarray(state, dtype=np.float64)
    names = state_columns()
    dims = list(CENSORED_LOAD_DIMS) if dims is None else list(dims)
    out = {}
    for d in dims:
        hits = int((np.abs(s[:, d]) == float(cap)).sum())
        if hits:
            out[names[d]] = hits
    return out


def check_channels_live(state, source: str) -> None:
    """Refuse a state whose load/current channels are not a live instrument reading — gate G-T's
    non-constant clause (module docstring; "the v4 protocol (text held until unblinding; SHA-256 in the README)" §3 and its
    Sep 6 16:10 amendment). Raises `TelemetryError` NAMING the offending columns: the operator's
    next move is to go and look at that servo, so a bare count is not enough.

    Two shapes, one fault. NaN is what the sidecar writes when the observation carries no `.load` /
    `.current` key (tools/telemetry_sidecar.py:27 — "a missing key is NaN in its column, never a
    shifted row"), i.e. the `so_follower.py` load patch is not in place; the project notes "Money /
    accounts" (2) records that this patch dies on any `pip install -U lerobot`, which is how a
    silently dead stream actually reaches the bench. A FROZEN column is a bus that stopped updating
    and left the last good value repeated. Neither is a measurement, and both produce the same
    "0 contact events" as a clean approach that never pressed on the rack.

    NaN is refused on ALL 18 channels; constancy only on the 12 load/current channels [6:18], which
    is what §3 specifies — a position channel that never moves is a real trajectory (v4 ep 20 holds
    the jaw at 13.2 through transit), not a dead instrument.
    """
    s = np.asarray(state, dtype=np.float64)
    names = state_columns()
    if s.ndim != 2 or s.shape[1] != len(names):
        raise TelemetryError(f"{source}: expected an (n, 18) state, got {s.shape}")
    nan = [names[k] for k in range(len(names)) if np.isnan(s[:, k]).any()]
    if nan:
        raise TelemetryError(f"{source}: NaN in channel(s) {nan} — the sidecar is not seeing the "
                             f"servos (gate G-T; a missing .load/.current key writes NaN, "
                             f"telemetry_sidecar.py:27)")
    flat = [names[k] for k in range(6, 18) if s[:, k].std() == 0.0]
    if flat:
        raise TelemetryError(f"{source}: constant load/current column(s) {flat} — the sidecar is "
                             f"not seeing the servos (gate G-T, the non-constant clause)")


def check_action_live(action, source: str) -> None:
    """Refuse a NaN action stream. Same silent shape on the other input: `(A[t] - P[t]) . u_hat <=
    -DEADBAND_DEG` is False at every tick when A is NaN, so `first_opposing_command` reports the
    ORDINARY reason "no opposing command before the end of the trace" and the trial drops out of
    the median latency unremarked — endpoint 2 computed over fewer trials than it says.

    Constant action COLUMNS are deliberately NOT refused, unlike the load/current channels: gate
    G-T's non-constant clause is about the instrument, and a commanded joint that never moves is a
    real trajectory. Gating it here would invent a criterion the pre-registration does not have.
    """
    a = np.asarray(action, dtype=np.float64)
    bad = [i for i in range(a.shape[1]) if np.isnan(a[:, i]).any()] if a.ndim == 2 else []
    if bad:
        names = [state_columns()[i].split(".")[0] for i in bad]
        raise TelemetryError(f"{source}: NaN in the action stream, dim(s) {bad} {names} — every "
                             f"latency would come back undefined with an ordinary-looking reason")


def load_telemetry(path) -> tuple[np.ndarray, dict]:
    """(state (n, 18) float64 in dataset layout, meta). Columns are found by NAME."""
    path = Path(path)
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise TelemetryError(f"{path.name}: no rows")
    cols = state_columns()
    have = set(rows[0].keys())
    missing = [c for c in cols if c not in have]
    if missing:
        raise TelemetryError(f"{path.name}: missing column(s) {missing} — not a sidecar CSV")
    state = np.empty((len(rows), len(cols)), dtype=np.float64)
    for k, name in enumerate(cols):
        state[:, k] = [float(r[name]) if r[name] not in ("", None) else np.nan for r in rows]
    check_channels_live(state, path.name)
    fi_col, t_col = ts.HEADER[0], ts.HEADER[1]
    meta = {
        "path": str(path), "n": len(rows),
        "frame_index": [int(float(r[fi_col])) for r in rows] if fi_col in have else None,
        "t_mono": [float(r[t_col]) for r in rows] if t_col in have else None,
    }
    return state, meta


def state_to_telemetry_rows(state, fps: int = FPS, t0: float = 0.0) -> list[list]:
    """Dataset observation.state (n, 18) -> sidecar rows [frame_index, t_mono, *18] under
    `telemetry_sidecar.HEADER`. The k-th channel of the header is dataset dim k."""
    s = np.asarray(state, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] != len(state_columns()):
        raise TelemetryError(f"expected an (n, 18) state, got {s.shape}")
    return [[i, t0 + i / fps] + [float(v) for v in s[i]] for i in range(len(s))]


# --- the baseline -----------------------------------------------------------------

LOOP_HZ_MIX_TOL = 0.02     # the largest relative spread of loop rates pooled into ONE baseline


def loop_rate(t_mono) -> float | None:
    """Samples per second on the sidecar's own clock, (n - 1) / (t[-1] - t[0]); None without one."""
    if t_mono is None or len(t_mono) < 2:
        return None
    t = np.asarray(t_mono, dtype=np.float64)
    span = float(t[-1] - t[0])
    return float((len(t) - 1) / span) if span > 0 else None


@dataclass
class Baseline:
    mean: np.ndarray
    std: np.ndarray
    source: str
    n_rows: int
    loop_hz: float | None = None   # the rate its replays ran at; None when not measured

    def summary(self) -> dict:
        return {"source": self.source, "n_rows": int(self.n_rows), "loop_hz": self.loop_hz,
                "mean": [float(x) for x in self.mean], "std": [float(x) for x in self.std]}

    @classmethod
    def from_states(cls, states, source: str) -> "Baseline":
        s = np.asarray(states, dtype=np.float64)
        if s.ndim != 2 or len(s) < MIN_BASELINE_ROWS:
            raise TelemetryError(f"baseline: {0 if s.ndim != 2 else len(s)} rows < {MIN_BASELINE_ROWS} ({source})")
        check_channels_live(s, f"baseline ({source})")
        return cls(mean=s.mean(axis=0), std=s.std(axis=0), source=source, n_rows=len(s))

    @classmethod
    def from_csvs(cls, paths) -> "Baseline":
        """Per-block mu/sigma over every row of the block's certification-replay telemetry CSVs,
        and the loop rate they ran at.

        THE RATE IS PART OF THE INSTRUMENT. Block B1's replays ran at 18.523 / 18.626 / 18.601 Hz;
        `tools/replay_telemetry.py` paces to an absolute deadline since Sep 7, so later replays run
        at ~20 Hz, where the arm moves ~8 % faster through the same demonstration. The two kinds
        are told apart by `loop_hz` in every output, and CSVs whose rates differ by more than
        LOOP_HZ_MIX_TOL are refused rather than pooled into one mu/sigma (gate, don't narrate)."""
        paths = [Path(p) for p in paths]
        if not paths:
            raise TelemetryError("baseline: no telemetry CSVs given — a per-block baseline is required "
                                 "(the certification replays' *_telemetry.csv)")
        loaded = [load_telemetry(p) for p in paths]
        rates = [loop_rate(m["t_mono"]) for _s, m in loaded]
        known = [(p.name, r) for p, r in zip(paths, rates) if r is not None]
        if len(known) >= 2:
            lo, hi = min(r for _n, r in known), max(r for _n, r in known)
            if hi / lo - 1.0 > LOOP_HZ_MIX_TOL:
                raise TelemetryError(
                    "baseline: the CSVs ran at different loop rates "
                    + ", ".join(f"{n} {r:.3f} Hz" for n, r in known)
                    + f" (spread {100 * (hi / lo - 1.0):.1f} % > {100 * LOOP_HZ_MIX_TOL:.0f} %) — replays "
                      "recorded before and after the Sep 7 pacing fix are different instruments; "
                      "build the baseline from one kind")
        b = cls.from_states(np.concatenate([st for st, _m in loaded]),
                            source="csv: " + ", ".join(p.name for p in paths))
        if len(known) == len(loaded):
            intervals = sum(len(m["t_mono"]) - 1 for _s, m in loaded)
            span = sum(float(m["t_mono"][-1]) - float(m["t_mono"][0]) for _s, m in loaded)
            b.loop_hz = float(intervals / span) if span > 0 else None
        return b

    @classmethod
    def from_json(cls, path) -> "Baseline":
        """A baseline saved by a previous run. Gated like the others: this is the one route to
        `load_z` that does not pass through `from_states`, so an unchecked mu/sigma here puts a dead
        instrument straight into the z-score. NaN is the silent shape (z is NaN, no events, exit 0
        — the same failure as a dead trial); a zero sigma on a load/current channel is the loud one
        (z = |x| / 1e-6). Zero sigma on a POSITION channel is not gated, for the reason in
        `check_channels_live`: a joint that never moves is a real trajectory."""
        d = json.load(open(path))
        mean = np.asarray(d["mean"], dtype=np.float64)
        std = np.asarray(d["std"], dtype=np.float64)
        src = f"json: {Path(path).name} ({d.get('source', '?')})"
        names = state_columns()
        if mean.shape != (len(names),) or std.shape != (len(names),):
            raise TelemetryError(f"{src}: mean/std are {mean.shape}/{std.shape}, need ({len(names)},)")
        bad = [names[k] for k in range(len(names)) if np.isnan(mean[k]) or np.isnan(std[k])]
        if bad:
            raise TelemetryError(f"{src}: NaN in the baseline mu/sigma for channel(s) {bad} — every z "
                                 f"would be NaN and every trial would report zero contact events")
        zero = [names[k] for k in range(6, 18) if std[k] == 0.0]
        if zero:
            raise TelemetryError(f"{src}: zero sigma on load/current channel(s) {zero} — the baseline it "
                                 f"came from was not seeing the servos (gate G-T)")
        hz = d.get("loop_hz")
        return cls(mean=mean, std=std, source=src, n_rows=int(d.get("n_rows", 0)),
                   loop_hz=None if hz is None else float(hz))


# --- the detector -------------------------------------------------------------------

@dataclass
class Event:
    onset: int
    end: int            # exclusive
    peak_idx: int
    peak_z: float

    @property
    def dwell_ticks(self) -> int:
        return self.end - self.onset


def find_events(z, thr: float = ab.CONTACT_Z, min_ticks: int = ONSET_TICKS, off_ticks: int = OFF_TICKS) -> list[Event]:
    """Runs of >= min_ticks ticks with z > thr; an event ends at the first tick of a run of
    >= off_ticks ticks with z <= thr (or at the end of the trace)."""
    z = np.asarray(z, dtype=np.float64)
    over = z > thr
    events, n, i = [], len(z), 0
    while i < n:
        if not over[i]:
            i += 1
            continue
        j = i
        while j < n and over[j]:
            j += 1
        if j - i < min_ticks:
            i = j
            continue
        onset = i
        # inside the event: extend across dips shorter than off_ticks
        end = j
        k = j
        while k < n:
            below = 0
            while k < n and not over[k]:
                below += 1
                k += 1
            if below >= off_ticks or k >= n:
                break
            while k < n and over[k]:
                k += 1
            end = k
        seg = z[onset:end]
        pk = onset + int(np.argmax(seg))
        events.append(Event(onset=onset, end=end, peak_idx=pk, peak_z=float(seg.max())))
        i = end
    return events


def first_opposing_command(state, action, onset: int, fps: int = FPS, lookback: int = LOOKBACK_TICKS,
                           motion_min: float = MOTION_MIN_DEG, deadband: float = DEADBAND_DEG,
                           t=None, hard_end: int | None = None, resp_max_s: float | None = None,
                           rule: str = DEFAULT_RULE) -> dict:
    """The first OPPOSING command after a load onset, under `rule` (module docstring, "THE RULE,
    VERSIONED"): (A[t] - P[t]) · û <= -deadband, û the unit observed motion over the lookback
    before onset, arm joints only.

    RULE_PREREGISTERED (the default) is 8958af5's search, line for line: `ref = max(onset -
    lookback, 0)` in ticks, and the first sample t >= onset anywhere in the rest of the trace. A
    command already opposing AT the onset therefore scores latency 0.00 s — that is the rule's
    number — and is labelled `lead` with `lead_ticks` so it is never read as a response. It takes
    no horizon: passing `hard_end` or `resp_max_s` with it is a ValueError, not a quiet amendment.

    RULE_AMENDED is the Sep 7 proposal: the lookback is `lookback / fps` seconds of the real clock,
    the search starts strictly after the onset, and it stops at `hard_end` (exclusive) and at
    `resp_max_s` (RESP_MAX_S, read at call time) from the onset. A lead is undefined WITH A REASON.
    Measured on T0-SMOKE (Sep 7): the event at tick 15 (dwell 0.55 s) is handed a reversal 2.55 s
    after it ended by the pre-registered search and none by the amended one; the event at tick 103
    is a 3-tick lead, 0.00 s pre-registered and undefined amended.

    Out of range, stationary or absent is `None` WITH A REASON, which the median skips, never 0.
    """
    check_rule(rule)
    pre = rule == RULE_PREREGISTERED
    if pre and (hard_end is not None or resp_max_s is not None):
        raise ValueError("the pre-registered rule has no response horizon (8958af5 searched to the end "
                         "of the trace); pass rule=RULE_AMENDED to bound the search")
    pos = np.asarray(state, dtype=np.float64)[:, ARM_DIMS]
    act = np.asarray(action, dtype=np.float64)[:, ARM_DIMS]
    n = len(pos)
    if t is None:
        t = np.arange(n, dtype=np.float64) / float(fps)
    t = np.asarray(t, dtype=np.float64)
    if pre:
        ref = max(int(onset) - int(lookback), 0)                        # 8958af5: in ticks
    else:
        ref = int(np.searchsorted(t, t[onset] - lookback / float(fps), side="left"))
    u = pos[onset] - pos[ref]
    norm = float(np.linalg.norm(u))
    out = {"rule_version": rule, "frame": None, "latency_s": None, "reason": None, "lead": False,
           "lead_ticks": 0, "lead_s": 0.0,
           "approach_direction": [float(x) for x in u], "approach_motion_deg": norm}
    if norm < motion_min:
        out["reason"] = f"stationary at onset ({norm:.2f} deg over {onset - ref} ticks < {motion_min})"
        return out
    uhat = u / norm
    # explicit: BLAS matmul on a strided slice warned on the real data
    proj = ((act - pos) * uhat).sum(axis=1)
    opposing = proj <= -deadband

    if opposing[onset]:                       # already backing off when the load crossed threshold
        k = onset
        while k > 0 and opposing[k - 1]:
            k -= 1
        out.update(lead=True, lead_ticks=int(onset - k + 1), lead_s=float(t[onset] - t[k]))
        if pre:
            out.update(frame=int(onset), latency_s=0.0)      # t >= onset: the rule's 0.00, labelled
            return out
        out["reason"] = (f"the command was already opposing at the onset tick, and had been for "
                         f"{out['lead_ticks']} ticks ({out['lead_s']:.2f} s): a LEAD, not a response")
        return out

    # From here the onset sample is known NOT to oppose, so starting at onset + 1 is identical to
    # 8958af5's `t >= onset` for the pre-registered rule and is the amendment's "strictly after".
    lo = onset + 1
    if pre:
        hi = n
    else:
        rmax = RESP_MAX_S if resp_max_s is None else float(resp_max_s)
        hi = int(np.searchsorted(t, t[onset] + rmax, side="right"))
        hi = min(hi, n if hard_end is None else min(n, int(hard_end)))
    if lo >= hi:
        out["reason"] = ("no sample after the onset before the end of the trace" if pre else
                         f"no sample after the onset inside the response horizon ({rmax:.1f} s, and "
                         f"before the event's own end)")
        return out
    hits = np.where(opposing[lo:hi])[0]
    if len(hits):
        k = lo + int(hits[0])
        out.update(frame=k, latency_s=float(t[k] - t[onset]))
    else:
        out["reason"] = ("no opposing command before the end of the trace" if pre else
                         f"no opposing command within {rmax:.1f} s of the onset and before the end of "
                         f"the event's response window")
    return out


def response_horizon(t, ev: Event, resp_max_s: float | None = None, grace_s: float | None = None) -> int:
    """RULE_AMENDED only. Exclusive upper bound on THIS event's response search: whichever comes
    first of the event's end plus RESP_GRACE_S, and RESP_MAX_S from the onset (both read at call
    time, so the sensitivity band can be computed by setting the declared constants)."""
    rmax = RESP_MAX_S if resp_max_s is None else float(resp_max_s)
    grace = RESP_GRACE_S if grace_s is None else float(grace_s)
    t = np.asarray(t, dtype=np.float64)
    end_t = _edge_time(t, ev.end)
    by_grace = int(np.searchsorted(t, end_t + grace, side="right"))
    by_max = int(np.searchsorted(t, float(t[ev.onset]) + rmax, side="right"))
    return int(min(by_grace, by_max, len(t)))


def describe_event(ev: Event, state, action, baseline: Baseline, fps: int = FPS, t=None,
                   rule: str = DEFAULT_RULE) -> dict:
    check_rule(rule)
    s = np.asarray(state, dtype=np.float64)
    if t is None:
        t = np.arange(len(s), dtype=np.float64) / float(fps)
    t = np.asarray(t, dtype=np.float64)
    dt = _dt(t)
    zc = np.abs((s[ev.peak_idx, ab.DISTAL_LOAD] - baseline.mean[ab.DISTAL_LOAD]) / (baseline.std[ab.DISTAL_LOAD] + 1e-6))
    dim = ab.DISTAL_LOAD[int(np.argmax(zc))]
    joint = state_columns()[dim].split(".")[0]
    excess = np.abs(s[ev.onset:ev.end, dim] - baseline.mean[dim])
    # the event's own channel, at its own cap — and only a channel the cap is written to has one
    capped = dim in CENSORED_LOAD_DIMS
    at_cap = (np.abs(s[ev.onset:ev.end, dim]) == MAX_TORQUE_LIMIT) if capped else np.zeros(ev.end - ev.onset, bool)
    end_t = _edge_time(t, ev.end)
    d = {
        "onset": ev.onset, "end": ev.end, "onset_s": float(t[ev.onset]), "end_s": end_t,
        "dwell_s": end_t - float(t[ev.onset]), "joint": joint, "dim": dim,
        "peak_z": ev.peak_z, "peak_s": float(t[ev.peak_idx]), "peak_load": float(s[ev.peak_idx, dim]),
        "impulse_load_s": float((excess * dt[ev.onset:ev.end]).sum()),
        "censored_ticks_in_event": int(at_cap.sum()),
        "peak_is_lower_bound": bool(capped and (abs(float(s[ev.peak_idx, dim])) == MAX_TORQUE_LIMIT or at_cap.any())),
        "latency_s": None, "opposing_frame": None, "latency_reason": "no action stream",
        "lead": None, "lead_ticks": None, "lead_s": None,
    }
    if action is not None:
        if rule == RULE_PREREGISTERED:
            r = first_opposing_command(s, action, ev.onset, fps=fps, t=t, rule=rule)
        else:
            r = first_opposing_command(s, action, ev.onset, fps=fps, t=t, rule=rule,
                                       hard_end=response_horizon(t, ev))
        d.update(latency_s=r["latency_s"], opposing_frame=r["frame"], latency_reason=r["reason"],
                 approach_motion_deg=r["approach_motion_deg"],
                 lead=r["lead"], lead_ticks=r["lead_ticks"], lead_s=r["lead_s"])
    return d


def analyze(state, action, baseline: Baseline, fps: int = FPS, source: str = "trial",
            t_mono=None, rule: str = DEFAULT_RULE) -> dict:
    """All contact events of one trial. `action` (n, 6) or None; must align 1:1 with the rows.
    `rule` picks the latency search (RULE_PREREGISTERED unless asked); the output names it.

    Gate G-T is enforced HERE and not only in `load_telemetry`, because this is the choke point
    every contact number flows through — the manipulation check reaches it from the parquet, never
    from a CSV (module docstring, "THE TELEMETRY GATE"). The structural complaints (shape,
    alignment) are raised first: a misaligned action stream is the more fundamental fault, and
    saying "constant load column" about a state that is not even the right shape misleads."""
    check_rule(rule)
    s = np.asarray(state, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] != 18:
        raise TelemetryError(f"expected an (n, 18) state, got {s.shape}")
    if action is not None:
        a = np.asarray(action, dtype=np.float64)
        if a.ndim != 2 or len(a) != len(s):
            raise TelemetryError(f"action stream ({a.shape}) does not align 1:1 with the telemetry rows "
                                 f"({len(s)}) — the sidecar's frame index is the recorded frame index")
        if a.shape[1] < 6:
            raise TelemetryError(f"action stream has {a.shape[1]} dims, need 6")
    check_channels_live(s, source)
    if action is not None:
        check_action_live(action, source)
    t, clock = sample_times(len(s), t_mono=t_mono, fps=fps)
    z = ab.load_z(s, baseline.mean, baseline.std)
    events = find_events(z)
    rack, hover = rack_window(s, action), hover_window(s)
    described = []
    for e in events:
        d = describe_event(e, s, action, baseline, fps=fps, t=t, rule=rule)
        d["in_rack_window"] = bool(rack is not None and rack[0] <= e.onset < rack[1])
        d["in_hover_window"] = bool(hover is not None and hover[0] <= e.onset < hover[1])
        described.append(d)
    cens = censored_ticks(s)
    # the z_max is a lower bound when the channel that carries it is a capped one, sitting on its cap
    zi = int(np.argmax(z)) if len(z) else 0
    zc = (np.abs((s[zi, ab.DISTAL_LOAD] - baseline.mean[ab.DISTAL_LOAD])
                 / (baseline.std[ab.DISTAL_LOAD] + 1e-6)) if len(z) else np.zeros(len(ab.DISTAL_LOAD)))
    zdim = ab.DISTAL_LOAD[int(np.argmax(zc))]
    return {
        "rule": {"thr": ab.CONTACT_Z, "min_ticks": ONSET_TICKS, "off_ticks": OFF_TICKS},
        "rule_version": rule,
        "distal_load_dims": list(ab.DISTAL_LOAD),
        "baseline": {"source": baseline.source, "n_rows": int(baseline.n_rows), "loop_hz": baseline.loop_hz},
        "fps": fps, "clock": clock, "n_frames": int(len(s)),
        "duration_s": float(t[-1]) if len(t) > 1 else 0.0,
        "loop_hz": float((len(t) - 1) / t[-1]) if len(t) > 1 and t[-1] > 0 else None,
        "z_max": float(z.max()) if len(z) else float("nan"),
        "z_max_frame": zi if len(z) else None,
        "z_max_channel": state_columns()[zdim] if len(z) else None,
        "censored": cens,
        "z_max_is_lower_bound": bool(len(z) and zdim in CENSORED_LOAD_DIMS
                                     and abs(float(s[zi, zdim])) == MAX_TORQUE_LIMIT),
        "windows": {"rack": None if rack is None else [float(t[rack[0]]), _edge_time(t, rack[1])],
                    "hover": None if hover is None else [float(t[hover[0]]), _edge_time(t, hover[1])]},
        "events": described,
    }


# --- the manipulation check -----------------------------------------------------------

def _manifest() -> v4_manifest.Manifest:
    return v4_manifest.Manifest.load(MANIFEST_PATH)


def holdout_episodes() -> list[int]:
    m = _manifest()
    kept = sorted(e for e in m.assigned if e not in m.voided)
    return split.pick_holdouts(m, kept, seed=split.SEED)


def training_episodes() -> list[int]:
    """Exactly as tests/conftest.py `v4_training_episodes` derives them."""
    m = _manifest()
    return m.training_episodes(holdout_episodes())


def hover_window(state) -> tuple[int, int] | None:
    """Jaws-open hover at the pick site: first open -> grasp (v4_audit's approach signature)."""
    pan, grip = state[:, 0], state[:, 5]
    opened = np.where((pan > v4_audit.PICK_SITE_PAN) & (grip > v4_audit.JAW_OPEN))[0]
    gi = v4_audit.grasp_index(pan, grip)
    if not len(opened) or gi is None or gi <= opened[0]:
        return None
    return int(opened[0]), int(gi)


def rack_window(state, action=None) -> tuple[int, int] | None:
    """Rack phase up to the RELEASE COMMAND. Measured on v4 ep 0 (Sep 6): the commanded jaw
    crosses 20 deg at 7.05 s, the gripper load jumps -120 -> 274 at 7.15 s, the observed jaw crosses
    20 deg at 7.25 s. Cut on the observed jaw and the release's own load spike sits inside the
    window — the specified rule then 'detected' the release in 30/30 CLEAN rack phases (joint =
    gripper). The cut is the rising edge of the commanded jaw: the first rack tick where the command
    exceeds its running minimum since rack entry by RELEASE_CMD_RISE_DEG. Without an action stream
    the observed jaw (> JAW_OPEN) is the only signal there is."""
    pan, wflex, grip = state[:, 0], state[:, 3], state[:, 5]
    rack = v4_audit.rack_slice(pan, wflex)
    if len(rack) < 10:
        return None
    start, last = int(rack[0]), int(rack[-1])
    end = last + 1
    if action is not None:
        cmd = np.asarray(action, dtype=np.float64)[:, 5]
        runmin = np.minimum.accumulate(cmd[start:last + 1])
        rise = np.where(cmd[start:last + 1] - runmin > RELEASE_CMD_RISE_DEG)[0]
        if len(rise):
            end = start + int(rise[0])
    else:
        rel = [int(i) for i in rack if i > start and grip[i] > v4_audit.JAW_OPEN]
        if rel:
            end = rel[0]
    return (start, end) if end > start else None


def stall_reference(state, action, window) -> int | None:
    """Kinematic contact reference, independent of the load channels: first run of >= ONSET_TICKS
    ticks in the window where the command leads the position by >= STALL_GAP_DEG on
    lift/elbow/wrist_flex while the arm moves < STALL_SPEED_DEG_PER_TICK."""
    if window is None or action is None:
        return None
    a, b = window
    pos = np.asarray(state, dtype=np.float64)[:, STALL_DIMS]
    act = np.asarray(action, dtype=np.float64)[:, STALL_DIMS]
    gap = np.linalg.norm(act - pos, axis=1)
    speed = np.r_[np.inf, np.linalg.norm(np.diff(pos, axis=0), axis=1)]
    flag = (gap >= STALL_GAP_DEG) & (speed < STALL_SPEED_DEG_PER_TICK)
    run = 0
    for t in range(a, min(b, len(flag))):
        run = run + 1 if flag[t] else 0
        if run >= ONSET_TICKS:
            return t - ONSET_TICKS + 1
    return None


def _carry_window(state) -> tuple[int | None, int | None, str | None]:
    """(grasp index, rack-entry index, None), or (None, None, why the carry is not there)."""
    s = np.asarray(state, dtype=np.float64)
    pan, wflex, grip = s[:, 0], s[:, 3], s[:, 5]
    rack = v4_audit.rack_slice(pan, wflex)
    if len(rack) < 10:
        return None, None, "the arm never entered the rack area"
    gi = v4_audit.grasp_index(pan, grip)
    if gi is None:
        return None, None, "no grasp"
    entry = int(rack[0])
    if entry <= gi:
        return None, None, "rack entry precedes the grasp"
    return int(gi), entry, None


def carry_jaw_min(state) -> float | None:
    """The smallest observed jaw width between the grasp and rack entry — the number reached_T
    thresholds, and the number JAW_HELD_CALIBRATION is derived from. None when there is no carry."""
    gi, entry, _why = _carry_window(state)
    return None if gi is None else float(np.asarray(state, dtype=np.float64)[gi:entry, 5].min())


def reached_T(state, jaw_held_min: float = JAW_HELD_MIN_DEG) -> tuple[bool, str | None]:
    """Stage T (transported) FROM TELEMETRY ALONE — a PROXY for the adjudicated stage letter.

    §1 restricts co-primary 2 to "trials reaching T", and T is a stage of "the existing adjudicated
    taxonomy" ("the v4 protocol (text held until unblinding; SHA-256 in the README)":56-59): the video decides. This proxy is
    reported beside every trial's scalars (`reached_T_telemetry`) so disagreements with the video
    can be listed; it replaces the adjudicated letter in the scored column only when
    `--reached-T-from-telemetry` is passed, which is a ratification item.

    THE RULE: the arm entered the rack area (>= 10 samples of `v4_audit.rack_slice`) after a grasp,
    and the jaw never fell below `jaw_held_min` between the grasp and rack entry. A jaw closed on
    the cap holds its width; a jaw closed on nothing sits at its closed stop.

    ITS HONEST LIMIT: telemetry cannot see the tube, and the demonstrations cannot validate the
    threshold (they contain no negative; JAW_HELD_CALIBRATION). What discriminates is CHAMF-GATE-02
    and the 12 rack-reaching closed-jaw rollouts named in the tests.
    """
    gi, entry, why = _carry_window(state)
    if gi is None:
        return False, why
    lo = float(np.asarray(state, dtype=np.float64)[gi:entry, 5].min())
    if lo < jaw_held_min:
        return False, (f"the jaw collapsed to {lo:.2f} deg during the carry (< {jaw_held_min}) — "
                       f"nothing was held")
    return True, None


REACHED_T_ADJUDICATED = "adjudicated"   # DEFAULT: the scored column is left blank; §1's T is the video's
REACHED_T_TELEMETRY = "telemetry"       # OPT-IN: the jaw-width proxy goes INTO the scored column


def _plural(n: int, one: str, many: str) -> str:
    return one if n == 1 else many


def trial_scalars(state, action, baseline: Baseline, t_mono=None, label: str | None = None,
                  fps: int = FPS, source: str = "trial", rule: str = DEFAULT_RULE,
                  reached_T_source: str = REACHED_T_ADJUDICATED, stamp: str | None = None) -> dict:
    """One trial -> the three scores.csv columns, plus the denominators that make them readable.

    THE RULE is `rule` — RULE_PREREGISTERED unless the caller asks for the amendment — and the
    result names it (`rule_version`). The anchor, the baseline and DISTAL_LOAD are the same under
    both; the rules differ only in which command counts as the first opposing one (module docstring).

    `contact_latency_s` = the MEDIAN over the trial's events with a defined latency (§1 "per-trial
    median latency"). An event whose latency is undefined is not in the median — the frozen
    code's own `_median` skips None, and the Sep 6 17:20 amendment ("Eval Protocol — v4 merged
    (Sep 6).md":441-443) describes a trial whose latencies all came back undefined as having
    "dropped out of the median". A trial with none contributes `None` AND ITS REASONS — an empty
    cell, never a zero. Under the pre-registered rule a lead is in the median at 0.00 s and is
    counted in `n_leads_in_median`.

    `contact_impulse` = the MEDIAN over the trial's events. §1 names the median for the latency and
    no aggregator for the impulse; the median is the reading that applies §1's word to both, and it
    is printed as a convention with `contact_impulse_sum` beside it — a ratification item.

    §1 does not restrict the median to the rack window, so neither does this; the rack-window
    denominators are reported (`n_latencies_in_rack_window`, `latency_from_rack_window`).

    `reached_T` is None (an empty cell) by default, so `tools/v4_bench_analysis.py` restricts on the
    ADJUDICATED stage letter, as §1 says. The telemetry proxy is always reported beside it as
    `reached_T_telemetry`, and goes into `reached_T` only with reached_T_source=REACHED_T_TELEMETRY.
    """
    check_rule(rule)
    if reached_T_source not in (REACHED_T_ADJUDICATED, REACHED_T_TELEMETRY):
        raise ValueError(f"reached_T_source {reached_T_source!r} is neither {REACHED_T_ADJUDICATED!r} "
                         f"nor {REACHED_T_TELEMETRY!r}")
    res = analyze(state, action, baseline, fps=fps, source=source, t_mono=t_mono, rule=rule)
    ok_T, why_T = reached_T(state)
    ev = res["events"]
    defined = [e for e in ev if e["latency_s"] is not None]
    lat = [e["latency_s"] for e in defined]
    imp = [e["impulse_load_s"] for e in ev if e["impulse_load_s"] is not None]
    reasons = [e["latency_reason"] for e in ev if e["latency_s"] is None and e["latency_reason"]]
    in_rack = [e for e in ev if e.get("in_rack_window")]
    n_lat_rack = sum(1 for e in in_rack if e["latency_s"] is not None)
    tel_T = int(bool(ok_T))
    sc = {
        "label": label, "stamp": stamp, "rule_version": rule,
        "contact_latency_s": float(np.median(lat)) if lat else None,
        "contact_impulse": float(np.median(imp)) if imp else None,
        "reached_T": tel_T if reached_T_source == REACHED_T_TELEMETRY else None,
        "reached_T_source": reached_T_source,
        "reached_T_telemetry": tel_T,
        "reached_T_telemetry_reason": why_T,
        "contact_impulse_sum": float(np.sum(imp)) if imp else None,
        "n_events": len(ev),
        "n_latencies": len(lat),
        "n_leads_in_median": sum(1 for e in defined if e.get("lead")),
        "n_events_in_rack_window": len(in_rack),
        "n_latencies_in_rack_window": n_lat_rack,
        "latency_from_rack_window": n_lat_rack > 0,
        "latency_reasons": reasons,
        "clock": res["clock"], "loop_hz": res["loop_hz"], "duration_s": res["duration_s"],
        "censored": res["censored"],
        "impulse_is_lower_bound": any(e["peak_is_lower_bound"] for e in ev),
        "baseline": dict(res["baseline"]),
    }
    sc["conventions"] = scalar_conventions(sc)
    return sc


def scalar_conventions(sc: dict) -> list[str]:
    """The conventions that produced a trial's scalars, as the lines the CLI prints beside them."""
    n, m = sc["n_latencies"], sc["n_events"]
    out = []
    if sc["rule_version"] == RULE_PREREGISTERED:
        out.append(f"rule {RULE_PREREGISTERED}: the pre-registered search (8958af5) — the first opposing "
                   f"command at or after the onset, to the end of the trace; lookback {LOOKBACK_TICKS} ticks")
    else:
        out.append(f"rule {RULE_AMENDED} (UNRATIFIED Sep 7 amendment): search bounded by RESP_MAX_S "
                   f"{RESP_MAX_S:g} s and the event's end + RESP_GRACE_S {RESP_GRACE_S:g} s; leads "
                   f"excluded; lookback {LOOKBACK_TICKS / FPS:g} s of the real clock")
    out.append(f"contact_latency_s = the median of the {n} defined {_plural(n, 'latency', 'latencies')} "
               f"of {m} {_plural(m, 'event', 'events')}; an undefined latency is not in the median")
    if sc["n_leads_in_median"]:
        k = sc["n_leads_in_median"]
        out.append(f"{k} of those {n} {_plural(k, 'is a LEAD', 'are LEADS')} (the command already opposing at "
                   f"the onset), scored 0.00 s by the pre-registered rule — not a response")
    out.append(f"contact_impulse = the median over the {m} {_plural(m, 'event', 'events')} (§1 names no "
               f"aggregator for the impulse; a ratification item)")
    out.append(f"{sc['n_latencies_in_rack_window']} of the {n} {_plural(n, 'latency', 'latencies')} came from "
               f"the rack window (§1 does not restrict the median to it)")
    if sc["reached_T_source"] == REACHED_T_TELEMETRY:
        out.append(f"reached_T = the telemetry proxy ({sc['reached_T_telemetry']}), by --reached-T-from-"
                   f"telemetry — it overrides the adjudicated stage letter in the analysis")
    else:
        out.append(f"reached_T left blank: the analysis restricts on the adjudicated stage letter (§1); "
                   f"the telemetry proxy says {sc['reached_T_telemetry']}")
    return out


def _cell(v):
    """A CSV cell: an undefined value is EMPTY, never 0 — `v4_bench_analysis` reads any non-empty
    cell as a number and a 0 latency would enter the paired test as 'responded instantly'."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return v


def write_scalar_rows(path, scalars) -> None:
    """The three promised columns, keyed by the ledger's BARE label — a scores.csv-compatible fragment.

    Deliberately NOT a whole scores.csv: `stage` and `mechanism` are adjudicated from video and
    this tool has no business writing them. The header is `label` + SCALAR_COLUMNS, all of which
    are `v4_bench_analysis.SCORE_COLUMNS` names, and the label is the ledger's own (the CLI strips
    the run stamp with `split_label_stamp`), so `merge_scalar_rows` joins it on `label` without
    renaming anything. An undefined value is an empty cell (`_cell`).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label"] + list(SCALAR_COLUMNS))
        for s in scalars:
            w.writerow([s.get("label") or ""] + [_cell(s.get(k)) for k in SCALAR_COLUMNS])


_STAMP_RE = re.compile(r"^(?P<label>.+)_(?P<stamp>\d{8}_\d{6})$")


def split_label_stamp(name) -> tuple[str, str | None]:
    """'<label>_<YYYYMMDD_HHMMSS>[_telemetry][.csv]' -> (label, stamp); without a stamp, (stem, None).

    tools/run_scored_trial.sh:69-70 names every run `${LABEL}_${STAMP}` with STAMP=%Y%m%d_%H%M%S and
    :145 writes `${RUN}_telemetry.csv`; the ledger keeps `label` and `stamp` in separate columns
    (`v4_bench_analysis.LEDGER_COLUMNS`) and scores.csv is keyed by the bare label."""
    stem = Path(str(name)).name
    if stem.endswith(".csv"):
        stem = stem[:-len(".csv")]
    if stem.endswith("_telemetry"):
        stem = stem[:-len("_telemetry")]
    m = _STAMP_RE.match(stem)
    return (m.group("label"), m.group("stamp")) if m else (stem, None)


def merge_scalar_rows(score_rows, fragment_rows) -> list[dict]:
    """scores.csv rows with their SCALAR_COLUMNS filled from `--scalars-out` fragments, by label.

    Refuses (TelemetryError) rather than guessing: a fragment label that is not in scores.csv, a
    label in more than one fragment row, a label twice in scores.csv, and a cell that already holds
    a DIFFERENT non-empty value (these columns are never hand-entered, so a disagreement means two
    sources). Every other column is copied through untouched; nothing is written in place."""
    frag: dict[str, dict] = {}
    for r in fragment_rows:
        lab = r.get("label") or ""
        if lab in frag:
            raise TelemetryError(f"label {lab!r} appears in more than one scalar fragment row — which run "
                                 f"counts is a manual call (the ledger's DUPLICATE_LABEL)")
        frag[lab] = r
    labels = [r.get("label") for r in score_rows]
    dup = sorted({lab for lab in labels if labels.count(lab) > 1})
    if dup:
        raise TelemetryError(f"scores.csv holds label(s) {dup} more than once")
    unknown = sorted(set(frag) - set(labels))
    if unknown:
        raise TelemetryError(f"scalar fragment label(s) {unknown} are not in scores.csv — a stamped or "
                             f"misspelt label joins nothing")
    merged = []
    for r in score_rows:
        row = dict(r)
        f = frag.get(r.get("label"))
        if f is not None:
            for k in SCALAR_COLUMNS:
                new, old = f.get(k, "") or "", row.get(k, "") or ""
                if old != "" and str(old) != str(new):
                    raise TelemetryError(f"{r.get('label')}: scores.csv already holds {k}={old!r} and the "
                                         f"fragment says {new!r} — these columns are never hand-entered")
                row[k] = new
        merged.append(row)
    return merged


WINDOW_OF = {"RESIST": ("rack", rack_window), "NUDGE": ("hover", hover_window)}


def _median(vals) -> float | None:
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.median(v)) if v else None


def _episode_in_window(res: dict, state, action, kind: str, fps: int) -> dict:
    win = rack_window(state, action) if kind == "rack" else hover_window(state)
    row = {"window": kind, "window_s": None if win is None else [win[0] / fps, win[1] / fps],
           "detected": False, "onset_s": None, "joint": None, "peak_z": None, "dwell_s": None,
           "impulse_load_s": None, "latency_s": None, "latency_reason": None,
           "stall_s": None, "timing_error_s": None, "z_max_in_window": None, "n_events_in_window": 0}
    if win is None:
        row["latency_reason"] = f"no {kind} window found in the episode"
        return row
    a, b = win
    z = ab.load_z(state, res["_baseline"].mean, res["_baseline"].std)
    row["z_max_in_window"] = float(z[a:b].max()) if b > a else None
    inside = [e for e in res["events"] if a <= e["onset"] < b]
    row["n_events_in_window"] = len(inside)
    st = stall_reference(state, action, win) if kind == "rack" else None
    row["stall_s"] = None if st is None else st / fps
    if inside:
        e = inside[0]
        row.update(detected=True, onset_s=e["onset_s"], joint=e["joint"], peak_z=e["peak_z"], dwell_s=e["dwell_s"],
                   impulse_load_s=e["impulse_load_s"], latency_s=e["latency_s"], latency_reason=e["latency_reason"])
        if kind == "rack" and st is not None:
            row["timing_error_s"] = e["onset_s"] - st / fps
        if kind == "hover":
            row["timing_error_s"] = None  # no kinematic reference exists for a graze
            row["onset_into_window_s"] = e["onset_s"] - a / fps
    return row


def manipulation_check(states: dict, actions: dict, fps: int = FPS) -> dict:
    """The specified rule on the TRAINING RESIST/NUDGE episodes, with the CLEAN control."""
    hold = holdout_episodes()
    if hold != HOLDOUT_HANDOFF:
        raise SplitChanged(f"derived hold-out {hold} != pre-registered {HOLDOUT_HANDOFF} — something upstream "
                           f"changed (manifest or split); stop and report, do not proceed")
    m = _manifest()
    train = training_episodes()
    used = sorted(e for e in train if e in states and e in actions)
    assert not (set(used) & set(hold)) and not (set(used) & set(m.voided))
    if not used:
        raise TelemetryError("manipulation check: none of the training episodes is present")
    baseline = Baseline.from_states(np.concatenate([np.asarray(states[e], dtype=np.float64) for e in used]),
                                    source=f"training episodes ({len(used)}), all frames")
    per_episode = {}
    for e in used:
        s = np.asarray(states[e], dtype=np.float64); a = np.asarray(actions[e], dtype=np.float64)
        res = analyze(s, a, baseline, fps=fps, source=f"episode {e}")
        res["_baseline"] = baseline
        per_episode[e] = {"type": m.type_of(e), "n_events_total": len(res["events"]),
                          "rack": _episode_in_window(res, s, a, "rack", fps),
                          "hover": _episode_in_window(res, s, a, "hover", fps)}

    def _agg(eps, kind):
        rows = [per_episode[e][kind] for e in eps]
        det = [r for r in rows if r["detected"]]
        return {
            "n": len(rows), "detected": len(det), "recall": (len(det) / len(rows)) if rows else None,
            "no_window": sum(r["window_s"] is None for r in rows),
            "onset_s": {e: per_episode[e][kind]["onset_s"] for e in eps},
            "median_timing_error_s": _median([r["timing_error_s"] for r in det]),
            "timing_error_s": [r["timing_error_s"] for r in det],
            "median_latency_s": _median([r["latency_s"] for r in det]),
            "latency_s": [r["latency_s"] for r in det],
            "latency_reasons": [r["latency_reason"] for r in det if r["latency_s"] is None],
            "median_peak_z": _median([r["peak_z"] for r in det]),
            "median_dwell_s": _median([r["dwell_s"] for r in det]),
            "median_impulse_load_s": _median([r["impulse_load_s"] for r in det]),
            "z_max_in_window": [r["z_max_in_window"] for r in rows],
            "joints": sorted({r["joint"] for r in det if r["joint"]}),
            "stall_found": sum(r["stall_s"] is not None for r in rows),
        }

    by_type, controls = {}, {}
    for t, (kind, _) in WINDOW_OF.items():
        eps = [e for e in used if m.type_of(e) == t]
        by_type[t] = {"window": kind, **_agg(eps, kind)}
        ctrl = [e for e in used if m.type_of(e) == CONTROL_TYPE]
        controls[t] = {CONTROL_TYPE: {"window": kind, **_agg(ctrl, kind)}}
    return {
        "rule": {"thr": ab.CONTACT_Z, "min_ticks": ONSET_TICKS, "off_ticks": OFF_TICKS},
        "baseline": baseline.summary(),
        "episodes_used": used, "holdout_excluded": hold, "voided_excluded": list(m.voided),
        "by_type": by_type, "controls": controls, "per_episode": per_episode,
    }


# --- dataset plumbing -------------------------------------------------------------------

def load_v4_dataset(root=None, *, exclude=tuple(HOLDOUT_HANDOFF)):
    """(states_by_episode, actions_by_episode, state_names) or None if the dataset is not here.
    Gates that the dataset's state names equal the sidecar's channel order.

    THE SPENT HOLD-OUT NEVER REACHES THIS PROCESS'S DATA. The episodes in `exclude` — by default,
    and at minimum, HOLDOUT_HANDOFF, used once on Sep 7 and spent — are removed by a filter INSIDE
    `pd.read_parquet` (pyarrow `filters`), so no DataFrame, dict or array built here holds a row of
    theirs; the parquet reader may decode the pages that contain them before it drops them. Their
    absence is asserted after grouping, and an `exclude` that omits any hold-out episode is a
    ValueError: no caller can opt the split back in."""
    import pandas as pd

    exclude = sorted({int(e) for e in exclude})
    omitted = sorted(set(HOLDOUT_HANDOFF) - set(exclude))
    if omitted:
        raise ValueError(f"exclude omits hold-out episode(s) {omitted}: the hold-out {HOLDOUT_HANDOFF} "
                         f"is spent and is never read again")
    if root is None:
        cands = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/lerobot/faithqin/so101-tube-insert-v4_[0-9]*")))
        if not cands:
            return None
        root = cands[0]
    root = Path(root)
    info = root / "meta" / "info.json"
    if not info.exists():
        return None
    names = list(json.load(open(info))["features"]["observation.state"]["names"])
    if names != state_columns():
        raise TelemetryError(f"dataset state names {names} != sidecar channel order {state_columns()} — "
                             f"every channel would be mislabelled")
    pqs = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    if not pqs:
        return None
    df = pd.concat(pd.read_parquet(p, columns=["episode_index", "frame_index", "observation.state", "action"],
                                   filters=[("episode_index", "not in", exclude)])
                   for p in pqs)
    states, actions = {}, {}
    for e, g in df.groupby("episode_index"):
        g = g.sort_values("frame_index")
        states[int(e)] = np.stack(g["observation.state"].to_numpy()).astype(np.float64)
        actions[int(e)] = np.stack(g["action"].to_numpy()).astype(np.float64)
    leaked = sorted(set(states) & set(exclude))
    if leaked:
        raise TelemetryError(f"excluded episode(s) {leaked} came back from the parquet reader despite the "
                             f"read-time filter — stop; nothing may be computed from them")
    return states, actions, names


def load_actions_parquet(path, episode: int | None) -> np.ndarray:
    import pandas as pd

    df = pd.read_parquet(path, columns=["episode_index", "frame_index", "action"])
    eps = sorted(int(x) for x in df["episode_index"].unique())
    if episode is None:
        if len(eps) != 1:
            raise TelemetryError(f"{path}: {len(eps)} episodes {eps}; pass --episode")
        episode = eps[0]
    g = df[df["episode_index"] == episode].sort_values("frame_index")
    if g.empty:
        raise TelemetryError(f"{path}: episode {episode} not present (have {eps})")
    return np.stack(g["action"].to_numpy()).astype(np.float64)


# --- CLI -----------------------------------------------------------------------------------

def _fmt(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def _print_check(res: dict) -> None:
    print(f"MANIPULATION CHECK — rule |z| > {res['rule']['thr']} for >= {res['rule']['min_ticks']} ticks; "
          f"baseline {res['baseline']['source']} ({res['baseline']['n_rows']} rows)")
    print(f"training episodes used: {len(res['episodes_used'])}; hold-out excluded {res['holdout_excluded']}; "
          f"voided excluded {res['voided_excluded']}")
    for t, r in res["by_type"].items():
        c = res["controls"][t][CONTROL_TYPE]
        print(f"\n{t} — window: {r['window']}")
        print(f"  recall {r['detected']}/{r['n']} = {_fmt(r['recall'])}   |   CLEAN control in the same window: "
              f"{c['detected']}/{c['n']} = {_fmt(c['recall'])}  (a threshold without a control is a bug)")
        print(f"  median timing error (onset - stall) {_fmt(r['median_timing_error_s'])} s   CLEAN {_fmt(c['median_timing_error_s'])} s")
        print(f"  median latency to first opposing command {_fmt(r['median_latency_s'])} s   CLEAN {_fmt(c['median_latency_s'])} s")
        print(f"  median peak z {_fmt(r['median_peak_z'], 1)} / dwell {_fmt(r['median_dwell_s'])} s / impulse "
              f"{_fmt(r['median_impulse_load_s'], 0)} load·s;  joints {r['joints']}   CLEAN joints {c['joints']}")
        print(f"  z_max in window  {t}: {[round(x, 1) for x in r['z_max_in_window'] if x is not None]}")
        print(f"  z_max in window  CLEAN: {[round(x, 1) for x in c['z_max_in_window'] if x is not None]}")


def _panel_module():
    """Imported LAZILY and only behind an opt-in flag. `tools/v4_contact_response.py` imports this
    module, so a top-level import would be circular — and, more to the point, the proposal must not
    be on the default path's import graph at all."""
    import v4_contact_response as cr
    return cr


def _print_secondary_panel(state, action, t_mono) -> None:
    cr = _panel_module()
    p = cr.secondary_panel(state, action, t_mono=t_mono)
    print(f"\n  --- SECONDARY PANEL (opt-in) — {p['status']}")
    print(f"      window {p['rack_window_s']} s, clock {p['clock']}")
    for key, v in p["secondaries"].items():
        print(f"      {key:14s} {v['value']}")
        for line in _wrap(v["caveat"], 92):
            print(f"        {line}")


def _wrap(text: str, width: int) -> list[str]:
    import textwrap
    return textwrap.wrap(text, width)


def _print_anchor_evaluability(a) -> int:
    """The measurement behind the demotion proposal. TRAINING EPISODES ONLY — the hold-out is
    filtered out inside the parquet reader by `load_v4_dataset` (never materialised here), and its
    absence is asserted again before any statistic runs."""
    cr = _panel_module()
    ds = load_v4_dataset()
    if ds is None:
        print("no v4 dataset on this machine", file=sys.stderr)
        return 2
    states, actions, _ = ds
    hold = holdout_episodes()
    if hold != HOLDOUT_HANDOFF:
        raise SplitChanged(f"derived hold-out {hold} != pre-registered {HOLDOUT_HANDOFF}")
    train = set(training_episodes())
    states = {e: v for e, v in states.items() if e in train}
    actions = {e: v for e, v in actions.items() if e in train}
    assert not set(states) & set(hold), "hold-out episode reached the measurement"
    m = _manifest()
    types = {e: m.type_of(e) for e in states}
    print("PROPOSAL — NOT RATIFIED. Is the PRE-REGISTERED anchor ('distal-load onset',\n"
          "'the v4 protocol (text held until unblinding; SHA-256 in the README)':56-59) a contact landmark under ANY baseline?\n"
          f"Training episodes only ({len(states)}); hold-out {hold} filtered out inside the parquet "
          f"reader (never materialised).\n")
    for dims, tag in ((None, "pre-registered DISTAL_LOAD [8, 9, 10, 11]"),
                      ([8, 9, 10], "the three uncensored distal channels [8, 9, 10]")):
        res = cr.anchor_evaluability(states, actions, types, dims=dims)
        print(f"  channels: {tag}   pooled sigma {[round(x, 1) for x in res['pooled_sigma']]}")
        for k, b in res["baselines"].items():
            fired = sum(x for x, _n in b["fired"].values())
            total = sum(n for _x, n in b["fired"].values())
            per = "  ".join(f"{lab} {x}/{n}" for lab, (x, n) in b["fired"].items())
            print(f"    {k:9s} fires {fired}/{total}   on the window's own edge {b['on_edge']}/{b['n_fired']}"
                  f"   rho(onset, deepest press) {_fmt(b['spearman_onset_vs_deepest'], 3)}")
            print(f"              {per}")
        print()
    print("  READ IT AS: 'not evaluable as pre-registered'. NOT as 'load carries no contact\n"
          "  signal' — that sentence is not what this measures (protocol §2, the forbidden list).")
    return 0


def _read_rows(path, header) -> list[dict]:
    """CSV rows, refusing any header but `header` (the analysis refuses the same way)."""
    with open(path, newline="") as fh:
        r = csv.DictReader(fh)
        got = list(r.fieldnames or [])
        rows = list(r)
    if got != list(header):
        raise TelemetryError(f"{Path(path).name}: header {got} != expected {list(header)}")
    return rows


def _merge_cli(a) -> int:
    import v4_bench_analysis as ba
    if Path(a.merged_out).resolve() == Path(a.into).resolve():
        raise TelemetryError("--merged-out must be a new file: the adjudicated scores.csv is never overwritten")
    scores = _read_rows(a.into, ba.SCORE_COLUMNS)
    frags = [row for f in a.merge_scalars for row in _read_rows(f, ["label"] + list(SCALAR_COLUMNS))]
    merged = merge_scalar_rows(scores, frags)
    Path(a.merged_out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.merged_out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ba.SCORE_COLUMNS)
        w.writeheader()
        w.writerows(merged)
    print(f"merged {len(frags)} fragment row(s) into {len(merged)} score row(s) by label -> {a.merged_out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--telemetry", help="a trial's *_telemetry.csv")
    ap.add_argument("--baseline", nargs="*", default=[], help="the block's certification-replay telemetry CSVs")
    ap.add_argument("--baseline-json", help="a baseline saved by a previous run (mean/std/source)")
    ap.add_argument("--actions-parquet", help="the recorded episode's parquet (for the latency)")
    ap.add_argument("--episode", type=int, default=None)
    ap.add_argument("--manipulation-check", action="store_true")
    ap.add_argument("--json", help="write the full result here (with --scalars-out, the scalars too)")
    ap.add_argument("--scalars-out", help="write the three promised scores.csv columns "
                                          "(label + " + ", ".join(SCALAR_COLUMNS) + ") here")
    ap.add_argument("--label", help="the trial label for --scalars-out (default: the ledger's bare label, "
                                    "i.e. the telemetry file's stem without the run stamp and _telemetry)")
    ap.add_argument("--merge-scalars", nargs="+", metavar="FRAGMENT",
                    help="fill scores.csv's contact columns from these --scalars-out fragments, by label")
    ap.add_argument("--into", help="with --merge-scalars: the adjudicated scores.csv (read, never written)")
    ap.add_argument("--merged-out", help="with --merge-scalars: the merged scores.csv to write (a new file)")
    # OPT-IN, DEFAULT OFF — departures from the pre-registered method (module docstring). The
    # default path computes RULE_PREREGISTERED against the given baseline and leaves reached_T to
    # the adjudicated stage letter.
    ap.add_argument("--rule", choices=RULES, default=DEFAULT_RULE,
                    help=f"the latency rule (default {DEFAULT_RULE}; {RULE_AMENDED} is the unratified "
                         f"Sep 7 amendment)")
    ap.add_argument("--reached-T-from-telemetry", action="store_true",
                    help="write the telemetry jaw-width proxy INTO the reached_T column (unratified: it "
                         "then overrides the adjudicated stage letter in the analysis)")
    ap.add_argument("--secondary-panel", action="store_true",
                    help="ALSO print the (unratified) SECONDARY panel beside the endpoint")
    ap.add_argument("--anchor-evaluability", action="store_true",
                    help="desk, training episodes only: is 'distal-load onset' a contact landmark "
                         "under ANY baseline? The measurement behind the demotion proposal")
    a = ap.parse_args(argv)
    try:
        if a.merge_scalars:
            if not a.into or not a.merged_out:
                ap.error("--merge-scalars needs --into <scores.csv> and --merged-out <new file>")
            return _merge_cli(a)
        if a.anchor_evaluability:
            return _print_anchor_evaluability(a)
        if a.manipulation_check:
            ds = load_v4_dataset()
            if ds is None:
                print("no v4 dataset on this machine", file=sys.stderr); return 2
            states, actions, _ = ds
            missing = [e for e in training_episodes() if e not in states]
            if missing:
                raise TelemetryError(f"training episodes missing from the dataset on disk: {missing}")
            res = manipulation_check(states, actions)
            _print_check(res)
        else:
            if not a.telemetry:
                ap.error("--telemetry or --manipulation-check")
            if a.baseline:
                bl = Baseline.from_csvs(a.baseline)
            elif a.baseline_json:
                bl = Baseline.from_json(a.baseline_json)
            else:
                raise TelemetryError("no baseline: pass the block's certification-replay telemetry CSVs "
                                     "(--baseline) — there is no silent fallback to dataset statistics")
            state, meta = load_telemetry(a.telemetry)
            act = load_actions_parquet(a.actions_parquet, a.episode) if a.actions_parquet else None
            res = analyze(state, act, bl, t_mono=meta["t_mono"], rule=a.rule)
            res["telemetry"] = meta["path"]
            print(f"{Path(a.telemetry).name}: {res['n_frames']} frames over {res['duration_s']:.2f} s "
                  f"({_fmt(res['loop_hz'], 3)} Hz, clock: {res['clock']}), z_max {res['z_max']:.2f}"
                  f"{' (LOWER BOUND: ' + str(res['z_max_channel']) + ' at its torque cap)' if res['z_max_is_lower_bound'] else ''}, "
                  f"{len(res['events'])} event(s); rule {res['rule_version']}")
            print(f"  baseline {bl.source} ({bl.n_rows} rows, {_fmt(bl.loop_hz, 3)} Hz)")
            if res["censored"]:
                print(f"  CENSORED at |{MAX_TORQUE_LIMIT:g}| (Max_Torque_Limit, written to the gripper only, "
                      f"so_follower.py:180-185): {res['censored']} — peaks there are lower bounds")
            for e in res["events"]:
                if e["latency_s"] is None:
                    note = f" ({e['latency_reason']})"
                elif e.get("lead"):
                    note = (f" [LEAD: already opposing {e['lead_ticks']} ticks at the onset — the pre-registered "
                            f"rule scores it 0.00 s; not a response]")
                else:
                    note = ""
                print(f"  onset {e['onset_s']:6.2f} s  {e['joint']:<12} peak z {e['peak_z']:.1f} @ {e['peak_s']:.2f} s  "
                      f"dwell {e['dwell_s']:.2f} s  impulse {e['impulse_load_s']:.0f} load·s  "
                      f"{'rack' if e['in_rack_window'] else '    '}  latency {_fmt(e['latency_s'])} s{note}")
            if a.scalars_out:
                default_label, stamp = split_label_stamp(a.telemetry)
                label = a.label or default_label
                sc = trial_scalars(state, act, bl, t_mono=meta["t_mono"], label=label, stamp=stamp, rule=a.rule,
                                   reached_T_source=(REACHED_T_TELEMETRY if a.reached_T_from_telemetry
                                                     else REACHED_T_ADJUDICATED))
                write_scalar_rows(a.scalars_out, [sc])
                res["scalars"] = sc
                print(f"\n  label {label}" + (f" (run stamp {stamp})" if stamp else ""))
                print(f"  {'contact_latency_s':20s} {_fmt(sc['contact_latency_s'])} s")
                print(f"  {'contact_impulse':20s} {_fmt(sc['contact_impulse'], 0)} load·s"
                      f"{'  (LOWER BOUND: an event sat on the gripper torque cap)' if sc['impulse_is_lower_bound'] else ''}")
                print(f"  {'reached_T':20s} {'(blank)' if sc['reached_T'] is None else sc['reached_T']}"
                      f"   telemetry proxy: {sc['reached_T_telemetry']}"
                      f"{'  (' + str(sc['reached_T_telemetry_reason']) + ')' if sc['reached_T_telemetry_reason'] else ''}")
                for line in sc["conventions"]:
                    print(f"    convention: {line}")
                for r in sc["latency_reasons"]:
                    print(f"    undefined: {r}")
                print(f"  wrote {a.scalars_out}")
            if a.secondary_panel:
                _print_secondary_panel(state, act, meta["t_mono"])
        if a.json:
            out = {k: v for k, v in res.items() if not k.startswith("_")}
            if "per_episode" in out:
                out["per_episode"] = {str(k): v for k, v in out["per_episode"].items()}
            Path(a.json).parent.mkdir(parents=True, exist_ok=True)
            json.dump(out, open(a.json, "w"), indent=1, default=float)
            print(f"\nwrote {a.json}")
        return 0
    except (TelemetryError, SplitChanged) as e:
        print(f"GATE FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
