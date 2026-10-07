#!/usr/bin/env python
"""Instrumented replay: which joint loses tracking, and only under load?

Born Sep 1 2026. A deterministic replay of known-good episode 0 grasped the
tube and then missed the rack badly — unprecedented since Aug 15 (grasp has
always implied seat on a replay). Faith's live observation localized the fault
to the LOADED phase: approach and grasp run the arm free; transport and
insertion carry the tube. The elbow had shown degraded goal-closing under
strain for two days (1.3–3.9 deg short, stiction both directions, static duty
200–550 at near-zero current).

`lerobot-replay` logs nothing, so "which joint, and is it load-phase-locked?"
was unanswerable. That answer decides a servo swap — a plant change with
study-comparability consequences — so it gets an instrument, not a guess.

Streams the episode's demonstrated actions at the dataset fps (exactly what
lerobot-replay does) while sampling Present_Position every tick, then reports
per-joint mean/max tracking error split at the demonstrated gripper close.

Usage (moves the arm — tube at start, hands clear):
    python tools/replay_tracking.py                     # episode 0, v3
    python tools/replay_tracking.py --episode 3 --json-out out.json

Pure logic (phase split, per-phase errors, verdict) unit-tested in
tests/test_replay_tracking.py; nothing in tests/ touches hardware.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_DATASET = "faithqin/so101-tube-insert-v3_20260827_110242"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Pre-declared fault bar, with its rationale: 1.5 deg is ~4x the validated
# homing spread (0.4 deg, Aug 30) and about half the residual that made a
# deterministic replay sail past the tube entirely (3.6-5.6 deg, Aug 29/31).
# Backlash-scale noise (<~0.5 deg) must pass. Do not adjust after a result.
FAULT_DEG = 1.5

# shoulder_lift settles against its mechanical stop beyond the -100 goal clamp
# (home_arm.py:59-61); its command-vs-actual error is structural, not tracking.
STRUCTURAL_JOINTS = {"shoulder_lift"}


def find_load_onset(cmd: np.ndarray, gripper_idx: int, closed_below: float, min_run: int = 5):
    """First sustained gripper close AFTER a sustained open — the grasp.

    Revised Sep 1 after the first live run: THIS RIG'S GRIPPER STARTS NEAR
    CLOSED (~1.0 normalized at home), opens during the approach, and closes on
    the grasp. "First closed tick" is therefore tick 0 of every episode, which
    classified entire episodes as loaded. The grasp is the close that follows
    the open. `min_run` consecutive ticks are required on both the open and the
    close, so single noisy dips cannot fake either transition.
    """
    g = np.asarray(cmd, dtype=float)[:, gripper_idx]
    closed = g < closed_below

    # 1. find the end of the first sustained OPEN stretch
    run, opened_at = 0, None
    for i, c in enumerate(closed):
        run = 0 if c else run + 1
        if run >= min_run:
            opened_at = i
            break
    if opened_at is None:
        return None  # never opened: no demonstrated grasp transition

    # 2. first sustained close after it
    run = 0
    for i in range(opened_at + 1, len(closed)):
        run = run + 1 if closed[i] else 0
        if run >= min_run:
            return i - min_run + 1
    return None


def settled_mask(cmd_joint: np.ndarray, vel_thresh: float = 0.25, settle_ticks: int = 4) -> np.ndarray:
    """True where the command has been quasi-static long enough to have arrived.

    A P-controlled servo legitimately lags a MOVING command by degrees at
    20 Hz; scoring those ticks flags healthy dynamics (the first live run
    faulted five joints at once this way). A tick is settled when the command
    changed less than `vel_thresh` per tick for the last `settle_ticks` ticks —
    where a healthy joint has arrived and only a real offset remains.
    """
    c = np.asarray(cmd_joint, dtype=float)
    slow = np.abs(np.diff(c, prepend=c[0])) < vel_thresh
    m = np.zeros(len(c), dtype=bool)
    # A command that is static from tick 0 is settled at tick 0 — the arm was
    # homed to it before the episode. Seed the run so the prefix counts.
    run = settle_ticks - 1
    for i, sl in enumerate(slow):
        run = run + 1 if sl else 0
        m[i] = run >= settle_ticks
    return m


def phase_errors(cmd: np.ndarray, actual: np.ndarray, joints: list[str], load_onset):
    """Per-joint |cmd - actual| summarized separately for free vs loaded phase.

    Per joint and per phase on purpose: one pooled number would average the
    elbow's loaded-phase lag away under five healthy joints — the same mistake
    as reporting a mean instead of the tail on latency.
    """
    cmd = np.asarray(cmd, dtype=float)
    actual = np.asarray(actual, dtype=float)
    err = np.abs(cmd - actual)
    out = {}
    for k, j in enumerate(joints):
        # settled ticks only: healthy dynamic lag is not a fault (Sep 1 fix)
        sm = settled_mask(cmd[:, k])
        e = np.where(sm, err[:, k], np.nan)
        free = e[:load_onset] if load_onset is not None else e
        loaded = e[load_onset:] if load_onset is not None else np.array([])
        free = free[~np.isnan(free)]
        loaded = loaded[~np.isnan(loaded)] if loaded.size else loaded
        out[j] = {
            "free_mean": float(free.mean()) if free.size else float("nan"),
            "free_max": float(free.max()) if free.size else float("nan"),
            "loaded_mean": float(loaded.mean()) if loaded.size else float("nan"),
            "loaded_max": float(loaded.max()) if loaded.size else float("nan"),
        }
    return out


def verdict(report: dict) -> dict:
    """FAULT joints, and whether the fault lives only in the loaded phase.

    load_locked=True means every faulted joint is clean while free and bad
    under load — the signature that indicts torque authority rather than
    calibration, commands, or backlash.
    """
    faulted, load_locked = [], True
    for j, r in report.items():
        if j in STRUCTURAL_JOINTS:
            continue
        loaded_bad = (not np.isnan(r["loaded_mean"])) and r["loaded_mean"] > FAULT_DEG
        free_bad = (not np.isnan(r["free_mean"])) and r["free_mean"] > FAULT_DEG
        if loaded_bad or free_bad:
            faulted.append(j)
            if free_bad:
                load_locked = False
    return {
        "status": "FAULT" if faulted else "PASS",
        "joints": faulted,
        "load_locked": load_locked if faulted else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _arms import FOLLOWER_CAL, FOLLOWER_PORT, open_bus

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(args.dataset, episodes=[args.episode])
    fps = ds.meta.fps
    row = ds.meta.episodes[args.episode]
    lo, hi = row["dataset_from_index"], row["dataset_to_index"]
    cmd = np.stack([np.asarray(ds[i]["action"], dtype=float) for i in range(lo, hi)])
    print(f"[track] episode {args.episode}: {len(cmd)} ticks @ {fps} fps", flush=True)

    onset = find_load_onset(cmd, gripper_idx=JOINTS.index("gripper"), closed_below=10.0)
    print(f"[track] demonstrated grasp closes at tick {onset} "
          f"({onset / fps:.1f} s)" if onset is not None else "[track] gripper never closes",
          flush=True)

    bus = open_bus(FOLLOWER_PORT, FOLLOWER_CAL)
    try:
        bus.enable_torque()
    except Exception:
        for j in JOINTS:
            bus.write("Torque_Enable", j, 1, normalize=False)

    actual = np.zeros_like(cmd)
    period = 1.0 / fps
    t0 = time.perf_counter()
    try:
        for i, a in enumerate(cmd):
            for k, j in enumerate(JOINTS):
                bus.write("Goal_Position", j, float(a[k]), normalize=True)
            for k, j in enumerate(JOINTS):
                actual[i, k] = float(bus.read("Present_Position", j, normalize=True))
            # absolute-deadline pacing, so read time does not slow the replay
            next_t = t0 + (i + 1) * period
            dt = next_t - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
    finally:
        # Sep 1: a mid-stream crash (gripper overload) previously left every
        # servo holding torque, jammed gripper included. Cleanup that only
        # runs on success is not cleanup. Torque is kept on the arm joints
        # (parked-holding is this rig's normal), but the bus must close and
        # the GRIPPER must release — it is the joint that jams.
        try:
            bus.write("Torque_Enable", "gripper", 0, normalize=False)
        except Exception:
            pass
        try:
            bus.disconnect(disable_torque=False)
        except Exception:
            pass

    rep = phase_errors(cmd, actual, JOINTS, onset)
    v = verdict(rep)

    print()
    print("=" * 66)
    print("REPLAY TRACKING  (cmd vs actual, deg)")
    print("=" * 66)
    print(f"{'joint':<15}{'free mean':>10}{'free max':>10}{'load mean':>11}{'load max':>10}")
    for j in JOINTS:
        r = rep[j]
        tag = "  <-- STRUCTURAL (stop-seated)" if j in STRUCTURAL_JOINTS else (
            "  <-- FAULT" if j in v["joints"] else "")
        print(f"{j:<15}{r['free_mean']:>10.2f}{r['free_max']:>10.2f}"
              f"{r['loaded_mean']:>11.2f}{r['loaded_max']:>10.2f}{tag}")
    print()
    print(f"  VERDICT   {v['status']}"
          + (f" — {', '.join(v['joints'])}, "
             + ("LOAD-LOCKED (clean free, bad loaded: torque authority)"
                if v["load_locked"] else "both phases (calibration/command path suspect)")
             if v["joints"] else " — all joints track within the pre-declared 1.5 deg"))
    print("=" * 66)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"dataset": args.dataset, "episode": args.episode,
                       "load_onset_tick": onset, "fault_deg": FAULT_DEG,
                       "report": rep, "verdict": v}, f, indent=2)
        print(f"[track] wrote {args.json_out}")
    return 0 if v["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
