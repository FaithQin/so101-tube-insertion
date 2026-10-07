"""Offline boundary-race falsification for the grasp-phase chunk lock.

Faith's test design (Aug 27, 2026): before spending bench trials on the
CAPSTONE_GRASP_LOCK probe, replay the COMMANDED action traces of recorded
rollout trials through the pure GraspLock state machine and ask, per trial:

    would the lock have engaged BEFORE the chunk boundary nearest the
    closing onset — or does the re-plan win the race?

Verdicts per trace:
    PROTECTED     lock engaged strictly before the next chunk boundary
    RACED         closing onset began, but the boundary fired at or before
                  the lock tick (the wrapper refills/replans BEFORE observe()
                  each tick, so a boundary ON the lock tick still replans)
    NEVER_LOCKED  no closing onset inside the grasp band in this trace

Boundaries are ticks 0, n, 2n, ... (n = n_action_steps): the queue empties
and a re-plan (or lock-extension) happens at the START of those ticks.

Usage against a v3.0-format rollout dataset (each episode = one trial):

    conda activate lerobot
    python tools/replay_grasp_lock.py <dataset_root> --n-action-steps 25

Prints a per-trial table and the protected/raced tally. Pure offline; touches
no hardware. Unit-tested in tests/test_replay_grasp_lock.py on synthetic
traces with ground truth known by construction.
"""

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from grasp_lock import GraspLock  # noqa: E402

LIFT_IDX, GRIP_IDX = 1, 5  # dataset action order: pan, lift, elbow, wflex, wroll, gripper


def analyze_trace(cmds, n_action_steps):
    """cmds: iterable of (grip_cmd, lift_cmd) per tick. Returns dict with
    onset_tick (first tick of the decreasing run that locked, or None),
    lock_tick, next_boundary (first boundary AFTER onset), margin_ticks
    (next_boundary - lock_tick), verdict."""
    gl = GraspLock()
    lock_tick = None
    prev_locked = False
    for t, (g, lift) in enumerate(cmds):
        locked = gl.observe(g, lift)
        if locked and not prev_locked and lock_tick is None:
            lock_tick = t
        prev_locked = locked
    if lock_tick is None:
        return {"onset_tick": None, "lock_tick": None, "next_boundary": None,
                "margin_ticks": None, "verdict": "NEVER_LOCKED"}
    # onset = first tick of the ONSET_TICKS-long decreasing run ending at lock
    from grasp_lock import ONSET_TICKS
    onset_tick = lock_tick - (ONSET_TICKS - 1)
    # first boundary strictly after onset began; a boundary AT the lock tick
    # still replans first (wrapper refill runs before observe)
    b = ((onset_tick // n_action_steps) + 1) * n_action_steps
    if b <= lock_tick:
        return {"onset_tick": onset_tick, "lock_tick": lock_tick,
                "next_boundary": b, "margin_ticks": b - lock_tick,
                "verdict": "RACED"}
    return {"onset_tick": onset_tick, "lock_tick": lock_tick,
            "next_boundary": b, "margin_ticks": b - lock_tick,
            "verdict": "PROTECTED"}


def load_trials(root):
    import pandas as pd
    df = pd.concat([pd.read_parquet(p) for p in
                    sorted(glob.glob(str(Path(root) / "data/chunk-*/file-*.parquet")))])
    for i, g in df.groupby("episode_index"):
        import numpy as np
        act = np.stack(g["action"].to_numpy())
        yield int(i), list(zip(act[:, GRIP_IDX].tolist(), act[:, LIFT_IDX].tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="rollout dataset root (v3.0 format)")
    ap.add_argument("--n-action-steps", type=int, default=25)
    args = ap.parse_args()
    tally = {"PROTECTED": 0, "RACED": 0, "NEVER_LOCKED": 0}
    print(f"{'trial':>5}  {'onset':>6}  {'lock':>5}  {'boundary':>8}  {'margin':>6}  verdict")
    for i, cmds in load_trials(args.root):
        r = analyze_trace(cmds, args.n_action_steps)
        tally[r["verdict"]] += 1
        print(f"{i:>5}  {str(r['onset_tick']):>6}  {str(r['lock_tick']):>5}  "
              f"{str(r['next_boundary']):>8}  {str(r['margin_ticks']):>6}  {r['verdict']}")
    n_lockable = tally["PROTECTED"] + tally["RACED"]
    print(f"\ntally: {tally}")
    if n_lockable:
        print(f"lock wins the race in {tally['PROTECTED']}/{n_lockable} trials "
              f"({100 * tally['PROTECTED'] / n_lockable:.0f}%)")


if __name__ == "__main__":
    main()
