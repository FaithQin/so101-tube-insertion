#!/usr/bin/env python
"""Certification replay that writes an 18-channel telemetry CSV — the contact endpoint's baseline.

    python tools/replay_telemetry.py --episode 0                 # v4 ep 0, writes the baseline
    python tools/replay_telemetry.py --episode 0 --out tools/scored_logs/CERT-B1_telemetry.csv

OPERATOR PRECONDITIONS — this MOVES THE ARM through a full demonstrated episode:
  * hands clear of the workspace, and the tube on its mark exactly as for a scored trial;
  * home first (`python tools/home_arm.py --pose v4`) — the replay refuses to start if the arm is
    more than `--max-start-gap` degrees from the episode's first action, because closing a large
    gap in one command is how a replay sweeps through the rack;
  * torque stays ON throughout and the arm is left holding at disconnect
    (`disable_torque_on_disconnect=False`), the same discipline as `tools/home_arm.py` — a
    full-bus release lets the unpowered elbow sag out of training support (Aug 16).

WHY THIS EXISTS (Sep 6 2026, bench morning). `tools/v4_contact_events.py:44` records the gap:
"the certification replays are run by `tools/replay_tracking.py`, which logs positions only and
does not set CAPSTONE_TELEMETRY_CSV; until the replay path writes a telemetry CSV there is no
per-block baseline to load." Contact response is a CO-PRIMARY endpoint of the frozen protocol
("the v4 protocol (text held until unblinding; SHA-256 in the README)"), and its per-block mu/sigma come from the certification
replays of that power cycle. No baseline, no endpoint.

WHY NOT EXTEND replay_tracking.py. That tool drives the arm through `tools/_arms.py`, which is in
RANGE mode; production runs in DEGREES (`SOFollowerConfig.use_degrees = True`). A baseline in one
unit applied to trials recorded in the other is silently wrong in every z-score — the
"plumbing, not policy" failure this project has already paid for once. So this tool drives the SAME
patched `SOFollower` a trial drives (`lerobot/robots/so_follower/so_follower.py`, whose
`_read_bus_state` adds `.load` and `.current` for all six joints unconditionally) and tees its
`get_observation()` through the SAME `tools/telemetry_sidecar.py` a trial tees through. The header
is never retyped here: it is read from `telemetry_sidecar.HEADER`, so the k-th column is the k-th
dataset dim by construction. `tests/test_replay_telemetry.py` pins that identity.

GATE, DON'T NARRATE. Two refusals, both at the bench where they are still fixable rather than at
analysis time where they are not:
  * an observation carrying no load/current channels (the follower patch is not applied) aborts the
    replay instead of writing 12 columns of NaN;
  * after writing, the CSV is handed to `v4_contact_events.Baseline.from_csvs`, the exact consumer.
    If it would be rejected later — too few rows, a NaN, a constant load column — this exits
    non-zero now. A constant load column is the telemetry gate of handoff section 4: it means the
    sidecar is not seeing the servos.

No cameras are opened: a replay is open-loop, so this needs no camera TCC grant and no Terminal
bridge. It is therefore NOT a perception check — a replay that seats certifies the mechanical
plant, not the cameras (replay-as-arbiter).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import telemetry_sidecar as ts  # noqa: E402  (pure: header + CSV writer, no hardware)
import v4_contact_events as ce  # noqa: E402  (pure: the baseline the trials will be scored against)

FOLLOWER_PORT = "/dev/tty.usbmodem5C4C1245641"   # WHITE arm, jaws (the project notes "Environment & hardware facts")

# STALL GUARD. The frozen protocol's chamferless section requires it in as many words: "Abort the
# episode on a sustained stall at the rack, not at the 45 s cap", because "a chamferless insertion
# JAMS rather than slides" and this rig "has already latched a servo twice on stalled presses"
# (the project notes: "No load-guard-free presses"). The elbow's gearbox is already worn (Sep 6), so a jam
# that keeps pressing is the way to lose it.
#
# A bare load limit would misfire: measured over the three clean coned certification replays today,
# normal peaks reach 644-675 on the arm joints (p50 60-204). So the signal is not high load, it is
# high load WITHOUT MOTION, sustained -- which is what a jam looks like and a fast reach does not.
STALL_LOAD = 700.0        # above every peak in three clean replays (max 675, wrist_flex)
STALL_MOVE_DEG = 0.5      # a joint moving more than this is tracking, not jammed
STALL_TICKS = 6           # 0.3 s at 20 fps; shorter than the servos' 2.0 s Protection_Time latch
GUARDED_JOINTS = [j for j in ts.JOINTS if j != "gripper"]   # the gripper's own patch guards it
DEFAULT_DATASET = "faithqin/so101-tube-insert-v4"
ACTION_NAMES = [f"{j}.pos" for j in ts.JOINTS]
MAX_START_GAP_DEG = 15.0


class ReplayTelemetryError(RuntimeError):
    """The replay cannot produce a baseline the trials can be scored against."""


class Stalled(ReplayTelemetryError):
    """A joint pushed hard without moving. The episode was aborted and the arm relieved."""


# ---------------------------------------------------------------------------
# pure
# ---------------------------------------------------------------------------
def missing_channels(obs) -> list[str]:
    """Channel names the observation does not carry. Empty means the follower patch is live."""
    return [name for name in ts.HEADER[2:] if obs.get(name) is None]


def is_stalling(load_abs: float, moved_deg: float, load_thresh: float = STALL_LOAD,
                move_eps: float = STALL_MOVE_DEG) -> bool:
    """Pressing hard and not moving -- a jam. High load alone is normal during a fast reach."""
    return abs(load_abs) > load_thresh and abs(moved_deg) < move_eps


def start_gap_deg(obs, first_action) -> float:
    """Largest per-joint distance between where the arm IS and the episode's first commanded pose."""
    return float(max(abs(float(obs[f"{j}.pos"]) - float(first_action[k])) for k, j in enumerate(ts.JOINTS)))


def actions_for_episode(root, episode: int) -> np.ndarray:
    """The episode's demonstrated actions, (n, 6), in frame order — what `lerobot-replay` streams."""
    import pandas as pd

    files = sorted(glob.glob(str(Path(root) / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise ReplayTelemetryError(f"no parquet under {root}")
    df = pd.concat(pd.read_parquet(f, columns=["episode_index", "frame_index", "action"]) for f in files)
    d = df[df["episode_index"] == int(episode)].sort_values("frame_index")
    if d.empty:
        raise ReplayTelemetryError(f"episode {episode} is not in {root}")
    return np.stack(d["action"].to_numpy()).astype(np.float64)


def replay_loop(robot, actions, sidecar, fps: int, sleep=None, clock=None,
                max_start_gap_deg: float | None = None, log=print, stall_guard: bool = True) -> int:
    """Stream `actions` to `robot` at `fps`, teeing every observation through `sidecar`.

    One row per action, and the row holds the observation taken BEFORE that tick's action — the
    same pairing a recorded frame has, so the baseline is not offset by a tick against the trials.
    """
    sleep = sleep or time.sleep
    clock = clock or time.perf_counter
    period = 1.0 / float(fps)
    n = 0
    prev_pos: dict = {}
    stalls: dict = {}
    deadline = None
    for i, a in enumerate(actions):
        t_start = clock()
        if deadline is None:
            deadline = t_start
        obs = robot.get_observation()
        if i == 0:
            gone = missing_channels(obs)
            if gone:
                raise ReplayTelemetryError(
                    f"the observation carries no {gone[0].split('.')[-1]} channels ({len(gone)} of 18 missing, "
                    f"e.g. {gone[:3]}) — the so_follower load/current patch is not applied to this "
                    f"environment, so the sidecar would write NaN columns and the contact endpoint "
                    f"would have no baseline. Re-apply tools/so_follower_load.patch (the project notes)."
                )
            if max_start_gap_deg is not None:
                gap = start_gap_deg(obs, a)
                if gap > max_start_gap_deg:
                    raise ReplayTelemetryError(
                        f"the arm is {gap:.1f} deg from the episode's first pose (limit "
                        f"{max_start_gap_deg:.1f}) — home it first with `python tools/home_arm.py "
                        f"--pose v4`; closing a gap this large in one command sweeps the arm through "
                        f"the workspace"
                    )
        if stall_guard:
            for j in GUARDED_JOINTS:
                pos, load = obs.get(f"{j}.pos"), obs.get(f"{j}.load")
                if pos is None or load is None:
                    continue
                moved = float(pos) - prev_pos.get(j, float(pos))
                prev_pos[j] = float(pos)
                stalls[j] = stalls.get(j, 0) + 1 if is_stalling(float(load), moved) else 0
                if stalls[j] >= STALL_TICKS:
                    for k2 in GUARDED_JOINTS:            # stop pressing before raising
                        here = obs.get(f"{k2}.pos")
                        if here is not None:
                            robot.send_action({f"{k2}.pos": float(here)})
                    raise Stalled(
                        f"{j} pushed at |load| {abs(float(load)):.0f} (> {STALL_LOAD:.0f}) without "
                        f"moving for {STALL_TICKS} ticks at frame {i} of {len(actions)} — a JAM. "
                        f"Episode aborted and the arm relieved, per the protocol's chamferless stop "
                        f"rule. Clear the obstruction before running anything else."
                    )
        sidecar.observe(obs)
        robot.send_action({name: float(a[k]) for k, name in enumerate(ACTION_NAMES)})
        sidecar.frame_recorded()
        n += 1
        if n % (fps * 5) == 0:
            log(f"  {n}/{len(actions)} frames ({n / fps:.0f}s)")
        # ABSOLUTE DEADLINE, not "period minus the work I just did" (Sep 7 2026). `time.sleep`
        # sleeps AT LEAST what it is asked for; the old form restarted the clock every tick, so
        # each tick's scheduling overshoot was kept and the replay ran slow. Measured on the five
        # replays this tool has written: 18.523 / 18.626 / 18.601 / 18.568 / 18.546 Hz against the
        # dataset's 20 fps. Those replays are the mu/sigma BASELINE for the contact endpoint, so
        # the drift is not cosmetic. The deadline advances by exactly one period per frame and
        # absorbs an overshoot on the next sleep; a tick that overruns a whole period re-bases
        # instead, so a late replay never sprints to catch up.
        deadline += period
        late = clock()
        if deadline < late:
            deadline = late
        sleep(max(0.0, deadline - late))
    return n


def verify_baseline(paths) -> "ce.Baseline":
    """Hand the CSV to its real consumer. Raises `ce.TelemetryError` on anything it would reject."""
    return ce.Baseline.from_csvs([Path(p) for p in paths])


# ---------------------------------------------------------------------------
# hardware — main() only; nothing above opens a bus
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--root", default=None, help="dataset root (default: the v4 dataset in the HF cache)")
    ap.add_argument("--port", default=FOLLOWER_PORT)
    ap.add_argument("--fps", type=int, default=None, help="default: the dataset's own fps")
    ap.add_argument("--out", default=None, help="telemetry CSV (default: tools/scored_logs/CERT-<stamp>_telemetry.csv)")
    ap.add_argument("--max-start-gap", type=float, default=MAX_START_GAP_DEG)
    a = ap.parse_args(argv)

    import json

    import v4_holdout_loss as hl

    root = Path(a.root) if a.root else hl.resolve_local_root(DEFAULT_DATASET)
    if root is None or not Path(root).exists():
        print(f"ABORT: {DEFAULT_DATASET} is not on this machine", file=sys.stderr)
        return 2
    fps = a.fps or int(json.load(open(Path(root) / "meta" / "info.json"))["fps"])
    actions = actions_for_episode(root, a.episode)
    out = Path(a.out) if a.out else (Path(__file__).resolve().parent / "scored_logs" /
                                     f"CERT-{time.strftime('%Y%m%d_%H%M%S')}_telemetry.csv")
    print(f"replay: {Path(root).name} episode {a.episode}, {len(actions)} frames at {fps} fps "
          f"({len(actions) / fps:.0f}s)\n  telemetry -> {out}")

    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.robots.so_follower.so_follower import SOFollower

    robot = SOFollower(SO101FollowerConfig(port=a.port, id="follower", disable_torque_on_disconnect=False))
    sidecar = ts.Sidecar(str(out))
    robot.connect()
    try:
        n = replay_loop(robot, actions, sidecar, fps=fps, max_start_gap_deg=a.max_start_gap)
    except Stalled as exc:
        sidecar.close()
        robot.disconnect()
        print(f"REPLAY ABORTED (STALL): {exc}", file=sys.stderr)
        return 5
    except ReplayTelemetryError as exc:
        sidecar.close()
        robot.disconnect()
        print(f"REPLAY REFUSED: {exc}", file=sys.stderr)
        return 3
    finally:
        sidecar.close()
        if robot.is_connected:
            robot.disconnect()

    print(f"replayed {n} frames -> {out}")
    try:
        base = verify_baseline([out])
    except ce.TelemetryError as exc:
        print(f"BASELINE REJECTED: {exc}\n"
              f"  The CSV was written but cannot be used as a contact baseline. Fix this before the "
              f"first scored trial — it is the telemetry gate.", file=sys.stderr)
        return 4
    names = ce.state_columns()
    sigma = "  ".join(f"{names[k].split('.')[0][:5]} {base.std[k]:.1f}" for k in range(6, 12))
    print(f"BASELINE OK: {base.n_rows} rows; load sigma  {sigma}")
    print(f"  use it with:  python tools/v4_contact_events.py --telemetry <trial>_telemetry.csv --baseline {out} ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
