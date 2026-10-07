"""⚠️ DO NOT USE ON THE ARM until tools/_arms.py normalizes in DEGREES like SOFollower.
Found Sep 3, 2026: this tool drives the servos in RANGE_M100_100 while the datasets are in
DEGREES (same zero, different scale: pan 10%, lift 5%, roll 44%). Its Sep 2 runs are void.

Replay one recorded episode over the bus with a per-joint command offset.

Production path: Goal_Position via sync_write, paced at the dataset fps, on the
follower's live calibration — the same path lerobot-replay and the rollout
wrapper use. Writes Goal_Position ONLY. Never touches EEPROM. Torque is left
ON at the end (the arm holds; run tools/home_arm.py afterwards as usual).

Why (Sep 2, 2026): the elbow lands a constant +3.07 deg above its commanded
goal at two poses (analysis/anchor_results.json). This replays v3 ep0 with the
elbow command shifted by -3.07 so that, if the fault is the elbow's settle
behaviour, the arm lands where the demonstration did.

    python tools/home_arm.py --pose v3
    python tools/replay_offset.py --dataset faithqin/so101-tube-insert-v3_20260827_110242 \
        --episode 0 --offset elbow_flex=-3.07
    # scene capture BEFORE any reset (camera via the Terminal bridge), then the
    # 0-offset control on the same power cycle:
    python tools/replay_offset.py --dataset ... --episode 0

12V ON BEFORE USB. Port to itself. Never scroll the terminal it runs in.
"""

import argparse
import csv
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _arms import FOLLOWER_CAL, FOLLOWER_PORT, JOINTS, open_bus  # noqa: E402

CACHE = Path("~/.cache/huggingface/lerobot").expanduser()


def parse_offsets(specs):
    """['elbow_flex=-3.07', ...] -> {'elbow_flex': -3.07}. Unknown joint -> ValueError."""
    out = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"offset must look like joint=value, got {spec!r}")
        joint, value = spec.split("=", 1)
        joint = joint.strip()
        if joint not in JOINTS:
            raise ValueError(f"unknown joint {joint!r}; choose from {JOINTS}")
        out[joint] = float(value)
    return out


def apply_offsets(actions, offsets):
    """Add each offset (normalized joint units, the dataset's own) to its column."""
    out = np.array(actions, dtype=float, copy=True)
    for joint, value in offsets.items():
        out[:, JOINTS.index(joint)] += value
    return out


def load_episode(repo_id, episode):
    root = CACHE / repo_id
    info = json.loads((root / "meta" / "info.json").read_text())
    names = info["features"]["action"]["names"]
    expected = [f"{j}.pos" for j in JOINTS]
    if names != expected:
        raise RuntimeError(f"action layout {names} != {expected}")
    files = sorted(glob.glob(str(root / "data" / "chunk-*" / "file-*.parquet")))
    df = pd.concat(pd.read_parquet(f, columns=["action", "episode_index"]) for f in files)
    g = df[df.episode_index == episode]
    if g.empty:
        raise RuntimeError(f"episode {episode} not found in {repo_id}")
    return np.stack(g["action"].to_numpy()).astype(np.float64), int(info["fps"])


RECORD_HEADER = ["tick"] + [f"cmd_{j}" for j in JOINTS] + [f"rep_{j}" for j in JOINTS]


def record_rows(commanded, reported):
    """Pair per-tick commanded and reported dicts into CSV rows (normalized units)."""
    return [[k] + [c[j] for j in JOINTS] + [r[j] for j in JOINTS]
            for k, (c, r) in enumerate(zip(commanded, reported))]


def replay(bus, actions, fps, record=False):
    """Drive the actions at fps via sync_write. With record=True, sync_read the
    reported positions every tick (the rollout's own observation path) so the
    live tracking error can be compared with the dataset's recording-time
    observation.state. Returns timing + motion summary (+ rows)."""
    start = {j: bus.read("Present_Position", j, normalize=False) for j in JOINTS}
    bus.enable_torque()
    period = 1.0 / fps
    stamps, commanded, reported = [], [], []
    t0 = time.perf_counter()
    for k, a in enumerate(actions):
        cmd = {j: float(a[i]) for i, j in enumerate(JOINTS)}
        bus.sync_write("Goal_Position", cmd)
        stamps.append(time.perf_counter())
        if record:
            commanded.append(cmd)
            reported.append({j: float(v) for j, v in bus.sync_read("Present_Position").items()})
        while time.perf_counter() - t0 < (k + 1) * period:
            time.sleep(0.001)
    time.sleep(0.5)
    end = {j: bus.read("Present_Position", j, normalize=False) for j in JOINTS}
    final_reported = {j: bus.read("Present_Position", j, normalize=True) for j in JOINTS}
    dt = np.diff(stamps)
    return dict(ticks=len(actions), wall_s=stamps[-1] - stamps[0],
                tick_ms_mean=float(dt.mean() * 1e3), tick_ms_max=float(dt.max() * 1e3),
                start_raw=start, end_raw=end, final_reported=final_reported,
                final_commanded={j: float(actions[-1][i]) for i, j in enumerate(JOINTS)},
                rows=record_rows(commanded, reported) if record else [])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True, help="repo id under the local lerobot cache")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--offset", action="append", default=[], help="joint=value, repeatable")
    p.add_argument("--dry-run", action="store_true", help="print the plan, touch nothing")
    p.add_argument("--record", default=None, help="CSV path: per-tick commanded + reported positions")
    args = p.parse_args()

    offsets = parse_offsets(args.offset)
    actions, fps = load_episode(args.dataset, args.episode)
    shifted = apply_offsets(actions, offsets)
    print(f"episode {args.episode} of {args.dataset}: {len(actions)} ticks at {fps} Hz "
          f"({len(actions) / fps:.1f} s); offsets {offsets or 'none'}")
    if args.dry_run:
        return 0

    bus = open_bus(FOLLOWER_PORT, FOLLOWER_CAL)
    try:
        summary = replay(bus, shifted, fps, record=bool(args.record))
    finally:
        bus.disconnect(disable_torque=False)
    if args.record:
        with open(args.record, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(RECORD_HEADER)
            w.writerows(summary["rows"])
        print(f"tracking record: {len(summary['rows'])} ticks -> {args.record}")

    moved = {j: int(summary["end_raw"][j]) - int(summary["start_raw"][j]) for j in JOINTS}
    print(f"replayed {summary['ticks']} ticks in {summary['wall_s']:.2f} s; "
          f"tick {summary['tick_ms_mean']:.1f} ms mean / {summary['tick_ms_max']:.1f} ms max")
    print("per-joint raw motion start->end:", moved)
    print("final commanded vs reported (normalized units):")
    for j in JOINTS:
        c, r = summary["final_commanded"][j], summary["final_reported"][j]
        print(f"  {j:14s} commanded {c:8.2f}  reported {r:8.2f}  settle error {r - c:+.2f}")
    if sum(abs(v) for v in moved.values()) < 60:
        print("REPLAY FAILED: the arm did not move")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
