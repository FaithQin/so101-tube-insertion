"""Compare a replay's live tracking (from `replay_offset.py --record`) against
the recording-time baseline in the dataset, joint by joint.

The dataset holds both the command (action) and what the follower reported
(observation.state[:6]) at every tick, so it IS the pre-fault baseline for
"how far behind its command does each joint run". A joint whose live-minus-
recording error is several degrees while the others sit near zero is the
fault, located with a baseline (the discipline Sep 1 lacked).

Rest poses are excluded by the window: at rest the leader presses the follower
past its position limits (v3, all 50 episodes: elbow reports -9 deg and
wrist_flex -5 deg below the command at the final tick), which is structure,
not tracking.

    python tools/compare_tracking.py /tmp/test3_control_track.csv \
        --dataset faithqin/so101-tube-insert-v3_20260827_110242 --episode 0 --window 20 200
"""

import argparse
import csv
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CACHE = Path("~/.cache/huggingface/lerobot").expanduser()


def load_record(path):
    with open(path, newline="") as fh:
        r = csv.DictReader(fh)
        rows = list(r)
    cmd = np.array([[float(x[f"cmd_{j}"]) for j in JOINTS] for x in rows])
    rep = np.array([[float(x[f"rep_{j}"]) for j in JOINTS] for x in rows])
    return cmd, rep


def load_recording(repo_id, episode):
    root = CACHE / repo_id
    files = sorted(glob.glob(str(root / "data" / "chunk-*" / "file-*.parquet")))
    df = pd.concat(pd.read_parquet(f, columns=["action", "observation.state", "episode_index"]) for f in files)
    g = df[df.episode_index == episode]
    cmd = np.stack(g["action"].to_numpy()).astype(float)
    rep = np.stack(g["observation.state"].to_numpy()).astype(float)[:, :6]
    return cmd, rep


def compare(cmd, rec_rep, live_rep, window):
    lo, hi = window
    n = min(len(cmd), len(rec_rep), len(live_rep))
    lo, hi = max(0, lo), min(hi, n)
    if hi <= lo:
        raise ValueError(f"empty window {window} for {n} ticks")
    out = {}
    for i, j in enumerate(JOINTS):
        c, r, l = cmd[lo:hi, i], rec_rep[lo:hi, i], live_rep[lo:hi, i]
        d = l - r
        out[j] = dict(n=int(hi - lo),
                      rec_minus_cmd_mean=float(np.mean(r - c)),
                      live_minus_cmd_mean=float(np.mean(l - c)),
                      live_minus_rec_mean=float(np.mean(d)),
                      live_minus_rec_p90abs=float(np.percentile(np.abs(d), 90)),
                      live_minus_rec_max=float(d[np.argmax(np.abs(d))]))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("record")
    p.add_argument("--dataset", required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--window", type=int, nargs=2, default=(20, 200), help="ticks [lo, hi) to compare")
    a = p.parse_args()
    live_cmd, live_rep = load_record(a.record)
    rec_cmd, rec_rep = load_recording(a.dataset, a.episode)
    cmd_gap = np.abs(live_cmd[: min(len(live_cmd), len(rec_cmd))] - rec_cmd[: min(len(live_cmd), len(rec_cmd))]).max(axis=0)
    print("max |live command - recorded command| per joint (offsets show up here):",
          {j: round(float(v), 2) for j, v in zip(JOINTS, cmd_gap)})
    out = compare(rec_cmd, rec_rep, live_rep, tuple(a.window))
    print(f"\nwindow ticks {a.window}, per joint, normalized units (~degrees):")
    print(f"{'joint':14s} {'rec-cmd':>9s} {'live-cmd':>9s} {'LIVE-REC':>9s} {'p90|d|':>8s} {'max':>7s}")
    for j, s in out.items():
        print(f"{j:14s} {s['rec_minus_cmd_mean']:+9.2f} {s['live_minus_cmd_mean']:+9.2f} "
              f"{s['live_minus_rec_mean']:+9.2f} {s['live_minus_rec_p90abs']:8.2f} {s['live_minus_rec_max']:+7.2f}")
    print("\nRead LIVE-REC: how far, on average, tonight's reported joint sits from where the same joint")
    print("reported at recording time under the same commands. Near zero = tracks as it did on Aug 27.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
