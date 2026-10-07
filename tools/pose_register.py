#!/usr/bin/env python
"""Register the follower against the DATASET's own frames, by eye.

Sep 2 2026. A day was lost measuring the follower against the leader arm and
patching calibration on that basis; the leader comparison had no pre-fault
baseline, so its deltas were never evidence of change, and the patches left the
arm unable to touch the tube. The only frame that matters is the one the
episodes were recorded in.

Procedure (per pose):
  1. `--capture-target EP TICK` extracts the recorded frame and its
     observation.state from the dataset.
  2. `--overlay` blends that frame over a fresh live capture and writes it to
     analysis/register_overlay.png. The operator poses the TORQUE-OFF follower
     until its silhouette sits on the recorded arm, re-running --overlay to check.
  3. `--read` reads the follower's reported state and prints dataset-minus-
     reported per joint: the correction each joint's reading needs.
  4. Repeat at 2-3 poses. `--verdict` on the saved readings says whether the
     offsets are CONSTANT (a calibration fix) or POSE_DEPENDENT (a servo/gear
     problem no calibration fixes).

Camera capture goes through the Terminal bridge (TCC); this script only reads
the PNG the bridge wrote. Pure logic is unit-tested in tests/test_pose_register.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
DATASET = "faithqin/so101-tube-insert-v3_20260827_110242"
LIVE_PNG = "tools/camera_probe_frames/scene_check_live.png"
STATE_FILE = "analysis/register_state.json"


def joint_offsets(recorded: dict, reported: dict) -> dict:
    """dataset - reported, per joint. Positive = the follower must report HIGHER."""
    return {j: float(recorded[j]) - float(reported[j]) for j in recorded if j in reported}


def consistency_verdict(per_pose: list[dict], tol_deg: float) -> dict:
    """CALIBRATION_CONSTANT if every joint's offset agrees across poses within
    tol_deg of its median; POSE_DEPENDENT otherwise — the discriminator between
    a fixable calibration constant and a servo/gear fault."""
    if len(per_pose) < 2:
        raise ValueError("consistency needs at least two poses")
    joints = [j for j in per_pose[0] if all(j in p for p in per_pose)]
    median = {j: float(np.median([p[j] for p in per_pose])) for j in joints}
    bad = [j for j in joints if max(abs(p[j] - median[j]) for p in per_pose) > tol_deg]
    return {
        "status": "POSE_DEPENDENT" if bad else "CALIBRATION_CONSTANT",
        "median": median,
        "inconsistent_joints": bad,
        "n_poses": len(per_pose),
    }


def blend(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """alpha*a + (1-alpha)*b, both visible. alpha in (0,1) exclusive."""
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be strictly between 0 and 1 so both images show")
    out = alpha * a.astype(float) + (1.0 - alpha) * b.astype(float)
    return np.clip(out, 0, 255).astype(np.uint8)


def _load_state():
    return json.load(open(STATE_FILE)) if os.path.exists(STATE_FILE) else {"targets": {}, "readings": []}


def _save_state(s):
    os.makedirs("analysis", exist_ok=True)
    json.dump(s, open(STATE_FILE, "w"), indent=2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture-target", nargs=2, type=int, metavar=("EP", "TICK"))
    ap.add_argument("--overlay", action="store_true", help="blend target over the latest live capture")
    ap.add_argument("--read", action="store_true", help="read follower state, record offsets vs target")
    ap.add_argument("--verdict", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()
    st = _load_state()

    if args.capture_target:
        import av
        from PIL import Image
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ep, tick = args.capture_target
        ds = LeRobotDataset(DATASET, episodes=[ep])
        lo = ds.meta.episodes[ep]["dataset_from_index"]
        state = {j: float(v) for j, v in zip(JOINTS, ds[lo + tick]["observation.state"][:6])}
        v = os.path.expanduser(f"~/.cache/huggingface/lerobot/{DATASET}/videos/observation.images.front/chunk-000/file-000.mp4")
        c = av.open(v); frame = None
        for i, fr in enumerate(c.decode(c.streams.video[0])):
            if i == tick:
                frame = fr.to_ndarray(format="rgb24"); break
        c.close()
        os.makedirs("analysis", exist_ok=True)
        Image.fromarray(frame).save("analysis/register_target.png")
        st["targets"]["current"] = {"ep": ep, "tick": tick, "state": state}
        _save_state(st)
        print(f"target ep{ep} tick{tick}: {json.dumps({k: round(v,1) for k,v in state.items()})}")
        print("wrote analysis/register_target.png")

    if args.overlay:
        from PIL import Image
        t = np.asarray(Image.open("analysis/register_target.png").convert("RGB"))
        live = np.asarray(Image.open(LIVE_PNG).convert("RGB").resize((t.shape[1], t.shape[0])))
        out = blend(t, live, args.alpha)
        Image.fromarray(out).save("analysis/register_overlay.png")
        print("wrote analysis/register_overlay.png  (recorded arm + live arm, 50/50)")

    if args.read:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _arms import FOLLOWER_CAL, FOLLOWER_PORT, open_bus
        bus = open_bus(FOLLOWER_PORT, FOLLOWER_CAL)
        reported = {j: float(bus.read("Present_Position", j, normalize=True)) for j in JOINTS}
        bus.disconnect(disable_torque=False)
        tgt = st["targets"]["current"]
        off = joint_offsets(tgt["state"], reported)
        st["readings"].append({"ep": tgt["ep"], "tick": tgt["tick"], "reported": reported, "offset": off})
        _save_state(st)
        print(f"pose ep{tgt['ep']} tick{tgt['tick']}  (dataset - reported):")
        for j in JOINTS:
            print(f"  {j:<14} recorded {tgt['state'][j]:>8.2f}  reported {reported[j]:>8.2f}  offset {off[j]:>+7.2f}")

    if args.verdict:
        per_pose = [r["offset"] for r in st["readings"]]
        v = consistency_verdict(per_pose, tol_deg=1.5)
        print(json.dumps(v, indent=2))
        return 0 if v["status"] == "CALIBRATION_CONSTANT" else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
