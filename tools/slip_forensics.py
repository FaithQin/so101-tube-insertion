#!/usr/bin/env python
"""Locate WHEN the horn slip happened, from frame-0 arm posture across trials.

Born Sep 1 2026, while the follower was being recalibrated after a measured
~10 deg elbow / ~7 deg wrist_flex physical slip. Every trial video begins at
the same commanded home pose, so frame 0 is a physical posture record taken at
identical commands. A horn slip moves the gripper's frame-0 image position by
many pixels; the chronology must show a STEP at the slip event.

Why it matters: the step's location decides which trials ran on a bent plant.
Everything after it is suspect; everything before it survives. That includes
the Aug 29-30 Block 2 probes (the "v3-A 0/2, v3-B 0/2" replication) and the
two counted Aug 31 B trials.

Method, and its deliberate constraints:
* Gripper tip proxy = centroid of vivid-red jaw pixels in the UPPER part of
  the frame only. The rack is also red and lives lower; without the crop it
  dominates and the metric measures nothing (the rack never moves).
* Red is chroma-based (R dominant over G and B), never brightness, so the
  day/night lighting differences the v2 audit documented cannot read as motion.
* v2-homed and v3-homed trials are separate groups — their commanded homes
  differ by design, so pooling would manufacture a fake step at the boundary.
* Step detection compares MEDIANS of a trailing and leading window and demands
  the jump be sustained: one noisy detection is not a slip event.

Usage (offline; reads videos on disk, touches no hardware):
    python tools/slip_forensics.py --json-out analysis/slip_timeline.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

ROLLOUT_GLOB = os.path.expanduser("~/.cache/huggingface/lerobot/faithqin/rollout_*")
FRONT_VIDEO = "videos/observation.images.front/chunk-000/file-000.mp4"

# Vivid-red chroma rule: red channel dominant by margin over both green and
# blue, and not a washed-out highlight. Tuned for the jaw tips; validated by
# eyeballing extracted frames (see analysis/slip_frames/).
RED_MARGIN = 60
RED_MIN = 120

# A slip of ~10 deg at the elbow moves the gripper tens of px at this camera
# distance; homing repeatability is a few px. 15 px separates them safely.
STEP_MIN_JUMP_PX = 15.0
STEP_MIN_RUN = 3


def red_centroid(frame: np.ndarray, top_frac: float = 0.55):
    """(cx, cy, n_pixels) of vivid-red pixels in the top `top_frac` of frame.

    None when no meaningful red cluster exists there. Chroma-based on purpose:
    a bright grey highlight has R high but not DOMINANT, and must not match.
    """
    h = frame.shape[0]
    crop = frame[: int(h * top_frac)].astype(int)
    r, g, b = crop[..., 0], crop[..., 1], crop[..., 2]
    mask = (r > RED_MIN) & (r - g > RED_MARGIN) & (r - b > RED_MARGIN)
    n = int(mask.sum())
    if n < 30:
        return None
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean()), n


def find_step(values: list[float], min_jump: float, min_run: int):
    """Index where a sustained level change begins, else None.

    Compares the median of the `min_run` values before each candidate index to
    the median of the `min_run` values from it. Medians, so a single outlier
    frame cannot fake or hide a jump; sustained, so noise cannot trigger.
    """
    v = np.asarray(values, dtype=float)
    for k in range(min_run, len(v) - min_run + 1):
        before = float(np.median(v[k - min_run : k]))
        after = float(np.median(v[k : k + min_run]))
        if abs(after - before) >= min_jump:
            # The median flips as soon as a majority of the leading window has
            # shifted — one index early for a run-of-3. Refine to the first
            # value actually on the far side, which is the true onset.
            while k < len(v) and abs(v[k] - before) < min_jump / 2:
                k += 1
            return k
    return None


def group_key(name: str) -> str:
    """v2-homed vs v3-homed trials — different commanded homes by design.

    v2 group: every rollout with the `_v2_` tag (Aug 25 trials, lock probes).
    v3 group: the B2* families (Block 2 probes, B2R/B2V re-runs, Block 2B).
    """
    return "v2" if re.search(r"_v2_", name) else "v3"


def stamp_of(name: str) -> str:
    m = re.search(r"(\d{8}_\d{6})", name)
    return m.group(1) if m else name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--frames-dir", default="analysis/slip_frames",
                    help="save each frame-0 PNG here for eyeball validation")
    args = ap.parse_args()

    import av

    rows = []
    for d in sorted(glob.glob(ROLLOUT_GLOB)):
        v = os.path.join(d, FRONT_VIDEO)
        if not os.path.isfile(v):
            continue
        name = os.path.basename(d)
        try:
            c = av.open(v)
            frame = next(c.decode(c.streams.video[0])).to_ndarray(format="rgb24")
            c.close()
        except Exception as e:
            print(f"[slip] {name}: unreadable ({type(e).__name__})", flush=True)
            continue
        cent = red_centroid(frame)
        rows.append({
            "name": name, "stamp": stamp_of(name), "group": group_key(name),
            "cx": None if cent is None else round(cent[0], 1),
            "cy": None if cent is None else round(cent[1], 1),
            "n_red": 0 if cent is None else cent[2],
        })
        if args.frames_dir:
            os.makedirs(args.frames_dir, exist_ok=True)
            try:
                from PIL import Image
                Image.fromarray(frame).save(os.path.join(args.frames_dir, f"{name}.png"))
            except Exception:
                pass

    rows.sort(key=lambda r: r["stamp"])
    out = {"rows": rows, "steps": {}}
    for grp in ("v2", "v3"):
        g = [r for r in rows if r["group"] == grp and r["cx"] is not None]
        print(f"\n=== group {grp} ({len(g)} trials with a detected gripper) ===")
        for r in g:
            print(f"  {r['stamp']}  {r['name']:<46} tip=({r['cx']:.0f},{r['cy']:.0f}) n={r['n_red']}")
        for axis in ("cx", "cy"):
            k = find_step([r[axis] for r in g], STEP_MIN_JUMP_PX, STEP_MIN_RUN)
            if k is not None:
                out["steps"][f"{grp}_{axis}"] = {
                    "first_shifted_trial": g[k]["name"],
                    "before_median": float(np.median([r[axis] for r in g[max(0, k - STEP_MIN_RUN):k]])),
                    "after_median": float(np.median([r[axis] for r in g[k:k + STEP_MIN_RUN]])),
                }
                print(f"  STEP in {axis}: begins at {g[k]['name']}")
        if not any(key.startswith(grp) for key in out["steps"]):
            print("  no sustained step found in this group")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[slip] wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
