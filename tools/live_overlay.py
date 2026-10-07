#!/usr/bin/env python
"""Live registration view: the recorded target frame ghosted over the camera.

Run from a TCC-granted Terminal (camera). Opens the FRONT camera at the same
index and mode the trial runner uses, blends each frame 50/50 with
analysis/register_target.png (written by pose_register.py --capture-target),
and shows the result in a window. Pose the torque-off follower until the two
silhouettes coincide, then run `pose_register.py --read` (bus only, no camera
contention). Press q in the window to quit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose_register import blend  # noqa: E402

FRONT_INDEX = 1
WIDTH, HEIGHT = 1280, 720
TARGET = "analysis/register_target.png"


def fit_to(frame: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if frame.shape[0] == h and frame.shape[1] == w:
        return frame
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)


def main() -> int:
    target_bgr = cv2.imread(TARGET)
    if target_bgr is None:
        print(f"no target at {TARGET}; run pose_register.py --capture-target EP TICK first")
        return 2
    cap = cv2.VideoCapture(FRONT_INDEX)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print("front camera failed to open (index 1) — is another process holding it?")
        return 3
    cv2.namedWindow("LIVE REGISTRATION  (ghost = recorded)  q to quit", cv2.WINDOW_NORMAL)
    import time
    last_load = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        # reload the target ~1x/s so pose_register.py --capture-target can
        # switch the ghost without restarting this window
        if time.time() - last_load > 1.0:
            t = cv2.imread(TARGET)
            if t is not None:
                target_bgr = t
            last_load = time.time()
        live = fit_to(frame, target_bgr.shape[:2])
        out = blend(target_bgr, live, 0.5)
        cv2.imshow("LIVE REGISTRATION  (ghost = recorded)  q to quit", out)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
