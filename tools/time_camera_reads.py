#!/usr/bin/env python
"""Measure what frame rate each camera ACTUALLY delivers at the rollout config.

READ-ONLY diagnostic, no robot needed. Run in a TCC-granted terminal (Faith's).

Why this exists: lerobot's `async_read` blocks until a FRESH frame arrives
(camera_opencv.py:566, `new_frame_event.wait`), so the record/rollout loop can
never run faster than the slowest camera's real delivery rate. A 30 fps loop
warning of ~12.4 Hz means a camera is delivering ~12.4 fps — typically
auto-exposure stretching shutter time in dim light, or a bad format
negotiation at 720p — NOT policy or servo-bus cost.

Usage:
    python tools/time_camera_reads.py                # rollout config: front=1 720p, wrist=0 480p
    python tools/time_camera_reads.py --front 2      # override indices after a probe
"""

import argparse
import statistics
import time

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

parser = argparse.ArgumentParser()
parser.add_argument("--front", type=int, default=1)
parser.add_argument("--wrist", type=int, default=0)
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument(
    "--auto-fourcc",
    action="store_true",
    help="let AVFoundation pick the pixel format (lerobot's default) instead of forcing MJPG. "
    "Uncompressed 720p@30 needs 55 MB/s — more than USB 2.0 carries — so auto-negotiation "
    "landing on YUYV caps the front camera at ~20 fps. Forcing MJPG is the fix under test.",
)
parser.add_argument(
    "--watch",
    action="store_true",
    help="continuously print each camera's delivered fps + brightness every 2 s (Ctrl-C to stop). "
    "Use to test the auto-exposure theory: shine a light on the scene and watch whether fps rises — "
    "AE holds image brightness constant by stretching shutter time, so a dim-ish scene gives a "
    "healthy-looking image at 20 fps and more light buys the fps back.",
)
parser.add_argument(
    "--only",
    choices=["front", "wrist"],
    default=None,
    help="test a single camera (for isolation experiments with the other unplugged)",
)
args = parser.parse_args()

fourcc = None if args.auto_fourcc else "MJPG"
print(f"pixel format: {'auto-negotiated' if fourcc is None else fourcc}")
CAMS = {
    "front": OpenCVCameraConfig(index_or_path=args.front, width=1280, height=720, fps=30, fourcc=fourcc),
    "wrist": OpenCVCameraConfig(index_or_path=args.wrist, width=640, height=480, fps=30, fourcc=fourcc),
}
if args.only:
    CAMS = {args.only: CAMS[args.only]}

cams = {}
for name, cfg in CAMS.items():
    cam = OpenCVCamera(cfg)
    cam.connect()
    cams[name] = cam
    print(f"{name}: connected index={cfg.index_or_path} requested {cfg.width}x{cfg.height}@{cfg.fps}")

time.sleep(1.0)  # let auto-exposure settle and read threads spin up

if args.watch:
    print("watching delivered fps (Ctrl-C to stop) — now change the lighting and look for movement:")
    try:
        while True:
            counts = {name: 0 for name in cams}
            bright = {}
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 2.0:
                for name, cam in cams.items():
                    try:
                        frame = cam.async_read(timeout_ms=200)
                        counts[name] += 1
                        bright[name] = frame.mean()
                    except TimeoutError:
                        pass
            dt = time.perf_counter() - t0
            print(
                "   ".join(
                    f"{n}: {counts[n]/dt:5.1f} fps (bright {bright.get(n, float('nan')):5.0f})"
                    for n in cams
                ),
                flush=True,
            )
    except KeyboardInterrupt:
        pass
    finally:
        for cam in cams.values():
            cam.disconnect()
    raise SystemExit(0)

try:
    # --- Per-camera solo delivery rate (waits for a fresh frame each read) ---
    for name, cam in cams.items():
        waits = []
        cam.async_read(timeout_ms=1000)  # sync to a frame boundary
        t_start = time.perf_counter()
        n = 0
        while time.perf_counter() - t_start < args.seconds:
            t0 = time.perf_counter()
            frame = cam.async_read(timeout_ms=1000)
            waits.append((time.perf_counter() - t0) * 1e3)
            n += 1
        fps = n / (time.perf_counter() - t_start)
        bright = float(frame.mean())
        print(
            f"{name}: delivered {fps:5.1f} fps | fresh-frame wait "
            f"mean {statistics.mean(waits):5.1f} ms  p95 {statistics.quantiles(waits, n=20)[-1]:5.1f} ms  "
            f"max {max(waits):5.1f} ms | last-frame brightness mean {bright:.0f} "
            f"({'DIM — suspect auto-exposure' if bright < 60 else 'ok'})"
        )

    # --- Combined tick, same order the robot reads them ---
    ticks = []
    for _ in range(150):
        t0 = time.perf_counter()
        for cam in cams.values():
            cam.async_read(timeout_ms=1000)
        ticks.append((time.perf_counter() - t0) * 1e3)
    mean_ms = statistics.mean(ticks)
    print(
        f"\ncombined both-camera tick: mean {mean_ms:.1f} ms  p95 "
        f"{statistics.quantiles(ticks, n=20)[-1]:.1f} ms  -> loop ceiling ≈ {1000 / mean_ms:.1f} Hz"
    )
    print("(add ~4 ms for bus reads + engine; measured 3.0 + 0.7 ms on Aug 15)")
finally:
    for cam in cams.values():
        cam.disconnect()
