#!/usr/bin/env python
"""Bypass lerobot: measure what each camera delivers through raw OpenCV and PyAV.

Run in a TCC-granted terminal. Separates three suspects the wrapper can't:
  1. the camera hardware (PyAV forces an explicit 30 fps AVFoundation session,
     and errors listing the REAL supported rates if the format can't do it),
  2. OpenCV's AVFoundation negotiation (raw cv2 matrix, several orders),
  3. lerobot's wrapper (if raw cv2 hits 30 but lerobot doesn't, it's the wrapper).

Usage:
    python tools/raw_camera_test.py --front 1 --wrist 0
"""

import argparse
import time

import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--front", type=int, default=1)
parser.add_argument("--wrist", type=int, default=0)
parser.add_argument("--seconds", type=float, default=4.0)
args = parser.parse_args()


def measure(cap, seconds):
    # warmup
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1.0:
        cap.read()
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        ok, _ = cap.read()
        if ok:
            n += 1
    return n / (time.perf_counter() - t0)


def cv2_case(idx, label, w=None, h=None, fps=None, fps_after_warmup=False):
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print(f"  [{label}] failed to open")
        return
    try:
        if w:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if fps and not fps_after_warmup:
            cap.set(cv2.CAP_PROP_FPS, fps)
        if fps_after_warmup:
            cap.read()
            cap.set(cv2.CAP_PROP_FPS, fps)
        delivered = measure(cap, args.seconds)
        got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        claim = cap.get(cv2.CAP_PROP_FPS)
        print(
            f"  [{label}] delivered {delivered:5.1f} fps   "
            f"(driver claims {claim:.0f}, actual {got_w}x{got_h})"
        )
    finally:
        cap.release()


print("=== raw OpenCV (AVFoundation), fresh session per case ===")
print(f"front (index {args.front}):")
cv2_case(args.front, "720p, fps=30 set before", 1280, 720, 30)
cv2_case(args.front, "720p, fps never set     ", 1280, 720, None)
cv2_case(args.front, "720p, fps=30 set AFTER 1st read", 1280, 720, 30, fps_after_warmup=True)
cv2_case(args.front, "480p, fps=30            ", 640, 480, 30)
cv2_case(args.front, "1080p, fps=30           ", 1920, 1080, 30)
cv2_case(args.front, "no props at all         ")
print(f"wrist (index {args.wrist}):")
cv2_case(args.wrist, "480p, fps=30            ", 640, 480, 30)
cv2_case(args.wrist, "480p, fps never set     ", 640, 480, None)
cv2_case(args.wrist, "no props at all         ")

print("\n=== PyAV / AVFoundation with an explicit 30 fps demand ===")
print("(a refusal here prints the device's REAL supported rates — that's data, not failure)")
import av  # noqa: E402
import av.logging  # noqa: E402

av.logging.set_level(av.logging.VERBOSE)

for name, idx, size in (("front", args.front, "1280x720"), ("wrist", args.wrist, "640x480")):
    for fr in ("30", "20"):
        try:
            container = av.open(
                format="avfoundation",
                file=str(idx),
                options={"framerate": fr, "video_size": size},
            )
            stream = container.streams.video[0]
            n = 0
            t0 = time.perf_counter()
            for _packet in container.demux(stream):
                n += 1
                if time.perf_counter() - t0 > args.seconds:
                    break
            dt = time.perf_counter() - t0
            print(f"  {name} @{size} framerate={fr}: delivered {n/dt:5.1f} fps")
            container.close()
        except Exception as e:  # noqa: BLE001
            print(f"  {name} @{size} framerate={fr}: REFUSED -> {e}")
