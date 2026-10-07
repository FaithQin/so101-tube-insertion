"""Which camera is on which index, and is it actually producing an image? Read-only.

Faster and more informative than `lerobot-find-cameras opencv`, which walks 60
indices and tells you nothing about whether the frames are black.

For each index it reports resolution, fps, and mean pixel value, saves a PNG so
you can identify the camera by looking at it, and tests whether width/height/fps
requests actually take effect on this platform.

No writes to any camera setting that persists. Nothing touches the arms.

    conda activate lerobot
    python probe_cameras.py

macOS notes that cost an evening on Aug 8, 2026:
  - Camera access is a per-app permission. Grant it to Terminal (or iTerm, or
    Code) under System Settings > Privacy & Security > Camera, then FULLY QUIT
    and relaunch that app — TCC is evaluated at process launch, so a new tab
    keeps the old denial. Without it every index fails and lerobot reports
    "No cameras were detected", which reads exactly like a USB fault.
  - Disconnect the iPhone first. Continuity Camera occupies an index and shifts
    every camera after it.
  - AVFOUNDATION reports CAP_PROP_FOURCC as 0, so the FOURCC column is blank
    here and garbage in lerobot's own scan. It is not a diagnostic on this Mac.
"""

from pathlib import Path

import cv2

MAX_INDEX = 8
WARMUP_FRAMES = 20  # let auto-exposure converge before measuring anything
OUT_DIR = Path(__file__).parent / "camera_probe_frames"


def fourcc_str(code: int) -> str:
    text = "".join([chr((int(code) >> 8 * i) & 0xFF) for i in range(4)])
    return text if text.strip("\x00") else "(unreported)"


OUT_DIR.mkdir(exist_ok=True)
print(f"saving frames to {OUT_DIR}\n")

found = 0
for idx in range(MAX_INDEX):
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap.release()
        continue

    found += 1
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))

    # WARMUP. Auto-exposure has not converged on the first frame after opening,
    # so reading immediately reports a brightness the camera never actually
    # settles at. lerobot waits warmup_s=1 for the same reason
    # (cameras/opencv/camera_opencv.py:180). Without this the mean is fiction —
    # it read 248 here while the settled value was 171.
    ok, frame = False, None
    for _ in range(WARMUP_FRAMES):
        ok, frame = cap.read()
    if ok and frame is not None:
        mean = float(frame.mean())
        note = f"mean pixel {mean:5.1f}"
        if mean < 10.0:
            note += "  <-- BLACK: lens cap, no light, or pointed at nothing"
        cv2.imwrite(str(OUT_DIR / f"index_{idx}.png"), frame)
    else:
        note = "READ FAILED (opens but yields no frame)"

    print(f"index {idx}: {width}x{height} @ {fps:.0f}fps  fourcc={fourcc}  {note}")

    # Does this platform honour requested settings, or silently ignore them?
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    got_fps = cap.get(cv2.CAP_PROP_FPS)
    applied = "APPLIED" if (got_w, got_h) == (640, 480) else "IGNORED"
    print(f"          requested 640x480@30 MJPG -> got {got_w}x{got_h}@{got_fps:.0f}  [{applied}]")

    cap.release()

print(f"\n{found} camera(s) responded. Open {OUT_DIR} to see which is which.")
print("640x480 is the 300K (wrist). ~1920x1080 is the 2MP (fixed) — or the")
print("MacBook's built-in, so confirm by looking at the saved frames.")

# Which modes does each camera ACTUALLY support, and what do they cost on the bus?
# A camera silently falls back to a mode it supports, so "I asked for 320x240"
# is not the same as "it is running at 320x240".
MODES = [(320, 240), (640, 480), (800, 600), (1024, 768), (1280, 720), (1920, 1080)]
USB2_CEILING_MB = 40.0

print("\n=== supported modes and raw bandwidth cost ===")
print("(raw = uncompressed YUYV. A mode whose raw cost exceeds the bus but still")
print(" delivers frames is proof the camera switched to compressed MJPG.)\n")

for idx in range(MAX_INDEX):
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap.release()
        continue

    print(f"camera {idx}:")
    for req_w, req_h in MODES:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
        got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        got_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        ok = False
        for _ in range(5):  # a mode switch also needs a few frames to take effect
            ok, _ = cap.read()

        raw_mb = got_w * got_h * 2 * got_fps / 1e6
        exact = (got_w, got_h) == (req_w, req_h)
        verdict = "supported " if exact else f"-> fell back to {got_w}x{got_h}"
        stream = "OK    " if ok else "NO FRAME"
        hint = "  MUST be MJPG (raw exceeds the bus)" if ok and raw_mb > USB2_CEILING_MB else ""
        print(f"  {req_w:>4}x{req_h:<4} {verdict:<26} {stream}  {got_fps:>4.0f}fps  raw {raw_mb:5.1f} MB/s{hint}")

    cap.release()

print(f"\nRaw-bandwidth budget: ~{USB2_CEILING_MB:.0f} MB/s per USB 2.0 controller.")
print("A high mode that streams fine is proof that camera is compressing (MJPG),")
print("so compressed high-res can cost LESS bus than uncompressed low-res.")
print()
print("⚠️  BUT compression does NOT let both kit cameras share one controller.")
print("   Tested Aug 8, 2026: they fail together in lerobot at EVERY resolution")
print("   combination, and halving the wrist's bandwidth changed nothing — which")
print("   rules out throughput as the mechanism. Pairing one with the MacBook's")
print("   camera (a different controller) works. The fix is a USB-C→USB-A adapter")
print("   to split them, not a bitrate you can tune.")
