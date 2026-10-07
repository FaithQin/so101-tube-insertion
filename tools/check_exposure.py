"""Is this camera's image actually readable, and can its exposure be locked?

Two jobs, both about the one camera setting lerobot cannot control.

1. LIVE READABILITY CHECK (default). Big view of one camera with the numbers
   that decide whether a policy could learn from it:
       mean     - overall brightness. 60-180 is healthy.
       blown    - % of pixels >= 250. These carry NO information; the sensor
                  clipped. If the object sits in a blown region it is gone.
       crushed  - % of pixels <= 5. Same problem at the dark end.
       stddev   - contrast. Under ~20 means a nearly flat image.
   Put the object IN THE GRIPPER'S VIEW and apply lerobot's own test:
   could you do the task from this image alone?

2. EXPOSURE LOCK TEST (--test-exposure). `OpenCVCameraConfig` has no exposure
   field, and the ACT literature warns that auto-exposure makes brightness
   inconsistent between episodes. This probes whether the AVFoundation backend
   honours CAP_PROP_AUTO_EXPOSURE / CAP_PROP_EXPOSURE anyway. It does not trust
   the read-back — it measures whether the picture actually changes.

    conda activate lerobot
    python check_exposure.py 0
    python check_exposure.py 0 --test-exposure

Keys: q or ESC quit, s save a frame to camera_probe_frames/
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).parent / "camera_probe_frames"

parser = argparse.ArgumentParser()
parser.add_argument("index", type=int, help="camera index (from probe_cameras.py)")
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--test-exposure", action="store_true", help="probe manual exposure support")
args = parser.parse_args()

cap = cv2.VideoCapture(args.index, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    raise SystemExit(f"camera {args.index} did not open. Run probe_cameras.py for current indices.")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)


def stats(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    total = gray.size
    return {
        "mean": float(gray.mean()),
        "blown": 100.0 * int((gray >= 250).sum()) / total,
        "crushed": 100.0 * int((gray <= 5).sum()) / total,
        "stddev": float(gray.std()),
    }


def settle(n=10):
    """Auto-exposure needs a few frames to react. Return stats after it settles."""
    frame = None
    for _ in range(n):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
    return (stats(frame), frame) if frame is not None else (None, None)


if args.test_exposure:
    print(f"\n=== exposure control probe: camera {args.index} ===")
    base, _ = settle()
    if base is None:
        raise SystemExit("no frames")
    print(f"auto-exposure baseline: mean {base['mean']:.1f}  blown {base['blown']:.1f}%")

    print(f"\ncurrent CAP_PROP_AUTO_EXPOSURE = {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")
    print(f"current CAP_PROP_EXPOSURE      = {cap.get(cv2.CAP_PROP_EXPOSURE)}")

    # 0.25 is the V4L2 "manual" convention; 1 and 0 are used by other backends.
    changed = False
    for auto_val in (0.25, 1, 0):
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_val)
        print(f"\n-- AUTO_EXPOSURE set to {auto_val} (reads back {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)})")
        for exp in (-8, -6, -4, 10, 50, 200):
            cap.set(cv2.CAP_PROP_EXPOSURE, exp)
            after, _ = settle(8)
            delta = after["mean"] - base["mean"]
            mark = "  <-- IMAGE CHANGED" if abs(delta) > 8.0 else ""
            if abs(delta) > 8.0:
                changed = True
            print(f"   EXPOSURE={exp:>5} -> mean {after['mean']:6.1f}  (delta {delta:+6.1f}){mark}")

    print("\nVERDICT:", end=" ")
    if changed:
        print("manual exposure APPEARS TO WORK on this backend.")
        print("  Exposure can be locked -> worth patching into OpenCVCameraConfig,")
        print("  and worth reporting upstream (lerobot has no exposure field).")
    else:
        print("manual exposure is IGNORED by AVFoundation on this Mac.")
        print("  The only levers are physical: scene brightness, camera angle,")
        print("  and a mid-tone surface to keep the meter off the rail.")
    cap.release()
    raise SystemExit(0)


print("live readability check — put the object in the gripper's view. q quits, s saves.")
cv2.namedWindow(f"camera {args.index}", cv2.WINDOW_NORMAL)

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        continue

    s = stats(frame)
    # Clipping is REGIONAL. A large blown area only matters if it covers the
    # object. Judge by the red overlay below, not by this number alone.
    if s["blown"] > 15.0:
        verdict, colour = "large clipped area - is the OBJECT red? if not, fine", (0, 165, 255)
    elif s["stddev"] < 20.0:
        verdict, colour = "WEAK: almost no contrast in this view", (0, 165, 255)
    elif not 40 <= s["mean"] <= 200:
        verdict, colour = "MARGINAL: brightness outside the healthy band", (0, 165, 255)
    else:
        verdict, colour = "OK: could you do the task from this image?", (0, 255, 0)

    cv2.putText(frame, f"mean {s['mean']:5.1f}  blown {s['blown']:4.1f}%  crushed {s['crushed']:4.1f}%",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"stddev {s['stddev']:5.1f}", (8, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, verdict, (8, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

    # Mark clipped pixels in red so you can see WHERE the detail is being lost.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame[gray >= 250] = (0, 0, 255)

    cv2.imshow(f"camera {args.index}", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):
        break
    if key == ord("s"):
        OUT_DIR.mkdir(exist_ok=True)
        cv2.imwrite(str(OUT_DIR / f"exposure_{args.index}.png"), frame)
        print(f"saved -> {OUT_DIR / f'exposure_{args.index}.png'}")

cap.release()
cv2.destroyAllWindows()
