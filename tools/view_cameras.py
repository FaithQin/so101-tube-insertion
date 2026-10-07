"""Live camera view for aiming and framing. Read-only, touches no arm.

Opens every camera it finds in its own window at the resolution you will
actually record at, and overlays what you need while pointing the thing:

  - index, resolution, and live mean pixel value
  - an EXPOSURE warning when the frame is blown out or crushed
  - rule-of-thirds guides, so the grasp point lands mid-frame not at an edge

Use it to aim the lamp-stand camera at all 4 taped grid positions and the wrist
camera down into the gripper's working volume.

    conda activate lerobot
    python view_cameras.py                        # all cameras, 640x480
    python view_cameras.py 0 1                    # only these indices
    python view_cameras.py 1:1280x720 0:640x480   # per-camera resolution

Per-camera sizing matters here. Measured Aug 8, 2026: the 300K only supports
640x480 (18.4 MB/s raw, no lower mode exists), and the 2MP switches to
compressed MJPG at 1280x720 and above. Both at 640x480 raw is ~37 MB/s and
starves the bus; 2MP at 1280x720 plus 300K at 640x480 fits comfortably AND
gives the fixed view more resolution. Run probe_cameras.py for the mode table.

Keys, with a camera window focused:
    q or ESC   quit
    s          save a snapshot of every window to camera_probe_frames/

Runs at the recording resolution on purpose. Framing judged at 1080p and then
recorded at 640x480 is not the same framing.
"""

import argparse
from pathlib import Path

import cv2

OUT_DIR = Path(__file__).parent / "camera_probe_frames"
MAX_INDEX = 8

parser = argparse.ArgumentParser()
parser.add_argument("specs", nargs="*", help="camera specs: N or N:WxH (e.g. 1:1280x720 0:640x480)")
parser.add_argument("--width", type=int, default=640, help="default width for bare indices")
parser.add_argument("--height", type=int, default=480, help="default height for bare indices")
parser.add_argument("--fps", type=int, default=30)
args = parser.parse_args()


def parse_spec(spec: str) -> tuple[int, tuple[int, int]]:
    if ":" in spec:
        idx, size = spec.split(":", 1)
        w, h = size.lower().split("x")
        return int(idx), (int(w), int(h))
    return int(spec), (args.width, args.height)


def open_camera(idx: int, size: tuple[int, int]):
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    return cap


if args.specs:
    targets = dict(parse_spec(s) for s in args.specs)
else:
    targets = {i: (args.width, args.height) for i in range(MAX_INDEX)}
cams = {}
starved = []
for idx, size in targets.items():
    cap = open_camera(idx, size)
    if cap is None:
        continue
    # Opening is not the same as streaming. Prove it yields a frame before
    # claiming it works — two cameras on one USB 2.0 bus can open fine and
    # then starve, which looks exactly like a dead camera.
    got = any(cap.read()[0] for _ in range(10))
    if not got:
        starved.append(idx)
        cap.release()
        continue
    cams[idx] = cap
    cv2.namedWindow(f"camera {idx}", cv2.WINDOW_NORMAL)

if starved:
    print(
        f"\n!! camera(s) {starved} opened but produced NO FRAMES.\n"
        "   If they work alone but not together, that is USB 2.0 bandwidth\n"
        "   contention, not a broken camera. Both kit cameras share one bus.\n"
        "   Try, in order:\n"
        f"     python {Path(__file__).name} {starved[0]}                     # alone, to confirm\n"
        f"     python {Path(__file__).name} 1:1280x720 0:640x480   # push the 2MP into MJPG\n"
        "   Run probe_cameras.py for the per-camera mode and bandwidth table.\n"
    )

if not cams:
    raise SystemExit(
        "No cameras opened.\n"
        "  - Grant Camera permission to your terminal app, then FULLY QUIT and relaunch it\n"
        "    (System Settings > Privacy & Security > Camera)\n"
        "  - Disconnect the iPhone; Continuity Camera shifts every index after it"
    )

print(f"live: {sorted(cams)}   q/ESC quit, s snapshot")
misses = dict.fromkeys(cams, 0)

while True:
    for idx, cap in cams.items():
        ok, frame = cap.read()
        if not ok or frame is None:
            misses[idx] += 1
            if misses[idx] in (30, 300):
                print(f"camera {idx}: {misses[idx]} dropped reads - starving on the USB bus?")
            continue
        misses[idx] = 0

        h, w = frame.shape[:2]
        mean = float(frame.mean())

        # rule-of-thirds guides
        for i in (1, 2):
            cv2.line(frame, (w * i // 3, 0), (w * i // 3, h), (0, 255, 0), 1)
            cv2.line(frame, (0, h * i // 3), (w, h * i // 3), (0, 255, 0), 1)

        if mean > 200:
            note, colour = "EXPOSURE: blown out - aim away from lights/windows", (0, 0, 255)
        elif mean < 40:
            note, colour = "EXPOSURE: too dark - add light on the bench", (0, 0, 255)
        else:
            note, colour = "exposure OK", (0, 255, 0)

        cv2.putText(frame, f"idx {idx}  {w}x{h}  mean {mean:5.1f}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, note, (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
        cv2.imshow(f"camera {idx}", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):
        break
    if key == ord("s"):
        OUT_DIR.mkdir(exist_ok=True)
        for idx, cap in cams.items():
            ok, frame = cap.read()
            if ok and frame is not None:
                cv2.imwrite(str(OUT_DIR / f"view_{idx}.png"), frame)
        print(f"saved snapshots to {OUT_DIR}")

for cap in cams.values():
    cap.release()
cv2.destroyAllWindows()
