"""Pre-probe scene certification: is the tube+rack where training saw them?

Captures one frame from the FRONT camera, feature-match aligns it to the
v2 training-start reference (different aspect/zoom is fine — ORB+RANSAC
homography on the background), then measures the two placement landmarks:

    blue tube cap centroid    — tube placement
    orange funnel centroid    — rack placement (joined the ritual Aug 18:
                                the rack was off by a nudge on Aug 17 and
                                replays missed right until it was matched)

Prints centroid deltas in REFERENCE pixels and writes a 50/50 blend to
camera_probe_frames/overlay_ref_vs_live.png — open it and look. Aug 18
certified baseline: cap sub-pixel, rack ~2 px. Single-digit px is fine;
tens of px means nudge the physical thing and re-run.

    conda activate lerobot
    python scene_check.py            # front camera at today's index 1
    python scene_check.py 2          # after probe_cameras.py says it moved

Camera indices shift after reboot/replug — run probe_cameras.py first if
the capture opens the wrong camera (the overlay will be garbage, not
subtly wrong, so you will know).

Read-only for the arm; touches no servo.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
REF_PATH = HERE / "v2_training_start_reference.png"
OUT_DIR = HERE / "camera_probe_frames"

# HSV ranges + search regions (reference-frame pixels), from the Aug 18
# certification session. Regions exclude the blue grid dots / clutter.
BLUE_CAP = {"lo": (100, 120, 60), "hi": (125, 255, 255), "region": (200, 250, 620, 500), "min_area": 60}
ORANGE_FUNNEL = {"lo": (8, 120, 120), "hi": (22, 255, 255), "region": (430, 420, 900, 560), "min_area": 40}

parser = argparse.ArgumentParser()
parser.add_argument("index", type=int, nargs="?", default=1, help="front camera index (probe_cameras.py)")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--warmup", type=int, default=20, help="frames to discard while auto-exposure settles")
args = parser.parse_args()

ref = cv2.imread(str(REF_PATH))
if ref is None:
    sys.exit(f"reference image missing: {REF_PATH}")

cap = cv2.VideoCapture(args.index, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    sys.exit(
        f"camera {args.index} did not open. Camera permission is a per-app TCC "
        "grant; indices shift per session — run probe_cameras.py."
    )
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
for _ in range(args.warmup):
    cap.read()
ok, live = cap.read()
cap.release()
if not ok or live is None:
    sys.exit(f"camera {args.index} opened but delivered no frame.")
OUT_DIR.mkdir(exist_ok=True)
cv2.imwrite(str(OUT_DIR / "scene_check_live.png"), live)

# Align live -> reference on background features.
orb = cv2.ORB_create(4000)
g_live, g_ref = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (live, ref))
k1, d1 = orb.detectAndCompute(g_live, None)
k2, d2 = orb.detectAndCompute(g_ref, None)
if d1 is None or d2 is None:
    sys.exit("no features — is this the right camera / is the lens covered?")
matches = sorted(
    cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2), key=lambda m: m.distance
)[:400]
src = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
if H is None:
    sys.exit("homography failed — wrong camera or scene unrecognizable.")
n_inl = int(inl.sum())
print(f"homography inliers: {n_inl}/{len(matches)}")
if n_inl < 60:
    print("⚠️  weak alignment — treat the numbers below with suspicion.")
warped = cv2.warpPerspective(live, H, (ref.shape[1], ref.shape[0]))


def centroid(img, lo, hi, region, min_area):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(lo), np.array(hi))
    mask = np.zeros_like(m)
    x0, y0, x1, y1 = region
    mask[y0:y1, x0:x1] = 255
    m = cv2.bitwise_and(m, mask)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > min_area]
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    M = cv2.moments(c)
    return np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])


for label, spec in [("tube (blue cap)   ", BLUE_CAP), ("rack (orange funnel)", ORANGE_FUNNEL)]:
    r = centroid(ref, spec["lo"], spec["hi"], spec["region"], spec["min_area"])
    l = centroid(warped, spec["lo"], spec["hi"], spec["region"], spec["min_area"])
    if r is None or l is None:
        print(f"{label}: NOT FOUND (ref={r is not None}, live={l is not None}) — check placement/lighting")
        continue
    d = l - r
    print(
        f"{label}: ref=({r[0]:6.1f},{r[1]:6.1f})  live=({l[0]:6.1f},{l[1]:6.1f})  "
        f"delta=({d[0]:+5.1f},{d[1]:+5.1f}) px  |d|={np.hypot(*d):.1f}"
    )


def tube_angle(img, cap):
    """Long-axis angle of the tube (deg) from Hough lines in a window right of
    the cap. 0 = horizontal as trained; negative = tip rotated up-right in the
    image. Added Aug 22 after a knocked tube (cap ~28 px off, body rotated)
    passed the cap-only check. Validated on synthetic rotations of the
    reference: reads ~17° at 20°, ~27° at 30°; noisy below ~8°, so only
    |delta| > 5° is flagged. Needs the arm OUT of the window (home pose)."""
    if cap is None:
        return None
    x0, y0 = int(cap[0]) - 10, int(cap[1]) - 70
    win = img[max(y0, 0):y0 + 140, max(x0, 0):x0 + 200]
    g = cv2.GaussianBlur(cv2.cvtColor(win, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    lines = cv2.HoughLinesP(cv2.Canny(g, 40, 120), 1, np.pi / 360, threshold=25, minLineLength=50, maxLineGap=8)
    if lines is None:
        return None
    angs = []
    for x1, y1, x2, y2 in lines[:, 0]:
        a = (np.degrees(np.arctan2(y2 - y1, x2 - x1)) + 90) % 180 - 90
        if abs(a) < 60:  # tube-ish; excludes the vertical grid tape
            angs.append(a)
    return float(np.median(angs)) if angs else None


spec = BLUE_CAP
a_ref = tube_angle(ref, centroid(ref, spec["lo"], spec["hi"], spec["region"], spec["min_area"]))
a_live = tube_angle(warped, centroid(warped, spec["lo"], spec["hi"], spec["region"], spec["min_area"]))
if a_ref is None or a_live is None:
    print("tube angle        : NOT MEASURED (no long edges found — arm in the window, or tube missing)")
else:
    flag = "  ⚠️  ROTATED — re-place the tube" if abs(a_live - a_ref) > 5 else ""
    print(f"tube angle        : ref={a_ref:+5.1f}°  live={a_live:+5.1f}°  delta={a_live - a_ref:+5.1f}°{flag}")

blend = cv2.addWeighted(ref, 0.5, warped, 0.5, 0)
cv2.imwrite(str(OUT_DIR / "overlay_ref_vs_live.png"), blend)
diff = cv2.absdiff(cv2.GaussianBlur(ref, (5, 5), 0), cv2.GaussianBlur(warped, (5, 5), 0)).sum(axis=2)
print("mean abs scene diff (0-765):", round(float(diff.mean()), 1))
print(f"overlay written: {OUT_DIR / 'overlay_ref_vs_live.png'} — LOOK AT IT before trusting numbers")
