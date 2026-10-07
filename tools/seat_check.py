"""Blind mechanical seat check for the pre-registered priming confirmation.
Captures one front-camera frame, finds the blue cap, classifies:
  SEAT  cap centroid in the rack/funnel region (470<x<640, 400<y<540)
  MISS  cap centroid elsewhere in the workspace (incl. mat)
  UNKNOWN  no cap found (occluded/out of frame)
Region logic identical to the tail classifier validated on all 46 v3
episodes (Aug 27). Saves the frame for re-scoring. No hypothesis input.
"""
import sys, time
from pathlib import Path
import cv2
import numpy as np

OUT = Path(__file__).parent / "seat_checks"
OUT.mkdir(exist_ok=True)
idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1

cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
for _ in range(15):
    cap.read()
ok, img = cap.read()
cap.release()
if not ok or img is None:
    print("SEATCHECK UNKNOWN (no frame)")
    sys.exit(1)

ts = time.strftime("%H%M%S")
cv2.imwrite(str(OUT / f"seat_{ts}.png"), img)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
m = cv2.inRange(hsv, np.array([100, 120, 60]), np.array([125, 255, 255]))
m[:250, :] = 0
cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = [c for c in cnts if cv2.contourArea(c) > 40]
if not cnts:
    print(f"SEATCHECK UNKNOWN (no cap) frame=seat_{ts}.png")
    sys.exit(0)
M = cv2.moments(max(cnts, key=cv2.contourArea))
x, y = M["m10"] / M["m00"], M["m01"] / M["m00"]
verdict = "SEAT" if (470 < x < 640 and 400 < y < 540) else "MISS"
print(f"SEATCHECK {verdict} cap=({x:.0f},{y:.0f}) frame=seat_{ts}.png")
