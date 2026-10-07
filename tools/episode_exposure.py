"""How much did exposure drift across a recorded episode? Read-only, offline.

You cannot run a live exposure check while `lerobot-teleoperate` or
`lerobot-record` holds the camera — one process per camera. So measure it
afterwards, from the encoded video the policy will actually train on.

The kit cameras have no exposure lock (probed Aug 8, 2026: AVFoundation ignores
CAP_PROP_AUTO_EXPOSURE entirely). Auto-exposure therefore keeps re-metering
mid-episode as the white arm sweeps through frame. This tells you how bad it is.

    conda activate lerobot
    python episode_exposure.py ~/.cache/huggingface/lerobot/<repo_id>/videos/.../episode_000000.mp4

Read the RANGE, not the mean. A swing under ~15 is fine. Over ~40 means the
frames early and late in an episode look like different lighting conditions,
and `--dataset.image_transforms.enable=true` at training time starts to matter.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

if len(sys.argv) < 2:
    raise SystemExit(__doc__)

path = Path(sys.argv[1]).expanduser()
if not path.exists():
    raise SystemExit(f"not found: {path}\nLook under ~/.cache/huggingface/lerobot/<repo_id>/videos/")

cap = cv2.VideoCapture(str(path))
if not cap.isOpened():
    raise SystemExit(f"could not open {path}")

means, blown = [], []
while True:
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    means.append(float(gray.mean()))
    blown.append(100.0 * int((gray >= 250).sum()) / gray.size)
cap.release()

if not means:
    raise SystemExit("no frames decoded")

means, blown = np.array(means), np.array(blown)
swing = means.max() - means.min()

print(f"\n{path.name}   {len(means)} frames")
print(f"  brightness  min {means.min():6.1f}   max {means.max():6.1f}   mean {means.mean():6.1f}")
print(f"  SWING       {swing:6.1f}")
print(f"  clipped     max {blown.max():5.1f}% of frame")

# Where in the episode does it move? Coarse sparkline over ten buckets.
buckets = np.array_split(means, 10)
lo, hi = means.min(), max(means.max(), means.min() + 1e-6)
bars = "".join(" ▁▂▃▄▅▆▇█"[int(8 * (b.mean() - lo) / (hi - lo))] for b in buckets)
print(f"  over time   |{bars}|  (start -> end)")

print()
if swing < 15:
    print("  STABLE. Auto-exposure is not meaningfully moving. Nothing to do.")
elif swing < 40:
    print("  MODERATE drift. Usable, but enable image augmentation at training time:")
    print("    --dataset.image_transforms.enable=true")
else:
    print("  LARGE drift. Early and late frames look like different lighting.")
    print("  Fix the scene first: diffuse the light, mid-tone surface, keep the")
    print("  meter off the rail. Augmentation alone will not cover this much.")
