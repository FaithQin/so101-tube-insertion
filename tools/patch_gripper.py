"""Patch ONLY the gripper range_min/range_max in the calibration files.

Everything else — homing offsets, all other joints — is left untouched. Writes a
timestamped .bak of each file first.

This exists so a bad gripper sweep does NOT require a full recalibration. Get the
new endpoints from gripper_measure.py.

⚠️ AFTER PATCHING: at the next lerobot-calibrate / lerobot-teleoperate prompt,
press plain ENTER — NOT `c`. Plain ENTER writes the file's values into the
motors, which is what you want. `c` throws the patch away and starts a fresh
calibration.

Usage:
    conda activate lerobot
    python patch_gripper.py --follower 2048 3449 --leader 1960 3190
    python patch_gripper.py --show                  # just print current values
"""

import argparse
import json
import shutil
from pathlib import Path

from _arms import FOLLOWER_CAL, JOINTS, LEADER_CAL

GOOD_WIDTH = {"follower": (1300, 1500), "leader": (1150, 1330)}

parser = argparse.ArgumentParser()
parser.add_argument("--follower", nargs=2, type=int, metavar=("MIN", "MAX"))
parser.add_argument("--leader", nargs=2, type=int, metavar=("MIN", "MAX"))
parser.add_argument("--show", action="store_true", help="print current calibration and exit")
parser.add_argument("--force", action="store_true", help="apply even if width looks wrong")
args = parser.parse_args()

TARGETS = {"follower": FOLLOWER_CAL, "leader": LEADER_CAL}


def dump(path, label):
    data = json.loads(Path(path).read_text())
    print(f"\n{label}  ({Path(path).name})")
    for j in JOINTS:
        v = data[j]
        print(f"  {j:<14} homing {v['homing_offset']:>6}   "
              f"range {v['range_min']:>5}-{v['range_max']:<5}  width {v['range_max'] - v['range_min']}")


if args.show or (not args.follower and not args.leader):
    for name, path in TARGETS.items():
        dump(path, name.upper())
    if not args.show:
        print("\nNothing to patch. Pass --follower MIN MAX and/or --leader MIN MAX.")
    raise SystemExit(0)

for name, path in TARGETS.items():
    new = getattr(args, name)
    if not new:
        continue
    lo, hi = new
    width = hi - lo
    good_lo, good_hi = GOOD_WIDTH[name]
    if not (good_lo <= width <= good_hi) and not args.force:
        print(f"REFUSING {name}: width {width} outside known-good {good_lo}-{good_hi}. "
              f"Re-measure, or pass --force if you're sure.")
        continue

    shutil.copy2(path, path.with_suffix(".json.bak"))
    data = json.loads(Path(path).read_text())
    old_lo, old_hi = data["gripper"]["range_min"], data["gripper"]["range_max"]
    data["gripper"]["range_min"] = lo
    data["gripper"]["range_max"] = hi
    Path(path).write_text(json.dumps(data, indent=4) + "\n")
    print(f"{name.upper():<10} {old_lo}-{old_hi} (width {old_hi - old_lo})  ->  "
          f"{lo}-{hi} (width {width})   backup: {path.with_suffix('.json.bak').name}")

for name, path in TARGETS.items():
    dump(path, name.upper())

print("\n⚠️  At the next lerobot prompt press plain ENTER, NOT `c`.")
print("⚠️  Once teleop looks right, re-copy calibration-backup/:")
print('    cp -R ~/.cache/huggingface/lerobot/calibration/. '
      '"<repo>/calibration-backup/"')
