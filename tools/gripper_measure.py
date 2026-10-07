"""Measure SAFE gripper endpoints on both arms. Read-only except torque-off.

Why this exists: on Aug 7, 2026 the lerobot-calibrate range sweep hand-forced
both grippers roughly DOUBLE their safe travel. Teleop then commanded the
follower jaws into a hard clamp and the servo flashed overload. The endpoint has
to be safe UNDER TORQUE, not merely reachable by hand.

This disables torque on the GRIPPER ONLY (the rest of the arm keeps holding, so
nothing sags), streams raw + normalized gripper position, and tracks the min/max
you reach by hand. Feed the result to patch_gripper.py.

Frame note: range_min/max are RAW positions recorded AFTER homing offsets are
written (motors_bus.py:822). Endpoints from an older calibration are in a
different frame and can NEVER be transplanted. Compare WIDTHS instead — widths
are offset-invariant.

  Known-good widths:  leader ~1216-1230,  follower ~1393-1401
  A width near double that means the sweep over-travelled.

Usage:
    conda activate lerobot
    python gripper_measure.py [n_samples]     # no arg = run until Ctrl-C
"""

import json
import sys
import time
from pathlib import Path

from _arms import ARMS, open_bus

max_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 0

buses, seen = {}, {}
for label, (port, cal_path) in ARMS.items():
    bus = open_bus(port, cal_path)
    bus.disable_torque("gripper")          # gripper only — arm keeps holding
    buses[label] = bus
    seen[label] = [10**9, -10**9]
    g = json.loads(Path(cal_path).read_text())["gripper"]
    width = g["range_max"] - g["range_min"]
    print(f"{label}  current calib: {g['range_min']}-{g['range_max']}  width {width}")

print("\nGripper torque is OFF on both arms. Move BOTH by hand:")
print("  follower jaws : fully open  ->  closed only until resistance RISES. DO NOT force.")
print("  leader trigger: fully released -> comfortably squeezed. DO NOT crush.")
print("Ctrl-C when done.\n")

n = 0
try:
    while max_samples == 0 or n < max_samples:
        n += 1
        parts = []
        for label, bus in buses.items():
            raw = bus.read("Present_Position", "gripper", normalize=False)
            norm = bus.read("Present_Position", "gripper", normalize=True)
            seen[label][0] = min(seen[label][0], raw)
            seen[label][1] = max(seen[label][1], raw)
            parts.append(f"{label} raw {raw:>5} norm {norm:>6.1f}")
        print("   |   ".join(parts))
        time.sleep(0.3)
except KeyboardInterrupt:
    pass
finally:
    print("\n=== RAW RANGE REACHED BY HAND — use as range_min / range_max ===")
    for label, (lo, hi) in seen.items():
        if lo <= hi:
            w = hi - lo
            flag = "" if 1150 < w < 1450 else "   <-- WIDTH LOOKS WRONG, re-measure"
            print(f"  {label}  min {lo}  max {hi}  width {w}{flag}")
    print("\nNext: python patch_gripper.py --follower MIN MAX --leader MIN MAX")
    for bus in buses.values():
        bus.disconnect(disable_torque=False)
