"""Poison v3: deliberate interrupted motion, no process kills.
Drives the first 2.0 s (40 ticks @ 20 Hz) of v2 ep0's actions directly over
the bus, then stops commanding and disconnects leaving the hold to home_arm.
Prints per-joint motion delta so a non-moving arm CANNOT pass silently."""
import glob
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _arms import FOLLOWER_CAL, FOLLOWER_PORT, JOINTS, open_bus  # noqa: E402

V2 = Path("~/.cache/huggingface/lerobot/faithqin/so101-tube-insert-v2_20260816_035314").expanduser()
df = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(V2 / "data/chunk-*/file-*.parquet")))])
g = df[df.episode_index == 0]
acts = np.stack(g["action"].to_numpy())[:40]  # first 2.0 s

bus = open_bus(FOLLOWER_PORT, FOLLOWER_CAL)
try:
    start = {j: bus.read("Present_Position", j, normalize=False) for j in JOINTS}
    for j in JOINTS:
        bus.write("Torque_Enable", j, 1, normalize=False)
    t0 = time.perf_counter()
    for k, a in enumerate(acts):
        for i, j in enumerate(JOINTS):
            bus.write("Goal_Position", j, float(a[i]), normalize=True)
        while time.perf_counter() - t0 < (k + 1) * 0.05:
            time.sleep(0.002)
    time.sleep(0.3)
    end = {j: bus.read("Present_Position", j, normalize=False) for j in JOINTS}
    moved = {j: int(end[j]) - int(start[j]) for j in JOINTS}
    total = sum(abs(v) for v in moved.values())
    print("per-joint raw-tick delta:", moved)
    if total < 60:
        print(f"POISON FAILED: arm did not move (total |delta| = {total} ticks)")
        sys.exit(1)
    print(f"POISON DELIVERED: 40 ticks executed, total motion {total} raw ticks")
finally:
    bus.disconnect()
