#!/usr/bin/env python
"""Watch the gripper's protection registers and live strain while you clamp.

READ-ONLY. Dumps the gripper's protection thresholds once, then streams
position / load / current / temperature at ~10 Hz. Clamp on the object and
watch what climbs in the seconds before the LED flashes red -- that names
the threshold that is tripping, so it can be tuned from measurement.

Usage (12V on first, then USB, no other session holding the port):
    python tools/watch_protection.py
"""

import time

from _arms import FOLLOWER_PORT, open_bus

THRESHOLDS = (
    "Max_Torque_Limit",
    "Protection_Current",
    "Protective_Torque",
    "Protection_Time",
    "Overload_Torque",
    "Over_Current_Protection_Time",
    "Unloading_Condition",
    "LED_Alarm_Condition",
)
LIVE = ("Present_Position", "Present_Load", "Present_Current", "Present_Temperature")

bus = open_bus(FOLLOWER_PORT)
try:
    print("gripper protection thresholds:")
    for reg in THRESHOLDS:
        try:
            print(f"  {reg:28s} = {bus.read(reg, 'gripper', normalize=False)}")
        except Exception as e:
            print(f"  {reg:28s} read failed: {e}")

    print("\nnow clamp on the object; Ctrl-C to stop.")
    print(f"{'t (s)':>7}  {'pos':>5}  {'load':>6}  {'current':>7}  {'temp':>4}")
    t0 = time.time()
    while True:
        vals = {reg: bus.read(reg, "gripper", normalize=False) for reg in LIVE}
        print(
            f"{time.time() - t0:7.1f}  {vals['Present_Position']:5d}  "
            f"{vals['Present_Load']:6d}  {vals['Present_Current']:7d}  "
            f"{vals['Present_Temperature']:4d}"
        )
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    bus.disconnect(disable_torque=False)
