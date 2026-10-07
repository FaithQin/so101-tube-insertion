#!/usr/bin/env python
"""Controlled grip experiment: which protection actually fires, and at what values.

Does four things in one run, printing everything:
  1. Dumps the gripper's protection thresholds (verifies whether the
     Protective_Torque=50 patch actually stuck in EPROM).
  2. If Protective_Torque still reads 20, tries to write 50 itself
     (EPROM unlock via Lock register) and re-reads to confirm.
  3. Enables torque on the GRIPPER ONLY and closes onto whatever is between
     the jaws with a bounded lead, streaming position/load/current/temp at
     20 Hz for --hold seconds -- through the moment protection fires, if it
     does. The last rows before the flash name the guilty threshold.
  4. Releases and disables gripper torque on exit, even on Ctrl-C.

Place the vial between the jaws (jaws just touching it) before running.

Usage:
    python tools/grip_test.py            # lead 40 counts, hold 8 s
    python tools/grip_test.py --lead 60 --hold 10
"""

import argparse
import time

from _arms import FOLLOWER_PORT, open_bus

THRESHOLDS = (
    "Max_Torque_Limit",
    "Protection_Current",
    "Protective_Torque",
    "Protection_Time",
    "Overload_Torque",
    "Over_Current_Protection_Time",
)

parser = argparse.ArgumentParser()
parser.add_argument("--lead", type=int, default=40, help="counts to command past contact")
parser.add_argument("--hold", type=float, default=8.0, help="seconds to hold the grip")
args = parser.parse_args()

bus = open_bus(FOLLOWER_PORT)
try:
    print("thresholds as found:")
    for reg in THRESHOLDS:
        print(f"  {reg:30s} = {bus.read(reg, 'gripper', normalize=False)}")

    if bus.read("Protective_Torque", "gripper", normalize=False) != 50:
        print("\nProtective_Torque is NOT 50 -- writing it here, with EPROM unlock:")
        bus.write("Lock", "gripper", 0)
        bus.write("Protective_Torque", "gripper", 50)
        bus.write("Lock", "gripper", 1)
        readback = bus.read("Protective_Torque", "gripper", normalize=False)
        print(f"  readback after write: {readback}  ({'STUCK' if readback == 50 else 'DID NOT STICK'})")

    p0 = bus.read("Present_Position", "gripper", normalize=False)
    goal = p0 - args.lead
    print(f"\ncontact pos {p0}, commanding {goal} (lead {args.lead}), holding {args.hold}s")
    print(f"{'t (s)':>7}  {'pos':>5}  {'load':>6}  {'current':>7}  {'temp':>4}")

    bus.write("Torque_Enable", "gripper", 1)
    bus.write("Goal_Position", "gripper", goal, normalize=False)
    t0 = time.time()
    fault_at = None
    while time.time() - t0 < args.hold:
        try:
            row = {
                "pos": bus.read("Present_Position", "gripper", normalize=False),
                "load": bus.read("Present_Load", "gripper", normalize=False),
                "cur": bus.read("Present_Current", "gripper", normalize=False),
                "temp": bus.read("Present_Temperature", "gripper", normalize=False),
            }
            print(
                f"{time.time() - t0:7.2f}  {row['pos']:5d}  {row['load']:6d}  "
                f"{row['cur']:7d}  {row['temp']:4d}"
            )
        except Exception as e:
            fault_at = time.time() - t0
            print(f"{fault_at:7.2f}  << servo stopped answering: {type(e).__name__} >>")
            break
        time.sleep(0.05)

    if fault_at is not None:
        print(f"\nPROTECTION FIRED at t={fault_at:.2f}s -- the rows above name the culprit.")
    else:
        print("\nheld the full duration with no fault.")
finally:
    try:
        bus.write("Goal_Position", "gripper", p0, normalize=False)
        time.sleep(0.5)
        bus.write("Torque_Enable", "gripper", 0)
    except Exception:
        print("(release failed -- power cycle the follower before the next session)")
    bus.disconnect(disable_torque=False)
