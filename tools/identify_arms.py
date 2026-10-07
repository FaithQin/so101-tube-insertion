"""Which physical arm is on which serial port? Read-only.

Compares each port's live Homing_Offset registers against the two calibration
files. Those registers are written into the servos at calibration time, so they
identify the arm regardless of which USB port it's plugged into.

No writes, no torque changes.

    conda activate lerobot
    python identify_arms.py
"""

import json
from pathlib import Path

from _arms import ARMS, JOINTS, open_bus

expected = {}
for label, (_port, cal_path) in ARMS.items():
    data = json.loads(Path(cal_path).read_text())
    expected[label] = {j: data[j]["homing_offset"] for j in JOINTS}

for label, (port, _cal) in ARMS.items():
    print(f"\n=== {port} ===")
    try:
        bus = open_bus(port)
    except Exception as e:
        print(f"  CANNOT CONNECT: {type(e).__name__}: {e}")
        print("  Bus dark? 12V must be ON and applied BEFORE usb.")
        continue

    live = {}
    for j in JOINTS:
        try:
            live[j] = bus.read("Homing_Offset", j, normalize=False)
        except Exception as e:
            live[j] = f"ERR {type(e).__name__}"

    print(f"  live homing offsets: {live}")

    match = [lbl for lbl, exp in expected.items() if exp == live]
    if len(match) == 1:
        print(f"  >>> IDENTIFIED: {match[0]}")
    elif not match:
        print("  >>> NO MATCH against either calibration file.")
        print("      Expected after a recalibration you haven't written to the motors yet —")
        print("      run lerobot-calibrate and press plain ENTER to write the file.")
        for lbl, exp in expected.items():
            diff = {j: (live.get(j), exp[j]) for j in JOINTS if live.get(j) != exp[j]}
            print(f"      vs {lbl}: differs on {diff}")
    else:
        print(f"  >>> AMBIGUOUS, matched: {match}")

    bus.disconnect(disable_torque=False)

print("\nGround truth without any file: run watch_gripper.py on one port and")
print("squeeze that arm's jaws by hand — the number should move.")
