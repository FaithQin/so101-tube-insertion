"""Stream gripper position from ONE port. Ground truth for "which arm is this?"

Run it, then squeeze that arm's jaws (or trigger) by hand. If the number moves,
that port is that physical arm — a check that owes nothing to any file.

Usage:
    conda activate lerobot
    python watch_gripper.py                 # defaults to the follower
    python watch_gripper.py leader
    python watch_gripper.py /dev/tty.usbmodemXXXX
"""

import sys
import time

from _arms import FOLLOWER_PORT, LEADER_PORT, open_bus

arg = sys.argv[1] if len(sys.argv) > 1 else "follower"
port = {"follower": FOLLOWER_PORT, "leader": LEADER_PORT}.get(arg, arg)

bus = open_bus(port)
print(f"{port}\nMove this arm's gripper by hand. Ctrl-C to stop.\n")
try:
    while True:
        raw = bus.read("Present_Position", "gripper", normalize=False)
        print(f"  gripper raw: {raw:5d}   {'#' * (raw // 60)}")
        time.sleep(0.25)
except KeyboardInterrupt:
    pass
finally:
    bus.disconnect(disable_torque=False)
