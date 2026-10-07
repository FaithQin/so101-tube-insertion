"""Reseat the gripper jaw into the pose distribution the policy was trained from.

Why this exists (Aug 29, 2026): after ACT-B episodes the jaws end in a
relieved/faulted state and settle around 2.00, while v2 frame-0 across all 50
training episodes is [1.14, 1.64] (mean 1.25, sd 0.09). `home_arm.py` does not
correct it, because a residual of 0.7 is comfortably inside its flat 2.5
tolerance — but 2.00 is roughly 8 sigma outside the distribution the policy
actually saw, so the preflight gate refuses to launch.

Commanding 1.3 again does not help; that is precisely what home_arm already
does. The jaw has to be driven OPEN first to unseat it, then walked back down
slowly. This is the remedy specified in the backlog task "Fix wrapper gripper
creep path #2".

Torque stays ON at the end: training frame-0s were captured with the jaw held,
and a full release lets it drift back out of support — the same mistake as the
Aug 16 elbow-sag bug.

Bus-only, no cameras. Needs the serial port to itself.

    python tools/reseat_gripper.py     # exit 0 if the jaw lands in support
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _arms import FOLLOWER_CAL, FOLLOWER_PORT, open_bus  # noqa: E402

# v2 frame-0 gripper range across all 50 training episodes. Must match
# home_gate.SUPPORT["gripper"] — tests/test_home_gate.py re-derives it from the
# parquet, so if the dataset changes, that test fails first.
SUPPORT_LO, SUPPORT_HI = 1.14, 1.64
TARGET = 1.3  # what home_arm commands; v2 frame-0 mean is 1.25
OPEN_TO = 10.0
STEP = 0.2
CLOSE_HZ = 50


def reseat(bus) -> float:
    """Open the jaw, then close it slowly to TARGET. Returns the settled pose."""
    bus.write("Goal_Position", "gripper", OPEN_TO, normalize=True)
    time.sleep(0.6)

    pos = OPEN_TO
    while pos > TARGET:
        pos = max(TARGET, pos - STEP)
        bus.write("Goal_Position", "gripper", pos, normalize=True)
        time.sleep(1 / CLOSE_HZ)

    time.sleep(0.5)
    return float(bus.read("Present_Position", "gripper", normalize=True))


def main() -> int:
    bus = open_bus(FOLLOWER_PORT, FOLLOWER_CAL)
    try:
        try:
            bus.enable_torque()
        except Exception:
            bus.write("Torque_Enable", "gripper", 1, normalize=False)

        before = float(bus.read("Present_Position", "gripper", normalize=True))
        after = reseat(bus)
        ok = SUPPORT_LO <= after <= SUPPORT_HI
        print(
            f"gripper {before:.2f} -> {after:.2f} "
            f"(training support [{SUPPORT_LO}, {SUPPORT_HI}]) "
            f"{'RESEAT OK' if ok else 'RESEAT INSUFFICIENT'}"
        )
        return 0 if ok else 1
    finally:
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    sys.exit(main())
