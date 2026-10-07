"""Shared config + bus helper for the diagnostic scripts in this folder.

Ports are derived from the CH343 chip serial, NOT the USB port, so they are
stable across cable swaps and hubs. Verified Aug 7, 2026.
"""

import json
from pathlib import Path

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

CAL_ROOT = Path.home() / ".cache/huggingface/lerobot/calibration"

FOLLOWER_PORT = "/dev/tty.usbmodem5C4C1245641"   # WHITE, gripper jaws
LEADER_PORT = "/dev/tty.usbmodem5C821069931"     # BLACK, trigger handle

FOLLOWER_CAL = CAL_ROOT / "robots/so_follower/follower.json"
LEADER_CAL = CAL_ROOT / "teleoperators/so_leader/leader.json"

ARMS = {
    "FOLLOWER (white, jaws)": (FOLLOWER_PORT, FOLLOWER_CAL),
    "LEADER   (black, trigger)": (LEADER_PORT, LEADER_CAL),
}

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def motors():
    """Motor map matching SOFollower/SOLeader: gripper is 0-100, the rest -100..100."""
    return {
        j: Motor(i + 1, "sts3215",
                 MotorNormMode.RANGE_0_100 if j == "gripper" else MotorNormMode.RANGE_M100_100)
        for i, j in enumerate(JOINTS)
    }


def load_calibration(cal_path):
    raw = json.loads(Path(cal_path).read_text())
    return {j: MotorCalibration(**raw[j]) for j in JOINTS}


def open_bus(port, cal_path=None):
    """Open a read-only connection. Pass cal_path to get normalized reads."""
    bus = FeetechMotorsBus(
        port=port,
        motors=motors(),
        calibration=load_calibration(cal_path) if cal_path else None,
    )
    bus.connect()
    return bus
