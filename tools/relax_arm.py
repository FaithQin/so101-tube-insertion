#!/usr/bin/env python
"""Cut torque on the follower, even when a servo is missing the bus handshake. Read-mostly.

    python tools/relax_arm.py                # follower; wrist_flex first, then the rest; verified by readback

WHY (Sep 14 2026, 12:47, the lab notebook (private)). After three certification replays the arm was left holding
for ~15 min; wrist_flex reached 70 C -- past the 69 C brownout on the ledger -- and stopped
answering the handshake. The only torque-off path in these diagnostics, `_arms.open_bus` ->
`bus.connect()`, then FAILED ("Missing motor IDs: 4"): it needs every servo to answer, which a
browning-out servo does not, at exactly the moment torque has to come off. What worked by hand:
`FeetechMotorsBus(...).connect(handshake=False)`, then `disable_torque(motor, num_retry=5)` for
each joint, wrist_flex first, then a readback of Torque_Enable on all six. This is that recipe.

WHEN. After every certification replay (`replay_telemetry.py` leaves the arm HOLDING at
disconnect, by design), before any pause with torque on, and as the first response to a servo
that reads hot. Never between `home_arm.py` and a rollout: the rollout needs the held pose, and
the runners home seconds before launch. Home first if the arm is extended over the rack -- a
released arm sags, and a sag onto the rack decertifies the scene.

GATE, DON'T NARRATE: exit 1 unless every joint reads Torque_Enable 0 after the writes.
Tested under a fake bus in tests/test_relax_arm.py; no port opens at import.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from lerobot.motors.feetech import FeetechMotorsBus

from _arms import FOLLOWER_PORT, JOINTS, motors  # _arms opens no bus at import (open_bus() only)

FIRST = ("wrist_flex",)   # the joint that browned out on Sep 14; the ledger's 69 C brownouts were the elbow
NUM_RETRY = 5             # a servo mid-brownout drops replies; one write is not a cut


def port_holders(port: str) -> list[int]:
    """PIDs holding the serial port open (lsof -t). Empty when free or when lsof is unavailable."""
    try:
        out = subprocess.run(["lsof", "-t", port], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [int(x) for x in out.split() if x.strip().isdigit()]


class RelaxFailed(RuntimeError):
    """A joint still reports torque ON after the writes and the readback."""


def relax_order(joints=JOINTS, first=FIRST) -> list[str]:
    head = [j for j in first if j in joints]
    return head + [j for j in joints if j not in head]


def relax(bus, joints=JOINTS, first=FIRST, num_retry=NUM_RETRY) -> dict[str, int]:
    """Disable torque joint by joint (with retries), then verify by readback. Raises RelaxFailed."""
    for j in relax_order(joints, first):
        bus.disable_torque(j, num_retry=num_retry)
    still = {}
    for j in joints:
        still[j] = int(bus.read("Torque_Enable", j, normalize=False, num_retry=num_retry))
    on = [j for j, v in still.items() if v != 0]
    if on:
        raise RelaxFailed(f"torque still ON after {num_retry} retries: {', '.join(on)} -- "
                          f"power-cycle that arm (USB out -> 12V out -> 12V in -> USB in) before anything else")
    return still


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", default=FOLLOWER_PORT)
    a = ap.parse_args(argv)

    # Sep 14 17:45: run under a live rollout, this cut torque mid-episode. Never open a held port.
    holders = port_holders(a.port)
    if holders:
        print(f"RELAX REFUSED: {a.port} is held by pid(s) {', '.join(map(str, holders))} -- a rollout or replay "
              f"is using the arm. Stop that process first (kill the whole tree: launcher, runner, rollout), then relax.")
        return 2

    bus = FeetechMotorsBus(port=a.port, motors=motors(), calibration=None)
    bus.connect(handshake=False)   # NOT the handshake path: it raises on the servo that most needs this
    try:
        try:
            still = relax(bus)
        except RelaxFailed as exc:
            print(f"RELAX FAILED: {exc}")
            return 1
        for idx, j in enumerate(JOINTS, start=1):
            cur = temp = "?"
            try:
                cur = bus.read("Present_Current", j, normalize=False, num_retry=NUM_RETRY)
                temp = f"{bus.read('Present_Temperature', j, normalize=False, num_retry=NUM_RETRY)}C"
            except Exception:
                pass
            print(f"  id {idx} {j:15s} torque {still[j]}  cur {str(cur):>4}  {temp:>4}")
        print("RELAXED: Torque_Enable reads 0 on all six. Let it cool; ping before the next certification.")
        return 0
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)   # already off; a second num_retry=0 walk could raise


if __name__ == "__main__":
    sys.exit(main())
