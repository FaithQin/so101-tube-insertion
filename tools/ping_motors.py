"""Which servos are alive, and is any of them faulted? Read-only.

Answers the two questions you actually have when something stops responding:
"is the motor there at all?" and "is it there but latched?" — which look
identical from teleop, and have completely different fixes.

    conda activate lerobot
    python ping_motors.py                 # both arms
    python ping_motors.py follower

⚠️ Needs the serial port to itself. Stop teleop / record / watch_load first.
12V ON BEFORE USB, or the board enumerates and the servo bus stays dark.

READING THE OUTPUT
------------------
no response        The motor is not on the bus. Power, wiring, or a dead servo.
                   If NO motor responds, the bus is dark — that is the
                   12V-after-USB failure, not six dead servos.
Status != 0        The motor is present and reporting a fault. Overload latches
                   here and will refuse writes until a full power cycle:
                   USB out -> 12V out -> 12V in -> USB in.
Torque_Enable 0    Not a fault — the joint is simply released and will backdrive
                   (stiffly; 1/345 gearing is hard to move by hand even limp).

Never run `lerobot-setup-motors` on this pre-assembled kit — manual §3.5, it
requires disassembly. Nothing here writes to a motor.
"""

import argparse
import sys

from _arms import FOLLOWER_PORT, JOINTS, LEADER_PORT, open_bus

# Standard Feetech STS/SCS status bits (Status register, addr 65). A nonzero
# value means a fault regardless of whether the label below is exact for this
# firmware — treat the number as authoritative and the name as a hint.
STATUS_BITS = [
    (0x01, "voltage"),
    (0x02, "sensor"),
    (0x04, "overheat"),
    (0x08, "current"),
    (0x10, "angle"),
    (0x20, "OVERLOAD"),
]

FIELDS = [
    ("pos", "Present_Position"),
    ("load", "Present_Load"),
    ("cur", "Present_Current"),
    ("temp", "Present_Temperature"),
    ("volt", "Present_Voltage"),
    ("torq", "Torque_Enable"),
    ("stat", "Status"),
]


def decode_status(value):
    if value is None:
        return "unreadable"
    if value == 0:
        return "ok"
    names = [name for bit, name in STATUS_BITS if value & bit]
    return f"FAULT 0x{value:02x} ({', '.join(names) or 'unknown bit'})"


def check_arm(label: str, port: str) -> int:
    print(f"\n=== {label}  {port} ===")
    try:
        bus = open_bus(port)
    except Exception as exc:
        print(f"  cannot open the port: {exc}")
        print("  -> is another process holding it? is 12V on, applied BEFORE USB?")
        return 1

    faults = 0
    try:
        try:
            found = bus.broadcast_ping() or {}
        except Exception as exc:
            print(f"  broadcast_ping failed ({exc}); falling back to per-motor ping")
            found = {}

        if found:
            print(f"  broadcast: {len(found)}/6 responded — ids {sorted(found)}")
        else:
            print("  broadcast: nothing responded")

        for idx, joint in enumerate(JOINTS, start=1):
            alive = True
            try:
                if bus.ping(joint) is None:
                    alive = False
            except Exception:
                alive = False

            if not alive:
                print(f"  id {idx} {joint:15s} no response")
                faults += 1
                continue

            vals = {}
            for short, reg in FIELDS:
                try:
                    vals[short] = bus.read(reg, joint, normalize=False)
                except Exception:
                    vals[short] = None

            status = decode_status(vals["stat"])
            if status not in ("ok",):
                faults += 1
            print(
                f"  id {idx} {joint:15s} pos {str(vals['pos']):>5}  "
                f"load {str(vals['load']):>5}  cur {str(vals['cur']):>4}  "
                f"{str(vals['temp']):>3}C  {str(vals['volt']):>3}  "
                f"torque {vals['torq']}  {status}"
            )
    finally:
        bus.disconnect(disable_torque=False)

    if faults == 0:
        print("  all six motors present and clean.")
    return faults


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("arm", nargs="?", default="both",
                        choices=["both", "follower", "leader"])
    args = parser.parse_args()

    targets = []
    if args.arm in ("both", "follower"):
        targets.append(("FOLLOWER (white, jaws)", FOLLOWER_PORT))
    if args.arm in ("both", "leader"):
        targets.append(("LEADER (black, trigger)", LEADER_PORT))

    total = sum(check_arm(label, port) for label, port in targets)

    if total:
        print(f"\n{total} problem(s). In order:")
        print("  1. Any FAULT line — full power cycle of that arm:")
        print("       USB out -> 12V out -> 12V in -> USB in")
        print("     A USB reconnect alone does NOT clear a latched overload.")
        print("  2. Every motor silent — the bus is dark. Same power cycle, same order.")
        print("  3. One motor silent, others fine — check that servo's cable at both")
        print("     ends. Do NOT run lerobot-setup-motors; it needs disassembly.")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
