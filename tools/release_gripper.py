"""Get the jaws open again. Run this when the gripper is stuck clamped.

`watch_load.py --close` deliberately leaves torque ON at the end so a held object
does not drop. That is correct during a measurement and useless afterwards — it
leaves the jaws clamped with no way out. This is the way out.

    conda activate lerobot
    cd "<repo>/tools"
    python release_gripper.py              # limp: torque off, open by hand
    python release_gripper.py --open       # drive the jaws open, then go limp

Stop any other process on the port first (watch_load.py, teleop, record). The
Feetech bus is half-duplex serial and will not share.

IF THIS SCRIPT ITSELF FAILS TO WRITE, the servo has LATCHED an overload. A
latched servo refuses every write, including Torque_Enable=0, so no software can
help. Full power cycle of the FOLLOWER, in this order:

    USB out  ->  12V out  ->  12V in  ->  USB in

A USB reconnect alone does NOT clear the latch. Neither does replugging 12V while
USB stays connected — that leaves the board enumerated and the servo bus dark.
"""

import argparse
import time

from _arms import FOLLOWER_PORT, JOINTS, LEADER_PORT, open_bus

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("arm", nargs="?", default="follower",
                    help="follower | leader | /dev/tty.usbmodemXXXX")
parser.add_argument("--motor", default="gripper")
parser.add_argument("--all", action="store_true",
                    help="release EVERY joint, not just --motor. Use when the whole arm "
                         "is locked — e.g. after growbot-arm/arm.py, whose exit freezes "
                         "the pose and thereby engages torque on all six.")
parser.add_argument("--open", action="store_true",
                    help="drive the jaws open in small steps before releasing torque")
parser.add_argument("--step", type=int, default=10, help="counts per opening step")
parser.add_argument("--counts", type=int, default=400,
                    help="total counts to open (default 400)")
args = parser.parse_args()

port = {"follower": FOLLOWER_PORT, "leader": LEADER_PORT}.get(args.arm, args.arm)
bus = open_bus(port)

def state():
    """Torque is TWO registers on these servos, not one. Report both."""
    out = {}
    for reg in ("Torque_Enable", "Lock", "Present_Position", "Present_Load"):
        try:
            out[reg] = bus.read(reg, args.motor, normalize=False)
        except Exception:
            out[reg] = None
    return out


try:
    before = state()
    print(f"{port}  {args.motor}")
    print(f"  before: Torque_Enable={before['Torque_Enable']}  Lock={before['Lock']}  "
          f"pos={before['Present_Position']}  load={before['Present_Load']}")

    if args.open:
        # Step outward rather than commanding the target in one jump: a single
        # large goal on a jammed joint is how you stall it again.
        pos = before["Present_Position"]
        bus.enable_torque(args.motor)
        goal, target = pos, pos + args.counts
        while goal < target:
            goal = min(goal + args.step, target)
            bus.write("Goal_Position", args.motor, goal, normalize=False)
            time.sleep(0.04)
        now = bus.read("Present_Position", args.motor, normalize=False)
        print(f"  opened to {now} (commanded {goal})")
        if abs(now - pos) < args.counts // 4:
            print("  ⚠️  barely moved — likely latched. Power cycle:")
            print("      USB out -> 12V out -> 12V in -> USB in")

    # Use lerobot's own API. It clears Torque_Enable AND Lock; writing only
    # Torque_Enable by hand leaves Lock set and the joint still held.
    targets = JOINTS if args.all else [args.motor]
    stuck = []
    for motor in targets:
        bus.disable_torque(motor)

    for motor in targets:
        te = lk = None
        try:
            te = bus.read("Torque_Enable", motor, normalize=False)
            lk = bus.read("Lock", motor, normalize=False)
        except Exception:
            pass
        ok = te == 0 and lk == 0
        if not ok:
            stuck.append(motor)
        if args.all:
            print(f"  {motor:15s} Torque_Enable={te} Lock={lk}  {'released' if ok else 'STILL HELD'}")
        else:
            print(f"  after:  Torque_Enable={te}  Lock={lk}")

    if not stuck:
        print(f"\n✅ torque is genuinely OFF on {len(targets)} joint(s) "
              "(both registers confirmed).")
        print("   It will STILL feel stiff — the follower is 1/345 geared and is hard")
        print("   to backdrive unpowered even when released. That stiffness is normal")
        print("   and is NOT the servo holding. Move it firmly by hand.")
    else:
        print(f"\n❌ registers did not clear on {stuck} — latched. Power cycle:")
        print("   USB out -> 12V out -> 12V in -> USB in")

except Exception as exc:
    print(f"\nwrite failed: {exc}\n")
    print("That is the signature of a LATCHED overload. Software cannot clear it.")
    print("Power cycle the follower:  USB out -> 12V out -> 12V in -> USB in")

finally:
    bus.disconnect(disable_torque=False)
