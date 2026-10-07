"""Measure gripper LOAD and CURRENT — the two numbers the policy cannot see.

`SOFollower.get_observation()` reads only `Present_Position` (so_follower.py:183),
so a trained policy is blind to how hard it is squeezing. The servo is not: the
STS3215 reports `Present_Load` (addr 60) and `Present_Current` (addr 69) the
whole time. This makes those readable, and turns "squeeze gently" into a number.

WHY IT EXISTS. On Aug 8, 2026 ten scissors episodes were recorded with sustained
over-grip and the gripper latched `[RxPacketError] Overload error!` on motor id 6,
killing the session. The recorded `action` is the COMMANDED position, so a policy
trained on that data learns to command the same stall. The SO-101 leader gives no
force feedback, so the operator cannot feel any of this through the trigger.
The servo can. That asymmetry is the whole reason for this file.

⚠️ IT NEEDS THE PORT TO ITSELF. The Feetech bus is half-duplex serial: two
processes reading the same /dev/tty interleave packets and corrupt each other.
STOP any lerobot-teleoperate / lerobot-record run before starting this.

TWO MODES
---------
Passive (default) — pure read, writes nothing:

    conda activate lerobot
    python watch_load.py

    Only meaningful while the servo is already holding torque against something.
    With torque off it reads ~0, because Present_Load reflects applied PWM.
    Use it to sanity-check the register, not to pick a threshold.

Guided closure (--close) — the one that actually answers the question:

    python watch_load.py --close --max-load 300 --csv scissors.csv

    Enables gripper torque, then steps the jaws closed a few counts at a time,
    printing load and current at each step. Place the object between the jaws
    first. It STOPS on its own at --max-load, and you can Ctrl-C the instant the
    object looks stressed. Nothing is ever commanded in one jump, so it cannot
    reproduce the Aug 8 stall.

WHAT TO WRITE DOWN, per object (scissors, screwdriver, pen, reading glasses):
  - |load| where the object is held FIRMLY but not deforming  -> your threshold
  - |load| at the first sign of deformation                    -> your ceiling
A safe grasp threshold sits below the lower of those, with margin. If no single
value is both firm on the scissors and safe on the glasses, that is not a failure
of this script — that is the load-sensing argument becoming a measurement, and it
belongs in the gap analysis.

THE NUMBERS THAT BOUND YOU (so_follower.py:168-171, gripper only):
    Max_Torque_Limit   500   50% of max torque
    Protection_Current 250   exceed this and the servo LATCHES an overload
    Overload_Torque     25   torque it drops to once overloaded
A latched servo refuses writes — including Torque_Enable=0 — so clearing it
needs a FULL power cycle: USB out, 12V out, 12V in, then USB.

ENCODING. `Present_Load` is sign-magnitude with the sign bit at position 10
(feetech/tables.py, STS_SMS_SERIES_ENCODINGS_TABLE). lerobot's bus decodes that
for you, so the value printed is already signed — magnitude 0-1023, sign = which
direction the servo is pushing. Do not mask bits yourself.
"""

import argparse
import time

from _arms import FOLLOWER_PORT, LEADER_PORT, open_bus

PROTECTION_CURRENT = 250  # so_follower.py:170 — the latch threshold
LOAD_FULL_SCALE = 1023

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("arm", nargs="?", default="follower",
                    help="follower | leader | /dev/tty.usbmodemXXXX")
parser.add_argument("--motor", default="gripper", help="which joint to watch")
parser.add_argument("--hz", type=float, default=8.0, help="samples per second")
parser.add_argument("--csv", metavar="PATH", help="append samples to this CSV")
parser.add_argument("--close", action="store_true",
                    help="enable torque and step the jaws closed until --max-load")
parser.add_argument("--max-load", type=int, default=300,
                    help="stop closing at this |load| (default 300)")
parser.add_argument("--step", type=int, default=4,
                    help="raw position counts per closing step (default 4)")
parser.add_argument("--max-current", type=int, default=150,
                    help=f"stop at this current; {PROTECTION_CURRENT} is the latch (default 150)")
args = parser.parse_args()

if args.max_current >= PROTECTION_CURRENT:
    parser.error(f"--max-current must stay under the {PROTECTION_CURRENT} latch threshold")

if args.max_load >= LOAD_FULL_SCALE:
    parser.error("--max-load must be well under 1023; 250-400 is the useful band")

port = {"follower": FOLLOWER_PORT, "leader": LEADER_PORT}.get(args.arm, args.arm)
bus = open_bus(port)


def try_read(register):
    """Not every servo series exposes every register; degrade instead of dying."""
    try:
        return bus.read(register, args.motor, normalize=False)
    except Exception:
        return None


has_current = try_read("Present_Current") is not None

csv_file = None
if args.csv:
    csv_file = open(args.csv, "a")
    if csv_file.tell() == 0:
        csv_file.write("t,pos_raw,load,current\n")

print(f"{port}  motor={args.motor}")
if not has_current:
    print("note: Present_Current unavailable on this motor; showing load only.")

closing = False
if args.close:
    print(f"\nCLOSING MODE. step {args.step}, stop at |load| {args.max_load} "
          f"or current {args.max_current}")
    print("Place the object between the jaws now. Ctrl-C stops immediately.")
    input("press ENTER to begin closing... ")
    # Read the START POSITION *AFTER* the prompt. Placing the object moves the
    # jaws, and a position captured before the prompt is stale by however far you
    # pushed them — which turns the first "step" into a slam of that whole
    # distance. Measured Aug 9, 2026: stale start 2056 vs actual 2334 produced a
    # 286-count first move and overshot the load ceiling by 67%.
    start = try_read("Present_Position")
    if start is None:
        raise SystemExit("cannot read Present_Position — is 12V on, and applied BEFORE USB?")
    print(f"start pos {start} (re-read after placement)")
    bus.enable_torque(args.motor)
    bus.write("Goal_Position", args.motor, start, normalize=False)
    closing = True
    stalled_for = 0
else:
    print("\nPASSIVE MODE — writing nothing. Reads ~0 unless the servo is already")
    print("holding torque against something. Use --close to pick a threshold.")
print("Ctrl-C to stop.\n")

peak_load = 0
peak_current = 0
prev_pos = None
t0 = time.time()
stop_reason = "interrupted"

try:
    while True:
        pos = try_read("Present_Position")
        load = try_read("Present_Load")
        current = try_read("Present_Current") if has_current else None

        if load is None or pos is None:
            print("  unreadable — is the arm powered and connected?")
            time.sleep(1.0)
            continue

        mag = abs(load)
        peak_load = max(peak_load, mag)

        bar = "#" * min(int(mag / LOAD_FULL_SCALE * 40), 40)
        line = f"  pos {pos:5d}  load {load:+6d} |{mag:4d}|  peak {peak_load:4d}  {bar:<40}"

        if current is not None:
            peak_current = max(peak_current, current)
            pct = current / PROTECTION_CURRENT * 100
            line += f"  cur {current:4d} ({pct:3.0f}% of latch)"
            if pct >= 70:
                line += "  <-- approaching latch"

        print(line)

        if csv_file:
            csv_file.write(f"{time.time() - t0:.3f},{pos},{load},"
                           f"{current if current is not None else ''}\n")
            csv_file.flush()

        if closing:
            # Every stop condition is checked BEFORE commanding the next step, so
            # the jaws never advance into a state already known to be unsafe.
            if mag >= args.max_load:
                stop_reason = f"reached --max-load ({mag} >= {args.max_load})"
                break
            if current is not None and current >= args.max_current:
                stop_reason = (f"current {current} reached --max-current "
                               f"{args.max_current} ({current / PROTECTION_CURRENT * 100:.0f}% of latch)")
                break

            # Step from the LIVE position, never from an accumulator. If the jaws
            # are blocked, `pos` stops changing, so the goal stops advancing and
            # the servo is never asked for more than one step of extra travel.
            # An accumulator drifts away from reality and slams the difference.
            goal = pos - args.step
            if goal <= 0:
                stop_reason = "jaws reached the closed end of travel with no contact"
                break
            bus.write("Goal_Position", args.motor, goal, normalize=False)

            # Blocked but not loaded = something is wrong (jam, no torque, stale
            # calibration). Contact shows up as load, not as silence.
            moved = prev_pos is None or abs(pos - prev_pos) >= 2
            stalled_for = 0 if moved else stalled_for + 1
            if stalled_for >= 12:
                stop_reason = f"jaws stopped moving at pos {pos} with only |load| {mag}"
                break
        prev_pos = pos

        time.sleep(1.0 / args.hz)

except KeyboardInterrupt:
    stop_reason = "interrupted by you"

finally:
    if closing:
        # Freeze where we are. Do NOT disable torque: that drops whatever is held.
        held = try_read("Present_Position")
        if held is not None:
            bus.write("Goal_Position", args.motor, held, normalize=False)
    print(f"\nstopped: {stop_reason}")
    print(f"peak |load| {peak_load}", end="")
    if has_current:
        print(f"   peak current {peak_current} "
              f"({peak_current / PROTECTION_CURRENT * 100:.0f}% of the {PROTECTION_CURRENT} latch)",
              end="")
    print()
    if closing:
        print("\njaws are STILL HOLDING. Torque was left ON deliberately — disabling")
        print("it here would drop whatever is in the gripper. To open them:")
        print("    python release_gripper.py           # limp, open by hand")
        print("    python release_gripper.py --open    # drive open, then limp")
    if csv_file:
        csv_file.close()
        print(f"wrote {args.csv}")
    bus.disconnect(disable_torque=False)
