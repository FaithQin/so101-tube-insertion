"""Pre-launch gate for a scored trial. Every check ABORTS; none of them warn.

The Aug 29 2026 session lost roughly an hour to checks that only printed:
`home_arm`'s residual warning was narrated three times and launched through
(the elbow was at 74.1 deg, below the v2 training floor, and a deterministic
replay then missed the tube entirely), and the tube placement delta was echoed
into a summary without being enforced. The lesson, written into this module:
if a condition should stop a launch, it stops it here; if it should not, it
does not belong in this file.

    from preflight import gate
    ok, reasons = gate(scene_text, home_text, ping_text)
    if not ok:
        abort("; ".join(reasons))

Parsers deliberately raise on missing input rather than returning a default.
A check that silently degrades to "pass" when its input is malformed is worse
than no check, because it reads as green in the log.
"""

import ast
import math
import os
import re
from pathlib import Path

import home_arm_shift
import home_gate

# Tube placement tolerance in reference pixels. The certified scene baseline is
# ~2 px (cap sub-pixel, rack ~2 px); single digits are fine, tens of px mean the
# physical thing moved. The Aug 29 episode that ran at 10.2 px is the reason
# this is enforced rather than printed.
TUBE_DELTA_MAX_PX = 8.0

# Where the cap must be when the start is NOT the D mark. Sep 14 2026, protocol §7 15:26: the PI
# redefined S1 ROT-45 as the tube rotated 45° about its MIDPOINT, tip toward the rack, so the cap
# leaves the mark; the target is her placement at 15:08, measured by scene_check
# (`live=( 381.6, 320.2)`), in REFERENCE px. Selected by NAME through CAPSTONE_TUBE_TARGET, which
# run_pair.sh exports for S1 pairs; the same 8 px band applies around the target. Without a target
# the cap is gated on the mark, as before. An unknown name aborts -- it never defaults to the mark:
# P02 trial 1 ran on a horizontal tube under an S1 pair (15:03) with every gate green.
TUBE_TARGETS = {"S1-ROT45": (382.0, 320.0)}

# Elbow is the tracked reliability risk: two bus-drops, a 69 C brownout, and
# published misbehavior at 71 C with no trustworthy built-in shutdown. 60 C is
# the standing gate; it hit 58 C by the end of Block 1 on Aug 29.
ELBOW_TEMP_MAX_C = 60

# ...but the brownout is a property of the SERVO, not of the elbow, and this gate read only the
# elbow. Sep 6 2026, measured: with the arm parked holding the v4 home pose during software work,
# wrist_flex drew a steady 57 current at load 260 and reached 68 C -- one degree under the 69 C
# brownout on the ledger -- while the elbow sat at 57 C. The gate would have passed that trial and
# named the elbow's 57 C as the reason it was safe. Every joint is now gated at the same limit.
# the project notes' pre-session ritual already says why this happens: "the elbow hold has a THERMAL
# BUDGET -- home right before the run, never park the arm holding during long diagnostics."
JOINT_TEMP_MAX_C = ELBOW_TEMP_MAX_C

# ...and "every joint" has to mean every SERVO, not every line the parser managed to read. Sep 7
# night audit, reproduced Sep 12 2026 on the Sep 6 16:14 ping receipt: a joint whose line said
# "no response" (a servo off the bus, ping_motors.py:97) or "NoneC" (its temperature read raised,
# ping_motors.py:105-106 and :114) fell out of the parse, the gate demanded only the elbow, and
# the CLI printed PREFLIGHT PASS on a servo nobody had read. An unread servo is not a cool one.
# This is the list ping_motors walks (_arms.JOINTS, ping_motors.py:88), written out rather than
# imported because _arms imports lerobot's bus at module top (_arms.py:10-11);
# tests/test_preflight.py reads _arms.py with ast and holds the two equal.
SERVO_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")

# Scored trials run 45 s, not the probe's 60 s: a v2 scored trial holds 875
# frames (~45 s at the measured rate) and the locked protocol says "Max 45 s".
# Success is defined at rollout end, so episode length is part of the
# definition and must match the baseline the primary is compared against.
SCORED_EPISODE_TIME_S = 45


def parse_tube_delta(scene_text: str) -> float:
    # Same LINE only. With re.S this pattern ran on from a "tube ...: NOT FOUND" line into the
    # rack's "|d|=1.5" and gated a missing tube green at 1.5 px (found Sep 6 2026 by the
    # carry-start tests; the Aug 30 arm-over-the-tube case had no rack line to fall into).
    m = re.search(r"tube \(blue cap\)[^\n]*?\|d\|=\s*([\d.]+)", scene_text)
    if not m:
        raise ValueError("no tube |d| in scene_check output — cannot gate placement")
    return float(m.group(1))


def parse_tube_live(scene_text: str) -> tuple[float, float]:
    """The live cap centroid in reference px, from the tube line. Same-line match, like parse_tube_delta."""
    m = re.search(r"tube \(blue cap\)[^\n]*?live=\(\s*([\d.]+),\s*([\d.]+)\)", scene_text)
    if not m:
        raise ValueError("no live tube centroid in scene_check output — cannot gate placement against a target")
    return float(m.group(1)), float(m.group(2))


def parse_rack_delta(scene_text: str) -> float:
    """The rack's displacement, for a PRE-GRASPED start (Sep 6 2026). The tube is in the jaws
    then, so its placement mark says nothing; the rack must be where training had it."""
    m = re.search(r"rack \(orange funnel\)[^\n]*?\|d\|=\s*([\d.]+)", scene_text)
    if not m:
        raise ValueError("no rack |d| in scene_check output — cannot gate the rack for a carry start")
    return float(m.group(1))


def parse_settled_pose(home_text: str) -> dict:
    m = re.search(r"settled:\s*(\{[^}]*\})", home_text)
    if not m:
        raise ValueError("no 'settled:' line in home_arm output — cannot gate the pose")
    return {k: float(v) for k, v in ast.literal_eval(m.group(1)).items()}


def parse_elbow_temp(ping_text: str) -> int:
    m = re.search(r"elbow_flex.*?(\d+)C", ping_text)
    if not m:
        raise ValueError("no elbow temperature in ping_motors output — cannot gate thermal")
    return int(m.group(1))


def parse_joint_temps(ping_text: str) -> dict:
    """Every READABLE joint's temperature from ping_motors output, by name. Same line shape the
    elbow parser uses (`  id 3 elbow_flex      pos 2948  load 80  cur 0   50C  126  torque 0  ok`),
    matched per LINE so a joint cannot borrow the next one's reading. A joint with no reading is
    left out of the dict, never defaulted -- gate() refuses it by name against SERVO_JOINTS.

    A joint named on two lines raises. `ping_motors.py` with no argument pings the LEADER too,
    after the follower and under the same joint names (ping_motors.py:127-137), and keeping the
    last line let the leader's reading stand in for the follower's: Sep 12, a follower wrist at
    68C read as the leader's 52C and gated green. The runners ping `follower` only
    (run_scored_trial.sh:93), so a hand-made capture is the way in -- refused all the same."""
    out, seen = {}, set()
    for line in ping_text.splitlines():
        m = re.search(r"\bid\s+\d+\s+([a-z_]+)\b(.*)", line)
        if not m:
            continue
        joint = m.group(1)
        if joint in seen:
            raise ValueError(
                f"{joint} appears twice in ping_motors output (both arms captured?) — cannot tell "
                "which reading is the follower's; cannot gate thermal"
            )
        seen.add(joint)
        t = re.search(r"(\d+)C", m.group(2))
        if t:
            out[joint] = int(t.group(1))
    if not out:
        raise ValueError("no joint temperatures in ping_motors output — cannot gate thermal")
    return out


def is_main_checkout(path) -> bool:
    """Refuse to run from a git worktree copy.

    Aug 27 INFRA #3: a probe was invalidated because the wrapper it ran came
    from a stale `.worktrees/` copy. Project state lives in the main
    checkout; a worktree's tools can silently lag it.
    """
    return ".worktrees" not in str(Path(path).resolve())


def gate(
    scene_text: str, home_text: str, ping_text: str, dataset: str = home_gate.DEFAULT_DATASET,
    tube_target: tuple | None = None,
) -> tuple[bool, list]:
    """Return (ok, reasons). Reports EVERY failure, not just the first — a
    one-at-a-time gate costs an extra trip to the bench per problem.

    `dataset` selects WHICH training distribution the pose is gated against.
    v2 and v3 differ materially (elbow 75.47-84.44 vs 78.11-90.42), so running a
    v3 policy against v2 bounds both rejects valid starts and admits invalid
    ones."""
    reasons = []

    if dataset in home_gate.GRIPPER_ON_TUBE_DATASETS:
        # PRE-GRASPED start (S2, a modified task): the tube is in the jaws, so its placement
        # mark is meaningless and the check is the RACK instead -- judged from the frame-0
        # pose the runner photographs before homing to the carry pose (arm clear, rack
        # visible). A rack the check cannot see raises, and the CLI turns that into an abort.
        delta = parse_rack_delta(scene_text)
        if delta > TUBE_DELTA_MAX_PX:
            reasons.append(f"placement: rack |d|={delta:.1f} px > {TUBE_DELTA_MAX_PX:.0f} px")
    elif tube_target is not None:
        # A stratum whose start is off the mark (S1 ROT-45 since Sep 14 15:26): the cap is gated
        # against THAT target, same band. The name is looked up for the message only.
        tx, ty = float(tube_target[0]), float(tube_target[1])
        name = next((k for k, v in TUBE_TARGETS.items() if tuple(v) == (tx, ty)), "target")
        lx, ly = parse_tube_live(scene_text)
        delta = math.hypot(lx - tx, ly - ty)
        if delta > TUBE_DELTA_MAX_PX:
            reasons.append(f"placement: tube |d|={delta:.1f} px from the {name} target ({tx:.0f}, {ty:.0f}) "
                           f"> {TUBE_DELTA_MAX_PX:.0f} px")
    else:
        delta = parse_tube_delta(scene_text)
        if delta > TUBE_DELTA_MAX_PX:
            reasons.append(f"placement: tube |d|={delta:.1f} px > {TUBE_DELTA_MAX_PX:.0f} px")

    # Gate the pose the POLICY will observe, not the one home_arm leaves behind.
    # SOFollower.connect() shifts wrist_roll by about -1.62 deg (measured Aug 30),
    # and home_arm now pre-compensates for that — so its settled pose is
    # deliberately outside the training band and only lands inside after connect.
    # Judging the pre-connect pose rejects a correct setup and, worse, accepts an
    # uncompensated one that will drift out of band the moment the rollout starts.
    pose = parse_settled_pose(home_text)
    # A joint the settled line does not carry is not a joint in band. home_gate.check_pose judges
    # the joints the pose HAS (home_gate.py:203-205), so until Sep 12 a settled line with no
    # elbow_flex in it gated green. home_arm prints all six or no settled line at all
    # (home_arm.py:249-254), so only a truncated or hand-edited capture gets here -- and the gate
    # must not be the thing that trusts it. shoulder_lift is exempt from BOUNDS at frame 0, not
    # from being read.
    unread_pose = [j for j in home_gate.support_for(dataset) if j not in pose]
    if unread_pose:
        reasons.append(f"home: {', '.join(unread_pose)} not in the settled line — unread, not in band")
    pose = home_arm_shift.predict_post_connect(pose)
    ok_pose, offenders = home_gate.check_pose(pose, dataset=dataset)
    if not ok_pose:
        reasons.append(f"home: {home_gate.describe(offenders)}")

    temps = parse_joint_temps(ping_text)
    hot = {j: t for j, t in temps.items() if t > JOINT_TEMP_MAX_C}
    if hot:
        worst = max(hot, key=hot.get)
        rest = "".join(f"; {j} {t}C" for j, t in sorted(hot.items(), key=lambda kv: -kv[1]) if j != worst)
        reasons.append(f"thermal: {worst} temp {hot[worst]}C > {JOINT_TEMP_MAX_C}C{rest}")
    # Every servo, not just the elbow (this line used to raise for a missing elbow and wave the
    # other five through). The reason says "thermal" on purpose: the runners key their repairs on
    # the joint NAME -- run_scored_trial.sh:109-121 reseats the jaw on "gripper" and re-homes on
    # "elbow_flex" -- and only "thermal" short-circuits them as human-only first
    # (run_scored_trial.sh:102-104). A servo off the bus needs the power cycle, not a reseat.
    unread = [j for j in SERVO_JOINTS if j not in temps]
    if unread:
        reasons.append(f"thermal: {', '.join(unread)} temperature unreadable — an unread servo is not a cool one")

    return (not reasons), reasons


if __name__ == "__main__":
    import sys

    try:
        scene, home, ping = (Path(p).read_text() for p in sys.argv[1:4])
        ds = sys.argv[4] if len(sys.argv) > 4 else home_gate.DEFAULT_DATASET
        target = None
        tname = os.environ.get("CAPSTONE_TUBE_TARGET")
        if tname:
            if tname not in TUBE_TARGETS:
                print(f"PREFLIGHT ABORT — unknown tube target '{tname}' (CAPSTONE_TUBE_TARGET); known: "
                      f"{sorted(TUBE_TARGETS)}")
                sys.exit(1)
            target = TUBE_TARGETS[tname]
        ok, reasons = gate(scene, home, ping, dataset=ds, tube_target=target)
    except (ValueError, OSError) as exc:
        # A malformed or missing capture aborts. Never degrade to "pass": a
        # check that reads green on broken input is worse than no check.
        print(f"PREFLIGHT ABORT — could not evaluate a gate: {exc}")
        sys.exit(1)
    print(f"PREFLIGHT PASS ({ds})" if ok else "PREFLIGHT ABORT — " + "; ".join(reasons))
    sys.exit(0 if ok else 1)
