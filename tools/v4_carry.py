#!/usr/bin/env python
"""The PRE-GRASPED start pose (`v4-carry`): derived from the v4 training set, not guessed.

    python tools/v4_carry.py            # the table, the k spread, the exclusions, the snippets

WHY (Sep 6 2026, eval redesign, handoff §3 item 2 / §6): S2 PRE-GRASPED is the stratum where
the tube starts IN THE JAWS so every trial reaches contact -- the only place the load channels
can matter. It is a MODIFIED TASK (the pick is done by hand) and is labelled so everywhere the
pose appears: home_arm.HOME_POSES["v4-carry"], TRAIN_HOME_V4_CARRY in rollout_30hz_stale_ok.py,
home_gate.FRAME0["v4-carry"]. tests/test_v4_carry.py re-derives every number from the parquet.

METHOD, over the 50 TRAINING episodes of so101-tube-insert-v4 (tests/conftest.v4_training_episodes;
hold-out and voided takes excluded): carry frame = grasp_index + k, with grasp_index imported from
tools/v4_audit.py (first pick-site frame where a previously open jaw reads <= 16) and k the first
frame after it where shoulder_lift has moved >= LIFT_RISE_DEG (2 deg) TOWARD LIFTED.

SIGN CHECK -- measured, because "risen 2 deg above" is ambiguous for a negative-signed joint.
shoulder_lift is -105.34 at frame 0 (folded home, against its stop) and +45.35 on average at the
grasp frame (the arm reached DOWN to the table). Lifting the tube brings the value back toward
home, i.e. numerically DOWN. Both readings were computed from the parquet:

    lift >= lift_at_grasp + 2   fires in  0 / 50 episodes
    lift <= lift_at_grasp - 2   fires in 50 / 50 episodes   <- LIFT_DIRECTION = -1

Excluded episodes: 0 (no episode lacks a grasp, none fails to lift). The exclusion path exists,
is counted, and is tested on synthetic episodes -- silently dropping episodes is how a table
becomes a fiction.

k spread: [6, 20] frames = 0.30-1.00 s at 20 FPS, mean 14.56, sd 2.58, median 15.

CARRY POSE (mean / sd / min / max):

    shoulder_pan   67.09   3.17   61.14   77.93
    shoulder_lift  45.35   7.20   23.43   57.89
    elbow_flex    -35.84  13.68  -59.91   10.33
    wrist_flex     78.49   7.08   59.78   92.75
    wrist_roll     49.66   7.78   25.71   60.35
    gripper        13.16   0.58   12.13   15.42     v4 frame 0: 1.11 (closed on NOTHING)

The spread is wide on purpose: the pick site varies with the tube's start zone, so the carry
pose is a distribution the mean sits in the middle of. The elbow sd of 13.68 deg is the data.

THE GRIPPER (the silent bug this module exists to prevent). home_gate gates every frame-0
dataset's gripper on (0, GRIPPER_CLOSED_MAX=3.0) because at frame 0 the jaw is closed on
nothing. At the carry pose it is closed ON THE TUBE and reads the tube's width, 12.13-15.42.
A 0-3.0 bound would have refused every S2 start on the bench. `v4-carry` is therefore gated on
its observed carry range, which still refuses an empty jaw (~1.1) and an open one (>= 16).

But position alone cannot see the OTHER empty jaw: `home_arm --pose v4-carry` with no tube
parks the empty jaw at 13.2, squarely inside that band. What separates the two is contact
LOAD: training carry gripper |Present_Load| is 98-120 raw (mean 118.0, sd 5.7); the follower
patch measured the free-motion bias at 56-80 and frame-0 closed-on-nothing reads 52-92. So the
scripted close verifies BOTH position and load, and home_arm refuses `--pose v4-carry` without
`--close-on-tube` -- the pose with an empty jaw is not a trial start and must not print a
`settled:` line for preflight to pass.

THE SCRIPTED CLOSE mirrors the production mechanism that produced training's carry load:
tools/so_follower_load.patch bounds every gripper command to `present - lead`, lead 4.0 in free
air and 1.5 once |load| >= 115 (released below 90), so demand never pegs Max_Torque_Limit and
the 2 s Protection_Time latch never fires. Same numbers here (tested against the patch text).
The jaw is opened to OPEN_FOR_PLACEMENT (35; training grasps opened to 24.8-38.8), the operator
places the tube and presses ENTER, the jaw closes under the bounded lead until it stalls, and
the settled (position, load) is verified. Nothing here touches a bus at import; the only bus
call sites take the bus as an argument, so the suite exercises the parameters and every
refusal on a fake bus. Replay validation of the pose needs the arm -- morning, with Faith.

shoulder_lift at the carry pose (+45) is inside the goal clamp and actively holds the extended
arm -- there is no stop to settle against. So for `v4-carry` it is GATED (home_gate) and NOT
released (settle_joints_for returns () and both homing paths route through it). Releasing it
there would drop the arm, holding the tube, onto the rig.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import home_gate  # noqa: E402  (pure: tables and bounds, no hardware)

TAG = "v4-carry"
FPS = 20
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# "Lifted" = shoulder_lift moves from ~+45 (grasp, reached down) back toward -105 (home): DOWN.
LIFT_DIRECTION = -1
LIFT_RISE_DEG = 2.0

# Documented derivation bookkeeping; the tests re-derive these from the parquet.
EXCLUDED_EPISODES = 0
K_SPREAD = {"min": 6, "max": 20, "mean": 14.56, "sd": 2.58}

# --- scripted close ---------------------------------------------------------------------
# Mirrors tools/so_follower_load.patch (send_action): the bounded closing lead and its contact
# hysteresis. tests/test_v4_carry.py reads these back out of the patch text.
CLOSE_LEAD_FREE = 4.0
CLOSE_LEAD_CONTACT = 1.5
CONTACT_ENGAGE_LOAD = 115.0
CONTACT_RELEASE_LOAD = 90.0
# A settled jaw must show contact load. 90 is the patch's own "contact lost" threshold: above the
# 56-80 free-air bias, below every training carry frame (min |load| 98).
CARRY_LOAD_MIN = CONTACT_RELEASE_LOAD
# Where the jaw is opened for the operator to place the tube (training pre-grasp openings 24.8-38.8).
OPEN_FOR_PLACEMENT = 35.0
OPEN_TOL = 3.0
# The close aims BELOW the tube, as the leader trigger did (action gripper 0.0-1.14 at carry);
# the bounded lead, not the target, sets the grip force.
CLOSE_TARGET = 1.0
CLOSE_HZ = 20
CLOSE_MAX_S = 4.0
STALL_EPS = 0.05          # normalized gripper units per tick
STALL_TICKS = 5           # 0.25 s without motion = the jaw has met something (or nothing)
SETTLE_S = 0.5            # the patch's relief window; lets load settle before it is judged


class CarryRefused(RuntimeError):
    """The jaw is not holding the tube at the carry pose. Abort the trial start."""


@dataclass(frozen=True)
class CarryGrip:
    position: float   # normalized gripper reading after the close
    load: float       # |Present_Load|, raw


# ---------------------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------------------
def carry_index(pan, lift, grip, *, rise_deg: float = LIFT_RISE_DEG):
    """(grasp_index, k) for one episode, or (None, reason) when it must be EXCLUDED."""
    import numpy as np

    import v4_audit

    pan, lift, grip = (np.asarray(a, dtype=float) for a in (pan, lift, grip))
    g = v4_audit.grasp_index(pan, grip)
    if g is None:
        return None, "no grasp (v4_audit.grasp_index is None)"
    lifted = np.where(LIFT_DIRECTION * (lift[g:] - lift[g]) >= rise_deg)[0]
    if not len(lifted):
        return None, f"lift never moved {rise_deg:g} deg toward lifted after the grasp"
    return int(g), int(lifted[0])


def settle_joints_for(pose_tag: str, default: tuple) -> tuple:
    """Joints whose torque may be released after homing to `pose_tag`. Frame-0 poses release
    shoulder_lift to settle against its stop; the carry pose releases NOTHING (see docstring)."""
    return () if pose_tag == TAG else tuple(default)


def derive(root: Path, episodes) -> dict:
    """Carry-pose statistics over `episodes` of the dataset at `root` (parquet only)."""
    import glob

    import numpy as np
    import pandas as pd

    files = sorted(glob.glob(str(Path(root) / "data" / "chunk-*" / "file-*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet under {root}")
    df = pd.concat(
        pd.read_parquet(f, columns=["frame_index", "episode_index", "observation.state"]) for f in files
    )
    episodes = sorted(int(e) for e in episodes)
    used, excluded, k_of, lift_at_grasp, rise_fires, preopen, gload, carry = [], {}, {}, {}, 0, {}, {}, []
    for e in episodes:
        d = df[df["episode_index"] == e].sort_values("frame_index")
        s = np.stack(d["observation.state"].to_numpy())
        g, k = carry_index(s[:, 0], s[:, 1], s[:, 5])
        if g is None:
            excluded[e] = k
            continue
        if np.any(s[g:, 1] >= s[g, 1] + LIFT_RISE_DEG):   # the OTHER reading, counted for the sign check
            rise_fires += 1
        used.append(e)
        k_of[e] = k
        lift_at_grasp[e] = float(s[g, 1])
        preopen[e] = float(s[:g, 5].max())
        gload[e] = float(abs(s[g + k, 11])) if s.shape[1] >= 12 else float("nan")
        carry.append(s[g + k, :6])
    c = np.stack(carry) if carry else np.zeros((0, 6))
    ks = np.array(list(k_of.values()), dtype=float)
    return {
        "fps": FPS,
        "n_requested": len(episodes),
        "used": used,
        "excluded": excluded,
        "k": k_of,
        "k_stats": {
            "min": int(ks.min()), "max": int(ks.max()), "mean": float(ks.mean()), "sd": float(ks.std()),
            "median": float(np.median(ks)), "min_s": float(ks.min() / FPS), "max_s": float(ks.max() / FPS),
        } if len(ks) else {},
        "lift_at_grasp_mean": float(np.mean(list(lift_at_grasp.values()))) if lift_at_grasp else float("nan"),
        "n_numeric_rise_fires": rise_fires,
        "pre_grasp_open": {"min": min(preopen.values()), "max": max(preopen.values())} if preopen else {},
        "gripper_load_abs": {
            "min": float(np.nanmin(list(gload.values()))), "mean": float(np.nanmean(list(gload.values()))),
            "max": float(np.nanmax(list(gload.values()))),
        } if gload else {},
        "n_episodes": len(used),
        "joints": JOINTS,
        "mean": dict(zip(JOINTS, c.mean(0))),
        "std": dict(zip(JOINTS, c.std(0))),
        "min": dict(zip(JOINTS, c.min(0))),
        "max": dict(zip(JOINTS, c.max(0))),
    }


# ---------------------------------------------------------------------------------------
# Scripted close (bus passed in; never opened here)
# ---------------------------------------------------------------------------------------
def bounded_goal(present: float, target: float, lead: float) -> float:
    """The patch's rule: a close may never be commanded more than `lead` below `present`."""
    return max(target, present - lead)


def contact_state(load_abs: float, in_contact: bool) -> bool:
    if load_abs >= CONTACT_ENGAGE_LOAD:
        return True
    if load_abs < CONTACT_RELEASE_LOAD:
        return False
    return in_contact


def _read(bus) -> tuple[float, float]:
    pos = float(bus.read("Present_Position", "gripper", normalize=True))
    load = abs(float(bus.read("Present_Load", "gripper", normalize=False)))
    return pos, load


def close_on_tube(bus, *, sleep=None, hz: int = CLOSE_HZ, max_s: float = CLOSE_MAX_S) -> CarryGrip:
    """Close the jaw on whatever is between it under the bounded lead until it stalls.
    Returns the settled reading; judge it with verify_carry_grip. Torque stays ON."""
    import time

    sleep = sleep or time.sleep
    in_contact, stalled, last = False, 0, None
    for _ in range(int(max_s * hz)):
        present, load = _read(bus)
        in_contact = contact_state(load, in_contact)
        lead = CLOSE_LEAD_CONTACT if in_contact else CLOSE_LEAD_FREE
        bus.write("Goal_Position", "gripper", bounded_goal(present, CLOSE_TARGET, lead), normalize=True)
        stalled = stalled + 1 if last is not None and abs(present - last) < STALL_EPS else 0
        last = present
        if stalled >= STALL_TICKS:
            break
        sleep(1 / hz)
    else:
        raise CarryRefused(f"jaw did not settle within {max_s:g} s (still moving) -- not holding anything steady")
    sleep(SETTLE_S)
    present, load = _read(bus)
    return CarryGrip(position=present, load=load)


def verify_carry_grip(grip: CarryGrip) -> None:
    """Gate, don't narrate: raise unless the jaw is on the tube inside the training carry band."""
    lo, hi = home_gate.support_for(TAG)["gripper"]
    if grip.position <= home_gate.GRIPPER_CLOSED_MAX:
        raise CarryRefused(
            f"EMPTY jaw: gripper {grip.position:.2f} <= {home_gate.GRIPPER_CLOSED_MAX} -- closed on nothing; "
            f"the tube is not in the jaws"
        )
    if not lo <= grip.position <= hi:
        raise CarryRefused(
            f"gripper {grip.position:.2f} outside the training carry band [{lo:.2f}, {hi:.2f}] -- "
            f"the jaw did not close on the tube the way training did (misplaced tube?)"
        )
    if grip.load < CARRY_LOAD_MIN:
        raise CarryRefused(
            f"no contact load: |Present_Load| {grip.load:.0f} < {CARRY_LOAD_MIN:.0f} -- the jaw sits at "
            f"{grip.position:.2f} in free air, not on the tube"
        )


def is_holding(present: float, load_abs: float) -> bool:
    """Already on the tube: inside the training carry band WITH contact load. The runners'
    gate loop re-homes up to three times; on attempt 2 the tube is already in the jaws, and
    opening them there drops it onto the rig. Position alone is not enough (the empty jaw
    parks at 13.2 in free air): the load is what says the tube is there."""
    lo, hi = home_gate.support_for(TAG)["gripper"]
    return lo <= present <= hi and load_abs >= CARRY_LOAD_MIN


def _say(text: str) -> None:
    import contextlib
    import subprocess

    with contextlib.suppress(Exception):
        subprocess.run(["say", text], check=False, timeout=15)


def operator_prompt(msg: str, *, out_path: str = "/dev/tty", in_path: str = "/dev/tty", speak=True) -> str:
    """Put the placement prompt in front of the operator. The runners redirect home_arm's
    stdout and stderr to ${RUN}.home, so a plain input() prompt would be written into a file
    nobody is reading; this writes to the terminal itself, speaks, and reads ENTER from the
    terminal. `speak` may be a callable (tests) or a bool (bench: macOS `say`)."""
    try:
        with open(out_path, "w") as f:
            f.write(msg + "\n")
    except OSError:
        print(msg, flush=True)
    if speak:
        (speak if callable(speak) else _say)("Place the tube in the jaws, then press enter.")
    try:
        with open(in_path) as f:
            return f.readline()
    except OSError:
        return input()


def place_and_close(bus, pose: dict, *, prompt, sleep=None) -> CarryGrip:
    """Open the jaw, ask the operator to place the tube, close, verify. `pose` is the arm's
    settled pose as the POLICY will see it; the close only makes sense at the carry pose.
    A jaw that is already holding the tube (band AND load) is verified and kept as it is."""
    import time

    sleep = sleep or time.sleep
    arm = {j: v for j, v in pose.items() if j != "gripper"}
    ok, offenders = home_gate.check_pose(arm, dataset=TAG)
    if not ok:
        raise CarryRefused(f"arm is not at the carry pose: {home_gate.describe(offenders)}")
    present, load = _read(bus)
    if is_holding(present, load):
        grip = CarryGrip(position=present, load=load)
        verify_carry_grip(grip)
        return grip
    bus.write("Goal_Position", "gripper", OPEN_FOR_PLACEMENT, normalize=True)
    sleep(0.8)
    present, _ = _read(bus)
    if present < OPEN_FOR_PLACEMENT - OPEN_TOL:
        raise CarryRefused(f"jaw did not open for placement: {present:.2f} < {OPEN_FOR_PLACEMENT - OPEN_TOL:.1f}")
    prompt(
        f"Jaw open at {present:.1f}. Place the tube in the jaws at the carry pose (cap image-left), "
        f"clear your hand, then press ENTER to close."
    )
    grip = close_on_tube(bus, sleep=sleep)
    verify_carry_grip(grip)
    return grip


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------
def _snippets(s: dict) -> str:
    m = s["mean"]
    pose = ", ".join(f"{m[j]:.1f}" for j in JOINTS)
    wrapper = "\n".join(f'    "{j}.pos": {m[j]:.1f},' for j in JOINTS)
    gate = "\n".join(
        f'        "{j}": ({m[j]:.2f}, {s["std"][j]:.2f}, {s["min"][j]:.2f}, {s["max"][j]:.2f}),' for j in JOINTS
    )
    return (
        f'home_arm.HOME_POSES:      "{TAG}": [{pose}],\n'
        f"rollout wrapper:          TRAIN_HOME_V4_CARRY = {{\n{wrapper}\n}}\n"
        f'home_gate.FRAME0:         "{TAG}": {{\n{gate}\n    }}'
    )


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default=None, help="dataset root (default: the v4 dataset in the HF cache)")
    a = p.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
    from conftest import V4_DATASET_ROOT, v4_training_episodes

    root = Path(a.root) if a.root else V4_DATASET_ROOT
    s = derive(root, v4_training_episodes())
    print(f"dataset: {root}")
    print(f"episodes requested {s['n_requested']}, used {s['n_episodes']}, excluded {len(s['excluded'])}: {s['excluded']}")
    print(f"sign check: lift at grasp mean {s['lift_at_grasp_mean']:.2f}; numeric +{LIFT_RISE_DEG:g} deg fires in "
          f"{s['n_numeric_rise_fires']} episodes; LIFT_DIRECTION={LIFT_DIRECTION}")
    k = s["k_stats"]
    print(f"k: [{k['min']}, {k['max']}] frames = {k['min_s']:.2f}-{k['max_s']:.2f} s @ {FPS} FPS, "
          f"mean {k['mean']:.2f} sd {k['sd']:.2f} median {k['median']:.1f}")
    print(f"pre-grasp opening: {s['pre_grasp_open']['min']:.1f}-{s['pre_grasp_open']['max']:.1f}; "
          f"carry gripper |load|: {s['gripper_load_abs']['min']:.0f}-{s['gripper_load_abs']['max']:.0f} "
          f"(mean {s['gripper_load_abs']['mean']:.1f}); CARRY_LOAD_MIN={CARRY_LOAD_MIN:.0f}")
    print(f"\n{'CARRY POSE':14s} {'mean':>8s} {'sd':>7s} {'min':>8s} {'max':>8s}")
    for j in JOINTS:
        print(f"{j:14s} {s['mean'][j]:8.2f} {s['std'][j]:7.2f} {s['min'][j]:8.2f} {s['max'][j]:8.2f}")
    print("\n" + _snippets(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
