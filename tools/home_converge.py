"""Close the last degree of homing, by adding the measured shortfall back into the command.

Sep 6 2026, bench morning. Homing commands the training pose and accepts wherever the servo
settles. Under gravity a proportional controller settles SHORT: measured on the arm today, the
follower parks about 0.9 deg below the v4 wrist_flex target and 1-3 deg below the elbow target,
every time, from six consecutive homings. That is fine against `home_arm.TOL_DEFAULT` (2.5 deg)
and NOT fine against the preflight gate, whose windows come from the training data and are about
one sigma wide -- wrist_flex's is [73.23, 74.02] against a training sd of 0.17, and the arm lands
at 72.9. Every scored trial today would abort at the home gate.

The reason the gap exists at all is that the training poses were set by TELEOP: the operator
watched the arm sag and pushed the leader further to compensate. Homing has no operator, so it
has to do the same thing in code -- measure the shortfall, add it to the command, repeat.

    cmd <- clamp(cmd + (target - actual), target +- MAX_OVERSHOOT)

Bounded three ways, because pressing a servo that is not moving is how this rig latched a gripper
and browned out an elbow (the project notes, "Elbow servo is a tracked reliability risk"; "No
load-guard-free presses"):
  * MAX_OVERSHOOT caps how far past the target the command may ever go;
  * LOAD_MAX stops a joint that is pushing without moving -- the stall signature, measured today
    at load 368 with 0.44 deg of travel for 4.9 deg of command;
  * MIN_GAIN_DEG stops a joint that two passes could not move, so a jammed joint gets two tries,
    not six.

It converges the joints the GATE actually judges. shoulder_lift is excluded (it is commanded past
its -100 clamp deliberately and rests on its stop -- `home_gate.PASSIVE_SETTLE_JOINTS`), and so is
the gripper (its target is a jaw state, not a pose, and it is the joint with the latch).

This does not touch the under-load droop during a rollout, which is the plant's, not homing's.
"""
from __future__ import annotations

CONVERGE_TOL_DEG = 0.15      # tighter than the gate windows, which are ~1 training sigma
MAX_PASSES = 6
MAX_OVERSHOOT_DEG = 6.0      # wrist_flex needed +3.5 on the arm today; the elbow never converged
LOAD_MAX = 250               # the stall guard; today's elbow stall sat at 368 with no motion
MIN_GAIN_DEG = 0.05          # a pass that moves less than this counts as "did not move"
STALL_PASSES = 2             # two such passes and the joint is declared stuck

SKIP_JOINTS = frozenset({"shoulder_lift", "gripper"})


class Stalled(RuntimeError):
    """A joint pushed without moving. Stop commanding it; report, do not press."""


def converge_command(target: float, actual: float, command: float,
                     max_overshoot: float = MAX_OVERSHOOT_DEG) -> float:
    """The next command: the current one plus the shortfall, clamped around the target."""
    return min(max(command + (target - actual), target - max_overshoot), target + max_overshoot)


def is_converged(target: float, actual: float, tol: float = CONVERGE_TOL_DEG) -> bool:
    return abs(target - actual) <= tol


def joints_to_converge(targets: dict, skip=SKIP_JOINTS) -> list:
    return [j for j in targets if j not in skip]


def plan_pass(joint: str, target: float, actual: float, command: float, load: float,
              stalls: int, tol: float = CONVERGE_TOL_DEG, load_max: float = LOAD_MAX,
              max_overshoot: float = MAX_OVERSHOOT_DEG) -> tuple:
    """(action, value, reason) for one joint on one pass.

    action is "done" (inside tolerance), "stall" (pushing without moving, or over the load guard),
    or "command" with the next value.
    """
    if is_converged(target, actual, tol):
        return "done", actual, f"{joint} within {tol:g} deg"
    if abs(load) > load_max:
        return "stall", actual, (f"{joint} is {target - actual:+.2f} deg short at load {load:.0f} "
                                 f"(> {load_max:g}) — pushing without moving; not pressing further")
    if stalls >= STALL_PASSES:
        return "stall", actual, (f"{joint} did not move on {stalls} passes and is still "
                                 f"{target - actual:+.2f} deg short — treating it as stuck")
    nxt = converge_command(target, actual, command, max_overshoot)
    if abs(nxt - command) < 1e-9:
        return "stall", actual, f"{joint} is at the overshoot limit ({max_overshoot:g} deg) and still short"
    return "command", nxt, f"{joint} {target - actual:+.2f} short -> command {nxt:.2f}"


def converge(targets: dict, read, write, settle, log=print, tol: float = CONVERGE_TOL_DEG,
             max_passes: int = MAX_PASSES, load_max: float = LOAD_MAX,
             max_overshoot: float = MAX_OVERSHOOT_DEG) -> dict:
    """Drive `targets` to within `tol`. `read(joint) -> (position, load)`; `write(joint, value)`;
    `settle()` waits. Returns {joint: {"actual","command","passes","stalled","reason"}}."""
    state = {j: {"command": targets[j], "stalls": 0, "passes": 0, "stalled": False, "reason": ""}
             for j in joints_to_converge(targets)}
    for _ in range(max_passes):
        active = [j for j, s in state.items() if not s["stalled"]]
        if not active:
            break
        moved = False
        for j in active:
            pos, load = read(j)
            prev = state[j].get("actual")
            if prev is not None and abs(pos - prev) < MIN_GAIN_DEG:
                state[j]["stalls"] += 1
            state[j]["actual"] = pos
            action, value, reason = plan_pass(j, targets[j], pos, state[j]["command"], load,
                                              state[j]["stalls"], tol, load_max, max_overshoot)
            state[j]["reason"] = reason
            if action == "done":
                state[j]["stalled"] = False
                state[j]["converged"] = True
            elif action == "stall":
                state[j]["stalled"] = True
                log(f"  converge: {reason}")
            else:
                state[j]["command"] = value
                state[j]["passes"] += 1
                write(j, value)
                moved = True
        if not moved:
            break
        settle()
    for j, s in state.items():
        pos, _ = read(j)
        s["actual"] = pos
        s["converged"] = is_converged(targets[j], pos, tol)
    return state


def report(state: dict, targets: dict) -> str:
    parts = []
    for j, s in sorted(state.items()):
        mark = "ok" if s.get("converged") else ("STUCK" if s["stalled"] else "short")
        parts.append(f"{j} {s['actual']:.2f}/{targets[j]:.2f} {mark}" + (f" (+{s['passes']})" if s["passes"] else ""))
    return "converge: " + "  ".join(parts)
