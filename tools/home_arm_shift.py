"""The pose shift caused by `SOFollower.connect()`, shared by home_arm and the gate.

Measured Aug 30 2026 over three runs, homing to v3 then connecting as a rollout
does: wrist_roll -1.62 / -1.66 / -1.57, every other joint under 0.3 deg. The arm
is steady on its own (held -2.08 for 30 s with torque on), so it is the connect
and configure path relaxing the joint, with backlash taking it from there.

shoulder_lift is deliberately absent. Its -5.69 shift is the passive settle to
its true training value (-99.6 clamped before connect, -105.3 after, against a
-105.19 training mean), so it was always correct and must not be compensated.

home_arm drives to HOME - shift so the arm lands on HOME after connect; the
preflight gate adds the shift back so it judges what the policy will observe.
Both read this one table, or they drift apart.
"""

CONNECT_SHIFT = {
    "wrist_roll": -1.62,
}


def predict_post_connect(pose: dict) -> dict:
    """Given a pose measured BEFORE the rollout connects, predict what the
    policy will observe at frame 0."""
    return {j: v + CONNECT_SHIFT.get(j, 0.0) for j, v in pose.items()}
