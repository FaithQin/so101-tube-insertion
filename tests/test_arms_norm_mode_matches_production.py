"""The diagnostics' bus helper must normalize joints the way the production
robot does, or every comparison against dataset values is a unit mismatch.

Found Sep 3, 2026 00:10: `tools/_arms.py` builds joints in RANGE_M100_100 while
`SOFollowerConfig.use_degrees` defaults to True (DEGREES). Both modes share the
same zero (the calibration midpoint, motors_bus.py:875/905) but not the scale:
RANGE is 200/range_width per tick, DEGREES is 360/4095 per tick. With the Aug 7
follower calibration the gap per unit of value is pan 10%, shoulder_lift 5.4%,
elbow 1%, wrist_flex 0.1%, wrist_roll 44%. At the demo's grasp/transport poses
that is ~7 deg of pan, ~3 deg of lift, and up to ~39 deg of wrist_roll. The
Sep 1 anchor drive (`replay_tracking.py`), the Sep 2 registration
(`pose_register.py`), and the Sep 2 `replay_offset.py` runs all read or drove
the arm through this helper and compared against DEGREES-normalized dataset
values; their offsets are the unit gap, not the plant. Production replays and
policies never touch the helper and are unaffected.

Marked xfail(strict) until `_arms.motors()` is switched to the production mode
AND home_arm / home_gate / preflight are re-verified on the bench in that mode.
Do not flip this before a bench session that depends on the current homing.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


@pytest.mark.xfail(strict=True, reason="_arms.py normalizes RANGE_M100_100; production SOFollower uses DEGREES (use_degrees=True)")
def test_arms_helper_uses_the_production_norm_mode():
    from lerobot.motors import MotorNormMode
    from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig
    import _arms

    production = MotorNormMode.DEGREES if SOFollowerConfig.use_degrees else MotorNormMode.RANGE_M100_100
    modes = {name: m.norm_mode for name, m in _arms.motors().items() if name != "gripper"}
    assert all(mode == production for mode in modes.values()), modes
