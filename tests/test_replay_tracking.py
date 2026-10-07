"""Regression: the instrumented replay must isolate WHICH joint loses tracking, WHEN.

Born Sep 1 2026. A deterministic replay of known-good episode 0 grasped the
tube and then missed the rack badly — unprecedented (grasp-then-seat has been
the invariant since Aug 15). Faith's live observation localized the fault to
the LOADED phase: approach/grasp run the arm free, transport/insertion run it
loaded. The elbow had shown degraded goal-closing under strain for two days.

`lerobot-replay` logs nothing, so "which joint, and only-under-load?" was
unanswerable — and that answer decides a servo swap, which is a plant change
with study-comparability consequences. This tool replays while sampling
Present_Position each tick and reports per-joint tracking error split by
episode phase.

What is pinned here (pure functions; no hardware in tests/):

1. Phase splitting must come from the GRIPPER'S COMMANDED CLOSE, not a time
   guess. The episode's loaded phase begins where the demonstrated gripper
   closes on the tube. A hardcoded timestamp would rot the moment a different
   episode is used.
2. Per-joint, per-phase error — a single pooled number would average the
   elbow's loaded-phase lag away under five healthy joints (the same mistake
   as reporting the mean instead of p90 on latency).
3. The verdict must name the joint and the phase, and must not fire on
   healthy noise (backlash-scale error everywhere is a PASS).
"""

import sys

import numpy as np
from conftest import TOOLS

sys.path.insert(0, str(TOOLS))

import replay_tracking as rt  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def _healthy_traj(n=200):
    """Commanded ramps; actual tracks within 0.3 deg everywhere."""
    rng = np.random.default_rng(0)
    cmd = np.cumsum(rng.normal(0, 0.2, size=(n, 6)), axis=0) + 40.0
    act = cmd + rng.normal(0, 0.15, size=(n, 6))
    return cmd, act


# --------------------------------------------------------------------------
# 1. Phase boundary from the demonstrated gripper close
# --------------------------------------------------------------------------


def test_load_onset_is_where_the_gripper_closes():
    cmd, _ = _healthy_traj()
    g = np.full(200, 30.0)
    g[120:] = 2.0  # demonstrated close at tick 120
    cmd[:, 5] = g
    assert rt.find_load_onset(cmd, gripper_idx=5, closed_below=10.0) == 120


def test_load_onset_ignores_a_transient_dip():
    """One noisy tick below threshold is not a grasp."""
    cmd, _ = _healthy_traj()
    g = np.full(200, 30.0)
    g[60] = 5.0        # transient
    g[140:] = 2.0      # the real close
    cmd[:, 5] = g
    assert rt.find_load_onset(cmd, gripper_idx=5, closed_below=10.0, min_run=5) == 140


def test_load_onset_none_when_never_closed():
    cmd, _ = _healthy_traj()
    cmd[:, 5] = 30.0
    assert rt.find_load_onset(cmd, gripper_idx=5, closed_below=10.0) is None


# --------------------------------------------------------------------------
# 2. Per-joint, per-phase error
# --------------------------------------------------------------------------


def test_phase_errors_split_free_vs_loaded():
    cmd, act = _healthy_traj()
    act[120:, 2] += 3.0  # elbow lags 3 deg ONLY in the loaded phase
    rep = rt.phase_errors(cmd, act, JOINTS, load_onset=120)
    assert rep["elbow_flex"]["loaded_mean"] > 2.5
    assert rep["elbow_flex"]["free_mean"] < 0.5
    assert rep["shoulder_pan"]["loaded_mean"] < 0.5


def test_phase_errors_with_no_onset_report_free_only():
    cmd, act = _healthy_traj()
    rep = rt.phase_errors(cmd, act, JOINTS, load_onset=None)
    assert np.isnan(rep["elbow_flex"]["loaded_mean"])
    assert rep["elbow_flex"]["free_mean"] < 0.5


# --------------------------------------------------------------------------
# 3. Verdict: names the joint, load-phase-locked, quiet on healthy noise
# --------------------------------------------------------------------------


def test_verdict_names_a_load_locked_joint():
    cmd, act = _healthy_traj()
    act[120:, 2] += 3.0
    rep = rt.phase_errors(cmd, act, JOINTS, load_onset=120)
    v = rt.verdict(rep)
    assert v["status"] == "FAULT"
    assert "elbow_flex" in v["joints"]
    assert v["load_locked"] is True


def test_verdict_flags_free_phase_fault_as_not_load_locked():
    cmd, act = _healthy_traj()
    act[:, 2] += 3.0  # lags everywhere, both phases
    rep = rt.phase_errors(cmd, act, JOINTS, load_onset=120)
    v = rt.verdict(rep)
    assert v["status"] == "FAULT"
    assert v["load_locked"] is False


def test_verdict_passes_healthy_backlash_noise():
    cmd, act = _healthy_traj()
    rep = rt.phase_errors(cmd, act, JOINTS, load_onset=120)
    assert rt.verdict(rep)["status"] == "PASS"


def test_fault_threshold_is_declared_not_tuned():
    """Pre-declared, with a rationale: 1.5 deg is ~4x the validated homing
    spread (0.4) and half the residual that made a replay sail past the tube
    (3.6/5.6). Do not adjust after seeing a result."""
    assert rt.FAULT_DEG == 1.5


# --------------------------------------------------------------------------
# 4. Fixes from the first live run (Sep 1) — fixtures had encoded assumptions
# --------------------------------------------------------------------------
#
# The instrument's first live run produced a garbage verdict (5 joints FAULT,
# whole episode "loaded") for two reasons its own tests had blessed, because
# the fixtures encoded assumptions instead of the rig:
#
# (a) THIS RIG'S GRIPPER STARTS NEAR CLOSED (~1.0 normalized at home), opens
#     during approach, then closes on the grasp. "First closed tick" is tick 0
#     on every episode. The onset must be first sustained close AFTER a
#     sustained open.
# (b) A P-controlled servo legitimately lags a MOVING command by degrees at
#     20 Hz. Scoring every tick against a static threshold flags healthy
#     dynamics. Error must be scored on SETTLED ticks only — where the command
#     has been quasi-static long enough that a healthy joint has arrived.


def test_load_onset_close_after_open_on_the_rig_profile():
    """Gripper: closed at home (1.0) -> opens (30) -> closes on grasp (1.5)."""
    cmd, _ = _healthy_traj()
    g = np.full(200, 1.0)     # closed at home, from tick 0
    g[30:120] = 30.0          # opens during approach
    g[120:] = 1.5             # the real grasp
    cmd[:, 5] = g
    assert rt.find_load_onset(cmd, gripper_idx=5, closed_below=10.0) == 120


def test_load_onset_none_when_gripper_never_reopens():
    """Closed the whole episode = no demonstrated grasp transition."""
    cmd, _ = _healthy_traj()
    cmd[:, 5] = 1.0
    assert rt.find_load_onset(cmd, gripper_idx=5, closed_below=10.0) is None


def test_settled_mask_excludes_fast_command_segments():
    cmd = np.zeros((100, 1))
    cmd[40:60, 0] = np.linspace(0, 30, 20)  # fast ramp
    cmd[60:, 0] = 30.0                      # settled again
    m = rt.settled_mask(cmd[:, 0], vel_thresh=0.5, settle_ticks=4)
    assert m[:40].all(), "static prefix must be settled"
    assert not m[41:60].any(), "ramp must be excluded"
    assert m[70:].all(), "post-ramp plateau must re-settle"


def test_settled_mask_requires_the_dwell_not_just_one_slow_tick():
    cmd = np.zeros(20)
    cmd[10] = 5.0  # single spike: neighbours have high |delta|
    m = rt.settled_mask(cmd, vel_thresh=0.5, settle_ticks=4)
    assert not m[10:14].any(), "ticks within the dwell window of motion are unsettled"


def test_healthy_dynamic_lag_does_not_fault():
    """A joint that lags a moving command but lands every plateau is healthy."""
    n = 200
    cmd = np.zeros((n, 6))
    cmd[:, 2] = np.concatenate([np.zeros(50), np.linspace(0, 20, 50), np.full(100, 20.0)])
    act = cmd.copy()
    act[50:100, 2] -= 4.0   # big lag during the ramp only
    cmd[:, 5] = 30.0        # keep gripper open: single free phase
    rep = rt.phase_errors(cmd, act, JOINTS, load_onset=None)
    assert rep["elbow_flex"]["free_mean"] < 0.5, "dynamic lag leaked into settled error"


def test_true_offset_still_faults_on_settled_ticks():
    """A slipped/failed joint misses the plateau too — that must still flag."""
    cmd, act = _healthy_traj()
    act[:, 2] += 3.0        # constant offset, present when settled
    cmd[:, 5] = 30.0
    rep = rt.phase_errors(cmd, act, JOINTS, load_onset=None)
    assert rep["elbow_flex"]["free_mean"] > 2.5
    assert rt.verdict(rep)["status"] == "FAULT"


# --------------------------------------------------------------------------
# 5. A crash mid-replay must not leave the arm holding torque (Sep 1, live)
# --------------------------------------------------------------------------
#
# The gripper-overload run raised inside the streaming loop, before the
# disconnect line — so torque stayed ON across the whole arm, with the jammed
# gripper straining at raw ~1611, and Faith found the jaws locked when asked
# to free them by hand. Cleanup that only runs on success is not cleanup.


def test_streaming_loop_releases_in_a_finally():
    import inspect

    src = inspect.getsource(rt.main)
    assert "finally" in src, (
        "replay loop has no finally: a crash mid-stream leaves every servo "
        "holding torque, including a jammed gripper"
    )
    finally_part = src.split("finally", 1)[1]
    assert "disconnect" in finally_part, "the finally block must disconnect the bus"
