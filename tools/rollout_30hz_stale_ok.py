#!/usr/bin/env python
"""lerobot-rollout wrapper: run the control loop at the TARGET fps even when
cameras deliver fewer frames than that, by serving the latest available frame
instead of blocking for a fresh one.

WHY (Aug 15, 2026): both icspring cameras deliver 20.0/22.6 fps instead of 30
(measured at the native AVFoundation level, zero OS drops — cause unknown,
post-Aug-14 regression on the Mac's USB camera path). lerobot's async_read
blocks until a FRESH frame, so with two cameras read sequentially the rollout
loop sinks to ~12.3 Hz — a tempo the 30 Hz-trained policies never saw, which
invalidates every trial. With the sync-engine queue-skip patch, ACT only
CONSUMES an observation at chunk boundaries (once per n_action_steps=100 ticks
= every 3.3 s); every other read exists for the rollout recording. Serving the
newest available frame lets actions step at the trained 33.3 ms tempo while
cameras refresh at whatever they can.

KNOWN COSTS (accepted, documented for the writeup):
- rollout-recording videos contain ~1/3 duplicated frames (scoring unaffected);
- chunk-boundary observations are up to ~50 ms stale (vs 3.3 s of open-loop
  execution per chunk — negligible);
- 50 ms camera exposures blur motion slightly more than training's 33 ms.

USAGE: exactly like lerobot-rollout, same args:
    python tools/rollout_30hz_stale_ok.py --strategy.type=episodic ...

This wrapper is ROLLOUT-ONLY on purpose. Never use it for lerobot-record:
training data must never contain duplicated frames.
"""

import os
import sys
from pathlib import Path

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera


def async_read_latest(self, timeout_ms: float = 200):
    """Return the newest available frame; block only if none has EVER arrived."""
    if self.thread is None or not self.thread.is_alive():
        raise RuntimeError(f"{self} read thread is not running.")

    # Grab whatever is fresh within ~5 ms, otherwise fall through to the cache.
    got_new = self.new_frame_event.wait(timeout=0.005)
    with self.frame_lock:
        frame = self.latest_frame
        if got_new:
            self.new_frame_event.clear()

    if frame is None:
        # Startup only: no frame has ever been captured — block like upstream.
        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(
                f"Timed out waiting for first frame from camera {self} after {timeout_ms} ms."
            )
        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()
        if frame is None:
            raise RuntimeError(f"Internal error: event set but no frame for {self}.")

    return frame


OpenCVCamera.async_read = async_read_latest
print(
    "[rollout_30hz_stale_ok] OpenCVCamera.async_read patched: serving latest "
    "available frame (non-blocking); control loop paces at target fps.",
    file=sys.stderr,
)

# ---------------------------------------------------------------------------
# Per-stage timing instrumentation: prints a breakdown every ~90 ticks so the
# slow stage of the control loop is measured, not inferred.
# ---------------------------------------------------------------------------
import time as _time  # noqa: E402
from collections import defaultdict  # noqa: E402

_stage_totals: dict[str, float] = defaultdict(float)
_stage_counts: dict[str, int] = defaultdict(int)


def _timed(stage, fn):
    def wrapper(*a, **k):
        t0 = _time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            _stage_totals[stage] += _time.perf_counter() - t0
            _stage_counts[stage] += 1

    return wrapper


def _report_maybe():
    n = _stage_counts.get("add_frame", 0)
    if n and n % 90 == 0:
        parts = []
        for stage in sorted(_stage_totals, key=_stage_totals.get, reverse=True):
            c = _stage_counts[stage] or 1
            parts.append(f"{stage} {1000 * _stage_totals[stage] / c:5.1f}ms")
            _stage_totals[stage] = 0.0
            _stage_counts[stage] = 0
        print(f"[timing per tick] {'  '.join(parts)}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Phantom-arrow-key guard. Lived here as a private copy from Aug 18 until
# Sep 5 2026 — which is precisely why `lerobot-record` never had it, and why a
# scroll-forged Right arrow cost three v4 episodes. One module, two callers now;
# see tools/arrow_guard.py for the full account and tools/record_arrow_safe.py
# for the recording entry point.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from arrow_guard import install as _install_arrow_guard  # noqa: E402

_install_arrow_guard()

import lerobot.rollout.strategies.core as _core  # noqa: E402
import lerobot.rollout.strategies.episodic as _episodic  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.rollout.robot_wrapper import ThreadSafeRobot  # noqa: E402
from lerobot.rollout.strategies.core import RolloutStrategy  # noqa: E402

ThreadSafeRobot.get_observation = _timed("get_obs", ThreadSafeRobot.get_observation)
ThreadSafeRobot.send_action = _timed("send_act", ThreadSafeRobot.send_action)

# ---------------------------------------------------------------------------
# Telemetry sidecar (Sep 6 2026): tee all 18 servo channels to CSV, one row per
# recorded frame, whatever width the policy sees. An A-side rollout records a
# 6-dim state, so without this the load/current at contact — the evaluation's
# mechanism endpoints — would exist for B trials only. Write-only; the policy
# path is untouched. Inert unless CAPSTONE_TELEMETRY_CSV is set (the runners set it).
# ---------------------------------------------------------------------------
import atexit as _atexit  # noqa: E402

from telemetry_sidecar import Sidecar as _Sidecar  # noqa: E402

_telemetry = _Sidecar.from_env()
# Wall-clock loop rate (Sep 6 2026, handoff §3 item 7): first/last recorded frame on the
# monotonic clock, one `[loop] loop_hz=...` line at exit; the runners copy it into the
# `.verify` receipt. Until then the 13.4 / 17 Hz figures stay barred (tools/loop_timer.py).
from loop_timer import LoopTimer as _LoopTimer  # noqa: E402

_loop = _LoopTimer()
_atexit.register(_loop.report)
_get_obs_timed = ThreadSafeRobot.get_observation


def _get_obs_tee(*a, **k):
    obs = _get_obs_timed(*a, **k)
    _telemetry.observe(obs)
    return obs


ThreadSafeRobot.get_observation = _get_obs_tee
_atexit.register(_telemetry.close)
if _telemetry.path:
    print(f"[telemetry] 18-channel sidecar -> {_telemetry.path}", file=sys.stderr, flush=True)
RolloutStrategy._process_observation_and_notify = _timed(
    "obs_proc", RolloutStrategy._process_observation_and_notify
)
_core.send_next_action = _timed("next_act", _core.send_next_action)
_episodic.send_next_action = _core.send_next_action
_episodic.build_dataset_frame = _timed("build_frame", _episodic.build_dataset_frame)

_orig_add_frame = LeRobotDataset.add_frame


def _add_frame_timed(self, frame):
    t0 = _time.perf_counter()
    try:
        return _orig_add_frame(self, frame)
    finally:
        _stage_totals["add_frame"] += _time.perf_counter() - t0
        _stage_counts["add_frame"] += 1
        _telemetry.frame_recorded()
        _loop.frame_recorded()
        _report_maybe()


LeRobotDataset.add_frame = _add_frame_timed
print("[rollout_30hz_stale_ok] per-stage timing instrumentation active.", file=sys.stderr)

# ---------------------------------------------------------------------------
# Robust between-episode homing: upstream returns to the pose the arm held at
# CONNECT time (hw.initial_position) with a fast 1 s interpolation and no
# verification. If the arm connected in a non-home pose (e.g. slumped after a
# warmup replay), every reset re-homes to that corrupted pose and later
# episodes start out-of-distribution (observed Aug 15: ep2+ overreached into
# the walls). Replace it: always return to the TRAINING home pose (measured
# across training episodes' frame-0 states), slower, and verify with a retry.
# ---------------------------------------------------------------------------
TRAIN_HOME_V1 = {
    "shoulder_pan.pos": 2.4,
    "shoulder_lift.pos": -105.2,
    "elbow_flex.pos": 82.9,
    "wrist_flex.pos": 73.4,
    "wrist_roll.pos": -2.2,
    "gripper.pos": 1.1,
}
# v2 frame-0 means, measured across all 50 episodes' parquet (Aug 16) —
# enforced against the dataset by tests/test_home_pose_matches_training.py.
TRAIN_HOME_V2 = {
    "shoulder_pan.pos": 2.5,
    "shoulder_lift.pos": -105.2,
    "elbow_flex.pos": 79.7,
    "wrist_flex.pos": 73.4,
    "wrist_roll.pos": -1.5,
    "gripper.pos": 1.3,
}
# v3 frame-0 means, measured across all 50 episodes of
# so101-tube-insert-v3_20260827_110242 — the same source home_gate.FRAME0["v3"]
# is derived from, enforced by tests/test_home_pose_matches_training.py.
#
# Added Sep 3 2026. Its absence was the Aug-16 v1->v2 bug recurring as v2->v3:
# this table held only v1 and v2, the default below was "v2", and no v3 runner
# set CAPSTONE_HOME_POSE.
#
# BLAST RADIUS, corrected Sep 5 by the pre-push review — the first account was
# wrong. TRAIN_HOME is consumed ONLY by _return_to_initial_position, which
# lerobot 0.6.1 calls from three sites, all AFTER an episode (episodic.py:168
# reset, :190 re-record, core.py:128 teardown). The reset branch is gated on
# `recorded_episodes < num_episodes - 1` and every runner passes
# --dataset.num_episodes=1, so it never fires. The RECORDED start pose comes
# from home_arm.py --pose $DATASET_TAG, pinned separately. Measured across all
# 20 v3-era rollout parquets: 19 homed to v3, frame-0 elbow mean 82.04 (sd
# 1.38) vs the v3 training mean 82.34 — -0.30 deg, not the "~1 deg low" first
# claimed; a real v2 re-home would sit at -2.61. Only P05-02 (a re-record) was
# genuinely affected. The fix is still right — it prevents the bug the moment
# anything runs multi-episode or a v4 tag lands in the table below.
TRAIN_HOME_V3 = {
    "shoulder_pan.pos": 1.9,
    "shoulder_lift.pos": -105.2,
    "elbow_flex.pos": 82.3,
    "wrist_flex.pos": 73.8,
    "wrist_roll.pos": -1.9,
    "gripper.pos": 1.0,
}
# v4 frame-0 means over the 50 TRAINING episodes of so101-tube-insert-v4_20260904_220924
# (hold-out and voided takes excluded) -- the same source as home_gate.FRAME0["v4"] and
# tests/test_v4_support.py, which re-derives them from the parquet. Added Sep 6 2026, the
# night the v4 policies came off the Hub: elbow 85.6 vs v3's 82.3 (+3.3 deg), gripper 1.1
# vs 1.0. Homing a v4 policy to the v3 pose would have been the v2->v3 bug a third time.
TRAIN_HOME_V4 = {
    "shoulder_pan.pos": 1.6,
    "shoulder_lift.pos": -105.3,
    "elbow_flex.pos": 85.6,
    "wrist_flex.pos": 73.7,
    "wrist_roll.pos": -1.7,
    "gripper.pos": 1.1,
}
# v4-carry (Sep 6 2026): the PRE-GRASPED start, a MODIFIED TASK -- the tube starts IN the jaws
# (S2 stratum of the merged protocol). Mean CARRY frame over the 50 v4 TRAINING episodes
# (tools/v4_carry.py; tests/test_v4_carry.py re-derives it from the parquet), the same 1-decimal
# values as home_arm.HOME_POSES["v4-carry"]. The gripper value is the tube's width (13.2), not a
# closed jaw. shoulder_lift here (+45) is actively holding the extended arm with no stop under
# it, so the post-episode re-home must NOT release it: see _settle_joints_for below.
TRAIN_HOME_V4_CARRY = {
    "shoulder_pan.pos": 67.1,
    "shoulder_lift.pos": 45.3,
    "elbow_flex.pos": -35.8,
    "wrist_flex.pos": 78.5,
    "wrist_roll.pos": 49.7,
    "gripper.pos": 13.2,
}
_HOME_POSE = os.environ.get("CAPSTONE_HOME_POSE", "v2")
TRAIN_HOME = {"v1": TRAIN_HOME_V1, "v2": TRAIN_HOME_V2, "v3": TRAIN_HOME_V3, "v4": TRAIN_HOME_V4,
              "v4-carry": TRAIN_HOME_V4_CARRY}[_HOME_POSE]
# Settle ONLY shoulder_lift (its true training start, -105.2, is beyond the
# -100 goal clamp — it rests against the hard stop). Training frame-0s were
# captured under teleop with torque ON everywhere else; a full-bus release
# let the unpowered elbow sag to 88.4 vs training range 75.5–84.4 (Aug 16).
SETTLE_JOINTS = ("shoulder_lift",)
# ...except at the carry pose, where shoulder_lift holds the arm up and releasing it drops the
# arm (and the tube) onto the rig. tools/v4_carry.settle_joints_for returns () for "v4-carry"
# and the literal SETTLE_JOINTS for every frame-0 pose; both homing paths route through it.
from v4_carry import settle_joints_for as _settle_joints_for  # noqa: E402  (pure; no bus)
_HOME_TOL = 3.0
# shoulder_lift's training home (-105.2) is beyond the -100 goal clamp; the arm
# settles there passively, so give it extra slack.
_HOME_SLACK = {"shoulder_lift.pos": 4.0}
# Gripper (Aug 22): re-homing closes the jaws on nothing. The follower bounds
# each close command to 4.0 below present (~56 counts) in "free air", the
# firmware latches at a sustained ~35 counts, and the fault-recovery path opens
# the jaw +2 per event — so repeated "go to 1.3" re-sends crept the frame-0
# gripper state to +8.2/+9.5/+10.8 across the 17:28 block. Give the gripper
# slack so a near-closed jaw is NOT re-commanded; the interpolation already
# ends at the training value. Torque stays ON (training held it at 1.3).
_HOME_SLACK["gripper.pos"] = 3.0


def _return_to_training_home(hw, duration_s: float = 2.5, fps: int = 50) -> None:
    from lerobot.utils.robot_utils import precise_sleep  # local import, matches upstream

    robot = hw.robot_wrapper
    try:
        obs = robot.get_observation()
        cur = {k: float(obs[k]) for k in TRAIN_HOME if k in obs}
        steps = max(int(duration_s * fps), 1)
        for s in range(1, steps + 1):
            t = s / steps
            robot.send_action({k: cur[k] * (1 - t) + TRAIN_HOME[k] * t for k in cur})
            precise_sleep(1 / fps)
        precise_sleep(0.4)
        for _ in range(2):
            obs = robot.get_observation()
            residuals = {k: float(obs[k]) - TRAIN_HOME[k] for k in cur}
            worst = max(abs(r) - _HOME_SLACK.get(k, 0.0) for k, r in residuals.items())
            if worst <= _HOME_TOL:
                break
            for k, r in residuals.items():
                if abs(r) - _HOME_SLACK.get(k, 0.0) > _HOME_TOL:
                    robot.send_action({k: TRAIN_HOME[k]})
            precise_sleep(0.6)
        # Training episodes started from shoulder_lift RESTING against its
        # stop (-105.2, beyond the -100 command clamp) with every OTHER joint
        # actively held by teleop. Release torque on SETTLE_JOINTS only; a
        # full-bus release lets the elbow sag out of training support.
        try:
            inner = getattr(robot, "robot", None) or getattr(robot, "_robot", robot)
            # v4-carry: shoulder_lift is +45 there, holding the extended arm with no stop
            # under it -- releasing it drops the arm. settle_joints_for empties the tuple.
            for _j in _settle_joints_for(_HOME_POSE, SETTLE_JOINTS):
                inner.bus.write("Torque_Enable", _j, 0, normalize=False)
            precise_sleep(0.5)
        except Exception as te:
            print(f"[rollout_30hz_stale_ok] settle release failed: {te}", file=sys.stderr)
        # Diagnostics (Aug 22): per-joint residuals plus the gripper's raw load, so a
        # drifting start state is measured in the log rather than inferred later.
        try:
            obs = robot.get_observation()
            residuals = {k.removesuffix(".pos"): round(float(obs[k]) - TRAIN_HOME[k], 1) for k in cur}
            _inner = getattr(robot, "robot", None) or getattr(robot, "_robot", robot)
            grip_load = _inner.bus.read("Present_Load", "gripper", normalize=False)
            print(f"[rollout_30hz_stale_ok] re-home residuals {residuals} gripper_load {grip_load}",
                  file=sys.stderr)
        except Exception as de:
            print(f"[rollout_30hz_stale_ok] re-home diagnostics failed: {de}", file=sys.stderr)
        print(f"[rollout_30hz_stale_ok] re-homed to {_HOME_POSE} training pose "
              f"(worst residual {worst:.1f}); released {_settle_joints_for(_HOME_POSE, SETTLE_JOINTS)} "
              f"to settle, other joints held", file=sys.stderr)
    except Exception as e:  # never kill a rollout over homing
        print(f"[rollout_30hz_stale_ok] re-home failed: {e}", file=sys.stderr)


def _return_patched(hw, duration_s: float = 2.5, fps: int = 50) -> None:
    _return_to_training_home(hw, duration_s=duration_s, fps=fps)


RolloutStrategy._return_to_initial_position = staticmethod(_return_patched)
print("[rollout_30hz_stale_ok] between-episode reset pinned to TRAINING home pose "
      "(verified, with retry).", file=sys.stderr)

# ---------------------------------------------------------------------------
# SmolVLA support: checkpoints saved by remote lerobot 0.6.2 store the
# tokenizer as a relative subfolder name ("tokenizer") that local 0.6.1
# cannot resolve. Inject an override pointing at the same tokenizer's
# canonical source (the policy config's vlm_model_name).
# ---------------------------------------------------------------------------
import lerobot.rollout.context as _ctx  # noqa: E402

_orig_mppp = _ctx.make_pre_post_processors


def _mppp_with_tokenizer_fix(*args, **kwargs):
    policy_cfg = kwargs.get("policy_cfg") or (args[0] if args else None)
    vlm = getattr(policy_cfg, "vlm_model_name", None)
    if vlm:
        overrides = dict(kwargs.get("preprocessor_overrides") or {})
        overrides.setdefault("tokenizer_processor", {"tokenizer_name": vlm})
        kwargs["preprocessor_overrides"] = overrides
        print(f"[rollout_30hz_stale_ok] tokenizer override -> {vlm}", file=sys.stderr)
    return _orig_mppp(*args, **kwargs)


_ctx.make_pre_post_processors = _mppp_with_tokenizer_fix

# ---------------------------------------------------------------------------
# Grasp-phase chunk locking (Aug 25, 2026 — Faith's insight, opt-in via
# CAPSTONE_GRASP_LOCK=1). The 40-trial v2 baseline showed the dominant failure
# is a correct descent discarded by the NEXT chunk's re-plan veering left at
# the moment of clamping. When tools/grasp_lock.py detects closing onset (from
# the policy's own commanded gripper+lift, in degrees), we extend the CURRENT
# cached chunk instead of re-planning: the queue is refilled from the full
# chunk_size prediction (n_action_steps at a time) until the close completes
# or the chunk exhausts. Default OFF: without the env var, rollout behavior is
# byte-identical. Probe-only until it earns a protocol amendment.
# ---------------------------------------------------------------------------
if os.environ.get("CAPSTONE_GRASP_LOCK") == "1":
    from grasp_lock import GraspLock  # noqa: E402  (tools/ is on sys.path when run as a script)
    from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402
    from lerobot.rollout.inference.sync import SyncInferenceEngine  # noqa: E402

    _GL = GraspLock()
    _GL_LIFT_IDX, _GL_GRIP_IDX = 1, 5  # dataset action order: pan, lift, elbow, wflex, wroll, gripper

    _orig_predict_chunk = ACTPolicy.predict_action_chunk

    def _gl_caching_predict(self, batch, *a, **k):
        chunk = _orig_predict_chunk(self, batch, *a, **k)
        self._gl_full_chunk = chunk  # (B, chunk_size, dim), policy output space
        self._gl_consumed = self.config.n_action_steps  # first n go to the queue
        return chunk

    ACTPolicy.predict_action_chunk = _gl_caching_predict

    _orig_act_reset = ACTPolicy.reset

    def _gl_reset(self):
        _GL.reset()
        self._gl_full_chunk = None
        self._gl_consumed = 0
        return _orig_act_reset(self)

    ACTPolicy.reset = _gl_reset

    _orig_get_action = SyncInferenceEngine.get_action

    def _gl_get_action(self, obs_frame):
        policy = self._policy
        queue = getattr(policy, "_action_queue", None)
        if (
            _GL.locked
            and queue is not None
            and len(queue) == 0
            and getattr(policy, "_gl_full_chunk", None) is not None
        ):
            full = policy._gl_full_chunk
            consumed = policy._gl_consumed
            chunk_size = full.shape[1]
            if consumed < chunk_size:
                n = min(policy.config.n_action_steps, chunk_size - consumed)
                queue.extend(full[:, consumed : consumed + n].transpose(0, 1))
                policy._gl_consumed = consumed + n
                print(f"[grasp_lock] LOCKED: extended current chunk (+{n} ticks, "
                      f"{policy._gl_consumed}/{chunk_size} consumed)", file=sys.stderr)
            else:
                _GL.force_release()
                print("[grasp_lock] chunk exhausted — lock released, re-planning", file=sys.stderr)
        action = _orig_get_action(self, obs_frame)
        if action is not None:
            was = _GL.locked
            now = _GL.observe(float(action[_GL_GRIP_IDX]), float(action[_GL_LIFT_IDX]))
            if now and not was:
                print("[grasp_lock] closing onset detected — chunk locked", file=sys.stderr)
        return action

    SyncInferenceEngine.get_action = _gl_get_action
    print("[rollout_30hz_stale_ok] GRASP-PHASE CHUNK LOCKING ACTIVE "
          "(CAPSTONE_GRASP_LOCK=1) — probe configuration, not the scored protocol.",
          file=sys.stderr)

from lerobot.scripts.lerobot_rollout import main  # noqa: E402

if __name__ == "__main__":
    main()
