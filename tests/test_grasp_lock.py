"""Grasp-phase chunk locking (Aug 25, 2026, Faith's insight after the 40-trial
v2 baseline): the dominant failure was a correct descent discarded by the next
chunk's re-plan veering left at the moment of clamping. The lock denies
re-planning while a close is in progress: on closing onset, the current chunk
executes to its end. Pure state machine here; wrapper wiring is env-gated
(CAPSTONE_GRASP_LOCK=1) so default rollout behavior is byte-identical."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from grasp_lock import CLOSED_T, ONSET_DELTA, OPEN_T, GraspLock  # noqa: E402


def feed(gl, trace):
    for grip, lift in trace:
        gl.observe(grip, lift)
    return gl


def test_closing_onset_locks():
    gl = GraspLock()
    # high approach: open gripper, lift above band -> must NOT arm
    feed(gl, [(40, 80), (40, 70)])
    assert not gl.locked
    # descend into grasp band with gripper open -> armed; then closing onset
    feed(gl, [(40, 45), (40, 40), (30, 38), (20, 37)])
    assert gl.locked, "two consecutive closing ticks in the grasp band must lock"


def test_onset_deadband_rejects_command_jitter():
    """ONSET_DELTA=0.5 is a noise deadband. Every closing trace in this file
    steps the gripper by 10 deg, so the deadband could be zeroed with the whole
    suite green (proved by mutation, Sep 5 2026). At 0.0 any strictly
    decreasing float is a "closing" tick: two ticks of 0.01 deg command jitter
    while the policy hovers in the grasp band arm the lock during the APPROACH,
    the chunk is spent open-loop through the phase where vision correction
    matters most, and the genuine close then arrives inside a chunk the lock
    already spent. Brackets the deadband from both sides: 0.01 deg/tick must
    not lock, 0.6 deg/tick must."""
    jitter = GraspLock()
    feed(jitter, [(40.0, 40.0)])  # armed: gripper open, lift inside the band
    feed(jitter, [(40.0 - 0.01 * k, 39.0) for k in range(1, 21)])
    assert not jitter.locked, (
        "20 ticks of 0.01 deg gripper jitter locked the chunk mid-approach -- "
        f"the onset deadband is gone (ONSET_DELTA={ONSET_DELTA})"
    )
    real = GraspLock()
    feed(real, [(40.0, 40.0)])
    feed(real, [(40.0 - 0.6 * k, 39.0) for k in range(1, 4)])
    assert real.locked, (
        "a genuine 0.6 deg/tick close did not lock -- the deadband is now too "
        f"wide to see a real close (ONSET_DELTA={ONSET_DELTA})"
    )


def test_the_onset_run_must_be_consecutive():
    """The word "consecutive" lived only in an assertion message: no trace here
    ever interleaved a non-closing tick, so deleting the `self._closing_run = 0`
    reset in the ARMED branch -- the one line that makes ONSET_TICKS mean
    consecutive rather than cumulative -- kept the suite green. Two unrelated
    downward twitches minutes apart would then latch the lock, and
    replay_grasp_lock.py's onset arithmetic (`lock_tick - (ONSET_TICKS - 1)`)
    assumes the run IS consecutive, so the offline PROTECTED/RACED tally that
    gates the bench probe would be computed against a fictitious onset tick."""
    gl = GraspLock()
    feed(gl, [(40, 40)])  # armed
    feed(gl, [(30, 39)])  # one closing tick
    feed(gl, [(30, 39)])  # gripper holds -- the run must reset here
    feed(gl, [(20, 38)])  # a second, unrelated closing tick
    assert not gl.locked, (
        "two NON-consecutive closing ticks locked the chunk: the closing run "
        "accumulates instead of resetting when the gripper holds still"
    )
    feed(gl, [(10, 38)])  # now genuinely consecutive -> lock
    assert gl.locked


def test_close_completion_releases_and_rearms():
    """The release trace used to be `[(5, 36)] * GraspLock.RELEASE_TICKS` --
    derived from the constant under test, so it passed for ANY value and
    RELEASE_TICKS could be dropped to 1 with the suite green. At 1 the lock
    drops on the first tick the jaws reach the tube, so the next chunk boundary
    re-plans at exactly the moment of clamping: the veer-left failure the lock
    exists to deny, reproduced while the log says "lock engaged". Counted
    literally now -- 4 closed ticks must NOT release, the 5th must."""
    gl = GraspLock()
    feed(gl, [(40, 40), (30, 38), (20, 37)])
    assert gl.locked
    feed(gl, [(5, 36)] * 4)
    assert gl.locked, (
        "the lock released after only 4 closed ticks -- the 5-tick hold is what "
        f"carries it through the clamp (RELEASE_TICKS={GraspLock.RELEASE_TICKS})"
    )
    gl.observe(5, 36)
    assert not gl.locked, (
        "the lock survived 5 consecutive closed ticks -- the close is complete "
        f"and re-planning must resume (RELEASE_TICKS={GraspLock.RELEASE_TICKS})"
    )
    # second grasp attempt (re-open, re-close) must lock again
    feed(gl, [(40, 40), (40, 39), (28, 38), (18, 37)])
    assert gl.locked, "the lock must re-arm for every subsequent grasp attempt"


def test_hysteresis_band_does_not_count_as_a_completed_close():
    """open >15 / closed <8 is a hysteresis band, but every release trace here
    used grip 5 -- 3 deg clear of CLOSED_T -- and every re-open used 40, so
    CLOSED_T could be raised to 15.0, collapsing it onto OPEN_T, with the suite
    green. The release counter would then start ~7 deg of jaw travel early,
    mid-close while the tube is still being gathered, and the lock would expire
    DURING the clamp instead of after it: the same bench signature as a lock
    that never engaged, plus a degenerate boundary where one command is both
    "closed" and not "open"."""
    gl = GraspLock()
    feed(gl, [(40, 40), (30, 38), (20, 37)])
    assert gl.locked
    # 12 deg sits inside the band: still gathering the tube, not clamped on it.
    feed(gl, [(12, 36)] * (GraspLock.RELEASE_TICKS * 2))
    assert gl.locked, (
        "a grip command of 12 deg -- inside the 8..15 hysteresis band -- counted "
        f"as a completed close and released the lock mid-clamp (CLOSED_T={CLOSED_T})"
    )
    assert (OPEN_T, CLOSED_T) == (15.0, 8.0), (
        f"hysteresis thresholds moved to open>{OPEN_T} / closed<{CLOSED_T}; the "
        "module docstring and the eval sheet both say open >15 / closed <8"
    )


def test_no_lock_when_open_above_band():
    gl = GraspLock()
    # closing motion while lift is HIGH (e.g. carrying/retreat) must not lock
    feed(gl, [(40, 80), (30, 78), (20, 75), (10, 72)])
    assert not gl.locked


def test_force_release_returns_to_idle():
    """`assert not gl.locked` is satisfied by ANY non-LOCKED state, and the old
    follow-up trace re-opened the gripper to 40 first -- so force_release()
    could land in ARMED instead of IDLE with the suite green. ARMED skips the
    "gripper must re-open first" precondition, so on a stubborn close the
    wrapper (which calls force_release() exactly when a locked chunk exhausts)
    re-locks on the next two decreasing ticks and extends the next chunk too:
    lock chained after lock, never re-planning -- the woodpecker loop with the
    lock's name on it. Distinguished now by feeding closing ticks from an
    ALREADY-CLOSED gripper, which only ARMED would act on."""
    gl = GraspLock()
    feed(gl, [(40, 40), (30, 38), (20, 37)])
    assert gl.locked
    gl.force_release()  # wrapper calls this when the cached chunk exhausts
    assert not gl.locked
    feed(gl, [(12, 37), (10, 37)])
    assert not gl.locked, (
        "force_release() left the machine ARMED, not IDLE: two more decreasing "
        "ticks re-locked without the gripper ever re-opening above OPEN_T, so a "
        "failing close would chain lock after lock and never re-plan"
    )
    # a real re-grasp -- open above OPEN_T, then close -- must still lock
    feed(gl, [(40, 40), (30, 38), (20, 37)])
    assert gl.locked


def test_reset_clears_state():
    gl = GraspLock()
    feed(gl, [(40, 40), (30, 38), (20, 37)])
    gl.reset()
    assert not gl.locked


# ---------------------------------------------------------------------------
# tools/rollout_30hz_stale_ok.py -- the wiring.
#
# Until Sep 5 2026 the only coverage of the wrapper was two substring scans,
# `"CAPSTONE_GRASP_LOCK" in src` and `"grasp_lock" in src`. A mutation sweep
# walked through all three of these with the full suite green:
#   * `== "1"` -> `!= "0"` on the gate -- exactly the edit someone makes to run
#     one probe block without exporting the var, then forgets to revert. The
#     lock silently turns ON for every scored trial: a control policy the Eval
#     Protocol never locked, one stderr line as the only tell, and a whole
#     paired A/B block unusable.
#   * `policy._gl_consumed = consumed + n` -> `= n` -- every refill re-serves
#     the SAME stale slice, `consumed < chunk_size` never goes false, so the
#     chunk-exhaustion release never fires: an indefinite open-loop press into
#     the rack, on an elbow servo already tracked as a brownout/latch risk.
#   * `_GL_LIFT_IDX, _GL_GRIP_IDX = 1, 5` -> `5, 1` -- the detector reads the
#     wrong two action columns, locks on the descent and never at the close,
#     and the probe retires the v2 hypothesis on a mis-wired detector while
#     replay_grasp_lock.py's own (correct) indices still report PROTECTED.
#
# So these RUN the wrapper instead of reading it: a subprocess imports it with
# lerobot's two patch targets (SyncInferenceEngine.get_action,
# ACTPolicy.predict_action_chunk) replaced by stubs, then drives the installed
# _gl_get_action with a synthetic action chunk. No policy, no robot, no camera,
# no hardware. It has to be a subprocess: importing the wrapper rewrites lerobot
# classes process-wide, and this pytest process must stay clean.
# ---------------------------------------------------------------------------

PY_BIN = "/opt/homebrew/Caskroom/miniforge/base/envs/lerobot/bin/python"
TOOLS = Path(__file__).resolve().parent.parent / "tools"

_PROBE = r'''
import importlib.util, json, sys
from collections import deque
from pathlib import Path

MODE, TOOLS = sys.argv[1], Path(sys.argv[2])
sys.path.insert(0, str(TOOLS))

import torch
import lerobot.policies.act.modeling_act as _act
import lerobot.rollout.inference.sync as _sync

PRISTINE = (_sync.SyncInferenceEngine.get_action, _act.ACTPolicy.predict_action_chunk)

def load():
    spec = importlib.util.spec_from_file_location(
        "wrapper_under_test", TOOLS / "rollout_30hz_stale_ok.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

if MODE == "gate-off":
    load()
    now = (_sync.SyncInferenceEngine.get_action, _act.ACTPolicy.predict_action_chunk)
    print(json.dumps({"patched": now != PRISTINE}))
    sys.exit(0)

CHUNK, N = 100, 25

def stub_get_action(self, obs_frame):
    q = self._policy._action_queue
    return q.popleft().squeeze(0) if q else None

_sync.SyncInferenceEngine.get_action = stub_get_action
_act.ACTPolicy.predict_action_chunk = lambda self, batch, *a, **k: self._gl_full_chunk

mod = load()
gate_opened = _sync.SyncInferenceEngine.get_action is not stub_get_action

class Policy:
    def __init__(self, chunk):
        self.config = type("cfg", (), {"n_action_steps": N})()
        self._action_queue = deque()
        self._gl_full_chunk = chunk
        self._gl_consumed = 0

class Engine:
    def __init__(self, policy):
        self._policy = policy

def chunk_of(grip, lift):
    c = torch.zeros(1, CHUNK, 6)
    for t in range(CHUNK):
        c[0, t, 0] = float(t)   # row id in the pan column: a re-served slice shows up
        c[0, t, 1] = lift(t)
        c[0, t, 5] = grip(t)
    return c

def run(chunk, ticks):
    mod._GL.reset()
    p = Policy(chunk)
    p._action_queue.extend(chunk[:, 0:N].transpose(0, 1))  # what _gl_caching_predict does
    p._gl_consumed = N
    eng, served, lock_tick = Engine(p), [], None
    for t in range(ticks):
        a = _sync.SyncInferenceEngine.get_action(eng, {"obs": None})
        served.append(None if a is None else int(round(float(a[0]))))
        if lock_tick is None and mod._GL.locked:
            lock_tick = t
    return served, lock_tick, p._gl_consumed

# a real close: gripper 40 -> 20 at 1 deg/tick, lift parked in the grasp band
served, lock_tick, consumed = run(
    chunk_of(lambda t: max(20.0, 40.0 - t), lambda t: 38.0), 3 * N)
# a plain descent: gripper held OPEN, lift coming down 1 deg/tick
_, descent_lock, _ = run(
    chunk_of(lambda t: 40.0, lambda t: max(20.0, 49.0 - t)), 3 * N)

print(json.dumps({"gate_opened": gate_opened, "served": served,
                  "lock_tick": lock_tick, "consumed": consumed,
                  "descent_lock_tick": descent_lock}))
'''


def _probe(mode, gate):
    env = dict(os.environ)
    env.pop("CAPSTONE_GRASP_LOCK", None)
    env.pop("CAPSTONE_HOME_POSE", None)  # the wrapper indexes TRAIN_HOME with it
    if gate is not None:
        env["CAPSTONE_GRASP_LOCK"] = gate
    out = subprocess.run([PY_BIN, "-c", _PROBE, mode, str(TOOLS)],
                         capture_output=True, text=True, timeout=600, env=env)
    assert out.returncode == 0, f"grasp-lock probe ({mode}) crashed:\n{out.stderr[-2000:]}"
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def gl_probe():
    """One gate-ON subprocess shared by the wiring tests -- importing lerobot
    costs ~4 s and there is nothing per-test about it."""
    return _probe("gate-on", "1")


def test_wrapper_is_off_unless_the_env_var_is_set():
    """Imports the wrapper with CAPSTONE_GRASP_LOCK unset and checks lerobot's
    classes come back untouched. `"CAPSTONE_GRASP_LOCK" in src` could not tell
    `== "1"` from `!= "0"`; this can."""
    assert _probe("gate-off", None)["patched"] is False, (
        "importing tools/rollout_30hz_stale_ok.py with CAPSTONE_GRASP_LOCK "
        "UNSET patched SyncInferenceEngine.get_action / "
        "ACTPolicy.predict_action_chunk -- the gate's polarity is inverted, so "
        "every scored trial would silently run the chunk-locking probe "
        "configuration instead of the protocol's n_action_steps=25/50"
    )


def test_wrapper_watches_the_gripper_and_lift_action_columns(gl_probe):
    assert gl_probe["gate_opened"] is True, (
        "CAPSTONE_GRASP_LOCK=1 did not patch SyncInferenceEngine.get_action -- "
        "the wrapper no longer installs the lock at all"
    )
    assert gl_probe["lock_tick"] == 2, (
        "a 1 deg/tick gripper close inside the grasp band did not lock the "
        f"chunk on its 2nd closing tick (locked at tick {gl_probe['lock_tick']}). "
        "_GL_LIFT_IDX/_GL_GRIP_IDX are reading the wrong action columns; the "
        "dataset action order is pan, lift, elbow, wflex, wroll, gripper"
    )
    assert gl_probe["descent_lock_tick"] is None, (
        "a plain descent with the gripper held OPEN locked the chunk at tick "
        f"{gl_probe['descent_lock_tick']} -- lift and gripper are swapped, so "
        "the lock fires on every approach and never at the actual close"
    )


def test_locked_refills_advance_through_the_cached_chunk(gl_probe):
    """While LOCKED the wrapper refills the queue from the cached chunk, n
    ticks at a time. Each refill must serve the NEXT slice."""
    assert gl_probe["lock_tick"] is not None, (
        "the lock never engaged on the probe's closing trace, so the refill "
        "path was never exercised -- fix the detector wiring first (see "
        "test_wrapper_watches_the_gripper_and_lift_action_columns)"
    )
    served, expected = gl_probe["served"], list(range(75))
    assert len(served) == len(expected)
    bad = next((i for i, (a, b) in enumerate(zip(served, expected)) if a != b), None)
    assert served == expected, (
        "while LOCKED the wrapper re-served a stale slice: control tick "
        f"{bad} replayed cached chunk row {served[bad]} instead of row "
        f"{expected[bad]}. _gl_consumed is not accumulating, so the chunk never "
        "exhausts, the release never fires, and a failed close becomes an "
        "open-loop press into the rack with no escape path"
    )
    assert gl_probe["consumed"] == 75, (
        f"_gl_consumed is {gl_probe['consumed']} after 3 chunk slices; it must "
        "count 75 for the chunk-exhaustion release to ever fire"
    )


def test_pure_module_has_no_lerobot_dependency():
    src = (Path(__file__).parent.parent / "tools" / "grasp_lock.py").read_text()
    assert "lerobot" not in src, "grasp_lock.py must stay a pure, dependency-free state machine"
