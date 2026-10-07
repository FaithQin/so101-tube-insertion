"""Regression: the four LOCAL site-packages patches are still applied.

Every one of these dies silently on `pip install -U lerobot` and each one,
missing, has already cost a session:

1. configs/default.py — JobConfig.timeout "8h" (cost guardrail; upstream "2d").
2. robots/.../so_follower.py — Present_Load/Present_Current in the observation
   (18-dim state; the load-sensing A/B is unrecoverable without it) AND
   gripper Max_Torque_Limit 300 (upstream 500 trips the overload latch).
3. rollout/inference/sync.py — queue-skip (per-tick image preprocessing capped
   the loop at ~15 Hz on MPS).
4. rollout/context.py — .load/.current routed into observation.state for
   18-dim policies (ACT-B crashed 6-vs-18 without it, Aug 15).

References with full diffs/markers live in tools/ (so_follower_load.patch,
sync_queue_skip_patch.py.txt, rollout_context_load_state_patch.txt).
"""

import re


def _read(site_packages, rel):
    p = site_packages / rel
    assert p.exists(), f"{rel} not found under {site_packages}"
    return p.read_text()


def test_timeout_guardrail(site_packages):
    src = _read(site_packages, "configs/default.py")
    m = re.search(r"timeout[^\n=]*=\s*\"(\w+)\"", src)
    assert m, "JobConfig.timeout not found in configs/default.py"
    assert m.group(1) == "8h", (
        f"JobConfig.timeout is \"{m.group(1)}\" — the 8h cost guardrail is gone "
        f"(upstream default 2d; a hung job burns two days of GPU credit)"
    )


# --------------------------------------------------------------------------
# so_follower: run the patched methods against a stub bus.
#
# Every so_follower assertion here used to be a source grep, and every one of
# them was mutation-proven blind: the name being present says nothing about
# which motor a register reaches or which register a channel is filled from.
# --------------------------------------------------------------------------

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


class _StubBus:
    """Just enough FeetechMotorsBus to run the patched methods off-arm.

    Records every write with the motor it was addressed to — the thing a
    source grep can never answer.
    """

    def __init__(self, reads):
        self.motors = {m: object() for m in JOINTS}
        self.reads = reads
        self.writes = []  # [(register, motor, value)]
        self.sync_writes = []  # [(register, {motor: value})]
        self.is_connected = True
        self.raise_on_sync_read = 0

    def torque_disabled(self):
        import contextlib

        return contextlib.nullcontext()

    def configure_motors(self):
        pass

    def write(self, register, motor, value, **kwargs):
        self.writes.append((register, motor, value))

    def read(self, register, motor, **kwargs):
        return self.reads[register][motor]

    def sync_read(self, register, num_retry=0, **kwargs):
        if self.raise_on_sync_read > 0:
            self.raise_on_sync_read -= 1
            raise RuntimeError("simulated gripper-overload bus fault")
        return dict(self.reads[register])

    def sync_write(self, register, values, **kwargs):
        self.sync_writes.append((register, dict(values)))


def _bench_follower(site_packages, reads):
    """A REAL SOFollower wired to a stub bus: no serial port, no arm, no camera."""
    from pathlib import Path
    from types import SimpleNamespace

    import lerobot.robots.so_follower.so_follower as mod

    # Never grade a different install than the fixture is reading.
    assert Path(mod.__file__).resolve() == (
        site_packages / "robots/so_follower/so_follower.py"
    ).resolve(), f"imported so_follower is {mod.__file__}, not the one under {site_packages}"

    follower = mod.SOFollower.__new__(mod.SOFollower)
    follower.id = "bench"
    follower.cameras = {}
    follower.bus = _StubBus(reads)
    follower.config = SimpleNamespace(
        num_read_retries=1,
        max_relative_target=None,
        position_p_coefficient=16,
        position_i_coefficient=0,
        position_d_coefficient=32,
    )
    return mod, follower, follower.bus


def test_so_follower_load_observation(site_packages):
    """RUN the observation read and check WHERE each register's value lands.

    The old version asserted only that `sync_read("Present_Load")` and
    `sync_read("Present_Current")` appear in the source. Both calls survive a
    one-word copy-paste slip on the `.current` line (`for motor, val in
    loads.items()`), so the suite stayed green while dims 12-17 of every newly
    recorded episode became an exact duplicate of dims 6-11 — observation.state
    still 18-dim, nothing crashing, and the whole research question answered on
    a dataset whose current channel carries no information. The same grep also
    survived folding the two dict-splats into one per-motor comprehension,
    which keeps both names but interleaves the channels away from the
    pos/load/current order ACT-B and SmolVLA-B trained on (v3 meta/info.json).

    So: a distinct sentinel per register, and assert the mapping AND the order.
    """
    reads = {
        "Present_Position": {m: 10.0 + i for i, m in enumerate(JOINTS)},
        "Present_Load": {m: 100.0 + i for i, m in enumerate(JOINTS)},
        "Present_Current": {m: 1000.0 + i for i, m in enumerate(JOINTS)},
    }
    _, follower, _bus = _bench_follower(site_packages, reads)
    obs = follower._read_bus_state()

    expected = {}
    for reg, suffix in (
        ("Present_Position", "pos"),
        ("Present_Load", "load"),
        ("Present_Current", "current"),
    ):
        expected.update({f"{m}.{suffix}": reads[reg][m] for m in JOINTS})

    wrong = {k: (obs.get(k), v) for k, v in expected.items() if obs.get(k) != v}
    assert not wrong, (
        "observation channels are wired to the wrong register — {key: (got, want)} "
        f"{wrong}. Sentinels: .pos=10x, .load=100x, .current=1000x. A .current "
        f"reading 100x is the load value copied into the current channel; every "
        f"episode recorded like this has a duplicate, uninformative current "
        f"channel and cannot be retrofitted."
    )
    assert list(obs) == list(expected), (
        f"observation.state channel ORDER changed:\n  got  {list(obs)}\n  want "
        f"{list(expected)}\nv3 (meta/info.json) is 6 .pos, then 6 .load, then 6 "
        f".current; any other order permutes dims 6-17 away from what ACT-B and "
        f"SmolVLA-B were trained on, silently."
    )


def test_so_follower_gripper_stall_protection(site_packages):
    """CALL configure() and read back which motor each register reached.

    The gripper stall design as of Aug 16: Max_Torque_Limit stays at UPSTREAM
    500 (lowering to 300 made every bounded grip peg the cap and trip the 2 s
    Protection_Time latch — the Aug 13 ledger note saying 300 is STALE). Real
    stall safety = Protection_Time pinned to stock 200 (2.0 s) + the
    send_action force-floor + _relieve_gripper recovery (executed in the test
    below this one).

    The old version regexed those two write() lines out of the source, which is
    the "ABORT appears in a case branch, never that the branch exits" bug
    again: mistyping the guard to `if motor == "grippers":` makes the whole
    block DEAD — no protection register reaches any motor — while both regexes
    still match and `_relieve_gripper` is still defined. Feetech EEPROM applies
    at power-on, so a dead branch means the arm boots with whatever a
    diagnostic left in Protection_Time (the project notes: possibly 0 = instant trip).
    """
    _, follower, bus = _bench_follower(site_packages, {})
    follower.configure()

    protection = ("Max_Torque_Limit", "Protection_Time", "Protection_Current", "Overload_Torque")
    gripper = {reg: val for reg, motor, val in bus.writes if motor == "gripper"}
    assert set(protection) <= set(gripper), (
        f"configure() never wrote {sorted(set(protection) - set(gripper))} to the "
        f"gripper — the `if motor == \"gripper\"` branch did not run. Motors "
        f"actually written: {sorted({m for _, m, _ in bus.writes})}. The servo "
        f"keeps whatever a diagnostic left in EEPROM and applies it at power-on."
    )
    assert gripper["Max_Torque_Limit"] == 500, (
        f"gripper Max_Torque_Limit is {gripper['Max_Torque_Limit']}, not 500 — a "
        f"lower cap pegs on every bounded grip and trips the Protection_Time "
        f"latch (measured Aug 13-16; the cap must exceed P*(lead+slack) = 416)"
    )
    assert gripper["Protection_Time"] == 200, (
        f"gripper Protection_Time is {gripper['Protection_Time']}, not 200 (2.0 s) "
        f"— diagnostics leave this EPROM register at 0 (instant trip) or 250"
    )
    leaked = sorted({(reg, motor) for reg, motor, _ in bus.writes if motor != "gripper" and reg in protection})
    assert not leaked, (
        f"stall-protection registers were written to non-gripper motors: {leaked}. "
        f"The guard is meant to scope these to the gripper alone."
    )


def test_so_follower_gripper_recovery_paths_are_live(site_packages, monkeypatch):
    """The three things the docstring above calls the REAL stall safety, executed.

    None of them was ever asserted, and three sibling mutations proved it: the
    force-floor flipped to `min(...)` (which commands continuous closing), the
    relief command reversed to `relief - 2.0` (the one recovery path closes the
    jaws HARDER), and the get_observation try/except deleted (a mid-grip
    overload kills the session instead of relieving and retrying) all left the
    full suite green.
    """
    reads = {
        "Present_Position": {m: 50.0 for m in JOINTS},
        "Present_Load": {m: 0.0 for m in JOINTS},
        "Present_Current": {m: 0.0 for m in JOINTS},
    }

    # (1) relief must OPEN the jaws.
    mod, follower, bus = _bench_follower(site_packages, reads)
    monkeypatch.setattr(mod.time, "sleep", lambda *_a, **_k: None)
    follower._last_gripper_present = 30.0
    follower._relieve_gripper()
    goals = [v["gripper"] for reg, v in bus.sync_writes if reg == "Goal_Position"]
    assert goals and goals[-1] > 30.0, (
        f"_relieve_gripper commanded {goals[-1] if goals else None} from a present "
        f"of 30.0 — the one recovery path from an overload fault must open the "
        f"jaws; commanding lower closes them harder and holds the strain"
    )

    # (2) the force-floor must clamp a violent close.
    _mod, follower, bus = _bench_follower(site_packages, reads)
    sent = follower.send_action({f"{m}.pos": 0.0 for m in JOINTS})
    assert sent["gripper.pos"] > 0.0 and sent["gripper.pos"] >= 45.0, (
        f"send_action passed a fully-closed command through as "
        f"{sent['gripper.pos']} from a present of 50.0 — the bounded closing "
        f"lead is gone, so a sustained error saturates output and the firmware "
        f"latches after Protection_Time (2 s)"
    )

    # (3) a bus fault mid-grip must be relieved and retried, not fatal.
    _mod, follower, bus = _bench_follower(site_packages, reads)
    bus.raise_on_sync_read = 1
    obs = follower.get_observation()
    assert obs["gripper.pos"] == 50.0, f"retry after relief returned {obs.get('gripper.pos')}"
    assert any(reg == "Goal_Position" for reg, _ in bus.sync_writes), (
        "get_observation let a mid-grip bus fault propagate — no relief command "
        "was sent, so the session dies with the strain still commanded"
    )


def test_sync_engine_queue_skip(site_packages):
    """DRIVE the engine: queue-served ticks must skip preprocessing and still
    be POSTprocessed.

    The old version grepped for "LOCAL PATCH (capstone)" and "_action_queue",
    which live in a comment and a getattr — the whole body of the fast path can
    be rewritten under them. The realistic mis-reapplication (the project notes
    requires re-applying this by hand after every `pip install -U lerobot`,
    which is exactly when a hunk gets lost) is dropping the postprocessor call.
    The postprocessor is the action UNnormalizer, and with n_action_steps
    locked at 25 (ACT-A) / 50 (ACT-B) the queue serves 24 of every 25 ticks —
    those would reach make_robot_action, and the servos, as raw ~zero-mean
    normalized values interpreted as joint angles.
    """
    from collections import deque
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np
    import torch

    import lerobot.rollout.inference.sync as sync_mod

    assert Path(sync_mod.__file__).resolve() == (
        site_packages / "rollout/inference/sync.py"
    ).resolve(), f"imported sync engine is {sync_mod.__file__}, not the one under {site_packages}"

    calls = []

    class _Policy:
        def __init__(self, ensemble=None):
            self._action_queue = deque()
            self.config = SimpleNamespace(temporal_ensemble_coeff=ensemble, use_amp=False)

        def select_action(self, observation):
            calls.append("select_action")
            return torch.tensor([[7.0, 8.0]])

    def _preprocessor(observation):
        calls.append("preprocess")
        return observation

    def _postprocessor(action):
        # Stand-in for the real unnormalizer: an affine map, so a skipped
        # postprocess step is visible in the numbers and not just in a flag.
        calls.append("postprocess")
        return action * 10.0 + 1.0

    def _engine(policy):
        return sync_mod.SyncInferenceEngine(
            policy=policy,
            preprocessor=_preprocessor,
            postprocessor=_postprocessor,
            dataset_features={"action": {"names": ["a.pos", "b.pos"]}},
            ordered_action_keys=["b.pos", "a.pos"],
            task="insert the tube",
            device="cpu",
            robot_type="so101_follower",
        )

    # --- queue-served tick: skip the preprocessing, keep the unnormalizer ---
    policy = _Policy()
    policy._action_queue.append(torch.tensor([[2.0, 3.0]]))
    calls.clear()
    out = _engine(policy).get_action({"observation.state": np.zeros(6, dtype=np.float32)})
    assert "postprocess" in calls, (
        f"the queue-served tick ran {calls} — it never called the postprocessor, "
        f"which is the action UNnormalizer. With n_action_steps locked at 25 "
        f"(ACT-A) / 50 (ACT-B) the queue serves 24 of every 25 ticks, so those "
        f"commands reach the servos as raw ~zero-mean normalized values "
        f"interpreted as joint angles."
    )
    assert torch.equal(out, torch.tensor([31.0, 21.0])), (
        f"queue-served action is {out.tolist()}, expected [31.0, 21.0] "
        f"(postprocessed [2,3] -> [21,31], reordered to b.pos,a.pos). Raw "
        f"[3.0, 2.0] means the action is not being unnormalized on the fast path."
    )
    assert calls == ["postprocess"], (
        f"queue-served tick ran {calls}, expected ['postprocess'] only — the "
        f"skip is not skipping, which is the ~15 Hz MPS regression this patch "
        f"exists to fix"
    )

    # --- empty queue: the full pipeline still runs (the control) ---
    calls.clear()
    out = _engine(_Policy()).get_action({"observation.state": np.zeros(6, dtype=np.float32)})
    assert calls == ["preprocess", "select_action", "postprocess"], (
        f"slow path ran {calls} — the full pipeline must still run when the "
        f"queue is empty, otherwise the policy never sees an observation"
    )
    assert torch.equal(out, torch.tensor([81.0, 71.0])), f"slow-path action is {out.tolist()}"

    # --- temporal ensembling: the fast path must NOT engage ---
    policy = _Policy(ensemble=0.01)
    policy._action_queue.append(torch.tensor([[2.0, 3.0]]))
    calls.clear()
    _engine(policy).get_action({"observation.state": np.zeros(6, dtype=np.float32)})
    assert "select_action" in calls, (
        "the queue skip engaged with temporal ensembling ON. The ensembler lives "
        "inside select_action; skipping it silently disables ensembling on the "
        "very arm of the n-probe grid that exists to test it."
    )


# --------------------------------------------------------------------------
# context.py state routing: EXECUTE the decision, don't grep for its words.
#
# Every assertion in this section used to be a source grep, and all four were
# mutation-proven blind — the strings the greps looked for live in the prose
# comment block above the code, in a logger call, and in an exception message,
# so the routing rule itself could be gutted with the suite still green.
# --------------------------------------------------------------------------


def _run_state_routing(site_packages, policy_state_dim, max_state_dim=None, obs_features=None):
    """Cut the routing decision out of context.py, run it, return its locals.

    The rule is inline in `init_rollout_context` (it cannot import from this
    repo — see tools/state_routing.py for the reference implementation), so it
    cannot be called directly without a robot, a policy and two cameras.
    Instead: slice the source between the `_state_suffixes` seed and the
    `action_features_hw` line — which is the decision PLUS the
    `observation_features_hw` comprehension it feeds — and exec that with stub
    inputs. What comes back is what the rollout and, crucially, what
    dataset_features would be built from.
    """
    import os
    import textwrap
    from types import SimpleNamespace

    src = _read(site_packages, "rollout/context.py")
    lines = src.splitlines(True)
    start = next((i for i, ln in enumerate(lines) if ln.startswith("    _state_suffixes = ")), None)
    end = next((i for i, ln in enumerate(lines) if ln.startswith("    action_features_hw = ")), None)
    assert start is not None and end is not None and start < end, (
        "could not find the state-routing block in rollout/context.py "
        "(anchors: `    _state_suffixes = ` .. `    action_features_hw = `) — "
        "the local patch is gone or the upstream function was restructured"
    )
    block = textwrap.dedent("".join(lines[start:end]))

    if obs_features is None:
        obs_features = {f"{m}.pos": float for m in JOINTS}
        obs_features.update({f"{m}.load": float for m in JOINTS})
        obs_features.update({f"{m}.current": float for m in JOINTS})
        obs_features["front"] = (480, 640, 3)

    input_features = {}
    if policy_state_dim is not None:
        input_features["observation.state"] = SimpleNamespace(shape=(policy_state_dim,))
    policy_config = SimpleNamespace(input_features=input_features)
    if max_state_dim is not None:
        policy_config.max_state_dim = max_state_dim

    class _Recorder:
        def __init__(self):
            self.lines = []

        def info(self, msg, *args):
            self.lines.append(msg % args if args else msg)

        warning = info

    ns = {
        "os": os,
        "logger": _Recorder(),
        "all_obs_features": obs_features,
        "policy_config": policy_config,
    }
    exec(compile(block, str(site_packages / "rollout/context.py"), "exec"), ns)  # noqa: S102
    return ns


def _routing(site_packages, **kwargs):
    """(decision, state keys the filter yields) for one policy shape."""
    ns = _run_state_routing(site_packages, **kwargs)
    assert "_widen" in ns, (
        "rollout/context.py made no _widen decision — the local state-routing "
        "patch is gone (pip install -U lerobot?). Re-apply it from "
        "tools/rollout_context_load_state_patch.txt before any rollout."
    )
    return ns["_widen"], [k for k, v in ns["observation_features_hw"].items() if v is float]


def test_context_load_state_filter(site_packages):
    """ACT-B's 18 state keys must actually come out of the filter.

    The old assertion was `".load" in src and ".current" in src and
    "_state_suffixes" in src` — and both suffix strings appear in the prose
    comment at context.py:350 describing the patch, so the widened tuple (the
    only line that does any work) could be truncated to `(".pos", ".vel",
    ".load")` with the suite green. That yields 12 state keys: a 12-vs-18
    crash on an ACT-B rollout, and — worse — a silent 12-dim recording under
    CAPSTONE_FORCE_WIDE_STATE=1, which looks plausible in the meta because the
    state is still wider than 6.
    """
    _widen, wide = _routing(site_packages, policy_state_dim=18)
    expected = (
        [f"{m}.pos" for m in JOINTS] + [f"{m}.load" for m in JOINTS] + [f"{m}.current" for m in JOINTS]
    )
    assert wide == expected, (
        f"an 18-dim policy routes {len(wide)} state keys, not 18:\n  got  {wide}\n"
        f"  want {expected}\nmissing {sorted(set(expected) - set(wide))} — on a "
        f"rollout that is the 6-vs-18 normalizer crash; on a DAgger session it "
        f"writes a dataset with those channels dropped at the source."
    )
    _widen, narrow = _routing(site_packages, policy_state_dim=6)
    assert narrow == [f"{m}.pos" for m in JOINTS], (
        f"a 6-dim policy (ACT-A) routes {narrow} — widening is supposed to be "
        f"conditional; feeding 18 dims into a 6-dim normalizer crashes it"
    )


def test_context_state_routing_handles_padded_policies(site_packages):
    """pi0.5 declares observation.state.shape == max_state_dim (32) whatever it
    trained on. pi05-tube-A-v3 trained on the 6-dim noload dataset, so dims 6-31
    were zero in every training frame. Without the padding exclusion the filter
    sees 32 > 6, routes load+current in, and hands the policy real values in
    coordinates it learned as always-zero.

    NOTHING CRASHES — the shapes line up. It is a silent distribution shift, and
    it would have presented as "pi0.5 doesn't work on this task". Found Aug 31
    2026 by reading the rollout path before the first pi0.5 bench session.

    The old version asserted the words "CAPSTONE_STATE_ROUTING" and
    "max_state_dim" were in the file — the "*)" regex failure verbatim: both
    survive in a logger call and a getattr, so deleting the entire
    padding-exclusion branch (a clean revert of the Aug 31 fix) kept the suite
    green. Run the rule against the real checkpoint's declared widths instead.
    """
    widen, padded = _routing(site_packages, policy_state_dim=32, max_state_dim=32)
    assert widen is False and padded == [f"{m}.pos" for m in JOINTS], (
        f"pi05-tube-A-v3 (declared 32, max_state_dim 32, trained on 6-dim noload) "
        f"routes {len(padded)} state keys: {padded}. 32 is padding, not "
        f"dimensionality — the padding-exclusion branch is gone, and load/current "
        f"are being fed into dims the policy learned as always-zero. Nothing will "
        f"crash; it will just look like pi0.5 doesn't work."
    )
    # Exact equality only: a policy that pads to 32 but genuinely declares 18
    # really is 18-dim and must still widen.
    widen, _genuine = _routing(site_packages, policy_state_dim=18, max_state_dim=32)
    assert widen is True, (
        "a genuinely 18-dim policy that pads to 32 was routed narrow — the "
        "padding exclusion has become a threshold instead of an equality test "
        "and now starves the B side"
    )


def test_context_state_routing_has_explicit_override(site_packages, monkeypatch):
    """DAgger builds dataset_features from this same filtered dict, so recording
    under 6-dim ACT-A writes a 6-dim dataset — load and current dropped at the
    source, unretrofittable, and the B-side twin foreclosed forever.
    CAPSTONE_FORCE_WIDE_STATE is the only way to record wide under a narrow
    policy.

    The old assertion was `"CAPSTONE_FORCE_WIDE_STATE" in src`. Deleting the two
    lines that let `_force` decide leaves the variable read, validated and
    logged — every string the grep wanted — while the pin decides nothing and
    routing silently falls back to inference. Set the variable and check the
    keys that come out, in both directions. (tools/smolvla_dryrun.py and
    run_smolvla_trial.sh:220 both pin it to 1 for the B side, because
    smolvla-tube-B-v3 declares [6] while carrying 18-dim normalizer stats.)
    """
    monkeypatch.setenv("CAPSTONE_FORCE_WIDE_STATE", "1")
    widen, forced_wide = _routing(site_packages, policy_state_dim=6)
    assert widen is True and len(forced_wide) > 6, (
        f"CAPSTONE_FORCE_WIDE_STATE=1 under a 6-dim policy still routed "
        f"{len(forced_wide)} state keys: {forced_wide}. The override is inert — "
        f"SmolVLA-B trials die 6-vs-18 on the first tick, and a correction "
        f"session records narrow with the load channel gone for good."
    )

    monkeypatch.setenv("CAPSTONE_FORCE_WIDE_STATE", "0")
    widen, forced_narrow = _routing(site_packages, policy_state_dim=18)
    assert widen is False and forced_narrow == [f"{m}.pos" for m in JOINTS], (
        f"CAPSTONE_FORCE_WIDE_STATE=0 under an 18-dim policy routed "
        f"{len(forced_narrow)} state keys, not 6 — the override is a real switch "
        f"in both directions or it is not an override"
    )

    monkeypatch.delenv("CAPSTONE_FORCE_WIDE_STATE")
    widen, inferred = _routing(site_packages, policy_state_dim=6)
    assert widen is False and len(inferred) == 6, (
        f"unset must mean INFER, not force-wide: a 6-dim policy routed "
        f"{len(inferred)} keys with the variable unset"
    )


def test_context_state_routing_rejects_unparseable_override(site_packages, monkeypatch):
    """A typo'd override must raise, not fall back to inference. Falling back
    would mean a session the operator believes is recording 18-dim records 6.

    The old assertion looked for "is not a boolean" — which lives in the
    MESSAGE, so swapping the `raise ValueError(...)` for a `logger.warning(...)`
    plus `_force = None` left it intact while restoring exactly the fallback the
    docstring forbids. Run it with a typo and require the exception.
    """
    monkeypatch.setenv("CAPSTONE_FORCE_WIDE_STATE", "ture")
    try:
        _widen, keys = _routing(site_packages, policy_state_dim=6)
    except ValueError as e:
        assert "CAPSTONE_FORCE_WIDE_STATE" in str(e), (
            f"the override rejection does not name the variable: {e}"
        )
        return
    raise AssertionError(
        "CAPSTONE_FORCE_WIDE_STATE='ture' did not raise; routing fell back to "
        f"inference and produced {len(keys)} state keys. One mistyped character "
        "in the pin now costs the load channel of an entire correction session, "
        "with a warning buried in the rollout log."
    )
