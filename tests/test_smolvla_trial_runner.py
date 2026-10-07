"""Regression: the SmolVLA trial runner, pinned against tonight's two bugs.

Written Sep 3 2026 (evening). No SmolVLA runner existed before this file --
`ls tools/*smolvla*` returned nothing -- while the Sep 4 10:00 block schedules
three trials per side.

Two bugs found TODAY define what this file tests, and a third defines HOW.

**Bug 1 -- the empty prompt.** `run_pi05_trial.sh` passed `--dataset.*` flags but
neither `--task` nor `--dataset.single_task`. `rollout/configs.py:244` leaves
`task: str = ""`, neither propagation branch fires (configs.py:365-370), and
`context.py:562` hands the inference engine `task=""`. SmolVLA is
language-conditioned the same way, so the same omission produces the same silent
distribution shift. The runner must pass a task string, it must be read from the
dataset rather than typed, and it must be non-empty.

**Bug 2 -- the state-width pin.** `smolvla-tube-B-v3` declares
`observation.state [6]` while carrying 18-dim normalizer stats; the live routing
in `context.py:400-407` reads the DECLARED width, so without
`CAPSTONE_FORCE_WIDE_STATE=1` a 6-dim state meets an 18-dim normalizer and
raises on the first inference tick. Its A-side twin declares the same `[6]` and
genuinely IS 6-dim, so it must be pinned to `0`. Nothing but the pin separates
them.

**How -- strip the comments before scanning.** The Sep 3 night audit found
`test_pi05_trial_runner.py::test_runner_forces_narrow_state_routing_explicitly`
was satisfied by a COMMENT (`run_pi05_trial.sh:22` contains the literal
`CAPSTONE_FORCE_WIDE_STATE=0`), so deleting the real `export` left the test
green. Substring-scanning raw source is the bug class. Every scan below runs on
`_code()`, and `test_the_comment_stripper_actually_strips` pins the stripper
itself.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import TOOLS

sys.path.insert(0, str(TOOLS))

import smolvla_dryrun as sd  # noqa: E402

RUNNER = TOOLS / "run_smolvla_trial.sh"
HUB = Path.home() / ".cache/huggingface/hub"
PY_BIN = "/opt/homebrew/Caskroom/miniforge/base/envs/lerobot/bin/python"


def _src() -> str:
    assert RUNNER.exists(), f"{RUNNER} missing"
    return RUNNER.read_text()


def _code() -> str:
    """The runner with shell comments stripped.

    A `#` opens a comment only when unquoted and at line start or after
    whitespace -- so `${REPO##*/}` survives. Same stripper as
    `test_pi05_trial_runner._code`, duplicated on purpose: importing it would
    bind this file's assertions to the other runner's path.
    """
    out = []
    for line in _src().splitlines():
        q = None
        cut = len(line)
        for i, ch in enumerate(line):
            if q:
                if ch == q:
                    q = None
            elif ch in "'\"":
                q = ch
            elif ch == "#" and (i == 0 or line[i - 1].isspace()):
                cut = i
                break
        out.append(line[:cut])
    return "\n".join(out)


def _train_config(side: str) -> dict:
    snaps = sorted((HUB / f"models--faithqin--smolvla-tube-{side}-v3" / "snapshots").glob("*"))
    if not snaps or not (snaps[-1] / "train_config.json").exists():
        pytest.skip(f"smolvla-tube-{side}-v3 train_config.json not in the Hub cache")
    return json.loads((snaps[-1] / "train_config.json").read_text())


# ---------------------------------------------------------------------------
# 0. The scanner itself
# ---------------------------------------------------------------------------


def test_the_comment_stripper_actually_strips():
    """If this ever regresses, every scan below becomes satisfiable by prose."""
    code = _code()
    assert "#!/bin/zsh" not in code
    assert "${REPO##*/}" in code or "##*/" not in _src(), "the stripper ate a ## expansion"


def test_runner_exists_and_is_executable():
    assert os.access(RUNNER, os.X_OK), "run_smolvla_trial.sh is not executable"


# ---------------------------------------------------------------------------
# 1. BUG 1 -- a non-empty task string, equal to the dataset's
# ---------------------------------------------------------------------------


def test_runner_passes_a_task_string_at_all():
    code = _code()
    assert "--dataset.single_task=" in code or "--task=" in code, (
        "runner passes no task string: SmolVLA would receive an EMPTY language "
        "prompt on hardware, exactly as pi0.5 was about to"
    )


def test_runner_reads_the_task_from_the_dataset_instead_of_hardcoding_it():
    """A literal in the script goes stale silently the day the dataset changes."""
    code = _code()
    m = re.search(r"--(?:dataset\.single_task|task)=(\S+)", code)
    assert m, "could not parse the task argument out of the runner"
    assert "$" in m.group(1), (
        f"task argument {m.group(1)!r} is a literal; it must expand a variable "
        f"resolved from the dataset's meta/tasks.parquet"
    )


def test_the_task_the_runner_resolves_is_non_empty_and_is_the_trained_one():
    """Executes the runner's own resolver, then compares against the parquet.

    This is the test that would have caught the pi0.5 bug: the pi0.5 rehearsal
    fed the trained string from a python constant while the runner fed nothing,
    so nothing compared the two. Here the runner and the dry-run share one
    resolver and this asserts on its output.
    """
    import pandas as pd

    for name, side in sd.SIDES.items():
        try:
            resolved = sd.trained_task(side.dataset)
        except sd.TaskUnavailable:
            pytest.skip(f"{side.dataset} meta/tasks.parquet not reachable")
        assert resolved.strip(), f"side {name} resolves an EMPTY task string"

        out = subprocess.run(
            [PY_BIN, str(TOOLS / "smolvla_dryrun.py"), "--print", "task", "--side", name],
            capture_output=True, text=True, timeout=180,
        )
        assert out.returncode == 0, f"--print task failed for side {name}: {out.stderr[-500:]}"
        assert out.stdout.strip() == resolved

        parquet = sd.tasks_parquet_path(side.dataset)
        trained = list(pd.read_parquet(parquet).index)[0]
        assert resolved == trained, (
            f"side {name} would pass {resolved!r} but trained on {trained!r}"
        )


def test_runner_aborts_when_the_task_cannot_be_resolved():
    """An unresolvable prompt must stop the run, not launch with an empty one."""
    code = _code()
    assert re.search(r'-z\s+"?\$\{?TASK', code), (
        "runner does not guard against an empty TASK: the pi0.5 failure mode "
        "was an empty prompt that nothing objected to"
    )


# ---------------------------------------------------------------------------
# 2. BUG 2 -- the per-side CAPSTONE_FORCE_WIDE_STATE pin
# ---------------------------------------------------------------------------


def test_runner_pins_the_force_variable_explicitly():
    code = _code()
    assert "CAPSTONE_FORCE_WIDE_STATE" in code, (
        "runner does not pin CAPSTONE_FORCE_WIDE_STATE; smolvla-tube-B-v3 would "
        "route narrow into an 18-dim normalizer and raise on tick 1"
    )
    assert re.search(r"export\s+CAPSTONE_FORCE_WIDE_STATE=", code), (
        "the pin is not exported into the rollout's environment"
    )


def test_the_pin_is_resolved_per_side_not_fixed():
    """One hardcoded value would be right for exactly one of the two twins."""
    code = _code()
    m = re.search(r"export\s+CAPSTONE_FORCE_WIDE_STATE=(\S+)", code)
    assert m, "could not parse the exported pin"
    assert "$" in m.group(1), (
        f"pin is hardcoded to {m.group(1)!r}; A needs 0 and B needs 1, and both "
        f"checkpoints declare state width [6] so nothing else separates them"
    )


@pytest.mark.parametrize("side,expected", [("A", "0"), ("B", "1")])
def test_the_resolved_pin_is_narrow_for_a_and_wide_for_b(side, expected):
    """Executes the resolver the runner calls, rather than reading the table."""
    out = subprocess.run(
        [PY_BIN, str(TOOLS / "smolvla_dryrun.py"), "--print", "force", "--side", side],
        capture_output=True, text=True, timeout=180,
    )
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.strip() == expected


# ---------------------------------------------------------------------------
# The stubbed-runner harness — behaviour, without the arm
# ---------------------------------------------------------------------------
#
# Every scan above could be satisfied by a name sitting in the right
# neighbourhood, and the Sep 5 audit proved it on this file: an inverted `-n`,
# `$?` in place of `${pipestatus[1]}`, a dropped `"$DATASET_TAG"`, a hardcoded
# `--side A`, transposed camera indices, `--fps=30`, the scene photographed
# before the arm was homed, `unset CAPSTONE_GRASP_LOCK_V2`. Eight real bugs,
# the full suite green through every one of them, because the words were all
# still there.
#
# So RUN the runner. A copy of it, in a scratch directory, with its ONE line of
# interpreter path (`B=...`) repointed at a stub `python` that records the argv
# and environment of every call and answers with its own argv, and with `say`
# shadowed on PATH so a pytest run cannot speak over a bench session. Nothing
# else about the script is changed.
#
# This is safe in the default `pytest tests/` — unlike the executing tests in
# section 5, which run the real resolvers. Under the stubs the only external
# commands left are `date`, `mkdir`, `sed`, `grep`, `tee` and `ls`: no serial
# port, no camera, no speaker, no Hub, no ledger row.

import functools  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402

# The single line the harness rewrites. Asserted to appear exactly once, so
# moving the runner to another interpreter fails loudly here instead of
# silently testing a script nobody edited.
_INTERP_LINE = "B=/opt/homebrew/Caskroom/miniforge/base/envs/lerobot/bin"

_STUB_PY = r"""#!/bin/sh
# Stand-in for the lerobot env's python. Appends one record per call --
# \0-separated fields, \36-terminated -- carrying the three environment
# variables the runner is supposed to control, then the argv. Answers with its
# own argv so the runner's $(...) captures come back non-empty. Exits non-zero
# when CAPSTONE_STUB_FAIL matches, which is how a gate is made to fail.
{
  printf 'GRASP=%s\0' "${CAPSTONE_GRASP_LOCK-__UNSET__}"
  printf 'FORCE=%s\0' "${CAPSTONE_FORCE_WIDE_STATE-__UNSET__}"
  printf 'HOME_POSE=%s\0' "${CAPSTONE_HOME_POSE-__UNSET__}"
  printf 'ARGV\0'
  printf '%s\0' "$@"
  printf '\36'
} >> "$CAPSTONE_STUB_LOG"
case "$*" in
  *"--print gen "*) printf 'v4\n' ;;   # the side table's generation (Sep 6): a real tag, not argv
  *) printf '%s\n' "$*" ;;
esac
case "$*" in
  *"${CAPSTONE_STUB_FAIL:-__no_tool_is_named_this__}"*) exit "${CAPSTONE_STUB_RC:-1}" ;;
esac
exit 0
"""

_STUB_SAY = r"""#!/bin/sh
printf '%s\n' "$*" >> "$CAPSTONE_SAY_LOG"
"""


class _StubbedRun:
    """What one stubbed execution of the runner actually did."""

    def __init__(self, proc, calls, spoken):
        self.rc = proc.returncode
        self.out = proc.stdout + proc.stderr
        self.calls = calls  # [{"env": {...}, "argv": [...]}, ...], in order
        self.spoken = spoken

    def tools_in_order(self):
        return [c["argv"][0] if c["argv"][0] != "-c" else "python -c" for c in self.calls]

    def calls_to(self, tool):
        return [c for c in self.calls if c["argv"] and c["argv"][0].endswith(tool)]

    def only_call_to(self, tool):
        hits = self.calls_to(tool)
        assert len(hits) == 1, (
            f"expected exactly one {tool} call, the runner made {len(hits)}; "
            f"it ran {self.tools_in_order()} and exited {self.rc}"
        )
        return hits[0]


@functools.lru_cache(maxsize=None)
def _stub_run(side="B", label="HARNESS-TEST", nsteps="25", fail_on=None, env=()):
    """Execute the runner with every `$B/python` and `say` replaced by a stub.

    `env` is a tuple of pairs so this stays cacheable — the whole file's
    behavioural tests share four or five runs, at ~0.3 s each.
    """
    src = _src()
    assert src.count(_INTERP_LINE) == 1, (
        f"expected exactly one `{_INTERP_LINE}` line in the runner, found "
        f"{src.count(_INTERP_LINE)}: the harness cannot stub the interpreter, "
        f"and a test that cannot stub it must not run the real one"
    )
    work = Path(tempfile.mkdtemp(prefix="smolvla-runner-harness-"))
    try:
        tools = work / "proj" / "tools"
        tools.mkdir(parents=True)
        script = tools / RUNNER.name
        script.write_text(src.replace(_INTERP_LINE, 'B="${CAPSTONE_STUB_BIN:?}"'))
        script.chmod(0o755)
        stub = work / "bin"
        stub.mkdir()
        for name, body in (("python", _STUB_PY), ("say", _STUB_SAY)):
            (stub / name).write_text(body)
            (stub / name).chmod(0o755)
        calls_log, say_log = work / "calls", work / "spoken"
        calls_log.touch()
        say_log.touch()
        environ = {
            "PATH": f"{stub}:/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": os.environ["HOME"],
            "CAPSTONE_STUB_BIN": str(stub),
            "CAPSTONE_STUB_LOG": str(calls_log),
            "CAPSTONE_SAY_LOG": str(say_log),
            "CAPSTONE_ALLOW_WORKTREE": "1",
        }
        if fail_on:
            environ["CAPSTONE_STUB_FAIL"] = fail_on
        environ.update(dict(env))
        proc = subprocess.run(
            ["/bin/zsh", str(script), side, label, nsteps],
            capture_output=True, text=True, timeout=120, env=environ,
        )
        calls = []
        for record in calls_log.read_bytes().split(b"\x1e"):
            fields = [f.decode() for f in record.split(b"\0")]
            if "ARGV" not in fields:
                continue
            cut = fields.index("ARGV")
            calls.append({
                "env": dict(f.split("=", 1) for f in fields[:cut]),
                "argv": [f for f in fields[cut + 1:] if f],
            })
        return _StubbedRun(proc, calls, say_log.read_text().splitlines())
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _hatch_misbehaviour():
    """Run the runner both ways under the stubs. None when the hatch behaves.

    Shared with `_require_safe_to_execute_the_runner` in section 5, which has
    to decide whether it is safe to execute the REAL runner from pytest. A text
    check cannot make that decision — it passed for an inverted condition.
    """
    with_marker = _stub_run(env=(("CAPSTONE_RESOLVE_ONLY", "1"),))
    if with_marker.calls_to("home_arm.py"):
        return (
            "CAPSTONE_RESOLVE_ONLY=1 did NOT stop the runner — it went on to run "
            f"{with_marker.tools_in_order()}. The escape hatch is inverted or gone: "
            "the 10-second pre-flight would consume a gated trial, and the "
            "executing tests in this file would drive the arm."
        )
    if with_marker.rc != 0:
        return f"CAPSTONE_RESOLVE_ONLY=1 exited {with_marker.rc}, not 0"
    if "RESOLVE-ONLY" not in with_marker.out:
        return "CAPSTONE_RESOLVE_ONLY=1 stopped without announcing it"
    real = _stub_run()
    if not real.calls_to("home_arm.py"):
        return (
            "with CAPSTONE_RESOLVE_ONLY unset the runner STILL stopped before the "
            f"gates (it ran {real.tools_in_order()}, exit {real.rc}) — a real trial "
            "would exit 0 having launched nothing, announcing 'NO trial was executed'"
        )
    return None


def test_runner_gates_on_check_state_width_and_aborts_on_failure():
    """The gate has to STOP the runner, not just print a verdict.

    Was: `"exit"` somewhere in the six lines after the check_state_width line.
    That proves an `exit` is nearby, not that anything reaches it — and on Sep 5
    it did not. Reverting `${pipestatus[1]}` to the pre-fix `$?` reads the
    status of `sed`, the last command of
    `check_state_width.py ... | sed 's/^/  /'`, which is always 0; the gate
    printed its verdict, the `exit 4` four lines below became unreachable, and
    the full suite stayed green. So make the gate fail, and watch.
    """
    code = _code()
    assert "check_state_width.py" in code, (
        "runner skips the state-width gate; a declared-vs-stats disagreement "
        "would surface as a RuntimeError mid-episode instead of before launch"
    )
    failed = _stub_run(fail_on="check_state_width.py")
    assert failed.rc != 0, (
        f"check_state_width.py exited non-zero and the runner still returned "
        f"{failed.rc}: the state-width gate is decorative. A wrong "
        f"CAPSTONE_FORCE_WIDE_STATE reaches the bench and raises on the first "
        f"inference tick, mid-episode. (Is the gate reading ${{pipestatus[1]}}? "
        f"`$?` after a pipeline is sed's status, which is always 0.)"
    )
    assert not failed.calls_to("home_arm.py"), (
        f"a failed state-width gate did not stop the runner — it went on to run "
        f"{failed.tools_in_order()}"
    )
    assert not failed.calls_to("rollout_30hz_stale_ok.py"), (
        "a failed state-width gate still launched the rollout"
    )
    passing = _stub_run()
    assert passing.calls_to("rollout_30hz_stale_ok.py"), (
        f"with every gate passing the runner never reached the rollout — it "
        f"stopped after {passing.tools_in_order()} (exit {passing.rc})"
    )


def test_the_gate_checks_the_same_artifact_the_rollout_loads():
    """Gating repo X and launching path Y proves nothing."""
    code = _code()
    m_gate = re.search(r"check_state_width\.py\s+\"?\$\{?(\w+)", code)
    m_load = re.search(r"--policy\.path=\"?\$\{?(\w+)", code)
    assert m_gate and m_load, "could not parse the gated path and the loaded path"
    assert m_gate.group(1) == m_load.group(1), (
        f"gate checks ${m_gate.group(1)} but the rollout loads ${m_load.group(1)}"
    )


# ---------------------------------------------------------------------------
# 3. The rest of the launch line
# ---------------------------------------------------------------------------


def test_runner_requests_sync_inference():
    """RTC was the Sep 4 plan and was REVERSED the same day, on the bench.

    Under --inference.type=rtc the three SmolVLA trials (SV-A-01, SV-B-01,
    SV-A-02) all commanded a fully-extended, gripper-open pose from tick 0 and
    hovered there for 44 s. Root cause, reproduced offline with the real
    ActionQueue: RTC's _replace_actions_queue discards the first `real_delay`
    actions of EVERY chunk, including the first one, when nothing was executed
    during inference. On MPS the first inference took 15 ticks (log:
    "indexes_diff=0, real_delay=15"), so the arm's first command was chunk[15]
    -- a pose planned 0.75 s into the reach, executed from rest. The sync
    engine has no queue discard and serves chunk[0]; ACT and pi0.5 already run
    on it. See test_rtc_action_queue_discards_leading_actions_on_first_chunk.
    """
    code = _code()
    assert "--inference.type=sync" in code, "SmolVLA must run on the sync engine (see docstring)"
    assert "--inference.type=rtc" not in code, "RTC re-enabled: chunk[real_delay] would be served first"


def test_rtc_action_queue_discards_leading_actions_on_first_chunk():
    """Characterises the upstream behaviour that forced sync (lerobot 0.6.1).

    A fresh RTC ActionQueue, merged with a 50-step chunk at real_delay=15 and
    action_index_before_inference=0, serves chunk[15] first, not chunk[0].
    If this test ever FAILS, upstream fixed the first-chunk discard and RTC
    may be worth re-evaluating on MPS -- declare that first, then change the
    runner.
    """
    import torch
    from lerobot.policies.rtc import ActionQueue
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    chunk = torch.arange(50, dtype=torch.float32).unsqueeze(1).repeat(1, 6)  # row k == k
    q = ActionQueue(RTCConfig())
    q.merge(chunk.clone(), chunk.clone(), 15, q.get_action_index())
    first = q.get()
    assert first is not None and float(first[0]) == 15.0, (
        f"expected chunk[15] served first (upstream discard), got {None if first is None else float(first[0])}"
    )
    q0 = ActionQueue(RTCConfig())
    q0.merge(chunk.clone(), chunk.clone(), 0, q0.get_action_index())
    assert float(q0.get()[0]) == 0.0, "real_delay=0 must serve chunk[0]"


def test_runner_passes_a_rename_map():
    """`RolloutConfig.rename_map` defaults to {} (rollout/configs.py:259) and
    context.py:537-546 uses it to OVERRIDE the map saved with the checkpoint.
    Omitting the flag therefore replaces a correct map with an empty one.

    Read off the argv the rollout is actually launched with: `--rename_map` in a
    comment, or on a continuation line that lost its trailing backslash, would
    both satisfy a substring scan of the file."""
    launch = _stub_run().only_call_to("rollout_30hz_stale_ok.py")["argv"]
    assert any(a.startswith("--rename_map=") for a in launch), (
        f"no --rename_map on the launch line {launch}: SmolVLA would mask the "
        f"absent towers instead of raising, and run blind at full speed"
    )


def test_the_launch_line_wires_each_camera_to_the_device_it_was_recorded_on():
    """The map above pins the RENAMING. Nothing pinned which device feeds front.

    `tools/smolvla_dryrun.py:88` says it outright — "swapping front and wrist
    runs fine and is silently wrong" — and the project notes records that macOS camera
    indices shift after a reboot or a replug, which is exactly the moment
    somebody edits these two digits. Transposed, the wrist stream arrives as
    camera1: the rollout runs at full speed, writes a plausible video and scores
    MISS, and the A/B is decided on a wiring fault. No test in tests/ mentioned
    index_or_path at all. So pin the geometry against the recording's own
    feature shapes, and the device indices against run_scored_trial.sh — the
    certified ACT rig config, which this file already refuses to let drift. If
    the indices genuinely moved, tools/probe_cameras.py says so and every runner
    changes together.
    """
    launch = _stub_run().only_call_to("rollout_30hz_stale_ok.py")["argv"]
    spec = next((a for a in launch if a.startswith("--robot.cameras=")), None)
    assert spec, f"the rollout is launched with no cameras at all: {launch}"

    cams = {}
    for name in sd.CAMERA_SHAPES:
        m = re.search(name + r":\s*\{([^}]*)\}", spec)
        assert m, f"no `{name}` camera on the launch line: {spec}"
        cams[name] = {
            k.strip(): v.strip() for k, v in (p.split(":", 1) for p in m.group(1).split(","))
        }

    for name, (height, width, _) in sd.CAMERA_SHAPES.items():
        opened = (int(cams[name]["height"]), int(cams[name]["width"]))
        assert opened == (height, width), (
            f"{name} is opened at {opened[1]}x{opened[0]} but the v3 recording "
            f"stores observation.images.{name} as {width}x{height}"
        )

    act = (TOOLS / "run_scored_trial.sh").read_text()
    for name in sd.CAMERA_SHAPES:
        index = cams[name]["index_or_path"]
        assert f"{name}: {{type: opencv, index_or_path: {index}," in act, (
            f"the SmolVLA runner opens `{name}` on OpenCV index {index}, which is "
            f"not what run_scored_trial.sh — the certified ACT rig config — uses. "
            f"Transposed indices run fine and are silently wrong "
            f"(tools/smolvla_dryrun.py:88): the trial scores MISS and SmolVLA "
            f"takes the blame. If the indices really moved, re-run "
            f"tools/probe_cameras.py and change every runner together"
        )


def test_rename_map_matches_the_checkpoints_own_training_map():
    code = _code()
    m = re.search(r"rename_map=(['\"])(.+?)\1", code, re.S)
    assert m, "could not parse the --rename_map argument"
    pairs = {}
    for part in m.group(2).strip().strip("{}").split(","):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        pairs[k.strip()] = v.strip()
    trained = _train_config("A")["rename_map"]
    assert pairs == trained, f"runner maps {pairs}, training used {trained}"
    assert pairs == _train_config("B")["rename_map"], "the two twins disagree on the map"


def test_the_control_loop_is_paced_at_the_dataset_fps():
    """`--fps` had no assertion anywhere; only `--inference.type=sync` was pinned.

    The v3 recording is 20 fps and both twins trained on it. `--fps=30` executes
    every action chunk at 1.5x the trained rate while `--dataset.fps` still
    stamps the saved video 20, so even the recording misreports its own timing:
    the arm overshoots the seat, the trial scores MISS, and SmolVLA takes the
    blame for the wrapper's name. 30 is a plausible thing to type here — the
    launcher is called rollout_30hz_stale_ok.py. Read the rate off the dataset,
    never off a constant in this file.
    """
    from conftest import V3_DATASET_ROOT

    info = V3_DATASET_ROOT / "meta" / "info.json"
    if not info.exists():
        pytest.skip(f"v3 recording not in the local cache ({V3_DATASET_ROOT})")
    trained_fps = json.loads(info.read_text())["fps"]

    launch = _stub_run().only_call_to("rollout_30hz_stale_ok.py")["argv"]
    for flag in ("--fps", "--dataset.fps"):
        got = [a for a in launch if a.startswith(f"{flag}=")]
        assert got == [f"{flag}={trained_fps}"], (
            f"{flag} is {got or 'unset'}; the v3 recording both twins trained on is "
            f"{trained_fps} fps. Pacing anything else runs each action chunk off the "
            f"trained rate — the arm overshoots the seat and the trial scores MISS "
            f"for a reason that has nothing to do with the policy"
        )


def test_runner_reuses_the_same_gates_as_the_other_runners():
    """Three names, in the order the runner actually executes them.

    Was: three name-presence scans with no ordering constraint — in a file that
    elsewhere proves it knows how to assert order. Swapping the two lines
    reverts the Aug 30 lesson written into the comment directly above them: the
    scene is photographed before the arm is homed, an arm parked over the tube
    occludes the cap, and scene_check reports an occlusion failure as a
    PLACEMENT failure. `*placement*` is a human-only gate with no retry, so the
    block aborts and Faith re-seats a tube that was already correct — bench
    minutes at the start of a gated block, spent on the wrong thing.
    """
    run = _stub_run()
    order = run.tools_in_order()
    for gate in ("preflight.py", "home_arm.py", "scene_check.py"):
        assert any(t.endswith(gate) for t in order), (
            f"SmolVLA runner skips {gate} — it ran {order}"
        )
    first = lambda gate: next(i for i, t in enumerate(order) if t.endswith(gate))  # noqa: E731
    assert first("home_arm.py") < first("scene_check.py"), (
        f"scene_check.py runs BEFORE home_arm.py: the scene is photographed with "
        f"the arm still parked over the tube, which occludes the cap and reports "
        f"an occlusion failure as a placement failure (observed Aug 30). "
        f"Order was {order}"
    )
    assert first("scene_check.py") < first("preflight.py"), (
        f"preflight.py reads the scene capture before scene_check.py writes it. "
        f"Order was {order}"
    )


def test_runner_gates_on_the_side_tables_generation_not_v2():
    """The support window and the home pose follow the SIDE TABLE'S generation (Sep 6: v4 by
    default, `CAPSTONE_POLICY_GEN=v3` for the Sep 3-4 policies), never a constant in this file
    and never home_gate.DEFAULT_DATASET ("v2"). Under the stub harness `--print gen` answers
    "v4", so that is what must reach preflight's 4th argument and home_arm's --pose.

    History, kept because it is the bug: both v3 twins trained on v3; v2's frame-0 bounds
    reject valid v3 starts.

    Was: `DATASET_TAG=v3` is assigned somewhere in the file. It is — and the tag
    still never reached the gate. `tools/preflight.py:117` reads the support
    window from `sys.argv[4]` and falls back to `home_gate.DEFAULT_DATASET`,
    which is `"v2"`, so dropping the fourth argument gates every SmolVLA trial
    on v2's frame-0 bounds while the arm is homed to v3 and a v3-trained policy
    launches — and prints a reassuring `PREFLIGHT PASS (v2)` while doing it.
    v3's elbow frame-0 range is [78.11, 90.42] against v2's [75.47, 84.44] and
    its gripper [0.79, 1.57] against [1.14, 1.64] (conftest), so it both rejects
    in-distribution starts and admits out-of-distribution ones. Assert on the
    argv preflight is handed, and on the pose the arm is homed to.
    """
    import home_gate

    run = _stub_run()
    args = run.only_call_to("preflight.py")["argv"][1:]
    assert len(args) >= 4, (
        f"preflight.py is called with {len(args)} arguments {args}; the 4th is the "
        f"training-support window, and without it preflight.py:117 silently falls "
        f"back to home_gate.DEFAULT_DATASET={home_gate.DEFAULT_DATASET!r}"
    )
    assert args[3] == "v4", (
        f"preflight gates on the {args[3]!r} support window; the runner must pass the side "
        f"table's generation (the stub answers 'v4' for --print gen), not a constant or v2"
    )
    homed = run.calls_to("home_arm.py")
    assert homed, f"the runner never homes the arm — it ran {run.tools_in_order()}"
    for call in homed:
        assert args[3] in call["argv"], (
            f"home_arm.py is posed with {call['argv']} but preflight gates on "
            f"{args[3]!r}: the arm would be homed to one dataset's pose and judged "
            f"against another's support window"
        )


def test_runner_does_not_enable_the_grasp_lock():
    """CAPSTONE_GRASP_LOCK is an ACT-specific heuristic on the sync action
    queue. SmolVLA now runs on that same queue (sync, since Sep 4), but the
    lock was never pre-declared for SmolVLA and stays OFF, matching the pi0.5
    runner.

    Was: `"unset CAPSTONE_GRASP_LOCK" in code`. That is a PREFIX — a rename or a
    typo to `unset CAPSTONE_GRASP_LOCK_V2` unsets nothing that exists and still
    satisfies it, which removes the only barrier between an exported lock left
    over from an ACT session in the same shell and a SmolVLA rollout. So export
    one, and check what the rollout is actually handed.
    """
    code = _code()
    assert "CAPSTONE_GRASP_LOCK=1" not in code
    leaked = _stub_run(env=(("CAPSTONE_GRASP_LOCK", "1"),))
    launched = leaked.only_call_to("rollout_30hz_stale_ok.py")
    assert launched["env"]["GRASP"] == "__UNSET__", (
        f"CAPSTONE_GRASP_LOCK={launched['env']['GRASP']!r} reached the rollout. An "
        f"ACT session in the same shell would apply the grasp-phase chunk lock to a "
        f"policy it was never pre-declared for, on the same sync action queue, and "
        f"the trial would be uninterpretable rather than obviously broken"
    )


def test_runner_refuses_a_side_that_is_not_A_or_B():
    """Executed, not parsed: a bad side must abort before any gate runs, so this
    can be run safely while the arm is busy in the other window."""
    out = subprocess.run(
        ["/bin/zsh", str(RUNNER), "Q", "BADSIDE-TEST", "50"],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "CAPSTONE_ALLOW_WORKTREE": "1"},
    )
    assert out.returncode != 0, "an unknown side did not abort"
    assert "side" in (out.stdout + out.stderr).lower()
    assert "home_arm" not in out.stdout, "a bad side reached the hardware gates"


def test_runner_refuses_a_non_integer_horizon():
    out = subprocess.run(
        ["/bin/zsh", str(RUNNER), "A", "BADN-TEST", "fifty"],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "CAPSTONE_ALLOW_WORKTREE": "1"},
    )
    assert out.returncode != 0, "a non-integer n_action_steps did not abort"


def test_the_horizon_argument_reaches_the_policy():
    """$3 is validated — then it has to actually be USED.

    Nothing in this file tied `$NSTEPS` to the launch line, so pinning
    `--policy.n_action_steps=50` passed every test while the banner, the
    `.verify` receipt and column 4 of trials_smolvla.csv all went on reporting
    the horizon that was ASKED for. Every receipt agrees and every one of them
    is wrong: a 25-vs-50 row pair in the ledger would be a comparison that never
    happened, and the per-policy config would be chosen off it. n_action_steps
    is live under sync — it is the number of steps of each chunk executed
    open-loop — so this is the one knob the n-grid turns.
    """
    run = _stub_run(nsteps="25")
    launch = run.only_call_to("rollout_30hz_stale_ok.py")["argv"]
    pinned = [a for a in launch if a.startswith("--policy.n_action_steps=")]
    assert pinned == ["--policy.n_action_steps=25"], (
        f"asked for n_action_steps=25; the rollout was launched with "
        f"{pinned or 'no horizon argument at all'}. The ledger and the .verify "
        f"receipt would both record 25 for a rollout that ran at something else"
    )
    assert "n_action_steps=25" in run.out, (
        f"the horizon handed to the rollout is not the one the console and the "
        f"receipt report; the runner said:\n{run.out[:600]}"
    )


# ---------------------------------------------------------------------------
# 4. Ownership boundaries -- Window A is on the bench tonight
# ---------------------------------------------------------------------------


def test_act_runner_left_untouched():
    """run_scored_trial.sh runs Block 2B at 10:00 and must not change tonight."""
    act = (TOOLS / "run_scored_trial.sh").read_text()
    for foreign in ("smolvla", "rename_map", "camera1", "inference.type"):
        assert foreign not in act, (
            f"the ACT runner grew a SmolVLA concern ({foreign}); it runs Block 2B "
            f"tomorrow morning and must stay stable"
        )


def test_pi05_runner_left_untouched():
    p = (TOOLS / "run_pi05_trial.sh").read_text()
    assert "smolvla" not in p.lower(), "the pi0.5 runner grew a SmolVLA concern"


def test_smolvla_rows_do_not_land_in_the_act_trials_csv():
    """`tools/figures_data.py:83` classifies any non-`act-tube-B` policy as
    ACT-A, so SmolVLA rows appended to trials.csv would publish as ACT-A points
    in fig4. Separate ledger until that classifier is fixed."""
    code = _code()
    assert "scored_logs/trials.csv" not in code, (
        "SmolVLA rows must not be appended to the ACT ledger while "
        "figures_data.py:83 misclassifies non-ACT policies"
    )
    assert "trials_smolvla.csv" in code, "no SmolVLA ledger is written"


def test_usage_validation_precedes_every_line_that_could_touch_the_arm():
    """Structural, so it fails BEFORE the suite drives hardware.

    `test_runner_refuses_a_side_that_is_not_A_or_B` executes the runner, which
    is only safe while the argument checks are the first thing that happens.
    If a later edit moved them below the gates, that test would home the arm
    during a routine `pytest tests/` — possibly mid-session in another window.
    This one fails on the ordering itself.
    """
    lines = _code().splitlines()
    first_validation = next((i for i, ln in enumerate(lines) if 'case "$SIDE"' in ln), None)
    assert first_validation is not None, (
        "no `case \"$SIDE\"` validation found: the runner accepts any side, and "
        "test_runner_refuses_a_side_that_is_not_A_or_B would drive the gates"
    )
    risky = [
        i
        for i, ln in enumerate(lines)
        if any(t in ln for t in ("home_arm.py", "ping_motors.py", "scene_check.py",
                                 "rollout_30hz_stale_ok.py", "reseat_gripper.py", "$B/python"))
    ]
    assert risky, "no launch lines found; the parser is out of date"
    assert first_validation < min(risky), (
        f"argument validation is at line {first_validation + 1} but the first "
        f"command that can reach hardware is at line {min(risky) + 1}"
    )


def test_usage_aborts_are_silent():
    """No `say` before the gates: the suite exercises the abort path, and a
    spoken 'abort' from a pytest run would land in the middle of a bench
    session in the other window."""
    lines = _code().splitlines()
    first_gate = next((i for i, ln in enumerate(lines) if "home_arm.py" in ln), None)
    cut = next((i for i, ln in enumerate(lines) if "CAPSTONE_ALLOW_WORKTREE" in ln), None)
    assert first_gate is not None and cut is not None, "the runner's shape changed"
    header = lines[:cut]
    assert not any(re.search(r"\bsay\b", ln) for ln in header), (
        "an argument-validation abort speaks; pytest would talk over the bench"
    )
    assert first_gate > 0


# ---------------------------------------------------------------------------
# 5. The resolution block — the only stretch that hardware gating hid
# ---------------------------------------------------------------------------
#
# Between the usage gates and `home_arm.py` sit the four `--print` captures, the
# `${RAW_SIDE:u}` normalisation, the empty-TASK guard and the state-width gate.
# None of it could be executed in a test, because reaching it and continuing
# means homing the arm. CAPSTONE_RESOLVE_ONLY stops immediately after the
# state-width gate, so that stretch runs for real, on both sides, in the suite —
# and Faith gets a 10-second pre-flight that consumes no trial.


def test_resolve_only_stops_before_the_first_line_that_touches_hardware():
    """Structural AND behavioural. If this block moved below the gates — or
    stopped selecting on the marker — the executing tests underneath would home
    the arm from a routine `pytest tests/`.

    The structural half used to be all there was: the marker appears somewhere,
    and the word `exit` appears within five lines of it. Both survive inverting
    the condition to `[ -z "${CAPSTONE_RESOLVE_ONLY:-}" ]`, which turns the
    escape hatch inside out — the 10-second pre-flight falls straight through to
    home_arm.py, scene_check.py and a full 45 s rollout, consuming a gated trial
    and writing a ledger row, while a REAL trial exits 0 at the state-width gate
    announcing that "NO trial was executed". The suite stayed green through it,
    and `_require_safe_to_execute_the_runner` below decided from these same two
    checks that it was safe to execute the runner from pytest. So run the hatch,
    both ways.
    """
    lines = _code().splitlines()
    stop = next((i for i, ln in enumerate(lines) if "CAPSTONE_RESOLVE_ONLY" in ln), None)
    assert stop is not None, "no CAPSTONE_RESOLVE_ONLY escape hatch"
    first_hw = next((i for i, ln in enumerate(lines) if "home_arm.py" in ln), None)
    assert first_hw is not None and stop < first_hw, (
        f"CAPSTONE_RESOLVE_ONLY is at line {stop + 1}, first hardware line at "
        f"{first_hw + 1 if first_hw else '?'} — the escape hatch is too late"
    )
    # It must actually stop the runner, and ONLY when the marker is set.
    problem = _hatch_misbehaviour()
    assert problem is None, problem


EXEC_ENV = "CAPSTONE_EXEC_RUNNER_TESTS"


def _require_safe_to_execute_the_runner():
    """Three conditions, all required, before this suite will RUN the runner.

    Recorded incident, Sep 3 2026 — twice in one hour:

    1. The executing tests below were written before the CAPSTONE_RESOLVE_ONLY
       hatch existed (the TDD red phase). `pytest` ran the runner straight into
       `home_arm.py` and homed the arm three times.
    2. The fix — "skip unless the marker is present" — was then defeated by a
       mutation that kept the marker but replaced `exit 0` with `:`. The guard
       looked present, so the tests ran, and the runner fell through again.

    Nothing was damaged either time (no trial consumed, no ledger row, no
    dataset; the scene gate aborted both runs) but the boundary was broken by a
    test, which is the worst way to break it.

    So the default is now OFF. `pytest tests/` — what Faith runs before every
    session, sometimes with the arm powered and the other window on the bench —
    never executes this script. Turn it on deliberately, when the bench is known
    idle:

        CAPSTONE_EXEC_RUNNER_TESTS=1 pytest tests/test_smolvla_trial_runner.py

    The static tests still cover this file on every run; only the three
    subprocess tests are gated. The operational equivalent is the pre-flight
    Faith already runs: `CAPSTONE_RESOLVE_ONLY=1 tools/run_smolvla_trial.sh ...`
    """
    if not os.environ.get(EXEC_ENV):
        pytest.skip(
            f"{EXEC_ENV} not set — refusing to execute the trial runner. "
            f"It reaches hardware if its CAPSTONE_RESOLVE_ONLY hatch is ever "
            f"broken, and this suite runs while the arm is powered."
        )
    code = _code()
    lines = code.splitlines()
    stop = next((i for i, ln in enumerate(lines) if "CAPSTONE_RESOLVE_ONLY" in ln), None)
    if stop is None:
        pytest.skip("runner has no CAPSTONE_RESOLVE_ONLY hatch — would drive the arm")
    problem = _hatch_misbehaviour()
    if problem is not None:
        pytest.skip(f"refusing to execute the runner — {problem}")
    first_hw = next((i for i, ln in enumerate(lines) if "home_arm.py" in ln), None)
    if first_hw is None or stop >= first_hw:
        pytest.skip("CAPSTONE_RESOLVE_ONLY hatch sits at or below the hardware gates")


# 25 s, not 180. Everything this resolves is cached and offline
# (HF_HUB_OFFLINE=1): it takes ~0.9 s when it works. A 180 s budget only bought
# a 3-minute stall before failing -- and it did stall once, during the Sep 4
# session, then passed alone in 0.87 s. This is the PRE-PUSH gate; it must not
# be the slow step, and an environment hiccup must not read as a broken runner.
_RESOLVE_TIMEOUT_S = 25


def _resolve_only(side_arg: str, label: str = "VERIFY-TEST"):
    _require_safe_to_execute_the_runner()
    try:
        return subprocess.run(
            ["/bin/zsh", str(RUNNER), side_arg, label, "50"],
            capture_output=True, text=True, timeout=_RESOLVE_TIMEOUT_S,
            env={**os.environ, "CAPSTONE_RESOLVE_ONLY": "1", "HF_HUB_OFFLINE": "1"},
        )
    except subprocess.TimeoutExpired:
        pytest.skip(
            f"runner resolution exceeded {_RESOLVE_TIMEOUT_S}s -- everything it needs is "
            f"cached and offline, so this is an environment stall, not a runner defect. "
            f"Re-run this file alone to confirm."
        )


@pytest.mark.parametrize("side_arg,side,pin", [("A", "A", "0"), ("B", "B", "1")])
def test_resolve_only_runs_the_real_resolution_for_both_sides(side_arg, side, pin):
    """Executes the actual capture block: repo, snapshot, pin, prompt, gate.

    Offline on purpose — it doubles as the proof that a dropped connection at
    10:00 cannot stop a launch, since every value resolves from cache.
    """
    out = _resolve_only(side_arg)
    combined = out.stdout + out.stderr
    assert out.returncode == 0, f"resolve-only failed for side {side}:\n{combined[-800:]}"
    assert f"side={side}" in combined
    assert f"faithqin/smolvla-tube-{side}-v3" in combined
    assert f"CAPSTONE_FORCE_WIDE_STATE={pin}" in combined
    assert "Pick up the test tube and insert it into the rack" in combined
    # the state-width gate really ran, and really passed
    assert "declared=" in combined and "stats=" in combined
    # and nothing downstream of it did
    assert "home_arm" not in combined, "resolve-only reached the hardware gates"
    assert "TRIAL" not in out.stdout.upper().replace("NO TRIAL", "")


def test_resolve_only_accepts_a_lowercase_side():
    """`${RAW_SIDE:u}` — typing `b` at 10:00 must not be a silent A-side run."""
    out = _resolve_only("b")
    combined = out.stdout + out.stderr
    assert out.returncode == 0, combined[-500:]
    assert "side=B" in combined and "smolvla-tube-B-v3" in combined
    assert "CAPSTONE_FORCE_WIDE_STATE=1" in combined


def test_every_resolver_call_is_asked_about_the_side_being_run():
    """The four `--print` captures are column-aligned and copy-pasted.

    Leaving one un-parameterised — `--side A` where `--side "$SIDE"` belongs —
    is an ordinary edit, and nothing fails: SNAP is resolved separately and
    stays per-side, so the correct B checkpoint still runs while REPO_ID says A.
    REPO_ID is the console banner, `policy=` in the .verify receipt and column 3
    of trials_smolvla.csv — the column any later analysis splits the
    load-sensing A/B on. The log would be self-consistent and wrong.

    The one assertion tying REPO_ID to $SIDE lives in
    test_resolve_only_runs_the_real_resolution_for_both_sides, which is SKIPPED
    unless CAPSTONE_EXEC_RUNNER_TESTS is set — so on `pytest tests/` nothing
    covered it. This runs under the stub harness, so it runs every time, and it
    covers `${RAW_SIDE:u}` without the Hub cache.
    """
    for given, expected in (("A", "A"), ("B", "B"), ("b", "B")):
        run = _stub_run(side=given)
        asked = run.calls_to("smolvla_dryrun.py")
        assert len(asked) == 5, (
            f"expected five --print resolutions (gen, repo, snapshot, force, task), saw "
            f"{[c['argv'] for c in asked]}"
        )
        for call in asked:
            argv = call["argv"]
            assert "--side" in argv, f"{argv} resolves without a side at all"
            what = argv[argv.index("--print") + 1]
            side = argv[argv.index("--side") + 1]
            assert side == expected, (
                f"run as side {given!r} (normalised to {expected}), but --print "
                f"{what} was resolved for side {side!r}. That value reaches the "
                f"banner, the .verify receipt and the ledger's policy column while "
                f"the other side's trial is the one that actually runs"
            )


def _policy_method_calls(path) -> set:
    """Method names invoked on the local name `policy`, via ast.

    Comments and docstrings are STRING/COMMENT tokens and never become Call
    nodes, so this cannot be satisfied by prose -- which is the whole point
    (three tests in this repo have now been found green on a comment).
    """
    import ast

    return {
        node.func.attr
        for node in ast.walk(ast.parse(Path(path).read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "policy"
    }


def test_the_rehearsal_exercises_the_SAME_engine_the_runner_launches():
    """A dry-run that rehearses a different engine certifies nothing.

    Sep 4: the runner was switched from RTC to sync after RTC's ActionQueue was
    found to discard the first chunk's leading actions -- but smolvla_dryrun.py
    still called `predict_action_chunk(batch, inference_delay=..., 
    prev_chunk_left_over=...)`, the RTC entry point. The sync engine calls
    `policy.select_action(batch)` (rollout/inference/sync.py:130). So the
    "PLUMBING OK" that cleared SV-A-01r was measured on a path the bench no
    longer takes.

    This is the failure that produced the pi0.5 empty-prompt bug: a rehearsal
    that builds its own pipeline instead of mirroring the real call site.
    """
    runner = _code()
    uses_sync = "--inference.type=sync" in runner
    assert uses_sync, "runner engine changed -- update this test and the rehearsal together"

    # ast, not a substring scan. `select_action` appears in smolvla_dryrun.py's
    # module docstring and in three inline comments; the old
    # `assert "select_action" in dry` was satisfied by any one of those four
    # prose mentions alone, on a rehearsal that never called it (Sep 5 review).
    called = _policy_method_calls(TOOLS / "smolvla_dryrun.py")
    assert "select_action" in called, (
        "runner launches the SYNC engine but smolvla_dryrun.py never CALLS "
        f"policy.select_action (calls found: {sorted(called) or 'none'}) -- "
        "the rehearsal is exercising a different engine and its verdict does not transfer"
    )
    assert "predict_action_chunk" not in called, (
        "smolvla_dryrun.py calls policy.predict_action_chunk -- that is the RTC entry "
        "point, which discards the first chunk's leading actions; the bench runs sync"
    )
    assert "init_rtc_processor" not in called, (
        "the rehearsal enables RTC on the policy; SmolVLAPolicy.select_action asserts "
        "when RTC is on (modeling_smolvla.py:254), so this rehearses a policy the bench "
        "never builds"
    )
