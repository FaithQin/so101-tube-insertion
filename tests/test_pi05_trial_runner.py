"""Regression: the pi0.5 trial runner must carry the two things ACT does not need.

Written Aug 31 2026, before the first pi0.5 bench session.

pi0.5 cannot reuse `run_scored_trial.sh` unchanged, for two reasons that are
both silent failures rather than crashes:

1. **rename_map.** The rig publishes `observation.images.front` and
   `...wrist`; pi0.5 declares `base_0_rgb`, `left_wrist_0_rgb`,
   `right_wrist_0_rgb`, `empty_camera_0`. Training used an explicit
   rename_map. Without it at rollout, NONE of the policy's declared image keys
   are present -- and `_preprocess_images` (modeling_pi05.py:1009) pads every
   missing image with -1 and masks it to zero rather than raising. The policy
   would run blind, at full speed, producing actions from four blank towers.

2. **Sanitized checkpoint path.** The Hub config carries six lerobot fields
   0.6.1 cannot parse, so `--policy.path=<repo_id>` dies with a draccus
   DecodingError. The runner must point at the locally sanitized copy.

`run_scored_trial.sh` is deliberately NOT modified -- it runs the ACT blocks and
must not change the night before a session.
"""

import re

import pytest

from conftest import TOOLS

RUNNER = TOOLS / "run_pi05_trial.sh"


def _src():
    assert RUNNER.exists(), f"{RUNNER} missing"
    return RUNNER.read_text()


def _code():
    """The runner with shell comments stripped.

    Added Sep 3 2026 after the night audit found that
    test_runner_forces_narrow_state_routing_explicitly was satisfied by a
    COMMENT (run_pi05_trial.sh:22 contains the literal
    "CAPSTONE_FORCE_WIDE_STATE=0"), so deleting the real export left the test
    green. Substring-scanning source text is the bug class; strip the prose
    first. A `#` opens a comment only when unquoted and at line start or after
    whitespace -- so `${REPO##*/}` survives.
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


def test_runner_exists_and_is_executable():
    import os
    assert os.access(RUNNER, os.X_OK), "run_pi05_trial.sh is not executable"


def test_runner_passes_rename_map():
    src = _src()
    assert "rename_map" in src, (
        "no --rename_map: pi0.5 would receive four blank padded image towers "
        "and run blind without raising"
    )


def test_rename_map_maps_front_to_base_and_wrist_to_left_wrist():
    src = _src()
    m = re.search(r"rename_map=(['\"])(.+?)\1", src, re.S)
    assert m, "could not parse the --rename_map argument"
    body = m.group(2)
    assert "observation.images.front" in body and "base_0_rgb" in body
    assert "observation.images.wrist" in body and "left_wrist_0_rgb" in body
    # Parse into pairs and assert on the MAPPING, not on substring offsets.
    # The previous check was `front < base < lw or front < base`, which reduces
    # to `front < base` -- a map that swapped front and wrist passed it
    # (night audit, Sep 3 2026).
    pairs = {}
    for part in body.strip().strip("{}").split(","):
        if ":" not in part:
            continue
        src_key, dst_key = part.split(":", 1)
        pairs[src_key.strip()] = dst_key.strip()
    assert pairs == {
        "observation.images.front": "observation.images.base_0_rgb",
        "observation.images.wrist": "observation.images.left_wrist_0_rgb",
    }, f"rename_map does not map front->base and wrist->left_wrist: {pairs}"


_GATE_RUNS: dict = {}


def _run_gates(preflight_rc=0, preflight_msg="OK", inherit=None, side="A"):
    """RUN the runner against stub binaries. Returns (rc, stdout, calls, policy_dir).

    Added Sep 5 2026 after mutation testing walked through every text scan in
    this file at once: the suite stayed green with DATASET_TAG flipped to v2,
    with preflight's exit status swallowed by a `| tee` pipeline, with `exit 3`
    replaced by an echo, with --policy.path reverted to the un-parsable Hub id,
    with `export` dropped from the FORCE_WIDE_STATE pin, and with `unset
    CAPSTONE_GRASP_LOCK` commented out. Naming a gate script in the source
    proves neither which pose it is handed nor that anyone reads its answer.

    The script under test is byte-identical except for its `B=<bin>` line, which
    is repointed at a stub `python` that records every argv, returns the
    preflight status we ask for, and -- on the rollout call -- dumps the
    CAPSTONE environment it was ACTUALLY handed. `say` is stubbed too, and
    HERE/PROJ resolve inside a temp dir, so this stays silent, hardware-free,
    and writes nothing into the checkout.
    """
    key = (preflight_rc, preflight_msg, tuple(sorted((inherit or {}).items())), side)
    if key in _GATE_RUNS:
        return _GATE_RUNS[key]

    import os
    import shutil
    import subprocess
    import tempfile

    stub_py = """#!/bin/zsh
print -rn -- "PY $*" | tr '\\n' ' ' >> "$STUB_CALLS"
print -r -- "" >> "$STUB_CALLS"
case "$*" in
  *preflight.py*) print -r -- "$STUB_PREFLIGHT_MSG"; exit $STUB_PREFLIGHT_RC ;;
  *materialize_local_checkpoint*) print -r -- "$STUB_POLICY_DIR"; exit 0 ;;
  *"pi05_sides.py --print snapshot"*) print -r -- "$STUB_POLICY_DIR"; exit 0 ;;
  *"pi05_sides.py --print gen"*) print -r -- "v4"; exit 0 ;;
  *"pi05_sides.py --print force --side B"*) print -r -- "1"; exit 0 ;;
  *"pi05_sides.py --print force"*) print -r -- "0"; exit 0 ;;
  *"pi05_sides.py --print repo --side B"*) print -r -- "faithqin/pi05-tube-B-v4"; exit 0 ;;
  *"pi05_sides.py --print repo"*) print -r -- "faithqin/pi05-tube-A-v4"; exit 0 ;;
  *"pi05_sides.py --print task"*) print -r -- "Pick up the test tube and insert it into the rack"; exit 0 ;;
  *rollout_30hz_stale_ok.py*)
      print -r -- LAUNCHED >> "$STUB_CALLS"
      env | grep '^CAPSTONE' | sed 's/^/LAUNCH_ENV /' >> "$STUB_CALLS"
      exit 0 ;;
esac
exit 0
"""
    stub_say = '#!/bin/zsh\nprint -r -- "SAY $*" >> "$STUB_CALLS"\nexit 0\n'

    work = tempfile.mkdtemp(prefix="pi05_gate_")
    try:
        bin_dir = os.path.join(work, "bin")
        os.makedirs(bin_dir)
        os.makedirs(os.path.join(work, "tools"))
        calls = os.path.join(work, "calls.log")
        open(calls, "w").close()
        for name, body in (("python", stub_py), ("say", stub_say)):
            with open(os.path.join(bin_dir, name), "w") as f:
                f.write(body)
            os.chmod(os.path.join(bin_dir, name), 0o755)

        patched, n = re.subn(r"(?m)^B=\S+$", lambda m: "B=" + bin_dir, _src())
        assert n == 1, "run_pi05_trial.sh no longer has a single `B=<bindir>` line"
        script = os.path.join(work, "tools", "run_pi05_trial.sh")
        with open(script, "w") as f:
            f.write(patched)
        os.chmod(script, 0o755)

        policy_dir = os.path.join(work, "sanitized_ckpt")
        # Strip the operator's own CAPSTONE_* so the run is deterministic; the
        # only ones the child may see are the ones the runner itself sets.
        env = {k: v for k, v in os.environ.items() if not k.startswith("CAPSTONE")}
        env.update(inherit or {})
        env.update({
            "PATH": bin_dir + os.pathsep + os.environ["PATH"],
            "STUB_CALLS": calls,
            "STUB_PREFLIGHT_RC": str(preflight_rc),
            "STUB_PREFLIGHT_MSG": preflight_msg,
            "STUB_POLICY_DIR": policy_dir,
            "CAPSTONE_ALLOW_WORKTREE": "1",
        })
        r = subprocess.run(["/bin/zsh", script, side, "PI05-GATETEST", "50"],
                           capture_output=True, text=True, timeout=180,
                           env=env, cwd=work)
        with open(calls) as f:
            result = (r.returncode, r.stdout, f.read(), policy_dir)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    _GATE_RUNS[key] = result
    return result


def test_runner_uses_a_sanitized_local_checkpoint_not_the_hub_id():
    """EXECUTED, not scanned: the rollout must load the directory the sanitizer
    printed, never the Hub repo id.

    The old assertion was `"materialize_local_checkpoint" in src or "sanitiz"
    in src.lower()` over RAW source, and "sanitized" appears in the header
    comment and in the section banner -- so reverting the launch line to
    --policy.path="$REPO_ID" (the exact pre-fix form the test was written to
    forbid) left it green, sanitizing block and all. That revert does not fail
    honestly: draccus dies on the six keys 0.6.1 cannot parse, stderr lands in
    ${RUN}.log, the scorer then finds no dataset and no video, and the last
    line still says "miss" -- an infrastructure crash spoken as a policy
    verdict, after the gates and the homing are already spent.
    """
    rc, stdout, calls, policy_dir = _run_gates()
    launch = [ln for ln in calls.splitlines() if "rollout_30hz_stale_ok.py" in ln]
    assert launch, f"the rollout never launched on a passing gate (rc={rc})"
    m = re.search(r"--policy\.path=(\S+)", launch[0])
    assert m, f"no --policy.path on the launch line: {launch[0]}"
    assert m.group(1) == policy_dir, (
        f"rollout loads --policy.path={m.group(1)!r}; it must load the sanitized "
        f"local checkpoint the runner just materialized ({policy_dir!r}). A Hub "
        f"repo id here dies in draccus and is scored as a MISS."
    )


def test_runner_reuses_the_same_gates_as_the_act_runner():
    """Same preflight, same abort-not-warn discipline -- asserted by RUNNING it.

    A pi0.5 trial launched from an out-of-support home pose is as worthless as
    an ACT one, and three separate mutations survived the old
    `for gate in (...): assert gate in src` loop (Sep 5 audit):

      * DATASET_TAG=v3 -> v2. That one value drives both `home_arm.py --pose`
        and preflight's support window, so the pose and the window are wrong
        TOGETHER and preflight passes green -- the Sep 3 production bug
        ("every v3 rollout re-homed to the v2 pose") reintroduced, reading on
        the bench as "pi0.5 cannot do the task".
      * `preflight.py ...); RC=$?` -> `... | tee -a "${RUN}.pre"); RC=$?`, the
        natural edit for keeping the gate reasons in a log. zsh reports the
        LAST command of a pipeline, so RC becomes tee's 0 and placement,
        support window and thermal are bypassed at once -- on an elbow that has
        already browned out twice.
      * `exit 3` -> an echo. The abort still speaks and still prints "TRIAL
        ABORTED -- nothing launched", then falls through and drives the arm.
    """
    src = _src()
    for gate in ("preflight.py", "home_arm.py", "scene_check.py"):
        assert gate in src, f"pi0.5 runner skips {gate}"

    rc, stdout, calls, _ = _run_gates()
    homes = [ln for ln in calls.splitlines() if "home_arm.py" in ln]
    assert homes, "the runner never homed the arm"
    for line in homes:
        assert re.search(r"--pose\s+v4\b", line), (
            f"homed with {line.strip()!r} -- the pose must be the side table's generation "
            f"(the stub answers 'v4' for --print gen), never a constant: v4's elbow is 3.3 deg above v3's"
        )
    pre = [ln for ln in calls.splitlines() if "preflight.py" in ln]
    assert pre, "preflight never ran"
    assert pre[-1].split()[-1] == "v4", (
        f"preflight gated against the {pre[-1].split()[-1]!r} support window, not the side "
        f"table's generation -- a stale window both rejects valid starts and admits invalid ones"
    )
    homed_for = [ln.split("=", 1)[1] for ln in calls.splitlines()
                 if ln.startswith("LAUNCH_ENV CAPSTONE_HOME_POSE=")]
    assert homed_for == ["v4"], (
        f"the rollout process was handed CAPSTONE_HOME_POSE={homed_for or 'nothing'} -- "
        f"unset or wrong, the wrapper re-homes to its v2 default between episodes"
    )

    # ...and a failed gate must STOP the run, not narrate it.
    rc, stdout, calls, _ = _run_gates(preflight_rc=1,
                                      preflight_msg="FAIL thermal: elbow 63.1C")
    launched = "LAUNCHED" in calls.splitlines()
    narrated = "TRIAL ABORTED" in stdout
    assert not launched, (
        "preflight FAILED and the rollout launched anyway" + (
            " -- the abort branch even printed 'TRIAL ABORTED' first, so it speaks "
            "and then falls through instead of exiting"
            if narrated else
            " -- the gate's exit status is never read (one `| tee` added to that "
            "pipeline is enough to do this, since zsh reports the LAST command)"
        ) + ": a 45 s contact-rich press would run from a rejected pose or a hot elbow"
    )
    assert rc == 3, f"a failed gate exited {rc}, not 3"
    spoke = any(ln.startswith("SAY abort") for ln in calls.splitlines())
    assert narrated and spoke, (
        "the abort neither printed nor spoke a reason"
    )


def test_runner_does_not_enable_the_grasp_lock():
    """CAPSTONE_GRASP_LOCK patches ACT's action-queue path. It has no defined
    behaviour for pi0.5's flow-matching chunks and must stay off.

    Sep 5: the old check forbade one literal in the source and nothing more, so
    commenting out `unset CAPSTONE_GRASP_LOCK` -- the natural bench edit ("let
    me see what the lock does to pi0.5") -- kept it green. The lock does not
    have to be set IN the runner to apply: run_probe_episode.sh:12 exports it,
    and rollout_30hz_stale_ok.py:320 acts on whatever is in the environment.
    Its only trace would be one line inside ${RUN}.log that nothing greps, and
    the trial would still land in trials.csv as a plain pi0.5 result. So export
    it into the runner and prove the child process does not see it.
    """
    src = _src()
    assert "CAPSTONE_GRASP_LOCK=1" not in src, (
        "grasp lock must not be enabled for pi0.5 — it is an ACT-queue patch"
    )
    rc, stdout, calls, _ = _run_gates(inherit={"CAPSTONE_GRASP_LOCK": "1"})
    leaked = [ln for ln in calls.splitlines()
              if ln.startswith("LAUNCH_ENV CAPSTONE_GRASP_LOCK")]
    assert not leaked, (
        f"an inherited grasp lock reached the rollout process ({leaked}) -- ACT's "
        f"grasp-phase chunk locking would be applied to pi0.5's flow-matching "
        f"chunks and the trial would still score as a plain pi0.5 result"
    )


def test_runner_forces_narrow_state_routing_explicitly():
    """pi0.5 declares state width 32 (max_state_dim padding) while training on
    6-dim. The context.py fix handles this, but the runner states it explicitly
    so a regression in that patch cannot silently widen the state.

    _code() (Sep 3) stopped a COMMENT from satisfying the pin; Sep 5 found the
    other half of the same bug still open -- a bare, unexported
    `CAPSTONE_FORCE_WIDE_STATE=0` satisfies the scan while never reaching the
    child (`zsh -c 'X=0; env | grep X'` prints nothing). Assert the rollout's
    OWN environment instead. This is the belt for the day someone re-applies
    the older rollout_context_load_state_patch.txt, whose rule is
    "declared 32 > 6 -> widen": pi0.5 would then receive load+current in state
    dims 6-17 that it trained as always-zero, with no crash and no shape error.
    """
    # Sep 6: the pin is PER SIDE, resolved by pi05_sides.py -- A narrow (0), B wide (1). B carries
    # 18-dim normalizer statistics behind the same declared [32]; a narrow B meets a 6-dim state
    # against an 18-dim normalizer on its first tick.
    for side, want in (("A", "0"), ("B", "1")):
        rc, stdout, calls, _ = _run_gates(side=side)
        pins = [ln.split(" ", 1)[1] for ln in calls.splitlines()
                if ln.startswith("LAUNCH_ENV CAPSTONE_FORCE_WIDE_STATE=")]
        assert pins == [f"CAPSTONE_FORCE_WIDE_STATE={want}"], (
            f"side {side}: the rollout process was handed "
            f"{pins or 'no CAPSTONE_FORCE_WIDE_STATE at all'} -- the pin must come from the side "
            f"table, and an unexported assignment satisfies a text scan but not the child environment"
        )


def test_act_runner_left_untouched():
    """run_scored_trial.sh must not have grown pi0.5 concerns."""
    act = (TOOLS / "run_scored_trial.sh").read_text()
    assert "rename_map" not in act, (
        "the ACT runner was modified for pi0.5; it runs Block 2B and must stay stable"
    )


def test_runner_passes_the_trained_language_prompt():
    """pi0.5 is language-conditioned; an empty prompt is a silent distribution shift.

    Found by the Sep 3 2026 night audit. The runner passes --dataset.* flags but
    neither --task nor --dataset.single_task. In lerobot 0.6.1 that means
    `task: str = ""` (rollout/configs.py:244) stays empty, NEITHER propagation
    branch fires (configs.py:365-370), and context.py:562 hands the inference
    engine `task=""`. Pi05PrepareStateTokenizerProcessorStep then builds
    "Task: , State: ..." -- while tools/pi05_dryrun.py feeds the real trained
    string, so the rehearsal passes and the bench run gets a different input.
    That is plumbing presenting as a policy result: it would have read as
    "pi0.5 cannot do the task."
    """
    code = _code()
    assert "--dataset.single_task=" in code or "--task=" in code, (
        "runner passes no task string: pi0.5 would receive an EMPTY language "
        "prompt on hardware while pi05_dryrun.py feeds the trained one"
    )


def test_the_runners_task_string_is_the_one_the_policy_trained_on():
    """The prompt is READ from the side's dataset by pi05_sides.py (Sep 6), never typed here.
    The runner must pass that resolved value and refuse an empty one."""
    import subprocess
    import sys

    import pytest

    sys.path.insert(0, str(TOOLS))
    import pi05_sides as ps

    side = ps.active_sides({})["A"]
    try:
        trained = ps.trained_task(side.dataset)
    except ps.TaskUnavailable:
        pytest.skip(f"{side.dataset} tasks.parquet not reachable")
    # 60 s, not 180: the answer takes 0.27 s wall (Sep 6). The old budget bought a
    # 3-minute stall in the pre-session suite before failing; with the hard exit
    # below a hang can only be a regression, and it should fail LOUD and fast.
    out = subprocess.run([sys.executable, str(TOOLS / "pi05_sides.py"), "--print", "task", "--side", "A"],
                         capture_output=True, text=True, timeout=_PRINT_TIMEOUT_S)
    assert out.returncode == 0 and out.stdout.strip() == trained, out.stderr[-300:]
    code = _code()
    assert re.search(r'--dataset\.single_task="\$TASK"', code), "runner does not pass the resolved $TASK"
    assert re.search(r'-z\s+"?\$\{?TASK', code), "runner does not refuse an empty prompt"


# ---------------------------------------------------------------------------
# The --print path must not reach interpreter teardown (Sep 6 2026)
# ---------------------------------------------------------------------------
#
# `pi05_sides.py --print task --side A` printed the right prompt and then sat
# for 180 s in the full suite (03:xx, 1 failed / 506 passed), and passed alone
# in 1.7 s. Reproduced at a desk with no suite at all: 2 hangs in ~155 plain
# subprocess runs (~1 %/invocation). `sample <pid>` of the stuck process:
#
#   Py_Exit -> exit -> __cxa_finalize_ranges
#     -> ~shared_ptr<arrow::internal::ThreadPool> -> ThreadPool::Shutdown
#     -> std::condition_variable::wait            (main thread)
#   ThreadPool::LaunchWorkersUnlocked worker -> condition_variable::wait
#
# i.e. arrow's global thread pool (spawned by pandas.read_parquet inside
# trained_task) and its C++ static destructor miss each other's wake-up while
# libc runs exit(). The answer is already on stdout; only the exit is lost.
# run_pi05_trial.sh:70 captures that answer with no timeout, so on the bench
# it is an indefinite stall at "resolving the prompt" before any gate speaks.
#
# The fix: once the answer is flushed, `os._exit` -- no Python finalisation, no
# atexit, no C++ destructors. These tests pin the mechanism (atexit is never
# reached), the flush (a pipe-backed stdout is block-buffered, so a hard exit
# without the flush prints NOTHING), and the exit status the runner keys on.

_PRINT_TIMEOUT_S = 60


def _clean_env(**extra):
    import os
    env = {k: v for k, v in os.environ.items() if not k.startswith("CAPSTONE")}
    env.update(extra)
    return env


@pytest.mark.parametrize("field", ["force", "task"])
def test_print_exits_before_interpreter_teardown(field, tmp_path):
    """Run the script as __main__ with an atexit marker armed: the marker must
    never be written. Reaching atexit means reaching exit() and the arrow
    destructor race behind it. `force` loads numpy only; `task` is the path
    that actually spawns the arrow workers."""
    import subprocess
    import sys

    marker = tmp_path / "atexit_ran"
    prog = (
        "import atexit, runpy, sys\n"
        f"atexit.register(lambda: open({str(marker)!r}, 'w').write('ran'))\n"
        f"sys.argv = [{str(TOOLS / 'pi05_sides.py')!r}, '--print', {field!r}, '--side', 'A']\n"
        f"runpy.run_path({str(TOOLS / 'pi05_sides.py')!r}, run_name='__main__')\n"
    )
    out = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True,
                         timeout=_PRINT_TIMEOUT_S, env=_clean_env())
    assert out.returncode == 0, out.stderr[-400:]
    assert out.stdout.strip(), "the answer never reached the pipe"
    assert not marker.exists(), (
        f"--print {field}: the process ran its atexit handlers, i.e. it went through "
        f"interpreter teardown and libc exit() -- the path on which arrow's ThreadPool "
        f"destructor hangs ~1 % of the time. The --print path must os._exit once flushed."
    )


@pytest.mark.parametrize("side", ["A", "B"])
@pytest.mark.parametrize("field", ["repo", "dataset", "force", "state_dim", "gen", "task", "checkpoint"])
def test_every_print_field_arrives_complete_through_a_pipe(field, side):
    """The runner reads each field through `$(...)` -- a pipe. Every offline
    field must arrive byte-for-byte as print_field resolves it in-process.
    (`snapshot` is excluded: it materialises a Hub checkpoint.)"""
    import subprocess
    import sys

    sys.path.insert(0, str(TOOLS))
    import pi05_sides as ps

    env = _clean_env(CAPSTONE_CHECKPOINT="005000")
    want = ps.print_field(field, ps.active_sides({"CAPSTONE_CHECKPOINT": "005000"})[side]) \
        if field != "checkpoint" else "005000"
    out = subprocess.run([sys.executable, str(TOOLS / "pi05_sides.py"), "--print", field, "--side", side],
                         capture_output=True, text=True, timeout=_PRINT_TIMEOUT_S, env=env)
    assert out.returncode == 0, out.stderr[-400:]
    assert out.stdout == want + "\n", (
        f"--print {field} --side {side} delivered {out.stdout!r}, expected {want + chr(10)!r} -- "
        f"a hard exit without flushing leaves a block-buffered pipe empty and the runner "
        f"launches with an empty value"
    )


def test_print_abort_keeps_its_exit_status_and_reason():
    """The runner's `|| { say abort; exit }` keys on a non-zero status; the hard
    exit must carry main()'s status, and the reason must already be on stderr."""
    import subprocess
    import sys

    out = subprocess.run([sys.executable, str(TOOLS / "pi05_sides.py"), "--print", "task", "--side", "B"],
                         capture_output=True, text=True, timeout=_PRINT_TIMEOUT_S,
                         env=_clean_env(CAPSTONE_POLICY_GEN="v3"))
    assert out.returncode == 5, f"side B in v3 must abort with 5, got {out.returncode}"
    assert out.stdout == "", f"an aborted resolve must print nothing the runner could capture: {out.stdout!r}"
    assert out.stderr.startswith("ABORT: side B does not exist in generation v3"), out.stderr[-300:]
