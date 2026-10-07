"""Regression: every pre-launch check must ABORT the trial, in code.

Born from the Aug 29 2026 session, where the failure mode was not a missing
check but a check that only printed. In one block:

  * ran `home_arm`, saw "WARNING: adjusted residual exceeds 2.5", wrote "one
    thing I'm watching", and launched anyway — three times. The elbow was at
    74.1 deg, below the v2 training floor; a deterministic replay then missed
    the tube entirely.
  * emitted the tube placement delta into a summary without enforcing it, and
    launched an episode with the tube 10.2 px off.
  * launched from a stale worktree copy once already (Aug 27 INFRA #3).

So the gate lives here, is composed from parsed tool output, and returns
reasons rather than warnings. If a condition should not stop a launch, it does
not belong in this file.
"""

import ast
import re
import sys

import pytest
from conftest import PROJECT, TOOLS

sys.path.insert(0, str(TOOLS))

import preflight  # noqa: E402

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def _joint_line(ping: str, joint: str) -> str:
    """The one ping_motors line for `joint` -- the reading shape (ping_motors.py:111-116) or the
    no-response shape (ping_motors.py:97)."""
    [line] = [ln for ln in ping.splitlines() if re.search(rf"\bid \d+ {joint}\b", ln)]
    return line


SCENE_OK = """homography inliers: 342/400
tube (blue cap)   : ref=( 360.0, 355.8)  live=( 357.7, 356.6)  delta=( -2.3, +0.7) px  |d|=2.4
rack (orange funnel): ref=( 537.4, 481.0)  live=( 538.8, 481.1)  delta=( +1.5, +0.2) px  |d|=1.5
tube angle        : ref= +0.0°  live= +0.0°  delta= +0.0°
"""

SCENE_TUBE_OFF = SCENE_OK.replace("|d|=2.4", "|d|=10.2")

# elbow 79.7 is the v2 frame-0 mean. This fixture used 77.1 before the Aug 29
# tightening; that sits 1.2 sigma low and is now correctly rejected.
HOME_OK = (
    "settled: {'shoulder_pan': 2.8, 'shoulder_lift': -99.3, 'elbow_flex': 79.7, "
    "'wrist_flex': 73.4, 'wrist_roll': -0.9, 'gripper': 1.2}"
)
HOME_ELBOW_LOW = HOME_OK.replace("'elbow_flex': 79.7", "'elbow_flex': 74.1")

# A real ping receipt, verbatim: the one a scored trial launched on, Sep 6 2026 16:14
# (tools/scored_logs/h83j2kp6_20260906_161402.ping:1-10). Inlined, not read from disk:
# scored_logs/ is gitignored (.gitignore:23), and a thermal test that skips when its input is
# absent is the Sep 7 audit's LOW finding. Until Sep 12 this fixture held TWO joints and the gate
# passed it -- four servos nobody had read, gated green. See "a servo nobody read" below.
PING_OK = """
=== FOLLOWER (white, jaws)  /dev/tty.usbmodem5C4C1245641 ===
  broadcast: nothing responded
  id 1 shoulder_pan    pos  2013  load    60  cur    1   44C  125  torque 1  ok
  id 2 shoulder_lift   pos   779  load   -56  cur    0   47C  124  torque 0  ok
  id 3 elbow_flex      pos  3033  load   -44  cur    1   47C  126  torque 1  ok
  id 4 wrist_flex      pos  2945  load    44  cur    1   52C  123  torque 1  ok
  id 5 wrist_roll      pos  2039  load    72  cur    3   47C  124  torque 1  ok
  id 6 gripper         pos  2063  load     0  cur    0   46C  125  torque 1  ok
  all six motors present and clean.
"""
_ELBOW_LINE = _joint_line(PING_OK, "elbow_flex")
PING_HOT = PING_OK.replace(_ELBOW_LINE, _ELBOW_LINE.replace(" 47C", " 62C"))  # the elbow's line only


# ---------------------------------------------------------------- parsing


def test_parse_tube_delta():
    assert preflight.parse_tube_delta(SCENE_OK) == pytest.approx(2.4)


def _preflight_cli(tmp_path, scene: str, home: str, ping: str, dataset: str = "v2"):
    """Run tools/preflight.py the way the launcher runs it; return (exit status, stdout).

    Every gating test in this file asserts on gate()'s return VALUE. The
    launcher reads none of that: run_scored_trial.sh does
    `OUT=$(... preflight.py ...); RC=$?` and branches on RC alone. So the exit
    status needs a harness of its own, or the module can go back to narrating
    without a single red test.
    """
    import subprocess

    args = []
    for name, text in (("scene", scene), ("home", home), ("ping", ping)):
        p = tmp_path / f"run.{name}"
        p.write_text(text)
        args.append(str(p))
    r = subprocess.run(
        [sys.executable, str(TOOLS / "preflight.py"), *args, dataset],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return r.returncode, r.stdout.strip()


def test_parse_tube_delta_missing_is_an_error_not_a_pass(tmp_path):
    """Raising in-process is only half of "an error, not a pass".

    The raise has to become a non-zero EXIT, and that conversion lives in the
    `except` handler at the bottom of preflight.py, which nothing used to
    exercise. Turn its exit(1) into exit(0) — the plausible "the message
    already says ABORT" edit — and a capture the gate cannot read prints
    "PREFLIGHT ABORT — could not evaluate a gate: ..." and gates GREEN. That is
    the measured Aug 30 case (the arm parked over the tube, scene_check
    reported it missing) and it reverses this module's own promise never to
    degrade to pass.
    """
    with pytest.raises(ValueError):
        preflight.parse_tube_delta("homography inliers: 12/400\n")

    rc, out = _preflight_cli(tmp_path, "homography inliers: 12/400\n", HOME_OK, PING_OK)
    assert rc == 1, (
        f"preflight.py printed {out!r} and exited {rc} on a capture it could not read — "
        "the wrapper branches on $? alone, so a 0 here launches a scored trial with the "
        "placement gate never evaluated at all"
    )
    assert "could not evaluate" in out


def test_parse_settled_pose():
    pose = preflight.parse_settled_pose(HOME_OK)
    assert pose["elbow_flex"] == pytest.approx(79.7)
    assert pose["shoulder_lift"] == pytest.approx(-99.3)
    assert len(pose) == 6


def test_parse_settled_pose_missing_is_an_error():
    with pytest.raises(ValueError):
        preflight.parse_settled_pose("HOMED (worst adjusted residual 1.2)")


def test_parse_elbow_temp():
    assert preflight.parse_elbow_temp(PING_OK) == 47
    assert preflight.parse_elbow_temp(PING_HOT) == 62


def test_parse_elbow_temp_missing_is_an_error():
    with pytest.raises(ValueError):
        preflight.parse_elbow_temp("  broadcast: nothing responded\n")


# ---------------------------------------------------------------- gating


def test_clean_preflight_passes():
    ok, reasons = preflight.gate(SCENE_OK, HOME_OK, PING_OK)
    assert ok, reasons


def test_tube_out_of_place_aborts(tmp_path):
    """The refusal has to reach the launcher, and the only wire is the exit status.

    `sys.exit(0 if ok else 1)` collapsing to `sys.exit(0)` is a one-token edit
    that leaves the module still PRINTING "PREFLIGHT ABORT — placement: ..."
    while the wrapper sets GATED=1 and runs the trial with the tube 10.2 px
    off. That is the pre-fix "narrate" form of this whole module, and the
    in-process (ok, reasons) assertions below cannot see it.
    """
    ok, reasons = preflight.gate(SCENE_TUBE_OFF, HOME_OK, PING_OK)
    assert not ok
    assert any("placement" in r for r in reasons)

    rc, out = _preflight_cli(tmp_path, SCENE_TUBE_OFF, HOME_OK, PING_OK)
    assert rc == 1, f"preflight.py printed {out!r} but exited {rc} — the launcher gates on $?"
    assert "ABORT" in out
    # ...and the other direction, or a gate that always exits 1 would look fine here
    assert _preflight_cli(tmp_path, SCENE_OK, HOME_OK, PING_OK) == (0, "PREFLIGHT PASS (v2)")


# ------------------------------------------- running the wrapper, not grepping it
#
# Sep 5 audit. Everything this file said about run_scored_trial.sh was a
# substring grep, and a grep proves a word is present, never that the script
# acts on it. Five one-line edits a tired person would plausibly make all kept
# the greps green: `RC=$?` moved behind the `echo` (RC then reports the echo,
# so every gate in the file becomes narration), a dropped `break`, a dropped
# `exit 3`, a dropped `exit 2`, a dropped `"$DATASET_TAG"` argv. So we cut the
# real section out of the real script and RUN it under zsh with $B/python
# replaced by a recorder.


class _Ran:
    """What one executed section of run_scored_trial.sh actually did."""

    def __init__(self, rc, out, calls, spoken):
        self.rc, self.out, self.calls, self.spoken = rc, out, calls, spoken

    def __repr__(self):
        return f"_Ran(rc={self.rc}, spoken={self.spoken}, calls={self.calls}, out={self.out!r})"


# Stands in for $B/python: records argv, and replays a scripted exit code for
# each tools/preflight.py call so the loop's control flow becomes observable.
_STUB_PYTHON = r"""#!/bin/zsh
print -r -- "$@" >> "$CALLS"
if [[ "$1" == */preflight.py ]]; then
  n=$(( $(cat "$NCALL") + 1 )); print -r -- $n > "$NCALL"
  rcs=(${=GATE_RC}); rc=$rcs[n]; [[ -z $rc ]] && rc=$rcs[-1]
  print -r -- "$GATE_OUT"
  exit $rc
fi
exit 0
"""


def _section(start: str, end: str) -> str:
    """Cut one region out of run_scored_trial.sh so it can be executed alone."""
    src = (TOOLS / "run_scored_trial.sh").read_text()
    i = src.index(start)
    j = src.index(end, i)
    return src[i:j]


def _run_shell(
    tmp_path,
    body: str,
    dataset: str = "v2",
    gate_out: str = "PREFLIGHT PASS (v2)",
    gate_rc: str = "0",
    label: str = "P001-A",
    mode: str = "off",
    bin_dir=None,
    cwd=None,
) -> _Ran:
    """Execute `body` under zsh with the wrapper's own preamble variables set.

    `say` is captured instead of spoken. `gate_rc` is one exit code per
    preflight.py call ("1 1 1", "0 1 1", ...), the last one repeating.
    """
    import os
    import subprocess

    work = tmp_path / "sh"
    stub_bin = work / "bin"
    stub_bin.mkdir(parents=True, exist_ok=True)
    stub = stub_bin / "python"
    stub.write_text(_STUB_PYTHON)
    stub.chmod(0o755)
    calls, ncall, spoken = work / "calls", work / "ncall", work / "spoken"
    calls.write_text("")
    spoken.write_text("")
    ncall.write_text("0")
    script = "\n".join(
        [
            "set -u",
            f"say() {{ print -r -- \"$@\" >> '{spoken}' }}",
            f"B='{bin_dir or stub_bin}'",
            f"RUN='{work}/run'",
            f"DATASET_TAG={dataset}",
            f"START_POSE={dataset}",     # the loop's start-pose inputs (Sep 6): frame 0, no close flag
            "HOME_FLAGS=()",
            f"LABEL={label}",
            f"MODE={mode}",
            "NSTEPS=25",
            f"POLICY=faithqin/act-tube-A-{dataset}",
            body,
            "print -r -- REACHED_THE_END",
        ]
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("CAPSTONE_")}
    env.update(CALLS=str(calls), NCALL=str(ncall), GATE_OUT=gate_out, GATE_RC=gate_rc)
    r = subprocess.run(
        ["/bin/zsh", "-c", script],
        cwd=str(cwd or work),
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    return _Ran(
        r.returncode, r.stdout, calls.read_text().splitlines(), spoken.read_text().splitlines()
    )


def test_elbow_below_training_support_aborts(tmp_path):
    """The measured Aug 29 failure: elbow 74.1 deg produced a total replay miss.

    The second half pins the argument that decides WHICH training support the
    pose is judged against. Nothing else in the suite covers that call site:
    drop the 4th argv from the preflight call and the gate silently falls back
    to home_gate.DEFAULT_DATASET ("v2") while home_arm.py --pose v3 homed to
    v3 — the Sep 3 "gated on one version, homed to another" bug in its mirror
    image. The windows barely overlap (v2 elbow [77.53, 81.93], v3
    [79.51, 85.17]), so the fallback refuses an in-distribution v3 start of
    84.5 three times and admits a 78.0 that v3 never saw.
    """
    ok, reasons = preflight.gate(SCENE_OK, HOME_ELBOW_LOW, PING_OK)
    assert not ok
    assert any("elbow_flex" in r for r in reasons)

    # the tag is load-bearing, not decorative
    home_84 = HOME_OK.replace("'elbow_flex': 79.7", "'elbow_flex': 84.5")
    assert preflight.gate(SCENE_OK, home_84, PING_OK, dataset="v3")[0]
    assert not preflight.gate(SCENE_OK, home_84, PING_OK, dataset="v2")[0]

    ran = _run_shell(
        tmp_path, _section("GATED=0", "# --- launch"), dataset="v3", gate_out="PREFLIGHT PASS (v3)"
    )
    homed = [c for c in ran.calls if "home_arm.py" in c]
    gated = [c for c in ran.calls if "preflight.py" in c]
    assert homed and homed[-1].split()[-1] == "v3", f"the loop homed with {homed!r}"
    assert gated, f"the gate loop never ran preflight.py: {ran.calls!r}"
    assert gated[-1].split()[-1] == "v3", (
        f"run_scored_trial.sh homed to v3 but gated with argv {gated[-1].split()[1:]!r} — "
        "without the dataset tag preflight falls back to v2 bounds and judges a v3 start "
        "against the wrong training support"
    )


def test_hot_elbow_aborts():
    ok, reasons = preflight.gate(SCENE_OK, HOME_OK, PING_HOT)
    assert not ok
    assert any("temp" in r.lower() for r in reasons)


def test_all_failures_are_reported_not_just_the_first():
    """A gate that stops at the first problem hides the others and costs a
    second round trip to the bench."""
    ok, reasons = preflight.gate(SCENE_TUBE_OFF, HOME_ELBOW_LOW, PING_HOT)
    assert not ok
    assert len(reasons) >= 3


def test_gate_is_not_fooled_by_a_shoulder_lift_settle():
    """shoulder_lift always reads ~5-6 deg off by design; gating it would abort
    every launch."""
    ok, _ = preflight.gate(SCENE_OK, HOME_OK, PING_OK)
    assert ok


# ---------------------------------------------------------------- worktree


def test_worktree_path_is_refused(tmp_path):
    """Aug 27 INFRA #3: a probe was invalidated by a stale worktree wrapper.

    Three things have to hold, and the one-line version pinned only the first:

      * the literal worktree path is refused;
      * refusal survives a path that merely RESOLVES into a worktree. Both
        worktree fixtures — this one and test_main_checkout_is_accepted — fed
        already-absolute, already-resolved strings, so dropping the
        `Path(path).resolve()` was invisible to them. That drops every
        relative, symlinked or `$(pwd -L)` spelling straight through, which is
        how Aug 27 happened in the first place;
      * run_scored_trial.sh EXITS on the refusal. Delete `exit 2` and the
        script says "abort. worktree.", prints the ABORT line, and then runs
        the whole scored trial from the stale copy anyway — a probe that is not
        obviously wrong, just quietly invalid.
    """
    import os

    bad = "~/x/.worktrees/liber-moss/tools"
    assert not preflight.is_main_checkout(bad)

    real = tmp_path / ".worktrees" / "liber-moss" / "tools"
    real.mkdir(parents=True)
    link = tmp_path / "tools"
    link.symlink_to(real)
    assert not preflight.is_main_checkout(link), (
        f"{link} was accepted — it points at {real}, so is_main_checkout is judging the "
        "spelling of the path instead of where it lands"
    )

    guard = _section('if [ -z "${CAPSTONE_ALLOW_WORKTREE:-}" ]', "# Which training distribution")
    real_bin = os.path.dirname(sys.executable)
    stale = _run_shell(
        tmp_path,
        f'PROJ="~/x/.worktrees/liber-moss"\n{guard}',
        bin_dir=real_bin,
        cwd=PROJECT,
    )
    assert stale.rc == 2 and "REACHED_THE_END" not in stale.out, (
        f"the worktree guard printed {stale.out.strip()!r} and then kept going "
        f"(rc={stale.rc}) — a full scored trial runs from the stale worktree tools"
    )
    good = _run_shell(tmp_path, f'PROJ="{PROJECT}"\n{guard}', bin_dir=real_bin, cwd=PROJECT)
    assert good.rc == 0, f"the guard refuses the MAIN checkout too (rc={good.rc}) — nothing runs"


def test_main_checkout_is_accepted():
    assert preflight.is_main_checkout(str(PROJECT))


def test_scored_episode_length_matches_the_v2_protocol():
    """v2 scored trials hold 875 frames ~= 45 s and the protocol says 'Max 45 s'.
    Running scored trials at the probe's 60 s would break comparability with
    the baseline the primary is measured against."""
    assert preflight.SCORED_EPISODE_TIME_S == 45


def test_scored_trial_uses_a_rollout_prefixed_repo_id(tmp_path):
    """lerobot rejects a rollout dataset whose name does not start with
    'rollout_' (rollout/context.py). Block 2 trial 1 on Aug 29 died on this
    after the gates had already passed and the cameras were open.

    Grading the `REPO=` assignment graded the wrong end of the wire. Inline a
    bare `--dataset.repo_id="faithqin/${LABEL}"` at the call site and the
    assignment — and the grep — survive untouched while the launch hands
    lerobot a name with no prefix. So read the argv the rollout wrapper is
    actually given.
    """
    import re

    ran = _run_shell(tmp_path, _section("# --- launch", "# --- verify"), label="P001-A")
    launch = [c for c in ran.calls if "rollout_30hz_stale_ok.py" in c]
    assert len(launch) == 1, f"the launch section did not run the rollout wrapper: {ran.calls!r}"
    m = re.search(r"--dataset\.repo_id=(\S+)", launch[0])
    assert m, f"no --dataset.repo_id in the launch argv: {launch[0]!r}"
    assert m.group(1).startswith("faithqin/rollout_"), (
        f"the scored trial launches with repo_id={m.group(1)!r} — lerobot aborts at dataset "
        "setup, after preflight has passed, the arm has been homed up to three times and the "
        "cameras are open"
    )
    assert "P001-A" in m.group(1), (
        f"repo_id={m.group(1)!r} does not carry the trial label — trials would overwrite "
        "each other's dataset and the scoring glob would find the wrong episode"
    )


def test_scored_trial_runs_45_second_episodes():
    src = (TOOLS / "run_scored_trial.sh").read_text()
    assert "--dataset.episode_time_s=45" in src, (
        "scored trials must run 45 s to match the v2 protocol and baseline"
    )


def test_scored_trial_retries_homing_before_aborting(tmp_path):
    """A single home is not reliable: the elbow under-reaches when hot and
    over-reaches from some starting poses (74.1-85.3 measured Aug 29), and the
    gripper creeps open after ACT-B episodes. Re-homing costs ~5 s.

    `"for attempt in 1 2 3" in src` proved the loop was written, never that it
    works, and two one-line edits kept it green. Moving `RC=$?` behind the
    `echo` — the classic $? clobber, on the ONE line where preflight's verdict
    enters the wrapper — makes RC the echo's status, i.e. always 0, and every
    gate in this file becomes narration again. Deleting the `break` latches
    GATED on attempt 1 and then launches from attempt 3's unvetted pose. So run
    the loop and count what it did.
    """
    loop = _section("GATED=0", "# --- launch")

    # a gate that never passes must spend all three attempts and then abort
    hard = _run_shell(
        tmp_path,
        loop,
        gate_out="PREFLIGHT ABORT — home: elbow_flex=74.10 outside [77.53, 81.93]",
        gate_rc="1 1 1",
    )
    assert hard.rc == 3 and "REACHED_THE_END" not in hard.out, (
        f"preflight refused three times and the wrapper launched anyway (rc={hard.rc}). "
        f"RC must be captured from preflight itself, not from whatever ran last.\n{hard.out}"
    )
    assert sum("home_arm.py" in c for c in hard.calls) == 3, (
        f"home was attempted {sum('home_arm.py' in c for c in hard.calls)}x, expected 3 — "
        "one home is not reliable enough to spend a scored trial on"
    )

    # a gate that passes must STOP: the pose it vetted is the pose that rolls out
    quick = _run_shell(tmp_path, loop, gate_out="PREFLIGHT PASS (v2)", gate_rc="0 1 1")
    assert "REACHED_THE_END" in quick.out and quick.rc == 0, "a clean gate did not reach the launch"
    assert sum("home_arm.py" in c for c in quick.calls) == 1, (
        f"the loop re-homed {sum('home_arm.py' in c for c in quick.calls)}x after gating green — "
        "GATED is latched, so the rollout starts from a pose nothing checked, and it burns two "
        "extra elbow holds against a documented thermal budget"
    )


def test_abort_speaks_the_cause_not_the_category(tmp_path):
    """'abort, preflight' tells the operator nothing and the TTS mangles it.
    The spoken message must name the failing gate — and then stop the trial.

    Grepping for the cue strings let the abort path rot while every cue stayed
    in the file. Deleting `REASON="$OUT"` (it looks redundant beside the echo
    two lines up) leaves REASON empty, every `case` arm misses, and every abort
    speaks "abort. unknown gate." — the same non-information this test exists
    to forbid, one costume over. Deleting `exit 3` leaves the abort speaking
    the right cause and then falling through into the launch: a 45 s scored
    episode from the pose the gate just rejected, written into trials.csv as if
    it were valid, which poisons the paired comparison rather than merely
    wasting a slot. So drive the loop and listen.
    """
    loop = _section("GATED=0", "# --- launch")
    for reason, expected in (
        (
            "PREFLIGHT ABORT — home: elbow_flex=74.10 outside [77.53, 81.93]",
            "abort. elbow out of range.",
        ),
        ("PREFLIGHT ABORT — placement: tube |d|=10.2 px > 8 px", "abort. tube placement."),
        ("PREFLIGHT ABORT — thermal: elbow temp 62C > 60C", "abort. elbow too hot."),
        (
            "PREFLIGHT ABORT — could not evaluate a gate: no tube |d| in scene_check output",
            "abort. a check could not be read.",
        ),
    ):
        ran = _run_shell(tmp_path, loop, gate_out=reason, gate_rc="1")
        assert ran.spoken == [expected], (
            f"the gate said {reason!r} and the operator heard {ran.spoken!r} — expected "
            f"[{expected!r}]. 'abort. unknown gate.' means the reason never reached the case."
        )
        assert ran.rc == 3 and "REACHED_THE_END" not in ran.out, (
            f"the wrapper spoke {expected!r} and then kept going (rc={ran.rc}) — a scored "
            f"trial launches from the pose the gate just rejected\n{ran.out}"
        )


def test_scene_is_captured_after_the_arm_is_homed():
    """Aug 30: scene_check reported the tube NOT FOUND because the arm had been
    parked over it since the previous session. The parser raises rather than
    passing, so it failed safe — but it burns a trial slot and reads as a
    placement problem when it is an occlusion problem. Home first, then look."""
    src = (TOOLS / "run_scored_trial.sh").read_text()
    home_i = src.index("home_arm.py")
    scene_i = src.index("scene_check.py")
    assert home_i < scene_i, (
        "run_scored_trial.sh captures the scene before homing; a parked arm can "
        "occlude the tube and the placement gate then reads garbage"
    )


# ---------------------------------------------------------------- thermal, every joint
PING_ALL = """  id 1 shoulder_pan    pos  2009  load    64  cur    2   45C  125  torque 1  ok
  id 2 shoulder_lift   pos   778  load   -44  cur    0   48C  124  torque 1  ok
  id 3 elbow_flex      pos  3056  load   208  cur   37   57C  126  torque 1  ok
  id 4 wrist_flex      pos  2951  load   260  cur   57   68C  123  torque 1  ok
  id 5 wrist_roll      pos  2027  load    52  cur    1   48C  124  torque 1  ok
  id 6 gripper         pos  2065  load   -84  cur    6   49C  125  torque 1  ok
"""


def test_parse_joint_temps_reads_every_joint_from_its_own_line():
    t = preflight.parse_joint_temps(PING_ALL)
    assert t == {"shoulder_pan": 45, "shoulder_lift": 48, "elbow_flex": 57,
                 "wrist_flex": 68, "wrist_roll": 48, "gripper": 49}
    with pytest.raises(ValueError, match="temperature"):
        preflight.parse_joint_temps("nothing here")


# These three took the v4 frame-0 mean as their home until Sep 12, through the v4_frame0_stats
# fixture, which SKIPS when the v4 parquet is not on the machine (conftest.py:104-105). The thermal
# gate never reads the pose, so the fixture bought nothing and cost the coverage: measured Sep 12
# with the parquet pointed away, putting the elbow-only gate back
# (`temps = {"elbow_flex": parse_elbow_temp(ping_text)}`) left this file at 21 passed, 3 skipped --
# the Sep 6 bug restored, green. They now gate on HOME_OK (v2), which needs no data at all.
def test_a_hot_wrist_aborts_even_when_the_elbow_is_cool():
    """Measured Sep 6: the arm was parked holding the v4 home pose during software work and
    wrist_flex reached 68C — one degree under the 69C brownout on the ledger — while the elbow sat
    at 57C. The old gate read the elbow only: it would have passed this trial and named the
    elbow's 57C as the reason it was safe. The brownout is a property of the servo, not the joint."""
    ok, reasons = preflight.gate(SCENE_OK, HOME_OK, PING_ALL)
    assert not ok, "a 68C joint passed the thermal gate"
    thermal = [r for r in reasons if r.startswith("thermal:")]
    assert thermal and "wrist_flex" in thermal[0] and "68C" in thermal[0], reasons
    assert preflight.parse_elbow_temp(PING_ALL) == 57      # and the elbow really was cool


def test_every_joint_cool_still_passes():
    cool = PING_ALL.replace("68C", "47C").replace("57C", "50C")
    ok, reasons = preflight.gate(SCENE_OK, HOME_OK, cool)
    assert ok, reasons


def test_the_gate_still_refuses_output_it_cannot_read():
    with pytest.raises(ValueError, match="thermal"):
        preflight.gate(SCENE_OK, HOME_OK, "the bus did not answer\n")


# ---------------------------------------------------------------- thermal, a servo nobody read
#
# Sep 7 night audit [HIGH], reproduced Sep 12 on the Sep 6 16:14 receipt (PING_OK above): take ONE
# joint's temperature out and gate() returned (True, []), the CLI printed "PREFLIGHT PASS (v2)" and
# exited 0. parse_joint_temps skipped every line it could not read, and the gate demanded a
# reading from the elbow alone -- the Sep 6 elbow-only bug ("the lab notebook (private)":2444), one layer
# down. An unread servo is not a cool one. ping_motors prints exactly two shapes for a servo it
# could not read, and the elbow's documented failure mode (two bus drops, tools/preflight.py:34) is the first of them:
#   "no response"  a servo off the bus                        (ping_motors.py:96-99)
#   "NoneC"        the temperature read raised; str(None)+"C"  (ping_motors.py:102-106, :114)
# The third shape, a missing line, is a truncated capture.


def _unread(ping: str, joint: str, shape: str) -> str:
    line = _joint_line(ping, joint)
    if shape == "no response":
        idx = re.search(r"\bid (\d+)", line).group(1)
        return ping.replace(line, f"  id {idx} {joint:15s} no response")
    if shape == "NoneC":
        return ping.replace(line, re.sub(r" \d+C", "NoneC", line))
    assert shape == "line removed"
    return ping.replace(line + "\n", "")


@pytest.mark.parametrize("shape", ["no response", "NoneC", "line removed"])
@pytest.mark.parametrize("joint", JOINTS)
def test_a_joint_whose_temperature_cannot_be_read_is_a_refusal_naming_it(joint, shape):
    ping = _unread(PING_OK, joint, shape)
    assert joint not in preflight.parse_joint_temps(ping), f"the {shape!r} fixture is still readable"
    ok, reasons = preflight.gate(SCENE_OK, HOME_OK, ping)
    thermal = [r for r in reasons if r.startswith("thermal:")]
    assert not ok and thermal, (
        f"{joint} ({shape}) gated {ok}, reasons {reasons} — every other servo on the Sep 6 receipt "
        f"is at 44-52C, so this is a PASS on a servo nobody read"
    )
    assert joint in thermal[0], f"the refusal does not name {joint}: {thermal}"
    named = [j for j in JOINTS if j != joint and re.search(rf"\b{j}\b", thermal[0])]
    assert not named, f"only {joint} was unread, but the refusal names {named}: {thermal}"


def test_an_unread_servo_stops_the_launcher_without_a_repair(tmp_path):
    """The refusal has to cross the wire: preflight's exit status, then the runner's `case "$OUT"`.

    The runner keys its repairs on the joint NAME (run_scored_trial.sh:109-121): a reason that
    contains "gripper" runs reseat_gripper.py, one with "elbow_flex" re-homes. Only a reason that
    also says "thermal" is short-circuited as human-only first (run_scored_trial.sh:102-104). A
    servo off the bus needs the power cycle (USB out -> 12V out -> 12V in -> USB in), not a jaw
    reseat and two more holds against the elbow's thermal budget. So run the real preflight on the
    receipt, hand its real stdout and status to the real loop, and count what the loop did."""
    loop = _section("GATED=0", "# --- launch")
    for joint in ("gripper", "elbow_flex"):
        rc, out = _preflight_cli(tmp_path, SCENE_OK, HOME_OK, _unread(PING_OK, joint, "no response"))
        assert rc == 1 and "ABORT" in out and joint in out, (
            f"{joint} did not answer and preflight.py printed {out!r}, exit {rc} — the runner "
            "branches on $? alone, so 0 launches a scored trial on a servo nobody read"
        )
        ran = _run_shell(tmp_path, loop, gate_out=out, gate_rc=str(rc))
        assert ran.rc == 3 and "REACHED_THE_END" not in ran.out, f"{joint} unread and it launched: {ran!r}"
        assert not [c for c in ran.calls if "reseat_gripper.py" in c], f"{joint} unread ran a reseat: {ran!r}"
        assert sum("home_arm.py" in c for c in ran.calls) == 1, (
            f"{joint} unread and the loop re-homed — an unreadable servo is a human-only gate: {ran!r}"
        )


def test_a_second_arms_reading_cannot_stand_in_for_the_followers():
    """`ping_motors.py` with no argument pings BOTH arms, follower first (ping_motors.py:127-137),
    and the leader's lines carry the same joint names. parse_joint_temps kept the LAST line per
    name, so reproduced Sep 12: a follower wrist at 68C read as the leader's 52C and gated green,
    and a follower elbow that had dropped off the bus borrowed the leader's elbow and gated green.
    Every runner pings `follower` only (run_scored_trial.sh:93, run_pi05_trial.sh:121,
    run_smolvla_trial.sh:213), so only a hand-made capture reaches this -- the gate still must not
    be the thing that trusts one."""
    leader = PING_OK.replace(
        "=== FOLLOWER (white, jaws)  /dev/tty.usbmodem5C4C1245641 ===",
        "=== LEADER (black, trigger)  /dev/tty.usbmodem5C821069931 ===",
    )
    wrist = _joint_line(PING_OK, "wrist_flex")
    hot_follower = PING_OK.replace(wrist, wrist.replace(" 52C", " 68C"))
    dropped_follower = _unread(PING_OK, "elbow_flex", "no response")
    for follower in (hot_follower, dropped_follower):
        with pytest.raises(ValueError, match="thermal"):
            preflight.gate(SCENE_OK, HOME_OK, follower + leader)
    assert not preflight.gate(SCENE_OK, HOME_OK, hot_follower)[0]      # ...and alone, each refuses
    assert not preflight.gate(SCENE_OK, HOME_OK, dropped_follower)[0]


def test_the_required_servos_are_the_ones_ping_motors_walks():
    """ping_motors prints one line per entry of _arms.JOINTS (ping_motors.py:32, :88). If the gate
    demanded a shorter list, a servo outside it would be the unread one that passes. Read with ast:
    _arms imports lerobot's bus at module top (_arms.py:10-11), which a gate test does not need."""
    tree = ast.parse((TOOLS / "_arms.py").read_text())
    [joints] = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "JOINTS" for t in node.targets)
    ]
    assert tuple(joints) == JOINTS == tuple(preflight.SERVO_JOINTS)


# ---------------------------------------------------------------- home, a joint nobody read


@pytest.mark.parametrize("joint", JOINTS)
def test_a_settled_line_missing_a_joint_is_a_refusal_naming_it(joint):
    """The pose gate had the same shape. home_gate.check_pose judges the joints the settled line
    HAS (home_gate.py:203-205), so reproduced Sep 12: a settled line with no elbow_flex in it, on
    the Sep 6 receipt, gated (True, []). home_arm prints all six joints or no settled line at all
    (home_arm.py:249-254, inside contextlib.suppress), so today only a truncated or hand-edited
    capture reaches this. shoulder_lift is exempt from BOUNDS at frame 0 (home_gate.py:157), not
    from being read: at the carry pose it is the joint that defines "lifted" (home_gate.py:163)."""
    pose = preflight.parse_settled_pose(HOME_OK)
    del pose[joint]
    ok, reasons = preflight.gate(SCENE_OK, "settled: " + repr(pose), PING_OK)
    home = [r for r in reasons if r.startswith("home:")]
    assert not ok and home, f"a settled line without {joint} gated {ok}, reasons {reasons}"
    assert joint in home[0], f"the refusal does not name {joint}: {home}"
    named = [j for j in JOINTS if j != joint and re.search(rf"\b{j}\b", home[0])]
    assert not named, f"only {joint} was missing, but the refusal names {named}: {home}"


# ---------------------------------------------------------------------------
# S1 ROT-45 placement target (Sep 14 2026, protocol §7 15:26). The PI redefined S1 as the tube rotated
# 45° about its MIDPOINT, tip toward the rack, so the cap LEAVES the D mark: it sits at (382, 320)
# reference px, where Faith placed it at 15:08. The cap-on-the-mark gate would refuse every valid S1
# start (42 px off) and pass every wrong one (a horizontal tube on the mark -- which is exactly what
# P02 trial 1 ran on at 15:03, unnoticed by any gate). The target is selected by NAME through
# CAPSTONE_TUBE_TARGET, exported by run_pair.sh for S1 pairs; the constant lives in preflight.
# ---------------------------------------------------------------------------
SCENE_S1_OK = SCENE_OK.replace("live=( 357.7, 356.6)  delta=( -2.3, +0.7) px  |d|=2.4",
                               "live=( 384.1, 318.9)  delta=(+24.1,-36.9) px  |d|=44.1")


def test_s1_target_is_the_pi_placement():
    assert preflight.TUBE_TARGETS["S1-ROT45"] == (382.0, 320.0)


def test_parse_tube_live_reads_the_live_centroid():
    assert preflight.parse_tube_live(SCENE_OK) == pytest.approx((357.7, 356.6))
    with pytest.raises(ValueError):
        preflight.parse_tube_live("tube (blue cap)   : NOT FOUND (ref=True, live=False)")


def test_s1_gate_passes_the_rotated_start_and_refuses_a_tube_on_the_mark():
    ok, reasons = preflight.gate(SCENE_S1_OK, HOME_OK, PING_OK, tube_target=preflight.TUBE_TARGETS["S1-ROT45"])
    assert ok, reasons                                   # 2.4 px from the S1 target
    ok, reasons = preflight.gate(SCENE_OK, HOME_OK, PING_OK, tube_target=preflight.TUBE_TARGETS["S1-ROT45"])
    assert not ok and any("placement" in r and "S1" in r for r in reasons), reasons   # the S0 start, 43.9 px away
    # and without a target the S0 rule is unchanged: the rotated start is 44.1 px off the mark
    ok, reasons = preflight.gate(SCENE_S1_OK, HOME_OK, PING_OK)
    assert not ok and any("placement" in r for r in reasons)


def test_cli_selects_the_target_by_name_from_the_environment(tmp_path):
    import os
    import subprocess
    files = []
    for name, text in (("scene", SCENE_S1_OK), ("home", HOME_OK), ("ping", PING_OK)):
        p = tmp_path / f"run.{name}"; p.write_text(text); files.append(str(p))
    base = {k: v for k, v in os.environ.items() if not k.startswith("CAPSTONE")}
    def run(env_extra):
        r = subprocess.run([sys.executable, str(TOOLS / "preflight.py"), *files, "v2"], env={**base, **env_extra},
                           capture_output=True, text=True)
        return r.returncode, r.stdout.strip()
    rc, out = run({"CAPSTONE_TUBE_TARGET": "S1-ROT45"})
    assert rc == 0 and out.startswith("PREFLIGHT PASS"), out
    rc, out = run({})
    assert rc == 1 and "placement" in out, out          # same scene, no target: the S0 rule refuses it
    rc, out = run({"CAPSTONE_TUBE_TARGET": "S9-NOPE"})
    assert rc == 1 and "S9-NOPE" in out, out            # an unknown name aborts, never defaults to the mark
