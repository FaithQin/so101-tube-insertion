"""Regression: the pre-launch gate must REPAIR what it can, not just refuse.

Born from Aug 29, 23:20. The gate correctly caught the gripper settling at 2.00
(outside the v2 frame-0 training range [1.14, 1.64] — the jaws end ACT-B
episodes relieved, and home_arm does not correct it because 0.7 is inside its
flat 2.5 residual tolerance). It then announced "abort" three times and stopped,
having fixed nothing. Faith, correctly: "it's such a waste of time to just tell
me it doesn't work then not fix itself — then when does it get fixed?"

The split this file pins down:

  SOFTWARE-REPAIRABLE — retry with a repair, do not bother the operator:
    * gripper outside training support -> reseat the jaw (open, pause, close
      slowly), then re-home and re-gate.
    * elbow outside training support   -> re-home; it lands differently from
      different starting poses (74.1-85.3 measured across one session).

  HUMAN-ONLY — abort IMMEDIATELY, do not burn retries that cannot help:
    * tube placement  -> someone has to move the tube.
    * elbow too hot   -> someone has to point a fan at it.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from conftest import TOOLS

sys.path.insert(0, str(TOOLS))

import home_gate  # noqa: E402
import reseat_gripper  # noqa: E402

TRIAL_SH = (TOOLS / "run_scored_trial.sh").read_text()


# ---------------------------------------------- running the runner's own loop
#
# Sep 5 audit. Every assertion in this file used to be a substring scan of
# run_scored_trial.sh, and a mutation sweep walked through all of them: dropping
# the `break` after `GATED=1`, dropping the `break` from the human-only case,
# commenting out the reseat invocation, and swapping the placement/thermal
# spoken strings ALL left the full suite green. The tokens they grep for are in
# comments, in echoes, or simply elsewhere in the file.
#
# So the gate loop is lifted VERBATIM out of the runner and executed under zsh
# with a scripted preflight, the same technique test_runner_dataset_tags.py uses
# on the DATASET_TAG case block. Prose cannot satisfy a process exit code.

# preflight.py's real stdout, one line per verdict (preflight.py:122-124).
PREFLIGHT_PASS = "PREFLIGHT PASS (v2)"
ELBOW_BAD = "PREFLIGHT ABORT — home: elbow_flex=74.10 outside [77.53, 81.93]"
GRIPPER_BAD = "PREFLIGHT ABORT — home: gripper=8.00 outside [0.00, 3.00]"
PLACEMENT_BAD = "PREFLIGHT ABORT — placement: tube |d|=10.2 px > 8 px"
THERMAL_BAD = "PREFLIGHT ABORT — thermal: elbow temp 62C > 60C"
UNREADABLE = (
    "PREFLIGHT ABORT — could not evaluate a gate: no 'settled:' line in "
    "home_arm output — cannot gate the pose"
)

# A `python` that records which tool it was asked to run and replays a scripted
# preflight verdict. No serial port, no cameras, no arm.
_PY_STUB = """#!/bin/zsh
tool="${1:t}"
print -r -- "$tool" >> "$MUT_CALLS"
if [[ "$tool" == preflight.py ]]; then
  n=$(( $(<"$MUT_N") + 1 ))
  print -r -- "$n" > "$MUT_N"
  if [[ -f "$MUT_SEQ/$n.msg" ]]; then
    cat "$MUT_SEQ/$n.msg"
    exit "$(<"$MUT_SEQ/$n.rc")"
  fi
  print -r -- "PREFLIGHT ABORT — attempt $n was never scripted"
  exit 1
fi
exit 0
"""


def _gate_section() -> str:
    """The runner's retry loop and abort block, lifted verbatim.

    From `GATED=0` to the launch banner — i.e. lines 69-116 of the runner,
    including the human-only short-circuit, the repair branches and the spoken
    abort. Sliced rather than re-typed so the tests can never drift from the
    script they are grading.
    """
    start = TRIAL_SH.find("GATED=0")
    end = TRIAL_SH.find("# --- launch")
    assert 0 <= start < end, "cannot find the gate loop in run_scored_trial.sh"
    section = TRIAL_SH[start:end]
    assert "for attempt in" in section, "the gate loop no longer retries at all"
    return section


def _run_gate(outcomes):
    """Execute that loop for real, with preflight scripted attempt by attempt.

    `outcomes` is [(exit_code, stdout_line), ...] — one entry per preflight
    call, in order. An unscripted extra call is itself a failure signal: it
    means the loop ran an attempt the test did not expect.

    Returns the tools the loop ACTUALLY invoked, what it spoke, the final
    GATED value and the script's exit status.
    """
    tmp = Path(tempfile.mkdtemp(prefix="gate-"))
    try:
        seq = tmp / "seq"
        seq.mkdir()
        for i, (rc, msg) in enumerate(outcomes, 1):
            (seq / f"{i}.rc").write_text(str(rc))
            (seq / f"{i}.msg").write_text(msg + "\n")

        bin_dir = tmp / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "python"
        stub.write_text(_PY_STUB)
        stub.chmod(0o755)

        calls, spoken, count = tmp / "calls", tmp / "say", tmp / "n"
        calls.touch()
        spoken.touch()
        count.write_text("0")

        script = "\n".join([
            'say() { print -r -- "$*" >> "$MUT_SAY"; }',  # the runner's TTS, captured
            "set -u",                                      # as the runner runs it
            'B="$MUT_BIN"',
            'RUN="$MUT_TMP/run"',
            "DATASET_TAG=v2",
            "START_POSE=v2",       # the loop's start-pose inputs (Sep 6): frame 0, no close flag
            "HOME_FLAGS=()",
            _gate_section(),
            'print -r -- "MUT_GATED=$GATED"',              # only reached if it did not abort
        ])
        env = {
            **os.environ,
            "MUT_BIN": str(bin_dir), "MUT_TMP": str(tmp), "MUT_SEQ": str(seq),
            "MUT_CALLS": str(calls), "MUT_SAY": str(spoken), "MUT_N": str(count),
        }
        r = subprocess.run(["/bin/zsh", "-c", script], cwd=tmp, env=env,
                           capture_output=True, text=True, timeout=60)
        m = re.search(r"^MUT_GATED=(.*)$", r.stdout, re.M)
        return {
            "rc": r.returncode,
            "out": r.stdout,
            "gated": m.group(1) if m else None,
            "calls": calls.read_text().split(),
            "spoken": [ln for ln in spoken.read_text().splitlines() if ln],
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------- running the repair tool

# Where the jaw actually settles after an ACT-B episode (Aug 29). This is the
# pose reseat_gripper exists to get OUT of, so it is the number every assertion
# about the open stroke has to clear.
STUCK_AT = 2.00


class _FakeBus:
    """A Feetech bus that records instead of moving.

    `settles_at` is what Present_Position reads back once the close walk is
    done — the part the code cannot control and therefore the part worth
    varying. Torque state is tracked so the tests can ask what the process left
    the servo in, rather than which keyword argument is spelled in the source.
    """

    def __init__(self, settles_at: float, start_at: float = STUCK_AT):
        self.settles_at = settles_at
        self.start_at = start_at
        self.goals = []            # every Goal_Position commanded, in order
        self.writes = []           # (register, motor, value)
        self.torque_on = False
        self.torque_at_disconnect = None

    def enable_torque(self):
        self.torque_on = True

    def write(self, reg, motor, value, normalize=True):
        self.writes.append((reg, motor, value))
        if reg == "Goal_Position":
            self.goals.append(float(value))
        elif reg == "Torque_Enable":
            self.torque_on = bool(value)

    def read(self, reg, motor, normalize=True):
        # Stuck where the episode left it until something is commanded.
        return self.settles_at if self.goals else self.start_at

    def disconnect(self, disable_torque=True):
        self.torque_at_disconnect = self.torque_on and not disable_torque


class _NoSleep:
    """Stands in for `time` inside reseat_gripper only — the real close walk is
    ~2 s of sleeps per call and the suite runs before every bench session."""

    @staticmethod
    def sleep(_seconds):
        pass


def _run_reseat_main(settles_at: float):
    """Run reseat_gripper.main() against a fake bus. Returns (exit_code, bus)."""
    bus = _FakeBus(settles_at)
    real_open, real_time = reseat_gripper.open_bus, reseat_gripper.time
    reseat_gripper.open_bus = lambda *a, **k: bus
    reseat_gripper.time = _NoSleep
    try:
        return reseat_gripper.main(), bus
    finally:
        reseat_gripper.open_bus, reseat_gripper.time = real_open, real_time


# ------------------------------------------------- the repair tool itself


def test_gripper_reseat_tool_exists():
    assert (TOOLS / "reseat_gripper.py").exists(), (
        "no gripper reseat tool — the gate can detect the creep but not fix it"
    )


def test_reseat_targets_the_training_range_not_just_the_home_value():
    """Grade the VERDICT the tool returns, not the digits in the file.

    Sep 5 audit: this asserted `"1.14" in src and "1.64" in src`, and both
    numbers also appear in the module docstring ("[1.14, 1.64]"). So retuning
    the executable constants to v3's gripper frame-0 (0.79, 1.57) — the single
    most likely edit to this file now that v3 policies are what is on the bench,
    and home_gate.FRAME0 hands you the pair — left the prose intact, the suite
    green, and the tool grading its own repair against a distribution the trial
    is not running. A jaw settling at 0.85 would have printed RESEAT OK.
    """
    assert (reseat_gripper.SUPPORT_LO, reseat_gripper.SUPPORT_HI) == (1.14, 1.64), (
        f"reseat grades against [{reseat_gripper.SUPPORT_LO}, "
        f"{reseat_gripper.SUPPORT_HI}] — the v2 frame-0 gripper range is "
        f"[1.14, 1.64] and that is what the gate checks. v3's is "
        f"{home_gate.FRAME0['v3']['gripper'][2:]}; do not swap one in silently."
    )
    # ...and prove those constants are the ones the verdict is computed from.
    for settled, want, why in [
        (1.25, 0, "the v2 frame-0 mean must read OK"),
        (0.85, 1, "0.85 is inside v3's [0.79, 1.57] and outside v2's [1.14, 1.64]"),
        (1.60, 0, "1.60 is inside v2's [1.14, 1.64] and outside v3's [0.79, 1.57]"),
        (STUCK_AT, 1, "the unrepaired jaw must never read OK"),
    ]:
        rc, _ = _run_reseat_main(settled)
        assert rc == want, (
            f"gripper settled at {settled} and reseat exited {rc} (expected "
            f"{want}) — {why}. The support constants are not the ones being used."
        )


def test_reseat_opens_before_closing():
    """A jaw stuck open at 2.00 cannot be closed by commanding 1.3 — that is
    exactly what home_arm already does and it does not work. It has to be
    driven open first to unseat it, then walked back down.

    Sep 5 audit: this asserted `"OPEN_TO" in src and "STEP" in src`, which only
    proves two identifiers are spelled. Retuning OPEN_TO to 2.0 — plausible as a
    "do not slam the jaw wide" edit, since 2.0 clears home_gate's measured
    contact point — makes reseat command the exact position the jaw is already
    stuck at, i.e. a 1.5 s no-op that prints REPAIRED three times and fixes
    nothing, and it kept the suite green. So drive the function and look at the
    positions it actually commands.
    """
    bus = _FakeBus(settles_at=1.25)
    reseat_gripper.reseat(bus)

    assert bus.goals, "reseat commanded no Goal_Position at all — it moves nothing"
    assert bus.goals[0] > STUCK_AT and bus.goals[0] > home_gate.GRIPPER_CLOSED_MAX, (
        f"reseat's FIRST command is {bus.goals[0]}, and the jaw it is repairing "
        f"is already sitting at {STUCK_AT} (home_gate calls anything under "
        f"{home_gate.GRIPPER_CLOSED_MAX} still closed). It never unseats the jaw "
        f"— this is home_arm's no-op with extra steps. The OPEN stroke is what "
        f"does the work; OPEN_TO must clear where the jaw is stuck."
    )
    assert max(bus.goals) == bus.goals[0], "the open must come first, not mid-walk"
    assert bus.goals[-1] == pytest.approx(reseat_gripper.TARGET), (
        f"reseat finished at {bus.goals[-1]}, not TARGET={reseat_gripper.TARGET}"
    )
    assert all(0 < a - b <= reseat_gripper.STEP + 1e-9 for a, b in zip(bus.goals, bus.goals[1:])), (
        f"the close is not a slow monotone walk in STEP={reseat_gripper.STEP} "
        f"increments: {bus.goals}"
    )


def test_reseat_keeps_torque_on():
    """Training frame-0s were captured with the jaw held. A full release lets it
    drift back out of support — the Aug 16 elbow-sag bug in another costume.

    Sep 5 audit: this asserted `"disable_torque=False" in src`. Adding an
    explicit `bus.write("Torque_Enable", "gripper", 0)` one line above the
    disconnect — a very plausible "do not leave the gripper energised after a
    fault" edit given the HX-30HM Protection_Time history — leaves that keyword
    untouched and kept the suite green while the jaw was released the instant
    the repair finished. The failure is invisible because RESEAT OK is measured
    BEFORE the release. So ask the bus what state the process left it in.
    """
    rc, bus = _run_reseat_main(settles_at=1.25)
    assert rc == 0
    assert ("Torque_Enable", "gripper", 0) not in bus.writes, (
        "reseat_gripper explicitly releases the gripper before exiting — the jaw "
        "it just walked into support drifts straight back out, and 'RESEAT OK' "
        "was printed before the release"
    )
    assert bus.torque_at_disconnect is True, (
        "reseat_gripper exited with the gripper's torque OFF. Training frame-0s "
        "were captured with the jaw HELD; the next preflight will refuse the "
        "drifted jaw and the trial burns all three attempts having fixed nothing"
    )


# ------------------------------------------------- wiring into the gate


def test_trial_script_repairs_the_gripper_instead_of_only_aborting():
    """Sep 5 audit: this asserted `"reseat_gripper.py" in TRIAL_SH`, which any
    mention satisfies — a comment included. Commenting the invocation out (how
    this bug class actually lands: disabled while chasing a serial-port
    conflict, never restored) kept the suite green, left the `REPAIRED:` echo
    above it still printing, and aborted the trial having repaired nothing. So
    run the loop and count the invocation.
    """
    r = _run_gate([(1, GRIPPER_BAD), (0, PREFLIGHT_PASS)])
    assert r["calls"].count("reseat_gripper.py") == 1, (
        f"the gripper gate failed and tools/reseat_gripper.py was never RUN — "
        f"the loop invoked {r['calls']}. The trial script detects the creep, "
        f"announces a repair, and repairs nothing."
    )
    assert r["gated"] == "1" and r["rc"] == 0, (
        f"the repaired arm did not go on to launch (GATED={r['gated']}, "
        f"rc={r['rc']})"
    )
    # the repair belongs to the gripper gate, not to every failure
    r = _run_gate([(1, ELBOW_BAD), (0, PREFLIGHT_PASS)])
    assert "reseat_gripper.py" not in r["calls"], (
        "an elbow failure reseated the gripper — the repair branches are crossed"
    )


def test_trial_script_retries_after_a_repair():
    """Sep 5 audit: this pinned the loop HEADER text (`for attempt in 1 2 3`).
    Deleting the `break` after `GATED=1` kept it green, and turns the gate into
    an advisory print: all three attempts run, GATED is set on the first pass
    and never cleared, so a trial whose attempt-1 gate passed but whose
    attempt-3 gate FAILED still falls through and records a SCORED row in
    trials.csv. That is verbatim the Aug 29 "narrated three times and launched
    through" failure preflight.py was written to end. Run the loop and count.
    """
    # a repairable failure is retried, and the attempt that passes ends it
    r = _run_gate([(1, GRIPPER_BAD), (1, GRIPPER_BAD), (0, PREFLIGHT_PASS)])
    assert r["calls"].count("preflight.py") == 3
    assert r["gated"] == "1" and r["rc"] == 0, "a repaired arm must still launch"

    # the loop must STOP the moment the gate passes. Nothing clears GATED once
    # it is set, so an attempt that runs after a pass can only ever be a gate
    # whose verdict is then ignored.
    r = _run_gate([(0, PREFLIGHT_PASS), (1, ELBOW_BAD), (1, THERMAL_BAD)])
    assert r["calls"].count("preflight.py") == 1, (
        f"attempt 1 gated CLEAN and the loop kept going — preflight ran "
        f"{r['calls'].count('preflight.py')}x. GATED is never cleared, so the "
        f"later failing attempts (elbow out of support, elbow over 60 C) are "
        f"gathered and discarded, and the trial launches anyway. The `break` "
        f"after `GATED=1` is what makes this a gate instead of a print."
    )
    assert r["calls"].count("home_arm.py") == 1, (
        f"home_arm ran {r['calls'].count('home_arm.py')}x after the gate had "
        f"already passed — every extra cycle is ~5 s of elbow-hold thermal "
        f"budget against 'home right before the run, never park it holding'"
    )
    assert r["rc"] == 0

    # three failures give up rather than looping forever or launching
    r = _run_gate([(1, ELBOW_BAD)] * 3)
    assert r["calls"].count("preflight.py") == 3
    assert r["gated"] != "1" and r["rc"] == 3, (
        f"three failed gates did not abort the trial (GATED={r['gated']}, "
        f"rc={r['rc']})"
    )


def test_human_only_gates_abort_immediately():
    """Retrying a home three times cannot move a tube or cool a servo. Burning
    ~15 s of homing to re-announce a fact the operator already heard is the
    same waste, in the other direction.

    Sep 5 audit: this asserted `"HUMAN_ONLY" in TRIAL_SH` — a token that appears
    ONLY in the comment above the branch. The behaviour is the `break` inside
    `case "$OUT" in *placement*|*thermal*)`, and `;;` ends the case, not the
    loop: dropping that break kept the suite green while a thermal abort ran two
    EXTRA home_arm cycles, driving and holding the elbow preflight had just
    measured above 60 C. Narrowing the pattern to `*placement*)` alone — so only
    thermal loses the short-circuit — was green too, because the assertion
    cannot see the pattern list at all. Execute the loop, one reason at a time.
    """
    for reason in (PLACEMENT_BAD, THERMAL_BAD):
        r = _run_gate([(1, reason)] * 3)
        assert r["calls"].count("preflight.py") == 1, (
            f"a human-only gate burned {r['calls'].count('preflight.py')} "
            f"attempts on {reason!r} — no amount of re-homing moves a tube or "
            f"cools a servo, and `;;` ends the case, not the loop"
        )
        assert r["calls"].count("home_arm.py") == 1, (
            f"{reason!r} triggered {r['calls'].count('home_arm.py')} home_arm "
            f"cycles. On a thermal abort those extra cycles drive and HOLD the "
            f"elbow that preflight just measured over 60 C — the gate that "
            f"exists to protect the arm's tracked reliability risk is cooking it"
        )
        assert r["rc"] == 3

    # ...and the short-circuit must stay specific: a repairable gate still retries
    r = _run_gate([(1, GRIPPER_BAD)] * 3)
    assert r["calls"].count("preflight.py") == 3, (
        "a software-repairable gate stopped short-circuiting too — the "
        "human-only pattern list has swallowed the retryable cases"
    )


def test_spoken_reason_names_the_failing_gate():
    """Sep 5 audit: this asserted the three phrases appear somewhere in the file,
    never that each is bound to the gate it names. Swapping the SPOKEN strings
    between the adjacent `*placement*)` and `*thermal*)` branches — the same
    shape as the confirmed 'v3' -> TRAIN_HOME_V2 mapping bug — leaves all three
    phrases in the file and kept the suite green while every spoken abort named
    the wrong cause. The operator is across the room acting on the voice line
    alone; that is the stated design intent. So make it speak, and listen.
    """
    for reason, want in [
        (ELBOW_BAD, "abort. elbow out of range."),
        (GRIPPER_BAD, "abort. gripper out of range."),
        (PLACEMENT_BAD, "abort. tube placement."),
        (THERMAL_BAD, "abort. elbow too hot."),
        (UNREADABLE, "abort. a check could not be read."),
    ]:
        r = _run_gate([(1, reason)] * 3)
        assert r["spoken"] == [want], (
            f"preflight reported {reason!r} and the arm announced "
            f"{r['spoken']} — expected [{want!r}]. The operator acts on the "
            f"voice line from across the room, so a crossed branch sends them "
            f"to cool a servo that is fine while the tube stays where it drifted."
        )
    assert 'say "abort. preflight."' not in TRIAL_SH


def test_repair_is_reported_so_it_never_happens_silently():
    """A silent repair is how a drifting arm stops being visible. Aug 29's whole
    lesson was that invisible conditions get launched through.

    Sep 5 audit: `"REPAIRED" in TRIAL_SH` is satisfied by the echo alone, so
    commenting out the reseat invocation INVERTED this test — it went on
    certifying a scored_logs line that positively asserts a repair which never
    happened, which is strictly worse than a missing one. Report and repair have
    to stand or fall together.
    """
    repaired = _run_gate([(1, GRIPPER_BAD), (0, PREFLIGHT_PASS)])
    assert any("REPAIRED" in ln for ln in repaired["out"].splitlines()), (
        f"the gripper was repaired and the log never says so:\n{repaired['out']}"
    )
    assert "reseat_gripper.py" in repaired["calls"], (
        f"the log claims 'REPAIRED' but no repair tool ran — the "
        f"${{LABEL}}_${{STAMP}} entry lies to whoever reads it next. "
        f"Tools invoked: {repaired['calls']}"
    )

    # ...and a gate nothing can repair must not claim one
    aborted = _run_gate([(1, THERMAL_BAD)] * 3)
    assert "REPAIRED" not in aborted["out"], (
        f"a thermal abort reported a repair:\n{aborted['out']}"
    )
