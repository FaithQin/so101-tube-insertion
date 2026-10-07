"""`tools/run_cert.sh` -- the certification replay as ONE gated command, run under stubs.

Sep 14 2026. G-R needs a replay to SEAT before every block and after any failed episode (protocol
:245, :291). By hand this was six chat round trips per pass, and the morning's one certification
miss was the pass that skipped the scene check. This runs the sequence in order and REFUSES at
each gate, in code, before the arm is torqued:

  1. ping -> cert_gate temps (<= 52 C, every servo read)           else exit 3, no motion
  2. home_arm --pose v4 -> cert_gate home (HOMED, settled)         else exit 3, arm relieved
  3. scene_check -> preflight (tube <= 8 px, pose in v4 support)   else exit 3, no replay
  4. replay_telemetry --episode 0 --out <label>_<stamp>_telemetry.csv   non-zero -> exit 5
  5. cert_summary line
  6. home_arm (retract) + relax_arm -- ALWAYS after any motion. The first version prompted for the
     verdict here and held torque while it waited: CERT-B3-02 (14:47) sat at the rack three minutes,
     wrist_flex 45 -> 51 C. And a second terminal to type into is not how the bench is driven: the
     operator calls the seat to the chat, as on Sep 6 and this morning.
  7. print the one command that records the operator's verdict:
        python tools/cert_gate.py record <run> SEAT|MISS     -> <run>.seat
  exit 0 when the replay completed (verdict pending), 3 gate abort, 5 replay failed.

The stub `python` records every argv and answers per script; `say` is shadowed.
"""
import os
import re
import subprocess
import sys
import tempfile

import pytest

from conftest import TOOLS

SCRIPT = "run_cert.sh"

_STUB_PY = r"""#!/bin/zsh
{ print -rn -- "CALL"; for a in "$@"; do print -rn -- $'\t'"$a"; done; print -r -- ""; } >> "$STUB_CALLS"
case "$*" in
  *ping_motors.py*)   print -r -- "  id 3 elbow_flex      pos 3031  load 0  cur 0   41C  126  torque 1  ok"; exit 0 ;;
  *"cert_gate.py temps"*) print -r -- "${STUB_TEMPS_MSG:-TEMPS OK}"; exit "${STUB_TEMPS_RC:-0}" ;;
  *"cert_gate.py home"*)
      n=$(( $(cat "$STUB_HOME_N" 2>/dev/null || echo 0) + 1 )); print -r -- "$n" > "$STUB_HOME_N"
      if [ "$n" -le "${STUB_HOME_FAIL_TIMES:-0}" ]; then print -r -- "CERT GATE REFUSED — home: elbow 86.4 outside 85.6 +- 0.5"; exit 1; fi
      print -r -- "HOME OK"; exit 0 ;;
  *"cert_gate.py receipt"*) print -r -- "label=$4"; exit 0 ;;
  *home_arm.py*)      print -r -- "HOMED (worst adjusted residual 1.2)"; print -r -- "settled: {'elbow_flex': 85.6}"; exit 0 ;;
  *scene_check.py*)   print -r -- "tube (blue cap)   : ref=( 360.0, 355.8)  live=( 359.1, 354.3)  delta=( -1.0, -1.6) px  |d|=1.8"; exit 0 ;;
  *preflight.py*)     print -r -- "${STUB_PREFLIGHT_MSG:-PREFLIGHT PASS (v4)}"; exit "${STUB_PREFLIGHT_RC:-0}" ;;
  *replay_telemetry.py*)
      out=""; prev=""; for a in "$@"; do [ "$prev" = "--out" ] && out="$a"; prev="$a"; done
      [ -n "$out" ] && print -r -- "frame_index,t_mono" > "$out"
      print -r -- "${STUB_REPLAY_MSG:-BASELINE OK: 203 rows}"; exit "${STUB_REPLAY_RC:-0}" ;;
  *cert_summary.py*)  print -r -- "CERT-T  loop 19.99 Hz  ee_proxy 29.93  jaw_min 13.42 (held)"; exit 0 ;;
  *relax_arm.py*)     print -r -- "RELAXED"; exit 0 ;;
esac
exit 0
"""
_STUB_SAY = '#!/bin/zsh\nprint -r -- "SAY $*" >> "$STUB_CALLS"\nexit 0\n'


class Run:
    def __init__(self, rc, out, calls_text, work):
        self.rc, self.out, self.work = rc, out, work
        self.calls = [ln.split("\t")[1:] for ln in calls_text.splitlines() if ln.startswith("CALL")]
        self.spoken = [ln[4:] for ln in calls_text.splitlines() if ln.startswith("SAY ")]

    def scripts(self):
        """The tool each python call ran, in order (plus the cert_gate subcommand)."""
        seq = []
        for c in self.calls:
            tool = next((os.path.basename(a) for a in c if a.endswith(".py")), None)
            if tool == "cert_gate.py":
                tool += ":" + c[c.index(next(a for a in c if a.endswith("cert_gate.py"))) + 1]
            if tool:
                seq.append(tool)
        return seq

    def receipts(self, ext):
        d = os.path.join(self.work, "tools", "scored_logs")
        return sorted(f for f in os.listdir(d) if f.endswith(ext)) if os.path.isdir(d) else []


def run_cert(label="CERT-T", env_extra=None, timeout=20) -> Run:
    src = (TOOLS / SCRIPT).read_text()
    work = tempfile.mkdtemp(prefix="run_cert_")
    bin_dir = os.path.join(work, "bin"); os.makedirs(bin_dir)
    os.makedirs(os.path.join(work, "tools", "scored_logs"))
    for name, body in (("python", _STUB_PY), ("say", _STUB_SAY)):
        p = os.path.join(bin_dir, name)
        with open(p, "w") as f:
            f.write(body)
        os.chmod(p, 0o755)
    patched, n = re.subn(r"(?m)^B=\S+$", "B=" + bin_dir, src)
    assert n == 1, f"{SCRIPT} must have exactly one `B=<bindir>` line"
    script = os.path.join(work, "tools", SCRIPT)
    with open(script, "w") as f:
        f.write(patched)
    os.chmod(script, 0o755)
    calls = os.path.join(work, "calls.log"); open(calls, "w").close()
    env = {k: v for k, v in os.environ.items() if not k.startswith("CAPSTONE")}
    env.update({"STUB_CALLS": calls, "STUB_HOME_N": os.path.join(work, "home_n"), "PATH": bin_dir + ":" + env.get("PATH", "")})
    env.update(env_extra or {})
    proc = subprocess.run(["/bin/zsh", script, label], cwd=work, env=env, stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    return Run(proc.returncode, proc.stdout, open(calls).read(), work)


def test_runs_every_step_in_order_relieves_and_exits_0_with_the_record_command():
    r = run_cert()
    assert r.rc == 0, r.out
    assert r.scripts() == ["ping_motors.py", "cert_gate.py:temps", "home_arm.py", "cert_gate.py:home",
                           "scene_check.py", "preflight.py", "replay_telemetry.py", "cert_summary.py",
                           "home_arm.py", "relax_arm.py"], r.out
    assert r.receipts("_telemetry.csv") and r.receipts(".ping") and r.receipts(".scene") and r.receipts(".home")
    assert not r.receipts(".seat"), "the verdict is the operator's, recorded afterwards"
    assert re.search(r"cert_gate\.py record \S+CERT-T_\d{8}_\d{6} SEAT\|MISS", r.out), r.out
    assert "SEAT or MISS?" not in r.out, "no prompt: the operator calls the seat to the chat"


def test_hot_servo_refuses_before_any_motion():
    r = run_cert(env_extra={"STUB_TEMPS_RC": "1", "STUB_TEMPS_MSG": "TEMPS REFUSED: wrist_flex 57C > 52C"})
    assert r.rc == 3, r.out
    assert "home_arm.py" not in r.scripts() and "replay_telemetry.py" not in r.scripts()
    assert "wrist_flex 57C" in r.out


def test_bad_scene_refuses_before_the_replay_and_relieves():
    r = run_cert(env_extra={"STUB_PREFLIGHT_RC": "1", "STUB_PREFLIGHT_MSG": "PREFLIGHT ABORT — placement: tube |d|=10.2 px > 8 px"})
    assert r.rc == 3, r.out
    assert "replay_telemetry.py" not in r.scripts() and "10.2 px" in r.out
    assert r.scripts()[-2:] == ["home_arm.py", "relax_arm.py"], "a refused pass still relieves the torqued arm"


def test_failed_replay_exits_5_and_relieves():
    r = run_cert(env_extra={"STUB_REPLAY_RC": "5", "STUB_REPLAY_MSG": "REPLAY ABORTED (STALL): elbow_flex pushed"})
    assert r.rc == 5, r.out
    assert "cert_summary.py" not in r.scripts() and "record" not in r.out
    assert r.scripts()[-2:] == ["home_arm.py", "relax_arm.py"]


# Sep 17 2026 07:25: the certification re-homes until the elbow settles inside cert_gate's band.
def test_rehomes_until_the_elbow_band_passes_then_replays():
    r = run_cert(env_extra={"STUB_HOME_FAIL_TIMES": "2"})
    assert r.rc == 0, r.out
    homes_before_replay = [x for x in r.scripts()[:r.scripts().index("replay_telemetry.py")] if x == "home_arm.py"]
    assert len(homes_before_replay) == 3, r.scripts()          # two refused, third passed
    assert r.scripts().count("cert_gate.py:home") == 3


def test_gives_up_after_the_home_budget_and_relieves():
    r = run_cert(env_extra={"STUB_HOME_FAIL_TIMES": "99"})
    assert r.rc == 3, r.out
    assert "replay_telemetry.py" not in r.scripts()
    assert r.scripts().count("cert_gate.py:home") == 5, r.scripts()
    assert r.scripts()[-1] == "relax_arm.py"
