"""The PRE-GRASPED start (S2) reaches the bench through ONE switch, `CAPSTONE_START_POSE=v4-carry`,
and every runner then homes, closes, gates, re-homes and reports against the CARRY tables.

Handoff Sep 6 §3 item 2 / §6 S2: "tube starts in the jaws at the `v4-carry` pose; a modified
task, labelled; every trial reaches contact". Sep 6 morning: the pose tables landed
(tests/test_v4_carry.py) but no runner could select them -- `home_arm.py --pose $DATASET_TAG`,
`preflight.py ... $DATASET_TAG`, `CAPSTONE_HOME_POSE=$DATASET_TAG`, all keyed on the policy's
generation. A pre-grasped trial would have homed to frame 0 with the tube in a closed jaw.

What the switch must do, RUN under stubs (the three runners are copied into a scratch dir with
their `B=<bin>` line repointed at a stub `python` that records every argv and the CAPSTONE
environment, and `say` shadowed):

  1. Default (unset): nothing changes -- frame-0 home, no --close-on-tube, preflight on the
     generation tag, CAPSTONE_HOME_POSE = the tag, `start_pose=<tag>` in the receipt.
  2. `v4-carry`: the scene is still photographed from FRAME 0 (arm clear, rack visible), then
     the gate loop homes with `--pose v4-carry --close-on-tube`, preflight is handed
     `v4-carry`, the wrapper is told CAPSTONE_HOME_POSE=v4-carry, the receipt says so.
  3. Anything else aborts before the arm is touched (exit 2, no home, no launch).
  4. Under a carry start the gripper "repair" (reseat the jaw on nothing) must NOT run -- the
     jaw is holding the tube; the loop re-homes and re-prompts instead.

preflight for a carry dataset gates the RACK, not the tube: the tube is in the jaws, so its
placement mark says nothing, and the scene is judged from the frame-0 pose the runner
photographs before the carry home. A rack the check cannot see is an abort, not a pass.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

import pytest

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))

import home_gate  # noqa: E402
import preflight  # noqa: E402
import v4_carry  # noqa: E402

TAG = "v4-carry"
TUBE_LOAD = 118.0
FREE_AIR_LOAD = 70.0

RUNNERS = {
    "act": ("run_scored_trial.sh", ["faithqin/act-tube-A-v4", "25", "off", "SP-TEST"], {}),
    "smolvla": ("run_smolvla_trial.sh", ["B", "SP-TEST", "50"], {"CAPSTONE_CHECKPOINT": "005000"}),
    "pi05": ("run_pi05_trial.sh", ["A", "SP-TEST", "50"], {"CAPSTONE_CHECKPOINT": "020000"}),
}

_STUB_PY = r"""#!/bin/zsh
{
  print -rn -- "CALL"
  for a in "$@"; do print -rn -- $'\t'"$a"; done
  print -r -- $'\t'"HOME_POSE=${CAPSTONE_HOME_POSE-__UNSET__}"
} >> "$STUB_CALLS"
case "$*" in
  *preflight.py*)
    n=$(( $(<"$STUB_N") + 1 )); print -r -- "$n" > "$STUB_N"
    parts=("${(@s:;:)STUB_PREFLIGHT_SEQ}")
    entry="${parts[$n]:-1:PREFLIGHT ABORT — attempt $n was never scripted}"
    print -r -- "${entry#*:}"; exit "${entry%%:*}" ;;
  *"--print gen"*) print -r -- "v4" ;;
  *"--print repo"*"--side B"*) print -r -- "faithqin/stub-tube-B-v4" ;;
  *"--print repo"*) print -r -- "faithqin/stub-tube-A-v4" ;;
  *"--print force"*"--side B"*) print -r -- "1" ;;
  *"--print force"*) print -r -- "0" ;;
  *"--print state_dim"*"--side B"*) print -r -- "18" ;;
  *"--print state_dim"*) print -r -- "6" ;;
  *"--print snapshot"*) print -r -- "$STUB_DIR" ;;
  *"--print checkpoint"*) print -r -- "005000" ;;
  *"--print task"*) print -r -- "Pick up the test tube and insert it into the rack" ;;
  *materialize_local_checkpoint*) print -r -- "$STUB_DIR" ;;
  *rollout_30hz_stale_ok.py*) print -r -- "LAUNCHED" >> "$STUB_CALLS"; print -r -- "[loop] loop_hz=nan frames=0 wall_s=0.00"; print -r -- "[loop] loop_hz=19.73 frames=890 wall_s=45.06" ;;
esac
exit 0
"""
_STUB_SAY = '#!/bin/zsh\nprint -r -- "SAY $*" >> "$STUB_CALLS"\nexit 0\n'


class Run:
    def __init__(self, rc, stdout, calls, receipt):
        self.rc, self.stdout, self.receipt = rc, stdout, receipt
        self.calls = [ln.split("\t")[1:] for ln in calls.splitlines() if ln.startswith("CALL")]
        self.spoken = [ln[4:] for ln in calls.splitlines() if ln.startswith("SAY ")]
        self.launched = "LAUNCHED" in calls.splitlines()

    def to(self, tool):
        return [c for c in self.calls if c and c[0].endswith(tool)]

    def launch_home_pose(self):
        hits = [c for c in self.calls if c and c[0].endswith("rollout_30hz_stale_ok.py")]
        assert len(hits) == 1, f"expected one rollout launch, saw {len(hits)}"
        return [x for x in hits[0] if x.startswith("HOME_POSE=")][0].split("=", 1)[1]


_CACHE: dict = {}


def run_runner(family: str, env_extra=None, preflight_seq="0:PREFLIGHT PASS (stub)") -> Run:
    key = (family, tuple(sorted((env_extra or {}).items())), preflight_seq)
    if key in _CACHE:
        return _CACHE[key]
    script_name, argv, base_env = RUNNERS[family]
    src = (TOOLS / script_name).read_text()
    work = tempfile.mkdtemp(prefix="start_pose_")
    try:
        bin_dir = os.path.join(work, "bin")
        os.makedirs(bin_dir)
        os.makedirs(os.path.join(work, "tools"))
        for name, body in (("python", _STUB_PY), ("say", _STUB_SAY)):
            with open(os.path.join(bin_dir, name), "w") as f:
                f.write(body)
            os.chmod(os.path.join(bin_dir, name), 0o755)
        patched, n = re.subn(r"(?m)^B=\S+$", lambda m: "B=" + bin_dir, src)
        assert n == 1, f"{script_name} no longer has a single `B=<bindir>` line"
        script = os.path.join(work, "tools", script_name)
        with open(script, "w") as f:
            f.write(patched)
        os.chmod(script, 0o755)
        calls = os.path.join(work, "calls.log")
        counter = os.path.join(work, "n")
        open(calls, "w").close()
        with open(counter, "w") as f:
            f.write("0")
        env = {k: v for k, v in os.environ.items() if not k.startswith("CAPSTONE")}
        env.update(base_env)
        env.update(env_extra or {})
        env.update({
            "PATH": bin_dir + os.pathsep + os.environ["PATH"],
            "STUB_CALLS": calls, "STUB_N": counter, "STUB_DIR": os.path.join(work, "snap"),
            "STUB_PREFLIGHT_SEQ": preflight_seq,
            "CAPSTONE_ALLOW_WORKTREE": "1",
        })
        r = subprocess.run(["/bin/zsh", script, *argv], capture_output=True, text=True, timeout=180, env=env, cwd=work)
        with open(calls) as f:
            log = f.read()
        receipts = [p for p in os.listdir(os.path.join(work, "tools", "scored_logs"))
                    if p.endswith(".verify")] if os.path.isdir(os.path.join(work, "tools", "scored_logs")) else []
        receipt = open(os.path.join(work, "tools", "scored_logs", receipts[0])).read() if receipts else ""
        result = Run(r.returncode, r.stdout + r.stderr, log, receipt)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    _CACHE[key] = result
    return result


# ---------------------------------------------------------------------------
# 1. the three runners, executed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("family", sorted(RUNNERS))
def test_default_start_is_frame0_and_says_so_in_the_receipt(family):
    r = run_runner(family)
    assert r.launched, f"{family}: did not launch on a passing gate (rc {r.rc})\n{r.stdout[-800:]}"
    homes = r.to("home_arm.py")
    assert homes and all("--close-on-tube" not in c for c in homes), homes
    assert all(c[c.index("--pose") + 1] == "v4" for c in homes), homes
    pre = r.to("preflight.py")
    assert pre and all(c[-2] == "v4" for c in pre), pre     # the dataset arg precedes the HOME_POSE record
    assert r.launch_home_pose() == "v4"
    assert re.search(r"^start_pose=v4$", r.receipt, re.M), r.receipt


@pytest.mark.parametrize("family", sorted(RUNNERS))
def test_carry_start_homes_closes_gates_and_rehomes_on_the_carry_tables(family):
    r = run_runner(family, {"CAPSTONE_START_POSE": TAG})
    assert r.launched, f"{family}: carry start did not launch (rc {r.rc})\n{r.stdout[-800:]}"
    homes = r.to("home_arm.py")
    assert len(homes) >= 2, homes
    first, loop = homes[0], homes[1:]
    # The scene is photographed from FRAME 0 -- arm clear of the rack, jaws empty -- so the
    # very first home is the frame-0 pose without the close...
    assert first[first.index("--pose") + 1] == "v4" and "--close-on-tube" not in first, first
    # ...and every gate-loop home is the carry pose WITH the scripted close.
    for c in loop:
        assert c[c.index("--pose") + 1] == TAG and "--close-on-tube" in c, c
    pre = r.to("preflight.py")
    assert pre and all(c[-2] == TAG for c in pre), pre
    assert r.launch_home_pose() == TAG
    assert re.search(rf"^start_pose={TAG}$", r.receipt, re.M), r.receipt
    assert "MODIFIED TASK" in r.stdout


@pytest.mark.parametrize("family", sorted(RUNNERS))
@pytest.mark.parametrize("bad", ["v4-hover", "v3-carry", "carry"])
def test_an_unknown_start_pose_aborts_before_the_arm_is_touched(family, bad):
    r = run_runner(family, {"CAPSTONE_START_POSE": bad})
    assert r.rc == 2, f"{family}: CAPSTONE_START_POSE={bad!r} exited {r.rc}\n{r.stdout[-600:]}"
    assert not r.to("home_arm.py") and not r.launched
    assert any("start pose" in s for s in r.spoken), r.spoken


@pytest.mark.parametrize("family", sorted(RUNNERS))
def test_carry_start_never_reseats_the_jaw_on_nothing(family):
    seq = "1:PREFLIGHT ABORT — home: gripper=8.00 outside [12.13, 15.42];0:PREFLIGHT PASS (v4-carry)"
    carry = run_runner(family, {"CAPSTONE_START_POSE": TAG}, preflight_seq=seq)
    assert carry.launched and len(carry.to("preflight.py")) == 2, carry.stdout[-600:]
    assert not carry.to("reseat_gripper.py"), "a carry start opened and re-closed the jaw on nothing"
    plain = run_runner(family, preflight_seq=seq.replace("(v4-carry)", "(v4)"))
    assert plain.launched and plain.to("reseat_gripper.py"), "the frame-0 gripper repair went missing"


# ---------------------------------------------------------------------------
# 2. preflight: the rack, not the tube, for a carry dataset
# ---------------------------------------------------------------------------
SCENE_OK = """homography inliers: 342/400
tube (blue cap)   : ref=( 360.0, 355.8)  live=( 357.7, 356.6)  delta=( -2.3, +0.7) px  |d|=2.4
rack (orange funnel): ref=( 537.4, 481.0)  live=( 538.8, 481.1)  delta=( +1.5, +0.2) px  |d|=1.5
tube angle        : ref= +0.0°  live= +0.0°  delta= +0.0°
"""
SCENE_TUBE_GONE = SCENE_OK.replace(
    "tube (blue cap)   : ref=( 360.0, 355.8)  live=( 357.7, 356.6)  delta=( -2.3, +0.7) px  |d|=2.4",
    "tube (blue cap)   : NOT FOUND (ref=True, live=False) — check placement/lighting")
SCENE_RACK_OFF = SCENE_TUBE_GONE.replace("|d|=1.5", "|d|=11.0")
SCENE_RACK_GONE = SCENE_TUBE_GONE.replace(
    "rack (orange funnel): ref=( 537.4, 481.0)  live=( 538.8, 481.1)  delta=( +1.5, +0.2) px  |d|=1.5",
    "rack (orange funnel): NOT FOUND (ref=True, live=False) — check placement/lighting")
# All six joints, in the shape of a real receipt (tools/scored_logs/h83j2kp6_20260906_161402.ping).
# It was one elbow line until Sep 12, when preflight began refusing any joint it could not read
# a temperature for. Only the elbow carries 50C, so the ".replace('50C', '62C')" below still
# overheats exactly one joint.
PING_OK = (
    "  id 1 shoulder_pan    pos  2013  load    60  cur    1   44C  125  torque 1  ok\n"
    "  id 2 shoulder_lift   pos   779  load   -56  cur    0   47C  124  torque 0  ok\n"
    "  id 3 elbow_flex      pos  2948  load    80  cur    0   50C  126  torque 0  ok\n"
    "  id 4 wrist_flex      pos  2945  load    44  cur    1   48C  123  torque 1  ok\n"
    "  id 5 wrist_roll      pos  2039  load    72  cur    3   47C  124  torque 1  ok\n"
    "  id 6 gripper         pos  2063  load     0  cur    0   46C  125  torque 1  ok\n"
)


def _home_line(pose: dict) -> str:
    return "settled: " + repr({j: round(float(v), 1) for j, v in pose.items()})   # float(): numpy repr is not a literal


@pytest.fixture(scope="module")
def carry_home(v4_carry_stats):
    return _home_line(v4_carry_stats["mean"])


def test_parse_rack_delta():
    assert preflight.parse_rack_delta(SCENE_OK) == pytest.approx(1.5)
    with pytest.raises(ValueError, match="rack"):
        preflight.parse_rack_delta(SCENE_RACK_GONE)


def test_carry_dataset_gates_the_rack_and_ignores_the_tube(carry_home):
    assert TAG in home_gate.GRIPPER_ON_TUBE_DATASETS
    ok, reasons = preflight.gate(SCENE_TUBE_GONE, carry_home, PING_OK, dataset=TAG)
    assert ok, reasons
    ok, reasons = preflight.gate(SCENE_RACK_OFF, carry_home, PING_OK, dataset=TAG)
    assert not ok and any("rack" in r and "placement" in r for r in reasons), reasons


def test_carry_dataset_cannot_pass_without_seeing_the_rack(carry_home):
    with pytest.raises(ValueError, match="rack"):
        preflight.gate(SCENE_RACK_GONE, carry_home, PING_OK, dataset=TAG)


def test_a_missing_tube_never_borrows_the_racks_delta():
    """Found Sep 6 while writing the carry tests: parse_tube_delta used re.S, so on a
    'tube: NOT FOUND' line it ran on into the rack line and returned the RACK's 1.5 px --
    a missing tube gated green. The Aug 30 case had no rack line to fall into."""
    assert "|d|=2.4" not in SCENE_TUBE_GONE and "|d|=1.5" in SCENE_TUBE_GONE
    with pytest.raises(ValueError, match="tube"):
        preflight.parse_tube_delta(SCENE_TUBE_GONE)


def test_frame0_datasets_still_gate_the_tube(v4_frame0_stats):
    home = _home_line(v4_frame0_stats["mean"])
    ok, _ = preflight.gate(SCENE_OK, home, PING_OK, dataset="v4")
    assert ok
    with pytest.raises(ValueError, match="tube"):
        preflight.gate(SCENE_TUBE_GONE, home, PING_OK, dataset="v4")
    ok, reasons = preflight.gate(SCENE_OK.replace("|d|=2.4", "|d|=10.2"), home, PING_OK, dataset="v4")
    assert not ok and any("tube" in r for r in reasons)


def test_carry_gate_still_refuses_a_frame0_pose_and_a_hot_elbow(v4_frame0_stats, carry_home):
    ok, reasons = preflight.gate(SCENE_TUBE_GONE, _home_line(v4_frame0_stats["mean"]), PING_OK, dataset=TAG)
    assert not ok and any(r.startswith("home:") for r in reasons), reasons
    ok, reasons = preflight.gate(SCENE_TUBE_GONE, carry_home, PING_OK.replace("50C", "62C"), dataset=TAG)
    assert not ok and any("thermal" in r for r in reasons), reasons


# ---------------------------------------------------------------------------
# 3. the close on a re-home attempt: keep the tube, do not re-open over the rig
# ---------------------------------------------------------------------------
class FakeBus:
    def __init__(self, start_at, stalls_at, load, moves=True):
        self.pos, self.stalls_at, self.load, self.moves = start_at, stalls_at, load, moves
        self.goals = []

    def write(self, reg, motor, value, normalize=True):
        if reg == "Goal_Position":
            self.goals.append(float(value))
            if self.moves:
                self.pos = max(float(value), self.stalls_at) if value < self.pos else float(value)

    def read(self, reg, motor, normalize=True):
        if reg == "Present_Position":
            return self.pos
        return -self.load if abs(self.pos - self.stalls_at) < 1e-6 else -FREE_AIR_LOAD


def test_place_and_close_keeps_a_tube_it_is_already_holding(v4_carry_stats):
    """Attempt 2 of the gate loop re-homes with the tube in the jaws. Opening them there drops
    the tube onto the rig; the close must recognise a held tube and only verify it."""
    bus = FakeBus(start_at=13.0, stalls_at=13.0, load=TUBE_LOAD)
    prompts = []
    grip = v4_carry.place_and_close(bus, dict(v4_carry_stats["mean"]), prompt=prompts.append, sleep=lambda _s: None)
    assert bus.goals == [] and prompts == [], (bus.goals, prompts)
    assert grip.position == pytest.approx(13.0) and grip.load == pytest.approx(TUBE_LOAD)


def test_an_empty_jaw_parked_at_tube_width_is_not_mistaken_for_a_held_tube(v4_carry_stats):
    bus = FakeBus(start_at=13.2, stalls_at=13.0, load=TUBE_LOAD)   # free-air load until it closes on the tube
    prompts = []
    v4_carry.place_and_close(bus, dict(v4_carry_stats["mean"]), prompt=prompts.append, sleep=lambda _s: None)
    assert prompts and bus.goals and bus.goals[0] == pytest.approx(v4_carry.OPEN_FOR_PLACEMENT)


def test_is_holding_needs_both_the_band_and_the_load():
    assert v4_carry.is_holding(13.2, TUBE_LOAD)
    assert not v4_carry.is_holding(13.2, FREE_AIR_LOAD)
    assert not v4_carry.is_holding(1.1, TUBE_LOAD)
    assert not v4_carry.is_holding(20.0, TUBE_LOAD)


# ---------------------------------------------------------------------------
# 4. the prompt reaches the operator although home_arm's stdout is a file
# ---------------------------------------------------------------------------
def test_operator_prompt_writes_to_the_tty_speaks_and_reads_the_reply(tmp_path):
    out, inp = tmp_path / "tty.out", tmp_path / "tty.in"
    inp.write_text("\n")
    spoken = []
    reply = v4_carry.operator_prompt("Place the tube, then press ENTER.", out_path=str(out),
                                     in_path=str(inp), speak=spoken.append)
    assert "Place the tube" in out.read_text()
    assert reply == "\n"
    assert spoken and "tube" in spoken[0].lower()


def test_home_arm_prompts_through_the_tty_not_a_redirected_stdin():
    src = (TOOLS / "home_arm.py").read_text()
    m = re.search(r"place_and_close\(([^)]*)\)", src)
    assert m and "prompt=v4_carry.operator_prompt" in m.group(1), (
        "home_arm passes a different prompt; the runners redirect its stdout to ${RUN}.home, so "
        "a plain input() prompt is never seen by the operator"
    )
