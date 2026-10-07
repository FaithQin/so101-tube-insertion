"""The PRE-GRASPED start pose (`v4-carry`) is DERIVED from the carry frames of the v4 training
set, differs from the frame-0 pose everywhere it appears, and its gate catches an EMPTY jaw.

Handoff Sep 6 §3 item 2 / §6 S2: "tube starts in the jaws at the `v4-carry` pose; a modified
task, labelled; every trial reaches contact". Measured over the 50 TRAINING episodes (Sep 6):

    carry frame = grasp_index + k, k = first frame after grasp where shoulder_lift has moved
    >= 2 deg TOWARD LIFTED. Lifted means the value DECREASES: lift is +45.35 at grasp (the arm
    reached down to the table) and -105.34 at frame 0 (folded home). The numeric "+2 deg"
    reading fires in 0/50 episodes; the "-2 deg" reading fires in 50/50, k in [6, 20] frames
    (0.30-1.00 s at 20 FPS), mean 14.56 sd 2.58. Excluded episodes: 0.

    shoulder_pan   67.09 sd  3.17 [ 61.14,  77.93]      v4 frame 0:   1.65
    shoulder_lift  45.35 sd  7.20 [ 23.43,  57.89]                  -105.34
    elbow_flex    -35.84 sd 13.68 [-59.91,  10.33]                   85.61
    wrist_flex     78.49 sd  7.08 [ 59.78,  92.75]                   73.65
    wrist_roll     49.66 sd  7.78 [ 25.71,  60.35]                   -1.71
    gripper        13.16 sd  0.58 [ 12.13,  15.42]                    1.11  (closed on NOTHING)

Three things this file exists to catch, none of which the frame-0 template covers:

  1. The gripper. home_gate gates frame-0 datasets on `(0, GRIPPER_CLOSED_MAX=3.0)` because at
     frame 0 the jaw is closed on nothing. At the carry pose it is closed ON THE TUBE and reads
     the tube's width (12.13-15.42). A 0-3.0 bound would refuse every legitimate S2 start.
  2. The empty jaw. Position alone cannot tell "jaw at 13.2 on the tube" from "jaw parked at
     13.2 in free air", which is exactly what homing to the carry pose WITHOUT the scripted
     close produces. Training carry gripper |load| is 98-120 (raw); free-air bias is 56-80
     (tools/so_follower_load.patch). The close verifies contact load, and `--pose v4-carry`
     refuses to run without it.
  3. shoulder_lift. At frame 0 it rests against its stop beyond the goal clamp and is exempt
     from the gate and released to settle. At the carry pose it is +45 -- actively holding the
     extended arm, nothing to settle against. It must be GATED (it is the joint that defines
     "lifted") and must NOT be released (the arm would fall onto the rig, holding the tube).

Every number is re-derived from the parquet by the `v4_carry_stats` fixture; a stale table
fails here. Replay validation of the pose itself needs the arm (morning, with Faith).
"""
import re
import subprocess
import sys

import pytest

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))

import home_gate  # noqa: E402
import v4_carry  # noqa: E402

JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TAG = "v4-carry"
PY = sys.executable

# Test-only bench values (no rig fact here; the real numbers come from the fixture).
EMPTY_JAW = 1.1          # v4 frame-0 gripper mean: closed on nothing
OPEN_JAW = 20.0          # v4_audit.JAW_OPEN
FREE_AIR_LOAD = 70.0     # inside the patch's measured 56-80 free-motion bias
TUBE_LOAD = 118.0        # training carry mean


# ---------------------------------------------------------------------------
# Source readers (the template's: read the tables back out of the SOURCE so a stale
# table fails; a hyphenated tag needs its own wrapper regex).
# ---------------------------------------------------------------------------
def _pose_from_source(path, tag: str) -> dict:
    src = path.read_text()
    m = re.search(rf'"{tag}":\s*\[([^\]]+)\]', src)
    assert m, f"{path.name} has no {tag} pose"
    return dict(zip(JOINT_ORDER, [float(x) for x in m.group(1).split(",")]))


def _train_home_from_wrapper(name: str) -> dict:
    src = (TOOLS / "rollout_30hz_stale_ok.py").read_text()
    m = re.search(rf"{name}\s*=\s*\{{(.*?)\}}", src, re.S)
    assert m, f"rollout_30hz_stale_ok.py has no {name} table"
    return {k: float(v) for k, v in re.findall(r'"(\w+)\.pos":\s*([-\d.]+)', m.group(1))}


# ---------------------------------------------------------------------------
# 1. The derivation itself
# ---------------------------------------------------------------------------
def test_derivation_uses_every_training_episode_and_counts_exclusions(v4_carry_stats):
    s = v4_carry_stats
    assert s["n_requested"] == 50
    assert len(s["excluded"]) == v4_carry.EXCLUDED_EPISODES, (
        f"excluded {s['excluded']} but the module claims {v4_carry.EXCLUDED_EPISODES}"
    )
    assert len(s["used"]) + len(s["excluded"]) == s["n_requested"]
    assert s["n_episodes"] == len(s["used"]) == 50


def test_lifted_means_shoulder_lift_DECREASES(v4_carry_stats):
    """The sign check. lift is ~+45 at grasp and -105 at home; the arm lifts by moving back
    toward home, i.e. numerically DOWN. The literal '+2 deg' reading fires in no episode."""
    s = v4_carry_stats
    assert v4_carry.LIFT_DIRECTION == -1
    assert s["lift_at_grasp_mean"] > 20.0, s["lift_at_grasp_mean"]
    assert s["mean"]["shoulder_lift"] <= s["lift_at_grasp_mean"] - v4_carry.LIFT_RISE_DEG
    assert s["n_numeric_rise_fires"] == 0, (
        f"the numeric +2 deg reading fired in {s['n_numeric_rise_fires']} episodes"
    )


def test_k_spread_matches_the_documented_spread(v4_carry_stats):
    k = v4_carry_stats["k_stats"]
    assert k["min"] == v4_carry.K_SPREAD["min"]
    assert k["max"] == v4_carry.K_SPREAD["max"]
    assert abs(k["mean"] - v4_carry.K_SPREAD["mean"]) < 0.01
    assert abs(k["sd"] - v4_carry.K_SPREAD["sd"]) < 0.01
    assert k["min_s"] == pytest.approx(k["min"] / v4_carry.FPS)
    assert k["max_s"] == pytest.approx(k["max"] / v4_carry.FPS)


def test_carry_index_on_synthetic_episodes():
    """The pure selector, on trajectories whose answer is known by construction."""
    import numpy as np

    n = 60
    pan = np.full(n, 68.0)
    grip = np.r_[np.full(20, 30.0), np.full(40, 13.0)]     # opens, then closes at frame 20
    lift = np.full(n, 45.0)
    lift[30:] = 42.0                                       # lifted 3 deg at frame 30 -> k = 10
    assert v4_carry.carry_index(pan, lift, grip) == (20, 10)

    rising = np.full(n, 45.0)
    rising[30:] = 48.0                                     # moves the WRONG way: excluded
    g, why = v4_carry.carry_index(pan, rising, grip)
    assert g is None and "lift" in why

    never_open = np.full(n, 13.0)                          # no grasp at all: excluded
    g, why = v4_carry.carry_index(pan, lift, never_open)
    assert g is None and "grasp" in why


def test_derive_excludes_and_counts_rather_than_dropping(tmp_path):
    """A synthetic two-episode dataset where one episode never lifts: it must land in
    `excluded` with a reason and the pose must be computed from the other one alone."""
    import numpy as np
    import pandas as pd

    def episode(idx, lifts: bool):
        n = 60
        pan = np.full(n, 68.0)
        grip = np.r_[np.full(20, 30.0), np.full(40, 13.0)]
        lift = np.full(n, 45.0)
        if lifts:
            lift[30:] = 40.0
        state = np.zeros((n, 18))
        state[:, 0], state[:, 1], state[:, 2], state[:, 5] = pan, lift, 7.0 + idx, grip
        state[:, 11] = -118.0
        return pd.DataFrame({
            "frame_index": np.arange(n), "episode_index": idx,
            "observation.state": list(state),
        })

    root = tmp_path / "ds"
    (root / "data" / "chunk-000").mkdir(parents=True)
    pd.concat([episode(0, True), episode(1, False)]).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    s = v4_carry.derive(root, [0, 1])
    assert s["used"] == [0]
    assert list(s["excluded"]) == [1] and "lift" in s["excluded"][1]
    assert s["mean"]["elbow_flex"] == pytest.approx(7.0)     # episode 0 only, not 7.5
    assert s["k"] == {0: 10}


# ---------------------------------------------------------------------------
# 2. The pose is selectable in all three places and matches the parquet
# ---------------------------------------------------------------------------
def test_home_arm_has_a_v4_carry_pose_that_matches_the_carry_frames(v4_carry_stats):
    pose = _pose_from_source(TOOLS / "home_arm.py", TAG)
    for joint, mean in v4_carry_stats["mean"].items():   # every joint, lift included: it is commanded here
        assert abs(pose[joint] - mean) <= 2.5, f"home_arm {TAG} {joint}={pose[joint]} vs carry mean {mean:.2f}"


def test_wrapper_train_home_v4_carry_matches_and_is_selectable(v4_carry_stats):
    home = _train_home_from_wrapper("TRAIN_HOME_V4_CARRY")
    for joint, mean in v4_carry_stats["mean"].items():
        assert abs(home[joint] - mean) <= 0.11, f"TRAIN_HOME_V4_CARRY {joint}={home[joint]} vs {mean:.2f}"
    src = (TOOLS / "rollout_30hz_stale_ok.py").read_text()
    assert re.search(r'"v4-carry":\s*TRAIN_HOME_V4_CARRY', src), (
        "TRAIN_HOME table does not offer v4-carry (CAPSTONE_HOME_POSE=v4-carry)"
    )


def test_home_gate_exposes_v4_carry_support_from_the_parquet(v4_carry_stats):
    for joint, mean in v4_carry_stats["mean"].items():
        m, sd, lo, hi = home_gate.FRAME0[TAG][joint]
        assert abs(m - mean) < 0.05, f"{TAG} {joint}: gate mean {m} vs dataset {mean:.2f}"
        assert abs(sd - v4_carry_stats["std"][joint]) < 0.05
        assert abs(lo - v4_carry_stats["min"][joint]) < 0.05
        assert abs(hi - v4_carry_stats["max"][joint]) < 0.05
    assert TAG in home_gate.SUPPORT_BY_DATASET


def test_home_arm_and_wrapper_agree_on_the_carry_pose():
    home = _pose_from_source(TOOLS / "home_arm.py", TAG)
    wrapper = _train_home_from_wrapper("TRAIN_HOME_V4_CARRY")
    for joint in JOINT_ORDER:
        assert abs(home[joint] - wrapper[joint]) < 0.05, f"{joint}: home_arm {home[joint]} vs wrapper {wrapper[joint]}"


def test_v4_carry_differs_materially_from_v4_frame0(v4_frame0_stats, v4_carry_stats):
    """The carry pose cannot be a copy of the frame-0 table: the whole arm has moved."""
    for joint, floor in (("shoulder_lift", 100.0), ("elbow_flex", 100.0), ("shoulder_pan", 50.0),
                         ("wrist_roll", 40.0), ("gripper", 8.0)):
        d = abs(v4_frame0_stats["mean"][joint] - v4_carry_stats["mean"][joint])
        assert d > floor, f"{joint}: carry and frame-0 means differ by only {d:.2f} -- is the table a copy?"
    assert home_gate.FRAME0[TAG]["shoulder_lift"][0] != home_gate.FRAME0["v4"]["shoulder_lift"][0]


def test_every_pose_table_labels_the_carry_pose_a_modified_task():
    """Handoff: 'Label it a modified task everywhere it appears.'"""
    for path in ("home_arm.py", "rollout_30hz_stale_ok.py", "home_gate.py", "v4_carry.py"):
        src = (TOOLS / path).read_text()
        assert re.search(r"modified task", src, re.I), f"{path} does not label v4-carry a modified task"


# ---------------------------------------------------------------------------
# 3. The gripper decision
# ---------------------------------------------------------------------------
def test_carry_gripper_gate_admits_the_measured_carry_range_and_refuses_an_empty_jaw(v4_carry_stats):
    lo, hi = home_gate.SUPPORT_BY_DATASET[TAG]["gripper"]
    assert lo == pytest.approx(v4_carry_stats["min"]["gripper"], abs=0.05)
    assert hi == pytest.approx(v4_carry_stats["max"]["gripper"], abs=0.05)
    for v in (v4_carry_stats["min"]["gripper"], v4_carry_stats["mean"]["gripper"], v4_carry_stats["max"]["gripper"]):
        assert home_gate.in_training_support("gripper", v, dataset=TAG), f"carry gripper {v:.2f} refused"
    # The failure modes that matter: closed on nothing, and never closed.
    assert not home_gate.in_training_support("gripper", EMPTY_JAW, dataset=TAG)
    assert not home_gate.in_training_support("gripper", home_gate.GRIPPER_CLOSED_MAX, dataset=TAG)
    assert not home_gate.in_training_support("gripper", OPEN_JAW, dataset=TAG)
    assert hi < 16.0, "carry band reaches v4_audit's 'closed' threshold -- it would admit an open jaw"


def test_frame0_datasets_keep_the_closed_on_nothing_gripper_gate():
    for ds in ("v2", "v3", "v4"):
        assert home_gate.SUPPORT_BY_DATASET[ds]["gripper"] == (0.0, home_gate.GRIPPER_CLOSED_MAX), ds
        assert home_gate.in_training_support("gripper", EMPTY_JAW, dataset=ds)
        assert not home_gate.in_training_support("gripper", 13.2, dataset=ds)


# ---------------------------------------------------------------------------
# 4. shoulder_lift: gated at the carry pose, still passive at frame 0
# ---------------------------------------------------------------------------
def test_shoulder_lift_is_gated_at_the_carry_pose_and_passive_everywhere_else(v4_carry_stats):
    lift = v4_carry_stats["mean"]["shoulder_lift"]
    assert home_gate.in_training_support("shoulder_lift", lift, dataset=TAG)
    assert not home_gate.in_training_support("shoulder_lift", -105.2, dataset=TAG), (
        "a folded (frame-0) shoulder_lift passed the CARRY gate -- the arm is at home, not carrying"
    )
    for ds in ("v2", "v3", "v4"):
        assert home_gate.in_training_support("shoulder_lift", -105.2, dataset=ds)
        assert home_gate.in_training_support("shoulder_lift", 45.0, dataset=ds)   # passive: never gated


def test_check_pose_refuses_the_frame0_pose_under_the_carry_gate_and_admits_the_carry_mean(
    v4_frame0_stats, v4_carry_stats
):
    ok, offenders = home_gate.check_pose(dict(v4_frame0_stats["mean"]), dataset=TAG)
    assert not ok and {"shoulder_lift", "elbow_flex", "gripper"} <= set(offenders)
    ok, offenders = home_gate.check_pose(dict(v4_carry_stats["mean"]), dataset=TAG)
    assert ok, home_gate.describe(offenders)


def test_settle_release_is_skipped_at_the_carry_pose_in_both_homing_paths():
    """Releasing shoulder_lift torque is only safe against its stop. Both scripts keep their
    literal SETTLE_JOINTS = ("shoulder_lift",) (pinned by test_home_pose_matches_training) and
    route it through the selector, which empties it for the carry pose."""
    assert v4_carry.settle_joints_for(TAG, ("shoulder_lift",)) == ()
    for tag in ("v1", "v2", "v3", "v4"):
        assert v4_carry.settle_joints_for(tag, ("shoulder_lift",)) == ("shoulder_lift",)
    for path in ("home_arm.py", "rollout_30hz_stale_ok.py"):
        src = (TOOLS / path).read_text()
        assert re.search(r"for\s+_?j\s+in\s+(?:v4_carry\.|_)?settle_joints_for\(", src), (
            f"{path}: the settle loop does not go through settle_joints_for -- at the carry pose "
            f"it would cut torque on a shoulder_lift that is holding the extended arm"
        )


# ---------------------------------------------------------------------------
# 5. The scripted close: parameters and refusals (never motion -- it drives the bus)
# ---------------------------------------------------------------------------
class FakeBus:
    """Reads what the jaw would do; records every command. `stalls_at` is where the jaw stops
    (tube width, or ~1 on nothing); `load` is the |Present_Load| it reports once there."""

    def __init__(self, start_at: float, stalls_at: float, load: float, moves: bool = True):
        self.pos, self.stalls_at, self.load, self.moves = start_at, stalls_at, load, moves
        self.goals, self.writes = [], []

    def write(self, reg, motor, value, normalize=True):
        assert motor == "gripper", motor
        self.writes.append((reg, motor, value))
        if reg == "Goal_Position":
            self.goals.append(float(value))
            if self.moves:
                self.pos = max(float(value), self.stalls_at) if value < self.pos else float(value)

    def read(self, reg, motor, normalize=True):
        assert motor == "gripper", motor
        if reg == "Present_Position":
            return self.pos
        if reg == "Present_Load":
            return -self.load if abs(self.pos - self.stalls_at) < 1e-6 else -FREE_AIR_LOAD
        raise KeyError(reg)


def _no_sleep(_s):
    pass


def _patch_lead_and_hysteresis() -> dict:
    src = (TOOLS / "so_follower_load.patch").read_text()
    m = re.search(r"lead\s*=\s*([\d.]+)\s+if in_contact else\s*([\d.]+)", src)
    e = re.search(r"if load >= (\d+):", src)
    r = re.search(r"elif load < (\d+):", src)
    assert m and e and r, "cannot find the follower patch's bounded-lead rule"
    return {"contact": float(m.group(1)), "free": float(m.group(2)), "engage": float(e.group(1)), "release": float(r.group(1))}


def test_close_mirrors_the_follower_patch_bounded_lead():
    """The production path produced training's carry load (~118 raw) with THIS rule; the
    scripted close must reproduce it, not invent its own."""
    p = _patch_lead_and_hysteresis()
    assert v4_carry.CLOSE_LEAD_FREE == p["free"]
    assert v4_carry.CLOSE_LEAD_CONTACT == p["contact"]
    assert v4_carry.CONTACT_ENGAGE_LOAD == p["engage"]
    assert v4_carry.CONTACT_RELEASE_LOAD == p["release"]
    assert v4_carry.CARRY_LOAD_MIN == p["release"]


def test_close_parameters_sit_inside_the_training_data(v4_carry_stats):
    s = v4_carry_stats
    assert v4_carry.FPS == 20
    assert OPEN_JAW <= v4_carry.OPEN_FOR_PLACEMENT <= s["pre_grasp_open"]["max"], (
        "placement opening is outside what the training grasps opened to"
    )
    assert v4_carry.CARRY_LOAD_MIN <= s["gripper_load_abs"]["min"], (
        "some training carry frames would fail the contact-load check"
    )
    assert v4_carry.CARRY_LOAD_MIN > 80.0, "must sit above the patch's measured 56-80 free-air load bias"
    assert v4_carry.CLOSE_TARGET < s["min"]["gripper"], "the close must aim below the tube, as the trigger did"


def test_close_on_tube_settles_on_the_tube_with_bounded_commands():
    bus = FakeBus(start_at=35.0, stalls_at=13.0, load=TUBE_LOAD)
    grip = v4_carry.close_on_tube(bus, sleep=_no_sleep)
    assert grip.position == pytest.approx(13.0) and grip.load == pytest.approx(TUBE_LOAD)
    v4_carry.verify_carry_grip(grip)          # must not raise
    assert bus.goals and min(bus.goals) >= v4_carry.CLOSE_TARGET
    # Never more than the free-air lead below the position read on that tick.
    pos = 35.0
    for g in bus.goals:
        assert g >= pos - v4_carry.CLOSE_LEAD_FREE - 1e-9, f"goal {g} is more than the lead below {pos}"
        pos = max(g, 13.0)
    # In contact the bound tightens to the contact lead (the patch's second speed).
    assert bus.goals[-1] == pytest.approx(13.0 - v4_carry.CLOSE_LEAD_CONTACT)


def test_close_refuses_an_empty_jaw():
    bus = FakeBus(start_at=35.0, stalls_at=EMPTY_JAW, load=85.0)   # closed on itself, trigger-hard
    grip = v4_carry.close_on_tube(bus, sleep=_no_sleep)
    with pytest.raises(v4_carry.CarryRefused, match=r"(?i)empty|closed on nothing"):
        v4_carry.verify_carry_grip(grip)


def test_verify_refuses_a_jaw_parked_at_tube_width_in_free_air():
    """THE case position alone cannot see: home_arm without the close leaves the empty jaw at 13.2."""
    with pytest.raises(v4_carry.CarryRefused, match=r"(?i)load|free air"):
        v4_carry.verify_carry_grip(v4_carry.CarryGrip(position=13.2, load=FREE_AIR_LOAD))


def test_close_refuses_a_jaw_that_never_closed():
    bus = FakeBus(start_at=35.0, stalls_at=35.0, load=60.0, moves=False)
    grip = v4_carry.close_on_tube(bus, sleep=_no_sleep)
    with pytest.raises(v4_carry.CarryRefused, match=r"(?i)outside|did not close"):
        v4_carry.verify_carry_grip(grip)


def test_close_refuses_when_the_jaw_never_settles():
    class Creeping(FakeBus):
        def read(self, reg, motor, normalize=True):
            if reg == "Present_Position":
                self.pos -= 0.5      # keeps moving forever: no stall, no equilibrium
                return self.pos
            return -FREE_AIR_LOAD

    # moves=False: FakeBus.write would otherwise re-park the jaw at `stalls_at` on every command,
    # which reads as a stall after 5 ticks and the refusal never fires (the fake, not the code).
    with pytest.raises(v4_carry.CarryRefused, match=r"(?i)settle"):
        v4_carry.close_on_tube(Creeping(35.0, 13.0, TUBE_LOAD, moves=False), sleep=_no_sleep)


def test_place_and_close_opens_first_prompts_once_then_closes(v4_carry_stats):
    bus = FakeBus(start_at=13.2, stalls_at=13.0, load=TUBE_LOAD)
    prompts = []
    grip = v4_carry.place_and_close(bus, dict(v4_carry_stats["mean"]), prompt=prompts.append, sleep=_no_sleep)
    assert bus.goals[0] == pytest.approx(v4_carry.OPEN_FOR_PLACEMENT)
    assert prompts and len(prompts) == 1
    assert grip.position == pytest.approx(13.0)


def test_place_and_close_refuses_before_prompting_if_the_jaw_did_not_open(v4_carry_stats):
    bus = FakeBus(start_at=13.2, stalls_at=13.0, load=TUBE_LOAD, moves=False)
    prompts = []
    with pytest.raises(v4_carry.CarryRefused, match=r"(?i)open"):
        v4_carry.place_and_close(bus, dict(v4_carry_stats["mean"]), prompt=prompts.append, sleep=_no_sleep)
    assert prompts == [], "asked the operator to place a tube into a jaw that never opened"


def test_place_and_close_refuses_off_the_carry_pose_without_commanding_anything(v4_frame0_stats):
    bus = FakeBus(start_at=13.2, stalls_at=13.0, load=TUBE_LOAD)
    with pytest.raises(v4_carry.CarryRefused, match=r"(?i)carry pose"):
        v4_carry.place_and_close(bus, dict(v4_frame0_stats["mean"]), prompt=lambda _m: None, sleep=_no_sleep)
    assert bus.goals == []


def test_importing_v4_carry_touches_no_bus():
    src = (TOOLS / "v4_carry.py").read_text()
    assert not re.search(r"^\s*(?:from|import)\s+_arms", src, re.M), "v4_carry imports the bus helper at module level"
    assert not re.search(r"^[^\n#]*open_bus\(", src, re.M), "v4_carry opens a bus at import"


# ---------------------------------------------------------------------------
# 6. home_arm wiring: OFF by default; v4-carry REQUIRES the close; refusal prints no pose
# ---------------------------------------------------------------------------
def _home_arm_src() -> str:
    return (TOOLS / "home_arm.py").read_text()


def test_home_arm_close_flag_is_off_by_default():
    assert re.search(r'"--close-on-tube",\s*action="store_true"', _home_arm_src())


def test_home_arm_refuses_bad_flag_combinations_before_opening_the_bus():
    src = _home_arm_src()
    refusal = re.search(r"parser\.error\([^)]*close-on-tube", src)
    assert refusal, "home_arm has no parser.error for the close flag"
    assert refusal.start() < src.index("open_bus("), "the refusal comes AFTER the bus is opened"
    # Executed: argparse exits 2 before any import that could reach the serial port.
    for argv in (["--pose", TAG], ["--pose", "v4", "--close-on-tube"]):
        r = subprocess.run([PY, str(TOOLS / "home_arm.py"), *argv], capture_output=True, text=True, timeout=120)
        assert r.returncode == 2, f"{argv}: rc {r.returncode}\n{r.stderr}"
        assert "close-on-tube" in r.stderr


def test_home_arm_refusal_exits_nonzero_before_the_settled_line():
    """run_scored_trial.sh ignores home_arm's exit code and preflight keys off 'settled:'.
    A refused close must therefore never print that line."""
    src = _home_arm_src()
    close = src.index("place_and_close(")
    handler = re.search(r"except v4_carry\.CarryRefused[^\n]*\n(?:.*\n){0,6}?\s*sys\.exit\(1\)", src)
    assert handler and handler.start() > close, "CarryRefused is not turned into sys.exit(1)"
    assert handler.end() < src.index('print("settled:"'), "the refusal is handled after 'settled:' is printed"
