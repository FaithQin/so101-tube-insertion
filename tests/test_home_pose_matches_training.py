"""Regression: the homing pose must match the CURRENT dataset's measured
frame-0 states, and torque settle must be selective.

Born from the Aug 16 2026 overshoot debug. Two compounding bugs:

1. home_arm.py / rollout wrapper HOME targets were measured on the V1 dataset
   (elbow 82.9); v2 episodes actually start at elbow 79.7 +/- 2.2.
2. Both homing paths released torque on ALL joints so shoulder_lift settles
   against its stop (-105.2, correct) -- but the unpowered elbow then sags
   under gravity. Probe 154411's saved episode started at elbow 88.4: outside
   the entire v2 training range (75.5..84.4). Training frame-0s were captured
   under teleop with torque ON. Fix: release ONLY shoulder_lift; keep every
   other joint holding its target.

These tests parse the tool sources (importing them would open the serial
bus) and compare against frame-0 statistics computed from the dataset itself.
"""

import ast
import re

from conftest import TOOLS

JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Joints that settle passively against a hard stop (beyond the goal clamp);
# their commanded target legitimately differs from the dataset frame-0 value.
PASSIVE_SETTLE_JOINTS = {"shoulder_lift"}

# |home_target - dataset_frame0_mean| tolerance, in normalized units. The
# tightest joint std in v2 is 0.1 (lift); the loosest ~2.2 (elbow). One-sigma
# on the loose joints, generous on the tight ones.
TOL = 2.5


def _home_arm_pose() -> dict:
    src = (TOOLS / "home_arm.py").read_text()
    m = re.search(r"\"v2\":\s*\[([^\]]+)\]", src)
    assert m, "home_arm.py has no v2 pose list — v1-only HOME is the Aug 16 regression"
    vals = [float(x) for x in m.group(1).split(",")]
    return dict(zip(JOINT_ORDER, vals))


def _wrapper_pose() -> dict:
    src = (TOOLS / "rollout_30hz_stale_ok.py").read_text()
    m = re.search(r"TRAIN_HOME_V2\s*=\s*\{([^}]+)\}", src)
    assert m, "rollout wrapper has no TRAIN_HOME_V2 — v1-only TRAIN_HOME is the Aug 16 regression"
    pose = dict(re.findall(r"\"(\w+)\.pos\":\s*(-?[\d.]+)", m.group(1)))
    return {k: float(v) for k, v in pose.items()}


def test_home_arm_v2_pose_matches_dataset_frame0(v2_frame0_stats):
    pose = _home_arm_pose()
    for joint, mean in v2_frame0_stats["mean"].items():
        if joint in PASSIVE_SETTLE_JOINTS:
            continue
        assert abs(pose[joint] - mean) <= TOL, (
            f"home_arm.py v2 {joint}={pose[joint]} vs dataset frame-0 mean {mean:.1f} "
            f"(std {v2_frame0_stats['std'][joint]:.1f}) — homing outside training support"
        )


def test_wrapper_v2_pose_matches_dataset_frame0(v2_frame0_stats):
    pose = _wrapper_pose()
    for joint, mean in v2_frame0_stats["mean"].items():
        if joint in PASSIVE_SETTLE_JOINTS:
            continue
        assert abs(pose[joint] - mean) <= TOL, (
            f"wrapper TRAIN_HOME_V2 {joint}={pose[joint]} vs dataset frame-0 mean {mean:.1f} "
            f"— between-episode reset outside training support"
        )


def test_home_arm_and_wrapper_agree():
    home, wrapper = _home_arm_pose(), _wrapper_pose()
    for joint in JOINT_ORDER:
        assert abs(home[joint] - wrapper[joint]) < 0.05, (
            f"{joint}: home_arm.py ({home[joint]}) and wrapper ({wrapper[joint]}) disagree — "
            f"episode 1 and episodes 2+ would start from different poses"
        )


def _settle_joints(path) -> tuple:
    src = (TOOLS / path).read_text()
    m = re.search(r"SETTLE_JOINTS\s*=\s*(\([^)]*\))", src)
    assert m, (
        f"{path}: no SETTLE_JOINTS constant. Full-bus disable_torque lets the elbow sag "
        f"out of training support (measured 88.4 vs range 75.5..84.4 on Aug 16)."
    )
    return ast.literal_eval(m.group(1))


def test_home_arm_settles_only_shoulder_lift():
    assert _settle_joints("home_arm.py") == ("shoulder_lift",)


def test_wrapper_settles_only_shoulder_lift():
    assert _settle_joints("rollout_30hz_stale_ok.py") == ("shoulder_lift",)


# test_wrapper_strips_arrow_keys lived here until Sep 5 2026. It scanned the
# wrapper's source for "_PYNPUT_KEY_NAMES" and for the literal name of a local
# function -- the same prose-scanning class this repo keeps finding, and it
# would have gone red on a refactor that kept the behaviour. The guard now
# lives in tools/arrow_guard.py and is covered by tests/test_arrow_guard.py,
# which EXECUTES the filter through both real entry points (rollout AND
# record) and checks that n/r/q/ESC still get through. Strictly stronger.


def test_wrapper_rehome_does_not_recommand_a_closed_gripper():
    """Aug 22 17:28 block: the between-episode re-home closed the gripper on
    nothing and then re-sent "go to 1.3" for any residual > 3. The follower bounds
    each close to 4.0 below present (~56 counts) in free air, the firmware latches
    at a sustained ~35 counts, and the fault-recovery path opens the jaw +2 per
    event — frame-0 gripper state crept +8.2 / +9.5 / +10.8 across episodes 2-4.
    The wrapper must give the gripper residual slack (>= 3.0) so a near-closed
    jaw is not re-commanded, and must keep torque ON (training held it at 1.3)."""
    src = (TOOLS / "rollout_30hz_stale_ok.py").read_text()
    m = re.search(r"_HOME_SLACK\[\"gripper\.pos\"\]\s*=\s*([\d.]+)", src) or re.search(
        r"_HOME_SLACK\s*=\s*\{[^}]*\"gripper\.pos\":\s*([\d.]+)", src
    )
    assert m, "wrapper _HOME_SLACK has no gripper entry — re-home will re-command a closed jaw and creep it open"
    assert float(m.group(1)) >= 3.0, f"gripper re-home slack {m.group(1)} < 3.0 — still re-commands a closed jaw"
    assert "gripper" not in _settle_joints("rollout_30hz_stale_ok.py"), (
        "gripper must stay torque-ON after re-home (training frame-0 held it at 1.3)"
    )


# --- v3 (Sep 3 2026) --------------------------------------------------------
# The v1->v2 bug of Aug 16 recurred as v2->v3 and nothing caught it, because
# the guards above only ever named v2. rollout_30hz_stale_ok.py defined
# TRAIN_HOME for v1 and v2 only and defaulted CAPSTONE_HOME_POSE to "v2", and
# no v3 runner set the variable.
#
# What that had ALREADY broken was first written down wrong, and corrected
# Sep 5: TRAIN_HOME reaches only _return_to_initial_position, which runs AFTER
# an episode, and every runner passes --dataset.num_episodes=1. Measured across
# all 20 v3-era rollout parquets, frame-0 elbow was 82.04 (sd 1.38) vs the v3
# training mean 82.34 -- NOT "~1 deg low"; only the one re-record (P05-02) was
# affected. See the the lab notebook (private), Sep 3->4 Fault 1.
#
# The tests below stand on their own merit: a v3 runner must still resolve to
# the v3 pose, because the moment a run is multi-episode -- or a v4 tag lands
# in that table -- the default becomes load-bearing.


def _wrapper_pose_v3() -> dict:
    src = (TOOLS / "rollout_30hz_stale_ok.py").read_text()
    m = re.search(r"TRAIN_HOME_V3\s*=\s*\{([^}]+)\}", src)
    assert m, (
        "rollout wrapper has no TRAIN_HOME_V3 — v3 rollouts silently re-home to "
        "the v2 pose (Sep 3 2026 regression)"
    )
    pose = dict(re.findall(r'"(\w+)\.pos":\s*(-?[\d.]+)', m.group(1)))
    return {k: float(v) for k, v in pose.items()}


def _v3_frame0_means() -> dict:
    import sys
    sys.path.insert(0, str(TOOLS))
    import home_gate
    return {j: st[0] for j, st in home_gate.FRAME0["v3"].items()}


def test_wrapper_v3_pose_matches_v3_dataset_frame0():
    pose = _wrapper_pose_v3()
    for joint, mean in _v3_frame0_means().items():
        if joint in PASSIVE_SETTLE_JOINTS:
            continue
        assert abs(pose[joint] - mean) <= TOL, (
            f"wrapper TRAIN_HOME_V3 {joint}={pose[joint]} vs v3 frame-0 mean {mean:.2f} "
            f"— between-episode reset outside v3 training support"
        )


def _train_home_table() -> dict:
    """The {tag: constant-name} mapping behind `TRAIN_HOME = {...}[_HOME_POSE]`.

    Parsed with ast, not a substring scan. The substring version asserted only
    that the KEY '"v3"' appeared between the braces, so mapping
    `"v3": TRAIN_HOME_V2` -- the Sep 3 bug verbatim -- left the whole suite
    green. Verified by mutation Sep 5: 364 passed with that edit in place.
    """
    src = (TOOLS / "rollout_30hz_stale_ok.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "TRAIN_HOME" for t in node.targets):
            continue
        sub = node.value
        assert isinstance(sub, ast.Subscript), (
            "TRAIN_HOME is no longer a `{table}[tag]` lookup — this test can no "
            "longer prove which pose a tag resolves to; rewrite it against the new shape"
        )
        assert isinstance(sub.slice, ast.Name) and sub.slice.id == "_HOME_POSE", (
            "TRAIN_HOME is not keyed by _HOME_POSE — the env var no longer selects the pose"
        )
        assert isinstance(sub.value, ast.Dict), "TRAIN_HOME lookup table is not a dict literal"
        table = {}
        for k, v in zip(sub.value.keys, sub.value.values):
            assert isinstance(k, ast.Constant) and isinstance(v, ast.Name), (
                "TRAIN_HOME table entries must be `\"tag\": TRAIN_HOME_TAG` name references"
            )
            table[k.value] = v.id
        return table
    raise AssertionError("no `TRAIN_HOME = ...` assignment in rollout_30hz_stale_ok.py")


def test_wrapper_reads_the_pose_tag_from_the_environment():
    """First link of the chain: CAPSTONE_HOME_POSE -> _HOME_POSE."""
    src = (TOOLS / "rollout_30hz_stale_ok.py").read_text()
    assert re.search(r'_HOME_POSE\s*=\s*os\.environ\.get\(\s*"CAPSTONE_HOME_POSE"', src), (
        "_HOME_POSE no longer reads CAPSTONE_HOME_POSE from the environment — "
        "the runners' export would stop selecting the pose"
    )


def test_wrapper_resolves_every_pose_tag_to_its_OWN_table():
    """Second link: each tag must resolve to the table of the SAME version.

    This is the property the whole changeset exists for -- 'gated on v3, then
    silently re-homed to v2'. The value test
    (test_wrapper_v3_pose_matches_v3_dataset_frame0) proves TRAIN_HOME_V3
    matches the v3 dataset; this proves CAPSTONE_HOME_POSE=v3 actually
    reaches it. Neither test alone closes the chain.
    """
    table = _train_home_table()
    for tag in ("v1", "v2", "v3"):
        assert tag in table, (
            f'TRAIN_HOME has no "{tag}" key — CAPSTONE_HOME_POSE={tag} raises KeyError at import'
        )
        assert table[tag] == f"TRAIN_HOME_{tag.upper()}", (
            f'CAPSTONE_HOME_POSE={tag} resolves to {table[tag]}, not TRAIN_HOME_{tag.upper()} '
            f"— preflight gates on {tag} and the wrapper then re-homes somewhere else"
        )


def test_v3_runners_EXPORT_the_home_pose_bound_to_their_dataset_tag():
    """A runner must EXPORT CAPSTONE_HOME_POSE, and bind it to $DATASET_TAG.

    Both halves are load-bearing and neither was checked before (Sep 5 review,
    both verified by mutation):
      - drop the word `export` and the bare zsh assignment never reaches the
        child `python tools/rollout_30hz_stale_ok.py`;
      - hard-code `v2` and a v3 runner re-homes to v2 -- the original bug.
    Binding to $DATASET_TAG, rather than to the literal v3, is what keeps the
    home pose and the preflight gate reading the same tag when v4 lands.
    """
    for runner in ("run_pi05_trial.sh", "run_scored_trial.sh", "run_smolvla_trial.sh"):
        path = TOOLS / runner
        # No silent `continue`: a renamed runner must fail here, not vanish.
        assert path.exists(), (
            f"{runner} is missing — if it was renamed, rename it here too; a runner "
            f"that drops out of this loop is an unpinned home pose nobody notices"
        )
        # code only -- a comment mentioning the variable must not satisfy the pin
        # (the "tests that scan prose" class; caught in this very file, Sep 4)
        src = "\n".join(line.split("#")[0] for line in path.read_text().splitlines())
        m = re.search(r"^\s*export\s+CAPSTONE_HOME_POSE=(\S+)", src, re.M)
        assert m, (
            f"{runner} never EXPORTS CAPSTONE_HOME_POSE — the rollout wrapper runs in a "
            f"child process and will re-home to its v2 default after preflight gated on v3"
        )
        value = m.group(1).strip("\"'")
        # Sep 6 2026: the start pose became its own variable (START_POSE = the dataset tag,
        # or <tag>-carry for the PRE-GRASPED start). The invariant is unchanged: the export
        # must expand THE SAME variable preflight is handed, never a literal pose.
        pre = re.search(r'preflight\.py "\$\{RUN\}\.scene" "\$\{RUN\}\.home" "\$\{RUN\}\.ping" "(\$\{?\w+\}?)"', src)
        assert pre, f"{runner}: cannot find the preflight call and the tag it is handed"
        gated_on = pre.group(1).strip("{}").replace("${", "$")
        assert value.replace("${", "$").rstrip("}") == gated_on and gated_on in ("$DATASET_TAG", "$START_POSE"), (
            f"{runner} exports CAPSTONE_HOME_POSE={value} but preflight gated on {pre.group(1)} — "
            f"the home pose and the gate must read the same variable, never a hard-coded pose"
        )


def test_the_elbow_approaches_its_target_from_above():
    """Sep 6 2026. The Aug 30 anti-backlash rule undershoots and comes back UP, which assumes the
    joint can lift into its target. The elbow's gearbox wore out (it steps in notches by hand, one
    per gear tooth) and it can no longer do that: six consecutive homes from below landed 82.3-83.6
    for an 85.6 target and the motor stalled at load 550 / current 245 without moving, at raised
    proportional AND raised integral gain. Descending from 90.2 landed 85.8 and passed the gate.
    Flip this back only with a bench measurement that says the elbow can lift again."""
    import re

    src = (TOOLS / "home_arm.py").read_text()
    m = re.search(r"APPROACH_FROM_ABOVE\s*=\s*\{([^}]*)\}", src)
    assert m, "home_arm has no per-joint approach direction"
    assert "elbow_flex" in m.group(1), "the elbow no longer approaches from above"
    assert "shoulder_lift" not in m.group(1), "shoulder_lift settles against its stop; it is excluded"
    # the pre-position must actually use it, not just declare it
    assert re.search(r"HOME\[j\]\s*\+\s*BACKLASH_MARGIN\s+if\s+j\s+in\s+APPROACH_FROM_ABOVE", src), (
        "APPROACH_FROM_ABOVE is declared but the pre-position still undershoots every joint"
    )
    assert re.search(r"bus\.write\(\"Goal_Position\", j, _approach_from\(j\)", src), (
        "the pre-position loop does not go through _approach_from"
    )
