"""The one-way door: a dataset recorded 6-dim can never be retrofitted to 18.

Sep 3 audit finding #1 flagged that CAPSTONE_FORCE_WIDE_STATE supplies the
MECHANISM for wide recording but nothing ENFORCES it, so a session could
silently record 6-dim and foreclose every B-arm policy trained on that data.

Verified Sep 4, before recording v4: `lerobot-record` does NOT go through
rollout/context.py's policy-dependent state filter. lerobot/scripts/
lerobot_record.py builds its dataset features from `robot.observation_features`
directly, which on the patched SOFollower already carries .pos + .load +
.current (18 dims). So recording is wide by construction and the env var is
irrelevant to it.

These tests pin BOTH halves of that, because if either changes the door opens:
  1. the so_follower patch still exposes .load/.current, and
  2. the record script still reads robot.observation_features directly.

DAgger is the path that DOES route through context.py -- it is covered by the
CAPSTONE_FORCE_WIDE_STATE pin, not by this test.
"""
import ast
import inspect

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def test_so_follower_observation_features_include_load_and_current():
    """Parsed with ast -- instantiating the robot would open the serial bus."""
    import ast
    import inspect

    from lerobot.robots.so_follower import so_follower as mod

    tree = ast.parse(inspect.getsource(mod))
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "SOFollower")
    fns = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}

    assert "_load_ft" in fns, (
        "SOFollower._load_ft is gone -- the capstone 18-dim patch was lost "
        "(pip install -U?). Recording would be 6-dim: ONE-WAY DOOR."
    )
    load_src = ast.unparse(fns["_load_ft"])
    assert ".load" in load_src and ".current" in load_src, "_load_ft no longer yields .load/.current"

    obs_src = ast.unparse(fns["observation_features"])
    assert "_motors_ft" in obs_src and "_load_ft" in obs_src, (
        f"observation_features no longer merges _load_ft -- recording narrows to 6-dim. Got: {obs_src}"
    )


def test_record_script_does_not_filter_the_observation_features():
    """lerobot_record.py must pass robot.observation_features through unfiltered.

    If it ever starts building its features the way rollout/context.py does
    (a policy-declared-width filter), recording silently narrows to 6-dim.
    """
    from lerobot.scripts import lerobot_record

    src = inspect.getsource(lerobot_record)
    tree = ast.parse(src)
    passes_raw = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "observation" and isinstance(kw.value, ast.Attribute) \
                        and kw.value.attr == "observation_features":
                    passes_raw = True
    assert passes_raw, (
        "lerobot_record.py no longer passes robot.observation_features directly into "
        "create_initial_features -- recording may have started filtering the state. "
        "Verify the recorded observation.state width before recording anything."
    )
    assert "_state_suffixes" not in src and "CAPSTONE_STATE_ROUTING" not in src, (
        "the record script now references the routing filter; recording width is no longer "
        "guaranteed wide"
    )
