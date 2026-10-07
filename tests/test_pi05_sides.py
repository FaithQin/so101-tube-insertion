"""The pi0.5 runner resolves its side (A/B), generation, checkpoint and pin from ONE table.

Until Sep 6 2026 `run_pi05_trial.sh` served exactly one policy: pi05-tube-A-v3, hardcoded, with
CAPSTONE_FORCE_WIDE_STATE=0 hardcoded beside it. v4 has an A and a B; B carries 18-dim
normalizer statistics behind the same declared [32] (max_state_dim padding), so the pin must be
per side -- A narrow, B wide -- or B meets a 6-dim state on its first tick. `pi05_sides.py` is
the single source of truth, mirroring `smolvla_dryrun.py --print`.
"""
import os
import subprocess
import sys

import pytest

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))
import pi05_sides as ps  # noqa: E402

PY = sys.executable


def test_v4_is_the_default_generation_and_v3_stays_reachable():
    v4 = ps.active_sides({})
    assert v4["A"].repo_id == "faithqin/pi05-tube-A-v4" and v4["A"].dataset == "faithqin/so101-tube-insert-v4-noload"
    assert v4["B"].repo_id == "faithqin/pi05-tube-B-v4" and v4["B"].dataset == "faithqin/so101-tube-insert-v4"
    v3 = ps.active_sides({"CAPSTONE_POLICY_GEN": "v3"})
    assert v3["A"].repo_id == "faithqin/pi05-tube-A-v3_2026-08-30_03-06-05"
    assert "B" not in v3                      # pi0.5-B never existed in v3; a v3 B must not resolve
    with pytest.raises(ValueError, match="CAPSTONE_POLICY_GEN"):
        ps.active_sides({"CAPSTONE_POLICY_GEN": "v2"})


def test_the_pin_is_narrow_for_a_and_wide_for_b():
    v4 = ps.active_sides({})
    assert (v4["A"].force, v4["A"].state_dim) == ("0", 6)     # declared 32 == max_state_dim would widen; pin narrow
    assert (v4["B"].force, v4["B"].state_dim) == ("1", 18)    # mutation: copy A's pin to B
    assert ps.print_field("force", v4["B"]) == "1" and ps.print_field("gen", v4["B"]) == "v4"


def test_snapshot_is_the_sanitized_checkpoint_dir_not_the_hub_snapshot(tmp_path, monkeypatch):
    seen = {}
    def fake_resolve(repo, ckpt=None):
        seen["resolve"] = (repo, ckpt)
        return str(tmp_path / "snap")
    monkeypatch.setattr(ps, "resolve_snapshot", fake_resolve)
    monkeypatch.setattr(ps, "materialize_local_checkpoint",
                        lambda snap, out: (seen.setdefault("args", (snap, str(out))) and out, ["x"]))
    side = ps.active_sides({})["A"]
    d = ps.sanitized_policy_dir(side, "005000")
    assert seen["args"][0] == str(tmp_path / "snap")
    # the weights fetched are the CHOSEN checkpoint's -- a top-level fetch under a checkpoint-named
    # directory would be a silent substitution (mutation: resolve_snapshot(repo, None))
    assert seen["resolve"] == ("faithqin/pi05-tube-A-v4", "005000")
    assert d.endswith("pi05-tube-A-v4--005000")               # one sanitized dir per repo+checkpoint
    assert "pi05_local_sanitized" in d


def test_print_snapshot_aborts_for_v4_without_a_checkpoint():
    env = {k: v for k, v in os.environ.items() if not k.startswith("CAPSTONE")}
    out = subprocess.run([PY, str(TOOLS / "pi05_sides.py"), "--print", "snapshot", "--side", "A"],
                         capture_output=True, text=True, timeout=120, env=env)
    assert out.returncode == 5, out.stderr[-400:]
    assert "CAPSTONE_CHECKPOINT" in out.stderr and out.stdout.strip() == ""


def test_the_trained_prompt_resolves_for_both_sides_and_is_the_same_prompt():
    prompts = set()
    for side in ps.active_sides({}).values():
        try:
            prompts.add(ps.trained_task(side.dataset))
        except ps.TaskUnavailable:
            pytest.skip(f"{side.dataset} tasks.parquet not reachable")
    assert len(prompts) == 1 and next(iter(prompts)).strip()
