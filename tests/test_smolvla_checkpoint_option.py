"""The SmolVLA runner benches the checkpoint the hold-out verdict chose, never silently the final one.

Sep 5 2026: `tools/v4_holdout_loss.py` showed both SmolVLA-v4 runs OVERFIT by 20k (+39 %/+45 %,
all 8 hold-out episodes worse); the minimum is the 5k checkpoint. The top-level weights in
`faithqin/smolvla-tube-{A,B}-v4` ARE the 20k checkpoint, so `run_smolvla_trial.sh`, which
resolves its snapshot through `smolvla_dryrun.py --print snapshot`, would have benched the wrong
policy. The runner and the rehearsal share one resolver, so the choice lives there:

  CAPSTONE_POLICY_GEN   v4 (default) | v3        which generation's repos and datasets
  CAPSTONE_CHECKPOINT   005000 | 5000 | top       which weights; for a v4 side it MUST be set
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))
import smolvla_dryrun as sd  # noqa: E402

PY = sys.executable


def test_default_generation_is_v4_and_v3_stays_reachable():
    v4 = sd.active_sides({})
    assert v4["A"].repo_id == "faithqin/smolvla-tube-A-v4"
    assert v4["A"].dataset == "faithqin/so101-tube-insert-v4-noload"
    assert v4["B"].repo_id == "faithqin/smolvla-tube-B-v4"
    assert v4["B"].dataset == "faithqin/so101-tube-insert-v4"
    # v4-B declares [6] and carries 18-dim statistics exactly like v3-B: the pin is unchanged
    assert (v4["A"].force, v4["A"].state_dim) == ("0", 6)
    assert (v4["B"].force, v4["B"].state_dim) == ("1", 18)
    v3 = sd.active_sides({"CAPSTONE_POLICY_GEN": "v3"})
    assert v3["A"].repo_id == "faithqin/smolvla-tube-A-v3" and v3 is sd.SIDES  # the historical table
    with pytest.raises(ValueError, match="CAPSTONE_POLICY_GEN"):
        sd.active_sides({"CAPSTONE_POLICY_GEN": "v5"})


def test_checkpoint_from_env_forces_an_explicit_choice_for_v4():
    with pytest.raises(sd.CheckpointUnavailable, match="005000"):
        sd.checkpoint_from_env({}, gen="v4")                     # unset -> abort, names the verdict
    with pytest.raises(sd.CheckpointUnavailable, match="005000"):
        sd.checkpoint_from_env({"CAPSTONE_CHECKPOINT": ""}, gen="v4")
    assert sd.checkpoint_from_env({"CAPSTONE_CHECKPOINT": "top"}, gen="v4") is None   # deliberate final
    assert sd.checkpoint_from_env({"CAPSTONE_CHECKPOINT": "5000"}, gen="v4") == "005000"
    assert sd.checkpoint_from_env({"CAPSTONE_CHECKPOINT": "005000"}, gen="v4") == "005000"
    with pytest.raises(sd.CheckpointUnavailable, match="not a step"):
        sd.checkpoint_from_env({"CAPSTONE_CHECKPOINT": "latest"}, gen="v4")
    assert sd.checkpoint_from_env({}, gen="v3") is None          # v3 trials were the final weights


def test_resolve_snapshot_targets_the_checkpoint_subfolder(tmp_path, monkeypatch):
    calls = []

    def fake_snapshot_download(repo_id, allow_patterns=None, **kw):
        calls.append((repo_id, allow_patterns))
        root = tmp_path / repo_id.replace("/", "--")
        if allow_patterns:
            d = root / "checkpoints" / "005000" / "pretrained_model"
            d.mkdir(parents=True, exist_ok=True)
            (d / "model.safetensors").write_bytes(b"x")
        else:
            root.mkdir(parents=True, exist_ok=True)
        return str(root)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    p = sd.resolve_snapshot("faithqin/smolvla-tube-A-v4", "005000")
    assert p.endswith(os.path.join("checkpoints", "005000", "pretrained_model"))
    assert calls[-1][1] == ["checkpoints/005000/pretrained_model/*"]    # mutation: download everything
    assert sd.resolve_snapshot("faithqin/smolvla-tube-A-v4", None) == str(tmp_path / "faithqin--smolvla-tube-A-v4")
    with pytest.raises(sd.CheckpointUnavailable, match="020000"):
        sd.resolve_snapshot("faithqin/smolvla-tube-A-v4", "020000")     # fake tree has no 020000


def test_print_snapshot_aborts_for_v4_without_a_checkpoint():
    """The runner does `SNAP=$(... --print snapshot ...) || exit 4` -- the abort must be a non-zero exit."""
    env = {k: v for k, v in os.environ.items() if k not in ("CAPSTONE_CHECKPOINT", "CAPSTONE_POLICY_GEN")}
    out = subprocess.run([PY, str(TOOLS / "smolvla_dryrun.py"), "--print", "snapshot", "--side", "A"],
                         capture_output=True, text=True, timeout=120, env=env)
    assert out.returncode == 5, out.stderr[-400:]
    assert "CAPSTONE_CHECKPOINT" in out.stderr and "005000" in out.stderr
    assert out.stdout.strip() == ""          # nothing a `$(...)` could mistake for a path


def test_runner_records_the_resolved_checkpoint_path():
    src = (TOOLS / "run_smolvla_trial.sh").read_text()
    assert "checkpoint: $SNAP" in src        # the trial log carries the exact directory benched
    assert "CAPSTONE_CHECKPOINT" in src      # and the usage tells the operator to set it


def test_prompt_resolver_prefers_the_local_dataset_copy_including_the_recorders_twin(tmp_path, monkeypatch):
    """A runner that must reach the Hub to learn its own prompt is a bench-day hazard (Sep 6: the
    full suite's copy of this call timed out under load). The v4 datasets live under the recorder's
    `<repo>_<YYYYMMDD>_<HHMMSS>` twins, which the exact-path lookup missed."""
    cache = tmp_path / "lerobot"
    twin = cache / "faithqin" / "so101-tube-insert-v4-noload_20260904_220924" / "meta"
    twin.mkdir(parents=True); (twin / "tasks.parquet").write_bytes(b"x")
    bak = cache / "faithqin" / "so101-tube-insert-v4-noload_20260904_220924.bak-task-1" / "meta"
    bak.mkdir(parents=True); (bak / "tasks.parquet").write_bytes(b"x")

    def no_hub(*a, **k):
        raise AssertionError("the Hub was consulted although a local copy exists")
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_hub)
    got = sd.tasks_parquet_path("faithqin/so101-tube-insert-v4-noload", cache=cache)
    assert got == twin / "tasks.parquet"                       # mutation: exact path only -> Hub
    exact = cache / "faithqin" / "so101-tube-insert-v4-noload" / "meta"
    exact.mkdir(parents=True); (exact / "tasks.parquet").write_bytes(b"x")
    assert sd.tasks_parquet_path("faithqin/so101-tube-insert-v4-noload", cache=cache) == exact / "tasks.parquet"
