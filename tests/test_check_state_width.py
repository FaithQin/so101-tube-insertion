"""A checkpoint whose declared state width contradicts its own normalizer must
abort BEFORE the arm moves.

Written Sep 3 2026 after the night audit found that
`faithqin/smolvla-tube-B-v3` declares `observation.state.shape = [6]` with
`max_state_dim = 32` while carrying 18-dim normalizer statistics. The live
routing logic in `rollout/context.py` reads the DECLARED width, so it routes
narrow, and a 6-dim state meets an 18-dim normalizer: RuntimeError on the first
inference tick, mid-episode, on a gated bench day.

Its A-side twin declares the same [6] and genuinely is 6-dim, so neither the
declaration, the padding width, nor the policy family separates them. Only the
stats do.

This gate is deliberately standalone: `preflight.py` takes scene/home/ping and
no policy path, and `run_scored_trial.sh` must not change the night before a
session (tests/test_pi05_trial_runner.py::test_act_runner_left_untouched).
"""

import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import check_state_width as csw  # noqa: E402

HUB = Path("~/.cache/huggingface/hub").expanduser()


def _make_ckpt(d: Path, declared, max_state_dim, stats_width):
    d.mkdir(parents=True, exist_ok=True)
    cfg = {"input_features": {"observation.state": {"shape": [declared]}}}
    if max_state_dim is not None:
        cfg["max_state_dim"] = max_state_dim
    (d / "config.json").write_text(json.dumps(cfg))
    if stats_width is not None:
        save_file(
            {"observation.state.mean": torch.zeros(stats_width),
             "observation.state.std": torch.ones(stats_width)},
            str(d / "policy_preprocessor_step_5_normalizer_processor.safetensors"),
        )
    return d


def test_reads_both_widths(tmp_path):
    d = _make_ckpt(tmp_path / "ck", declared=6, max_state_dim=32, stats_width=18)
    w = csw.read_widths(d)
    assert w["declared"] == 6 and w["max_state_dim"] == 32 and w["stats"] == 18


def test_consistent_checkpoint_is_ok(tmp_path):
    d = _make_ckpt(tmp_path / "a", declared=6, max_state_dim=32, stats_width=6)
    v = csw.evaluate(d, n_stock=6)
    assert v["disagrees"] is False
    assert v["widen"] is False
    assert v["required_force"] is None
    assert v["exit_code"] == 0


def test_smolvla_b_shape_is_flagged_and_names_the_fix(tmp_path):
    d = _make_ckpt(tmp_path / "b", declared=6, max_state_dim=32, stats_width=18)
    v = csw.evaluate(d, n_stock=6)
    assert v["disagrees"] is True
    assert v["widen"] is True, "stats say 18 > 6, so the state must be widened"
    assert v["required_force"] == "1", "operator must be told to pin =1"
    assert v["exit_code"] != 0, "a disagreement must abort, not warn"


def test_an_explicit_force_satisfies_the_gate(tmp_path):
    d = _make_ckpt(tmp_path / "c", declared=6, max_state_dim=32, stats_width=18)
    v = csw.evaluate(d, n_stock=6, force="1")
    assert v["exit_code"] == 0, "pinning the right value clears the gate"
    v_wrong = csw.evaluate(d, n_stock=6, force="0")
    assert v_wrong["exit_code"] != 0, "pinning the WRONG value must still abort"


def test_act_b_consistent_wide_checkpoint_passes(tmp_path):
    d = _make_ckpt(tmp_path / "d", declared=18, max_state_dim=None, stats_width=18)
    v = csw.evaluate(d, n_stock=6)
    assert v["disagrees"] is False and v["widen"] is True and v["exit_code"] == 0


def test_pi05_padded_checkpoint_passes_narrow(tmp_path):
    """declared 32 (padding) vs stats 6: a disagreement, but the stats say
    narrow, so the required pin is 0 -- and it must still abort rather than
    silently route, because the metadata is wrong either way."""
    d = _make_ckpt(tmp_path / "e", declared=32, max_state_dim=32, stats_width=6)
    v = csw.evaluate(d, n_stock=6)
    assert v["widen"] is False
    assert v["required_force"] == "0"


def test_missing_normalizer_cannot_silently_pass(tmp_path):
    """No stats to check against is unknown, not safe."""
    d = _make_ckpt(tmp_path / "f", declared=6, max_state_dim=32, stats_width=None)
    v = csw.evaluate(d, n_stock=6)
    assert v["stats"] is None
    assert v["exit_code"] != 0, "an unreadable normalizer must abort, not assume"


@pytest.mark.skipif(
    not list(HUB.glob("models--faithqin--smolvla-tube-B-v3/snapshots/*/config.json")),
    reason="smolvla-tube-B-v3 not in the local hub cache",
)
def test_against_the_real_smolvla_b_checkpoint():
    """Verify against reality, not fixtures."""
    snap = sorted(HUB.glob("models--faithqin--smolvla-tube-B-v3/snapshots/*"))[-1]
    v = csw.evaluate(snap, n_stock=6)
    assert v["declared"] == 6 and v["stats"] == 18
    assert v["disagrees"] is True and v["widen"] is True
    assert v["required_force"] == "1" and v["exit_code"] != 0


@pytest.mark.skipif(
    not list(HUB.glob("models--faithqin--smolvla-tube-A-v3/snapshots/*/config.json")),
    reason="smolvla-tube-A-v3 not in the local hub cache",
)
def test_the_a_side_twin_is_not_dragged_wide():
    snap = sorted(HUB.glob("models--faithqin--smolvla-tube-A-v3/snapshots/*"))[-1]
    v = csw.evaluate(snap, n_stock=6)
    assert v["declared"] == 6 and v["stats"] == 6
    assert v["disagrees"] is False and v["widen"] is False and v["exit_code"] == 0
