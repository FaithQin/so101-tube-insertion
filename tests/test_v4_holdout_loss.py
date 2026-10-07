"""The hold-out scorer gates before it scores, and its arithmetic is frame-weighted.

Every test here is hardware- and dataset-free: the heavy path (decode frames,
run the policy) is not exercised; the parts that decide WHAT gets scored and
HOW the numbers combine are. Each test names the mutation that turns it red.
"""
import json
import sys

import numpy as np
import pytest
import torch

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))
import v4_holdout_loss as hl  # noqa: E402


# --- what is on the Hub ------------------------------------------------------

def test_parse_checkpoint_steps_reads_only_full_checkpoints_and_dedups():
    files = [
        "README.md", "model.safetensors", "train_config.json",
        "checkpoints/010000/pretrained_model/model.safetensors",
        "checkpoints/010000/pretrained_model/config.json",
        "checkpoints/005000/pretrained_model/model.safetensors",
        "checkpoints/020000/pretrained_model/config.json",   # weights missing -> not scorable
        "checkpoints/last/pretrained_model/model.safetensors",  # not a step
    ]
    # mutation: matching config.json instead of model.safetensors adds 20000
    assert hl.parse_checkpoint_steps(files) == [5000, 10000]


# --- which episodes ----------------------------------------------------------

def test_holdout_is_kept_minus_train_and_train_must_be_a_subset_of_kept():
    kept = [0, 1, 2, 3, 5, 7]
    assert hl.holdout_from(kept, [0, 1, 5]) == [2, 3, 7]
    with pytest.raises(ValueError, match="not kept"):
        hl.holdout_from(kept, [0, 4])  # 4 was voided: a train list naming it is a label bug


def test_train_reference_is_type_matched_disjoint_and_seeded():
    types = {0: "CLEAN", 1: "CLEAN", 2: "NUDGE", 3: "CLEAN", 4: "NUDGE", 5: "RESIST",
             6: "CLEAN", 7: "RESIST", 8: "CLEAN", 9: "NUDGE"}
    holdouts = [2, 5, 8]           # NUDGE, RESIST, CLEAN
    train = [0, 1, 3, 4, 6, 7, 9]
    ref = hl.pick_train_reference(types.__getitem__, train, holdouts, seed=1)
    assert len(ref) == len(holdouts)
    # PAIRED: ref[i] has the type of sorted(holdouts)[i], so a truncated smoke run (--episodes-limit)
    # still compares like with like. mutation: sort the reference by episode index
    assert [types[e] for e in ref] == [types[e] for e in sorted(holdouts)]
    assert not set(ref) & set(holdouts)
    assert all(e in train for e in ref)
    assert ref == hl.pick_train_reference(types.__getitem__, train, holdouts, seed=1)
    with pytest.raises(ValueError, match="no training episode of type"):
        hl.pick_train_reference(types.__getitem__, [0, 1], [2], seed=1)  # no NUDGE in train


def test_episode_frame_indices_honour_membership_and_stride():
    col = [3, 3, 3, 3, 3, 7, 7, 7, 3]   # episode 3 has frames at 0-4 and (out of order) 8
    assert hl.episode_frame_indices(col, 3, stride=1) == [0, 1, 2, 3, 4, 8]
    assert hl.episode_frame_indices(col, 3, stride=2) == [0, 2, 4]      # mutation: stride on raw index
    assert hl.episode_frame_indices(col, 7, stride=1) == [5, 6, 7]
    assert hl.episode_frame_indices(col, 9, stride=1) == []


# --- the gates ---------------------------------------------------------------

def _consistent():
    train = [0, 1, 3, 5]
    return dict(
        train_cfg={"dataset": {"repo_id": "faithqin/ds", "episodes": train}},
        dataset_repo="faithqin/ds",
        split_train=train,
        holdouts=[2, 4],
        local_info={"total_episodes": 6, "total_frames": 1200},
        hub_info={"total_episodes": 6, "total_frames": 1200},
    )


def test_gate_passes_on_consistent_inputs():
    assert hl.gate(**_consistent()) == []


def test_gate_names_every_failure_mode():
    k = _consistent(); k["train_cfg"]["dataset"]["repo_id"] = "faithqin/other"
    assert any("repo_id" in p for p in hl.gate(**k))

    k = _consistent(); k["train_cfg"]["dataset"]["episodes"] = [0, 1, 3]
    assert any("dataset.episodes" in p for p in hl.gate(**k))

    k = _consistent(); k["holdouts"] = [2, 3]           # 3 is in the training list
    assert any("overlap" in p for p in hl.gate(**k))

    k = _consistent(); k["local_info"]["total_frames"] = 1199
    assert any("local dataset" in p for p in hl.gate(**k))


# --- the arithmetic ----------------------------------------------------------

def test_frame_weighted_mean_weights_by_frames():
    # plain mean of [1, 3] is 2; frame-weighted with 1 vs 3 frames is 2.5
    assert hl.frame_weighted_mean([(1.0, 1), (3.0, 3)]) == pytest.approx(2.5)
    with pytest.raises(ValueError):
        hl.frame_weighted_mean([])


def test_images_to_float_scales_uint8_cameras_only():
    batch = {
        "observation.images.front": torch.full((2, 3, 4, 4), 255, dtype=torch.uint8),
        "observation.images.wrist": torch.zeros((2, 3, 4, 4), dtype=torch.float32),
        "observation.state": torch.full((2, 6), 255, dtype=torch.uint8),
    }
    out = hl.images_to_float(batch, ["observation.images.front", "observation.images.wrist"])
    assert out["observation.images.front"].dtype == torch.float32
    assert float(out["observation.images.front"].max()) == pytest.approx(1.0)  # mutation: drop /255
    assert out["observation.images.wrist"].dtype == torch.float32
    assert out["observation.state"].dtype == torch.uint8                     # not a camera


def test_verdict_flags_a_holdout_minimum_before_the_last_checkpoint():
    curve = [
        {"step": 10000, "holdout": 0.100, "train_ref": 0.090},
        {"step": 20000, "holdout": 0.080, "train_ref": 0.060},
        {"step": 30000, "holdout": 0.085, "train_ref": 0.040},
        {"step": 40000, "holdout": 0.090, "train_ref": 0.030},
    ]
    v = hl.verdict(curve, rise_frac=0.05)
    assert v["best_step"] == 20000 and v["last_step"] == 40000
    assert v["overfit"] is True                    # 0.090 >= 0.080 * 1.05
    assert v["rise_frac"] == pytest.approx(0.125)  # mutation: absolute instead of relative

    flat = [dict(r, holdout=0.080) for r in curve]  # no rise at all
    assert hl.verdict(flat, rise_frac=0.05)["overfit"] is False

    small = [{"step": 1, "holdout": 0.080, "train_ref": 0.05},
             {"step": 2, "holdout": 0.083, "train_ref": 0.04}]   # +3.75% < 5%
    assert hl.verdict(small, rise_frac=0.05)["overfit"] is False
    assert hl.verdict(small, rise_frac=0.05)["best_step"] == 1


# --- loading exactly what training saved -------------------------------------

class _Cfg:
    def __init__(self, type_, vlm=None):
        self.type = type_
        if vlm is not None:
            self.vlm_model_name = vlm


def test_preprocessor_overrides_mirror_the_bench():
    rename = {"observation.images.front": "observation.images.camera1"}
    o = hl.preprocessor_overrides(_Cfg("smolvla", vlm="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"),
                                  rename, "mps")
    assert o["device_processor"] == {"device": "mps"}
    assert o["rename_observations_processor"] == {"rename_map": rename}
    assert o["tokenizer_processor"] == {"tokenizer_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"}
    o = hl.preprocessor_overrides(_Cfg("act"), {}, "mps")
    assert "tokenizer_processor" not in o          # mutation: always add the override


def test_resolve_local_root_prefers_exact_then_newest_twin_and_skips_backups(tmp_path):
    cache = tmp_path / "lerobot"
    def mk(name, eps):
        d = cache / "faithqin" / name / "meta"; d.mkdir(parents=True)
        (d / "info.json").write_text(json.dumps({"total_episodes": eps}))
        return d.parent
    assert hl.resolve_local_root("faithqin/ds", cache) is None
    old = mk("ds_20260901_000000", 20)
    bak = mk("ds_20260904_000000.bak-task-20260905_114542", 20)
    new = mk("ds_20260904_000000", 64)
    assert hl.resolve_local_root("faithqin/ds", cache) == new      # newest timestamp, not the .bak
    assert bak != new and old != new
    exact = mk("ds", 64)
    assert hl.resolve_local_root("faithqin/ds", cache) == exact    # exact name wins


def test_report_table_lists_every_checkpoint_and_the_verdict():
    curve = [
        {"step": 10000, "holdout": 0.100, "train_ref": 0.090, "per_type": {"CLEAN": 0.1}},
        {"step": 20000, "holdout": 0.080, "train_ref": 0.060, "per_type": {"CLEAN": 0.08}},
    ]
    text = hl.report_table("faithqin/act-tube-A-v4", curve, hl.verdict(curve, 0.05))
    assert "10000" in text and "20000" in text and "0.080" in text
    assert "best hold-out" in text
    # gap = hold-out minus train-ref: POSITIVE means the policy does worse on unseen episodes
    assert "+0.0100" in text and "hold-ref" in text   # mutation: ref-hold


# --- resuming a partial run --------------------------------------------------

def _settings(**over):
    s = {"repo": "faithqin/pi05-tube-A-v4", "dataset": "faithqin/ds", "stride": 6, "seed": 1,
         "holdout": [11, 19], "train_ref": [29, 12]}
    s.update(over)
    return s


def test_merge_results_keeps_scored_steps_and_orders_the_union():
    existing = dict(_settings(), checkpoints=[{"step": 10000, "holdout": 0.2}, {"step": 5000, "holdout": 0.1}])
    merged = hl.merge_results(existing, _settings(), [{"step": 20000, "holdout": 0.3}])
    assert [c["step"] for c in merged["checkpoints"]] == [5000, 10000, 20000]
    assert hl.steps_to_skip(existing, [5000, 10000, 20000]) == [5000, 10000]   # mutation: skip nothing


def test_merge_results_refuses_a_different_run():
    existing = dict(_settings(), checkpoints=[{"step": 5000, "holdout": 0.1}])
    for bad in (dict(stride=3), dict(seed=2), dict(holdout=[11, 25]), dict(repo="faithqin/other")):
        with pytest.raises(ValueError, match="settings differ"):
            hl.merge_results(existing, _settings(**bad), [])
    # a re-scored step replaces the old row rather than duplicating it
    merged = hl.merge_results(existing, _settings(), [{"step": 5000, "holdout": 0.15}])
    assert merged["checkpoints"] == [{"step": 5000, "holdout": 0.15}]
