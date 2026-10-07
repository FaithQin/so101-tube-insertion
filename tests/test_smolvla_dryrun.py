"""Regression: the SmolVLA offline dry-run must mirror the REAL call site.

Written Sep 3 2026 (evening), the night before the first SmolVLA bench block.

Why this file exists
--------------------
`tools/pi05_dryrun.py` PASSED while the production path was about to feed pi0.5
an EMPTY language prompt. It passed because it built its own pipeline: it
assembled an observation dict by hand, put the trained task string straight into
it, and called `pre(obs)`. The bench path does something different --
`build_dataset_frame` -> `prepare_observation_for_inference` ->
`obs_batch["task"] = [task]` -> preprocessor -- and the task it carries comes
from `cfg.dataset.single_task`, which nobody was passing. A rehearsal that
builds its own pipeline cannot catch a bug in the pipeline it did not build.

So the rule this file enforces is: **mirror the real path literally.** Every
assertion here is about the dry-run agreeing with
`site-packages/lerobot/rollout/context.py` and
`site-packages/lerobot/rollout/inference/rtc.py`, not about it being
self-consistent.

The three things that would break the Sep 4 block, all pinned here:

1. **An empty prompt.** SmolVLA is language-conditioned exactly like pi0.5.
   `rollout/configs.py:244` defaults `task` to `""`, and `context.py:562` reads
   `cfg.dataset.single_task if cfg.dataset else cfg.task`. The dry-run and the
   runner must resolve the SAME string from the dataset, and the dry-run must
   prove it survives tokenization.
2. **A missing rename_map.** `RolloutConfig.rename_map` defaults to an EMPTY
   dict (`rollout/configs.py:259`), and `context.py:537-546` passes it as
   `preprocessor_overrides["rename_observations_processor"]["rename_map"]` --
   which REPLACES the map saved with the checkpoint. Omit `--rename_map` and
   the checkpoint's own correct map is overwritten with `{}`, the policy's
   declared `camera1`/`camera2` never appear, and SmolVLA runs on padded
   towers.
3. **The wrong state width.** `smolvla-tube-B-v3` declares `[6]` and carries
   18-dim normalizer stats. The live routing in `context.py:400-407` reads the
   DECLARED width, so B must be launched with `CAPSTONE_FORCE_WIDE_STATE=1`;
   its A-side twin declares the same `[6]` and genuinely IS 6-dim, so A must be
   pinned to `0`. One source of truth for that mapping, shared by the runner,
   the dry-run and this file.

Pure functions and metadata only here. Nothing in tests/ loads the 868 MB
checkpoints -- the model path is exercised by RUNNING the tool.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import TOOLS, V3_DATASET_ROOT

sys.path.insert(0, str(TOOLS))

import smolvla_dryrun as sd  # noqa: E402

HUB = Path.home() / ".cache/huggingface/hub"


def _train_config(side: str) -> dict:
    """The checkpoint's OWN train_config.json, from the Hub cache."""
    snaps = sorted((HUB / f"models--faithqin--smolvla-tube-{side}-v3" / "snapshots").glob("*"))
    if not snaps:
        pytest.skip(f"smolvla-tube-{side}-v3 not in the Hub cache")
    cfg = snaps[-1] / "train_config.json"
    if not cfg.exists():
        pytest.skip(f"train_config.json not cached for smolvla-tube-{side}-v3")
    return json.loads(cfg.read_text())


# ---------------------------------------------------------------------------
# 1. Side metadata is the single source of truth for the force pin
# ---------------------------------------------------------------------------


def test_a_side_is_pinned_narrow_and_b_side_wide():
    """The measurement, from tools/check_state_width.py's header (Sep 3 2026):

        smolvla-tube-A-v3   declared [6]   max 32   stats [6]    -> narrow
        smolvla-tube-B-v3   declared [6]   max 32   stats [18]   -> WIDE

    Both declare 6, so only the pin separates them.
    """
    assert sd.SIDES["A"].force == "0"
    assert sd.SIDES["B"].force == "1"


def test_side_state_widths_match_the_normalizer_measurement():
    assert sd.SIDES["A"].state_dim == 6
    assert sd.SIDES["B"].state_dim == 18


def test_sides_point_at_the_v3_twins_not_the_v2_or_b16_runs():
    assert sd.SIDES["A"].repo_id == "faithqin/smolvla-tube-A-v3"
    assert sd.SIDES["B"].repo_id == "faithqin/smolvla-tube-B-v3"


def test_each_side_names_the_dataset_it_actually_trained_on():
    """Read from the checkpoint's own train_config.json, never from this note."""
    for name, side in sd.SIDES.items():
        trained_on = _train_config(name)["dataset"]["repo_id"]
        assert side.dataset == trained_on, (
            f"side {name} claims dataset {side.dataset!r} but the checkpoint "
            f"trained on {trained_on!r}"
        )


# ---------------------------------------------------------------------------
# 2. The camera map comes from the checkpoint, not from a note
# ---------------------------------------------------------------------------


def test_rename_map_matches_the_checkpoints_own_train_config():
    """The handoff said 'front->camera1, wrist->camera2 -- verify it'.

    Verified here against the map the training run actually used. A swapped map
    runs fine, produces actions, and is silently wrong.
    """
    for name in ("A", "B"):
        assert sd.RENAME_MAP == _train_config(name)["rename_map"], (
            f"dry-run rename_map disagrees with smolvla-tube-{name}-v3's training map"
        )


def test_rename_map_targets_the_keys_the_policy_declares():
    for name in ("A", "B"):
        declared = set(_train_config(name)["policy"]["input_features"])
        for dst in sd.RENAME_MAP.values():
            assert dst in declared, f"{dst} is not an input feature of smolvla-tube-{name}-v3"


# ---------------------------------------------------------------------------
# 3. The task string -- the bug that shipped in the pi0.5 path
# ---------------------------------------------------------------------------


def test_trained_task_is_non_empty_and_read_from_the_dataset():
    for name, side in sd.SIDES.items():
        try:
            task = sd.trained_task(side.dataset)
        except sd.TaskUnavailable:
            pytest.skip(f"{side.dataset} meta/tasks.parquet not reachable")
        assert task, f"side {name} resolved an EMPTY task string"
        assert task == "Pick up the test tube and insert it into the rack"


def test_trained_task_matches_the_datasets_parquet_read_independently():
    """Pin to the parquet, not to a literal, so it cannot go stale."""
    import pandas as pd

    parquet = V3_DATASET_ROOT / "meta" / "tasks.parquet"
    if not parquet.exists():
        pytest.skip("v3 dataset cache not present")
    trained = list(pd.read_parquet(parquet).index)[0]
    try:
        assert sd.trained_task(sd.SIDES["B"].dataset) == trained
    except sd.TaskUnavailable:
        pytest.skip("B-side dataset meta/tasks.parquet not reachable")


def test_an_empty_task_is_refused_rather_than_passed_through():
    """The pi0.5 failure mode was an empty prompt that nothing objected to."""
    with pytest.raises(sd.TaskUnavailable):
        sd.require_task("")
    with pytest.raises(sd.TaskUnavailable):
        sd.require_task("   ")
    assert sd.require_task("Pick up the test tube") == "Pick up the test tube"


def test_prompt_check_rejects_an_empty_or_mismatched_decoded_prompt():
    """Proves the assertion the dry-run makes on the real preprocessed batch."""
    task = "Pick up the test tube and insert it into the rack"
    # Tokenizer round-trips add BOS/EOS/newline padding -- containment, not equality.
    assert sd.prompt_survived_tokenization("<bos>" + task + "\n<eos>", task)
    assert not sd.prompt_survived_tokenization("", task)
    assert not sd.prompt_survived_tokenization("<bos>Task: <eos>", task)
    assert not sd.prompt_survived_tokenization("<bos>put the block in the box<eos>", task)


# ---------------------------------------------------------------------------
# 4. State routing -- the live rule, not the reference one
# ---------------------------------------------------------------------------


def test_hw_observation_features_match_what_the_patched_follower_publishes():
    """18 scalars + 2 cameras, taken from the v3 recording's own info.json.

    That file was written by `hw_to_dataset_features(robot.observation_features)`
    during recording, with this exact patched follower -- so it is a measurement
    of the bench, not a guess. (Window B must never open the serial port to ask.)
    """
    info = json.loads((V3_DATASET_ROOT / "meta" / "info.json").read_text())
    recorded = info["features"]["observation.state"]["names"]
    feats = sd.hw_observation_features()
    scalars = [k for k, v in feats.items() if v is float]
    assert scalars == recorded
    assert feats["front"] == tuple(info["features"]["observation.images.front"]["shape"])
    assert feats["wrist"] == tuple(info["features"]["observation.images.wrist"]["shape"])


def test_narrow_pin_routes_six_pos_keys_and_drops_load_and_current():
    keys = sd.routed_state_keys("0")
    assert len(keys) == 6
    assert all(k.endswith(".pos") for k in keys)
    assert not any(k.endswith((".load", ".current")) for k in keys)


def test_wide_pin_routes_all_eighteen_in_recorded_order():
    keys = sd.routed_state_keys("1")
    info = json.loads((V3_DATASET_ROOT / "meta" / "info.json").read_text())
    assert keys == info["features"]["observation.state"]["names"]


def test_each_sides_pin_produces_the_width_its_normalizer_expects():
    """The whole point of the pin: B's 18-dim normalizer must get 18 numbers."""
    for name, side in sd.SIDES.items():
        assert len(sd.routed_state_keys(side.force)) == side.state_dim, (
            f"side {name} pinned to {side.force} routes the wrong state width"
        )


def test_routing_ignores_the_declared_width_exactly_like_the_live_code():
    """`context.py:401` checks `_force` FIRST. Both twins declare 6; the pin is
    the only thing that separates them, so a declared width of 6 must not be
    able to narrow the B side."""
    assert len(sd.routed_state_keys("1", policy_state_dim=6, max_state_dim=32)) == 18
    assert len(sd.routed_state_keys("0", policy_state_dim=6, max_state_dim=32)) == 6


def test_an_unparseable_pin_raises_rather_than_inferring():
    """Matches `context.py:395-398`: a typo must not silently become 'narrow'."""
    with pytest.raises(ValueError):
        sd.routed_state_keys("wide")


# ---------------------------------------------------------------------------
# 5. The observation the policy sees -- built the way rtc.py builds it
# ---------------------------------------------------------------------------


def _fake_raw_observation():
    """A robot-shaped observation: named scalars + HWC uint8 camera frames."""
    values = {k: float(i) for i, (k, v) in enumerate(sd.hw_observation_features().items()) if v is float}
    values["front"] = np.zeros((8, 12, 3), dtype=np.uint8)
    values["wrist"] = np.zeros((6, 8, 3), dtype=np.uint8)
    return values


def test_rollout_observation_carries_the_task_as_a_list_like_rtc_does():
    """`rtc.py:301` sets `obs_batch["task"] = [self._task]` AFTER
    `prepare_observation_for_inference`. Mirror the list, not the bare string."""
    task = "Pick up the test tube and insert it into the rack"
    batch = sd.build_rollout_observation(_fake_raw_observation(), force="1", task=task, device="cpu")
    assert batch["task"] == [task]
    assert batch["robot_type"] == sd.ROBOT_TYPE


def test_rollout_observation_state_width_follows_the_pin():
    task = "t"
    wide = sd.build_rollout_observation(_fake_raw_observation(), force="1", task=task, device="cpu")
    narrow = sd.build_rollout_observation(_fake_raw_observation(), force="0", task=task, device="cpu")
    assert wide["observation.state"].shape[-1] == 18
    assert narrow["observation.state"].shape[-1] == 6


def test_rollout_observation_keeps_raw_camera_names_for_the_preprocessor():
    """The rename is done by `rename_observations_processor` inside the
    preprocessor, exactly as on the bench. Renaming here as well would rename
    twice and leave the policy with no images at all."""
    batch = sd.build_rollout_observation(_fake_raw_observation(), force="0", task="t", device="cpu")
    assert "observation.images.front" in batch
    assert "observation.images.wrist" in batch
    for dst in sd.RENAME_MAP.values():
        assert dst not in batch


def test_rollout_observation_images_are_channel_first_and_scaled():
    """`prepare_observation_for_inference` divides uint8 by 255 and permutes to
    CHW. If the dry-run hands the policy HWC uint8, normalization differs from
    the bench and the verdict is meaningless."""
    batch = sd.build_rollout_observation(_fake_raw_observation(), force="0", task="t", device="cpu")
    front = batch["observation.images.front"]
    assert front.shape == (1, 3, 8, 12)
    assert front.dtype.is_floating_point
    assert float(front.max()) <= 1.0


def test_missing_camera_raises_instead_of_being_padded_away():
    """SmolVLA masks absent image towers rather than raising: it would run blind
    at full speed. The dry-run must stop instead."""
    values = _fake_raw_observation()
    del values["wrist"]
    with pytest.raises(KeyError):
        sd.build_rollout_observation(values, force="0", task="t", device="cpu")


# ---------------------------------------------------------------------------
# 6. Scoring guards (a dry-run that cannot fail is not an arbiter)
# ---------------------------------------------------------------------------


def test_degenerate_chunks_are_rejected():
    ok, _ = sd.check_chunk_sane(np.tile(np.arange(6.0), (10, 1)), 6)
    assert not ok, "a constant-over-time chunk must not pass"
    ok, _ = sd.check_chunk_sane(np.full((10, 6), np.nan), 6)
    assert not ok
    ok, _ = sd.check_chunk_sane(np.random.RandomState(0).randn(10, 6), 6)
    assert ok


def test_normalized_error_scales_by_each_joints_own_spread():
    pred = np.array([1.0, 1.0])
    truth = np.array([0.0, 0.0])
    assert sd.normalized_error(pred, truth, np.array([100.0, 100.0])) == pytest.approx(0.01)
    assert sd.normalized_error(pred, truth, np.array([1.0, 1.0])) == pytest.approx(1.0)


def test_threshold_is_pre_declared_and_matches_the_pi05_rehearsal():
    """Same bar as pi0.5's, declared before any number came back, so the two
    architectures are judged identically."""
    import pi05_dryrun

    assert sd.AGREEMENT_THRESHOLD == pi05_dryrun.AGREEMENT_THRESHOLD


# ---------------------------------------------------------------------------
# 7. Preprocessor overrides -- ALL of them, including the launcher's
# ---------------------------------------------------------------------------


class _Cfg:
    vlm_model_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


def test_overrides_cover_device_rename_and_tokenizer():
    """The bench applies three, not two.

    `context.py:537-546` supplies device and rename_map;
    `tools/rollout_30hz_stale_ok.py:293-301` injects the tokenizer name because
    a 0.6.2-saved SmolVLA checkpoint records its tokenizer as the relative
    folder "tokenizer", which local 0.6.1 cannot resolve. Measured Sep 3 2026:
    without it `make_pre_post_processors` raises outright, so a rehearsal that
    skips it is not rehearsing a path the bench can take.
    """
    ov = sd.preprocessor_overrides(_Cfg(), "cpu")
    assert set(ov) == {"device_processor", "rename_observations_processor", "tokenizer_processor"}
    assert ov["device_processor"] == {"device": "cpu"}
    assert ov["rename_observations_processor"] == {"rename_map": sd.RENAME_MAP}
    assert ov["tokenizer_processor"] == {"tokenizer_name": _Cfg.vlm_model_name}


def test_tokenizer_override_uses_the_checkpoints_own_vlm_name():
    """Read from the checkpoint, not from this note."""
    for name in ("A", "B"):
        vlm = _train_config(name)["policy"]["vlm_model_name"]
        cfg = type("C", (), {"vlm_model_name": vlm})()
        assert sd.preprocessor_overrides(cfg, "mps")["tokenizer_processor"] == {
            "tokenizer_name": vlm
        }


def test_no_tokenizer_override_when_the_policy_has_no_vlm():
    """ACT has no `vlm_model_name`; inventing the key would break its pipeline."""
    assert "tokenizer_processor" not in sd.preprocessor_overrides(object(), "cpu")
