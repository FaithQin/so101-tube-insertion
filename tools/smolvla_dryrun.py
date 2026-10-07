#!/usr/bin/env python
"""Rehearse the SmolVLA rollout at a desk, on the REAL call site, before 10:00.

Usage
-----
    python tools/smolvla_dryrun.py --side A          # 6 frames, default seed
    python tools/smolvla_dryrun.py --side B --n 8 --json-out b.json
    python tools/smolvla_dryrun.py --print task --side B     # used by the runner

Why this is not `pi05_dryrun.py` with the names changed
------------------------------------------------------
`tools/pi05_dryrun.py` PASSED on Aug 31 while the production path was about to
hand pi0.5 an EMPTY language prompt, and the reason is structural, not a slip:
the rehearsal built its OWN pipeline. It assembled

    obs = {"observation.images.front": ..., "observation.state": ..., "task": TASK}
    batch = pre(obs)

with `TASK` a python constant, and never touched the two functions the bench
actually calls. The bench builds its observation like this instead
(`rollout/inference/rtc.py:297-303`):

    obs_batch = build_dataset_frame(hw_features, obs, prefix="observation")
    obs_batch = prepare_observation_for_inference(obs_batch, device, task, robot_type)
    obs_batch["task"] = [task]
    preprocessed = preprocessor(obs_batch)
    actions = policy.predict_action_chunk(preprocessed, inference_delay=delay,
                                          prev_chunk_left_over=prev)

and `task` there is `cfg.dataset.single_task if cfg.dataset else cfg.task`
(`rollout/context.py:562`) -- which nobody was passing, so it was `""`. A
rehearsal that builds its own pipeline cannot catch a bug in the pipeline it did
not build.

So this file mirrors the real call site LITERALLY:

* the same `build_dataset_frame` / `prepare_observation_for_inference` pair, fed
  a robot-shaped observation (named scalars + HWC uint8 frames), not a
  pre-assembled policy batch;
* the same `make_pre_post_processors(..., pretrained_path=...)` with all three
  overrides the bench applies -- device and rename_map from
  `context.py:537-546`, plus the tokenizer_name injection from
  `tools/rollout_30hz_stale_ok.py:293-301` without which processor
  construction raises outright under local lerobot 0.6.1;
* the same RTC wiring `context.py:259-268` performs, and the same first-tick
  call shape -- SYNC since Sep 4: `select_action(batch)` inside
  torch.inference_mode(), sync.py:125-131 (was RTC's predict_action_chunk);
* the task string resolved from `meta/tasks.parquet` by the SAME function
  `run_smolvla_trial.sh` calls, so the runner and the rehearsal cannot disagree
  about the prompt -- which is exactly how pi0.5 slipped through.

And it ASSERTS, per probe frame, that the prompt survived tokenization: the
`observation.language.tokens` tensor is decoded with the checkpoint's own
tokenizer and must still contain the trained sentence. An empty prompt fails
here instead of on the bench.

What a pass means
-----------------
Only that observations reach SmolVLA the way training did. It says nothing about
whether SmolVLA can do the task. A FAIL is decisive: do not go to the bench.

Hardware: none. This file never opens a serial port or a camera. The bench
observation contract is DECLARED below and verified against the v3 recording's
own `meta/info.json` in `tests/test_smolvla_dryrun.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from state_routing import force_from_env  # noqa: E402

# lerobot.utils.constants.ACTION. Spelled literally so this module stays
# importable (and its gate testable) without pulling in lerobot.
ACTION_KEY = "action"

# --- what the bench publishes ---------------------------------------------
# Straight from the checkpoints' own `train_config.json` (both twins agree).
# NOT reconstructed by hand: swapping front and wrist runs fine and is silently
# wrong. Load-bearing at rollout because `RolloutConfig.rename_map` defaults to
# an empty dict (`rollout/configs.py:259`) and `context.py:537-546` uses it to
# OVERRIDE the map saved with the checkpoint -- omitting `--rename_map` replaces
# a correct map with `{}`.
RENAME_MAP = {
    "observation.images.front": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2",
}

# `SOFollower.name`; what `robot_wrapper.robot_type` returns on the bench and
# what the v3 recording carries as `robot_type`.
ROBOT_TYPE = "so_follower"

# The patched follower's `observation_features`: 6 .pos, then 6 .load, then 6
# .current (see the `_load_ft` local patch), then the cameras. Declared here
# rather than read from a dataset so the test can CHECK it against the v3
# recording's `meta/info.json` -- reading it from that file would make the test
# circular, and asking the robot would mean opening the serial port.
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CAMERA_SHAPES = {"front": (720, 1280, 3), "wrist": (480, 640, 3)}

# The 18-dim recording IS the raw robot observation; `-noload` is byte-identical
# to it with the load/current columns dropped (verified Sep 3 2026: first six
# state dims and all actions equal across all 10,479 frames). So one dataset
# supplies probe frames for both sides and the routing filter does the
# narrowing, exactly as it does on the arm.
PROBE_DATASET = "faithqin/so101-tube-insert-v3"
PROBE_DATASET_ROOT = (
    Path.home() / ".cache/huggingface/lerobot/faithqin/so101-tube-insert-v3_20260827_110242"
)

# Same bar as the pi0.5 rehearsal, pre-declared so it cannot be loosened once a
# number comes back, and identical across architectures so the four-way
# comparison is judged the same way.
AGREEMENT_THRESHOLD = 0.25


@dataclass(frozen=True)
class Side:
    """One arm of the SmolVLA A/B, and the pin it must be launched with.

    Measured Sep 3 2026 (`tools/check_state_width.py` header):

        smolvla-tube-A-v3   declared [6]   max 32   stats [6]    -> narrow
        smolvla-tube-B-v3   declared [6]   max 32   stats [18]   -> WIDE

    BOTH declare 6. The live routing (`context.py:400-407`) reads the declared
    width, so B routes narrow by inference and a 6-dim state meets an 18-dim
    normalizer: RuntimeError on the first tick. `_force` is checked first
    (context.py:401), so the pin -- and only the pin -- separates the twins.
    """

    name: str
    repo_id: str
    dataset: str  # the dataset THIS side trained on (train_config.json)
    force: str  # the CAPSTONE_FORCE_WIDE_STATE value to launch with
    state_dim: int  # width its normalizer statistics expect
    gen: str = "v3"  # policy generation: which repos/datasets, and whether a checkpoint must be chosen


# The Sep 3-4 trials' table, kept verbatim: tests pin it and the v3 rows in the trial CSV
# were produced with it. Select it with CAPSTONE_POLICY_GEN=v3.
SIDES = {
    "A": Side("A", "faithqin/smolvla-tube-A-v3", "faithqin/so101-tube-insert-v3-noload", "0", 6),
    "B": Side("B", "faithqin/smolvla-tube-B-v3", "faithqin/so101-tube-insert-v3", "1", 18),
}

# v4 (Sep 5): the study's policies. v4-B declares observation.state [6] and carries 18-dim
# normalizer statistics exactly like v3-B (verified on the Hub, Sep 6), so the pin is unchanged.
SIDES_V4 = {
    "A": Side("A", "faithqin/smolvla-tube-A-v4", "faithqin/so101-tube-insert-v4-noload", "0", 6, "v4"),
    "B": Side("B", "faithqin/smolvla-tube-B-v4", "faithqin/so101-tube-insert-v4", "1", 18, "v4"),
}

GENERATIONS = {"v3": SIDES, "v4": SIDES_V4}


def policy_gen(env=None) -> str:
    env = os.environ if env is None else env
    gen = env.get("CAPSTONE_POLICY_GEN", "v4")
    if gen not in GENERATIONS:
        raise ValueError(f"CAPSTONE_POLICY_GEN={gen!r}; expected one of {sorted(GENERATIONS)}")
    return gen


def active_sides(env=None) -> dict:
    """The side table the runner and the rehearsal both use: v4 unless CAPSTONE_POLICY_GEN=v3."""
    return GENERATIONS[policy_gen(env)]


class CheckpointUnavailable(RuntimeError):
    """No checkpoint was chosen for a generation that requires the choice, or it is not on the Hub."""


def checkpoint_from_env(env=None, gen: str = "v4"):
    """Which weights to bench: a zero-padded step ("005000"), or None for the top-level (final) weights.

    For a v4 side an UNSET value aborts: `tools/v4_holdout_loss.py` (Sep 5) showed both SmolVLA-v4
    runs overfit by 20k, and the top-level weights ARE the 20k checkpoint. The operator states the
    choice -- `CAPSTONE_CHECKPOINT=005000`, or `=top` to bench the final weights deliberately.
    v3 trials were run on the final weights before any hold-out existed; unset keeps that behaviour.
    """
    env = os.environ if env is None else env
    raw = (env.get("CAPSTONE_CHECKPOINT") or "").strip()
    if not raw:
        if gen == "v4":
            raise CheckpointUnavailable(
                "CAPSTONE_CHECKPOINT is unset. The v4 hold-out verdict (the lab notebook (private), Sep 5) is "
                "'bench checkpoints/005000, not the final weights' -- set CAPSTONE_CHECKPOINT=005000, "
                "or CAPSTONE_CHECKPOINT=top to bench the top-level (20k) weights deliberately"
            )
        return None
    if raw == "top":
        return None
    if not raw.isdigit():
        raise CheckpointUnavailable(f"CAPSTONE_CHECKPOINT={raw!r} is not a step (e.g. 005000) or 'top'")
    return f"{int(raw):06d}"


class TaskUnavailable(RuntimeError):
    """The trained prompt could not be resolved, or resolved to nothing."""


# ---------------------------------------------------------------------------
# The prompt — the bug that shipped in the pi0.5 path
# ---------------------------------------------------------------------------


def require_task(task: str | None) -> str:
    """Refuse an empty prompt instead of passing it along.

    `prepare_observation_for_inference` (policies/utils.py:136) writes
    `observation["task"] = task if task else ""` without complaint, and
    SmolVLA's tokenizer step happily tokenizes the empty string. Nothing
    downstream objects, which is precisely why pi0.5 was about to run blind on
    language.
    """
    if task is None or not str(task).strip():
        raise TaskUnavailable(
            "resolved an EMPTY task string; SmolVLA is language-conditioned and "
            "an empty prompt is a silent distribution shift, not a crash"
        )
    return str(task)


def tasks_parquet_path(repo_id: str, cache: Path | None = None) -> Path:
    """Locate a dataset's `meta/tasks.parquet`: the local copy first -- the exact `<cache>/<repo>`
    dir, else the NEWEST `<repo>_<YYYYMMDD>_<HHMMSS>` twin the recorder writes (never a `.bak-*`
    copy) -- and only then the Hub. A runner that needs the network to learn its own prompt is a
    bench-day hazard (Sep 6: the full suite's copy of this lookup timed out under load)."""
    cache = Path.home() / ".cache/huggingface/lerobot" if cache is None else Path(cache)
    local = cache / repo_id / "meta" / "tasks.parquet"
    if local.exists():
        return local
    parent, name = (cache / repo_id).parent, Path(repo_id).name
    twins = [d for d in parent.glob(f"{name}_*")
             if re.fullmatch(rf"{re.escape(name)}_\d{{8}}_\d{{6}}", d.name) and (d / "meta" / "tasks.parquet").exists()]
    if twins:
        return max(twins, key=lambda d: d.name) / "meta" / "tasks.parquet"
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - huggingface_hub is a hard dep
        raise TaskUnavailable(f"huggingface_hub unavailable: {exc}") from exc
    for offline in (True, False):
        try:
            return Path(
                hf_hub_download(
                    repo_id, "meta/tasks.parquet", repo_type="dataset", local_files_only=offline
                )
            )
        except Exception:  # noqa: BLE001 - any failure means "try the next source"
            continue
    raise TaskUnavailable(f"no meta/tasks.parquet reachable for {repo_id}")


def trained_task(dataset_repo_id: str) -> str:
    """The exact prompt a policy trained on, read from the dataset.

    Read, never typed. A literal in the runner goes stale the day the dataset
    changes and nothing notices until the numbers come out wrong.
    """
    import pandas as pd

    path = tasks_parquet_path(dataset_repo_id)
    tasks = list(pd.read_parquet(path).index)
    if not tasks:
        raise TaskUnavailable(f"{path} lists no tasks")
    return require_task(tasks[0])


def prompt_survived_tokenization(decoded: str, task: str) -> bool:
    """Did the trained sentence actually reach the model?

    Compared by containment after collapsing whitespace: the tokenizer round
    trip adds BOS/EOS markers and the SmolVLA newline step appends "\\n", so
    equality would fail on a correct prompt. An empty or substituted prompt
    still fails.
    """
    if not task.strip():
        return False
    squash = lambda s: re.sub(r"\s+", " ", s).strip()  # noqa: E731
    return squash(task) in squash(decoded)


# ---------------------------------------------------------------------------
# State routing — a transcription of the LIVE rule, not the reference one
# ---------------------------------------------------------------------------


def hw_observation_features() -> dict[str, type | tuple]:
    """What `robot.observation_features` returns on the bench, declared.

    Order matters: `hw_to_dataset_features` names `observation.state` in dict
    order, so this must match the recording. Pinned against the v3 recording's
    `meta/info.json` in the test suite.
    """
    feats: dict[str, type | tuple] = {}
    for suffix in (".pos", ".load", ".current"):
        for j in JOINTS:
            feats[f"{j}{suffix}"] = float
    feats.update(CAMERA_SHAPES)
    return feats


def routed_state_keys(
    force: str | None,
    policy_state_dim: int | None = 6,
    max_state_dim: int | None = 32,
) -> list[str]:
    """Which scalar keys land in `observation.state`.

    A transcription of `rollout/context.py:378-407` -- the code that actually
    runs during a rollout -- NOT of `tools/state_routing.py`. The reference
    implementation was extended today with a `stats_state_dim` argument that
    wins over the declared width; the live file has no such argument and still
    decides from the declared width. Mirroring the reference here would rehearse
    a pipeline the bench will never take.

    Both twins declare 6, so inference alone routes both narrow. `_force` is
    checked first (context.py:401), which is why the runner pins it.
    """
    parsed = force_from_env({"CAPSTONE_FORCE_WIDE_STATE": force} if force is not None else {})
    feats = hw_observation_features()
    suffixes = (".pos", ".vel")
    n_stock = sum(1 for k, v in feats.items() if v is float and k.endswith(suffixes))

    if parsed is not None:
        widen = parsed
    elif policy_state_dim is None:
        widen = False
    elif max_state_dim is not None and policy_state_dim == max_state_dim:
        widen = False  # padding is not dimensionality
    else:
        widen = policy_state_dim > n_stock

    if widen:
        suffixes = (".pos", ".vel", ".load", ".current")
    return [k for k, v in feats.items() if v is float and k.endswith(suffixes)]


# ---------------------------------------------------------------------------
# The observation — built the way rtc.py builds it
# ---------------------------------------------------------------------------


def build_rollout_observation(values: dict, force: str | None, task: str, device: str) -> dict:
    """`rtc.py:297-301`, verbatim, on a robot-shaped observation.

    Args:
        values: named scalars (`shoulder_pan.pos`, ...) plus HWC uint8 camera
            frames under their raw names (`front`, `wrist`) -- what the follower
            hands the rollout loop. A missing key raises here rather than being
            padded into a blank tower and masked to zero.
        force: the `CAPSTONE_FORCE_WIDE_STATE` value in effect.
        task: the trained prompt.
        device: torch device string.
    """
    import torch
    from lerobot.policies.utils import prepare_observation_for_inference
    from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

    routed = set(routed_state_keys(force))
    observation_features_hw = {
        k: v
        for k, v in hw_observation_features().items()
        if isinstance(v, tuple) or (v is float and k in routed)
    }
    hw_features = hw_to_dataset_features(observation_features_hw, "observation")
    frame = build_dataset_frame(hw_features, values, prefix="observation")
    batch = prepare_observation_for_inference(frame, torch.device(device), task, ROBOT_TYPE)
    # rtc.py:301 overwrites the bare string with a one-element list.
    batch["task"] = [task]
    return batch


def robot_values_from_dataset_item(item: dict, state_names: list[str]) -> dict:
    """Turn a LeRobotDataset frame back into what the follower would publish.

    The dataset stores `observation.state` as one concatenated vector and images
    as CHW float in [0, 1]; the robot publishes named scalars and HWC uint8
    frames, and `prepare_observation_for_inference` is what divides by 255 and
    permutes. Feeding the dataset's tensors straight in would skip that step and
    normalize differently from the bench.
    """
    state = np.asarray(item["observation.state"], dtype=np.float32)
    values = {name: float(state[i]) for i, name in enumerate(state_names)}
    for cam in CAMERA_SHAPES:
        chw = np.asarray(item[f"observation.images.{cam}"], dtype=np.float32)
        hwc = np.transpose(chw, (1, 2, 0))
        values[cam] = np.clip(np.rint(hwc * 255.0), 0, 255).astype(np.uint8)
    return values


def preprocessor_overrides(policy_cfg, device: str) -> dict:
    """Every override the bench applies, and no others.

    `context.py:537-546` applies two -- device and rename_map. The launcher adds
    a THIRD: `tools/rollout_30hz_stale_ok.py:293-301` injects
    `tokenizer_processor.tokenizer_name = policy_cfg.vlm_model_name`, because
    SmolVLA checkpoints saved by remote lerobot 0.6.2 record the tokenizer as a
    relative subfolder name ("tokenizer") that local 0.6.1 cannot resolve.

    Measured Sep 3 2026: WITHOUT that third override,
    `make_pre_post_processors(pretrained_path=<snapshot>)` raises

        ValueError: Failed to instantiate processor step 'tokenizer_processor'
        ... tokenizer is not a local folder and is not a valid model identifier

    So a rehearsal that loads the bundled tokenizer some other way is exercising
    a pipeline the bench cannot even construct. Mirror all three.
    """
    overrides = {
        "device_processor": {"device": device},
        "rename_observations_processor": {"rename_map": RENAME_MAP},
    }
    vlm = getattr(policy_cfg, "vlm_model_name", None)
    if vlm:
        overrides["tokenizer_processor"] = {"tokenizer_name": vlm}
    return overrides


# ---------------------------------------------------------------------------
# Scoring (same shape as the pi0.5 rehearsal, so the two are comparable)
# ---------------------------------------------------------------------------


def normalized_error(pred, truth, spread) -> float:
    """Mean absolute error scaled by each joint's own range of motion."""
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    spread = np.asarray(spread, dtype=float)
    safe = np.where(spread > 1e-6, spread, 1.0)
    return float(np.mean(np.abs(pred - truth) / safe))


def check_chunk_sane(chunk, n_joints: int) -> tuple[bool, str]:
    """Reject output that could score well while being meaningless."""
    chunk = np.asarray(chunk, dtype=float)
    if chunk.ndim != 2 or chunk.shape[1] != n_joints:
        return False, f"action chunk is {chunk.shape}, expected (T, {n_joints})"
    if not np.all(np.isfinite(chunk)):
        return False, "action chunk is not finite (NaN or inf present)"
    if np.allclose(chunk.std(axis=0), 0.0):
        return False, "action chunk is constant over time — degenerate output"
    return True, "ok"


def action_queue(policy):
    """The action queue `select_action` just populated, across architectures.

    ACT (`policies/act/modeling_act.py:98`), pi0 and pi0.5 expose
    `_action_queue`. **SmolVLAPolicy has neither** -- it keeps its chunk in
    `self._queues[ACTION]` (`policies/smolvla/modeling_smolvla.py:172`,
    extended :267, popped :269), and `_action_queue` appears zero times in that
    module. Reading only the ACT-shaped name (borrowed from the local
    `sync.py` queue-skip patch, which was written for ACT) made this whole gate
    dead code for every SmolVLA rehearsal: `all sane` was unconditionally True
    and `PLUMBING SUSPECT` was unreachable. Found by the Sep 5 pre-push review;
    both Sep 4 verdicts (A 7.2x, B 6.4x) had no sanity gate behind them.

    Returns None when neither shape is present -- which the caller must treat
    as a FAILURE, never as a pass.
    """
    q = getattr(policy, "_action_queue", None)
    if q is not None:
        return q
    queues = getattr(policy, "_queues", None)
    if isinstance(queues, dict):
        for key in ("action", ACTION_KEY):
            if key in queues:
                return queues[key]
    return None


def assemble_chunk(policy, raw_first, post):
    """(T, n_joints) chunk behind the tick-0 command, or None if unreadable.

    `raw_first` is the RAW return of `policy.select_action` -- not the posted
    one. The previous version stacked the already-posted first action and then
    ran `post` over the whole stack, so row 0 (the row every other row is
    compared against for time-constancy) was post-processed twice.

    Never re-infers: the queue is read exactly as `select_action` left it.
    """
    import torch  # deferred: importing this module must stay cheap for tests

    q = action_queue(policy)
    if q is None or len(q) == 0:
        return None
    stacked = torch.stack([raw_first.squeeze(0)] + [a.squeeze(0) for a in list(q)]).unsqueeze(0)
    return np.asarray(post(stacked)[0].to("cpu"), dtype=float)


def chunk_verdict(chunk, n_joints: int) -> tuple[bool, str]:
    """`check_chunk_sane`, with "I could not read the chunk" as a FAILURE.

    There is deliberately no `len(chunk) > 1` escape: a one-action chunk is not
    a chunk, and the escape is precisely what let the dead gate report sane.
    """
    if chunk is None:
        return False, (
            "action queue unreadable — the policy exposes neither `_action_queue` nor "
            "`_queues[ACTION]`, so the chunk could not be checked for degeneracy"
        )
    return check_chunk_sane(chunk, n_joints)


def trivial_baseline(state) -> np.ndarray:
    """Command where you already are. Any fitted policy must beat this."""
    return np.asarray(state, dtype=float)


def relative_skill(policy_err: float, baseline_err: float) -> float:
    return float(policy_err / max(baseline_err, 1e-6))


def diagnose(policy_err: float, baseline_err: float) -> str:
    ratio = relative_skill(policy_err, baseline_err)
    if ratio > 1.5:
        return (
            f"Policy error ({policy_err:.3f}) is {ratio:.1f}x the trivial "
            f"copy-the-state baseline ({baseline_err:.3f}) -- WORSE THAN DOING "
            f"NOTHING. That points at the observation pipeline, not model "
            f"quality: camera mapping, state routing, normalization, prompt."
        )
    if ratio > 0.85:
        return (
            f"Policy error ({policy_err:.3f}) is comparable to the trivial "
            f"baseline ({baseline_err:.3f}, ratio {ratio:.2f}). The metric scale "
            f"is doing little work on this frame set, so the verdict is "
            f"uninformative rather than damning. Interpret with care."
        )
    return (
        f"Policy error ({policy_err:.3f}) beats the trivial baseline "
        f"({baseline_err:.3f}) by {1 / max(ratio, 1e-6):.1f}x -- real skill on "
        f"training frames. The pipeline delivers observations the way training did."
    )


def verdict(agreement: float) -> str:
    return "PLUMBING OK" if agreement < AGREEMENT_THRESHOLD else "PLUMBING SUSPECT"


def pick_probe_frames(n_episodes: int, episode_len: int, n: int, seed: int) -> list[tuple[int, int]]:
    """Spread probes across episodes and across time, biased away from frame 0.

    Every episode starts near home, where any policy looks competent.
    """
    rng = random.Random(seed)
    eps = rng.sample(range(n_episodes), k=min(n, n_episodes))
    while len(eps) < n:
        eps.append(rng.randrange(n_episodes))
    frames = []
    for i, ep in enumerate(eps[:n]):
        frac = 0.15 + 0.7 * (i / max(n - 1, 1))
        jitter = rng.uniform(-0.07, 0.07)
        f = int(min(max((frac + jitter), 0.0), 0.99) * episode_len)
        frames.append((ep, min(f, episode_len - 1)))
    return frames


# ---------------------------------------------------------------------------
# Metadata printing — the runner's single source of truth
# ---------------------------------------------------------------------------


def resolve_snapshot(repo_id: str, checkpoint: str | None = None) -> str:
    """Local directory of the weights to bench.

    `checkpoint=None` -> the repo's top-level (final) weights. `checkpoint="005000"` -> the
    `checkpoints/005000/pretrained_model` subfolder, fetched on its own (the top-level 900 MB is not
    needed) and REQUIRED to hold model.safetensors -- a missing checkpoint aborts rather than
    falling back to the final weights, which is the exact substitution the hold-out verdict forbids.
    """
    from pathlib import Path

    import huggingface_hub

    if checkpoint is None:
        return huggingface_hub.snapshot_download(repo_id)
    sub = f"checkpoints/{checkpoint}/pretrained_model"
    root = huggingface_hub.snapshot_download(repo_id, allow_patterns=[f"{sub}/*"])
    d = Path(root) / "checkpoints" / checkpoint / "pretrained_model"
    if not (d / "model.safetensors").exists():
        raise CheckpointUnavailable(f"{repo_id} has no {sub}/model.safetensors on the Hub")
    return str(d)


PRINTABLE = ("task", "force", "snapshot", "checkpoint", "repo", "dataset", "state_dim", "gen")


def print_field(field: str, side: Side) -> str:
    if field == "task":
        return trained_task(side.dataset)
    if field == "force":
        return side.force
    if field == "snapshot":
        return resolve_snapshot(side.repo_id, checkpoint_from_env(None, side.gen))
    if field == "checkpoint":
        return checkpoint_from_env(None, side.gen) or "top"
    if field == "repo":
        return side.repo_id
    if field == "dataset":
        return side.dataset
    if field == "state_dim":
        return str(side.state_dim)
    if field == "gen":
        return side.gen
    raise ValueError(f"unknown field {field!r}; expected one of {PRINTABLE}")


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run(args) -> int:
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.rollout.inference.rtc import supports_rtc_inference
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS

    side = active_sides()[args.side]

    # 1. The prompt, resolved exactly as run_smolvla_trial.sh resolves it.
    task = trained_task(side.dataset)
    print(f"[dryrun] side {side.name}  policy {side.repo_id}", flush=True)
    print(f"[dryrun] prompt from {side.dataset}: {task!r}", flush=True)

    # 2. Probe frames come from the 18-dim recording — the raw robot observation.
    ds = LeRobotDataset(PROBE_DATASET, root=str(PROBE_DATASET_ROOT))
    state_names = ds.meta.features["observation.state"]["names"]
    n_eps = ds.meta.total_episodes
    ep_len = ds.meta.total_frames // max(n_eps, 1)
    stats = ds.meta.stats["action"]
    spread = np.asarray(stats["max"], dtype=float) - np.asarray(stats["min"], dtype=float)
    print(f"[dryrun] {n_eps} episodes, ~{ep_len} frames each", flush=True)
    print(f"[dryrun] action spread per joint: {np.round(spread, 1)}", flush=True)

    # 3. Policy + RTC wiring, mirroring context.py:246-271.
    snap = resolve_snapshot(side.repo_id, checkpoint_from_env(None, side.gen))
    cfg = PreTrainedConfig.from_pretrained(snap)
    cfg.device = args.device
    policy = SmolVLAPolicy.from_pretrained(snap, config=cfg)
    # Capability probe only -- kept so a checkpoint that could never run either
    # engine is still caught early. The sync path does not require it.
    if not supports_rtc_inference(policy):
        print("[dryrun] note: checkpoint does not advertise RTC inference (sync path unaffected)",
              file=sys.stderr)
    # NOT enabling RTC: context.py:261-270 sets rtc_config and calls
    # init_rtc_processor ONLY when cfg.inference is an RTC config. The runner
    # launches --inference.type=sync, so the bench policy has RTC off -- and
    # SmolVLAPolicy.select_action asserts if RTC is enabled
    # (modeling_smolvla.py:254). Enabling it here would rehearse a policy the
    # bench never builds.
    policy = policy.to(args.device)
    policy.eval()

    # 4. Processors, with the two overrides context.py:537-546 applies. No
    #    dataset_stats: with `pretrained_path` set, make_pre_post_processors
    #    loads the checkpoint's own normalizer and ignores them (factory.py:180).
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=snap,
        pretrained_revision=cfg.pretrained_revision,
        preprocessor_overrides=preprocessor_overrides(cfg, args.device),
    )
    tokenizer = None
    for step in getattr(pre, "steps", ()):
        if getattr(step, "input_tokenizer", None) is not None:
            tokenizer = step.input_tokenizer
    if tokenizer is None:
        print("[dryrun] ABORT: no tokenizer step in the preprocessor", file=sys.stderr)
        return 2

    n_joints = cfg.output_features["action"].shape[0]
    expected_state = side.state_dim
    probes = pick_probe_frames(n_eps, ep_len, args.n, args.seed)
    print(f"[dryrun] probing {len(probes)} frames: {probes}\n", flush=True)

    rows, prompt_failures, width_failures = [], [], []
    for ep, f in probes:
        row = ds.meta.episodes[ep]
        lo, hi = row["dataset_from_index"], row["dataset_to_index"]
        item = ds[min(lo + f, hi - 1)]

        values = robot_values_from_dataset_item(item, state_names)
        obs = build_rollout_observation(values, side.force, task, args.device)

        policy.reset()
        pre.reset()
        post.reset()
        with torch.no_grad():
            batch = pre(obs)

            # (a) Did the prompt reach the model? Decode the tensor, not the dict.
            decoded = tokenizer.batch_decode(batch[OBS_LANGUAGE_TOKENS], skip_special_tokens=True)[0]
            if not prompt_survived_tokenization(decoded, task):
                prompt_failures.append((ep, f, decoded))

            # (b) Did the pin produce the width this checkpoint's normalizer wants?
            got = int(batch["observation.state"].shape[-1])
            if got != expected_state:
                width_failures.append((ep, f, got))

            # (c) The SYNC call site, verbatim: rollout/inference/sync.py:125-131
            #     runs `select_action` inside torch.inference_mode(). The runner
            #     launches --inference.type=sync (switched Sep 4 after RTC's
            #     ActionQueue was found to discard the first chunk's leading
            #     actions), so this is the call the bench actually makes.
            #     `select_action` fills the policy's internal queue from one
            #     chunk and returns its FIRST action -- the tick-0 command.
            raw_first = policy.select_action(batch)
            first_action = post(raw_first)
            #     The chunk behind it, for the degeneracy check only. Read from
            #     the queue the same call just populated; never re-inferred.
            #     An unreadable queue is a FAILURE, not a pass -- see
            #     tests/test_smolvla_dryrun_gate.py.
            chunk_np = assemble_chunk(policy, raw_first, post)
        first_np = np.asarray(first_action.squeeze(0).to("cpu"), dtype=float)

        ok, why = chunk_verdict(chunk_np, n_joints)
        truth = np.asarray(item["action"], dtype=float)
        err = normalized_error(first_np, truth, spread)
        base = normalized_error(
            trivial_baseline(np.asarray(item["observation.state"], dtype=float)[:n_joints]),
            truth,
            spread,
        )
        rows.append(
            {"episode": ep, "frame": f, "sane": ok, "why": why, "norm_err": err, "baseline_err": base}
        )
        flag = " " if ok else "  <-- DEGENERATE"
        print(f"[dryrun] ep {ep:>2} frame {f:>3}: policy {err:.3f}   trivial {base:.3f}{flag}", flush=True)
        if not ok:
            print(f"           {why}", flush=True)

    errs = [r["norm_err"] for r in rows]
    baselines = [r["baseline_err"] for r in rows]
    agreement = float(np.mean(errs))
    baseline_agreement = float(np.mean(baselines))
    all_sane = all(r["sane"] for r in rows)
    v = verdict(agreement) if all_sane else "PLUMBING SUSPECT"
    if prompt_failures or width_failures:
        v = "PLUMBING SUSPECT"

    print()
    print("=" * 70)
    print(f"SmolVLA-{side.name} OFFLINE DRY-RUN  (real call site, no hardware)")
    print("=" * 70)
    print(f"  policy           {side.repo_id}")
    print(f"  device           {args.device}")
    print(f"  inference        sync (policy.select_action -> chunk[0] served first)")
    print(f"  CAPSTONE_FORCE_WIDE_STATE={side.force}  ->  state width {expected_state}")
    print(f"  prompt           {task!r}")
    print(f"  prompt reached policy   {len(rows) - len(prompt_failures)}/{len(rows)} frames")
    print(f"  state width correct     {len(rows) - len(width_failures)}/{len(rows)} frames")
    print(f"  frames           {len(rows)} across {len({r['episode'] for r in rows})} episodes")
    print(f"  mean norm err    {agreement:.3f}   (threshold {AGREEMENT_THRESHOLD}, pre-declared)")
    print(f"  trivial baseline {baseline_agreement:.3f}   (copy current state -> action)")
    print(f"  relative skill   {relative_skill(agreement, baseline_agreement):.2f}   (<1 = beats trivial)")
    print(f"  worst frame      {max(errs):.3f}")
    print(f"  all sane         {all_sane}")
    print()
    print(f"  VERDICT          {v}")
    print()
    for ep, f, decoded in prompt_failures:
        print(f"  PROMPT LOST at ep {ep} frame {f}: model saw {decoded!r}")
    for ep, f, got in width_failures:
        print(f"  STATE WIDTH {got} at ep {ep} frame {f}, normalizer wants {expected_state}")
    print("  " + diagnose(agreement, baseline_agreement).replace(". ", ".\n  "))
    print()
    if v == "PLUMBING OK":
        print("  Observations and the prompt reach SmolVLA the way training did.")
        print("  This says NOTHING about whether SmolVLA can do the task -- only")
        print("  that the pipeline is not the reason if it cannot.")
    else:
        print("  Do NOT go to the bench with this side. Check, in order: the")
        print("  prompt (--dataset.single_task), the pin (CAPSTONE_FORCE_WIDE_STATE),")
        print("  the rename_map, then normalization.")
    print("=" * 70)

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(
                {
                    "side": side.name,
                    "repo_id": side.repo_id,
                    "dataset": side.dataset,
                    "force": side.force,
                    "state_dim": expected_state,
                    "task": task,
                    "device": args.device,
                    "threshold": AGREEMENT_THRESHOLD,
                    "mean_norm_err": agreement,
                    "mean_baseline_err": baseline_agreement,
                    "relative_skill": relative_skill(agreement, baseline_agreement),
                    "prompt_failures": prompt_failures,
                    "width_failures": width_failures,
                    "diagnosis": diagnose(agreement, baseline_agreement),
                    "verdict": v,
                    "rows": rows,
                },
                fh,
                indent=2,
            )
        print(f"\n[dryrun] wrote {args.json_out}")

    return 0 if v == "PLUMBING OK" else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--side", choices=sorted(SIDES), required=True)
    ap.add_argument("--print", dest="print_field", choices=PRINTABLE, default=None,
                    help="print one metadata field and exit (used by run_smolvla_trial.sh)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    if args.print_field:
        try:
            print(print_field(args.print_field, active_sides()[args.side]))
        except (TaskUnavailable, CheckpointUnavailable, ValueError) as exc:
            print(f"ABORT: {exc}", file=sys.stderr)
            return 5
        return 0
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
