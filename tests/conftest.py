"""Shared paths for the capstone regression suite.

Run with the lerobot env's python:
    /opt/homebrew/Caskroom/miniforge/base/envs/lerobot/bin/python -m pytest tests/ -v

Every test here is hardware-free on purpose: no serial ports, no cameras, no
TCC. The suite is safe to run any time, including mid-session with the arm
powered. Pre-session ritual: run it BEFORE touching the arm.
"""

from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent
TOOLS = PROJECT / "tools"

# The only calibration lerobot actually reads:
LIVE_CALIBRATION = Path.home() / ".cache/huggingface/lerobot/calibration"
CALIBRATION_BACKUP = PROJECT / "calibration-backup"

V2_DATASET_ROOT = (
    Path.home()
    / ".cache/huggingface/lerobot/faithqin/so101-tube-insert-v2_20260816_035314"
)

V4_DATASET_ROOT = next(iter(sorted(
    Path.home().glob(".cache/huggingface/lerobot/faithqin/so101-tube-insert-v4_[0-9]*")
)), Path.home() / ".cache/huggingface/lerobot/faithqin/so101-tube-insert-v4")

V3_DATASET_ROOT = (
    Path.home()
    / ".cache/huggingface/lerobot/faithqin/so101-tube-insert-v3_20260827_110242"
)


@pytest.fixture(scope="session")
def site_packages() -> Path:
    import lerobot

    return Path(lerobot.__file__).resolve().parent


def _frame0_stats(root: Path, tag: str, episodes=None) -> dict:
    import glob

    import numpy as np
    import pandas as pd

    files = sorted(glob.glob(str(root / "data" / "chunk-*" / "file-*.parquet")))
    assert files, f"{tag} dataset parquet not found under {root}"
    df = pd.concat(
        pd.read_parquet(f, columns=["frame_index", "episode_index", "observation.state"])
        for f in files
    )
    f0 = df[df["frame_index"] == 0].sort_values("episode_index")
    if episodes is not None:  # v4: the TRAINING episodes only (hold-out + voided excluded)
        f0 = f0[f0["episode_index"].isin(list(episodes))]
    states = np.stack(f0["observation.state"].to_numpy())[:, :6]
    joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    return {
        "n_episodes": len(f0),
        "joints": joints,
        "mean": dict(zip(joints, states.mean(0))),
        "std": dict(zip(joints, states.std(0))),
        "min": dict(zip(joints, states.min(0))),
        "max": dict(zip(joints, states.max(0))),
    }


@pytest.fixture(scope="session")
def v2_frame0_stats():
    """Per-joint mean/std/min/max of observation.state at frame 0 across all
    v2 training episodes. This is the pose the policy expects to start from."""
    return _frame0_stats(V2_DATASET_ROOT, "v2")


@pytest.fixture(scope="session")
def v3_frame0_stats():
    """Same, for v3. NOT interchangeable with v2 — measured Aug 29, v3's elbow
    frame-0 range is [78.11, 90.42] against v2's [75.47, 84.44], and its gripper
    is [0.79, 1.57] against [1.14, 1.64]. Gating a v3 policy on v2's bounds
    rejects perfectly in-distribution poses and admits out-of-distribution ones."""
    return _frame0_stats(V3_DATASET_ROOT, "v3")


def v4_training_episodes() -> list[int]:
    """The 50 episodes the v4 policies trained on, derived exactly as the scorer derives them:
    manifest -> kept -> seeded A4 hold-out -> the rest (tools/v4_training_split.py)."""
    import sys

    sys.path.insert(0, str(TOOLS))
    import v4_manifest, v4_training_split as split  # noqa: E402

    m = v4_manifest.Manifest.load(TOOLS / "v4_manifest.json")
    kept = sorted(e for e in m.assigned if e not in m.voided)
    return m.training_episodes(split.pick_holdouts(m, kept, seed=split.SEED))


@pytest.fixture(scope="session")
def v4_frame0_stats():
    """Frame-0 support of the v4 TRAINING set (50 episodes; the hold-out and voided takes are not
    what the policies saw). Sep 6: elbow 85.61 sd 2.77 [79.60, 91.30] -- 3.3 deg above v3."""
    if not (V4_DATASET_ROOT / "meta" / "info.json").exists():
        pytest.skip("v4 dataset not on this machine")
    return _frame0_stats(V4_DATASET_ROOT, "v4", episodes=v4_training_episodes())


@pytest.fixture(scope="session")
def v4_carry_stats():
    """The CARRY pose of the v4 TRAINING set (50 episodes): the frame `grasp_index + k` where
    the jaws have closed on the tube and shoulder_lift has moved >= 2 deg toward lifted
    (tools/v4_carry.py -- the single derivation; the S2 PRE-GRASPED stratum homes here).
    Sep 6: lift +45.35 sd 7.20 at carry vs -105.34 at frame 0; gripper 13.16 ON the tube vs
    1.11 closed on nothing. Same keys as v4_frame0_stats plus the derivation's bookkeeping
    (k per episode, exclusions, gripper load), so a stale table anywhere fails in the test."""
    if not (V4_DATASET_ROOT / "meta" / "info.json").exists():
        pytest.skip("v4 dataset not on this machine")
    import sys

    sys.path.insert(0, str(TOOLS))
    import v4_carry  # noqa: E402

    return v4_carry.derive(V4_DATASET_ROOT, v4_training_episodes())

# Sep 14 2026 18:20 -- EXCLUDED, not fixed: tests/test_v4_secondary_panel.py asserts over the blinded rollout
# recordings on disk (by token) and embeds the sealed token, so it cannot be opened until the key is
# opened (working notes (private), section 5). Today's eight new recordings made two of its
# data assertions fail ("no rollout episode re-presses more than twice"; the retry secondary's duration
# dependence). Those are facts about the bench, not regressions. Rewrite it after scoring, with
# synthetic fixtures; until then it is not collected. Remove this line the day it is rewritten.
collect_ignore = ["test_v4_secondary_panel.py"]
