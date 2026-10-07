"""Training configs must not quietly duplicate every checkpoint to W&B, nor inherit a run id.

Sep 5-6 2026: the v4 configs were derived field-for-field from v3 and carried
`wandb.disable_artifact: false` and v3's `wandb.run_id`. Every checkpoint was therefore
uploaded twice (Hub + W&B): 124.8 GB of W&B artifacts, 93.5 GB of them pi0.5 twins, and the
account went over quota. The inherited run ids only stayed harmless because the W&B project
name changed -- in the same project they would have RESUMED the v3 runs.
"""
import json
from pathlib import Path

import pytest

from conftest import PROJECT

CONFIGS = sorted((PROJECT / "tools" / "train_configs").glob("**/*.json"))
TRAIN_CONFIGS = [p for p in CONFIGS if "wandb" in json.load(open(p))]


@pytest.mark.parametrize("path", TRAIN_CONFIGS, ids=lambda p: p.name)
def test_checkpoints_are_not_duplicated_to_wandb(path):
    w = json.load(open(path))["wandb"]
    assert w.get("disable_artifact") is True, f"{path.name}: wandb.disable_artifact must be true -- the Hub holds the checkpoints"


@pytest.mark.parametrize("path", TRAIN_CONFIGS, ids=lambda p: p.name)
def test_no_inherited_wandb_run_id(path):
    w = json.load(open(path))["wandb"]
    assert w.get("run_id") in (None, ""), f"{path.name}: wandb.run_id must be null -- W&B assigns it at launch"


def test_the_five_v4_configs_are_covered():
    assert len(TRAIN_CONFIGS) >= 5
