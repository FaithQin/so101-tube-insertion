"""The pre-training gate refuses to emit a split unless every label checks out."""
import sys

import numpy as np
import pytest

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))
import v4_training_split as split  # noqa: E402
from v4_manifest import Manifest  # noqa: E402


def _m():
    planned = ["CLEAN", "CLEAN"] + ["CLEAN", "NUDGE", "CLEAN", "RESIST", "CLEAN", "LEFT-MIMIC",
                                    "CLEAN", "ROT-20°", "CLEAN", "ROT-45°", "CLEAN", "CLEAN", "CLEAN", "CLEAN"]
    return Manifest(planned)


def test_holdout_has_one_of_every_type_plus_three_clean_and_avoids_the_edges():
    m = _m()
    kept = list(range(len(m.planned)))
    h = split.pick_holdouts(m, kept)
    types = [m.type_of(e) for e in h]
    for t in ("NUDGE", "RESIST", "LEFT-MIMIC", "ROT-20°", "ROT-45°"):
        assert types.count(t) == 1, (t, types)
    assert types.count("CLEAN") == split.HOLDOUT_CLEAN
    assert len(h) == 8
    assert not set(h) & {0, 1}, "never the first two"
    assert not set(h) & set(kept[-4:]), "never the last four kept"


def test_holdout_is_reproducible():
    m = _m(); kept = list(range(len(m.planned)))
    assert split.pick_holdouts(m, kept) == split.pick_holdouts(m, kept)


def test_gate_fails_on_a_label_mismatch_and_on_a_wrong_task(tmp_path, monkeypatch):
    """A CLEAN-shaped trajectory under a graze label, and an episode whose task is
    not the prompt, must each stop the split from being emitted."""
    import pandas as pd
    m = Manifest(["CLEAN", "NUDGE"])
    T = 200
    clean = np.zeros((T, 18)); clean[:, 3] = 73
    clean[40:80, 0] = 66; clean[40:60, 5] = 34; clean[60:80, 5] = 14           # open, close at the site
    clean[100:140, 0] = 15; clean[100:140, 3] = -5; clean[100:120, 5] = 13; clean[120:140, 5] = 30
    states = {0: clean, 1: clean}                                             # ep 1 is a CLEAN labelled NUDGE
    root = tmp_path; (root / "meta" / "episodes" / "c").mkdir(parents=True)
    pd.DataFrame({"episode_index": [0, 1], "tasks": [[split.PROMPT], [""]]}).to_parquet(
        root / "meta" / "episodes" / "c" / "f.parquet", index=False)
    problems = split.gate(states, m, root)
    assert any("ep 1" in p and ("GRAZE NOT REGISTERED" in p or "LABEL" in p) for p in problems), problems
    assert any("ep 1: task is" in p for p in problems), problems
