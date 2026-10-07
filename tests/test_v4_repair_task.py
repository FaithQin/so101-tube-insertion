"""The empty-task repair, exercised on a synthetic dataset in the v3 layout."""
import json
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))
import v4_repair_task as rep  # noqa: E402

P = rep.PROMPT


def _mini(tmp_path):
    root = tmp_path / "ds"
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    pd.DataFrame({"task_index": [0, 1]}, index=pd.Index([P, ""], name="task")).to_parquet(root / "meta" / "tasks.parquet")
    pd.DataFrame({"episode_index": [0, 0, 1, 1, 2],
                  "task_index": [0, 0, 1, 1, 1],
                  "observation.state": [[1.0] * 3] * 5}).to_parquet(root / "data" / "chunk-000" / "file-000.parquet", index=False)
    pd.DataFrame({"episode_index": [0, 1, 2],
                  # ep 1 as the real data stores it (a one-element list holding ""),
                  # ep 2 as a genuinely empty list
                  "tasks": [np.array([P], dtype=object), np.array([""], dtype=object), np.array([], dtype=object)],
                  "length": [2, 2, 1],
                  "stats/task_index/min": [[0.0], [1.0], [1.0]],
                  "stats/task_index/count": [[2], [2], [1]]}).to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)
    (root / "meta" / "info.json").write_text(json.dumps({"total_tasks": 2, "total_episodes": 3}))
    (root / "meta" / "stats.json").write_text(json.dumps({"task_index": {"min": [0.0], "max": [1.0], "count": [5]}}))
    return root


def test_inspect_finds_exactly_the_empty_episodes(tmp_path):
    info = rep.inspect(_mini(tmp_path))
    assert info["empty_episodes"] == [1, 2] and info["empty_frames"] == 3 and info["bad_idx"] == 1


def test_apply_repairs_every_place_the_task_lives_and_backs_up_first(tmp_path):
    root = _mini(tmp_path)
    after = rep.apply(root)
    assert after["empty_episodes"] == [] and after["empty_frames"] == 0 and after["bad_idx"] is None
    d = pd.read_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    assert set(d["task_index"]) == {0}, "per-frame task_index not rewritten"
    e = pd.read_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    assert all(list(t) == [P] for t in e["tasks"]), "per-episode tasks not rewritten"
    assert list(e["stats/task_index/min"].map(lambda v: list(v))) == [[0.0]] * 3, "episode stats not refreshed"
    assert list(e["stats/task_index/count"].map(lambda v: list(v))) == [[2], [2], [1]], "counts must be untouched"
    t = pd.read_parquet(root / "meta" / "tasks.parquet")
    assert list(t.index) == [P] and list(t["task_index"]) == [0]
    assert json.loads((root / "meta" / "info.json").read_text())["total_tasks"] == 1
    assert json.loads((root / "meta" / "stats.json").read_text())["task_index"]["max"] == [0.0]
    # a backup exists and still holds the broken state
    bk = list(tmp_path.glob("ds.bak-task-*"))
    assert len(bk) == 1
    assert set(pd.read_parquet(bk[0] / "data" / "chunk-000" / "file-000.parquet")["task_index"]) == {0, 1}


def test_apply_is_a_noop_when_nothing_is_empty(tmp_path):
    root = _mini(tmp_path)
    rep.apply(root)
    again = rep.apply(root)
    assert again["empty_episodes"] == [] and "backup" not in again, "a clean dataset must not be re-backed-up"


def test_observations_are_byte_identical_after_repair(tmp_path):
    root = _mini(tmp_path)
    before = pd.read_parquet(root / "data" / "chunk-000" / "file-000.parquet")["observation.state"].map(tuple).tolist()
    rep.apply(root)
    after = pd.read_parquet(root / "data" / "chunk-000" / "file-000.parquet")["observation.state"].map(tuple).tolist()
    assert before == after


def test_the_empty_task_detector_sees_the_one_element_empty_string():
    """np.array(['']) prints as [] -- the first version tested len()==0 and
    found zero empty episodes on the real dataset (17 of 20 were)."""
    assert rep._is_empty_tasks(np.array([""], dtype=object))
    assert rep._is_empty_tasks(np.array([], dtype=object))
    assert rep._is_empty_tasks([""]) and rep._is_empty_tasks([None]) and rep._is_empty_tasks([])
    assert not rep._is_empty_tasks(np.array([P], dtype=object))


def test_stats_are_fixed_even_when_the_file_has_no_healthy_episode(tmp_path):
    """meta/episodes is split across files; eps 3-19 shared none with a healthy row,
    so a copy-from-a-healthy-row approach left their stats claiming task 1."""
    root = _mini(tmp_path)
    # move the two empty episodes into their OWN file, with no healthy row beside them
    e = pd.read_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    e[e.episode_index == 0].to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)
    e[e.episode_index >= 1].to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-001.parquet", index=False)
    after = rep.apply(root)
    assert after["stale_stats"] == [], after
    e1 = pd.read_parquet(root / "meta" / "episodes" / "chunk-000" / "file-001.parquet")
    assert all(float(np.asarray(v).max()) == 0.0 for v in e1["stats/task_index/min"])


def test_a_second_apply_after_a_partial_repair_finishes_the_job(tmp_path):
    """The real Sep 5 sequence: tasks fixed, stats still stale, '' row already gone."""
    root = _mini(tmp_path)
    rep.apply(root)
    # simulate stale stats left behind
    p = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    e = pd.read_parquet(p); e["stats/task_index/min"] = [[0.0], [1.0], [1.0]]; e.to_parquet(p, index=False)
    assert rep.inspect(root)["stale_stats"] == [1, 2]
    after = rep.apply(root)
    assert after["stale_stats"] == [] and after["empty_episodes"] == []
