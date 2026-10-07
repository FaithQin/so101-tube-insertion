#!/usr/bin/env python
"""Repair episodes recorded with an EMPTY task string. Dry-run unless --apply.

    python tools/v4_repair_task.py            # report only
    python tools/v4_repair_task.py --apply    # backup meta/ + data/, then rewrite

WHY (Sep 5 2026): the v4 run crashed after ep 2 (phantom arrow) and was
resumed; the resume command carried no --dataset.single_task, so every
episode recorded on a resume -- 3-19, 17 of 20 -- has `tasks = []` and
task_index -> "". ACT ignores language. SmolVLA is language-conditioned and
would train on "" then be evaluated on the real prompt: the pi0.5
empty-prompt bug, baked into the training data.

WHAT IT TOUCHES (metadata only; observations, actions and videos are never
opened):
  data/**.parquet            per-frame task_index  : empty -> prompt
  meta/episodes/**.parquet   per-episode `tasks`   : []    -> [prompt]
                             stats/task_index/*    : copied from a healthy episode
  meta/tasks.parquet         drop the "" row
  meta/info.json             total_tasks
  meta/stats.json            task_index aggregate, if present

A full copy of meta/ and data/ is written beside the dataset first, and the
result is verified by re-reading every file. Tests: tests/test_v4_repair_task.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROMPT = "Pick up the test tube and insert it into the rack"
DEFAULT_ROOT = "~/.cache/huggingface/lerobot/faithqin/so101-tube-insert-v4_20260904_220924"


def _is_empty_tasks(cell) -> bool:
    """True for [], [''], [None], np.array(['']) — the real data stores the empty
    task as a one-element list holding '' (numpy PRINTS that as [], which is how
    the first version of this script found zero empty episodes)."""
    try:
        items = list(cell)
    except TypeError:
        return cell is None or cell == ""
    return all(x is None or str(x) == "" for x in items)


def inspect(root: Path, prompt: str = PROMPT) -> dict:
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    idx_of = {t: int(i) for t, i in tasks["task_index"].items()}
    if prompt not in idx_of:
        raise SystemExit(f"[repair] the prompt {prompt!r} is not in tasks.parquet: {idx_of}")
    good, bad = idx_of[prompt], idx_of.get("")
    eps = pd.concat([pd.read_parquet(p) for p in
                     sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))])
    empty_eps = sorted(int(e) for e, t in zip(eps["episode_index"], eps["tasks"]) if _is_empty_tasks(t))
    frames = 0
    for p in sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True)):
        d = pd.read_parquet(p, columns=["task_index"])
        frames += int((d["task_index"] == bad).sum()) if bad is not None else 0
    # every task_index stat must agree with an all-`good` column: std == 0, the rest == good
    stat_cols = [c for c in eps.columns if c.startswith("stats/task_index/") and not c.endswith("/count")]
    def _stale(row) -> bool:
        for c in stat_cols:
            want = 0.0 if c.endswith("/std") else float(good)
            a = np.asarray(row[c], dtype=float)
            if a.size and not np.allclose(a, want):
                return True
        return False
    stale_stats = sorted(int(r["episode_index"]) for _, r in eps.iterrows() if _stale(r))
    return {"good_idx": good, "bad_idx": bad, "empty_episodes": empty_eps, "empty_frames": frames,
            "total_tasks": len(idx_of), "stale_stats": stale_stats}


def _zero_like(v):
    a = np.asarray(v, dtype=float)
    return (a * 0.0).tolist() if a.ndim else 0.0


def apply(root: Path, prompt: str = PROMPT) -> dict:
    info = inspect(root, prompt)
    if info["bad_idx"] is None and not info["empty_episodes"] and not info["stale_stats"]:
        return info
    good, bad = info["good_idx"], info["bad_idx"]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = root.parent / f"{root.name}.bak-task-{stamp}"
    n = 1
    while backup.exists():                       # two applies in one second must not collide
        n += 1
        backup = root.parent / f"{root.name}.bak-task-{stamp}-{n}"
    backup.mkdir()
    shutil.copytree(root / "meta", backup / "meta")
    shutil.copytree(root / "data", backup / "data")

    if bad is not None:
        # data: per-frame task_index
        for p in sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True)):
            d = pd.read_parquet(p)
            d.loc[d["task_index"] == bad, "task_index"] = good
            d.to_parquet(p, index=False)
        # tasks.parquet: drop the empty row
        t = pd.read_parquet(root / "meta" / "tasks.parquet")
        t = t[t.index != ""]
        t.to_parquet(root / "meta" / "tasks.parquet")
        ij = root / "meta" / "info.json"
        j = json.loads(ij.read_text())
        j["total_tasks"] = int(len(t))
        ij.write_text(json.dumps(j, indent=4))

    # episodes: tasks -> [prompt] where empty, and task_index stats set ANALYTICALLY.
    # After the rewrite every frame's task_index is `good`, so every episode's
    # min/max/mean/q* equal `good` and std is 0 -- no reference episode needed.
    # (The first version copied a healthy row from the SAME file; meta/episodes is
    # split across files and eps 3-19 shared none with a healthy row.)
    for p in sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True)):
        e = pd.read_parquet(p)
        e["tasks"] = [([prompt] if _is_empty_tasks(t) else list(t)) for t in e["tasks"]]
        for c in e.columns:
            if c.startswith("stats/task_index/") and not c.endswith("/count"):
                if c.endswith("/std"):
                    e[c] = [_zero_like(v) for v in e[c]]
                else:
                    e[c] = [(np.asarray(_zero_like(v)) + good).tolist() for v in e[c]]
        e.to_parquet(p, index=False)

    # stats.json aggregate, if present
    sj = root / "meta" / "stats.json"
    if sj.exists():
        st = json.loads(sj.read_text())
        if "task_index" in st:
            for k, v in st["task_index"].items():
                if k == "count":
                    continue
                st["task_index"][k] = _zero_like(v) if k == "std" else (np.asarray(_zero_like(v)) + good).tolist()
            sj.write_text(json.dumps(st, indent=4))

    after = inspect(root, prompt)
    after["backup"] = str(backup)
    return after


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--task", default=PROMPT)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    root = Path(os.path.expanduser(a.root))
    before = inspect(root, a.task)
    print(f"[repair] {root.name}")
    print(f"  prompt task_index {before['good_idx']}  empty task_index {before['bad_idx']}")
    print(f"  episodes with an EMPTY task: {before['empty_episodes']}")
    print(f"  frames with the empty task : {before['empty_frames']}")
    print(f"  episodes with STALE task_index stats: {before['stale_stats']}")
    if before["bad_idx"] is None and not before["empty_episodes"] and not before["stale_stats"]:
        print("  nothing to repair"); return 0
    if not a.apply:
        print("  DRY RUN — re-run with --apply to fix (meta/ and data/ are backed up first)")
        return 1
    after = apply(root, a.task)
    print(f"  backup: {after['backup']}")
    print(f"  after : empty episodes {after['empty_episodes']}  empty frames {after['empty_frames']}  "
          f"total_tasks {after['total_tasks']}")
    ok = (not after["empty_episodes"] and after["empty_frames"] == 0 and after["bad_idx"] is None
          and not after["stale_stats"])
    print("  VERIFIED" if ok else "  ✗ VERIFICATION FAILED — restore from the backup")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
