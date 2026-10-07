#!/usr/bin/env python
"""Derive the ACT-A ablation dataset: identical episodes, state truncated to
the first 6 dims (joint positions only -- drops the 6 load + 6 current dims).

The A/B experiment trains two policies from ONE recording; lerobot's
input_features config can only exclude whole feature keys, not slice inside
observation.state, so the A side gets this derived copy instead.

Truncates state in all three places it lives:
  - data/chunk-*/file-*.parquet          observation.state column
  - meta/episodes/chunk-*/file-*.parquet stats/observation.state/* vectors
  - meta/stats.json                      aggregate stats
and rewrites meta/info.json (shape + names). Videos and everything else are
copied untouched. Action space is already 6-dim and is not touched.

Usage:
    python tools/make_noload_dataset.py <src_root> <dst_root>
"""

import glob
import json
import shutil
import sys

import numpy as np
import pandas as pd

KEEP = 6  # first 6 dims are the joint positions

src, dst = sys.argv[1], sys.argv[2]
shutil.copytree(src, dst)

# 1. info.json
info_path = dst + "/meta/info.json"
info = json.load(open(info_path))
feat = info["features"]["observation.state"]
assert feat["shape"] == [18], f"expected 18-dim state, got {feat['shape']}"
feat["shape"] = [KEEP]
feat["names"] = feat["names"][:KEEP]
json.dump(info, open(info_path, "w"), indent=4)
print("info.json:", feat["shape"], feat["names"])

# 2. data parquets
for p in glob.glob(dst + "/data/chunk-*/file-*.parquet"):
    df = pd.read_parquet(p)
    df["observation.state"] = df["observation.state"].map(lambda v: np.asarray(v[:KEEP]))
    df.to_parquet(p)
    print("truncated:", p.split(dst)[-1], f"({len(df)} frames)")

# 3. per-episode stats vectors
for p in glob.glob(dst + "/meta/episodes/chunk-*/file-*.parquet"):
    ep = pd.read_parquet(p)
    for col in [c for c in ep.columns if c.startswith("stats/observation.state/")]:
        if col.endswith("/count"):
            continue
        ep[col] = ep[col].map(lambda v: np.asarray(v)[:KEEP] if np.asarray(v).ndim >= 1 else v)
    ep.to_parquet(p)
    print("truncated stats:", p.split(dst)[-1])

# 4. aggregate stats.json
stats_path = dst + "/meta/stats.json"
stats = json.load(open(stats_path))
if "observation.state" in stats:
    for k, v in stats["observation.state"].items():
        if isinstance(v, list) and len(v) == 18:
            stats["observation.state"][k] = v[:KEEP]
    json.dump(stats, open(stats_path, "w"), indent=4)
    print("stats.json truncated")

print("done:", dst)
