#!/usr/bin/env python
"""The pre-training gate + the A4 hold-out split. Gate, don't narrate.

    python tools/v4_training_split.py            # gates, then prints the split
    python tools/v4_training_split.py --json     # machine-readable

GATES (any failure -> exit 1, nothing emitted):
  1. every kept episode passes tools/v4_audit.py against ITS manifest label
     (structure, approach-signature vs label, open jaws on graze types)
  2. every kept episode's task string is the prompt
  3. kept counts == the planned counts, per type

HOLD-OUT (A4, re-picked at the end against the final manifest): one of every
category plus 3 CLEAN = 8 episodes; never eps 0-1, never the last four kept
(fatigue); seeded 20260904 so it is reproducible. Emits the training list for
--dataset.episodes and the hold-out list, both by RECORDED index.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4_audit, v4_manifest  # noqa: E402

PROMPT = "Pick up the test tube and insert it into the rack"
SEED = 20260904
HOLDOUT_CLEAN = 3
NEVER_FIRST = 2      # eps 0-1
NEVER_LAST = 4       # the last four kept: recorded most fatigued


def pick_holdouts(manifest, kept: list[int], seed: int = SEED) -> list[int]:
    kept = sorted(kept)
    eligible = [e for e in kept[NEVER_FIRST:] if e not in kept[-NEVER_LAST:]]
    rng = random.Random(seed)
    out = []
    # "one of EVERY category plus 3 CLEAN" = the five RECOVERY categories + 3 CLEAN = 8,
    # leaving 50 = 30 CLEAN + 20 recovery, identical to v3 (Start Schedule, A4).
    types = sorted(set(manifest.type_of(e) for e in kept) - {"CLEAN"})
    for t in types:
        pool = [e for e in eligible if manifest.type_of(e) == t and e not in out]
        out.append(rng.choice(pool))
    clean_pool = [e for e in eligible if manifest.type_of(e) == "CLEAN" and e not in out]
    out += rng.sample(clean_pool, HOLDOUT_CLEAN)
    return sorted(out)


def gate(states, manifest, root) -> list[str]:
    problems = []
    kept = [e for e in sorted(states) if e in manifest.assigned and e not in manifest.voided]
    for e in kept:
        r = v4_audit.audit(states[e], manifest.type_of(e), e)
        if not r.ok:
            problems += [f"ep {e} ({manifest.type_of(e)}): {f}" for f in r.failures]
    eps = pd.concat([pd.read_parquet(p) for p in
                     sorted(glob.glob(str(Path(root) / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))])
    for e, t in zip(eps["episode_index"], eps["tasks"]):
        if int(e) in kept and [str(x) for x in t] != [PROMPT]:
            problems.append(f"ep {int(e)}: task is {[str(x) for x in t]!r}, not the prompt")
    have = Counter(manifest.type_of(e) for e in kept)
    if have != Counter(manifest.planned):
        problems.append(f"kept counts {dict(have)} != planned {dict(Counter(manifest.planned))}")
    unassigned = [e for e in states if e not in manifest.assigned]
    if unassigned:
        problems.append(f"episodes on disk with no manifest entry: {unassigned}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    ds = v4_audit.load_dataset()
    if ds is None:
        print("no v4 dataset on this machine", file=sys.stderr); return 2
    states, manifest = ds
    root = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/lerobot/faithqin/so101-tube-insert-v4_*")))[0]
    problems = gate(states, manifest, root)
    if problems:
        print("PRE-TRAINING GATE FAILED — nothing emitted:", file=sys.stderr)
        for p in problems: print("  ✗ " + p, file=sys.stderr)
        return 1
    kept = [e for e in sorted(states) if e in manifest.assigned and e not in manifest.voided]
    holdouts = pick_holdouts(manifest, kept)
    train = manifest.training_episodes(holdouts)
    train = [e for e in train if e in states]
    out = {
        "recorded": len(states), "voided": manifest.voided, "kept": len(kept),
        "holdout": holdouts, "holdout_types": {e: manifest.type_of(e) for e in holdouts},
        "train": train, "train_types": dict(Counter(manifest.type_of(e) for e in train)),
        "dataset_episodes_arg": "--dataset.episodes=" + json.dumps(train, separators=(",", ":")),
    }
    if a.json:
        print(json.dumps(out, indent=1)); return 0
    print(f"GATE PASSED — {len(kept)} kept episodes, every one matches its label, every task is the prompt")
    print(f"recorded {out['recorded']} · voided {manifest.voided} · kept {len(kept)}")
    print(f"\nHOLD-OUT ({len(holdouts)}, seed {SEED}, never 0-1, never the last four kept):")
    for e in holdouts: print(f"  {e:>2} {manifest.type_of(e)}")
    print(f"\nTRAIN ({len(train)}): {dict(Counter(manifest.type_of(e) for e in train))}")
    print(f"\n{out['dataset_episodes_arg']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
