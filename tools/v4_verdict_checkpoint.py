#!/usr/bin/env python
"""Print the checkpoint the PRE-DECLARED hold-out rule picks for a v4 policy.

    python tools/v4_verdict_checkpoint.py faithqin/smolvla-tube-A-v4     -> 005000
    python tools/v4_verdict_checkpoint.py faithqin/pi05-tube-A-v4        -> 020000

The rule (the lab notebook (private), Sep 5 evening; tools/v4_holdout_loss.py `verdict`): bench the FINAL
step unless the final hold-out loss sits >= 5 % above the curve's minimum -- then the curve
overfit and the minimum's checkpoint is benched. The verdict lives in
analysis/v4_holdout/<name>.json, written by the scorer; this prints `%06d` of the chosen step,
which is exactly what CAPSTONE_CHECKPOINT wants (tools/smolvla_dryrun.checkpoint_from_env).

Exits 2 with NOTHING on stdout when the file or its verdict is missing, so a runner handed the
empty value aborts ("CAPSTONE_CHECKPOINT is unset") instead of benching whatever sits at the
top level. Why (Sep 6 2026): tools/run_pair.sh must not carry checkpoint choices in its own
text -- the verdict JSON is the single source, and a policy whose curve has not been scored
(pi0.5-B until its chain finishes) cannot be launched by accident.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HOLDOUT_DIR = Path(__file__).resolve().parent.parent / "analysis" / "v4_holdout"


def verdict_checkpoint(verdict: dict) -> str:
    """The rule, on the scorer's verdict dict: the minimum's step if it overfit, else the last."""
    step = verdict["best_step"] if verdict["overfit"] else verdict["last_step"]
    return f"{int(step):06d}"


def checkpoint_for(repo: str, holdout_dir: Path = HOLDOUT_DIR) -> str:
    path = Path(holdout_dir) / f"{repo.split('/')[-1]}.json"
    if not path.exists():
        raise FileNotFoundError(f"no hold-out curve for {repo}: {path} (run tools/v4_holdout_all.sh)")
    data = json.load(open(path))
    if "verdict" not in data:
        raise KeyError(f"{path} has no verdict yet")
    return verdict_checkpoint(data["verdict"])


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: v4_verdict_checkpoint.py <repo_id>", file=sys.stderr)
        return 2
    try:
        print(checkpoint_for(argv[0]))
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
