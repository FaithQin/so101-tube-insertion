#!/usr/bin/env python3
"""Abort when a checkpoint's declared state width contradicts its own normalizer.

Why this exists
---------------
`faithqin/smolvla-tube-B-v3` declares `input_features["observation.state"].shape
= [6]` with `max_state_dim = 32`, while its bundled preprocessor carries 18-dim
normalizer statistics (measured Sep 3 2026). The live routing logic in
`rollout/context.py` decides from the DECLARED width, so it routes narrow, and a
6-dim state then meets an 18-dim normalizer: RuntimeError on the first inference
tick -- mid-episode, on a gated bench day.

Its A-side twin `smolvla-tube-A-v3` declares the same `[6]` and genuinely IS
6-dim. So nothing about the declaration, the padding width, or the policy family
separates them. Only the normalizer stats do, and those are the ground truth by
construction: they are the buffers the state is about to be normalized against.

Measured widths, Sep 3 2026:

    smolvla-tube-A-v3   declared [6]   max 32     stats [6]    -> narrow, ok
    smolvla-tube-B-v3   declared [6]   max 32     stats [18]   -> WIDE, MISMATCH
    pi05-tube-A-v3      declared [32]  max 32     stats [6]    -> narrow, MISMATCH
    act-tube-A-v3       declared [6]   max None   stats [6]    -> narrow, ok
    act-tube-B-v3       declared [18]  max None   stats [18]   -> WIDE, ok

Gate, don't narrate: a mismatch exits non-zero and prints the exact
CAPSTONE_FORCE_WIDE_STATE value to pin. Passing the right value clears the gate;
passing the wrong one still aborts.

Deliberately standalone. `preflight.py` takes scene/home/ping and no policy
path, and `run_scored_trial.sh` must not change the night before a session.

Usage
-----
    python tools/check_state_width.py <checkpoint-dir>
    python tools/check_state_width.py <checkpoint-dir> --force "$CAPSTONE_FORCE_WIDE_STATE"

Exit codes: 0 safe to launch · 4 metadata disagreement or unreadable normalizer.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from state_routing import should_widen_state, state_width_disagrees  # noqa: E402

EXIT_MISMATCH = 4


def read_widths(snapshot: str | Path) -> dict:
    """Declared width, padding width, and normalizer-stats width of a checkpoint."""
    snapshot = Path(snapshot)
    cfg_path = snapshot / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"no config.json under {snapshot}")
    cfg = json.loads(cfg_path.read_text())
    feat = (cfg.get("input_features") or {}).get("observation.state") or {}
    shape = feat.get("shape")
    declared = int(shape[0]) if shape else None
    max_state_dim = cfg.get("max_state_dim")

    stats = None
    for st in sorted(glob.glob(str(snapshot / "*normalizer*.safetensors"))):
        try:
            from safetensors import safe_open

            with safe_open(st, "pt") as f:
                for k in f.keys():
                    if k.endswith("observation.state.mean"):
                        stats = int(f.get_tensor(k).shape[0])
        except Exception:  # unreadable is unknown, never "fine"
            continue
    return {"declared": declared, "max_state_dim": max_state_dim, "stats": stats}


def evaluate(snapshot: str | Path, n_stock: int = 6, force: str | None = None) -> dict:
    """Routing decision plus the gate verdict for one checkpoint."""
    w = read_widths(snapshot)
    declared, maxsd, stats = w["declared"], w["max_state_dim"], w["stats"]

    disagrees = state_width_disagrees(declared, stats)
    widen = should_widen_state(
        policy_state_dim=declared, n_stock=n_stock,
        max_state_dim=maxsd, stats_state_dim=stats,
    )
    required_force = None
    if disagrees:
        required_force = "1" if widen else "0"

    if stats is None:
        # Cannot check. Unknown is not safe.
        exit_code, reason = EXIT_MISMATCH, (
            "no readable observation.state normalizer stats: cannot confirm the "
            "declared width, and an unverifiable width is not a safe one"
        )
    elif not disagrees:
        exit_code, reason = 0, "declared width matches the normalizer stats"
    elif force is not None and force.strip() == required_force:
        exit_code, reason = 0, (
            f"metadata disagrees (declared {declared} vs stats {stats}) but "
            f"CAPSTONE_FORCE_WIDE_STATE={required_force} pins the correct routing"
        )
    else:
        exit_code, reason = EXIT_MISMATCH, (
            f"declared observation.state width {declared} contradicts this "
            f"checkpoint's own normalizer stats width {stats}; the live "
            f"context.py routing reads the DECLARED width, so it would route "
            f"{'narrow' if not widen else 'wide'} when this checkpoint needs "
            f"{'wide' if widen else 'narrow'} -> "
            f"pin CAPSTONE_FORCE_WIDE_STATE={required_force}"
        )

    return {**w, "disagrees": disagrees, "widen": widen,
            "required_force": required_force, "exit_code": exit_code, "reason": reason}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("snapshot", help="path to the checkpoint snapshot directory")
    ap.add_argument("--n-stock", type=int, default=6, help="stock .pos/.vel width (SO-101: 6)")
    ap.add_argument("--force", default=None, help="the CAPSTONE_FORCE_WIDE_STATE value in effect")
    a = ap.parse_args(argv)

    v = evaluate(a.snapshot, n_stock=a.n_stock, force=a.force)
    print(f"declared={v['declared']} max_state_dim={v['max_state_dim']} stats={v['stats']}")
    print(f"routing={'WIDE (pos+vel+load+current)' if v['widen'] else 'narrow (pos+vel)'}")
    if v["exit_code"] == 0:
        print(f"OK — {v['reason']}")
    else:
        print(f"ABORT — {v['reason']}", file=sys.stderr)
    return v["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
