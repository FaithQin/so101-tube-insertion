#!/usr/bin/env python
"""Prove the pi0.5 observation pipeline at a desk, before spending bench time.

Replay as arbiter, applied offline. Feed pi0.5 frames it was TRAINED on and
check it roughly reproduces the demonstrated action. A fitted policy should not
be far off on its own training data. If it is, the plumbing is wrong — camera
mapping, state width, normalization — and no amount of bench time fixes that.

Why this exists (Aug 31 2026)
-----------------------------
pi0.5 has never driven the arm, and every previous "the policy doesn't work"
conclusion in this project turned out to be plumbing:

* phantom arrow keys re-recording episodes (Aug 17)
* the elbow homing 1.4 deg below the training floor (Aug 29)
* `SOFollower.connect()` moving wrist_roll ~1.62 deg (Aug 29)
* `.load`/`.current` being routed into a 6-dim policy's state (found tonight,
  by reading the rollout path — it does not crash, it just silently feeds
  pi0.5 real values in coordinates it trained as zero)

Bench time before Sep 6 is the scarcest resource in the project. This burns
none of it.

What a pass means
-----------------
Only that observations reach the policy the way training did. It says nothing
about whether pi0.5 can do the task. A FAIL, on the other hand, is decisive:
do not go to the bench until it passes.

Usage
-----
    python tools/pi05_dryrun.py                    # 6 frames, default seed
    python tools/pi05_dryrun.py --n 12 --json-out out.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys

import numpy as np

# Straight out of the checkpoint's train_config.json. NOT reconstructed by
# hand: getting front/wrist backwards runs fine and is silently wrong.
RENAME_MAP = {
    "observation.images.front": "observation.images.base_0_rgb",
    "observation.images.wrist": "observation.images.left_wrist_0_rgb",
}

DEFAULT_REPO_ID = "faithqin/pi05-tube-A-v3_2026-08-30_03-06-05"
DEFAULT_DATASET = "faithqin/so101-tube-insert-v3-noload"
TASK = "Pick up the test tube and insert it into the rack"

# Pre-declared so it cannot be loosened once a number comes back. Mean absolute
# error, normalized by each joint's own action spread across the dataset. 0.25
# means "typically within a quarter of the range that joint uses" -- loose
# enough for an imperfect policy on a contact-rich task, tight enough that a
# swapped camera or a mis-widened state cannot slip through.
AGREEMENT_THRESHOLD = 0.25


def apply_rename(obs: dict) -> dict:
    """Rename dataset camera keys to the ones the policy declares.

    Raises when a mapped camera is absent: two cameras in, two renamed out. A
    missing one means the dataset or the rig changed, and that must stop the
    run rather than be quietly padded away by `_preprocess_images`.
    """
    out = dict(obs)
    for src, dst in RENAME_MAP.items():
        if src not in out:
            raise ValueError(
                f"observation is missing {src!r}; the rename_map expects both "
                f"cameras present ({', '.join(RENAME_MAP)})"
            )
        out[dst] = out.pop(src)
    return out


def normalized_error(pred: np.ndarray, truth: np.ndarray, spread: np.ndarray) -> float:
    """Mean absolute error scaled by each joint's own range of motion.

    An absolute degree threshold is meaningless across joints: 1 deg of error on
    a joint that sweeps 100 deg is nothing, and the same degree on a joint that
    moves 2 deg is enormous.
    """
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    spread = np.asarray(spread, dtype=float)
    # A joint that never moves cannot be scored relatively; fall back to a unit
    # scale rather than dividing by zero and reporting inf.
    safe = np.where(spread > 1e-6, spread, 1.0)
    return float(np.mean(np.abs(pred - truth) / safe))


def check_chunk_sane(chunk: np.ndarray, n_joints: int) -> tuple[bool, str]:
    """Reject output that could score well while being meaningless."""
    chunk = np.asarray(chunk, dtype=float)
    if chunk.ndim != 2 or chunk.shape[1] != n_joints:
        return False, f"action chunk is {chunk.shape}, expected (T, {n_joints})"
    if not np.all(np.isfinite(chunk)):
        return False, "action chunk is not finite (NaN or inf present)"
    if np.allclose(chunk.std(axis=0), 0.0):
        return False, "action chunk is constant over time — degenerate output"
    return True, "ok"


def pick_probe_frames(n_episodes: int, episode_len: int, n: int, seed: int) -> list[tuple[int, int]]:
    """Choose (episode, frame) pairs spread across episodes and across time.

    Deliberately biased away from frame 0: every episode starts near home,
    where any policy looks competent. The interesting frames are mid-episode,
    at approach and insertion.
    """
    rng = random.Random(seed)
    eps = rng.sample(range(n_episodes), k=min(n, n_episodes))
    while len(eps) < n:
        eps.append(rng.randrange(n_episodes))
    frames = []
    for i, ep in enumerate(eps[:n]):
        # Sweep the episode: early, middle, late, rather than uniform noise.
        frac = 0.15 + 0.7 * (i / max(n - 1, 1))
        jitter = rng.uniform(-0.07, 0.07)
        f = int(min(max((frac + jitter), 0.0), 0.99) * episode_len)
        frames.append((ep, min(f, episode_len - 1)))
    return frames


def trivial_baseline(state: np.ndarray) -> np.ndarray:
    """The dumbest possible predictor: command where you already are.

    On a position-controlled arm at 20 Hz the demonstrated action at frame t is
    very close to the observed state at frame t -- the commanded position barely
    moves in 50 ms. Any fitted policy must beat this. It is the control that
    makes an absolute error number interpretable.
    """
    return np.asarray(state, dtype=float)


def relative_skill(policy_err: float, baseline_err: float) -> float:
    """policy error / trivial-baseline error. <1 is skill, >1 is worse than nothing."""
    return float(policy_err / max(baseline_err, 1e-6))


def diagnose(policy_err: float, baseline_err: float) -> str:
    """Turn two numbers into the actual conclusion.

    Deliberately separate from `verdict`, which is judged against the
    pre-declared threshold and does not move. This explains the verdict; it
    never overrides it.
    """
    ratio = relative_skill(policy_err, baseline_err)
    if ratio > 1.5:
        return (
            f"Policy error ({policy_err:.3f}) is {ratio:.1f}x the trivial "
            f"copy-the-state baseline ({baseline_err:.3f}) -- WORSE THAN DOING "
            f"NOTHING. This points at the observation pipeline, not at model "
            f"quality: check camera mapping, state routing, normalization."
        )
    if ratio > 0.85:
        return (
            f"Policy error ({policy_err:.3f}) is comparable to the trivial "
            f"baseline ({baseline_err:.3f}, ratio {ratio:.2f}). The metric SCALE "
            f"is doing little work here -- this frame set cannot separate a good "
            f"policy from a lazy one, so the verdict is uninformative rather "
            f"than damning. Interpret with care."
        )
    return (
        f"Policy error ({policy_err:.3f}) beats the trivial baseline "
        f"({baseline_err:.3f}) by {1 / max(ratio, 1e-6):.1f}x -- real skill on "
        f"training frames. The pipeline delivers observations the way training did."
    )


def verdict(agreement: float) -> str:
    return "PLUMBING OK" if agreement < AGREEMENT_THRESHOLD else "PLUMBING SUSPECT"


# ---------------------------------------------------------------------------
# The run itself
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260831)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    import tempfile
    from pathlib import Path

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pi05_latency_bench import (
        apply_runtime_config,
        materialize_local_checkpoint,
        resolve_snapshot_dir,
    )

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors

    print(f"[dryrun] dataset {args.dataset}", flush=True)
    ds = LeRobotDataset(args.dataset)
    n_eps = ds.meta.total_episodes
    ep_len = ds.meta.total_frames // max(n_eps, 1)
    print(f"[dryrun] {n_eps} episodes, ~{ep_len} frames each", flush=True)

    # Per-joint action spread over the whole dataset — the scale everything is
    # judged against.
    stats = ds.meta.stats["action"]
    spread = np.asarray(stats["max"], dtype=float) - np.asarray(stats["min"], dtype=float)
    print(f"[dryrun] action spread per joint: {np.round(spread, 1)}", flush=True)

    print("[dryrun] loading pi0.5 (sanitized copy)...", flush=True)
    snap = resolve_snapshot_dir(args.repo_id)
    cache_name = "models--" + args.repo_id.replace("/", "--")
    local_dir, dropped = materialize_local_checkpoint(
        snap, Path(tempfile.gettempdir()) / "pi05_local_sanitized" / cache_name
    )
    cfg = PreTrainedConfig.from_pretrained(local_dir)
    apply_runtime_config(cfg, args.dtype, args.device)
    policy = PI05Policy.from_pretrained(local_dir, config=cfg).to(args.device)
    policy.eval()
    # Load the checkpoint's OWN saved processors (its normalizer stats shipped
    # with it). Rebuilding from ds.meta.stats risks a different normalization
    # than training used.
    # Mirror rollout/context.py:537-546 exactly. The saved preprocessor pins
    # device='cuda' (written on an H200) and carries the identity rename map;
    # the rollout overrides both, so the dry-run must too or it exercises a
    # pipeline the bench will never take. The POSTprocessor's device is 'cpu'
    # by design -- the robot reads actions from the CPU -- so it is left alone.
    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=str(local_dir),
        preprocessor_overrides={
            "device_processor": {"device": args.device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    print(f"[dryrun] loaded (dropped inert 0.6.2 fields: {', '.join(dropped)})", flush=True)

    n_joints = cfg.output_features["action"].shape[0]
    probes = pick_probe_frames(n_eps, ep_len, args.n, args.seed)
    print(f"[dryrun] probing {len(probes)} frames: {probes}\n", flush=True)

    rows = []
    for ep, f in probes:
        # lerobot 0.6.1 exposes episode bounds on meta.episodes as
        # dataset_from_index / dataset_to_index. The older per-dataset index
        # attribute does not exist on LeRobotDataset here (checked Aug 31 2026);
        # the guard test pins this so an upgrade surfaces as a test failure
        # rather than a crash mid-probe.
        row = ds.meta.episodes[ep]
        lo, hi = row["dataset_from_index"], row["dataset_to_index"]
        idx = min(lo + f, hi - 1)
        item = ds[idx]

        obs = {
            "observation.images.front": item["observation.images.front"],
            "observation.images.wrist": item["observation.images.wrist"],
            "observation.state": item["observation.state"],
            "task": TASK,
        }
        # Validate both cameras are present, then hand RAW keys to the
        # pipeline -- rename_observations_processor does the mapping, exactly
        # as on the bench. Renaming here too would rename twice.
        apply_rename(obs)

        with torch.no_grad():
            batch = pre(obs)
            chunk = policy.predict_action_chunk(batch)
            # ENTRY-POINT NOTE (measured Sep 5 2026, resolved). The bench runs
            # the SYNC engine -- run_pi05_trial.sh passes no --inference.type and
            # lerobot defaults to SyncInferenceConfig (rollout/configs.py:234) --
            # and sync.py:129 calls `policy.select_action(observation)`, not this.
            # That is the same shape of divergence that made the SmolVLA
            # rehearsal certify a path the bench never took.
            #
            # Here it is BENIGN, and it was measured rather than assumed:
            # PI05Policy.select_action calls predict_action_chunk internally and
            # pops the queue, so the two are not independent paths. Comparing
            # first actions on the real checkpoint, with a self-comparison
            # control for pi0.5's flow-matching sampling noise:
            #     chunk-vs-chunk (same entry point, twice)  0.105 mean max|d|
            #     chunk-vs-select (rehearsal vs bench)      0.094
            #     ratio 0.90 -- the cross-entry difference is SMALLER than the
            #     noise floor. pi0.5's PLUMBING OK transfers.
            # Two consequences: (a) do not "fix" this by swapping the call
            # without re-measuring; (b) the dry-run is STOCHASTIC at ~0.1
            # normalized units per call, so its headline ratio is not
            # reproducible to two decimals -- never quote it as if it were.
            #
            # predict_action_chunk emits NORMALIZED actions. The real rollout
            # unnormalizes every tick (rollout/inference/sync.py:112); skipping
            # it here compares normalized numbers against degrees and can only
            # ever return SUSPECT.
            chunk = post(chunk)
        chunk_np = np.asarray(chunk[0], dtype=float)

        ok, why = check_chunk_sane(chunk_np, n_joints)
        truth = np.asarray(item["action"], dtype=float)
        err = normalized_error(chunk_np[0], truth, spread)
        # Control: what does "just command where you already are" score?
        base = normalized_error(
            trivial_baseline(np.asarray(item["observation.state"], dtype=float)), truth, spread
        )
        rows.append(
            {"episode": ep, "frame": f, "sane": ok, "why": why, "norm_err": err, "baseline_err": base}
        )
        flag = " " if ok else "  <-- DEGENERATE"
        print(
            f"[dryrun] ep {ep:>2} frame {f:>3}: policy {err:.3f}   trivial {base:.3f}{flag}",
            flush=True,
        )
        if not ok:
            print(f"           {why}", flush=True)

    errs = [r["norm_err"] for r in rows]
    baselines = [r["baseline_err"] for r in rows]
    agreement = float(np.mean(errs))
    baseline_agreement = float(np.mean(baselines))
    all_sane = all(r["sane"] for r in rows)
    v = verdict(agreement) if all_sane else "PLUMBING SUSPECT"

    print()
    print("=" * 68)
    print("pi0.5 OFFLINE DRY-RUN  (replay as arbiter, no hardware)")
    print("=" * 68)
    print(f"  policy        {args.repo_id}")
    print(f"  device        {args.device} / {args.dtype}")
    print(f"  frames        {len(rows)} across {len({r['episode'] for r in rows})} episodes")
    print(f"  mean norm err {agreement:.3f}   (threshold {AGREEMENT_THRESHOLD}, pre-declared)")
    print(f"  trivial baseline {baseline_agreement:.3f}   (copy current state -> action)")
    print(f"  relative skill   {relative_skill(agreement, baseline_agreement):.2f}   (<1 = beats trivial)")
    print(f"  worst frame   {max(errs):.3f}")
    print(f"  all sane      {all_sane}")
    print()
    print(f"  VERDICT       {v}")
    print()
    print("  " + diagnose(agreement, baseline_agreement).replace(". ", ".\n  "))
    print()
    if v == "PLUMBING OK":
        print("  Observations reach the policy the way training did. This says")
        print("  NOTHING about whether pi0.5 can do the task -- only that the")
        print("  pipeline is not the reason if it cannot.")
    else:
        print("  Do NOT go to the bench. On its own training frames the policy")
        print("  does not reproduce the demonstrated action, which points at the")
        print("  observation pipeline: camera mapping, state width, or")
        print("  normalization. Check CAPSTONE_STATE_ROUTING in the rollout log.")
    print("=" * 68)

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(
                {
                    "repo_id": args.repo_id,
                    "dataset": args.dataset,
                    "device": args.device,
                    "dtype": args.dtype,
                    "threshold": AGREEMENT_THRESHOLD,
                    "mean_norm_err": agreement,
                    "mean_baseline_err": baseline_agreement,
                    "relative_skill": relative_skill(agreement, baseline_agreement),
                    "diagnosis": diagnose(agreement, baseline_agreement),
                    "verdict": v,
                    "rows": rows,
                },
                fh,
                indent=2,
            )
        print(f"\n[dryrun] wrote {args.json_out}")

    return 0 if v == "PLUMBING OK" else 1


if __name__ == "__main__":
    sys.exit(main())
