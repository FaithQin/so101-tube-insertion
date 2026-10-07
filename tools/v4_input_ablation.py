#!/usr/bin/env python
"""Input-ablation gate: does a B policy's predicted chunk MOVE when load/current is taken away?

    python tools/v4_input_ablation.py faithqin/act-tube-B-v4 --checkpoint 40000
    python tools/v4_input_ablation.py faithqin/smolvla-tube-B-v4 --checkpoint 5000 --stride 6

Pre-bench, no hardware (Sep 6 2026, eval redesign, comparator §G item 4). For every held-out
frame (the 8 A4 episodes, never trained on) the checkpoint predicts an action chunk from the real
observation and from three edited copies of it:

  mean     the 12 load/current dims frozen at their training mean   -> "what if the channels were flat"
  shuffle  load/current taken from another frame of the same episode -> "what if they were wrong"
  pos      the 6 POSITION dims frozen at their mean (control)         -> what a used input looks like

Reported in DEGREES (the checkpoint's own postprocessor unnormalises the chunks), split into
CONTACT frames (any distal load channel -- elbow, wrist_flex, wrist_roll, gripper -- more than
CONTACT_Z training standard deviations from its mean) and FREE frames, and per recovery type.

Verdict, pre-declared: if the position control moves the chunk but the load ablation moves it by
less than FLOOR_DEG and less than a tenth of the control, the policy IGNORES the channels and the
load-sensing hypothesis is false by construction for that checkpoint. Flow-matching policies are
sampled with the SAME noise for every variant of a frame, so the difference is the input's.
State layout: [0:6] pos, [6:12] load, [12:18] current (so_follower patch; tools/v4_audit.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4_holdout_loss as hl  # noqa: E402
import v4_manifest  # noqa: E402
import v4_training_split as split  # noqa: E402

DISTAL_LOAD = [8, 9, 10, 11]  # elbow, wrist_flex, wrist_roll, gripper load
CONTACT_Z = 2.0
FLOOR_DEG = 0.05
OUT_DIR = Path(__file__).resolve().parent.parent / "analysis" / "v4_ablation"
MODES = ("mean", "shuffle", "pos")


# --- pure -------------------------------------------------------------------

def ablate(state, mean, mode: str, rng=None):
    """A copy of `state` (n, 18) with one block of dims replaced; the input is never mutated."""
    src = np.asarray(state, dtype=np.float32)
    s = np.array(src, copy=True)
    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    if mode == "mean":
        s[:, 6:18] = mean[6:18]
    elif mode == "pos":
        s[:, 0:6] = mean[0:6]
    elif mode == "shuffle":
        rng = np.random.default_rng(0) if rng is None else rng
        n = len(s)
        if n > 1:
            shift = int(rng.integers(1, n))  # a cyclic shift: every row is a real row, none stays put
            s[:, 6:18] = src[np.roll(np.arange(n), shift), 6:18]
    else:
        raise ValueError(f"mode {mode!r}: expected one of {MODES}")
    return s


def flatten_state(raw) -> tuple[np.ndarray, tuple]:
    """(n, 18) view of a batch state that may carry a time axis ((B, T, 18) for the VLAs); + the shape."""
    a = np.asarray(raw, dtype=np.float32)
    if a.shape[-1] != 18:
        raise ValueError(f"expected an 18-dim state on the last axis, got shape {a.shape}")
    return a.reshape(-1, 18), a.shape


def restore_state(flat, shape) -> np.ndarray:
    return np.asarray(flat, dtype=np.float32).reshape(shape)


def load_z(state, mean, std) -> np.ndarray:
    """Per-frame max |z| over the distal LOAD channels only (not currents, not proximal joints)."""
    s = np.asarray(state, dtype=np.float64)[:, DISTAL_LOAD]
    m = np.asarray(mean, dtype=np.float64).reshape(-1)[DISTAL_LOAD]
    sd = np.asarray(std, dtype=np.float64).reshape(-1)[DISTAL_LOAD] + 1e-6
    return np.abs((s - m) / sd).max(axis=1)


def contact_mask(z, thr: float = CONTACT_Z) -> np.ndarray:
    return np.asarray(z, dtype=np.float64) > thr


def summarize(deltas, mask, types) -> dict:
    d = np.asarray(deltas, dtype=np.float64); m = np.asarray(mask, dtype=bool)
    by_type = defaultdict(list)
    for x, t in zip(d, types):
        by_type[t].append(float(x))
    return {
        "contact_mean_deg": float(d[m].mean()) if m.any() else float("nan"),
        "free_mean_deg": float(d[~m].mean()) if (~m).any() else float("nan"),
        "all_mean_deg": float(d.mean()) if len(d) else float("nan"),
        "n_contact": int(m.sum()), "n_free": int((~m).sum()),
        "per_type": {t: {"mean_deg": float(np.mean(v)), "n": len(v)} for t, v in sorted(by_type.items())},
    }


def verdict(load_contact_deg: float, load_free_deg: float, pos_control_deg: float, floor_deg: float = FLOOR_DEG) -> dict:
    ratio = float(load_contact_deg) / max(float(load_free_deg), 1e-9)
    out = {"load_contact_deg": load_contact_deg, "load_free_deg": load_free_deg, "pos_control_deg": pos_control_deg,
           "contact_to_free_ratio": ratio, "floor_deg": floor_deg}
    if pos_control_deg < floor_deg:
        out.update(uses_load_channels=None, message=(
            f"UNINTERPRETABLE: the position CONTROL ablation moved the chunk only {pos_control_deg:.3f} deg -- "
            f"an input the policy certainly uses must move it before the load result means anything"))
    elif load_contact_deg < floor_deg and load_contact_deg < 0.1 * pos_control_deg:
        out.update(uses_load_channels=False, message=(
            f"IGNORES the load/current channels: freezing them moves the chunk {load_contact_deg:.3f} deg on contact "
            f"frames (control {pos_control_deg:.2f} deg) -- the load-sensing hypothesis is false by construction "
            f"for this checkpoint"))
    else:
        out.update(uses_load_channels=True, message=(
            f"USES the load/current channels: {load_contact_deg:.3f} deg on contact frames vs {load_free_deg:.3f} deg "
            f"on free frames (x{ratio:.1f}); control {pos_control_deg:.2f} deg"))
    return out


# --- heavy ------------------------------------------------------------------

def _to_deg(post, chunk: torch.Tensor) -> np.ndarray:
    out = post(chunk)
    return np.asarray(out.detach().to("cpu"), dtype=np.float64)[0]


def run(repo: str, step: int, stride: int, device: str, seed: int, episodes_limit: int | None, log=print) -> dict:
    from torch.utils.data import DataLoader, Subset

    policy, pre, post, cfg, train_cfg, dropped = hl.load_checkpoint(repo, step, device, with_post=True)
    ds_repo = train_cfg["dataset"]["repo_id"]
    if not ds_repo.endswith("so101-tube-insert-v4"):
        raise SystemExit(f"{repo} trained on {ds_repo}: an A policy has no load/current channels to ablate")
    manifest = v4_manifest.Manifest.load(hl.MANIFEST)
    kept = sorted(e for e in manifest.assigned if e not in manifest.voided)
    holdouts = split.pick_holdouts(manifest, kept, seed=split.SEED)
    if episodes_limit:
        holdouts = holdouts[:episodes_limit]
    root = hl.resolve_local_root(ds_repo)
    if root is None:
        raise SystemExit(f"{ds_repo} not on this machine")
    ds = hl.build_dataset(ds_repo, root, holdouts, cfg, train_cfg.get("tolerance_s", 1e-4))
    st = ds.meta.stats["observation.state"]
    mean = np.asarray(st["mean"], dtype=np.float32).reshape(-1); std = np.asarray(st["std"], dtype=np.float32).reshape(-1)
    assert mean.shape == (18,), f"expected an 18-dim state, got {mean.shape}"
    cams = list(ds.meta.camera_keys)
    col = [int(x) for x in ds.hf_dataset["episode_index"]]
    is_flow = cfg.type in ("smolvla", "pi05")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    rng = np.random.default_rng(seed)
    rows = []
    for e in holdouts:
        idx = hl.episode_frame_indices(col, e, stride)
        dl = DataLoader(Subset(ds, idx), batch_size=1, shuffle=False, num_workers=0)
        ep_type = manifest.type_of(e); t0 = time.time()
        # a per-episode pool for the shuffle variant: the load/current of every strided frame of this episode
        pool = np.stack([flatten_state(ds[i]["observation.state"])[0][-1] for i in idx])  # latest obs step
        shuffled_pool = ablate(pool, mean, "shuffle", rng)
        for k, batch in enumerate(dl):
            raw = batch["observation.state"].numpy()
            flat, shape = flatten_state(raw)
            z = float(load_z(flat[-1:], mean, std)[0])
            variants = {"real": flat, "mean": ablate(flat, mean, "mean"), "pos": ablate(flat, mean, "pos")}
            sh = np.array(flat, copy=True)
            sh[:, 6:18] = shuffled_pool[k, 6:18]
            variants["shuffle"] = sh
            variants = {name: restore_state(v, shape) for name, v in variants.items()}
            noise = None
            if is_flow:
                chunk = getattr(cfg, "chunk_size", 50); adim = getattr(cfg, "max_action_dim", 32)
                noise = torch.randn(1, chunk, adim, generator=gen).to(device)
            chunks = {}
            with torch.no_grad():
                for name, state in variants.items():
                    b = dict(batch); b["observation.state"] = torch.from_numpy(np.asarray(state, dtype=np.float32))
                    b = pre(hl.images_to_float(b, cams))
                    out = policy.predict_action_chunk(b, noise=noise) if is_flow else policy.predict_action_chunk(b)
                    chunks[name] = _to_deg(post, out)
            real = chunks["real"]
            rows.append({"episode": e, "type": ep_type, "z": z, "contact": bool(z > CONTACT_Z),
                         "d_mean": float(np.abs(real - chunks["mean"]).mean()),
                         "d_shuffle": float(np.abs(real - chunks["shuffle"]).mean()),
                         "d_pos": float(np.abs(real - chunks["pos"]).mean()),
                         "chunk_spread_deg": float(real.std(axis=0).mean())})
        n_c = sum(r["contact"] for r in rows if r["episode"] == e)
        log(f"  ep {e:>2} {ep_type:<10} {len(idx):>3} frames, {n_c:>3} contact  ({time.time()-t0:.0f}s)")
    mask = np.array([r["contact"] for r in rows]); types = [r["type"] for r in rows]
    summ = {m: summarize([r[f"d_{m}"] for r in rows], mask, types) for m in MODES}
    v = verdict(summ["mean"]["contact_mean_deg"], summ["mean"]["free_mean_deg"], summ["pos"]["all_mean_deg"])
    return {"repo": repo, "step": step, "policy_type": cfg.type, "dataset": ds_repo, "stride": stride, "seed": seed,
            "contact_z": CONTACT_Z, "holdout": holdouts, "n_frames": len(rows), "dropped_fields": dropped,
            "summary": summ, "verdict": v, "chunk_spread_deg": float(np.mean([r["chunk_spread_deg"] for r in rows])),
            "frames": rows}


def report(res: dict) -> str:
    s = res["summary"]; v = res["verdict"]
    lines = [f"### {res['repo']} @ {res['step']} -- input ablation on {res['n_frames']} held-out frames "
             f"({s['mean']['n_contact']} contact, {s['mean']['n_free']} free; contact = distal load |z| > {res['contact_z']})",
             "", "| ablation | contact frames | free frames | all |", "|---|---|---|---|"]
    for m, label in (("mean", "load/current -> training mean"), ("shuffle", "load/current -> another frame"),
                     ("pos", "POSITIONS -> mean (control)")):
        lines.append(f"| {label} | {s[m]['contact_mean_deg']:.3f} deg | {s[m]['free_mean_deg']:.3f} deg | {s[m]['all_mean_deg']:.3f} deg |")
    lines += ["", "per type (load -> mean): " + ", ".join(f"{t} {d['mean_deg']:.3f} deg (n={d['n']})"
                                                       for t, d in s["mean"]["per_type"].items()),
              f"chunk's own spread: {res['chunk_spread_deg']:.2f} deg", "", f"VERDICT: {v['message']}"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo")
    ap.add_argument("--checkpoint", type=int, required=True, help="step whose weights to ablate (e.g. 40000, 5000)")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=split.SEED)
    ap.add_argument("--episodes-limit", type=int)
    ap.add_argument("--out")
    a = ap.parse_args()
    res = run(a.repo, a.checkpoint, a.stride, a.device, a.seed, a.episodes_limit)
    out = Path(a.out) if a.out else OUT_DIR / f"{a.repo.split('/')[-1]}_{a.checkpoint:06d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(); print(report(res)); print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
