#!/usr/bin/env python
"""Hold-out loss on CONTACT frames only, A vs B on the identical frame set.

    python tools/v4_contact_loss.py --a faithqin/act-tube-A-v4 --a-checkpoint 40000 \\
                                    --b faithqin/act-tube-B-v4 --b-checkpoint 40000 --stride 3

Why (Sep 6 2026, eval redesign, comparator §G item 5): the whole-episode hold-out loss is
indistinguishable between A and B in every family (Sep 5). If load helps, it helps on the few
frames where a distal servo is loaded, and those are drowned by ~700 frames of free-space
transit. This labels every held-out frame from the 18-dim twin's OWN load channels (the -noload
twin carries the same frames by (episode, frame_index)), scores A and B with the training loss on
that identical frame set -- one frame per batch, and for the flow-matching policies the noise is
seeded from (episode, frame) so both arms draw the same noise -- and reports contact vs free.
Losses compare A vs B within one family only (ACT: L1+KL; VLAs: flow-matching MSE).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4_holdout_loss as hl  # noqa: E402
import v4_input_ablation as ab  # noqa: E402
import v4_manifest  # noqa: E402
import v4_training_split as split  # noqa: E402

FULL_DS = "faithqin/so101-tube-insert-v4"  # the 18-dim twin: where the load channels live
OUT_DIR = Path(__file__).resolve().parent.parent / "analysis" / "v4_contact_loss"


# --- pure -------------------------------------------------------------------

def contact_lookup(episodes, frames, states, mean, std, thr: float = ab.CONTACT_Z) -> dict:
    """{(episode, frame): contact?} from the distal LOAD channels' |z| against training stats."""
    z = ab.load_z(states, mean, std)
    return {(int(e), int(f)): bool(v > thr) for e, f, v in zip(episodes, frames, z)}


def split_means(rows) -> dict:
    out = {}
    for key, flag in (("contact", True), ("free", False)):
        sel = [float(r["loss"]) for r in rows if r["contact"] == flag]
        out[key] = {"mean": float(np.mean(sel)) if sel else float("nan"), "n": len(sel)}
    return out


def pair_table(a_rows, b_rows) -> dict:
    """A vs B on the frames BOTH scored, split by contact; B_minus_A < 0 = B predicts better."""
    a = {(r["episode"], r["frame"]): r for r in a_rows}
    b = {(r["episode"], r["frame"]): r for r in b_rows}
    keys = sorted(set(a) & set(b))
    out = {"n_paired": len(keys)}
    for key, flag in (("contact", True), ("free", False)):
        ks = [k for k in keys if a[k]["contact"] == flag]
        la = np.array([a[k]["loss"] for k in ks], dtype=np.float64)
        lb = np.array([b[k]["loss"] for k in ks], dtype=np.float64)
        out[key] = {"n": len(ks),
                    "A": float(la.mean()) if len(ks) else float("nan"),
                    "B": float(lb.mean()) if len(ks) else float("nan"),
                    "B_minus_A": float((lb - la).mean()) if len(ks) else float("nan"),
                    "frames_B_better": int((lb < la).sum())}
    return out


def aggregate_seeds(results) -> dict:
    """B − A over several noise-seed runs of the SAME pair: averaged per frame (only frames present in
    every seed and both arms), then per episode on contact frames — the unit that settled π0.5 on
    Sep 6 (one draw said 63 % of frames, another 48 %; three draws per episode said 8/8)."""
    if not results:
        raise ValueError("no seed results to aggregate")
    per_seed = []
    for r in results:
        a = {(x["episode"], x["frame"]): x for x in r["A"]["rows"]}
        b = {(x["episode"], x["frame"]): x for x in r["B"]["rows"]}
        per_seed.append({k: (b[k]["loss"] - a[k]["loss"], bool(a[k]["contact"])) for k in set(a) & set(b)})
    keys = set.intersection(*(set(d) for d in per_seed))
    frame_diff = {k: float(np.mean([d[k][0] for d in per_seed])) for k in keys}
    contact = {k: per_seed[0][k][1] for k in keys}
    by_ep = {}
    for k, v in frame_diff.items():
        if contact[k]:
            by_ep.setdefault(k[0], []).append(v)
    per_episode = {e: float(np.mean(v)) for e, v in sorted(by_ep.items())}
    d = np.array(list(per_episode.values()), dtype=np.float64)
    contact_vals = [v for k, v in frame_diff.items() if contact[k]]
    free_vals = [v for k, v in frame_diff.items() if not contact[k]]
    return {
        "n_seeds": len(results), "n_episodes": int(len(d)),
        "n_frames_contact": len(contact_vals), "n_frames_free": len(free_vals),
        "per_episode": per_episode,
        "mean": float(d.mean()) if len(d) else float("nan"),
        "se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan"),
        "episodes_B_better": int((d < 0).sum()),
        "contact_frame_mean": float(np.mean(contact_vals)) if contact_vals else float("nan"),
        "free_mean": float(np.mean(free_vals)) if free_vals else float("nan"),
    }


def report_aggregate(agg: dict, files) -> str:
    eps = ", ".join(f"{e} {v:+.4f}" for e, v in agg["per_episode"].items())
    return (f"### {len(files)}-seed aggregate: {', '.join(Path(f).name for f in files)}\n"
            f"contact frames: per-episode B − A = {agg['mean']:+.4f} ± {agg['se']:.4f} (SE), "
            f"B better in {agg['episodes_B_better']}/{agg['n_episodes']} episodes; "
            f"frame mean {agg['contact_frame_mean']:+.4f} over {agg['n_frames_contact']} frames; "
            f"free frames {agg['free_mean']:+.4f} over {agg['n_frames_free']}\n"
            f"per episode: {eps}")


# --- heavy ------------------------------------------------------------------

def build_lookup(holdouts: list[int]) -> tuple[dict, str]:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    root = hl.resolve_local_root(FULL_DS)
    if root is None:
        raise SystemExit(f"{FULL_DS} not on this machine")
    st = LeRobotDatasetMetadata(FULL_DS, root=str(root)).stats["observation.state"]
    mean = np.asarray(st["mean"], dtype=np.float32).reshape(-1); std = np.asarray(st["std"], dtype=np.float32).reshape(-1)
    df = pd.concat(pd.read_parquet(p, columns=["episode_index", "frame_index", "observation.state"])
                   for p in sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True)))
    df = df[df["episode_index"].isin(holdouts)]
    states = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    return contact_lookup(df["episode_index"].to_numpy(), df["frame_index"].to_numpy(), states, mean, std), root.name


def score(repo: str, step: int, stride: int, device: str, seed: int, holdouts: list[int], lookup: dict, log=print) -> list[dict]:
    from torch.utils.data import DataLoader, Subset

    policy, pre, cfg, train_cfg, _ = hl.load_checkpoint(repo, step, device)
    ds_repo = train_cfg["dataset"]["repo_id"]
    root = hl.resolve_local_root(ds_repo)
    ds = hl.build_dataset(ds_repo, root, holdouts, cfg, train_cfg.get("tolerance_s", 1e-4))
    cams = list(ds.meta.camera_keys)
    col = [int(x) for x in ds.hf_dataset["episode_index"]]
    rows = []
    for e in holdouts:
        idx = hl.episode_frame_indices(col, e, stride)
        dl = DataLoader(Subset(ds, idx), batch_size=1, shuffle=False, num_workers=0)
        t0 = time.time()
        with torch.no_grad():
            for batch in dl:
                frame = int(batch["frame_index"].reshape(-1)[0])
                torch.manual_seed(seed * 1000003 + e * 10007 + frame)  # paired noise across A and B
                res = policy.forward(pre(hl.images_to_float(batch, cams)))
                loss = res[0] if isinstance(res, tuple) else res["loss"]
                rows.append({"episode": e, "frame": frame, "loss": float(loss), "contact": lookup.get((e, frame), False)})
        log(f"    {repo.split('/')[-1]} ep {e:>2}: {len(idx)} frames ({time.time()-t0:.0f}s)")
    return rows


def report(res: dict) -> str:
    t = res["pair"]
    lines = [f"### {res['A']['repo']} @{res['A']['step']} vs {res['B']['repo']} @{res['B']['step']} -- hold-out loss by contact "
             f"({t['n_paired']} paired frames; contact = distal load |z| > {res['contact_z']})", "",
             "| frames | n | A | B | B - A | frames B better |", "|---|---|---|---|---|---|"]
    for key in ("contact", "free"):
        r = t[key]
        lines.append(f"| {key} | {r['n']} | {r['A']:.4f} | {r['B']:.4f} | {r['B_minus_A']:+.4f} | {r['frames_B_better']}/{r['n']} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a"); ap.add_argument("--a-checkpoint", type=int)
    ap.add_argument("--b"); ap.add_argument("--b-checkpoint", type=int)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=split.SEED)
    ap.add_argument("--out")
    ap.add_argument("--aggregate", nargs="+", metavar="JSON",
                    help="no scoring: aggregate several seed runs of ONE pair per episode (writes <first>_aggregate.json)")
    a = ap.parse_args()
    if a.aggregate:
        results = [json.load(open(f)) for f in a.aggregate]
        agg = aggregate_seeds(results)
        out = Path(a.out) if a.out else Path(a.aggregate[0]).with_name(Path(a.aggregate[0]).stem.split("_seed")[0] + "_aggregate.json")
        json.dump({"files": a.aggregate, **agg}, open(out, "w"), indent=1)
        print(report_aggregate(agg, a.aggregate)); print(f"\nwrote {out}")
        return 0
    if not (a.a and a.b and a.a_checkpoint and a.b_checkpoint):
        ap.error("--a/--a-checkpoint/--b/--b-checkpoint are required unless --aggregate is used")
    manifest = v4_manifest.Manifest.load(hl.MANIFEST)
    kept = sorted(e for e in manifest.assigned if e not in manifest.voided)
    holdouts = split.pick_holdouts(manifest, kept, seed=split.SEED)
    lookup, root_name = build_lookup(holdouts)
    n_contact = sum(lookup.values())
    print(f"contact lookup from {root_name}: {n_contact}/{len(lookup)} held-out frames are contact (|z| > {ab.CONTACT_Z})")
    res = {"contact_z": ab.CONTACT_Z, "stride": a.stride, "seed": a.seed, "holdout": holdouts}
    for side, repo, step in (("A", a.a, a.a_checkpoint), ("B", a.b, a.b_checkpoint)):
        print(f"== {side}: {repo} @{step}")
        rows = score(repo, step, a.stride, a.device, a.seed, holdouts, lookup)
        res[side] = {"repo": repo, "step": step, "split": split_means(rows), "rows": rows}
        if a.device == "mps":
            torch.mps.empty_cache()
    res["pair"] = pair_table(res["A"]["rows"], res["B"]["rows"])
    out = Path(a.out) if a.out else OUT_DIR / f"{a.a.split('/')[-1]}_vs_{a.b.split('/')[-1]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(); print(report(res)); print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
