#!/usr/bin/env python
"""Offline hold-out loss per checkpoint: the overfitting curve W&B's train loss cannot show.

    python tools/v4_holdout_loss.py faithqin/act-tube-A-v4
    python tools/v4_holdout_loss.py faithqin/smolvla-tube-B-v4 --stride 3 --batch 8
    python tools/v4_holdout_loss.py faithqin/pi05-tube-A-v4 --stride 4 --batch 4
    python tools/v4_holdout_loss.py faithqin/act-tube-A-v4 --checkpoints 10000 --stride 30 --episodes-limit 2   # smoke

For every checkpoint the run pushed (checkpoints/NNNNNN/pretrained_model) this loads the
weights AND the processors that run saved -- so normalisation is the training normalisation,
never a re-estimate -- and runs the training loss (policy.forward in eval mode, exactly what
lerobot_train's eval step does) over

  * the 8 A4 hold-out episodes, which no v4 policy has seen, and
  * a type-matched reference sample of 8 TRAINING episodes (same types, same seed),

frame-weighted, per episode. The gap between the two is the memorisation signal; the
checkpoint with the lowest hold-out loss is the one to bench first.

GATES (any failure -> exit 1, nothing scored):
  1. the checkpoint's train_config.json trained on the dataset being scored (repo_id)
  2. its dataset.episodes == the list tools/v4_training_split.py emits (kept minus hold-out)
  3. hold-out / training overlap is empty (true by construction; re-checked anyway)
  4. the local dataset copy carries the Hub dataset's episode and frame counts (no stale twin)

Numbers are comparable ACROSS CHECKPOINTS OF ONE RUN only. ACT = L1 + kl_weight*KL;
SmolVLA / pi0.5 = flow-matching MSE with noise drawn from a seed reset per checkpoint, so
every checkpoint of a run sees the same noise. Never compare an ACT number to a SmolVLA one.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4_manifest  # noqa: E402
import v4_training_split as split  # noqa: E402

MANIFEST = Path(__file__).resolve().parent / "v4_manifest.json"
LEROBOT_CACHE = Path.home() / ".cache/huggingface/lerobot"
OUT_DIR = Path(__file__).resolve().parent.parent / "analysis" / "v4_holdout"
RISE_FRAC = 0.05  # pre-declared: final hold-out loss >= 5 % above its minimum = overfit signal
CKPT_RE = re.compile(r"^checkpoints/(\d+)/pretrained_model/model\.safetensors$")
TWIN_RE = r"^{name}_\d{{8}}_\d{{6}}$"  # the recorder's `<repo>_<YYYYMMDD>_<HHMMSS>` cache dirs


# --- pure: what to score -----------------------------------------------------

def parse_checkpoint_steps(files) -> list[int]:
    """Steps that have WEIGHTS on the Hub. A config.json without model.safetensors is not scorable."""
    return sorted({int(m.group(1)) for f in files for m in [CKPT_RE.match(f)] if m})


def holdout_from(kept: list[int], train: list[int]) -> list[int]:
    """Hold-out = kept minus training, by definition. A training episode that is not kept is a label bug."""
    stray = sorted(set(train) - set(kept))
    if stray:
        raise ValueError(f"training list names episodes that are not kept: {stray}")
    return sorted(set(kept) - set(train))


def pick_train_reference(type_of, train: list[int], holdouts: list[int], seed: int) -> list[int]:
    """One TRAINING episode per hold-out episode, of the same type, seeded. Same composition ->
    the hold-out-vs-train gap measures memorisation, not a type mix. Returned PAIRED with
    sorted(holdouts) -- ref[i] matches hold-out i -- so a truncated run still compares like with like."""
    rng = random.Random(seed)
    out: list[int] = []
    for h in sorted(holdouts):
        t = type_of(h)
        pool = [e for e in train if type_of(e) == t and e not in out]
        if not pool:
            raise ValueError(f"no training episode of type {t} left to pair with hold-out {h}")
        out.append(rng.choice(pool))
    return out


def episode_frame_indices(episode_col, episode: int, stride: int = 1) -> list[int]:
    """Dataset indices of `episode`'s frames, every `stride`-th one, counted within the episode."""
    idx = [i for i, e in enumerate(episode_col) if int(e) == episode]
    return idx[::stride]


def resolve_local_root(repo_id: str, cache: Path = LEROBOT_CACHE) -> Path | None:
    """The exact `<cache>/<repo_id>` if present, else the NEWEST `<repo>_<timestamp>` twin the
    recorder wrote -- never a `.bak-*` copy. None if nothing is on this machine."""
    exact = cache / repo_id
    if (exact / "meta" / "info.json").exists():
        return exact
    parent, name = (cache / repo_id).parent, Path(repo_id).name
    pat = re.compile(TWIN_RE.format(name=re.escape(name)))
    cands = [d for d in parent.glob(f"{name}_*") if pat.match(d.name) and (d / "meta" / "info.json").exists()]
    return max(cands, key=lambda d: d.name) if cands else None


# --- pure: the gates ---------------------------------------------------------

def gate(train_cfg: dict, dataset_repo: str, split_train: list[int], holdouts: list[int],
         local_info: dict, hub_info: dict) -> list[str]:
    problems = []
    ds = train_cfg.get("dataset", {})
    if ds.get("repo_id") != dataset_repo:
        problems.append(f"train_config.json dataset.repo_id={ds.get('repo_id')!r} but scoring {dataset_repo!r}")
    trained = sorted(ds.get("episodes") or [])
    if trained != sorted(split_train):
        missing = sorted(set(split_train) - set(trained)); extra = sorted(set(trained) - set(split_train))
        problems.append(f"train_config.json dataset.episodes != the split tools/v4_training_split.py emits "
                        f"(missing {missing}, extra {extra})")
    overlap = sorted(set(holdouts) & set(trained))
    if overlap:
        problems.append(f"hold-out/training overlap: {overlap} -- these episodes were trained on")
    for k in ("total_episodes", "total_frames"):
        if local_info.get(k) != hub_info.get(k):
            problems.append(f"local dataset copy {k}={local_info.get(k)} but the Hub dataset has {hub_info.get(k)} "
                            f"-- a stale twin, not what training saw")
    return problems


# --- pure: the arithmetic ----------------------------------------------------

def frame_weighted_mean(rows) -> float:
    """rows = (loss, n_frames). A batch mean of batch means would weight the ragged last batch wrongly."""
    rows = list(rows)
    total = sum(n for _, n in rows)
    if not rows or total == 0:
        raise ValueError("nothing to average")
    return sum(loss * n for loss, n in rows) / total


def images_to_float(batch: dict, camera_keys) -> dict:
    """What lerobot_train does between the dataloader and the preprocessor: uint8 -> float in [0, 1]."""
    out = dict(batch)
    for k in camera_keys:
        if k in out and out[k].dtype == torch.uint8:
            out[k] = out[k].to(dtype=torch.float32) / 255.0
    return out


def verdict(curve: list[dict], rise_frac: float = RISE_FRAC) -> dict:
    best = min(curve, key=lambda r: r["holdout"])
    last = curve[-1]
    rise = (last["holdout"] - best["holdout"]) / best["holdout"] if best["holdout"] > 0 else 0.0
    overfit = best["step"] != last["step"] and rise >= rise_frac
    if overfit:
        msg = (f"OVERFIT SIGNAL: hold-out loss is {rise*100:.1f}% above its minimum (step {best['step']}) "
               f"at the final checkpoint -- bench step {best['step']}, not the final")
    else:
        msg = f"no overfit signal (final hold-out within {rise_frac*100:.0f}% of the minimum, pre-declared)"
    return {"best_step": best["step"], "best_holdout": best["holdout"], "last_step": last["step"],
            "last_holdout": last["holdout"], "rise_frac": rise, "overfit": overfit, "message": msg}


def report_table(repo: str, curve: list[dict], v: dict) -> str:
    types = sorted({t for r in curve for t in r.get("per_type", {})})
    head = "| step | hold-out | train-ref | gap (hold-ref) |" + "".join(f" {t} |" for t in types)
    rule = "|---|---|---|---|" + "---|" * len(types)
    lines = [f"### {repo} -- hold-out loss per checkpoint (8 A4 hold-out eps vs 8 type-matched training eps)",
             "gap = hold-out minus train-ref: positive = worse on unseen episodes (the memorisation signal)", "",
             head, rule]
    for r in curve:
        gap = r["holdout"] - r["train_ref"]
        row = f"| {r['step']} | {r['holdout']:.4f} | {r['train_ref']:.4f} | {gap:+.4f} |"
        row += "".join(f" {r.get('per_type', {}).get(t, float('nan')):.4f} |" for t in types)
        lines.append(row)
    lines += ["", f"best hold-out at step {v['best_step']} ({v['best_holdout']:.4f}); "
                  f"final step {v['last_step']} ({v['last_holdout']:.4f}) -> {v['message']}"]
    return "\n".join(lines)


RUN_KEYS = ("repo", "dataset", "stride", "seed", "holdout", "train_ref")  # what makes two runs comparable


def merge_results(existing: dict, settings: dict, new_rows: list[dict]) -> dict:
    """Union of an earlier (partial) run and new checkpoint rows, ordered by step. Refuses when any
    setting that defines the numbers differs -- a stride-3 row must never sit beside a stride-6
    one. A re-scored step replaces its old row."""
    diff = [k for k in RUN_KEYS if existing.get(k) != settings.get(k)]
    if diff:
        raise ValueError(f"settings differ from the existing results ({', '.join(diff)}) -- not the same run")
    rows = {c["step"]: c for c in existing.get("checkpoints", [])}
    rows.update({c["step"]: c for c in new_rows})
    out = dict(existing)
    out.update(settings)
    out["checkpoints"] = [rows[s] for s in sorted(rows)]
    return out


def steps_to_skip(existing: dict, steps: list[int]) -> list[int]:
    done = {c["step"] for c in existing.get("checkpoints", [])}
    return [s for s in steps if s in done]


def preprocessor_overrides(policy_cfg, rename_map: dict, device: str) -> dict:
    """The overrides the bench applies (tools/smolvla_dryrun.py): device, the run's rename_map, and
    for SmolVLA the tokenizer by VLM name -- the checkpoint records it as a relative folder that
    lerobot 0.6.1 cannot resolve. pi0.5 bundles its tokenizer; materialize_local_checkpoint
    absolutises that path instead."""
    overrides = {
        "device_processor": {"device": device},
        "rename_observations_processor": {"rename_map": dict(rename_map)},
    }
    vlm = getattr(policy_cfg, "vlm_model_name", None)
    if vlm:
        overrides["tokenizer_processor"] = {"tokenizer_name": vlm}
    return overrides


# --- heavy: load exactly what training saved --------------------------------

def load_checkpoint(repo: str, step: int, device: str, with_post: bool = False):
    """(policy, preprocessor, policy_cfg, train_cfg, dropped_fields) for one Hub checkpoint;
    with_post=True inserts the postprocessor after the preprocessor (v4_input_ablation needs it
    to read predicted chunks in degrees)."""
    from huggingface_hub import snapshot_download
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    sub = f"checkpoints/{step:06d}/pretrained_model"
    ckpt = Path(snapshot_download(repo, allow_patterns=[f"{sub}/*", "train_config.json"])) / sub
    tc = ckpt / "train_config.json"
    if not tc.exists():
        tc = ckpt.parent.parent.parent / "train_config.json"
    train_cfg = json.load(open(tc))
    dropped: list[str] = []
    if json.load(open(ckpt / "config.json")).get("type") == "pi05":
        from pi05_latency_bench import materialize_local_checkpoint  # 0.6.2 -> 0.6.1 config, bundled tokenizer
        out = Path(tempfile.gettempdir()) / "v4_holdout_sanitized" / f"{repo.replace('/', '--')}--{step:06d}"
        ckpt, dropped = materialize_local_checkpoint(ckpt, out)
    cfg = PreTrainedConfig.from_pretrained(str(ckpt))
    cfg.device = device
    if cfg.type == "pi05":
        cfg.dtype = "bfloat16"  # tools/pi05_latency_bench.apply_runtime_config: cast via config, never .to(dtype)
    policy = get_policy_class(cfg.type).from_pretrained(str(ckpt), config=cfg).to(device)
    policy.eval()
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=str(ckpt),
        preprocessor_overrides=preprocessor_overrides(cfg, train_cfg.get("rename_map") or {}, device),
    )
    if with_post:
        return policy, pre, post, cfg, train_cfg, dropped
    return policy, pre, cfg, train_cfg, dropped


def build_dataset(repo_id: str, root: Path, episodes: list[int], policy_cfg, tolerance_s: float):
    """LeRobotDataset the way lerobot.datasets.factory.make_dataset builds it for training."""
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id, root=str(root))
    delta = resolve_delta_timestamps(policy_cfg, meta)
    return LeRobotDataset(repo_id, root=str(root), episodes=episodes, delta_timestamps=delta,
                          return_uint8=True, tolerance_s=tolerance_s)


def score_episodes(policy, pre, ds, episodes: list[int], *, stride: int, batch_size: int, workers: int,
                   seed: int, log=print) -> dict[int, tuple[float, int]]:
    """{episode: (frame-weighted loss, n_frames)}. Seed reset here so every checkpoint draws the same noise."""
    from torch.utils.data import DataLoader, Subset

    col = [int(x) for x in ds.hf_dataset["episode_index"]]
    cams = list(ds.meta.camera_keys)
    torch.manual_seed(seed)
    out = {}
    for e in episodes:
        idx = episode_frame_indices(col, e, stride)
        if not idx:
            raise RuntimeError(f"episode {e} has no frames in the loaded dataset")
        dl = DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False, num_workers=workers)
        rows, t0 = [], time.time()
        with torch.no_grad():
            for batch in dl:
                n = batch["action"].shape[0]
                batch = pre(images_to_float(batch, cams))
                res = policy.forward(batch)
                loss = res[0] if isinstance(res, tuple) else res["loss"]
                rows.append((float(loss), n))
        out[e] = (frame_weighted_mean(rows), sum(n for _, n in rows))
        log(f"    ep {e:>2}: loss {out[e][0]:.4f} over {out[e][1]} frames ({time.time()-t0:.0f}s)")
    return out


# --- main --------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", help="model repo, e.g. faithqin/act-tube-A-v4")
    ap.add_argument("--checkpoints", help="comma-separated steps (default: every checkpoint with weights)")
    ap.add_argument("--stride", type=int, default=1, help="score every Nth frame of each episode")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=split.SEED)
    ap.add_argument("--episodes-limit", type=int, help="smoke: score only the first N hold-out and N reference eps")
    ap.add_argument("--rise-frac", type=float, default=RISE_FRAC)
    ap.add_argument("--out", help=f"JSON path (default {OUT_DIR}/<repo name>.json)")
    ap.add_argument("--append", action="store_true",
                    help="resume: keep steps already in the JSON (same settings only), score the rest")
    a = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    # 1. the split, derived the way training derived it: manifest -> kept -> seeded hold-out
    manifest = v4_manifest.Manifest.load(MANIFEST)
    kept = sorted(e for e in manifest.assigned if e not in manifest.voided)
    holdouts = split.pick_holdouts(manifest, kept, seed=split.SEED)
    train = manifest.training_episodes(holdouts)
    assert holdouts == holdout_from(kept, train)

    # 2. what the run says it trained on, and what is on this machine
    files = HfApi().list_repo_files(a.repo)
    steps = parse_checkpoint_steps(files)
    if a.checkpoints:
        want = [int(s) for s in a.checkpoints.split(",")]
        absent = [s for s in want if s not in steps]
        if absent:
            print(f"no weights on the Hub for steps {absent}; available: {steps}", file=sys.stderr); return 2
        steps = want
    if not steps:
        print(f"{a.repo} has no checkpoints/NNNNNN/pretrained_model/model.safetensors yet", file=sys.stderr); return 2
    top_cfg = json.load(open(hf_hub_download(a.repo, "train_config.json")))
    ds_repo = top_cfg["dataset"]["repo_id"]
    root = resolve_local_root(ds_repo)
    if root is None:
        print(f"{ds_repo} is not in {LEROBOT_CACHE} -- download it first (hf download {ds_repo} --repo-type dataset "
              f"--local-dir {LEROBOT_CACHE / ds_repo})", file=sys.stderr); return 2
    local_info = json.load(open(root / "meta" / "info.json"))
    hub_info = json.load(open(hf_hub_download(ds_repo, "meta/info.json", repo_type="dataset")))

    problems = gate(top_cfg, ds_repo, train, holdouts, local_info, hub_info)
    if problems:
        print("HOLD-OUT GATE FAILED -- nothing scored:", file=sys.stderr)
        for p in problems: print("  x " + p, file=sys.stderr)
        return 1
    ref = pick_train_reference(manifest.type_of, train, holdouts, a.seed)
    if a.episodes_limit:
        holdouts, ref = holdouts[:a.episodes_limit], ref[:a.episodes_limit]
    print(f"GATE PASSED -- {a.repo} trained on {ds_repo} episodes == the split ({len(train)}); "
          f"hold-out {holdouts} disjoint; local copy {root.name} matches the Hub "
          f"({local_info['total_episodes']} eps / {local_info['total_frames']} frames)")
    print(f"hold-out  : {[(e, manifest.type_of(e)) for e in holdouts]}")
    print(f"train-ref : {[(e, manifest.type_of(e)) for e in ref]}")
    print(f"checkpoints {steps} · stride {a.stride} · batch {a.batch} · device {a.device} · seed {a.seed}\n")

    out_path = Path(a.out) if a.out else OUT_DIR / f"{a.repo.split('/')[-1]}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    settings = {"repo": a.repo, "dataset": ds_repo, "stride": a.stride, "seed": a.seed,
                "holdout": holdouts, "train_ref": ref}
    result = dict(settings, dataset_root=str(root), types={e: manifest.type_of(e) for e in holdouts + ref},
                  checkpoints=[])
    if a.append and out_path.exists():
        result = merge_results(json.load(open(out_path)), settings, [])  # refuses a different run
        skip = steps_to_skip(result, steps)
        steps = [s for s in steps if s not in skip]
        print(f"--append: {out_path.name} already has {skip}; scoring {steps or 'nothing new'}\n")
    ds_hold = ds_ref = None
    for step in steps:
        t0 = time.time()
        print(f"== step {step} ==")
        policy, pre, cfg, train_cfg, dropped = load_checkpoint(a.repo, step, a.device)
        if dropped:
            print(f"   (dropped inert 0.6.2 fields: {', '.join(dropped)})")
        ck_problems = gate(train_cfg, ds_repo, train, holdouts, local_info, hub_info)
        if ck_problems:
            print(f"checkpoint {step}'s own train_config.json fails the gate:", file=sys.stderr)
            for p in ck_problems: print("  x " + p, file=sys.stderr)
            return 1
        if ds_hold is None:  # delta_timestamps depend on the policy family only; build once
            ds_hold = build_dataset(ds_repo, root, holdouts, cfg, train_cfg.get("tolerance_s", 1e-4))
            ds_ref = build_dataset(ds_repo, root, ref, cfg, train_cfg.get("tolerance_s", 1e-4))
        print("  hold-out:")
        h = score_episodes(policy, pre, ds_hold, holdouts, stride=a.stride, batch_size=a.batch,
                           workers=a.workers, seed=a.seed)
        print("  train-ref:")
        r = score_episodes(policy, pre, ds_ref, ref, stride=a.stride, batch_size=a.batch,
                           workers=a.workers, seed=a.seed)
        by_type = defaultdict(list)
        for e, (loss, n) in h.items():
            by_type[manifest.type_of(e)].append((loss, n))
        row = {"step": step,
               "holdout": frame_weighted_mean(h.values()), "train_ref": frame_weighted_mean(r.values()),
               "per_type": {t: frame_weighted_mean(v) for t, v in sorted(by_type.items())},
               "per_episode_holdout": {str(e): v[0] for e, v in h.items()},
               "per_episode_train_ref": {str(e): v[0] for e, v in r.items()},
               "n_holdout_frames": sum(n for _, n in h.values()), "n_train_ref_frames": sum(n for _, n in r.values()),
               "seconds": round(time.time() - t0)}
        result = merge_results(result, settings, [row])
        print(f"  => hold-out {row['holdout']:.4f}  train-ref {row['train_ref']:.4f}  "
              f"gap(hold-ref) {row['holdout']-row['train_ref']:+.4f}  ({row['seconds']}s)\n")
        json.dump(result, open(out_path, "w"), indent=1)  # partial results survive a crash
        del policy, pre
        if a.device == "mps":
            torch.mps.empty_cache()

    if not result["checkpoints"]:
        print("nothing scored", file=sys.stderr); return 2
    result["verdict"] = verdict(result["checkpoints"], a.rise_frac)
    json.dump(result, open(out_path, "w"), indent=1)
    print(report_table(a.repo, result["checkpoints"], result["verdict"]))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
