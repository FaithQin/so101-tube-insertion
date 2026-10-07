#!/usr/bin/env python
"""Measure how long pi0.5 takes to predict ONE action chunk, and say plainly
whether that fits the robot's budget.

Why this exists (Aug 31, 2026)
------------------------------
`faithqin/pi05-tube-A-v3_2026-08-30_03-06-05` was fine-tuned on HF Jobs and has
never run on hardware. Before it can drive the arm, one number decides the
entire deployment path:

    chunk budget = n_action_steps / fps = 50 / 20 = 2.5 s

The policy serves 50 actions from one forward pass, so it has 2.5 s of robot
time to produce the next chunk. That is fifty times more forgiving than ACT's
50 ms per-tick budget, and it is the reason "a 3B VLA on a laptop" is not
automatically absurd. If the measured p90 fits with headroom, pi0.5 can run
locally on MPS. If it does not, inference moves to a rented GPU and the arm
talks to it over an SSH tunnel.

What is actually measured
-------------------------
`PI05Policy.predict_action_chunk`, which is where essentially all the cost sits:
one PaliGemma prefix pass over the image towers plus `num_inference_steps=10`
flow-matching denoising passes through the action expert. Normalization and
tokenization run on CPU ahead of it and are timed separately.

Two facts that make the synthetic observation faithful, both read out of
lerobot 0.6.1 source rather than assumed:

1. `PI05Policy._preprocess_images` (modeling_pi05.py:1009) creates every image
   feature that is MISSING from the batch as a fully-padded tensor and encodes
   it anyway -- only the attention mask goes to zero. So the rig's two real
   cameras cost exactly what four cost. Latency does not depend on how many
   cameras are plugged in.
2. `TokenizerProcessorStep` runs with `padding="max_length"` and
   `tokenizer_max_length=200`, so the token tensor is 200 wide whatever the
   prompt says. Prompt content cannot change the timing.

Pixel content likewise cannot change the timing of a fixed-shape transformer,
so synthetic observations give the same number as real ones. The report says
"synthetic" out loud regardless, because a benchmark that quietly passes itself
off as a rollout is how you end up believing a policy works.

Usage
-----
    python tools/pi05_latency_bench.py --device mps --dtype bfloat16
    python tools/pi05_latency_bench.py --device cuda --iters 20 --json-out out.json

Pure functions are unit-tested in tests/test_pi05_latency_bench.py; nothing in
that suite loads the 9.4 GB checkpoint or needs a GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

# Headroom the policy must leave for everything that is not the policy: camera
# reads, bus writes, preprocessing, and (on a rented GPU) the network hop. The
# Aug 29 instrumentation measured non-policy stages summing to 8.8 ms/tick, but
# a chunked async path also has to absorb jitter, so the bar is a flat 20%.
HEADROOM_FRACTION = 0.8

DEFAULT_REPO_ID = "faithqin/pi05-tube-A-v3_2026-08-30_03-06-05"

# The rig, per Eval Protocol -- ACT AB.md and the verified 19.3 Hz control loop.
DEFAULT_FPS = 20


# ---------------------------------------------------------------------------
# Pure functions -- the part that decides what the number MEANS
# ---------------------------------------------------------------------------


def chunk_budget_s(n_action_steps: int, fps: int) -> float:
    """Robot-time bought by one chunk: how long the policy has to make the next.

    A chunk of `n_action_steps` actions consumed at `fps` Hz plays for
    `n_action_steps / fps` seconds. That, not the control period, is the
    deadline for chunked policies.
    """
    if n_action_steps <= 0:
        raise ValueError(f"n_action_steps must be positive, got {n_action_steps}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    return n_action_steps / fps


def amortized_per_tick_ms(latency_s: float, n_action_steps: int) -> float:
    """Chunk latency spread over the actions it produced, in milliseconds.

    Useful only for comparing against ACT's per-tick numbers. It is NOT the
    quantity that has to fit -- a chunk arrives all at once or not at all.
    """
    if n_action_steps <= 0:
        raise ValueError(f"n_action_steps must be positive, got {n_action_steps}")
    return latency_s / n_action_steps * 1000.0


def verdict(p90_s: float, budget_s: float) -> str:
    """FITS / MARGINAL / EXCEEDS, judged on the tail and with headroom."""
    if budget_s <= 0:
        raise ValueError(f"budget_s must be positive, got {budget_s}")
    if p90_s <= HEADROOM_FRACTION * budget_s:
        return "FITS"
    if p90_s <= budget_s:
        return "MARGINAL"
    return "EXCEEDS"


def latency_stats(samples: list[float]) -> dict:
    """Summarize timing samples, tail included.

    p90 rather than the mean, deliberately. The Aug 29 session lost hours to a
    "12-15 Hz loop" that turned out to be 47 slow ticks out of 1200 -- a tail
    misread as a tempo. Here the tail is the thing that stalls the arm, so it
    gets reported next to the median instead of being averaged away.
    """
    if not samples:
        raise ValueError("latency_stats requires at least one sample")
    if any(s <= 0 for s in samples):
        raise ValueError(f"non-positive latency sample in {samples} -- timer misuse, not speed")

    import numpy as np

    arr = np.asarray(samples, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
        "min": float(arr.min()),
    }


def synthetic_batch_spec(input_features: dict, batch_size: int = 1) -> dict:
    """Tensor shapes for a synthetic observation, derived from the policy config.

    Driven by `input_features` rather than hardcoded, so a retrained policy with
    a different camera set cannot silently be benchmarked at the wrong shape.
    Raises if the config resolved to zero image towers -- that means the
    rename_map or `empty_cameras` flag was lost, which is precisely the mismatch
    that would make an inference launch fail after the model was already loaded.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    spec: dict[str, tuple] = {}
    visual = 0
    for key, feat in input_features.items():
        ftype = feat["type"] if isinstance(feat, dict) else getattr(feat, "type", None)
        shape = feat["shape"] if isinstance(feat, dict) else getattr(feat, "shape", None)
        ftype = getattr(ftype, "name", ftype)  # FeatureType enum or plain str

        if str(ftype).upper().endswith("VISUAL"):
            spec[key] = (batch_size, *tuple(shape))
            visual += 1
        elif str(ftype).upper().endswith("STATE"):
            spec[key] = (batch_size, *tuple(shape))

    if visual == 0:
        raise ValueError(
            "policy config declares no VISUAL inputs -- the rename_map or "
            "empty_cameras setting was lost; benchmarking this would measure "
            "a model that cannot run"
        )
    return spec


def sanitize_pi05_config(cfg: dict) -> tuple[dict, list[str]]:
    """Drop lerobot-0.6.2 config fields that 0.6.1's PI05Config cannot parse.

    HF Jobs trains on the remote image (lerobot 0.6.2); the Mac runs 0.6.1 with
    four hand-applied site-packages patches that `pip install -U lerobot`
    destroys. So `PI05Policy.from_pretrained` on the Mac dies with a draccus
    DecodingError naming six unknown fields.

    Upgrading the Mac would take the ACT A/B rig down with it, so instead this
    strips exactly the fields it has reasoned about, and ONLY when their values
    prove they are inert for this checkpoint. If the checkpoint actually uses
    visual memory, proprioceptive memory, or a nonzero RTC training delay, then
    0.6.1 genuinely cannot express that model and this raises rather than
    quietly producing a different policy than the one that was trained.

    Anything not in the reasoned-about list is left in place: an unrecognized
    future field is not automatically safe to discard, and should surface as a
    loud error rather than a silent behaviour change.

    Returns (clean_config, dropped_field_names).
    """
    # field -> the value that makes it inert
    inert_when = {
        "use_visual_memory": False,
        "use_proprioceptive_memory": False,
        "rtc_training_max_delay": 0,
    }
    # These only describe HOW memory works; they are inert whenever both memory
    # flags are off, whatever their own values.
    memory_detail_fields = [
        "memory_frames",
        "memory_stride",
        "memory_temporal_attention_every",
    ]

    for field, inert_value in inert_when.items():
        if field in cfg and cfg[field] != inert_value:
            raise ValueError(
                f"cannot sanitize: {field}={cfg[field]!r} is active in this checkpoint "
                f"(inert value is {inert_value!r}). lerobot 0.6.1 cannot express this "
                f"model -- run it on lerobot 0.6.2 instead of stripping the field."
            )

    clean = dict(cfg)
    dropped: list[str] = []
    for field in list(inert_when) + memory_detail_fields:
        if field in clean:
            del clean[field]
            dropped.append(field)
    return clean, dropped


def resolve_snapshot_dir(repo_id: str, cache_dir=None):
    """Locate the already-downloaded snapshot for `repo_id`.

    Delegates to huggingface_hub so HF_HOME, HF_HUB_CACHE and an explicit
    cache_dir are all honoured. Aug 31: the first version hardcoded
    `~/.cache/huggingface/hub` and so failed on the RunPod pod, which sets
    HF_HOME to /workspace/.cache/huggingface -- and then told the user to run
    the `hf download` they had just successfully run.

    `local_files_only=True` because the point is to find bytes already on disk.
    """
    from pathlib import Path

    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(repo_id, local_files_only=True, cache_dir=cache_dir))
    except Exception as e:
        # hub's own message names neither the repo nor the cache it searched,
        # which is what made the RunPod failure so confusing.
        from huggingface_hub.constants import HF_HUB_CACHE

        raise RuntimeError(
            f"no local snapshot for {repo_id!r} under {HF_HUB_CACHE}. "
            f"If it was just downloaded, HF_HOME may differ between the "
            f"download and this process. Underlying error: {e}"
        ) from e


def apply_runtime_config(config, dtype: str, device: str):
    """Set dtype and device ON THE CONFIG, which is the only correct way here.

    PI05Pytorch is constructed with `precision=config.dtype` and then casts
    itself selectively -- weights to bfloat16, prefix/suffix embeddings to
    match. Calling `.to(dtype=...)` on the finished policy instead casts
    everything indiscriminately while the noise tensor that `sample_actions`
    generates internally stays float32, and the first matmul of the denoising
    loop dies mixing dtypes. Aug 31: that is exactly what killed the first run
    on both MPS and CUDA, and it was misread as "bf16 is broken on Metal".
    """
    if dtype not in ("bfloat16", "float32"):
        raise ValueError(f"pi0.5 supports bfloat16 or float32, got {dtype!r}")
    config.dtype = dtype
    config.device = device
    return config


def absolutize_tokenizer(preproc: dict, checkpoint_dir) -> dict:
    """Rewrite a relative bundled-tokenizer reference to an absolute path.

    The checkpoint bundles its PaliGemma tokenizer in `tokenizer/` -- which is
    why inference never touches Google's gated repo -- but the saved
    preprocessor names it as the bare relative string "tokenizer". transformers
    resolves that against the CURRENT WORKING DIRECTORY, so loading the
    processors from anywhere else fails with

        OSError: tokenizer is not a local folder and is not a valid model
        identifier

    whose "if this is a private repository" hint sends you chasing an auth
    problem that does not exist.

    Only a bare relative name that resolves to a real directory inside the
    checkpoint is rewritten: an absolute path is left alone, and so is a
    genuine Hub id like `google/paligemma-3b-pt-224`.
    """
    import copy
    from pathlib import Path

    checkpoint_dir = Path(checkpoint_dir)
    out = copy.deepcopy(preproc)
    for step in out.get("steps", []):
        for section in ("config", "artifacts"):
            block = step.get(section)
            if not isinstance(block, dict):
                continue
            name = block.get("tokenizer_name")
            if not isinstance(name, str) or not name:
                continue
            if name.startswith("/") or "/" in name:
                continue  # absolute path, or a Hub repo id like org/model
            block["tokenizer_name"] = str(checkpoint_dir / name)
    return out


def materialize_local_checkpoint(snapshot_dir, out_dir):
    """Build a 0.6.1-loadable checkpoint dir beside the HF cache, non-destructively.

    Every file is symlinked back to the immutable HF snapshot except
    `config.json`, which is rewritten with the forward-compat fields stripped.
    Nothing in ~/.cache/huggingface is modified, so a later `hf download` or a
    run on lerobot 0.6.2 still sees the original checkpoint.
    """
    import shutil
    from pathlib import Path

    snapshot_dir = Path(snapshot_dir)
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    for item in snapshot_dir.iterdir():
        if item.name == "config.json":
            continue
        (out_dir / item.name).symlink_to(item.resolve())

    with open(snapshot_dir / "config.json") as f:
        cfg = json.load(f)
    clean, dropped = sanitize_pi05_config(cfg)
    with open(out_dir / "config.json", "w") as f:
        json.dump(clean, f, indent=2)

    # The bundled tokenizer is referenced relatively; make it absolute so the
    # processors load from any working directory.
    pre_path = snapshot_dir / "policy_preprocessor.json"
    if pre_path.exists():
        with open(pre_path) as f:
            preproc = json.load(f)
        fixed = absolutize_tokenizer(preproc, out_dir)
        if fixed != preproc:
            (out_dir / "policy_preprocessor.json").unlink(missing_ok=True)
            with open(out_dir / "policy_preprocessor.json", "w") as f:
                json.dump(fixed, f, indent=2)

    return out_dir, dropped


def format_report(
    stats: dict,
    budget_s: float,
    n_action_steps: int,
    fps: int,
    device: str,
    dtype: str,
    synthetic: bool,
    repo_id: str = DEFAULT_REPO_ID,
    load_s: float | None = None,
    preprocess_ms: float | None = None,
) -> str:
    v = verdict(stats["p90"], budget_s)
    obs_kind = "SYNTHETIC observations (shape-faithful; see module docstring)" if synthetic else "real dataset observations"

    lines = [
        "=" * 68,
        "pi0.5 CHUNK LATENCY",
        "=" * 68,
        f"  policy         {repo_id}",
        f"  device         {device}",
        f"  dtype          {dtype}",
        f"  host           {platform.platform()}",
        f"  observations   {obs_kind}",
        "",
        f"  chunk          {n_action_steps} actions @ {fps} Hz",
        f"  BUDGET         {budget_s:.3f} s   (deadline for the next chunk)",
        f"  headroom bar   {HEADROOM_FRACTION * budget_s:.3f} s   ({int(HEADROOM_FRACTION * 100)}% of budget)",
        "",
        f"  samples        n = {stats['n']}",
        f"  p50            {stats['p50']:.3f} s",
        f"  p90            {stats['p90']:.3f} s   <-- judged on this",
        f"  max            {stats['max']:.3f} s",
        f"  min            {stats['min']:.3f} s",
        f"  mean           {stats['mean']:.3f} s",
        "",
        f"  amortized      {amortized_per_tick_ms(stats['p50'], n_action_steps):.1f} ms per action"
        f"   (ACT on MPS = 43 ms per tick)",
    ]
    if load_s is not None:
        lines.append(f"  model load     {load_s:.1f} s")
    if preprocess_ms is not None:
        lines.append(f"  preprocess     {preprocess_ms:.1f} ms per chunk (CPU, excluded above)")

    lines += [
        "",
        f"  VERDICT        {v}",
        "",
        {
            "FITS": "  Chunk lands with headroom. Local rollout on this device is viable.",
            "MARGINAL": "  Fits the budget but not the headroom. A stall costs the arm a\n"
            "  chunk of dead time. Treat as unproven until run on hardware.",
            "EXCEEDS": "  Does not fit. The arm would run dry between chunks. Move\n"
            "  inference to a faster device, or raise n_action_steps.",
        }[v],
        "=" * 68,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The measurement itself
# ---------------------------------------------------------------------------


def _resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_model_batch(policy, spec: dict, device, tokenizer_max_length: int):
    """Construct exactly the tensors `predict_action_chunk` consumes.

    pi0.5 does not take state as a separate model argument -- the state is
    discretized into the token stream by `Pi05PrepareStateTokenizerProcessorStep`
    before the model ever sees it (which is why `sample_actions` has no state
    parameter). So the model-side batch is images plus language tokens.
    """
    import torch

    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    batch: dict = {}
    batch_size = 1
    for key, shape in spec.items():
        if not key.startswith("observation.images."):
            continue
        batch_size = shape[0]
        # lerobot hands the policy float images in [0, 1]; mid-grey is as
        # representative as anything, and content cannot change timing.
        batch[key] = torch.full(shape, 0.5, dtype=torch.float32, device=device)

    # Padded to max_length by the real pipeline, so always this wide.
    batch[OBS_LANGUAGE_TOKENS] = torch.ones(
        (batch_size, tokenizer_max_length), dtype=torch.long, device=device
    )
    batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(
        (batch_size, tokenizer_max_length), dtype=torch.bool, device=device
    )
    return batch


def _synchronize(device: str) -> None:
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def run_benchmark(args) -> int:
    import torch

    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    device = _resolve_device(args.device)

    print(f"[bench] device={device} dtype={args.dtype} repo={args.repo_id}", flush=True)
    print("[bench] loading policy (9.4 GB checkpoint -- this takes a while)...", flush=True)

    from lerobot.configs.policies import PreTrainedConfig

    def _load(path):
        cfg = PreTrainedConfig.from_pretrained(path)
        apply_runtime_config(cfg, args.dtype, device)
        return PI05Policy.from_pretrained(path, config=cfg)

    t0 = time.perf_counter()
    try:
        policy = _load(args.repo_id)
    except Exception as e:
        if "are not valid for PI05Config" not in str(e):
            raise
        # lerobot version skew: the checkpoint was written by 0.6.2 on HF Jobs,
        # this env is 0.6.1. Upgrading would destroy the four site-packages
        # patches the ACT rig depends on, so sanitize a local copy instead.
        import tempfile
        from pathlib import Path

        print(f"\n[bench] version skew detected: {e}", flush=True)
        cache_name = "models--" + args.repo_id.replace("/", "--")
        snapshot = resolve_snapshot_dir(args.repo_id)
        # Sanitized copy sits beside the real cache, never inside it.
        sanitized_root = Path(
            os.environ.get("PI05_SANITIZED_ROOT", Path(tempfile.gettempdir()) / "pi05_local_sanitized")
        )
        local_dir, dropped = materialize_local_checkpoint(snapshot, sanitized_root / cache_name)
        print(f"[bench] sanitized copy at {local_dir}", flush=True)
        print(f"[bench] dropped inert 0.6.2 fields: {', '.join(dropped)}", flush=True)
        print("[bench] HF cache untouched; original checkpoint intact.\n", flush=True)
        policy = _load(local_dir)

    policy = policy.to(device)
    policy.eval()
    load_s = time.perf_counter() - t0
    print(f"[bench] loaded in {load_s:.1f} s", flush=True)

    cfg = policy.config
    n_action_steps = args.n_action_steps or cfg.n_action_steps
    budget = chunk_budget_s(n_action_steps, args.fps)

    spec = synthetic_batch_spec(cfg.input_features)
    towers = [k for k in spec if k.startswith("observation.images.")]
    print(f"[bench] {len(towers)} image towers: {', '.join(t.rsplit('.', 1)[-1] for t in towers)}", flush=True)
    print(f"[bench] budget {budget:.3f} s ({n_action_steps} actions @ {args.fps} Hz)", flush=True)

    batch = _build_model_batch(policy, spec, device, cfg.tokenizer_max_length)

    print(f"[bench] warmup x{args.warmup}...", flush=True)
    for _ in range(args.warmup):
        with torch.no_grad():
            policy.predict_action_chunk(batch)
        _synchronize(device)

    samples: list[float] = []
    for i in range(args.iters):
        _synchronize(device)
        t = time.perf_counter()
        with torch.no_grad():
            policy.predict_action_chunk(batch)
        _synchronize(device)
        dt = time.perf_counter() - t
        samples.append(dt)
        print(f"[bench] iter {i + 1}/{args.iters}: {dt:.3f} s", flush=True)

    stats = latency_stats(samples)
    report = format_report(
        stats=stats,
        budget_s=budget,
        n_action_steps=n_action_steps,
        fps=args.fps,
        device=device,
        dtype=args.dtype,
        synthetic=True,
        repo_id=args.repo_id,
        load_s=load_s,
    )
    print()
    print(report)

    if args.json_out:
        payload = {
            "repo_id": args.repo_id,
            "device": device,
            "dtype": args.dtype,
            "host": platform.platform(),
            "n_action_steps": n_action_steps,
            "fps": args.fps,
            "budget_s": budget,
            "image_towers": towers,
            "load_s": load_s,
            "samples_s": samples,
            "stats": stats,
            "verdict": verdict(stats["p90"], budget),
            "synthetic": True,
        }
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[bench] wrote {args.json_out}")

    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--fps", type=int, default=DEFAULT_FPS)
    p.add_argument("--n-action-steps", type=int, default=None, help="defaults to the policy config")
    p.add_argument("--json-out", default=None)
    return run_benchmark(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
