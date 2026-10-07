"""Regression: the pi0.5 latency benchmark must measure the RIGHT thing.

Born from the Aug 31 2026 session. `faithqin/pi05-tube-A-v3_2026-08-30_03-06-05`
was fine-tuned on HF Jobs and has NEVER run on hardware. Before it can drive the
arm, one number decides the whole deployment path: how long one action chunk
takes to predict. A chunk is `n_action_steps` actions served at `fps`, so the
policy has exactly that much robot-time to produce the next one.

Three ways this benchmark could lie, all pinned here:

1. **Wrong budget.** The unit is the CHUNK, not the tick. pi0.5 ships
   `chunk_size=50` / `n_action_steps=50` and the rig runs 20 FPS, so the budget
   is 2.5 s -- fifty times more forgiving than ACT's 50 ms tick. Comparing a
   2 s chunk latency against a 50 ms tick budget would wrongly kill a workable
   policy; comparing it against nothing at all would wrongly ship a broken one.

2. **Too few image towers.** The saved `train_config.json` declares FOUR visual
   inputs -- `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb` and
   `empty_camera_0` -- because training ran with `empty_cameras=1` and a
   rename_map. The rig only has two real cameras. A benchmark that feeds two
   images measures a cheaper model than the one that will actually run: image
   tokens dominate the PaliGemma prefix. Feed every declared visual key or the
   number is optimistic and useless.

3. **Reporting the mean.** Flow matching runs `num_inference_steps=10` denoising
   passes per chunk, and a chunk that misses its deadline stalls the arm. The
   honest statistic is p90, not the mean -- the same lesson the Aug 29 session
   learned when `1/dt` on single overrunning ticks was read as a sustained
   12-15 Hz loop that never existed.

Hardware-free by construction: pure functions plus a shape spec. Nothing here
loads the 9.4 GB checkpoint or touches a GPU.
"""

import sys

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))

import pi05_latency_bench as bench  # noqa: E402

# Straight out of the saved train_config.json for
# faithqin/pi05-tube-A-v3_2026-08-30_03-06-05. If the policy is retrained with a
# different recipe these must be re-read from the new config, not edited to fit.
PI05_N_ACTION_STEPS = 50
PI05_CHUNK_SIZE = 50
PI05_IMAGE_RESOLUTION = (224, 224)
PI05_STATE_DIM = 32  # max_state_dim padding, NOT the 6 real joints
PI05_ACTION_DIM = 6
RIG_FPS = 20

PI05_VISUAL_KEYS = [
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
    "observation.images.empty_camera_0",
]


# --------------------------------------------------------------------------
# 1. The budget is the chunk, not the tick
# --------------------------------------------------------------------------


def test_chunk_budget_is_action_steps_over_fps():
    """50 actions served at 20 Hz buys the policy 2.5 s to make the next chunk."""
    assert bench.chunk_budget_s(PI05_N_ACTION_STEPS, RIG_FPS) == 2.5


def test_chunk_budget_matches_act_tick_when_chunk_is_one():
    """Degenerate case: a 1-step chunk IS the per-tick budget (ACT-style, 50 ms)."""
    assert bench.chunk_budget_s(1, RIG_FPS) == 0.05


def test_chunk_budget_rejects_nonsense():
    for bad in [(0, 20), (50, 0), (-1, 20), (50, -20)]:
        try:
            bench.chunk_budget_s(*bad)
        except ValueError:
            continue
        raise AssertionError(f"chunk_budget_s{bad} should have raised ValueError")


def test_amortized_per_tick_is_reported_in_ms():
    """A 2.0 s chunk of 50 actions amortizes to 40 ms of compute per action."""
    assert bench.amortized_per_tick_ms(2.0, 50) == 40.0


# --------------------------------------------------------------------------
# 2. The verdict must have headroom, and must be driven by p90
# --------------------------------------------------------------------------


def test_verdict_fits_only_with_headroom():
    """Camera reads, bus writes and the network hop all have to fit alongside.
    80% of budget is the line: measured non-policy stage time was 8.8 ms/tick."""
    budget = 2.5
    assert bench.verdict(1.0, budget) == "FITS"
    assert bench.verdict(1.9, budget) == "FITS"  # 76% of budget
    assert bench.verdict(2.1, budget) == "MARGINAL"  # over 80%, under budget
    assert bench.verdict(2.49, budget) == "MARGINAL"
    assert bench.verdict(2.5, budget) == "MARGINAL"  # exactly on the line
    assert bench.verdict(2.51, budget) == "EXCEEDS"
    assert bench.verdict(9.0, budget) == "EXCEEDS"


def test_verdict_boundary_is_exactly_eighty_percent():
    assert bench.verdict(2.0, 2.5) == "FITS"
    assert bench.verdict(2.0001, 2.5) == "MARGINAL"


def test_latency_stats_reports_p90_and_max_not_just_mean():
    """One stalled chunk stalls the arm. The tail is the number that matters."""
    samples = [1.0] * 9 + [10.0]
    stats = bench.latency_stats(samples)
    assert stats["n"] == 10
    assert stats["p50"] == 1.0
    assert stats["max"] == 10.0
    assert stats["mean"] == 1.9
    # p90 must expose the tail, not smooth it away like the mean does.
    assert stats["p90"] >= 1.0
    assert stats["p90"] > stats["p50"]


def test_latency_stats_p50_on_even_sample_count():
    assert bench.latency_stats([1.0, 2.0, 3.0, 4.0])["p50"] == 2.5


def test_latency_stats_rejects_empty():
    try:
        bench.latency_stats([])
    except ValueError:
        return
    raise AssertionError("latency_stats([]) should have raised ValueError")


def test_latency_stats_rejects_warmup_contamination():
    """A negative or zero duration means the timer was misused, not that the
    model was fast. Fail loudly rather than publish a flattering number."""
    try:
        bench.latency_stats([1.0, 0.0, 1.0])
    except ValueError:
        return
    raise AssertionError("latency_stats should reject non-positive samples")


# --------------------------------------------------------------------------
# 3. Every declared visual key must be fed, or the number is optimistic
# --------------------------------------------------------------------------


def _pi05_input_features():
    """Mirror of the policy config's input_features, as saved on the Hub."""
    feats = {
        k: {"type": "VISUAL", "shape": [3, *PI05_IMAGE_RESOLUTION]} for k in PI05_VISUAL_KEYS
    }
    feats["observation.state"] = {"type": "STATE", "shape": [PI05_STATE_DIM]}
    return feats


def test_batch_spec_includes_every_declared_visual_key():
    """The rig has two cameras; the policy declares four image towers. Feeding
    two would measure a model that does not exist."""
    spec = bench.synthetic_batch_spec(_pi05_input_features())
    for key in PI05_VISUAL_KEYS:
        assert key in spec, f"{key} missing from batch spec -- latency would be understated"


def test_batch_spec_counts_four_image_towers():
    spec = bench.synthetic_batch_spec(_pi05_input_features())
    visual = [k for k in spec if k.startswith("observation.images.")]
    assert len(visual) == 4, f"expected 4 image towers, got {len(visual)}: {visual}"


def test_batch_spec_state_is_padded_to_thirty_two():
    """pi0.5 pads state to max_state_dim=32. Feeding the 6 real joints would
    change the tensor shape the model sees."""
    spec = bench.synthetic_batch_spec(_pi05_input_features())
    assert spec["observation.state"] == (1, PI05_STATE_DIM)


def test_batch_spec_images_carry_batch_dim_and_channels_first():
    spec = bench.synthetic_batch_spec(_pi05_input_features())
    assert spec["observation.images.base_0_rgb"] == (1, 3, 224, 224)


def test_batch_spec_honours_batch_size():
    spec = bench.synthetic_batch_spec(_pi05_input_features(), batch_size=4)
    assert spec["observation.images.base_0_rgb"] == (4, 3, 224, 224)
    assert spec["observation.state"] == (4, PI05_STATE_DIM)


def test_batch_spec_rejects_features_with_no_visual_keys():
    """A config that resolved to zero image towers means the rename_map or
    empty_cameras flag was lost -- exactly the failure this session flagged."""
    try:
        bench.synthetic_batch_spec({"observation.state": {"type": "STATE", "shape": [32]}})
    except ValueError:
        return
    raise AssertionError("synthetic_batch_spec should reject a spec with no visual inputs")


# --------------------------------------------------------------------------
# 4. The report has to be honest about what it measured
# --------------------------------------------------------------------------


def test_report_names_the_device_and_dtype():
    """A number without its device is unusable -- MPS and an RTX 3090 are the
    whole question tonight."""
    report = bench.format_report(
        stats={"n": 10, "mean": 1.0, "p50": 1.0, "p90": 1.2, "max": 1.3, "min": 0.9},
        budget_s=2.5,
        n_action_steps=50,
        fps=20,
        device="mps",
        dtype="bfloat16",
        synthetic=True,
    )
    assert "mps" in report
    assert "bfloat16" in report
    assert "FITS" in report


def test_report_flags_synthetic_observations():
    """Latency is content-independent for fixed shapes, but the report must say
    so out loud rather than let a synthetic number pass as a real rollout."""
    report = bench.format_report(
        stats={"n": 10, "mean": 1.0, "p50": 1.0, "p90": 1.2, "max": 1.3, "min": 0.9},
        budget_s=2.5,
        n_action_steps=50,
        fps=20,
        device="mps",
        dtype="bfloat16",
        synthetic=True,
    )
    assert "synthetic" in report.lower()


# --------------------------------------------------------------------------
# 5. Version skew: the checkpoint was written by lerobot 0.6.2, local is 0.6.1
# --------------------------------------------------------------------------
#
# Found Aug 31 2026 while running this very benchmark. `PI05Policy.from_pretrained`
# died with:
#
#   DecodingError: The fields `use_visual_memory`, `use_proprioceptive_memory`,
#   `memory_frames`, `memory_stride`, `memory_temporal_attention_every`,
#   `rtc_training_max_delay` are not valid for PI05Config
#
# HF Jobs trains on the remote image (lerobot 0.6.2); the Mac runs 0.6.1 with
# four hand-applied site-packages patches that `pip install -U lerobot` destroys.
# So the fix is NOT to upgrade the Mac -- it is to drop forward-compat fields
# that are provably inert for this checkpoint, and to REFUSE when they are not.
# Silently stripping an active field would change the model without saying so.

_V062_CONFIG_FIELDS = [
    "use_visual_memory",
    "use_proprioceptive_memory",
    "memory_frames",
    "memory_stride",
    "memory_temporal_attention_every",
    "rtc_training_max_delay",
]


def _v062_config():
    """The real config.json shipped with pi05-tube-A-v3, abridged."""
    return {
        "type": "pi05",
        "chunk_size": 50,
        "n_action_steps": 50,
        "max_state_dim": 32,
        "num_inference_steps": 10,
        "use_visual_memory": False,
        "use_proprioceptive_memory": False,
        "memory_frames": 6,
        "memory_stride": 30,
        "memory_temporal_attention_every": 4,
        "rtc_config": None,
        "rtc_training_max_delay": 0,
    }


def test_sanitizer_drops_exactly_the_six_unparseable_fields():
    clean, dropped = bench.sanitize_pi05_config(_v062_config())
    assert sorted(dropped) == sorted(_V062_CONFIG_FIELDS)
    for f in _V062_CONFIG_FIELDS:
        assert f not in clean


def test_sanitizer_preserves_everything_else_byte_for_byte():
    """Dropping a field that matters would change the model silently."""
    clean, _ = bench.sanitize_pi05_config(_v062_config())
    assert clean["chunk_size"] == 50
    assert clean["n_action_steps"] == 50
    assert clean["max_state_dim"] == 32
    assert clean["num_inference_steps"] == 10
    assert clean["type"] == "pi05"
    assert "rtc_config" in clean  # 0.6.1 DOES understand this one


def test_sanitizer_refuses_when_visual_memory_is_actually_on():
    """If the checkpoint really uses visual memory, 0.6.1 cannot run it at all
    and pretending otherwise would produce a wrong policy, not a slow one."""
    cfg = _v062_config()
    cfg["use_visual_memory"] = True
    try:
        bench.sanitize_pi05_config(cfg)
    except ValueError as e:
        assert "use_visual_memory" in str(e)
        return
    raise AssertionError("sanitizer must refuse a checkpoint that uses visual memory")


def test_sanitizer_refuses_when_proprioceptive_memory_is_actually_on():
    cfg = _v062_config()
    cfg["use_proprioceptive_memory"] = True
    try:
        bench.sanitize_pi05_config(cfg)
    except ValueError:
        return
    raise AssertionError("sanitizer must refuse a checkpoint that uses proprioceptive memory")


def test_sanitizer_refuses_nonzero_rtc_training_delay():
    cfg = _v062_config()
    cfg["rtc_training_max_delay"] = 3
    try:
        bench.sanitize_pi05_config(cfg)
    except ValueError:
        return
    raise AssertionError("sanitizer must refuse a nonzero rtc_training_max_delay")


def test_sanitizer_is_a_noop_on_an_already_valid_config():
    cfg = {"type": "pi05", "chunk_size": 50}
    clean, dropped = bench.sanitize_pi05_config(cfg)
    assert dropped == []
    assert clean == cfg


def test_sanitizer_does_not_mutate_its_input():
    cfg = _v062_config()
    before = dict(cfg)
    bench.sanitize_pi05_config(cfg)
    assert cfg == before


def test_sanitizer_leaves_unknown_future_fields_alone():
    """An unrecognized field is NOT automatically safe to drop. The sanitizer
    handles only fields it has reasoned about; anything else must surface as a
    real error rather than be silently discarded."""
    cfg = _v062_config()
    cfg["some_future_field_that_changes_everything"] = True
    clean, dropped = bench.sanitize_pi05_config(cfg)
    assert "some_future_field_that_changes_everything" in clean
    assert "some_future_field_that_changes_everything" not in dropped


# --------------------------------------------------------------------------
# 6. dtype must come from the config, never from a blanket .to(dtype=...)
# --------------------------------------------------------------------------
#
# Found Aug 31 2026, on BOTH devices, from the same root cause:
#
#   MPS:  MPSNDArrayMatrixMultiplication ... Destination NDArray and Accumulator
#         NDArray cannot have different datatype
#   CUDA: RuntimeError: mat1 and mat2 must have the same dtype, but got Float
#         and BFloat16   (at action_in_proj, inside the flow-matching loop)
#
# The first read of the MPS crash was "bf16 is broken on Metal". It was not.
# The benchmark called `policy.to(dtype=torch.bfloat16)`, which casts every
# weight -- including the action expert's `action_in_proj` -- while the noise
# tensor that `sample_actions` generates internally stays float32. The first
# matmul of the denoising loop then mixes dtypes.
#
# PI05Pytorch already handles this correctly on its own: it is built with
# `precision=config.dtype` (modeling_pi05.py:421), calls `self.to(bfloat16)`
# selectively (:246), and casts prefix/suffix embeddings to match the weights
# (:136-137, :575-578). A blanket external cast defeats that machinery.
#
# So dtype is a CONFIG setting here, not a tensor operation. This is a source
# check rather than a behavioural one because reproducing the crash needs the
# 9.4 GB checkpoint and a GPU, and the suite is hardware-free on purpose.

import inspect  # noqa: E402


def test_benchmark_never_blanket_casts_policy_dtype():
    """`.to(dtype=...)` on the loaded policy is the bug, not the fix.

    Parsed with `ast` rather than grepped, so that prose ABOUT the bug -- this
    docstring, the module docstring, the comment block above -- cannot trip it,
    and so that a real call cannot hide behind odd formatting.
    """
    import ast

    tree = ast.parse(inspect.getsource(bench))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "to"):
            continue
        if any(kw.arg == "dtype" for kw in node.keywords):
            offenders.append(ast.dump(func))

    assert not offenders, (
        "benchmark casts a tensor/module dtype via .to(dtype=...), which mixes "
        f"float32 noise with bfloat16 weights in the denoising loop: {offenders}"
    )


def test_dtype_is_applied_to_the_config_object():
    """The requested dtype must reach the model as config.dtype, so PI05Pytorch
    can do its own selective casting."""

    class FakeConfig:
        dtype = "float32"
        device = None

    cfg = bench.apply_runtime_config(FakeConfig(), dtype="bfloat16", device="cuda")
    assert cfg.dtype == "bfloat16"
    assert cfg.device == "cuda"


def test_apply_runtime_config_rejects_unknown_dtype():
    class FakeConfig:
        dtype = "float32"
        device = None

    try:
        bench.apply_runtime_config(FakeConfig(), dtype="int8", device="cuda")
    except ValueError:
        return
    raise AssertionError("apply_runtime_config should reject a dtype pi0.5 cannot use")


# --------------------------------------------------------------------------
# 7. Snapshot lookup must honour HF_HOME, not assume ~/.cache/huggingface
# --------------------------------------------------------------------------
#
# Found Aug 31 2026 on the RunPod box, by Faith running the staged script.
# The version-skew fallback located the downloaded checkpoint by hardcoding
# `Path.home() / ".cache/huggingface/hub"`. The pod sets HF_HOME to
# /workspace/.cache/huggingface, so the glob found nothing and the tool
# reported "no local snapshot ... run `hf download`" -- immediately after
# `hf download` had in fact just downloaded all 9.39 GB successfully.
#
# The misleading part is that the error blamed the user for a step they had
# already completed. Resolution belongs to huggingface_hub, which already knows
# about HF_HOME, HF_HUB_CACHE and an explicit cache_dir.


def test_no_hardcoded_home_cache_path():
    """No string LITERAL in executable code may name the default HF cache.

    Docstrings are excluded on purpose -- the prose explaining this bug has to
    be allowed to mention the path it is about. Same lesson as the dtype test.
    """
    import ast

    tree = ast.parse(inspect.getsource(bench))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            d = ast.get_docstring(node, clean=False)
            if d:
                docstrings.add(d)

    offenders = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and ".cache/huggingface" in node.value
        and node.value not in docstrings
    ]
    assert not offenders, (
        "snapshot lookup hardcodes the default HF cache; this breaks anywhere "
        f"HF_HOME is set (RunPod sets /workspace/.cache/huggingface): {offenders}"
    )


def test_snapshot_resolver_exists_and_delegates_to_hub():
    """Resolution must go through huggingface_hub so every cache-location
    environment variable is honoured for free."""
    src = inspect.getsource(bench.resolve_snapshot_dir)
    assert "snapshot_download" in src
    assert "local_files_only" in src, (
        "resolver must not hit the network -- the file is already downloaded"
    )


def test_snapshot_resolver_raises_a_useful_error_when_truly_absent():
    """A genuinely missing repo should still fail, but the message must not
    tell the user to re-run a download they already ran."""
    try:
        bench.resolve_snapshot_dir("faithqin/definitely-not-a-real-repo-xyz")
    except Exception as e:
        assert "definitely-not-a-real-repo-xyz" in str(e)
        return
    raise AssertionError("resolver should raise for a repo with no local snapshot")


# --------------------------------------------------------------------------
# 8. The bundled tokenizer is referenced RELATIVELY (found Aug 31 2026)
# --------------------------------------------------------------------------
#
# The checkpoint bundles its PaliGemma tokenizer in a `tokenizer/` subfolder --
# which is why inference does not hit Google's gated repo. But the saved
# preprocessor references it as the bare relative name "tokenizer":
#
#   {"registry_name": "tokenizer_processor",
#    "config": {..., "tokenizer_name": "tokenizer"},
#    "artifacts": {"tokenizer_name": "tokenizer"}}
#
# transformers resolves that against the CURRENT WORKING DIRECTORY, not against
# the checkpoint. So loading the processors from anywhere else fails with:
#
#   OSError: tokenizer is not a local folder and is not a valid model identifier
#
# and the "if this is a private repository" hint sends you chasing an auth
# problem that does not exist. This would have broken run_pi05_trial.sh on the
# bench tomorrow, not just the dry-run.
#
# The sanitized copy already symlinks `tokenizer/`, so the fix is to rewrite the
# reference to an absolute path while materializing.


def _preproc_with_relative_tokenizer():
    return {
        "steps": [
            {"registry_name": "normalizer_processor", "config": {}},
            {
                "registry_name": "tokenizer_processor",
                "config": {"max_length": 200, "tokenizer_name": "tokenizer"},
                "artifacts": {"tokenizer_name": "tokenizer"},
            },
        ]
    }


def test_absolutize_rewrites_config_tokenizer_name():
    out = bench.absolutize_tokenizer(_preproc_with_relative_tokenizer(), "/tmp/ckpt")
    step = out["steps"][1]
    assert step["config"]["tokenizer_name"] == "/tmp/ckpt/tokenizer"


def test_absolutize_rewrites_artifacts_tokenizer_name():
    out = bench.absolutize_tokenizer(_preproc_with_relative_tokenizer(), "/tmp/ckpt")
    step = out["steps"][1]
    assert step["artifacts"]["tokenizer_name"] == "/tmp/ckpt/tokenizer"


def test_absolutize_leaves_an_already_absolute_path_alone():
    d = _preproc_with_relative_tokenizer()
    d["steps"][1]["config"]["tokenizer_name"] = "/somewhere/else/tokenizer"
    d["steps"][1]["artifacts"]["tokenizer_name"] = "/somewhere/else/tokenizer"
    out = bench.absolutize_tokenizer(d, "/tmp/ckpt")
    assert out["steps"][1]["config"]["tokenizer_name"] == "/somewhere/else/tokenizer"


def test_absolutize_leaves_a_hub_repo_id_alone():
    """A real Hub id like `google/paligemma-3b-pt-224` must not be turned into
    a bogus local path."""
    d = _preproc_with_relative_tokenizer()
    d["steps"][1]["config"]["tokenizer_name"] = "google/paligemma-3b-pt-224"
    d["steps"][1]["artifacts"]["tokenizer_name"] = "google/paligemma-3b-pt-224"
    out = bench.absolutize_tokenizer(d, "/tmp/ckpt")
    assert out["steps"][1]["config"]["tokenizer_name"] == "google/paligemma-3b-pt-224"


def test_absolutize_does_not_mutate_its_input():
    d = _preproc_with_relative_tokenizer()
    import copy

    before = copy.deepcopy(d)
    bench.absolutize_tokenizer(d, "/tmp/ckpt")
    assert d == before


def test_absolutize_is_a_noop_without_a_tokenizer_step():
    d = {"steps": [{"registry_name": "normalizer_processor", "config": {}}]}
    assert bench.absolutize_tokenizer(d, "/tmp/ckpt") == d
