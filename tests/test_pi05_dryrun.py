"""Regression: the pi0.5 offline dry-run must be a real arbiter, not a smoke test.

Written Aug 31 2026, the night before the first pi0.5 bench session. pi0.5 has
never driven the arm, and every prior "the policy doesn't work" conclusion in
this project turned out to be plumbing: the phantom arrow keys (Aug 17), the
elbow below the training floor (Aug 29), the connect() wrist shift (Aug 29),
the load/current routing found tonight. Bench time is the scarcest resource
before Sep 6, so the plumbing gets proven at a desk first.

The idea is the same one that already works here: **replay as arbiter.** Feed
the policy a frame it was TRAINED on and check it reproduces roughly the action
that was demonstrated. On a training frame a fitted policy should not be far
off. If it is, the observation pipeline is wrong — wrong camera mapping, wrong
state width, wrong normalization — and no amount of bench time will fix it.

Four ways this dry-run could reassure falsely, all pinned here:

1. **Wrong camera mapping.** The dataset stores `observation.images.front` and
   `...wrist`; the policy wants `base_0_rgb` and `left_wrist_0_rgb`. Training
   used an explicit rename_map. Getting it backwards (wrist into the base slot)
   still runs, still produces actions, and is silently wrong.
2. **Grading on the wrong baseline.** "Close to the demonstrated action" means
   nothing without a scale. The comparison must be against the spread of the
   actions themselves, not an absolute degree threshold.
3. **A single lucky frame.** One frame can agree by accident, especially early
   in an episode where the arm is near home. Multiple frames across multiple
   episodes, including mid-episode ones.
4. **Passing when the policy output is degenerate.** All-zeros, all-NaN, or a
   constant chunk can look "close" under a loose metric.

Pure functions only here; the model-loading path is exercised by running the
tool, not by the suite (nothing in tests/ loads 8.7 GB).
"""

import sys

import numpy as np
from conftest import TOOLS

sys.path.insert(0, str(TOOLS))

import pi05_dryrun  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


# --------------------------------------------------------------------------
# 1. Camera mapping must come from the training config, not be reinvented
# --------------------------------------------------------------------------

TRAIN_RENAME_MAP = {
    "observation.images.front": "observation.images.base_0_rgb",
    "observation.images.wrist": "observation.images.left_wrist_0_rgb",
}


def test_rename_map_matches_training_exactly():
    assert pi05_dryrun.RENAME_MAP == TRAIN_RENAME_MAP


def test_apply_rename_moves_front_to_base_and_wrist_to_left_wrist():
    obs = {
        "observation.images.front": "FRONT",
        "observation.images.wrist": "WRIST",
        "observation.state": [0.0] * 6,
    }
    out = pi05_dryrun.apply_rename(obs)
    assert out["observation.images.base_0_rgb"] == "FRONT"
    assert out["observation.images.left_wrist_0_rgb"] == "WRIST"
    assert "observation.images.front" not in out
    assert out["observation.state"] == [0.0] * 6


def test_apply_rename_refuses_to_swap_the_cameras():
    """A backwards mapping runs fine and is silently wrong. Catch it here."""
    obs = {
        "observation.images.front": "FRONT",
        "observation.images.wrist": "WRIST",
    }
    out = pi05_dryrun.apply_rename(obs)
    assert out["observation.images.base_0_rgb"] != "WRIST", "front/wrist are swapped"


def test_apply_rename_raises_on_a_missing_camera():
    """Two cameras in, two renamed out. A missing one means the dataset or the
    rig changed, and the run must stop rather than pad it away."""
    try:
        pi05_dryrun.apply_rename({"observation.images.front": "FRONT"})
    except ValueError as e:
        assert "wrist" in str(e)
        return
    raise AssertionError("apply_rename must raise when a mapped camera is absent")


# --------------------------------------------------------------------------
# 2. Agreement is scored against the action spread, not an absolute threshold
# --------------------------------------------------------------------------


def test_normalized_error_is_relative_to_the_action_spread():
    """1 degree of error on a joint that moves 100 degrees is nothing; the same
    degree on a joint that moves 2 degrees is large.

    The mixed-sign probe at the bottom is the important one. Every probe in this
    section used to have pred >= truth, so `np.mean(np.abs(pred - truth) / safe)`
    could be quietly reduced to `np.mean((pred - truth) / safe)` and stay green.
    A SIGNED mean is the dangerous direction: on real 6-joint output the joints
    that overshoot cancel the ones that undershoot, so a broken pipeline -- a
    swapped front/wrist camera, a mis-widened state, the wrong normalizer, the
    exact failures this tool exists to detect -- scores near 0.0 and prints
    PLUMBING OK.
    """
    pred = np.array([1.0])
    truth = np.array([0.0])
    wide = pi05_dryrun.normalized_error(pred, truth, spread=np.array([100.0]))
    narrow = pi05_dryrun.normalized_error(pred, truth, spread=np.array([2.0]))
    assert wide < narrow
    assert np.isclose(wide, 0.01)
    assert np.isclose(narrow, 0.5)

    mixed = pi05_dryrun.normalized_error(
        np.array([1.0, -1.0]), np.zeros(2), spread=np.array([1.0, 1.0])
    )
    assert np.isclose(mixed, 1.0), (
        f"one joint overshooting by 1 and one undershooting by 1 scored {mixed} "
        "-- the errors CANCELLED, so this is a signed mean, not a mean ABSOLUTE "
        "error. A pipeline that pushes different joints in opposite directions "
        "would score ~0 and be reported as PLUMBING OK."
    )


def test_normalized_error_is_zero_for_an_exact_match():
    x = np.array([10.0, -5.0, 3.0])
    assert pi05_dryrun.normalized_error(x, x, spread=np.array([1.0, 1.0, 1.0])) == 0.0


def test_normalized_error_guards_against_zero_spread():
    """A joint that never moves would divide by zero and report inf or nan."""
    e = pi05_dryrun.normalized_error(np.array([0.5]), np.array([0.0]), spread=np.array([0.0]))
    assert np.isfinite(e)


# --------------------------------------------------------------------------
# The rehearsal harness: main() is RUN, not grepped
# --------------------------------------------------------------------------
#
# Sections 3, 6, 8 and 9 below used to assert that some substring appeared in
# `inspect.getsource(pi05_dryrun)`. That is the guard that has now been wrong
# eight times in this project: source text pins a NAME, never the BEHAVIOUR.
# `chunk = post(chunk)` -> `post(chunk)` keeps the substring "post(" and throws
# the unnormalized actions away; `dataset_from_index` also appears in the
# COMMENT two lines above the line that uses it; and no test in this file ever
# named the inference entry point at all.
#
# So main() is executed here for real. torch, lerobot, the 8.7 GB checkpoint and
# the dataset are all replaced by stubs inside a SUBPROCESS -- subprocess, not
# monkeypatch, because stubbing lerobot in-process would leak into every other
# test in the suite. What runs is pi05_dryrun's own control flow: probe
# indexing, the inference call, the postprocessor, the sanity gate, the verdict,
# the exit code. The stub records what the tool did to it, and the arithmetic is
# rigged so each mutation lands on its own identifiable number: row k of the
# emitted chunk is (truth + k) / SCALE, so
#
#   post(chunk)[0]   (correct)                     scores exactly 0.000
#   post(chunk)[-1]  (index slip)                  scores (T - 1) / SPREAD
#   chunk[0]         (post's return value dropped) scores ~SCALE x truth
#
# `_decode_scored_err` turns the observed number back into which of those it is.

_REHEARSAL_RUNNER = r'''
"""Execute pi05_dryrun.main() against stub torch/lerobot modules.

Written by tests/test_pi05_dryrun.py; not a standalone tool. MODE "healthy"
emits a chunk that moves over time; MODE "degenerate" emits the same action at
every timestep, which check_chunk_sane must reject.
"""
import json
import sys
import types
from contextlib import contextmanager

TOOLS, OUT, REPORT, MODE, DEVICE = sys.argv[1:6]
sys.path.insert(0, TOOLS)

import numpy as np

N_EPS, EP_LEN, SPREAD, SCALE, T, SEED, N = 5, 100, 20.0, 10.0, 8, 20260831, 6

rec = {"fetched": [], "calls": [], "to_device": None, "pre_kwargs": None,
       "n_eps": N_EPS, "ep_len": EP_LEN, "spread": SPREAD, "scale": SCALE,
       "T": T, "seed": SEED, "n": N, "mode": MODE}


def truth_for(idx):
    """The demonstrated action at dataset row idx: distinct per row AND joint."""
    return np.array([idx + 10.0 * j for j in range(6)], dtype=float)


def state_for(idx):
    """observation.state trails the action by 1 deg, so the trivial baseline
    scores 1/SPREAD and the run has a meaningful control."""
    return truth_for(idx) - 1.0


class _Meta:
    def __init__(self):
        self.total_episodes = N_EPS
        self.total_frames = N_EPS * EP_LEN
        self.stats = {"action": {"min": np.zeros(6), "max": np.full(6, SPREAD)}}
        self.episodes = [
            {"dataset_from_index": e * EP_LEN, "dataset_to_index": (e + 1) * EP_LEN}
            for e in range(N_EPS)
        ]


class _Dataset:
    def __init__(self, repo_id):
        self.meta = _Meta()

    def __getitem__(self, idx):
        rec["fetched"].append(int(idx))
        return {
            "observation.images.front": "front-row-%d" % idx,
            "observation.images.wrist": "wrist-row-%d" % idx,
            "observation.state": state_for(idx),
            "action": truth_for(idx),
        }


class _Cfg:
    output_features = {"action": types.SimpleNamespace(shape=(6,))}

    @classmethod
    def from_pretrained(cls, path):
        return cls()


class _Policy:
    """Shaped like PI05Policy: both entry points exist, and both are recorded."""

    @classmethod
    def from_pretrained(cls, path, config=None):
        return cls()

    def to(self, device):
        rec["to_device"] = device
        return self

    def eval(self):
        pass

    def _chunk(self, batch):
        # NORMALIZED actions, as pi0.5 emits: post() multiplies by SCALE.
        truth = np.asarray(batch["observation.state"], dtype=float) + 1.0
        if MODE == "degenerate":
            rows = [truth / SCALE for _ in range(T)]      # never moves in time
        else:
            rows = [(truth + k) / SCALE for k in range(T)]
        return np.stack(rows)[None, ...]                  # (1, T, 6)

    def predict_action_chunk(self, batch):
        rec["calls"].append("predict_action_chunk")
        return self._chunk(batch)

    def select_action(self, batch):
        rec["calls"].append("select_action")
        return self._chunk(batch)


def _make_pre_post(cfg, pretrained_path=None, preprocessor_overrides=None, **kw):
    rec["pre_kwargs"] = {
        "pretrained_path": str(pretrained_path),
        "preprocessor_overrides": preprocessor_overrides,
    }
    return (lambda obs: dict(obs)), (lambda x: np.asarray(x, dtype=float) * SCALE)


@contextmanager
def _no_grad():
    yield


def _mod(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


_mod("torch", no_grad=_no_grad)
_mod("pi05_latency_bench",
     apply_runtime_config=lambda cfg, dtype, device: None,
     materialize_local_checkpoint=lambda snap, out: (out, ["inert_0_6_2_field"]),
     resolve_snapshot_dir=lambda repo_id: "/stub/snapshot/" + repo_id)
_mod("lerobot")
_mod("lerobot.configs")
_mod("lerobot.configs.policies", PreTrainedConfig=_Cfg)
_mod("lerobot.datasets")
_mod("lerobot.datasets.lerobot_dataset", LeRobotDataset=_Dataset)
_mod("lerobot.policies")
_mod("lerobot.policies.pi05")
_mod("lerobot.policies.pi05.modeling_pi05", PI05Policy=_Policy)
_mod("lerobot.policies.factory", make_pre_post_processors=_make_pre_post)

import pi05_dryrun

sys.argv = ["pi05_dryrun", "--device", DEVICE, "--n", str(N), "--seed", str(SEED),
            "--json-out", REPORT]
try:
    rec["rc"] = pi05_dryrun.main()
except BaseException as exc:          # a crash is a finding too -- report it
    rec["rc"] = None
    rec["error"] = "%s: %s" % (type(exc).__name__, exc)
with open(OUT, "w") as fh:
    json.dump(rec, fh)
'''


def _rehearse(tmp_path, mode="healthy", device="cpu"):
    """Run pi05_dryrun.main() end to end on stubs. Returns (record, report).

    `record` is what the stubs saw -- which dataset rows were read, which
    inference method was called, which processor overrides were passed.
    `report` is the tool's own --json-out: the numbers and the verdict a human
    would read off the terminal.
    """
    import json
    import subprocess

    runner = tmp_path / "rehearse_pi05.py"
    runner.write_text(_REHEARSAL_RUNNER)
    record, report = tmp_path / "record.json", tmp_path / "report.json"
    r = subprocess.run(
        [sys.executable, str(runner), str(TOOLS), str(record), str(report), mode, device],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert record.exists(), (
        "the pi0.5 dry-run rehearsal never finished:\n" + r.stdout[-3000:] + r.stderr[-3000:]
    )
    rec = json.loads(record.read_text())
    assert rec.get("error") is None, "pi05_dryrun.main() raised %s" % rec.get("error")
    rec["stdout"] = r.stdout
    return rec, json.loads(report.read_text())


def _decode_scored_err(err, rec):
    """Name which wrong number the rehearsal just measured.

    The scored value is meant to be the FIRST action of the chunk, after the
    postprocessor. Every plausible slip produces its own signature.
    """
    for k in range(1, rec["T"]):
        if np.isclose(err, k / rec["spread"]):
            return (
                f"{err:.3f} is chunk row {k}: the scored action is {k} ticks in "
                f"the FUTURE (at 20 Hz, {k * 50} ms), not the one predicted for "
                f"this frame -- an index slip in `normalized_error(chunk_np[0], ...)`."
            )
    if err > 1.0:
        return (
            "an error this large is normalized-vs-degrees: the postprocessor's "
            "return value was thrown away (`post(chunk)` instead of "
            "`chunk = post(chunk)`), which is harness bug #2 of Aug 31 -- the "
            "version that could only ever return PLUMBING SUSPECT."
        )
    return "no known mutation signature matches this number."


# --------------------------------------------------------------------------
# 3. Degenerate output must fail, however close the metric says it is
# --------------------------------------------------------------------------


def test_degenerate_all_zeros_is_rejected_and_gates_the_verdict(tmp_path):
    """All-zeros is degenerate -- and the gate must actually GATE the verdict.

    The first half is the original test. The second half is the Sep 5 SmolVLA
    bug (tests/test_smolvla_dryrun_gate.py) rehearsed for pi0.5: there
    `check_chunk_sane` was live code that nothing ever consulted, `all_sane`
    could never be False, and both Sep 4 verdicts were reported with no sanity
    gate behind them. Deleting `if all_sane` here is the identical bug, and the
    four degeneracy tests in this section could not see it -- every one of them
    calls check_chunk_sane directly and none of them runs the tool.

    So the rehearsal emits a chunk that is constant in time but scores
    PERFECTLY (0.000, far under the pre-declared threshold). The only thing that
    can turn that into PLUMBING SUSPECT is the sanity gate.
    """
    chunk = np.zeros((50, 6))
    ok, why = pi05_dryrun.check_chunk_sane(chunk, n_joints=6)
    assert not ok and "constant" in why.lower()

    rec, report = _rehearse(tmp_path, mode="degenerate")
    assert report["mean_norm_err"] < pi05_dryrun.AGREEMENT_THRESHOLD, (
        f"the degenerate chunk scored {report['mean_norm_err']:.3f}, over the "
        f"{pi05_dryrun.AGREEMENT_THRESHOLD} threshold. It is rigged to score "
        "0.000 precisely so that ONLY the sanity gate can produce a SUSPECT "
        "verdict here; a number this large means the SCORING path changed -- "
        "read test_dryrun_scores_the_postprocessed_first_action_of_"
        "predict_action_chunk first, it says which mutation this is."
    )
    assert not any(row["sane"] for row in report["rows"]), (
        "check_chunk_sane did not flag a chunk that is identical at every "
        "timestep"
    )
    assert report["verdict"] == "PLUMBING SUSPECT", (
        f"a time-constant action chunk scored {report['mean_norm_err']:.3f} and "
        f"was graded {report['verdict']!r}. check_chunk_sane is dead decoration "
        "again: the verdict no longer consults all_sane, so an all-zeros, "
        "all-NaN or unsliced 32-wide chunk now clears pi0.5 for the bench."
    )
    assert rec["rc"] == 1, "a degenerate dry-run must exit non-zero"


def test_degenerate_constant_chunk_is_rejected():
    chunk = np.tile(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), (50, 1))
    ok, why = pi05_dryrun.check_chunk_sane(chunk, n_joints=6)
    assert not ok and "constant" in why.lower()


def test_nan_chunk_is_rejected():
    chunk = np.zeros((50, 6))
    chunk[3, 2] = np.nan
    ok, why = pi05_dryrun.check_chunk_sane(chunk, n_joints=6)
    assert not ok and "finite" in why.lower()


def test_wrong_action_width_is_rejected():
    """pi0.5 pads internally to 32 and slices back to 6. A 32-wide chunk
    reaching the robot would be an unsliced tensor."""
    ok, why = pi05_dryrun.check_chunk_sane(np.random.randn(50, 32), n_joints=6)
    assert not ok and "6" in why


def test_a_plausible_varying_chunk_passes():
    rng = np.random.default_rng(0)
    chunk = np.cumsum(rng.normal(0, 0.5, size=(50, 6)), axis=0) + 40.0
    ok, why = pi05_dryrun.check_chunk_sane(chunk, n_joints=6)
    assert ok, why


# --------------------------------------------------------------------------
# 4. Frame selection must span episodes and reach mid-episode
# --------------------------------------------------------------------------


def test_probe_frames_span_multiple_episodes():
    frames = pi05_dryrun.pick_probe_frames(n_episodes=50, episode_len=200, n=6, seed=7)
    assert len({ep for ep, _ in frames}) >= 4, "probe frames cluster in too few episodes"


def test_probe_frames_are_not_all_near_the_start():
    """Frame 0 is near home in every episode, where any policy looks good."""
    frames = pi05_dryrun.pick_probe_frames(n_episodes=50, episode_len=200, n=6, seed=7)
    assert max(f for _, f in frames) > 60, "no mid-episode frames — the easy region only"


def test_probe_frames_are_deterministic_for_a_seed():
    a = pi05_dryrun.pick_probe_frames(n_episodes=50, episode_len=200, n=6, seed=3)
    b = pi05_dryrun.pick_probe_frames(n_episodes=50, episode_len=200, n=6, seed=3)
    assert a == b


def test_probe_frames_stay_in_range():
    frames = pi05_dryrun.pick_probe_frames(n_episodes=12, episode_len=90, n=10, seed=1)
    for ep, f in frames:
        assert 0 <= ep < 12
        assert 0 <= f < 90


# --------------------------------------------------------------------------
# 5. The verdict must not be graded on a curve after the fact
# --------------------------------------------------------------------------


def test_verdict_threshold_is_declared_as_a_constant():
    """Pre-declared, so it cannot be loosened once a number comes back.

    The value itself is pinned, not just its type and range: `isinstance(...,
    float)` and `0.0 < t < 1.0` accept any number in the band, so 0.25 could be
    quietly widened to 0.40 and stay green -- and the only cross-check
    (test_smolvla_dryrun.py) asserts the two tools' constants are EQUAL, so
    moving both together is green too. 0.40 is not an abstract loosening: it
    flips the historical pre-fix result 0.379 from PLUMBING SUSPECT to PLUMBING
    OK, which is grading on a curve after seeing the result -- the one move this
    project's methodology exists to prevent.
    """
    assert isinstance(pi05_dryrun.AGREEMENT_THRESHOLD, float)
    assert pi05_dryrun.AGREEMENT_THRESHOLD == 0.25, (
        f"the pre-declared threshold moved to {pi05_dryrun.AGREEMENT_THRESHOLD}. "
        "It was declared at 0.25 on Aug 31 BEFORE any number came back; a "
        "re-grade needs a written argument in the the lab notebook (private), not an edit."
    )
    assert pi05_dryrun.verdict(0.379) == "PLUMBING SUSPECT", (
        "0.379 was the first (pre-fix) dry-run result and must still fail: a "
        "threshold that admits it is a threshold that would have cleared pi0.5 "
        "for the bench on a harness that was itself broken."
    )


def test_verdict_uses_the_declared_threshold():
    t = pi05_dryrun.AGREEMENT_THRESHOLD
    assert pi05_dryrun.verdict(t - 0.01) == "PLUMBING OK"
    assert pi05_dryrun.verdict(t + 0.01) == "PLUMBING SUSPECT"


# --------------------------------------------------------------------------
# 6. Episode indexing API — pinned after it moved (Aug 31 2026)
# --------------------------------------------------------------------------


def test_dryrun_reads_frames_from_inside_the_probed_episode(tmp_path):
    """`LeRobotDataset.episode_data_index` does not exist in lerobot 0.6.1;
    episode bounds live on `meta.episodes` as dataset_from_index /
    dataset_to_index.

    This used to be pinned by grepping the source for those two names -- but
    both appear in the COMMENT two lines above the line that uses them, so the
    prose alone satisfied the grep and `lo, hi = row["dataset_to_index"],
    row["dataset_from_index"]` stayed green. Transposed, `idx = min(lo + f,
    hi - 1)` collapses to from_index - 1 with no exception: every probe lands on
    the LAST frame of the PREVIOUS episode, a seated end-of-episode pose where
    the arm is nearly static -- precisely the easy region section 4 exists to
    exclude. So the rehearsal records which rows the tool actually read.
    """
    import inspect

    src = inspect.getsource(pi05_dryrun)
    assert "episode_data_index" not in src, "using an API that does not exist in 0.6.1"

    rec, _ = _rehearse(tmp_path)
    probes = pi05_dryrun.pick_probe_frames(
        n_episodes=rec["n_eps"], episode_len=rec["ep_len"], n=rec["n"], seed=rec["seed"]
    )
    expected = [ep * rec["ep_len"] + f for ep, f in probes]
    assert rec["fetched"] == expected, (
        f"for probes {probes} the dry-run read dataset rows {rec['fetched']}, "
        f"but episode e occupies rows [e*{rec['ep_len']}, (e+1)*{rec['ep_len']}) "
        f"so it should have read {expected}. Rows that are all one BELOW an "
        "episode start mean dataset_from_index / dataset_to_index are "
        "transposed: every probe scored the previous episode's final, seated "
        "frame instead of the approach/insertion frame it selected."
    )


# --------------------------------------------------------------------------
# 7. A verdict needs a control (added Aug 31 2026, after the first result)
# --------------------------------------------------------------------------
#
# The first dry-run returned mean normalized error 0.379 against a pre-declared
# threshold of 0.25 -> PLUMBING SUSPECT. That number is not interpretable on its
# own: it could mean the pipeline is wrong, or that 0.25 was simply too tight
# for this policy and task.
#
# The threshold does NOT move. Re-grading a pre-declared bar after seeing the
# result is exactly the failure this project's methodology exists to prevent.
# Instead the run gains a CONTROL that makes the number mean something.
#
# The control: on a position-controlled arm at 20 Hz, the demonstrated action at
# frame t is very close to the observed state at frame t -- the commanded
# position barely moves in 50 ms. So "copy the current state" is a trivial
# predictor that any fitted policy must beat. If the trivial baseline scores far
# BETTER than pi0.5, pi0.5 is worse than doing nothing and the pipeline is the
# suspect. If the trivial baseline scores comparably badly, the metric itself is
# mis-scaled and the verdict is uninformative rather than damning.


def test_trivial_baseline_copies_state_to_action():
    state = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 1.0])
    assert np.allclose(pi05_dryrun.trivial_baseline(state), state)


def test_relative_skill_is_one_when_policy_matches_baseline():
    assert np.isclose(pi05_dryrun.relative_skill(0.30, 0.30), 1.0)


def test_relative_skill_below_one_means_better_than_trivial():
    """Lower error than the baseline is skill."""
    assert pi05_dryrun.relative_skill(0.10, 0.40) < 1.0


def test_relative_skill_above_one_means_worse_than_doing_nothing():
    assert pi05_dryrun.relative_skill(0.40, 0.10) > 1.0


def test_relative_skill_handles_a_perfect_baseline():
    """If the baseline is exactly 0, the ratio must stay finite."""
    assert np.isfinite(pi05_dryrun.relative_skill(0.3, 0.0))


def test_diagnosis_calls_out_worse_than_trivial():
    """Including at the BAND EDGE, which is where a loosening would happen.

    All three diagnose probes used to sit far from both edges (ratios 7.6, 1.05,
    0.25), so `if ratio > 1.5` could be widened to `if ratio > 3.0` and stay
    green -- and then a policy scoring 2x worse than the copy-your-own-position
    control gets described as "comparable to the trivial baseline ...
    uninformative rather than damning" instead of "WORSE THAN DOING NOTHING ...
    check camera mapping, state routing, normalization". Relative skill is the
    figure that now carries the verdict, and the diagnosis sentence is what gets
    pasted into the session log.
    """
    d = pi05_dryrun.diagnose(policy_err=0.379, baseline_err=0.05)
    assert "worse than" in d.lower()
    assert "worse than" in pi05_dryrun.diagnose(0.152, 0.10).lower(), (
        "ratio 1.52 is past the 1.5 band edge and must read WORSE THAN DOING "
        "NOTHING; the upper band has been widened"
    )
    assert "worse than" not in pi05_dryrun.diagnose(0.148, 0.10).lower(), (
        "ratio 1.48 is inside the 1.5 band edge and must NOT be called worse "
        "than doing nothing; the upper band has been tightened"
    )


def test_diagnosis_calls_out_a_miscalibrated_metric():
    """Baseline just as bad -> the scale is wrong, not necessarily the policy.

    Both edges of this middle band are pinned: below 0.85 the tool claims real
    skill, above 1.5 it claims worse than nothing, and moving either edge
    silently rewrites the conclusion a borderline run reports.
    """
    d = pi05_dryrun.diagnose(policy_err=0.379, baseline_err=0.360)
    assert "metric" in d.lower() or "scale" in d.lower()
    edge = pi05_dryrun.diagnose(0.086, 0.10).lower()
    assert "metric" in edge or "scale" in edge, (
        "ratio 0.86 is only just better than trivial and must stay in the "
        "uninformative band, not be reported as real skill"
    )


def test_diagnosis_confirms_real_skill():
    """And only below the 0.85 edge, so the claim keeps its meaning."""
    d = pi05_dryrun.diagnose(policy_err=0.10, baseline_err=0.40)
    assert "beats" in d.lower() or "skill" in d.lower()
    edge = pi05_dryrun.diagnose(0.084, 0.10).lower()
    assert "beats" in edge or "skill" in edge, (
        "ratio 0.84 is past the 0.85 band edge and must read as real skill; "
        "the lower band has been moved"
    )


# --------------------------------------------------------------------------
# 8. The dry-run must UNNORMALIZE before comparing (found Aug 31 2026)
# --------------------------------------------------------------------------
#
# The first dry-run reported mean normalized error 0.379 -> PLUMBING SUSPECT.
# The suspect was the harness, not the pipeline: it compared
# `predict_action_chunk` output directly against ground-truth actions in
# degrees. But that output is in NORMALIZED space. pi0.5's postprocessor is
# `[unnormalize, AbsoluteActions, to_cpu]` (processor_pi05.py:150), and the real
# rollout applies it every tick (`rollout/inference/sync.py:112`).
#
# Comparing normalized predictions against degree-valued truth guarantees a
# large error regardless of how good the policy is -- a harness that can only
# ever return SUSPECT. It would have sent a false "do not go to the bench".
#
# Two invariants pinned here:
#   * the postprocessor is applied to the chunk before scoring
#   * processors come from the CHECKPOINT (its own saved normalizer stats),
#     not rebuilt from dataset statistics, which can differ


def test_dryrun_scores_the_postprocessed_first_action_of_predict_action_chunk(tmp_path):
    """The number the whole gate rests on must be post(predict_action_chunk())[0].

    Three mutations this catches, none of which `"post(" in src` could see:

    * `chunk = post(chunk)` -> `post(chunk)`. The call is still in the source, so
      the old grep is still satisfied, but the return value is discarded and
      `chunk` stays in NORMALIZED space while `truth` is in degrees. That is
      harness bug #2 of Aug 31 verbatim -- the harness that could only ever
      return PLUMBING SUSPECT, and that reported 0.379 three times before the
      fix moved the number to 0.035.
    * `chunk_np[0]` -> `chunk_np[-1]`. A one-character slip in the only line
      that turns model output into the number the gate rests on: it scores a
      command 2.5 s in the future against the present, pins the tool at SUSPECT,
      and points the operator at the observation pipeline.
    * `predict_action_chunk` -> `select_action`. No test in this file named
      either entry point, in either direction. They are not interchangeable:
      select_action asserts RTC is off, slices to n_action_steps and serves one
      (B, D) action from a queue, while this rehearsal is written around the
      whole (B, T, D) chunk.

    The stub emits row k = (truth + k) / SCALE, so the correct read scores
    exactly 0.000 and each mistake scores its own signature.
    """
    rec, report = _rehearse(tmp_path)
    assert rec["calls"] == ["predict_action_chunk"] * len(report["rows"]), (
        f"the dry-run's inference calls were {sorted(set(rec['calls']))}. The "
        "rehearsal must take predict_action_chunk -- the (B, T, D) chunk path "
        "it post-processes and slices; select_action serves a single sliced "
        "action from a queue and exercises a different rank of tensor."
    )
    scored = [row["norm_err"] for row in report["rows"]]
    assert all(np.isclose(e, 0.0) for e in scored), (
        f"scored errors {np.round(scored, 3).tolist()}, but post(chunk)[0] IS "
        f"the demonstrated action here, so every one must be 0.000. "
        f"{_decode_scored_err(scored[0], rec)}"
    )
    assert report["verdict"] == "PLUMBING OK" and rec["rc"] == 0


def test_dryrun_loads_processors_from_the_checkpoint():
    """`make_pre_post_processors(cfg, pretrained_path=...)` loads the saved
    normalizer that shipped with the checkpoint. Rebuilding from dataset stats
    risks a different normalization than training used."""
    import inspect

    src = inspect.getsource(pi05_dryrun)
    assert "pretrained_path" in src, (
        "processors must be loaded from the checkpoint, not rebuilt from "
        "dataset statistics"
    )


# --------------------------------------------------------------------------
# 9. The dry-run must mirror the ROLLOUT's processor construction exactly
# --------------------------------------------------------------------------
#
# Found Aug 31 2026: loading the checkpoint's saved processors failed with
#
#   ValueError: Failed to instantiate processor step 'device_processor' with
#   config: {'device': 'cuda', ...}. Requested device 'cuda' but CUDA is not
#   available.
#
# The saved preprocessor pins device=cuda (it was written on an H200). The real
# rollout already handles this at rollout/context.py:537-546:
#
#   preprocessor_overrides={
#       "device_processor": {"device": cfg.device},
#       "rename_observations_processor": {"rename_map": cfg.rename_map},
#   }
#
# Two consequences:
#   * the dry-run must pass the same overrides, or it tests a path the bench
#     will never take;
#   * the camera rename belongs to `rename_observations_processor`, NOT to a
#     hand-rolled dict rewrite. Doing both would rename twice.
#
# The postprocessor's own device_processor is pinned to 'cpu' and is CORRECT --
# overriding it would move actions off the CPU the robot reads them from.


def test_dryrun_overrides_the_preprocessor_device_with_the_RUNS_device(tmp_path):
    """The override must carry THIS RUN's device, not a hard-coded string.

    The old guard only checked that the identifiers "preprocessor_overrides" and
    "device_processor" appeared in the source, so `{"device": args.device}` could
    revert to `{"device": "cuda"}` -- the value the checkpoint actually saved,
    written on the H200 -- and stay green while reproducing the Aug 31 crash:
    ValueError: Failed to instantiate processor step 'device_processor' ...
    Requested device 'cuda' but CUDA is not available. So the rehearsal runs with
    --device cpu and reads back what the processors were actually built with.

    The rename map is checked here too: the rollout renames inside
    rename_observations_processor (rollout/context.py:537-546), so the dry-run
    must hand it the training map rather than rewriting keys by hand.
    """
    rec, _ = _rehearse(tmp_path, device="cpu")
    overrides = rec["pre_kwargs"]["preprocessor_overrides"]
    got = overrides["device_processor"]["device"]
    assert got == "cpu", (
        f"the preprocessor was built for device {got!r} while the run was "
        "--device cpu. The checkpoint's saved preprocessor pins 'cuda'; "
        "anything but the run's own device is the Aug 31 'Requested device "
        "cuda but CUDA is not available' crash, on the Mac, the night before "
        "a bench session."
    )
    assert overrides["rename_observations_processor"]["rename_map"] == TRAIN_RENAME_MAP, (
        "rename_observations_processor did not get the training rename map -- "
        "the dry-run is exercising a camera mapping the bench never takes"
    )
    assert rec["to_device"] == "cpu", "the policy itself must move to the run's device"


def test_dryrun_renames_via_the_processor_not_by_hand():
    """The rollout renames inside rename_observations_processor. The dry-run
    must do the same or it is exercising a different pipeline."""
    import inspect

    src = inspect.getsource(pi05_dryrun)
    assert "rename_observations_processor" in src


def test_dryrun_does_not_override_the_postprocessor_device():
    """postprocessor device_processor is 'cpu' by design; the robot reads
    actions from the CPU."""
    import inspect

    src = inspect.getsource(pi05_dryrun)
    assert "postprocessor_overrides" not in src, (
        "postprocessor device is 'cpu' on purpose — do not override it"
    )
