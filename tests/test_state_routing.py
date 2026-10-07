"""Regression: deciding when to route .load/.current into observation.state.

`rollout/context.py` (local patch, Aug 15 2026) widens the observation filter
from `.pos`/`.vel` to also include `.load`/`.current` when the loaded policy
expects a wider state than the joint-position keys provide. Without it ACT-B
crashed on a 6-vs-18 normalizer mismatch while ACT-A silently worked.

Two failures of that inference rule, both found Aug 31 2026:

1. **pi0.5 pads, and padding is not dimensionality.** `pi05-tube-A-v3` trained
   on `so101-tube-insert-v3-noload`, whose `observation.state` is **6-dim**
   (verified in the parquet). `Pi05PrepareStateTokenizerProcessorStep` pads it
   to `max_state_dim=32` before the model sees it, so dims 6-31 were **zero in
   every training frame**. But the saved config declares
   `input_features["observation.state"].shape = [32]`, and 32 > 6, so the
   inference rule fires and routes load+current in. The policy would then
   receive real load/current values in dims 6-17 -- coordinates it learned as
   always-zero -- and pad the rest.

   **This does not crash.** The shapes line up. It is a silent distribution
   shift that would have looked like "pi0.5 just doesn't work on this task,"
   and it would have burned a bench session to discover.

2. **DAgger recording needs the wide filter regardless of the policy.**
   `dataset_features` is built from the same filtered dict
   (`context.py:384-389`), so collecting corrections under a 6-dim policy
   writes a 6-dim dataset with load and current dropped at the source and
   unretrofittable. An explicit override is the only way to record 18-dim while
   rolling out a 6-dim policy.

The decision is a pure function so it can be tested without a robot, a policy,
or site-packages. `context.py` embeds the same logic; test_sitepackages_patches
checks the markers are present.
"""

import sys

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))

import state_routing  # noqa: E402

# The rig: six joints, each contributing one .pos key. No .vel on a pure arm.
N_STOCK = 6


# --------------------------------------------------------------------------
# The original behaviour must survive -- this is what ACT-B depends on
# --------------------------------------------------------------------------


def test_act_b_still_widens():
    """ACT-B trained on 18-dim state and CRASHES without the wide filter."""
    assert state_routing.should_widen_state(policy_state_dim=18, n_stock=N_STOCK) is True


def test_act_a_still_narrow():
    """ACT-A trained on 6-dim; widening would feed it 18 into a 6-dim normalizer."""
    assert state_routing.should_widen_state(policy_state_dim=6, n_stock=N_STOCK) is False


def test_no_policy_state_is_narrow():
    """No declared state feature -> stock behaviour, never speculative widening."""
    assert state_routing.should_widen_state(policy_state_dim=None, n_stock=N_STOCK) is False


# --------------------------------------------------------------------------
# The pi0.5 padding bug
# --------------------------------------------------------------------------


def test_pi05_padded_state_does_not_widen():
    """THE BUG. 32 > 6, but 32 is `max_state_dim` padding, not real width."""
    assert (
        state_routing.should_widen_state(
            policy_state_dim=32, n_stock=N_STOCK, max_state_dim=32
        )
        is False
    )


def test_padding_exclusion_is_exact_not_a_threshold():
    """Only a state dim EQUAL to max_state_dim is treated as padding. A policy
    that genuinely declares 18 while padding to 32 is really 18-dim."""
    assert (
        state_routing.should_widen_state(
            policy_state_dim=18, n_stock=N_STOCK, max_state_dim=32
        )
        is True
    )


def test_padding_exclusion_does_not_leak_to_unpadded_policies():
    """ACT has no max_state_dim at all; its 18 must still widen."""
    assert (
        state_routing.should_widen_state(
            policy_state_dim=18, n_stock=N_STOCK, max_state_dim=None
        )
        is True
    )


def test_a_six_dim_policy_that_pads_is_still_narrow():
    assert (
        state_routing.should_widen_state(policy_state_dim=6, n_stock=N_STOCK, max_state_dim=32)
        is False
    )


# --------------------------------------------------------------------------
# The explicit override, for DAgger recording
# --------------------------------------------------------------------------


def test_force_true_overrides_a_narrow_policy():
    """DAgger under 6-dim ACT-A must still RECORD 18-dim, or the corrections
    can never train a B-side twin. This is a one-way door."""
    assert (
        state_routing.should_widen_state(policy_state_dim=6, n_stock=N_STOCK, force=True) is True
    )


def test_force_false_overrides_a_wide_policy():
    """Symmetric, so the override is a real switch and not a one-way nudge."""
    assert (
        state_routing.should_widen_state(policy_state_dim=18, n_stock=N_STOCK, force=False)
        is False
    )


def test_force_beats_the_padding_exclusion():
    """A hypothetical pi0.5-B trained on 18-dim would declare 32 and be caught
    by the padding rule. The override is the documented escape hatch."""
    assert (
        state_routing.should_widen_state(
            policy_state_dim=32, n_stock=N_STOCK, max_state_dim=32, force=True
        )
        is True
    )


# --------------------------------------------------------------------------
# Env parsing -- the override reaches context.py through the environment
# --------------------------------------------------------------------------


def test_env_unset_is_none_not_false():
    """None means 'infer'; False means 'forced narrow'. Collapsing them would
    make an unset variable silently disable ACT-B's widening."""
    assert state_routing.force_from_env({}) is None


def test_env_truthy_values():
    for v in ("1", "true", "TRUE", "yes", "on"):
        assert state_routing.force_from_env({"CAPSTONE_FORCE_WIDE_STATE": v}) is True, v


def test_env_falsy_values():
    for v in ("0", "false", "FALSE", "no", "off"):
        assert state_routing.force_from_env({"CAPSTONE_FORCE_WIDE_STATE": v}) is False, v


def test_env_garbage_raises_rather_than_guessing():
    """A typo'd value must not silently mean 'infer' -- that is how a DAgger
    session records 6-dim while the operator believes it is recording 18."""
    try:
        state_routing.force_from_env({"CAPSTONE_FORCE_WIDE_STATE": "yeah"})
    except ValueError:
        return
    raise AssertionError("an unparseable override value must raise, not default")


def test_empty_string_is_treated_as_unset():
    assert state_routing.force_from_env({"CAPSTONE_FORCE_WIDE_STATE": ""}) is None


# --------------------------------------------------------------------------
# The normalizer-stats discriminator (found by the Sep 3 2026 night audit)
#
# smolvla-tube-B-v3 declares observation.state.shape = [6] with
# max_state_dim = 32, while its OWN bundled normalizer carries 18-dim stats.
# The declared width therefore routes it NARROW, and a 6-dim state hits an
# 18-dim normalizer: RuntimeError on the first inference tick.
#
# Measured on the real checkpoints (Sep 3 2026):
#     smolvla-tube-A-v3   declared [6]   max 32     stats [6]    -> narrow
#     smolvla-tube-B-v3   declared [6]   max 32     stats [18]   -> WIDE
#     pi05-tube-A-v3      declared [32]  max 32     stats [6]    -> narrow
#     act-tube-A-v3       declared [6]   max None   stats [6]    -> narrow
#     act-tube-B-v3       declared [18]  max None   stats [18]   -> WIDE
#
# Both SmolVLA checkpoints declare 6, so the declared width cannot tell them
# apart. The normalizer stats are the only signal, and they are also what the
# policy is actually normalized against -- so when the two disagree, the stats
# decide. This is additive: omitting stats_state_dim reproduces the old answers
# exactly, which is why test_a_six_dim_policy_that_pads_is_still_narrow above
# still passes unchanged.
# --------------------------------------------------------------------------

REAL_CHECKPOINTS = [
    # (name,              declared, max_state_dim, stats, expect_wide)
    ("smolvla-tube-A-v3", 6, 32, 6, False),
    ("smolvla-tube-B-v3", 6, 32, 18, True),
    ("pi05-tube-A-v3", 32, 32, 6, False),
    ("act-tube-A-v3", 6, None, 6, False),
    ("act-tube-B-v3", 18, None, 18, True),
]


def test_every_real_checkpoint_routes_correctly():
    """The five checkpoints that exist, by their measured widths."""
    wrong = []
    for name, declared, maxsd, stats, expect in REAL_CHECKPOINTS:
        got = state_routing.should_widen_state(
            policy_state_dim=declared, n_stock=N_STOCK,
            max_state_dim=maxsd, stats_state_dim=stats,
        )
        if got is not expect:
            wrong.append(f"{name}: declared={declared} max={maxsd} stats={stats} -> {got}, want {expect}")
    assert not wrong, "\n".join(wrong)


def test_smolvla_b_widens_on_its_stats_despite_declaring_six():
    """The bug this discriminator exists for. Without it: 6-vs-18 crash."""
    assert (
        state_routing.should_widen_state(
            policy_state_dim=6, n_stock=N_STOCK, max_state_dim=32, stats_state_dim=18
        )
        is True
    )


def test_smolvla_a_stays_narrow_on_matching_stats():
    """Same declared width as B. Must NOT be dragged wide with it."""
    assert (
        state_routing.should_widen_state(
            policy_state_dim=6, n_stock=N_STOCK, max_state_dim=32, stats_state_dim=6
        )
        is False
    )


def test_pi05_stays_narrow_via_stats_too():
    """pi0.5 declares 32 and trained on 6. The stats say 6, so narrow -- and
    now for a better reason than the max_state_dim heuristic."""
    assert (
        state_routing.should_widen_state(
            policy_state_dim=32, n_stock=N_STOCK, max_state_dim=32, stats_state_dim=6
        )
        is False
    )


def test_stats_argument_is_additive_and_changes_nothing_when_omitted():
    """Backwards compatibility: every prior case must answer identically."""
    for declared, maxsd in ((18, None), (6, None), (6, 32), (32, 32), (None, None)):
        assert state_routing.should_widen_state(
            policy_state_dim=declared, n_stock=N_STOCK, max_state_dim=maxsd
        ) is state_routing.should_widen_state(
            policy_state_dim=declared, n_stock=N_STOCK, max_state_dim=maxsd,
            stats_state_dim=None,
        )


def test_force_still_beats_the_stats_discriminator():
    """The DAgger override outranks every inference path, including this one."""
    assert state_routing.should_widen_state(
        policy_state_dim=6, n_stock=N_STOCK, max_state_dim=32,
        stats_state_dim=18, force=False,
    ) is False
    assert state_routing.should_widen_state(
        policy_state_dim=6, n_stock=N_STOCK, max_state_dim=32,
        stats_state_dim=6, force=True,
    ) is True


def test_declared_stats_disagreement_is_detectable():
    """A checkpoint whose declared width and normalizer stats disagree is
    malformed metadata, and the caller must be able to gate on it rather than
    silently rely on the stats fallback."""
    assert state_routing.state_width_disagrees(declared=6, stats=18) is True
    assert state_routing.state_width_disagrees(declared=6, stats=6) is False
    assert state_routing.state_width_disagrees(declared=32, stats=6) is True
    # unknown either side cannot disagree
    assert state_routing.state_width_disagrees(declared=None, stats=18) is False
    assert state_routing.state_width_disagrees(declared=6, stats=None) is False
