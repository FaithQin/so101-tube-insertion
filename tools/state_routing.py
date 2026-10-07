"""Decide whether `.load`/`.current` are routed into `observation.state`.

This is the reference implementation of the rule embedded in the local
`rollout/context.py` patch. It lives here so the decision can be unit-tested
without a robot, a policy, or site-packages -- see tests/test_state_routing.py.
`context.py` carries the same logic inline (it cannot import from this repo),
and tests/test_sitepackages_patches.py checks the markers are still present.

Background
----------
The Aug 15 2026 patch widened the observation filter from `.pos`/`.vel` to also
include `.load`/`.current` whenever the loaded policy declared a wider
`observation.state` than the joint-position keys provide. That was the right
call for ACT-B, which trained on 18-dim state and crashed with a 6-vs-18
normalizer mismatch without it, while ACT-A silently worked.

Two problems with inferring it purely from the declared width, both found
Aug 31 2026:

**Padding is not dimensionality.** pi0.5 declares
`input_features["observation.state"].shape = [max_state_dim]` -- 32 -- no matter
what it trained on. `pi05-tube-A-v3` trained on `so101-tube-insert-v3-noload`,
whose state is 6-dim; `Pi05PrepareStateTokenizerProcessorStep` pads 6 -> 32, so
dims 6-31 are zero in every training frame. The naive rule sees 32 > 6, routes
load+current in, and hands the policy real values in coordinates it learned as
always-zero. Shapes still line up, so nothing crashes -- it is a silent
distribution shift that presents as "the policy just doesn't work."

**Recording needs the wide filter even under a narrow policy.** DAgger builds
`dataset_features` from this same filtered dict, so collecting corrections
under 6-dim ACT-A writes a 6-dim dataset with load and current dropped at the
source. That cannot be retrofitted; it permanently forecloses the B-side twin.
Hence an explicit override.
"""

from __future__ import annotations

ENV_VAR = "CAPSTONE_FORCE_WIDE_STATE"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def state_width_disagrees(declared: int | None, stats: int | None) -> bool:
    """True when a checkpoint's declared state width contradicts its own stats.

    Malformed metadata, not a judgement call: the normalizer buffers are what
    the policy is actually normalized against, so a mismatch means the declared
    width cannot be trusted. Callers should GATE on this (abort before the arm
    moves) rather than silently leaning on the fallback in
    `should_widen_state`. Unknown on either side cannot disagree.
    """
    if declared is None or stats is None:
        return False
    return declared != stats


def should_widen_state(
    policy_state_dim: int | None,
    n_stock: int,
    max_state_dim: int | None = None,
    force: bool | None = None,
    stats_state_dim: int | None = None,
) -> bool:
    """True when `.load`/`.current` should join `observation.state`.

    Args:
        policy_state_dim: width the loaded policy declares, or None if it
            declares no state feature.
        n_stock: how many scalar state features the stock `.pos`/`.vel` filter
            yields for this robot (6 on the SO-101).
        max_state_dim: the policy's padding width, when it pads (pi0/pi0.5).
            A declared width EQUAL to this is padding, not evidence of
            dimensionality.
        force: explicit override. None means infer. True/False decide outright,
            and are the only way to record wide under a narrow policy.
        stats_state_dim: width of the checkpoint's own normalizer statistics
            (the shape of `observation.state.mean` in its preprocessor
            safetensors), when known. When this disagrees with the declared
            width, it WINS -- see below.
    """
    if force is not None:
        return force
    # The declared width is not always trustworthy. `smolvla-tube-B-v3` declares
    # 6 while carrying 18-dim normalizer stats (measured Sep 3 2026), so the
    # declared width routes it narrow and a 6-dim state hits an 18-dim
    # normalizer: RuntimeError on the first tick. Its A-side twin declares the
    # same 6 and genuinely IS 6-dim, so nothing about the declaration, the
    # padding width, or the policy family can separate them -- only the stats.
    # And the stats are the ground truth here by construction: they are the
    # buffers the state is about to be normalized against.
    if stats_state_dim is not None and state_width_disagrees(policy_state_dim, stats_state_dim):
        return stats_state_dim > n_stock
    if policy_state_dim is None:
        return False
    # Padding is not dimensionality. Exact equality only: a policy that pads to
    # 32 but genuinely declares 18 really is 18-dim and must still widen.
    if max_state_dim is not None and policy_state_dim == max_state_dim:
        return False
    return policy_state_dim > n_stock


def force_from_env(env: dict) -> bool | None:
    """Parse the override out of the environment.

    Returns None when unset -- distinct from False, which forces the narrow
    filter. Collapsing the two would make an unset variable silently disable
    ACT-B's widening.

    Raises on an unparseable value rather than falling back to inference: a
    typo must not quietly become "record 6-dim" during a DAgger session the
    operator believes is recording 18.
    """
    raw = env.get(ENV_VAR)
    if raw is None or raw.strip() == "":
        return None
    v = raw.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(
        f"{ENV_VAR}={raw!r} is not a boolean. Use one of "
        f"{sorted(_TRUE)} or {sorted(_FALSE)}, or unset it to infer."
    )
