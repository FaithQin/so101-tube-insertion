"""The rehearsal's own degeneracy gate — tested by EXECUTING it.

Sep 5 2026 pre-push review, HIGH. `smolvla_dryrun.py` read the action chunk
back with `getattr(policy, "_action_queue", None)`. That attribute belongs to
ACT (`policies/act/modeling_act.py:98`) and pi0/pi0.5; **SmolVLAPolicy has no
`_action_queue` at all** — it keeps its chunk in `self._queues[ACTION]`
(`policies/smolvla/modeling_smolvla.py:172`, extended :267, popped :269).

So `q` was always None, the chunk collapsed to the single first action, and
`... if len(chunk_np) > 1 else (True, "")` hard-coded `ok = True`.
`check_chunk_sane` — the only guard against a wrong-width, non-finite, or
time-constant chunk — was dead code, `all_sane` could never be False, and the
`PLUMBING SUSPECT` fallback could never fire. **Both dry-run verdicts reported
on Sep 4 (A 7.2x, B 6.4x) had no sanity gate behind them.**

The previous guard against exactly this was a source-text grep. These tests run
the code instead: a stub policy shaped like each architecture's real queue, and
a stub post-processor, so no checkpoint and no hardware are needed.
"""

import sys
from collections import deque

import numpy as np
import pytest

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))
import smolvla_dryrun as dry  # noqa: E402

ACTION = "action"
NJ = 6


class _Stub:
    """A policy exposing whichever queue attribute we are testing."""

    def __init__(self, *, act_queue=None, smolvla_queue=None):
        if act_queue is not None:
            self._action_queue = deque(act_queue)
        if smolvla_queue is not None:
            self._queues = {ACTION: deque(smolvla_queue)}


def _t(vals):
    import torch

    return torch.tensor(vals, dtype=torch.float32).unsqueeze(0)  # (1, NJ)


def _identity_post(x):
    return x


def _ramp(n=8, jitter=1.0):
    """A healthy chunk: moves over time, finite, right width."""
    return [_t([j * jitter + i for j in range(NJ)]) for i in range(n)]


# --- the queue lookup ------------------------------------------------------

def test_action_queue_finds_the_SMOLVLA_queue():
    """SmolVLAPolicy has no _action_queue; this is the shape that was missed."""
    p = _Stub(smolvla_queue=_ramp(3))
    q = dry.action_queue(p)
    assert q is not None, (
        "action_queue() cannot see _queues[ACTION] — every SmolVLA dry-run's "
        "degeneracy gate is dead code again (the Sep 4 bug)"
    )
    assert len(q) == 3


def test_action_queue_still_finds_the_ACT_queue():
    """ACT/pi0/pi0.5 keep theirs in _action_queue; do not regress them."""
    p = _Stub(act_queue=_ramp(4))
    q = dry.action_queue(p)
    assert q is not None and len(q) == 4


def test_action_queue_returns_None_when_the_policy_exposes_neither():
    assert dry.action_queue(_Stub()) is None


# --- the gate itself -------------------------------------------------------

def test_a_healthy_chunk_passes():
    p = _Stub(smolvla_queue=_ramp(7))
    chunk = dry.assemble_chunk(p, _t([0.0] * NJ), _identity_post)
    assert chunk is not None and chunk.shape == (8, NJ)
    ok, why = dry.chunk_verdict(chunk, NJ)
    assert ok, why


def test_a_time_constant_chunk_is_DEGENERATE():
    """The failure the gate exists for: the policy emits one pose forever."""
    const = [_t([1.0] * NJ) for _ in range(7)]
    p = _Stub(smolvla_queue=const)
    ok, why = dry.chunk_verdict(dry.assemble_chunk(p, _t([1.0] * NJ), _identity_post), NJ)
    assert not ok, "a chunk that never changes over time was reported sane"
    assert "constant" in why


def test_a_nonfinite_chunk_is_DEGENERATE():
    bad = _ramp(6)
    bad[2] = _t([float("nan")] * NJ)
    p = _Stub(smolvla_queue=bad)
    ok, why = dry.chunk_verdict(dry.assemble_chunk(p, _t([0.0] * NJ), _identity_post), NJ)
    assert not ok and "finite" in why


def test_an_UNREADABLE_queue_is_a_hard_failure_not_a_pass():
    """A sanity check that cannot find its input must never report sane.

    This is the exact regression: `_action_queue` absent -> chunk collapses to
    one action -> `if len(chunk_np) > 1 else (True, "")` -> PLUMBING OK.
    """
    p = _Stub()  # neither attribute — what SmolVLAPolicy looked like to the old code
    chunk = dry.assemble_chunk(p, _t([0.0] * NJ), _identity_post)
    assert chunk is None
    ok, why = dry.chunk_verdict(chunk, NJ)
    assert not ok, "an unreadable action queue was reported SANE — the gate is dead again"
    assert "queue" in why.lower()


def test_a_single_action_chunk_fails_loudly():
    """No `len(chunk_np) > 1` escape hatch: one action is not a chunk."""
    p = _Stub(smolvla_queue=[])  # select_action popped the only action
    ok, why = dry.chunk_verdict(dry.assemble_chunk(p, _t(list(range(NJ))), _identity_post), NJ)
    assert not ok, "a 1-action chunk slipped through — the escape hatch is back"


def test_the_first_action_is_not_post_processed_TWICE():
    """chunk[0] must be post(raw_first), not post(post(raw_first)).

    The old assembly stacked the ALREADY-posted first action and ran post over
    the whole thing, so row 0 — the row `check_chunk_sane` compares every other
    row against — was unnormalized twice.
    """
    doubled = []

    def post(x):
        doubled.append(x)
        return x * 2.0

    raw_first = _t([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    p = _Stub(smolvla_queue=_ramp(3))
    chunk = dry.assemble_chunk(p, raw_first, post)
    np.testing.assert_allclose(
        chunk[0], np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0]),
        err_msg="chunk[0] is not post(raw_first) — the first action was post-processed twice",
    )


def test_a_wrong_width_chunk_is_DEGENERATE():
    p = _Stub(smolvla_queue=[_t([0.0] * NJ)[:, :3] + i for i in range(4)])
    ok, why = dry.chunk_verdict(dry.assemble_chunk(p, _t([0.0] * NJ)[:, :3], _identity_post), NJ)
    assert not ok and "expected" in why
