"""Aggregating contact-frame pairs over noise seeds, per episode — the unit that settled π0.5 (Sep 6 03:05).

One flow-matching draw put π0.5-B better on 63 % of contact frames, another on 48 %; averaging
three draws per frame and then per episode gave −0.0062 ± 0.0012 with B better in 8/8 episodes.
`aggregate_seeds` is that computation, so the morning's checkpoint-matched pair is one command.
"""
import sys

import numpy as np
import pytest

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))
import v4_contact_loss as cl  # noqa: E402


def _result(diffs_by_key, contact_keys):
    """A minimal result dict in the shape v4_contact_loss writes: A rows at 0, B rows at 0 + diff."""
    a = [{"episode": e, "frame": f, "loss": 1.0, "contact": (e, f) in contact_keys} for (e, f) in diffs_by_key]
    b = [{"episode": e, "frame": f, "loss": 1.0 + d, "contact": (e, f) in contact_keys} for (e, f), d in diffs_by_key.items()]
    return {"A": {"rows": a}, "B": {"rows": b}}


def test_aggregate_averages_seeds_per_frame_then_per_episode_on_contact_frames_only():
    keys = {(11, 0): None, (11, 1): None, (19, 0): None, (19, 1): None}
    contact = {(11, 0), (11, 1), (19, 0)}                      # (19, 1) is a free frame
    s0 = _result({(11, 0): -0.02, (11, 1): -0.02, (19, 0): +0.01, (19, 1): -0.50}, contact)
    s1 = _result({(11, 0): -0.01, (11, 1): -0.01, (19, 0): -0.03, (19, 1): -0.50}, contact)
    agg = cl.aggregate_seeds([s0, s1])
    # ep 11: frames average (-0.015, -0.015) -> -0.015 ; ep 19: frame (19,0) averages -0.01
    assert agg["per_episode"] == {11: pytest.approx(-0.015), 19: pytest.approx(-0.01)}
    assert agg["episodes_B_better"] == 2 and agg["n_episodes"] == 2 and agg["n_seeds"] == 2
    assert agg["mean"] == pytest.approx(-0.0125)
    assert agg["free_mean"] == pytest.approx(-0.50)             # mutation: pool free frames into contact
    assert agg["contact_frame_mean"] == pytest.approx(np.mean([-0.015, -0.015, -0.01]))


def test_aggregate_uses_only_frames_present_in_every_seed_and_both_arms():
    contact = {(11, 0), (11, 1)}
    s0 = _result({(11, 0): -0.02, (11, 1): -0.02}, contact)
    s1 = _result({(11, 0): -0.02}, contact)                     # seed 1 lacks frame (11, 1)
    agg = cl.aggregate_seeds([s0, s1])
    assert agg["per_episode"] == {11: pytest.approx(-0.02)} and agg["n_frames_contact"] == 1   # mutation: union
    with pytest.raises(ValueError, match="seed"):
        cl.aggregate_seeds([])
