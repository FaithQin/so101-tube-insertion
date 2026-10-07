"""The input-ablation gate: does a B policy's predicted chunk MOVE when its load/current is taken away?

Pre-bench, no hardware. If freezing the 12 load/current channels at their training mean leaves
the predicted action chunk unchanged on contact frames, the policy is ignoring the channels and
the load-sensing hypothesis is false by construction -- known before a single bench hour. The
control ablation freezes the 6 POSITION dims instead, which every policy certainly uses, so a
reader can see what "moves" looks like on the same scale.

State layout (so_follower patch, tools/v4_audit.py): [0:6] pos, [6:12] load, [12:18] current.
"""
import sys

import numpy as np
import pytest

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))
import v4_input_ablation as ab  # noqa: E402


def _state(n=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 18)).astype(np.float32)


def test_mean_ablation_replaces_only_the_load_and_current_dims():
    s = _state(); mean = np.arange(18, dtype=np.float32)
    out = ab.ablate(s, mean, mode="mean")
    assert np.array_equal(out[:, :6], s[:, :6])                      # positions untouched
    assert np.allclose(out[:, 6:18], np.broadcast_to(mean[6:18], (5, 12)))
    assert out is not s and np.array_equal(s, _state())               # input not mutated


def test_position_control_ablation_replaces_only_the_position_dims():
    s = _state(); mean = np.zeros(18, dtype=np.float32)
    out = ab.ablate(s, mean, mode="pos")
    assert np.allclose(out[:, :6], 0.0) and np.array_equal(out[:, 6:], s[:, 6:])   # mutation: swap the slices


def test_shuffle_ablation_draws_load_current_from_other_frames_of_the_same_episode():
    s = _state(n=6)
    out = ab.ablate(s, np.zeros(18), mode="shuffle", rng=np.random.default_rng(1))
    assert np.array_equal(out[:, :6], s[:, :6])
    rows = {tuple(r) for r in s[:, 6:18].round(6)}
    assert all(tuple(r) in rows for r in out[:, 6:18].round(6))       # every row is a real row
    assert not np.array_equal(out[:, 6:18], s[:, 6:18])               # and the order changed
    with pytest.raises(ValueError, match="mode"):
        ab.ablate(s, np.zeros(18), mode="bogus")


def test_contact_mask_uses_the_distal_load_channels_only():
    s = np.zeros((4, 18), dtype=np.float32); mean = np.zeros(18); std = np.ones(18)
    s[1, 8] = 3.0     # elbow load spike -> contact
    s[2, 6] = 9.0     # shoulder_pan load -- NOT a distal channel
    s[3, 14] = 9.0    # elbow CURRENT, not load
    z = ab.load_z(s, mean, std)
    assert list(ab.contact_mask(z, thr=2.0)) == [False, True, False, False]   # mutation: use all 6 loads


def test_summary_separates_contact_from_free_frames_and_reports_degrees():
    deltas = np.array([0.1, 5.0, 0.2, 4.0])          # mean |dAction| per frame, degrees
    mask = np.array([False, True, False, True])
    types = ["CLEAN", "RESIST", "CLEAN", "RESIST"]
    r = ab.summarize(deltas, mask, types)
    assert r["contact_mean_deg"] == pytest.approx(4.5) and r["free_mean_deg"] == pytest.approx(0.15)
    assert r["n_contact"] == 2 and r["n_free"] == 2
    assert r["per_type"]["RESIST"]["mean_deg"] == pytest.approx(4.5)   # mutation: pool types


def test_verdict_is_ignored_only_when_the_load_ablation_is_below_the_control():
    # the policy provably uses positions (control moves 3 deg) but load ablation moves 0.02 deg
    v = ab.verdict(load_contact_deg=0.02, load_free_deg=0.01, pos_control_deg=3.0, floor_deg=0.05)
    assert v["uses_load_channels"] is False and "IGNORES" in v["message"]
    v = ab.verdict(load_contact_deg=0.8, load_free_deg=0.1, pos_control_deg=3.0, floor_deg=0.05)
    assert v["uses_load_channels"] is True and v["contact_to_free_ratio"] == pytest.approx(8.0)
    # a broken control (positions also do nothing) must not be read as "ignores load"
    v = ab.verdict(load_contact_deg=0.02, load_free_deg=0.01, pos_control_deg=0.03, floor_deg=0.05)
    assert v["uses_load_channels"] is None and "control" in v["message"].lower()


def test_verdict_ratio_clause_separates_a_small_but_real_effect_from_noise():
    # below the absolute floor but NOT below a tenth of the control: a small real effect, not "ignores"
    v = ab.verdict(load_contact_deg=0.04, load_free_deg=0.01, pos_control_deg=0.30, floor_deg=0.05)
    assert v["uses_load_channels"] is True                            # mutation: drop the 0.1*control clause
    v = ab.verdict(load_contact_deg=0.02, load_free_deg=0.01, pos_control_deg=0.30, floor_deg=0.05)
    assert v["uses_load_channels"] is False


def test_state_with_a_time_axis_is_flattened_and_restored():
    """SmolVLA/pi0.5 batches carry observation.state as (B, n_obs_steps, 18); ACT as (B, 18).
    Sep 6: the first SmolVLA run died with IndexError on axis 1 of size 1."""
    raw3 = np.arange(18, dtype=np.float32).reshape(1, 1, 18)
    flat, shape = ab.flatten_state(raw3)
    assert flat.shape == (1, 18) and shape == (1, 1, 18)
    assert ab.restore_state(flat, shape).shape == (1, 1, 18)
    raw2 = np.arange(36, dtype=np.float32).reshape(2, 18)
    flat2, shape2 = ab.flatten_state(raw2)
    assert flat2.shape == (2, 18) and ab.restore_state(flat2, shape2).shape == (2, 18)
    with pytest.raises(ValueError, match="18"):
        ab.flatten_state(np.zeros((1, 6), dtype=np.float32))
