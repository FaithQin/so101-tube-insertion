"""Hold-out loss on CONTACT frames only: is the offline A/B null an averaging artifact?

The whole-episode hold-out loss is indistinguishable between A and B (Sep 5). A load-sensing
effect would live on the few frames where a distal servo is loaded; averaged over ~700 frames
of free-space transit it is invisible. This scorer labels every held-out frame from the 18-dim
twin's own load channels (the same frames the -noload twin carries, by (episode, frame_index)),
scores A and B on the identical frame set with paired noise, and reports contact vs free.
"""
import sys

import numpy as np
import pytest

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))
import v4_contact_loss as cl  # noqa: E402


def test_contact_lookup_keys_by_episode_and_frame_using_distal_load_z():
    mean = np.zeros(18); std = np.ones(18)
    states = np.zeros((4, 18), dtype=np.float32)
    states[1, 8] = 3.0    # elbow load spike (contact)
    states[3, 6] = 9.0    # shoulder_pan load: not distal -> free
    eps = [11, 11, 19, 19]; frames = [0, 1, 0, 1]
    look = cl.contact_lookup(eps, frames, states, mean, std, thr=2.0)
    assert look[(11, 0)] is False and look[(11, 1)] is True
    assert look[(19, 0)] is False and look[(19, 1)] is False           # mutation: any load channel


def test_split_means_are_frame_weighted_and_keyed_by_contact():
    rows = [{"episode": 11, "frame": 0, "loss": 1.0, "contact": False},
            {"episode": 11, "frame": 1, "loss": 3.0, "contact": True},
            {"episode": 19, "frame": 0, "loss": 5.0, "contact": True}]
    s = cl.split_means(rows)
    assert s["contact"]["mean"] == pytest.approx(4.0) and s["contact"]["n"] == 2
    assert s["free"]["mean"] == pytest.approx(1.0) and s["free"]["n"] == 1


def test_pair_table_matches_a_and_b_on_the_same_frames_only():
    a = [{"episode": 11, "frame": 0, "loss": 1.0, "contact": False},
         {"episode": 11, "frame": 1, "loss": 2.0, "contact": True},
         {"episode": 11, "frame": 2, "loss": 9.0, "contact": True}]     # B has no frame 2
    b = [{"episode": 11, "frame": 0, "loss": 1.5, "contact": False},
         {"episode": 11, "frame": 1, "loss": 1.0, "contact": True}]
    t = cl.pair_table(a, b)
    assert t["n_paired"] == 2                                          # mutation: use all of A's rows
    assert t["contact"]["A"] == pytest.approx(2.0) and t["contact"]["B"] == pytest.approx(1.0)
    assert t["contact"]["B_minus_A"] == pytest.approx(-1.0)
    assert t["free"]["B_minus_A"] == pytest.approx(0.5)
    assert t["contact"]["frames_B_better"] == 1 and t["free"]["frames_B_better"] == 0
