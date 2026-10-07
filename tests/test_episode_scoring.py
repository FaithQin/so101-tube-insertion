"""Regression: trial outcomes are scored from the episode's OWN recording, and
the classifier must not confuse a held cap with a seated one.

Born from the Aug 29 2026 session, where the live seat check produced two wrong
verdicts in a row:

1. It fires AFTER the episode's reset phase, so it can photograph a scene the
   operator has already reset. Success is defined "at rollout end", and the
   only synchronized evidence of that moment is the recorded video.
2. It asks only "is the blue cap inside the rack region", so a cap held in the
   gripper directly above the funnel scored as SEAT. On the discarded trial 3
   those false seats lasted 0.1 s and 0.5 s while the arm was carrying the tube.

It also pins the start-state guard: that same episode BEGAN with the tube
already seated and was reset ~7 s into recording, with a hand in frame, so the
policy observed 7 s of a scene that exists nowhere in training. Nothing caught
it; the operator did.
"""

import re
import sys

import numpy as np

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))

import score_episode  # noqa: E402

H, W = 720, 1280
FPS = 20.0


def _frame(cx=None, cy=None, size=24):
    """A black frame with one saturated-blue blob (the tube cap) at (cx, cy)."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    if cx is not None:
        h = size // 2
        img[cy - h : cy + h, cx - h : cx + h] = (255, 0, 0)  # BGR pure blue
    return img


SEATED = (533, 465)  # measured seat centroid, Aug 29
START = (360, 356)  # reference start position
HELD_HIGH = (533, 300)  # cap in the gripper, above the funnel and above the mask cut


def test_seat_region_matches_the_live_checker():
    """The two scorers must agree, or verdicts stop being comparable across the
    study. seat_check.py is the one validated on all 46 v3 episodes."""
    src = (TOOLS / "seat_check.py").read_text()
    nums = [int(n) for n in re.findall(r"(\d+)\s*<\s*[xy]\s*<\s*(\d+)", src)[0]] if False else None
    assert "470" in src and "640" in src and "400" in src and "540" in src, (
        "seat_check.py region constants changed — update score_episode.py to match"
    )
    assert score_episode.SEAT_X == (470, 640)
    assert score_episode.SEAT_Y == (400, 540)


def test_classify_seated_cap():
    c, seat = score_episode.classify_frame(_frame(*SEATED))
    assert seat and c is not None


def test_classify_cap_at_start_is_not_a_seat():
    c, seat = score_episode.classify_frame(_frame(*START))
    assert c is not None and not seat


def test_classify_empty_frame_is_unknown():
    c, seat = score_episode.classify_frame(_frame())
    assert c is None and not seat


def test_top_of_frame_is_masked():
    """The mask zeroes the top 250 rows so the arm's own hardware cannot be
    mistaken for the cap."""
    c, _ = score_episode.classify_frame(_frame(533, 100))
    assert c is None


def test_final_verdict_comes_from_the_last_frame_not_the_best_one():
    """A tube that seats and is then knocked loose is a MISS: success is defined
    at rollout end. This is the rule early-stopping would quietly break."""
    frames = [_frame(*SEATED)] * 100 + [_frame(*START)] * 20
    r = score_episode.score_frames(frames, fps=FPS)
    assert r["final"] == "MISS"
    assert r["seat_runs"], "the transient seat should still be reported"


def test_held_cap_over_the_funnel_is_flagged_as_transient():
    """0.1-0.5 s of 'seat' while the gripper carries the tube across the funnel
    must not read as a seat that held."""
    frames = [_frame(*START)] * 40 + [_frame(*SEATED)] * 2 + [_frame(*HELD_HIGH)] * 40
    r = score_episode.score_frames(frames, fps=FPS)
    assert r["final"] == "MISS"
    assert all(d < 0.5 for _, _, d in r["seat_runs"]), "transient seats must be short"
    assert not r["held_to_end"]


def test_sustained_seat_to_the_end_is_a_seat():
    frames = [_frame(*START)] * 150 + [_frame(*SEATED)] * 900
    r = score_episode.score_frames(frames, fps=FPS)
    assert r["final"] == "SEAT"
    assert r["held_to_end"]
    assert r["seat_runs"][0][0] == 7.5


def test_start_state_guard_flags_an_episode_that_began_seated():
    """The discarded trial 3: recording began with the tube in the rack."""
    frames = [_frame(*SEATED)] * 126 + [_frame(*START)] * 900
    r = score_episode.score_frames(frames, fps=FPS)
    assert r["infra_start_seated"], "an episode that begins seated must be INFRA, not scored"


def test_clean_start_is_not_flagged():
    frames = [_frame(*START)] * 150 + [_frame(*SEATED)] * 900
    r = score_episode.score_frames(frames, fps=FPS)
    assert not r["infra_start_seated"]


def test_verdict_is_none_when_infra():
    """An invalid episode must not also carry a scoreable verdict — that is how
    a bad trial sneaks into a block."""
    frames = [_frame(*SEATED)] * 126 + [_frame(*START)] * 900
    r = score_episode.score_frames(frames, fps=FPS)
    assert r["scoreable"] is False
