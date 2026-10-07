"""Regression: the live registration overlay reuses the tested blend and
never opens a camera in tests.

Sep 2 2026: posing the arm against a static overlay PNG was guess-and-check
— Faith: "I wish there was a way that I could live see what it looks like."
This tool shows the recorded ghost blended over the live feed continuously.
"""

import inspect
import sys

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))

import live_overlay as lo  # noqa: E402


def test_uses_the_tested_blend_not_a_second_copy():
    src = inspect.getsource(lo)
    assert "from pose_register import blend" in src
    assert "def blend(" not in src, "second blend implementation would drift from the tested one"


def test_front_camera_index_and_mode_match_the_trial_runner():
    """Same camera, same requested mode as run_scored_trial.sh, so the
    overlay shows what the policy sees."""
    assert lo.FRONT_INDEX == 1
    assert lo.WIDTH == 1280 and lo.HEIGHT == 720


def test_fit_resizes_live_frame_to_target_shape():
    import numpy as np
    live = np.zeros((480, 640, 3), dtype=np.uint8)
    out = lo.fit_to(live, (720, 1280))
    assert out.shape == (720, 1280, 3)
