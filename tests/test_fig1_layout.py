"""fig1 (the public README's headline figure) must be readable and must not
claim a video check that never happened.

Oct 7 2026: Faith read the public repo and found the fig1 title and subtitle
too close to read (title baseline 18 pt above the plot, subtitle box ending
about 3 pt into the title's box). The subtitle also said "live scoring, every
trial on video", which a reader takes as "video-scored", but the video
re-scoring pass never ran.

The suite's env (lerobot) has no matplotlib, so the first two tests read the
layout constants straight from the source with `ast` and run everywhere. The
third renders the figure and measures it, and runs wherever matplotlib is
installed (the figure env).
"""

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAKE_FIGURES = ROOT / "tools" / "make_figures.py"

FIG1_SUBTITLE = ("v2 paired eval, Aug 25 2026 · share of trials reaching at least each stage"
                 " · live scores (the video re-scoring pass never ran)")


def _constants():
    """Module-level literal assignments in make_figures.py, without importing it."""
    out = {}
    for node in ast.parse(MAKE_FIGURES.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def test_fig1_subtitle_says_live_scores_and_that_the_video_pass_never_ran():
    c = _constants()
    assert c.get("FIG1_SUBTITLE") == FIG1_SUBTITLE
    assert "every trial on video" not in MAKE_FIGURES.read_text()


def test_fig1_title_baseline_clears_the_subtitle_baseline_by_one_and_a_half_title_heights():
    c = _constants()
    gap = c["FIG1_TITLE_PAD_PT"] - c["FIG1_SUBTITLE_PAD_PT"]
    # Before Oct 7 the baseline gap was about 8 pt for a 12.5 pt title.
    assert gap >= 1.5 * c["TITLE_FONT_PT"]
    # The subtitle's descenders must also clear the top of the plot.
    assert c["FIG1_SUBTITLE_PAD_PT"] >= 0.5 * c["SUBTITLE_FONT_PT"]


def _text_named(ax, prefix):
    hits = [t for t in ax.get_children() if hasattr(t, "get_text") and t.get_text().startswith(prefix)]
    assert len(hits) == 1, prefix
    return hits[0]


def test_fig1_rendered_title_and_subtitle_are_separated_and_overlap_nothing():
    pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")
    sys.path.insert(0, str(ROOT / "tools"))
    import make_figures as mf

    fig, ax, _ = mf.draw_fig1(mf.fd.v2_trials())
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pt = 72 / fig.dpi
    title = _text_named(ax, "Where each arm")
    sub = _text_named(ax, "v2 paired eval")
    assert sub.get_text() == FIG1_SUBTITLE

    def baseline(t):
        return t.get_transform().transform(t.get_position())[1]

    gap_pt = (baseline(title) - baseline(sub)) * pt
    assert gap_pt >= 1.5 * title.get_fontsize()

    tb, sb = title.get_window_extent(r), sub.get_window_extent(r)
    assert (tb.y0 - sb.y1) * pt >= 4            # visible white space between the two lines
    assert sb.y0 >= ax.bbox.y1                   # subtitle sits wholly above the plot
    assert fig.bbox.x0 <= sb.x0 and sb.x1 <= fig.bbox.x1 and tb.y1 <= fig.bbox.y1
    others = [t.get_window_extent(r) for t in ax.texts if t is not sub]
    others.append(ax.get_legend().get_window_extent(r))
    for bb in others:
        assert not bb.overlaps(tb) and not bb.overlaps(sb)
    import matplotlib.pyplot as plt
    plt.close(fig)
