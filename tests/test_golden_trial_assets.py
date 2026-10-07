"""Write-up assets for the two Sep 6 policy trials, and the ground truth they are cut from.

Sep 12 2026. The two rollout datasets behind the first blinded bench trial (P01 trial 1,
token h83j2kp6) and the unblinded smoke trial (T0-SMOKE) existed in exactly one place:
~/.cache/huggingface/lerobot/. Rollouts are never pushed to the Hub, and /private/tmp had
already been wiped once (Sep 10) and taken a night of drafts with it. They are now copied
to analysis/bench_sep6/rollouts/ with a SHA256SUMS manifest, and
`tools/golden_trial_assets.py` cuts the write-up assets from those copies.

What these tests hold:
  * BLINDING. Every string that lands in a sealed trial's image, caption or filename is
    checked for arm/policy words. The sealed caption is fixed verbatim. The tool never
    names trials.csv or the sealed directory (checked with ast, not grep).
  * The first close is v4_audit.grasp_index, and the RE-grasp is found even when the
    first close never gets below v4_audit.JAW_SHUT (P01 trial 1's bottomed at 14.3, so
    jaw_reopen_cycles reports 0 for a trial that visibly reopened and re-grasped).
  * The side-by-side video keeps every frame, in order, at 20 fps, <= 1280 px, H.264.
  * The contact sheet and joint plot put each frame / landmark where its time says.
  * The preserved copies still match their manifest, and the four frame counts agree.
"""

import ast
import hashlib
import re
import sys
from fractions import Fraction

import numpy as np
import pytest

from conftest import PROJECT, TOOLS

sys.path.insert(0, str(TOOLS))
import golden_trial_assets as gta  # noqa: E402
import v4_audit  # noqa: E402

FPS = 20
ARM_WORDS = re.compile(r"\bACT\b|\b[AB]-v\d|(?i:noload|\bload\b|\barm [AB]\b|polic|smolvla|pi0)")


# --- blinding ---------------------------------------------------------------

def _fake_landmarks():
    return {"first_close": 74, "regrasp": 159, "rack_entry": 208}


def test_sealed_trial_caption_is_exactly_the_blinded_form():
    t = gta.trial("h83j2kp6")
    assert t.caption == "P01 trial 1 (token h83j2kp6, arm sealed)"


def test_only_the_smoke_trial_is_unblinded():
    assert gta.UNBLINDED == {"T0-SMOKE"}
    assert {t.token for t in gta.TRIALS} == {"h83j2kp6", "T0-SMOKE"}


def test_no_text_or_filename_for_a_sealed_trial_names_an_arm_or_policy():
    sealed = [t for t in gta.TRIALS if t.token not in gta.UNBLINDED]
    assert sealed, "the blinded trial must be in the config"
    for t in sealed:
        texts = gta.asset_texts(t, _fake_landmarks(), seat_s=11.7, x_end_s=15.7,
                                settle=0.35, duration_s=43.5)
        strings = [t.label, t.caption, *gta.output_names(t).values()]
        stack = list(texts.values())
        while stack:
            v = stack.pop()
            if isinstance(v, dict):
                stack.extend(v.values())
            elif isinstance(v, (list, tuple)):
                stack.extend(v)
            else:
                strings.append(v)
        assert len(strings) > 8
        for s in strings:
            assert not ARM_WORDS.search(s), f"sealed trial text names an arm/policy: {s!r}"
            gta.assert_blind_safe(t, s)  # the tool's own runtime guard must accept it


def test_runtime_guard_refuses_arm_words_on_a_sealed_trial_only():
    sealed = gta.trial("h83j2kp6")
    with pytest.raises(ValueError):
        gta.assert_blind_safe(sealed, "P01 trial 1 (policy X)")
    gta.assert_blind_safe(gta.trial("T0-SMOKE"), "T0-SMOKE smoke trial (ACT-A-v4)")


def test_the_tool_never_names_trials_csv_or_the_sealed_directory():
    tree = ast.parse((TOOLS / "golden_trial_assets.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            v = node.value
            assert "trials.csv" not in v, v
            assert v != "sealed" and "/sealed" not in v and "sealed/" not in v, v


# --- score parsing ------------------------------------------------------------

SEAT_SCORE = ("  frames=871 dur=43.5s\n  FINAL: cap=(523, 465) -> SEAT\n"
              "  seated  11.7s ->  43.5s (31.9s)  [holds to end]\nVERDICT SEAT\n")
LOST_SCORE = ("  frames=871 dur=43.5s\n  FINAL: cap=(284, 658) -> MISS\n"
              "  seated   6.5s ->   6.6s (0.1s)  [LOST]\n"
              "  seated  39.5s ->  39.6s (0.1s)  [LOST]\nVERDICT MISS\n")


def test_parse_score_takes_the_seat_that_holds_to_end():
    s = gta.parse_score(SEAT_SCORE)
    assert (s.frames, s.duration_s, s.seat_s, s.verdict) == (871, 43.5, 11.7, "SEAT")


def test_parse_score_ignores_transient_lost_seats():
    s = gta.parse_score(LOST_SCORE)
    assert s.seat_s is None and s.verdict == "MISS" and s.frames == 871


# --- timing -----------------------------------------------------------------

def test_contact_sheet_times_span_zero_to_seat_plus_two():
    ts = gta.contact_sheet_times(seat_s=11.7, last_t_s=43.5)
    assert len(ts) == 8
    assert ts[0] == 0.0 and ts[-1] == pytest.approx(13.7)
    assert np.allclose(np.diff(ts), 13.7 / 7)


def test_contact_sheet_times_clamp_to_the_recording_and_need_a_seat():
    assert gta.contact_sheet_times(seat_s=42.5, last_t_s=43.5)[-1] == pytest.approx(43.5)
    with pytest.raises(ValueError):
        gta.contact_sheet_times(seat_s=None, last_t_s=43.5)


def test_frame_at_rounds_to_the_nearest_frame_and_clamps():
    assert gta.frame_at(13.7, FPS, 871) == 274
    assert gta.frame_at(13.7 / 7, FPS, 871) == 39
    assert gta.frame_at(99.0, FPS, 871) == 870
    assert gta.frame_at(-1.0, FPS, 871) == 0
    assert gta.frame_at(0.08, FPS, 871) == 2   # 1.6 frames: nearest, not truncated


def test_side_by_side_geometry_fits_1280_even_and_undistorted():
    h, fw, ww = gta.side_by_side_geometry((1280, 720), (640, 480), max_width=1280)
    assert fw + ww <= 1280 and fw + ww > 1280 - 12
    assert h % 2 == 0 and fw % 2 == 0 and ww % 2 == 0
    assert abs(fw / h - 1280 / 720) / (1280 / 720) < 0.01
    assert abs(ww / h - 640 / 480) / (640 / 480) < 0.01
    h2, fw2, ww2 = gta.side_by_side_geometry((1280, 720), (640, 480), max_width=640)
    assert fw2 + ww2 <= 640


def test_encoder_choice_prefers_quicktime_playable_h264():
    assert gta.choose_encoder({"libx264", "h264_videotoolbox", "mpeg4"}) == "libx264"
    assert gta.choose_encoder({"h264_videotoolbox", "mpeg4"}) == "h264_videotoolbox"
    assert gta.choose_encoder({"mpeg4"}) == "mpeg4"
    with pytest.raises(RuntimeError):
        gta.choose_encoder(set())


# --- landmarks ----------------------------------------------------------------

def _recovery_trace():
    """Approach, close at the cap end that only reaches 14.3, reopen, re-close, carry, rack."""
    seg = [  # (frames, pan, wrist_flex, grip)
        (20, 2.0, 73.0, 1.1),     # parked
        (20, 68.0, 74.0, 32.0),   # jaws open at the pick site
        (10, 68.0, 72.0, 15.0),   # first close (<=16)...
        (10, 68.0, 72.0, 14.3),   # ...bottoms above JAW_SHUT=10
        (20, 65.0, 88.0, 31.4),   # lift away empty, jaws reopen
        (20, 68.0, 81.0, 12.6),   # re-grasp
        (10, 30.0, 82.0, 12.1),   # carry
        (20, 18.0, 5.0, 11.3),    # rack phase
        (10, 18.0, -4.0, 34.0),   # release
    ]
    pan, wf, grip = [], [], []
    for n, p, w, g in seg:
        pan += [p] * n
        wf += [w] * n
        grip += [g] * n
    st = np.zeros((len(pan), 6))
    st[:, 0], st[:, 3], st[:, 5] = pan, wf, grip
    return st


def test_first_close_is_grasp_index_and_the_regrasp_is_found_without_jaw_shut():
    st = _recovery_trace()
    pan, wf, grip = st[:, 0], st[:, 3], st[:, 5]
    # the reason this helper exists: the audit's reopen counter never sees this recovery
    assert v4_audit.jaw_reopen_cycles(pan, grip)[0] == 0
    lm = gta.landmarks(st)
    assert lm["first_close"] == v4_audit.grasp_index(pan, grip) == 40
    assert lm["regrasp"] == 80
    assert lm["rack_entry"] == v4_audit.rack_slice(pan, wf)[0] == 110
    assert gta.pick_closes(pan, grip) == [40, 80]


def test_a_single_clean_grasp_has_no_regrasp():
    st = _recovery_trace()
    st[60:80, 5] = 14.0  # never reopens
    lm = gta.landmarks(st)
    assert lm["first_close"] == 40 and lm["regrasp"] is None and lm["rack_entry"] == 110


def test_reconcile_counts_names_the_odd_source():
    assert gta.reconcile_counts({"front": 871, "wrist": 871, "parquet": 871, "csv": 871}) == []
    probs = gta.reconcile_counts({"front": 871, "wrist": 870, "parquet": 871, "csv": 871})
    assert len(probs) == 1 and "wrist" in probs[0] and "870" in probs[0]


def test_place_labels_never_overlaps_within_a_row():
    items = [(100, 90), (150, 90), (160, 60), (400, 50), (420, 80), (193, 20)]  # 193: 3 px past row 0
    rows = gta.place_labels(items, gap=6)
    assert len(rows) == len(items)
    for r in set(rows):
        spans = sorted((x, x + w) for (x, w), rr in zip(items, rows) if rr == r)
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            assert b0 >= a1 + 6
    assert rows[0] == 0 and rows[3] == 0


# --- manifest -----------------------------------------------------------------

def _write_manifest(root, files):
    lines = []
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        lines.append(f"{hashlib.sha256(data).hexdigest()}  {rel}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def test_verify_manifest_catches_tamper_missing_and_unlisted(tmp_path):
    _write_manifest(tmp_path, {"a/x.bin": b"abc", "a/b/y.json": b"{}"})
    assert gta.verify_manifest(tmp_path) == []
    (tmp_path / "a/x.bin").write_bytes(b"abd")
    (tmp_path / "a/b/y.json").unlink()
    (tmp_path / "a/z.txt").write_bytes(b"new")
    probs = gta.verify_manifest(tmp_path)
    joined = "\n".join(probs)
    assert len(probs) == 3
    assert "a/x.bin" in joined and "a/b/y.json" in joined and "a/z.txt" in joined


# --- rendering (synthetic video, no real data) ----------------------------------

def _gray_video(path, n, w, h, level):
    import av

    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=FPS)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        s.options = {"crf": "0", "preset": "ultrafast"}
        for k in range(n):
            arr = np.full((h, w, 3), level(k), dtype=np.uint8)
            fr = av.VideoFrame.from_ndarray(arr, format="rgb24").reformat(format="yuv420p")
            for p in s.encode(fr):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def _decode_all(path):
    import av

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(s)]
        return s, frames


def test_side_by_side_keeps_every_frame_in_order_at_20fps(tmp_path):
    n = 25
    _gray_video(tmp_path / "front.mp4", n, 64, 36, lambda k: 10 * k)
    _gray_video(tmp_path / "wrist.mp4", n, 48, 36, lambda k: 240 - 9 * k)
    out = tmp_path / "sbs.mp4"
    info = gta.render_side_by_side(tmp_path / "front.mp4", tmp_path / "wrist.mp4", out,
                                   fps=FPS, max_width=96, encoder="libx264")
    assert info["frames"] == n
    s, frames = _decode_all(out)
    assert len(frames) == n
    assert s.codec_context.name == "h264" and s.codec_context.pix_fmt == "yuv420p"
    assert Fraction(s.average_rate) == FPS
    assert s.width <= 96 and s.width % 2 == 0 and s.height % 2 == 0
    fw = info["front_width"]
    for k, fr in enumerate(frames):
        assert abs(fr[:, : fw - 2].mean() - 10 * k) < 8, f"front frame {k} out of order"
        assert abs(fr[:, fw + 2:].mean() - (240 - 9 * k)) < 8, f"wrist frame {k} out of order"


def test_contact_sheet_uses_the_frames_its_stamps_claim(tmp_path):
    n = 20
    _gray_video(tmp_path / "front.mp4", n, 64, 36, lambda k: 12 * k)
    times = gta.contact_sheet_times(seat_s=0.5, last_t_s=(n - 1) / FPS)  # clamps to 0.95 s
    out = tmp_path / "sheet.png"
    tiles = gta.render_contact_sheet(tmp_path / "front.mp4", times, FPS, n, out,
                                     title="T", subtitle="S", tile_width=64)
    # 0.95 s * k/7 * 20 fps = 2.714 k frames, rounded to nearest (computed by hand, not by the tool)
    assert [t["index"] for t in tiles] == [0, 3, 5, 8, 11, 14, 16, 19]
    assert [t["stamp"] for t in tiles] == [f"{t['index'] / FPS:.2f} s" for t in tiles]
    from PIL import Image

    im = Image.open(out)
    assert im.mode == "RGB"
    a = np.asarray(im).astype(float)
    for t in tiles:
        x0, y0, x1, y1 = t["box"]
        patch = a[y1 - 6: y1 - 2, x1 - 6: x1 - 2]  # bottom-right corner, clear of the stamp
        assert abs(patch.mean() - 12 * t["index"]) < 6  # one frame off = 12 levels


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_joint_plot_draws_landmarks_where_their_times_say(tmp_path, theme):
    n = 300
    st = np.full((n, 6), 10.0)
    st[0] = 0.0  # every panel's range is [0, 10]; the trace sits on the top edge
    lm = {"first_close": 40, "regrasp": 100, "rack_entry": 160}
    out = tmp_path / "joints.png"
    texts = {"title": "t", "subtitle": "s",
             "rules": {"first_close": "a", "regrasp": "b", "rack_entry": "c", "seat": "d"}}
    lay = gta.render_joint_plot(st, FPS, lm, seat_s=9.0, x_end_s=12.0, out=out,
                                texts=texts, theme=theme)
    from PIL import Image

    im = Image.open(out)
    assert im.mode == "RGB"
    a = np.asarray(im)
    surface = gta.hex_rgb(gta.THEMES[theme]["surface"])
    assert tuple(int(v) for v in a[1, 1]) == surface
    l, t_, r, _ = lay["plot_box"]
    (_, gap_top), (gap_bottom, _) = lay["panels"][0], lay["panels"][1]
    # Tick labels of neighbouring panels need at least one label-height of air between them.
    # The first Sep 12 render put one panel's "0" about half a label above the next panel's
    # "100", and it read as one axis instead of two.
    ticks = lay["tick_boxes"]
    assert len({j for j, _ in ticks}) == 6
    for j, (ax0, ay0, ax1, ay1) in ticks:
        for k, (bx0, by0, bx1, by1) in ticks:
            if j != k and ax0 < bx1 and bx0 < ax1:
                air = max(by0 - ay1, ay0 - by1)
                assert air >= max(ay1 - ay0, by1 - by0), (j, k, air)
    y = (gap_top + gap_bottom) // 2  # between panels 1 and 2: only the landmark rules cross here
    for name, t_s in [("first_close", 2.0), ("regrasp", 5.0), ("rack_entry", 8.0), ("seat", 9.0)]:
        x = lay["rules"][name]
        assert abs(x - (l + t_s / 12.0 * (r - l))) <= 1
        ink = gta.hex_rgb(gta.THEMES[theme]["ink" if name == "seat" else "ink2"])
        assert tuple(int(v) for v in a[y, x]) == ink, f"no {name} rule drawn at x={x}"
        assert tuple(int(v) for v in a[y, x - 4]) == surface, f"{name} rule is not a thin line"
    xs = [lay["rules"][k] for k in ("first_close", "regrasp", "rack_entry", "seat")]
    assert xs == sorted(xs)


# --- the real, preserved ground truth (skips where the copies are absent) --------

ROLLOUTS = PROJECT / "analysis" / "bench_sep6" / "rollouts"


@pytest.mark.skipif(not (ROLLOUTS / "SHA256SUMS").exists(), reason="preserved rollouts not on this machine")
def test_preserved_rollouts_still_match_their_manifest():
    assert gta.verify_manifest(ROLLOUTS) == []


@pytest.mark.skipif(not (ROLLOUTS / "SHA256SUMS").exists(), reason="preserved rollouts not on this machine")
@pytest.mark.parametrize("token,expected", [("h83j2kp6", 871), ("T0-SMOKE", 873)])
def test_video_parquet_and_telemetry_frame_counts_agree(token, expected):
    counts = gta.frame_counts(gta.trial(token))
    assert set(counts) == {"front video", "wrist video", "parquet rows", "telemetry csv rows"}
    assert gta.reconcile_counts(counts) == []
    assert set(counts.values()) == {expected}
