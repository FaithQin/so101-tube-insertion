"""The demo cut is assembled from archived rollouts only, with in/out times taken from data.

Born Sep 12 2026 (cut from archived rollouts, no bench); re-cut Sep 22 for
a stranger (a PhD student or founder opening a cold-email link) who has 90 seconds to learn
three things: what the task is, whether it works, and how it fails. The cut is a public-facing
exhibit, so the failure modes that matter are the ones that would make it say something the
record does not, or say it in a way the stranger cannot follow:

1. A clip mapped to the wrong rollout directory. The v2 sheet names run ids
   (`trial_B_v2_20260825_181012`); directories append a stamp. A loose prefix match could pick
   a neighbour, so resolution must be exact and unique, and the recomputed forensic grasp-close
   time must reproduce the sheet's own column for every v2 clip.
2. A blinded Sep 6 bench recording slipping into the cut. Sources are whitelisted to the
   unblinded families (v2 trials, the Aug 22 probe, the Aug 30 Block 2 probe).
3. In/out points chosen by eye. Every window is an event from the episode's own data plus a
   declared offset; the event detectors are pinned here on synthetic traces.
4. A rough cut that is not real time, too wide, or has captions that never reach the frame.
5. Evidence bookkeeping leaking on screen. Faith, Sep 22: a stranger does not need dates,
   dataset versions, trial numbers, arm labels, run ids or operator resets; those live in
   EDL.md. Captions are one line of at most eight plain words, the cards exist, the clip list
   is the approved structure, the end card's rate is the sheet's own fraction, and the word
   "clean" never describes an episode again (the old C01 used it for "uncontaminated data").
"""

import re
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import PROJECT, TOOLS

sys.path.insert(0, str(TOOLS))

import demo_cut as dc  # noqa: E402

ROLLOUTS = Path.home() / ".cache/huggingface/lerobot/faithqin"
needs_cache = pytest.mark.skipif(
    not (ROLLOUTS / "rollout_B2V-A-02_20260830_233508").is_dir(),
    reason="archived rollouts live only on the capstone Mac",
)
# Faith, Oct 7 2026 (afternoon): every on-screen word is set in Arial, the deck's font, so the
# video, clips and slides match. Arial ships with macOS.
VIDEO_FONT = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
needs_font = pytest.mark.skipif(not VIDEO_FONT.is_file(), reason="Arial ships with macOS")


# --- 3. event detectors -------------------------------------------------------------------

def test_commanded_closes_needs_an_open_before_each_close():
    # starts shut (not a close), opens, shuts (close 1), dithers in the dead band (not a close),
    # opens, shuts (close 2)
    grip = [0, 0, 30, 30, 5, 5, 12, 5, 30, 30, 4]
    assert dc.commanded_closes(grip) == [4, 10]


def test_commanded_closes_dead_band_is_hysteresis_not_a_threshold():
    # 30 -> 10 -> 30 -> 10: never below 8, so never closed
    assert dc.commanded_closes([30, 10, 30, 10, 30]) == []


def test_home_return_ignores_the_folded_start_pose():
    fps = 20
    lift = np.full(200, 20.0)
    lift[:30] = -100.0          # episode opens folded (before 2 s): not a return
    lift[150:] = -104.0         # the real return
    assert dc.home_return_index(lift, fps=fps) == 150


def test_home_return_none_when_never_commanded():
    assert dc.home_return_index(np.full(100, 10.0)) is None


def test_pick_exits_are_downward_crossings_of_the_pick_site_pan():
    pan = [5, 40, 70, 70, 50, 20, 70, 71, 30]
    assert dc.pick_exits(pan) == [4, 8]


def test_rack_entry_requires_both_pan_and_wrist_flex():
    pan = np.array([70, 30, 30, 30, 30])
    wflex = np.array([60, 60, 60, -15, -15])   # pan alone is transport, not rack
    assert dc.rack_entry_index(pan, wflex) == 3
    assert dc.rack_entry_index(pan, np.full(5, 60)) is None


def test_cap_displaced_ignores_noise_single_outliers_and_occlusion():
    caps = [(360, 355)] * 10 + [(362, 356)] * 5 + [(400, 300)] + [(361, 355)] * 5 \
        + [None] * 3 + [(380, 330)] * 8
    # onset is the first frame of the sustained move (index 24), not the lone outlier (15)
    assert dc.cap_displaced_index(caps, radius_px=8, sustain=5) == 24


def test_cap_displaced_none_when_the_tube_never_moves():
    assert dc.cap_displaced_index([(362, 356)] * 50, radius_px=8, sustain=5) is None


def test_cap_lost_needs_a_sustained_absence_after_start():
    caps = [None] * 30 + [(450, 360)] * 10 + [None] * 5 + [(440, 350)] * 5 + [None] * 40
    # the early 30-frame occlusion is before `start`; the 5-frame gap is too short
    assert dc.cap_lost_index(caps, start=35, min_missing=20) == 50


def test_held_seat_start_is_the_run_that_reaches_the_end():
    runs = [(1.0, 1.5, 0.55), (7.4, 43.45, 36.1)]
    assert dc.held_seat_start(runs, duration_s=43.5, fps=20) == pytest.approx(7.4)
    assert dc.held_seat_start([(13.8, 13.8, 0.05)], duration_s=43.7, fps=20) is None


def test_lift_reversals_count_turning_points_beyond_the_threshold():
    lift = [0, 10, 0, 10, 0, 3, 0]   # two 10-degree swings, then a 3-degree wobble
    assert len(dc.lift_reversals(lift, min_deg=6.0)) == 3


# --- window resolution ----------------------------------------------------------------------

def test_resolve_window_applies_offsets_and_clamps_to_the_episode():
    ev = {"grasp_close": 2.3, "home_return": 8.65}
    assert dc.resolve_window(ev, ("grasp_close", -2.0), ("home_return", 0.5), 43.5) == \
        pytest.approx((0.3, 9.15))
    assert dc.resolve_window(ev, ("grasp_close", -5.0), ("home_return", 99.0), 43.5) == \
        pytest.approx((0.0, 43.5))


def test_resolve_window_refuses_a_missing_event_or_an_empty_window():
    with pytest.raises(KeyError):
        dc.resolve_window({"grasp_close": 2.3}, ("rack_entry", 0.0), ("grasp_close", 1.0), 43.5)
    with pytest.raises(ValueError):
        dc.resolve_window({"a": 5.0, "b": 4.0}, ("a", 0.0), ("b", 0.0), 43.5)


def test_resolve_window_refuses_an_event_that_was_not_detected():
    with pytest.raises(ValueError):
        dc.resolve_window({"rack_entry": None, "start": 0.0}, ("start", 0.0),
                          ("rack_entry", 1.0), 43.5)


# --- 1. source mapping ------------------------------------------------------------------------

def test_resolve_rollout_is_exact_and_unique(tmp_path):
    (tmp_path / "rollout_trial_A_v2_20260825_164252_20260825_164259").mkdir()
    (tmp_path / "rollout_trial_A_v2_20260825_1642520_20260825_164300").mkdir()  # prefix neighbour
    got = dc.resolve_rollout("trial_A_v2_20260825_164252", root=tmp_path)
    assert got.name == "rollout_trial_A_v2_20260825_164252_20260825_164259"
    with pytest.raises(FileNotFoundError):
        dc.resolve_rollout("trial_A_v2_20260825_999999", root=tmp_path)
    (tmp_path / "rollout_trial_A_v2_20260825_164252_20260825_170000").mkdir()
    with pytest.raises(RuntimeError):
        dc.resolve_rollout("trial_A_v2_20260825_164252", root=tmp_path)


SHEET_HEADER = (
    "| # | Block | Policy | n | Start | Stage (live) | Stage (video) | Mechanism | Notes | Run id / ep | grasp-close t | home-return t |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)


def test_v2_sheet_parser_reads_run_id_stage_and_forensic_close(tmp_path):
    sheet = tmp_path / "sheet.md"
    sheet.write_text(
        SHEET_HEADER
        + "|  6 | 2 | ACT-B | 25 | primed | S | | — | seated | trial_B_v2_20260825_171233 / ep0 | 2.6 s (lift +40.1, 3.0° high) | 9.8 s |\n"
        + "| 29 | 6 | ACT-B | 25 | primed | INFRA — re-run | | INFRA | camera | trial_B_v2_20260825_201626 / ep0 | 2.7 s (lift +39.7, 2.6° high) | n/a (INFRA) |\n"
    )
    rows = dc.v2_sheet_rows(sheet)
    assert rows["6"]["run_id"] == "trial_B_v2_20260825_171233"
    assert rows["6"]["episode"] == 0
    assert rows["6"]["policy"] == "ACT-B"
    assert rows["6"]["stage"] == "S"
    assert rows["6"]["grasp_close_s"] == pytest.approx(2.6)
    assert rows["6"]["home_return_s"] == pytest.approx(9.8)
    assert rows["6"]["line"] == 3
    assert rows["29"]["stage"].startswith("INFRA")
    assert rows["29"]["home_return_s"] is None


def test_seated_fraction_counts_S_rows_over_valid_trials_and_drops_infra_reruns(tmp_path):
    sheet = tmp_path / "sheet.md"
    sheet.write_text(
        SHEET_HEADER
        + "|  1 | 1 | ACT-A | 25 | primed | R | | PECK | pecked | trial_A_v2_20260825_164252 / ep0 | 2.4 s | never |\n"
        + "|  6 | 2 | ACT-B | 25 | primed | S | | — | seated | trial_B_v2_20260825_171233 / ep0 | 2.6 s | 9.8 s |\n"
        + "| 29 | 6 | ACT-B | 25 | primed | INFRA — re-run | | INFRA | camera | trial_B_v2_20260825_201626 / ep0 | 2.7 s | n/a (INFRA) |\n"
        + "| 29b | 6 | ACT-B | 25 | primed | R | | PECK | re-run | trial_B_v2_20260825_202049 / ep0 | 2.7 s | never |\n"
    )
    assert dc.seated_fraction(dc.v2_sheet_rows(sheet)) == (1, 3)


def test_rate_phrase_rounds_to_one_in_n():
    assert dc.rate_phrase(11, 40) == "about 1 in 4"
    assert dc.rate_phrase(10, 40) == "about 1 in 4"
    assert dc.rate_phrase(20, 40) == "about 1 in 2"
    with pytest.raises(ValueError):
        dc.rate_phrase(0, 40)


# --- 2. blinding guard ------------------------------------------------------------------------

@pytest.mark.parametrize("run_id", [
    "trial_A_v2_20260825_164252", "trial_B_v2_20260825_181012",
    "probe_a25_v2_20260822_165905", "B2V-A-02_20260830_233508",
])
def test_unblinded_sources_pass(run_id):
    dc.assert_unblinded(run_id)


@pytest.mark.parametrize("run_id", [
    "q7w2e9r4_20260906_161433",               # an 8-char bench token: arm is sealed
    "so101-tube-insert-v4-noload_20260904_220924",  # a training dataset, not a rollout
    "trial_A_v2_20260825_164252/../sealed",
])
def test_blinded_or_non_rollout_sources_are_refused(run_id):
    with pytest.raises(ValueError):
        dc.assert_unblinded(run_id)


def test_every_clip_in_the_manifest_is_an_unblinded_rollout_in_story_order():
    order = [c.section for c in dc.CLIPS]
    for c in dc.CLIPS:
        dc.assert_unblinded(dc.clip_run_id(c))
    ranks = [dc.SECTIONS.index(s) for s in order]
    assert ranks == sorted(ranks), "story order: it works, it recovers, how it fails"
    assert set(order) == set(dc.SECTIONS)


# --- 5. the public-facing words (Sep 22 re-cut) -----------------------------------------------
# Banned on screen (handoff Sep 22 §1): dates, dataset versions, trial numbers, ACT-A/ACT-B,
# run ids, operator resets. Plus "clean", which the old cut used to mean "uncontaminated data".

BANNED_ON_SCREEN = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b"   # Aug 22
    r"|\b20\d{2}\b|\d{8}_\d{6}"                                                          # years, stamps
    r"|\bv\d\b|\btrials?\b|\bACT-|\bB2V|\bep\s?\d|\bepisodes?\b|\bprobe\b|\brun id\b"
    r"|\breset\b|\boperator\b|\bclean\b",
    re.IGNORECASE,
)


def _on_screen_text():
    return ([c.caption for c in dc.CLIPS] + list(dc.SECTION_TITLE.values())
            + list(dc.TITLE_CARD.lines) + list(dc.END_CARD.lines))


def test_nothing_on_screen_carries_the_evidence_bookkeeping():
    texts = _on_screen_text()
    assert len(texts) >= len(dc.CLIPS) + len(dc.SECTIONS) + 2
    for text in texts:
        m = BANNED_ON_SCREEN.search(text)
        assert m is None, f"{text!r} carries {m and m.group(0)!r}; that belongs in EDL.md"


def test_captions_are_one_line_of_at_most_eight_words():
    assert dc.CAPTION_MAX_WORDS == 8
    for c in dc.CLIPS:
        assert "\n" not in c.caption, (c.clip_id, c.caption)
        assert 1 <= len(c.caption.split()) <= dc.CAPTION_MAX_WORDS, (c.clip_id, c.caption)


def test_section_titles_are_plain_words_without_numbering():
    for title in dc.SECTION_TITLE.values():
        assert "/" not in title and not re.search(r"\d", title), title
        assert title == title.capitalize(), title


def test_the_word_clean_never_describes_an_episode_again():
    for c in dc.CLIPS:
        for field in (c.caption, c.happens, c.why, c.label):
            assert not re.search(r"\bclean", field, re.IGNORECASE), (c.clip_id, field)
    assert not re.search(r"\bclean", dc.NOTES, re.IGNORECASE)


def test_clip_list_is_the_structure_faith_approved_on_sep_22():
    expected = [
        ("C01", "works", "32", None),
        ("C02", "recovers", "28", None),
        ("C03", "fails", "1", None),
        ("C04", "fails", "2", None),
        ("C05", "fails", None, "B2V-A-02_20260830_233508"),
    ]
    assert [(c.clip_id, c.section, c.sheet_trial, c.run_id) for c in dc.CLIPS] == expected
    assert dc.SECTIONS == ["works", "recovers", "fails"]


def test_title_and_end_cards_exist_and_the_end_card_states_the_rate_and_the_link():
    assert dc.TITLE_CARD.seconds > 0 and dc.END_CARD.seconds > 0
    title = " ".join(dc.TITLE_CARD.lines).lower()
    assert "tube" in title and "rack" in title   # Oct 7: Faith dropped "from my demonstrations"
    end = " ".join(dc.END_CARD.lines)
    assert dc.REPO_URL in end
    assert dc.REPO_URL == "github.com/FaithQin/so101-tube-insertion"
    assert "failure" in end.lower()


def test_the_end_card_rate_is_the_sheets_seated_fraction_not_a_recollection():
    """'About 1 in 4' is 11 S rows of 40 valid v2 trials (trial 29 was an INFRA re-run,
    replaced by 29b). Recomputed from the sheet so the card can never drift from the record."""
    seated, valid = dc.seated_fraction(dc.v2_sheet_rows(PROJECT / dc.V2_SHEET))
    assert (seated, valid) == (11, 40)
    assert dc.rate_phrase(seated, valid) in " ".join(dc.END_CARD.lines)


# --- 4. rendering -----------------------------------------------------------------------------

@needs_font
def test_caption_is_burned_into_the_bottom_band_only():
    rgb = np.full((720, 1280, 3), 128, dtype=np.uint8)
    out = dc.caption_frame(rgb, "How it fails", "Misses the grab, keeps pecking")
    assert out.shape == (720, 1280, 3)
    bottom = out[-60:]
    assert np.median(bottom) < 64, "a dark band must sit behind the caption, or it is unreadable"
    blank = dc.caption_frame(rgb, "How it fails", " ")
    assert not np.array_equal(out[-40:], blank[-40:]), "the caption text itself must be drawn"
    assert np.array_equal(out[200:500, 300:1000], rgb[200:500, 300:1000])


@needs_font
def test_caption_frame_carries_no_source_time_stamp():
    """Keep test: every word on screen answers task / works / how it fails. A source timestamp
    answers none of them; it stays in the contact sheets and EDL.md."""
    rgb = np.full((720, 1280, 3), 128, dtype=np.uint8)
    out = dc.caption_frame(rgb, "It works", "Grabs the tube, carries it, drops it in")
    assert np.array_equal(out[:120], rgb[:120])


@needs_font
def test_caption_frame_downscales_wide_sources_to_even_dims_within_1280():
    rgb = np.zeros((1080, 1921, 3), dtype=np.uint8)
    out = dc.caption_frame(rgb, "S", "c")
    assert out.shape[1] <= 1280
    assert out.shape[0] % 2 == 0 and out.shape[1] % 2 == 0


@needs_font
def test_card_frame_is_a_dark_1280x720_frame_with_the_text_drawn():
    out = dc.card_frame(("Succeeds about 1 in 4 times.", "Full results and failure breakdown:"))
    assert out.shape == (720, 1280, 3) and out.dtype == np.uint8
    assert np.median(out) < 40, "cards are dark so the eye rests between clips"
    assert out.max() > 200, "the text must be drawn"
    assert not np.array_equal(out, dc.card_frame((" ",)))


def test_card_frame_counts_are_on_the_20_fps_grid():
    assert dc.card_frame_count(4.0) == 80
    assert dc.card_frame_count(6.0) == 120
    assert dc.cards_runtime() == pytest.approx(dc.TITLE_CARD.seconds + dc.END_CARD.seconds)


def test_encoder_choice_prefers_a_real_h264_and_admits_none():
    assert dc.pick_h264_encoder({"libx264", "h264_videotoolbox", "mpeg4"}) == "libx264"
    assert dc.pick_h264_encoder({"libopenh264"}) == "libopenh264"
    assert dc.pick_h264_encoder({"mpeg4", "libsvtav1"}) is None


def test_contact_sheet_samples_land_exactly_on_the_events_they_label():
    # a tile is labelled with an event only when it IS that frame, so every in-window event
    # must be one of the sampled frames; out-of-window events must not be
    times = dc.sample_times(4.85, 16.2, [5.35, 7.3, 13.8, 14.15, 20.0], n=8)
    assert times == sorted(set(times))
    for t in (5.35, 7.3, 13.8, 14.15):
        assert t in times
    assert 20.0 not in times
    assert all(4.85 - 1e-9 <= t < 16.2 for t in times)
    assert 8 <= len(times) <= 12
    assert all(abs(t * 20 - round(t * 20)) < 1e-6 for t in times)   # on the 20 fps grid


def test_contact_sheet_samples_span_the_whole_window():
    # the fill must cover the clip end to end, not bunch up after the first events
    times = dc.sample_times(10.0, 17.0, [10.5, 11.0, 11.5], n=8)
    assert times[0] == pytest.approx(10.0) and times[-1] == pytest.approx(16.95)
    gaps = np.diff(times)
    assert gaps.max() <= 2 * (7.0 / 7) + 1e-9


def test_total_runtime_sums_windows():
    assert dc.total_runtime([(0.3, 9.15), (5.0, 10.2)]) == pytest.approx(14.05)


# --- data-backed checks ---------------------------------------------------------------------
# Video events (seat runs, cap track) cost ~2 s per rollout to decode, so the full analysis is
# written once to a tracked events.json by `python tools/demo_cut.py` and the suite checks that
# file against the primary record cheaply: every joint event re-read from parquet, and the one
# video claim the cut's finale leans on re-decoded from its own recording.

EVENTS_JSON = PROJECT / "analysis/demo_cut/events.json"


def test_tracked_events_name_each_clips_own_source():
    """events.json is keyed by clip id; after a re-cut a stale file would silently anchor the new
    clips on the old clips' events. The run id inside each entry must be the clip's own."""
    stored = dc.load_events(EVENTS_JSON)
    assert set(stored) == {c.clip_id for c in dc.CLIPS}
    for c in dc.CLIPS:
        assert stored[c.clip_id]["run_id"] == dc.clip_run_id(c), c.clip_id


@needs_cache
def test_recomputed_grasp_close_reproduces_the_v2_sheet_for_every_v2_clip():
    rows = dc.v2_sheet_rows(PROJECT / dc.V2_SHEET)
    checked = 0
    for c in dc.CLIPS:
        if c.sheet_trial is None:
            continue
        row = rows[c.sheet_trial]
        ev = dc.joint_events(dc.resolve_rollout(row["run_id"]), row["episode"])
        # the sheet rounds to 0.1 s
        assert abs(ev["grasp_close"] - row["grasp_close_s"]) <= 0.05 + 1e-9, c.clip_id
        checked += 1
    assert checked == sum(1 for c in dc.CLIPS if c.sheet_trial) >= 4


@needs_cache
def test_tracked_events_match_a_fresh_read_of_the_parquet():
    stored = dc.load_events(EVENTS_JSON)
    for c in dc.CLIPS:
        fresh = dc.joint_events(dc.resolve_rollout(dc.clip_run_id(c)), c.episode)
        for key, value in fresh.items():
            assert stored[c.clip_id][key] == pytest.approx(value), (c.clip_id, key)


@needs_cache
def test_the_works_clip_is_the_one_seated_run_with_a_single_grasp_and_no_re_aim():
    """Sep 22 search over all 45 unblinded episodes: five seated runs have one commanded close
    before the seat and the minimal three lift turning points (grasp, carry apex, insertion);
    of those only trial 32's scorer note records no correction at all. The second commanded
    close in every seated run is the jaws shutting on the way home after release."""
    c = dc.CLIPS[0]
    assert c.section == "works"
    d = dc.resolve_rollout(dc.clip_run_id(c))
    ev = dc.joint_events(d, c.episode)
    vev = dc.video_events(d, c.episode, rack_entry_s=ev["rack_entry"])
    assert vev["final"] == "SEAT"
    act, _ = dc._episode_arrays(d, c.episode)
    closes_s = [i / dc.FPS for i in dc.commanded_closes(act[:, 5])]
    assert sum(t < vev["seat_hold_start"] for t in closes_s) == 1, closes_s
    assert ev["n_closes"] == 2 and closes_s[1] > vev["seat_hold_start"]
    assert ev["lift_reversals"] == 3


@needs_cache
def test_the_funnel_clip_seat_runs_are_what_its_recording_says():
    """Write-up §3 quotes six seat runs, longest 0.6 s, final MISS for B2V-A-02."""
    ev = dc.video_events(dc.resolve_rollout("B2V-A-02_20260830_233508"), 0)
    assert len(ev["seat_runs"]) == 6
    assert max(d for _, _, d in ev["seat_runs"]) == pytest.approx(0.6)
    assert ev["final"] == "MISS"


def test_resolved_cut_with_cards_runs_40_to_50_seconds():
    plan = dc.plan(dc.load_events(EVENTS_JSON))
    assert [p["clip_id"] for p in plan] == [c.clip_id for c in dc.CLIPS]
    clips = dc.total_runtime([(p["in_s"], p["out_s"]) for p in plan])
    assert 30.0 <= clips <= 40.0
    assert 40.0 <= clips + dc.cards_runtime() <= 50.0


@needs_cache
def test_reversal_detector_reproduces_the_findings_log_counts():
    """the lab notebook (private):231 (39 on the Aug 22 probe's ep 1) and :333-334 (66, 67, 67,
    67, 67 on the Aug 22 n=25 block): the peck count in the events table uses this detector."""
    probe = dc.resolve_rollout("probe_a25_v2_20260822_165905")
    act, _ = dc._episode_arrays(probe, 1)
    assert len(dc.lift_reversals(act[:, 1])) == 39
    block = dc.resolve_rollout("block_a25_v2_20260822_172819")
    counts = [len(dc.lift_reversals(dc._episode_arrays(block, e)[0][:, 1])) for e in range(5)]
    assert counts == [66, 67, 67, 67, 67]


# --- 6. Oct 7 polish (Faith, ARMLab interview day) --------------------------------------------
# Faith, Oct 7 2026: "change the title to just show for 2 seconds", "last slide can just show
# for 3 seconds", and "I don't like the font used in the video, it feels very basic and plain
# but in a bad way, try Times New Roman". Clips, caption words and layout are unchanged.

def test_title_card_shows_for_two_seconds():
    assert dc.TITLE_CARD.seconds == 2.0
    assert dc.card_frame_count(dc.TITLE_CARD.seconds) == 40



# Faith, Oct 7 2026: "for the demo video don't say 'a 3D printed arm' ... just say LeRobot SO-101
# arms". The arm is named by its official name, and nothing on a card sells it by price or build.
def test_title_card_names_the_arm_lerobot_so101_and_nothing_else():
    assert dc.TITLE_CARD.lines[0] == "A LeRobot SO-101 arm learns to put a tube in a rack"
    assert len(dc.TITLE_CARD.lines) == 1, "Oct 7: the title card is one line, no 'from my demonstrations'"
    for line in dc.TITLE_CARD.lines + dc.END_CARD.lines:
        for banned in ("3D", "printed", "$"):
            assert banned.lower() not in line.lower(), (banned, line)

def test_end_card_shows_for_three_seconds():
    assert dc.END_CARD.seconds == 3.0
    assert dc.card_frame_count(dc.END_CARD.seconds) == 60


def test_cut_is_35_70_s_of_clips_plus_5_s_of_cards():
    """Only the cards changed on Oct 7: the five windows still sum to 35.70 s, so the cut runs
    40.70 s (was 45.70 s with the 4 s title and 6 s end card), inside the 40-50 s bound."""
    plan = dc.plan(dc.load_events(EVENTS_JSON))
    clips = dc.total_runtime([(p["in_s"], p["out_s"]) for p in plan])
    assert clips == pytest.approx(35.70)
    assert dc.cards_runtime() == pytest.approx(5.0)
    assert clips + dc.cards_runtime() == pytest.approx(40.70)


@needs_font
def test_the_video_font_is_arial_regular_at_every_size_used():
    assert Path(dc.FONT_PATH) == VIDEO_FONT
    for size in (dc.SECTION_SIZE, dc.CAPTION_SIZE, dc.CARD_SIZE):
        font = dc._font(size)
        assert font.getname() == ("Arial", "Regular"), font.getname()
        assert font.size == size


@needs_font
def test_every_text_render_draws_in_arial(monkeypatch):
    """Spy on PIL's text call: the section title, the caption and every card line must each be
    drawn with an explicit Arial font, never PIL's default and never a system fallback."""
    from PIL import ImageDraw
    drawn = []
    real_text = ImageDraw.ImageDraw.text

    def spy(self, xy, text, *args, **kwargs):
        drawn.append((text, kwargs.get("font")))
        return real_text(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", spy)
    rgb = np.full((720, 1280, 3), 128, dtype=np.uint8)
    dc.caption_frame(rgb, "How it fails", "Misses the grab, keeps pecking")
    dc.card_frame(dc.TITLE_CARD.lines)
    dc.card_frame(dc.END_CARD.lines)
    assert [t for t, _ in drawn] == ["How it fails", "Misses the grab, keeps pecking",
                                     *dc.TITLE_CARD.lines, *dc.END_CARD.lines]
    for text, font in drawn:
        assert font is not None, f"{text!r} drawn in PIL's default font"
        assert font.getname()[0] == "Arial", (text, font.getname())


def test_a_missing_font_is_refused_not_swapped_for_another(monkeypatch):
    """A fallback chain (the old Helvetica -> Arial -> PIL default) would render a public video in
    a font nobody chose without a word. A missing font must stop the render."""
    dc._font.cache_clear()
    monkeypatch.setattr(dc, "FONT_PATH", "/nonexistent/Arial.ttf")
    try:
        with pytest.raises(OSError):
            dc._font(34)
    finally:
        dc._font.cache_clear()


@needs_font
def test_every_caption_and_card_line_fits_one_line_at_1280_wide():
    """Times New Roman sets narrower than Helvetica, but the longest caption is pinned anyway:
    a caption that runs off the frame edge is a caption the stranger cannot read."""
    margin = 24                                        # caption_frame's left inset, mirrored right
    widest = max(dc.CLIPS, key=lambda c: dc._font(dc.CAPTION_SIZE).getlength(c.caption))
    assert margin + dc._font(dc.CAPTION_SIZE).getlength(widest.caption) <= dc.MAX_WIDTH - margin
    for title in dc.SECTION_TITLE.values():
        assert margin + dc._font(dc.SECTION_SIZE).getlength(title) <= dc.MAX_WIDTH - margin
    for line in dc.TITLE_CARD.lines + dc.END_CARD.lines:
        assert dc._font(dc.CARD_SIZE).getlength(line) <= dc.MAX_WIDTH - 2 * margin, line
