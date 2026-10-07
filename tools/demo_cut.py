#!/usr/bin/env python
"""The demo cut: an edit decision list, contact sheets and a captioned mp4, from archived rollouts.

Sep 12 2026: cut from archived rollouts, no bench. Every scored trial and
probe is on disk in both cameras (`the Sep 2 assessment note (private):330-338` lists the
cut). This tool touches no hardware: it reads rollout parquet and video from the local cache.

Re-cut Sep 22 2026 for a stranger with 90 seconds (a PhD student or founder opening a
cold-email link) who needs three answers: what is the task, does it work, how does it fail.
Five clips between a title card and an end card, about 41 s (Oct 7: cards trimmed to 2 s and
3 s, every on-screen word set in Times New Roman, at Faith's request). Everything that was on screen to
satisfy a skeptical reviewer (dates, dataset versions, trial numbers, arm labels, run ids,
operator resets) now lives only in EDL.md, which is the audit trail; captions are one line of
at most eight plain words. The v2 A/B comparison was a null and is not a point this video makes.

    python tools/demo_cut.py                 # analyse, write events.json, EDL.md, sheets, mp4
    python tools/demo_cut.py --no-video      # everything except the mp4

WHAT IT REFUSES TO DO BY EYE
----------------------------
* In/out points. Each clip is `(event, offset)` pairs; events come from the episode's own
  data: the v2 sheet's forensic grasp-close (commanded gripper open >15 -> closed <8,
  `Eval Trials — v2 paired blocks.md:81-83`), home return (commanded lift < -95 after 2 s,
  :84-91), pick-site exit (pan crossing `v4_audit.PICK_SITE_PAN`), rack entry
  (`v4_audit.rack_slice`), seat runs (`score_episode.score_frames` on the front video), and the
  cap track from `score_episode.classify_frame` (displacement / loss).
* Source mapping. A v2 clip names a sheet trial; its run id is parsed from the sheet and
  resolved to exactly one `rollout_<run id>[_<stamp>]` directory. The suite checks that the
  recomputed grasp-close reproduces the sheet's column for every v2 clip.
* Blinding. Sources are whitelisted to unblinded families; a Sep 6 bench token cannot pass.

What the eye DID decide, and says so: the words. The "what happens" and "why it belongs" text
restates the v2 sheet's video-scored notes and the the lab notebook (private), cited by line, and renders
only into EDL.md; where the video contradicts a written claim, the EDL's notes say so
(B2V-A-02, see NOTES below). The captions describe the frame in plain words and nothing else.

Timebase: video time = frame index / 20. The loop ran ~19.3 Hz on Aug 22
(`the lab notebook (private):242-244`), so 20 fps playback is ~3% faster than the wall clock.
Tests: tests/test_demo_cut.py.
"""

from __future__ import annotations

import argparse
import functools
import glob
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v4_audit import PICK_SITE_PAN, grasp_index, rack_slice  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
ROLLOUT_ROOT = Path("~/.cache/huggingface/lerobot/faithqin").expanduser()
OUT_DIR = PROJECT / "analysis" / "demo_cut"
V2_SHEET = "Eval Trials — v2 paired blocks.md"
FPS = 20
MAX_WIDTH = 1280

# v2 sheet forensic definitions (Eval Trials — v2 paired blocks.md:81-91)
GRIP_OPEN_ABOVE = 15.0
GRIP_CLOSED_BELOW = 8.0
HOME_LIFT_BELOW = -95.0
HOME_AFTER_S = 2.0
# the lab notebook (private):231 counts shoulder_lift reversals ">6°"
REVERSAL_DEG = 6.0

UNBLINDED = re.compile(
    r"^(trial_[AB]_v2_\d{8}_\d{6}"        # v2 paired blocks: the sheet names the arm per row
    r"|probe_a25_v2_\d{8}_\d{6}"          # Aug 22 ACT-A-v2 probe
    r"|B2V-A-02_\d{8}_\d{6})$"            # Aug 30 Block 2 probe, arm in the label
)

SECTIONS = ["works", "recovers", "fails"]
SECTION_TITLE = {
    "works": "It works",
    "recovers": "It recovers",
    "fails": "How it fails",
}
CAPTION_MAX_WORDS = 8            # Faith, Sep 22: one line, about eight words, the same shape every time
REPO_URL = "github.com/FaithQin/so101-tube-insertion"   # the public repo (name approved Sep 22)
FUNNEL_RUN_ID = "B2V-A-02_20260830_233508"              # the source behind NOTES 1 and its evidence sheet
H264_PREFERENCE = ("libx264", "h264_videotoolbox", "libopenh264")


@dataclass(frozen=True)
class Card:
    """A full-frame text card: `lines` centred on a dark 1280x720 frame for `seconds`."""
    lines: tuple
    seconds: float


# Faith, Oct 7 2026: "change the title to just show for 2 seconds" (was 4 s) and "last slide
# can just show for 3 seconds" (was 6 s).
TITLE_CARD = Card(("A 3D-printed arm learns to put a tube in a rack,",
                   "from my demonstrations"), 2.0)
# "about 1 in 4" is the v2 sheet's seated fraction, 11 S rows of 40 valid trials; the suite
# recomputes it from the sheet (seated_fraction / rate_phrase) so this literal cannot drift.
END_CARD = Card(("Succeeds about 1 in 4 times.",
                 "Full results and failure breakdown:",
                 REPO_URL), 3.0)


@dataclass(frozen=True)
class Clip:
    clip_id: str
    section: str
    sheet_trial: str | None      # v2 sheet row; run id is parsed from the sheet
    run_id: str | None           # non-sheet sources
    episode: int
    in_at: tuple                 # (event, offset_s)
    out_at: tuple
    label: str
    caption: str
    happens: str
    why: str
    policy: str | None = None     # non-sheet sources; sheet rows carry their own
    camera: str = "front"


CLIPS = [
    Clip("C01", "works", "32", None, 0,
         ("grasp_close", -2.0), ("home_return", 0.5),
         "v2 trial 32, ACT-B",
         "Grabs the tube, carries it, drops it in",
         "Reach, one commanded close on the cap, lift and carry, lower into the funnel, the tube "
         "drops in and stays; folded home.",
         "The one seated v2 trial whose scorer note records no correction at all: 'grasped the "
         "cap's top half first try; guided to the rack and seated in one go' (v2 sheet :129). The "
         "kinematics agree: one commanded close before the seat and the minimal three lift turning "
         "points (grasp, carry apex, insertion). The Sep 22 search over all 45 unblinded episodes "
         "found four other seated runs with that profile (trials 6, 19, 23, 38), and each of their "
         "notes records something at the rack: an adjustment, a pause, resistance, or an off-centre "
         "push that the bevel guided in. The second commanded close (8.9 s) is the jaws shutting on "
         "the way home after release, not a second grab. See NOTES 3."),
    Clip("C02", "recovers", "28", None, 0,
         ("rack_entry", -1.0), ("seat_hold_start", 1.5),
         "v2 trial 28, ACT-B",
         "Hits the rack, adjusts, gets it in",
         "Carry arrives at the rack's top-left corner, the tube is held against it under push, "
         "nudged toward the centre, drops in; arm leaves.",
         "A contact recovery the camera can see: 'at the rack hit the top-left corner, nudged "
         "centerward after resistance, seated' (v2 sheet :124). The missed first grab (commanded "
         "closes at 2.95 and 8.45 s) is before the window; the failure section carries that pattern."),
    Clip("C03", "fails", "1", None, 0,
         ("grasp_close", -1.0), ("grasp_close", 5.0),
         "v2 trial 1, ACT-A, PECK",
         "Misses the grab, keeps pecking",
         "First close on air beside the cap; jaws re-open and undulate above it; tube never moves.",
         "The woodpecker, the project's named symptom (v2 sheet :97; mechanism definition :44-46)."),
    Clip("C04", "fails", "2", None, 0,
         ("grasp_close", -0.5), ("cap_displaced", 2.0),
         "v2 trial 2, ACT-A, SLIP",
         "Grabs the tip, tube twists out",
         "Close on the cap tip; the tube rotates out of the grip and is left angled.",
         "SLIP family, first of three near-identical trials (v2 sheet :98-100); one instance stands "
         "for the family so a stranger sees three failure types, not five."),
    Clip("C05", "fails", None, FUNNEL_RUN_ID, 0,
         ("rack_entry", -0.5), ("seat_run_3_end", 1.0),
         "Aug 30 probe B2V-A-02, ACT-A-v3",
         "Reaches the rack, can't get it in",
         "Carry to the funnel, tip at the mouth without entering, withdraw still holding, swing "
         "back toward the pick site, return, release; the tube lies flat beside the rack.",
         "The v3 insertion failure (the results draft (private):95-96). Recorded Aug 30 23:35, "
         "inside the valid-plant bracket (last verified-healthy Aug 31 15:18, the lab notebook (private) — "
         "Capstone.md:1168-1169; the Sep 2 assessment note (private):48). See NOTES 1: the "
         "recording does not show a seat, and the honest line is 'reaches the funnel and cannot "
         "insert'.",
         policy="ACT-A-v3"),
]


# --- event detectors (pure) ----------------------------------------------------------------

def commanded_closes(grip, open_above=GRIP_OPEN_ABOVE, closed_below=GRIP_CLOSED_BELOW):
    """Indices where the commanded gripper goes from open (>15) to closed (<8), with hysteresis.

    The first one is the v2 sheet's forensic grasp-close t (v2 sheet :81-83)."""
    out, opened = [], False
    for i, v in enumerate(grip):
        if v > open_above:
            opened = True
        elif v < closed_below and opened:
            out.append(i)
            opened = False
    return out


def home_return_index(lift, fps=FPS, after_s=HOME_AFTER_S, below=HOME_LIFT_BELOW):
    """First tick after 2 s where the commanded shoulder_lift drops below -95 (v2 sheet :84-91)."""
    lift = np.asarray(lift, dtype=float)
    for i in range(int(round(after_s * fps)), len(lift)):
        if lift[i] < below:
            return i
    return None


def pick_exits(pan, pick_pan=PICK_SITE_PAN):
    """Indices where the measured pan leaves the pick site (crosses from > to <= 55)."""
    pan = np.asarray(pan, dtype=float)
    return [i for i in range(1, len(pan)) if pan[i - 1] > pick_pan and pan[i] <= pick_pan]


def rack_entry_index(pan, wflex):
    """First frame of `v4_audit.rack_slice` (wrist_flex < 20 and pan < 40)."""
    idx = rack_slice(pan, wflex)
    return int(idx[0]) if len(idx) else None


def _ref_cap(caps, n=10):
    seen = [c for c in caps if c is not None][:n]
    if not seen:
        return None
    return np.median(np.asarray(seen, dtype=float), axis=0)


def cap_displaced_index(caps, radius_px=8.0, sustain=5):
    """Onset of the first sustained move of the cap away from its opening position.

    Reference = median of the first ten detections. Undetected frames (the jaws occlude the
    cap) neither count nor break a run; a lone outlier is never an onset."""
    ref = _ref_cap(caps)
    if ref is None:
        return None
    run_start, run_len = None, 0
    for i, c in enumerate(caps):
        if c is None:
            continue
        if np.hypot(c[0] - ref[0], c[1] - ref[1]) > radius_px:
            if run_start is None:
                run_start = i
            run_len += 1
            if run_len >= sustain:
                return run_start
        else:
            run_start, run_len = None, 0
    return None


def cap_lost_index(caps, start, min_missing=20):
    """First index >= start that begins >= min_missing consecutive undetected frames."""
    run_start, run_len = None, 0
    for i in range(start, len(caps)):
        if caps[i] is None:
            if run_start is None:
                run_start = i
            run_len += 1
            if run_len >= min_missing:
                return run_start
        else:
            run_start, run_len = None, 0
    return None


def held_seat_start(runs, duration_s, fps=FPS):
    """Start of the seat run that holds to the last frame (score_episode.py:130 tolerance)."""
    for a, b, _d in runs:
        if abs(b - (duration_s - 1.0 / fps)) < 0.2:
            return a
    return None


def lift_reversals(lift, min_deg=REVERSAL_DEG):
    """Turning points of a zig-zag with `min_deg` hysteresis (indices of the extremes)."""
    x = np.asarray(lift, dtype=float)
    pts, direction, lo, hi = [], 0, 0, 0
    for i in range(1, len(x)):
        if direction == 0:
            if x[i] - x[lo] > min_deg:
                direction, hi = 1, i
            elif x[hi] - x[i] > min_deg:
                direction, lo = -1, i
            else:
                lo = i if x[i] < x[lo] else lo
                hi = i if x[i] > x[hi] else hi
        elif direction == 1:
            if x[i] >= x[hi]:
                hi = i
            elif x[hi] - x[i] > min_deg:
                pts.append(hi)
                direction, lo = -1, i
        else:
            if x[i] <= x[lo]:
                lo = i
            elif x[i] - x[lo] > min_deg:
                pts.append(lo)
                direction, hi = 1, i
    return pts


# --- windows -------------------------------------------------------------------------------

def resolve_window(events, spec_in, spec_out, duration_s):
    """(in_s, out_s) from (event, offset) specs, clamped to the episode."""
    times = []
    for name, offset in (spec_in, spec_out):
        t = events[name]                      # KeyError: the event was never computed
        if t is None:
            raise ValueError(f"event {name!r} was not detected in this episode")
        times.append(min(max(float(t) + float(offset), 0.0), float(duration_s)))
    if times[0] >= times[1]:
        raise ValueError(f"empty window {times[0]:.2f} -> {times[1]:.2f} s")
    return times[0], times[1]


def sample_times(a, b, event_times, n=8, max_n=12):
    """Contact-sheet frame times on the 20 fps grid: every event inside [a, b), then even fill.

    Tiles are labelled only with events they sit on exactly, so a label can never point at a
    frame up to a second away from the event it names."""
    snap = lambda t: round(t * FPS) / FPS  # noqa: E731
    last = snap(b - 1.0 / FPS)
    events = sorted({snap(t) for t in event_times if t is not None and a - 1e-9 <= t < b - 1e-9})
    chosen = set(events[:max_n])
    target = max(n, min(max_n, len(chosen) + 2))
    first = snap(a)
    grid = [snap(first + i / FPS) for i in range(int(round((last - first) * FPS)) + 1)]
    for end in (first, last):
        if len(chosen) < target:
            chosen.add(end)
    while len(chosen) < min(target, len(grid)):
        # farthest-point fill: the grid frame farthest from everything already chosen
        chosen.add(max(grid, key=lambda g: min(abs(g - c) for c in chosen)))
    return sorted(chosen)


def total_runtime(windows):
    return float(sum(b - a for a, b in windows))


# --- sources ---------------------------------------------------------------------------------

def assert_unblinded(run_id):
    if not isinstance(run_id, str) or not UNBLINDED.match(run_id):
        raise ValueError(f"{run_id!r} is not an unblinded rollout source for the demo cut")


def resolve_rollout(run_id, root=None):
    root = Path(root) if root is not None else ROLLOUT_ROOT
    pat = re.compile(r"^rollout_" + re.escape(run_id) + r"(_\d{8}_\d{6})?$")
    hits = sorted(p for p in root.iterdir() if p.is_dir() and pat.match(p.name)) if root.is_dir() else []
    if not hits:
        raise FileNotFoundError(f"no rollout directory for {run_id} under {root}")
    if len(hits) > 1:
        raise RuntimeError(f"{run_id} is ambiguous: {[h.name for h in hits]}")
    return hits[0]


def _seconds(cell):
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*s\b", cell)
    return float(m.group(1)) if m else None


def v2_sheet_rows(path):
    """Trial rows of the v2 sheet keyed by trial number ('29b' kept as its own key)."""
    rows = {}
    for lineno, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.startswith("|"):
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) < 12 or not re.match(r"^\d+b?$", c[0]):
            continue
        m = re.match(r"^(\S+)\s*/\s*ep(\d+)$", c[9])
        if not m:
            continue
        rows[c[0]] = dict(trial=c[0], policy=c[2], stage=c[5], mechanism=c[7], notes=c[8],
                          run_id=m.group(1), episode=int(m.group(2)),
                          grasp_close_s=_seconds(c[10]), home_return_s=_seconds(c[11]),
                          line=lineno)
    return rows


def seated_fraction(rows):
    """(seated, valid) over the sheet's trial rows: S in the live Stage column over every row
    that is not an INFRA re-run (v2: 11 of 40; trial 29 was INFRA and 29b replaced it)."""
    valid = [r for r in rows.values() if not r["stage"].upper().startswith("INFRA")]
    seated = [r for r in valid if r["stage"] == "S"]
    return len(seated), len(valid)


def rate_phrase(seated, valid):
    """'about 1 in N' for the end card, N = round(valid / seated)."""
    if seated <= 0 or valid <= 0:
        raise ValueError("a rate needs at least one seat and one valid trial")
    return f"about 1 in {round(valid / seated)}"


@functools.lru_cache(maxsize=1)
def _project_sheet():
    return v2_sheet_rows(PROJECT / V2_SHEET)


def clip_run_id(clip):
    return _project_sheet()[clip.sheet_trial]["run_id"] if clip.sheet_trial else clip.run_id


# --- reading a rollout -----------------------------------------------------------------------

def _episode_meta(rollout_dir, episode):
    import pandas as pd
    files = sorted(glob.glob(str(Path(rollout_dir) / "meta/episodes/*/*.parquet")))
    meta = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return meta[meta.episode_index == episode].iloc[0]


def _episode_arrays(rollout_dir, episode):
    import pandas as pd
    files = sorted(glob.glob(str(Path(rollout_dir) / "data/*/*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    g = df[df.episode_index == episode].sort_values("frame_index")
    return np.stack(g["action"].values), np.stack(g["observation.state"].values)


def joint_events(rollout_dir, episode):
    """Events from the recorded action (commanded) and observation.state (measured) streams."""
    act, st = _episode_arrays(rollout_dir, episode)
    n = len(act)
    t = lambda i: None if i is None else i / FPS  # noqa: E731
    closes = commanded_closes(act[:, 5])
    exits = pick_exits(st[:, 0])
    first_exit = exits[0] if exits else n
    pick_closes = [i for i in closes if i < first_exit]
    return {
        "n_frames": n,
        "duration_s": n / FPS,
        "grasp_close": t(closes[0] if closes else None),
        "close_2": t(closes[1] if len(closes) > 1 else None),
        "n_closes": len(closes),
        "last_pick_close": t(pick_closes[-1] if pick_closes and exits else None),
        "v4_grasp_index": t(grasp_index(st[:, 0], st[:, 5])),
        "pick_exit": t(exits[0] if exits else None),
        "pick_exit_2": t(exits[1] if len(exits) > 1 else None),
        "rack_entry": t(rack_entry_index(st[:, 0], st[:, 3])),
        "home_return": t(home_return_index(act[:, 1])),
        "lift_reversals": len(lift_reversals(act[:, 1])),
    }


def iter_frames(rollout_dir, episode, camera="front", t_in=0.0, t_out=None):
    """(episode time s, rgb ndarray) for frames of `episode` with t_in <= t < t_out."""
    import av
    meta = _episode_meta(rollout_dir, episode)
    key = f"videos/observation.images.{camera}"
    path = Path(rollout_dir) / "videos" / f"observation.images.{camera}" / \
        f"chunk-{int(meta[key + '/chunk_index']):03d}" / f"file-{int(meta[key + '/file_index']):03d}.mp4"
    t0, t1 = float(meta[key + "/from_timestamp"]), float(meta[key + "/to_timestamp"])
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        for fr in container.decode(stream):
            ts = float(fr.pts * stream.time_base)
            if ts < t0 - 1e-6:
                continue
            if ts >= t1 - 1e-6:
                break
            te = round((ts - t0) * FPS) / FPS
            if te < t_in - 1e-6:
                continue
            if t_out is not None and te >= t_out - 1e-6:
                break
            yield te, fr.to_ndarray(format="rgb24")
    finally:
        container.close()


def video_events(rollout_dir, episode, rack_entry_s=None):
    """Seat runs via score_episode.score_frames, plus cap-track events, from the front video."""
    import score_episode as se
    caps = []

    def tapped():
        for _t, rgb in iter_frames(rollout_dir, episode, "front"):
            bgr = np.ascontiguousarray(rgb[..., ::-1])
            caps.append(se.classify_frame(bgr)[0])
            yield bgr

    r = se.score_frames(tapped(), fps=FPS)
    runs = [[round(a, 4), round(b, 4), round(d, 4)] for a, b, d in r["seat_runs"]]
    lost = None
    if rack_entry_s is not None:
        li = cap_lost_index(caps, start=int(round(rack_entry_s * FPS)), min_missing=FPS)
        lost = None if li is None else li / FPS
    disp = cap_displaced_index(caps, radius_px=8.0, sustain=5)
    return {
        "final": r["final"],
        "final_cap": list(r["final_cap"]) if r["final_cap"] else None,
        "seat_runs": runs,
        "held_to_end": r["held_to_end"],
        "infra_start_seated": r["infra_start_seated"],
        "seat_hold_start": held_seat_start(r["seat_runs"], r["duration_s"], FPS),
        "seat_run_3_end": runs[2][1] if len(runs) >= 3 else None,
        "cap_displaced": None if disp is None else disp / FPS,
        "cap_lost_after_rack": lost,
    }


def analyze(clip):
    run_id = clip_run_id(clip)
    assert_unblinded(run_id)
    d = resolve_rollout(run_id)
    ev = joint_events(d, clip.episode)
    ev.update(video_events(d, clip.episode, rack_entry_s=ev["rack_entry"]))
    ev["run_id"], ev["rollout_dir"] = run_id, d.name
    return ev


def load_events(path):
    return json.loads(Path(path).read_text())


def plan(events):
    out = []
    for c in CLIPS:
        ev = events[c.clip_id]
        a, b = resolve_window(ev, c.in_at, c.out_at, ev["duration_s"])
        out.append(dict(clip_id=c.clip_id, section=c.section, run_id=ev.get("run_id"),
                        rollout_dir=ev.get("rollout_dir"), episode=c.episode, camera=c.camera,
                        in_s=round(a, 2), out_s=round(b, 2), dur_s=round(b - a, 2)))
    return out


# --- rendering -------------------------------------------------------------------------------

# Faith, Oct 7 2026: Helvetica "feels very basic and plain but in a bad way, try Times New Roman".
# One font, no fallback chain: a missing font stops the render instead of quietly swapping in
# another face (the old Helvetica -> Arial -> PIL-default chain could do exactly that).
FONT_PATH = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
SECTION_SIZE = 24      # section title above the caption
CAPTION_SIZE = 34      # the one-line caption
CARD_SIZE = 40         # title and end card lines


@functools.lru_cache(maxsize=8)
def _font(size):
    from PIL import ImageFont
    # Checked by path, not left to Pillow: truetype() given a missing path quietly searches the
    # system font folders by file name and can load a different file than the one named here.
    if not Path(FONT_PATH).is_file():
        raise FileNotFoundError(f"video font {FONT_PATH} is missing; refusing to render in another")
    return ImageFont.truetype(FONT_PATH, size)


def _fit_width(img, max_width=MAX_WIDTH):
    from PIL import Image
    w, h = img.size
    if w > max_width:
        h = int(round(h * max_width / w))
        w = max_width
    w, h = w - (w % 2), h - (h % 2)
    return img.resize((w, h), Image.BILINEAR) if (w, h) != img.size else img


def caption_frame(rgb, section, caption):
    """Burn a section title + one-line caption into a band at the bottom. Nothing else: the
    source time and every other piece of bookkeeping live in the contact sheets and EDL.md."""
    from PIL import Image, ImageDraw
    img = _fit_width(Image.fromarray(np.asarray(rgb, dtype=np.uint8)))
    w, h = img.size
    lines = caption.split("\n")
    band = 50 + 42 * len(lines)
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.rectangle([0, h - band, w, h], fill=(0, 0, 0, 185))
    out = Image.alpha_composite(img.convert("RGBA"), overlay)
    d = ImageDraw.Draw(out)
    d.text((24, h - band + 10), section, font=_font(SECTION_SIZE), fill=(255, 196, 64, 255))
    for k, line in enumerate(lines):
        d.text((24, h - band + 42 + 42 * k), line, font=_font(CAPTION_SIZE), fill=(255, 255, 255, 255))
    return np.asarray(out.convert("RGB"))


def card_frame(lines, size=(MAX_WIDTH, 720), bg=(16, 16, 16)):
    """A dark full frame with `lines` centred: the title card and the end card."""
    from PIL import Image, ImageDraw
    w, h = size
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    font, gap = _font(CARD_SIZE), 60
    y = (h - gap * len(lines)) // 2
    for line in lines:
        x0, y0, x1, y1 = d.textbbox((0, 0), line, font=font)
        d.text(((w - (x1 - x0)) // 2 - x0, y - y0), line, font=font, fill=(240, 240, 240))
        y += gap
    return np.asarray(img, dtype=np.uint8)


def card_frame_count(seconds, fps=FPS):
    return int(round(seconds * fps))


def cards_runtime():
    return float(TITLE_CARD.seconds + END_CARD.seconds)


def pick_h264_encoder(available=None):
    if available is None:
        import av
        available = set()
        for name in H264_PREFERENCE:
            try:
                av.codec.Codec(name, "w")
                available.add(name)
            except Exception:
                pass
    for name in H264_PREFERENCE:
        if name in available:
            return name
    return None


KEY_EVENTS = ("grasp_close", "last_pick_close", "pick_exit", "rack_entry", "seat_hold_start")


def _event_marks(ev, clip):
    """The clip's own anchors, the task milestones, and seat-run onsets - snapped to frames."""
    names = dict.fromkeys([clip.in_at[0], clip.out_at[0], *KEY_EVENTS])
    marks = [(round(ev[n] * FPS) / FPS, n) for n in names if ev.get(n) is not None]
    marks += [(round(a * FPS) / FPS, f"seat run {k + 1}")
              for k, (a, _b, _d) in enumerate(ev.get("seat_runs", []))]
    return sorted(marks)


SHEET_CROP = (150, 0, 870, 720)   # arm base, pick mark (~362,355), funnel (~527,464), mat below


def contact_sheet(samples, title, marks, cols=4, tile=(320, 320), crop=SHEET_CROP):
    """samples: [(t, rgb)]; tiles cropped to the work area, labelled with the events they ARE."""
    from PIL import Image, ImageDraw
    rows = (len(samples) + cols - 1) // cols
    head = 34
    sheet = Image.new("RGB", (cols * tile[0], head + rows * tile[1]), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    d.text((8, 8), title, font=_font(15), fill=(255, 255, 255))
    for k, (t, rgb) in enumerate(samples):
        im = Image.fromarray(rgb).crop(crop).resize(tile, Image.BILINEAR)
        di = ImageDraw.Draw(im)
        near = [name for mt, name in marks if abs(mt - t) < 1e-6]
        label = f"t={t:.2f}s" + (("  " + ", ".join(near)) if near else "")
        di.rectangle([0, 0, tile[0], 20], fill=(0, 0, 0))
        di.text((4, 3), label, font=_font(13), fill=(255, 220, 90) if near else (255, 255, 255))
        sheet.paste(im, ((k % cols) * tile[0], head + (k // cols) * tile[1]))
    return sheet


def seat_run_evidence(rollout_dir, episode, runs, crop=(380, 220, 800, 580)):
    """One tile per seat run (its middle frame), seat box and classifier centroid drawn."""
    from PIL import Image, ImageDraw
    import score_episode as se
    wanted = [round(((a + b) / 2) * FPS) / FPS for a, b, _ in runs]
    got = {}
    for t, rgb in iter_frames(rollout_dir, episode, "front", t_in=min(wanted), t_out=max(wanted) + 0.05):
        for w in wanted:
            if abs(t - w) < 1e-6:
                got[w] = rgb
    cw, ch = crop[2] - crop[0], crop[3] - crop[1]
    head = 34
    sheet = Image.new("RGB", (3 * cw, head + 2 * ch), (20, 20, 20))
    ImageDraw.Draw(sheet).text(
        (8, 8), f"{Path(rollout_dir).name} front: the middle frame of each score_episode seat run "
        f"(green = seat box, score_episode.py:37-38; magenta = cap centroid)",
        font=_font(15), fill=(255, 255, 255))
    for k, ((a, b, dur), w) in enumerate(zip(runs, wanted)):
        rgb = got[w]
        cap, seat = se.classify_frame(np.ascontiguousarray(rgb[..., ::-1]))
        im = Image.fromarray(rgb)
        di = ImageDraw.Draw(im)
        di.rectangle([se.SEAT_X[0], se.SEAT_Y[0], se.SEAT_X[1], se.SEAT_Y[1]], outline=(0, 255, 0), width=3)
        if cap:
            di.ellipse([cap[0] - 9, cap[1] - 9, cap[0] + 9, cap[1] + 9], outline=(255, 0, 255), width=4)
        im = im.crop(crop)
        di = ImageDraw.Draw(im)
        di.rectangle([0, 0, cw, 22], fill=(0, 0, 0))
        di.text((5, 3), f"run {k + 1}: {a:.2f}-{b:.2f} s ({dur:.2f} s), frame t={w:.2f}",
                font=_font(14), fill=(255, 255, 255))
        sheet.paste(im, ((k % 3) * cw, head + (k // 3) * ch))
    return sheet


# --- EDL --------------------------------------------------------------------------------------

NOTES = """\
## Notes the cut depends on

1. **B2V-A-02 never seats on video - the "six seat runs" are not seats.** `score_episode.py`
   counts a run whenever the blue-cap centroid sits inside its seat box (470<x<640, 400<y<540,
   `tools/score_episode.py:37-38`), and its own docstring warns that a cap merely passing over
   the funnel scores (`tools/score_episode.py:9-11`). The recording shows (evidence sheet
   `B2V-A-02_seat_runs_evidence.png`, middle frame of every run):
   - runs 1-3 (13.80-15.20 s, including the 0.60 s "longest"): the tube falling out of the jaws
     and lying flat on the mat just image-above the rack, cap inside the box;
   - runs 4-6 (19.95-20.00 s, 34.30-34.80 s): the **operator's hand** carrying the tube through
     the box during two mid-episode resets (hand in frame ~19-24 s and ~29.5-38 s).
   The tube tip reached the funnel rim twice (~7.5-8.0 s and ~27.2 s) and did not enter. So
   `the results draft (private):95-96` ("A reaches the seat and cannot hold it ... six separate
   seat runs, longest 0.6 s"), `Eval Design — DAgger single-arm before-after.md:39`,
   `the Sep 2 assessment note (private):48-49` and
   `Protocol — pi0.5 first hardware session.md:153` describe the instrument's count, not the
   event. The honest line is "A reaches the funnel and cannot insert"; C05's caption says
   "Reaches the rack, can't get it in". The final MISS verdict stands. Not edited in those
   files - flagged.
2. **The v2 baseline is a null and this video makes no A/B point.** A 6/20 vs B 5/20, stopped
   at 40 trials (v2 sheet :69-78). Arm labels are banned on screen; the source map above records
   them. Both seated clips happen to be ACT-B and all three failures ACT-A because C01 was chosen
   by the data search in note 3 and the failures by mechanism family; at this n it means nothing,
   and the end card's rate pools both arms (11 of 40).
3. **How C01 was chosen (Sep 22).** Faith asked for one run with no missed grip and no correction
   at the rack. All 45 unblinded episodes were scanned with this module's detectors: 12 seat on
   video (the 11 sheet S rows and the Aug 22 probe's ep 0). Five have one commanded close before
   the seat and the minimal three lift turning points (trials 6, 19, 23, 32, 38); the other seven
   show a re-aim above the funnel (five turning points: the Aug 22 probe episode that was the old
   C01, and trials 5, 20, 22) or repeated grabs (trials 14, 27, 28). Only trial 32's scorer note
   records nothing at the rack. In every seated run the second commanded close is the jaws
   shutting on the way home after release, so "2 closes" in the events table is not two grabs.
4. **Blinded Sep 6 bench recordings are excluded** until the key is unsealed; the source
   whitelist in `tools/demo_cut.py` refuses them.
5. **Timebase.** In/out are video time (frame/20). The rollout loop ran ~19.3 Hz
   (`the lab notebook (private):242-244`), so the cut plays ~3% fast against wall clock. The
   cards are 20 fps frames too.
6. **On-screen words.** Captions are one line of at most eight plain words in the same shape;
   dates, dataset versions, trial numbers, arm labels, run ids and operator resets are banned on
   screen and live in this file instead (`tests/test_demo_cut.py`, section 5).
"""


def _fmt(t):
    return "-" if t is None else f"{t:.2f}"


def render_edl(plan_rows, events, sheet_rows):
    by_id = {c.clip_id: c for c in CLIPS}
    total = total_runtime([(p["in_s"], p["out_s"]) for p in plan_rows])
    L = []
    L.append("# Demo cut - edit decision list\n")
    L.append("Generated by `python tools/demo_cut.py` from archived rollouts in "
             "`~/.cache/huggingface/lerobot/faithqin/` (no bench). Every in/out point is an "
             "event from the episode's own data plus a declared offset; events are in "
             "`events.json` beside this file and re-checked by `tests/test_demo_cut.py`. "
             "Story: it works -> it recovers -> how it fails, between a title card and an end card. "
             f"**Total runtime {total + cards_runtime():.2f} s**: {total:.2f} s of clips over "
             f"{len(plan_rows)} clips plus {cards_runtime():.0f} s of cards, front camera, real time "
             "(20 fps source, 20 fps cut; see note 5).\n")
    L.append("## Source map\n")
    L.append("| clip | record | policy | run id | rollout directory | sheet grasp-close | recomputed |")
    L.append("|---|---|---|---|---|---|---|")
    for p in plan_rows:
        c, ev = by_id[p["clip_id"]], events[p["clip_id"]]
        if c.sheet_trial:
            r = sheet_rows[c.sheet_trial]
            mech = "seated" if r["mechanism"] in ("—", "-") else r["mechanism"]
            rec = f"v2 trial {c.sheet_trial} (`{V2_SHEET}:{r['line']}`), {mech}"
            pol, sheet_gc = r["policy"], f"{r['grasp_close_s']:.1f} s"
        else:
            rec, pol, sheet_gc = c.label, c.policy, "n/a"
        L.append(f"| {c.clip_id} | {rec} | {pol} | `{ev['run_id']}` | `{ev['rollout_dir']}` ep {c.episode} "
                 f"| {sheet_gc} | {_fmt(ev['grasp_close'])} s |")
    L.append("\nPolicy for the non-sheet source: B2V-A-02 is ACT-A on v3 "
             "(`analysis/figures/numbers.json:320-324`, `the Sep 2 assessment note (private):48`).\n")
    L.append("## Edit decision list\n")
    L.append("| # | section | source | cam | in (s) | out (s) | dur (s) | anchors | what happens | why it belongs | caption |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for p in plan_rows:
        c, ev = by_id[p["clip_id"]], events[p["clip_id"]]
        anchors = (f"in = {c.in_at[0]} {c.in_at[1]:+.1f} ({_fmt(ev[c.in_at[0]])}); "
                   f"out = {c.out_at[0]} {c.out_at[1]:+.1f} ({_fmt(ev[c.out_at[0]])})")
        cap = c.caption.replace("\n", " ")
        L.append(f"| {c.clip_id} | {SECTION_TITLE[c.section]} | `{ev['rollout_dir']}` ep {c.episode} | "
                 f"{c.camera} | {p['in_s']:.2f} | {p['out_s']:.2f} | {p['dur_s']:.2f} | {anchors} | "
                 f"{c.happens} | {c.why} | {cap} |")
    L.append(f"\n**Clips: {total:.2f} s.** Contact sheets: `<clip>_<run id>.png` in this folder (work-area "
             "crop, tiles labelled only with the events they sit on). Evidence for note 1: "
             "`B2V-A-02_seat_runs_evidence.png`. Cut: `demo_cut.mp4` (H.264 via PyAV, "
             "1280x720, 20 fps, captions burned in; mp4s are gitignored, `.gitignore:42`).\n")
    L.append("## Cards\n")
    L.append("| card | seconds | text |")
    L.append("|---|---|---|")
    L.append(f"| title | {TITLE_CARD.seconds:.0f} | {' / '.join(TITLE_CARD.lines)} |")
    L.append(f"| end | {END_CARD.seconds:.0f} | {' / '.join(END_CARD.lines)} |")
    L.append("")
    L.append("## Events per source (seconds, video time)\n")
    L.append("| clip | grasp-close (cmd) | closes | v4_audit.grasp_index (jaw <=16 at pick site; not a hold) | pick exit | rack entry | "
             "seat runs (start-end, dur) | held seat from | cap displaced | cap lost after rack | "
             "home return | lift reversals >6 deg | final |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in plan_rows:
        ev = events[p["clip_id"]]
        runs = "; ".join(f"{a:.2f}-{b:.2f} ({d:.2f})" for a, b, d in ev["seat_runs"]) or "none"
        L.append(f"| {p['clip_id']} | {_fmt(ev['grasp_close'])} | {ev['n_closes']} | {_fmt(ev['v4_grasp_index'])} | "
                 f"{_fmt(ev['pick_exit'])} | {_fmt(ev['rack_entry'])} | {runs} | {_fmt(ev['seat_hold_start'])} | "
                 f"{_fmt(ev['cap_displaced'])} | {_fmt(ev['cap_lost_after_rack'])} | {_fmt(ev['home_return'])} | "
                 f"{ev['lift_reversals']} | {ev['final']} |")
    L.append("\nDefinitions: grasp-close and home return are the v2 sheet's forensic columns "
             f"(`{V2_SHEET}:81-91`); pick exit = measured pan crossing {PICK_SITE_PAN:.0f} deg; rack entry = "
             "first frame of `v4_audit.rack_slice`; seat runs and final = `score_episode.score_frames`; "
             "held seat from = onset of the seat run that lasts to the final frame (the cap enters the "
             "seat box while still in the jaws; the release follows); "
             "cap displaced = first sustained (5 detections) move >8 px from the opening cap position; "
             "cap lost = first >=1 s run with no cap detection after rack entry.\n")
    L.append(NOTES)
    return "\n".join(L) + "\n"


# --- main -------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--no-video", action="store_true", help="skip the mp4")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cache = {}
    events = {}
    for c in CLIPS:
        key = (clip_run_id(c), c.episode)
        if key not in cache:
            print(f"[demo_cut] analysing {key[0]} ep {key[1]}", flush=True)
            cache[key] = analyze(c)
        events[c.clip_id] = cache[key]
    (out / "events.json").write_text(json.dumps(events, indent=2) + "\n")
    rows = plan(events)

    encoder = None if args.no_video else pick_h264_encoder()
    writer = stream = None
    if encoder:
        import av
        writer = av.open(str(out / "demo_cut.mp4"), mode="w")
        stream = writer.add_stream(encoder, rate=FPS)
        stream.width, stream.height, stream.pix_fmt = 1280, 720, "yuv420p"
        if encoder == "libx264":
            stream.options = {"crf": "20", "preset": "medium"}
    frames_written = 0

    def write_frame(img, times=1):
        nonlocal frames_written
        if writer is None:
            return
        if img.shape[:2] != (720, 1280):
            raise RuntimeError(f"frame {img.shape} is not 1280x720")
        for _ in range(times):
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                writer.mux(packet)
            frames_written += 1

    write_frame(card_frame(TITLE_CARD.lines), card_frame_count(TITLE_CARD.seconds))
    by_id = {c.clip_id: c for c in CLIPS}
    for p in rows:
        c, ev = by_id[p["clip_id"]], events[p["clip_id"]]
        d = resolve_rollout(ev["run_id"])
        marks = _event_marks(ev, c)
        n_tiles = 12 if p["dur_s"] > 8.0 else 8
        want = set(sample_times(p["in_s"], p["out_s"], [m for m, _ in marks], n=n_tiles))
        samples = []
        for t, rgb in iter_frames(d, c.episode, c.camera, p["in_s"], p["out_s"]):
            if t in want:
                samples.append((t, rgb))
            if writer is not None:
                write_frame(caption_frame(rgb, SECTION_TITLE[c.section], c.caption))
        title = (f"{c.clip_id}  {SECTION_TITLE[c.section]}  |  {c.label}  |  {ev['rollout_dir']} ep {c.episode} "
                 f"{c.camera}  |  in {p['in_s']:.2f} s -> out {p['out_s']:.2f} s")
        contact_sheet(samples, title, marks).save(out / f"{c.clip_id}_{ev['run_id']}.png", optimize=True)
        print(f"[demo_cut] {c.clip_id} {p['in_s']:.2f}-{p['out_s']:.2f} s ({p['dur_s']:.2f} s)", flush=True)
    write_frame(card_frame(END_CARD.lines), card_frame_count(END_CARD.seconds))
    if writer is not None:
        for packet in stream.encode():
            writer.mux(packet)
        writer.close()

    b2v = next(ev for ev in events.values() if ev["run_id"] == FUNNEL_RUN_ID)
    seat_run_evidence(resolve_rollout(b2v["run_id"]), 0, b2v["seat_runs"]).save(
        out / "B2V-A-02_seat_runs_evidence.png", optimize=True)
    (out / "EDL.md").write_text(render_edl(rows, events, _project_sheet()))
    total = total_runtime([(p["in_s"], p["out_s"]) for p in rows])
    print(f"[demo_cut] clips {total:.2f} s + cards {cards_runtime():.0f} s = {total + cards_runtime():.2f} s; "
          f"encoder={encoder}; frames written={frames_written} ({frames_written / FPS:.2f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
