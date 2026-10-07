#!/usr/bin/env python
"""Cut the write-up assets for the Sep 6 policy trials from their preserved rollouts.

    python tools/golden_trial_assets.py                    # both trials
    python tools/golden_trial_assets.py --token T0-SMOKE   # one

INPUTS (read-only)
  analysis/bench_sep6/rollouts/<rollout>/   the preserved LeRobot rollout datasets, verified
                                            against analysis/bench_sep6/rollouts/SHA256SUMS
                                            before anything is cut (a mismatch aborts)
  tools/scored_logs/<trial>.score           the scorer's verdict -- the seat time comes from here
  tools/scored_logs/<trial>_telemetry.csv   only its row count is used (frame reconciliation)

OUTPUTS (analysis/bench_sep6/clips/)
  <label>_front_wrist.mp4      front | wrist side by side, every frame, real time (20 fps),
                               <= 1280 px wide, H.264 yuv420p + faststart so QuickTime plays it
                               (gitignored: analysis/**/*.mp4)
  <label>_contact_sheet.png    8 front frames evenly spaced 0 s -> seat + 2 s, each stamped with
                               the time of the frame actually shown
  <label>_joints.png           observation.state dims 0-5 vs time, one panel per joint, with the
  <label>_joints_dark.png      first close, re-grasp, rack entry and seat marked; light and dark
  <label>_recovery_strip.png   first close / re-grasp / rack entry (trials with recovery_strip)

BLINDING. P01 trial 1 (token h83j2kp6) belongs to a sealed arm. Its caption is fixed as
"P01 trial 1 (token h83j2kp6, arm sealed)", and every string this tool draws or names for a
token outside UNBLINDED passes `assert_blind_safe` first. The tool reads only joint POSITIONS
(state dims 0-5) and never reports the state's width. It never opens the scoring ledger or the
sealed directory (tests/test_golden_trial_assets.py checks that with ast).

DECODER. ffmpeg is not on PATH. PyAV (the lerobot env's `av`, which bundles libdav1d for the
AV1 source and libx264 for output) does all decoding and encoding.

UNITS. Body joints are degrees and the gripper is 0-100 % of its calibrated range:
lerobot so_follower.py:50 picks DEGREES when config.use_degrees (config_so_follower.py:42
defaults it True) and so_follower.py:59 gives the gripper RANGE_0_100.

Written Sep 12 2026. Tests: tests/test_golden_trial_assets.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4_audit  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
BENCH = PROJECT / "analysis" / "bench_sep6"
ROLLOUTS = BENCH / "rollouts"
CLIPS = BENCH / "clips"
SCORED = PROJECT / "tools" / "scored_logs"
MANIFEST = "SHA256SUMS"
MAX_WIDTH = 1280

# grasp_index's own "closed" threshold is the literal 16.0 at tools/v4_audit.py:159.
GRASP_CLOSED = 16.0

JOINTS = (
    ("shoulder_pan", "shoulder pan", "degrees"),
    ("shoulder_lift", "shoulder lift", "degrees"),
    ("elbow_flex", "elbow flex", "degrees"),
    ("wrist_flex", "wrist flex", "degrees"),
    ("wrist_roll", "wrist roll", "degrees"),
    ("gripper", "gripper", "% of range"),
)


# --- the trials ---------------------------------------------------------------

@dataclass(frozen=True)
class Trial:
    token: str
    label: str            # filename stem
    caption: str
    rollout: str          # directory under ROLLOUTS
    score: str            # file under SCORED
    telemetry: str        # file under SCORED
    recovery_strip: bool = False
    strip_notes: tuple = ()      # one line per strip panel, written after viewing the frames
    strip_crops: tuple = ()      # (x0, y0, x1, y1) per panel in front-camera pixels; display only


TRIALS = (
    Trial(
        token="h83j2kp6",
        label="h83j2kp6",
        caption="P01 trial 1 (token h83j2kp6, arm sealed)",
        rollout="rollout_h83j2kp6_20260906_161433",
        score="h83j2kp6_20260906_161402.score",
        telemetry="h83j2kp6_20260906_161402_telemetry.csv",
        recovery_strip=True,
        # Verified against the decoded front frames on Sep 12 (native-resolution crops): at
        # frames 74-90 the red jaw pad sits LEFT of the blue cap, the cap fully visible beside
        # it; frame 100 (5.00 s) shows the gripper lifted with the tube still on the table; at
        # frames 159-180 the pad has moved right and overlaps the cap (the far jaw is not
        # visible from this camera); frame 208 shows the tube tip at the rack hole. The bench
        # notes are Faith's, from the trial itself.
        strip_notes=(
            "Jaws shut to the left of the cap; the lift that follows leaves the tube on the "
            "table (5.00 s). Bench note: grasp too far left.",
            "Jaws reopened and closed again further right, the jaw pad now over the cap. "
            "Bench note: poor hold, "
            "the tube almost dangling through transport.",
            "Start of the rack phase (v4_audit.rack_slice): tube tip at the hole. "
            "Scorer: seated 11.7 s, held to end.",
        ),
        # display crops (front-camera pixels): the jaws and cap for 1-2, tube and rack for 3
        strip_crops=((210, 180, 594, 396), (210, 180, 594, 396), (300, 200, 940, 560)),
    ),
    Trial(
        token="T0-SMOKE",
        label="T0-SMOKE",
        caption="T0-SMOKE smoke trial (ACT-A-v4)",
        rollout="rollout_T0-SMOKE_20260906_145318",
        score="T0-SMOKE_20260906_145245.score",
        telemetry="T0-SMOKE_20260906_145245_telemetry.csv",
    ),
)

UNBLINDED = frozenset({"T0-SMOKE"})

_ARM_WORDS = re.compile(r"\bACT\b|\b[AB]-v\d|(?i:noload|\bload\b|\barm [AB]\b|polic|smolvla|pi0)")


def trial(token: str) -> Trial:
    for t in TRIALS:
        if t.token == token:
            return t
    raise KeyError(token)


def assert_blind_safe(t: Trial, text: str) -> None:
    """Refuse any arm/policy word in text drawn or named for a sealed trial."""
    if t.token in UNBLINDED:
        return
    if _ARM_WORDS.search(text):
        raise ValueError(f"refusing to draw arm/policy text for sealed token {t.token}: {text!r}")


def output_names(t: Trial) -> dict:
    names = {
        "video": f"{t.label}_front_wrist.mp4",
        "sheet": f"{t.label}_contact_sheet.png",
        "joints": f"{t.label}_joints.png",
        "joints_dark": f"{t.label}_joints_dark.png",
    }
    if t.recovery_strip:
        names["strip"] = f"{t.label}_recovery_strip.png"
    return names


def asset_texts(t: Trial, lm: dict, seat_s: float, x_end_s: float, settle: float,
                duration_s: float, fps: int = v4_audit.FPS) -> dict:
    """Every string any image for this trial will carry."""
    def at(i):
        return f"{i / fps:.2f} s"

    rules = {}
    if lm.get("first_close") is not None:
        rules["first_close"] = f"first close {at(lm['first_close'])}"
    if lm.get("regrasp") is not None:
        rules["regrasp"] = f"re-grasp {at(lm['regrasp'])}"
    if lm.get("rack_entry") is not None:
        rules["rack_entry"] = f"rack entry {at(lm['rack_entry'])}"
    rules["seat"] = f"seated {seat_s:.1f} s (scorer)"

    panels = []
    for key, name in (("first_close", "first close"), ("regrasp", "re-grasp"),
                      ("rack_entry", "rack entry")):
        if lm.get(key) is not None:
            panels.append(f"{len(panels) + 1} · {name} · {at(lm[key])}")

    return {
        "sheet_title": t.caption,
        "sheet_subtitle": (f"Front camera · 8 frames evenly spaced from 0 s to seat + 2 s "
                           f"({seat_s + 2:.1f} s) · scorer seat {seat_s:.1f} s · "
                           f"each stamp is the time of the frame shown"),
        "title": f"{t.caption} · joint positions",
        "subtitle": (f"observation.state dims 0-5 at {fps} Hz · 0-{x_end_s:.1f} s of "
                     f"{duration_s:.1f} s shown; after {x_end_s:.1f} s every joint stays within "
                     f"{settle:.2f} of its final value · degrees, gripper in % of range"),
        "x_label": "time since episode start (s)",
        "rules": rules,
        "strip_title": f"{t.caption} · the recovery",
        "strip_subtitle": ("Front camera, cropped and scaled to equal width. "
                           "First close = v4_audit.grasp_index; "
                           "re-grasp = the next pick-site close after the jaws reopened; "
                           "rack entry = first frame of v4_audit.rack_slice."),
        "strip_panels": panels,
        "strip_notes": list(t.strip_notes),
    }


# --- ground truth ---------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(root) -> list:
    """Problems between root/SHA256SUMS and the files under root; [] when intact."""
    root = Path(root)
    listed = {}
    for line in (root / MANIFEST).read_text().splitlines():
        if line.strip():
            digest, rel = line.split(None, 1)
            listed[rel.strip()] = digest
    problems = []
    for rel, digest in sorted(listed.items()):
        p = root / rel
        if not p.is_file():
            problems.append(f"MISSING {rel}")
        elif _sha256(p) != digest:
            problems.append(f"CHANGED {rel}: sha256 no longer matches the manifest")
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name not in (MANIFEST, ".DS_Store"):
            rel = p.relative_to(root).as_posix()
            if rel not in listed:
                problems.append(f"UNLISTED {rel}")
    return problems


@dataclass(frozen=True)
class Score:
    frames: int | None
    duration_s: float | None
    seat_s: float | None
    verdict: str | None


def parse_score(text: str) -> Score:
    """Format: tools/score_episode.py:125 (frames/dur) and :130-131 (seated ... [tag])."""
    m = re.search(r"frames=(\d+)\s+dur=([\d.]+)s", text)
    held = re.findall(r"seated\s+([\d.]+)s\s*->\s*[\d.]+s\s*\([\d.]+s\)\s*\[holds to end\]", text)
    v = re.search(r"^VERDICT\s+(\S+)", text, re.M)
    return Score(frames=int(m.group(1)) if m else None,
                 duration_s=float(m.group(2)) if m else None,
                 seat_s=float(held[-1]) if held else None,
                 verdict=v.group(1) if v else None)


def rollout_fps(root: Path) -> int:
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = int(info["fps"])
    for cam in ("front", "wrist"):
        vinfo = info["features"][f"observation.images.{cam}"]["info"]
        if int(vinfo["video.fps"]) != fps:
            raise ValueError(f"{cam} video fps {vinfo['video.fps']} != dataset fps {fps}")
    return fps


def load_positions(root: Path) -> np.ndarray:
    """Joint positions only (state dims 0-5). Nothing wider leaves this function."""
    import pandas as pd

    files = sorted((root / "data").rglob("*.parquet"))
    df = pd.concat(pd.read_parquet(f, columns=["frame_index", "observation.state"]) for f in files)
    df = df.sort_values("frame_index")
    return np.stack(df["observation.state"].to_numpy())[:, :6].astype(float)


def video_path(root: Path, cam: str) -> Path:
    return root / "videos" / f"observation.images.{cam}" / "chunk-000" / "file-000.mp4"


def count_decoded_frames(path) -> int:
    import av

    with av.open(str(path)) as c:
        return sum(1 for _ in c.decode(c.streams.video[0]))


def frame_counts(t: Trial) -> dict:
    import pandas as pd

    root = ROLLOUTS / t.rollout
    return {
        "front video": count_decoded_frames(video_path(root, "front")),
        "wrist video": count_decoded_frames(video_path(root, "wrist")),
        "parquet rows": int(sum(len(pd.read_parquet(f, columns=["frame_index"]))
                                for f in sorted((root / "data").rglob("*.parquet")))),
        "telemetry csv rows": int(len(pd.read_csv(SCORED / t.telemetry, usecols=["frame_index"]))),
    }


def reconcile_counts(counts: dict) -> list:
    if len(set(counts.values())) <= 1:
        return []
    majority, _ = Counter(counts.values()).most_common(1)[0]
    agree = [k for k, v in counts.items() if v == majority]
    return [f"{k}: {v} frames, but {', '.join(agree)} say {majority}"
            for k, v in counts.items() if v != majority]


# --- landmarks ----------------------------------------------------------------

def pick_closes(pan, grip) -> list:
    """Every pick-site close of a jaw that had opened, with grasp_index's thresholds.

    The first is v4_audit.grasp_index. Unlike v4_audit.jaw_reopen_cycles, a reopen counts
    whatever the close bottomed at -- P01 trial 1's first close only reached 14.3, above
    JAW_SHUT, so the audit's cycle counter never saw its re-grasp.
    """
    pan = np.asarray(pan, dtype=float)
    grip = np.asarray(grip, dtype=float)
    closes, opened = [], False
    for i in range(len(grip)):
        at_site = pan[i] > v4_audit.PICK_SITE_PAN
        if at_site and grip[i] > v4_audit.JAW_OPEN:
            opened = True
        if opened and at_site and grip[i] <= GRASP_CLOSED:
            closes.append(i)
            opened = False
    return closes


def landmarks(state) -> dict:
    st = np.asarray(state, dtype=float)[:, :6]
    pan, wflex, grip = st[:, 0], st[:, 3], st[:, 5]
    closes = pick_closes(pan, grip)
    rack = v4_audit.rack_slice(pan, wflex)
    rack_entry = int(rack[0]) if len(rack) else None
    before = [c for c in closes if rack_entry is None or c < rack_entry]
    return {
        "first_close": closes[0] if closes else None,
        "regrasp": before[-1] if len(before) >= 2 else None,
        "rack_entry": rack_entry,
    }


# --- timing and geometry ----------------------------------------------------------

def contact_sheet_times(seat_s, last_t_s, n=8, pad_s=2.0) -> list:
    if seat_s is None:
        raise ValueError("no held seat in the score: the contact sheet window is undefined")
    end = min(seat_s + pad_s, last_t_s)
    return [end * k / (n - 1) for k in range(n)]


def frame_at(t_s, fps, n_frames) -> int:
    return int(min(max(math.floor(t_s * fps + 0.5), 0), n_frames - 1))


def _even(v) -> int:
    v = int(math.floor(v))
    return v - v % 2


def side_by_side_geometry(front_wh, wrist_wh, max_width=MAX_WIDTH):
    """(common height, front width, wrist width): all even, summing to <= max_width."""
    (fw, fh), (ww, wh) = front_wh, wrist_wh
    h = _even(min(max_width / (fw / fh + ww / wh), fh, wh))
    while True:
        a, b = _even(h * fw / fh), _even(h * ww / wh)
        if a + b <= max_width:
            return h, a, b
        h -= 2


def choose_encoder(available) -> str:
    """H.264 first (QuickTime's native codec); MPEG-4 part 2 also plays; nothing else does."""
    for name in ("libx264", "h264_videotoolbox", "mpeg4"):
        if name in available:
            return name
    raise RuntimeError("no QuickTime-playable encoder in this PyAV build")


def available_encoders() -> set:
    from av.codec.codec import Codec

    found = set()
    for name in ("libx264", "h264_videotoolbox", "mpeg4"):
        try:
            Codec(name, "w")
            found.add(name)
        except Exception:  # noqa: BLE001
            pass
    return found


def place_labels(items, gap=6) -> list:
    """Row index per (x, width) label so no two labels in a row come within `gap`."""
    order = sorted(range(len(items)), key=lambda i: items[i][0])
    ends, rows = [], [0] * len(items)
    for i in order:
        x, w = items[i]
        for r, end in enumerate(ends):
            if x >= end + gap:
                rows[i], ends[r] = r, x + w
                break
        else:
            rows[i] = len(ends)
            ends.append(x + w)
    return rows


# --- drawing helpers ----------------------------------------------------------

# Reference palette (dataviz skill references/palette.md): chart surfaces, ink, hairlines,
# categorical slot 1 for the one series. validate_palette.js: ALL CHECKS PASS, both modes.
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "series": "#2a78d6"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "series": "#3987e5"},
}

_FONT_FILES = ("/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/Helvetica.ttc")


def hex_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    for path in _FONT_FILES:
        try:
            return ImageFont.truetype(path, size, index=1 if bold else 0)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _wrap(text: str, font, width: int) -> list:
    lines, cur = [], ""
    for word in text.split():
        trial_line = f"{cur} {word}".strip()
        if font.getlength(trial_line) <= width or not cur:
            cur = trial_line
        else:
            lines.append(cur)
            cur = word
    return lines + ([cur] if cur else [])


def decode_frames(path, indices) -> dict:
    """{index: PIL.Image} for the requested frame indices (decoded in order)."""
    import av

    want, got = set(int(i) for i in indices), {}
    with av.open(str(path)) as c:
        for i, fr in enumerate(c.decode(c.streams.video[0])):
            if i in want:
                got[i] = fr.to_image()
            if len(got) == len(want):
                break
    missing = want - set(got)
    if missing:
        raise ValueError(f"{path}: frames {sorted(missing)} not in the video")
    return got


def _save_png(im, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".partial.png")
    im.save(tmp, optimize=True)
    tmp.replace(out)


# --- renderers ----------------------------------------------------------------

def render_side_by_side(front, wrist, out, fps, max_width=MAX_WIDTH, encoder="libx264") -> dict:
    import av

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".partial.mp4")
    with av.open(str(front)) as fc, av.open(str(wrist)) as wc:
        fs, ws = fc.streams.video[0], wc.streams.video[0]
        h, fw, ww = side_by_side_geometry((fs.width, fs.height), (ws.width, ws.height), max_width)
        with av.open(str(tmp), "w", format="mp4", container_options={"movflags": "+faststart"}) as oc:
            s = oc.add_stream(encoder, rate=fps)
            s.width, s.height, s.pix_fmt = fw + ww, h, "yuv420p"
            s.time_base = Fraction(1, fps)
            if encoder == "libx264":
                s.options = {"crf": "18", "preset": "medium", "profile": "high"}
            else:
                s.bit_rate = 8_000_000
            n = 0
            for f_fr, w_fr in zip(fc.decode(fs), wc.decode(ws)):
                a = f_fr.reformat(width=fw, height=h, format="rgb24", interpolation="AREA").to_ndarray()
                b = w_fr.reformat(width=ww, height=h, format="rgb24", interpolation="AREA").to_ndarray()
                vf = av.VideoFrame.from_ndarray(np.hstack([a, b]), format="rgb24").reformat(format="yuv420p")
                vf.pts, vf.time_base = n, Fraction(1, fps)
                for p in s.encode(vf):
                    oc.mux(p)
                n += 1
            for p in s.encode():
                oc.mux(p)
    tmp.replace(out)
    return {"frames": n, "width": fw + ww, "height": h, "front_width": fw, "wrist_width": ww,
            "encoder": encoder}


def render_contact_sheet(video, times, fps, n_frames, out, title, subtitle, tile_width=440,
                         cols=4, theme="light") -> list:
    from PIL import Image, ImageDraw

    th = THEMES[theme]
    idx = [frame_at(t, fps, n_frames) for t in times]
    frames = decode_frames(video, idx)
    src_w, src_h = next(iter(frames.values())).size
    tile_h = round(tile_width * src_h / src_w)
    pad, gap = max(12, tile_width // 14), max(4, tile_width // 55)
    f_title, f_sub = _font(max(12, tile_width // 18), bold=True), _font(max(10, tile_width // 27))
    f_stamp = _font(max(10, tile_width // 22), bold=True)
    rows = math.ceil(len(idx) / cols)
    grid_w = cols * tile_width + (cols - 1) * gap
    sub_lines = _wrap(subtitle, f_sub, grid_w)
    head = f_title.size + 8 + len(sub_lines) * (f_sub.size + 4) + pad // 2
    W = grid_w + 2 * pad
    H = pad + head + rows * tile_h + (rows - 1) * gap + pad
    im = Image.new("RGB", (W, H), hex_rgb(th["surface"]))
    d = ImageDraw.Draw(im)
    d.text((pad, pad), title, font=f_title, fill=hex_rgb(th["ink"]))
    y = pad + f_title.size + 8
    for line in sub_lines:
        d.text((pad, y), line, font=f_sub, fill=hex_rgb(th["ink2"]))
        y += f_sub.size + 4
    tiles = []
    for k, i in enumerate(idx):
        r, c = divmod(k, cols)
        x0 = pad + c * (tile_width + gap)
        y0 = pad + head + r * (tile_h + gap)
        im.paste(frames[i].resize((tile_width, tile_h), Image.LANCZOS), (x0, y0))
        stamp = f"{i / fps:.2f} s"
        sw, sh = f_stamp.getlength(stamp), f_stamp.size
        m = max(3, tile_width // 80)
        d.rectangle((x0 + m, y0 + m, x0 + m + sw + 2 * m, y0 + m + sh + 2 * m), fill=(11, 11, 11))
        d.text((x0 + 2 * m, y0 + 1.5 * m), stamp, font=f_stamp, fill=(255, 255, 255))
        tiles.append({"index": i, "time_s": i / fps, "stamp": stamp,
                      "box": (x0, y0, x0 + tile_width, y0 + tile_h)})
    _save_png(im, Path(out))
    return tiles


def _nice_step(raw: float) -> float:
    """The 1-2-5 step closest (in ratio) to raw."""
    mag = 10 ** math.floor(math.log10(raw))
    cands = [s * m for m in (mag / 10, mag, mag * 10) for s in (1, 2, 5)]
    return min(cands, key=lambda c: abs(math.log(c / raw)))


def _axis(lo: float, hi: float, target: int = 3, pad_frac: float = 0.08):
    """Padded domain around the data, and the 1-2-5 ticks inside it."""
    if hi - lo < 1e-9:
        lo, hi = lo - 1, hi + 1
    pad = (hi - lo) * pad_frac
    a, b = lo - pad, hi + pad
    step = _nice_step((b - a) / target)
    ticks = [k * step for k in range(math.ceil(a / step - 1e-9), math.floor(b / step + 1e-9) + 1)]
    return a, b, ticks


def _fmt_tick(v: float) -> str:
    return f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"


def render_joint_plot(state, fps, lm, seat_s, x_end_s, out, texts, theme="light", scale=2) -> dict:
    from PIL import Image, ImageDraw

    th = {k: hex_rgb(v) for k, v in THEMES[theme].items()}
    S = scale
    st = np.asarray(state, dtype=float)[:, :6]
    n_show = min(len(st), int(math.floor(x_end_s * fps)) + 1)
    t = np.arange(n_show) / fps

    W = 960 * S
    left, right = 150 * S, 40 * S
    f_title, f_sub = _font(16 * S, bold=True), _font(12 * S)
    f_joint, f_tick, f_rule = _font(12 * S, bold=True), _font(11 * S), _font(11 * S)
    pad = 24 * S
    plot_l, plot_r = left, W - right

    def x_of(ts):
        return plot_l + ts / x_end_s * (plot_r - plot_l)

    rule_keys = [k for k in ("first_close", "regrasp", "rack_entry") if lm.get(k) is not None]
    rule_t = {k: lm[k] / fps for k in rule_keys}
    rule_t["seat"] = seat_s
    rules = {k: int(round(x_of(v))) for k, v in rule_t.items()}
    labels = [(k, texts["rules"].get(k, k)) for k in rules]
    items = [(rules[k] + 4 * S, int(f_rule.getlength(s))) for k, s in labels]
    rows = place_labels(items, gap=8 * S)
    n_rows = max(rows) + 1 if rows else 0

    sub_lines = _wrap(texts["subtitle"], f_sub, W - 2 * pad)
    y = pad
    title_y = y
    y += f_title.size + 8 * S
    sub_y = y
    y += len(sub_lines) * (f_sub.size + 4 * S) + 14 * S
    label_top = y
    row_h = f_rule.size + 6 * S
    y += n_rows * row_h + 4 * S
    plot_t = y
    panel_h, panel_gap = 88 * S, 30 * S
    panels = []
    for j in range(6):
        y0 = plot_t + j * (panel_h + panel_gap)
        panels.append((y0, y0 + panel_h))
    plot_b = panels[-1][1]
    H = plot_b + 56 * S

    im = Image.new("RGB", (W, H), th["surface"])
    d = ImageDraw.Draw(im)
    d.text((pad, title_y), texts["title"], font=f_title, fill=th["ink"])
    for k, line in enumerate(sub_lines):
        d.text((pad, sub_y + k * (f_sub.size + 4 * S)), line, font=f_sub, fill=th["ink2"])

    # 1) per panel: hairline grid at the y ticks, tick labels, joint name + unit
    traces, tick_boxes = [], []
    for j, (y0, y1) in enumerate(panels):
        vals = st[:n_show, j]
        lo, hi, ticks = _axis(float(vals.min()), float(vals.max()))

        def y_of(v, y0=y0, y1=y1, lo=lo, hi=hi):
            return y1 - (v - lo) / (hi - lo) * (y1 - y0)

        for tk in ticks:
            yy = int(round(y_of(tk)))
            d.line((plot_l, yy, plot_r, yy), fill=th["grid"], width=S)
            s = _fmt_tick(tk)
            pos = (plot_l - 8 * S - f_tick.getlength(s), yy - f_tick.size // 2 - S)
            d.text(pos, s, font=f_tick, fill=th["muted"])
            tick_boxes.append((j, d.textbbox(pos, s, font=f_tick)))
        _, name, unit = JOINTS[j]
        d.text((pad, y0 + panel_h // 2 - f_joint.size), name, font=f_joint, fill=th["ink"])
        d.text((pad, y0 + panel_h // 2 + 2 * S), unit, font=f_tick, fill=th["muted"])
        traces.append([(x_of(tt), y_of(v)) for tt, v in zip(t, vals)])

    # 2) landmark rules, from their label row down through every panel (over grid, under traces)
    label_boxes = []
    for (k, s), (lx, lw), r in zip(labels, items, rows):
        ly = label_top + r * row_h
        seat = k == "seat"
        d.line((rules[k], ly, rules[k], plot_b), fill=th["ink"] if seat else th["ink2"],
               width=(3 if seat else 2) * S // 2)
        d.text((lx, ly), s, font=f_rule, fill=th["ink"])
        label_boxes.append((lx, ly, lx + lw, ly + f_rule.size))

    # 3) the traces
    for pts in traces:
        if len(pts) > 1:
            d.line(pts, fill=th["series"], width=2 * S, joint="curve")

    # x axis: one hairline under the last panel, 1-2-5 ticks in seconds
    d.line((plot_l, plot_b, plot_r, plot_b), fill=th["axis"], width=S)
    step = _nice_step(x_end_s / 7)
    for k in range(int(math.floor(x_end_s / step + 1e-9)) + 1):
        tk = k * step
        xx = int(round(x_of(tk)))
        d.line((xx, plot_b, xx, plot_b + 5 * S), fill=th["axis"], width=S)
        s = _fmt_tick(tk)
        d.text((xx - f_tick.getlength(s) / 2, plot_b + 8 * S), s, font=f_tick, fill=th["muted"])
    xl = texts.get("x_label", "time (s)")
    d.text(((plot_l + plot_r) / 2 - f_tick.getlength(xl) / 2, plot_b + 28 * S), xl,
           font=f_tick, fill=th["ink2"])

    _save_png(im, Path(out))
    return {"size": (W, H), "plot_box": (plot_l, plot_t, plot_r, plot_b), "panels": panels,
            "rules": rules, "label_boxes": label_boxes, "tick_boxes": tick_boxes, "scale": S}


def render_recovery_strip(video, lm, fps, out, texts, crops=(), panel_width=640,
                          theme="light") -> list:
    from PIL import Image, ImageDraw

    th = {k: hex_rgb(v) for k, v in THEMES[theme].items()}
    keys = [k for k in ("first_close", "regrasp", "rack_entry") if lm.get(k) is not None]
    if len(keys) != 3:
        raise ValueError(f"recovery strip needs first close, re-grasp and rack entry; got {keys}")
    frames = decode_frames(video, [lm[k] for k in keys])
    pad, gap = 32, 16
    f_title, f_sub = _font(26, bold=True), _font(17)
    f_panel, f_note = _font(19, bold=True), _font(16)
    imgs = []
    for j, k in enumerate(keys):
        fr = frames[lm[k]]
        if j < len(crops) and crops[j]:
            fr = fr.crop(crops[j])
        imgs.append(fr.resize((panel_width, round(panel_width * fr.size[1] / fr.size[0])),
                              Image.LANCZOS))
    ph = max(i.size[1] for i in imgs)
    W = 2 * pad + 3 * panel_width + 2 * gap
    sub_lines = _wrap(texts["strip_subtitle"], f_sub, W - 2 * pad)
    notes = texts.get("strip_notes") or [""] * 3
    note_lines = [_wrap(n, f_note, panel_width) for n in notes]
    head = f_title.size + 10 + len(sub_lines) * (f_sub.size + 5) + 18
    foot = 12 + f_panel.size + 8 + max(len(n) for n in note_lines) * (f_note.size + 5) + pad
    H = pad + head + ph + foot
    im = Image.new("RGB", (W, H), th["surface"])
    d = ImageDraw.Draw(im)
    d.text((pad, pad), texts["strip_title"], font=f_title, fill=th["ink"])
    y = pad + f_title.size + 10
    for line in sub_lines:
        d.text((pad, y), line, font=f_sub, fill=th["ink2"])
        y += f_sub.size + 5
    boxes = []
    for j, img in enumerate(imgs):
        x0, y0 = pad + j * (panel_width + gap), pad + head
        im.paste(img, (x0, y0))
        ty = y0 + ph + 12
        d.text((x0, ty), texts["strip_panels"][j], font=f_panel, fill=th["ink"])
        ty += f_panel.size + 8
        for line in note_lines[j]:
            d.text((x0, ty), line, font=f_note, fill=th["ink2"])
            ty += f_note.size + 5
        boxes.append({"key": keys[j], "index": lm[keys[j]], "box": (x0, y0, x0 + img.size[0], y0 + img.size[1])})
    _save_png(im, Path(out))
    return boxes


# --- the run --------------------------------------------------------------------

def run(t: Trial, clips: Path = CLIPS, encoder: str | None = None) -> dict:
    report = {"token": t.token, "caption": t.caption, "problems": []}
    root = ROLLOUTS / t.rollout
    fps = rollout_fps(root)
    score = parse_score((SCORED / t.score).read_text())
    if score.seat_s is None:
        raise ValueError(f"{t.score}: no seat that holds to end -- nothing to cut against")
    st = load_positions(root)
    counts = frame_counts(t)
    report["counts"] = counts
    report["problems"] += reconcile_counts(counts)
    if score.frames is not None and score.frames not in counts.values():
        report["problems"].append(f"scorer says frames={score.frames}, sources say {counts}")
    report["score"] = score

    lm = landmarks(st)
    report["landmarks"] = {k: (v, None if v is None else v / fps) for k, v in lm.items()}
    last_t = (len(st) - 1) / fps
    x_end = min(last_t, score.seat_s + 4.0)
    tail = st[frame_at(x_end, fps, len(st)):]
    settle = float(np.abs(tail - st[-1]).max()) if len(tail) else 0.0
    texts = asset_texts(t, lm, score.seat_s, x_end, settle, score.duration_s or last_t, fps)
    stack = list(texts.values()) + [t.label, t.caption, *output_names(t).values()]
    while stack:
        v = stack.pop()
        if isinstance(v, dict):
            stack.extend(v.values())
        elif isinstance(v, (list, tuple)):
            stack.extend(v)
        else:
            assert_blind_safe(t, v)

    names = output_names(t)
    clips = Path(clips)
    enc = encoder or choose_encoder(available_encoders())
    report["video"] = render_side_by_side(video_path(root, "front"), video_path(root, "wrist"),
                                          clips / names["video"], fps, MAX_WIDTH, enc)
    times = contact_sheet_times(score.seat_s, last_t)
    report["sheet"] = render_contact_sheet(video_path(root, "front"), times, fps, len(st),
                                           clips / names["sheet"], texts["sheet_title"],
                                           texts["sheet_subtitle"])
    for theme, key in (("light", "joints"), ("dark", "joints_dark")):
        report[key] = render_joint_plot(st, fps, lm, score.seat_s, x_end, clips / names[key],
                                        texts, theme=theme)
    if t.recovery_strip:
        report["strip"] = render_recovery_strip(video_path(root, "front"), lm, fps,
                                                clips / names["strip"], texts, t.strip_crops)
    report["outputs"] = {k: str(clips / v) for k, v in names.items()}
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--token", choices=[t.token for t in TRIALS], default=None)
    ap.add_argument("--clips", default=str(CLIPS))
    args = ap.parse_args()

    bad = verify_manifest(ROLLOUTS)
    if bad:
        print("[golden_trial_assets] preserved rollouts do NOT match SHA256SUMS -- refusing:",
              *bad, sep="\n  ", file=sys.stderr)
        return 2
    print("preserved rollouts: every file matches SHA256SUMS")
    worst = 0
    for t in TRIALS:
        if args.token and t.token != args.token:
            continue
        r = run(t, Path(args.clips))
        print(f"\n{r['caption']}")
        print("  frames: " + ", ".join(f"{k} {v}" for k, v in r["counts"].items())
              + f", scorer {r['score'].frames}")
        print(f"  scorer: seat {r['score'].seat_s} s, verdict {r['score'].verdict}")
        print("  landmarks: " + ", ".join(
            f"{k} {'-' if v[0] is None else f'frame {v[0]} ({v[1]:.2f} s)'}"
            for k, v in r["landmarks"].items()))
        v = r["video"]
        print(f"  video: {v['frames']} frames {v['width']}x{v['height']} {v['encoder']}")
        print("  contact sheet frames: " + ", ".join(x["stamp"] for x in r["sheet"]))
        for p in r["problems"]:
            print(f"  PROBLEM {p}")
            worst = 1
        for k, p in r["outputs"].items():
            print(f"  {k}: {p}")
    return worst


if __name__ == "__main__":
    sys.exit(main())
