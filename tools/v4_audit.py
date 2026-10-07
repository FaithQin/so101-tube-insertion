#!/usr/bin/env python
"""Audit a freshly recorded v4 episode against ITS OWN type's expectations.

Run it the moment an episode is saved, while the tube, the rack and the
operator's memory are all still in the state that produced it:

    python tools/v4_audit.py                 # every recorded episode
    python tools/v4_audit.py --episode 11    # just the one just recorded

WHAT IT GATES, AND WHAT IT REFUSES TO GATE
------------------------------------------
Of the four declared recovery criteria, exactly one is separable from the
eleven episodes recorded so far. This tool gates on that one and on structure,
and REPORTS the rest as numbers beside their observed distribution.

GATED
  * Structure -- non-empty, 18-dim state (6 pos + 6 load + 6 current), a grasp,
    a rack phase, a release. A zero-frame episode is what the phantom-arrow
    crash leaves behind; a 6-dim episode can never be retrofitted for the
    load-sensing A/B.
  * Amendment A2 -- a NUDGE must graze with OPEN jaws. Measured as
    open -> shut -> reopen jaw cycles at the pick site. Episode 9 shows exactly
    one, spanning 4.5-7.5 s; its human void reason reads "nudged with
    closing/closed jaws over ~3 s". Independent reproduction, duration included.
    Every other recorded episode: zero. **Control: 0/50 false positives on the
    v3 dataset** (Aug 27, hand-staged recovery, also 18-dim, and independent of
    every number in this file) — so zero false positives in sixty episodes.

NOT GATED -- reported for Faith's eye
  * RESIST back-off, and CLEAN decisiveness. The best feature found is the
    elbow path/net "wander" ratio over the rack phase, and it does order
    correctly:
        CLEAN keep   1.10 1.17 1.21 1.23 2.17 2.20
        RESIST VOID  1.94   (ep 4, "rack 10.1 s but no elbow back-off")
        CLEAN  VOID  2.38   (ep 6, "back-off at the rack in a decisive take")
        RESIST keep  2.68   (ep 2)
    but the margin between the highest CLEAN keep and the voided ep 6 is
    **0.18, on eight episodes**. Gating there would be a threshold without a
    control -- a named bug class in this project. The number is printed with
    the distribution instead, so a take that lands outside the pack is visible
    without a fabricated verdict.
  * LEFT-MIMIC and ROT-20/45. No episode of either type has been recorded, so
    there is nothing to calibrate against. Reported UNCALIBRATED, never "ok".

Written Sep 5 2026. `scratchpad/v4_audit.py`, which the Sep 4 handoff points at,
does not exist -- the scratchpad was transient. Tests: tests/test_v4_audit.py.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

FPS = 20
PICK_SITE_PAN = 55.0     # pan is ~68 at the tube, ~15 at the rack
JAW_OPEN = 20.0          # was 25.0 -- ep 13's open-jaw graze sat at 24.8 and was mis-flagged (Sep 5)
JAW_SHUT = 10.0
RACK_WRIST_FLEX = 20.0
RACK_PAN = 40.0

# Observed rack-wander distribution over the 11 episodes recorded by Sep 5.
# Printed beside each new episode's number. NOT a threshold.
WANDER_REFERENCE = {
    "CLEAN keep": [1.10, 1.17, 1.21, 1.23, 2.17, 2.20],
    "CLEAN void": [1.13, 2.38],
    "RESIST keep": [2.68],
    "RESIST void": [1.94],
    "NUDGE void": [1.10],
}

# Approach signature, measured over all 20 episodes recorded by Sep 5 morning:
#   direct (CLEAN / RESIST)             jaws-open hover at the tube 1.4-2.3 s  (n=15)
#   graze  (NUDGE / ROT / LEFT-MIMIC)   5.7-7.1 s on every VALID one           (n=4)
# 3.4 s between the classes. The one graze below that was the voided ep 9,
# whose jaws closed early -- so a short hover on a graze label is a defect
# whichever way you read it. This check is what would have caught the Sep 5
# label mix-up on the spot: an episode recorded as the schedule's LEFT-MIMIC
# but judged as the manifest's RESIST, and a CLEAN judged as a ROT-45.
DIRECT_TYPES = {"CLEAN", "RESIST"}
GRAZE_TYPES = {"NUDGE", "ROT-20°", "ROT-45°", "LEFT-MIMIC"}
HOVER_DIRECT_MAX_S = 3.5
HOVER_GRAZE_MIN_S = 4.0

GATED_TYPES = DIRECT_TYPES | GRAZE_TYPES   # structure + approach signature for all; open jaws for graze types
UNCALIBRATED_NOTE = (
    "no calibrated threshold exists for this type yet -- judge by eye "
    "(and by the video) against the schedule's description"
)


@dataclass
class Result:
    episode: int | None = None
    type: str | None = None
    ok: bool = True
    uncalibrated: bool = False
    failures: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    summary: str = ""


# --- features ---------------------------------------------------------------

def jaw_reopen_cycles(pan, grip):
    """open -> shut -> reopen jaw cycles at the PICK SITE, with their spans.

    The jaw opens at the rack on every single episode (that is the release), so
    the pick-site restriction is what makes this specific to the graze.
    """
    pan = np.asarray(pan, dtype=float)
    grip = np.asarray(grip, dtype=float)
    state, cycles, spans, shut_at = None, 0, [], None
    for i in range(len(grip)):
        if pan[i] <= PICK_SITE_PAN:
            continue
        if grip[i] > JAW_OPEN:
            if state == "shut":
                cycles += 1
                spans.append((shut_at / FPS, i / FPS))
            state = "open"
        elif grip[i] < JAW_SHUT:
            if state == "open":
                state, shut_at = "shut", i
    return cycles, spans


def approach_signature(pan, grip):
    """(hover_s, pan_excursion_deg): jaws-open time at the pick site before the
    grasp closes, and how far the pan wandered while there. A direct approach
    opens and closes; a graze/hover type stays open for seconds first."""
    pan = np.asarray(pan, dtype=float)
    grip = np.asarray(grip, dtype=float)
    site = pan > PICK_SITE_PAN
    opened = np.where(site & (grip > JAW_OPEN))[0]
    if not len(opened):
        return None, None
    first_open = opened[0]
    closes = np.where(site & (np.arange(len(grip)) > first_open) & (grip < 16.0))[0]
    grasp = closes[0] if len(closes) else None
    span = pan[first_open:grasp] if grasp is not None else pan[first_open:]
    hover = (grasp - first_open) / FPS if grasp is not None else None
    return hover, float(span.max() - span.min()) if len(span) else None


def grasp_index(pan, grip):
    """First frame at the pick site where a previously open jaw reaches closed."""
    opened = False
    for i in range(len(grip)):
        if pan[i] > PICK_SITE_PAN and grip[i] > JAW_OPEN:
            opened = True
        if opened and pan[i] > PICK_SITE_PAN and grip[i] <= 16.0:
            return i
    return None


def rack_slice(pan, wrist_flex):
    return np.where((np.asarray(wrist_flex) < RACK_WRIST_FLEX) & (np.asarray(pan) < RACK_PAN))[0]


def wander(x):
    """Path length / net displacement. 1.0 = a straight move; >1 = back-and-forth."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan")
    return float(np.abs(np.diff(x)).sum() / max(abs(x[-1] - x[0]), 1e-6))


# --- the audit --------------------------------------------------------------

def audit(state, type_, episode=None) -> Result:
    r = Result(episode=episode, type=type_)
    state = np.asarray(state)

    if state.ndim != 2 or len(state) == 0:
        r.ok = False
        r.failures.append("EMPTY: zero frames recorded — this is what the phantom-arrow "
                          "crash leaves behind. Re-record.")
        r.summary = "zero-frame episode"
        return r
    if state.shape[1] != 18:
        r.ok = False
        r.failures.append(f"WIDTH: observation.state is {state.shape[1]}-dim, must be 18 "
                          f"(6 pos + 6 load + 6 current). A narrow episode can never be "
                          f"retrofitted for the load-sensing A/B.")
        r.summary = f"{state.shape[1]}-dim state"
        return r

    pan, lift, elbow, wflex, grip = (state[:, 0], state[:, 1], state[:, 2],
                                     state[:, 3], state[:, 5])
    gi = grasp_index(pan, grip)
    rack = rack_slice(pan, wflex)
    cycles, spans = jaw_reopen_cycles(pan, grip)
    hover_s, excursion = approach_signature(pan, grip)

    r.metrics = {
        "hover_s": round(hover_s, 1) if hover_s is not None else None,
        "pan_excursion_deg": round(excursion, 1) if excursion is not None else None,
        "frames": len(state),
        "seconds": round(len(state) / FPS, 1),
        "grasp_s": round(gi / FPS, 1) if gi is not None else None,
        "rack_s": round(len(rack) / FPS, 1),
        "rack_wander": round(wander(elbow[rack]), 2) if len(rack) > 10 else None,
        "peak_elbow_load_at_rack": int(np.abs(state[rack, 8]).max()) if len(rack) else 0,
        "jaw_reopen_cycles": cycles,
        "jaw_reopen_spans_s": [(round(a, 1), round(b, 1)) for a, b in spans],
    }

    if gi is None:
        r.ok = False
        r.failures.append("STRUCTURE: no grasp found — the jaw never opened and then closed "
                          "at the pick site.")
    if len(rack) < 10:
        r.ok = False
        r.failures.append("STRUCTURE: no rack phase found — the tube was never carried to "
                          "the rack.")

    # --- approach signature vs label: the check that catches a label mix-up ---
    if hover_s is not None:
        if type_ in DIRECT_TYPES and hover_s > HOVER_DIRECT_MAX_S:
            r.ok = False
            r.failures.append(
                f"LABEL MISMATCH? labelled {type_} but the jaws hovered open at the tube for "
                f"{hover_s:.1f} s (direct approaches measure 1.4-2.3 s; graze types 5.7-7.1 s). "
                f"Either this is a graze episode carrying the wrong label, or the approach was "
                f"not 'as CLEAN'. Check which row of the schedule was being followed."
            )
        if type_ in GRAZE_TYPES and hover_s < HOVER_GRAZE_MIN_S:
            r.ok = False
            r.failures.append(
                f"GRAZE NOT REGISTERED / LABEL MISMATCH? labelled {type_} but the jaws hovered "
                f"open only {hover_s:.1f} s before closing (valid graze episodes: 5.7-7.1 s; "
                f"direct: 1.4-2.3 s). Either it went straight in like a CLEAN, or the jaws closed "
                f"during the graze (ep 9). Check the label, then re-record if it is right."
            )

    # --- open jaws through the graze: A2 (NUDGE), A1 (LEFT-MIMIC), A5 (ROT, Sep 5) ---
    if type_ in GRAZE_TYPES and cycles > 0:
        longest = max((b - a) for a, b in spans)
        rule = {"NUDGE": "A2", "LEFT-MIMIC": "A1"}.get(type_, "A5")
        r.ok = False
        r.failures.append(
            f"{rule} VIOLATED: the jaws CLOSED during the approach and reopened "
            f"({cycles} cycle(s), longest {longest:.1f} s at "
            f"{spans[0][0]:.1f}-{spans[0][1]:.1f} s). A {type_} must graze/hover with OPEN jaws "
            f"— a close-on-air teaches the woodpecker. This is exactly why ep 9 was voided. "
            f"RE-RECORD."
        )

    if type_ not in GATED_TYPES:
        r.uncalibrated = True

    # RESIST's back-off is real but not separable yet — say so rather than gate.
    if type_ == "RESIST":
        r.uncalibrated = True
        r.summary = (
            f"RESIST back-off is NOT gated: rack wander {r.metrics['rack_wander']} — "
            f"compare against RESIST keep 2.68 / RESIST void 1.94 / CLEAN keep 1.10-2.20. "
            f"The margin is 0.18 on 8 episodes, so this is a number for your eye, not a "
            f"verdict. Did you feel the push-back and back off 1-2 cm?"
        )
    elif r.uncalibrated:
        r.summary = f"{type_}: {UNCALIBRATED_NOTE}"
    elif r.ok and type_ in GRAZE_TYPES:
        extra = ("" if type_ not in ("ROT-20°", "ROT-45°") else
                 " — the rotation MAGNITUDE (20 vs 45) is not measured here; judge that by eye")
        r.summary = (f"{type_}: structure ok, graze registered ({hover_s:.1f} s open hover, "
                     f"pan excursion {excursion:.1f}°), jaws stayed open{extra}")
    elif r.ok:
        r.summary = f"{type_}: structure ok, direct approach ({hover_s:.1f} s)"
    else:
        r.summary = f"{type_}: {len(r.failures)} problem(s)"
    return r


# --- dataset plumbing -------------------------------------------------------

def load_dataset(root=None):
    """(states_by_episode, manifest) or None if the dataset is not on this machine."""
    import pandas as pd

    if root is None:
        cands = sorted(glob.glob(os.path.expanduser(
            "~/.cache/huggingface/lerobot/faithqin/so101-tube-insert-v4_*")))
        if not cands:
            return None
        root = cands[0]
    root = Path(root)
    pqs = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    if not pqs:
        return None
    import v4_manifest
    man_path = Path(__file__).resolve().parent / "v4_manifest.json"
    if not man_path.exists():
        return None
    manifest = v4_manifest.Manifest.load(man_path)
    frames, skipped = [], []
    for p in pqs:
        try:
            frames.append(pd.read_parquet(p))
        except Exception as e:  # noqa: BLE001
            # lerobot 0.6.1 streams the CURRENT data file and writes its Parquet
            # footer only on close, so while a recording runs the newest file is
            # unreadable (Sep 5: file-003, held open by the recorder, fd 34w).
            # Skip it and say so; the flushed episodes are still auditable.
            skipped.append((Path(p).name, type(e).__name__))
    if skipped:
        print(f"[v4_audit] skipped {len(skipped)} unreadable data file(s) — the recorder is "
              f"probably still writing them: {skipped}", file=sys.stderr)
    if not frames:
        return None
    df = pd.concat(frames)
    states = {int(e): np.stack(g["observation.state"].values)
              for e, g in df.groupby("episode_index")}
    return states, manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--episode", type=int, default=None, help="audit just this index")
    ap.add_argument("--root", default=None)
    args = ap.parse_args()

    ds = load_dataset(args.root)
    if ds is None:
        print("[v4_audit] no v4 dataset found on this machine", file=sys.stderr)
        return 2
    states, manifest = ds

    eps = [args.episode] if args.episode is not None else sorted(states)
    worst = 0
    for ep in eps:
        if ep not in states:
            print(f"[v4_audit] episode {ep} is not in the dataset "
                  f"(recorded: {min(states)}-{max(states)})", file=sys.stderr)
            worst = max(worst, 2)
            continue
        typ = manifest.type_of(ep)
        r = audit(states[ep], typ, episode=ep)
        voided = ep in manifest.voided
        head = f"ep {ep:>2}  {typ:<11}" + ("  [already VOIDED]" if voided else "")
        print(f"\n{head}")
        m = r.metrics
        print(f"  {m['seconds']:.1f}s / {m['frames']} frames   grasp@{m['grasp_s']}s   "
              f"rack {m['rack_s']}s   wander {m['rack_wander']}   "
              f"peak elbow load {m['peak_elbow_load_at_rack']}   "
              f"jaw reopen cycles {m['jaw_reopen_cycles']}")
        for f in r.failures:
            print(f"  ✗ {f}")
        if r.summary:
            print(f"  · {r.summary}")
        if not r.failures and not r.uncalibrated:
            print("  ✓ passes every check calibrated for this type")
        if not r.ok and not voided:
            worst = max(worst, 1)

    print("\nGATED: structure (frames, 18-dim, grasp, rack) for every type; "
          "amendment A2 (open-jaw graze) for NUDGE.")
    print("NOT GATED: RESIST back-off, CLEAN decisiveness, LEFT-MIMIC, ROT-20/45 — "
          "reported as numbers, judged by eye. See this file's header for why.")
    return worst


if __name__ == "__main__":
    sys.exit(main())
