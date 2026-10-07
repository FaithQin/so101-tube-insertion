#!/usr/bin/env python
"""The numbers a certification replay is logged with, computed from its own telemetry CSV.

    python tools/cert_summary.py tools/scored_logs/CERT-B3-01_telemetry.csv [more.csv ...]

Prints one line per file:  <name>  loop 19.99 Hz  ee_proxy 29.93  jaw_min 13.42 (held)

Definitions -- the ones already published, not new ones:
  loop_hz   (rows - 1) / (t_mono[last] - t_mono[first]); the receipts' wall-clocked definition.
  ee_proxy  mean over frames 103..202 of shoulder_lift + elbow_flex + wrist_flex position
            (the lab notebook (private), Sep 14 12:37). Higher = the arm rides higher. The recording (v4 ep 0
            observation.state) reads 30.12; the three Sep 6 passes 29.27; the Sep 14 12:37 passes
            29.79 / 29.46 / 30.34 (seat, miss, seat).
  jaw_min   minimum gripper position over frames 96..136, where the recorded episode holds the
            tube; "held" when >= v4_contact_events.JAW_HELD_MIN_DEG. CHAMF-GATE-02, the one
            measured drop, read 1.07.

Refuses a CSV without all 203 frames of v4 episode 0: the windows are defined on that episode,
and a stalled or truncated replay is not a certification pass. Descriptive only -- the seat is
the gate (G-R), and Faith calls it. No hardware. tests/test_cert_summary.py.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v4_contact_events as ce  # noqa: E402  (pure: the published jaw floor)

EPISODE_FRAMES = 203
EE_WINDOW = (103, 202)      # inclusive, the last 100 frames
CARRY_WINDOW = (96, 136)    # inclusive
EE_JOINTS = ("shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos")


class CertSummaryError(RuntimeError):
    """The CSV is not a complete certification pass."""


def summarize(path) -> dict:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    if n != EPISODE_FRAMES:
        raise CertSummaryError(f"{Path(path).name}: {n} rows, not the {EPISODE_FRAMES} frames of v4 "
                               f"episode 0 -- a stalled or truncated replay is not a certification pass")
    t0, t1 = float(rows[0]["t_mono"]), float(rows[-1]["t_mono"])
    loop_hz = (n - 1) / (t1 - t0)
    a, b = EE_WINDOW
    ee = [sum(float(r[j]) for j in EE_JOINTS) for r in rows[a:b + 1]]
    c, d = CARRY_WINDOW
    jaw = [float(r["gripper.pos"]) for r in rows[c:d + 1]]
    jaw_min = min(jaw)
    return {"name": Path(path).name.replace("_telemetry.csv", ""), "rows": n, "loop_hz": loop_hz,
            "ee_proxy": sum(ee) / len(ee), "jaw_min": jaw_min,
            "jaw_held": jaw_min >= ce.JAW_HELD_MIN_DEG}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", nargs="+")
    a = ap.parse_args(argv)
    rc = 0
    for p in a.csv:
        try:
            s = summarize(p)
        except (CertSummaryError, OSError, KeyError, ValueError) as exc:
            print(f"{Path(p).name}: REFUSED -- {exc}")
            rc = 1
            continue
        held = "held" if s["jaw_held"] else f"DROPPED (< {ce.JAW_HELD_MIN_DEG:.1f})"
        print(f"{s['name']}  loop {s['loop_hz']:.2f} Hz  ee_proxy {s['ee_proxy']:.2f}  jaw_min {s['jaw_min']:.2f} ({held})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
