#!/usr/bin/env python
"""The gates a certification replay passes BEFORE the arm is torqued, and the receipt of the seat call.

    python tools/cert_gate.py temps   <label>.ping            # every servo read, all <= 52 C      exit 0/1
    python tools/cert_gate.py home    <label>.home            # HOMED, a settled line, no WARNING   exit 0/1
    python tools/cert_gate.py record  tools/scored_logs/<label>_<stamp> SEAT|MISS   # writes <run>.seat from the run's files
    python tools/cert_gate.py receipt --label L --stamp S --verdict SEAT|MISS \
        --scene <label>.scene --ping <label>.ping --summary "<cert_summary line>"   # prints a receipt (used by record)

WHY (Sep 14 2026). Certification replays ran outside `preflight.py`, so its thermal gate never saw
them (the lab notebook (private) 12:47, gap 2), and the pre-session 52 C rule and "scene-check before every
replay" (12:37) were chat discipline. The morning's one miss was the pass without a scene check;
the 70 C event was an arm left holding. `tools/run_cert.sh` calls these so both refusals are code.
Reuses preflight's parsers: a malformed capture refuses, never passes. No hardware.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import preflight  # noqa: E402

CERT_TEMP_MAX_C = 52   # the pre-session rule (the project notes; handoff Sep 14 §2 item 6), stricter than preflight's 60
# Sep 17 2026 07:25: every seated certification pass settled the elbow at 85.6 at home; the three straight
# misses settled at 86.2-86.9 and then held the grasp 1.5 deg high under double the load (the worn gearbox
# on another tooth). preflight's +-1 sigma window passes both, so certification gets its own band and
# run_cert.sh re-homes (bounded) until the elbow lands in it.
CERT_ELBOW_TARGET = 85.6
CERT_ELBOW_TOL = 0.5
VERDICTS = ("SEAT", "MISS")


def gate_temps(ping_text: str, max_c: int = CERT_TEMP_MAX_C) -> tuple[bool, str]:
    try:
        temps = preflight.parse_joint_temps(ping_text)
    except ValueError as exc:
        return False, f"temps: {exc}"
    unread = [j for j in preflight.SERVO_JOINTS if j not in temps]
    if unread:
        return False, f"temps: {', '.join(unread)} temperature unreadable — an unread servo is not a cool one"
    hot = {j: t for j, t in temps.items() if t > max_c}
    if hot:
        worst = max(hot, key=hot.get)
        rest = "".join(f"; {j} {t}C" for j, t in sorted(hot.items(), key=lambda kv: -kv[1]) if j != worst)
        return False, f"temps: {worst} {hot[worst]}C > {max_c}C{rest} — wait, or relax the arm (tools/relax_arm.py)"
    hottest = max(temps, key=temps.get)
    return True, f"temps OK: hottest {hottest} {temps[hottest]}C <= {max_c}C"


def gate_home(home_text: str, elbow_band: bool = False) -> tuple[bool, str]:
    warn = re.search(r"^WARNING.*$", home_text, re.M)
    if warn:
        return False, f"home: {warn.group(0)}"
    if not re.search(r"^HOMED\b", home_text, re.M):
        return False, "home: no HOMED line in home_arm output"
    try:
        pose = preflight.parse_settled_pose(home_text)
    except ValueError as exc:
        return False, f"home: {exc}"
    if elbow_band:
        e = pose.get("elbow_flex")
        if e is None:
            return False, "home: no elbow_flex in the settled line"
        if abs(e - CERT_ELBOW_TARGET) > CERT_ELBOW_TOL:
            return False, (f"home: elbow settled at {e:.1f}, outside {CERT_ELBOW_TARGET} +- {CERT_ELBOW_TOL} "
                           f"(the band every seated certification pass landed in) — re-home")
        return True, f"home OK: elbow {e:.1f} in band"
    return True, "home OK"


def receipt(label: str, stamp: str, verdict: str, scene_text: str, ping_text: str, summary_line: str) -> str:
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}, got {verdict!r}")
    tube = preflight.parse_tube_delta(scene_text)
    rack = preflight.parse_rack_delta(scene_text)
    m = re.search(r"tube angle[^\n]*?delta=\s*([+-]?[\d.]+)°", scene_text)
    angle = m.group(1) if m else "nan"
    temps = preflight.parse_joint_temps(ping_text)
    hottest = max(temps, key=temps.get)
    s = re.search(r"loop ([\d.]+) Hz\s+ee_proxy ([\d.]+)\s+jaw_min ([\d.]+)", summary_line)
    loop_hz, ee, jaw = s.groups() if s else ("nan", "nan", "nan")
    lines = [f"label={label}", f"stamp={stamp}", f"verdict={verdict}", f"tube_px={tube}", f"rack_px={rack}",
             f"angle_deg={float(angle) if angle != 'nan' else 'nan'}", f"hottest_joint={hottest}",
             f"hottest_c={temps[hottest]}", f"loop_hz={loop_hz}", f"ee_proxy={ee}", f"jaw_min={jaw}"]
    return "\n".join(lines) + "\n"


def record(run_prefix, verdict: str) -> tuple[int, str]:
    """Write <run>.seat for a finished certification run from its own files. One verdict per pass, ever:
    the operator calls the seat to the chat (Sep 6, Sep 14 morning), and this is the one place it lands."""
    run = Path(run_prefix)
    seat = Path(str(run) + ".seat")
    if verdict not in VERDICTS:
        return 1, f"verdict must be one of {VERDICTS}, got {verdict!r}"
    if seat.exists():
        return 1, f"{seat.name} already holds a verdict — a pass has one verdict; never overwrite a receipt:\n{seat.read_text()}"
    scene, ping, tel = (Path(str(run) + ext) for ext in (".scene", ".ping", "_telemetry.csv"))
    missing = [f.name for f in (scene, ping, tel) if not f.exists()]
    if missing:
        return 1, f"not a finished certification run: missing {', '.join(missing)}"
    import cert_summary
    try:
        cs = cert_summary.summarize(tel)
    except (cert_summary.CertSummaryError, OSError, KeyError, ValueError) as exc:
        return 1, f"not a certification pass: {exc}"
    held = "held" if cs["jaw_held"] else "DROPPED"
    summary = f"{cs['name']}  loop {cs['loop_hz']:.2f} Hz  ee_proxy {cs['ee_proxy']:.2f}  jaw_min {cs['jaw_min']:.2f} ({held})"
    parts = run.name.split("_")
    if len(parts) < 3:
        return 1, f"run name {run.name!r} is not <label>_<YYYYMMDD>_<HHMMSS>"
    label, stamp = "_".join(parts[:-2]), "_".join(parts[-2:])
    try:
        text = receipt(label, stamp, verdict, scene.read_text(), ping.read_text(), summary)
    except ValueError as exc:
        return 1, f"could not build the receipt: {exc}"
    seat.write_text(text)
    return 0, text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("temps"); t.add_argument("ping")
    h = sub.add_parser("home"); h.add_argument("home"); h.add_argument("--elbow-band", action="store_true")
    r = sub.add_parser("receipt")
    for k in ("label", "stamp", "verdict", "scene", "ping", "summary"):
        r.add_argument(f"--{k}", required=True)
    rec = sub.add_parser("record"); rec.add_argument("run"); rec.add_argument("verdict")
    a = ap.parse_args(argv)
    if a.cmd == "record":
        rc, text = record(a.run, a.verdict)
        print(text if rc == 0 else f"RECORD REFUSED — {text}")
        return rc
    try:
        if a.cmd == "temps":
            ok, why = gate_temps(Path(a.ping).read_text())
        elif a.cmd == "home":
            ok, why = gate_home(Path(a.home).read_text(), elbow_band=a.elbow_band)
        else:
            print(receipt(a.label, a.stamp, a.verdict, Path(a.scene).read_text(), Path(a.ping).read_text(), a.summary), end="")
            return 0
    except (OSError, ValueError) as exc:
        print(f"CERT GATE ABORT — could not evaluate: {exc}")
        return 1
    print(("CERT GATE PASS — " if ok else "CERT GATE REFUSED — ") + why)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
