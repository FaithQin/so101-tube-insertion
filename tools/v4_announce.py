#!/usr/bin/env python
"""Speak and print the TYPE of each episode the moment the recorder starts it.

    python tools/v4_announce.py            # tails the recorder's bridge output

Sep 5 2026: nine episodes were recorded against `Start Schedule — v4.md` by row
number while `tools/v4_manifest.json` had 11-14 pencilled in as redos. The
manifest is the only label the training data has, and nothing put it in front
of the operator at the moment she needed it — she is holding the leader with
her eyes on the arm, not on a JSON file. So: when lerobot logs
"Recording episode N", look N up in the manifest and SAY it (the runners
already use `say` for gates), and print it large.

Tests: tests/test_v4_announce.py (the line -> utterance mapping; `say` is
never called there).
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4_manifest  # noqa: E402

MANIFEST = Path(__file__).resolve().parent / "v4_manifest.json"
BRIDGE_OUT = Path("/tmp/capstone/bridge/v4rec.out")

_REC = re.compile(r"Recording episode (\d+)")
_RERECORD = re.compile(r"Re-record episode")

SPOKEN = {  # what the operator needs to hear, nothing more
    "CLEAN": "clean. straight in.",
    "RESIST": "resist. jam at the funnel, feel it, back off two centimetres, re-align.",
    "NUDGE": "nudge. open-jaw graze on the way in, one centimetre right, then regrasp.",
    "LEFT-MIMIC": "left mimic. hover left of the cap, jaws open, correct right, one close.",
    "ROT-20°": "rotate twenty. head graze, jaws open, come up, adjust, regrasp.",
    "ROT-45°": "rotate forty-five. head graze, jaws open, come up, adjust, regrasp.",
}


def utterance(line: str, manifest: v4_manifest.Manifest) -> str | None:
    """The sentence to speak for one recorder log line, or None."""
    if _RERECORD.search(line):
        return "re-recording."
    m = _REC.search(line)
    if not m:
        return None
    idx = int(m.group(1))
    t = manifest.type_of(idx)
    if t is None:
        return f"episode {idx}. NOT IN THE MANIFEST. stop."
    redo = [v for v in manifest.voided if manifest.type_of(v) == t]
    return f"episode {idx}. {SPOKEN.get(t, t)}"


def follow(path: Path, start_at_end: bool = True):
    """An iterator of NEW lines. The starting line count is taken EAGERLY, here,
    not on the first next() -- so a line appended between creating the
    iterator and consuming it is still seen (the first version's test hung on
    exactly that). Starts at the end of an existing file (a stale log from an
    earlier run must not be replayed: the first version spoke nine old
    episodes at once on startup) and resets if the file is truncated or
    replaced (the bridge rm -f's it when a new recording launches)."""
    seen = len(path.read_text(errors="replace").splitlines()) if (start_at_end and path.exists()) else 0
    return _follow_from(path, seen)


def _follow_from(path: Path, seen: int):
    while True:
        if path.exists():
            lines = path.read_text(errors="replace").splitlines()
            if len(lines) < seen:          # truncated / replaced: new run
                seen = 0
            for line in lines[seen:]:
                yield line
            seen = len(lines)
        time.sleep(0.5)


def speak(text: str) -> None:
    """One voice at a time: wait for the previous utterance to finish."""
    subprocess.run(["say", text], check=False)


def main() -> int:
    manifest = v4_manifest.Manifest.load(MANIFEST)
    print(f"[announce] watching {BRIDGE_OUT} (new lines only)", flush=True)
    for line in follow(BRIDGE_OUT):
        u = utterance(line, manifest)
        if u:
            print(f"\n{'=' * 64}\n  {u.upper()}\n{'=' * 64}\n", flush=True)
            speak(u)


if __name__ == "__main__":
    sys.exit(main())
