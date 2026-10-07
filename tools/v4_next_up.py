#!/usr/bin/env python
"""The authoritative NEXT-UP list: recorded index -> type -> what to do.

    python tools/v4_next_up.py            # next 12
    python tools/v4_next_up.py --all      # everything still to record

Read THIS at the bench, never `Start Schedule — v4.md` by row number (A6).
The table's positional index stops being the recorded index at the first void;
this list is derived from `tools/v4_manifest.json`, the only label the
training data has. The one-line spec is the schedule's own text for that type.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4_manifest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "tools" / "v4_manifest.json"
SCHEDULE = ROOT / "Start Schedule — v4.md"


def specs() -> tuple[dict[str, str], dict[int, str]]:
    """(by type: first row's 'what to do'; by ROW: every row's) — a redo must show
    the VOIDED row's own variant (ep 4 was the ANGLE-miss RESIST, not the LATERAL)."""
    by_type: dict[str, str] = {}
    by_row: dict[int, str] = {}
    for line in SCHEDULE.read_text().splitlines():
        if line.startswith("|") and line[1:5].strip().isdigit():
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) >= 5:
                by_row[int(cells[0])] = cells[4]
                by_type.setdefault(cells[1], cells[4])
    return by_type, by_row


def recorded_count() -> int:
    cands = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/lerobot/faithqin/so101-tube-insert-v4_*")))
    if not cands:
        return 0
    info = Path(cands[0]) / "meta" / "info.json"
    return json.loads(info.read_text())["total_episodes"] if info.exists() else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    m = v4_manifest.Manifest.load(MANIFEST)
    by_type, by_row = specs()
    rec = recorded_count()
    last = max(m.assigned)
    redo_of = {}
    for v in m.voided:
        redo_of.setdefault(m.type_of(v), []).append(v)
    print(f"recorded {rec} · voided {len(m.voided)} {m.voided} · last index {last} · "
          f"to go {last - rec + 1}\n")
    end = last + 1 if a.all else min(rec + a.n, last + 1)
    for i in range(rec, end):
        t = m.type_of(i)
        tag, s = "", by_type.get(t, "")
        if redo_of.get(t):
            v = redo_of[t].pop(0)
            tag = f"   ← redo of {v}"
            s = by_row.get(v, s)            # the voided row's own variant
        print(f"  {i:>2}  {t:<11}{tag}")
        s = re.sub(r"\s+", " ", s)
        print(f"      {s[:150]}{'…' if len(s) > 150 else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
