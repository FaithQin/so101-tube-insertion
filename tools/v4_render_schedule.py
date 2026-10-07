#!/usr/bin/env python
"""Regenerate the episode table in `Start Schedule — v4.md` FROM the manifest.

    python tools/v4_render_schedule.py          # rewrite the table in place

The table is a VIEW of `tools/v4_manifest.json`, never a source. Its index
column is the RECORDED index; voided rows are marked; redo rows say which
episode they replace and carry that episode's own variant text. Hold-out
stars are gone: A4 re-picks hold-outs at the end. Sep 5 2026: the operator
followed a static table by row number while the manifest had redos pencilled
in -- nine wrong labels. A table that is regenerated from the manifest cannot
disagree with it, and tests/test_v4_manifest.py checks that it does not.
"""
from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v4_manifest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "tools" / "v4_manifest.json"
SCHEDULE = ROOT / "Start Schedule — v4.md"
ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|([^|]*)\|([^|]*)\|([^|]*)\|\s*$")


def parse_rows(text: str) -> list[tuple[int, str, str, str, str]]:
    out = []
    for line in text.splitlines():
        m = ROW.match(line)
        if m:
            out.append((int(m.group(1)), m.group(2).strip(), m.group(3).strip(),
                        m.group(4).strip(), m.group(5).strip()))
    return out


def render(text: str, m: v4_manifest.Manifest, today: str) -> str:
    rows = parse_rows(text)
    if not rows:
        raise SystemExit("no table rows found")
    old_text = {i: (start, what) for i, _, _, start, what in rows}
    # per-type variant texts, in their original order (RESIST alternates LATERAL / ANGLE)
    variants: dict[str, list[str]] = {}
    starts: dict[str, str] = {}
    for _, t, _, start, what in rows:
        variants.setdefault(t, [])
        if what not in variants[t]:
            variants[t].append(what)
        starts.setdefault(t, start)
    counter: dict[str, int] = {}
    redo_of: dict[str, list[int]] = {}
    for v in m.voided:
        redo_of.setdefault(m.type_of(v), []).append(v)
    pending_redos = {t: list(vs) for t, vs in redo_of.items()}

    new_rows = []
    for i in range(max(m.assigned) + 1):
        t = m.type_of(i)
        if i in m.voided and (i not in old_text or t not in variants):
            # an unplanned index (ep 63, type EXTRA): no schedule text exists for it
            start, what = "—", m.reasons.get(i, "")[:120]
            flag = "VOID — unplanned"
        elif i in old_text and i in m.voided:
            start, what = old_text[i]
            flag = f"VOID — {m.reasons.get(i, '')[:60]}".rstrip()
        elif i in old_text and i < min(m.voided + [10**9]) or (i in old_text and i < 20):
            start, what = old_text[i]                       # already recorded: keep its text
            flag = ""
        else:
            src = None
            if pending_redos.get(t):
                src = pending_redos[t].pop(0)
            if src is not None and src in old_text:
                start, what = old_text[src]
                flag = f"← redo of {src}"
            elif t in variants:
                k = counter.get(t, 0); counter[t] = k + 1
                vs = variants[t]
                start, what = starts[t], vs[k % len(vs)]
                flag = ""
            else:
                start, what, flag = "—", m.reasons.get(i, ""), "unplanned"
        new_rows.append(f"| {i:>2} | {t} | {flag} | {start} | {what} |")

    # replace the contiguous block of table rows
    lines = text.splitlines()
    idx = [n for n, l in enumerate(lines) if ROW.match(l)]
    first, last = idx[0], idx[-1]
    header_note = (f"> **Table regenerated from `tools/v4_manifest.json` on {today}.** The index column "
                   f"is the RECORDED index. Voided rows are marked and their type is re-owed at a later "
                   f"index (A3); redo rows say which episode they replace. Hold-outs are re-picked at the "
                   f"end (A4). Regenerate with `python tools/v4_render_schedule.py`; never edit rows by hand.")
    # drop an earlier regenerated-note if present
    lines = [l for l in lines if not l.startswith("> **Table regenerated from")]
    idx = [n for n, l in enumerate(lines) if ROW.match(l)]
    first, last = idx[0], idx[-1]
    out = lines[:first] + new_rows + lines[last + 1:]
    # the note goes ABOVE the header row (never between the header and its |---| line,
    # which breaks the markdown table)
    hdr = next(n for n, l in enumerate(out) if l.startswith("| ep # |"))
    out[hdr] = out[hdr].replace("| ⭐ |", "| Note |")
    out[hdr:hdr] = [header_note, ""]
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def main() -> int:
    m = v4_manifest.Manifest.load(MANIFEST)
    text = SCHEDULE.read_text()
    SCHEDULE.write_text(render(text, m, date.today().isoformat()))
    rows = parse_rows(SCHEDULE.read_text())
    print(f"rendered {len(rows)} rows (0-{rows[-1][0]}); voided {m.voided}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
