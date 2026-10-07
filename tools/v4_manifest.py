"""Episode manifest for v4: which recorded index carries which declared type.

`lerobot-record` APPENDS. There is no way to re-record a middle episode in
place, and mutating recorded data is the one irreversible mistake this project
guards hardest against. So a bad take is VOIDED, never deleted, and its declared
TYPE goes back on the queue to be recorded again at a later index.

That breaks the schedule's positional index->type mapping the moment it happens,
and nothing else in the repo tracks it. This module is that record:

    manifest.type_of(9)              -> "NUDGE"      (what ep 9 WAS)
    manifest.void(9, reason=...)     -> ep 9 excluded, NUDGE re-owed
    manifest.next_type(recorded=11)  -> "NUDGE"      (record it again at 11)
    manifest.assign(11, "NUDGE")
    manifest.training_episodes(holdouts=[...])  -> the --dataset.episodes list

Voided episodes stay in the dataset and stay in the manifest. They are excluded
from training by index, exactly like the held-out reserve.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


class Manifest:
    def __init__(self, types: list[str], assigned: dict | None = None, voided: list | None = None,
                 reasons: dict | None = None, redo_queue: list | None = None):
        self.planned = list(types)
        # index -> type. Seeded positionally; diverges once a void happens.
        self.assigned = {int(k): v for k, v in (assigned or {}).items()} or {
            i: t for i, t in enumerate(self.planned)
        }
        self.voided = sorted(int(v) for v in (voided or []))
        self.reasons = {int(k): v for k, v in (reasons or {}).items()}
        # Types owed BECAUSE something was voided, oldest first. Kept separate
        # from `owed()`: a voided RESIST is owed even when the schedule already
        # has unrecorded RESISTs, and asking only "is this type owed?" cannot
        # tell the two apart (it returned the same type forever).
        self.redo_queue = list(redo_queue) if redo_queue is not None else [
            self.assigned[i] for i in self.voided if i in self.assigned
        ]

    # --- queries -----------------------------------------------------------
    def type_of(self, idx: int) -> str | None:
        """What this index WAS, whether or not it was voided."""
        return self.assigned.get(idx)

    def owed(self, recorded: int) -> Counter:
        """Types still needed: planned, minus what is assigned and not voided."""
        have = Counter(t for i, t in self.assigned.items() if i < recorded and i not in self.voided)
        return Counter({t: n for t, n in (Counter(self.planned) - have).items() if n > 0})

    def pending(self, recorded: int) -> list[str]:
        """Planned types not yet recorded, in the schedule's own order.

        Preserves the seeded interleave (eps 0-1 CLEAN, no two non-CLEAN
        adjacent). Serving "whichever type has the most owed" instead would
        always return CLEAN and bunch every recovery episode at the end of the
        session, where the operator is most fatigued.
        """
        have = Counter(t for i, t in self.assigned.items()
                       if i < recorded and i not in self.voided)
        out = []
        for t in self.planned:
            if have.get(t, 0) > 0:
                have[t] -= 1
            else:
                out.append(t)
        return out

    def next_type(self, recorded: int) -> str | None:
        """The type the next recorded episode should carry.

        Voided types come first — redo them while the setup is fresh — then the
        schedule's remaining order.
        """
        if self.redo_queue:                        # redos first, while the setup is fresh
            return self.redo_queue[0]
        pending = self.pending(recorded)           # then the schedule's own order
        return pending[0] if pending else None

    def training_episodes(self, holdouts: list[int]) -> list[int]:
        drop = set(self.voided) | set(int(h) for h in holdouts)
        return [i for i in sorted(self.assigned) if i not in drop]

    # --- mutation ----------------------------------------------------------
    def void(self, idx: int, reason: str) -> None:
        if idx not in self.assigned:
            raise KeyError(f"episode {idx} was never assigned a type")
        if idx not in self.voided:
            self.voided.append(idx); self.voided.sort()
            self.redo_queue.append(self.assigned[idx])
        self.reasons[idx] = reason

    def _owed_redos(self, recorded: int) -> list[str]:
        """Types still owed BECAUSE of a void, oldest void first, never more per
        type than the schedule is actually short. Counts only indices that have
        been RECORDED (< recorded): the pre-planned tail in `assigned` is a plan,
        not data, and counting it made every type look fully supplied (Sep 5)."""
        have = Counter(t for i, t in self.assigned.items()
                       if i < recorded and i not in self.voided)
        short = Counter(self.planned) - have
        out: list[str] = []
        for v in self.voided:
            t = self.assigned.get(v)
            if t is not None and out.count(t) < short.get(t, 0):
                out.append(t)
        return out

    def unvoid(self, idx: int) -> None:
        """Reverse a void made against the wrong label (Sep 5, ep 19). Call
        replan_tail() afterwards; the queue is derived state."""
        if idx in self.voided:
            self.voided.remove(idx)
        self.reasons.pop(idx, None)

    def relabel(self, idx: int, actual: str) -> None:
        """Correct a recorded index to the type that was ACTUALLY recorded.

        Sep 5 2026: the operator followed `Start Schedule — v4.md` by row
        number while this manifest had 11-14 pencilled in as redos of 4/6/9/10.
        Nine episodes were judged against the wrong labels, one good CLEAN was
        voided, and the redos were reported done when none had been recorded.
        The manifest records truth, not plan. Call replan_tail() afterwards.
        """
        self.assigned[idx] = actual

    def assign(self, idx: int, type_: str) -> None:
        self.assigned[idx] = type_
        if self.redo_queue and self.redo_queue[0] == type_:
            self.redo_queue.pop(0)

    def replan_tail(self, recorded: int) -> None:
        """Re-derive every index >= `recorded`: redo types first, then the
        schedule's own order. Call after any void.

        Sep 5 2026: voiding 16 and 19 left the stale positional tail in place,
        so type_of(21) still said ROT-20 while the ROT-45 redo was what had to
        be recorded next -- and the audit would have judged it against the wrong
        type. Recorded indices are never touched.
        """
        for idx in list(self.assigned):
            if idx >= recorded:
                del self.assigned[idx]
        self.redo_queue = self._owed_redos(recorded)
        idx = recorded
        while True:
            t = self.next_type(idx)
            if t is None:
                break
            self.assign(idx, t)
            idx += 1

    # --- persistence -------------------------------------------------------
    def save(self, path) -> None:
        Path(path).write_text(json.dumps({
            "planned": self.planned,
            "assigned": {str(k): v for k, v in sorted(self.assigned.items())},
            "voided": self.voided,
            "reasons": {str(k): v for k, v in sorted(self.reasons.items())},
            "redo_queue": self.redo_queue,
        }, indent=2))

    @classmethod
    def load(cls, path) -> "Manifest":
        d = json.loads(Path(path).read_text())
        return cls(d["planned"], d.get("assigned"), d.get("voided"), d.get("reasons"),
                   d.get("redo_queue"))
