"""The v4 manifest: episode index -> declared type -> status.

Recording appends; you cannot re-record a middle episode in place. So a bad
take is VOIDED (never deleted -- recorded data is never mutated) and its TYPE
is pushed back onto the queue to be recorded again later. The manifest is the
only thing that knows which recorded index carries which declared type, and it
is what emits the --dataset.episodes list for training.

Born Sep 4 2026, after eps 4/6/9/10 of v4 were voided mid-session and the
index->type mapping stopped matching the schedule's positional order.
"""
import sys

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))
import v4_manifest as M


def test_a_fresh_manifest_matches_the_schedule_positionally():
    m = M.Manifest(types=["CLEAN", "CLEAN", "RESIST", "CLEAN"])
    assert m.type_of(0) == "CLEAN"
    assert m.type_of(2) == "RESIST"
    assert m.voided == []
    assert m.training_episodes(holdouts=[]) == [0, 1, 2, 3]


def test_voiding_pushes_the_type_back_onto_the_queue():
    m = M.Manifest(types=["CLEAN", "CLEAN", "RESIST", "CLEAN"])
    m.void(2, reason="nudged with closed jaws")
    assert 2 in m.voided
    # the next episode recorded takes the voided type, not the next positional one
    assert m.next_type(recorded=4) == "RESIST"
    m.assign(4, "RESIST")
    assert m.type_of(4) == "RESIST"
    # a voided episode is excluded from training even if it is not a holdout
    assert m.training_episodes(holdouts=[]) == [0, 1, 3, 4]


def test_voided_episodes_are_never_deleted_only_excluded():
    m = M.Manifest(types=["CLEAN", "RESIST"])
    m.void(0, reason="two approaches")
    assert m.type_of(0) == "CLEAN", "the record of what a voided episode WAS must survive"
    assert 0 not in m.training_episodes(holdouts=[])


def test_holdouts_and_voids_are_both_excluded_and_do_not_double_count():
    m = M.Manifest(types=["CLEAN"] * 6)
    m.void(1, reason="x")
    assert m.training_episodes(holdouts=[3]) == [0, 2, 4, 5]


def test_counts_report_what_is_still_owed_per_type():
    m = M.Manifest(types=["CLEAN", "CLEAN", "RESIST", "NUDGE"])
    m.void(2, reason="x")
    # three episodes recorded (0,1,2); index 3 (NUDGE) not reached yet
    owed = m.owed(recorded=3)
    assert owed["RESIST"] == 1, "the voided RESIST must still be owed"
    assert owed["NUDGE"] == 1, "NUDGE was never recorded, still owed"
    assert owed.get("CLEAN", 0) == 0
    # once NUDGE is recorded at 3, only the voided RESIST remains owed
    assert m.owed(recorded=4) == {"RESIST": 1}


def test_roundtrip_through_disk_preserves_voids_and_assignments(tmp_path):
    m = M.Manifest(types=["CLEAN", "RESIST", "CLEAN"])
    m.void(1, reason="no back-off")
    m.assign(3, "RESIST")
    p = tmp_path / "manifest.json"
    m.save(p)
    m2 = M.Manifest.load(p)
    assert m2.voided == [1]
    assert m2.type_of(3) == "RESIST"
    assert m2.training_episodes(holdouts=[]) == m.training_episodes(holdouts=[])


def test_redo_queue_replays_each_voided_type_exactly_once_in_order():
    """Voiding four episodes of three different types must yield those four
    types back, in the order they were voided — not the same type repeatedly.

    First implementation asked "is this type still owed?", which is true for any
    type the schedule still has unrecorded copies of, so voiding one RESIST out
    of five made next_type() return RESIST forever.
    """
    types = ["CLEAN"] * 6 + ["RESIST"] * 3 + ["NUDGE"] * 2      # 11 planned
    m = M.Manifest(types=types)
    # void one RESIST, two CLEANs and one NUDGE, in that order
    m.void(6, reason="resist")      # RESIST
    m.void(0, reason="clean a")     # CLEAN
    m.void(9, reason="nudge")       # NUDGE
    m.void(1, reason="clean b")     # CLEAN
    got = []
    rec = 11
    for _ in range(4):
        ty = m.next_type(recorded=rec)
        m.assign(rec, ty)
        got.append(ty)
        rec += 1
    assert got == ["RESIST", "CLEAN", "NUDGE", "CLEAN"], got
    # after the redos, nothing is owed and the training set excludes the voids
    assert sum(m.owed(recorded=rec).values()) == 0
    assert m.training_episodes(holdouts=[]) == [2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14]


def test_after_redos_it_resumes_the_schedule_order_not_the_commonest_type():
    """The interleave is the point: eps 0-1 CLEAN, no two non-CLEAN adjacent.

    First implementation picked whichever type had the most still owed, which is
    always CLEAN, so after a redo it queued six CLEANs in a row and the recovery
    episodes all bunched at the end — exactly the fatigue-clustering the
    schedule's seeded interleave exists to prevent.
    """
    planned = ["CLEAN", "RESIST", "CLEAN", "LEFT-MIMIC", "CLEAN", "ROT-20°", "CLEAN", "NUDGE"]
    m = M.Manifest(types=planned)
    m.void(1, reason="bad resist")          # RESIST goes back on the queue
    rec = 8                                  # 0..7 recorded
    got = []
    for _ in range(3):
        ty = m.next_type(recorded=rec); m.assign(rec, ty); got.append(ty); rec += 1
    # only the voided RESIST is owed, then nothing
    assert got[0] == "RESIST", got
    assert got[1] is None and got[2] is None, f"nothing left to record, got {got}"


def test_unrecorded_schedule_positions_are_served_in_schedule_order():
    planned = ["CLEAN", "CLEAN", "RESIST", "CLEAN", "LEFT-MIMIC", "CLEAN", "ROT-20°"]
    m = M.Manifest(types=planned, assigned={0: "CLEAN", 1: "CLEAN"})   # only 2 recorded
    got = []
    rec = 2
    for _ in range(5):
        ty = m.next_type(recorded=rec); m.assign(rec, ty); got.append(ty); rec += 1
    assert got == ["RESIST", "CLEAN", "LEFT-MIMIC", "CLEAN", "ROT-20°"], got


def test_replan_tail_puts_redos_first_then_the_schedule_in_order_and_keeps_counts():
    """After a void, every index >= recorded must be re-derived: redo types first
    (while the setup is fresh), then the schedule's own order. Sep 5: voiding
    16 (CLEAN) and 19 (ROT-45) left the old tail in place, so type_of(21) said
    ROT-20 while the ROT-45 redo was what had to be recorded next."""
    from v4_manifest import Manifest
    planned = ["CLEAN", "CLEAN", "RESIST", "CLEAN", "NUDGE", "CLEAN", "ROT-45°", "CLEAN"]
    m = Manifest(planned)                      # positional: 0..7
    m.void(4, "bad nudge")
    m.void(6, "bad rot")
    m.replan_tail(recorded=7)                  # 0..6 recorded, 7 is next
    tail = [m.type_of(i) for i in range(7, 7 + len(m.pending(7)))]
    assert tail[:2] == ["NUDGE", "ROT-45°"], f"redos must come first: {tail}"
    assert tail == ["NUDGE", "ROT-45°", "CLEAN"], tail
    # counts: kept + tail == planned, per type
    from collections import Counter
    kept = Counter(m.type_of(i) for i in range(7) if i not in m.voided)
    assert kept + Counter(tail) == Counter(planned)
    assert max(m.assigned) == len(planned) + len(m.voided) - 1, "tail length must grow by one per void"
    assert m.redo_queue == [], "replan consumes the redo queue"


def test_replan_tail_does_not_touch_recorded_indices():
    from v4_manifest import Manifest
    m = Manifest(["CLEAN", "RESIST", "CLEAN"])
    m.void(1, "x")
    before = {i: m.type_of(i) for i in range(2)}
    m.replan_tail(recorded=2)
    assert {i: m.type_of(i) for i in range(2)} == before


def test_unvoid_restores_the_episode_and_removes_its_redo():
    """Sep 5: ep 19 was voided against the WRONG label (the manifest said
    ROT-45, the operator had recorded the schedule's CLEAN). Unvoiding must
    put it back in training and take its type back off the redo queue."""
    from v4_manifest import Manifest
    m = Manifest(["CLEAN", "ROT-45°", "CLEAN"])
    m.void(1, "mislabelled")
    assert 1 in m.voided and m.redo_queue == ["ROT-45°"]
    m.unvoid(1); m.replan_tail(recorded=3)
    assert 1 not in m.voided and m.redo_queue == [] and 1 not in m.reasons
    assert m.training_episodes(holdouts=[]) == [0, 1, 2]
    assert max(m.assigned) == 2, "an un-voided episode must not leave a phantom redo in the tail"


def test_relabel_records_what_was_ACTUALLY_recorded():
    """The manifest is the record of index -> type. When the operator records
    a different type from the one assigned (Sep 5: she followed the schedule
    table, the manifest had redos pencilled in), the manifest must be corrected
    to the truth, and the redo owed by a void must come back onto the queue."""
    from v4_manifest import Manifest
    m = Manifest(["CLEAN", "RESIST", "CLEAN", "LEFT-MIMIC"])
    m.void(1, "no back-off")                  # RESIST owed
    m.assign(2, "RESIST")                      # plan: 2 is the redo -> queue empties
    assert m.redo_queue == []
    m.relabel(2, "CLEAN")                      # ...but she recorded a CLEAN at 2
    m.replan_tail(recorded=3)
    assert m.type_of(2) == "CLEAN"
    assert m.type_of(3) == "RESIST", "the RESIST redo is owed again and comes first"
    assert m.type_of(4) == "LEFT-MIMIC"
    assert max(m.assigned) == 4


def test_replan_tail_ignores_the_pre_planned_tail_when_counting_what_is_owed():
    """The bug behind the Sep 5 re-plan: `assigned` carries the whole planned
    tail (0..61 on day one), so counting it as 'have' made every type look fully
    supplied and the redo queue came back empty. Only recorded indices count."""
    from v4_manifest import Manifest
    planned = ["CLEAN", "RESIST", "CLEAN", "NUDGE", "CLEAN"]
    m = Manifest(planned)                      # assigned pre-filled 0..4, like the real file
    m.void(1, "no back-off")                   # only 0..2 recorded so far
    m.replan_tail(recorded=3)
    assert m.redo_queue == [], "consumed by the replan"
    assert m.type_of(3) == "RESIST", f"redo must come first, got {[m.type_of(i) for i in range(3,6)]}"
    assert [m.type_of(i) for i in range(3, 6)] == ["RESIST", "NUDGE", "CLEAN"]
    assert max(m.assigned) == 5


def test_the_schedule_table_IS_the_manifest():
    """The table is a VIEW of the manifest (tools/v4_render_schedule.py). Its
    index column must be the recorded index and its type column the manifest's
    type -- row for row, including voids. Sep 5: nine episodes were recorded
    against a static table while the manifest had redos pencilled in."""
    from conftest import PROJECT
    from v4_manifest import Manifest
    import v4_render_schedule as r
    m = Manifest.load(PROJECT / "tools" / "v4_manifest.json")
    rows = r.parse_rows((PROJECT / "Start Schedule — v4.md").read_text())
    table = {i: t for i, t, *_ in rows}
    assert table == m.assigned, "run tools/v4_render_schedule.py — the table has drifted from the manifest"
    flags = {i: f for i, _, f, *_ in rows}
    for v in m.voided:
        assert flags[v].startswith("VOID"), f"voided ep {v} is not marked in the table"
    assert not any("⭐" in f for f in flags.values()), "hold-out stars must be gone (A4 re-picks at the end)"


def test_the_manifest_plan_matches_the_ORIGINAL_58_row_sequence():
    """The plan itself is unchanged by voids: the manifest's `planned` list is the
    seeded 58-row sequence, and the kept + remaining rows must sum to it."""
    from collections import Counter
    from conftest import PROJECT
    from v4_manifest import Manifest
    m = Manifest.load(PROJECT / "tools" / "v4_manifest.json")
    assert len(m.planned) == 58
    kept = Counter(t for i, t in m.assigned.items() if i not in m.voided)
    assert kept == Counter(m.planned), "kept + remaining rows no longer sum to the plan"
