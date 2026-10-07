"""The figure numbers must equal the numbers already published in the Findings
Log and the trial sheet. If a figure and the log ever disagree, this fails."""

import csv
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import figures_data as fd  # noqa: E402


def test_v2_sheet_parses_to_the_published_tally():
    rows = fd.v2_trials()
    assert len(rows) == 40
    by = {p: [r for r in rows if r["policy"] == p] for p in ("ACT-A", "ACT-B")}
    assert len(by["ACT-A"]) == 20 and len(by["ACT-B"]) == 20
    assert sum(r["success"] for r in by["ACT-A"]) == 6      # A 6/20
    assert sum(r["success"] for r in by["ACT-B"]) == 5      # B 5/20
    assert Counter(r["stage"] for r in by["ACT-A"]) == {"R": 7, "G": 5, "T": 2, "S": 6}
    assert Counter(r["stage"] for r in by["ACT-B"]) == {"R": 15, "S": 5}
    assert all(r["close_offset_deg"] is not None for r in rows)


def test_stage_split_numbers():
    cum = fd.stage_cumulative(fd.v2_trials())
    assert cum["ACT-A"]["S"] == pytest.approx(0.30)
    assert cum["ACT-A"]["G"] == pytest.approx(0.65)   # 13 of 20 held the tube at some point
    assert cum["ACT-B"]["S"] == pytest.approx(0.25)
    assert cum["ACT-B"]["G"] == pytest.approx(0.25)   # B held it only in its 5 successes
    assert cum["ACT-A"]["R"] == cum["ACT-B"]["R"] == 1.0


@pytest.mark.skipif(not (fd.CACHE / fd.V2).exists(), reason="dataset cache not present")
def test_durations_match_the_aug_30_entry():
    v2 = fd.duration_summary(fd.episode_durations(fd.V2))
    v3 = fd.duration_summary(fd.episode_durations(fd.V3))
    assert v2["n"] == 50 and v3["n"] == 50
    # Pinned at the precision the write-up publishes (0.1 s), compared exactly.
    # A tolerance wider than the reported precision cannot catch a one-decimal
    # error, which is how "v2 max 19.8" and "v3 median 10.2" survived (Sep 3).
    assert (round(v2["median"], 1), round(v2["p90"], 1), round(v2["max"], 1)) == (11.9, 15.8, 19.7)
    assert (round(v3["median"], 1), round(v3["p90"], 1), round(v3["max"], 1)) == (10.1, 12.0, 13.8)
    assert v2["long_tail"] == 2
    assert v3["long_tail"] == 0


SCORE_LOGS = fd.ROOT / "tools/scored_logs"


def _require_score_receipts(n: int):
    """Skip when the .score receipts are not on this machine.

    `tools/scored_logs/*` is gitignored (only the two CSVs are tracked), so a
    fresh clone has the ledger but none of the per-trial receipts these two
    tests read. Before Sep 5 they raised FileNotFoundError there: the suite
    passed ONLY on the bench laptop, and anywhere else the pre-session ritual
    reported two failures that were pure environment. A gate that cries wolf
    trains you to ignore it.

    Skipping is right for hardware-produced artifacts; it is NOT a verdict on
    whether they should be tracked. They are ~84K of primary evidence behind
    every write-up claim, and the .home receipts are what made the Sep 5 Fault-1
    correction reproducible -- that call belongs to the public-repo curation
    task, not to this test.
    """
    have = len(list(SCORE_LOGS.glob("*.score")))
    if have < n:
        pytest.skip(
            f"only {have} .score receipts on this machine (need {n}); "
            f"tools/scored_logs/* is gitignored, so this is a fresh clone, not a regression"
        )


def test_non_act_policies_are_never_labelled_act_a():
    """A pi0.5 or SmolVLA row in trials.csv must not be plotted as an ACT-A point.

    figures_data classified any policy whose name lacked 'act-tube-B' as ACT-A,
    so the first pi0.5 probe (P05-01, Sep 3 2026) entered the v3 ACT scatter as
    a ninth ACT-A point and turned this module's own scatter test red. fig4 is
    the pre-registered ACT A/B comparison; pi0.5 and SmolVLA are probes and are
    not part of it. An unrecognised policy must be excluded, never defaulted.
    """
    assert fd.classify_policy("faithqin/act-tube-A-v3") == "ACT-A"
    assert fd.classify_policy("faithqin/act-tube-B-v3") == "ACT-B"
    assert fd.classify_policy("faithqin/pi05-tube-A-v3_2026-08-30_03-06-05") is None
    assert fd.classify_policy("faithqin/smolvla-tube-A-v3") is None
    assert fd.classify_policy("faithqin/smolvla-tube-B-v3") is None
    _require_score_receipts(8)
    labels = {p["label"] for p in fd.v3_final_positions()}
    assert not any(l.startswith("P05-") for l in labels), "pi0.5 probe leaked into the ACT scatter"


def test_v3_final_positions_match_the_scatter_claim():
    _require_score_receipts(8)
    pts = fd.v3_final_positions()
    assert len(pts) == 8                                   # 6 A + 2 B valid trials
    assert Counter(p["policy"] for p in pts) == {"ACT-A": 6, "ACT-B": 2}
    xs = [p["x"] for p in pts]; ys = [p["y"] for p in pts]
    assert (min(xs), max(xs)) == (253, 568)
    assert (min(ys), max(ys)) == (340, 715)
    assert next(p for p in pts if p["label"] == "B2V-A-02")["seat_runs"] == 6


@pytest.mark.skipif(not (fd.CACHE / fd.V2).exists(), reason="dataset cache not present")
def test_skeleton_duration_table_matches_the_data():
    """The write-up publishes these to 0.1 s, so the guard must resolve 0.1 s.

    Found in the Sep 3 night audit: the §2 table said v2 max 19.8 and v3 median
    10.2 while the parquet says 19.7 and 10.1. The existing pin used abs=0.15 --
    wider than the precision the numbers are reported at, so it could not tell
    10.1 from 10.2. A tolerance looser than the claim it guards is not a guard.
    """
    import re

    text = (ROOT / "the results draft (private)").read_text()
    published = {}
    for line in text.splitlines():
        m = re.match(r"\|\s*(v2|v3)\b[^|]*\|\s*([\d.]+) s\s*\|\s*([\d.]+) s\s*\|\s*([\d.]+) s\s*\|", line)
        if m:
            published[m.group(1)] = tuple(float(m.group(i)) for i in (2, 3, 4))
    assert set(published) == {"v2", "v3"}, published

    for name, repo in (("v2", fd.V2), ("v3", fd.V3)):
        s = fd.duration_summary(fd.episode_durations(repo))
        computed = (round(s["median"], 1), round(s["p90"], 1), round(s["max"], 1))
        assert published[name] == pytest.approx(computed, abs=0.05), (
            f"{name}: skeleton says {published[name]}, data says {computed}"
        )


def test_a_v4_trial_never_enters_the_v3_scatter(tmp_path):
    """Sep 6 2026, caught by the suite the moment the first v4 trial landed. The v4 bench writes
    into the same tools/scored_logs/trials.csv, so `faithqin/act-tube-A-v4` — a genuine ACT-A
    policy — became a ninth point in a figure whose caption says eight, with 56 trials queued
    behind it. This is the same bug as `classify_policy`'s pi0.5 case, one level up: the question
    is not only WHICH ARM but WHICH GENERATION."""
    assert fd.policy_generation("faithqin/act-tube-A-v4") == "v4"
    assert fd.policy_generation("faithqin/act-tube-A-v3_2026-08-30_03-06-05") == "v3"
    assert fd.policy_generation("faithqin/act-tube-B-v2") == "v2"
    assert fd.policy_generation("faithqin/something-unversioned") is None

    logs = tmp_path / "logs"
    logs.mkdir()
    rows = [("V3-A", "faithqin/act-tube-A-v3", "20260903_120000"),
            ("V4-A", "faithqin/act-tube-A-v4", "20260906_120000")]
    with open(tmp_path / "trials.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "policy", "n", "lock", "stamp", "verdict", "dur_s", "seat_runs"])
        for label, policy, stamp in rows:
            w.writerow([label, policy, 25, "off", stamp, "SEAT", "43.6", 1])
            (logs / f"{label}_{stamp}.score").write_text("  FINAL: cap=(300, 400) -> SEAT\n")

    got = fd.v3_final_positions(csv=tmp_path / "trials.csv", logs=logs)
    assert [r["label"] for r in got] == ["V3-A"], got
    assert [r["label"] for r in fd.v3_final_positions(csv=tmp_path / "trials.csv", logs=logs,
                                                      generation="v4")] == ["V4-A"]
