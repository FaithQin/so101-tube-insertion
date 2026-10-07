"""The recovery-signature auditor, and the limits it must admit to.

Faith records a recovery episode, the auditor checks it from the parquet before the
setup is disturbed, and says whether the signature registered. Four of the
first eleven v4 episodes were voided that way. `scratchpad/v4_audit.py`, which
the Sep 4 handoff tells the next session to run, does not exist -- the
scratchpad was transient. This is that tool, rebuilt in the repo with tests.

**The honest half.** Of the four declared recovery criteria, exactly ONE is
separable from the eleven episodes recorded so far:

  * A2 (NUDGE grazes with OPEN jaws) -- separates perfectly. Episode 9 shows one
    open->shut->reopen jaw cycle at the pick site spanning 4.5-7.5 s; the human's
    void reason reads "nudged with closing/closed jaws over ~3 s". Independent
    reproduction, duration included. Every other episode: zero cycles.

  * RESIST back-off, and CLEAN decisiveness, are NOT separable. The best feature
    found (elbow path/net "wander" over the rack phase) orders correctly --
    CLEAN-keep 1.10-2.20, ep4 RESIST-void 1.94, ep6 CLEAN-void 2.38, ep2
    RESIST-keep 2.68 -- but the margin between the highest CLEAN keep (2.20) and
    the voided ep6 (2.38) is **0.18 on eight episodes**. A gate on that is a
    threshold without a control, which is a named bug class in this project.

So the auditor GATES on A2 and on structure, and REPORTS the rest as numbers
beside their observed distribution for Faith to judge. It must never imply it
checked something it cannot check -- that is what these tests enforce.
"""

import sys

import numpy as np
import pytest

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))
import v4_audit  # noqa: E402

FPS = 20


def _traj(events):
    """Build a (T, 18) state array from (duration_s, pan, grip, wrist_flex) segments.

    wrist_flex matters: the rack phase is `wrist_flex < 20 and pan < 40`, which
    is how the real episodes look -- the wrist swings from ~73 at the pick site
    to ~-5 to present the tube at the funnel. A fixture that holds it constant
    has no rack phase at all.
    """
    rows = []
    for secs, pan, grip, wflex in events:
        for i in range(int(secs * FPS)):
            r = np.zeros(18)
            r[0], r[5], r[3] = pan, grip, wflex
            r[1] = -105.0
            # give the elbow a little motion so wander() is defined
            r[2] = 85.0 - 0.05 * i
            rows.append(r)
    return np.asarray(rows)


#                     secs  pan  grip  wflex
CLEAN_LIKE = _traj([(1, 0, 1, 73), (2, 65, 34, 73), (1, 68, 14, 73),
                    (2, 15, 13, -5), (1, 15, 30, -5), (1, 0, 2, 73)])
NUDGE_BAD = _traj([  # graze with the jaws CLOSING -- the ep-9 defect
    (1, 0, 1, 73), (1.5, 65, 34, 73), (3, 66, 2, 73), (1.5, 66, 35, 73),
    (1, 68, 14, 73), (2, 15, 13, -5), (1, 15, 30, -5), (1, 0, 2, 73),
])
NUDGE_GOOD = _traj([  # graze with the jaws OPEN throughout the approach
    (1, 0, 1, 73), (1.5, 65, 34, 73), (3, 66, 33, 73), (1.5, 66, 35, 73),
    (1, 68, 14, 73), (2, 15, 13, -5), (1, 15, 30, -5), (1, 0, 2, 73),
])


# --- the A2 detector -------------------------------------------------------

def test_a_closing_jaw_graze_is_detected():
    cycles, spans = v4_audit.jaw_reopen_cycles(NUDGE_BAD[:, 0], NUDGE_BAD[:, 5])
    assert cycles == 1, "the ep-9 defect (close on the way in, then reopen) was not seen"
    (start, end), = spans
    assert 2.5 <= end - start <= 3.5, f"span {end - start:.1f}s should be the ~3 s closure"


def test_an_open_jaw_graze_is_clean():
    assert v4_audit.jaw_reopen_cycles(NUDGE_GOOD[:, 0], NUDGE_GOOD[:, 5])[0] == 0


def test_a_plain_approach_and_grasp_is_clean():
    assert v4_audit.jaw_reopen_cycles(CLEAN_LIKE[:, 0], CLEAN_LIKE[:, 5])[0] == 0


def test_the_jaw_cycle_ignores_the_release_at_the_rack():
    """The jaw opens at the rack every single episode. That is not a graze."""
    assert v4_audit.jaw_reopen_cycles(CLEAN_LIKE[:, 0], CLEAN_LIKE[:, 5])[0] == 0, (
        "the rack release was counted as a pick-site jaw cycle"
    )


# --- the gate --------------------------------------------------------------

def test_nudge_with_closing_jaws_FAILS():
    r = v4_audit.audit(NUDGE_BAD, "NUDGE")
    assert not r.ok, "a closing-jaw NUDGE passed -- amendment A2 is not enforced"
    assert any("A2" in f for f in r.failures), r.failures


def test_nudge_with_open_jaws_PASSES():
    r = v4_audit.audit(NUDGE_GOOD, "NUDGE")
    assert r.ok, r.failures


def test_a_clean_episode_is_not_judged_against_the_nudge_rule():
    """Judge each episode against ITS OWN type. A CLEAN has no graze to check."""
    r = v4_audit.audit(CLEAN_LIKE, "CLEAN")
    assert r.ok, r.failures
    assert not any("A2" in f for f in r.failures)


def test_a_zero_frame_episode_FAILS_loudly():
    """The arrow-key crash writes an empty episode; it must never read as fine."""
    r = v4_audit.audit(np.zeros((0, 18)), "CLEAN")
    assert not r.ok and any("frame" in f.lower() for f in r.failures), r.failures


def test_a_six_dim_episode_FAILS():
    """v4 must be 18-dim (6 pos + 6 load + 6 current). A 6-dim take is unusable
    for the load-sensing A/B and can never be retrofitted."""
    r = v4_audit.audit(CLEAN_LIKE[:, :6], "CLEAN")
    assert not r.ok and any("18" in f for f in r.failures), r.failures


# --- the tool must admit what it cannot check ------------------------------

def test_graze_types_are_gated_on_the_approach_signature():
    """Sep 5 morning: 11, 13, 15, 17 gave the calibration (5.7-7.1 s open hover
    vs 1.4-2.3 s for every direct approach). A CLEAN-shaped trajectory under a
    graze label must FAIL, not pass, and the ROT magnitude is still called out
    as unmeasured."""
    for t in ("LEFT-MIMIC", "ROT-20°", "ROT-45°", "NUDGE"):
        r = v4_audit.audit(CLEAN_LIKE, t)
        assert not r.ok and any("GRAZE NOT REGISTERED" in f for f in r.failures), (t, r.failures)
        good = v4_audit.audit(NUDGE_GOOD, t)
        assert good.ok, (t, good.failures)
    assert "MAGNITUDE" in v4_audit.audit(NUDGE_GOOD, "ROT-45°").summary


def test_a_direct_label_on_a_graze_trajectory_is_flagged_as_a_LABEL_mismatch():
    """The Sep 5 mix-up, in miniature: ep 11 was a LEFT-MIMIC judged as a RESIST."""
    for t in ("CLEAN", "RESIST"):
        r = v4_audit.audit(NUDGE_GOOD, t)
        assert any("LABEL MISMATCH" in f for f in r.failures), (t, r.failures)


def test_the_open_jaw_rule_now_covers_ROT_and_LEFT_MIMIC():
    """A1 (LEFT-MIMIC) and A5 (ROT, Faith's Sep 5 method) are the same rule as A2."""
    for t, rule in (("ROT-20°", "A5"), ("ROT-45°", "A5"), ("LEFT-MIMIC", "A1"), ("NUDGE", "A2")):
        r = v4_audit.audit(NUDGE_BAD, t)
        assert any(f"{rule} VIOLATED" in f for f in r.failures), (t, r.failures)


def test_resist_backoff_is_reported_but_NOT_gated():
    """The wander margin is 0.18 on 8 episodes. Reporting it is useful; gating
    on it would be a threshold without a control."""
    r = v4_audit.audit(CLEAN_LIKE, "RESIST")
    assert r.uncalibrated, "RESIST back-off is being gated on an unvalidated threshold"
    assert "wander" in r.summary.lower() or "back-off" in r.summary.lower(), r.summary


# --- regression against the real recorded episodes -------------------------

def test_the_auditor_reproduces_the_human_calls_on_the_real_dataset():
    """The eleven episodes on disk are the only ground truth that exists.

    The auditor must flag ep 9 on A2 and must NOT flag the seven kept episodes.
    It is NOT required to reproduce the ep 4 / 6 / 10 voids -- those rest on
    criteria this tool explicitly does not gate, and claiming otherwise is the
    failure mode this file exists to prevent.
    """
    ds = v4_audit.load_dataset()
    if ds is None:
        pytest.skip("v4 dataset not on this machine")
    states, manifest = ds
    a2_flagged = set()
    for ep, st in states.items():
        typ = manifest.type_of(ep)
        r = v4_audit.audit(st, typ)
        if any("A2" in f for f in r.failures):
            a2_flagged.add(ep)
    assert 9 in a2_flagged, "ep 9's closing-jaw graze is no longer detected"

    # The label cross-check must be QUIET on the true labels...
    label_flags = {ep for ep, st in states.items()
                   if any("LABEL MISMATCH" in f or "GRAZE NOT REGISTERED" in f
                          for f in v4_audit.audit(st, manifest.type_of(ep)).failures)
                   and ep not in manifest.voided}
    assert not label_flags, f"label cross-check fires on kept episodes {sorted(label_flags)}"
    # ...and LOUD on the labels the manifest wrongly carried for 11-19 on Sep 5 morning.
    wrong = {11: "RESIST", 12: "CLEAN", 13: "NUDGE", 14: "CLEAN", 15: "LEFT-MIMIC",
             16: "CLEAN", 17: "ROT-20°", 18: "CLEAN", 19: "ROT-45°"}
    caught = {ep for ep, t in wrong.items() if ep in states
              and any("LABEL MISMATCH" in f or "GRAZE NOT REGISTERED" in f
                      for f in v4_audit.audit(states[ep], t).failures)}
    assert {11, 19} <= caught, (
        f"the cross-check would NOT have caught the Sep 5 mix-up (caught {sorted(caught)}); "
        f"11 was a LEFT-MIMIC judged as RESIST and 19 a CLEAN judged as ROT-45"
    )
    kept = [e for e in states if e not in manifest.voided]
    assert not (a2_flagged & set(kept)), (
        f"the A2 gate fires on kept episodes {sorted(a2_flagged & set(kept))} -- false positives "
        f"would make Faith re-record good takes"
    )


def test_the_A2_detector_has_a_CONTROL_and_does_not_fire_on_v3():
    """False-positive control on a dataset the detector was never tuned on.

    "A threshold without a control" is a named bug class here, and the A2 gate
    is the one behavioural check this tool actually gates on — so it needs one.
    v3 (50 episodes, Aug 27, hand-staged recovery, also 18-dim) is independent
    of every number in v4_audit.py.

    Measured Sep 5: **0/50 v3 episodes fire.** With 1/1 on the true positive
    (v4 ep 9) and 0/10 on the other v4 episodes, that is zero false positives in
    sixty episodes. If this ever starts firing on v3, the pick-site/jaw
    thresholds have drifted and the gate is no longer specific — do not raise
    the threshold to silence it without re-deriving from the data.
    """
    import glob
    import os

    import numpy as np
    import pandas as pd

    root = os.path.expanduser(
        "~/.cache/huggingface/lerobot/faithqin/so101-tube-insert-v3_20260827_110242")
    pqs = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    if not pqs:
        pytest.skip("v3 dataset not on this machine")
    df = pd.concat([pd.read_parquet(p) for p in pqs])
    fired = []
    for e, g in df.groupby("episode_index"):
        st = np.stack(g["observation.state"].values)
        if v4_audit.jaw_reopen_cycles(st[:, 0], st[:, 5])[0]:
            fired.append(int(e))
    assert not fired, (
        f"the A2 gate fires on {len(fired)}/{df['episode_index'].nunique()} ordinary v3 "
        f"episodes {fired[:10]} — it is no longer specific to a closing-jaw graze, and "
        f"would make Faith re-record good takes"
    )
