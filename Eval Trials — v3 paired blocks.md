# Eval Trials — ACT-A-v3 vs ACT-B-v3, paired blocks (pre-declared Aug 29, 2026)

Scored trials only. Probes and protocol-validation episodes are NOT in here —
they live in the the lab notebook (private). Rules: `Eval Protocol — ACT AB.md` plus the v2
amendments in `Eval Trials — v2 paired blocks.md` (scoring taxonomy inherited
verbatim: Stage R/G/T/Rel/S × Mechanism NG/PECK/SLIP/DT/JAM/KR/INFRA), plus the
Aug 29 amendments below.

**Written before trial 1. Nothing below this line changes once scoring starts.**

---

## PRE-REGISTRATION (declared Aug 29, 2026, ~22:30 — before any scored trial)

### The question

**Primary:** does adding servo load + current to the observation space improve
contact-rich insertion, measured on the v3-targeted dataset?
**v3-A** = vision + 6-dim joint positions. **v3-B** = + load + current (18-dim).

⚠️ **Correction to the Morning Report (Aug 29).** That document labels v3-A vs
v3-B as attributing "the data intervention." It does not. It holds data constant
at v3 and varies the observation space, so it attributes **load sensing** — the
project's locked thesis. The *data* intervention is v3-A vs v2-A, which is
available only ACROSS sessions against the existing 40-trial v2 table, unpaired,
and is reported as a caveated secondary (see "Secondary" below).

### Arms and n — FIXED

| Arm | Policy | n | Pairing |
|---|---|---|---|
| **Primary** | v3-A vs v3-B | **40 pairs (80 trials)** | paired, same start position, back-to-back |
| **System** | v3-A + grasp lock | **20 trials** | unpaired, labelled as the stacked system |
| ~~Stretch~~ | ~~v2-A + lock~~ | — | **dropped Aug 29** |

**Scope chosen by Faith (Option C) on Aug 29** over a 20-pair alternative: the
larger primary buys the power to call a null honestly rather than reporting
"underpowered".

### Run order — degrades gracefully

- **Day 1:** primary pairs 1–20, then the full 20-trial lock arm.
- **Day 2:** primary pairs 21–40.

Rationale: if hardware stops the study on Day 2, what is already in hand is a
complete 20-pair primary **plus** the complete lock arm. The extension is pure
upside; a stop costs pairs, not the differentiator.

### Conditions — declared, not negotiable mid-study

1. **Split by PAIRS, never by arm.** Each pair runs v3-A then v3-B
   back-to-back on the same start position. Splitting A-Day-1 / B-Day-2 would
   confound the contrast with across-day drift and destroy the comparison.
2. **Fixed n. No adding trials after peeking** (STEP, arXiv:2503.10966).
3. **Stop rule:** a third elbow bus-drop ends the block. Report the truncation
   and its cause. A hardware stop is not a peek and does not compromise the
   pre-registration.
4. **Episode length 45 s** (`--dataset.episode_time_s=45`), matching the locked
   protocol's "Max 45 s" and the v2 scored trials (875 frames ≈ 45 s). Success
   is defined *at rollout end*, so episode length is part of the definition.
   **Early-stop (`n` key) is forbidden in scored trials** — it silently
   redefines success. It remains fine in probes.
5. **Inference config: PENDING BLOCK 2.** v2's locked configs were A n=25 /
   B n=50, chosen on a pre-declared probe grid. v3 has never run on hardware.
   Block 2 selects the config by the same rule; **absent a reason to differ,
   default to v2 parity (A n=25, B n=50)**. Whatever Block 2 returns is written
   here before trial 1.

### Analysis — declared in advance

- **Exact mid-p McNemar** on discordant pairs for the primary.
- **Wilson 95% CIs** per arm, reported alongside — per-arm CIs overlap even for
  a doubling, so the paired test is what carries significance.
- Power note for the write-up, citable to Kress-Gazit et al.
  (arXiv:2409.09491): the study is powered for large effects; a null does not
  exclude effects of ≤15 points.
- **Lock arm reported intention-to-treat**: all 20 trials regardless of whether
  the lock engaged. Excluding zero-engagement trials would select on an
  outcome-correlated variable. **Engagement count and intervention count are
  logged per trial as secondary descriptives** — Block 1 showed these differ
  (see below).

### Secondary (labelled as such, not pre-registered as primary)

v3-A vs the Aug 25 v2-A table (6/20). **Across-session and unpaired.** The
Aug 29 probes measured v2-A at 8/10 pooled versus that 30% baseline
(Fisher p = 0.019), which — whatever its cause — is direct evidence that
session-level shifts here can be large. Report this comparison with that
caveat attached, or not at all.

---

## Gates — every one ABORTS the launch, in code (`tools/preflight.py`)

| Gate | Threshold | Why |
|---|---|---|
| Tube placement | \|d\| ≤ 8 px vs reference | Aug 29: an episode ran at 10.2 px because the value was printed, not enforced |
| Home pose | every joint inside v2 training frame-0 range; elbow **[75.5, 84.4]** | Aug 29: elbow at 74.1° (below the 75.47 floor) made a *deterministic replay* miss the tube entirely |
| Elbow temperature | ≤ 60 °C | Two bus-drops on the ledger; 69 °C brownout, published misbehavior at 71 °C, no trustworthy built-in shutdown |
| Start state | episode's own frame 0 must not show the cap in the rack region | Aug 29: one episode began with the tube already seated and was reset ~7 s into recording |

**The home gate is expressed in TRAINING SUPPORT, not in a residual.**
`home_arm`'s flat 2.5° tolerance is *tighter than the training data's own
spread* (elbow frame-0 sd = 2.20) yet says nothing about whether the pose is one
the policy ever saw. Bounds are the per-joint frame-0 min/max over all 50 v2
episodes; `tests/test_home_gate.py` re-derives them from the parquet so they
cannot go stale. Signed off by Faith, Aug 29.

## Scoring

Two-pass, as in v2: Faith scores live; **video is ground truth on disagreement.**
Amended Aug 29 — the video pass is now automated and synchronized:

- `tools/score_episode.py` scores the **episode's own recording**, not a live
  camera grab. The old live check fired *after* the reset phase, so it could
  photograph a scene the operator had already reset.
- It reports **seat runs**, so a 0.1 s cap-passing-over-the-funnel is
  distinguishable from a seat that held. The old checker scored a cap held in
  the gripper above the funnel as SEAT.
- An episode that **begins** seated is INFRA and carries no verdict at all.

Faith retains Stage / Mechanism / Notes. The run id and forensic
columns are filled separately. **Faith's live observation outranks the harness**: three silent harness
failures were caught by her eye on Aug 27, and two more on Aug 29.

## Prior on the lock arm (Block 1, Aug 29 — probe, not scored)

ON 5/5 vs OFF 3/5, **Fisher p = 0.444 — no effect demonstrated.** And the lock
*intervened* in only 3 of the 5 ON trials; two registered closing onsets with
zero chunk extensions and succeeded anyway. Chronologically the last 6 trials all
seated regardless of arm, with both OFF failures at cold moments — consistent
with warm-up rather than the lock. **The lock arm is run as an honest test of an
undemonstrated mechanism, not as a headline-number arm.**

Characterised regimes: inert when the policy never commands a close (0 onsets);
protects BAD closes during peck loops (15 onsets / 6 extensions on a discarded
trial).

---

## Trials

| # | Pair | Arm | Policy | n | Start condition (gates) | Stage (live) | Stage (video, FINAL) | Mechanism | Notes | Run id | onsets/extends |
|---|------|-----|--------|---|--------------------------|--------------|----------------------|-----------|-------|--------|----------------|
| | | | | | | | | | | | |

## INFRA log (re-run, not counted)

| # | Arm | Cause | Resolution |
|---|-----|-------|------------|
| | | | |
