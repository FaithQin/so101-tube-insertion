# Eval Trials: ACT-A-v2 vs ACT-B-v2, paired blocks (started Aug 23, 2026)

> **Note (Sep 22 2026).** This sheet is the dated record of the v2 evaluation as it was written on Aug 24 and 25, 2026. Its stop note quotes the claim allowed at the time. The later pre-registration for v4 forbids that wording, and the allowed form is "we can rule out an improvement larger than X pp at 95%". The README uses the allowed form. One process line under the scoring method named who summarized the live notes and was edited in this copy.

Scored trials only. Probes and protocol-validation episodes are NOT in here
(they live in the lab notebook (private)). Rules: `eval-protocol-act-ab.md` v2 plus
the amendments below.

## Amendments (dated, apply to BOTH policies identically)

**Aug 23:**
1. **Seat-primed start.** Before every block: replays of v2 episode 0 until one
   SEATS. After any failed episode: re-prime the same way before the next trial.
   Basis: seat-primed 5/6 vs unprimed 0/6 for ACT-A-v2 (the lab notebook (private)).
2. **Episodes are launched one at a time** (`home_arm.py` + a 1-episode run),
   so re-priming never interrupts a multi-episode process.
3. **Dot column = the single v2 training mark**. The pairing structure
   (AB, BA, ...) is kept.
4. **Configs:** ACT-A n=25, ACT-B n=25 (valid primed probes tied 0/2 vs 0/2.
   Tie rule + shared horizon removes the horizon×observation confound).

**Aug 24 (scoring methodology, decided before trial 1, the pre-registration):**
5. **Two-pass scoring instrument.** Pass 1: Faith scores live as the trial
   runs. Pass 2: Faith re-scores from the recorded videos (both cameras are
   recorded for every trial). **Video is ground truth**. On disagreement the
   video score stands. All videos are archived in the local dataset cache and
   independently re-scoreable.
6. **Two-axis scoring** replaces the single Category:
   - **Stage reached** (ordinal, cumulative). Operational definitions
     (added Aug 25 at Faith's request, before trial 1 was scored):
     - **R** reached: jaws entered the grasp zone (within ~3 cm / one cap
       width of the tube). Contact NOT required. A touch without a grasp is
       still R.
     - **G** grasped: tube held in closed jaws and lifted off the mat.
     - **T** transported: carried into the rack area.
     - **Rel** released: let go into/over the funnel.
     - **S** seated: standing in the hole unsupported at episode end.
   - Pass-1 Notes are dictated by Faith live and summarized before
     entry. The dictation is the source, the summary is the record.
   - **Failure mechanism** (one per trial. If several occurred, the one at
     the latest stage wins). Operational definitions:
     - **NG** never grasped: clean miss(es), without the peck loop. A
       touch, nudge, or push where the tube was never HELD is still NG
       (ruled Aug 25, trial 8): SLIP requires the jaws to have closed
       around the tube and taken hold, however briefly.
     - **PECK** the stall/peck loop: ≥3 descend/withdraw (or open/close)
       cycles without a grasp, occupying the rest of the episode (the
       "woodpecker". Stage is usually R).
     - **SLIP** tube leaves the jaws before transport begins (during the
       grasp or lift, still at the pick site).
     - **DT** tube leaves the jaws during transport (after leaving the pick
       zone, before rack contact).
     - **JAM** insertion fails after rack contact: jamming/wedging in
       Whitney's sense, including drops caused by rack contact.
     - **KR** rack or tube knocked over/displaced so the task is
       unachievable, as the PRIMARY event (ruled Aug 25, trial 17):
       displacement that occurs as a side effect of a grasp or insertion
       failure is captured by that failure's mechanism (SLIP/NG/JAM), never
       double-counted as KR.
     - **INFRA** infrastructure corrupted the trial (camera dropout, servo
       fault/latch, harness error). An INFRA trial does not count against
       the policy: it is logged in the footnote and RE-RUN.
     - **-** for successes.
7. **Metrics, in order:** PRIMARY = success rate (S / N) per policy with
   Clopper–Pearson 95% CIs. A vs B compared by McNemar's test on the matched
   pairs. SECONDARY = cumulative per-stage percentages (%R, %G, %T, %Rel, %S
   of all trials, ACT-paper style). TERTIARY = average progress score with
   π0/Chen weights (R .2 / G .4 / T .7 / Rel .8 / S 1.0), derived from Stage,
   reported as a dashboard number only, never load-bearing.

**Aug 25, ~21:30. EVAL STOPPED AT 40 TRIALS (8 of 16 pre-registered blocks), by
PI decision, documented before any further analysis:** the A/B contrast cannot
reach significance at the observed effect size (A 6/20, B 5/20, discordant
pairs 3–2) even at the full 80, and the v3 critical path (recording, training,
eval, all before a Sep 7–14 absence [as believed on Aug 25. The actual Taiwan
return is **Sep 11** (corrected Sep 2, 2026 by Faith). The stop decision stands
on its own reasoning. Only the date was wrong]) takes priority. Blocks 1–8
constitute the
v2 baseline. Claims from this table are limited to: "no difference detected.
The CI on the A−B gap spans approximately −25 to +33 points."

**Forensic column definitions (Aug 25, at Faith's request):**
- *grasp-close t*: first tick where the commanded gripper crosses from open
  (>15) to closed (<8), with the commanded lift/elbow at that tick and the
  offset from the demo grasp median (lift +37.1).
- *home-return t*: first tick after t=2 s where the commanded shoulder_lift
  drops below −95° ≈ arrival at the folded home posture. Read from the
  recorded action stream, so it is INDEPENDENT of episode length and of when
  (or whether) `n` is pressed. Derived from commanded joints (commanded and
  measured track within 1–3°, ≲0.1 s arrival difference). "never" = home was
  never commanded. On successes this doubles as task-completion time: clean
  single-cycle runs land ~9 s, runs with a recovery cycle land ~20 s. The
  spread is task duration, not measurement error.

## Trial sheet

| # | Block | Policy | n | Start condition | Stage (live) | Stage (video, FINAL) | Mechanism | Notes | Run id / ep | grasp-close t | home-return t |
|---|-------|--------|---|-----------------|--------------|----------------------|-----------|-------|-------------|---------------|---------------|
|  1 | 1 | ACT-A | 25 | replay seat 16:39, home_arm, scene 1.8 px/0.0° | R | | PECK | First close overshot left of cap. Open-jaw undulation. One more close attempt, still left. Froze open, left of cap. | trial_A_v2_20260825_164252 / ep0 | 2.4 s (lift +41.2, 4.1° high) | never |
|  2 | 1 | ACT-A | 25 | replay seat 16:47, home_arm, scene 0.4 px/+0.7° | G (marginal) | | SLIP | Jaws caught only the cap's top edge. Tube slipped, landed ~45° rotated. Re-descended and closed again (real recovery attempt, no peck) but couldn't handle the new angle. | trial_A_v2_20260825_165236 / ep0 | 2.8 s (lift +39.6, 2.5° high) | never |
|  3 | 1 | ACT-A | 25 | replay seat 16:55, home_arm, scene 0.9 px/0.0° | G (marginal) | | SLIP | Same as trial 2: caught cap tip only, slipped out to ~45° during close. Retried (down-close-up), no peck, couldn't recover. | trial_A_v2_20260825_165728 / ep0 | 2.4 s (lift +35.8, 1.3° low) | never |
|  4 | 1 | ACT-A | 25 | replay seat 16:59 (3 passes), home_arm, scene 1.6 px/0.0° | G (marginal) | | SLIP | Same family as 2-3: tip grasp, knocked tube ~20°. Retries consistently land slightly LEFT of cap. Couldn't recover. | trial_A_v2_20260825_170044 / ep0 | 2.2 s (lift +39.6, 2.5° high) | never |
|  5 | 1 | ACT-A | 25 | replay seat 17:05, home_arm, scene 0.6 px/+0.5° | S | | - | Approached slightly right, self-corrected up into the grasp. Two repositions at the rack, then slotted cleanly. | trial_A_v2_20260825_170703 / ep0 | 2.5 s (lift +37.9, 0.8° high) | 8.8 s |
|  6 | 2 | ACT-B | 25 | replay seat 17:10 (pass 1), home_arm, scene 2.6 px/0.0° | S | | - | Grasped first try (despite a 3° high close). Rack resistance on first insert attempt, modulated force without losing the tube, corrected, seated. | trial_B_v2_20260825_171233 / ep0 | 2.6 s (lift +40.1, 3.0° high) | 9.8 s |
|  7 | 2 | ACT-B | 25 | success-primed (trial 6 seat + replay seat 17:14), home_arm, scene 0.8 px/+0.6° | R | | NG | Repeated approaches: nearly right angle, then swung too far left each time. Never touched the tube (video pass: check PECK criterion, 14 reversals). | trial_B_v2_20260825_171626 / ep0 | 2.7 s (lift +39.3, 2.2° high) | never |
|  8 | 2 | ACT-B | 25 | re-primed (replay seat 17:19), home_arm, scene 0.9 px/0.0° | R | | NG | First grasp too far left, pushed tube slightly (never held). Repeated re-grasp attempts on good trajectories, each overcorrecting left. No peck. | trial_B_v2_20260825_172047 / ep0 | 2.5 s (lift +43.1, 6.0° high) | never |
|  9 | 2 | ACT-B | 25 | re-primed (replay seat 17:24), home_arm, scene 1.4 px/+1.0° | R | | SLIP-or-NG (video decides) | Several left-misses (up-down-close-redirect). Then contact at the tube's top, tube escaped and rolled into quadrant E. Arm kept grasping at empty quadrant D afterward. | trial_B_v2_20260825_172631 / ep0 | 2.4 s (lift +44.5, 7.4° high) | never |
| 10 | 2 | ACT-B | 25 | re-primed (replay seat 17:36), home_arm, scene 1.6 px/-0.4° | R | | NG | First attempt left of cap. Retries on good trajectories but always overcorrects left. Jaws fully open/re-close. Touched and nudged, never held. | trial_B_v2_20260825_173757 / ep0 | 2.5 s (lift +42.0, 4.9° high) | never |
| 11 | 3 | ACT-B | 25 | block-start replay seat 17:50, home_arm, scene 1.1 px/+0.9° | R | | SLIP | Jaw opens on descent, repeatedly overshoots even on on-target trajectories. Got the cap between the pads once, rotated the tube off 180° and lost it. No recovery. | trial_B_v2_20260825_175218 / ep0 | 2.9 s (lift +38.5, 1.4° high) | never |
| 12 | 3 | ACT-B | 25 | re-primed (replay seat 18:01), home_arm, scene 0.5 px/-0.5° | R | | NG | Right trajectory each descent, then jerks left at the moment of clamping. Brushed the cap once, never in the mouth. Always overshoots left. | trial_B_v2_20260825_180406 / ep0 | 2.6 s (lift +40.0, 2.9° high) | never |
| 13 | 3 | ACT-B | 25 | re-primed (replay seat 18:05), home_arm, scene 2.2 px/0.0° | R | | NG | Open/close cycles with real corrections (up, lateral nudges), no peck. Always overshoots above-left of the cap. Never touched. | trial_B_v2_20260825_180717 / ep0 | 2.4 s (lift +46.4, 9.3° high, day's worst) | never |
| 14 | 3 | ACT-B | 25 | re-primed (replay seat 18:08), home_arm, scene 2.4 px/+0.2° | S | | - | Left-jerk at the clamp again, but course-corrected, grasped. At the rack started too close in, corrected outward, seated cleanly. | trial_B_v2_20260825_181012 / ep0 | 2.4 s (lift +41.0, 3.9° high) | 20.8 s |
| 15 | 3 | ACT-B | 25 | success-primed (trial 14 seat), home_arm, scene 0.5 px/-0.9° | R | | NG | First attempt nudged the cap off 180° (never held). Then the constant cycle: correct trajectory, sudden left swing at the clamp, full close on air, reposition, repeat. | trial_B_v2_20260825_181202 / ep0 | 2.8 s (lift +42.4, 5.3° high) | never |
| 16 | 4 | ACT-A | 25 | block-start replay seat 19:17, home_arm, scene 0.3 px/+1.2° | R | | NG | Repeated grasp cycles (up-down, repositions, closes). Nudged the cap once (never held). Each attempt swings off left and overshoots. | trial_A_v2_20260825_191934 / ep0 | 2.8 s (lift +43.1, 6.0° high) | never |
| 17 | 4 | ACT-A | 25 | re-primed (replay seat 19:21), home_arm, scene 1.2 px/-0.7° | T | | JAM | Near-miss left, then GRASPED the cap. Dropped in transit and RE-GRASPED by the body (first recovery of the day). Forced insertion at a bad angle, tube hit the rack side, rolled off the quadrants. | trial_A_v2_20260825_192231 / ep0 | 2.6 s (lift +38.8, 1.7° high) | never |
| 18 | 4 | ACT-A | 25 | re-primed (replay seat 19:29), home_arm, scene 0.4 px/+2.4° | G (marginal) | | SLIP | Near-miss then left jerks. Several re-grasps too far left. Once held the cap without a firm grip, swung out of the jaws into quadrant E. No recovery. | trial_A_v2_20260825_193136 / ep0 | 2.8 s (lift +43.9, 6.8° high) | never |
| 19 | 4 | ACT-A | 25 | re-primed (replay seat 19:33), home_arm, scene 1.0 px/0.0° | S | | - | No left jerk. Precise first-try grasp of the cap. Slowed on rack approach, small trajectory adjustment, seated first try. | trial_A_v2_20260825_193516 / ep0 | 2.5 s (lift +34.1, 3.0° low) | 9.2 s |
| 20 | 4 | ACT-A | 25 | success-primed (trial 19 seat), home_arm, scene 0.7 px/-0.8° | S | | - | First-try grasp despite a 3.5° high close. At the rack positioned too far toward itself, corrected FORWARD after tube-rack contact, seated (A's first rack-contact recovery). | trial_A_v2_20260825_193715 / ep0 | 2.9 s (lift +40.6, 3.5° high) | 9.8 s |
| 21 | 5 | ACT-A | 25 | block-start replay seat 19:42, home_arm, scene 1.1 px/+1.2° | R | | NG | Perfect trajectories repeatedly ruined by the left jerk at the clamp. Micro-adjustments, open/close cycles. Touched the cap several times, never held. | trial_A_v2_20260825_194356 / ep0 | 2.9 s (lift +42.8, 5.7° high) | never |
| 22 | 5 | ACT-A | 25 | re-primed (replay seat 19:56), home_arm, scene 0.6 px/+0.9° | S | | - | First-try grasp. Re-aimed during the descent at the rack, seated. | trial_A_v2_20260825_195830 / ep0 | 2.5 s (lift +39.3, 2.2° high) | 9.2 s |
| 23 | 5 | ACT-A | 25 | success-primed (trial 22 seat), home_arm, scene 1.0 px/0.0° | S | | - | First-try grasp of the cap's top half. At the rack the tube sat slightly toward the arm, but the funnel bevel guided the push in, seated. | trial_A_v2_20260825_200024 / ep0 | 2.5 s (lift +37.5, 0.4° high) | 9.3 s |
| 24 | 5 | ACT-A | 25 | success-primed (trial 23 seat), home_arm, scene 1.5 px/+0.9° | R | | NG | On-point trajectories, left jerk at the clamp, continuous up-down. Nudged the tube rightward (making its own target harder), never held. | trial_A_v2_20260825_200216 / ep0 | 3.0 s (lift +40.4, 3.3° high) | never |
| 25 | 5 | ACT-A | 25 | re-primed (replay seat 20:04), home_arm, scene 0.4 px/+1.0° | R | | NG | Grasp landed left. Top jaw came down outside the tube and pushed it downward. Repeated up/re-angle/down cycles, never held, no recovery. | trial_A_v2_20260825_200521 / ep0 | 2.6 s (lift +40.3, 3.2° high) | never |
| 26 | 6 | ACT-B | 25 | block-start replay seat 20:06 (pass 1, post-failure, pattern's first exception), home_arm, scene 2.2 px/0.0° | R | | NG | Slow, indecisive first descent. On-point trajectories ruined by the left jerk at the clamp. Grasp-open-regrasp cycles. Never touched the tube. | trial_B_v2_20260825_200801 / ep0 | 2.8 s (lift +42.0, 4.9° high) | never |
| 27 | 6 | ACT-B | 25 | re-primed (replay seat 20:10), home_arm, scene 1.8 px/-0.8° | S | | - | Many left-jerk misses first. Eventually grasped the cap's top half, carried over, teetered on the rim, then seated. | trial_B_v2_20260825_201132 / ep0 | 2.9 s (lift +40.4, 3.3° high) | 22.6 s |
| 28 | 6 | ACT-B | 25 | success-primed (trial 27 seat), home_arm, scene 1.9 px/-0.5° | S | | - | First grasp attempt slightly left. Second spot-on. At the rack hit the top-left corner, nudged centerward after resistance, seated. | trial_B_v2_20260825_201410 / ep0 | 3.0 s (lift +40.5, 3.4° high) | 16.6 s |
| 29 | 6 | ACT-B | 25 | success-primed (trial 28 seat), home_arm, scene 1.8 px/0.0° | INFRA (re-run) | | INFRA | First grasp too low, moved the tube. Re-grasped successfully, trajectory to the rack looked perfect. Then the WRIST CAMERA DISCONNECTED mid-carry (2nd dropout in 3 days, same camera). | trial_B_v2_20260825_201626 / ep0 | 2.7 s (lift +39.7, 2.6° high) | n/a (INFRA) |
| 29b | 6 | ACT-B | 25 | re-run of 29 (INFRA), re-primed (replay seat 20:19), home_arm, scene 3.1 px/-0.5° | R | | NG | Left again. Nudged the cap once (never held). Near-perfect trajectories broken by the left jerk. Eyewitness hypothesis: the jerk is the next chunk arriving (boundary re-plan), to verify against boundary ticks. | trial_B_v2_20260825_202049 / ep0 | 2.8 s (lift +39.3, 2.2° high) | never |
| 30 | 6 | ACT-B | 25 | re-primed (replay seat 20:23), home_arm, scene 0.5 px/0.0° | R | | NG | First try overshot, top jaw nudged the tube downward. Repeated close-but-left attempts (near-perfect trajectory, then the veer). Never held. | trial_B_v2_20260825_202503 / ep0 | 2.5 s (lift +40.6, 3.5° high) | never |
| 31 | 7 | ACT-B | 25 | block-start replay seat 20:27 (pass 1), home_arm, scene 1.7 px/0.0° | R | | NG | First half (video): cap ahead-right of the pads in every frame t=2-24, never between the jaws, tube unmoved. Second half (eyewitness): same, too far left throughout. | trial_B_v2_20260825_202847 / ep0 | 2.3 s (lift +38.6, 1.5° high) | never |
| 32 | 7 | ACT-B | 25 | re-primed (replay seat 20:31), home_arm, scene 2.0 px/+3.4° | S | | - | Grasped the cap's top half first try. Guided to the rack and seated in one go. | trial_B_v2_20260825_203323 / ep0 | 2.6 s (lift +37.7, 0.6° high) | 9.4 s |
| 33 | 7 | ACT-B | 25 | success-primed (trial 32 seat), home_arm, scene 1.5 px/+0.6° | R | | PECK | Grasp attempts further left. Hovering up-down cycles, jaws closing only slightly, motion felt laggy (operator suspects heat). Never touched. | trial_B_v2_20260825_203445 / ep0 | 3.6 s (lift +43.0, 5.9° high) | never |
| 34 | 7 | ACT-B | 25 | re-primed (replay seat 20:38), home_arm, scene 1.4 px/-1.6° | R | | NG | On-trajectory then the left jerk, every attempt. Full descents miss the cap. Open/close and reposition cycles, never touched. | trial_B_v2_20260825_204025 / ep0 | 2.8 s (lift +39.3, 2.2° high) | never |
| 35 | 7 | ACT-B | 25 | re-primed (replay seat 20:42, pass 1), home_arm, scene 2.5 px/+3.4° | R | | NG | Repeatedly very close. Mid-clamp leftward nudge turns grasps into pushes. Nudged the tube out of jaw range. Never held. | trial_B_v2_20260825_204408 / ep0 | 2.9 s (lift +39.7, 2.6° high) | never |

| 36 | 8 | ACT-A | 25 | block-start replay seat 20:45 (pass 1), home_arm, scene 1.7 px/0.0° | G (marginal) | | SLIP | The left jerk at the close again. Held the cap's top quarter: weak grip, lift swung the tube into quadrant E. Then grasped at empty quadrant D. | trial_A_v2_20260825_204652 / ep0 | 2.4 s (lift +38.2, 1.1° high) | never |
| 37 | 8 | ACT-A | 25 | re-primed (replay seat 20:49), home_arm, scene 2.6 px/+0.8° | T | | JAM | First-try grasp of the cap. At the rack the tube tip angled too low, kept pushing without redirecting, tube fell out. Then grasped at the empty pick site. | trial_A_v2_20260825_205047 / ep0 | 2.7 s (lift +35.1, 2.0° low) | never |
| 38 | 8 | ACT-A | 25 | re-primed (replay seat 20:52). Pre-check saw pre-reset scene, start RETRO-CERTIFIED from frame 0 (tube 1.8 px) | S | | - | Perfect grasp. Paused at the rack to recheck trajectory, small shift, smooth seat. | trial_A_v2_20260825_205412 / ep0 | 2.5 s (lift +35.8, 1.3° low) | 9.2 s |
| 39 | 8 | ACT-A | 25 | success-primed (trial 38 seat), home_arm, scene 2.6 px/0.0° | R | | NG | Near-miss left. Each descent on-trajectory then the left jerk. Nudged the tube repeatedly, never held. | trial_A_v2_20260825_205622 / ep0 | 2.5 s (lift +35.5, 1.6° low) | never |
| 40 | 8 | ACT-A | 25 | re-primed (replay seat 20:58, pass 1), home_arm, scene 2.0 px/+0.7° | R | | NG | First grasp: top jaw pushed the tube into a ~45° spin (never held). Repeated re-grasps at the rotated tube, closes on air. No recovery from non-180° pose. | trial_A_v2_20260825_205929 / ep0 | 2.2 s (lift +36.2, 0.9° low) | never |
(Blocks 5–16 follow the protocol's AB/BA alternation. Rows added as reached.)
Voided runs (not trials, run before the Aug 24 instrument was pre-registered,
operator absent, voided at Faith's direction): `rollout_trial_A_v2_20260824_230210`
ep0, Aug 24 23:02. Video shows two off-center closes, tube batted ~15 cm,
pecking to 45 s. Kept as diagnostic only.
INFRA re-runs: trial 29 re-run after the Aug 25 20:16 wrist-camera dropout (`rollout_trial_B_v2_20260825_201626` kept as the INFRA record).
