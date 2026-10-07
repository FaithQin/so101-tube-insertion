# Start Schedule — v4 (position D), seeded 20260904 — amended by Faith before ep 0

**Written Sep 4, 2026, before episode 0.** Derived from `Methodology — recording protocols from the literature.md` §6, amended by Faith at 17:20 (v3's four recovery situations return at v3's counts, but the **arm** creates each bad state mid-episode; BACKWARD dropped — the coned tube self-centres; **RESIST** added) and at 19:05 (a held-out evaluation set, sized so training matches v3 exactly). v4 is arm B of the staged-vs-in-trajectory recovery-data comparison; v3 is arm A.

**58 episodes recorded · 50 trained on · 8 held out (14%).**
Recorded: 33 CLEAN + 6 ROT-20° + 4 ROT-45° + 5 NUDGE + 5 LEFT-MIMIC + 5 RESIST.
**Trained on: 30 CLEAN + 20 recovery = 50 — the same 30 + 20 split as v3**, so the v3↔v4 comparison carries no data-quantity confound. The reserve is what pushes the recording count to 58.
Seeded interleave: eps 0–1 CLEAN to warm in; no two non-CLEAN adjacent.

**Hold-out reserve — ⭐ in the table: 8 whole episodes, one from EVERY category plus 3 CLEAN.** Never from eps 0–1, never from the last four (those are recorded most fatigued and would bias the validation number). Recorded exactly like every other episode; the split happens at training time.

## Rules (declared before episode 0)

1. **ONE grasp strategy, held for all 58 (confirmed by Faith 17:15):** grip the **cap**; cap **image-left** on the mat; approach **from above, jaws open, close on the cap.** (Demo-SCORE: two *successful* grasp styles cost 15–35%.)
2. **Decisiveness rule KEPT for CLEAN episodes and for every nominal segment** (Chen et al. 2025: 90% of ACT's real insertion failures involve idling learned from recorded pauses). A hesitant CLEAN take → `r`, redo. v3 was right about this; it stays.
3. **Every bad state is created by the arm, mid-episode — never placed by hand.** NUDGE and ROT: the approach grazes the tube, then the demo notices and recovers. LEFT-MIMIC: unchanged from v3 (already in-trajectory). RESIST: the tube is carried to the rack and brought down at a wrong position or angle so the tip meets resistance; the operator feels it push back through the leader, backs off, re-aligns, inserts. Every episode continues to the seat (VBT; no policy is in the loop, so RaC's end-after-correction rule does not apply). Hesitation is permitted ONLY in the segment between the induced error and the correction — that is the behaviour v3 deleted.
4. **Magnitudes:** NUDGE ≈ **1.0 cm image-right** — ACT-v3's own measured first-contact displacement on non-grasp contacts (0.7 / 1.0 / 1.7 cm on B2-B-01, B2V-B-01, B2R-A-02; Sep 4; DART: match the policy's error). ROT ≈ 20° / ≈ 45° about the cap, as v3. RESIST: enough lateral/angular miss that the tip does not enter (a rim landing, or a tilt that jams the mouth) — the point is the felt resistance, then the correction.
5. **Slow, deliberate pre-contact alignment; do not rush the last 2 cm** (FACTR 2). A small consistent alignment at the rack mouth before the push (Qiao 1996; Park 2020).
6. **RESIST is the category that tests the B thesis.** Force information carries nothing before overlap (InsertionNet) and everything after (FMB 2/25 → 11/25 with F/T). Audit each RESIST episode for a load/current excursion at the rack before the seat and a back-off in shoulder_lift/elbow — the signature ACT-B/SmolVLA-B can learn from and ACT-A cannot see.
7. **Episode end as v2/v3: release into the funnel, jaws close, then reset.** Kept for comparability with the v3 policies the eval infrastructure is built on.
8. **Cameras never move.** If one moves, the dataset restarts (Xie et al.). Check the wrist-camera cable slack before ep 0 — it has loaded `wrist_flex` twice this week.
9. **Every episode from `home_arm.py --pose v3`**; `scene_check` ≤ 8 px before every episode (the tube starts on the mark for all types now).
10. Hard stop if hesitation creeps into CLEAN takes from fatigue. Temps ≤ 52 °C; elbow ≥ 60 °C aborts. **Do not scroll the recording Terminal** (scroll → phantom arrow keys).

## Amendments — declared Sep 4 22:50, after eps 0–10 and before any redo

**A1. LEFT-MIMIC: the jaws stay OPEN through the hover.** Approach left of the
cap, one beat of hover, correct rightward, then ONE decisive close on the cap.
Verified against your own v3 practice: all five v3 LEFT-MIMIC episodes (18, 32,
35, 42, 43) recorded **zero jaw closes before the grasp**, pan overshoot
1.3–4.0°. This is not a new rule, it is the existing one written down.
*Why it matters:* a close-on-air teaches "attempt, miss, retry" — which is the
woodpecker, the exact failure the policies already produce (6–23 pre-grasp
closes, never recovering). Teaching it deliberately would be actively harmful.

**A2. NUDGE: graze with OPEN jaws on the way in, and never vary it.** The
displacement must come from an open-jaw graze during the approach, not from
closed jaws pushing the tube. Hold this for all five NUDGE episodes (rule 1;
Demo-SCORE: two *successful but different* styles cost 15–35% absolute).
*Why open:* the nudge exists to reproduce what the POLICY does wrong, and the
policies approach jaws-open (~33) and close on the cap — so an open-jaw graze is
the displacement mode they actually cause. A closing-jaw nudge also inserts an
extra jaw-close event that both the taxonomy and the policy read as a grasp
attempt, muddying the signal the category exists to teach.
*Origin:* v4 ep9 reached the right magnitude (1.24 cm) with a genuine second
approach, but delivered it by closing the jaws and pushing over ~3 s. Voided for
consistency, not for failure.

**A3. Voided episodes are never deleted.** `tools/v4_manifest.json` (built by
`tools/v4_manifest.py`, TDD) is the record of which recorded index carries which
declared type. A bad take is voided; its type returns to a redo queue and is
recorded again at a later index; the voided episode stays in the dataset and is
excluded at training time by `--dataset.episodes`, exactly like the reserve.
**The schedule's positional index→type mapping stops being true the moment a
void happens — read the manifest, not the table, for what a recorded index is.**

**A5. ROT-20° / ROT-45°: turn the tube with the HEAD, jaws OPEN, then come back
up, adjust, and regrasp.** Declared by Faith at the bench Sep 5, before ep 13 (the
first ROT recorded). The rotation is produced by grazing the cap with the gripper
head — *not* by closing the jaws on it — because a close-on-air teaches the
woodpecker (same reasoning as A1 and A2). Then lift clear, adjust to the new
angle, ONE decisive close on the cap, insert, release. Verified on the first
three: eps 13, 15, 17 all hold the jaws open 6.3–7.1 s through the graze with zero
closes, pan excursion 13–22°. `tools/v4_audit.py` enforces the open jaws (rule
A5) and the graze itself (open hover ≥ 4 s); the rotation *magnitude* (20 vs 45)
is not measured and is judged by eye.

**A6. Follow the NEXT-UP list, never this table by row number.** Sep 5: nine
episodes were recorded against this table's rows while the manifest had 11–14
pencilled in as redos of 4/6/9/10 — nine wrong labels, one good CLEAN voided,
and the redos reported done when none had been recorded. The positional index in
this table stops being the recorded index at the FIRST void (A3). Before each
episode the type is spoken and printed by `tools/v4_announce.py`; the
authoritative list is `python tools/v4_next_up.py`. The dataset itself carries no
per-episode type — `tools/v4_manifest.json` is the ONLY label, so it is the only
thing the training split and the write-up's 30 + 20 accounting can come from.

**A4. Hold-out indices are re-selected at the END**, against the final manifest,
preserving the design: one of every recovery category plus 3 CLEAN. The indices
listed earlier in this document were chosen against the original positional
mapping and no longer carry those types.

## Record command (v3's, verbatim, repo → v4, 58 episodes)

⚠️ **`tools/record_arrow_safe.py`, not bare `lerobot-record`** (Sep 5). Identical
arguments; it installs the phantom-arrow guard first. `lerobot-record` falls back
to terminal keyboard input without an Accessibility grant, and a raw-mode terminal
turns mouse-wheel scroll into arrow escape sequences — a forged Right arrow during
the 15 s reset saves a **zero-frame episode and crashes the run**. It has cost
three episodes. Controls are unchanged except the arrows are dead:
**n = next · r = re-record · q / ESC = stop.**

```bash
/opt/homebrew/Caskroom/miniforge/base/envs/lerobot/bin/python tools/record_arrow_safe.py \
  --robot.type=so101_follower --robot.port=/dev/tty.usbmodem<follower-serial> --robot.id=follower \
  --teleop.type=so101_leader --teleop.port=/dev/tty.usbmodem<leader-serial> --teleop.id=leader \
  --robot.cameras='{front: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30}, wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \
  --dataset.fps=20 --display_data=true \
  --dataset.repo_id=faithqin/so101-tube-insert-v4 \
  --dataset.single_task="Pick up the test tube and insert it into the rack" \
  --dataset.num_episodes=58 --dataset.episode_time_s=60 --dataset.reset_time_s=15 \
  --dataset.push_to_hub=false
```
Camera indices verified Sep 4 (front=1, wrist=0). The 18-dim `so_follower` patch is active — `observation.state` records 6 pos + 6 load + 6 current; the noload twin is derived afterward with `tools/make_noload_dataset.py`.

## Held-out evaluation set — the mechanism, declared before recording

**No policy in this project has ever had one.** Every `train_config.json` on disk
carries `eval_split: 0.0`, and the only `episodes=` list ever passed (v2, ACT-A and
SmolVLA-A) excluded **episode 33 alone** — a quality exclusion of a known-bad take,
not a holdout. So every reported offline number, including both dry-runs' "PLUMBING
OK", is measured on frames the policy was **trained on** (`pi05_dryrun.py`'s own
header: *"Feed pi0.5 frames it was TRAINED on"*). That is a training-set reproduction
check, not a generalization check.

**Do NOT use lerobot's `--dataset.eval_split`.** It exists (`configs/default.py`:
*"Fraction of episodes held out per task for offline evaluation"*) but
`make_train_eval_datasets` takes **the LAST ceil(n × eval_split) episodes per task**.
This dataset has one task, so the reserve would be episodes 50–54 — four CLEAN and one
ROT-20°, i.e. drawn from one category, which is exactly what the standing rule forbids.

**Use explicit episode selection instead.** The ⭐ episodes are spread across CLEAN,
NUDGE, RESIST and ROT-20° by construction. Train on the other 49:

```
--dataset.episodes="[0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23,24,26,27,28,29,31,32,33,34,35,36,37,39,40,41,43,44,46,47,48,49,50,51,53,54,55,56,57]"
```

Held out: **5 (CLEAN), 20 (NUDGE), 25 (CLEAN), 30 (RESIST), 38 (ROT-20°), 42 (LEFT-MIMIC), 45 (ROT-45°), 52 (CLEAN)**
— 8 of 58, 14%. Whole episodes, never partial. `LeRobotDataset` builds its statistics
from the selected episodes only, so the normalizer sees no held-out frame either.

**What the reserve buys, and what it does not.** It gives a genuine offline
generalization number: re-point both dry-runs at held-out episodes and their error is
then measured on frames no policy ever saw, which is what makes "PLUMBING OK" mean
something about the policy rather than about memorization. It does **not** test
generalization to a new *start position* — v4, like v3, records at position D only, so
the eval condition matches the training condition by design. Spatial generalization
would need multiple positions and is a v5 question; say so rather than implying the
held-out set covers it.

**Record all 58 the same way.** The ⭐ marks change nothing about how an episode is
demonstrated; the split happens at training time.

## Per-episode table


> **Table regenerated from `tools/v4_manifest.json` on 2026-09-05.** The index column is the RECORDED index. Voided rows are marked and their type is re-owed at a later index (A3); redo rows say which episode they replace. Hold-outs are re-picked at the end (A4). Regenerate with `python tools/v4_render_schedule.py`; never edit rows by hand.

| ep # | Type | Note | Physical setup before the episode | Demonstration |
|---|---|---|---|---|
|  0 | CLEAN |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
|  1 | CLEAN |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
|  2 | RESIST |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) — exactly as CLEAN | approach, grasp, transport as CLEAN → at the rack, LATERAL miss: tip lands on the rack surface beside the funnel → feel the push-back through the leader → back off ≈1–2 cm → re-align → insert → release. Hesitation allowed ONLY between the resistance and the re-align (v2 eps ~45–48 precedent) |
|  3 | CLEAN |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
|  4 | RESIST | VOID — RESIST: rack 10.1 s but no elbow back-off registered | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) — exactly as CLEAN | approach, grasp, transport as CLEAN → at the rack, ANGLE miss: tube tilted so the tip jams at the funnel mouth → feel the push-back through the leader → back off ≈1–2 cm → re-align → insert → release. Hesitation allowed ONLY between the resistance and the re-align (v2 eps ~45–48 precedent) |
|  5 | CLEAN |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
|  6 | CLEAN | VOID — CLEAN: back-off at the rack in a decisive take | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
|  7 | CLEAN |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
|  8 | CLEAN |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
|  9 | NUDGE | VOID — NUDGE: nudged with closing/closed jaws over ~3 s, not an ope | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) — exactly as CLEAN | approach as CLEAN → on the way in, GRAZE the cap side with the jaws so the tube shifts ≈1 cm image-RIGHT (the arm does it) → notice → re-approach → grasp → transport → seat → release. Hesitation allowed ONLY between graze and regrip |
| 10 | CLEAN | VOID — CLEAN: two approaches before the grasp | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
| 11 | LEFT-MIMIC |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) (v3's own in-trajectory type — unchanged) | approach deliberately slightly LEFT of the cap, ONE beat of hover, correct rightward, grasp decisively, insert, release (verify pan overshoot from parquet, as v3 did) |
| 12 | CLEAN |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
| 13 | ROT-20° |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) — exactly as CLEAN | approach as CLEAN → graze the cap with the HEAD, jaws OPEN (A5), so the tube turns ≈20° → come back up → adjust to the new angle → ONE decisive close on the cap → lift → insert → release |
| 14 | CLEAN |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
| 15 | ROT-45° |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) — exactly as CLEAN | approach as CLEAN → graze the cap with the HEAD, jaws OPEN (A5), so the tube turns ≈45° (firmer than ROT-20°) → come back up → adjust to the new angle → ONE decisive close on the cap → lift → insert → release |
| 16 | CLEAN | VOID — CLEAN: tube released at the PICK SITE (gripper load 305->500 | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
| 17 | ROT-20° |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) — exactly as CLEAN | approach as CLEAN → graze the cap with the HEAD, jaws OPEN (A5), so the tube turns ≈20° → come back up → adjust to the new angle → ONE decisive close on the cap → lift → insert → release |
| 18 | CLEAN |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
| 19 | CLEAN |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
| 20 | RESIST | ← redo of 4 | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) — exactly as CLEAN | approach, grasp, transport as CLEAN → at the rack, ANGLE miss: tube tilted so the tip jams at the funnel mouth → feel the push-back through the leader → back off ≈1–2 cm → re-align → insert → release. Hesitation allowed ONLY between the resistance and the re-align (v2 eps ~45–48 precedent) |
| 21 | CLEAN | ← redo of 6 | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
| 22 | NUDGE | ← redo of 9 | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) — exactly as CLEAN | approach as CLEAN → on the way in, GRAZE the cap side with the jaws so the tube shifts ≈1 cm image-RIGHT (the arm does it) → notice → re-approach → grasp → transport → seat → release. Hesitation allowed ONLY between graze and regrip |
| 23 | CLEAN | ← redo of 10 | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
| 24 | CLEAN | ← redo of 16 | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
| 25 | NUDGE |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) — exactly as CLEAN | approach as CLEAN → on the way in, GRAZE the cap side with the jaws so the tube shifts ≈1 cm image-RIGHT (the arm does it) → notice → re-approach → grasp → transport → seat → release. Hesitation allowed ONLY between graze and regrip |
| 26 | CLEAN |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
| 27 | RESIST |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) — exactly as CLEAN | approach, grasp, transport as CLEAN → at the rack, LATERAL miss: tip lands on the rack surface beside the funnel → feel the push-back through the leader → back off ≈1–2 cm → re-align → insert → release. Hesitation allowed ONLY between the resistance and the re-align (v2 eps ~45–48 precedent) |
| 28 | CLEAN |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
| 29 | LEFT-MIMIC |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) (v3's own in-trajectory type — unchanged) | approach deliberately slightly LEFT of the cap, ONE beat of hover, correct rightward, grasp decisively, insert, release (verify pan overshoot from parquet, as v3 did) |
| 30 | CLEAN |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
| 31 | ROT-20° |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) — exactly as CLEAN | approach as CLEAN → graze the cap with the HEAD, jaws OPEN (A5), so the tube turns ≈20° → come back up → adjust to the new angle → ONE decisive close on the cap → lift → insert → release |
| 32 | CLEAN |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
| 33 | NUDGE |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) — exactly as CLEAN | approach as CLEAN → on the way in, GRAZE the cap side with the jaws so the tube shifts ≈1 cm image-RIGHT (the arm does it) → notice → re-approach → grasp → transport → seat → release. Hesitation allowed ONLY between graze and regrip |
| 34 | CLEAN |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
| 35 | RESIST |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) — exactly as CLEAN | approach, grasp, transport as CLEAN → at the rack, ANGLE miss: tube tilted so the tip jams at the funnel mouth → feel the push-back through the leader → back off ≈1–2 cm → re-align → insert → release. Hesitation allowed ONLY between the resistance and the re-align (v2 eps ~45–48 precedent) |
| 36 | CLEAN |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
| 37 | NUDGE |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) — exactly as CLEAN | approach as CLEAN → on the way in, GRAZE the cap side with the jaws so the tube shifts ≈1 cm image-RIGHT (the arm does it) → notice → re-approach → grasp → transport → seat → release. Hesitation allowed ONLY between graze and regrip |
| 38 | CLEAN |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
| 39 | RESIST |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) — exactly as CLEAN | approach, grasp, transport as CLEAN → at the rack, LATERAL miss: tip lands on the rack surface beside the funnel → feel the push-back through the leader → back off ≈1–2 cm → re-align → insert → release. Hesitation allowed ONLY between the resistance and the re-align (v2 eps ~45–48 precedent) |
| 40 | CLEAN |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
| 41 | ROT-20° |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) — exactly as CLEAN | approach as CLEAN → graze the cap with the HEAD, jaws OPEN (A5), so the tube turns ≈20° → come back up → adjust to the new angle → ONE decisive close on the cap → lift → insert → release |
| 42 | CLEAN |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
| 43 | ROT-20° |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) — exactly as CLEAN | approach as CLEAN → graze the cap with the HEAD, jaws OPEN (A5), so the tube turns ≈20° → come back up → adjust to the new angle → ONE decisive close on the cap → lift → insert → release |
| 44 | CLEAN |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
| 45 | LEFT-MIMIC |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) (v3's own in-trajectory type — unchanged) | approach deliberately slightly LEFT of the cap, ONE beat of hover, correct rightward, grasp decisively, insert, release (verify pan overshoot from parquet, as v3 did) |
| 46 | CLEAN |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
| 47 | LEFT-MIMIC |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) (v3's own in-trajectory type — unchanged) | approach deliberately slightly LEFT of the cap, ONE beat of hover, correct rightward, grasp decisively, insert, release (verify pan overshoot from parquet, as v3 did) |
| 48 | CLEAN |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
| 49 | CLEAN |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
| 50 | ROT-45° |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) — exactly as CLEAN | approach as CLEAN → graze the cap with the HEAD, jaws OPEN (A5), so the tube turns ≈45° (firmer than ROT-20°) → come back up → adjust to the new angle → ONE decisive close on the cap → lift → insert → release |
| 51 | CLEAN |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
| 52 | LEFT-MIMIC |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) (v3's own in-trajectory type — unchanged) | approach deliberately slightly LEFT of the cap, ONE beat of hover, correct rightward, grasp decisively, insert, release (verify pan overshoot from parquet, as v3 did) |
| 53 | CLEAN |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
| 54 | ROT-45° |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) — exactly as CLEAN | approach as CLEAN → graze the cap with the HEAD, jaws OPEN (A5), so the tube turns ≈45° (firmer than ROT-20°) → come back up → adjust to the new angle → ONE decisive close on the cap → lift → insert → release |
| 55 | CLEAN |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
| 56 | NUDGE |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) — exactly as CLEAN | approach as CLEAN → on the way in, GRAZE the cap side with the jaws so the tube shifts ≈1 cm image-RIGHT (the arm does it) → notice → re-approach → grasp → transport → seat → release. Hesitation allowed ONLY between graze and regrip |
| 57 | CLEAN |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
| 58 | ROT-45° |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) — exactly as CLEAN | approach as CLEAN → graze the cap with the HEAD, jaws OPEN (A5), so the tube turns ≈45° (firmer than ROT-20°) → come back up → adjust to the new angle → ONE decisive close on the cap → lift → insert → release |
| 59 | CLEAN |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
| 60 | CLEAN |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
| 61 | CLEAN |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) | DECISIVE: approach from above, grasp the cap ≈3.5 s from start, lift, transport, seat, release into the funnel, jaws close |
| 62 | ROT-20° |  | tube on the D mark, horizontal, cap image-left (scene_check ≤ 8 px) — exactly as CLEAN | approach as CLEAN → graze the cap with the HEAD, jaws OPEN (A5), so the tube turns ≈20° → come back up → adjust to the new angle → ONE decisive close on the cap → lift → insert → release |
| 63 | EXTRA | VOID — unplanned | — | EXTRA: unplanned 64th index — lerobot kept recording past num_episodes=63 under --resume; 37 frames (1.9 s), three Left  |

## Audit after recording (before any training)

- 58 episodes; every one ends seated (front-camera final frame in the seat region).
- Per-category signatures from parquet + video: NUDGE — cap shift ≈1 cm right before the grasp, second approach; ROT — tube angle change before the grasp (video; scene_check's angle is noise below ~8°, so ROT-20° is verified by eye); LEFT-MIMIC — pan overshoot > ~2.5° as v3 (v3 measured 2.46–5.63°); RESIST — load/current excursion at the rack before the seat + a shoulder_lift/elbow back-off, rack phase longer than CLEAN's ~4–5 s; CLEAN — none of these.
- Frame-0 pose per episode inside the v3 support window (`home_gate.FRAME0["v3"]`); v4's own window is then derived and added to `home_gate.FRAME0` (TDD).
- Hold-out ⭐ episodes listed by index for the training split.
- fps = 20, both cameras, 18-dim state. Then derive `so101-tube-insert-v4-noload`, push both, launch ACT-A/B first.
