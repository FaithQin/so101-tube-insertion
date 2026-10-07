# DAgger single-arm before/after — design (Faith's call, Aug 31 2026)

> **Erratum (Sep 22 2026).** The line below saying B2V-A-02 "touched the seat 6 times, longest 0.6 s" repeats an instrument count. The scorer counts a run whenever the cap's centroid sits inside its seat box, and on the recording those runs are the dropped tube lying beside the rack and the operator's hand during resets. The tube never entered the funnel. The honest line is that the policy reaches the funnel and cannot insert. Evidence: `analysis/demo_cut/EDL.md`, note 1, and `analysis/demo_cut/B2V-A-02_seat_runs_evidence.png`.

**Chosen over embedding DAgger in the A/B**, because separate correction sets
per arm would change two variables at once and make any difference
un-attributable. Faith identified that failure mode herself before a single
episode was recorded.

---

## The question this design answers

> Does adding on-policy corrections — states the policy itself reaches, labelled
> with a good action — fix the failure that staged recovery demonstrations did
> not?

**One variable: the training data.** Observation space, architecture, task, rig,
start states, and eval protocol are all held fixed.

| | policy | training data | observation space |
|---|---|---|---|
| **Before** | `act-tube-A-v3` | v3 (50 eps) | 6-dim, vision + joint pos |
| **After** | `act-tube-A-v3-dagger` | v3 ∪ D | 6-dim, **unchanged** |

`D` = DAgger correction episodes collected under `act-tube-A-v3`.

**This is NOT the load-sensing study.** That one is answered: v2, 40 trials,
A 6/20 vs B 5/20, null, with the stage split as the durable signal. Nothing here
re-opens it.

## Why A and not B

1. **A is the only policy DAgger can work on.** DAgger's premise is that the
   policy carries the episode most of the way and the human corrects the last
   part. A grasps (2/2 in Block 2) and fails at insertion — a well-posed
   correction target. B, on current evidence, never grasps, which would mean the
   human teleoperates from the first second of every episode. That is
   re-recording demonstrations with extra ceremony, not DAgger.
2. **A's failure is the one that was characterised.** Seat runs that are reached
   and lost (B2V-A-02 touched the seat 6 times, longest 0.6 s), scatter that
   rules out miscalibration, and the two missing behaviours named in the Aug 30
   entry: "you moved it — re-plan" and "it will not go in — back off."

⚠️ Caveat on point 1: **B's rate is not established.** Wilson 95% CI on 0/2 is
[0.000, 0.658]. Block 2B (6 trials) is what makes "B never grasps" sayable at
all. Run Block 2B before treating the B-can't-DAgger argument as settled.

## Still record at 18-dim

Even though this design does not compare A against B, **`D` must be recorded
with load and current present.** Reason: `rollout/context.py:355` widens the
observation filter only when the loaded policy's state is wider than the `.pos`
keys, and `dataset_features` is built from that same filtered dict
(`context.py:384-389`). Collecting under A (6-dim) therefore writes a 6-dim
dataset with load and current **dropped at the source, unretrofittable** — the
same one-way door as the original 18-dim prerequisite.

Recording 18-dim costs nothing now and preserves every future option: the
B-side twin, a post-capstone A/B on corrected data, the effort-channel analysis.
Recording 6-dim closes all of them permanently.

**Blocker: the filter fix must land (TDD) before episode 1.**

## Protocol

**Collection.** `--strategy.type=dagger` with `--teleop.type` set (required) and
a NEW `--dataset.repo_id`; v3 is never overwritten. Default
`record_autonomous=False`, so only correction windows become episodes.

Handover, verified in source — the leader drives itself, Faith does not position
it. `AUTONOMOUS → PAUSED` calls `teleop_smooth_move_to`, which enables leader
torque and interpolates it to the follower's last commanded pose over 2.0 s at
30 Hz (`common/control_utils.py:187`); `PAUSED → CORRECTING` calls
`disable_torque` and hands it over. **Do not grip the leader on pause** — it
motors under your hand for those 2 s.

**What to correct, declared in advance.** Only the two named failures:
1. the tube has been displaced by the arm → re-approach it
2. the tube is at the rack and will not seat → back off, re-align, re-insert

**Suspend the v3 decisiveness rule.** It is what deleted the behaviour last
time. A correction IS a hesitate-then-correct; hesitation is the signal, not a
defect. This is the single most important difference from the v3 recording
session, and it is written here before collection starts.

**Aggregation.** Fine-tune from the existing `act-tube-A-v3` checkpoint
(`--policy.path=...`), not a from-scratch 40k run. Pass
`--save_checkpoint_to_hub=true` — the π0.5 run's omission of it means no
intermediate checkpoints exist for that model.

**Hold out whole episodes** from `D` for eval, sized so the reserve does not
come entirely out of one correction category.

## Evaluation — pre-declare before trial 1

- **Paired by start position.** Before-policy and after-policy run
  back-to-back on the same start condition, exactly as the v3 pre-registration
  requires for A-vs-B. Splitting before-on-day-1 / after-on-day-2 would confound
  the contrast with across-day drift, which this project has measured to be
  large (v2-A pooled 8/10 on Aug 29 against a 6/20 Aug 25 baseline).
- **Fixed n, declared before the first trial. No adding trials after peeking.**
- **Exact mid-p McNemar** on discordant pairs; Wilson 95% CIs per arm reported
  alongside.
- Same gates (`tools/preflight.py`), same 45 s episodes, same video scoring
  (`tools/score_episode.py`), same INFRA rules.

## The honest limitation, to write down now

A before/after on one policy cannot separate "on-policy corrections helped" from
"more data helped." The controls that would separate them — a matched-size
arm of additional *staged* episodes, or a v3-minus-targeted ablation — cost
another collection session and another fine-tune each.

**Do not paper over this.** State it: the comparison shows whether the DAgger
remedy moved the number, not that on-policy-ness specifically was the active
ingredient. Given the Sep 6 wall, that is the right trade — but it is a stated
limitation, not an unnoticed one.

## Schedule

Zero scored trials exist. Sep 6 is the wall for bench work AND a complete draft;
Taiwan is Sep 7–11 (corrected Sep 2, 2026 by Faith — the return is the 11th, not the 14th, so Sep 12–14 are working days); ship Sep 15.

**Time-box: one collection session, one fine-tune.** If it is not producing by
Sep 3, stop and write. The write-up is the deliverable; DAgger is one section
of it.
