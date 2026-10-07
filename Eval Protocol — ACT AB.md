# Paired Evaluation Protocol — ACT-A vs ACT-B (tube insertion) — v2

Generated Aug 15 2026, seed 20260815. 80 trials: 40 per policy, 8 per dot per
policy. 16 blocks of 5 in consecutive pairs — each pair shares an IDENTICAL
start order (true pairing); pair policies alternate AB, BA, AB, BA... to
cancel time/servo-warmth drift. v2 fixes v1, whose blocks were shuffled
independently (pairing claim was wrong).

## Setup (once)
- 5 EVAL DOTS at zone centers A-E. Tube ON the dot, cap left, as trained.
- Rig untouched: cameras, rack, lighting (blinds closed, lamp on).

## Rules
- SUCCESS = tube standing in the target hole unsupported at rollout end.
- Max 45 s; never touch mid-rollout.
- Categories (one per trial, most severe event wins):
    S  = success (tube standing in target hole unsupported at rollout end)
    GM = grasped, missed insertion (reached the rack; failed to seat — Whitney: jamming vs wedging)
    NG = never grasped
    DT = dropped in transit
    KR = knocked rack/tube over
- 5-min servo rest between blocks; operator break every 4 blocks.
- WARMUP (added Aug 15, pre-trial-1, after the cold-sag discovery): before the
  first block of a session AND after any break >10 min, run 2 warmup replays
  (episode 0) or ~60 s of motion. Cold servos ride ~1 cm high vs the warm
  state the dataset was recorded in; replays confirmed grasp success returns
  by run 2-3 of a cold start. The 5-min inter-block rest is short enough to
  stay warm; a full cooldown is not.

## Inference config (locked Aug 16 ~00:30, pre-trial-1)

Chosen by a pre-declared rule on a 2-episodes-per-cell probe grid (all probes
Aug 15 23:07–Aug 16 00:25, dot D, warmup + verified homing before each; grade
per episode: grasp 1 / insertion attempt 1 / seated 2; higher sum wins per
policy; ties → n=25):

| config | ep1 | ep2 | sum |
|--------|-----|-----|-----|
| A n=25 | S (4) | no-grasp (0) | **4** |
| A n=50 | grasp+attempts (1) | no-grasp (0) | 1 |
| B n=25 | no-grasp (0) | no-grasp (0) | 0 |
| B n=50 | no-grasp (0) | grasp+attempt (2) | **2** |

**→ ACT-A rolls out with `--policy.n_action_steps=25`; ACT-B with
`--policy.n_action_steps=50`.** Per-policy configs are legitimate (each method
at its probe-selected setting); selection used probe episodes only, no scored
trials. No further config tuning — observed episode-to-episode variance
exceeds config deltas at feasible probe sample sizes; the paired 80-trial eval
is the instrument that resolves policy differences, not probes.

All rollouts run via `tools/rollout_30hz_stale_ok.py` (non-blocking camera
reads at 30 Hz loop; between-episode reset pinned to the verified training
home pose). Warmup (2 replays) + `tools/home_arm.py` before each block.

**Config-probe EXTENSION (declared Aug 16 ~01:00, before running, PI decision
overriding the stop-tuning recommendation — logged for honesty):**
ACT-A only: 10 episodes @ n=25 vs 10 @ n=10, PAIRED — both arms use the same
start sequence (D,B,C,E,A then D,E,B,A,C; 2 per dot per arm). Rule, declared
now: A's eval config = the arm with more S; tie → more grasps; tie → n=25
(hardware prior: n=10 runs ~26.3 Hz effective vs 28.4, and per-chunk inference
fires every 0.33 s). These are probes, not scored eval trials. B stays @ 50.

**Acknowledged confound + pre-registered secondary probe:** with per-policy
configs, the A/B contrast measures observation-space PLUS its interaction
with chunk horizon (no shared n exists where both policies function — probes:
25 is A's only working config, 50 is B's). If the primary 80-trial result is
close or surprising, run an EXPLORATORY cross-config set (2 blocks A@50,
2 blocks B@25, 20 trials total, same pairing rules, labeled secondary) to
bound the n-effect. Decided Aug 16 pre-trial-1.

Known scene deviation, applies to BOTH policies equally: the 5 blue eval dots
did not exist in training data. "Dot-as-cap-decoy" is a logged hypothesis
(2 sightings during probes) — post-eval masked-dots probe planned; not a
protocol change.

## Trial sheet
| # | Block | Policy | Dot | Category | Notes |
|---|-------|--------|-----|----------|-------|
|  1 |  1 | ACT-A | D |    |    |
|  2 |  1 | ACT-A | B |    |    |
|  3 |  1 | ACT-A | C |    |    |
|  4 |  1 | ACT-A | E |    |    |
|  5 |  1 | ACT-A | A |    |    |
|  6 |  2 | ACT-B | D |    |    |
|  7 |  2 | ACT-B | B |    |    |
|  8 |  2 | ACT-B | C |    |    |
|  9 |  2 | ACT-B | E |    |    |
| 10 |  2 | ACT-B | A |    |    |
| 11 |  3 | ACT-B | D |    |    |
| 12 |  3 | ACT-B | E |    |    |
| 13 |  3 | ACT-B | B |    |    |
| 14 |  3 | ACT-B | A |    |    |
| 15 |  3 | ACT-B | C |    |    |
| 16 |  4 | ACT-A | D |    |    |
| 17 |  4 | ACT-A | E |    |    |
| 18 |  4 | ACT-A | B |    |    |
| 19 |  4 | ACT-A | A |    |    |
| 20 |  4 | ACT-A | C |    |    |
| 21 |  5 | ACT-A | B |    |    |
| 22 |  5 | ACT-A | D |    |    |
| 23 |  5 | ACT-A | C |    |    |
| 24 |  5 | ACT-A | A |    |    |
| 25 |  5 | ACT-A | E |    |    |
| 26 |  6 | ACT-B | B |    |    |
| 27 |  6 | ACT-B | D |    |    |
| 28 |  6 | ACT-B | C |    |    |
| 29 |  6 | ACT-B | A |    |    |
| 30 |  6 | ACT-B | E |    |    |
| 31 |  7 | ACT-B | B |    |    |
| 32 |  7 | ACT-B | C |    |    |
| 33 |  7 | ACT-B | E |    |    |
| 34 |  7 | ACT-B | A |    |    |
| 35 |  7 | ACT-B | D |    |    |
| 36 |  8 | ACT-A | B |    |    |
| 37 |  8 | ACT-A | C |    |    |
| 38 |  8 | ACT-A | E |    |    |
| 39 |  8 | ACT-A | A |    |    |
| 40 |  8 | ACT-A | D |    |    |
| 41 |  9 | ACT-A | C |    |    |
| 42 |  9 | ACT-A | A |    |    |
| 43 |  9 | ACT-A | E |    |    |
| 44 |  9 | ACT-A | B |    |    |
| 45 |  9 | ACT-A | D |    |    |
| 46 | 10 | ACT-B | C |    |    |
| 47 | 10 | ACT-B | A |    |    |
| 48 | 10 | ACT-B | E |    |    |
| 49 | 10 | ACT-B | B |    |    |
| 50 | 10 | ACT-B | D |    |    |
| 51 | 11 | ACT-B | E |    |    |
| 52 | 11 | ACT-B | B |    |    |
| 53 | 11 | ACT-B | D |    |    |
| 54 | 11 | ACT-B | C |    |    |
| 55 | 11 | ACT-B | A |    |    |
| 56 | 12 | ACT-A | E |    |    |
| 57 | 12 | ACT-A | B |    |    |
| 58 | 12 | ACT-A | D |    |    |
| 59 | 12 | ACT-A | C |    |    |
| 60 | 12 | ACT-A | A |    |    |
| 61 | 13 | ACT-A | E |    |    |
| 62 | 13 | ACT-A | B |    |    |
| 63 | 13 | ACT-A | D |    |    |
| 64 | 13 | ACT-A | A |    |    |
| 65 | 13 | ACT-A | C |    |    |
| 66 | 14 | ACT-B | E |    |    |
| 67 | 14 | ACT-B | B |    |    |
| 68 | 14 | ACT-B | D |    |    |
| 69 | 14 | ACT-B | A |    |    |
| 70 | 14 | ACT-B | C |    |    |
| 71 | 15 | ACT-B | E |    |    |
| 72 | 15 | ACT-B | C |    |    |
| 73 | 15 | ACT-B | B |    |    |
| 74 | 15 | ACT-B | D |    |    |
| 75 | 15 | ACT-B | A |    |    |
| 76 | 16 | ACT-A | E |    |    |
| 77 | 16 | ACT-A | C |    |    |
| 78 | 16 | ACT-A | B |    |    |
| 79 | 16 | ACT-A | D |    |    |
| 80 | 16 | ACT-A | A |    |    |
