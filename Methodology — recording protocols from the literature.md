# Methodology — recording & evaluation protocols from the literature

**Written Sep 4, 2026, before v4 is recorded.** This is the reference the v4
recording protocol is derived from, and the record of where each rule came
from. It exists so that v4 does not repeat v3, and so the write-up can cite
the methodology instead of asserting it.

Every claim carries one of three tags, assigned on reading the
primary source:

- **[verified]** — read in the paper's own text; authors, title, venue, ID confirmed
- **[partial]** — the paper exists and says something close; the exact number or
  venue could not be confirmed from the primary source
- **[community]** — a blog, vendor post or forum thread; useful, not peer-reviewed

Anything not tagged is inference and is labelled as such. Section 9 lists
claims that circulated during the search and could **not** be verified — do
not cite them.

---

## 1. Demonstration quality and dataset composition

**One operator, one strategy, held constant across every episode.** [verified]
robomimic's Multi-Human set (300 demos, six operators) scored *below* its
Proficient-Human set (200 demos, one operator): Square with plain BC 78.7% →
52.7% despite 50% more data. *"This most likely stems from the presence of
suboptimal and multimodal data."* — Mandlekar et al., arXiv:2108.03298, CoRL
2021 (Oral). Gandhi et al. (arXiv:2210.08073, CoRL 2022): *"the ideal dataset
for imitation learning is homogenous and low-variance — reflecting a single,
optimal method for performing a task."*

We are single-operator, so the exposure is **within-operator drift** — the
strategy changing between session 1 and session 5.

**Fix the grasp geometry specifically.** [verified] Demo-SCORE (Chen, Lessing,
Liu, Finn, arXiv:2503.03707): every demonstration *succeeded*; they differed
only in grasp geometry (side midpoint vs. diagonal corner). Filtering to one
strategy gained **15–35% absolute**. Two successful-but-different grasps are
enough to cost double digits. For the tube: one grip height, one cap
orientation, one approach direction.

**Do not top up a good dataset with mediocre takes.** [verified] robomimic:
adding 100 "worse"-operator demos to 100 good ones degraded most algorithms;
only BC-RNN improved. CUPID (Agia et al., arXiv:2506.19121, CoRL 2025) shows
VLA-scale models do not simply learn to ignore low-quality behaviour either —
this applies to the SmolVLA and π0.5 arms.

**Vary the start state; never move a camera.** [verified] Xie, Lee, Xiao, Finn
(arXiv:2307.03659) ranked eleven environment factors: *"new camera positions
are the hardest to generalize to while new backgrounds are the easiest."*
Their "narrow" camera perturbation was ±2.5 cm. Lighting is among the easiest
factors — the night/day non-finding of Aug 29 is the expected result.

**The zone A–E schedule is already the field's protocol.** [verified] SmolVLA
(arXiv:2506.01844) on SO-100/SO-101: *"10 trajectories for each of 5 distinct
starting positions, resulting in a total of 50 demonstrations per dataset."*

**Diversity beats count past a threshold.** [verified] Lin et al.,
arXiv:2410.18647, ICLR 2025: *"once the number of demonstrations per
environment or object reaches a certain threshold, additional demonstrations
have minimal effect."* Do not cite their "50 demos per pair" at our setup —
that is 50 per environment-object pair across 32 pairs.

**50 episodes is convention, not an ablated optimum.** [verified, gap flagged]
The ACT paper contains no demo-count ablation. robomimic's hardest precision
task, Tool Hang, reached **3.3% on 200 real demonstrations**. The task tube is
the **uncut, coned** tube (the lab notebook (private), Aug 16 — the sawn-flat version was too
hard even under teleop and was abandoned before recording), so it keeps its
self-centering chamfer and sits well below Tool-Hang difficulty; the
calibration point stands for the chamferless variant that was tried and dropped.

**Keep the wrist camera and use crop augmentation.** [verified] robomimic
real-world Can: 73.3% → **43.3%** without the wrist camera → 26.7% without
pixel-shift randomization. Hsu et al. (arXiv:2203.12677, ICLR 2022 oral):
hand-centric views *"consistently improve training efficiency and
out-of-distribution generalization."*

**Quality filtering that works uses rollout feedback, not human judgement.**
[verified — the caveat is the finding] Demo-SCORE, CUPID and Gandhi all score
demos via policy rollouts or influence functions. Demo-SCORE: *"unreliable or
underrepresented strategies can be difficult even for people to discern."*
"Discard the hesitant takes by eye" is **not** what this literature validates.

---

## 2. Recovery data and covariate shift

**The canonical statement.** [verified] Ross & Bagnell, AISTATS 2010: errors
compound and regret grows quadratically in horizon. DAgger: Ross, Gordon,
Bagnell, AISTATS 2011, arXiv:1011.0686.

**DAgger correction data is policy-specific by construction, but the
practical picture has three parts.** [verified]

1. Formally: DAgger Algorithm 3.1 collects *"visited states by π_i"* — the
   data is defined relative to one learner. HG-DAgger (Kelly et al.,
   arXiv:1810.02890, ICRA 2019) and DART both say the same independently.
2. Empirically, *within one architecture*, corrections transfer nearly
   free: IWR (Mandlekar et al., arXiv:2012.06733) Table III — 87.3 ± 6.4
   trained on HG-DAgger's dataset vs 87.3 ± 5.0 on its own.
3. **Cross-architecture transfer is untested.** No paper harvests corrections
   against architecture A and trains architecture B. For a four-architecture
   study, per-policy corrections would mean up to eight correction sets.

**Consequence:** a *dataset* is architecture-agnostic; DAgger corrections are
not known to be. That is why v4 is a dataset, not a DAgger round. DAgger
remains the sharp *single-architecture* follow-up.

**Human-demonstrated recovery is a well-evidenced alternative to interactive
DAgger.** [verified]

- **DART** — Laskey, Lee, Fox, Dragan, Goldberg, CoRL 2017, arXiv:1703.09327.
  Inject small errors during human demonstrations so the supervisor must
  demonstrate recovery: *"This forces the supervisor to demonstrate how to
  recover from errors."* Real Toyota HSR: BC 49% → DART 79%. Off-policy,
  dataset-based, architecture-agnostic.
- **The magnitude is load-bearing, and larger is worse.** DART α=3 → 79%;
  α=6 → 72%: *"this level of noise was potentially too high for the human
  supervisor."* The rule: *"the noise should approximate the error of the
  trained robot's policy."*
- **VBT — Visual Backtracking Teleoperation** — Brandfonbrener, Tu, Singh,
  Welker, Boodoo, Matni, Varley, arXiv:2210.02343, ICRA 2023. *"1. Failure:
  the teleoperator first fails at the desired task. 2. Recovery: the
  teleoperator recovers from the failure and begins to attempt the task again.
  3. Success: the teleoperator successfully finishes the task."* All within
  one trajectory, so that *"differences between observations of failure and
  success are task-relevant."* Caveat: payoff measured for offline RL on
  deformable grasping, not BC on insertion.
- **RaC** — Hu, Wu, Enock, Li, Kadakia, Erickson, Kumar, arXiv:2509.07953.
  Human-in-the-loop *during policy rollout*: rewind to an in-distribution
  state, then correct. Success scales linearly with recovery maneuvers.
  DROID contains only 3.68% episodes with recovery behaviour.
- **RaC's argument against staged failures**, verbatim: *"one could simply
  instruct human teleoperators to artificially stage possible failure
  states… However, such behaviors produced by humans from contrived or
  'fake' states may not reflect the out-of-distribution errors that a learned
  policy would actually encounter."* This is v3's post-mortem, in print —
  but it is **asserted, not measured**.

**RaC's "end the episode after the correction" does NOT apply to pure
teleop.** (inference, from the two papers' stated rationales) RaC's Rule 2
exists because a *policy* is running when the human intervenes — the
post-takeover state is a mixture of policy-induced and human-induced. In v4
no policy is in the loop; every state is human-produced. VBT's guidance
applies: **knock, regrip, then transport and seat, in one episode.** A
BC-specific reason too: an episode ending after the regrip teaches "after a
recovery, stop."

**Staged failure states are workable only with an explicit detector.**
[verified] EgoRecovery (arXiv:2607.19745) stages failures but pairs them with a
learned recovery gate. Wong et al. (arXiv:2112.05251, CoRL 2021) train an
explicit error detector. The field does not expect detection to emerge from
staged data alone.

**Was v3's decisiveness rule wrong? The evidence now says probably not.**
[verified] Chen et al., *"Exploiting Policy Idling for Dexterous
Manipulation,"* arXiv:2508.15669, IROS 2025: **90% of ACT's failures on a real
connector-insertion task involved idling**, caused by *"small actions in areas
where the robot needs to perform high-precision motions, e.g., when preparing
to grasp an object or object insertion."* Their fix is upstream: do not record
the pause. Reactive Diffusion Policy (arXiv:2503.02881, RSS 2025) Table V:
chunk 8 → 2 dropped grasp 100% → 20% *"because of demonstration pauses."*
Diffusion Policy: *"single-step policies can easily overfit to this pausing
behavior."*

**The honest v3 post-mortem, after measuring (Sep 4):** ACT-v3's own
first-contact displacement on non-grasp contacts is **≈1.0 cm (0.7 / 1.0 /
1.7 cm, n=3; grasp-and-carry events excluded)**. v3's hand-staged 1.4–1.5 cm
sat at the top of that range — roughly right, not wildly off — so magnitude
is *not* the main suspect. What remains: the offsets were **staged by hand**
(no transition observed; RaC's contrived-state critique) and episodes were
compositionally wrong (all begin in the bad state). The decisiveness rule is
not on that list. (This document's first draft said the offsets were
"probably far larger" than the policy's error — corrected by measurement.) **No paper isolates "observing the
transition into the bad state" as the mechanism** — that remains our
inference.

**The gap v3 + v4 can fill.** Four differently-worded searches found no
controlled A/B of pre-staged failure-recovery data (v3) against in-trajectory
human-induced failure-recovery data (v4) on a contact-rich task with an
action-chunking policy. v3 is arm A of that comparison.

---

## 3. Contact-rich insertion specifics

**The nearest neighbour to the load-sensing thesis: FACTR 2.** [verified]
Oh, Liu, Tao, Han, Shaw, Funabashi, Salakhutdinov, Pathak, arXiv:2606.12406,
Appendix D.6 Table 10, real hardware, ACT:

| | NIST Insertion | Cap Screwing |
|---|---|---|
| ACT | 0.220 | 0.440 |
| ACT + raw joint current | **0.431** | 0.440 |
| ACT + learned external torque | 0.567 | 0.525 |

Raw current nearly doubled insertion progress and did nothing on screwing.
Raw current *"mixes free-space actuation effort with contact torque"*; a
clean external-torque estimate *"generally requires temporal context"* (they
use a 50-step history). A single-frame policy on raw current should expect
the weaker result.

**Force pays off only in the seating phase.** [verified] InsertionNet
(Spector & Di Castro, RA-L 2021, arXiv:2104.14223): *"force information can be
useless when there is no overlap between the peg and the hole."* FMB (Luo et
al., IJRR): insertion-only ablation 2/25 → 11/25 with F/T. **Per-phase scoring
must be built to detect a seating-phase-only effect.**

**Over-represent the pre-contact phase.** [verified] FACTR 2 Table 2:
up-sampling pre-contact beat contact-only 0.818 vs 0.670 average; NIST
Insertion 0.800 → 0.933. *"The moments before contact are especially
important as they determine reaching the correct pose and alignment."*

**Demonstrate alignment as a distinct, deliberate search.** [verified] Every
serious insertion paper splits the task into search-then-insert (Inoue et al.,
IROS 2017; Qiao, Moore, Knight, Robotica 1996, on chamferless assembly).
Van Wyk et al. (T-RO 2018): spiral turn spacing should equal the clearance;
at σ=2 mm spiral variance exploded ~31×. Park et al. (RA-L 2020): a tilted peg
posture cut completion-time std-dev by up to 91.7%.

**Backward learning — the highest-yield 30 minutes available.** [verified]
InsertionNet Algorithm 1: seat the peg, record pose L; perturb uniformly
within ±10 mm lateral and ±10°; approach until a force threshold trips; label
with the correction L − D. Sixteen real insertion tasks bootstrapped in under
ten minutes.

**The chamfer is a named phase of the mate, not a detail — and the task tube
has one.** [literature verified; task identity per the lab notebook (private), Aug 16]
Whitney, ASME J. Dyn. Sys. 1982 (DOI 10.1115/1.3149634) and MIT OCW 2.875
notes: Approach → **Chamfer Crossing** → one-point → two-point → line contact.
With the compliance centre at the peg tip, *"all that's left is chamfer
crossing force."* Lian et al. (IROS 2021, arXiv:2103.05140) measured the
tolerance area: *"rectangle pegs have high tolerance thanks to their chamfered
edges"*; *"USB has the lowest tolerance because of its small dimension and
sharp edges."*

**Correction to the project's own framing.** the project notes says the tube's
conical tip was "sawn off deliberately." The the lab notebook (private) (Aug 16, *Object-
identity correction*) records the opposite: the cut tube was too hard even
under teleop, Faith switched to the uncut coned tube before recording, and
every dataset and trial since has used it. So the passive funnel **is present**;
what the arm lacks is a compliance centre and any force sensing with which to
exploit it. The abandoned chamferless attempt is an honest data point that
Lian's tolerance-area result predicts — report it as that. **The write-up must
not say the chamfer was removed.** (This document's Sep 4 first draft, and the
three research prompts behind it, repeated the stale framing; corrected here.)

**Expectation calibration.** [verified] HIL-SERL real RAM insertion, 200
demos: Diffusion Policy 27%, flat BC **12%**. ACT: 20% on Thread Velcro.
ResiP: diffusion BC on insertion plateaus near 80% with 100k demos
(simulation). These baselines run on far more capable hardware; low success at
50 demos on a coned tube, on an arm with no compliance centre, is still the
expected outcome.

---

## 4. Observation space — the confound in the A/B

**robomimic §4.3, verbatim:** [verified] *"including end effector velocity
information, and joint information hurts agents trained on low-dim
observations substantially (49%–88% relative performance drop), while
image-based agents are more tolerant to the inclusion of this extra
information (2%–29% relative performance drop). We hypothesize that
performance drops might be due to overfitting to the presence of this extra
information not needed for solving these tasks."* Table 25: Square 82.0 →
64.7 (+EEF vel) → 58.0 (+joint). *"Information-hiding can be a powerful
paradigm."*

**There is a published prior that ACT-B underperforms ACT-A purely from added
observation dimensionality**, independent of whether load carries signal. The
current design cannot separate those two explanations. Mitigation: we are
image-based (2–29% penalty, not 49–88%). This goes in the methods section,
named by us.

**Novelty status of the load channel.** [community, absence-of-evidence] No
public report of anyone adding `Present_Load`/`Present_Current` to
`observation.state` on an SO-100/SO-101. LeRobot's stock `so_follower` reads
only `Present_Position`. Phrase as "no public report I could find."

---

## 5. Control rate, chunking, and inference latency

**Match rollout rate to recording rate before blaming the model.** [community
for the number; verified for the mechanism] Morishige/Classmethod, 2026-03-07:
same ACT checkpoint, Mac MPS ~15 Hz → 40% (4/10); CUDA ~30 Hz → 90% (9/10).
Real-Time Chunking (Black, Galliker, Levine, arXiv:2506.07339, NeurIPS 2025)
names *"pauses or out-of-distribution jerky movements at chunk boundaries"*
under latency.

**Applies to π0.5 and, to a lesser degree, SmolVLA — not ACT.** v3 is 20 fps
(verified from `meta/info.json`). ⚠️ *Corrected Sep 5: this paragraph previously
read "ACT and SmolVLA roll out at 20 Hz — matched." Only ACT is matched.*

Per-tick timings, aggregated from every scored trial log (the wrapper's
`[timing per tick]` windows; the 50 ms budget is one tick at 20 Hz):

| policy | next_action | whole tick | ticks sampled |
|---|---|---|---|
| ACT | 4.8 ms mean / 4.4 median | **8.8 ms** | 94 |
| SmolVLA (sync) | 14.5 / 14.4 | **18.5 ms** | 8 |
| π0.5 | 31.8 / 19.9 | **35.8 ms** | 29 |

ACT sits far enough inside the budget to hold a true 20 Hz. The other two do
not run out of budget on the *average* tick — they blow it on the *chunk
boundary*: with `n_action_steps=50`, one tick in fifty does a full forward pass
and the other 49 pop a queued action in ~0.2 ms, so a mean of 14.5 ms implies a
single boundary tick of roughly 0.7 s, and π0.5's 31.8 ms implies ~1.0 s. That
stall is what the effective-rate deficit is made of.

**Numbers to nail down before the write-up cites them.** The previously stated
"π0.5 effective 13.4 Hz (601 frames in ~45 s)" is *arithmetically consistent*
(601/45 = 13.4) and 601 frames is exactly P05-01's length — **but P05-01 is a
VOIDED trial**, and the ~45 s wall-clock is not recoverable from the artifacts:
the parquet `timestamp` column is nominal (`frame_index / fps`), so every
rollout reads back as exactly 20.00 Hz regardless of what the loop did. The
"~17 Hz SmolVLA" figure quoted in the Sep 4 handoff has the same problem — no
primary artifact on disk reproduces it. **Clock both directly (wall-clock at
loop start and end) before either number appears in the write-up**; the
per-tick table above is what is currently defensible, and the rate asymmetry is
real in direction and mechanism even where the exact figure is not yet sourced.

**Chunk length trades reactivity for temporal consistency.** [verified] ACT
Fig. 6a: 1% at k=1 → 44% at k=100, tapering at 200/400. Temporal ensembling
gained only +3.3% for ACT and hurt VINN. RDP Table V: ensembling coefficient
τ=0.2 → 30%, τ=0.5 → 0%, τ=0.8 → 100% — *"very hard to balance."* PACE
(arXiv:2606.00537): success is *"non-monotonic with respect to the execution
horizon"*. Bidirectional Decoding (arXiv:2408.17355): the correct title is
*"Improving Action Chunking via Guided Test-Time Sampling."*

**LeRobot's ACT defaults** [verified, read from the installed package]:
`chunk_size=100`, `n_action_steps=100`, `temporal_ensemble_coeff=None`,
tuned on ALOHA insertion. Ensembling forces `n_action_steps=1`.

---

## 5b. Held-out episodes — the convention, and what a held-out number is worth

Added Sep 4 after Faith raised overfitting from Andrew Ng's train/dev/test framing.
Literature checked Sep 4; every row below was read in the primary source.

### There is no single convention, and the two most-cited numbers disagree

| work | holds out demos? | fraction | where |
|---|---|---|---|
| **robomimic** (arXiv:2108.03298, CoRL 2021) | **yes** | **10%** | §3.2 + App B.2; code default `--ratio 0.1` [verified] |
| **ACT / ALOHA** (arXiv:2304.13705, RSS 2023) | **yes — code only, never in the paper** | **20%** | `tonyzhaozh/act`, `utils.py:114` `train_ratio = 0.8` [verified] |
| Diffusion Policy (arXiv:2303.04137) | nominal | **2%** (`val_ratio: 0.02`) | task configs; "validation" appears 0× in the paper [verified] |
| SmolVLA (arXiv:2506.01844) | **no** | — | [verified] |
| Open X-Embodiment (arXiv:2310.08864) | **no** | — | 3,600 physical trials instead [verified] |
| BridgeData V2 · DROID · π0 · π0.5 | **no** | — | `openpi/scripts/train.py` has no validation loop at all [verified] |
| LeRobot | optional, **off** | `eval_split: float = 0.0` | no doc recommends any fraction [verified] |

robomimic §3.2, verbatim: *"We also split all datasets and data subsets into
training (90%) and validation (10%) subsets… Models were not trained on the
validation subsets."* ACT states only *"we load the policy that achieves the
lowest validation loss"* (§III) — the 80/20 split exists solely in its code.

**The dominant practice in modern large-scale robot IL is to hold out nothing and
evaluate on physical rollouts.** That is the honest headline.

### The finding that matters more than the fraction: validation loss does not predict success

**robomimic §4.5** [verified]: selecting the checkpoint by best validation loss
gives *"10% to 100% decrease"* against the best policy. **Appendix G** is blunter —
on Square (PH) the best policy scores **80.7 ± 0.9** while the lowest-validation-loss
policy scores **2.7 ± 1.9**; on Transport (PH), **64.0 ± 2.8** versus **0.7 ± 0.9**.
And: *"Success rate can increase even while validation loss increases substantially…
This further shows that validation loss is a poor measure of policy performance."*
robomimic holds out 10% and then **does not use it for selection** — §4.5:
*"we evaluated every policy checkpoint online and reported the best one."*

**Also from Appendix G, and decisive for sizing:** raising validation from 10% to
30% *"does not improve policy selection"* while costing 20% of the training data.
There is no return on a generous holdout.

Caveat to state: robomimic's evidence is BC-RNN/BC in robosuite simulation, not ACT
or a VLA on hardware. It is the strongest published evidence, not a direct
measurement on our architectures. A 2026 preprint agrees but is **not peer-reviewed**
— Huang et al., *"Critical Interval MSE"*, arXiv:2606.29898: raw MSE correlates with
rollout performance at only **Spearman ρ = −0.61**.

**Consequence for this project: the held-out set is a leakage check and an
overfitting tripwire, not a policy-selection criterion and not a success-rate
predictor.** Do not pick a checkpoint by held-out loss. Do not report it as if it
forecast the bench.

### What v4 does, and why

**58 recorded · 50 trained · 8 held out = 13.8%** — between robomimic's stated 10%
and ACT's implemented 20%, but **the size is not derived from either.** It is
derived backwards from the experiment: v4 exists to compare staged (v3) against
in-trajectory (v4) recovery data, so the training set is pinned at **50 — exactly
v3's — and the reserve raises the recording count instead of cutting training.**
Had it cut training to 44, a data-quantity confound would sit inside the one
comparison the dataset was recorded to make. Say it that way in the write-up:
*"sized so the training set matches v3 exactly"* is defensible; *"13.8% because
that is standard"* would be false.

**Eight, not six**, because six left no LEFT-MIMIC and no ROT-45° in the reserve —
it could not have measured generalization on two of the five behaviours under test.
Eight buys one of every category plus 3 CLEAN.

**Scattered, not the tail.** `eval_split` takes the LAST ceil(n×split) episodes
(`factory.py:152`), i.e. the most fatigued stretch of the session, and in the seeded
schedule a CLEAN-heavy one. The reserve is sampled per category from episodes
2..N-5 and excluded with `--dataset.episodes`. **Stated cost:** `eval_steps`
requires `eval_split > 0`, so there is no live train-vs-validation curve and no
early-stopping signal — held-out loss is computed offline across saved checkpoints.
Given §4.5 above, that loss is a tripwire, not a selector, so the loss is small.

### Disclosed: the normalizer sees the held-out episodes

**Measured, not assumed.** `--dataset.episodes` filters which *frames are served*
(verified: 10,479 → 8,379 on a 40-of-50 subset) but `lerobot_train.py:314`
normalizes with `dataset.meta.stats`, which `dataset_metadata.py:219` loads from
`meta/stats.json` — written at **record time over every episode**. The two stat
dicts are byte-identical. **This is the same leak ACT's official implementation
has** (`norm_stats` computed over all episodes before the split).

Magnitude, computed from the per-episode statistics on v3 at the same ~14% ratio:

| | worst shift |
|---|---|
| normalization mean | **0.017 σ** |
| normalization std | **1.3%** |

So the held-out frames move the normalization constants by under two hundredths of
a standard deviation. It is real transductive leakage and it is disclosed rather
than engineered around; a clean fix exists if ever wanted (per-episode stats are
stored under `meta/episodes/**` as `stats/<key>/{mean,std,count}` and
`aggregate_stats()` recombines them count-weighted, so a training-only
`stats.json` can be built for a derived 50-episode dataset). **An earlier draft of
this section claimed the normalizer sees no held-out frame. That was wrong;
corrected here by measurement.**

### The real test set is the robot

In manipulation the physical rollout is the test set — the policy has never seen
that episode and success is scored on the task, not on action MSE. A held-out
demonstration set answers a narrower question: does the policy reproduce
demonstrated actions on frames it never saw. Both dry-runs currently answer an even
narrower one — `pi05_dryrun.py`'s header reads *"Feed pi0.5 frames it was TRAINED
on"* — so π0.5's 0.035 and SmolVLA's 0.010/0.011 are training-set reproduction
figures. Re-point them at held-out episodes after v4 and they become generalization
figures.

## 6. The v4 recording protocol, derived

Each rule names its source. Rules that reverse a v3 decision say so.

1. **Same rig, same cameras, same calibration, same single start position (the
   D mark) as v2 and v3 — 50 TRAINING episodes.** *(Corrected Sep 5: this said
   "50 episodes", which is the trained count, not the recorded one.)* v4 records
   more than it trains on so the held-out reserve does not shrink the training
   set: **62 recorded → 4 voided → 8 held out → 50 trained** (30 CLEAN + 20
   recovery, identical to v3, so the v3↔v4 comparison carries no data-quantity
   confound). Recorded index → declared type → status lives in
   `tools/v4_manifest.json`, **not** in the schedule's positional table — the
   moment an episode is voided the table stops being true. Camera position is the hardest factor
   to generalize across (Xie). v2/v3 deliberately used one position; the
   certified scene reference, the replay arbiter (v3 ep0) and every preflight
   gate are built on D, and the v3-vs-v4 comparison needs the same start.
   In-position diversity comes from the knock and backward categories. The
   five-zone schedule (SmolVLA's SO-101 protocol, 10 per position) is a v5
   consideration, not a v4 change — changing two things at once would make
   the staged-vs-in-trajectory comparison unreadable.
2. **One grasp strategy, written down before episode 1 and held for all 50:**
   grip height on the tube, cap orientation (image-left), approach direction.
   (Demo-SCORE, robomimic PH vs MH.)
3. **Slow, deliberate pre-contact alignment; do not rush the last 2 cm.**
   (FACTR 2 pre-contact up-sampling.) Demonstrate a small, consistent
   alignment search at the rack mouth — tilt-then-slide or a short spiral —
   rather than one confident push. (Qiao 1996; Van Wyk 2018; Park 2020.)
4. **Keep the decisiveness rule for nominal episodes.** *Reverses my Sep 3
   advice.* Pauses at the grasp/insert are what make ACT idle (Chen 2025:
   90% of insertion failures). Abort and re-record a hesitant nominal take;
   do not keep it.
5. **Recovery episodes are in-trajectory, arm-induced, and continue to
   seating — VBT, not RaC.** Approach → graze/knock/meet resistance → notice →
   correct → transport → seat → **release into the funnel, jaws close**, one
   episode. *Reverses v3's hand-staging.* RaC's end-after-correction rule does
   not transfer: no policy is in the loop.

   ⚠️ *Corrected Sep 5.* This rule previously said episodes end "while still
   holding the seated tube (standing rule)" — **the opposite of what v4 is
   actually recording.** `Start Schedule — v4.md` rule 7 declares "Episode end
   as v2/v3: release into the funnel, jaws close, then reset", and all 62 rows
   end in `release`. The standing rule it cited (README, "End every episode
   while still holding the object") was written for the **dead voice-fetch
   task**, where the release was a handover to a human hand: the policy cannot
   see gripper load, so it could only ever learn "open N timesteps after
   reaching the handoff pose" and would drop objects into empty air. Releasing
   into a **fixed, visually-determined rack funnel** is a different act — the
   target is in the image, and v2 and v3 both recorded it that way. Following
   the old wording mid-session would have made episodes 11+ inconsistent with
   the 11 already on disk.
6. **Knock ≈ 1.0 cm, image-right — ACT-v3's measured first-contact
   displacement** (non-grasp contacts 0.7 / 1.0 / 1.7 cm on B2-B-01,
   B2V-B-01, B2R-A-02; grasp-and-carry events excluded; front-camera cap
   track at 12.8 px/cm, Sep 4). DART: match the policy's own error; smaller
   beats larger. v3's 1.4–1.5 cm was at the top of that range, so this trims
   rather than reverses the magnitude — the real change is *who* produces the
   displacement: the arm, in-trajectory, not a hand before the episode.
7. **Recovery episodes may hesitate — the hesitation is the behaviour.** The
   decisiveness rule is suspended *only* for the recovery segment of a
   recovery episode, on the strength of RaC's "recovery is locally
   suboptimal" — this is the one rule that rests on [partial] evidence.
   Nominal segments of the same episode stay decisive.
8. ~~**Add a backward-learning block.**~~ **DROPPED by Faith, Sep 4 — replaced
   by RESIST.** The InsertionNet Alg. 1 proposal (seat by hand, record the pose,
   perturb ±10 mm / ±10°, demonstrate the correction back to seated) does not
   apply to this rig: **the coned tube self-centres**, so a partly-seated
   wrong-angle state does not arise. *(Note the tip was never sawn off —
   the project notes' task line is wrong about that; the uncut coned tube was used for
   every dataset and trial. the lab notebook (private), Aug 16.)*

   **RESIST (5 episodes) took its place**, and is the only category that
   exercises the load/current thesis directly: carry to the rack, meet
   resistance at a wrong position or angle, feel it through the leader, back
   off ≈1–2 cm, re-align, insert. It is the descendant of v2's insertion-recovery
   demos (eps ~45–48). The v4 categories are **CLEAN · RESIST · NUDGE ·
   LEFT-MIMIC · ROT-20° · ROT-45°** — there is no BACKWARD. Each is still sized
   as its own category so the held-out reserve draws one of every type.
9. **Reserve held-out episodes at recording time**, whole episodes, per
   category. Pass `--save_checkpoint_to_hub=true`. (Standing rule; the v3
   handoff.)
10. **Recovery data is a large fraction, not a garnish.** RaC: success scales
    linearly with recovery maneuvers; DROID has 3.68%. Target on the order
    of a third of episodes containing a recovery — decided before
    recording, written into the start schedule.
11. **Never move a camera; if one moves, the dataset restarts.** (Standing
    rule, now with Xie et al. behind it.)
12. **Per-phase scoring is designed before recording** — R/G/T/Rel/S stages —
    so a seating-phase-only load effect is detectable (InsertionNet, FMB).

---

## 7. Positioning against the nearest neighbours

**Phaser** — Chen, Tang, Zhang, Kosuge, Hirata, *"Phase-Conditioned
Imitation Learning with Autonomous Failure Recovery for Robust Deformable
Object Manipulation,"* arXiv:2605.29407, submitted 28 May 2026. [verified,
full text read Sep 4]

What they did: dual-arm **deformable** manipulation (hanging/removing a
T-shirt) with **6-axis F/T sensors** (wrenches ∈ ℝ¹²), hybrid impedance
control, haptic bilateral teleop, four RGB cameras. Contribution is a
**FiLM-conditioned ACT encoder** keyed on task phase to resolve state
aliasing, plus a separate **phase predictor** fusing vision+force+pose that
detects failures and triggers recovery. 56% → 87% via autonomous recovery.
15 Hz inference.

Their own stated limits (§V): (a) faster execution broke the system —
*"faster execution drives the manipulated object into out-of-distribution
states"*; future work is demos at varying speeds or online RL. (b)
Single-garment, RGB-dependent; future work is point clouds/depth.

Where this project differs — and the differences are the contribution:

| axis | Phaser | this project |
|---|---|---|
| force signal | two 6-axis F/T sensors + impedance control | servo `Present_Load`/`Present_Current` — free on every hobby servo, no added hardware |
| where force acts | mainly in the **detector** (phase predictor); in the policy as one token | in the **policy observation only** — the question is whether the free signal alone helps |
| task physics | deformable: snag, tension | rigid coned-tube insertion, chamfer intact; jam/wedge in two-point contact (Whitney) |
| design | a full system stack; no ablation isolates the force channel | **single-variable, pre-registered, paired A/B** — same arch, same data, one channel |
| architectures | one (ACT) | ACT and SmolVLA, both arms; π0.5 as breadth |
| arm cost | research dual-arm platform | $300 SO-ARM101, 3D-printed |
| recovery | autonomous, detector-triggered | dataset-level (v3 staged vs v4 in-trajectory) — a comparison nobody has measured |

They explicitly describe force-as-policy-input (FILIC) as *"open-loop,
lacking explicit mechanisms to recover"* — i.e. they moved on from the
question this project asks. Whether raw effort in the observation helps at all,
on the cheapest possible signal, with a design that can attribute the effect
to the channel, is not their result.

**FACTR 2** (arXiv:2606.12406) is the closer neighbour on the *thesis* — raw
current + ACT + insertion. Position against it honestly: they report the
headline; this project replicates it on a $300 arm with servo telemetry
rather than joint-torque-class hardware, across two architectures, under a
pre-registered protocol, with negative results reported. That is replication
+ extension + methodology, and it is stated as such.

**The contributions that are actually ours**, in the order a reviewer will
weigh them:

1. A pre-registered, paired, single-variable A/B on a $300 arm, with gates
   that abort in code and a same-cycle replay arbiter — and a record of three
   plumbing faults those gates caught before they became "results."
2. The staged-vs-in-trajectory recovery-data comparison (v3 vs v4), which
   RaC asserts and nobody has measured.
3. The observation-dimensionality confound named up front (robomimic §4.3),
   with the image-based mitigation.
4. Cross-architecture transfer of the load effect (ACT vs SmolVLA).
5. Per-phase failure taxonomy across three architectures on one task.

---

## 8. Citation table (verified against primary text unless marked)

| Paper | Venue | ID |
|---|---|---|
| Mandlekar et al. — What Matters in Learning from Offline Human Demonstrations (robomimic) | CoRL 2021 Oral | arXiv:2108.03298 |
| Zhao, Kumar, Levine, Finn — ACT / ALOHA | *no venue on arXiv record* (commonly RSS 2023) | arXiv:2304.13705 |
| Chi et al. — Diffusion Policy | RSS 2023 | arXiv:2303.04137 |
| Belkhale, Cui, Sadigh — Data Quality in Imitation Learning | NeurIPS 2023 | arXiv:2306.02437 |
| Gandhi, Karamcheti, Liao, Sadigh — Eliciting Compatible Demonstrations | CoRL 2022 | arXiv:2210.08073 |
| Chen, Lessing, Liu, Finn — Demo-SCORE | preprint | arXiv:2503.03707 |
| Agia et al. — CUPID | CoRL 2025 | arXiv:2506.19121 |
| Chen et al. — Exploiting Policy Idling | IROS 2025 (per arXiv comments) | arXiv:2508.15669 |
| Xie, Lee, Xiao, Finn — Decomposing the Generalization Gap | *no venue on record* | arXiv:2307.03659 |
| Lin et al. — Data Scaling Laws in IL | ICLR 2025 | arXiv:2410.18647 |
| Shukor et al. — SmolVLA | preprint | arXiv:2506.01844 |
| Ross & Bagnell — Efficient Reductions for IL | AISTATS 2010 | PMLR v9 |
| Ross, Gordon, Bagnell — DAgger | AISTATS 2011 | arXiv:1011.0686 |
| Kelly et al. — HG-DAgger | ICRA 2019 | arXiv:1810.02890 |
| Mandlekar et al. — IWR | preprint | arXiv:2012.06733 |
| Laskey et al. — DART | CoRL 2017 | arXiv:1703.09327 |
| Brandfonbrener et al. — VBT | ICRA 2023 | arXiv:2210.02343 |
| Hu et al. — RaC | preprint | arXiv:2509.07953 |
| Wong et al. — Error-Aware IL | CoRL 2021 | arXiv:2112.05251 |
| Ge et al. — EgoRecovery | preprint | arXiv:2607.19745 |
| Oh et al. — FACTR 2 | preprint | arXiv:2606.12406 |
| Liu et al. — FACTR | RSS 2025 | arXiv:2502.17432 |
| Luo et al. — FMB | IJRR 44(4) | arXiv:2401.08553 |
| Spector & Di Castro — InsertionNet | RA-L 2021 | arXiv:2104.14223 |
| Xue et al. — Reactive Diffusion Policy | RSS 2025 | arXiv:2503.02881 |
| Black, Galliker, Levine — Real-Time Chunking | NeurIPS 2025 | arXiv:2506.07339 |
| Liu et al. — Bidirectional Decoding | ICLR 2025 [partial] | arXiv:2408.17355 |
| Nie et al. — PACE | preprint | arXiv:2606.00537 |
| Whitney — Quasi-Static Assembly of Compliantly Supported Rigid Parts | ASME J. Dyn. Sys. 1982 | DOI 10.1115/1.3149634 |
| Lian et al. — Benchmarking Off-The-Shelf Solutions to Robotic Assembly | IROS 2021 | arXiv:2103.05140 |
| Van Wyk et al. — search strategies | IEEE T-RO 34(2) 2018 | DOI 10.1109/TRO.2018.2791591 |
| Qiao, Moore, Knight — chamferless assembly | Robotica 14(6) 1996 | DOI 10.1017/S0263574700018518 |
| Luo, Xu, Wu, Levine — HIL-SERL | preprint | arXiv:2410.21845 |
| Chen, Tang, Zhang, Kosuge, Hirata — Phaser (ACT + force + recovery) | preprint, T-Mech accepted (unconfirmed) [partial on venue] | arXiv:2605.29407 |
| Hsu et al. — Vision-Based Manipulators Need to Also See from Their Hands | ICLR 2022 oral | arXiv:2203.12677 |

Corrections applied: Gandhi et al. has no author named Lee. VBT is 2210.02343,
not 2210.02106. Bidirectional Decoding's title is *"Improving Action Chunking
via Guided Test-Time Sampling."*

---

## 9. Do not cite — searched for, could not verify

- **"LeRobot recommends holding out 10% of data or a maximum of 32 episodes."**
  Surfaced by web search Sep 4; **not present anywhere in the LeRobot source or in
  any of the 100 files under `docs/source/`.** The only `eval_split` mention in the
  docs sets it to 0.0. Treat as a search-engine confabulation.
- robomimic's project page says the best-validation policy is *"50 to 100% worse"*;
  the paper says **10% to 100%**. Cite the paper.

- "Chhatpar & Branicky: tilt strategy 100% over 60 trials, 7.1 s vs >40 s" —
  circulates in summaries; no openable source.
- "Adding a wrist camera to RT-2 improved success 15–25%" — vendor page only.
- Open X-Embodiment says nothing about *how* to record demonstrations; it
  pools 60 datasets. Its actual nuance is that RT-1-X did not beat RT-1 in the
  large-data setting.
- Kumar et al. (ICLR 2022) does **not** support "noisy demos are fine for BC."
- "Woodpecker effect" as a community term — traces to one HF blog post
  (sherryxychen, 2025-09-30), pick-and-place, zero independent uses.
  the project notes' "has a community name" should be softened.
- No peer-reviewed source on episode-boundary conventions (recording the
  release). The README rule is sound inference; say so.
- No controlled ablation of decisiveness-filtering against recovery ability.
- No cross-architecture correction-data transfer study.
- Any claim that "balanced mixing ratios are robust to different policy
  architectures" — Sirius holds architecture fixed.
- IntervenGen's "39×" figure — title/authors high-confidence, number not read
  from the primary source.
