# tools/ — arm and camera diagnostics

Read-only diagnostic scripts written Aug 7–8, 2026 while fixing the calibration
and bringing up the cameras. They exist because `lerobot`'s CLI can't answer
these questions.

**Always activate the env first** — the CLI and libraries live only there:

```bash
conda activate lerobot
cd "<repo>/tools"
```

12V must be ON and applied **before** USB, or the servo bus is dark.

| Script | Use it when |
|---|---|
| `identify_arms.py` | "Which arm is on which port?" Compares live motor registers to the calibration files. Run after any cable swap you're unsure about. |
| `watch_gripper.py` | Ground truth for the above — squeeze one arm's jaws and see the number move. Owes nothing to any file. |
| `watch_load.py` | **Turns "squeeze gently" into a number.** Reads `Present_Load` / `Present_Current`, which `get_observation()` does NOT expose. `--close` steps the jaws shut on an object and stops at a load ceiling, so you can measure a safe threshold per object without reproducing the Aug 8 overload latch. ⚠️ Needs the serial port to itself — stop teleop first. |
| `wrist_align.py` | After ANY recalibration. Live **normalized** wrist_roll on both arms; the only valid alignment test. |
| `gripper_measure.py` | After ANY recalibration, and any time the gripper stalls or flashes. Measures safe endpoints with gripper torque off. |
| `patch_gripper.py` | Fixes a bad gripper sweep **without** a full recalibration. Takes endpoints from `gripper_measure.py`. |
| `probe_cameras.py` | Before every recording session. Which index is which camera, is the frame black, do requested settings apply. Replaces `lerobot-find-cameras` for day-to-day use. |
| `view_cameras.py` | Aiming and framing. Live windows at recording resolution, with exposure warnings and rule-of-thirds guides. Use it to get all 4 grid positions in the fixed view. |
| `check_exposure.py` | "Can a policy actually read this view?" Live mean / % blown / % crushed / contrast for ONE camera, with clipped pixels painted red so you see *where* detail is lost. `--test-exposure` probes whether manual exposure can be locked on this backend. |
| `episode_exposure.py` | After recording: how much did auto-exposure drift *across* an episode? Runs on the encoded video, so it measures the frames the policy actually trains on. Read the SWING. |
| `_arms.py` | Shared ports/config. Not run directly. |

## The two mistakes these encode

**1. Never compare the two arms' raw `homing_offset` values to judge wrist_roll alignment.**
`homing_offset = 2047 − raw_at_ENTER`, and the raw encoder value depends on how each
servo's shaft was assembled into its wrist at the factory. Two arms read *different*
raw counts at the *same* physical orientation, so a nonzero gap at perfect alignment
is expected and tells you nothing. The valid test is behavioural — equal physical
orientation should give equal **normalized** position. That's `wrist_align.py`.

**2. A gripper endpoint must be safe UNDER TORQUE, not merely reachable by hand.**
The `lerobot-calibrate` range sweep records wherever you drag the joint, so hand-forcing
past the safe stop writes a position the servo will later stall trying to hold — jaws
clamped, servo flashing overload. Frame-independent check: compare range **widths**,
not endpoints, because `range_min`/`range_max` are raw values recorded *after* new
homing offsets are written and are meaningless across calibrations.

Known-good gripper widths: **leader ≈ 1216–1230, follower ≈ 1393–1401.**
Roughly double that means the sweep over-travelled.

**3. "No cameras were detected" on macOS usually means permission, not USB.**
Camera access is a per-app TCC grant. Give it to Terminal (or iTerm, or Code)
under System Settings → Privacy & Security → Camera, then **fully quit and
relaunch** that app — the grant is read at process launch, so a new tab keeps
the old denial. Until then every index fails to open and `lerobot-find-cameras`
reports no cameras and writes no `outputs/captured_images/`, which reads exactly
like a dead hub. Cost an hour of dock debugging on Aug 8, 2026. `probe_cameras.py`
surfaces the real error instead of swallowing it.

## Typical repair flow

```bash
python gripper_measure.py                                  # move both by hand, note min/max
python patch_gripper.py --follower 2048 3449 --leader 1960 3190
```

Then at the next `lerobot-calibrate` / `lerobot-teleoperate` prompt press plain
**ENTER**, not `c` — ENTER writes the patched file into the motors, `c` throws it away.

Once teleop looks right, re-copy the snapshot:

```bash
cp -R ~/.cache/huggingface/lerobot/calibration/. "<repo>/calibration-backup/"
```

Ports are hardcoded in `_arms.py`. They're derived from the CH343 chip serial, not the
USB port, so they survive cable swaps and hubs — they only change if an arm is replaced.

## v4 training — the offline hold-out check (Sep 5, 2026)

W&B shows the train loss, which cannot see overfitting. Every v4 run pushed all its
checkpoints (`save_checkpoint_to_hub`), so the check is done here, offline, on the 8 A4
hold-out episodes no policy has seen:

```bash
python v4_holdout_loss.py faithqin/act-tube-A-v4              # every checkpoint, every frame
python v4_holdout_loss.py faithqin/pi05-tube-A-v4 --stride 6 --batch 4
./v4_holdout_all.sh 3 0                                        # the four completed runs, sequentially
```

It loads the weights AND the processors each checkpoint saved (training normalisation, not a
re-estimate), runs `policy.forward` in eval mode exactly as `lerobot_train`'s eval step does,
and reports hold-out vs a type-matched sample of 8 training episodes, per checkpoint and per
recovery type. It refuses to score unless the run's `train_config.json` trained on the split
`v4_training_split.py` emits, the hold-out is disjoint, and the local dataset copy matches the
Hub's counts. Numbers compare across checkpoints of ONE run only (ACT is L1+KL, the VLAs are
flow-matching MSE). Results: `analysis/v4_holdout/<run>.json`.

> **⚠ Sep 12 — the hold-out is SPENT.** Episodes 11, 19, 25, 31, 35, 50, 51, 57 had their last
> use in the Sep 7 endpoint-2 instrument test. Do not load, score or plot them again, and every
> command in this section does. The verdicts are already on disk for all six runs
> (`analysis/v4_holdout/{act,smolvla,pi05}-tube-{A,B}-v4.json`), which is all
> `v4_verdict_checkpoint.py` reads. The same goes for `v4_input_ablation.py` and
> `v4_contact_loss.py` (below), and for the `v4_holdout_all.sh` line `run_pair.sh` prints when a
> verdict file is missing (the VLA checkpoint ABORT in `tools/run_pair.sh`).

## v4 bench — the pre-grasped start, the blinded pairs, the receipts (Sep 6, 2026)

Everything the Sep 6 paired A/B bench runs through, and the mistake each piece encodes.

```bash
tools/run_pair.sh <pair_id>                                 # one pair from the committed schedule, blinded at the console -- P01 is PENDING a decision (bench sheet, Progress)
CAPSTONE_RESERVE_AS=S0-CLEAN tools/run_pair.sh P26         # a reserve pair (P26-P28, all ACT) re-drawn for a voided pair: declare THAT pair's stratum
CAPSTONE_PAIR_TRIAL=2 tools/run_pair.sh P07                 # resume at trial 2 after a gate abort there
python tools/v4_schedule.py                                 # regenerate schedule.csv / schedule_6h.csv (seed 20260906, byte-identical)
python tools/v4_verdict_checkpoint.py faithqin/pi05-tube-B-v4   # the checkpoint the hold-out rule picks (015000)
python tools/v4_carry.py                                    # the PRE-GRASPED pose, derived from the training carry frames
python tools/home_arm.py --pose v4-carry --close-on-tube   # REFUSED since Sep 6 14:17 (norm_units.py, below) -- S2 is out
CAPSTONE_START_POSE=v4-carry tools/run_scored_trial.sh faithqin/act-tube-B-v4 25 off <label>   # a manual S2 trial -- aborts at preflight today, same refusal
python tools/v4_bench_analysis.py --ledger tools/scored_logs/trials.csv --pairs analysis/bench_sep6/sealed/pairs.csv --scores analysis/bench_sep6/scores.csv
python tools/replay_telemetry.py --episode 0                # certification replay -- MOVES THE ARM; writes the block's contact baseline
python tools/v4_contact_events.py --telemetry tools/scored_logs/<token>_<stamp>_telemetry.csv --baseline <CERT csvs> --actions-parquet <rollout parquet> --label <token> --scalars-out <token>_scalars.csv
```

- **`v4_schedule.py` / `run_pair.sh`** — the arm order and the labels are drawn once, seeded,
  and committed; the launcher reads them and prints only pair / stratum / trial k of 2, the
  gate lines, one arm-free HEALTH line (`trial_health`) and the verdict. The runner banner
  (`policy=`), `side=`, the `CAPSTONE_STATE_ROUTING WIDE/narrow` line and the state width all
  announce the arm, so they go to `analysis/bench_sep6/sealed/`
  with the token→arm key and `pairs.csv`, opened only after scoring. The blinding is
  procedural (the seed is public); what it protects against is a glance, not a determined reader.
- **`v4_verdict_checkpoint.py`** — VLA checkpoints come from `analysis/v4_holdout/<run>.json`
  at launch time, never from a literal in a script (SmolVLA's optimum is 5k, π0.5-B's 15k,
  π0.5-A's 20k: three answers from one rule). No curve → nothing on stdout → the runner aborts.
- **`v4_carry.py` + `--close-on-tube` + `CAPSTONE_START_POSE`** — the S2 start is a MODIFIED
  TASK (the pick is done by hand). Its gripper window is the tube's width (12.13–15.42), not
  the frame-0 "closed on nothing" bound, so an empty jaw parked at 13.2 is indistinguishable
  by position: the close verifies contact load too (|load| ≥ 90; training carry 98–120, free
  air 56–80). `shoulder_lift` is +45 there with no stop under it, so neither homing path
  releases it (the arm would drop, holding the tube). The runners photograph the scene from
  frame 0 first and `preflight` gates the RACK for a carry start, since the tube is in the jaws.
- **`preflight.parse_tube_delta`** — found Sep 6: with `re.S` a "tube: NOT FOUND" line ran on
  into the rack line and a MISSING tube gated green at the rack's 1.5 px. Same-line match now.
- **`loop_timer.py`** — every trial's `.verify` receipt carries `loop_hz=` (frames − 1 over the
  wall-clock span between the first and last recorded frame) beside `start_pose=`. The 13.4 /
  17 Hz figures in the notes were inferred, never measured, and stay barred until a receipt
  carries the number.
- **`v4_bench_analysis.py`** — paired Wilcoxon + Hodges–Lehmann, mid-p McNemar + Wilson + the
  MDE printed beside the binary, RMST to seat, pair-clustered bootstrap, the exhaustive
  disposition table over BOTH ledgers (`trials.csv`, `trials_smolvla.csv`). Statistics by hand
  (no scipy in the env), each tested against a hand-enumerated case.
- **`v4_contact_events.py`** — onset / peak / dwell / impulse from the telemetry CSV against the
  block's certification-replay baseline, and the latency to the first opposing command. The
  per-block baseline is written by `replay_telemetry.py` (next section) since Sep 6 13:57; the
  manipulation check (`--manipulation-check`) runs on the v4 TRAINING RESIST/NUDGE episodes offline.
  **Gate G-T is enforced on the TRIAL, not only the baseline** (`check_channels_live`, Sep 6 17:20
  amendment). Until then the refusal guarded the baseline alone, so a trial whose 12 load/current
  columns were NaN or frozen printed "0 event(s)" and exited 0 — indistinguishable from a trial
  where the arm never touched the rack, and that zero was eligible to enter co-primary endpoint 2.
  The same silent shape was closed on the NaN ACTION stream (every latency undefined, with an
  ordinary-looking reason) and on `--baseline-json`, the one route to `load_z` that bypasses
  `Baseline.from_states`.

Every piece is executed under stubs in `tests/` (`test_run_pair.py`, `test_v4_start_pose.py`,
`test_v4_carry.py`, `test_v4_loop_timer.py`, …) and was mutation-checked before it was committed.

## v4 bench, continued — the baseline replay, the unit guard, the endpoint-2 work (Sep 6–7, 2026)

Added during and after the Sep 6 bench. Two of these are unratified PROPOSALS and say so on every
run; one is not on the bench path at all.

```bash
python tools/replay_telemetry.py --episode 0                  # certification replay -- MOVES THE ARM; home first with --pose v4
python tools/v4_contact_events.py --telemetry <token>_<stamp>_telemetry.csv --baseline <CERT csvs> \
    --actions-parquet <rollout parquet> --label <token> --scalars-out <token>_scalars.csv
python tools/v4_contact_events.py --telemetry ... --baseline ... --secondary-panel    # OPT-IN: the unratified panel beside the endpoint
python tools/v4_contact_events.py --anchor-evaluability                               # desk, training episodes only (proposal)
python tools/v4_contact_response.py --training-check                                 # desk, training episodes only (proposal)
```

- **`replay_telemetry.py`** — the certification replay that writes the 18-channel baseline the
  contact endpoint z-scores against (`tools/scored_logs/CERT-<stamp>_telemetry.csv` by default).
  `replay_tracking.py` could not be extended to do it: it logs positions only, and it drives
  through `_arms.py` in RANGE mode while trials run in DEGREES, so its baseline would be in the
  wrong unit. This drives the same patched `SOFollower` a trial drives and tees through the same
  `telemetry_sidecar`. It refuses to start more than `--max-start-gap` (15°) from the episode's
  first pose, refuses an observation with no load/current channels (exit 3), aborts on a jam —
  an arm joint at |load| > 700 that has not moved 0.5° for 6 ticks (exit 5) — and hands the CSV to
  `Baseline.from_csvs` before it prints `BASELINE OK` (exit 4 if rejected). Opens no camera.
  ⚠ Its pacing changed on Sep 7 to an absolute deadline — the five replays it had written ran
  18.52–18.63 Hz against the dataset's 20 fps. Tested on a fake clock only; watch the first
  certification replay's rate on the arm.
- **`norm_units.py`** — why `home_arm.py --pose v4-carry` is REFUSED (exit 2, during argument
  checking, before the bus opens). `_arms.py` normalises RANGE_M100_100 and the pose tables are
  DEGREES: same zero, different scale, so the error grows with distance from the calibration
  midpoint — small at the frame-0 poses, about 40° on wrist_roll at the carry pose, with a hand in
  the workspace. A pose whose worst gated joint is off by more than 2.5° is refused
  (`shoulder_lift` and the gripper exempt). The guard, not the fix: the fix is `_arms.py` in
  DEGREES, tracked by the strict xfail in `tests/test_arms_norm_mode_matches_production.py`, and
  it changes v2/v3/v4 homing, so it needs its own bench check. Since Sep 12 `run_pair.sh` asks the
  same question before it seals an S2 pair (`carry_pose_offenders`) and refuses with nothing
  written, instead of sealing the pair and failing at the runner's homing gate.
- **`home_converge.py`** — **NOT on the bench path.** Nothing imports it, no test covers it, and
  no code in `tools/` or `tests/` names it (checked Sep 12). Its docstring reads like a live
  account ("Every scored trial today would abort at the home gate") and describes an iterative
  shortfall-correction home; the homing the bench actually runs is `home_arm.py`'s approach from
  ABOVE for `elbow_flex` and `wrist_flex` (`APPROACH_FROM_ABOVE`). Delete it or wire it in with a
  test — do not call it.
- **`telemetry_sidecar.py`** (Sep 6, pre-bench) — every trial tees all 18 servo channels, one row
  per recorded frame, to `<label>_<stamp>_telemetry.csv` (`CAPSTONE_TELEMETRY_CSV`, set by the
  runners), whatever the policy is allowed to see: an A trial's dataset stores only 6 dims.
  `HEADER` is the one definition of the column order; the replay and the contact tools read it
  rather than retyping it.
- **`v4_contact_events.py`, Sep 7.** Five engineering repairs that change no anchor, baseline or
  channel: times from the sidecar's own `t_mono`; the opposing-command search bounded to the
  event plus `RESP_GRACE_S` (1.0 s), capped at `RESP_MAX_S` (3.0 s) from onset; a command already
  opposing at onset reported as a LEAD, never 0.00 s; a load channel sitting exactly at the 500
  torque cap (in practice `gripper.load`) counted as censored, its peaks flagged as lower bounds;
  and the three scalars. **`--scalars-out`** writes
  `label,contact_latency_s,contact_impulse,reached_T` — a fragment, not a `scores.csv`, and
  nothing joins it yet. **`--label` must be the bare token**: the default is the file stem,
  `<token>_<stamp>`, which matches no ledger label (checked Sep 12 on T0-SMOKE). Without
  `--actions-parquet` every latency is undefined. **`--secondary-panel` and
  `--anchor-evaluability` default OFF and reach an unratified proposal** through a lazy import of
  `v4_contact_response.py`; `v4_bench_analysis.py` imports neither proposal module.
- **`v4_contact_response.py`** — **PROPOSAL, NOT RATIFIED.** The measurement behind withdrawing
  co-primary 2 (`anchor_evaluability`: does "distal-load onset" mark a contact under any baseline;
  training episodes only) and the descriptive panel proposed in its place (retries by a telemetry
  definition, stall rate, fixed-window impulse, `reached_T` beside the stage letter), each printed
  with its caveat. Its own CLI (`--telemetry`, `--actions-parquet`, `--episode`,
  `--training-check`, `--json`) prints the proposal banner first. Refuses hold-out episodes.
- **`v4_recovery_response.py`** — **PROPOSAL, NOT RATIFIED.** Library only, no CLI: the
  press → retreat → re-press depth statistic inside the rack window, the resistance-recovery
  alternative for co-primary 2. Its docstring says it is "imported by nothing"; since Sep 7
  `v4_contact_response.py` imports it lazily.
- **`v4_input_ablation.py`, `v4_contact_loss.py`** (Sep 6, pre-bench, no hardware) — does a B
  policy's predicted chunk move when load/current is frozen or shuffled; and hold-out loss on
  contact frames only, A vs B on the identical frames. **Both score the held-out episodes — spent;
  see the warning in the hold-out section.**
- **`pi05_sides.py`** (Sep 6) — the π0.5 runner's one side table (repo, dataset, state pin, prompt,
  checkpoint, sanitised snapshot), mirroring `smolvla_dryrun.py --print`: A pins narrow state,
  B wide, because both declare the same padded `[32]`.

Tests, all hardware-free: `test_replay_telemetry.py`, `test_norm_units.py`,
`test_telemetry_sidecar.py`, `test_v4_contact_endpoint_repairs.py`, `test_v4_secondary_panel.py`,
`test_v4_contact_response.py`, `test_v4_recovery_response.py`, `test_v4_input_ablation.py`,
`test_v4_contact_loss.py`, `test_pi05_sides.py`. What each one pins: `TESTING.md`.

## Sep 14, 2026 — relieving torque when a servo is browning out

```bash
python tools/relax_arm.py          # follower: connect WITHOUT the handshake, torque off wrist_flex first, verify by readback
```

- **`relax_arm.py`** — after the three Sep 14 certification replays the arm was left holding ~15 min;
  wrist_flex reached 70 °C (the ledger's brownout is 69 °C) and stopped answering the handshake, and
  the only torque-off path here (`_arms.open_bus` → `bus.connect()`) raised "Missing motor IDs: 4".
  A handshake needs every servo to answer, which is exactly what a browning-out servo does not do.
  This tool connects with `handshake=False`, writes `Torque_Enable 0` joint by joint with
  `num_retry=5` (wrist_flex first), reads `Torque_Enable` back on all six and exits 1 unless every
  one reads 0, then disconnects with `disable_torque=False` (already off; a second `num_retry=0`
  walk could raise). **Run it after every certification replay** — `replay_telemetry.py` leaves the
  arm holding by design — and never between `home_arm.py` and a rollout. Home first if the arm is
  extended over the rack: a released arm sags. `tests/test_relax_arm.py`, mutation-checked.

```bash
python tools/cert_summary.py tools/scored_logs/CERT-B3-01_telemetry.csv     # loop Hz, ee_proxy, jaw_min for the record
```

- **`cert_summary.py`** (Sep 14) — the numbers a certification pass is logged with, from its own
  telemetry CSV and nothing else: `loop_hz` (the receipts' definition), `ee_proxy` (mean of
  shoulder_lift + elbow_flex + wrist_flex over frames 103–202, the the lab notebook (private)'s Sep 14 12:37
  measure), `jaw_min` over the carry window 96–136 against `v4_contact_events.JAW_HELD_MIN_DEG`.
  Refuses a CSV short of the episode's 203 frames. Descriptive: the seat is still the gate, and
  Faith calls it. `tests/test_cert_summary.py` reproduces the published Sep 14 values from disk.

```bash
tools/run_cert.sh CERT-B3-03        # ONE gated certification replay (G-R), from a TCC-granted Terminal; relaxes the arm, exit 0 = ran
python tools/cert_gate.py record tools/scored_logs/CERT-B3-03_<stamp> SEAT|MISS   # the operator's verdict -> <run>.seat
python tools/cert_gate.py temps <label>.ping     # <= 52 C, every servo read       (called by run_cert.sh)
python tools/cert_gate.py home  <label>.home     # HOMED + settled, no WARNING     (called by run_cert.sh)
```

- **`run_cert.sh` + `cert_gate.py`** (Sep 14) — the certification replay as one gated command:
  ping → 52 °C gate → home → settled gate → scene check → the same `preflight.py` a scored trial
  passes → `replay_telemetry.py` → `cert_summary.py` → retract home → `relax_arm.py`, then it prints the
  `record` command and exits. The operator calls seat or miss to the chat (as on Sep 6 and Sep 14
  morning) and `cert_gate.py record` writes `<label>_<stamp>.seat`, once, from the run's own files.
  Every refusal is before the next motion; every exit after motion relieves the arm; nothing waits
  for a keystroke with torque on (the first version did: CERT-B3-02 held three minutes at the rack,
  wrist_flex 45 → 51 °C). Written after the morning's one certification miss turned out to be the
  pass with no scene check. Tests: `test_cert_gate.py`, `test_run_cert.py`.

```bash
python tools/prelaunch_check.py [--ping <file>.ping]   # GO / NO-GO before EVERY launcher or certification start
```

- **`prelaunch_check.py`** (Sep 14 17:44) — the four rules that were "known" and not gated when a pair was
  launched cold and then aborted under a live rollout: no bench process alive (by pid; a rollout's
  command line names the policy, so nothing here prints one), the follower port free, the newest
  `CERT-*.seat` a SEAT no older than 10 min (G-R plus the warm-up-after-a-break rule), and with `--ping`
  every servo ≤ 52 °C. Exit 1 lists every failing reason. `tests/test_prelaunch_check.py`.
- **`relax_arm.py`** now refuses when another process holds the port (`lsof -t`): stop the whole tree
  first (launcher, runner, rollout), then relax. Killing `run_pair.sh` does NOT kill its rollout child.
- **Scoring (`v4_bench_analysis.py`, Sep 14 evening)**: a re-run label whose earlier run carries a `MANUAL:`
  code in `invalidations.csv` is not a duplicate — the surviving row counts; `read_scores` accepts the
  six-column `scores.csv` (co-primary 2 withdrawn); `--stratum S0-CLEAN` restricts the analysed pool to
  the primary (§7 16:59: S1 is exploratory); the binary SEAT is the VIDEO adjudication (stage S), the
  scorer's ledger verdict only cross-checks (§7 16:49: `score_episode.py` is positively biased); an
  invalidation filed against a label with no ledger row reaches the pair's codes (P04). Run:
  `python tools/v4_bench_analysis.py --ledger tools/scored_logs/trials.csv --pairs analysis/bench_sep6/sealed/pairs.csv
  --scores analysis/bench_sep6/scores.csv --schedule analysis/bench_sep6/schedule_s0only.csv
  --invalidations analysis/bench_sep6/invalidations.csv --stratum S0-CLEAN --allow-seat-disagreement`
  — only after the key is opened.

```bash
python tools/bench_stop.py     # abort: stop launcher + runner + rollout by pid, verify, THEN relax the arm
```

- **`bench_stop.py`** (Sep 14 17:45) — the only sanctioned way to abort a running trial or replay.
  Killing `run_pair.sh` leaves the rollout python alive; this stops the whole tree, waits for it to be
  gone, and only then calls `relax_arm.py`. An aborted trial is still logged and, if it ran, voided.

## Sep 22, 2026 — the public snapshot: allowlist, scrub, gate

```bash
python tools/public_repo.py --dry-run                 # list the 227 files the allowlist admits
python tools/public_repo.py                           # copy + scrub into ~/dev/so101-tube-insertion, then gate
python tools/public_repo.py --gate-only               # re-run the gate on the copy
python tools/public_repo.py --review-images           # after LOOKING at every image in the copy
```

- **`public_repo.py`** — builds the public repo (`github.com/FaithQin/so101-tube-insertion`, a NEW repo with fresh
  history, Faith's decision Sep 22) from an explicit allowlist over `git ls-files`, so an untracked file can never
  leak and a held path (the blinding key and everything derived from it, the v4 bench sheet and protocol text,
  handoffs, home camera frames, calibration) is refused even if a rule would admit it. The scrub edits the COPY
  only: local paths, private-notes citations, tooling references, one account URL, two dated notes (the DAgger
  erratum, the v2 wording note), two masked serials, the W&B run ids. Every edit lands in the copy's
  `PROVENANCE.md`. The gate blocks on: held path, home or scratch path, tooling name, forbidden null phrasing,
  a bench token beside an arm label, an unreviewed image (hashes in `public_repo_reviewed_images.json`), an mp4,
  a file over 50 MiB, a non-noreply author, and any leftover draft marker on Faith's README sections. It never pushes, creates a
  repo, or changes visibility. Tests: `tests/test_public_repo.py`. Assets: `public_repo_assets/` (README
  template in Faith's voice, Apache-2.0 text). Neither this tool nor its test is copied into the public repo.
