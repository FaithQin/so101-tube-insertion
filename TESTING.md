# TESTING.md — the regression suite

Created Aug 16, 2026. **Rule: every bug that costs bench time gets a test the
same day**, if it is testable without hardware. The suite is the executable
version of the "blood-bought" rules scattered through the project notes and the
handoffs — docs drift, tests fail loudly.

## Run it

```bash
cd "<repo>" && /opt/homebrew/Caskroom/miniforge/base/envs/lerobot/bin/python -m pytest tests/ -q
```

Everything is hardware-free: no serial ports, no cameras, no TCC. Safe to run
any time. Run it: **before every bench session**, and **immediately after any
`pip install -U lerobot`** (the site-packages tests exist precisely for that).

**What green looked like on the Sep 12, 2026 evening baseline: 871 passed, 3 skipped,
1 xfailed** — 875 collected across 63 test files, before that night's fixes
added more. The 3 skips are `test_smolvla_trial_runner.py` refusing to execute
the real runner unless `CAPSTONE_EXEC_RUNNER_TESTS` is set
(it reaches hardware if its resolve-only hatch ever breaks, and the suite runs
while the arm is powered). The xfail is `test_arms_norm_mode_matches_production.py`,
strict: `_arms.py` normalises RANGE while production is DEGREES. Any other skip,
or an xfail that starts passing, is news.

## What is covered, and the bug each test encodes

| Test file | Bug it prevents |
|---|---|
| `test_home_pose_matches_training.py` | Aug 16: homing targets measured on v1 while probing v2 policies (elbow 82.9 vs v2's 79.7±2.2), AND full-bus torque release letting the unpowered elbow sag to 88.4 — outside the entire v2 training range. Asserts both homing paths carry a v2 pose matching the dataset's measured frame-0 means, agree with each other, and settle ONLY `shoulder_lift`. |
| `test_sitepackages_patches.py` | The four local site-packages patches die silently on every lerobot upgrade. Asserts: `JobConfig.timeout="8h"`; `Present_Load`/`Present_Current` in the observation; gripper stall design (Max_Torque_Limit 500 + Protection_Time pin + `_relieve_gripper` — NOT the stale "300" from the Aug 13 ledger); sync.py queue-skip; context.py 18-dim state filter. |
| `test_calibration_and_dataset.py` | Aug 7: over-swept gripper ranges (width bands, frame-independent); calibration-backup/ drifting out of sync with the live cache lerobot actually reads; v2 dataset invariants (fps=20, 50 eps, 18-dim state). |

First run of the suite caught a real drift: the project notes' ledger and
`tools/so_follower_load.patch` both still said Max_Torque_Limit=300, but the
live code deliberately keeps 500 (300 pegged and latched). Both were corrected
and the patch file regenerated from a live-vs-upstream diff on Aug 16.

## Added Sep 6–7, 2026 — the v4 bench and the endpoint-2 work

The files added between Aug 17 and Sep 5 are not listed one by one here; each
names its date and incident in its own docstring. From Sep 6 on:

| Test file | Bug it prevents |
|---|---|
| `test_telemetry_sidecar.py` | Sep 6: a rollout records the state THE POLICY saw, so an A trial kept no load/current and the contact endpoints had nothing to read. Pins the 18-channel CSV tee, one row per recorded frame, beside every trial. |
| `test_v4_support.py` | Aug 29 and Sep 3: a home/gate table that lacked the newest dataset re-homed every trial to the previous generation's pose. Pins v4's own frame-0 support (elbow 85.61 vs v3's 82.34), re-derived from the parquet. |
| `test_smolvla_checkpoint_option.py` | Sep 5: both SmolVLA-v4 runs overfit by 20k, and the top-level weights ARE the 20k checkpoint. Pins that a v4 side does not launch without `CAPSTONE_CHECKPOINT`. |
| `test_pi05_sides.py` | Sep 6: the π0.5 runner hardcoded one v3 policy and its state pin, and v4-A and v4-B declare the same padded `[32]`. Pins one side table with a per-side pin (A narrow, B wide). |
| `test_v4_input_ablation.py` | Sep 6, pre-bench: a B policy that ignores load/current makes the hypothesis false by construction. Pins the ablation gate — load frozen or shuffled vs a position-ablation control. |
| `test_v4_contact_loss.py`, `test_v4_contact_loss_aggregate.py` | Sep 6: whole-episode hold-out loss averages a contact effect away. Pins contact-frame-only scoring on an identical frame set with paired noise, and the per-episode aggregate over noise seeds. |
| `test_v4_schedule.py` | Sep 6: a schedule drawn at the bench cannot answer "was that decided before the trials?". Pins determinism against the committed `schedule.csv`, the allocation, arm order 3-of-6 per block, and the cut order. |
| `test_v4_bench_analysis.py` | Sep 6: no scipy in the env, so every statistic is written by hand. Each is checked on a hand-computed case, a planted effect it must recover, and a true null. |
| `test_v4_carry.py` | Sep 6: the PRE-GRASPED `v4-carry` pose must come from the training carry frames, and an EMPTY jaw parked at the tube's width must not pass its gate. |
| `test_v4_start_pose.py` | Sep 6: no runner could select the carry pose, so a pre-grasped trial would have homed to frame 0 with the tube in a closed jaw. Runs the three runners under stubs. |
| `test_run_pair.py` | Sep 6: the blinded pair launcher, executed under stub runners — no console line names the arm, pairs are atomic, a void is reported as one, and the pause between trials gates only on a real terminal. |
| `test_v4_loop_timer.py` | Sep 6: the 13.4 / 17 Hz figures were never measured. Pins a wall-clocked `loop_hz=` in every runner's `.verify` receipt. |
| `test_replay_telemetry.py` | Sep 6: the certification replay logged positions only, in RANGE units, so there was no contact baseline. Pins the channel order to the sidecar `HEADER`, the refusals (no load channels, far start, a jam) and, from Sep 7, pacing that does not drift — fake robot, fake clock. |
| `test_norm_units.py` | Sep 6: commanding `v4-carry` through the RANGE-mode bus puts wrist_roll ~40° off with a hand in the workspace. Pins that `home_arm` refuses it and still allows every frame-0 pose. |
| `test_v4_contact_endpoint_repairs.py` | Sep 7, tranche 1: five defects in co-primary 2's instrument — a hard-coded 20 fps clock, an unbounded opposing-command search, a lead reported as a 0.00 s response, a censored gripper channel, and no `reached_T` or per-trial scalars. |
| `test_v4_secondary_panel.py` | Sep 7, tranche 2, **UNRATIFIED**: the default path of `v4_contact_events.py` is still the pre-registered endpoint, the panel is unreachable without its flag, and no analysis script imports the proposal. |
| `test_v4_contact_response.py` | Sep 7, **UNRATIFIED proposal**: the plumbing of `v4_contact_response.py`, and the duration-matched control under which its kinematic stall does NOT survive. |
| `test_v4_recovery_response.py` | Sep 7, **UNRATIFIED proposal**: the press → retreat → re-press statistic in the rack window, and that the module refuses hold-out episodes. |

The last four never load the held-out episodes (11, 19, 25, 31, 35, 50, 51, 57),
which are spent as of Sep 7.

| `test_relax_arm.py` | Sep 14: wrist_flex hit 70 °C holding between certification passes and missed the bus handshake, and the only torque-off path (`_arms.open_bus` → `connect()`) raised "Missing motor IDs: 4" — it needs every servo to answer, at the moment one cannot. Pins `tools/relax_arm.py`: connect `handshake=False`, `disable_torque(j, num_retry=5)` wrist_flex first, readback of `Torque_Enable` on all six or exit 1, `disconnect(disable_torque=False)`. Fake bus; six mutations red (order, retries, readback, handshake, disconnect, open_bus). |

| `test_cert_summary.py` | Sep 14: the three numbers every certification pass is logged with (loop rate, end-effector height proxy over frames 103–202, jaw minimum over the carry window 96–136) were computed by an ad-hoc chat snippet. Pins `tools/cert_summary.py` to the published definitions, refuses a CSV short of 203 frames, and REPRODUCES the the lab notebook (private)'s Sep 14 12:37 values (29.79 / 29.46 / 30.34) from the CSVs on disk. Six mutations red (both windows, the loop off-by-one, the refusal, the jaw floor, a dropped joint). |

| `test_cert_gate.py` | Sep 14: certification replays ran outside `preflight.py`'s thermal gate and the scene check before them was chat discipline (the morning's one miss skipped it; the 70 °C event was an arm left holding). Pins `tools/cert_gate.py`: `temps` refuses > 52 °C or an unread servo (preflight's parsers, so both-arms captures refuse), `home` refuses a WARNING or a missing settled line, `receipt` records the verdict with the pass's gate readings and refuses anything but SEAT/MISS; `record` writes `<run>.seat` from the run's files once and never overwrites. Mutation-checked. |
| `test_run_cert.py` | Sep 14: G-R's replay-until-seat was six chat round trips per pass. Pins `tools/run_cert.sh` under stub runners: step order, refusal before motion on a hot servo, refusal before the replay on a bad scene, relieve (home + relax) after the replay BEFORE anyone is asked anything (CERT-B3-02 held torque three minutes waiting for a typed verdict), no prompt, the `record` command printed, exit 0 ran / 3 gate / 5 replay failed. Mutation-checked. |

| `test_prelaunch_check.py` | Sep 14 17:44: a pair launched on an arm 30 min unpowered (the warm-up rule lived in prose), the launcher was stopped while its rollout child kept running, and a torque-off was written under that live rollout. Pins `tools/prelaunch_check.py`: refuses on any bench process alive (by pid, never a command line), a held port, a newest `CERT-*.seat` that is not SEAT or is older than 10 min, a servo above 52 °C. Five mutations red. |
| `test_v4_bench_analysis.py` (Sep 14 additions) | A re-run label got DUPLICATE_LABEL on every row and a manual invalidation could not lift it (§7 15:26 re-ran `f3zyhuvc`). Now: rows with a MANUAL code are not duplicates; exactly one surviving row is the run that counts; two survivors are still duplicates. And `read_scores` accepts the six-column scores.csv (co-primary 2 withdrawn, §7 13:55). Three mutations red. |
| `test_relax_arm.py` (Sep 14 addition) | relax_arm ran under a live rollout at 17:45. Refuses when `lsof` shows the port held. Mutation red. |
| `test_schedule_s0only.py` | §7 16:59: eight ACT depth pairs re-declared S0-CLEAN in a derived schedule. Proves the derived file differs from `schedule.csv` in exactly those eight strata and nothing else, without printing an arm-bearing value. Two mutations red. |

| `test_bench_stop.py` | Sep 14 17:45: killing the launcher left its rollout child running. Pins `tools/bench_stop.py`: SIGTERM every bench pid, SIGKILL survivors, relax the arm only when none is left, refuse (exit 1) if one survives; never prints a command line. Three mutations red. |

| `test_v4_secondary_panel.py` — **NOT COLLECTED since Sep 14 18:20** (`collect_ignore` in `tests/conftest.py`) | It asserts over the blinded rollout recordings on disk and embeds the sealed token, so it cannot be opened before the key is; the Sep 14 bench recordings made two of its data assertions fail (a trial re-pressed more than twice). Rewrite with synthetic fixtures after scoring; then delete the `collect_ignore`. Until then the suite's headline count excludes it. |

## What is NOT covered (knowingly)

- Anything needing the arm or cameras (loop timing, fps delivery, TCC grants).
  Those live in `tools/` as interactive diagnostics — the pre-session ritual.
- `tools/home_converge.py`: no test and no importer (checked Sep 12). It is not
  on the bench path; see `tools/README.md`.
- Episode-boundary discipline (n-key, never Ctrl-C), tube placement, thermal
  warmups: procedure, not code. They live in the handoff + the project notes.
- `growbot-arm` (separate repo): out of scope here; give it its own `tests/`
  in its own session if wanted.

## Adding a test (the workflow)

1. Bug found and diagnosed → same session, write the test that would have
   caught it. Parse files/data rather than importing hardware-touching
   scripts (`home_arm.py` opens the serial bus at import).
2. Confirm it FAILS against the pre-fix state where practical, then fix.
3. One test file per bug *family*, docstring names the date and the incident.
   Tests exercise behaviour, or parse code with `ast`; they never grep prose.
   A zsh script has no `ast`, so it is executed under a stub harness
   (`test_run_pair.py`, `test_v4_start_pose.py`).
4. If the invariant is procedural (can't be asserted from disk), it goes into
   the project notes/handoff instead — don't write a vacuous test.
5. Mutation check before it counts: break the fix, watch the new test go red,
   restore, and record the exact mutation. A green test proves nothing until a
   mutation has made it red.
