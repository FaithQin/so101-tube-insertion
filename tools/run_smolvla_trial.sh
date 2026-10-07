#!/bin/zsh
# One SmolVLA probe episode, fully gated. Sibling of run_pi05_trial.sh.
#
#   run_smolvla_trial.sh <side A|B> <label> [n_action_steps]
#   CAPSTONE_CHECKPOINT=005000 run_smolvla_trial.sh B SV4-B-01 50
#
# WHICH WEIGHTS (Sep 6): the side table defaults to the v4 policies (CAPSTONE_POLICY_GEN=v3
# selects the Sep 3-4 table). For a v4 side CAPSTONE_CHECKPOINT is REQUIRED -- the hold-out
# verdict is "bench checkpoints/005000"; the repo's top-level weights are the overfit 20k
# checkpoint. `=top` benches them anyway, deliberately. Unset -> the resolver below aborts.
#
# A SEPARATE script on purpose, like the pi0.5 runner: run_scored_trial.sh
# drives the ACT blocks and must not change the night before a session. Same
# gates, same abort-not-warn discipline.
#
# ---------------------------------------------------------------------------
# Four things SmolVLA needs that ACT does not. Three of them fail SILENTLY.
# ---------------------------------------------------------------------------
#
#  1. THE PER-SIDE STATE PIN. smolvla-tube-B-v3 declares
#     observation.state [6] while carrying 18-dim normalizer statistics
#     (measured Sep 3 2026). The live routing in rollout/context.py:400-407
#     reads the DECLARED width, so B routes narrow by inference and a 6-dim
#     state meets an 18-dim normalizer: RuntimeError on the FIRST inference
#     tick, mid-episode, on a gated bench day. Its A-side twin declares the
#     same [6] and genuinely IS 6-dim. Nothing about the declaration, the
#     padding width, or the policy family separates them -- only the pin.
#     A is pinned to 0, B to 1, and tools/check_state_width.py verifies the
#     pin against the checkpoint's own normalizer before anything launches.
#     (This is the one failure that is loud. The other three are not.)
#
#  2. THE LANGUAGE PROMPT. SmolVLA is language-conditioned. lerobot 0.6.1
#     defaults task to "" (rollout/configs.py:244); neither propagation branch
#     fires (configs.py:365-370); context.py:562 hands the engine task="".
#     The tokenizer tokenizes the empty string without complaint. This runner
#     therefore READS the prompt from the training dataset's own
#     meta/tasks.parquet and aborts if it comes back empty -- it is never
#     typed here, so it cannot go stale.
#
#  3. --rename_map. The rig publishes observation.images.front / .wrist;
#     both checkpoints declare camera1 / camera2 / camera3 / empty_camera_0.
#     RolloutConfig.rename_map DEFAULTS TO AN EMPTY DICT
#     (rollout/configs.py:259) and context.py:537-546 passes it as a
#     preprocessor OVERRIDE -- so omitting the flag does not fall back to the
#     map saved with the checkpoint, it REPLACES that map with {}. SmolVLA
#     then masks the absent towers instead of raising and runs blind at full
#     speed. The map below is the one in both checkpoints' train_config.json.
#
#  4. THE LAUNCHER'S TOKENIZER FIX. tools/rollout_30hz_stale_ok.py:293-301
#     injects tokenizer_processor.tokenizer_name = vlm_model_name, because a
#     0.6.2-saved SmolVLA checkpoint records its tokenizer as the relative
#     folder "tokenizer" and local 0.6.1 cannot resolve it. Launching any
#     other way raises at processor construction. Do not "simplify" this to
#     plain lerobot-rollout.
#
# ---------------------------------------------------------------------------
# Inference engine: SYNC, not RTC (reversed Sep 4 2026, on the bench).
# ---------------------------------------------------------------------------
# --inference.type=rtc was the plan. Under it all three SmolVLA trials
# (SV-A-01, SV-B-01, SV-A-02) commanded a fully-extended, gripper-open pose
# from tick 0 and hovered for 44 s. Reproduced offline with the real
# ActionQueue: RTC's _replace_actions_queue discards the first `real_delay`
# actions of EVERY chunk -- including the first, when nothing was executed
# during inference. On MPS the first inference took 15 ticks (log:
# "indexes_diff=0, real_delay=15"), so the first command was chunk[15], a pose
# 0.75 s into the reach, executed from rest; every re-plan then discarded ~7
# more. The sync engine serves chunk[0] and is what ACT and pi0.5 run on.
# Cost: one ~0.25-0.45 s stall per chunk (better than pi0.5's 0.77 s).
# Under sync, n_action_steps IS live: the policy executes that many steps of
# each chunk open-loop before re-planning. 50 = the trained chunk size.
# Pinned by tests/test_smolvla_trial_runner.py.
#
# Ledger: writes scored_logs/trials_smolvla.csv, NOT trials.csv --
# tools/figures_data.py:83 classifies any non-`act-tube-B` policy as ACT-A, so
# SmolVLA rows in the ACT ledger would publish as ACT-A points in fig4. Merge
# deliberately once that classifier distinguishes architectures.
#
# Run this from a TCC-granted Terminal (cameras). Serial is direct.
set -u

RAW_SIDE="${1:-}"; LABEL="${2:-}"; NSTEPS="${3:-50}"
SIDE="${RAW_SIDE:u}"

# --- usage gates: BEFORE the hub, the env, the gates, and the arm -----------
# Deliberately silent (no `say`): these are typos, the operator is looking at
# the terminal, and the suite executes this path while the arm may be busy in
# another window.
case "$SIDE" in
  A|B) ;;
  *) echo "ABORT: side must be A or B (got '${RAW_SIDE:-<none>}')"
     echo "usage: run_smolvla_trial.sh <A|B> <label> [n_action_steps]"; exit 2 ;;
esac
case "$LABEL" in
  ''|*[!A-Za-z0-9_-]*)
     echo "ABORT: label must be non-empty and [A-Za-z0-9_-] only (got '${LABEL:-<none>}')"
     echo "       it becomes the rollout dataset name faithqin/rollout_<label>"; exit 2 ;;
esac
case "$NSTEPS" in
  ''|*[!0-9]*) echo "ABORT: n_action_steps must be a positive integer (got '$NSTEPS')"; exit 2 ;;
esac
if [ "$NSTEPS" -le 0 ]; then echo "ABORT: n_action_steps must be > 0 (got '$NSTEPS')"; exit 2; fi

HERE="${0:A:h}"
PROJ="${HERE:h}"
B=/opt/homebrew/Caskroom/miniforge/base/envs/lerobot/bin
cd "$PROJ" || exit 1

# Aug 27 INFRA #3: a probe was invalidated by a stale worktree copy of the
# wrapper. Project state lives in the main checkout.
if [ -z "${CAPSTONE_ALLOW_WORKTREE:-}" ]; then
  $B/python -c "import sys; sys.path.insert(0,'tools'); import preflight, sys as s; s.exit(0 if preflight.is_main_checkout('$PROJ') else 1)" || {
    say "abort. worktree."; echo "ABORT: running from a git worktree — use the main checkout"; exit 2; }
fi

# Which generation's support to home to and gate on: from the side table (v4 by default,
# CAPSTONE_POLICY_GEN=v3 for the Sep 3-4 policies) -- never a constant (Sep 6).
DATASET_TAG=$($B/python tools/smolvla_dryrun.py --print gen --side "$SIDE") || {
  say "abort. metadata."; echo "ABORT: could not resolve the policy generation"; exit 4; }

# --- start pose (Sep 6 2026) ---------------------------------------------------------------
# Frame 0 of the policy's generation by default. CAPSTONE_START_POSE=<gen>-carry selects the
# PRE-GRASPED start (protocol stratum S2, a MODIFIED TASK: the operator places the tube in the
# open jaws and home_arm closes on it before the policy runs): the gate loop homes to the
# carry pose with --close-on-tube, preflight gates that pose and the RACK (the tube is in the
# jaws, its placement mark says nothing), the wrapper re-homes there, and the receipt says so.
# The scene is still photographed from frame 0 first -- arm clear, rack visible.
START_POSE="${CAPSTONE_START_POSE:-$DATASET_TAG}"
HOME_FLAGS=()
case "$START_POSE" in
  "$DATASET_TAG") ;;
  "${DATASET_TAG}-carry")
    HOME_FLAGS=(--close-on-tube)
    echo "  start pose: $START_POSE -- PRE-GRASPED, a MODIFIED TASK (the jaws close on the tube before the policy runs)" ;;
  *) say "abort. start pose."
     echo "ABORT: CAPSTONE_START_POSE='$START_POSE' is not a start for a $DATASET_TAG policy (unset = frame 0, or ${DATASET_TAG}-carry)"
     exit 2 ;;
esac

STAMP=$(date +%Y%m%d_%H%M%S)
RUN="$HERE/scored_logs/${LABEL}_${STAMP}"
mkdir -p "$HERE/scored_logs"

# --- resolve the launch parameters from the artifacts, never from this file --
# tools/smolvla_dryrun.py is the single source of truth, so the rehearsal and
# the bench cannot disagree about the prompt, the pin, or the checkpoint. That
# disagreement is exactly how pi0.5 came within a day of running with an empty
# language prompt while its dry-run passed.
REPO_ID=$($B/python tools/smolvla_dryrun.py --print repo     --side "$SIDE") || {
  say "abort. metadata."; echo "ABORT: could not resolve the policy repo id"; exit 4; }
SNAP=$($B/python    tools/smolvla_dryrun.py --print snapshot --side "$SIDE") || {
  say "abort. checkpoint."; echo "ABORT: could not resolve the checkpoint snapshot"; exit 4; }
FORCE=$($B/python   tools/smolvla_dryrun.py --print force    --side "$SIDE") || {
  say "abort. metadata."; echo "ABORT: could not resolve the state-width pin"; exit 4; }
TASK=$($B/python    tools/smolvla_dryrun.py --print task     --side "$SIDE") || {
  say "abort. prompt."; echo "ABORT: could not read the trained prompt from the dataset"; exit 5; }

if [ -z "$TASK" ]; then
  say "abort. empty prompt."
  echo "ABORT: the trained prompt resolved EMPTY — SmolVLA would run with no"
  echo "       language conditioning and the trial would be uninterpretable"
  exit 5
fi

echo "=== $LABEL  side=$SIDE  policy=$REPO_ID  n=$NSTEPS  inference=sync  (gated)"
echo "  checkpoint: $SNAP"
echo "  prompt:     $TASK"
echo "  state pin:  CAPSTONE_FORCE_WIDE_STATE=$FORCE"

# --- gate 1: state width ----------------------------------------------------
# Checks the SAME directory the rollout loads below. Gating one artifact and
# launching another proves nothing.
$B/python tools/check_state_width.py "$SNAP" --force "$FORCE" | sed 's/^/  /'
if [ "${pipestatus[1]}" -ne 0 ]; then
  say "abort. state width."
  echo "ABORT: state-width gate failed — see the line above for the pin to use"
  exit 4
fi

# --- resolve-only escape hatch ----------------------------------------------
# Everything above this line is pure resolution: no serial, no cameras, no
# motion. Everything below it drives the arm. CAPSTONE_RESOLVE_ONLY=1 stops
# here, which does two jobs:
#
#   * gives Faith a ~10 s pre-flight that proves the exact launch parameters
#     for a side without consuming a gated trial;
#   * lets the regression suite EXECUTE the capture block above — the four
#     `--print` calls, the `${RAW_SIDE:u}` normalisation, the empty-TASK guard
#     and the state-width gate — which is otherwise untestable, because
#     reaching it and continuing means homing the arm.
#
# Added Sep 3 2026 after exactly that accident: an executing test was written
# for this hatch BEFORE the hatch existed, so the runner ran straight into
# `home_arm.py` from a `pytest` invocation and homed the arm three times. For a
# safety gate, the TDD red phase IS the hazard — the guard lands first, and
# tests/test_smolvla_trial_runner.py SKIPS every executing test when this
# marker is absent from the script.
if [ -n "${CAPSTONE_RESOLVE_ONLY:-}" ]; then
  echo "RESOLVE-ONLY: parameters above are verified and the state-width gate passed."
  echo "              NO gates were run, NO trial was executed, NO ledger row written."
  exit 0
fi

# --- gates 2-4: identical to the ACT and pi0.5 runners ----------------------
# Home BEFORE looking at the scene: an arm parked over the tube occludes the
# cap, and scene_check then reports a placement failure that is an occlusion
# failure (observed Aug 30).
$B/python tools/home_arm.py --pose "$DATASET_TAG" > "${RUN}.home" 2>&1
$B/python tools/scene_check.py  > "${RUN}.scene" 2>&1

GATED=0; REASON=""
for attempt in 1 2 3; do
  $B/python tools/home_arm.py --pose "$START_POSE" "${HOME_FLAGS[@]}" > "${RUN}.home" 2>&1
  $B/python tools/ping_motors.py follower > "${RUN}.ping" 2>&1
  OUT=$($B/python tools/preflight.py "${RUN}.scene" "${RUN}.home" "${RUN}.ping" "$START_POSE"); RC=$?
  echo "  $OUT"
  if [ "$RC" -eq 0 ]; then GATED=1; break; fi
  REASON="$OUT"
  case "$OUT" in
    *placement*|*thermal*) echo "  (human-only gate — not retrying)"; break ;;
  esac
  case "$OUT" in
    *gripper*)
      if [ "$START_POSE" = "$DATASET_TAG" ]; then
        echo "  REPAIRED: gripper outside training support — reseating the jaw"
        $B/python tools/reseat_gripper.py 2>&1 | sed 's/^/    /'
      else
        echo "  (carry start: the jaw is holding the tube, not reseated on nothing — re-place the tube when prompted)"
      fi ;;
    *elbow_flex*)
      echo "  REPAIRED: elbow outside training support — re-homing" ;;
  esac
  echo "  (attempt $attempt did not gate; retrying)"
done

if [ "$GATED" != "1" ]; then
  SPOKEN="abort. unknown gate."
  case "$REASON" in
    *elbow_flex*)  SPOKEN="abort. elbow out of range." ;;
    *gripper*)     SPOKEN="abort. gripper out of range." ;;
    *placement*)   SPOKEN="abort. tube placement." ;;
    *thermal*)     SPOKEN="abort. elbow too hot." ;;
    *evaluate*)    SPOKEN="abort. a check could not be read." ;;
  esac
  say "$SPOKEN"
  echo "TRIAL ABORTED — nothing launched, no trial consumed."
  exit 3
fi

# --- launch -----------------------------------------------------------------
unset CAPSTONE_GRASP_LOCK              # ACT action-queue patch; RTC has no queue to lock
export CAPSTONE_FORCE_WIDE_STATE="$FORCE"
export CAPSTONE_HOME_POSE="$START_POSE"  # else the wrapper re-homes to its v2 default
export CAPSTONE_TELEMETRY_CSV="${RUN}_telemetry.csv"  # all 18 servo channels per recorded frame, whatever the policy sees (Sep 6)
REPO="faithqin/rollout_${LABEL}"
say "recording"
$B/python tools/rollout_30hz_stale_ok.py \
  --strategy.type=episodic --fps=20 \
  --inference.type=sync \
  --policy.path="$SNAP" --policy.n_action_steps="$NSTEPS" \
  --rename_map='{observation.images.front: observation.images.camera1, observation.images.wrist: observation.images.camera2}' \
  --robot.type=so101_follower --robot.port=/dev/tty.usbmodem5C4C1245641 --robot.id=follower \
  --robot.cameras='{front: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30}, wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \
  --dataset.fps=20 --dataset.repo_id="$REPO" \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes=1 --dataset.episode_time_s=45 --dataset.reset_time_s=5 \
  --dataset.push_to_hub=false > "${RUN}.log" 2>&1

# --- verify + score ---------------------------------------------------------
# The routing line is the receipt that the pin took effect; the tokenizer line
# is the receipt that the launcher's fix fired. Both are read from the log
# rather than assumed.
{
  echo "start_pose=$START_POSE"
  # The wrapper module is loaded more than once per rollout (measured Sep 6: three [loop] lines
  # in one trial log), and the extra loads register atexit handlers that fire having recorded no
  # frames, printing "loop_hz=nan frames=0". `[0-9.]*` matched the EMPTY string after that first
  # nan, so the receipt carried "loop_hz=" on a trial whose real rate was 19.62. Require digits,
  # and take the LAST line -- the one written by the process that actually recorded the episode.
  echo "loop_hz=$(grep -oE 'loop_hz=[0-9]+\.[0-9]+' "${RUN}.log" | tail -1 | cut -d= -f2)"
  echo "side=$SIDE"
  echo "policy=$REPO_ID"
  echo "n_action_steps=$NSTEPS   (live under sync: steps executed open-loop per chunk)"
  echo "inference=sync  (RTC disabled Sep 4: ActionQueue discards real_delay leading actions on the first chunk)"
  echo "force_pin=$FORCE"
  echo "prompt=$TASK"
  echo "state_routing=$(grep -c 'CAPSTONE_STATE_ROUTING' "${RUN}.log")"
  echo "routing_line=$(grep -o 'CAPSTONE_STATE_ROUTING.*' "${RUN}.log" | head -1)"
  echo "tokenizer_line=$(grep -o 'tokenizer override.*' "${RUN}.log" | head -1)"
  echo "slow_ticks=$(grep -c 'running slower' "${RUN}.log")"
} | tee "${RUN}.verify"

DS=$(ls -dt ~/.cache/huggingface/lerobot/faithqin/${REPO##*/}_* 2>/dev/null | head -1)
V="$DS/videos/observation.images.front/chunk-000/file-000.mp4"
$B/python -c "
import sys; sys.path.insert(0,'tools')
import score_episode as s, csv, os
r = s.score_video('$V')
print(s.render(r))
row = ['$LABEL','$SIDE','$REPO_ID','$NSTEPS','sync','$FORCE','$STAMP',
       'INFRA' if r['infra_start_seated'] else r['final'],
       '%.1f' % r['duration_s'], len(r['seat_runs'])]
LEDGER = '$HERE/scored_logs/trials_smolvla.csv'
new = not os.path.exists(LEDGER)
with open(LEDGER,'a',newline='') as f:
    w=csv.writer(f)
    if new: w.writerow(['label','side','policy','n','inference','force','stamp','verdict','dur_s','seat_runs'])
    w.writerow(row)
print('VERDICT', row[7])
" | tee "${RUN}.score"

if grep -q "VERDICT SEAT" "${RUN}.score"; then say "seat"; elif grep -q "VERDICT INFRA" "${RUN}.score"; then say "infra"; else say "miss"; fi
