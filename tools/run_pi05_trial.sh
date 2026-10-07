#!/bin/zsh
# One pi0.5 probe episode, fully gated. Sibling of run_scored_trial.sh.
#
#   run_pi05_trial.sh <side A|B> <label> [n_action_steps]
#   CAPSTONE_CHECKPOINT=top run_pi05_trial.sh A P05-A-01 50
#
# WHICH POLICY (Sep 6): the side and everything that follows from it -- repo, sanitized
# checkpoint, state-width pin, prompt, home pose -- come from tools/pi05_sides.py, never from
# this file. v4 by default (CAPSTONE_POLICY_GEN=v3 = the Sep 3 policy). For a v4 side
# CAPSTONE_CHECKPOINT is REQUIRED (=top benches the final weights; the hold-out verdict for
# pi05-tube-A-v4 is top/20k; pi05-tube-B-v4 gets its own verdict once scored).
#
# Deliberately a SEPARATE script rather than a flag on run_scored_trial.sh:
# that runner drives the ACT blocks and must not change the night before a
# session. Same gates, same scoring, same abort-not-warn discipline.
#
# Two things pi0.5 needs that ACT does not, both SILENT failures if missed:
#
#  1. --rename_map. The rig publishes observation.images.front / .wrist; pi0.5
#     declares base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb /
#     empty_camera_0. Without the map NONE of its image keys are present, and
#     _preprocess_images pads every missing tower with -1 and masks it to zero
#     rather than raising. The policy would run BLIND, at full speed.
#
#  2. A sanitized local checkpoint. The Hub config carries six lerobot fields
#     0.6.1 cannot parse, so --policy.path=<repo_id> dies in draccus.
#
# The state-width pin is PER SIDE: pi0.5 declares state width 32 (max_state_dim padding)
# for both A and B, so the routing's "declared == max -> widen" rule cannot separate them.
# A trained on 6-dim (load/current landing in dims 6-17 would be a silent distribution
# shift); B carries 18-dim statistics (a narrow state meets an 18-dim normalizer: crash).
set -u
RAW_SIDE="${1:-}"; LABEL="${2:-}"; NSTEPS="${3:-50}"
SIDE="${RAW_SIDE:u}"
case "$SIDE" in
  A|B) ;;
  *) echo "ABORT: side must be A or B (got '${RAW_SIDE:-<none>}')"
     echo "usage: run_pi05_trial.sh <A|B> <label> [n_action_steps]"; exit 2 ;;
esac
case "$LABEL" in
  ''|*[!A-Za-z0-9_-]*)
     echo "ABORT: label must be non-empty and [A-Za-z0-9_-] only (got '${LABEL:-<none>}')"; exit 2 ;;
esac
case "$NSTEPS" in
  ''|*[!0-9]*) echo "ABORT: n_action_steps must be a positive integer (got '$NSTEPS')"; exit 2 ;;
esac
HERE="${0:A:h}"
PROJ="${HERE:h}"
B=/opt/homebrew/Caskroom/miniforge/base/envs/lerobot/bin
cd "$PROJ" || exit 1

if [ -z "${CAPSTONE_ALLOW_WORKTREE:-}" ]; then
  $B/python -c "import sys; sys.path.insert(0,'tools'); import preflight, sys as s; s.exit(0 if preflight.is_main_checkout('$PROJ') else 1)" || {
    say "abort. worktree."; echo "ABORT: running from a git worktree — use the main checkout"; exit 2; }
fi

STAMP=$(date +%Y%m%d_%H%M%S)
RUN="$HERE/scored_logs/${LABEL}_${STAMP}"
mkdir -p "$HERE/scored_logs"

# --- resolve the launch parameters from the side table, never from this file --
REPO_ID=$($B/python tools/pi05_sides.py --print repo       --side "$SIDE") || {
  say "abort. metadata."; echo "ABORT: could not resolve the policy repo id"; exit 4; }
DATASET_TAG=$($B/python tools/pi05_sides.py --print gen    --side "$SIDE") || {
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
FORCE=$($B/python tools/pi05_sides.py --print force        --side "$SIDE") || {
  say "abort. metadata."; echo "ABORT: could not resolve the state-width pin"; exit 4; }
CKPT=$($B/python tools/pi05_sides.py --print checkpoint    --side "$SIDE") || {
  say "abort. checkpoint."; echo "ABORT: no checkpoint chosen (CAPSTONE_CHECKPOINT)"; exit 4; }
TASK=$($B/python tools/pi05_sides.py --print task          --side "$SIDE") || {
  say "abort. prompt."; echo "ABORT: could not read the trained prompt from the dataset"; exit 5; }
if [ -z "$TASK" ]; then
  say "abort. empty prompt."
  echo "ABORT: the trained prompt resolved EMPTY -- pi0.5 would run with no language conditioning"
  exit 5
fi

echo "=== $LABEL  side=$SIDE  policy=$REPO_ID  checkpoint=$CKPT  n=$NSTEPS  (gated)"
echo "  dataset support: $DATASET_TAG    state pin: CAPSTONE_FORCE_WIDE_STATE=$FORCE"
echo "  prompt: $TASK"

# --- sanitized checkpoint (the chosen one, 0.6.1-parsable) ------------------
POLICY_DIR=$($B/python tools/pi05_sides.py --print snapshot --side "$SIDE") || {
  say "abort. checkpoint."; echo "ABORT: could not sanitize the checkpoint"; exit 4; }
echo "  policy path: $POLICY_DIR"

# --- gate 0: state width against the SAME directory the rollout loads --------
$B/python tools/check_state_width.py "$POLICY_DIR" --force "$FORCE" | sed 's/^/  /'
if [ "${pipestatus[1]}" -ne 0 ]; then
  say "abort. state width."
  echo "ABORT: state-width gate failed -- see the line above for the pin to use"
  exit 4
fi

# --- gates (identical to the ACT runner) -----------------------------------
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

# --- launch ----------------------------------------------------------------
unset CAPSTONE_GRASP_LOCK          # ACT action-queue patch; undefined for pi0.5
export CAPSTONE_FORCE_WIDE_STATE="$FORCE"  # per side, from pi05_sides.py: A narrow (0), B wide (1)
export CAPSTONE_HOME_POSE="$START_POSE"  # else the wrapper re-homes to its v2 default
export CAPSTONE_TELEMETRY_CSV="${RUN}_telemetry.csv"  # all 18 servo channels per recorded frame, whatever the policy sees (Sep 6)
REPO="faithqin/rollout_${LABEL}"
say "recording"
$B/python tools/rollout_30hz_stale_ok.py \
  --strategy.type=episodic --fps=20 \
  --policy.path="$POLICY_DIR" --policy.n_action_steps="$NSTEPS" \
  --rename_map='{observation.images.front: observation.images.base_0_rgb, observation.images.wrist: observation.images.left_wrist_0_rgb}' \
  --robot.type=so101_follower --robot.port=/dev/tty.usbmodem5C4C1245641 --robot.id=follower \
  --robot.cameras='{front: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30}, wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \
  --dataset.fps=20 --dataset.repo_id="$REPO" \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes=1 --dataset.episode_time_s=45 --dataset.reset_time_s=5 \
  --dataset.push_to_hub=false > "${RUN}.log" 2>&1

# --- verify + score --------------------------------------------------------
{
  echo "start_pose=$START_POSE"
  # The wrapper module is loaded more than once per rollout (measured Sep 6: three [loop] lines
  # in one trial log), and the extra loads register atexit handlers that fire having recorded no
  # frames, printing "loop_hz=nan frames=0". `[0-9.]*` matched the EMPTY string after that first
  # nan, so the receipt carried "loop_hz=" on a trial whose real rate was 19.62. Require digits,
  # and take the LAST line -- the one written by the process that actually recorded the episode.
  echo "loop_hz=$(grep -oE 'loop_hz=[0-9]+\.[0-9]+' "${RUN}.log" | tail -1 | cut -d= -f2)"
  echo "state_routing=$(grep -c 'CAPSTONE_STATE_ROUTING' "${RUN}.log")"
  echo "routing_line=$(grep -o 'CAPSTONE_STATE_ROUTING.*' "${RUN}.log" | head -1)"
  echo "slow_ticks=$(grep -c 'running slower' "${RUN}.log")"
} | tee "${RUN}.verify"

DS=$(ls -dt ~/.cache/huggingface/lerobot/faithqin/${REPO##*/}_* 2>/dev/null | head -1)
V="$DS/videos/observation.images.front/chunk-000/file-000.mp4"
$B/python -c "
import sys; sys.path.insert(0,'tools')
import score_episode as s, csv, os
r = s.score_video('$V')
print(s.render(r))
row = ['$LABEL','$REPO_ID','$NSTEPS','off','$STAMP',
       'INFRA' if r['infra_start_seated'] else r['final'],
       '%.1f' % r['duration_s'], len(r['seat_runs'])]
new = not os.path.exists('$HERE/scored_logs/trials.csv')
with open('$HERE/scored_logs/trials.csv','a',newline='') as f:
    w=csv.writer(f)
    if new: w.writerow(['label','policy','n','lock','stamp','verdict','dur_s','seat_runs'])
    w.writerow(row)
print('VERDICT', row[5])
" | tee "${RUN}.score"

if grep -q "VERDICT SEAT" "${RUN}.score"; then say "seat"; elif grep -q "VERDICT INFRA" "${RUN}.score"; then say "infra"; else say "miss"; fi
