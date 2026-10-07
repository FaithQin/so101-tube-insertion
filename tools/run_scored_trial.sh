#!/bin/zsh
# One SCORED trial, fully gated. Every check aborts; none of them warn.
#
#   run_scored_trial.sh <policy-repo> <n_action_steps> <on|off> <label>
#   run_scored_trial.sh faithqin/act-tube-A-v3 25 off P001-A
#
# Differences from run_probe_episode.sh, all of them deliberate:
#   * 45 s episodes, not 60 — a v2 scored trial holds 875 frames (~45 s) and the
#     locked protocol says "Max 45 s". Success is defined at rollout end, so
#     episode length is part of the definition and must match the baseline.
#   * preflight.py gates placement, home pose vs training support, and elbow
#     temperature BEFORE launching, and aborts on any of them.
#   * per-trial log and per-trial CSV row; nothing writes to a fixed filename
#     that a later trial would overwrite.
#   * scored from the episode's own video via score_episode.py, which also
#     flags an invalid start state as INFRA rather than scoring it.
#
# Run this from a TCC-granted Terminal (cameras). Serial is direct.
set -u
POLICY="$1"; NSTEPS="$2"; MODE="$3"; LABEL="$4"
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

# Which training distribution does THIS policy come from? v2 and v3 differ
# materially (elbow frame-0 75.47-84.44 vs 78.11-90.42; gripper 1.14-1.64 vs
# 0.79-1.57), so both the home pose and the support gate must follow the policy.
# Gating a v3 policy on v2 bounds rejects valid starts and admits invalid ones —
# caught Aug 29 when an elbow of 85.30, squarely inside v3, was refused.
case "$POLICY" in
  *v4*) DATASET_TAG=v4 ;;
  *v3*) DATASET_TAG=v3 ;;
  *v2*) DATASET_TAG=v2 ;;
  *)    say "abort. unknown dataset tag."
        echo "ABORT: cannot tell which dataset '$POLICY' was trained on (no v2/v3 in the name)."
        echo "       A v4 policy needs TRAIN_HOME_V4 in rollout_30hz_stale_ok.py, a v4 window in home_gate.FRAME0,"
        echo "       and a *v4*) branch here — never a silent default (Sep 3: every v3 trial re-homed to v2)."
        exit 2 ;;
esac
echo "  dataset support: $DATASET_TAG"

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

echo "=== $LABEL  policy=$POLICY  n=$NSTEPS  lock=$MODE"

# --- gates -----------------------------------------------------------------
# Home BEFORE looking at the scene. An arm parked over the tube (e.g. left there
# by the previous session) occludes the cap, and scene_check then reports it
# missing — which reads as a placement failure when it is an occlusion failure.
# Observed Aug 30. Homing first also means the scene is judged from the pose the
# policy will actually start in.
$B/python tools/home_arm.py --pose "$DATASET_TAG" > "${RUN}.home" 2>&1
$B/python tools/scene_check.py  > "${RUN}.scene" 2>&1

# Home up to 3 times before giving up. A single home is not reliable: the elbow
# under-reaches when hot and over-reaches from some starting poses (measured
# 74.1-85.3 across Aug 29), and the gripper creeps open after ACT-B episodes.
# Re-homing costs ~5 s and often lands inside support on the second try; the
# scene does not change between attempts, so it is captured once above.
GATED=0
REASON=""
for attempt in 1 2 3; do
  $B/python tools/home_arm.py --pose "$START_POSE" "${HOME_FLAGS[@]}" > "${RUN}.home" 2>&1
  $B/python tools/ping_motors.py follower > "${RUN}.ping" 2>&1
  OUT=$($B/python tools/preflight.py "${RUN}.scene" "${RUN}.home" "${RUN}.ping" "$START_POSE"); RC=$?
  echo "  $OUT"
  if [ "$RC" -eq 0 ]; then GATED=1; break; fi
  REASON="$OUT"

  # HUMAN_ONLY gates: no amount of re-homing moves a tube or cools a servo.
  # Short-circuit rather than burning ~15 s to re-announce a fact the operator
  # has already heard.
  case "$OUT" in
    *placement*|*thermal*) echo "  (human-only gate — not retrying)"; break ;;
  esac

  # SOFTWARE-REPAIRABLE gates: fix, then let the loop re-home and re-gate.
  # A gate that only refuses wastes the operator's time; the condition has to
  # get fixed somewhere, and here is where it is cheapest.
  case "$OUT" in
    *gripper*)
      if [ "$START_POSE" = "$DATASET_TAG" ]; then
        echo "  REPAIRED: gripper outside training support — reseating the jaw"
        $B/python tools/reseat_gripper.py 2>&1 | sed 's/^/    /'
      else
        echo "  (carry start: the jaw is holding the tube, not reseated on nothing — re-place the tube when prompted)"
      fi
      ;;
    *elbow_flex*)
      echo "  REPAIRED: elbow outside training support — re-homing (it lands differently from different starting poses)"
      ;;
  esac
  echo "  (attempt $attempt did not gate; retrying)"
done

if [ "$GATED" != "1" ]; then
  # Speak the CAUSE, not the category. "abort, preflight" tells the operator
  # nothing and the voice mangles the word; "abort, elbow out of range" is
  # immediately actionable from across the room.
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
if [ "$MODE" = "on" ]; then export CAPSTONE_GRASP_LOCK=1; else unset CAPSTONE_GRASP_LOCK; fi
export CAPSTONE_HOME_POSE="$START_POSE"  # else the wrapper re-homes to its v2 default
export CAPSTONE_TELEMETRY_CSV="${RUN}_telemetry.csv"  # all 18 servo channels per recorded frame, whatever the policy sees (Sep 6)
# lerobot refuses a rollout dataset whose name does not start with "rollout_"
# (rollout/context.py:446). Caught Aug 29 on Block 2 trial 1 — it aborted before
# the arm moved, so nothing was consumed, but the trial was lost to a restart.
REPO="faithqin/rollout_${LABEL}"
say "recording"
$B/python tools/rollout_30hz_stale_ok.py \
  --strategy.type=episodic --fps=20 \
  --policy.path="$POLICY" --policy.n_action_steps="$NSTEPS" \
  --robot.type=so101_follower --robot.port=/dev/tty.usbmodem5C4C1245641 --robot.id=follower \
  --robot.cameras='{front: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30}, wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \
  --dataset.fps=20 --dataset.repo_id="$REPO" \
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
  echo "lock_banner=$(grep -c 'GRASP-PHASE CHUNK LOCKING ACTIVE' "${RUN}.log")"
  echo "onsets=$(grep -c 'closing onset detected' "${RUN}.log")"
  echo "extends=$(grep -c 'LOCKED: extended' "${RUN}.log")"
  echo "slow_ticks=$(grep -c 'running slower' "${RUN}.log")"
} | tee "${RUN}.verify"

DS=$(ls -dt ~/.cache/huggingface/lerobot/faithqin/${REPO##*/}_* 2>/dev/null | head -1)
V="$DS/videos/observation.images.front/chunk-000/file-000.mp4"
$B/python -c "
import sys; sys.path.insert(0,'tools')
import score_episode as s, csv, os
r = s.score_video('$V')
print(s.render(r))
row = ['$LABEL','$POLICY','$NSTEPS','$MODE','$STAMP',
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
