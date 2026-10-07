#!/bin/zsh
# Usage: run_probe_episode.sh on|off  — one ACT-A-v2 probe episode from the
# MAIN checkout with the grasp lock armed or not. Logs to tools/probe_logs/
# for banner/loop-rate/engagement verification. Promoted from the Aug 27
# session (tests pending).
MODE="$1"
export CAPSTONE_HOME_POSE=v2   # this is a v2 probe runner; the wrapper must not guess (Sep 4 audit)
HERE="${0:A:h}"
LOG="$HERE/probe_logs/probe_${MODE}.log"
cd "$HERE/.." || exit 1
if [ "$MODE" = "on" ]; then
  export CAPSTONE_GRASP_LOCK=1
  REPO="faithqin/rollout_probe_lockON_a25_v2"
else
  unset CAPSTONE_GRASP_LOCK
  REPO="faithqin/rollout_probe_lockOFF_a25_v2"
fi
/opt/homebrew/Caskroom/miniforge/base/envs/lerobot/bin/python tools/rollout_30hz_stale_ok.py \
  --strategy.type=episodic --fps=20 \
  --policy.path=faithqin/act-tube-A-v2 --policy.n_action_steps=25 \
  --robot.type=so101_follower --robot.port=/dev/tty.usbmodem5C4C1245641 --robot.id=follower \
  --robot.cameras='{front: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30}, wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \
  --dataset.fps=20 --dataset.repo_id="$REPO" \
  --dataset.num_episodes=1 --dataset.episode_time_s=60 --dataset.reset_time_s=5 \
  --dataset.push_to_hub=false > "$LOG" 2>&1
echo "PROBE_EPISODE_DONE mode=$MODE rc=$?" >> "$LOG"
