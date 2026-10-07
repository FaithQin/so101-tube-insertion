#!/bin/zsh
# Score every checkpoint of the v4 runs against the A4 hold-out, one policy after another
# (they share the one MPS device -- never in parallel). One line per policy on stdout so a
# Monitor can follow it; the full log per policy is in /tmp/capstone/jobs/holdout_<name>.out
# and the numbers land in analysis/v4_holdout/<name>.json.
#
#   tools/v4_holdout_all.sh [stride=3] [workers=0] [batch=8] [policy ...]
#   tools/v4_holdout_all.sh 3 0 8 act-tube-A-v4 act-tube-B-v4 smolvla-tube-A-v4 smolvla-tube-B-v4
#   tools/v4_holdout_all.sh 6 0 4 pi05-tube-A-v4          # after the pi0.5 run completes (bf16, 3B)
set -u
PY=/opt/homebrew/Caskroom/miniforge/base/envs/lerobot/bin/python
ROOT="${0:A:h:h}"
STRIDE="${1:-3}"; WORKERS="${2:-0}"; BATCH="${3:-8}"; shift 3 2>/dev/null || true
POLICIES=("$@"); (( ${#POLICIES} )) || POLICIES=(act-tube-A-v4 act-tube-B-v4 smolvla-tube-A-v4 smolvla-tube-B-v4)
mkdir -p /tmp/capstone/jobs
for name in "${POLICIES[@]}"; do
  log=/tmp/capstone/jobs/holdout_${name}.out
  echo "[holdout] $name: start (stride $STRIDE, workers $WORKERS, batch $BATCH)  $(date '+%H:%M')"
  t0=$(date +%s)
  # --append: a re-run scores only the checkpoints not yet in analysis/v4_holdout/<name>.json
  # (same stride/seed/split only -- it refuses to mix), so the pi0.5 curve can be filled in
  # as its checkpoints land instead of waiting for the job to finish.
  if $PY "$ROOT/tools/v4_holdout_loss.py" "faithqin/$name" --stride "$STRIDE" --workers "$WORKERS" --batch "$BATCH" --append > "$log" 2>&1; then
    echo "[holdout] $name: DONE in $(( ($(date +%s)-t0)/60 ))m -- $(grep -E '^best hold-out' "$log")"
  else
    echo "[holdout] $name: FAILED (exit $?) after $(( ($(date +%s)-t0)/60 ))m -- $(grep -vE 'objc\[|AVF' "$log" | grep -E 'Error|error|GATE|Traceback' | tail -2 | tr '\n' ' ')"
  fi
done
echo "[holdout] all done  $(date '+%H:%M')"
