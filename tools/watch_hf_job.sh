#!/bin/zsh
# Watch one HF Jobs training run to completion. Emits ONE LINE per event on stdout
# (for Monitor): status changes, error signatures, stalls, and the final verdict.
#
#   tools/watch_hf_job.sh <job_id> <name> [stall_minutes=18]
#
# Faith's rule: watch every run
# to the end, detect stalls, report a failure the moment it happens. Exits when
# the job reaches a terminal state, so a Monitor on it ends by itself.
set -u
JOB="$1"; NAME="$2"; STALL_MIN="${3:-18}"
HF=/opt/homebrew/Caskroom/miniforge/base/envs/lerobot/bin/hf
LOG="/tmp/capstone/jobs/${NAME}.log"; mkdir -p /tmp/capstone/jobs
last_status=""; last_lines=0; last_progress=$(date +%s); start=$(date +%s)
while true; do
  # `hf jobs inspect` prints a TSV row whose status column reads {'stage': 'RUNNING', ...};
  # pull the stage by regex. (zsh: `status` is a READ-ONLY special variable — use `st`.)
  st=$($HF jobs inspect "$JOB" 2>/dev/null | grep -oE "'stage': '[A-Za-z_]+'" | head -1 | sed -E "s/.*'([A-Za-z_]+)'$/\1/")
  st=${st:-?}
  if [[ "$st" != "$last_status" ]]; then
    echo "[$NAME] status: ${last_status:-—} -> $st  (t+$(( ($(date +%s)-start)/60 ))m)"; last_status="$st"
  fi
  $HF jobs logs "$JOB" > "$LOG.new" 2>/dev/null && mv "$LOG.new" "$LOG"
  n=$(wc -l < "$LOG" 2>/dev/null | tr -d ' '); n=${n:-0}
  if (( n > last_lines )); then
    tail -n $(( n - last_lines )) "$LOG" | grep -E "Traceback|Error|error|OOM|out of memory|Killed|nan|NaN|inf loss" | grep -vE "error_bars|errors=|Errno 2.*wandb" | head -5 | sed "s/^/[$NAME] !! /"
    tail -n $(( n - last_lines )) "$LOG" | grep -E "step:[0-9]+|step=[0-9]+" | tail -1 | sed -E "s/^.*(step[:=][0-9]+[^ ]*).*(loss[:=][0-9.]+).*/[$NAME] \1 \2/" | grep -E "step" || true
    last_lines=$n; last_progress=$(date +%s)
  fi
  now=$(date +%s)
  if (( now - last_progress > STALL_MIN*60 )) && [[ "$st" == "RUNNING" ]]; then
    echo "[$NAME] !! STALL: no new log line for $(( (now-last_progress)/60 )) min (rule: $STALL_MIN)"; last_progress=$now
  fi
  case "$st" in
    COMPLETED) echo "[$NAME] ✓ COMPLETED after $(( (now-start)/60 )) min of watching"; exit 0 ;;
    ERROR|FAILED|CANCELED|CANCELLED) echo "[$NAME] ✗ $st after $(( (now-start)/60 )) min — last lines:"; tail -n 6 "$LOG" | sed "s/^/[$NAME]    /"; exit 1 ;;
  esac
  sleep 60
done
