#!/bin/zsh
# Blind seat check via Terminal bridge (camera TCC). Writes verdict to
# tools/seat_checks/seat_check.log; frames archived alongside.
HERE="${0:A:h}"
/opt/homebrew/Caskroom/miniforge/base/envs/lerobot/bin/python "$HERE/seat_check.py" 1 > "$HERE/seat_checks/seat_check.log" 2>&1
echo "SEATCHECK_DONE rc=$?" >> "$HERE/seat_checks/seat_check.log"
