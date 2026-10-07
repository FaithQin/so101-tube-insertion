#!/bin/zsh
# ONE certification replay (gate G-R), fully gated: every check aborts, none of them warn.
#
#   tools/run_cert.sh <label>          e.g. tools/run_cert.sh CERT-B3-02
#
# Run from a TCC-granted Terminal (scene_check opens the front camera). In order:
#   1. ping -> cert_gate temps: every servo read and <= 52 C           else exit 3, nothing moved
#   2. home_arm --pose v4 -> cert_gate home: HOMED, settled, no WARNING else exit 3, arm relieved
#   3. scene_check -> preflight: tube <= 8 px, pose in v4 support, <= 60 C  else exit 3, no replay
#   4. replay_telemetry --episode 0, telemetry to <label>_<stamp>_telemetry.csv (the block's baseline)
#                                                                     non-zero -> exit 5, no seat call
#   5. cert_summary: loop Hz / ee_proxy / jaw_min, for the record
#   6. retract to home, then relax torque -- ALWAYS, before anyone is asked anything (a replay leaves
#      the arm holding at the rack; Sep 14 12:47: 70 C after 15 min of exactly that; 14:47: 6 C in
#      3 min waiting for a typed verdict)
#   7. print the one command that records the operator's verdict, called to the chat:
#        python tools/cert_gate.py record tools/scored_logs/<label>_<stamp> SEAT|MISS   -> <run>.seat
#   exit 0 when the replay completed (verdict pending), 3 gate abort, 5 replay failed.
#
# WHY (Sep 14 2026). G-R needs a replay to SEAT before every block and after any failed episode
# (Eval Protocol -- v4 merged (Sep 6).md :245, :291). By hand this was six chat round trips per
# pass, and the morning's one miss was the pass that skipped the scene check. Gate, don't narrate.
set -u
LABEL="${1:-}"
HERE="${0:A:h}"; PROJ="${HERE:h}"
B=/opt/homebrew/Caskroom/miniforge/base/envs/lerobot/bin
cd "$PROJ" || exit 1
TTY="${CAPSTONE_TTY:-/dev/tty}"
[ -n "$LABEL" ] || { echo "usage: tools/run_cert.sh <label>   (e.g. CERT-B3-02)"; exit 2; }

STAMP=$(date +%Y%m%d_%H%M%S)
RUN="$HERE/scored_logs/${LABEL}_${STAMP}"
mkdir -p "$HERE/scored_logs"
echo "=== certification $LABEL  ($STAMP) ==="

relieve() {  # retract from wherever the arm is, then cut torque -- the Sep 14 12:47 rule
  $B/python tools/home_arm.py --pose v4 > "${RUN}.retract" 2>&1
  $B/python tools/relax_arm.py 2>&1 | tail -1 | sed 's/^/  /'
}

# --- 1. thermal, before torque -----------------------------------------------------------------
$B/python tools/ping_motors.py follower > "${RUN}.ping" 2>&1
OUT=$($B/python tools/cert_gate.py temps "${RUN}.ping"); RC=$?
echo "  $OUT"
if [ "$RC" -ne 0 ]; then
  say "abort. too hot for certification."
  echo "CERT ABORTED ($LABEL) — nothing moved. Wait for <= 52 C, then re-run."
  exit 3
fi

# --- 2. home until the elbow lands in the certification band (bounded) ---------------------------
# Sep 17 07:25: seated passes settle the elbow at 85.6; the left-miss passes at 86.2-86.9 (worn gearbox,
# another tooth). Re-home up to HOME_BUDGET times; each attempt is ~8 s against a ~60 s replay.
HOME_BUDGET=5
BAND_FLAG="--elbow-band"
if [ "${CAPSTONE_CERT_NO_BAND:-0}" = "1" ]; then   # Sep 17 07:30: operator's escape hatch when the elbow parks high; G-R itself needs only a seat
  BAND_FLAG=""; HOME_BUDGET=1
  echo "  (elbow band OFF for this pass: CAPSTONE_CERT_NO_BAND=1)"
fi
RC=1
for attempt in 1 2 3 4 5; do
  [ "$attempt" -le "$HOME_BUDGET" ] || break
  $B/python tools/home_arm.py --pose v4 > "${RUN}.home" 2>&1
  OUT=$($B/python tools/cert_gate.py home "${RUN}.home" $BAND_FLAG); RC=$?
  echo "  home attempt $attempt: $OUT"
  [ "$RC" -eq 0 ] && break
done
if [ "$RC" -ne 0 ]; then
  say "abort. elbow would not settle in band."
  echo "CERT ABORTED ($LABEL) — the elbow did not settle in the certification band in $HOME_BUDGET homes (${RUN}.home); arm relieved."
  echo "       This is the worn gearbox. Let the arm rest, then re-run; if it never lands, the plant cannot be certified today."
  relieve
  exit 3
fi

# --- 3. scene, then the same preflight a scored trial passes -----------------------------------
$B/python tools/scene_check.py > "${RUN}.scene" 2>&1
grep -E "tube \(blue cap\)|rack \(orange funnel\)|tube angle" "${RUN}.scene" | sed 's/^/  /'
OUT=$($B/python tools/preflight.py "${RUN}.scene" "${RUN}.home" "${RUN}.ping" v4); RC=$?
echo "  $OUT"
if [ "$RC" -ne 0 ]; then
  say "abort. scene."
  echo "CERT ABORTED ($LABEL) — no replay; arm relieved. Fix the cause above (nudge the tube / rack), then re-run."
  relieve
  exit 3
fi

# --- 4. the replay: open-loop, no policy, no camera --------------------------------------------
say "replay"
$B/python tools/replay_telemetry.py --episode 0 --out "${RUN}_telemetry.csv" 2>&1 | tee "${RUN}.replay" \
  | grep -E "^replay:|frames|BASELINE|REPLAY|ABORTED|REFUSED|REJECTED" | sed 's/^/  /'
RC=${pipestatus[1]}
if [ "$RC" -ne 0 ]; then
  say "replay failed"
  echo "CERT FAILED ($LABEL) — replay exit $RC (${RUN}.replay); no seat call taken; arm relieved."
  relieve
  exit 5
fi

# --- 5. the numbers the pass is logged with ------------------------------------------------------
SUMMARY=$($B/python tools/cert_summary.py "${RUN}_telemetry.csv")
echo "  $SUMMARY"

# --- 6. never leave it holding: retract and relax BEFORE anyone is asked anything -----------------
# The first version prompted here and held torque while it waited: CERT-B3-02 (Sep 14 14:47) sat at
# the rack for three minutes, wrist_flex 45 -> 51 C. The seat is the tube standing after release, so
# the arm's retreat cannot change it.
relieve

# --- 7. the verdict is the operator's, called to the chat; one command lands it in the receipt -----
say "call the seat"
echo "CERT $LABEL ran ($STAMP). The operator calls seat or miss to the chat; then:"
echo "  python tools/cert_gate.py record tools/scored_logs/${LABEL}_${STAMP} SEAT|MISS"
exit 0
