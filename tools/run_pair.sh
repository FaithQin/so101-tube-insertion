#!/bin/zsh
# One PAIR of the Sep 6 2026 bench, blinded at the console: the two arms in the order the
# seeded schedule drew, each trial labelled by its opaque token, through the existing runners.
#
#   tools/run_pair.sh <pair_id>                                  # e.g. tools/run_pair.sh P07
#   CAPSTONE_RESERVE_AS=S3-CHAMFERLESS tools/run_pair.sh P26     # a RESERVE pair, declared first
#   CAPSTONE_PAIR_TRIAL=2 tools/run_pair.sh P07                  # resume at trial 2 after a gate abort there
#
# What reaches the terminal: pair id / stratum / "trial k of 2", the runners' GATE lines
# (PREFLIGHT, REPAIRED, attempt, human-only, carry start, TRIAL ABORTED), the scorer's verdict
# lines, and ONE HEALTH line per trial that this launcher composes from the trial's own receipt
# (loop_hz / telemetry rows / slow ticks -- see trial_health). Everything else the runners print
# -- the policy id, `side=`, the state-width ABORT and its CAPSTONE_FORCE_WIDE_STATE pin, the
# CAPSTONE_STATE_ROUTING WIDE/narrow line, the state width -- goes to
# $SEALED/<pair>_<k>.console. The token -> arm map is written to $SEALED/<pair>.txt BEFORE each
# trial, and the pair's TWO analysis rows (pair_id,stratum,arm,label = tools/v4_bench_analysis.py
# --pairs) to $SEALED/pairs.csv in one append BEFORE the first one -- both arms or neither, so an
# abandoned pair is NOT_RUN in the disposition table and never half a pair the analysis refuses
# to read. The operator does not open $SEALED until scoring is done. What this blinding does and
# does not protect against: tools/v4_schedule.py, "TOKENS".
#
# Checkpoints are never written here: the VLA arms take theirs from the hold-out verdict JSON
# (tools/v4_verdict_checkpoint.py); ACT benches its top-level weights (final = the verdict,
# the lab notebook (private) Sep 5). Cadence per the frozen protocol: ACT n=25 both arms, VLAs n=50 sync.
# A pair whose first trial stops at a gate has consumed nothing: fix the cause, re-run the pair.
# A pair whose first trial RAN but cannot be scored (the scorer's ledger row says INFRA, or says
# nothing) is VOID and stops here with exit 6 -- trial 2 is never started, because pairs are
# atomic and its start state would be spent on a pair that is already discarded.
set -u
PAIR="${1:-}"
HERE="${0:A:h}"; PROJ="${HERE:h}"
B=/opt/homebrew/Caskroom/miniforge/base/envs/lerobot/bin
cd "$PROJ" || exit 1
SCHED="${CAPSTONE_SCHEDULE:-analysis/bench_sep6/schedule.csv}"
SEALED="${CAPSTONE_SEALED_DIR:-analysis/bench_sep6/sealed}"
TTY="${CAPSTONE_TTY:-/dev/tty}"
[ -n "$PAIR" ] || { echo "usage: tools/run_pair.sh <pair_id>   (pairs: $SCHED)"; exit 2; }

# --- the launcher owns the CAPSTONE_* environment -------------------------------------------
# A runner reads what this file sets, or nothing at all. A leftover export in the operator's
# shell changes a SCORED trial and NOTHING records it -- the console is blinded, the ledger row
# names the policy the runner thought it ran, and the receipt did not say. The ones that bite:
#   CAPSTONE_POLICY_GEN=v3     -> the Sep 3-4 weights instead of v4, gated on v3 support
#                                 (tools/smolvla_dryrun.py:169, tools/pi05_sides.py:73)
#   CAPSTONE_FORCE_WIDE_STATE  -> pins the state routing (tools/state_routing.py:38). The VLA
#                                 runners set it per side (run_smolvla_trial.sh:251,
#                                 run_pi05_trial.sh:159); run_scored_trial.sh NEVER sets it, so
#                                 ACT ran on whatever the shell held -- state_routing.py's own
#                                 "silent distribution shift that presents as the policy just
#                                 doesn't work".
#   CAPSTONE_ALLOW_WORKTREE    -> turns OFF the main-checkout gate (run_scored_trial.sh:28,
#                                 run_smolvla_trial.sh:110, run_pi05_trial.sh:52) -- the gate
#                                 that exists because of Aug 27 INFRA #3.
#   CAPSTONE_RESOLVE_ONLY      -> run_smolvla_trial.sh:197 exits 0 having run NOTHING, and a
#                                 zero exit is what this launcher reads as a completed trial.
#   CAPSTONE_CHECKPOINT        -> the weights (tools/smolvla_dryrun.py:193,
#                                 run_pi05_trial.sh:88); set per arm below, from the verdict.
#   CAPSTONE_GRASP_LOCK, CAPSTONE_HOME_POSE, CAPSTONE_TELEMETRY_CSV -> the rollout wrapper and
#                                 the sidecar (rollout_30hz_stale_ok.py:389, :266,
#                                 tools/telemetry_sidecar.py:56). All three runners set these;
#                                 the launcher does not get to depend on that.
# ALLOW-list, not deny-list: a knob added to a runner tomorrow is neutralised the day it is
# added. Neutralise rather than abort -- the rule the start pose already follows below (an S0
# pair with a stale CAPSTONE_START_POSE runs frame 0; it does not stop the bench) -- and the
# sealed record names what was cleared and what every trial was actually launched with.
typeset -a PAIR_ENV_KEEP CLEARED_ENV
PAIR_ENV_KEEP=(CAPSTONE_SCHEDULE CAPSTONE_SEALED_DIR CAPSTONE_TTY CAPSTONE_RESERVE_AS CAPSTONE_PAIR_TRIAL)
for _envvar in ${(ok)parameters}; do
  [[ $_envvar == CAPSTONE_* ]] || continue
  [[ ${(Pt)_envvar} == *export* ]] || continue
  (( ${PAIR_ENV_KEEP[(I)$_envvar]} )) && continue
  CLEARED_ENV+=("$_envvar=${(P)_envvar}")
  unset "$_envvar"
done

capstone_env() {  # every exported CAPSTONE_* a runner launched right now would see
  local v out=""
  for v in ${(ok)parameters}; do
    [[ $v == CAPSTONE_* ]] || continue
    [[ ${(Pt)v} == *export* ]] || continue
    out+=" $v=${(P)v}"
  done
  print -r -- "${out:- (none)}"
}

# --- the pair, from the committed schedule only -------------------------------------------
field() { $B/python tools/v4_schedule.py --schedule "$SCHED" --print "$1" --pair "$PAIR"; }
FAMILY=$(field family)     || { echo "ABORT: pair '$PAIR' is not in $SCHED"; exit 2; }
STRATUM=$(field stratum)   || exit 2
ROLE=$(field role)         || exit 2
ORDER=$(field arm_order)   || exit 2
TOK1=$(field token_first)  || exit 2
TOK2=$(field token_second) || exit 2
case "$ORDER" in AB|BA) ;; *) echo "ABORT: pair $PAIR has arm_order '$ORDER'"; exit 2 ;; esac

if [ "$STRATUM" = "RESERVE" ]; then
  # A reserve pair has no stratum until it is used (chamferless, or a re-draw of an
  # invalidated pair). The operator declares it; the launcher never guesses.
  case "${CAPSTONE_RESERVE_AS:-}" in
    S0-CLEAN|S1-ROT45|S2-PREGRASPED|S3-CHAMFERLESS) STRATUM="$CAPSTONE_RESERVE_AS" ;;
    *) echo "ABORT: $PAIR is a RESERVE pair ($ROLE) -- declare what it is being used for:"
       echo "       CAPSTONE_RESERVE_AS=S0-CLEAN|S1-ROT45|S2-PREGRASPED|S3-CHAMFERLESS tools/run_pair.sh $PAIR"
       exit 2 ;;
  esac
fi

# --- the start pose follows the stratum, never the shell ----------------------------------
# ...and an S2 PRE-GRASPED pair whose start pose cannot be commanded is REFUSED here, before a byte
# is written to $SEALED. The launcher used to export CAPSTONE_START_POSE=v4-carry, seal both rows
# and launch trial 1, and only the runner found out: home_arm.py:80-89 refuses `--pose v4-carry`
# during argument validation -- tools/_arms.py drives the bus in RANGE_M100_100, the pose table is
# in DEGREES, and on the saved calibration the carry wrist_roll lands +39.8 deg away and
# shoulder_pan +7.6 (tools/norm_units.py; computed Sep 12, no bus) -- so every homing attempt
# failed preflight and the runner ended "TRIAL ABORTED -- nothing launched, no trial consumed"
# (run_scored_trial.sh:91-139) over a pairs.csv that already held the pair. Launching P19-P24 would
# have sealed all six as S2-PREGRASPED and run none of them. Found by the Sep 7 night audit.
#
# The check is home_arm's own guard, asked early: norm_units.offenders() on HOME_POSES["v4-carry"]
# against _arms.FOLLOWER_CAL, with _arms.JOINTS -- the names home_arm.py:21,27,80-81 uses -- so the
# day _arms.py runs in the production norm mode and home_arm lets the pose through, this does too.
# home_arm.py is PARSED (ast), never imported: it runs argparse at import and then drives the arm.
# _arms is imported for its paths only; its bus opens in open_bus() (_arms.py:43-50), never at
# import. Anything that stops the check answering refuses as well: an unverified carry pose is not a
# commandable one. What S2 becomes instead is not this launcher's call -- the protocol's fallback is
# NUDGE-start, and adopting it is an amendment -- so the refusal names that route and stops.
CARRY_POSE=v4-carry
carry_pose_offenders() {  # prints why $CARRY_POSE cannot be commanded; 0 = it can, 1 = it cannot, else unknown
  $B/python -c '
import ast, sys
tools, tag = sys.argv[1], sys.argv[2]
sys.path.insert(0, tools)
try:
    from _arms import FOLLOWER_CAL, JOINTS
    import norm_units
    tree = ast.parse(open(f"{tools}/home_arm.py").read())
    poses = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and any(getattr(t, "id", None) == "HOME_POSES" for t in n.targets))
    bad = norm_units.offenders(dict(zip(JOINTS, poses[tag])), norm_units.load_calibration(FOLLOWER_CAL))
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}")
    sys.exit(3)
if bad:
    print(norm_units.describe(bad))
    sys.exit(1)
' "$HERE" "$CARRY_POSE" 2>/dev/null
}

case "$STRATUM" in
  S2-PREGRASPED)
    CARRY_WHY=$(carry_pose_offenders); CARRY_RC=$?
    if [ "$CARRY_RC" -ne 0 ]; then
      if [ "$CARRY_RC" -eq 1 ]; then
        echo "ABORT: $PAIR is S2-PREGRASPED, and its start pose $CARRY_POSE cannot be commanded on this rig:"
        echo "       $CARRY_WHY (tools/_arms.py drives RANGE_M100_100, the pose table is DEGREES)."
        echo "       tools/home_arm.py refuses this pose, so every trial of the pair would stop at its homing gate."
      else
        echo "ABORT: $PAIR is S2-PREGRASPED, and whether its start pose $CARRY_POSE can be commanded could not"
        echo "       be checked (${CARRY_WHY:-exit $CARRY_RC}) -- an unverified carry pose is not a commandable one."
      fi
      echo "       Nothing consumed: no trial was started, and nothing was written to the sealed record."
      echo "       The protocol's fallback when the carry pose does not validate is NUDGE-start (tube ~1 cm"
      echo "       image-right; Eval Protocol -- v4 merged (Sep 6).md, section 1 Conditions). Running S2 that way"
      echo "       is a protocol choice this launcher does not make: append it to the protocol's"
      echo "       Amendments log (section 7), dated and with its reason, before any S2 pair is launched."
      exit 2
    fi
    export CAPSTONE_START_POSE="$CARRY_POSE" ;;
  S1-ROT45)
    # Sep 14 15:26 (protocol §7): the PI redefined S1 as a rotation about the tube's midpoint, so the cap
    # sits at preflight.TUBE_TARGETS["S1-ROT45"], not on the mark. The runner's preflight reads the name.
    export CAPSTONE_TUBE_TARGET=S1-ROT45
    unset CAPSTONE_START_POSE ;;
  *) unset CAPSTONE_START_POSE ;;
esac

mkdir -p "$SEALED" && chmod 700 "$SEALED"

# The count goes to the console, the NAMES only to the sealed record: CAPSTONE_FORCE_WIDE_STATE
# carries the routing word this launcher's whole job is to keep off the operator's screen.
if (( ${#CLEARED_ENV} )); then
  print -r -- "$(date +%Y%m%d_%H%M%S) pair $PAIR  cleared from the operator's shell: ${(j: :)CLEARED_ENV}" >> "$SEALED/$PAIR.txt"
  echo "  cleared ${#CLEARED_ENV} leftover CAPSTONE_ setting(s) from the shell -- named in the sealed record"
fi

# --- checkpoints for BOTH arms up front, so an abort here names neither order ---------------
# (the resolver's stderr names the arm it could not resolve: it goes to the sealed dir)
typeset -A CKPT
case "$FAMILY" in
  act) ;;
  smolvla|pi05)
    for arm in A B; do
      CKPT[$arm]=$($B/python tools/v4_verdict_checkpoint.py "faithqin/${FAMILY}-tube-${arm}-v4" 2>> "$SEALED/${PAIR}_resolve.log") || {
        echo "ABORT: no hold-out verdict for one of faithqin/${FAMILY}-tube-{A,B}-v4 -- run tools/v4_holdout_all.sh 6 0 4 ${FAMILY}-tube-A-v4 ${FAMILY}-tube-B-v4 first"
        exit 2; }
    done ;;
  *) echo "ABORT: unknown family '$FAMILY' in $SCHED"; exit 2 ;;
esac

# CAPSTONE_CHECKPOINT is exported by run_one rather than prefixed here, so the receipt's
# "launched with:" line is the environment the runner really got, not a claim about it.
run_trial() {  # <arm> <token>
  case "$FAMILY" in
    act)     tools/run_scored_trial.sh "faithqin/act-tube-${1}-v4" 25 off "$2" ;;
    smolvla) tools/run_smolvla_trial.sh "$1" "$2" 50 ;;
    pi05)    tools/run_pi05_trial.sh "$1" "$2" 50 ;;
  esac
}

# Lines the operator may see. Gate lines are indented two spaces by the runners; the scorer's
# lines carry no policy. NOT here: the runner banner (policy=), side=, routing_line=,
# state_routing=, force_pin=, the state-width check, the SmolVLA/pi0.5 prompt line.
#
# ABORTS ARE LISTED ONE BY ONE, because three of them name the arm and a bare `^ABORT` passed
# all three:
#   * check_state_width.py:135 prints its ABORT to STDERR, so the `| sed 's/^/  /'` that indents
#     the gate's stdout (run_smolvla_trial.sh:172, run_pi05_trial.sh:106) never touches it and it
#     arrives at column 0 carrying the declared/stats widths, WIDE-vs-narrow, and
#     "pin CAPSTONE_FORCE_WIDE_STATE=<0|1>" -- which IS the arm (A pins 0, B pins 1,
#     run_pi05_trial.sh:159). One line, the whole blind.
#   * "ABORT: side must be A or B (got 'B')" (run_smolvla_trial.sh:90, run_pi05_trial.sh:37).
#   * "ABORT: cannot tell which dataset 'faithqin/act-tube-B-v4' ..." (run_scored_trial.sh:44).
# The rest name no arm and stay: they are the operator's only clue about why a trial stopped.
WL='^  (PREFLIGHT|REPAIRED|\(attempt|\(human-only|\(carry start|start pose:)|^TRIAL ABORTED|^VERDICT |^  (FINAL:|seated |never seated|\*\*\* INFRA|frames=)|^ABORT: (running from a git worktree|could not |no checkpoint chosen|the trained prompt resolved|state-width gate failed|label must be|n_action_steps must be|CAPSTONE_START_POSE=)'
# Second pass, because a whitelist matches the START of a line and cannot see the rest of it,
# and these runners are frozen for the bench: no line reaches the console carrying a field that
# names the arm, however its prefix got past WL.
DENY='(side|policy|policy_state|declared|stats|routing|force|force_pin|max_state_dim)=|CAPSTONE_(FORCE_WIDE_STATE|STATE_ROUTING)|tube-[AB]|-[AB]-v[0-9]|side must be|state \[[0-9]+\]'

# --- the pair's two analysis rows: both arms, ONE append, BEFORE the first trial ------------
# pairs.csv is what `tools/v4_bench_analysis.py --pairs` reads, and disposition() does not
# tolerate half a pair -- v4_bench_analysis.py:407-409
#     for pid, p in pairs.items():
#         if set(p["labels"]) != {"A", "B"}:
#             raise ValueError(f"pair {pid} must schedule exactly one A and one B trial")
# raises inside the loop over EVERY pair, so main() prints "REFUSED: ..." and exits 2: one
# abandoned pair would take the whole evening's table down, all 27 others with it. Aborts are
# EXPECTED today (a gate refusal, a hardware stop, a Ctrl-C at the pause), so both rows go down
# together before anything launches -- there is then no instant, and no kill, that leaves a pair
# the analysis cannot read. An abandoned pair then reads as exactly what it is: two labels, no
# ledger row, which disposition codes NOT_RUN (v4_bench_analysis.py:85, :437-441), the code that
# exists for it. Writing the rows only once each trial had scored would instead drop an
# abandoned pair out of the table entirely -- and would still hold half a pair through the pause
# between trial 1 and trial 2, which is where the operator's Ctrl-C lands. The rows are also the
# blinding key, and losing the key loses the trial: they are written first, never last.
seal_pair_rows() {  # <first-arm> <first-token> <second-arm> <second-token>
  local csv="$SEALED/pairs.csv" want have seen
  [ -f "$csv" ] || print -r -- "pair_id,stratum,arm,label" > "$csv"
  want="$PAIR,$STRATUM,$1,$2
$PAIR,$STRATUM,$3,$4"
  have=$(grep "^$PAIR," "$csv")
  if [ -n "$have" ]; then
    [ "$have" = "$want" ] && return 0          # the same pair again: a re-run or a resume
    # A reserve pair re-declared under another stratum would otherwise append a SECOND row for
    # the same arm, which raises "lists arm A twice" (v4_bench_analysis.py:403-404). Say so
    # without naming an arm or a token -- the console stays blinded.
    seen=$(print -r -- "$have" | cut -d, -f2 | sort -u | tr '\n' ' ')
    echo "ABORT: $csv already holds this pair's rows, sealed, under stratum: $seen"
    echo "       $PAIR is now being run as $STRATUM, and rewriting a sealed record is not on offer."
    echo "       Re-declare it as ${seen% }, or use a different reserve pair from $SCHED."
    return 2
  fi
  print -r -- "$want" >> "$csv"
}

# --- did the trial actually produce a verdict? ---------------------------------------------
# A zero exit does NOT mean the trial is scoreable. Every runner ends with
#   if grep -q "VERDICT SEAT" ...; then say "seat"; elif grep -q "VERDICT INFRA" ...; then
#   say "infra"; else say "miss"; fi
# (run_scored_trial.sh:193, run_smolvla_trial.sh:311, run_pi05_trial.sh:207) and `say` succeeds,
# so INFRA -- and a scorer that produced UNKNOWN off a video it never found (real: B2-A-01
# 20260829_230615, "UNKNOWN,0.0,0") -- exits 0 just like a SEAT. Pairs are ATOMIC (Eval Protocol
# -- v4 merged (Sep 6).md:277: "if one arm's trial is invalid, both are discarded and the pair
# re-drawn"), so a first trial that carries no verdict must stop the pair before it spends the
# second start state on a pair that is already void.
#
# The signal is the LEDGER ROW the scorer appends, not the console: csv.writer inside the same
# process that ran score_episode.py (run_scored_trial.sh:182-189, run_smolvla_trial.sh:150-158,
# run_pi05_trial.sh:195-202). A scorer that never got there leaves no row at all, and the test
# applied here is the one the analysis applies (v4_bench_analysis.py:423-426 -- INFRA, or
# anything that is not SEAT/MISS, invalidates), so the launcher and the disposition table cannot
# drift apart. A genuine MISS writes verdict "MISS" and runs on: a miss is data.
LEDGERS=(tools/scored_logs/trials.csv tools/scored_logs/trials_smolvla.csv)
ledger_verdict() {  # <token> <not-before stamp> -> that trial's verdict, empty if it wrote none
  $B/python -c '
import csv, sys
tok, since = sys.argv[1], sys.argv[2]
out = ""
for path in sys.argv[3:]:
    try:
        rows = list(csv.DictReader(open(path, newline="")))
    except OSError:
        continue
    for r in rows:                      # a re-run reuses the token: take rows from THIS trial on
        if r.get("label") == tok and (r.get("stamp") or "") >= since:
            out = r.get("verdict") or ""
print(out)
' "$1" "$2" "${LEDGERS[@]}" 2>> "$SEALED/${PAIR}_resolve.log"
}

# --- how the trial RAN, told without saying which arm ran it --------------------------------
# The whole `.verify` receipt is sealed, because it names the arm three ways (side=, policy=,
# force_pin= -- run_smolvla_trial.sh:280-284), and with it sealed NOTHING on this console
# separated a healthy trial from a degraded one: today's first trial held 19.62 Hz with 35 slow
# ticks, and a trial that fell to 12 Hz would have read identically. So the launcher opens the
# receipt ITSELF and prints three numbers, none of which can name an arm -- never by letting a
# runner's own line through:
#   loop_hz      the wall-clocked control rate the protocol requires reported (:54)
#   rows/frames  gate G-T, "row count == recorded frame count" (:290)
#   slow_ticks   ticks the loop ran under its 20 Hz budget
# Only the unambiguous half of G-T stops the pair: a sidecar that wrote NO rows at all. A
# row/frame mismatch, or a receipt with no numeric loop_hz, prints DEGRADED and runs on -- the
# sidecar has run on hardware exactly once (ACT, Sep 6: 873 rows, 873 frames) and never on a VLA
# runner, and that same trial's receipt carried an EMPTY `loop_hz=` at a real 19.62 Hz
# (T0-SMOKE_20260906_145245.verify). Neither may void a 15-minute pair on its own; the operator
# sees the numbers, and MANUAL:TELEMETRY_MISSING (v4_bench_analysis.py:88) is where that call
# lands.
trial_health() {  # <token> <not-before stamp>; one arm-free line, returns 1 when gate G-T failed
  local tok="$1" since="$2"
  local -a seen; seen=( "$HERE/scored_logs/${tok}"_*.verify(Nom) )   # newest first
  local v="${seen[1]:-}"
  # A re-run reuses its token (scored_logs holds B2-A-01 at both 20260829_230615 and _230727),
  # so the newest receipt for this token is not necessarily THIS trial's: the runner names it
  # <label>_<stamp> (run_scored_trial.sh:69-70) and the stamp must be at or after the one taken
  # before this trial launched. An older trial's healthy numbers vouching for a trial that wrote
  # none is exactly the failure this line is here to catch.
  if [ -n "$v" ]; then
    local base="${${v:t}%.verify}"
    [[ "${base#${tok}_}" < "$since" ]] && v=""
  fi
  if [ -z "$v" ]; then
    echo "  HEALTH  no .verify receipt from this trial  -> GATE G-T: nothing to check"
    return 1
  fi
  local tel="${v%.verify}_telemetry.csv" score="${v%.verify}.score"
  local hz slow frames rows=0
  hz=$(grep -oE '^loop_hz=[0-9]+\.[0-9]+' "$v" | tail -1 | cut -d= -f2)
  slow=$(grep -oE '^slow_ticks=[0-9]+' "$v" | tail -1 | cut -d= -f2)
  [ -f "$tel" ] && rows=$(( $(wc -l < "$tel") - 1 ))          # minus the header row
  [ "$rows" -lt 0 ] && rows=0
  frames=$(grep -oE '^  frames=[0-9]+' "$score" 2>/dev/null | head -1 | cut -d= -f2)
  local health="OK"
  if [ "$rows" -eq 0 ]; then
    health="GATE G-T: the telemetry sidecar wrote no rows"
  elif [ -n "$frames" ] && [ "$rows" -ne "$frames" ]; then
    health="DEGRADED: telemetry rows != recorded frames"
  elif [ -z "$hz" ]; then
    health="DEGRADED: no wall-clocked loop_hz on the receipt"
  fi
  echo "  HEALTH  loop_hz=${hz:-?}  telemetry=${rows}/${frames:-?} rows  slow_ticks=${slow:-?}  -> $health"
  [ "$rows" -gt 0 ]
}

run_one() {  # <k> <arm> <token>
  local k="$1" arm="$2" tok="$3"
  local console="$SEALED/${PAIR}_${k}.console"
  local stamp; stamp=$(date +%Y%m%d_%H%M%S)
  if [ -n "${CKPT[$arm]:-}" ]; then export CAPSTONE_CHECKPOINT="${CKPT[$arm]}"; else unset CAPSTONE_CHECKPOINT; fi
  print -r -- "$stamp trial $k of 2  token $tok  arm $arm  family $FAMILY  stratum $STRATUM  start_pose ${CAPSTONE_START_POSE:-frame0}  checkpoint ${CKPT[$arm]:-top-level}" >> "$SEALED/$PAIR.txt"
  print -r -- "    launched with:$(capstone_env)" >> "$SEALED/$PAIR.txt"
  echo "=== pair $PAIR  stratum $STRATUM  trial $k of 2 ==="
  run_trial "$arm" "$tok" 2>&1 | tee "$console" | grep --line-buffered -E "$WL" | grep --line-buffered -v -E "$DENY"
  local rc=${pipestatus[1]}
  if [ "$rc" -ne 0 ]; then
    local again="tools/run_pair.sh $PAIR"; [ "$k" = 2 ] && again="CAPSTONE_PAIR_TRIAL=2 $again"
    echo "pair $PAIR trial $k of 2: the runner stopped before scoring (exit $rc) -- nothing consumed."
    echo "fix the cause above, then re-run: $again"
    echo "(an abort line that names the arm is not printed -- it is in $console, which unblinds $PAIR)"
    return "$rc"
  fi
  local gt=0; trial_health "$tok" "$stamp" || gt=1
  local why="" verdict
  verdict=$(ledger_verdict "$tok" "$stamp")
  case "$verdict" in
    SEAT|MISS) ;;
    *) why="NOT SCOREABLE -- ${verdict:-no ledger row}" ;;
  esac
  # A trial with no telemetry is invalid under gate G-T, and pairs are atomic: same stop, same
  # instruction. The scoring failure is named first when both are true -- it is the larger one.
  [ "$gt" -eq 0 ] || why="${why:-INVALID -- gate G-T failed, see the HEALTH line above}"
  if [ -n "$why" ]; then
    print -r -- "$stamp trial $k of 2  token $tok  $why -- pair VOID" >> "$SEALED/$PAIR.txt"
    echo "pair $PAIR trial $k of 2: $why (token $tok)."
    echo "The trial ran but cannot be counted, and pairs are atomic, so this PAIR IS VOID."
    [ "$k" = 1 ] && echo "trial 2 of 2 was NOT started -- its start state is unspent."
    echo "Discard both halves, log the invalidation against token $tok (stamp $stamp), and re-draw the pair:"
    echo "DO NOT re-run $PAIR: its token is already in the ledger, so a re-run scores the same"
    echo "label twice and the analysis codes both trials DUPLICATE_LABEL -- ~11 min for nothing."
    echo "  CAPSTONE_RESERVE_AS=$STRATUM tools/run_pair.sh <a reserve pair from $SCHED>"
    return 6
  fi
  return 0
}

# --- the pause between the two trials is a GATE, not a message ------------------------------
# `read -r _ < "$TTY"` does not gate. Verified by running it under zsh (Sep 6): on EOF
# (/dev/null, an empty file, a Ctrl-D on a real terminal) `read` returns 1 and EXECUTION
# CONTINUES; when the open itself fails (the operator's terminal closed, no controlling tty, a
# bad CAPSTONE_TTY) zsh prints a diagnostic, `read` returns 1, and EXECUTION CONTINUES. There
# is no `set -e`. Trial 2 then launched onto a start state nobody had reset, possibly with a
# hand still in the workspace, and the pair read as clean afterwards -- worse than a lost pair.
# A REGULAR FILE is worse still: the prompt is written into it with `>` and read straight back
# as the operator's ENTER, so the gate answers itself.
#
# So the channel must be a TERMINAL -- checked before trial 1, while nothing has been consumed
# -- and the read must SUCCEED, checked between the trials, where the only safe move left is to
# stop and let the operator resume deliberately with CAPSTONE_PAIR_TRIAL=2.
#
# The prompt is NOT re-issued until an empty line arrives: an unbounded re-prompt loop is a way to
# hang the bench. Any line typed AFTER the prompt is the operator's ENTER; only failure and EOF stop.
# `{ exec 3<"$TTY" } 2>/dev/null` does NOT report the open: measured under zsh, it returns 1
# with fd 3 open on a live pty (openrc=1, `[ -t 3 ]` true, empty stderr) -- `exec` with only
# redirections is a null command and does not set a status of its own. So the gate decides on
# `[ -t 3 ]`, which is the property that actually matters and is false either way when the open
# failed. A test that trusted the exec status would have refused every real terminal.
pause_channel_is_a_terminal() {
  { exec 3<"$TTY" } 2>/dev/null
  if [ -t 3 ]; then exec 3<&-; return 0; fi
  exec 3<&-
  return 1
}

# --- ...and only an ENTER typed AFTER the prompt answers it ---------------------------------
# `read` takes the first line already in the terminal's input queue, and nothing emptied that
# queue: a newline typed at ANY moment during trial 1 -- the reflex ENTER while the arm runs, a key
# hit to wake the screen -- sat there for the minute the trial took and answered the prompt the
# instant it was printed. Trial 2 then launched onto a start state nobody had reset, the one thing
# this gate exists to stop, and tests/test_run_pair.py's pty fixture (which wrote its "\n" before
# the launcher had even started) called that a PASS. Found by the Sep 7 night audit; fixed Sep 12.
#
# So the queue is drained, on the same fd, BEFORE the prompt is printed, and only then read. Drain
# first, prompt second: an ENTER pressed the moment the prompt appears is never the one thrown
# away, and the window between the two is a single print. `sysread -t 0`, not `read -t 0`:
# measured under zsh 5.9 on a pty (Sep 12), `read -t 0 -k 1` returns 1 both for "nothing queued"
# and for a queued Ctrl-D, so a drain built on it stops at the EOF and leaves the ENTER of a
# Ctrl-D-then-ENTER queued to release the gate. sysread tells the two apart (4 = nothing there,
# 5 = an EOF, read and so consumed), and a stale Ctrl-D is stale input like any other. A line
# typed but not finished stays in the terminal's line buffer, where no read can see it; it cannot
# answer the gate either -- it still needs the ENTER typed after the prompt. Bounded, because a
# hung-up terminal answers EOF forever (measured: 4096 consecutive 5s) -- a closed channel, not a
# quiet one -- and the bound fails the gate rather than spinning.
PAUSE_DRAIN_MAX=4096
drain_pause_channel() {  # empty fd 3's input queue; 1 = it would not go quiet (hung up, stuck, error)
  local n=0 _stale
  while (( n < PAUSE_DRAIN_MAX )); do
    sysread -t 0 -i 3 -s 256 _stale
    case $? in
      0|5) (( n += 1 )) ;;   # a stale line, or a stale Ctrl-D: discarded
      4)   return 0 ;;       # nothing queued: the next line is typed after the prompt
      *)   return 1 ;;       # a read error on the terminal
    esac
  done
  return 1
}

pause_between() {
  { exec 3<"$TTY" } 2>/dev/null
  [ -t 3 ] || { exec 3<&-; return 1; }
  drain_pause_channel || { exec 3<&-; return 1; }
  print -r -- "trial 1 of 2 done. Reset the start state for trial 2 of 2 (same stratum: $STRATUM, same power cycle), then press ENTER." > "$TTY"
  read -r _ <&3 || { exec 3<&-; return 1; }
  exec 3<&-
  return 0
}

FIRST="${ORDER[1]}"; SECOND="${ORDER[2]}"
START_AT="${CAPSTONE_PAIR_TRIAL:-1}"
case "$START_AT" in 1|2) ;; *) echo "ABORT: CAPSTONE_PAIR_TRIAL must be 1 or 2, got '$START_AT'"; exit 2 ;; esac
# The terminal gate runs BEFORE the pair is sealed. Sealing first made "Nothing consumed" untrue:
# a reserve pair refused here still had its declared stratum written permanently into pairs.csv,
# and the corrected re-run then aborted on "already holds this pair's rows". There are only three
# reserve pairs. Found Sep 6 by the verification pass over the parallel launcher edits -- each of
# which was correct alone, the defect existing only in their composition.
if [ "$START_AT" = 1 ]; then
  # sysread lives in zsh/system; without it the drain cannot run and the gate would fail closed
  # AFTER trial 1 had spent its start state. Knowable now, so refused now.
  zmodload zsh/system 2>/dev/null || {
    echo "ABORT: zsh/system did not load, so the pause between the two trials cannot discard keys typed"
    echo "       during trial 1 before it asks for ENTER. Nothing consumed: no trial was started,"
    echo "       and this pair is NOT sealed."
    exit 2; }
  pause_channel_is_a_terminal || {
    echo "ABORT: the pause between the two trials has to gate on a human pressing ENTER, and"
    echo "       '$TTY' is not a terminal -- a read from it cannot block. Run the pair from a"
    echo "       terminal, or point CAPSTONE_TTY at one. Nothing consumed: no trial was started,"
    echo "       and this pair is NOT sealed -- a reserve pair keeps its stratum undeclared."
    exit 2; }
fi
seal_pair_rows "$FIRST" "$TOK1" "$SECOND" "$TOK2" || exit $?
if [ "$START_AT" = 1 ]; then
  run_one 1 "$FIRST" "$TOK1" || exit $?
  pause_between || {
    echo "pair $PAIR: trial 1 of 2 is done, but the ENTER gate could not be read ('$TTY' closed, or EOF)."
    echo "TRIAL 2 NOT LAUNCHED -- its start state is unspent, and nobody confirmed the reset."
    echo "Reset the start state, then resume: CAPSTONE_PAIR_TRIAL=2 tools/run_pair.sh $PAIR"
    exit 7; }
fi
run_one 2 "$SECOND" "$TOK2" || exit $?
echo "pair $PAIR complete: trials $TOK1 / $TOK2. The key is sealed in $SEALED/ -- do not open it until scoring is done."
