#!/usr/bin/env python
"""Go / no-go before EVERY launcher or certification start. Refuses; never narrates.

    python tools/prelaunch_check.py                     # runners, port, newest certification receipt
    python tools/prelaunch_check.py --ping <file>.ping  # ...plus every servo <= 52 C

WHY (Sep 14 2026, 17:44). A pair was launched on an arm that had sat unpowered for 30 minutes -- the warm-up
rule lived in prose. The launcher was then stopped while its rollout child kept running, and a torque-off
was written under that live rollout (the lab notebook (private) 17:50). Each rule here was "known"; none was a gate.

  1. No runner process alive: launcher, runner, rollout, replay, certification -- by pid only. A rollout's
     command line names the policy, so nothing here ever prints one.
  2. The follower port is free (lsof).
  3. The newest `tools/scored_logs/CERT-*.seat` says SEAT and is <= MAX_CERT_AGE_MIN old: G-R's "a seat
     before the next trial" and the warm-up-after-a-break rule in one check.
  4. With --ping: every servo <= 52 C (cert_gate.gate_temps).

Exit 0 = GO, 1 = NO-GO with every failing reason. Tests: tests/test_prelaunch_check.py (fakes; no hardware).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cert_gate  # noqa: E402

MAX_CERT_AGE_MIN = 10
FOLLOWER_PORT = "/dev/tty.usbmodem5C4C1245641"
RUNNER_PATTERNS = ("run_pair.sh", "run_scored_trial.sh", "run_smolvla_trial.sh", "run_pi05_trial.sh",
                   "rollout_30hz_stale_ok.py", "replay_telemetry.py", "run_cert.sh", "home_arm.py")
DEFAULT_LOGS = Path(__file__).resolve().parent / "scored_logs"


def runner_pids() -> list[int]:
    """PIDs of bench processes, by pattern. `pgrep -f` prints pids only -- never `-l`."""
    pids: set[int] = set()
    for pat in RUNNER_PATTERNS:
        try:
            out = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.TimeoutExpired):
            continue
        pids.update(int(x) for x in out.split() if x.strip().isdigit())
    return sorted(pids)


def port_holders(port: str = FOLLOWER_PORT) -> list[int]:
    try:
        out = subprocess.run(["lsof", "-t", port], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [int(x) for x in out.split() if x.strip().isdigit()]


def check_certification(logs_dir, now: float | None = None, max_age_min: int = MAX_CERT_AGE_MIN) -> tuple[bool, str]:
    now = time.time() if now is None else now
    receipts = sorted(Path(logs_dir).glob("CERT-*.seat"), key=lambda p: p.stat().st_mtime)
    if not receipts:
        return False, "no certification receipt (CERT-*.seat) on disk — run tools/run_cert.sh and record a SEAT first"
    newest = receipts[-1]
    kv = dict(ln.split("=", 1) for ln in newest.read_text().strip().splitlines() if "=" in ln)
    verdict = kv.get("verdict", "?")
    age_min = (now - newest.stat().st_mtime) / 60.0
    if verdict != "SEAT":
        return False, f"newest certification {newest.stem} is {verdict}, not SEAT — replay until one seats (G-R)"
    if age_min > max_age_min:
        return False, (f"newest certification {newest.stem} seated {age_min:.0f} min ago > {max_age_min} min — "
                       f"the arm has cooled; run a warm-up replay (tools/run_cert.sh) and record it")
    return True, f"certification OK: {newest.stem} SEAT {age_min:.0f} min ago"


def run(logs_dir=DEFAULT_LOGS, ping_text: str | None = None, port: str = FOLLOWER_PORT) -> tuple[int, str]:
    lines, ok = [], True
    pids = runner_pids()
    if pids:
        ok = False
        lines.append(f"NO-GO: bench process(es) alive, pid {', '.join(map(str, pids))} — stop the whole tree first")
    holders = port_holders(port)
    if holders:
        ok = False
        lines.append(f"NO-GO: {port} is held by pid {', '.join(map(str, holders))}")
    c_ok, c_why = check_certification(logs_dir)
    lines.append(("OK: " if c_ok else "NO-GO: ") + c_why)
    ok = ok and c_ok
    if ping_text is not None:
        t_ok, t_why = cert_gate.gate_temps(ping_text)
        lines.append(("OK: " if t_ok else "NO-GO: ") + t_why)
        ok = ok and t_ok
    lines.append("GO" if ok else "NO-GO — fix every line above before launching")
    return (0 if ok else 1), "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ping", default=None, help="a ping_motors.py capture to gate temperatures (<= 52 C)")
    ap.add_argument("--logs", default=str(DEFAULT_LOGS))
    a = ap.parse_args(argv)
    ping_text = Path(a.ping).read_text() if a.ping else None
    rc, out = run(logs_dir=a.logs, ping_text=ping_text)
    print(out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
