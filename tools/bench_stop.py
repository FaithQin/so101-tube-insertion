#!/usr/bin/env python
"""Stop the WHOLE bench process tree, verify it is gone, then relax the arm.

    python tools/bench_stop.py

WHY (Sep 14 2026, 17:45). `pkill -f run_pair.sh` and `pkill -f run_scored_trial.sh` killed the shells and
left the rollout python running as an orphan; `tools/relax_arm.py` then opened the port under it and
wrote torque-off, and the arm went limp and jerky mid-episode. Order matters: every launcher, runner,
rollout, replay and certification process (by pid, from `prelaunch_check.runner_pids`) gets SIGTERM,
then SIGKILL for whatever survives, and the arm is relaxed only when nothing is left. A process that
survives SIGKILL blocks the relax and is named by pid -- never by command line (it names the policy).
An aborted trial is still an aborted trial: log it in the bench record and re-draw if needed.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prelaunch_check import runner_pids  # noqa: E402

PY = "/opt/homebrew/Caskroom/miniforge/base/envs/lerobot/bin/python"


def _terminate(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def _kill(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _relax() -> int:
    return subprocess.run([PY, str(Path(__file__).resolve().parent / "relax_arm.py")]).returncode


def run() -> tuple[int, str]:
    lines = []
    pids = runner_pids()
    if not pids:
        lines.append("no bench process alive")
    else:
        lines.append(f"stopping pid {', '.join(map(str, pids))} (SIGTERM)")
        for pid in pids:
            _terminate(pid)
        time.sleep(3)
        left = runner_pids()
        if left:
            lines.append(f"still alive after SIGTERM: pid {', '.join(map(str, left))} -> SIGKILL")
            for pid in left:
                _kill(pid)
            time.sleep(2)
        left = runner_pids()
        if left:
            lines.append(f"REFUSED to relax: pid {', '.join(map(str, left))} survived SIGKILL — the port may still be held")
            return 1, "\n".join(lines)
        lines.append("tree stopped")
    rc = _relax()
    lines.append("arm relaxed" if rc == 0 else f"relax_arm exited {rc} — check the arm before anything else")
    return (0 if rc == 0 else 1), "\n".join(lines)


def main(argv=None) -> int:
    rc, out = run()
    print(out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
