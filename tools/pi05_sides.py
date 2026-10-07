#!/usr/bin/env python
"""pi0.5 side table and resolvers -- the single source of truth for run_pi05_trial.sh.

    python tools/pi05_sides.py --print repo|dataset|force|state_dim|gen|task|checkpoint|snapshot --side A|B

Mirrors `smolvla_dryrun.py --print`: the runner never types a repo id, a pin, a prompt or a
checkpoint; it asks here, so the rehearsal and the bench cannot disagree.

Sep 6 2026: until tonight the runner served exactly one policy, pi05-tube-A-v3, with
CAPSTONE_FORCE_WIDE_STATE=0 hardcoded beside it. v4 has an A and a B. Both declare
observation.state [32] (max_state_dim padding), so the live routing's "declared == max ->
widen" rule cannot tell them apart; only the pin can: A narrow (its statistics are 6-dim),
B wide (18-dim). `snapshot` returns the SANITIZED local checkpoint dir (lerobot 0.6.1 cannot
parse six 0.6.2 fields in the Hub config; tools/pi05_latency_bench.materialize_local_checkpoint
strips them and absolutises the bundled tokenizer) -- for the checkpoint CAPSTONE_CHECKPOINT
names; unset aborts for v4, exactly like the SmolVLA resolver.

Sep 6 2026, 03:xx -- the exit hang. `--print task --side A` printed the right prompt and then sat
for the full 180 s inside the pre-session suite (1 failed / 506 passed), and passed alone in 1.7 s.
Reproduced at a desk with no suite involved: 2 hangs in ~155 plain subprocess runs (~1 % per
invocation). `sample <pid>` of the stuck process, main thread:

    Py_Exit -> exit -> __cxa_finalize_ranges -> ~shared_ptr<arrow::internal::ThreadPool>
      -> ThreadPool::Shutdown -> std::condition_variable::wait

and the only other live thread is an arrow pool worker (ThreadPool::LaunchWorkersUnlocked) also
parked in condition_variable::wait. `trained_task` reads tasks.parquet through pandas, which
spawns arrow's global CPU pool; at libc exit() the pool's C++ static destructor and its worker
miss each other's wake-up. Python's own finalisation (atexit, logging.shutdown,
pyarrow's ensure_s3_finalized) has already completed by then -- the answer is on stdout, only
the exit is lost. run_pi05_trial.sh:70 captures `$(... --print task ...)` with no timeout, so on
the bench this is an indefinite stall at "resolving the prompt", before any gate speaks.

Hence `hard_exit`: flush stdout and stderr, then `os._exit` -- no interpreter teardown, no atexit,
no C++ destructors. Nothing on the --print path relies on teardown: the only files written
(`materialize_local_checkpoint`, pi05_latency_bench.py) are closed inside `with` blocks before
it returns. The flush is load-bearing: `$(...)` is a pipe, a pipe-backed stdout is block-buffered,
and a hard exit without it delivers an EMPTY value to the runner. Pinned by
tests/test_pi05_trial_runner.py (atexit never reached; every field intact through a pipe;
abort status preserved).
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pi05_latency_bench import materialize_local_checkpoint  # noqa: E402
from smolvla_dryrun import (  # noqa: E402
    CheckpointUnavailable,
    Side,
    TaskUnavailable,
    checkpoint_from_env,
    policy_gen,
    resolve_snapshot,
    trained_task,
)

SIDES_V3 = {
    "A": Side("A", "faithqin/pi05-tube-A-v3_2026-08-30_03-06-05", "faithqin/so101-tube-insert-v3-noload", "0", 6, "v3"),
}
SIDES_V4 = {
    "A": Side("A", "faithqin/pi05-tube-A-v4", "faithqin/so101-tube-insert-v4-noload", "0", 6, "v4"),
    "B": Side("B", "faithqin/pi05-tube-B-v4", "faithqin/so101-tube-insert-v4", "1", 18, "v4"),
}
GENERATIONS = {"v3": SIDES_V3, "v4": SIDES_V4}


def active_sides(env=None) -> dict:
    return GENERATIONS[policy_gen(env)]  # policy_gen validates CAPSTONE_POLICY_GEN


def sanitized_policy_dir(side: Side, checkpoint: str | None) -> str:
    """The directory the rollout loads: the chosen checkpoint, 0.6.1-parsable, one dir per repo+step."""
    snap = resolve_snapshot(side.repo_id, checkpoint)
    tag = f"{side.repo_id.replace('/', '--')}--{checkpoint or 'top'}"
    out = Path(tempfile.gettempdir()) / "pi05_local_sanitized" / tag
    d, _dropped = materialize_local_checkpoint(snap, out)
    return str(d)


PRINTABLE = ("repo", "dataset", "force", "state_dim", "gen", "task", "checkpoint", "snapshot")


def print_field(field: str, side: Side) -> str:
    if field == "repo":
        return side.repo_id
    if field == "dataset":
        return side.dataset
    if field == "force":
        return side.force
    if field == "state_dim":
        return str(side.state_dim)
    if field == "gen":
        return side.gen
    if field == "task":
        return trained_task(side.dataset)
    if field == "checkpoint":
        return checkpoint_from_env(None, side.gen) or "top"
    if field == "snapshot":
        return sanitized_policy_dir(side, checkpoint_from_env(None, side.gen))
    raise ValueError(f"unknown field {field!r}; expected one of {PRINTABLE}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", required=True, choices=["A", "B"])
    ap.add_argument("--print", dest="field", required=True, choices=PRINTABLE)
    a = ap.parse_args(argv)
    try:
        sides = active_sides()
        if a.side not in sides:
            raise ValueError(f"side {a.side} does not exist in generation {policy_gen()} (pi0.5-B is v4 only)")
        print(print_field(a.field, sides[a.side]))
    except (TaskUnavailable, CheckpointUnavailable, ValueError) as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 5
    return 0


def hard_exit(rc: int) -> None:
    """Deliver the answer, then leave WITHOUT interpreter or C++ teardown (see the module docstring).

    A flush that fails (closed pipe) means the runner did not receive the value, so the status
    goes non-zero: the runner's `|| { say abort; exit }` must fire rather than launch on nothing.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            rc = rc or 5
    os._exit(rc)


if __name__ == "__main__":
    hard_exit(main())
