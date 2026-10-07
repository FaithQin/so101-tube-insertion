"""Measure telemetry read costs on the real SO-101 bus: how much do extra
sync_read transactions cost, and what does the one-transaction block read
(tools/feetech_block_read.py) buy?

Paths measured (N iterations each, after warmup):
  pos1    position only                      (1 transaction)  — today's default
  patch3  position + load + current          (3 transactions) — the local capstone patch
  pr4     position + velocity + load + temp  (4 transactions) — upstream PR #3456 shape
  block1  registers 56-70 in ONE transaction — this proposal

Also verifies value equivalence: block-read raw values must equal the
per-register sync_read raw values taken back-to-back on an idle arm.

    conda activate lerobot
    python bench_block_read.py            # follower, N=200

Read-only; torque untouched. Needs the serial port to itself.
"""

import statistics
import time

from _arms import FOLLOWER_CAL, FOLLOWER_PORT, open_bus
from feetech_block_read import FIELDS, read_telemetry_block

N = 200
REGS3 = ["Present_Position", "Present_Load", "Present_Current"]
REGS4 = ["Present_Position", "Present_Velocity", "Present_Load", "Present_Temperature"]
FIELD_TO_REG = {"position": "Present_Position", "velocity": "Present_Velocity", "load": "Present_Load",
                "voltage": "Present_Voltage", "temperature": "Present_Temperature", "current": "Present_Current"}


def timed(fn, n=N, warmup=10):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return ts


def stats(ts):
    qs = statistics.quantiles(ts, n=100)
    return f"mean {statistics.mean(ts):6.2f} ms | p50 {statistics.median(ts):6.2f} | p99 {qs[98]:6.2f}"


def main():
    bus = open_bus(FOLLOWER_PORT, FOLLOWER_CAL)
    try:
        paths = {
            "pos1  (1 txn)": lambda: bus.sync_read("Present_Position", normalize=False),
            "patch3 (3 txn)": lambda: [bus.sync_read(r, normalize=False) for r in REGS3],
            "pr4   (4 txn)": lambda: [bus.sync_read(r, normalize=False) for r in REGS4],
            "block1 (1 txn)": lambda: read_telemetry_block(bus),
        }
        results = {}
        for name, fn in paths.items():
            results[name] = timed(fn)
            print(f"{name}: {stats(results[name])}")
        base = statistics.mean(results["pos1  (1 txn)"])
        blk = statistics.mean(results["block1 (1 txn)"])
        p3 = statistics.mean(results["patch3 (3 txn)"])
        p4 = statistics.mean(results["pr4   (4 txn)"])
        print(f"\nblock read overhead vs position-only: +{blk - base:.2f} ms "
              f"({blk / base:.2f}x); vs 3-txn saves {p3 - blk:.2f} ms; vs 4-txn saves {p4 - blk:.2f} ms")
        # equivalence: raw block values vs raw per-register reads, back to back
        blkv = read_telemetry_block(bus)
        mism = 0
        for field, reg in FIELD_TO_REG.items():
            ind = bus.sync_read(reg, normalize=False)
            for m, v in ind.items():
                if abs(int(blkv[m][field]) - int(v)) > 2:  # idle arm: allow ±2 counts jitter
                    print(f"MISMATCH {m}.{field}: block={blkv[m][field]} individual={v}")
                    mism += 1
        print("equivalence check:", "PASS (all fields match per-register reads)" if mism == 0 else f"{mism} mismatches")
    finally:
        bus.disconnect()


if __name__ == "__main__":
    main()
