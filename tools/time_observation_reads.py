#!/usr/bin/env python
"""Time the servo-bus reads that the load-sensing patch adds to get_observation().

READ-ONLY diagnostic. Answers: do the two extra sync_reads (Present_Load,
Present_Current) fit inside the 33.3 ms budget of a 30 fps record loop?

The patch's marginal cost is bus reads only — camera time is unchanged — so this
times the bus directly and needs no cameras, no calibration, and no TCC grant.

Usage (12V on first, then USB, arms idle):
    python tools/time_observation_reads.py
"""

import statistics
import time

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

PORT = "/dev/tty.usbmodem5C4C1245641"  # follower (WHITE)
N = 200
REGISTERS = ("Present_Position", "Present_Load", "Present_Current")
FPS_BUDGET_MS = 1000 / 30

robot = SOFollower(SOFollowerRobotConfig(port=PORT))
robot.bus.connect()
try:
    results = {}
    for reg in REGISTERS:
        robot.bus.sync_read(reg, normalize=False, num_retry=2)  # warm-up
        times_ms = []
        for _ in range(N):
            t0 = time.perf_counter()
            robot.bus.sync_read(reg, normalize=False, num_retry=2)
            times_ms.append((time.perf_counter() - t0) * 1e3)
        results[reg] = times_ms
        mean = statistics.mean(times_ms)
        p95 = statistics.quantiles(times_ms, n=20)[-1]
        worst = max(times_ms)
        print(f"{reg:20s}  mean {mean:5.2f} ms   p95 {p95:5.2f} ms   max {worst:5.2f} ms")

    patch_cost = statistics.mean(results["Present_Load"]) + statistics.mean(
        results["Present_Current"]
    )
    print(f"\npatch marginal cost (load + current): {patch_cost:.2f} ms per tick")
    print(f"30 fps budget: {FPS_BUDGET_MS:.1f} ms — patch uses {100 * patch_cost / FPS_BUDGET_MS:.0f}%")
    if patch_cost > 10:
        print("⚠️  >10 ms: consider dropping Present_Current, or the batched-read path.")
finally:
    robot.bus.disconnect()
