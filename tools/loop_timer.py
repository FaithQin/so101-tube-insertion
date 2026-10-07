"""Wall-clock rate of the recorded control loop, printed once at exit as a receipt line.

    [loop] loop_hz=19.73 frames=890 wall_s=45.06

Why (Sep 6 2026, handoff §3 item 7): the 13.4 Hz / 17 Hz control-rate figures in the notes were
never wall-clocked -- they were inferred from per-stage timings -- and stay BARRED from the
write-up until a trial reports its own rate. This is that measurement: the first and last
recorded frame's monotonic clock and the count between them; rate = (frames - 1) / span.
tools/rollout_30hz_stale_ok.py feeds it from `_add_frame_timed` (one call per recorded frame,
the same hook the telemetry sidecar uses) and registers `report` with atexit, so the line lands
in ${RUN}.log and the runners copy `loop_hz=` into the `.verify` receipt. Nothing here touches
the policy path; a clock is injected for the tests.
"""
from __future__ import annotations

import math
import sys
import time


class LoopTimer:
    def __init__(self, clock=time.perf_counter):
        self._clock = clock
        self.frames = 0
        self.t_first: float | None = None
        self.t_last: float | None = None

    def frame_recorded(self) -> None:
        now = self._clock()
        if self.t_first is None:
            self.t_first = now
        self.t_last = now
        self.frames += 1

    def wall_s(self) -> float:
        if self.t_first is None or self.t_last is None:
            return 0.0
        return self.t_last - self.t_first

    def rate_hz(self) -> float | None:
        """(frames - 1) intervals over the wall-clock span; None below two frames."""
        span = self.wall_s()
        if self.frames < 2 or span <= 0:
            return None
        return (self.frames - 1) / span

    def line(self) -> str:
        hz = self.rate_hz()
        return f"[loop] loop_hz={'nan' if hz is None else f'{hz:.2f}'} frames={self.frames} wall_s={self.wall_s():.2f}"

    def report(self, stream=None) -> None:
        print(self.line(), file=stream or sys.stderr, flush=True)


def parse_loop_hz(log_text: str) -> float | None:
    """The receipt's number back out of a log; None (not 0) when the line never printed."""
    import re

    m = re.search(r"loop_hz=([0-9.]+|nan)", log_text)
    if not m or m.group(1) == "nan":
        return None if not m else math.nan
    return float(m.group(1))
