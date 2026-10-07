"""Every scored trial wall-clocks its own control loop and writes the rate into its receipt.

Handoff Sep 6 §3 item 7 / §9: "The 13.4 Hz / 17 Hz control-rate figures stay barred until
wall-clocked". The wrapper stamps every recorded frame (tools/loop_timer.LoopTimer, fed from the
same add_frame hook as the telemetry sidecar), prints one `[loop] loop_hz=...` line at exit into
${RUN}.log, and each runner copies `loop_hz=` into the `.verify` receipt -- checked by RUNNING
the runners against a stub rollout that prints the line (harness: tests/test_v4_start_pose.py).
"""
import math
import re
import sys

import pytest

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))

import loop_timer  # noqa: E402
from test_v4_start_pose import RUNNERS, run_runner  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_rate_is_intervals_over_the_wall_clock_span():
    c = Clock()
    lt = loop_timer.LoopTimer(clock=c)
    for _ in range(901):            # 900 intervals of 50 ms = 45 s -> 20 Hz
        lt.frame_recorded()
        c.t += 0.05
    assert lt.frames == 901
    assert lt.wall_s() == pytest.approx(45.0)
    assert lt.rate_hz() == pytest.approx(20.0)
    assert lt.line() == "[loop] loop_hz=20.00 frames=901 wall_s=45.00"


def test_fewer_than_two_frames_is_no_rate_not_a_zero():
    lt = loop_timer.LoopTimer(clock=Clock())
    assert lt.rate_hz() is None and lt.line() == "[loop] loop_hz=nan frames=0 wall_s=0.00"
    lt.frame_recorded()
    assert lt.rate_hz() is None and "frames=1" in lt.line()


def test_parse_loop_hz_reads_the_receipt_back_and_distinguishes_missing_from_nan():
    assert loop_timer.parse_loop_hz("x\n[loop] loop_hz=19.73 frames=890 wall_s=45.06\n") == pytest.approx(19.73)
    assert loop_timer.parse_loop_hz("no such line") is None
    assert math.isnan(loop_timer.parse_loop_hz("[loop] loop_hz=nan frames=1 wall_s=0.00"))


def test_wrapper_feeds_the_timer_from_add_frame_and_reports_at_exit():
    src = (TOOLS / "rollout_30hz_stale_ok.py").read_text()
    body = re.search(r"def _add_frame_timed\(self, frame\):(.*?)\nLeRobotDataset\.add_frame = _add_frame_timed", src, re.S)
    assert body, "cannot find _add_frame_timed"
    assert re.search(r"_loop\.frame_recorded\(\)", body.group(1)), "add_frame does not stamp the loop timer"
    assert re.search(r"_atexit\.register\(_loop\.report\)", src), "the loop line is never printed at exit"


@pytest.mark.parametrize("family", sorted(RUNNERS))
def test_every_runner_copies_loop_hz_from_the_log_into_the_receipt(family):
    r = run_runner(family)
    assert r.launched, r.stdout[-600:]
    m = re.search(r"^loop_hz=(\S*)$", r.receipt, re.M)
    assert m, f"{family}: no loop_hz line in the receipt:\n{r.receipt}"
    assert m.group(1) == "19.73", f"{family}: receipt carries loop_hz={m.group(1)!r}, the stub rollout logged 19.73"
