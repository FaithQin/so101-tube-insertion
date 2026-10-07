"""The certification replay must produce a telemetry baseline in the SAME units and channel order
a trial does, or every contact z-score computed against it is silently wrong.

Sep 6 2026, bench morning. `tools/v4_contact_events.py:44` records the gap this file closes:
"the certification replays are run by `tools/replay_tracking.py`, which logs positions only and
does not set CAPSTONE_TELEMETRY_CSV; until the replay path writes a telemetry CSV there is no
per-block baseline to load." The contact response is a CO-PRIMARY endpoint of the frozen protocol
("the v4 protocol (text held until unblinding; SHA-256 in the README)"), so without a baseline it cannot be computed at all.

WHY A NEW TOOL RATHER THAN EXTENDING replay_tracking.py. `replay_tracking.py` drives the arm
through `tools/_arms.py`, which is in RANGE mode; production runs in DEGREES
(`SOFollowerConfig.use_degrees = True`). A baseline recorded in one unit and applied to trials
recorded in the other is exactly the "plumbing, not policy" failure this project has already paid
for once. `tools/replay_telemetry.py` therefore drives the SAME patched `SOFollower` a trial
drives, and tees its `get_observation()` through the SAME `telemetry_sidecar` a trial tees through.
These tests pin that identity: the header is read from `telemetry_sidecar.HEADER` AT CALL TIME, so
a renamed or permuted channel fails here rather than at analysis time.

Nothing in this file touches hardware: the robot is a fake, the clock is a fake, the sleep is a
fake. `main()` is the only place a bus is opened and no test executes it.
"""
import csv
import math
import re
import sys

import numpy as np
import pytest

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))

import replay_telemetry as rt  # noqa: E402
import telemetry_sidecar as ts  # noqa: E402
import v4_contact_events as ce  # noqa: E402

ACTION_NAMES = [f"{j}.pos" for j in ts.JOINTS]


class FakeRobot:
    """Records every call in order. `obs_for(i)` is the observation at tick i."""

    def __init__(self, n_ticks: int, drop_channels=()):
        self.n = n_ticks
        self.drop = set(drop_channels)
        self.calls = []          # ("obs", i) / ("act", dict) in the order they happened
        self._i = 0

    def get_observation(self):
        i = self._i
        self.calls.append(("obs", i))
        obs = {}
        for k, j in enumerate(ts.JOINTS):
            obs[f"{j}.pos"] = 100.0 + i + k
            if "load" not in self.drop:
                obs[f"{j}.load"] = 10.0 * k + (i % 7)      # varies with the tick: a real block does
            if "current" not in self.drop:
                obs[f"{j}.current"] = -5.0 * k - (i % 5)
        self._i += 1
        return obs

    def send_action(self, action):
        self.calls.append(("act", dict(action)))
        return action


def _actions(n):
    return np.array([[float(10 * i + k) for k in range(6)] for i in range(n)], dtype=np.float64)


def _read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# 1. the replay loop, through the real sidecar
# ---------------------------------------------------------------------------
def test_writes_one_row_per_action_with_the_sidecars_own_header(tmp_path):
    out = tmp_path / "cert_telemetry.csv"
    robot, sidecar = FakeRobot(5), ts.Sidecar(str(out))
    n = rt.replay_loop(robot, _actions(5), sidecar, fps=20, sleep=lambda _s: None, clock=lambda: 0.0)
    sidecar.close()
    assert n == 5
    with open(out, newline="") as fh:
        header = next(csv.reader(fh))
    # read from the sidecar AT CALL TIME: a permuted or renamed channel must fail here
    assert header == ts.HEADER
    rows = _read(out)
    assert [int(r["frame_index"]) for r in rows] == [0, 1, 2, 3, 4]
    assert float(rows[2]["elbow_flex.pos"]) == pytest.approx(100.0 + 2 + 2)
    assert float(rows[2]["gripper.load"]) == pytest.approx(10.0 * 5 + (2 % 7))


def test_the_row_holds_the_observation_taken_BEFORE_that_tick_s_action(tmp_path):
    """A recorded frame pairs the observation the policy saw with the action then taken; the
    baseline must be built from the same pairing, or it is offset by one tick."""
    out = tmp_path / "t.csv"
    robot, sidecar = FakeRobot(3), ts.Sidecar(str(out))
    rt.replay_loop(robot, _actions(3), sidecar, fps=20, sleep=lambda _s: None, clock=lambda: 0.0)
    sidecar.close()
    kinds = [c[0] for c in robot.calls]
    assert kinds == ["obs", "act"] * 3, kinds
    rows = _read(out)
    assert float(rows[0]["shoulder_pan.pos"]) == pytest.approx(100.0)   # tick 0's obs, not tick 1's


def test_actions_are_sent_under_the_dataset_s_own_joint_names_in_order(tmp_path):
    robot, sidecar = FakeRobot(3), ts.Sidecar(str(tmp_path / "t.csv"))
    rt.replay_loop(robot, _actions(3), sidecar, fps=20, sleep=lambda _s: None, clock=lambda: 0.0)
    sidecar.close()
    sent = [c[1] for c in robot.calls if c[0] == "act"]
    assert [list(a) for a in sent] == [ACTION_NAMES] * 3
    assert sent[1] == {name: float(10 * 1 + k) for k, name in enumerate(ACTION_NAMES)}


class _FakeClock:
    """A clock and a sleep that agree with each other: sleeping advances time, and each tick of
    work costs `work`, with `overshoot` extra seconds returned by every sleep — which is what
    `time.sleep` really does (it sleeps AT LEAST the requested time, plus scheduling latency)."""

    def __init__(self, work=0.01, overshoot=0.0):
        self.now, self.work, self.overshoot, self.slept = 0.0, work, overshoot, []

    def clock(self):
        self.now += self.work
        return self.now

    def sleep(self, s):
        self.slept.append(s)
        self.now += s + (self.overshoot if s > 0 else 0.0)


def test_the_loop_paces_at_the_dataset_fps(tmp_path):
    """Four ticks at 20 fps take four periods of wall clock, the work included. The per-sleep bound
    alone could not tell a paced loop from a bare `sleep(1 / fps)` (every sleep is exactly 1/20, so
    it passed); the elapsed-time assertion can. With 10 ms per clock read the deadline loop ends at
    0.21 s; a bare sleep(1/fps) ends at 4 x (0.02 + 0.05) = 0.28 s and the old relative form
    `sleep(period - (clock() - t_start))` at 0.24 s — both red (mutation-checked Sep 14)."""
    c = _FakeClock(work=0.01)
    robot, sidecar = FakeRobot(4), ts.Sidecar(str(tmp_path / "t.csv"))
    rt.replay_loop(robot, _actions(4), sidecar, fps=20, sleep=c.sleep, clock=c.clock)
    sidecar.close()
    assert len(c.slept) == 4
    assert all(0 < s <= 1 / 20 for s in c.slept), c.slept
    assert abs(c.now - 4 / 20) < 0.5 / 20, c.now


def test_the_loop_does_not_accumulate_sleep_overshoot_into_drift(tmp_path):
    """DEFECT (Sep 7 2026): `sleep(max(0.0, period - (clock() - t_start)))` restarts the clock
    every tick, so every millisecond `time.sleep` overshoots by is kept. Measured on the five
    certification / gate replays it wrote: 18.523, 18.626, 18.601, 18.568, 18.546 Hz against the
    dataset's 20 fps — a 7.4-8.0 % slow replay used as the mu/sigma baseline for trials that ran
    at 19.35-19.40 Hz. The deadline is absolute: an overshoot is absorbed by the NEXT sleep."""
    n, period = 60, 1 / 20
    c = _FakeClock(work=0.005, overshoot=0.004)      # 4 ms of scheduler latency per sleep
    robot, sidecar = FakeRobot(n), ts.Sidecar(str(tmp_path / "t.csv"))
    rt.replay_loop(robot, _actions(n), sidecar, fps=20, sleep=c.sleep, clock=c.clock)
    sidecar.close()
    hz = n / c.now
    assert 19.9 < hz <= 20.0, hz                     # the buggy loop lands at ~18.5 Hz here
    assert abs(c.now - n * period) < 2 * period


def test_a_tick_that_overruns_a_whole_period_does_not_make_the_next_ticks_sprint(tmp_path):
    """The other half of a deadline schedule: after a genuine overrun the loop must NOT fire a
    burst of zero-sleep ticks to catch up — a replay that sprints drives the arm faster than the
    demonstration it is reproducing."""
    c = _FakeClock(work=0.005)
    robot, sidecar = FakeRobot(6), ts.Sidecar(str(tmp_path / "t.csv"))

    real_clock = c.clock
    hits = {"n": 0}

    def clock():
        hits["n"] += 1
        if hits["n"] == 4:          # one tick stalls for a full second
            c.now += 1.0
        return real_clock()

    rt.replay_loop(robot, _actions(6), sidecar, fps=20, sleep=c.sleep, clock=clock)
    sidecar.close()
    assert sum(1 for s in c.slept if s == 0.0) <= 1, c.slept
    assert all(s <= 1 / 20 for s in c.slept), c.slept


def test_a_late_tick_never_sleeps_a_negative_time(tmp_path):
    slept = []
    robot, sidecar = FakeRobot(3), ts.Sidecar(str(tmp_path / "t.csv"))
    ticks = iter([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])   # every tick massively overruns
    rt.replay_loop(robot, _actions(3), sidecar, fps=20, sleep=slept.append, clock=lambda: next(ticks))
    sidecar.close()
    assert slept == [0.0, 0.0, 0.0], slept


# ---------------------------------------------------------------------------
# 2. gate, don't narrate — a baseline that would be rejected later is rejected NOW
# ---------------------------------------------------------------------------
def test_the_loop_refuses_an_observation_with_no_load_channels(tmp_path):
    """If the follower patch is not applied, get_observation carries only `.pos`, the sidecar fills
    12 NaN columns, and the CSV looks fine until analysis. Stop at the bench, where it is fixable."""
    robot, sidecar = FakeRobot(4, drop_channels=("load", "current")), ts.Sidecar(str(tmp_path / "t.csv"))
    with pytest.raises(rt.ReplayTelemetryError, match=r"(?i)load|current|patch"):
        rt.replay_loop(robot, _actions(4), sidecar, fps=20, sleep=lambda _s: None, clock=lambda: 0.0)


def test_verify_baseline_accepts_a_block_that_moves_and_reports_its_sigma(tmp_path):
    out = tmp_path / "cert_telemetry.csv"
    robot, sidecar = FakeRobot(ce.MIN_BASELINE_ROWS + 5), ts.Sidecar(str(out))
    rt.replay_loop(robot, _actions(ce.MIN_BASELINE_ROWS + 5), sidecar, fps=20,
                   sleep=lambda _s: None, clock=lambda: 0.0)
    sidecar.close()
    base = rt.verify_baseline([out])
    assert base.n_rows == ce.MIN_BASELINE_ROWS + 5
    assert len(base.std) == 18 and all(s > 0 for s in base.std[6:18])


def test_verify_baseline_refuses_a_constant_load_column(tmp_path):
    """The telemetry gate of handoff section 4: a load column that never moves means the sidecar is
    not seeing the servos. It must fail here, not silently become a zero-sigma divisor."""
    out = tmp_path / "flat.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(ts.HEADER)
        for i in range(ce.MIN_BASELINE_ROWS + 2):
            w.writerow([i, i / 20] + [float(i)] * 6 + [7.0] * 12)   # positions move, load/current do not
    with pytest.raises(ce.TelemetryError, match=r"(?i)constant"):
        rt.verify_baseline([out])


def test_verify_baseline_refuses_too_few_rows(tmp_path):
    out = tmp_path / "short.csv"
    robot, sidecar = FakeRobot(3), ts.Sidecar(str(out))
    rt.replay_loop(robot, _actions(3), sidecar, fps=20, sleep=lambda _s: None, clock=lambda: 0.0)
    sidecar.close()
    with pytest.raises(ce.TelemetryError, match=r"(?i)rows"):
        rt.verify_baseline([out])


# ---------------------------------------------------------------------------
# 3. the episode's actions come from the dataset, in frame order
# ---------------------------------------------------------------------------
def test_actions_for_episode_reads_the_parquet_in_frame_order(tmp_path):
    import pandas as pd

    root = tmp_path / "ds"
    (root / "data" / "chunk-000").mkdir(parents=True)
    rows = []
    for ep in (0, 1):
        for fi in range(4):
            rows.append({"episode_index": ep, "frame_index": fi,
                         "action": np.array([ep * 100 + fi + k for k in range(6)], dtype=np.float32)})
    pd.DataFrame(rows).sample(frac=1.0, random_state=0).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    a = rt.actions_for_episode(root, 1)
    assert a.shape == (4, 6)
    assert a[0][0] == pytest.approx(100.0) and a[3][0] == pytest.approx(103.0)   # sorted by frame_index
    with pytest.raises(rt.ReplayTelemetryError, match=r"(?i)episode"):
        rt.actions_for_episode(root, 99)


# ---------------------------------------------------------------------------
# 4. safety: nothing opens a bus on import, and the operator is told the rules
# ---------------------------------------------------------------------------
def test_importing_the_module_opens_no_bus_and_drives_nothing():
    src = (TOOLS / "replay_telemetry.py").read_text()
    assert not re.search(r"^\s*(?:from|import)\s+_arms", src, re.M), (
        "replay_telemetry imports the RANGE-mode helper; production is DEGREES and a baseline in "
        "the wrong unit is silently wrong"
    )
    body = src.split("def main(")[0]
    for forbidden in ("SOFollower(", "make_robot", ".connect(", "open_bus("):
        assert forbidden not in body, f"{forbidden} runs outside main() — importing this module would move the arm"


def test_the_docstring_states_the_operator_preconditions():
    doc = rt.__doc__ or ""
    for word in ("hands", "torque"):
        assert re.search(word, doc, re.I), f"the docstring does not tell the operator about {word}"


def test_the_replay_refuses_to_start_from_a_pose_far_from_the_first_action():
    """Closing a 100 deg gap in one command sweeps the arm through the workspace. The operator
    homes first; this is the gate that makes sure they did."""
    robot, sidecar = FakeRobot(3), ts.Sidecar(None)
    with pytest.raises(rt.ReplayTelemetryError, match=r"(?i)home|gap"):
        rt.replay_loop(robot, _actions(3), sidecar, fps=20, sleep=lambda _s: None,
                       clock=lambda: 0.0, max_start_gap_deg=15.0)
    assert [c[0] for c in robot.calls] == ["obs"], "it moved the arm before refusing"


def test_the_start_gap_is_the_largest_per_joint_distance():
    obs = {f"{j}.pos": 10.0 for j in ts.JOINTS}
    assert rt.start_gap_deg(obs, [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]) == pytest.approx(0.0)
    assert rt.start_gap_deg(obs, [10.0, 10.0, -25.0, 10.0, 10.0, 10.0]) == pytest.approx(35.0)


def test_missing_channels_names_what_the_patch_would_have_supplied():
    full = {n: 1.0 for n in ts.HEADER[2:]}
    assert rt.missing_channels(full) == []
    del full["elbow_flex.load"]
    assert rt.missing_channels(full) == ["elbow_flex.load"]


# ---------------------------------------------------------------------------
# 5. the stall guard — the protocol's chamferless stop rule, in code
# ---------------------------------------------------------------------------
class JammedRobot(FakeRobot):
    """Tracks normally until `jam_at`, then stops moving while its load climbs — a jam."""

    def __init__(self, n_ticks, jam_at, joint="elbow_flex", load=900.0):
        super().__init__(n_ticks)
        self.jam_at, self.jam_joint, self.jam_load = jam_at, joint, load

    def get_observation(self):
        i = self._i
        obs = super().get_observation()
        if i >= self.jam_at:
            obs[f"{self.jam_joint}.pos"] = 55.0                  # frozen
            obs[f"{self.jam_joint}.load"] = -self.jam_load       # pushing hard, sign-decoded
        return obs


def test_a_jam_aborts_the_episode_and_relieves_the_arm(tmp_path):
    robot = JammedRobot(40, jam_at=5)
    sidecar = ts.Sidecar(str(tmp_path / "t.csv"))
    with pytest.raises(rt.Stalled, match=r"(?i)elbow_flex.*JAM"):
        rt.replay_loop(robot, _actions(40), sidecar, fps=20, sleep=lambda _s: None, clock=lambda: 0.0)
    sidecar.close()
    # it stopped early, not at the end
    assert len([c for c in robot.calls if c[0] == "act"]) < 40
    # and the last thing it did was command every guarded joint to where it already was
    last = [c[1] for c in robot.calls if c[0] == "act"][-len(rt.GUARDED_JOINTS):]
    assert [list(a)[0] for a in last] == [f"{j}.pos" for j in rt.GUARDED_JOINTS]
    assert last[rt.GUARDED_JOINTS.index("elbow_flex")]["elbow_flex.pos"] == pytest.approx(55.0)


def test_high_load_while_still_moving_is_not_a_stall():
    """Measured today: the three clean coned replays peak at 644-675 on the arm joints during
    fast reaches. A bare load limit would abort a healthy episode."""
    assert not rt.is_stalling(675.0, moved_deg=2.0)
    assert not rt.is_stalling(900.0, moved_deg=2.0)      # moving fast under load: tracking
    assert not rt.is_stalling(300.0, moved_deg=0.0)      # still, but not pressing: settled
    assert rt.is_stalling(900.0, moved_deg=0.0)          # pressing and frozen: a jam


def test_the_threshold_sits_above_the_clean_replay_envelope():
    assert rt.STALL_LOAD > 675, "the guard would fire on a healthy replay"
    assert rt.STALL_LOAD < 1000, "the guard must fire below the servos' torque limit"
    assert rt.STALL_TICKS / 20 < 2.0, "must abort before the servos' 2.0 s Protection_Time latch"
    assert "gripper" not in rt.GUARDED_JOINTS, "the gripper has its own guard in the follower patch"


def test_the_guard_can_be_switched_off_for_the_pure_tests(tmp_path):
    robot = JammedRobot(20, jam_at=2)
    sidecar = ts.Sidecar(str(tmp_path / "t.csv"))
    n = rt.replay_loop(robot, _actions(20), sidecar, fps=20, sleep=lambda _s: None,
                       clock=lambda: 0.0, stall_guard=False)
    sidecar.close()
    assert n == 20
