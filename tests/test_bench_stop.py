"""`tools/bench_stop.py` -- stop EVERY bench process (launcher, runner, rollout, replay, certification) by
pid, wait until none is left, and only then relax the arm.

Sep 14 2026, 17:45: `pkill` on the launcher and runner shells left the rollout python running as an
orphan; `relax_arm.py` then wrote torque-off under it and the arm went limp mid-episode. The stop has
to be the whole tree, verified empty, before any bus command. Nothing here prints a command line.
"""
import sys

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))

import bench_stop  # noqa: E402


class FakeProcs:
    def __init__(self, pids, stubborn=()):
        self.alive = set(pids); self.stubborn = set(stubborn)
        self.term, self.kill9 = [], []
    def pids(self):
        return sorted(self.alive)
    def terminate(self, pid):
        self.term.append(pid)
        if pid not in self.stubborn:
            self.alive.discard(pid)
    def kill(self, pid):
        self.kill9.append(pid); self.alive.discard(pid)


def test_stops_the_whole_tree_then_relaxes(monkeypatch):
    fp = FakeProcs([100, 200, 300])
    relaxed = []
    monkeypatch.setattr(bench_stop, "runner_pids", fp.pids)
    monkeypatch.setattr(bench_stop, "_terminate", fp.terminate)
    monkeypatch.setattr(bench_stop, "_kill", fp.kill)
    monkeypatch.setattr(bench_stop, "_relax", lambda: relaxed.append(True) or 0)
    monkeypatch.setattr(bench_stop.time, "sleep", lambda s: None)
    rc, out = bench_stop.run()
    assert rc == 0 and sorted(fp.term) == [100, 200, 300] and relaxed == [True]
    assert "faithqin" not in out and "tube-" not in out


def test_escalates_to_kill_and_never_relaxes_while_something_lives(monkeypatch):
    fp = FakeProcs([100, 200], stubborn={200})
    order = []
    monkeypatch.setattr(bench_stop, "runner_pids", lambda: (order.append("pids"), fp.pids())[1])
    monkeypatch.setattr(bench_stop, "_terminate", fp.terminate)
    monkeypatch.setattr(bench_stop, "_kill", lambda pid: (order.append(("kill", pid)), fp.kill(pid)))
    monkeypatch.setattr(bench_stop, "_relax", lambda: (order.append("relax"), 0)[1])
    monkeypatch.setattr(bench_stop.time, "sleep", lambda s: None)
    rc, out = bench_stop.run()
    assert rc == 0 and ("kill", 200) in order
    assert order.index(("kill", 200)) < order.index("relax"), "relax only after the tree is empty"


def test_refuses_to_relax_if_a_process_survives_kill(monkeypatch):
    class Immortal(FakeProcs):
        def kill(self, pid): self.kill9.append(pid)          # never dies
    fp = Immortal([100], stubborn={100})
    relaxed = []
    monkeypatch.setattr(bench_stop, "runner_pids", fp.pids)
    monkeypatch.setattr(bench_stop, "_terminate", fp.terminate)
    monkeypatch.setattr(bench_stop, "_kill", fp.kill)
    monkeypatch.setattr(bench_stop, "_relax", lambda: relaxed.append(True) or 0)
    monkeypatch.setattr(bench_stop.time, "sleep", lambda s: None)
    rc, out = bench_stop.run()
    assert rc == 1 and not relaxed and "100" in out
