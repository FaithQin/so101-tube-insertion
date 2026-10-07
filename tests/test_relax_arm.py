"""`tools/relax_arm.py` cuts torque on the follower even when a servo is missing the bus handshake.

Sep 14 2026, 12:47 (the lab notebook (private)): after three certification replays the arm was left holding for
about 15 minutes; wrist_flex reached 70 C (past the 69 C brownout on the ledger) and missed the
handshake. The only torque-off path in the diagnostics, `tools/_arms.open_bus` -> `bus.connect()`,
then FAILED with "Missing motor IDs: 4" -- it needs every servo to answer, which is exactly what a
browning-out servo does not do, at exactly the moment torque has to come off. What worked, by hand:
`FeetechMotorsBus(...).connect(handshake=False)` then `disable_torque(motor, num_retry=5)`,
wrist_flex first, then a readback of Torque_Enable on all six. This file pins that recipe so the
next thermal event is one command, not a chat.

Everything runs against a fake bus: no port is opened. The bench check is the arm (memory:
"for anything physical, the test is the arm").
"""
import ast
import sys

import pytest

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))

import relax_arm  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class FakeBus:
    """Records what relax_arm asks of it. `stuck` joints keep Torque_Enable=1 whatever is written."""

    def __init__(self, stuck=()):
        self.stuck = set(stuck)
        self.torque = {j: 1 for j in JOINTS}
        self.temp = {j: 55 for j in JOINTS}
        self.current = {j: 40 for j in JOINTS}
        self.disable_calls = []          # (joint, num_retry) in call order
        self.connect_handshake = None
        self.disconnect_calls = []
        self.connected = False

    def connect(self, handshake=True):
        self.connect_handshake = handshake
        self.connected = True

    @property
    def is_connected(self):
        return self.connected

    def disconnect(self, disable_torque=True):
        self.disconnect_calls.append(disable_torque)
        self.connected = False

    def disable_torque(self, motors=None, num_retry=0):
        self.disable_calls.append((motors, num_retry))
        if motors not in self.stuck:
            self.torque[motors] = 0
            self.current[motors] = 0

    def read(self, data_name, motor, *, normalize=True, num_retry=0):
        return {"Torque_Enable": self.torque, "Present_Current": self.current,
                "Present_Temperature": self.temp}[data_name][motor]


def test_wrist_flex_is_cut_first_then_every_other_joint():
    bus = FakeBus()
    relax_arm.relax(bus)
    order = [j for j, _ in bus.disable_calls]
    assert order[0] == "wrist_flex", "the joint that browned out on Sep 14 comes off first"
    assert sorted(order) == sorted(JOINTS) and len(order) == 6
    assert all(v == 0 for v in bus.torque.values())


def test_every_torque_write_asks_for_retries():
    bus = FakeBus()
    relax_arm.relax(bus)
    retries = {n for _, n in bus.disable_calls}
    assert retries == {relax_arm.NUM_RETRY}
    assert relax_arm.NUM_RETRY >= 5, "a servo mid-brownout drops replies; one write is not a cut"


def test_readback_refuses_a_joint_whose_torque_stayed_on():
    bus = FakeBus(stuck={"elbow_flex"})
    with pytest.raises(relax_arm.RelaxFailed, match="elbow_flex"):
        relax_arm.relax(bus)


def test_main_connects_without_the_handshake_and_leaves_torque_off(monkeypatch, capsys):
    made = []

    def factory(**kw):
        bus = FakeBus()
        made.append(bus)
        return bus

    monkeypatch.setattr(relax_arm, "FeetechMotorsBus", factory)
    rc = relax_arm.main([])
    out = capsys.readouterr().out
    assert rc == 0
    bus = made[0]
    assert bus.connect_handshake is False, "connect() with the handshake is the path that failed at 12:47"
    assert bus.disconnect_calls == [False], "disconnect must not re-walk the bus with num_retry=0"
    assert "RELAXED" in out
    assert out.count("torque 0") == 6


def test_main_reports_failure_with_exit_1(monkeypatch, capsys):
    monkeypatch.setattr(relax_arm, "FeetechMotorsBus", lambda **kw: FakeBus(stuck={"wrist_flex"}))
    rc = relax_arm.main([])
    out = capsys.readouterr().out
    assert rc == 1
    assert "RELAX FAILED" in out and "wrist_flex" in out


def test_source_never_calls_the_handshake_path():
    """`_arms.open_bus` is the path that raised 'Missing motor IDs: 4'. AST, not a text scan."""
    tree = ast.parse((TOOLS / "relax_arm.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    names = set()
    for c in calls:
        f = c.func
        names.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None))
    assert "open_bus" not in names
    # and the connect it does make is explicitly handshake=False
    connects = [c for c in calls if isinstance(c.func, ast.Attribute) and c.func.attr == "connect"]
    assert connects, "main() must call bus.connect(...)"
    for c in connects:
        kw = {k.arg: k.value for k in c.keywords}
        assert "handshake" in kw and isinstance(kw["handshake"], ast.Constant) and kw["handshake"].value is False


# Sep 14 2026, 17:45: relax_arm was run while an orphaned rollout still held the follower port. macOS let
# the second open through; the torque-off writes landed under a live policy and the arm went limp and
# jerky mid-episode (the lab notebook (private), Sep 14 17:50). The tool must refuse, by pid, before opening anything.
def test_refuses_when_another_process_holds_the_port(monkeypatch, capsys):
    monkeypatch.setattr(relax_arm, "port_holders", lambda port: [17669])
    made = []
    monkeypatch.setattr(relax_arm, "FeetechMotorsBus", lambda **kw: made.append(kw) or FakeBus())
    rc = relax_arm.main([])
    out = capsys.readouterr().out
    assert rc == 2 and not made, "the bus must never be constructed while the port is held"
    assert "17669" in out and "REFUSED" in out


def test_proceeds_when_the_port_is_free(monkeypatch):
    monkeypatch.setattr(relax_arm, "port_holders", lambda port: [])
    monkeypatch.setattr(relax_arm, "FeetechMotorsBus", lambda **kw: FakeBus())
    assert relax_arm.main([]) == 0
