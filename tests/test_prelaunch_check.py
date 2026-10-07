"""`tools/prelaunch_check.py` -- the go/no-go the chat runs before EVERY launcher or certification start.

Sep 14 2026, 17:44: a pair was launched on an arm that had sat unpowered for 30 minutes (the warm-up rule
lived in prose), then the launcher was stopped while its rollout child kept running, and a torque-off was
written under that live rollout. Three rules that were "known" and not gated. This tool makes them refuse:

  1. no runner process alive (launcher, runner, rollout, replay, certification) -- checked by pid, never by
     printing a command line (a rollout's argv names the policy);
  2. the follower port is free (lsof);
  3. the newest certification receipt (`tools/scored_logs/CERT-*.seat`) says SEAT and is no older than
     MAX_CERT_AGE_MIN -- the warm-up rule after a break, and G-R's "a seat before the next trial";
  4. (when a ping file is given) every servo <= 52 C, via cert_gate.gate_temps.

Pure functions take their inputs; main() gathers them. Fake everything here; the bench check is the arm.
"""
import sys
import time

import pytest

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))

import prelaunch_check as pc  # noqa: E402


def _seat(tmp_path, name, verdict, age_min):
    p = tmp_path / f"{name}.seat"
    p.write_text(f"label={name.split('_')[0]}\nstamp={name.split('_',1)[1]}\nverdict={verdict}\n")
    t = time.time() - age_min * 60
    import os
    os.utime(p, (t, t))
    return p


def test_max_cert_age_is_the_warm_up_rule():
    assert pc.MAX_CERT_AGE_MIN == 10


def test_newest_seat_receipt_must_be_a_recent_seat(tmp_path):
    _seat(tmp_path, "CERT-B3-07_20260914_154526", "SEAT", age_min=40)
    _seat(tmp_path, "CERT-B3-08_20260914_155428", "MISS", age_min=30)
    ok, why = pc.check_certification(tmp_path, now=time.time())
    assert not ok and "MISS" in why                       # newest receipt (by mtime) is a miss
    _seat(tmp_path, "CERT-B3-09_20260914_155600", "SEAT", age_min=12)
    ok, why = pc.check_certification(tmp_path, now=time.time())
    assert not ok and "12" in why and "10" in why         # a seat, but stale: warm-up needed
    _seat(tmp_path, "CERT-B3-10_20260914_155700", "SEAT", age_min=2)
    ok, why = pc.check_certification(tmp_path, now=time.time())
    assert ok, why


def test_no_receipts_at_all_refuses(tmp_path):
    ok, why = pc.check_certification(tmp_path, now=time.time())
    assert not ok and "no certification" in why.lower()


def test_live_runner_or_held_port_refuses(monkeypatch, tmp_path):
    _seat(tmp_path, "CERT-X_20260914_170000", "SEAT", age_min=1)
    monkeypatch.setattr(pc, "runner_pids", lambda: [17669])
    monkeypatch.setattr(pc, "port_holders", lambda port: [])
    rc, out = pc.run(logs_dir=tmp_path, ping_text=None)
    assert rc == 1 and "17669" in out and "faithqin" not in out and "tube-" not in out
    monkeypatch.setattr(pc, "runner_pids", lambda: [])
    monkeypatch.setattr(pc, "port_holders", lambda port: [4242])
    rc, out = pc.run(logs_dir=tmp_path, ping_text=None)
    assert rc == 1 and "4242" in out


def test_hot_servo_refuses_and_all_clear_passes(monkeypatch, tmp_path):
    _seat(tmp_path, "CERT-X_20260914_170000", "SEAT", age_min=1)
    monkeypatch.setattr(pc, "runner_pids", lambda: [])
    monkeypatch.setattr(pc, "port_holders", lambda port: [])
    hot = "  id 4 wrist_flex      pos 2946  load 40  cur 1   57C  123  torque 1  ok\n" + "".join(
        f"  id {i} {j:15s} pos 0  load 0  cur 0   45C  124  torque 0  ok\n"
        for i, j in ((1, "shoulder_pan"), (2, "shoulder_lift"), (3, "elbow_flex"), (5, "wrist_roll"), (6, "gripper")))
    rc, out = pc.run(logs_dir=tmp_path, ping_text=hot)
    assert rc == 1 and "wrist_flex" in out
    rc, out = pc.run(logs_dir=tmp_path, ping_text=hot.replace("57C", "50C"))
    assert rc == 0 and "GO" in out
