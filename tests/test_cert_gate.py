"""`tools/cert_gate.py` -- the two checks a certification replay must pass BEFORE the arm is torqued,
and the receipt that records Faith's seat call.

Sep 14 2026. Certification replays run outside `preflight.py` (the lab notebook (private) 12:47, gap 2), so the
pre-session 52 C rule and the "scene-check before every replay" rule (12:37) were chat discipline.
The one miss this morning was the pass with no scene check; the 70 C event was an arm left holding.
This module makes both refusals code: `temps` refuses a replay when any servo reads above
CERT_TEMP_MAX_C or cannot be read; `home` refuses a home that WARNED or printed no settled line;
`receipt` writes the seat verdict and the pass's gate readings to one file, so "the seat is logged"
(protocol :291) is a file, not a chat line. Reuses preflight's parsers; no hardware.
"""
import sys

import pytest

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))

import cert_gate  # noqa: E402
import preflight  # noqa: E402

PING_OK = """
=== FOLLOWER (white, jaws)  /dev/tty.usbmodem5C4C1245641 ===
  broadcast: nothing responded
  id 1 shoulder_pan    pos  2014  load    44  cur    0   42C  125  torque 1  ok
  id 2 shoulder_lift   pos   778  load   -52  cur    0   44C  124  torque 0  ok
  id 3 elbow_flex      pos  3031  load     0  cur    0   41C  126  torque 1  ok
  id 4 wrist_flex      pos  2946  load    40  cur    1   52C  123  torque 1  ok
  id 5 wrist_roll      pos  2039  load    72  cur    3   45C  124  torque 1  ok
  id 6 gripper         pos  2062  load     0  cur    0   44C  125  torque 1  ok
  all six motors present and clean.
"""
HOME_OK = """homing to v4 training pose
current: {'shoulder_pan': 1.5}
residuals: {'shoulder_pan': -0.3, 'shoulder_lift': 5.7, 'elbow_flex': 0.0, 'wrist_flex': -0.2, 'wrist_roll': -0.3, 'gripper': -0.1}
HOMED (worst adjusted residual 1.2)
settled: {'shoulder_pan': 1.3, 'shoulder_lift': -99.6, 'elbow_flex': 85.6, 'wrist_flex': 73.5, 'wrist_roll': -0.4, 'gripper': 1.0}
"""
SCENE_OK = """homography inliers: 345/400
tube (blue cap)   : ref=( 360.0, 355.8)  live=( 359.1, 354.3)  delta=( -1.0, -1.6) px  |d|=1.8
rack (orange funnel): ref=( 537.4, 481.0)  live=( 538.0, 482.4)  delta=( +0.7, +1.5) px  |d|=1.6
tube angle        : ref= +0.0°  live= -0.4°  delta= -0.4°
"""


def test_cert_temp_limit_is_the_pre_session_rule_not_preflights():
    assert cert_gate.CERT_TEMP_MAX_C == 52
    assert cert_gate.CERT_TEMP_MAX_C < preflight.JOINT_TEMP_MAX_C


def test_temps_pass_at_the_limit_and_refuse_one_degree_over():
    ok, why = cert_gate.gate_temps(PING_OK)
    assert ok, why
    hot = PING_OK.replace("52C", "53C")
    ok, why = cert_gate.gate_temps(hot)
    assert not ok and "wrist_flex" in why and "53" in why


def test_temps_refuse_an_unread_servo():
    lines = [ln for ln in PING_OK.splitlines() if "elbow_flex" not in ln]
    ok, why = cert_gate.gate_temps("\n".join(lines))
    assert not ok and "elbow_flex" in why


def test_temps_refuse_both_arms_captured():
    ok, why = cert_gate.gate_temps(PING_OK + PING_OK)
    assert not ok and "twice" in why


def test_home_requires_homed_and_a_settled_line_and_no_warning():
    assert cert_gate.gate_home(HOME_OK)[0]
    warned = HOME_OK.replace("HOMED (worst adjusted residual 1.2)", "WARNING: adjusted residual 3.1 exceeds 2.5 — check for obstruction")
    ok, why = cert_gate.gate_home(warned)
    assert not ok and "WARNING" in why
    no_settled = "\n".join(ln for ln in HOME_OK.splitlines() if not ln.startswith("settled:"))
    ok, why = cert_gate.gate_home(no_settled)
    assert not ok and "settled" in why


def test_receipt_records_the_verdict_and_the_gate_readings():
    r = cert_gate.receipt("CERT-B3-02", "20260914_141500", "SEAT", SCENE_OK, PING_OK,
                          "CERT-B3-02  loop 19.99 Hz  ee_proxy 29.93  jaw_min 13.42 (held)")
    kv = dict(ln.split("=", 1) for ln in r.strip().splitlines())
    assert kv["label"] == "CERT-B3-02" and kv["stamp"] == "20260914_141500" and kv["verdict"] == "SEAT"
    assert float(kv["tube_px"]) == 1.8 and float(kv["rack_px"]) == 1.6 and float(kv["angle_deg"]) == -0.4
    assert kv["hottest_joint"] == "wrist_flex" and int(kv["hottest_c"]) == 52
    assert kv["loop_hz"] == "19.99" and kv["ee_proxy"] == "29.93" and kv["jaw_min"] == "13.42"


def test_receipt_refuses_a_verdict_that_is_not_seat_or_miss():
    with pytest.raises(ValueError, match="verdict"):
        cert_gate.receipt("X", "s", "seated", SCENE_OK, PING_OK, "X loop 20.00 Hz ee_proxy 1.00 jaw_min 1.00 (held)")


def test_cli_exit_codes(tmp_path, capsys):
    ping = tmp_path / "p.ping"; ping.write_text(PING_OK)
    assert cert_gate.main(["temps", str(ping)]) == 0
    ping.write_text(PING_OK.replace("52C", "61C"))
    assert cert_gate.main(["temps", str(ping)]) == 1
    assert "wrist_flex" in capsys.readouterr().out
    home = tmp_path / "h.home"; home.write_text(HOME_OK)
    assert cert_gate.main(["home", str(home)]) == 0
    home.write_text("nothing")
    assert cert_gate.main(["home", str(home)]) == 1


def _run_files(tmp_path, n_rows=203):
    import csv
    import telemetry_sidecar as ts
    run = tmp_path / "CERT-T_20260914_150000"
    (tmp_path / "CERT-T_20260914_150000.scene").write_text(SCENE_OK)
    (tmp_path / "CERT-T_20260914_150000.ping").write_text(PING_OK)
    with open(str(run) + "_telemetry.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(ts.HEADER)
        for i in range(n_rows):
            row = {c: 0.0 for c in ts.HEADER}
            row.update({"frame_index": i, "t_mono": 100 + i * 0.05, "shoulder_lift.pos": 10, "elbow_flex.pos": 10,
                        "wrist_flex.pos": 10, "gripper.pos": 13.0})
            w.writerow([row[c] for c in ts.HEADER])
    return run


def test_record_writes_the_seat_receipt_from_the_run_files(tmp_path, capsys):
    run = _run_files(tmp_path)
    assert cert_gate.main(["record", str(run), "SEAT"]) == 0
    kv = dict(ln.split("=", 1) for ln in (tmp_path / "CERT-T_20260914_150000.seat").read_text().strip().splitlines())
    assert kv["label"] == "CERT-T" and kv["stamp"] == "20260914_150000" and kv["verdict"] == "SEAT"
    assert kv["tube_px"] == "1.8" and kv["loop_hz"] == "20.00" and kv["ee_proxy"] == "30.00" and kv["jaw_min"] == "13.00"
    assert "verdict=SEAT" in capsys.readouterr().out


def test_record_refuses_a_second_verdict_and_a_bad_one(tmp_path, capsys):
    run = _run_files(tmp_path)
    assert cert_gate.main(["record", str(run), "seated"]) == 1
    assert cert_gate.main(["record", str(run), "MISS"]) == 0
    assert cert_gate.main(["record", str(run), "SEAT"]) == 1, "a pass has one verdict; never overwrite a receipt"
    assert "already" in capsys.readouterr().out


def test_record_refuses_a_run_without_its_files(tmp_path):
    run = _run_files(tmp_path, n_rows=150)     # stalled replay: not a certification pass
    assert cert_gate.main(["record", str(run), "MISS"]) == 1
    assert cert_gate.main(["record", str(tmp_path / "CERT-NOPE_1"), "MISS"]) == 1


# Sep 17 2026 07:25: three certification misses in a row. On every seated pass the home routine settled the
# elbow at 85.6; on the misses at 86.2-86.9, and the grasp then ran 1.5 deg high under double the load
# (worn gearbox on a different tooth). preflight's +-1 sigma window (2.74 deg) passes both. The
# certification home therefore gets its own band: the settled elbow within CERT_ELBOW_TOL of the seated
# passes' 85.6, and run_cert.sh re-homes until it lands there (bounded).
def test_cert_home_band_constants():
    assert cert_gate.CERT_ELBOW_TARGET == 85.6 and cert_gate.CERT_ELBOW_TOL == 0.5


def test_home_band_refuses_a_high_elbow_and_accepts_the_seated_one():
    ok, why = cert_gate.gate_home(HOME_OK, elbow_band=True)
    assert ok, why                                                       # settled elbow 85.6
    high = HOME_OK.replace("'elbow_flex': 85.6", "'elbow_flex': 86.3")
    ok, why = cert_gate.gate_home(high, elbow_band=True)
    assert not ok and "86.3" in why and "85.6" in why
    assert cert_gate.gate_home(high)[0], "without the band the old rule is unchanged"


def test_cli_home_band_flag(tmp_path):
    home = tmp_path / "h.home"
    home.write_text(HOME_OK.replace("'elbow_flex': 85.6", "'elbow_flex': 86.9"))
    assert cert_gate.main(["home", str(home)]) == 0
    assert cert_gate.main(["home", str(home), "--elbow-band"]) == 1
