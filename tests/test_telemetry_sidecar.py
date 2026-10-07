"""Every trial logs all 18 servo channels, whatever the policy is allowed to see.

Sep 6 2026, eval redesign: both fresh-eyes protocols put contact-response measures (load
spike -> first opposing command) at the centre of the evaluation, because that is where a
200-trial budget has real resolving power. They both assumed both arms log load/current.
They do not: a rollout records the observation THE POLICY SAW, so an A-side trial stores a
6-dim state and the load/current that would show the contact are gone. The rollout wrapper
already wraps get_observation and add_frame for timing; `telemetry_sidecar` tees the raw
18 channels per recorded frame to a CSV, write-only, never touching what reaches the policy.
"""
import csv
import sys

import pytest

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))
import telemetry_sidecar as ts  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def _obs(k=1.0):
    obs = {}
    for i, j in enumerate(JOINTS):
        obs[f"{j}.pos"] = 10.0 * i * k
        obs[f"{j}.load"] = 100.0 + i * k
        obs[f"{j}.current"] = 200.0 + i * k
    obs["front"] = "image-bytes"  # cameras must be ignored
    return obs


def test_row_carries_all_18_channels_in_joint_order():
    row = ts.telemetry_row(_obs(), frame_index=7, t_mono=12.5)
    assert row[:2] == [7, 12.5]
    assert row[2:8] == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]        # pos
    assert row[8:14] == [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]  # load   (mutation: drop load)
    assert row[14:20] == [200.0, 201.0, 202.0, 203.0, 204.0, 205.0]  # current
    assert len(row) == 20 and ts.HEADER[8] == "shoulder_pan.load" and ts.HEADER[14] == "shoulder_pan.current"


def test_missing_channels_are_recorded_as_nan_not_dropped():
    obs = _obs()
    del obs["elbow_flex.load"]
    row = ts.telemetry_row(obs, frame_index=0, t_mono=0.0)
    assert len(row) == 20 and row[10] != row[10]  # NaN, column position preserved (mutation: skip key)


def test_sidecar_writes_one_row_per_recorded_frame_and_nothing_without_a_path(tmp_path):
    path = tmp_path / "telemetry.csv"
    sc = ts.Sidecar(str(path))
    sc.observe(_obs(1.0)); sc.frame_recorded()
    sc.observe(_obs(2.0)); sc.observe(_obs(3.0)); sc.frame_recorded()   # two reads, one frame -> latest wins
    sc.close()
    rows = list(csv.reader(open(path)))
    assert rows[0] == ts.HEADER
    assert len(rows) == 3 and rows[1][0] == "0" and rows[2][0] == "1"
    assert float(rows[2][8]) == 100.0 * 1 + 0 * 3.0 and float(rows[2][9]) == 100.0 + 1 * 3.0  # frame 1 = obs k=3
    off = ts.Sidecar(None)
    off.observe(_obs()); off.frame_recorded(); off.close()
    assert list(tmp_path.iterdir()) == [path]                        # mutation: write a default file


def test_from_env_reads_the_runner_variable(monkeypatch, tmp_path):
    monkeypatch.delenv("CAPSTONE_TELEMETRY_CSV", raising=False)
    assert ts.Sidecar.from_env().path is None
    monkeypatch.setenv("CAPSTONE_TELEMETRY_CSV", str(tmp_path / "t.csv"))
    assert ts.Sidecar.from_env().path == str(tmp_path / "t.csv")


def test_the_rollout_wrapper_installs_the_sidecar_on_both_hooks():
    src = (TOOLS / "rollout_30hz_stale_ok.py").read_text()
    assert "telemetry_sidecar" in src
    assert "CAPSTONE_TELEMETRY_CSV" in src or "Sidecar.from_env" in src
    # get_observation feeds it, add_frame flushes a row: both hooks must be present
    assert ".observe(" in src and ".frame_recorded(" in src


@pytest.mark.parametrize("runner", ["run_scored_trial.sh", "run_smolvla_trial.sh", "run_pi05_trial.sh"])
def test_every_trial_runner_sets_the_telemetry_path(runner):
    src = (TOOLS / runner).read_text()
    assert "CAPSTONE_TELEMETRY_CSV" in src, f"{runner} does not export CAPSTONE_TELEMETRY_CSV"


def test_telemetry_csvs_are_not_gitignored():
    """tools/scored_logs/* is ignored except the ledgers; the telemetry CSVs are evidence and must be
    tracked the same way (the comparator review, Sep 6, caught this before the first bench trial)."""
    import subprocess
    from conftest import PROJECT
    probe = PROJECT / "tools" / "scored_logs" / "X_20260906_000000_telemetry.csv"
    out = subprocess.run(["git", "check-ignore", "-q", str(probe)], cwd=PROJECT)
    assert out.returncode == 1, "tools/scored_logs/*_telemetry.csv is gitignored -- the mechanism data would vanish"
