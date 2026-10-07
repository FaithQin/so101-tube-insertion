"""Feetech telemetry block read (Aug 25, 2026): registers 56-70 are contiguous
on STS servos, so ONE sync-read transaction of 15 bytes yields position,
velocity, load, voltage, temperature, and current for every motor — the answer
to upstream PR #3456's latency objection (3 extra sync_reads ~15-30 ms)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from feetech_block_read import BLOCK_ADDR, BLOCK_LEN, FIELDS, fields_within_block  # noqa: E402


def test_block_covers_all_fields():
    assert fields_within_block(), "every field must lie inside [BLOCK_ADDR, BLOCK_ADDR+BLOCK_LEN)"


def test_field_table_matches_installed_control_table():
    from lerobot.motors.feetech.tables import STS_SMS_SERIES_CONTROL_TABLE as CT
    expect = {"position": "Present_Position", "velocity": "Present_Velocity", "load": "Present_Load",
              "voltage": "Present_Voltage", "temperature": "Present_Temperature", "current": "Present_Current"}
    for field, reg in expect.items():
        assert FIELDS[field] == CT[reg], f"{field} must match control table {reg} {CT[reg]} != {FIELDS[field]}"


def test_block_bounds():
    assert BLOCK_ADDR == 56 and BLOCK_ADDR + BLOCK_LEN == 71, "block must span registers 56-70 inclusive"
