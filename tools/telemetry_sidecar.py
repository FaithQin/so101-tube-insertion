"""Tee all 18 servo channels (6 pos, 6 load, 6 current) to a CSV, one row per RECORDED frame.

Why (Sep 6 2026, eval redesign): the evaluation's most powerful endpoints are contact-response
measures computed from load/current at the moment of contact. A rollout records the observation
the POLICY saw, so an A-side (6-dim) trial stores no load/current at all. The follower reads all
18 channels every tick regardless; this module captures them beside the rollout, write-only, and
never touches what reaches the policy. Frame index = the order of `add_frame` calls, so rows align
1:1 with the recorded episode's frames.

    CAPSTONE_TELEMETRY_CSV=/path/to/<label>_telemetry.csv   (set by every trial runner)

Unset -> the sidecar is inert and writes nothing.
"""
from __future__ import annotations

import csv
import math
import os
import time

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CHANNELS = ["pos", "load", "current"]
HEADER = ["frame_index", "t_mono"] + [f"{j}.{c}" for c in CHANNELS for j in JOINTS]


def _channels(obs) -> list[float]:
    """The 18 values in HEADER order; a missing key is NaN in its column, never a shifted row."""
    out = []
    for c in CHANNELS:
        for j in JOINTS:
            v = obs.get(f"{j}.{c}")
            try:
                out.append(float(v) if v is not None else math.nan)
            except (TypeError, ValueError):
                out.append(math.nan)
    return out


def telemetry_row(obs, frame_index: int, t_mono: float) -> list:
    return [frame_index, t_mono] + _channels(obs)


class Sidecar:
    """observe() on every robot read (keeps the latest values); frame_recorded() writes a row."""

    def __init__(self, path: str | None):
        self.path = path
        self._latest: tuple[float, list[float]] | None = None
        self._n = 0
        self._fh = None
        self._writer = None

    @classmethod
    def from_env(cls, env=None) -> "Sidecar":
        env = os.environ if env is None else env
        p = (env.get("CAPSTONE_TELEMETRY_CSV") or "").strip()
        return cls(p or None)

    def observe(self, obs) -> None:
        if self.path is None or not isinstance(obs, dict):
            return
        self._latest = (time.monotonic(), _channels(obs))  # copy the numbers now: lerobot reuses the dict

    def frame_recorded(self) -> None:
        if self.path is None or self._latest is None:
            return
        if self._writer is None:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            self._fh = open(self.path, "w", newline="")
            self._writer = csv.writer(self._fh)
            self._writer.writerow(HEADER)
        t_mono, values = self._latest
        self._writer.writerow([self._n, f"{t_mono:.4f}"] + values)
        self._n += 1
        self._fh.flush()  # a crash mid-trial keeps every frame written so far

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None
