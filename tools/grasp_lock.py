"""Grasp-phase chunk lock: a pure state machine over the policy's own
COMMANDED gripper and shoulder_lift values (no vision, no extra sensors).

Motivation (Aug 25, 2026, measured on the 40-trial v2 baseline): the dominant
failure mode was a correct descent discarded at the moment of clamping by the
next action chunk's re-plan, which veered left. The lock detects closing onset
and — via the rollout wrapper — lets the CURRENT chunk run to its end instead
of re-planning mid-close. It re-arms after every completed or abandoned close,
so each re-grasp attempt is protected independently.

States: IDLE -> ARMED (gripper open, lift inside the grasp band)
             -> LOCKED (gripper commanded to close on >=2 consecutive ticks)
             -> IDLE  (close held shut for RELEASE_TICKS, or force_release()).

Thresholds mirror the measured v2 command ranges: gripper 0..47 with open >15 /
closed <8 (the eval sheet's hysteresis), grasp band = commanded lift < 50
(demos close at lift ~37; transport/retreat run higher).
Pure Python, no dependencies — unit-tested in tests/test_grasp_lock.py.
"""

OPEN_T = 15.0
CLOSED_T = 8.0
LIFT_BAND = 50.0
ONSET_TICKS = 2
ONSET_DELTA = 0.5


class GraspLock:
    RELEASE_TICKS = 5

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._state = "IDLE"
        self._prev_grip = None
        self._closing_run = 0
        self._closed_run = 0

    @property
    def locked(self) -> bool:
        return self._state == "LOCKED"

    def force_release(self) -> None:
        """Called by the wrapper when the cached chunk exhausts mid-lock."""
        self._state = "IDLE"
        self._closing_run = 0
        self._closed_run = 0

    def observe(self, grip_cmd: float, lift_cmd: float) -> bool:
        """Feed one tick of commanded gripper + lift; returns .locked."""
        if self._state == "IDLE":
            if grip_cmd > OPEN_T and lift_cmd < LIFT_BAND:
                self._state = "ARMED"
                self._closing_run = 0
        elif self._state == "ARMED":
            if lift_cmd >= LIFT_BAND:
                self._state = "IDLE"
            elif self._prev_grip is not None and grip_cmd < self._prev_grip - ONSET_DELTA:
                self._closing_run += 1
                if self._closing_run >= ONSET_TICKS:
                    self._state = "LOCKED"
                    self._closed_run = 0
            else:
                self._closing_run = 0
        elif self._state == "LOCKED":
            if grip_cmd < CLOSED_T:
                self._closed_run += 1
                if self._closed_run >= self.RELEASE_TICKS:
                    self._state = "IDLE"
            else:
                self._closed_run = 0
        self._prev_grip = grip_cmd
        return self.locked
