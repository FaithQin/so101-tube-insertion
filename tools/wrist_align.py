"""Live NORMALIZED wrist_roll on both arms. The ONLY valid alignment test.

Do NOT judge wrist_roll alignment by comparing the two arms' raw homing_offset
values. homing_offset = 2047 - raw_at_ENTER, and the raw encoder value depends
on how each servo's shaft was assembled into its wrist at the factory, so two
arms read DIFFERENT raw counts at the SAME physical orientation. A nonzero gap
at perfect alignment is expected and means nothing. (This mistake cost a wasted
recalibration on Aug 7, 2026 — see README "Bring-up lessons".)

The valid test is behavioural: at equal physical orientation, do both arms
report the same NORMALIZED position? That is what teleop passes leader->follower.

Usage:
    conda activate lerobot
    python wrist_align.py [n_samples]      # no arg = run until Ctrl-C

To fix a misalignment without touching the leader:
  1. Run this, rotate the LEADER wrist until its column reads ~0.00, leave it.
  2. Recalibrate ONLY the follower. Pose it mid-range, then align its wrist to
     the leader's current physical orientation as the LAST move before ENTER.
     (Hauling the arm into a mid-range pose rotates wrist_roll without you
     feeling it — that is why alignment must be last, not first.)
  3. Re-run this with both wrists still physically aligned. Delta must be ~0.

wrist_roll spans one full turn over -100..100, so 1 unit ~= 1.8 degrees.
"""

import sys
import time

from _arms import FOLLOWER_CAL, FOLLOWER_PORT, LEADER_CAL, LEADER_PORT, open_bus

max_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 0

buses = {
    "leader": open_bus(LEADER_PORT, LEADER_CAL),
    "follower": open_bus(FOLLOWER_PORT, FOLLOWER_CAL),
}

print("Align the two taped wrists to the SAME physical orientation, then read the delta.")
print("Delta near 0 = good. ~1 unit = ~1.8 deg. Ctrl-C to stop.\n")

n = 0
try:
    while max_samples == 0 or n < max_samples:
        n += 1
        lead = buses["leader"].read("Present_Position", "wrist_roll", normalize=True)
        foll = buses["follower"].read("Present_Position", "wrist_roll", normalize=True)
        delta = lead - foll
        verdict = "OK" if abs(delta) < 3 else ("close" if abs(delta) < 8 else "OFF")
        print(f"  leader {lead:>8.2f}   follower {foll:>8.2f}   "
              f"delta {delta:>8.2f}  ({delta * 1.8:>7.1f} deg)  {verdict}")
        time.sleep(0.3)
except KeyboardInterrupt:
    pass
finally:
    for bus in buses.values():
        bus.disconnect(disable_torque=False)
