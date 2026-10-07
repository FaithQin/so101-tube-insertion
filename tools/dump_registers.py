"""Dump every EEPROM/config register of every servo on one arm, and flag the
registers where one servo differs from the others. READ-ONLY.

Why (Sep 2, 2026): "EEPROM verified equal" after the Sep 1 calibration mess
covered only Homing_Offset and the two position limits. The raw register writes
that day could have touched any other EEPROM field (Protection_Time had been
found at 0 once before). The elbow then landed a constant +3.07 deg above its
goal at two poses with opposite gravity loading — the shape of a control
parameter (dead zone, P gain, startup force), not of a calibration offset.
This prints all of them, side by side, so a changed register on ONE servo is
visible in seconds.

    conda activate lerobot
    python tools/dump_registers.py            # follower
    python tools/dump_registers.py leader

12V ON BEFORE USB. Needs the port to itself. Nothing here writes to a motor,
and torque is left exactly as found.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import json  # noqa: E402

from _arms import FOLLOWER_CAL, FOLLOWER_PORT, JOINTS, LEADER_CAL, LEADER_PORT, open_bus  # noqa: E402
from lerobot.motors.feetech.tables import STS_SMS_SERIES_CONTROL_TABLE  # noqa: E402

# Registers that legitimately differ per servo (identity + calibration).
PER_SERVO_OK = frozenset({"ID", "Homing_Offset", "Min_Position_Limit", "Max_Position_Limit"})

# Live status, appended after the EEPROM block for context.
STATUS_REGISTERS = ("Torque_Enable", "Present_Position", "Present_Temperature",
                    "Present_Voltage", "Status")


def eeprom_registers(table=STS_SMS_SERIES_CONTROL_TABLE):
    """Every register whose address sits below Torque_Enable (the first RAM
    register), in table order. Derived, not hard-coded, so it tracks lerobot."""
    ram_start = table["Torque_Enable"][0]
    return [name for name, (addr, _) in table.items() if addr < ram_start]


def flag_outliers(table, ignore=PER_SERVO_OK):
    """table[register][joint] -> value. Return the registers whose readable
    values are not identical across joints, skipping per-servo registers."""
    out = {}
    for reg, vals in table.items():
        if reg in ignore:
            continue
        present = {j: v for j, v in vals.items() if v is not None}
        if len(set(present.values())) > 1:
            out[reg] = present
    return out


CAL_REGISTERS = {"Homing_Offset": "homing_offset",
                 "Min_Position_Limit": "range_min",
                 "Max_Position_Limit": "range_max"}


def calibration_mismatches(table, calibration):
    """Cross-check the live read against the calibration FILE for the registers
    whose true values the file holds. Returns [(joint, register, live, file)].
    An empty list means the instrument reproduced every known value; that is
    what licenses trusting the registers the file does not hold."""
    out = []
    for reg, key in CAL_REGISTERS.items():
        for joint, cal in calibration.items():
            live = table.get(reg, {}).get(joint)
            expected = cal[key] if isinstance(cal, dict) else getattr(cal, key)
            if live != expected:
                out.append((joint, reg, live, expected))
    return out


def read_table(bus, registers):
    table = {}
    for reg in registers:
        table[reg] = {}
        for joint in JOINTS:
            try:
                table[reg][joint] = bus.read(reg, joint, normalize=False)
            except Exception:
                table[reg][joint] = None
    return table


def print_table(table):
    w = max(len(r) for r in table) + 2
    print(f"{'register':{w}s}" + "".join(f"{j[:13]:>14s}" for j in JOINTS))
    for reg, vals in table.items():
        row = "".join(f"{('?' if vals[j] is None else vals[j]):>14}" for j in JOINTS)
        print(f"{reg:{w}s}{row}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("arm", nargs="?", default="follower", choices=["follower", "leader"])
    args = parser.parse_args()
    port = FOLLOWER_PORT if args.arm == "follower" else LEADER_PORT

    print(f"=== {args.arm.upper()}  {port}  (read-only) ===")
    try:
        bus = open_bus(port)
    except Exception as exc:
        print(f"cannot open the port: {exc}")
        print("-> 12V on BEFORE USB? another process holding the port?")
        return 1
    try:
        eeprom = read_table(bus, eeprom_registers())
        status = read_table(bus, STATUS_REGISTERS)
    finally:
        bus.disconnect(disable_torque=False)

    print("\n--- EEPROM / configuration ---")
    print_table(eeprom)
    print("\n--- live status ---")
    print_table(status)

    cal_path = FOLLOWER_CAL if args.arm == "follower" else LEADER_CAL
    calibration = json.loads(Path(cal_path).read_text())
    mism = calibration_mismatches(eeprom, calibration)
    n = len(CAL_REGISTERS) * len(calibration)
    print(f"\n--- instrument cross-check vs {cal_path.name} ---")
    if mism:
        print(f"MISMATCH on {len(mism)}/{n} known registers; do not trust the rest of this dump:")
        for joint, reg, live, exp in mism:
            print(f"  {joint} {reg}: live {live} vs file {exp}")
    else:
        print(f"MATCH: all {n} registers the file holds read back identically.")

    outliers = flag_outliers(eeprom)
    print("\n--- registers that differ across servos (ID / offsets / limits excluded) ---")
    if not outliers:
        print("none: every configuration register is identical on all six servos.")
    for reg, vals in outliers.items():
        counts = {}
        for v in vals.values():
            counts[v] = counts.get(v, 0) + 1
        common = max(counts, key=counts.get)
        odd = {j: v for j, v in vals.items() if v != common}
        print(f"  {reg}: common value {common}; differs on {odd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
