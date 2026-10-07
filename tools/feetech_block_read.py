"""One-transaction telemetry read for Feetech STS servos: registers 56-70 are
contiguous, so a single GroupSyncRead of 15 bytes returns position, velocity,
load, voltage, temperature, and current for every motor on the bus — instead
of one bus transaction per register.

Context: upstream lerobot PR #3456 (optional SO-follower telemetry) stalled on
"3 extra sync_read calls cost ~15-30 ms on a 6-motor bus." This block read
makes telemetry cost approximately the same as reading position alone.
Measured numbers: tools/bench_block_read.py. Field table is asserted against
lerobot's own control table in tests/test_block_read.py.
"""

BLOCK_ADDR = 56
BLOCK_LEN = 15  # registers 56..70 inclusive

# field -> (address, length), matching lerobot's STS_SMS_SERIES_CONTROL_TABLE
FIELDS = {
    "position": (56, 2),
    "velocity": (58, 2),
    "load": (60, 2),
    "voltage": (62, 1),
    "temperature": (63, 1),
    "current": (69, 2),
}


def fields_within_block() -> bool:
    return all(BLOCK_ADDR <= a and a + n <= BLOCK_ADDR + BLOCK_LEN for a, n in FIELDS.values())


# register name per field, for sign decoding via the bus's own encoding table
FIELD_REGISTER = {
    "position": "Present_Position",
    "velocity": "Present_Velocity",
    "load": "Present_Load",
    "voltage": "Present_Voltage",
    "temperature": "Present_Temperature",
    "current": "Present_Current",
}


def read_telemetry_block(bus, motors=None, num_retry: int = 0, decode_sign: bool = True):
    """One sync-read transaction; returns {motor_name: {field: value}}.

    With decode_sign=True (default) each field is sign-decoded exactly as
    lerobot's sync_read does (bus._decode_sign against the model's
    sign-magnitude encoding table — e.g. Present_Load uses sign bit 10), so
    values are drop-in identical to per-register sync_read(..., normalize=False).
    Uses only bus internals already exercised by sync_read: _sync_read + the
    sdk sync_reader.getData sub-window extraction.
    """
    names = bus._get_motors_list(motors)
    ids = [bus.motors[m].id for m in names]
    bus._sync_read(BLOCK_ADDR, BLOCK_LEN, ids, num_retry=num_retry,
                   raise_on_error=True, err_msg="telemetry block read failed")
    per_field = {}
    for field, (addr, n) in FIELDS.items():
        vals = {id_: bus.sync_reader.getData(id_, addr, n) for id_ in ids}
        if decode_sign:
            vals = bus._decode_sign(FIELD_REGISTER[field], vals)
        per_field[field] = vals
    return {name: {f: per_field[f][id_] for f in FIELDS} for name, id_ in zip(names, ids)}
