#!/usr/bin/env python
"""`lerobot-record`, with the phantom-arrow-key guard installed.

USE THIS INSTEAD OF `lerobot-record` FOR EVERY RECORDING SESSION. Same
arguments, same behaviour, one difference: a scroll-forged Left/Right arrow
cannot re-record or skip an episode. See tools/arrow_guard.py for why, and
tests/test_arrow_guard.py for the proof that both entry points are covered.

    /opt/homebrew/Caskroom/miniforge/base/envs/lerobot/bin/python tools/record_arrow_safe.py \\
        --robot.type=so101_follower ... --resume=true

Use that FULL interpreter path, not a bare `python`: on this Mac `python` is
miniforge base and cannot import lerobot at all.

Controls are unchanged, arrows included:
    Right / n = end this episode    Left / r = re-record    Esc / q = stop

⚠️ The scroll hazard is therefore still live: a raw-mode terminal turns
mouse-wheel scroll into arrow escape sequences, and a forged Right during the
15 s reset saves a zero-frame episode and crashes the run. DO NOT SCROLL THE
RECORDING TERMINAL. Granting Terminal.app Accessibility permission is the real
fix — pynput then binds to genuine key events instead of the TTY reader.

Importing this module installs the guard and does nothing else, so the test
suite can verify it without launching a recorder.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arrow_guard import install  # noqa: E402

# The only interpreter that can import lerobot on this machine.
ENV_PYTHON = "/opt/homebrew/Caskroom/miniforge/base/envs/lerobot/bin/python"


def _require_lerobot() -> None:
    """Fail with the fix, not a naked traceback.

    Bare `python` on this Mac is miniforge BASE, not the lerobot env, and cannot
    import lerobot at all (checked Sep 5: `which python` ->
    .../miniforge/base/bin/python). This has to run BEFORE install(), because
    install() imports lerobot.utils.keyboard_input itself — putting the check
    inside main() made it dead code, which is the same bug class this whole
    session has been fixing.
    """
    import importlib.util

    if importlib.util.find_spec("lerobot") is None:
        sys.exit(
            "[record_arrow_safe] lerobot is not importable under this interpreter\n"
            f"  running:  {sys.executable}\n"
            f"  use:      {ENV_PYTHON}\n"
            "  (bare `python` is miniforge base, not the lerobot env)"
        )


_require_lerobot()

# CAPSTONE_ARROW_MODE (Sep 5 evening):
#   live      every arrow works (Faith's default; the scroll hazard is live)
#   no-left   Left (re-record) is dropped; Right, Esc, n, r, q still work.
#             Chosen after ep 25 was re-recorded SIX times and ep 26 twice, each
#             re-record landing 4-5 s into the reset window like clockwork —
#             the reset routine brushing an input device. Right still ends an
#             episode; `r` re-records deliberately.
#   filtered  all four arrows dropped (the rollout wrapper's behaviour)
_MODE = os.environ.get("CAPSTONE_ARROW_MODE", "live").strip().lower()
if _MODE == "live":
    install(filter_arrows=False)
elif _MODE == "no-left":
    install(keys=frozenset({"left"}))
elif _MODE == "filtered":
    install()
else:
    sys.exit(f"[record_arrow_safe] CAPSTONE_ARROW_MODE={_MODE!r} is not one of live | no-left | filtered")


def main() -> None:
    # Imported here, AFTER the guard: keeps `import record_arrow_safe` cheap
    # for the tests and makes the ordering impossible to get wrong.
    from lerobot.scripts.lerobot_record import main as _record_main

    _record_main()


if __name__ == "__main__":
    main()
