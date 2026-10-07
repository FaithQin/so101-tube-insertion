#!/usr/bin/env python
"""Neutralise phantom arrow keys in BOTH of lerobot's keyboard backends.

WHY (Aug 18 2026, recurring Sep 4): lerobot's recording controls map
Right -> "end this episode" and Left -> "re-record the last episode"
(`utils/keyboard_input.py:153-171`). Two separate mechanisms forge those keys
without anyone touching an arrow key:

1. **pynput backend.** The listener is a GLOBAL hook. Logi Options+ horizontal
   scrolling emits Left/Right keystrokes, so a thumb-wheel scroll ANYWHERE on
   the Mac hit re-record/skip. Shredded the Aug 16 and Aug 17 probe sessions
   into 2-3 s fragments.
2. **terminal backend.** On macOS without Accessibility / Input Monitoring
   permission, `init_keyboard_listener` falls back to `TerminalKeyListener`,
   which reads the TTY in raw mode — and a raw-mode terminal converts
   mouse-wheel scroll into arrow escape sequences (`ESC [ C`). Scrolling the
   session's own terminal pressed re-record/skip (shredded the Aug 18 00:52
   probe even after the pynput map was stripped).

During RECORDING the second one is worse than during a rollout: a Right-arrow
inside the 15 s reset window saves a zero-frame episode and crashes the run
with `ValueError: You must add one or several frames with add_frame`. It has
cost three episodes.

WHAT SURVIVES: `n` (next), `r` (re-record), `q` (quit) and `Esc`.
`init_keyboard_listener` accepts those letters as equivalents of the arrows
(`keyboard_input.py:426-436`), so the operator keeps every control — only the
scroll-forgeable escape sequences are dropped. Use n / r / q / ESC.

This module is the single source of truth for both entry points
(`rollout_30hz_stale_ok.py` and `record_arrow_safe.py`). It was a copy inside
the rollout wrapper until Sep 5, which is exactly why recording never got it.
"""

from __future__ import annotations

import sys

ARROWS = frozenset(("left", "right", "up", "down"))
# Sep 17 2026 07:35: an "n -> end episode" translation lived here for 30 minutes and was REVERTED: the launcher's
# Terminal window takes keyboard focus when it opens, so a letter typed for the chat ended a scored trial at 2.7 s
# (P28 trial 2, second run). Letters are ignored again; a scored trial runs its full 45 s.

_INSTALLED_FLAG = "_capstone_arrow_guard_installed"


def install(kbi=None, *, filter_arrows: bool = True, announce: bool = True,
            keys: frozenset = ARROWS) -> bool:
    """Install the guard on lerobot's keyboard module. Returns False if already on.

    `filter_arrows=False` leaves the arrows LIVE and installs nothing. Faith
    drives recording with the arrow keys and finds them easier than n/r/q
    (Sep 5) — her workflow, her call. The two entry points therefore differ:

      * ROLLOUT  (rollout_30hz_stale_ok.py) filters. It has since Aug 18, the
        keys are not used during a probe, and scroll-shredded probe sessions
        are what motivated the guard.
      * RECORD   (record_arrow_safe.py) does NOT filter, because the operator
        uses Left/Right deliberately between episodes.

    The scroll hazard is unchanged for recording and is documented at the call
    site. Granting Terminal.app Accessibility permission is the real fix for
    the terminal backend: pynput then binds to genuine key events instead of
    the TTY reader that turns wheel-scroll into arrow escape sequences.

    Patches `create_key_listener` as a MODULE ATTRIBUTE, not the callers'
    bindings: `init_keyboard_listener` resolves it from module globals at call
    time (`keyboard_input.py:441`), so this works no matter which of
    `lerobot_record` / `episodic` / `dagger` imported what, and in any order.
    """
    if kbi is None:
        import lerobot.utils.keyboard_input as kbi  # noqa: PLC0415

    if getattr(kbi, _INSTALLED_FLAG, False):
        return False

    if not filter_arrows:
        if announce:
            print("[arrow_guard] arrows LIVE — Left=re-record, Right=next, Esc=stop. "
                  "Do not scroll this terminal: wheel-scroll forges arrow keys.",
                  file=sys.stderr)
        return False
    blocked = frozenset(keys)

    # Backend 1: drop the arrow entries from the pynput special-key map, so
    # `_resolve_pynput_key` returns None and the dispatcher is never called.
    key_names = getattr(kbi, "_PYNPUT_KEY_NAMES", None)
    if isinstance(key_names, dict):
        for key, name in list(key_names.items()):
            if name in blocked:
                del key_names[key]

    # Backend 2 (and belt-and-braces for backend 1): filter at the dispatch
    # layer both backends share.
    original = kbi.create_key_listener

    def guarded_create_key_listener(dispatch, **kwargs):
        def filtered(name):
            if name in blocked:
                print(f"[arrow_guard] ignored phantom '{name}' key "
                      f"(scroll-forged; no early-end key; ESC stops)", file=sys.stderr)
                return
            dispatch(name)
        return original(filtered, **kwargs)

    kbi.create_key_listener = guarded_create_key_listener
    setattr(kbi, _INSTALLED_FLAG, True)

    if announce:
        print(f"[arrow_guard] {sorted(blocked)} neutralized in BOTH keyboard backends "
              "(scroll-wheel phantom-key guard); letters are ignored; Esc stops.",
              file=sys.stderr)
    return True
