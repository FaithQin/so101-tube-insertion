"""The phantom-arrow-key guard, and the fact that RECORDING gets it too.

Sep 4 2026, twice, and once on Aug 18: `lerobot-record` logs

    pynput keyboard listener is not trusted (missing macOS Accessibility /
    Input Monitoring permission); falling back to terminal keyboard input.

A terminal in raw mode converts mouse-wheel scroll into arrow escape
sequences, and `init_keyboard_listener` maps Right -> exit_early and Left ->
rerecord_episode. **A stray scroll during the 15 s reset window saves a
zero-frame episode and crashes the run** with
`ValueError: You must add one or several frames with add_frame`. It has cost
three episodes.

`tools/rollout_30hz_stale_ok.py` has neutralised arrows since Aug 18 -- but
only for ROLLOUTS. `lerobot-record` does not go through that wrapper, so
recording ran unprotected through the entire v4 session.

The guard lives in one module now (`tools/arrow_guard.py`) precisely so the two
entry points cannot drift apart -- "the rehearsal diverging from the production
path" is a named bug class in this repo.

Note what is NOT filtered: `n` (next), `r` (re-record), `q` (quit) and `Esc`.
`init_keyboard_listener` accepts those letters as equivalents of the arrows
(keyboard_input.py:426-436), so the operator keeps every control; only the
scroll-forgeable escape sequences are dropped.
"""
import os
import subprocess
import sys
import types

import pytest

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))
import arrow_guard  # noqa: E402

PYBIN = sys.executable


def _fake_kbi():
    """A stand-in for lerobot.utils.keyboard_input, so tests mutate nothing real."""
    m = types.ModuleType("fake_kbi")
    m._PYNPUT_KEY_NAMES = {
        "K_LEFT": "left", "K_RIGHT": "right", "K_UP": "up", "K_DOWN": "down",
        "K_ESC": "esc", "K_SPACE": "space",
    }
    m.created = []

    def create_key_listener(dispatch, *, controls_help=""):
        m.created.append(dispatch)
        return dispatch
    m.create_key_listener = create_key_listener
    return m


def test_install_strips_arrows_from_the_pynput_key_map():
    k = _fake_kbi()
    arrow_guard.install(k)
    assert set(k._PYNPUT_KEY_NAMES.values()) == {"esc", "space"}, (
        "arrow names still resolve in the pynput backend"
    )


def test_arrows_never_reach_the_dispatcher():
    """The terminal backend: raw-mode scroll forges these escape sequences."""
    k = _fake_kbi()
    arrow_guard.install(k)
    seen = []
    dispatch = k.create_key_listener(seen.append)
    for name in ("left", "right", "up", "down"):
        dispatch(name)
    assert seen == [], f"phantom arrow keys reached the recorder: {seen}"


def test_the_operators_real_controls_still_work():
    """r / q / Esc pass through untouched; n is TRANSLATED to the end-episode control (below)."""
    k = _fake_kbi()
    arrow_guard.install(k)
    seen = []
    dispatch = k.create_key_listener(seen.append)
    for name in ("r", "q", "esc"):
        dispatch(name)
    assert seen == ["r", "q", "esc"]


def test_letters_are_ignored_there_is_no_early_end_key():
    """Sep 17 2026 07:35: an 'n -> end episode' translation was added at 07:06 and reverted 30 minutes later --
    the launcher's Terminal window takes focus when it opens, so a letter typed for the chat ended a scored
    trial at 2.7 s. Letters pass through untouched (lerobot's dispatcher ignores them); the arrow is dropped."""
    k = _fake_kbi()
    arrow_guard.install(k)
    seen = []
    dispatch = k.create_key_listener(seen.append)
    dispatch("n"); dispatch("right")
    assert seen == ["n"], seen


def test_install_is_idempotent():
    """The record wrapper may import a module that already installed it."""
    k = _fake_kbi()
    assert arrow_guard.install(k) is True
    assert arrow_guard.install(k) is False, "second install re-wrapped the dispatcher"
    seen = []
    dispatch = k.create_key_listener(seen.append)
    dispatch("left"); dispatch("n")
    assert seen == ["n"]


# --- the two entry points, on the REAL lerobot module ----------------------

_PROBE = """
import sys
sys.path.insert(0, {tools!r})

# Stand in for the REAL create_key_listener *before* the entry point imports.
# install() then wraps this, so the filtered dispatcher it builds comes back to
# us and can be exercised directly -- a full round-trip through the real guard
# on the real lerobot module. (The genuine function returns a listener object,
# or None when headless, so it cannot be round-tripped in a subprocess.)
import lerobot.utils.keyboard_input as kbi
captured = {{}}
def fake_original(dispatch, **kw):
    captured["dispatch"] = dispatch
    return object()
kbi.create_key_listener = fake_original

import {module}  # noqa  -- installs the guard

arrows = [n for n in kbi._PYNPUT_KEY_NAMES.values() if n in ("left","right","up","down")]
seen = []
kbi.create_key_listener(seen.append, controls_help="")
assert "dispatch" in captured, "the guard did not wrap create_key_listener on the module"
for k in ("left", "right", "up", "down", "n", "r", "q", "esc"):
    captured["dispatch"](k)
print("ARROWS_IN_MAP=" + repr(arrows))
print("DISPATCHED=" + repr(seen))
"""


def _probe(module: str):
    r = subprocess.run([PYBIN, "-c", _PROBE.format(tools=str(TOOLS), module=module)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"probe failed for {module}:\n{r.stdout}\n{r.stderr[-1500:]}"
    return r.stdout


def test_the_rollout_wrapper_is_guarded():
    """Rollouts still filter. The keys are not used during a probe, and
    scroll-shredded probe sessions (Aug 16-18) are what motivated the guard."""
    out = _probe("rollout_30hz_stale_ok")
    assert "ARROWS_IN_MAP=[]" in out, f"pynput map still resolves arrows\n{out}"
    assert "DISPATCHED=['n', 'r', 'q', 'esc']" in out, (
        f"arrows reached the dispatcher, or a real control was eaten\n{out}"
    )


def test_recording_leaves_the_ARROWS_LIVE():
    """Faith drives recording with the arrow keys (Sep 5) — her workflow.

    Left = re-record, Right = next. Filtering them out of the recorder would
    silently take away the controls she actually uses, which is worse than the
    scroll hazard it prevents: a lost control is discovered mid-session with the
    tube in the jaws. The hazard is handled by not scrolling that terminal, and
    properly by granting Terminal.app Accessibility permission.
    """
    out = _probe("record_arrow_safe")
    assert "DISPATCHED=['left', 'right', 'up', 'down', 'n', 'r', 'q', 'esc']" in out, (
        f"the recorder is swallowing arrow keys — Left/Right must reach lerobot\n{out}"
    )
    assert "ARROWS_IN_MAP=[]" not in out, (
        f"the recorder stripped the pynput arrow map — Left/Right would not work "
        f"under the pynput backend either\n{out}"
    )


def test_the_record_wrapper_does_not_reimplement_the_guard():
    """One guard, two callers. Two copies is how the rollout/record split happened."""
    for name in ("record_arrow_safe.py", "rollout_30hz_stale_ok.py"):
        src = (TOOLS / name).read_text()
        assert "arrow_guard" in src, f"{name} does not use the shared guard"
        assert "_PYNPUT_KEY_NAMES" not in src, (
            f"{name} carries its own copy of the pynput-map surgery -- the two copies "
            f"will drift, which is exactly how recording ended up unprotected"
        )


_ORDER_PROBE = """
import sys
sys.path.insert(0, {tools!r})

# Import the RECORDER FIRST. lerobot_record.py:157 does
#   from lerobot.utils.keyboard_input import init_keyboard_listener
# a from-import, so its binding is fixed at that moment. If the guard patched
# `init_keyboard_listener` it would be too late and silently ineffective.
# It patches `create_key_listener` instead, which init_keyboard_listener
# resolves from module globals at CALL time (keyboard_input.py:441) -- so the
# order cannot matter. This pins that, because "the guard silently stopped
# working" is indistinguishable from "the guard is on" at the bench.
import lerobot.scripts.lerobot_record  # noqa
import lerobot.utils.keyboard_input as kbi

captured = {{}}
def fake_original(dispatch, **kw):
    captured["dispatch"] = dispatch
    return object()
kbi.create_key_listener = fake_original

import rollout_30hz_stale_ok  # noqa  -- guard installs AFTER the recorder is imported

seen = []
kbi.init_keyboard_listener()
assert "dispatch" in captured, "init_keyboard_listener did not go through the guarded factory"
inner = captured["dispatch"]
# init_keyboard_listener wraps its own on_key around ours; drive the guarded one
for k in ("left", "right", "up", "down"):
    inner(k)
print("EVENTS=" + repr(sorted(k for k, v in kbi.init_keyboard_listener.__globals__.items() if False)))
print("ARROWS_IN_MAP=" + repr([n for n in kbi._PYNPUT_KEY_NAMES.values()
                               if n in ("left","right","up","down")]))
print("GUARD_FLAG=" + repr(getattr(kbi, "_capstone_arrow_guard_installed", False)))
"""


def test_the_guard_works_even_if_the_recorder_was_imported_first():
    """Checked on the ROLLOUT wrapper, the entry point that still filters.

    (Recording deliberately leaves the arrows live — see
    test_recording_leaves_the_ARROWS_LIVE — so it is not the subject here.)"""
    r = subprocess.run([PYBIN, "-c", _ORDER_PROBE.format(tools=str(TOOLS))],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"import-order probe failed:\n{r.stdout}\n{r.stderr[-1500:]}"
    assert "ARROWS_IN_MAP=[]" in r.stdout, r.stdout
    assert "GUARD_FLAG=True" in r.stdout, r.stdout


ENV_PYTHON = "/opt/homebrew/Caskroom/miniforge/base/envs/lerobot/bin/python"
DOCS_WITH_RECORD_COMMANDS = [
    "Start Schedule — v4.md",
    "working notes (private)",
    "working notes (private)",
]


def test_the_documented_record_command_pins_the_interpreter():
    """A bare `python` cannot import lerobot on this machine.

    Checked Sep 5: `which python` -> /opt/homebrew/Caskroom/miniforge/base/bin/python
    (miniforge BASE), and `python -c "import lerobot"` there raises
    ModuleNotFoundError. So a documented `python tools/record_arrow_safe.py ...`
    dies at the first line, mid-session, on the one command that must not fail.
    Every runner in tools/ already pins `$B/python` for exactly this reason;
    the docs must too.
    """
    root = TOOLS.parent
    offenders = []
    for rel in DOCS_WITH_RECORD_COMMANDS:
        path = root / rel
        if not path.exists():
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.lstrip()
            # a command line, not prose about one (prose is inside backticks)
            if stripped.startswith("python ") and "record_arrow_safe.py" in stripped:
                offenders.append(f"{rel}:{i}: {stripped[:80]}")
    assert not offenders, (
        "documented record command uses a bare `python`, which is miniforge base "
        f"and cannot import lerobot — use {ENV_PYTHON}:\n  " + "\n  ".join(offenders)
    )


BASE_PYTHON = "/opt/homebrew/Caskroom/miniforge/base/bin/python"


def test_the_wrapper_says_which_interpreter_to_use_when_lerobot_is_missing():
    """RUN it under the wrong interpreter and read what it prints.

    The first version of this test grepped the source for "ModuleNotFoundError"
    and ENV_PYTHON. It passed while the check was DEAD CODE: the guard sat
    inside main(), but `install()` runs at module import and imports
    lerobot.utils.keyboard_input itself, so the real failure was always a naked
    traceback from arrow_guard.py:53. Grepping proved the words were present,
    not that they were ever printed — the same bug class as everything else
    fixed tonight. Execute it instead.
    """
    if not os.path.exists(BASE_PYTHON):
        pytest.skip(f"{BASE_PYTHON} not on this machine")
    r = subprocess.run([BASE_PYTHON, str(TOOLS / "record_arrow_safe.py"), "--help"],
                       capture_output=True, text=True, timeout=120)
    out = r.stdout + r.stderr
    assert r.returncode != 0, "the wrapper ran under an interpreter without lerobot"
    assert "Traceback" not in out, (
        f"a bare traceback reached the operator instead of the one-line fix:\n{out[-600:]}"
    )
    assert ENV_PYTHON in out, f"the error does not name the interpreter to use:\n{out[-600:]}"


def test_the_wrapper_still_runs_under_the_right_interpreter():
    """The control for the test above — otherwise 'it errors' proves nothing."""
    r = subprocess.run([ENV_PYTHON, str(TOOLS / "record_arrow_safe.py"), "--help"],
                       capture_output=True, text=True, timeout=300)
    assert "usage:" in (r.stdout + r.stderr), (r.stdout + r.stderr)[-600:]


def test_every_documented_record_command_carries_the_task_string():
    """17 of the first 20 v4 episodes have an EMPTY task string.

    Episodes 0-2 carry "Pick up the test tube and insert it into the rack";
    3-19 carry "". The run crashed after ep 2 (phantom arrow), was resumed, and
    the resume command in the Sep 4 handoff had no --dataset.single_task — so
    every episode recorded on a resume, yesterday and this morning, got the
    empty prompt. ACT ignores language; SmolVLA is language-conditioned and
    would train on "" then be evaluated on the real prompt — the pi0.5
    empty-prompt bug, baked into the training data this time. Found Sep 5.
    """
    root = TOOLS.parent
    import re
    offenders = []
    for rel in DOCS_WITH_RECORD_COMMANDS:
        path = root / rel
        if not path.exists():
            continue
        text = path.read_text()
        for block in re.findall(r"```bash\n(.*?)```", text, re.S):
            if "record_arrow_safe.py" not in block:
                continue
            if "--dataset.single_task=" not in block:
                offenders.append(rel)
    assert not offenders, (
        "a documented record command has no --dataset.single_task; every episode it "
        f"records gets an empty prompt: {offenders}"
    )


def test_no_left_mode_drops_only_the_rerecord_arrow():
    """Sep 5 evening: ep 25 re-recorded six times, ep 26 twice, each re-record
    4-5 s into the reset window. Dropping Left alone removes the damaging key
    (a forged re-record throws away a finished take) while Right still ends an
    episode and `r` still re-records deliberately."""
    k = _fake_kbi()
    arrow_guard.install(k, keys=frozenset({"left"}))
    seen = []
    d = k.create_key_listener(seen.append)
    for name in ("left", "right", "up", "down", "n", "r", "q", "esc"):
        d(name)
    assert seen == ["right", "up", "down", "n", "r", "q", "esc"], seen
    assert "left" not in k._PYNPUT_KEY_NAMES.values() and "right" in k._PYNPUT_KEY_NAMES.values()


def test_the_record_wrapper_honours_CAPSTONE_ARROW_MODE():
    for mode, want_left, want_right in (("live", True, True), ("no-left", False, True), ("filtered", False, False)):
        probe = _PROBE.format(tools=str(TOOLS), module="record_arrow_safe")
        r = subprocess.run([PYBIN, "-c", probe], capture_output=True, text=True, timeout=300,
                           env={**os.environ, "CAPSTONE_ARROW_MODE": mode})
        assert r.returncode == 0, (mode, r.stderr[-800:])
        dispatched = r.stdout.split("DISPATCHED=")[1].splitlines()[0]
        assert ("'left'" in dispatched) == want_left, (mode, dispatched)
        assert ("'right'" in dispatched) == want_right, (mode, dispatched)
        assert "'n'" in dispatched and "'r'" in dispatched and "'esc'" in dispatched, (mode, dispatched)


def test_an_unknown_arrow_mode_refuses_to_start():
    r = subprocess.run([PYBIN, str(TOOLS / "record_arrow_safe.py"), "--help"], capture_output=True, text=True,
                       timeout=120, env={**os.environ, "CAPSTONE_ARROW_MODE": "sometimes"})
    assert r.returncode != 0 and "CAPSTONE_ARROW_MODE" in (r.stdout + r.stderr)
