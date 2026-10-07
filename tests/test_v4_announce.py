"""The announcer maps recorder log lines to what the operator must hear."""
import sys

from conftest import TOOLS

sys.path.insert(0, str(TOOLS))
import v4_announce  # noqa: E402
from v4_manifest import Manifest  # noqa: E402


def _m():
    m = Manifest(["CLEAN", "RESIST", "CLEAN", "NUDGE"])
    m.void(1, "x")
    m.replan_tail(recorded=3)          # 3 -> RESIST (redo), 4 -> NUDGE
    return m


def test_recording_line_speaks_the_manifest_type_not_the_schedule_row():
    m = _m()
    u = v4_announce.utterance("INFO ... ls/utils.py:147 Recording episode 3", m)
    assert u.startswith("episode 3. resist"), u


def test_a_type_missing_from_the_manifest_says_stop():
    u = v4_announce.utterance("Recording episode 99", _m())
    assert "NOT IN THE MANIFEST" in u and "stop" in u


def test_rerecord_and_noise_lines():
    m = _m()
    assert v4_announce.utterance("INFO Re-record episode", m) == "re-recording."
    assert v4_announce.utterance("INFO OpenCVCamera(1) connected.", m) is None


def test_every_type_has_a_spoken_form():
    for t in ("CLEAN", "RESIST", "NUDGE", "LEFT-MIMIC", "ROT-20°", "ROT-45°"):
        assert t in v4_announce.SPOKEN and v4_announce.SPOKEN[t]


def test_follow_skips_the_stale_log_and_resets_on_truncation(tmp_path):
    """Startup must not replay an old run; a new run (file replaced) must be seen."""
    import itertools
    log = tmp_path / "v4rec.out"
    log.write_text("Recording episode 11\nRecording episode 12\n")
    gen = v4_announce.follow(log)
    # append a NEW line and take exactly one item (the generator sleeps between polls)
    log.write_text(log.read_text() + "Recording episode 13\n")
    assert next(gen) == "Recording episode 13"
    # new run: the bridge replaces the file with a shorter one
    log.write_text("Recording episode 20\n")
    assert next(gen) == "Recording episode 20"
