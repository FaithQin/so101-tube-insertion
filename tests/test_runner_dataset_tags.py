"""A runner must never guess which dataset a policy came from.

Sep 3 2026: every v3 rollout re-homed to the v2 pose because the wrapper
defaulted CAPSTONE_HOME_POSE to "v2" and nothing set it. Sep 4 night audit:
run_scored_trial.sh still resolves the dataset tag as `*v3* -> v3, else v2`,
so the first v4 policy (act-tube-A-v4) would gate on v3 support? No -- on v2,
silently, and re-home to v2. Unknown tags must ABORT, not default.
"""
import re
import subprocess

from conftest import TOOLS


def _case_block(path):
    src = (TOOLS / path).read_text()
    m = re.search(r'case "\$POLICY" in(.*?)esac', src, re.S)
    assert m, f"{path}: no DATASET_TAG case block"
    return m.group(1)


def _resolve_tag(policy: str):
    """RUN the runner's own case block under zsh. Returns (rc, resolved_tag).

    Sep 5 pre-push review: the previous version graded this branch with
    `"ABORT" in body` and `"DATASET_TAG=v2" not in body`. Neither required the
    branch to actually STOP the script, and the second forbade exactly one
    literal -- defaulting to any other tag was unconstrained. It enforced the
    word, not the behaviour, in a test whose own thesis is "gate, don't
    narrate". Executing the block is the only assertion that cannot be
    satisfied by prose.
    """
    script = "\n".join([
        "say() { : }",          # the runner's TTS helper, stubbed
        'POLICY="$1"',
        # _case_block returns the BODY; re-wrap it in its own case/esac
        'case "$POLICY" in' + _case_block("run_scored_trial.sh") + "esac",
        'print -r -- "RESOLVED=${DATASET_TAG-<unset>}"',
    ])
    r = subprocess.run(["/bin/zsh", "-c", script, "zsh", policy],
                       capture_output=True, text=True, timeout=30)
    m = re.search(r"^RESOLVED=(.*)$", r.stdout, re.M)
    return r.returncode, (m.group(1) if m else None)


def test_scored_runner_resolves_the_tags_it_knows():
    for policy, want in [("act-tube-A-v3", "v3"), ("act-tube-B-v3", "v3"),
                         ("act-tube-A-v2", "v2"), ("smolvla-tube-B-v3", "v3"),
                         ("act-tube-A-v4", "v4"), ("act-tube-B-v4", "v4")]:   # v4 tables landed Sep 6
        rc, tag = _resolve_tag(policy)
        assert rc == 0 and tag == want, f"{policy} resolved to {tag!r} (rc={rc}), expected {want}"


def test_scored_runner_ABORTS_on_an_unknown_dataset_tag():
    """A policy of a generation with no support table must stop the launch, not silently pick a
    pose. (v4 was the example here until its tables landed on Sep 6; v6 is the unknown now.)"""
    for policy in ("act-tube-A-v6", "smolvla-tube-A-v5", "act-tube-A", "pi05-tube-A-v10"):
        rc, tag = _resolve_tag(policy)
        assert rc != 0, (
            f"{policy} did NOT abort (rc=0, DATASET_TAG={tag!r}) -- an unrecognised policy "
            f"silently picks a home pose and a preflight support window"
        )
        assert tag in (None, "<unset>"), (
            f"{policy} aborted but still assigned DATASET_TAG={tag!r} -- the branch must "
            f"exit before any assignment, or a later `set -e`-free path could use it"
        )


def test_the_default_branch_is_the_one_being_graded():
    r"""^\s*\*\) -- the DEFAULT branch only.

    Without the line anchor this matches the "*)" inside "*v3*)" and silently
    grades the wrong branch (caught Sep 4). Kept as a structural check beside
    the executing ones so a same-line default is still reported clearly.
    """
    block = _case_block("run_scored_trial.sh")
    default = re.search(r"^\s*\*\)\s*(.*?);;", block, re.S | re.M)
    assert default, "no default branch (or it is written on one line — split it)"
    assert re.search(r"\bexit\s+[1-9]", default.group(1)), (
        "the default branch does not exit non-zero -- it narrates instead of gating"
    )
    assert "DATASET_TAG=" not in default.group(1), (
        "the default branch assigns a DATASET_TAG -- unknown policies must abort, not default"
    )
    assert re.search(r"\*v2\*\)\s*DATASET_TAG=v2", block), "v2 must be matched explicitly, not by default"
    assert re.search(r"\*v3\*\)\s*DATASET_TAG=v3", block), "v3 pattern missing"


def test_probe_runner_pins_the_home_pose_or_is_retired():
    raw = (TOOLS / "run_probe_episode.sh").read_text()
    # code only -- a comment mentioning the variable must not satisfy the pin,
    # and it must be EXPORTed to reach the wrapper's child process (Sep 5 review).
    code = "\n".join(line.split("#")[0] for line in raw.splitlines())
    pinned = bool(re.search(r"^\s*export\s+CAPSTONE_HOME_POSE=\S", code, re.M))
    retired = "RETIRED" in "\n".join(raw.splitlines()[:5])
    assert pinned or retired, (
        "run_probe_episode.sh launches the rollout wrapper without pinning CAPSTONE_HOME_POSE "
        "-- it re-homes to v2 for every probe. Pin it or mark the script RETIRED in its header."
    )
