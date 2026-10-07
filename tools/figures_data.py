"""Source-verified numbers behind the write-up figures. Pure parsing, no plotting.

Every number a figure shows is computed here from the primary record on disk:
the v2 trial sheet, the v2/v3 dataset parquet, and the v3 score logs. The test
file pins these against the numbers already published in the the lab notebook (private), so a
figure can never drift from the log.
"""

import glob
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path("~/.cache/huggingface/lerobot/faithqin").expanduser()
V2 = "so101-tube-insert-v2_20260816_035314"
V3 = "so101-tube-insert-v3_20260827_110242"

STAGES = ["R", "G", "T", "S"]          # cumulative; Rel is implied by S and not scored separately
STAGE_ORDER = {s: i for i, s in enumerate(STAGES)}

SEAT_PX = (534, 464)                   # front-camera seat, the lab notebook (private) Aug 30 / seat_check region
START_PX = (363, 356)                  # the D mark at frame 0, placement-gate readings Aug 27-30


def v2_trials(path=ROOT / "Eval Trials — v2 paired blocks.md"):
    """The 40 scored v2 trials (INFRA re-runs excluded, the re-run kept)."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.startswith("|"):
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) < 12 or not re.match(r"^\d+b?$", c[0]):
            continue
        trial, _block, policy, _n, _start, stage_live, _stage_video, mech, _notes, _run, close, _home = c[:12]
        if mech.startswith("INFRA"):
            continue
        m = re.search(r"lift ([+-]?\d+\.\d+), (\d+\.\d+)° (high|low)", close)
        off = float(m.group(2)) * (1 if m.group(3) == "high" else -1) if m else None
        stage = stage_live.split(" ")[0]
        rows.append(dict(trial=trial, policy=policy, stage=stage, mechanism=mech,
                         close_offset_deg=off, success=(stage == "S")))
    return rows


def stage_cumulative(rows):
    """Fraction of each arm's trials that reached at least each stage."""
    out = {}
    for pol in ("ACT-A", "ACT-B"):
        sub = [r for r in rows if r["policy"] == pol]
        n = len(sub)
        out[pol] = {"n": n}
        for s in STAGES:
            out[pol][s] = sum(1 for r in sub if STAGE_ORDER[r["stage"]] >= STAGE_ORDER[s]) / n
    return out


def episode_durations(repo):
    files = sorted(glob.glob(str(CACHE / repo / "data" / "chunk-*" / "file-*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet under {CACHE / repo}")
    df = pd.concat(pd.read_parquet(f, columns=["timestamp", "episode_index"]) for f in files)
    return df.groupby("episode_index")["timestamp"].agg(lambda t: float(t.max() - t.min())).sort_index()


def duration_summary(d):
    med = float(d.median())
    return dict(n=int(len(d)), median=med, p90=float(d.quantile(0.9)), max=float(d.max()),
                long_tail=int((d > 1.5 * med).sum()))


def classify_policy(policy: str):
    """Map a trials.csv policy string to the ACT arm it belongs to, or None.

    Returns "ACT-A", "ACT-B", or None for anything that is not an ACT A/B
    trial (pi0.5, SmolVLA, or an unrecognised name). Returning None rather
    than defaulting is the whole point: the previous rule was
    `"ACT-B" if "act-tube-B" in policy else "ACT-A"`, which silently labelled
    every non-ACT policy as ACT-A. The first pi0.5 probe (P05-01, Sep 3 2026)
    therefore entered the v3 scatter as a ninth ACT-A point. A policy this
    function does not recognise is excluded, never guessed.
    """
    if "act-tube-B" in policy:
        return "ACT-B"
    if "act-tube-A" in policy:
        return "ACT-A"
    return None


def policy_generation(policy: str):
    """The dataset generation a trials.csv policy string was trained on, or None.

    Sep 6 2026: the SAME failure the classifier above documents, one level up. The v4 bench
    writes into the SAME `trials.csv`, so the first v4 trial of the day (T0-SMOKE,
    `faithqin/act-tube-A-v4`) entered the v3 scatter as a ninth ACT-A point -- it is an ACT-A
    policy, and nothing here asked which generation it came from. 56 more trials were queued
    behind it. A generation this function does not recognise is excluded, never guessed.
    """
    for gen in ("v4", "v3", "v2"):
        if f"-{gen}" in policy:
            return gen
    return None


def v3_final_positions(csv=ROOT / "tools/scored_logs/trials.csv", logs=ROOT / "tools/scored_logs",
                       generation="v3"):
    """Final cap position of every valid ACT A/B trial OF ONE GENERATION, from its own score log.

    Probe policies (pi0.5, SmolVLA) are excluded — fig4 is the pre-registered
    ACT comparison and the probes are not part of its fixed n. Other generations
    are excluded too: they share this ledger but not this figure.
    """
    out = []
    for row in pd.read_csv(csv).itertuples():
        if row.verdict in ("INFRA", "UNKNOWN"):
            continue
        if policy_generation(row.policy) != generation:
            continue                      # a different generation's trial, sharing the ledger
        score = logs / f"{row.label}_{row.stamp}.score"
        # The .score files are gitignored run artifacts (tools/scored_logs/*),
        # so they exist on the bench laptop and nowhere else. Skip a row whose
        # receipt is absent rather than raising -- otherwise every consumer,
        # including the pre-session test suite, dies on a fresh clone. Callers
        # that need completeness check len() against the expected count.
        if not score.exists():
            continue
        text = score.read_text()
        m = re.search(r"FINAL: cap=\((\d+), (\d+)\)", text)
        if not m:
            continue
        pol = classify_policy(row.policy)
        if pol is None:
            continue                      # probe, not part of the ACT A/B scatter
        out.append(dict(label=row.label, policy=pol,
                        x=int(m.group(1)), y=int(m.group(2)), seat_runs=int(row.seat_runs)))
    return out
