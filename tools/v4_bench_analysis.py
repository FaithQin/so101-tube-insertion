#!/usr/bin/env python
"""Paired analysis of the v4 bench (A = vision + 6 pos, B = + 6 load + 6 current), from the
trial ledger, the pair schedule and the adjudicated scores. Statistics by hand; no scipy.

    python tools/v4_bench_analysis.py --ledger tools/scored_logs/trials.csv \\
        --pairs analysis/bench_sep6/pairs.csv --scores analysis/bench_sep6/scores.csv \\
        [--schedule analysis/bench_sep6/schedule.csv] \\
        [--invalidations analysis/bench_sep6/invalidations.csv] [--out report.txt]

Why this exists (Sep 6 2026, handoff §3 item 5). The merged protocol (handoff §6) names the
endpoints and the tests: co-primary 1 the graded stage (R/G/T/Rel/S -> 0-5) by paired Wilcoxon +
Hodges-Lehmann with a pair-clustered bootstrap CI; co-primary 2 the contact response (latency,
impulse) restricted to trials reaching T; key secondary the binary SEAT (`held_to_end`) by mid-p
McNemar with Wilson CIs and "MDE printed beside it"; the grasp stage as negative control; RMST to
seat at 45 s; retries, guard fires, the paired failure-mechanism shift table; and "exhaustive
invalidation codes; disposition table published". scipy is not importable in the lerobot env
(checked Sep 6 02:xx by the orchestrator: ModuleNotFoundError) and installing anything into that
env is a one-way door -- five local site-packages patches die on any environment change
(the project notes "Money / accounts") -- so every routine here is written from the definition and
tested against a known answer in tests/test_v4_bench_analysis.py: hand-enumerated small cases,
planted-effect recovery, and true-null rejection rates. A statistical routine that has never
been run against a known answer is a threshold without a control (tools/v4_audit.py:29-41).

Modelling choices that belong in the pre-registration's language:
  * Wilcoxon: zero differences are DROPPED (Wilcoxon's rule), ties get average ranks. n <= 25
    uses the exact conditional null (a DP over achievable signed-rank sums on doubled ranks --
    exact with ties, no 2^n enumeration); above that the normal approximation with the tie
    correction sum(t^3 - t)/48. The result says which branch ran.
  * Hodges-Lehmann: median of the n(n+1)/2 Walsh averages; its CI cuts the k smallest and k
    largest Walsh averages where k is the largest integer with P(W+ <= k) <= alpha/2 under the
    exact untied null, so the attained confidence is reported, not the nominal one.
  * mid-p McNemar on the discordant pairs, X = #(B seated, A did not) ~ Bin(n_d, 1/2):
    one-sided mid-p = P(X > k) + 0.5 P(X = k); TWO-SIDED = 2 * min(lower, upper), capped at 1.
    The uncorrected exact p is printed beside it.
  * Wilson score intervals per arm; the paired risk difference gets the pair bootstrap CI.
  * MDE for the binary: Connor (1987) closed form for the paired test, alpha 0.05 two-sided,
    80 % power, at the OBSERVED discordance and at the pre-stated 0.30 / 0.50. Handoff §6
    states 0.28-0.37 at 28 pairs; the test reproduces that. Below 24 pairs at 0.30 it has no
    solution, and the report says so in words, never "nan". Beside it, always, the null 95 % CI
    half-width 1.96 * sqrt(psi / n) -- finite at every n >= 1, and the X of the protocol's only
    allowed null sentence (Eval Protocol -- v4 merged (Sep 6).md:76-77, :103-104).
  * RMST to seat at tau = 45 s: Kaplan-Meier on time-to-seat, where the seat time is the onset
    of the run that is HELD to the episode end (a seat that is lost is not a seat). A trial that
    never seats is censored at tau; a trial whose episode ended early without a seat (guard
    fire, elbow dropout) is censored at its end time. Lower RMST = seats sooner.
  * Bootstrap: the unit of resampling is the PAIR. `pair_bootstrap` takes an (n, 2) array and
    resamples its rows, so resampling trials instead of pairs is not expressible.
  * The analysis REFUSES to run with zero valid pairs (exit 2). An empty, well-formatted table
    the night after a bench day is how a session gets misreported. It still prints -- and writes
    to --out -- the two things the protocol publishes whatever happens: the disposition table
    (Eval Protocol -- v4 merged (Sep 6).md:68, :308) and the power statement at n = 0. No
    endpoint header is printed.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# --- the ledger and the receipts (tools/run_scored_trial.sh:62-63, :73-74, :135, :143, :158, :161)
LEDGER_COLUMNS = ["label", "policy", "n", "lock", "stamp", "verdict", "dur_s", "seat_runs"]
# run_smolvla_trial.sh keeps its own ledger (trials_smolvla.csv; figures_data.py:83 would file
# its rows as ACT-A) with a wider header; its rows are folded into the same disposition
# (Sep 6 2026) -- `inference` stands where `lock` does for ACT / pi0.5.
SMOLVLA_LEDGER = "trials_smolvla.csv"
SMOLVLA_LEDGER_COLUMNS = ["label", "side", "policy", "n", "inference", "force", "stamp", "verdict", "dur_s", "seat_runs"]
RECEIPTS = (".home", ".ping", ".scene", ".log", ".verify", ".score")
PAIR_COLUMNS = ["pair_id", "stratum", "arm", "label"]
# The committed seeded schedule (tools/v4_schedule.py COLUMNS, pinned in the tests) -- the universe
# of pairs. pairs.csv holds only the pairs run_pair.sh LAUNCHED (it seals a pair's rows just before
# its first trial, run_pair.sh:182-200), so a pair cut for time is in the schedule and nowhere else.
# Default = the file run_pair.sh:34 launches from. Read for pair_id / stratum / cut_rank only.
SCHEDULE_COLUMNS = ["pair_id", "play_index", "block", "family", "role", "stratum", "arm_order",
                    "token_first", "token_second", "cut_rank"]
DEFAULT_SCHEDULE = Path(__file__).resolve().parent.parent / "analysis" / "bench_sep6" / "schedule.csv"
SCORE_COLUMNS = ["label", "stage", "mechanism", "seat_time_s", "retries", "guard_fires",
                 "contact_latency_s", "contact_impulse", "reached_T"]
# Sep 14 2026 (protocol §7 13:55): co-primary 2 was withdrawn, so the three contact columns are optional.
SCORE_COLUMNS_REQUIRED = SCORE_COLUMNS[:6]

# Write-up -- Results Skeleton.md:108: "Stage x mechanism taxonomy: R/G/T/Rel/S x NG/PECK/SLIP/DT/JAM/KR/INFRA"
STAGES = {"NONE": 0, "R": 1, "G": 2, "T": 3, "Rel": 4, "S": 5}
MECHANISMS = ("OK", "NG", "PECK", "SLIP", "DT", "JAM", "KR", "INFRA")

# Exhaustive: every trial and pair disposition is one of these (MANUAL carries a MANUAL_CODES suffix).
INVALIDATION_CODES = (
    "INFRA_START_SEATED",   # ledger verdict INFRA: the scorer saw a seat inside the start window
    "UNSCOREABLE",          # ledger verdict UNKNOWN (no cap found / zero frames) or anything not SEAT/MISS
    "MISSING_RECEIPT",      # one of .home .ping .scene .log .verify .score is absent for this label_stamp
    "DUPLICATE_LABEL",      # the label was run more than once; which run counts is a manual call
    "NOT_IN_SCHEDULE",      # ledger row whose label is not in the pair schedule
    "ARM_UNKNOWN",          # the policy name does not say -A- or -B-
    "ARM_MISMATCH",         # schedule says A, policy name says B (or vice versa)
    "N_PARITY",             # the two trials of a pair ran with different n_action_steps
    "NOT_RUN",              # a scheduled pair with no ledger row for at least one arm
    "UNPAIRED",             # this trial is fine but its partner is not: pairs are atomic
    "MANUAL",               # operator invalidation from invalidations.csv (MANUAL:<code>)
)
MANUAL_CODES = ("HAND_IN_WORKSPACE", "OPERATOR_TOUCH", "POWER_CYCLE_BROKEN", "ELBOW_DROPOUT",
                "CAMERA_MOVED", "SCENE_DRIFT", "REPLAY_ARBITER_FAILED", "TELEMETRY_MISSING",
                "TELEMETRY_CONSTANT", "OTHER")
_ARM_RE = re.compile(r"-([AB])-v\d")


class NoValidPairs(RuntimeError):
    """Raised instead of producing an empty table."""


# --- normal distribution ------------------------------------------------------

def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def norm_ppf(p: float) -> float:
    """Inverse normal CDF: Acklam's rational approximation (|rel err| < 1.2e-9) plus one Halley
    step on erfc, which takes it to machine precision. Tested against z(.975) = 1.959964."""
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0, 1), got {p}")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p <= 1 - plow:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    e = norm_cdf(x) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


# --- Wilcoxon signed-rank -----------------------------------------------------

def _average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    ranks = np.empty(len(x), float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _signed_rank_null_counts(doubled_ranks) -> np.ndarray:
    """counts[s] = number of sign assignments whose doubled W+ equals s. Exact, O(n * sum)."""
    total = int(sum(doubled_ranks))
    counts = np.zeros(total + 1, dtype=object)
    counts[0] = 1
    for r in doubled_ranks:
        r = int(r)
        new = counts.copy()
        new[r:] += counts[:total + 1 - r]
        counts = new
    return counts


def _p_le(counts: np.ndarray, k2: int, n: int) -> float:
    """P(doubled W+ <= k2) under the null with n signs."""
    if k2 < 0:
        return 0.0
    return float(sum(counts[:k2 + 1]) / (2 ** n))


def wilcoxon_signed_rank(d, exact_max_n: int = 25) -> dict:
    """Two-sided paired Wilcoxon on differences d. Zeros dropped; average ranks for ties.
    n_used <= exact_max_n -> exact conditional null; else tie-corrected normal approximation."""
    d = np.asarray(d, float)
    nz = d[d != 0]
    n_zeros = int(len(d) - len(nz))
    n = int(len(nz))
    if n == 0:
        raise ValueError("every difference is zero: nothing to rank")
    ranks = _average_ranks(np.abs(nz))
    w_plus = float(ranks[nz > 0].sum())
    _, tie_sizes = np.unique(np.abs(nz), return_counts=True)
    tie_corr = float(sum(t ** 3 - t for t in tie_sizes) / 48.0)
    out = {"n_used": n, "n_zeros": n_zeros, "w_plus": w_plus, "w_minus": float(n * (n + 1) / 2 - w_plus),
           "tie_correction": tie_corr}
    if n <= exact_max_n:
        doubled = np.rint(2 * ranks).astype(int)
        counts = _signed_rank_null_counts(doubled)
        w2 = int(round(2 * w_plus))
        total = int(doubled.sum())
        p_le = _p_le(counts, w2, n)
        p_ge = _p_le(counts, total - w2, n)          # symmetry of the null
        out.update(method="exact", p=min(1.0, 2 * min(p_le, p_ge)))
    else:
        mean = n * (n + 1) / 4.0
        var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_corr
        z = (w_plus - mean) / math.sqrt(var)
        out.update(method="normal-approx (tie-corrected, zeros dropped)", z=z, p=2 * (1 - norm_cdf(abs(z))))
    return out


# --- Hodges-Lehmann -----------------------------------------------------------

def hodges_lehmann(d, conf: float = 0.95) -> dict:
    """Median of the Walsh averages (d_i + d_j)/2, i <= j, with the distribution-free CI from
    the exact signed-rank null on untied ranks 1..n."""
    d = np.asarray(d, float)
    n = len(d)
    if n == 0:
        raise ValueError("no differences")
    i, j = np.triu_indices(n)
    walsh = np.sort((d[i] + d[j]) / 2.0)
    m = len(walsh)
    est = float(np.median(walsh))
    counts = _signed_rank_null_counts(range(1, n + 1))
    alpha = 1 - conf
    k = -1
    while k + 1 <= m and _p_le(counts, k + 1, n) <= alpha / 2:
        k += 1
    if k < 0:
        ci, attained = (-math.inf, math.inf), 1.0
    else:
        ci, attained = (float(walsh[k]), float(walsh[m - 1 - k])), 1 - 2 * _p_le(counts, k, n)
    return {"estimate": est, "ci": ci, "conf_nominal": conf, "conf_attained": attained,
            "n": n, "n_walsh": m, "method": "median of Walsh averages; exact signed-rank CI"}


# --- mid-p McNemar, Wilson, MDE -------------------------------------------------

def midp_mcnemar(b: int, c: int) -> dict:
    """b = pairs where A seated and B did not; c = pairs where B seated and A did not.
    X = c ~ Bin(b + c, 1/2) under H0. See the module docstring for the two-siding rule."""
    n = b + c
    rule = "2*min(mid-p lower, mid-p upper), capped at 1"
    if n == 0:
        return {"n_discordant": 0, "b": b, "c": c, "p": 1.0, "p_one_sided_b_better": 0.5,
                "p_exact_uncorrected": 1.0, "two_sided_rule": rule}
    pmf = [math.comb(n, k) / 2 ** n for k in range(n + 1)]
    upper = sum(pmf[c + 1:]) + 0.5 * pmf[c]      # P(X > c) + .5 P(X = c): evidence B better
    lower = sum(pmf[:c]) + 0.5 * pmf[c]
    ex_upper, ex_lower = sum(pmf[c:]), sum(pmf[:c + 1])
    return {"n_discordant": n, "b": b, "c": c,
            "p": min(1.0, 2 * min(upper, lower)),
            "p_one_sided_b_better": upper,
            "p_exact_uncorrected": min(1.0, 2 * min(ex_upper, ex_lower)),
            "two_sided_rule": rule}


def wilson_interval(x: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("n must be positive")
    z = norm_ppf(1 - (1 - conf) / 2)
    p = x / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_power(n_pairs: int, discordance: float, delta: float, alpha: float = 0.05) -> float:
    """Connor (1987): power of the paired test for risk difference delta when a fraction
    `discordance` of pairs is discordant. Requires 0 < |delta| < discordance."""
    if not (0 < discordance <= 1):
        raise ValueError("discordance must be in (0, 1]")
    if not (0 < abs(delta) < discordance):
        raise ValueError("|delta| must be in (0, discordance)")
    za = norm_ppf(1 - alpha / 2)
    zb = (math.sqrt(n_pairs) * abs(delta) - za * math.sqrt(discordance)) / math.sqrt(discordance - delta ** 2)
    return norm_cdf(zb)


def mcnemar_mde(n_pairs: int, discordance: float, power: float = 0.80, alpha: float = 0.05) -> float:
    """Smallest paired risk difference detectable with `power` at `n_pairs` (bisection on Connor)."""
    if not (0 < discordance <= 1):
        raise ValueError("discordance must be in (0, 1]")
    lo, hi = 1e-6, discordance - 1e-9
    if mcnemar_power(n_pairs, discordance, hi, alpha) < power:
        return float("nan")                      # not detectable at any delta < discordance
    for _ in range(100):
        mid = (lo + hi) / 2
        if mcnemar_power(n_pairs, discordance, mid, alpha) < power:
            lo = mid
        else:
            hi = mid
    return hi


def null_ci_halfwidth(n_pairs: int, discordance: float, conf: float = 0.95) -> float:
    """Half-width of the CI of the paired risk difference B - A under the null: z * sqrt(psi / n),
    because Var(p_B - p_A) = (psi - delta^2) / n and delta = 0 under the null. It is the X in the
    protocol's only allowed null sentence, "we can rule out an improvement larger than X pp at 95 %"
    (Eval Protocol -- v4 merged (Sep 6).md:76-77, :103-104), and unlike Connor's MDE it exists at
    every n >= 1. Reproduces the protocol's own table (:134-135): +/-0.203 / +/-0.262 at 28 pairs,
    +/-0.240 / +/-0.310 at 20, psi = 0.30 / 0.50 (added Sep 12 2026, night audit Sep 7, :607)."""
    if n_pairs < 1:
        raise ValueError("no pairs: there is no bound to state")
    if not (0 < discordance <= 1):
        raise ValueError("discordance must be in (0, 1]")
    return norm_ppf(1 - (1 - conf) / 2) * math.sqrt(discordance / n_pairs)


# `Eval Protocol -- v4 merged (Sep 6).md:73-74`: the discordances its power statement is made at.
PRESTATED_DISCORDANCE = (0.30, 0.50)


def power_at(n_pairs: int, observed_discordance: float | None = None) -> dict:
    """The binary's power statement at the pool actually analysed: Connor's 80 %-power MDE and the
    null 95 % CI half-width, at the observed discordance (when there is one) and the pre-stated
    0.30 / 0.50. NaN = no number exists (MDE undetectable below the discordance, or no discordant
    pairs); inf = a half-width with no pairs behind it. `_power_lines` puts a sentence in their place."""
    def half(psi):
        return null_ci_halfwidth(n_pairs, psi) if n_pairs >= 1 else math.inf

    out = {"power": 0.80, "alpha": 0.05, "n_pairs": n_pairs, "observed_discordance": observed_discordance,
           "null_ci_halfwidth_95": {}}
    if observed_discordance is not None:
        has = observed_discordance > 0
        out["at_observed_discordance"] = mcnemar_mde(n_pairs, observed_discordance) if has else float("nan")
        out["null_ci_halfwidth_95"]["at_observed_discordance"] = half(observed_discordance) if has else float("nan")
    for psi in PRESTATED_DISCORDANCE:
        out[f"at_{psi:.2f}"] = mcnemar_mde(n_pairs, psi)
        out["null_ci_halfwidth_95"][f"at_{psi:.2f}"] = half(psi)
    return out


def paired_binary(a, b, conf: float = 0.95) -> dict:
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be aligned 1-d arrays, one entry per pair")
    n = len(a)
    bb, cc = int(np.sum(a & ~b)), int(np.sum(b & ~a))
    return {"n_pairs": n, "seat_A": int(a.sum()), "seat_B": int(b.sum()),
            "rate_A": float(a.mean()), "rate_B": float(b.mean()),
            "wilson_A": wilson_interval(int(a.sum()), n, conf), "wilson_B": wilson_interval(int(b.sum()), n, conf),
            "risk_difference": float(b.mean() - a.mean()),
            "discordance": (bb + cc) / n, "mcnemar": midp_mcnemar(bb, cc)}


# --- RMST ---------------------------------------------------------------------

def seat_time_from_score(r: dict) -> float:
    """Onset of the seat run held to the episode end (tools/score_episode.py:95-104); NaN otherwise."""
    if r.get("held_to_end") and r.get("seat_runs"):
        return float(r["seat_runs"][-1][0])
    return float("nan")


def rmst_time_to_seat(seat_times, tau: float = 45.0, censor_times=None) -> dict:
    """Kaplan-Meier restricted mean time-to-seat on [0, tau]. NaN = no seat (censored at tau, or
    at censor_times[i] if the episode ended earlier)."""
    t = np.asarray(seat_times, float)
    if np.any(t[~np.isnan(t)] > tau):
        raise ValueError(f"a seat time exceeds tau={tau}; it is not an observation on [0, tau]")
    if np.any(t[~np.isnan(t)] < 0):
        raise ValueError("negative seat time")
    event = ~np.isnan(t)
    end = np.full(len(t), float(tau))
    if censor_times is not None:
        for k, ct in enumerate(censor_times):
            if ct is not None and not (isinstance(ct, float) and math.isnan(ct)):
                end[k] = min(float(ct), tau)
    time = np.where(event, t, end)
    order = np.argsort(time, kind="mergesort")
    time, event = time[order], event[order]
    s, t_prev, rmst = 1.0, 0.0, 0.0
    curve = [(0.0, 1.0)]
    for tj in np.unique(time[event]):
        at_risk = int(np.sum(time >= tj))
        d = int(np.sum(event & (time == tj)))
        rmst += s * (tj - t_prev)
        s *= 1 - d / at_risk
        t_prev = tj
        curve.append((float(tj), s))
    rmst += s * (tau - t_prev)
    return {"rmst": float(rmst), "tau": tau, "n": int(len(t)), "n_events": int(event.sum()),
            "n_censored": int((~event).sum()), "survival": curve, "method": "Kaplan-Meier, censor at tau"}


# --- pair-clustered bootstrap ---------------------------------------------------

def pair_bootstrap(pairs, statistic, n_boot: int = 2000, seed: int = 0, conf: float = 0.95) -> dict:
    """Percentile CI of statistic(pairs) resampling ROWS of the (n_pairs, 2) array `pairs`.
    The A/B pairing inside a row can never be broken by this function."""
    pairs = np.asarray(pairs, float)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"pairs must be an (n_pairs, 2) array [A, B] per row, got shape {pairs.shape}")
    n = pairs.shape[0]
    if n == 0:
        raise ValueError("no pairs")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    reps = np.array([statistic(pairs[i]) for i in idx], float)
    reps = reps[~np.isnan(reps)]
    alpha = 1 - conf
    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": float(statistic(pairs)), "ci": (float(lo), float(hi)), "n_boot": n_boot,
            "n_replicates_finite": int(len(reps)), "n_pairs": n, "seed": seed, "unit": "pair"}


# --- disposition ----------------------------------------------------------------

def _read_csv(path: Path, columns) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
        header = rows[0].keys() if rows else csv.DictReader(open(path, newline="")).fieldnames
    header = list(header or [])
    if header != list(columns):
        raise ValueError(f"{path}: header {header} != expected {list(columns)}")
    return rows


def read_scores(path) -> list[dict]:
    """The adjudicated scores: the nine-column header, or the six-column one without the withdrawn
    contact endpoint's columns (protocol §7, Sep 14 13:55). Missing optional cells read as blank."""
    path = Path(path)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    if header not in (list(SCORE_COLUMNS), list(SCORE_COLUMNS_REQUIRED)):
        raise ValueError(f"{path}: header {header} != {list(SCORE_COLUMNS)} or {list(SCORE_COLUMNS_REQUIRED)}")
    for r in rows:
        for c in SCORE_COLUMNS:
            r.setdefault(c, "")
    return rows


def _arm_from_policy(policy: str):
    m = _ARM_RE.search(policy)
    return m.group(1) if m else None


def disposition(logs_dir, pairs_csv, invalidations_csv=None, schedule_csv=None) -> dict:
    """Every ledger row and every scheduled pair gets its codes; a pair is valid only when both
    trials carry none. Receipts are looked up as <logs_dir>/<label>_<stamp><ext>. With
    `schedule_csv`, every schedule pair that never reached pairs.csv is NOT_RUN (launched=False),
    and a pairs.csv pair the schedule does not hold is refused as the wrong schedule."""
    logs_dir = Path(logs_dir)
    ledger = _read_csv(logs_dir / "trials.csv", LEDGER_COLUMNS)
    if (logs_dir / SMOLVLA_LEDGER).exists():
        for r in _read_csv(logs_dir / SMOLVLA_LEDGER, SMOLVLA_LEDGER_COLUMNS):
            ledger.append({**{c: r[c] for c in LEDGER_COLUMNS if c in r}, "lock": r["inference"]})
    sched = _read_csv(Path(pairs_csv), PAIR_COLUMNS)
    manual = {}
    if invalidations_csv is not None:
        for r in _read_csv(Path(invalidations_csv), ["label", "stamp", "code", "note"]):
            if r["code"] not in MANUAL_CODES:
                raise ValueError(f"unknown invalidation code {r['code']!r}; allowed: {MANUAL_CODES}")
            manual.setdefault((r["label"], r["stamp"]), []).append(r["code"])

    pairs: dict[str, dict] = {}
    label_to_pair: dict[str, tuple[str, str]] = {}
    for r in sched:
        if r["arm"] not in ("A", "B"):
            raise ValueError(f"schedule arm must be A or B: {r}")
        p = pairs.setdefault(r["pair_id"], {"stratum": r["stratum"], "labels": {}, "A": None, "B": None, "codes": []})
        if r["arm"] in p["labels"]:
            raise ValueError(f"pair {r['pair_id']} lists arm {r['arm']} twice")
        p["labels"][r["arm"]] = r["label"]
        label_to_pair[r["label"]] = (r["pair_id"], r["arm"])
    for pid, p in pairs.items():
        if set(p["labels"]) != {"A", "B"}:
            raise ValueError(f"pair {pid} must schedule exactly one A and one B trial")
        p["launched"] = True

    # Until Sep 12 2026 the table was built from pairs.csv alone, so a pair cut for time never
    # appeared and "pairs scheduled" counted only the pairs that ran (night audit Sep 7, :396).
    planned: dict[str, dict] = {}
    if schedule_csv is not None:
        for r in _read_csv(Path(schedule_csv), SCHEDULE_COLUMNS):
            if r["pair_id"] in planned:
                raise ValueError(f"{schedule_csv}: pair {r['pair_id']} is listed twice")
            planned[r["pair_id"]] = r
        stray = sorted(set(pairs) - set(planned))
        if stray:
            raise ValueError(f"{pairs_csv} holds pair(s) {stray} that are not in the schedule {schedule_csv} "
                             "-- wrong --schedule? (run_pair.sh reads CAPSTONE_SCHEDULE when it is set)")

    label_counts = Counter(r["label"] for r in ledger)
    # A repeated label is a duplicate only among the rows NOT manually invalidated: a re-run whose
    # first run carries a MANUAL code is the run that counts (protocol §7, Sep 14 15:26). Two or
    # more surviving rows are still duplicates, all of them -- which one counts is then unresolved.
    surviving = Counter(r["label"] for r in ledger if not manual.get((r["label"], r["stamp"])))
    trials = []
    for r in ledger:
        codes = []
        if (label_counts[r["label"]] > 1 and surviving[r["label"]] > 1
                and not manual.get((r["label"], r["stamp"]))):
            codes.append("DUPLICATE_LABEL")
        pid_arm = label_to_pair.get(r["label"])
        if pid_arm is None:
            codes.append("NOT_IN_SCHEDULE")
        missing = [ext for ext in RECEIPTS if not (logs_dir / f"{r['label']}_{r['stamp']}{ext}").exists()]
        if missing:
            codes.append("MISSING_RECEIPT:" + "+".join(missing))
        if r["verdict"] == "INFRA":
            codes.append("INFRA_START_SEATED")
        elif r["verdict"] not in ("SEAT", "MISS"):
            codes.append("UNSCOREABLE")
        arm = _arm_from_policy(r["policy"])
        if arm is None:
            codes.append("ARM_UNKNOWN")
        elif pid_arm is not None and arm != pid_arm[1]:
            codes.append("ARM_MISMATCH")
        for code in manual.get((r["label"], r["stamp"]), []):
            codes.append(f"MANUAL:{code}")
        trials.append({**r, "pair_id": pid_arm[0] if pid_arm else None, "arm": pid_arm[1] if pid_arm else None,
                       "codes": codes})

    for pid, p in pairs.items():
        for arm in ("A", "B"):
            rows = [t for t in trials if t["label"] == p["labels"][arm]]
            counting = [t for t in rows if not any(c.startswith("MANUAL:") or c == "DUPLICATE_LABEL" for c in t["codes"])]
            if not rows and "NOT_RUN" not in p["codes"]:
                p["codes"].append("NOT_RUN")
            elif len(rows) == 1:
                p[arm] = rows[0]
            elif len(counting) == 1:
                p[arm] = counting[0]                 # the re-run that counts; the invalidated run stays listed
        if "NOT_RUN" in p["codes"]:
            # The half that DID run is coded too. Until Sep 12 2026 this `continue` came first, so
            # the only trial of a half-run pair (run_pair.sh:385-387 seals both rows, runs trial 1,
            # and the pair can stop there) carried no code and was invisible in a table the
            # protocol calls exhaustive (night audit Sep 7, :444). Same rule as below: UNPAIRED
            # only on a trial that is fine on its own; one with its own code keeps just that.
            for t in (p["A"], p["B"]):
                if t is not None and not t["codes"]:
                    t["codes"].append("UNPAIRED")
            continue
        ta, tb = p["A"], p["B"]
        if ta is not None and tb is not None and ta["n"] != tb["n"]:
            for t in (ta, tb):
                t["codes"].append("N_PARITY")
        partner_codes = {"A": tb["codes"] if tb is not None else ["DUPLICATE_LABEL"],
                         "B": ta["codes"] if ta is not None else ["DUPLICATE_LABEL"]}
        for arm, t in (("A", ta), ("B", tb)):
            if t is not None and not t["codes"] and partner_codes[arm]:
                t["codes"].append("UNPAIRED")
        seen = []
        for t in (ta, tb):
            for c in (t["codes"] if t is not None else ["DUPLICATE_LABEL"]):
                if c != "UNPAIRED" and c not in seen:
                    seen.append(c)
        p["codes"] = seen

    # An invalidation filed against a label that never wrote a ledger row (a rollout aborted before the
    # scorer ran: P04 / 7mzfkpy8, Sep 14 17:45) would otherwise vanish, and the pair would print a bare
    # NOT_RUN in a table the protocol calls exhaustive. Carry the code onto the pair.
    for pid, p in pairs.items():
        for arm in ("A", "B"):
            if p.get(arm) is None:
                for (lbl, _st), codes_ in manual.items():
                    if lbl == p["labels"].get(arm):
                        for c in codes_:
                            if f"MANUAL:{c}" not in p["codes"]:
                                p["codes"].append(f"MANUAL:{c}")

    for pid, r in planned.items():                   # never launched: in the schedule, not in pairs.csv
        if pid not in pairs:
            pairs[pid] = {"stratum": r["stratum"], "labels": {}, "A": None, "B": None, "codes": ["NOT_RUN"],
                          "launched": False, "cut_rank": r["cut_rank"]}

    valid = sorted(pid for pid, p in pairs.items() if not p["codes"])
    return {"trials": trials, "pairs": pairs, "valid_pairs": valid,
            "n_trials": len(trials), "n_pairs_scheduled": len(pairs), "n_pairs_valid": len(valid),
            "n_pairs_launched": sum(p["launched"] for p in pairs.values()),
            "trial_code_counts": Counter(c.split(":")[0] for t in trials for c in t["codes"]),
            "pair_code_counts": Counter(c.split(":")[0] for p in pairs.values() for c in p["codes"])}


# --- the analysis -----------------------------------------------------------------

def stage_to_int(s) -> int:
    if isinstance(s, str) and s in STAGES:
        return STAGES[s]
    try:
        v = int(s)
    except (TypeError, ValueError):
        raise ValueError(f"unknown stage {s!r}; use one of {list(STAGES)} or 0-5") from None
    if not 0 <= v <= 5:
        raise ValueError(f"stage {v} outside 0-5")
    return v


def _shift_block(d, seed, n_boot):
    """Wilcoxon + HL + pair bootstrap of the HL shift, for a vector of B - A differences given as pairs."""
    d = np.asarray(d, float)
    out = {"n": int(len(d)), "mean_diff": float(np.mean(d)), "hodges_lehmann": hodges_lehmann(d)}
    try:
        out["wilcoxon"] = wilcoxon_signed_rank(d)
    except ValueError as e:
        out["wilcoxon"] = {"method": "n/a", "p": float("nan"), "note": str(e)}
    return out


def _pairs_block(pairs_ab, seed, n_boot):
    pairs_ab = np.asarray(pairs_ab, float)
    d = pairs_ab[:, 1] - pairs_ab[:, 0]
    out = _shift_block(d, seed, n_boot)
    out["bootstrap"] = pair_bootstrap(pairs_ab, lambda p: hodges_lehmann(p[:, 1] - p[:, 0])["estimate"],
                                      n_boot=n_boot, seed=seed)
    return out


def analyze(rows, seed: int = 0, n_boot: int = 2000, tau: float = 45.0) -> dict:
    """rows: one dict per trial with pair_id, stratum, arm, label, stage, mechanism, seat,
    seat_time_s, retries, guard_fires, contact_latency_s, contact_impulse, reached_T."""
    by_pair: dict[str, dict] = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], {})[r["arm"]] = r
    pairs = {pid: p for pid, p in by_pair.items() if set(p) == {"A", "B"}}
    if not pairs:
        raise NoValidPairs("no valid pairs: refusing to produce a table from data that is not there")
    pids = sorted(pairs)
    A = [pairs[p]["A"] for p in pids]
    B = [pairs[p]["B"] for p in pids]
    for r in A + B:
        if r["mechanism"] not in MECHANISMS:
            raise ValueError(f"{r['label']}: mechanism {r['mechanism']!r} not in {MECHANISMS}")
    stage_a = np.array([stage_to_int(r["stage"]) for r in A])
    stage_b = np.array([stage_to_int(r["stage"]) for r in B])
    seat_a = np.array([bool(r["seat"]) for r in A])
    seat_b = np.array([bool(r["seat"]) for r in B])
    res = {"n_pairs": len(pids), "pair_ids": pids, "seed": seed, "n_boot": n_boot, "tau": tau,
           "strata": dict(Counter(pairs[p]["A"].get("stratum", "?") for p in pids))}

    res["graded_stage"] = _pairs_block(np.column_stack([stage_a, stage_b]), seed, n_boot)

    binary = paired_binary(seat_a, seat_b)
    binary["bootstrap"] = pair_bootstrap(np.column_stack([seat_a, seat_b]),
                                         lambda p: float(np.mean(p[:, 1]) - np.mean(p[:, 0])), n_boot=n_boot, seed=seed)
    disc = binary["discordance"]
    binary["mde_80"] = power_at(len(pids), disc)
    res["binary_seat"] = binary

    grasp_a, grasp_b = stage_a >= STAGES["G"], stage_b >= STAGES["G"]
    res["grasp_negative_control"] = paired_binary(grasp_a, grasp_b)

    st_a = np.array([float(r.get("seat_time_s", float("nan"))) for r in A])
    st_b = np.array([float(r.get("seat_time_s", float("nan"))) for r in B])
    for name, st, seat in (("A", st_a, seat_a), ("B", st_b, seat_b)):
        if np.any(seat & np.isnan(st)):
            raise ValueError(f"arm {name}: a seated trial has no seat_time_s")
        if np.any(~seat & ~np.isnan(st)):
            raise ValueError(f"arm {name}: a non-seated trial carries a seat_time_s")
    res["rmst_45s"] = {"A": rmst_time_to_seat(st_a, tau), "B": rmst_time_to_seat(st_b, tau)}
    res["rmst_45s"]["B_minus_A"] = pair_bootstrap(
        np.column_stack([st_a, st_b]),
        lambda p: rmst_time_to_seat(p[:, 1], tau)["rmst"] - rmst_time_to_seat(p[:, 0], tau)["rmst"],
        n_boot=n_boot, seed=seed)

    def reached_t(r, stage):
        v = r.get("reached_T")
        return bool(stage >= STAGES["T"]) if v in (None, "") else bool(int(v))
    keep = [k for k in range(len(pids)) if reached_t(A[k], stage_a[k]) and reached_t(B[k], stage_b[k])]
    contact = {"n_pairs_reaching_T": len(keep), "restriction": "both trials of the pair reached T"}
    # The fallback above is unchanged, and now COUNTED over every analysed trial (not only those the
    # `and` gets to): a blank reached_T is decided by the hand-scored stage letter, and co-primary 2 is
    # restricted on it. Silent until Sep 12 2026 (night audit Sep 7, :552-555; the reason
    # tools/v4_contact_events.py:750 computes reached_T at all). Printed every time, 0 included.
    blank = [r["label"] for a, b in zip(A, B) for r in (a, b) if r.get("reached_T") in (None, "")]
    contact["reached_T_fallback"] = {"n": len(blank), "of": len(A + B), "labels": blank}
    for key in ("contact_latency_s", "contact_impulse"):
        vals = np.array([[float(A[k].get(key, float("nan"))), float(B[k].get(key, float("nan")))] for k in keep], float)
        vals = vals[~np.isnan(vals).any(axis=1)] if len(vals) else vals
        name = "latency" if key.startswith("contact_latency") else "impulse"
        contact[name] = _pairs_block(vals, seed, n_boot) if len(vals) else {"n": 0, "note": "no telemetry pairs"}
    res["contact_response"] = contact

    ret_a = np.array([float(r.get("retries", 0)) for r in A])
    ret_b = np.array([float(r.get("retries", 0)) for r in B])
    res["retries"] = {"A_total": float(ret_a.sum()), "B_total": float(ret_b.sum()),
                      **_pairs_block(np.column_stack([ret_a, ret_b]), seed, n_boot)}
    res["guard_fires"] = {"A_total": float(sum(float(r.get("guard_fires", 0)) for r in A)),
                          "B_total": float(sum(float(r.get("guard_fires", 0)) for r in B))}
    res["mechanism_shift"] = Counter((a["mechanism"], b["mechanism"]) for a, b in zip(A, B))
    return res


# --- rendering ----------------------------------------------------------------------

def _f(x, nd=3):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _ci(ci, nd=3):
    return f"[{_f(ci[0], nd)}, {_f(ci[1], nd)}]"


def _shift_lines(name, blk, nd=3):
    hl, w, bs = blk["hodges_lehmann"], blk["wilcoxon"], blk.get("bootstrap")
    lines = [f"  {name}: n={blk['n']}  mean(B-A)={_f(blk['mean_diff'], nd)}  "
             f"HL={_f(hl['estimate'], nd)} {_ci(hl['ci'], nd)} (attained {hl['conf_attained']:.3f})",
             f"    Wilcoxon p={_f(w['p'], 4)}  method={w['method']}  "
             f"W+={_f(w.get('w_plus'), 1)} n_used={w.get('n_used')} zeros={w.get('n_zeros')}"]
    if bs:
        lines.append(f"    pair-bootstrap HL CI {_ci(bs['ci'], nd)}  (unit={bs['unit']}, n_boot={bs['n_boot']}, seed={bs['seed']})")
    return lines


NOT_DETECTABLE = ("not detectable at 80% power for any effect smaller than the discordance "
                  "(n={n} pairs, psi={psi:.3f})")


def _power_lines(p: dict) -> list[str]:
    """One row per discordance: the number where one exists, otherwise the sentence that is true.
    Until Sep 12 2026 this was a single line, and it printed the literal "nan" for the MDE at psi
    = 0.30 below 24 pairs -- the runnable core is 17 (night audit Sep 7, :607)."""
    n, obs = p["n_pairs"], p["observed_discordance"]
    rows = ([(f"psi={obs:.3f} (observed)", obs, "at_observed_discordance")] if obs is not None else []) + \
        [(f"psi={psi:.3f}", psi, f"at_{psi:.2f}") for psi in PRESTATED_DISCORDANCE]
    none = "not computable -- no discordant pairs; the pre-stated psi rows stand"
    L = [f"  MDE at 80% power, alpha 0.05 two-sided, n={n} pairs (Connor 1987):"]
    for label, psi, key in rows:
        v = p[key]
        L.append(f"    {label}: " + (none if psi == 0 else NOT_DETECTABLE.format(n=n, psi=psi) if math.isnan(v)
                                     else f"{v:.3f}"))
    L.append(f"  null 95% CI half-width of B-A, 1.96*sqrt(psi/n), n={n} pairs "
             "(the X of the allowed null sentence, protocol :76-77, :103-104):")
    for label, psi, key in rows:
        v = p["null_ci_halfwidth_95"][key]
        L.append(f"    {label}: " + (none if psi == 0 or math.isnan(v) else
                                     f"unbounded -- n={n} pairs, nothing was analysed" if math.isinf(v)
                                     else f"+/-{v:.3f}"))
    return L


def render(res: dict, disp: dict | None = None) -> str:
    # Sections print in the order `Eval Protocol -- v4 merged (Sep 6).md:56-64` pre-registers the
    # endpoints, every one headed "<role> -- <endpoint>". Until Sep 12 2026 co-primary 2 printed
    # fourth, after the key secondary and the negative control (night audit Sep 7, :615); the
    # order is pinned by test_endpoints_print_in_the_pre_registered_order_of_protocol_section_1.
    L = [f"PAIRS valid={res['n_pairs']}  strata={res['strata']}  seed={res['seed']}  n_boot={res['n_boot']}", ""]
    L.append("co-primary 1 -- graded stage (R/G/T/Rel/S -> 0-5), B - A")
    L += _shift_lines("stage", res["graded_stage"])
    c = res["contact_response"]
    L += ["", f"co-primary 2 -- contact response ({c['restriction']}): n_pairs={c['n_pairs_reaching_T']}"]
    fb = c["reached_T_fallback"]
    L.append(f"  reached_T: {fb['n']} of {fb['of']} trials FELL BACK to the hand-scored stage letter (stage >= T), "
             f"scores.csv left reached_T blank" + (f": {' '.join(fb['labels'])}" if fb["labels"] else ""))
    for name in ("latency", "impulse"):
        blk = c.get(name, {})
        L += _shift_lines(name, blk) if "hodges_lehmann" in blk else [f"  {name}: {blk.get('note', 'n/a')}"]
    b = res["binary_seat"]
    m, mde = b["mcnemar"], b["mde_80"]
    L += ["", "key secondary -- binary SEAT (held_to_end, 3.0 s hold)",
          f"  seat_A={b['seat_A']}/{b['n_pairs']} ({_f(b['rate_A'])}) Wilson {_ci(b['wilson_A'])}   "
          f"seat_B={b['seat_B']}/{b['n_pairs']} ({_f(b['rate_B'])}) Wilson {_ci(b['wilson_B'])}",
          f"  risk difference B-A={_f(b['risk_difference'])}  pair-bootstrap CI {_ci(b['bootstrap']['ci'])}",
          f"  discordant b(A only)={m['b']} c(B only)={m['c']}  mid-p McNemar p={_f(m['p'], 4)} "
          f"(exact uncorrected {_f(m['p_exact_uncorrected'], 4)}; two-sided = {m['two_sided_rule']})"] + \
        _power_lines(mde)
    g = res["grasp_negative_control"]
    L += ["", "negative control -- grasp stage reached (G or beyond)",
          f"  grasp_A={g['seat_A']}/{g['n_pairs']} Wilson {_ci(g['wilson_A'])}   grasp_B={g['seat_B']}/{g['n_pairs']} "
          f"Wilson {_ci(g['wilson_B'])}   diff={_f(g['risk_difference'])}  mid-p McNemar p={_f(g['mcnemar']['p'], 4)}"]
    r = res["rmst_45s"]
    L += ["", f"secondary -- RMST time-to-seat at {res['tau']} s (never seated = censored at tau; lower = sooner)",
          f"  A={_f(r['A']['rmst'], 2)} s (events {r['A']['n_events']}, censored {r['A']['n_censored']})   "
          f"B={_f(r['B']['rmst'], 2)} s (events {r['B']['n_events']}, censored {r['B']['n_censored']})   "
          f"B-A={_f(r['B_minus_A']['estimate'], 2)} pair-bootstrap CI {_ci(r['B_minus_A']['ci'], 2)}"]
    L += ["", f"secondary -- retries: A_total={res['retries']['A_total']:.0f} B_total={res['retries']['B_total']:.0f}"] + \
        _shift_lines("retries", res["retries"])
    L += ["", f"secondary -- guard fires: A_total={res['guard_fires']['A_total']:.0f} B_total={res['guard_fires']['B_total']:.0f}"]
    L += ["", "secondary -- paired failure-mechanism shift (A mechanism -> B mechanism : pairs)"]
    for (ma, mb), k in sorted(res["mechanism_shift"].items(), key=lambda kv: -kv[1]):
        L.append(f"  {ma:>5} -> {mb:<5} : {k}")
    if disp is not None:
        L += [""] + _disposition_lines(disp)
    return "\n".join(L)


def _disposition_lines(disp: dict) -> list[str]:
    L = [f"DISPOSITION  trials={disp['n_trials']}  pairs scheduled={disp['n_pairs_scheduled']}  "
         f"launched={disp['n_pairs_launched']}  valid={disp['n_pairs_valid']}"]
    # Once each. This loop read INVALIDATION_CODES + ("NOT_RUN",) until Sep 12 2026, and NOT_RUN
    # is already in INVALIDATION_CODES, so the table printed it twice (night audit Sep 7, :634).
    for code in INVALIDATION_CODES:
        kt, kp = disp["trial_code_counts"].get(code, 0), disp["pair_code_counts"].get(code, 0)
        if kt or kp:
            L.append(f"  {code:<20} trials={kt:<3} pairs={kp}")
    for t in disp["trials"]:
        if t["codes"]:
            L.append(f"    {t['label']}_{t['stamp']}  {t['verdict']:<7} {' '.join(t['codes'])}")
    for pid, p in sorted(disp["pairs"].items()):
        if p["codes"]:
            L.append(f"    pair {pid} ({p['stratum']}): {' '.join(p['codes'])}" +
                     ("" if p["launched"] else f"  (never launched; schedule cut_rank={p['cut_rank']})"))
    return L


def render_no_valid_pairs(disp: dict) -> str:
    """What a bench with zero valid pairs still publishes. Until Sep 12 2026 main() refused before
    rendering anything, so neither the disposition table nor the MDE came out (night audit Sep 7,
    :697), though the protocol publishes the table regardless (Eval Protocol -- v4 merged (Sep 6).md
    :68, :308). No endpoint header: nothing was analysed, and nothing here may read as a result."""
    return "\n".join(["NO VALID PAIR: no endpoint was analysed. Published regardless: the power statement and the "
                      "disposition table (protocol :68, :308). The run exits 2.", "",
                      "power at the pool actually analysed (n=0 valid pairs; the binary SEAT's MDE, protocol :61)"] +
                     _power_lines(power_at(0)) + [""] + _disposition_lines(disp))


# --- CLI ---------------------------------------------------------------------------

def _rows_from_files(disp: dict, scores_csv, allow_seat_disagreement: bool, stratum: str | None = None) -> list[dict]:
    """`stratum` restricts the pool (protocol §7 Sep 14 16:59: the primary pool is S0-CLEAN; S1 is
    exploratory). The binary SEAT is the VIDEO adjudication (stage S), not the scorer's ledger verdict:
    score_episode's SEAT is positively biased (§7 Sep 14 16:49) and video is ground truth (v2 amendment 5).
    The ledger verdict is a cross-check: a disagreement raises unless allowed, and is reported either way."""
    scores = {r["label"]: r for r in read_scores(scores_csv)}
    rows, missing, disagree = [], [], []
    for pid in disp["valid_pairs"]:
        p = disp["pairs"][pid]
        if stratum is not None and p["stratum"] != stratum:
            continue
        for arm in ("A", "B"):
            t = p[arm]
            s = scores.get(t["label"])
            if s is None:
                missing.append(t["label"])
                continue
            seat = stage_to_int(s["stage"]) == STAGES["S"]   # the adjudicated stage decides
            if (t["verdict"] == "SEAT") != seat:
                disagree.append(t["label"])

            def num(key, default=float("nan")):
                v = s.get(key, "")
                return default if v in ("", None) else float(v)
            rows.append({"pair_id": pid, "stratum": p["stratum"], "arm": arm, "label": t["label"],
                         "stage": s["stage"], "mechanism": s["mechanism"], "seat": seat,
                         "seat_time_s": num("seat_time_s"), "retries": num("retries", 0.0),
                         "guard_fires": num("guard_fires", 0.0), "contact_latency_s": num("contact_latency_s"),
                         "contact_impulse": num("contact_impulse"), "reached_T": s.get("reached_T", "")})
    if missing:
        raise ValueError(f"valid trials without an adjudicated score row: {missing}")
    if disagree and not allow_seat_disagreement:
        raise ValueError(f"adjudicated stage S disagrees with the ledger SEAT verdict on {disagree}; "
                         "decide which is right, record it in invalidations.csv or pass --allow-seat-disagreement")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ledger", required=True, help="tools/scored_logs/trials.csv; receipts are looked up beside it")
    ap.add_argument("--pairs", required=True, help="pair schedule csv: " + ",".join(PAIR_COLUMNS))
    ap.add_argument("--scores", required=True, help="adjudicated scores csv: " + ",".join(SCORE_COLUMNS))
    ap.add_argument("--schedule", default=None,
                    help=f"the seeded pair schedule (tools/v4_schedule.py); default {DEFAULT_SCHEDULE}")
    ap.add_argument("--invalidations", default=None, help="csv label,stamp,code,note; codes: " + ",".join(MANUAL_CODES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--tau", type=float, default=45.0)
    ap.add_argument("--allow-seat-disagreement", action="store_true")
    ap.add_argument("--stratum", default=None,
                    help="restrict the analysed pool to one stratum (protocol §7 Sep 14 16:59: the primary pool is S0-CLEAN)")
    a = ap.parse_args(argv)
    try:
        ledger = Path(a.ledger)
        if ledger.name != "trials.csv":
            raise ValueError("--ledger must point at a trials.csv (run_scored_trial.sh:156)")
        disp = disposition(ledger.parent, a.pairs, a.invalidations, schedule_csv=a.schedule or DEFAULT_SCHEDULE)
        if not disp["valid_pairs"]:
            _publish(render_no_valid_pairs(disp), a.out)       # the record first, then the refusal
            raise NoValidPairs(f"no valid pairs (trials={disp['n_trials']}, scheduled pairs={disp['n_pairs_scheduled']}; "
                               f"trial codes={dict(disp['trial_code_counts'])}, pair codes={dict(disp['pair_code_counts'])})")
        rows = _rows_from_files(disp, a.scores, a.allow_seat_disagreement, stratum=a.stratum)
        res = analyze(rows, seed=a.seed, n_boot=a.n_boot, tau=a.tau)
        fb = res["contact_response"]["reached_T_fallback"]
        if fb["n"]:
            print(f"WARNING: reached_T fell back to the hand-scored stage letter on {fb['n']} of {fb['of']} trials "
                  f"(co-primary 2's restriction): {' '.join(fb['labels'])}", file=sys.stderr)
    except (NoValidPairs, ValueError, FileNotFoundError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    _publish(render(res, disp), a.out)
    return 0


def _publish(text: str, out) -> None:
    print(text)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text + "\n")


if __name__ == "__main__":
    sys.exit(main())
