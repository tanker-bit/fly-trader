"""Statistical tests for cross-sectional signals.

A signal is {date_index: score array}; labels are {date_index: forward 21-day excess return array} for the same stocks.
Dates must be at least 21 trading days apart (non-overlapping labels), otherwise t-stats and p-values are overstated.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, norm


def _zrank(x):
    r = rankdata(np.asarray(x, float))
    r = r - r.mean()
    n = np.linalg.norm(r)
    return r / n if n > 0 else r * 0.0


def ic_series(sig, labels, dates):
    """Spearman IC per date (0 where the score is constant)."""
    out = []
    for i in dates:
        s, y = np.asarray(sig[i], float), np.asarray(labels[i], float)
        ok = np.isfinite(s) & np.isfinite(y)
        out.append(float(_zrank(s[ok]) @ _zrank(y[ok])) if ok.sum() >= 10 else np.nan)
    return np.array(out)


def tstat(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 3 or v.std(ddof=1) == 0:
        return np.nan
    return v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))


def bootstrap_ci(v, n=4000, seed=0, level=0.95):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    m = rng.choice(v, size=(n, len(v))).mean(1)
    lo, hi = np.percentile(m, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def holm(p):
    """Holm-Bonferroni adjusted p-values (controls family-wise error)."""
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, k in enumerate(o):
        run = max(run, (len(p) - rank) * p[k])
        adj[k] = min(1.0, run)
    return adj


def benjamini_hochberg(p):
    """BH adjusted p-values (controls false discovery rate; less strict than Holm)."""
    p = np.asarray(p, float)
    o = np.argsort(p)[::-1]
    adj = np.empty_like(p)
    run = 1.0
    for j, k in enumerate(o):
        rank = len(p) - j
        run = min(run, p[k] * len(p) / rank)
        adj[k] = run
    return adj


def cross_section_permutation_test(signals: dict, labels, dates, n_perm=2000, seed=0):
    """WARNING: anti-conservative on real data. It treats the stocks of one date as independent, but stocks share sector and
    market moves, so real monthly ICs vary about 4x more than this null allows (p-values come out near 0 for t-stats of 1.5).
    Kept only for synthetic checks; use signflip_test on the monthly IC series instead.

    Null: scores unrelated to returns within each date. Labels are shuffled WITHIN each date (the same shuffle is
    applied to every signal, so the joint null keeps the signals' mutual correlation).

    Returns {name: dict(ic, t, p, p_holm, p_bh, p_maxT)}: p two-sided on the mean IC; p_maxT is the Westfall-Young
    adjusted p (the observed |mean IC| against the null distribution of the LARGEST |mean IC| among all signals)."""
    names = list(signals)
    rng = np.random.default_rng(seed)
    null = np.zeros((n_perm, len(names)))
    obs = np.zeros((len(dates), len(names)))
    for di, i in enumerate(dates):
        y = np.asarray(labels[i], float)
        S = np.array([np.asarray(signals[nm][i], float) for nm in names])
        ok = np.isfinite(y) & np.all(np.isfinite(S), axis=0)
        if ok.sum() < 10:
            obs[di] = np.nan
            continue
        Z = np.array([_zrank(s[ok]) for s in S])               # k x n
        zy = _zrank(y[ok])
        obs[di] = Z @ zy
        idx = rng.random((n_perm, ok.sum())).argsort(1)         # n_perm random permutations
        null += (zy[idx] @ Z.T)
    n_d = np.sum(np.isfinite(obs[:, 0]))
    null /= max(n_d, 1)
    mean_ic = np.nanmean(obs, axis=0)
    p = np.array([(1 + np.sum(np.abs(null[:, k]) >= abs(mean_ic[k]))) / (n_perm + 1) for k in range(len(names))])
    mx = np.abs(null).max(1)
    p_max = np.array([(1 + np.sum(mx >= abs(mean_ic[k]))) / (n_perm + 1) for k in range(len(names))])
    ph, pb = holm(p), benjamini_hochberg(p)
    return {nm: dict(ic=float(mean_ic[k]), t=float(tstat(obs[:, k])), p=float(p[k]), p_holm=float(ph[k]),
                     p_bh=float(pb[k]), p_maxT=float(p_max[k])) for k, nm in enumerate(names)}


def power_curve(labels, dates, rhos=(0.0, 0.01, 0.02, 0.03, 0.05), n_trials=300, seed=0, alpha=0.05):
    """Plant a signal with true rank-IC ~ rho into the real labels and see how often the t-test (|t| > critical)
    finds it, with the same dates and stock counts as the real evaluation. rho = 0 is the PLACEBO: the false-positive
    rate there should be about alpha. Returns {rho: detection rate}."""
    rng = np.random.default_rng(seed)
    crit = norm.ppf(1 - alpha / 2)
    zys = {}
    for i in dates:
        y = np.asarray(labels[i], float)
        ok = np.isfinite(y)
        zys[i] = (ok, _zrank(y[ok]) * np.sqrt(ok.sum()))
    out = {}
    for rho in rhos:
        hits = 0
        for _ in range(n_trials):
            ics = []
            for i in dates:
                ok, zy = zys[i]
                s = rho * zy + np.sqrt(max(1 - rho ** 2, 0)) * rng.normal(size=len(zy))
                ics.append(float(_zrank(s) @ (zy / np.linalg.norm(zy))))
            hits += abs(tstat(ics)) > crit
        out[rho] = hits / n_trials
    return out


def min_detectable_ic(power):  # smallest planted IC detected >= 80% of the time
    """Smallest planted IC detected in >= 80% of trials (None if none in the grid)."""
    for rho in sorted(power):
        if rho > 0 and power[rho] >= 0.8:
            return rho
    return None


def signflip_test(ic_matrix, names, n_perm=20000, seed=0):
    """Test on the monthly IC series (dates x signals), which already contains the real cross-sectional correlation.
    Null: each month's IC is symmetric around zero, so its sign is randomly flipped (the SAME flips for every signal, so the
    joint null keeps the signals' mutual correlation). Needs months at least 21 days apart (independent labels).
    Returns {name: dict(ic, sd, t, p, p_holm, p_bh, p_maxT)}; p_maxT compares each |mean IC| with the null distribution of the
    LARGEST |mean IC| among all signals (family-wise control that respects correlated signals)."""
    M = np.asarray(ic_matrix, float)
    M = np.where(np.isfinite(M), M, 0.0)
    n, k = M.shape
    rng = np.random.default_rng(seed)
    obs = M.mean(0)
    null = np.empty((n_perm, k))
    for b in range(0, n_perm, 2000):
        e = min(n_perm, b + 2000)
        signs = rng.choice([-1.0, 1.0], size=(e - b, n))
        null[b:e] = signs @ M / n
    p = np.array([(1 + np.sum(np.abs(null[:, j]) >= abs(obs[j]) - 1e-15)) / (n_perm + 1) for j in range(k)])
    rms = np.sqrt((M ** 2).mean(0)) + 1e-12                 # sign flips leave each signal's RMS unchanged, so this studentizes
    tn = np.abs(null) * np.sqrt(n) / rms                    # null t-like statistics
    to = np.abs(obs) * np.sqrt(n) / rms
    mx = tn.max(1)                                          # largest |t| among ALL signals under the null
    pm = np.array([(1 + np.sum(mx >= to[j] - 1e-12)) / (n_perm + 1) for j in range(k)])
    ph, pb = holm(p), benjamini_hochberg(p)
    return {nm: dict(ic=float(obs[j]), sd=float(M[:, j].std(ddof=1)), t=float(tstat(M[:, j])), p=float(p[j]), p_holm=float(ph[j]),
                     p_bh=float(pb[j]), p_maxT=float(pm[j])) for j, nm in enumerate(names)}


def power_from_ics(placebo_ics, rhos=(0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1), n_trials=2000, seed=0, alpha=0.05):
    """Power using the REAL dispersion of the monthly IC series. `placebo_ics` are IC series of signals believed to have no
    skill (e.g. the raw features with |t| small); each trial picks one, removes its own mean, resamples the months with
    replacement, adds a planted IC `rho`, and applies the t-test. rho = 0 gives the false-positive rate (about alpha)."""
    rng = np.random.default_rng(seed)
    P = [np.asarray(v, float)[np.isfinite(v)] for v in placebo_ics]
    P = [v - v.mean() for v in P if len(v) > 5]
    crit = norm.ppf(1 - alpha / 2)
    out = {}
    for rho in rhos:
        hits = 0
        for _ in range(n_trials):
            v = P[rng.integers(len(P))]
            x = rng.choice(v, size=len(v)) + rho
            hits += abs(tstat(x)) > crit
        out[rho] = hits / n_trials
    return out
