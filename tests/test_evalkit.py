"""Tests for the evaluation kit: planted signals are found, placebos are not, corrections are right, portfolios use no future info."""
from __future__ import annotations

import numpy as np

from evalkit import stats, portfolio


def synth(n_dates=60, n=200, rho=0.0, seed=0):
    rng = np.random.default_rng(seed)
    labels, sig, tick = {}, {}, {}
    for i in range(n_dates):
        y = rng.normal(size=n)
        labels[i] = y
        sig[i] = rho * y / 1.0 + np.sqrt(1 - rho ** 2) * rng.normal(size=n) if rho else rng.normal(size=n)
        tick[i] = np.array([f"T{k}" for k in range(n)])
    return sig, labels, tick


def test_signflip_finds_planted_signal_and_not_noise():
    s1, lab, _ = synth(rho=0.3, seed=1)
    s0, _, _ = synth(rho=0.0, seed=2)
    d = list(range(60))
    M = np.column_stack([stats.ic_series(s1, lab, d), stats.ic_series(s0, lab, d)])
    res = stats.signflip_test(M, ["planted", "noise"], n_perm=4000)
    assert res["planted"]["p"] < 0.01 and res["planted"]["p_maxT"] < 0.01
    assert res["noise"]["p"] > 0.05 and res["planted"]["ic"] > 0.2


def test_signflip_p_matches_t_stat_for_correlated_stocks():
    # stocks share a common shock each month: ICs swing widely, and the test must NOT call a t of ~1.5 'p = 0.000'
    rng = np.random.default_rng(7)
    ic = rng.normal(0.02, 0.2, size=(120, 1))
    r = stats.signflip_test(ic, ["x"], n_perm=8000)["x"]
    from scipy.stats import norm
    assert abs(r["p"] - 2 * (1 - norm.cdf(abs(r["t"])))) < 0.05


def test_corrections():
    p = np.array([0.01, 0.04, 0.03, 0.5])
    h = stats.holm(p)
    assert np.allclose(h, [0.04, 0.09, 0.09, 0.5])          # Holm: 4*.01, 3*.03, then monotone, 1*.5
    b = stats.benjamini_hochberg(p)
    assert np.all(b >= p - 1e-12) and np.all(b <= 1) and b[0] <= b[3]


def test_power_uses_real_dispersion_and_placebo_rate():
    rng = np.random.default_rng(3)
    quiet = [rng.normal(0, 0.05, 120) for _ in range(5)]
    noisy = [rng.normal(0, 0.25, 120) for _ in range(5)]
    pq = stats.power_from_ics(quiet, rhos=(0.0, 0.02, 0.05), n_trials=600)
    pn = stats.power_from_ics(noisy, rhos=(0.0, 0.02, 0.05), n_trials=600)
    assert pq[0.0] < 0.12 and pn[0.0] < 0.12            # false positives near 5%
    assert pq[0.02] > pn[0.02] + 0.3                    # the same IC is far harder to detect when ICs swing widely
    assert pn[0.05] < 0.8                     # IC 0.05 with sd 0.25 over 120 months is only about 60% detectable


def test_portfolio_uses_only_given_scores_and_costs_reduce_return():
    s, lab, tk = synth(rho=0.3, seed=4)
    r = portfolio.long_short(s, lab, tk, list(range(60)), q=0.1, bps=(0, 50))
    assert r[0]["mean_m"] > 0 and r[50]["mean_m"] < r[0]["mean_m"]
    # a signal unrelated to returns earns about nothing
    s0, _, _ = synth(rho=0.0, seed=5)
    r0 = portfolio.long_short(s0, lab, tk, list(range(60)), q=0.1, bps=(0,))
    assert abs(r0[0]["mean_m"]) < 0.25
    # shifting scores by one date (using yesterday's score for today's label) destroys the planted signal
    sh = {i: s[i - 1] for i in range(1, 60)}
    rs = portfolio.long_short(sh, lab, tk, list(range(1, 60)), q=0.1, bps=(0,))
    assert rs[0]["mean_m"] < r[0]["mean_m"] / 2


def test_sector_demean_and_parser_sector_column():
    from evalkit.neutral import demean_by_group
    x = np.array([1.0, 3.0, 10.0, 14.0, np.nan])
    g = np.array(["a", "a", "b", "b", "b"])
    out = demean_by_group(x, g)
    assert np.allclose(out[:4], [-1, 1, -2, 2]) and np.isnan(out[4])
    from env import universe_us as uni
    html = ("<table><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th><th>Date added</th><th>CIK</th></tr>"
            "<tr><td>AAA</td><td>Aaa</td><td>Tech</td><td>Software</td><td>2015-01-05</td><td>0000000001</td></tr></table>")
    cur, _ = uni.parse_wikipedia(html)
    assert cur.loc[0, "sector"] == "Tech" and cur.loc[0, "sub_industry"] == "Software"


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            fn()
            print("ok", n)
    print("all evalkit tests passed")
