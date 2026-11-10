"""Honest evaluation suite for stock-ranking signals (and the framework any future model should be judged by).

    python -m experiments.eval_suite --panel data_fundamentals/panel.parquet            # a few minutes
    python -m experiments.eval_suite --panel ... --n-perm 4000 --out reports/eval_suite.md

What it does, all on the league's own walk-forward rounds and never touching the holdout:
  * Pools the MONTHLY decision dates (21 days apart, so labels never overlap) of every validation window: about 10x more
    dates than the per-round tables, so t-stats and p-values are meaningful.
  * Signals: every raw feature, simple factor composites (value, quality, growth, momentum, reversal), and walk-forward
    models refit each round (ridge on raw features, random sparse KC code + ridge, random dense features + ridge).
  * Sign-flip test per signal on the monthly IC series (valid with correlated stocks), with Holm, Benjamini-Hochberg and Westfall-Young maxT
    corrections across ALL signals tested, so a lucky feature among 30 is not reported as a finding.
  * Power curve: plants a known IC into placebo IC series with the real dispersion. Shows the smallest IC this data can detect and checks
    that the false-positive rate (rho = 0) is about 5%.
  * Cost-aware decile long-short backtest at several cost levels, with turnover and drawdown.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from env.features import FUND_FEATURES
from evalkit import portfolio, stats
from experiments.control_features import dense_code, fit, kc_code, score, sparse_proj
from tournament.data import GAP, Data

COMPOSITES = {
    "value (EY+BP+FCF)": ["earnings_yield", "book_to_price", "fcf_yield"],
    "quality (ROE+ROA+GM)": ["roe", "roa", "gross_margin"],
    "growth (rev+eps yoy)": ["rev_growth_yoy", "eps_growth_yoy"],
    "momentum 12-1": ["mom_12_1"],
    "reversal (-ret_1)": ["-ret_1"],
    "low leverage (-D/E)": ["-debt_to_equity"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data_fundamentals/panel.parquet")
    ap.add_argument("--train-days", type=int, default=1000)
    ap.add_argument("--val-days", type=int, default=252)
    ap.add_argument("--holdout", type=int, default=500)
    ap.add_argument("--units", type=int, default=1000)
    ap.add_argument("--kc-inputs", type=int, default=8)
    ap.add_argument("--sparsity", type=float, default=0.05)
    ap.add_argument("--max-rows", type=int, default=100000)
    ap.add_argument("--train-every", type=int, default=3)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--out", default="reports/eval_suite.md")
    ap.add_argument("--no-models", action="store_true", help="features and composites only (fast)")
    ap.add_argument("--sector-neutral", action="store_true",
                    help="demean labels and scores within GICS sector each date (needs the saved Wikipedia page)")
    ap.add_argument("--wiki-html", default="data_fundamentals/universe/wiki_page.html")
    a = ap.parse_args()

    d = Data(a.panel, a.train_days, a.val_days, a.holdout)
    feats = d.feats
    col = lambda nm: feats.index(nm)
    n_obs, k = len(feats), max(1, int(a.sparsity * a.units))
    sigs: dict[str, dict] = {f: {} for f in feats}
    for c in COMPOSITES:
        sigs[c] = {}
    if not a.no_models:
        for m in ("ridge raw", "random KC code + ridge", "random dense + ridge"):
            sigs[m] = {}
    dates, r = [], 0
    while d.round_valid(r):
        c, v0, v1 = d.round_geometry(r)
        val = list(range(v0, v1, GAP))
        dates += val
        for i in val:
            X = d.day[i][1].astype(np.float32)
            for j, f in enumerate(feats):
                sigs[f][i] = X[:, j]
            for cn, parts in COMPOSITES.items():
                z = np.zeros(len(X), np.float32)
                for p in parts:
                    z += (-1 if p.startswith("-") else 1) * X[:, col(p.lstrip("-"))]
                sigs[cn][i] = z
        if not a.no_models:
            tr = range(0, c + 1, a.train_every)
            Xt = np.vstack([d.day[i][1] for i in tr]).astype(np.float32)
            yt = np.concatenate([d.day[i][2] for i in tr]).astype(np.float32)
            ok = np.isfinite(yt)
            Xt, yt = Xt[ok], yt[ok]
            if len(Xt) > a.max_rows:
                sel = np.random.default_rng(r).choice(len(Xt), a.max_rows, replace=False)
                Xt, yt = Xt[sel], yt[sel]
            rng = np.random.default_rng([r, 11])
            P = sparse_proj(rng, a.units, n_obs, a.kc_inputs)
            Pd = rng.normal(size=(a.units, n_obs)).astype(np.float32)
            mods = {"ridge raw": (lambda Z: Z, fit(Xt, yt)),
                    "random KC code + ridge": (lambda Z: kc_code(Z, P, k), fit(kc_code(Xt, P, k), yt)),
                    "random dense + ridge": (lambda Z: dense_code(Z, Pd), fit(dense_code(Xt, Pd), yt))}
            for i in val:
                X = d.day[i][1].astype(np.float32)
                for name, (enc, m) in mods.items():
                    sigs[name][i] = score(enc(X), m)
        print(f"round {r} done ({len(val)} monthly dates)", flush=True)
        r += 1
    labels = {i: d.day[i][2] for i in dates}
    tick = {i: d.day[i][0] for i in dates}
    if a.sector_neutral:
        from evalkit.neutral import demean_by_group, sector_map
        sm = sector_map(a.wiki_html)
        grp = {i: np.array([sm.get(t, "UNKNOWN") for t in tick[i]]) for i in dates}
        print(f"sector-neutral: {np.mean([np.mean(grp[i] == 'UNKNOWN') for i in dates]):.0%} of stock-dates have no sector "
              f"(removed companies), treated as one group")
        labels = {i: demean_by_group(labels[i], grp[i]) for i in dates}
        sigs = {nm: {i: demean_by_group(v[i], grp[i]) for i in dates} for nm, v in sigs.items()}
    print(f"\n{len(sigs)} signals, {len(dates)} monthly decision dates {d.dates[dates[0]].date()} -> {d.dates[dates[-1]].date()}\n")

    names = list(sigs)
    ics = {s: stats.ic_series(sigs[s], labels, dates) for s in names}
    res = stats.signflip_test(np.column_stack([ics[s] for s in names]), names, n_perm=max(a.n_perm, 20000))
    # placebos: raw features and composites whose own IC is indistinguishable from zero keep the real cross-correlation noise
    placebo = [ics[s] for s in names if abs(res[s]["t"]) < 1.0 and s not in ("ridge raw", "random KC code + ridge", "random dense + ridge")]
    pw = stats.power_from_ics(placebo, n_trials=a.trials * 4)
    mdi = stats.min_detectable_ic(pw)
    order = sorted(res, key=lambda s: -abs(res[s]["t"]) if np.isfinite(res[s]["t"]) else 0)
    lines = [f"# Evaluation suite   ({datetime.now():%Y-%m-%d %H:%M})", "",
             f"{len(sigs)} signals, {len(dates)} non-overlapping monthly decision dates "
             f"({d.dates[dates[0]].date()} to {d.dates[dates[-1]].date()}); holdout untouched. "
             f"{'SECTOR-NEUTRAL (labels and scores demeaned within GICS sector).' if a.sector_neutral else 'Not sector-neutral.'}", "",
             "## Signal tests (mean monthly rank-IC; p from a sign-flip test on the monthly IC series)", "",
             "| signal | IC | IC sd | t | p | p Holm | p BH | p maxT | ICs>0 |", "|---|---|---|---|---|---|---|---|---|"]
    for s in order:
        x = res[s]
        lines.append(f"| {s} | {x['ic']:+.4f} | {x['sd']:.3f} | {x['t']:+.2f} | {x['p']:.3f} | {x['p_holm']:.3f} | {x['p_bh']:.3f} | "
                     f"{x['p_maxT']:.3f} | {np.mean(ics[s] > 0):.0%} |")
    lines += ["", "Read the corrected columns (Holm, maxT), not raw p: with this many signals a few will look good by chance.", "",
              "## Power: what could this data have detected? (real month-to-month IC dispersion)", "", "| planted IC | detection rate |", "|---|---|"]
    lines += [f"| {rho:.2f} | {v:.0%} |" for rho, v in pw.items()]
    lines += ["", f"False-positive rate at IC 0 (placebo): {pw[0.0]:.0%} (should be near 5%). "
              f"Smallest planted IC found in >= 80% of trials: {mdi if mdi else 'above the grid'}.", "",
              "## Decile long-short (net of costs, monthly, equal weight)", "",
              "| signal | gross/m | net Sharpe @0bp | @10bp | @25bp | maxDD @10bp | turnover |", "|---|---|---|---|---|---|---|"]
    top = [s for s in order[:10]]
    for s in top:
        pr = portfolio.long_short(sigs[s], labels, tick, dates, q=0.1, bps=(0, 10, 25))
        lines.append(f"| {s} | {pr[0]['mean_m']:+.2%} | {pr[0]['sharpe']:+.2f} | {pr[10]['sharpe']:+.2f} | {pr[25]['sharpe']:+.2f} | "
                     f"{pr[10]['maxdd']:+.1%} | {pr[10]['turnover']:.0%} |")
    lines += ["", "Caveats: the panel is not fully survivorship-free (delisted names are missing), costs are a flat bps per unit "
              "traded with no market impact or borrow cost on shorts, and the top-10 list above was chosen AFTER seeing the "
              "results, so its Sharpe ratios are optimistic."]
    text = "\n".join(lines)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    print(f"\nsaved to {a.out}")


if __name__ == "__main__":
    main()
