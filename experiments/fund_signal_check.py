"""Gate before the expensive league: do the fundamentals contain ANY signal at the monthly horizon?

    python -m experiments.fund_signal_check --panel data_fundamentals/panel.parquet          # about 1-3 minutes

1. Per feature: mean rank-IC with the 21-day excess return over NON-overlapping monthly decision dates, its t-stat, and the
   same for the first and second half of the sample (a real effect should have the same sign in both).
2. Walk-forward ridge, price features only vs price + fundamentals (refit every 252 dates on data whose 21-day label was
   already known). The paired difference and its t-stat answer: do fundamentals add anything a linear model can use?
The final holdout (last --holdout dates) is never touched. If step 2 shows no gain, a fly league on these inputs is unlikely
to find one either, and the better investment is better data, not more training.
"""
from __future__ import annotations

import argparse

import numpy as np

from env.features import FUND_FEATURES
from learning.ridge import fit_ridge, predict
from tournament.data import GAP, Data, rank_ic


def tstat(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return (a.mean(), a.mean() / (a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 2 and a.std() > 0 else np.nan, len(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data_fundamentals/panel.parquet")
    ap.add_argument("--holdout", type=int, default=500)
    ap.add_argument("--first", type=int, default=750, help="first decision date index (needs training history before it)")
    a = ap.parse_args()
    d = Data(a.panel, 1000, 252, a.holdout)
    n = d.n_usable
    price = [j for j, f in enumerate(d.feats) if f not in FUND_FEATURES]
    allj = list(range(len(d.feats)))
    dec = list(range(a.first, n, GAP))
    print(f"{len(d.feats)} features ({len(price)} price, {len(allj) - len(price)} fundamental); {len(dec)} monthly decisions "
          f"{d.dates[dec[0]].date()} -> {d.dates[dec[-1]].date()}; holdout untouched\n")
    print(f"{'feature':<18}{'IC':>8}{'t':>7}{'IC 1st half':>13}{'IC 2nd half':>13}")
    h = len(dec) // 2
    for j, f in enumerate(d.feats):
        ics = np.array([rank_ic(d.day[i][1][:, j], d.day[i][2]) for i in dec])
        m, t, _ = tstat(ics)
        print(f"{f:<18}{m:>+8.3f}{t:>+7.1f}{ics[:h].mean():>+13.3f}{ics[h:].mean():>+13.3f}")
    res = {"price only": [], "price + fundamentals": []}
    coefs, last_fit = {}, None
    for i in dec:
        if last_fit is None or i - last_fit >= 252:
            tr = range(0, i - GAP + 1, 3)
            X = np.vstack([d.day[k][1] for k in tr]).astype(float)
            y = np.concatenate([d.day[k][2] for k in tr])
            coefs = {"price only": (price, *fit_ridge(X[:, price], y)), "price + fundamentals": (allj, *fit_ridge(X[:, allj], y))}
            last_fit = i
        for name, (cols, c, xm) in coefs.items():
            res[name].append(rank_ic(predict(d.day[i][1][:, cols].astype(float), c, xm), d.day[i][2]))
    print("\nWalk-forward ridge (monthly decisions):")
    for name, v in res.items():
        m, t, k = tstat(v)
        print(f"  {name:<22} IC {m:+.4f}  t {t:+.2f}  (n={k})")
    diff = np.array(res["price + fundamentals"]) - np.array(res["price only"])
    m, t, k = tstat(diff)
    print(f"  {'fundamentals add':<22} IC {m:+.4f}  t {t:+.2f}  (paired, n={k})")
    print("\nReading: |t| < 2 is no evidence. Fundamentals are only worth a fly league if they add clearly in the paired line.")


if __name__ == "__main__":
    main()
