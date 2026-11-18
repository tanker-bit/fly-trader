"""Do plain fundamentals rules work at long horizons on the league's own rounds? (T5 step 5, part 1: core)

Signs are the textbook ones, fixed in advance (not picked from the audit, which would be in-sample): cheap, profitable, low accruals.
Each rule is scored on the league's validation dates for each round, against the forward excess return at the given horizon
(63 or 126 days), using only rows where the fundamentals are real (fund_known > 0). Nothing is trained. A date counts only if its
forward window ends before the holdout starts, so the sealed holdout is never touched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.fundamentals_audit import forward_excess
from tournament.data import rank_ic

RULES = {"earnings_yield": ["+earnings_yield"], "fcf_yield": ["+fcf_yield"], "roe": ["+roe"], "roa": ["+roa"],
         "-accruals": ["-accruals"], "book_to_price": ["+book_to_price"],
         "value+quality": ["+earnings_yield", "+fcf_yield", "+roe", "+roa", "-accruals"]}


def _term(data, Xi, term):
    sign = -1.0 if term[0] == "-" else 1.0
    col = Xi[:, data.feats.index(term[1:])]
    return sign * (col - np.nanmean(col)) / (np.nanstd(col) + 1e-12)       # z-score so terms can be averaged


def rule_score(data, Xi, terms):
    return np.sum([_term(data, Xi, t) for t in terms], axis=0)


def fund_rule_ics(data, labels_wide: pd.DataFrame, rounds, score_every: int, horizon: int, rules=RULES) -> pd.DataFrame:
    """Mean rank IC per round per rule vs the `horizon`-day forward excess return. labels_wide: date x ticker.
    Dates whose forward window would reach the holdout (i + horizon > n_usable) or with < 20 real-fundamentals stocks are skipped."""
    fk = data.feats.index("fund_known")
    rows = {}
    for r in rounds:
        c, v0, v1 = data.round_geometry(r)
        row = {name: [] for name in rules}
        for i in range(v0, v1, score_every):
            if i + horizon > data.n_usable:
                continue
            tick, X, _ = data.day[i]
            real = X[:, fk] > 0
            if real.sum() < 20:
                continue
            y = labels_wide.loc[data.dates[i]].reindex(tick).to_numpy(float)[real]
            for name, terms in rules.items():
                row[name].append(rank_ic(rule_score(data, X[real], terms), y))
        rows[r] = {name: (float(np.nanmean(v)) if v else np.nan) for name, v in row.items()}
    return pd.DataFrame(rows).T


def labels_for(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return forward_excess(panel, horizon).unstack("ticker")


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Across rounds: mean IC, rounds positive, and t of the mean vs zero. Rounds' forward windows overlap at long horizons, so
    the t is optimistic; read it as an upper bound."""
    n = table.count()
    t = table.mean() / (table.std(ddof=1) / np.sqrt(n))
    return pd.DataFrame({"mean_ic": table.mean(), "rounds_positive": (table > 0).sum(), "rounds": n, "t": t})


def main():
    import argparse
    from tournament.data import Data
    from tournament.store import Store

    ap = argparse.ArgumentParser(description="Plain fundamentals rules vs the reversal rule at long horizons, on the league's rounds.")
    ap.add_argument("--root", default="league_data")
    ap.add_argument("--horizons", default="63,126")
    ap.add_argument("--out", default=None, help="optional CSV prefix, one file per horizon")
    a = ap.parse_args()
    store = Store(a.root)
    cfg = store.load_config()
    print("loading panel...", flush=True)
    data = Data.from_cfg(cfg)
    from tournament.data import load_panel
    panel = load_panel(cfg["panel"])
    rounds = [r for r in range(0, 200) if data.round_valid(r)]
    rules = {**RULES, "reversal -ma50_dist": ["-ma50_dist"]}
    pd.set_option("display.width", 200)
    for h in [int(x) for x in a.horizons.split(",")]:
        t = fund_rule_ics(data, labels_for(panel, h), rounds, cfg["score_every"], h, rules)
        t = t.dropna(how="all")
        print(f"\n=== forward {h}-day excess return, rounds {t.index.min()}-{t.index.max()} ===\n")
        print(t.round(3).to_string())
        print("\n" + summarize(t).round(3).to_string())
        if a.out:
            t.to_csv(f"{a.out}_{h}d.csv")


if __name__ == "__main__":
    main()
