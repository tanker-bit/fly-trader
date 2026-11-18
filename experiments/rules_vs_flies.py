"""Do the flies beat one-line rules? (T5 step 4)

Core: score simple rules on exactly the validation dates the league used, round by round, with no training. Part 2 adds the
command line and the summary against the flies' own saved results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from learning.ridge import fit_ridge, predict
from tournament.data import rank_ic

RULES = ["-ma50_dist", "-ret_20", "-rsi_14", "momentum 12-1", "ridge"]


def rule_scores(data, name: str, Xi, ridge=None):
    """Score array for one day's stocks. `-name` = minus that feature (so a reversal rule); 'ridge' uses the fitted (coef, xm)."""
    if name == "ridge":
        return predict(Xi, *ridge)
    if name == "momentum 12-1":
        return Xi[:, data.feats.index("mom_12_1")]
    if name.startswith("-"):
        return -Xi[:, data.feats.index(name[1:])]
    return Xi[:, data.feats.index(name)]


def rule_ics(data, rounds, score_every: int, rules=RULES) -> pd.DataFrame:
    """Mean rank IC of each rule over each round's validation dates (the same dates the league scored: range(v0, v1, score_every)).
    Ridge is fitted on the round's training dates only (<= c), as the league's own ridge control is. -> DataFrame rounds x rules."""
    rows = {}
    for r in rounds:
        c, v0, v1 = data.round_geometry(r)
        idx = list(range(v0, v1, score_every))
        ridge = None
        if "ridge" in rules:
            X, y = data.stack_train(c)
            ridge = fit_ridge(X, y)
        row = {}
        for name in rules:
            row[name] = float(np.nanmean([rank_ic(rule_scores(data, name, data.day[i][1], ridge), data.day[i][2]) for i in idx]))
        rows[r] = row
    return pd.DataFrame(rows).T


def fly_ics(store, data, rounds) -> pd.DataFrame:
    """Per-round mean rank IC of the flies, recomputed from the league's saved scores (rounds/scores/rXX_<id>.npz).
    -> DataFrame rounds x ['fly mean', 'committee', 'n flies']. 'fly mean' averages every fly that scored the round."""
    from tournament.league import committee_scores
    rows = {}
    for r in rounds:
        per, dicts = [], []
        for aid in store.agent_ids():
            f = store.root / "rounds" / "scores" / f"r{r:02d}_{aid}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            d = {int(n[1:]): z[n].astype(float) for n in z.files}
            dicts.append(d)
            per.append(float(np.nanmean([rank_ic(v, data.day[i][2]) for i, v in d.items()])))
        if not dicts:
            continue
        com = committee_scores(dicts)
        rows[r] = {"fly mean": float(np.mean(per)), "committee": float(np.nanmean([rank_ic(v, data.day[i][2]) for i, v in com.items()])),
                   "n flies": len(per)}
    return pd.DataFrame(rows).T


def summarize(table: pd.DataFrame, rules) -> pd.DataFrame:
    """Across rounds: mean IC, and for each rule the mean (fly - rule) with a paired t. Negative = the rule beats the fly."""
    out = []
    for col in ["fly mean", "committee", *rules]:
        d = table["fly mean"] - table[col]
        n = len(d)
        t = float(d.mean() / (d.std(ddof=1) / np.sqrt(n))) if col not in ("fly mean",) and n > 2 and d.std(ddof=1) > 0 else np.nan
        out.append(dict(series=col, mean_ic=float(table[col].mean()), rounds_positive=int((table[col] > 0).sum()),
                        fly_minus_this=float(d.mean()) if col != "fly mean" else np.nan, paired_t=t))
    return pd.DataFrame(out).set_index("series")


def main():
    import argparse
    from tournament.data import Data
    from tournament.store import Store

    ap = argparse.ArgumentParser(description="Do the flies beat one-line rules? IC per finished round, no training for the rules.")
    ap.add_argument("--root", default="league_data")
    ap.add_argument("--rounds", default=None, help="e.g. 0-10 or 3,5,8 (default: every round with saved scores)")
    ap.add_argument("--out", default=None, help="optional CSV path for the per-round table")
    a = ap.parse_args()
    store = Store(a.root)
    cfg = store.load_config()
    print("loading panel...", flush=True)
    data = Data.from_cfg(cfg)
    if a.rounds:
        rounds = []
        for part in a.rounds.split(","):
            lo, _, hi = part.partition("-")
            rounds += list(range(int(lo), int(hi or lo) + 1))
    else:
        rounds = sorted({int(f.name[1:3]) for f in (store.root / "rounds" / "scores").glob("r*_*.npz")})
    rounds = [r for r in rounds if data.round_valid(r)]
    fl = fly_ics(store, data, rounds)
    ru = rule_ics(data, list(fl.index), cfg["score_every"])
    table = fl.join(ru)
    pd.set_option("display.width", 200)
    print("\nMean rank IC per round (higher = better)\n")
    print(table.drop(columns="n flies").round(3).to_string())
    print("\nAcross rounds (paired t: fly mean minus the rule; negative = rule is better)\n")
    print(summarize(table, RULES).round(3).to_string())
    if a.out:
        table.to_csv(a.out)
        print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
