"""Does blending signals beat any single one? (A2) No training: uses the flies' saved scores plus plain rules.

    python -m experiments.blend_test --root league_v7 --out blend

Per league round, on the round's own validation dates and only for stocks with real fundamentals (fund_known > 0), every series is
z-scored across stocks and scored with rank IC against (a) the league's own label and (b) the forward 63-day excess return.
Components: fly = equal-weight committee of every fly that scored the round (dead circuits left out); ridge = fitted on that round's
training dates only; fund = value+quality (earnings yield, FCF yield, ROE, ROA, minus accruals); rev = minus distance from the
50-day average (what the flies mostly learned). Blends are the equal-weight average of their z-scored parts, with no weights fitted.
Rounds' forward windows overlap, so the t-stats are optimistic upper bounds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.fund_rules import RULES as FUND_RULES, rule_score
from learning.ridge import fit_ridge, predict
from tournament.data import rank_ic

COMPONENTS = ("fly", "ridge", "fund", "rev")
BLENDS = {"fly": ("fly",), "ridge": ("ridge",), "fund": ("fund",), "rev": ("rev",),
          "fly+ridge": ("fly", "ridge"), "fly+fund": ("fly", "fund"), "ridge+fund": ("ridge", "fund"),
          "fly+ridge+fund": ("fly", "ridge", "fund")}
MIN_REAL = 20


def _z(s) -> np.ndarray:
    s = np.asarray(s, float)
    return (s - np.nanmean(s)) / (np.nanstd(s) + 1e-12)


def _committee(dicts, i, real):
    parts = [_z(np.asarray(d[i], float)[real]) for d in dicts if i in d and np.nanstd(np.asarray(d[i], float)) > 1e-9]
    return np.mean(parts, axis=0) if parts else None


def components_for_day(data, i, dicts, ridge):
    """({component: z-scored array over the real-fundamentals stocks}, y_league[real], real mask) or None when the day is unusable."""
    tick, X, y = data.day[i]
    real = X[:, data.feats.index("fund_known")] > 0
    if real.sum() < MIN_REAL:
        return None
    Xr = X[real]
    comps = {"ridge": _z(predict(Xr, *ridge)), "fund": _z(rule_score(data, Xr, FUND_RULES["value+quality"])),
             "rev": _z(-Xr[:, data.feats.index("ma50_dist")])}
    fly = _committee(dicts, i, real) if dicts else None
    if fly is not None:
        comps["fly"] = fly
    return comps, np.asarray(y, float)[real], real


def blend_ics(data, flies_by_round: dict, rounds, score_every: int, labels: dict | None = None, blends=BLENDS):
    """{label name: DataFrame rounds x blends of mean rank IC}, plus a DataFrame of the mean rank correlation between components.
    labels: {name: (date x ticker excess-return frame, horizon)}; 'league' (the league's own label) is always included."""
    labels = labels or {}
    names = ["league", *labels]
    rows = {n: {} for n in names}
    cor, ncor = {}, 0
    for r in rounds:
        c, v0, v1 = data.round_geometry(r)
        dicts = flies_by_round.get(r, [])
        ridge = fit_ridge(*data.stack_train(c))
        acc = {n: {b: [] for b in blends} for n in names}
        for i in range(v0, v1, score_every):
            got = components_for_day(data, i, dicts, ridge)
            if got is None:
                continue
            comps, y_league, real = got
            tick = data.day[i][0]
            ys = {"league": y_league}
            for n, (lw, h) in labels.items():
                if i + h <= data.n_usable and data.dates[i] in lw.index:
                    ys[n] = lw.loc[data.dates[i]].reindex(tick).to_numpy(float)[real]
            for b, parts in blends.items():
                if not all(p in comps for p in parts):
                    continue
                s = np.mean([comps[p] for p in parts], axis=0)
                for n, y in ys.items():
                    acc[n][b].append(rank_ic(s, y))
            if len(comps) == len(COMPONENTS):
                for a in COMPONENTS:
                    for b in COMPONENTS:
                        cor[(a, b)] = cor.get((a, b), 0.0) + rank_ic(comps[a], comps[b])
                ncor += 1
        for n in names:
            rows[n][r] = {b: (float(np.nanmean(v)) if v else np.nan) for b, v in acc[n].items()}
    tables = {n: pd.DataFrame(rows[n]).T for n in names}
    corr = pd.DataFrame({a: {b: cor.get((a, b), np.nan) / ncor if ncor else np.nan for b in COMPONENTS} for a in COMPONENTS})
    return tables, corr


def summarize(t: pd.DataFrame) -> pd.DataFrame:
    """Across rounds: mean IC, rounds positive, t of the mean vs zero, and the mean gap to the best single component (paired t)."""
    n = t.count()
    mean_t = t.mean() / (t.std(ddof=1) / np.sqrt(n))
    singles = [c for c in t.columns if c in COMPONENTS]
    best = t[singles].mean().idxmax() if singles else None
    gap = t.sub(t[best], axis=0) if best else None
    out = pd.DataFrame({"mean_ic": t.mean(), "rounds_positive": (t > 0).sum(), "rounds": n, "t": mean_t})
    if best:
        out[f"minus {best}"] = gap.mean()
        out["paired_t"] = gap.mean() / (gap.std(ddof=1) / np.sqrt(gap.count()))
        out.loc[best, ["paired_t"]] = np.nan
    return out


def load_flies(store, rounds) -> dict:
    out = {}
    for r in rounds:
        ds = []
        for aid in store.agent_ids():
            f = store.root / "rounds" / "scores" / f"r{r:02d}_{aid}.npz"
            if f.exists():
                z = np.load(f)
                ds.append({int(k[1:]): z[k].astype(float) for k in z.files})
        out[r] = ds
    return out


def main():
    import argparse

    from experiments.fund_rules import labels_for
    from tournament.data import Data, load_panel
    from tournament.store import Store

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="league_data")
    ap.add_argument("--horizons", default="63", help="extra forward-return labels in trading days (comma separated)")
    ap.add_argument("--out", default=None, help="CSV prefix")
    a = ap.parse_args()
    store = Store(a.root)
    cfg = store.load_config()
    print("loading panel...", flush=True)
    data = Data.from_cfg(cfg)
    panel = load_panel(cfg["panel"])
    rounds = [r for r in range(0, 200) if data.round_valid(r)
              and any((store.root / "rounds" / "scores").glob(f"r{r:02d}_*.npz"))]
    labels = {f"fwd{h}d": (labels_for(panel, int(h)), int(h)) for h in a.horizons.split(",") if h}
    print(f"{len(rounds)} rounds with saved fly scores: {rounds[0]}-{rounds[-1]}", flush=True)
    tables, corr = blend_ics(data, load_flies(store, rounds), rounds, cfg["score_every"], labels)
    pd.set_option("display.width", 220)
    for n, t in tables.items():
        t = t.dropna(how="all")
        title = "the league's own label" if n == "league" else f"forward {n[3:-1]}-day excess return"
        print(f"\n=== mean rank IC per round against {title} ===\n")
        print(t.round(3).to_string())
        print("\nacross rounds\n")
        print(summarize(t).round(3).to_string())
        if a.out:
            t.to_csv(f"{a.out}_{n}.csv")
    print("\nmean rank correlation between the components (low = they carry different information)\n")
    print(corr.round(2).to_string())


if __name__ == "__main__":
    main()
