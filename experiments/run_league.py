"""Run the fly league (persistent population, culling and breeding).

    python -m experiments.run_league --workers 4              # first call creates the league, runs 1 round
    python -m experiments.run_league --workers 4 --rounds 3   # more rounds (resumes where it stopped)
    python -m experiments.run_league --status                 # show the latest leaderboard
    python -m experiments.run_league --final                  # ONE-TIME test on the untouched last ~500 days

Set OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS to 1 first when using --workers > 1.
League settings are fixed at creation (stored in league_data/config.json); to change them, use a new --root.
All files live in --root (default league_data). Stopping with Ctrl+C is safe: rerun the same command to resume.
"""
from __future__ import annotations

import argparse

from tournament.data import Data
from tournament.summary import summary_text
from tournament.league import G, init_league, init_worker, run_final, run_round
from tournament.store import Store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="league_data")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--summary", action="store_true", help="cross-round table: flies vs ridge and the other controls")
    ap.add_argument("--health-check", action="store_true", help="before creating a league: are random flies silent on this panel?")
    ap.add_argument("--force", action="store_true")
    # settings used only when the league is first created
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--keep", type=int, default=10)
    ap.add_argument("--train-days", type=int, default=1000, help="initial training window (dates)")
    ap.add_argument("--val-days", type=int, default=252, help="validation window per round (dates)")
    ap.add_argument("--holdout", type=int, default=500, help="final untouched dates")
    ap.add_argument("--stocks-per-day", type=int, default=8)
    ap.add_argument("--score-every", type=int, default=5, help="score every k-th validation date (speed)")
    ap.add_argument("--sim-ms", type=int, default=20)
    ap.add_argument("--warm-ms", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--panel", default="data/panel.parquet")
    ap.add_argument("--neurons", default="data/mb_neurons.csv")
    ap.add_argument("--edges", default="data/mb_edges.csv")
    a = ap.parse_args()

    store = Store(a.root)
    if a.status:
        p = store.root / "LEADERBOARD.md"
        print(p.read_text() if p.exists() else "no rounds finished yet")
        return
    if a.summary:
        text = summary_text(store)
        (store.root / "SUMMARY.md").write_text(text)
        print(text)
        return
    if a.health_check:
        from tournament.league import health_check
        health_check(a)
        return
    if not store.has_config():
        n_obs = len(Data(a.panel, a.train_days, a.val_days, a.holdout).feats)
        cfg = dict(n_obs=n_obs, root=a.root, pop=a.pop, keep=a.keep, train_days=a.train_days, val_days=a.val_days,
                   holdout=a.holdout, stocks_per_day=a.stocks_per_day, score_every=a.score_every, sim_ms=a.sim_ms,
                   warm_ms=a.warm_ms, seed=a.seed, panel=a.panel, neurons=a.neurons, edges=a.edges)
        init_league(store, cfg)
        print(f"created league in {a.root}: {a.pop} random flies (keep {a.keep})")
    cfg = store.load_config()
    init_worker(cfg)                      # loads the data in this process too (also used for controls)
    data = G["data"]
    print(f"{len(data.dates)} labelled dates {data.dates[0].date()} -> {data.dates[-1].date()}; "
          f"league uses up to {data.dates[data.n_usable - 1].date()}, holdout after that")
    if a.final:
        run_final(store, cfg, data, a.workers, a.force)
        return
    for _ in range(a.rounds):
        r = store.load_state()["round"]
        if not data.round_valid(r):
            print(f"round {r} would run into the holdout: no more rounds. Use --final for the one-time test.")
            break
        run_round(store, cfg, data, a.workers)


if __name__ == "__main__":
    main()
