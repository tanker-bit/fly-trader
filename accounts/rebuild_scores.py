"""Rebuild a finished round's saved scores from the flies' stored weights (for leagues run before scores were saved).

    python -m accounts.rebuild_scores --root league_data --round 14

Only flies still alive whose weights were last saved after training through that round can be rebuilt (culled flies had their
weights deleted). Each rebuilt fly's mean IC is compared with the IC recorded in its history: they must match, otherwise the
fly is skipped. Writes league_data/rounds/scores/rXX_<id>.npz, the files accounts.league_report reads.
"""
from __future__ import annotations

import argparse

import numpy as np

from tournament.data import Data, rank_ic
from tournament.league import fly_scores, load_agent_obj
from tournament.store import Store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="league_data")
    ap.add_argument("--round", type=int, required=True)
    a = ap.parse_args()
    store = Store(a.root)
    cfg = store.load_config()
    data = Data.from_cfg(cfg)
    c, v0, v1 = data.round_geometry(a.round)
    sdir = store.root / "rounds" / "scores"
    sdir.mkdir(parents=True, exist_ok=True)
    done = 0
    for aid in store.agent_ids():
        rec = store.load_agent(aid)
        h = [x for x in rec["history"] if x["round"] == a.round]
        if rec["status"] != "alive" or not h or rec["last_round_done"] != a.round or rec["trained_idx"] != c \
                or store.load_weights(aid) is None:
            print(f"  skip {aid}: not rebuildable (culled, or weights not from the end of round {a.round})")
            continue
        ag = load_agent_obj(rec, store, cfg, data)
        sc = {i: fly_scores(ag, data.day[i][1]) for i in range(v0, v1, cfg["score_every"])}
        ic = float(np.mean([rank_ic(sc[i], data.day[i][2]) for i in sc]))
        if abs(ic - h[0]["ic"]) > 1e-4:
            print(f"  skip {aid}: rebuilt IC {ic:+.4f} != recorded {h[0]['ic']:+.4f}")
            continue
        np.savez_compressed(sdir / f"r{a.round:02d}_{aid}.npz", **{f"d{i}": v.astype(np.float32) for i, v in sc.items()})
        done += 1
        print(f"  {aid}: rebuilt, IC {ic:+.4f} matches history")
    print(f"\n{done} flies rebuilt into {sdir}")


if __name__ == "__main__":
    main()
