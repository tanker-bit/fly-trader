"""The league: a persistent population of flies that keeps training, gets ranked, culled and bred.

One ROUND =
  1. every living agent continues training (from its saved weights) on dates up to the round's cut,
  2. every agent is scored on the next validation window (after a 21-day gap, so no label overlaps),
  3. rank by mean validation IC over the last 3 rounds; keep the top `keep`, cull the rest,
  4. refill to `pop` with mutated children of the survivors (inheriting trained weights when the wiring is kept).
Everything is saved after each agent, so the run can be stopped and resumed. See experiments/run_league.py.
"""
from __future__ import annotations

import contextlib
import io
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

from env.portfolio import report
from learning.ridge import fit_ridge, predict
from tournament.data import GAP, Data, rank_ic
from tournament.genome import default_genome, mutate, random_genome
from tournament.store import Store

CODE_VERSION = "v6-silent-guard-committee"
G: dict = {}   # per-process globals (cfg, data, store)


# ---------------------------------------------------------------- agents
def build_agent(g: dict, cfg: dict, n_obs: int):
    from experiments.v3_counterfactual import CounterfactualFly
    return CounterfactualFly(neurons_csv=cfg["neurons"], edges_csv=cfg["edges"], n_obs=n_obs, sim_ms=cfg["sim_ms"],
                             warm_ms=cfg["warm_ms"], kc_sparsity=g["kc_sparsity"], kc_inputs=int(g["kc_inputs"]),
                             kc_drive=g["kc_drive"], w_kc_mbon=g["w_kc_mbon"], tau_plastic=g["tau_plastic"],
                             seed=int(g["wiring_seed"]), input_scale=0.5, has_position=False)


def new_record(aid, genome, born_round, parent=None, gen=0, inherited=False):
    return dict(id=aid, genome=genome, parent=parent, gen=gen, born_round=born_round, last_round_done=born_round - 1,
                status="alive", n_seen=0, trained_idx=-1, r_mean=0.0, r_var=1e-4, history=[],
                code_version=CODE_VERSION, inherited_weights=inherited, created=datetime.now().isoformat(timespec="seconds"))


def sel_score(rec) -> float:
    """Mean validation IC over the last 3 rounds. A SILENT agent (constant score, so IC is exactly 0) gets -9:
    in the 15-round baseline silent agents beat everyone in bad years (0 > negative) and kept being selected."""
    h = rec["history"][-3:]
    if not h or h[-1].get("silent"):
        return -9.0
    live = [x["ic"] for x in h if not x.get("silent")]
    return float(np.mean(live)) if live else -9.0


def fly_scores(ag, X):
    out = np.empty(len(X))
    for k, x in enumerate(X):
        ag.reset_state()
        ag.act(x)
        out[k] = ag.last_score
    return out


def load_agent_obj(rec, store, cfg, data):
    ag = build_agent(rec["genome"], cfg, len(data.feats))
    W = store.load_weights(rec["id"])
    if W is not None and W.shape == ag.syn.W.shape:
        ag.syn.W = W.astype(np.float64) * ag.mask
    ag.r_mean, ag.r_var = rec["r_mean"], rec["r_var"]
    return ag


def train_through(ag, rec, data, upto, cfg):
    """Continue online counterfactual-dopamine training over dates (trained_idx, upto]."""
    g = rec["genome"]
    rng = np.random.default_rng([int(g["wiring_seed"]), rec["n_seen"]])
    for i in range(rec["trained_idx"] + 1, upto + 1):
        _, X, y = data.day[i]
        for j in rng.choice(len(X), size=min(cfg["stocks_per_day"], len(X)), replace=False):
            ag.syn.lr = g["lr0"] / (1.0 + rec["n_seen"] / g["tau_anneal"])
            ag.reset_state()
            ag.act(X[j])
            ag.learn_counterfactual(ag.last_pre.copy(), y[j])
            rec["n_seen"] += 1
    rec["trained_idx"] = max(rec["trained_idx"], upto)
    rec["r_mean"], rec["r_var"] = float(ag.r_mean), float(ag.r_var)


# ---------------------------------------------------------------- worker tasks
def init_worker(cfg):
    G["cfg"] = cfg
    G["store"] = Store(cfg["root"])
    G["data"] = Data(cfg["panel"], cfg["train_days"], cfg["val_days"], cfg["holdout"])


def run_agent_round(aid, r):
    cfg, data, store = G["cfg"], G["data"], G["store"]
    rec = store.load_agent(aid)
    c, v0, v1 = data.round_geometry(r)
    ag = load_agent_obj(rec, store, cfg, data)
    t0 = time.time()
    train_through(ag, rec, data, c, cfg)
    sc = {i: fly_scores(ag, data.day[i][1]) for i in range(v0, v1, cfg["score_every"])}
    ics = [rank_ic(sc[i], data.day[i][2]) for i in sc]
    sdir = store.root / "rounds" / "scores"            # kept so the round can score a committee of these agents
    sdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sdir / f"r{r:02d}_{aid}.npz", **{f"d{i}": v.astype(np.float32) for i, v in sc.items()})
    rec["history"].append(dict(round=r, ic=float(np.mean(ics)), sd=float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0,
                               n=len(ics)))
    rec["last_round_done"] = r
    store.save_weights(aid, ag.syn.W)
    store.save_agent(rec)
    return dict(id=aid, ic=rec["history"][-1]["ic"], secs=time.time() - t0)


def run_agent_final(aid):
    """Train through the last label-safe date, then score the holdout decision dates. Saves nothing."""
    cfg, data, store = G["cfg"], G["data"], G["store"]
    rec = store.load_agent(aid)
    h0 = data.n_usable
    ag = load_agent_obj(rec, store, cfg, data)
    train_through(ag, rec, data, h0 - 1 - GAP, cfg)
    return aid, {i: fly_scores(ag, data.day[i][1]) for i in range(h0, len(data.dates), GAP)}


def _map(fn, jobs, cfg, workers):
    """Run fn(*job) for each job, in-process or in a pool. Yields results as they finish."""
    if workers <= 1:
        init_worker(cfg)
        for job in jobs:
            yield fn(*job)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(cfg,)) as ex:
            for f in as_completed([ex.submit(fn, *job) for job in jobs]):
                yield f.result()


# ---------------------------------------------------------------- controls
def control_scores(data, idxs, fit_upto, rng, extra_fly=None):
    """Score arrays per date index for the non-learning baselines (ridge fitted on dates <= fit_upto)."""
    X, y = data.stack_train(fit_upto)
    coef, xm = fit_ridge(X, y)
    out = {"ridge": {}, "momentum 12-1": {}, "reversal (-ret_1)": {}, "random": {}}
    for i in idxs:
        Xi = data.day[i][1]
        out["ridge"][i] = predict(Xi, coef, xm)
        out["momentum 12-1"][i] = Xi[:, data.i_mom]
        out["reversal (-ret_1)"][i] = -Xi[:, data.i_rev]
        out["random"][i] = rng.normal(size=len(Xi))
    return out


def untrained_control(data, cfg, idxs):
    ag = build_agent(default_genome(), cfg, len(data.feats))
    return {i: fly_scores(ag, data.day[i][1]) for i in idxs}


def _z(s):
    s = np.asarray(s, float)
    return (s - s.mean()) / (s.std() + 1e-12)


def committee_scores(score_dicts):
    """Average of per-date z-scored scores across agents: {date_idx: committee score array}."""
    idxs = score_dicts[0].keys()
    return {i: np.mean([_z(d[i]) for d in score_dicts], axis=0) for i in idxs}


def committee_stats(store, data, r, agent_ids):
    """IC of the equal-weight committee of `agent_ids` on round r's validation dates, next to their mean individual IC."""
    dicts = []
    for aid in agent_ids:
        f = store.root / "rounds" / "scores" / f"r{r:02d}_{aid}.npz"
        if f.exists():
            z = np.load(f)
            dicts.append({int(k[1:]): z[k].astype(float) for k in z.files})
    if not dicts:
        return None
    com = committee_scores(dicts)
    ind = [np.mean([rank_ic(d[i], data.day[i][2]) for i in d]) for d in dicts]
    return dict(n=len(dicts), ic=float(np.mean([rank_ic(com[i], data.day[i][2]) for i in com])),
                mean_individual_ic=float(np.mean(ind)))


def summarize(data, scores_by_idx):
    ics = [rank_ic(s, data.day[i][2]) for i, s in scores_by_idx.items()]
    return dict(ic=float(np.mean(ics)), sd=float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0)


# ---------------------------------------------------------------- selection and breeding
def select_and_breed(store, cfg, r):
    """Idempotent: safe to re-run after an interruption (children are only created up to `pop`)."""
    rng = np.random.default_rng([cfg["seed"], r])
    elig = [a for a in store.alive() if a["born_round"] <= r and a["last_round_done"] == r]
    elig.sort(key=lambda a: -sel_score(a))
    keep, culled = elig[:cfg["keep"]], elig[cfg["keep"]:]
    for a in culled:
        a["status"] = "culled"
        a["culled_round"] = r
        store.save_agent(a)
        store.drop_weights(a["id"])
    need = cfg["pop"] - len(store.alive())
    w = np.arange(len(keep), 0, -1, dtype=float)
    kids = []
    for _ in range(max(0, need)):
        parent = keep[int(rng.choice(len(keep), p=w / w.sum()))]
        inherit = bool(rng.random() < 0.7) and store.load_weights(parent["id"]) is not None
        st = store.load_state()
        cid = f"a{st['next_id']:04d}"
        st["next_id"] += 1
        st["trials"] += 1
        store.save_state(st)
        child = new_record(cid, mutate(parent["genome"], rng, rewire=not inherit, freeze_init=inherit, n_obs=cfg.get("n_obs", 13)),
                           born_round=r + 1, parent=parent["id"], gen=parent["gen"] + 1, inherited=inherit)
        if inherit:
            store.save_weights(cid, store.load_weights(parent["id"]))
            for k in ("n_seen", "trained_idx", "r_mean", "r_var"):
                child[k] = parent[k]
        store.save_agent(child)
        kids.append(child)
    return keep, culled, kids


# ---------------------------------------------------------------- reporting
def leaderboard_text(store, data, cfg, r, controls, keep_ids, committee=None):
    c, v0, v1 = data.round_geometry(r)
    st = store.load_state()
    rows = [a for a in (store.load_agent(i) for i in store.agent_ids())
            if a["history"] and a["history"][-1]["round"] == r]
    rows.sort(key=lambda a: -sel_score(a))
    n_eff = cfg["val_days"] / GAP
    sd = float(np.mean([a["history"][-1]["sd"] for a in rows])) if rows else 0.0
    n_tried = sum(1 for i in store.agent_ids() if store.load_agent(i)["born_round"] <= r)   # agents scored up to this round
    bar = sd * np.sqrt(2 * np.log(max(n_tried, 2))) / np.sqrt(n_eff)
    L = [f"# Round {r}   train through {data.dates[c].date()}   validate {data.dates[v0].date()} -> {data.dates[v1 - 1].date()}",
         f"agents scored {len(rows)}, agents tried so far {n_tried}, validation noise (se of mean IC) about +/-{sd / np.sqrt(n_eff):.3f}",
         "", f"{'rank':>4} {'id':<6}{'gen':>4} {'parent':<7}{'IC now':>8}{'3-rd avg':>9}{'rounds':>7}  {'lr0':>6}{'tau_an':>8}{'kc_sp':>6}{'drive':>6}  fate"]
    for k, a in enumerate(rows, 1):
        g = a["genome"]
        L.append(f"{k:>4} {a['id']:<6}{a['gen']:>4} {str(a['parent'] or '-'):<7}{a['history'][-1]['ic']:>+8.3f}{sel_score(a):>+9.3f}"
                 f"{len(a['history']):>7}  {g['lr0']:>6.3f}{g['tau_anneal']:>8.0f}{g['kc_sparsity']:>6.3f}{g['kc_drive']:>6.2f}  "
                 f"{'KEPT' if a['id'] in keep_ids else 'culled'}{'  SILENT' if a['history'][-1].get('silent') else ''}")
    L += ["", "Controls on the same validation window (no learning from this window):"]
    for name, v in controls.items():
        L.append(f"  {name:<20} IC {v['ic']:+.3f}")
    for label, c in (committee or {}).items():
        if c:
            L.append(f"  {label:<20} IC {c['ic']:+.3f}   (its {c['n']} members average {c['mean_individual_ic']:+.3f} individually)")
    best = rows[0]["history"][-1]["ic"] if rows else 0.0
    L += ["", f"Luck bar for the best of {n_tried} agents: {bar:.3f}.  Best IC this round: {best:+.3f}  -> "
              + ("above the bar" if best > bar else "NOT above the bar: this round's ranking is mostly noise.")]
    return "\n".join(L)


# ---------------------------------------------------------------- driver
def init_league(store, cfg):
    store.save_config(cfg)
    store.save_state(dict(round=0, next_id=0, trials=0))
    rng = np.random.default_rng(cfg["seed"])
    for _ in range(cfg["pop"]):
        st = store.load_state()
        aid = f"a{st['next_id']:04d}"
        st["next_id"] += 1
        st["trials"] += 1
        store.save_state(st)
        store.save_agent(new_record(aid, random_genome(rng, cfg.get('n_obs', 13)), born_round=0))


def run_round(store, cfg, data, workers):
    st = store.load_state()
    r = st["round"]
    c, v0, v1 = data.round_geometry(r)
    pending = [a["id"] for a in store.alive() if a["born_round"] <= r and a["last_round_done"] < r]
    survivors_in = [a["id"] for a in store.alive() if a["born_round"] < r]     # chosen BEFORE seeing this round
    print(f"\n=== round {r}: train through {data.dates[c].date()}, validate {data.dates[v0].date()} -> "
          f"{data.dates[v1 - 1].date()}; {len(pending)} agents to run, workers={workers} ===", flush=True)
    t0 = time.time()
    for k, res in enumerate(_map(run_agent_round, [(aid, r) for aid in pending], cfg, workers), 1):
        print(f"  [{k}/{len(pending)}] {res['id']}  IC {res['ic']:+.3f}  ({res['secs']:.0f}s)", flush=True)
    if "data" not in G:
        init_worker(cfg)
    rfile = store.root / "rounds" / f"round_{r:02d}.json"
    idxs = list(range(v0, v1, cfg["score_every"]))
    if rfile.exists():
        controls = store._read(rfile)["controls"]
    else:
        cs = control_scores(data, idxs, c, np.random.default_rng([cfg["seed"], r, 7]))
        controls = {k: summarize(data, v) for k, v in cs.items()}
        controls["fly untrained (default)"] = summarize(data, untrained_control(data, cfg, idxs))
    committee = {"Committee (kept last round)": committee_stats(store, data, r, survivors_in) if survivors_in else None,
                 "Committee (all scored)": committee_stats(store, data, r, [a["id"] for a in store.alive() if a["last_round_done"] == r])}
    store._write(rfile, dict(round=r, controls=controls, committee=committee))
    keep, culled, kids = select_and_breed(store, cfg, r)
    text = leaderboard_text(store, data, cfg, r, controls, {a["id"] for a in keep}, committee)
    (store.root / "LEADERBOARD.md").write_text(text)
    (store.root / "rounds" / f"leaderboard_{r:02d}.txt").write_text(text)
    print("\n" + text)
    print(f"\n{len(culled)} culled, {len(kids)} new children ({sum(k['inherited_weights'] for k in kids)} inherit weights). "
          f"round took {time.time() - t0:.0f}s", flush=True)
    st = store.load_state()
    st["round"] = r + 1
    store.save_state(st)


def run_final(store, cfg, data, workers, force=False):
    lock = store.root / "holdout_used.txt"
    if lock.exists() and not force:
        print("REFUSED: the final holdout was already used:\n" + lock.read_text() +
              "\nUsing it again turns it into a second validation set. Use --force only if you accept that.")
        return
    lock.write_text(f"used {datetime.now().isoformat(timespec='seconds')} at round {store.load_state()['round']}\n")
    h0 = data.n_usable
    agents = sorted(store.alive(), key=lambda a: -sel_score(a))[:cfg["keep"]]
    print(f"FINAL HOLDOUT {data.dates[h0].date()} -> {data.dates[-1].date()}  ({len(data.dates) - h0} days; "
          f"decisions every {GAP} days). Scoring {len(agents)} agents.", flush=True)
    idxs = list(range(h0, len(data.dates), GAP))
    days = data.dates[h0:]

    def frame(by_idx):
        f = pd.DataFrame(np.nan, index=days, columns=data.fwd1.columns)
        for i, s in by_idx.items():
            f.loc[data.dates[i], data.day[i][0]] = s
        return f

    signals = {}
    cs = control_scores(data, idxs, h0 - 1 - GAP, np.random.default_rng([cfg["seed"], 999]))
    for k, v in cs.items():
        signals[k] = frame(v)
    signals["fly untrained (default)"] = frame(untrained_control(data, cfg, idxs))
    all_scores = []
    for aid, by_idx in _map(run_agent_final, [(a["id"],) for a in agents], cfg, workers):
        signals[f"league {aid}"] = frame(by_idx)
        all_scores.append(by_idx)
        print(f"  scored {aid}", flush=True)
    if all_scores:
        signals[f"committee of top {len(all_scores)}"] = frame(committee_scores(all_scores))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report(signals, data.fwd1.reindex(days), data.y_frame(range(h0, len(data.dates))).reindex(days),
               rebalances=(GAP,))
    out = ("=== FINAL HOLDOUT, monthly rebalance, long-short, net Sharpe at 0/5/10/20 bps ===\n" + buf.getvalue() +
           f"\nVariants tried before this: {store.load_state()['trials']}. IC t-stats use non-overlapping monthly decisions "
           f"({len(idxs)} of them): treat |t| < 2 as no evidence.")
    (store.root / "FINAL_HOLDOUT.txt").write_text(out)
    print(out)


def health_check(a, n_flies=12, n_samples=60):
    """Draw random genomes, run each on real stock-days and report firing and score spread. Silent flies score the same
    for every stock (std 0) and can never earn a non-zero IC, so a high silent share means the genome bounds are wrong for
    this panel (for example after changing the number of inputs)."""
    data = Data(a.panel, a.train_days, a.val_days, a.holdout)
    cfg = dict(neurons=a.neurons, edges=a.edges, sim_ms=a.sim_ms, warm_ms=a.warm_ms)
    rng = np.random.default_rng(a.seed + 5)
    pick = rng.choice(data.n_usable, size=n_samples, replace=False)
    X = np.array([data.day[i][1][rng.integers(len(data.day[i][1]))] for i in pick])
    print(f"{len(data.feats)} inputs; {n_flies} random flies on {n_samples} real stock-days")
    print(f"{'kc_in':>6}{'drive':>7}{'kc_sp':>7}{'w':>7}{'MBON Hz':>9}{'score sd':>10}  status")
    silent = 0
    for _ in range(n_flies):
        g = random_genome(rng, len(data.feats))
        ag = build_agent(g, cfg, len(data.feats))
        hz, sc = [], []
        for x in X:
            ag.reset_state()
            ag.act(x)
            hz.append(ag.last_rates.mean())
            sc.append(ag.last_score)
        dead = np.std(sc) < 1e-12
        silent += dead
        print(f"{g['kc_inputs']:>6}{g['kc_drive']:>7.2f}{g['kc_sparsity']:>7.3f}{g['w_kc_mbon']:>7.3f}{np.mean(hz):>9.1f}{np.std(sc):>10.3f}  {'SILENT' if dead else 'ok'}")
    print(f"\n{silent}/{n_flies} silent. " + ("Raise the kc_drive floor or w_kc_mbon floor in tournament/genome.py." if silent > n_flies // 4 else "Bounds look usable."))
