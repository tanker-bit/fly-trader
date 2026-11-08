"""Cross-round summary of a league: the flies against ridge and the other controls, round by round.

Everything comes from files already on disk (agent histories + rounds/round_XX.json), so it also works on a league
that has finished. Honest-evaluation rules built in:
  * "survivors" = agents that were kept BEFORE the round started (their score in that round is out-of-sample for
    the selection); "children" = agents created at the end of the previous round.
  * rounds use disjoint validation windows, so they are treated as roughly independent samples for the t-stats.
  * selection value = survivors minus children; persistence = rank correlation of agents' IC between consecutive
    rounds. If both are ~0 the culling is not finding skill.
"""
from __future__ import annotations

import numpy as np

CONTROLS = ["ridge", "momentum 12-1", "reversal (-ret_1)", "random", "fly untrained (default)"]


def _spearman(a, b):
    if len(a) < 5:
        return np.nan
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if np.std(ra) > 0 and np.std(rb) > 0 else np.nan


def collect(store):
    recs = [store.load_agent(i) for i in store.agent_ids()]
    rounds = sorted({h["round"] for a in recs for h in a["history"]})
    out, prev = [], {}
    for r in rounds:
        here = [(a, next(h for h in a["history"] if h["round"] == r)) for a in recs if any(h["round"] == r for h in a["history"])]
        ics = {a["id"]: h["ic"] for a, h in here}
        surv = [h["ic"] for a, h in here if a["born_round"] < r]
        kids = [h["ic"] for a, h in here if a["born_round"] >= r]
        silent = sum(1 for a, h in here if h.get("silent") or h["ic"] == 0.0)
        both = [i for i in ics if i in prev]
        rf = store.root / "rounds" / f"round_{r:02d}.json"
        info = store._read(rf) if rf.exists() else {}
        row = dict(round=r, n=len(here), all=float(np.mean(list(ics.values()))),
                   surv=float(np.mean(surv)) if surv else np.nan, kids=float(np.mean(kids)) if kids else np.nan,
                   best=float(max(ics.values())), silent=silent,
                   persist=_spearman([prev[i] for i in both], [ics[i] for i in both]),
                   committee=((info.get("committee") or {}).get("Committee (kept last round)") or {}).get("ic", np.nan),
                   committee_all=((info.get("committee") or {}).get("Committee (all scored)") or {}).get("ic", np.nan))
        for k in CONTROLS:
            row[k] = (info.get("controls") or {}).get(k, {}).get("ic", np.nan)
        out.append(row)
        prev = ics
    return out


def _stat(name, a):
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    if len(a) < 2:
        return f"{name:<38} n/a"
    se = a.std(ddof=1) / np.sqrt(len(a))
    return f"{name:<38} mean {a.mean():+.4f}  se {se:.4f}  t {a.mean() / se if se > 0 else float('nan'):+.2f}  rounds>0 {int((a > 0).sum())}/{len(a)}"


def summary_text(store):
    rows = collect(store)
    if not rows:
        return "no rounds with results yet"
    g = lambda k: np.array([x[k] for x in rows], float)
    L = ["# Cross-round summary (IC = mean rank correlation with the 21-day excess return)", "",
         f"{'rd':>3}{'n':>4}{'all':>8}{'surv':>8}{'kids':>8}{'best':>8}{'dead':>5}{'comm':>8}{'ridge':>8}{'mom':>8}{'rev':>8}{'rand':>8}{'untr':>8}{'persist':>9}"]
    for x in rows:
        L.append(f"{x['round']:>3}{x['n']:>4}{x['all']:>+8.3f}{x['surv']:>+8.3f}{x['kids']:>+8.3f}{x['best']:>+8.3f}{x['silent']:>5}"
                 f"{x['committee']:>+8.3f}{x['ridge']:>+8.3f}{x['momentum 12-1']:>+8.3f}{x['reversal (-ret_1)']:>+8.3f}"
                 f"{x['random']:>+8.3f}{x['fly untrained (default)']:>+8.3f}{x['persist']:>+9.2f}")
    L += ["", "all = mean IC of every agent scored; surv = agents kept before the round; kids = new that round; dead = IC exactly 0 (silent);",
          "comm = committee of last round's survivors (needs scores saved by v6 code; blank for older runs); persist = rank corr of agents' IC with the previous round.", ""]
    L.append("Levels (mean over rounds):")
    for lab, k in (("fly, all agents", "all"), ("ridge", "ridge"), ("momentum 12-1", "momentum 12-1"), ("reversal", "reversal (-ret_1)"),
                   ("random", "random"), ("fly untrained", "fly untrained (default)"), ("committee of survivors", "committee")):
        L.append(_stat(lab, g(k)))
    L += ["", "Differences (what matters):"]
    L.append(_stat("fly - ridge", g("all") - g("ridge")))
    L.append(_stat("fly - untrained", g("all") - g("fly untrained (default)")))
    L.append(_stat("fly - random", g("all") - g("random")))
    L.append(_stat("fly - momentum", g("all") - g("momentum 12-1")))
    if not np.all(np.isnan(g("committee"))):
        L.append(_stat("committee - mean individual fly", g("committee") - g("surv")))
        L.append(_stat("committee - ridge", g("committee") - g("ridge")))
    L += ["", "Does the culling find skill?"]
    L.append(_stat("survivors - children (selection value)", g("surv") - g("kids")))
    L.append(_stat("persistence (rank corr, round to round)", g("persist")))
    L += ["", "Reading guide: |t| < 2 means no evidence. Persistence near 0 or negative and selection value near 0 mean the",
          "ranking inside a round is noise, so evolving among similar flies adds nothing. 'dead' agents scoring 0 should be 0 after v6."]
    return "\n".join(L)
