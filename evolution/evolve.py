"""The evolution loop: a population of flies, scored on validation data, culled and mutated.

Selection only ever sees the validation blocks. The holdout is touched once, at the end, by
`evaluate.holdout_report`. Fitness is deterministic for a given genome, so survivors are cached
and never re-scored; `n_trials` counts every distinct genome evaluated (needed for the luck bar).
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from evolution.evaluate import evaluate_genome, null_bar
from evolution.genome import Genome, crossover, mutate, random_genome

_SPLITS: dict = {}
_CFG: dict = {}


def _init_worker(splits, cfg):
    global _SPLITS, _CFG
    _SPLITS, _CFG = splits, cfg


def _eval_one(g: Genome) -> dict:
    return evaluate_genome(g, _SPLITS, _CFG)


def _score_all(genomes, splits, cfg, workers):
    if workers <= 1:
        _init_worker(splits, cfg)
        return [_eval_one(g) for g in genomes]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(splits, cfg)) as ex:
        return list(ex.map(_eval_one, genomes))


def evolve(splits: dict, cfg: dict, pop: int = 24, gens: int = 10, workers: int = 1, seed: int = 0,
           elites: int = 2, p_cross: float = 0.3, log=print) -> dict:
    rng = np.random.default_rng(seed)
    n_val_total = sum(len(v[1]) for v in splits["val"])
    default = Genome()
    population = [default] + [random_genome(rng) for _ in range(pop - 1)]
    scored: dict[Genome, dict] = {}
    history = []
    t0 = time.time()

    for gen in range(gens + 1):
        todo = [g for g in dict.fromkeys(population) if g not in scored]   # unique, not yet scored
        for g, res in zip(todo, _score_all(todo, splits, cfg, workers)):
            scored[g] = res
        ranked = sorted(set(population), key=lambda g: scored[g]["fitness"], reverse=True)
        fits = np.array([scored[g]["fitness"] for g in population])
        n_trials = len(scored)
        bar = null_bar(n_trials, n_val_total)
        best = scored[ranked[0]]["fitness"]
        history.append({"gen": gen, "best": best, "median": float(np.median(fits)),
                        "n_trials": n_trials, "luck_bar": bar})
        log(f"gen {gen:>2}  best val IC {best:+.3f}  median {np.median(fits):+.3f}  "
            f"trials {n_trials:>4}  luck bar {bar:.3f}  ({time.time() - t0:.0f}s)")
        if gen == gens:
            break

        # Truncation selection: top half survive; elites carry over unchanged; the rest are children.
        survivors = ranked[:max(2, pop // 2)]
        nxt = list(ranked[:elites])
        while len(nxt) < pop:
            a, b = (survivors[i] for i in rng.choice(len(survivors), size=2, replace=False))
            child = crossover(a, b, rng) if rng.random() < p_cross else a
            nxt.append(mutate(child, rng))
        population = nxt

    champion = max(scored, key=lambda g: scored[g]["fitness"])
    return {"champion": champion, "champion_fitness": scored[champion]["fitness"],
            "champion_val_ic": scored[champion]["val_ic"], "history": history,
            "n_trials": len(scored), "n_val_total": n_val_total, "default": default,
            "default_fitness": scored[default]["fitness"]}
