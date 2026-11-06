"""Is parallel fly evaluation actually scaling?  Find out in about a minute.

    python -m experiments.bench_workers --workers 1 4 8 10
    python -m experiments.bench_workers --workers 1 4 8 10 --one-thread

Runs the same fixed workload (build a fly, then simulate N samples) in k processes at the same
time and reports how much slower each sample gets as k grows. Perfect scaling: every process
keeps the same ms/sample. If ms/sample balloons at k=8-10, the processes are fighting for CPU
(too many math threads per process, or more workers than physical cores).

  Run it twice: once plain, once with --one-thread (limits each process to one math thread).
  If --one-thread is much faster, run the evolution with:
      set OMP_NUM_THREADS=1 && set MKL_NUM_THREADS=1 && set OPENBLAS_NUM_THREADS=1
  and use roughly (physical cores) workers, not (hardware threads).
"""
import os
import sys

if "--one-thread" in sys.argv:  # must happen before numpy is imported
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_v] = "1"

import argparse  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402


def work(args):
    neurons, edges, n_samples, w = args
    from learning.fly_agent import FlyAgent

    t0 = time.time()
    fly = FlyAgent(neurons_csv=neurons, edges_csv=edges, n_obs=13, sim_ms=20, warm_ms=15,
                   input_scale=0.5, has_position=False, w_kc_mbon=w, seed=0)
    build = time.time() - t0
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (n_samples, 13))
    t1 = time.time()
    for x in X:
        fly.reset_state()
        fly.act(x)
    return build, (time.time() - t1) / n_samples * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 10])
    ap.add_argument("--samples", type=int, default=300, help="samples per process")
    ap.add_argument("--w", type=float, default=0.08)
    ap.add_argument("--neurons", default="data/mb_neurons.csv")
    ap.add_argument("--edges", default="data/mb_edges.csv")
    ap.add_argument("--one-thread", action="store_true")
    a = ap.parse_args()

    print(f"cpu threads reported: {os.cpu_count()}   one-thread mode: {a.one_thread}   "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '(unset)')}")
    print(f"{'procs':>6}{'wall s':>8}{'build s':>9}{'ms/sample':>11}{'vs 1 proc':>11}{'total samples/s':>17}{'speedup':>9}")
    base_ms = base_rate = None
    for k in a.workers:
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=k) as ex:
            res = list(ex.map(work, [(a.neurons, a.edges, a.samples, a.w)] * k))
        wall = time.time() - t0
        build = float(np.mean([r[0] for r in res]))
        ms = float(np.mean([r[1] for r in res]))
        rate = k * 1000.0 / ms
        if base_ms is None:
            base_ms, base_rate = ms, rate
        print(f"{k:>6}{wall:>8.1f}{build:>9.1f}{ms:>11.1f}{ms / base_ms:>10.1f}x{rate:>17.0f}{rate / base_rate:>8.1f}x",
              flush=True)


if __name__ == "__main__":
    main()
