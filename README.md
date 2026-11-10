# fly-trader (STONKFLY)

**Can a simulated fruit-fly brain pick stocks?** This research project wires the real *Drosophila* mushroom-body connectome (hemibrain: Kenyon cells to output neurons, with dopamine-gated learning) into a stock-ranking pipeline, then tests it with unusually strict statistics.

> **Status: research only. No tradable edge has been demonstrated.** The honest result so far is that the fly performs about the same as a random sparse code and a ridge regression (roughly +0.01 rank IC), and nothing survives multiple-testing correction. This is not financial advice and nothing here trades real money.

## What is in it

- **Real connectome.** About 1,900 Kenyon cells, 70 output neurons (MBONs) and 300 dopamine neurons from neuPrint (hemibrain v1.2.1), simulated as a leaky integrate-and-fire network. Only the KC-to-MBON synapses learn.
- **Fly learning.** A three-factor rule (pre-synaptic activity x post-synaptic activity x dopamine), with dopamine as reward prediction error and counterfactual (full-information) updates.
- **A league of flies.** 20 flies per round; the top 10 are kept, the rest culled, children bred from winners and weights saved between rounds. Each round trains on the past, leaves a 21-day gap, and validates on the next window. An untouched final holdout (last 500 dates) is only opened once, with `--final`.
- **Point-in-time data.** S&P 500 membership including removed constituents (Wikipedia), SEC EDGAR XBRL fundamentals aligned by *filing date* with a one-day lag, and 11 fundamental features. Known gap: delisted names without prices are still missing (survivorship bias).
- **Honest evaluation kit (`evalkit/`).** Sign-flip tests on monthly IC series (valid for correlated stocks), Holm, Benjamini-Hochberg and studentized max-T corrections, power from real IC dispersion, sector-neutralization, and decile long-short with trading costs. 33 signals compared against the fly, including random-feature controls.
- **Every round has controls** on the same window: ridge, 12-1 momentum, reversal, random, untrained fly, plus a "luck bar" (the best IC expected from pure noise given how many agents were tried).

## Results so far

| Check | Outcome |
|---|---|
| League, 15 rounds | Best validation IC near +0.01 to +0.02 after round 0; below the luck bar |
| 33-signal evaluation suite | No signal significant after correction |
| Fly vs random KC code vs ridge | Indistinguishable |
| Final holdout | Not opened |

Negative results are kept on purpose. See `reports/` for the generated evaluation reports.

## Project layout

```
connectome/   neuPrint download and weight matrix
sim/          leaky integrate-and-fire simulator
learning/     fly agent, dopamine rule, ridge baseline
tournament/   league: train, validate, cull, breed, save weights
env/          prices, S&P 500 membership, SEC fundamentals, features
evalkit/      statistics and portfolio tests
experiments/  runnable scripts
reports/      generated evaluation reports
tests/        unit tests
```

## Quickstart (Windows Command Prompt)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
echo NEUPRINT_TOKEN=your_token_here> .env
python -m connectome.fetch_mb
python -m experiments.build_dataset_fund --user-agent "Your Name you@example.com"
python -m experiments.run_league --workers 4
python -m experiments.eval_suite --panel data_fundamentals/panel.parquet
```

- Get a free token at neuprint.janelia.org. `.env` is gitignored: never commit it.
- SEC EDGAR requires a real User-Agent with your name and contact email and at most 8 requests per second.
- A league round takes about 10 minutes with 4 workers on a 12-thread machine.

## Roadmap

- Survivorship-free universe (prices for removed S&P names).
- 3D and 2D live dashboard that replays saved out-of-sample scores (in progress; never places trades).
- Paper trading only if an edge survives rigorous testing.

## Data sources

neuPrint / Janelia hemibrain, Yahoo Finance (prices), SEC EDGAR (fundamentals), Wikipedia (index membership). Please respect each source's terms and rate limits.

## Disclaimer

For research and education only. Not investment advice. Backtests are not predictions. Past performance, simulated or not, does not guarantee future results.
