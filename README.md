# fly-trader

Train a simulated fruit fly brain (built from the real connectome) to trade stocks using
dopamine-style reinforcement learning. Gains are rewarded with a dopamine signal, losses are punished.
Long-term goal: paper trade, then (carefully) live trade through a brokerage API.

> **Disclaimer:** research project. Not financial advice. Markets are mostly noise and most
> strategies, including this one, will likely not beat buy-and-hold. Never risk money you can't lose.

## Idea

The fly mushroom body is a reward-learning circuit:

- **Kenyon cells (KCs):** sparse code of sensory input
- **Dopaminergic neurons (DANs):** signal reward / punishment
- **Mushroom body output neurons (MBONs):** drive behavior (here: buy / sell / hold)

The connectome fixes the wiring. Only the KC -> MBON synapses learn, via a three-factor rule:

```
dW = lr * dopamine * eligibility(pre, post)
```

Market features are encoded as input currents into KCs. MBON firing rates are decoded into a
position. Reward = risk-adjusted P&L after costs.

## Layout

```
connectome/   data download + subgraph extraction (hemibrain / FlyWire)
sim/          LIF neuron simulator
env/          Gymnasium trading environment + data loaders
learning/     dopamine-gated plasticity rules
baselines/    buy-and-hold (add PPO next)
broker/       paper/live adapters + risk limits (TODO)
experiments/  configs + notebooks
```

## Roadmap

- [x] Repo scaffold, trading env, buy-and-hold baseline
- [ ] Extract mushroom body subgraph from hemibrain (neuPrint) or FlyWire
- [ ] Wire connectome -> LIF sim -> trading env
- [ ] Dopamine-gated learning on KC->MBON synapses
- [ ] Walk-forward validation vs. buy-and-hold and PPO
- [ ] Paper trading (Alpaca / IBKR) with hard risk limits and kill switch
- [ ] Live trading (only if paper results justify it)

## Quickstart

```bash
pip install -r requirements.txt
python -m baselines.buy_and_hold
```

## Data sources

- FlyWire (~139k neurons): https://flywire.ai
- Hemibrain via neuPrint: https://neuprint.janelia.org
