"""Feature groups: which slice of the information a fly can see (diverse leagues, v8).

In a v7 league every fly sees every feature, so the population collapsed into near-copies (mean score correlation 0.90 after round 4).
A fly whose genome has a `groups` gene sees only the features of those groups; its circuit is wired on the visible features only
(n_obs = how many it sees), so no KC listens to a hidden input. A genome without the gene sees everything, exactly as before.

  price : returns over 1-120 days, 12-1 momentum, distance from the 50/200-day averages and the 52-week high, RSI
  risk  : volatility, volatility ratio, dollar volume
  fund  : the 11 point-in-time fundamentals and fund_known
Features outside every group stay visible to everyone. A fly needs at least MIN_VISIBLE visible features (a KC samples 3 or more).
"""
from __future__ import annotations

import numpy as np

GROUPS = {
    "price": ("ret_1", "ret_5", "ret_20", "ret_60", "ret_120", "mom_12_1", "ma50_dist", "ma200_dist", "rsi_14", "dist_52w_high"),
    "risk": ("vol_20", "vol_ratio", "volume_ratio"),
    "fund": ("earnings_yield", "book_to_price", "fcf_yield", "roe", "roa", "gross_margin", "rev_growth_yoy", "eps_growth_yoy",
             "debt_to_equity", "accruals", "days_since_filing", "fund_known"),
}
ALLOWED_SETS = (("price",), ("fund",), ("price", "risk"), ("price", "fund"), ("risk", "fund"), ("price", "risk", "fund"))
MIN_VISIBLE = 6
_GROUPED = {f for names in GROUPS.values() for f in names}


def visible_columns(groups, feats) -> np.ndarray:
    """Indices (into feats) of the features a fly with these groups can see."""
    want = {f for g in groups for f in GROUPS[g]}
    return np.array([i for i, f in enumerate(feats) if f in want or f not in _GROUPED], dtype=int)


def usable_sets(feats) -> list:
    """The allowed group sets that leave at least MIN_VISIBLE features on this panel (a panel without fundamentals loses the fund sets)."""
    return [s for s in ALLOWED_SETS if len(visible_columns(s, feats)) >= MIN_VISIBLE]


def view_cols(genome: dict, feats):
    """Visible column indices for this genome, or None when it has no `groups` gene (sees everything, no slicing)."""
    g = genome.get("groups")
    return visible_columns(g, feats) if g else None


def n_view(genome: dict, feats) -> int:
    cols = view_cols(genome, feats)
    return len(feats) if cols is None else len(cols)


def niche(genome: dict) -> tuple:
    """The feature slice as a hashable key; a fly without a `groups` gene is the 'all' niche."""
    return tuple(genome.get("groups") or ("all",))


def neighbor_set(groups, rng, feats, avoid=()) -> list:
    """A usable slice that differs from `groups` by exactly one group added or removed (any other usable slice if none is that close).
    Slices in `avoid` (e.g. ones that produced only silent flies this round) are skipped while any other choice exists."""
    cur = set(groups)
    sets = usable_sets(feats)
    avoid = {tuple(x) for x in avoid}
    sets = [s for s in sets if tuple(s) not in avoid] or sets
    near = [s for s in sets if len(cur ^ set(s)) == 1]
    pool = near or [s for s in sets if set(s) != cur] or sets
    return list(pool[int(rng.integers(len(pool)))])


def random_groups(rng, feats) -> list:
    sets = usable_sets(feats)
    return list(sets[int(rng.integers(len(sets)))])
