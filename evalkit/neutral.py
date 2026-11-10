"""Sector neutralization: remove the part of scores and returns that is just 'which sector', which makes monthly ICs swing
widely (stocks of a sector move together) and hides small real effects."""
from __future__ import annotations

import numpy as np


def demean_by_group(x, groups):
    """x minus the mean of x within each group (NaNs ignored). groups: array of labels, same length as x."""
    x = np.asarray(x, float)
    _, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    ok = np.isfinite(x)
    cnt = np.bincount(inv[ok], minlength=inv.max() + 1)
    sm = np.bincount(inv[ok], weights=x[ok], minlength=inv.max() + 1)
    mean = np.where(cnt > 0, sm / np.maximum(cnt, 1), 0.0)
    return x - mean[inv]


def sector_map(wiki_html_path):
    """{ticker: GICS sector} from a saved copy of Wikipedia's S&P 500 page (today's classification, so it is applied to the
    past as an approximation; sectors rarely change but GICS had a few re-classifications)."""
    from pathlib import Path
    from env import universe_us as uni
    cur, _ = uni.parse_wikipedia(Path(wiki_html_path).read_text(encoding="utf-8", errors="replace"))
    return dict(zip(cur["symbol"], cur["sector"]))
