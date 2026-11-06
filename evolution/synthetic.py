"""Synthetic stock-day samples with a PLANTED pattern, for checking that evolution works.

Each sample is (features, next-day excess return). Features are 13 values in [-1, 1] (like the
cross-sectionally ranked features in the real TSX panel). The return is

    y = 0.01 * (snr * f(x) + noise)        f(x) has zero mean and unit std

so `snr` sets how much of the return is learnable. Patterns, easiest to hardest for a linear model:

    none      f = 0                       pure noise: nothing to find (the overfitting detector)
    linear    f = x0                      one feature predicts the return
    sparse2   f = x0 - x1                 two features, opposite signs
    xor       f = sign(x0) * sign(x1)     interaction: zero correlation with each feature alone,
                                          so a linear model (ridge) cannot capture it
"""
from __future__ import annotations

import numpy as np

N_FEATURES = 13
PATTERNS = ("none", "linear", "sparse2", "xor")


def planted(pattern: str, X: np.ndarray) -> np.ndarray:
    """The true signal f(x), normalised to unit std (uniform [-1,1] has std 1/sqrt(3))."""
    if pattern == "none":
        return np.zeros(len(X))
    if pattern == "linear":
        return X[:, 0] * np.sqrt(3.0)
    if pattern == "sparse2":
        return (X[:, 0] - X[:, 1]) * np.sqrt(3.0 / 2.0)
    if pattern == "xor":
        return np.sign(X[:, 0]) * np.sign(X[:, 1])
    raise ValueError(f"unknown pattern {pattern!r}; choose from {PATTERNS}")


def make_block(pattern: str, n: int, seed: int, snr: float = 0.3):
    """Return (X, y, f): features, next-day return, and the noise-free planted signal."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, N_FEATURES))
    f = planted(pattern, X)
    y = 0.01 * (snr * f + rng.normal(size=n))
    return X, y, f


def make_splits(pattern: str, seed: int = 0, n_train: int = 800, n_val: int = 400, n_val_blocks: int = 2,
                n_hold: int = 2000, snr: float = 0.3) -> dict:
    """Train block, several validation blocks (used for selection), and a holdout never used in evolution."""
    base = 10_000 * (seed + 1)
    return {
        "train": make_block(pattern, n_train, base + 1, snr),
        "val": [make_block(pattern, n_val, base + 10 + i, snr) for i in range(n_val_blocks)],
        "holdout": make_block(pattern, n_hold, base + 99, snr),
        "pattern": pattern, "snr": snr,
    }
