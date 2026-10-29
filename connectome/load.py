"""Connectome loading and subgraph extraction.

Expected edge table columns: pre_id, post_id, syn_count, nt_type
(nt_type in {"ACH", "GABA", "GLUT", "DA", "OCT", "SER", ...}).

TODO: fill in real downloads.
  - Hemibrain: use neuprint-python (needs a free token in NEUPRINT_TOKEN).
  - FlyWire: download the connections table from the FlyWire Codex / Zenodo release.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

# Rough sign per neurotransmitter (excitatory +1, inhibitory -1).
NT_SIGN = {"ACH": 1.0, "GLUT": -1.0, "GABA": -1.0, "DA": 1.0, "OCT": 1.0, "SER": 1.0}


def load_edges(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    missing = {"pre_id", "post_id", "syn_count", "nt_type"} - set(df.columns)
    if missing:
        raise ValueError(f"edge table missing columns: {missing}")
    return df


def extract_subgraph(edges: pd.DataFrame, neuron_ids) -> pd.DataFrame:
    ids = set(neuron_ids)
    return edges[edges.pre_id.isin(ids) & edges.post_id.isin(ids)].reset_index(drop=True)


def to_weight_matrix(edges: pd.DataFrame, scale: float = 0.01):
    """Return (W, id_index) where W[post, pre] is a signed sparse weight matrix."""
    ids = np.unique(np.concatenate([edges.pre_id.values, edges.post_id.values]))
    index = {nid: i for i, nid in enumerate(ids)}
    pre = edges.pre_id.map(index).values
    post = edges.post_id.map(index).values
    sign = edges.nt_type.map(NT_SIGN).fillna(1.0).values
    w = sign * edges.syn_count.values * scale
    W = sparse.csr_matrix((w, (post, pre)), shape=(len(ids), len(ids)))
    return W, index
