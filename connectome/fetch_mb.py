"""Download the mushroom body subgraph (KCs, MBONs, DANs) from the hemibrain via neuPrint.

Setup:
  1. Sign in at https://neuprint.janelia.org and copy your auth token (account page).
  2. Create a file named .env in the repo root containing:   NEUPRINT_TOKEN=your_token_here
     (.env is gitignored, never commit it.)
  3. python -m connectome.fetch_mb

Outputs (gitignored, in data/):
  data/mb_neurons.csv   bodyId, type, cell_class
  data/mb_edges.csv     pre_id, post_id, syn_count, nt_type   (format expected by connectome/load.py)
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from neuprint import Client, NeuronCriteria as NC, fetch_adjacencies, fetch_neurons, set_default_client

DATASET = "hemibrain:v1.2.1"
SERVER = "neuprint.janelia.org"
MIN_WEIGHT = 3  # drop very weak connections

# Cell classes. Neurotransmitter is a coarse default per class (KC cholinergic, DAN dopaminergic).
# MBONs really are a mix (ACh / GABA / glutamate); refine per type later.
CLASSES = {
    "KC": {"type": "KC.*", "nt": "ACH"},
    "MBON": {"type": "MBON.*", "nt": "ACH"},  # TODO: per-type GABA/GLUT
    "DAN": {"type": "(PAM|PPL1).*", "nt": "DA"},
}


def main(out_dir: str = "data"):
    load_dotenv()
    token = os.environ.get("NEUPRINT_TOKEN")
    if not token:
        raise SystemExit("NEUPRINT_TOKEN not set. Put it in a .env file in the repo root.")
    client = Client(SERVER, dataset=DATASET, token=token)
    set_default_client(client)

    frames = []
    for cls, spec in CLASSES.items():
        df, _ = fetch_neurons(NC(type=spec["type"], regex=True), client=client)
        df = df[["bodyId", "type"]].copy()
        df["cell_class"] = cls
        df["nt_type"] = spec["nt"]
        frames.append(df)
        print(f"{cls}: {len(df)} neurons")
    neurons = pd.concat(frames, ignore_index=True).drop_duplicates("bodyId")

    ids = neurons.bodyId.tolist()
    _, conn = fetch_adjacencies(ids, ids, min_total_weight=MIN_WEIGHT, client=client)
    edges = conn.groupby(["bodyId_pre", "bodyId_post"], as_index=False).weight.sum()
    edges = edges.rename(columns={"bodyId_pre": "pre_id", "bodyId_post": "post_id", "weight": "syn_count"})
    edges["nt_type"] = edges.pre_id.map(neurons.set_index("bodyId").nt_type)
    print(f"{len(edges)} edges")

    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    neurons.to_csv(out / "mb_neurons.csv", index=False)
    edges.to_csv(out / "mb_edges.csv", index=False)
    print(f"saved to {out}/")


if __name__ == "__main__":
    main()
