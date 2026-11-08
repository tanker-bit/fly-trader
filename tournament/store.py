"""Files on disk: config, state, and one folder per agent (state.json + weights.npy)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


class Store:
    def __init__(self, root):
        self.root = Path(root)
        (self.root / "agents").mkdir(parents=True, exist_ok=True)
        (self.root / "rounds").mkdir(parents=True, exist_ok=True)

    def _write(self, path, obj):
        tmp = str(path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=1)
        os.replace(tmp, path)

    def _read(self, path):
        with open(path) as f:
            return json.load(f)

    def has_config(self):
        return (self.root / "config.json").exists()

    def load_config(self):
        return self._read(self.root / "config.json")

    def save_config(self, cfg):
        self._write(self.root / "config.json", cfg)

    def load_state(self):
        return self._read(self.root / "state.json")

    def save_state(self, st):
        self._write(self.root / "state.json", st)

    def agent_ids(self):
        return sorted(p.name for p in (self.root / "agents").iterdir() if (p / "state.json").exists())

    def load_agent(self, aid):
        return self._read(self.root / "agents" / aid / "state.json")

    def save_agent(self, rec):
        d = self.root / "agents" / rec["id"]
        d.mkdir(parents=True, exist_ok=True)
        self._write(d / "state.json", rec)

    def load_weights(self, aid):
        p = self.root / "agents" / aid / "weights.npy"
        return np.load(p) if p.exists() else None

    def save_weights(self, aid, W):
        d = self.root / "agents" / aid
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "weights.npy", np.asarray(W, dtype=np.float32))

    def drop_weights(self, aid):
        p = self.root / "agents" / aid / "weights.npy"
        if p.exists():
            p.unlink()

    def alive(self):
        return [r for r in (self.load_agent(i) for i in self.agent_ids()) if r["status"] == "alive"]
