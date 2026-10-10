"""Gary's belief model: what the enemy has, from what Gary has seen of it (the first piece of the
opponent model, docs/ARCHITECTURE.md §6.9).

Learned from pro games (train/belief.py): from a player's view of the enemy so far, the enemy's
real army. Inputs, the same in training and in play: game time; for each enemy unit and building
type, how many are in sight now and the most in sight at once over the last 30 s, 90 s, 3 min and
ever; seconds since an enemy army unit was in sight; the player's own supply and workers. Outputs:
the enemy's expected count of each army type, workers and town halls, and its army supply's median
and 80th percentile.

In play, BeliefMemory keeps that view, one look per second of game time (as the training data
samples it), and BeliefModel answers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

WINDOWS = (30, 90, 180)                 # "most in sight at once" over these many seconds
SINCE_CAP = 600                         # seconds since an enemy army unit was in sight, at most
# the enemy's types the inputs cover (by race), and the army types the outputs count, with supply
# x2 (BW counts half supply)
ENEMY_TYPES = {
    "Z": [37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 62, 103, 131, 132, 133, 135, 136, 137, 138, 139, 140,
          141, 142, 143, 144, 146, 149],
    "T": [0, 1, 2, 3, 5, 7, 8, 9, 11, 12, 30, 32, 34, 58, 106, 107, 109, 110, 111, 112, 113, 114, 115,
          116, 117, 120, 122, 123, 124, 125],
    "P": [61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 83, 84, 154, 155, 156, 157, 159, 160, 162, 163, 164,
          165, 166, 167, 169, 170, 171, 172],
}
ARMY_SUPPLY = {
    "Z": {37: 1, 38: 2, 39: 8, 43: 4, 44: 4, 45: 4, 46: 4, 47: 1, 62: 4, 103: 4},
    "T": {0: 2, 1: 2, 2: 4, 3: 4, 5: 4, 30: 4, 8: 4, 9: 4, 11: 4, 12: 12, 32: 2, 34: 2, 58: 4},
    "P": {65: 4, 66: 4, 61: 4, 67: 8, 68: 8, 63: 4, 70: 6, 71: 4, 72: 12, 83: 8, 60: 4},
}
WORKERS = {"Z": 41, "T": 7, "P": 64}
TOWN_HALLS = {"Z": (131, 132, 133), "T": (106,), "P": (154,)}


def _vector(t_s: float, now: np.ndarray, maxes: list[np.ndarray], ever: np.ndarray, since_s: float,
            supply_x2: float, workers: float) -> np.ndarray:
    return np.concatenate([[t_s / 600.0], np.log1p(now), *[np.log1p(m) for m in maxes], np.log1p(ever),
                           [min(since_s, SINCE_CAP) / 600.0, supply_x2 / 400.0, workers / 80.0]]).astype(np.float32)


def features_series(seen: np.ndarray, own_units: np.ndarray, eco: np.ndarray, frames: np.ndarray,
                    enemy: str, own: str) -> np.ndarray:
    """(T, F) inputs for every second of one player's game, from the macro training set's arrays:
    seen (T, 228) the enemy in sight each second, own_units (T, 228), eco (T, 4), frames (T,)."""
    types = ENEMY_TYPES[enemy]
    army = np.array([t in ARMY_SUPPLY[enemy] for t in types])
    s = seen[:, types].astype(np.float32)
    ever = np.maximum.accumulate(s, axis=0)
    out, last = [], -10 ** 6
    for t in range(len(s)):
        if s[t, army].sum() > 0:
            last = t
        maxes = [s[max(0, t - w + 1):t + 1].max(0) for w in WINDOWS]
        out.append(_vector(frames[t] / 24.0, s[t], maxes, ever[t], t - last, eco[t, 2], own_units[t, WORKERS[own]]))
    return np.stack(out)


class BeliefMemory:
    """The player's view of the enemy during play, one look per second of game time."""

    def __init__(self, enemy: str):
        self.enemy = enemy
        self.types = ENEMY_TYPES[enemy]
        self.army = np.array([t in ARMY_SUPPLY[enemy] for t in self.types])
        self.by_second: dict[int, np.ndarray] = {}     # second -> enemy units in sight (its first look)
        self.ever = np.zeros(len(self.types), np.float32)
        self.army_seen_s = -10 ** 6

    def look(self, frame: int, seen228: np.ndarray) -> None:
        """What is in sight at this frame (enemy unit counts by type id)."""
        sec = frame // 24
        if sec in self.by_second:
            return
        s = seen228[self.types].astype(np.float32)
        self.by_second[sec] = s
        self.ever = np.maximum(self.ever, s)
        if s[self.army].sum() > 0:
            self.army_seen_s = sec
        for old in [k for k in self.by_second if k <= sec - max(WINDOWS)]:
            del self.by_second[old]

    def vector(self, frame: int, supply_x2: float, workers: float) -> np.ndarray:
        sec = frame // 24
        last = self.by_second[max(self.by_second)] if self.by_second else np.zeros(len(self.types), np.float32)
        maxes = []
        for w in WINDOWS:
            inside = [v for k, v in self.by_second.items() if k > sec - w]
            maxes.append(np.max(inside, axis=0) if inside else np.zeros(len(self.types), np.float32))
        return _vector(sec, last, maxes, self.ever, sec - self.army_seen_s, supply_x2, workers)


class BeliefNet(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


class BeliefModel:
    def __init__(self, ckpt: dict):
        self.ckpt = ckpt
        self.enemy, self.race = ckpt["enemy"], ckpt["race"]
        self.army_types = ckpt["army_types"]
        self.quantiles = ckpt["quantiles"]
        self.net = BeliefNet(ckpt["n_in"], ckpt["n_out"])
        self.net.load_state_dict(ckpt["state"])
        self.net.eval()
        self.path = None

    @classmethod
    def load(cls, path: str | Path) -> "BeliefModel":
        m = cls(torch.load(path, map_location="cpu", weights_only=False))
        m.path = str(path)
        return m

    def memory(self) -> BeliefMemory:
        return BeliefMemory(self.enemy)

    def predict(self, x: np.ndarray) -> dict:
        """The enemy's expected army supply x2, its quantiles, and the expected count of each army type."""
        with torch.no_grad():
            out = self.net(torch.tensor(x[None]))[0].numpy()
        k = len(self.army_types) + 2
        counts = np.exp(out[:k])
        sup = np.array([ARMY_SUPPLY[self.enemy][t] for t in self.army_types], np.float32)
        q = out[k:] * self.ckpt["sup_scale"]
        return {"supply": float(counts[:len(sup)] @ sup), "counts": dict(zip(self.army_types, counts[:len(sup)].tolist())),
                **{f"q{int(qq * 100)}": float(v) for qq, v in zip(self.quantiles, q)}}
