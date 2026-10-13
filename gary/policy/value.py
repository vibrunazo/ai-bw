"""Gary's value network (ARCHITECTURE.md P3; docs/CONTROL.md step 7): how good a moment is.

Trained by train/value.py on pro TvZ games and Gary's own; judged as a critic by
eval/value_branches.py. Its view of a moment, the same in replays and in play:

  the macro model's   time, money, supply, own units, the enemy in sight now and the most seen, the
  features            decisions so far (gary/policy/macro.py encode; the build style left out:
                      the critic judges the state, not what Gary meant to do)
  extras              the belief model's estimate of the enemy's army, hidden or not (its expected
                      army supply and 50th/80th percentiles, drones, town halls: gary/policy/belief.py);
                      own army supply now and 30 s ago, workers gained over 30 s, the enemy army in
                      sight now and the most at once over 30 s (a fight going on, and how it goes)

Heads: win (will this player win), net60 and net180 (the score's net value, killed minus lost,
over the next 60 and 180 s, /1000), disc (the score's net value ahead discounted by DISC_HALF_S,
/1000, learned by bootstrapping: the next few seconds' score plus its own value after them).
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gary.policy.belief import ARMY_SUPPLY, WORKERS

EXTRAS = ("belief army", "belief q50", "belief q80", "belief drones", "belief halls", "own army",
          "own army 30 s ago", "workers gained 30 s", "enemy army in sight", "enemy army in sight 30 s")
LOOKBACK_S = 30
DISC_HALF_S = 60.0
GAMMA_S = 0.5 ** (1 / DISC_HALF_S)     # the discount per second
_T_ARMY = np.array(sorted(ARMY_SUPPLY["T"]))
_T_SUP = np.array([ARMY_SUPPLY["T"][t] for t in _T_ARMY], np.float32)
_Z_ARMY = np.array(sorted(ARMY_SUPPLY["Z"]))
_Z_SUP = np.array([ARMY_SUPPLY["Z"][t] for t in _Z_ARMY], np.float32)


def army_supply(counts228: np.ndarray, race: str) -> np.ndarray:
    """Army supply x2 of unit counts by type id, (..., 228) -> (...)."""
    types, sup = (_T_ARMY, _T_SUP) if race == "T" else (_Z_ARMY, _Z_SUP)
    return counts228[..., types] @ sup


def belief_outputs(belief, x: np.ndarray) -> np.ndarray:
    """(n, 5) from the belief model's inputs (n, F): enemy army supply x2 expected, its 50th and 80th
    percentiles, drones and town halls expected."""
    with torch.no_grad():
        out = belief.net(torch.tensor(x, dtype=torch.float32)).numpy()
    k = len(belief.army_types) + 2
    counts = np.exp(np.minimum(out[:, :k], 10))
    sup = np.array([ARMY_SUPPLY[belief.enemy][t] for t in belief.army_types], np.float32)
    q = np.maximum(out[:, k:] * belief.ckpt["sup_scale"], 0)
    return np.stack([counts[:, :len(sup)] @ sup, q[:, 0], q[:, 1], counts[:, k - 2], counts[:, k - 1]], 1)


def extras(belief5: np.ndarray, own_army: np.ndarray, own_army_ago: np.ndarray, workers_gained: np.ndarray,
           seen_army: np.ndarray, seen_army_max: np.ndarray) -> np.ndarray:
    """(n, len(EXTRAS)): the belief model's outputs and the trends (army supplies x2)."""
    return np.concatenate([np.log1p(belief5), np.log1p(np.stack([own_army, own_army_ago], 1)),
                           (workers_gained / 10.0)[:, None], np.log1p(np.stack([seen_army, seen_army_max], 1))],
                          1).astype(np.float32)


def no_style(spec, x: np.ndarray) -> np.ndarray:
    """The macro features with the style one-hot set to "none" (its last columns)."""
    x = x.copy()
    n = len(spec.styles) + 1
    x[:, -n:] = 0
    x[:, -1] = 1
    return x


class ValueNet(nn.Module):
    def __init__(self, n_in: int, hidden: int = 256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.win = nn.Linear(hidden, 1)
        self.net = nn.Linear(hidden, 3)          # net60, net180, disc (/1000)

    def forward(self, x):
        h = self.body(x)
        return self.win(h)[:, 0], self.net(h)


class ValueModel:
    def __init__(self, ckpt: dict):
        self.ckpt = ckpt
        self.net = ValueNet(ckpt["n_in"])
        self.net.load_state_dict(ckpt["state"])
        self.net.eval()
        self.mean = np.array(ckpt["mean"], np.float32)
        self.std = np.array(ckpt["std"], np.float32)
        self.path = None

    @classmethod
    def load(cls, path: str | Path) -> "ValueModel":
        m = cls(torch.load(path, map_location="cpu", weights_only=False))
        m.path = str(path)
        return m

    def predict(self, x: np.ndarray) -> dict:
        """x (n, F) -> win logit, net60, net180, disc (net value, not /1000), each (n,)."""
        with torch.no_grad():
            lw, ln = self.net(torch.tensor((np.atleast_2d(x) - self.mean) / self.std, dtype=torch.float32))
        ln = ln.numpy() * 1000
        return {"win": lw.numpy(), "net60": ln[:, 0], "net180": ln[:, 1], "disc": ln[:, 2]}


class ValueView:
    """The critic's view during play: looks once a second (the belief model's memory, the trends),
    and gives the features of the moment. Reads Gary's macro tracker; changes nothing in the game."""

    def __init__(self, g, belief):
        self.g, self.belief = g, belief
        self.mem = belief.memory()
        self.hist: dict[int, tuple[float, float, float]] = {}    # second -> own army, workers, enemy army in sight

    def look(self, obs: dict) -> None:
        sec = obs["frame"] // 24
        if sec in self.hist:
            return
        own = np.zeros(228, np.float32)
        seen = np.zeros(228, np.float32)
        for u in obs["units"]:
            if not 0 <= u["type"] < 228:
                continue
            if u["owner"] == self.g.slot:
                own[u["type"]] += 1 if u["completed"] else 0
            elif u["owner"] != 11:
                seen[u["type"]] += 1
        self.mem.look(obs["frame"], seen)
        self.hist[sec] = (float(army_supply(own, "T")), float(own[WORKERS["T"]]), float(army_supply(seen, "Z")))
        for old in [k for k in self.hist if k < sec - LOOKBACK_S]:
            del self.hist[old]

    def features(self, obs: dict) -> np.ndarray:
        """(1, F) for this moment (look() first)."""
        g = self.g
        sec = obs["frame"] // 24
        mine = [u for u in obs["units"] if u["owner"] == g.slot]
        tracker = copy.copy(g.macro.tracker)
        tracker.seen_max = g.macro.tracker.seen_max.copy()      # (its seen_max is kept by the macro)
        x = no_style(tracker.spec, tracker.features(obs, g.macro.in_production(mine)))
        now = self.hist[max(self.hist)]
        ago = self.hist[min(self.hist)]
        b = self.mem.vector(obs["frame"], obs["me"]["supply_used"] * 2, now[1])
        ex = extras(belief_outputs(self.belief, b[None]), np.array([now[0]]), np.array([ago[0]]),
                    np.array([now[1] - ago[1]]), np.array([now[2]]),
                    np.array([max(h[2] for k, h in self.hist.items() if k > sec - LOOKBACK_S)]))
        return np.concatenate([x, ex], 1)
