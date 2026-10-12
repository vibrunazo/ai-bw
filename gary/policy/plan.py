"""Gary's plan policy: the posture's mode (layer L1, docs/CONTROL.md step 7), as a small learned policy.

The modes are the ones v0.9-v0.18's posture rules (gary/agent/posture.py) move between:

  normal   the usual plan: no constraints
  rush     an early pool: the one-base defense build, no expansion, the army home, bunkers manned
  hold     the rush build done: the learned macro again, but no expansion, the army home
  flood    a zergling mass on its way: the zergling-flood build, no expansion, the army home

Inputs: the symbolic facts those rules decide on (what Gary has seen of the enemy, its own army and
bunkers, how long its bases have been calm, the time), the mode it is in, and the mode the rules
would choose now. Output: a probability per mode, every second: a strong vote for the rules' mode
(PRIOR) plus a learned correction that starts at zero. So it starts as the rules exactly (a copy
learned by imitation got 2 of 7 held-out mode changes right: the rules change mode a few times a
game, too few to learn their thresholds from), and training learns only where to leave them.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

MODES = ["normal", "rush", "hold", "flood"]
FACTS = ["minutes", "hatcheries_seen", "base_seen", "pool_seen", "early_pool_seen", "early_pool_sure", "scouted",
         "lings_now", "lings_most", "enemy_home", "enemy_most", "army", "marines", "bunker", "calm_s", "bases",
         "unknown_window", "pool_window", "flood_window"]
LING, POOL = 37, 142
CALM_CAP = 60.0


def facts(posture, obs: dict) -> dict:
    """The facts the plan decides on, from the posture's memory and what Gary sees now."""
    from gary import terran as T
    from gary.agent.fight import enemy_fighters, near_home
    from gary.agent.posture import FLOOD_FROM, FLOOD_UNTIL, POOL_BY
    from gary.policy.army import supply_x2
    g = posture.g
    f = obs["frame"]
    mine = [u for u in obs["units"] if u["owner"] == g.slot]
    tags = {u["tag"] for u in mine}
    inside = set().union(set(), *(m for b, m in g.loaded.items() if b in tags)) - tags
    fighters = enemy_fighters(obs, g.slot)
    lings_now = sum(1 for u in obs["units"] if u["owner"] not in (g.slot, 11) and u["type"] == LING)
    marks = getattr(posture, "ling_marks", None) or [(f, lings_now)]
    return {"minutes": f / 24 / 60,
            "hatcheries_seen": len(posture.hatcheries_seen),
            "base_seen": float(posture.base_seen), "pool_seen": float(posture.pool_seen),
            "early_pool_seen": float(posture.early_pool_seen), "early_pool_sure": float(posture.early_pool_sure),
            "scouted": float(getattr(g.army, "scout_done", True)),
            "lings_now": lings_now, "lings_most": max(n for _, n in marks),
            "enemy_home": sum(supply_x2(u["type"]) for u in near_home(mine, fighters)) / 2,
            "enemy_most": posture.most_enemy,
            "army": sum(supply_x2(u["type"]) for u in mine if u["type"] in g.army_types) / 2 + len(inside),
            "marines": sum(1 for u in mine if u["type"] == T.MARINE) + len(inside),
            "bunker": float(any(u["type"] == T.BUNKER and u["completed"] for u in mine)),
            "calm_s": min(CALM_CAP, (f - g.calm_since) / 24),
            "bases": sum(1 for u in mine if u["type"] == T.CC),
            "unknown_window": float(posture.unknown_by <= f <= POOL_BY),
            "pool_window": float(f <= POOL_BY),
            "flood_window": float(FLOOD_FROM <= f <= FLOOD_UNTIL)}


PRIOR = 6.0                          # the rules' mode's logit (the others 0): with no correction, p = 0.99


def vector(fx: dict, mode: str, rules: str) -> np.ndarray:
    """The facts, the mode Gary is in, the mode the rules would choose now (one-hot each)."""
    v = [fx[k] for k in FACTS] + [float(mode == m) for m in MODES] + [float(rules == m) for m in MODES]
    return np.array(v, np.float32)


class PlanNet(nn.Module):
    """logits = PRIOR x the rules' mode + a correction (an MLP whose output starts at zero)."""

    def __init__(self, n_in: int = len(FACTS) + 2 * len(MODES), hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, len(MODES)))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x_norm, rules_onehot):
        return PRIOR * rules_onehot + self.net(x_norm)


class PlanModel:
    """A trained plan policy: the mode's probabilities from the facts and the current mode."""

    def __init__(self, ckpt: dict):
        self.ckpt = ckpt
        self.mean = np.array(ckpt["mean"], np.float32)
        self.std = np.array(ckpt["std"], np.float32)
        self.net = PlanNet(len(self.mean))
        self.net.load_state_dict(ckpt["state"])
        self.net.eval()
        self.path = None

    @classmethod
    def from_rules(cls, mean: np.ndarray, std: np.ndarray) -> "PlanModel":
        """The policy as it starts: the rules' mode, and a correction of zero."""
        net = PlanNet(len(mean))
        return cls({"state": net.state_dict(), "mean": np.asarray(mean).tolist(), "std": np.asarray(std).tolist()})

    @classmethod
    def load(cls, path: str | Path) -> "PlanModel":
        m = cls(torch.load(path, map_location="cpu", weights_only=False))
        m.path = str(path)
        return m

    def logits(self, x: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(np.asarray(x, np.float32))
        return self.net((x - torch.as_tensor(self.mean)) / torch.as_tensor(self.std), x[..., -len(MODES):])

    def probs(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch.softmax(self.logits(x), -1).numpy()


def decide(model: PlanModel, x: np.ndarray, rng=None, temperature: float = 0.0) -> tuple[str, float]:
    """The mode: the likeliest (temperature 0, play), or drawn at this temperature (training).
    Returns it with its probability."""
    p = model.probs(x)
    if temperature and rng is not None:
        q = np.power(np.maximum(p, 1e-9), 1.0 / temperature)
        q /= q.sum()
        i = int(rng.choice(len(MODES), p=q))
    else:
        i = int(p.argmax())
    return MODES[i], float(p[i])
