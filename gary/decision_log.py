"""The learned layers' decisions in one game, for the league (eval/league.py) and the RL that trains on
them: each decision's inputs, what was drawn and its log-probability, saved when the game's process
ends. Off unless GARY_DECISION_LOG names the file (.npz).

Layers, by side (both sides of a headless game share the process): "army_z" (gary/bots/gary_z.py: where
a group goes, move or attack), "roles_z" (gary/bots/zerg_roles.py: each unit's role in a fight
snapshot), "army_t" (gary/agent/army.py, with draw), "roles_t" (gary/agent/roles.py ahead/v1: its
units' roles before contact). Arrays are stacked per layer: "<layer>/<field>".
"""

from __future__ import annotations

import atexit
import os
from collections import defaultdict

import numpy as np


class DecisionLog:
    def __init__(self):
        self.path = os.environ.get("GARY_DECISION_LOG")
        self.items: dict[str, list[dict]] = defaultdict(list)
        if self.path:
            atexit.register(self.save)

    @property
    def on(self) -> bool:
        return bool(self.path)

    def add(self, layer: str, **fields) -> None:
        if self.path:
            self.items[layer].append(fields)

    def save(self) -> None:
        out = {}
        for layer, items in self.items.items():
            for k in items[0]:
                out[f"{layer}/{k}"] = np.stack([np.asarray(it[k]) for it in items])
        if out:
            np.savez_compressed(self.path, **out)


LOG = DecisionLog()
