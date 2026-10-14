"""The learned early-defense switch (docs/CONTROL.md step 7): go into the rush defense now or not.

A small regression tree over the posture's facts (gary/policy/plan.py facts) predicting how much a
game gains from switching now instead of following the rules (train/early_switch.py, from paired
games: the same game played as the rules play it and with the switch forced at a given moment).
The posture asks it every few seconds while it isn't in the rush defense (gary/agent/posture.py
switch); it switches where the predicted gain is above MARGIN. Kept as plain JSON: readable, and
read without scikit-learn.
"""

from __future__ import annotations

import json
from pathlib import Path

MARGIN = 0.02                    # predicted gain (in game outcome, 0 lost to 1 won) needed to switch
SCOUT = ["drones_most", "extractor_seen", "pool_seen_at", "lings_seen_at"]   # (minutes; 0: not yet)


def facts(posture, obs: dict) -> dict:
    """What the switch decides on: the plan policy's facts (gary/policy/plan.py) and the scout's:
    the most drones seen at once, an extractor seen, when the pool and the first zerglings were
    first seen (the posture remembers them)."""
    from gary.policy.plan import facts as plan_facts
    fx = plan_facts(posture, obs)
    at = lambda f: f / 1440 if f is not None else 0.0
    fx.update(drones_most=getattr(posture, "drones_most", 0), extractor_seen=float(getattr(posture, "extractor_seen", False)),
              pool_seen_at=at(getattr(posture, "pool_seen_at", None)), lings_seen_at=at(getattr(posture, "lings_seen_at", None)))
    return fx


def names() -> list[str]:
    from gary.policy.plan import FACTS
    return FACTS + SCOUT


class Switch:
    def __init__(self, tree: dict, margin: float = MARGIN):
        self.tree, self.margin = tree, margin
        self.path = None

    @classmethod
    def load(cls, path: str | Path) -> "Switch":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        s = cls(d["tree"], d.get("margin", MARGIN))
        s.path = str(path)
        return s

    def gain(self, fx: dict) -> float:
        t, i = self.tree, 0
        while t["left"][i] >= 0:
            i = t["left"][i] if fx[t["feature"][i]] <= t["threshold"][i] else t["right"][i]
        return t["value"][i]

    def __call__(self, fx: dict) -> bool:
        return self.gain(fx) > self.margin

    @staticmethod
    def describe(tree: dict) -> list[str]:
        """The tree as readable rules, one line per leaf."""
        out = []

        def walk(i, conds):
            if tree["left"][i] < 0:
                out.append(f"{' and '.join(conds) or 'always'}: gain {tree['value'][i]:+.3f} ({tree['n'][i]} moments)")
                return
            f, x = tree["feature"][i], tree["threshold"][i]
            walk(tree["left"][i], conds + [f"{f} <= {x:g}"])
            walk(tree["right"][i], conds + [f"{f} > {x:g}"])
        walk(0, [])
        return out
