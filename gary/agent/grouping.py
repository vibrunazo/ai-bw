"""Grouping, a provider slot (docs/CONTROL.md): which of Gary's army units belong together, and on which control-group
key. The motor carries a plan out (assigns, adds) and uses it: an order whose units are exactly a key's members is
one key and one click, an order to others is selected unit by unit as before. Asked when army units arrive or the
clusters merge or split for good, not every turn.

Providers:
  clusters  scripted (the default): the army's clusters (up to 3, LearnedArmy's clustering) on keys 1-3, a cluster
            keeping the key holding most of its units; an arrival joins its cluster's key once it is there; a key is
            re-formed when its cluster has been merged or split for FOR_GOOD
  learned   (planned) imitated from the pros' own hotkey commands (eval/pro_groups.py: what they group, how often
            they re-form and recall)

Pros (100 modern games a race, after 5:00): the army on keys 1-3 (Terran also 9, 0, 4), assigned 6-10 times a
minute, recalled 41-71 times; 71% of their fight selections are recalls, Gary-T's 19%.
"""
from __future__ import annotations

import math

from gary.agent.gary import Gary

ARMY_KEYS = [1, 2, 3]
FOR_GOOD = 10 * 24                    # a cluster merged or split this long: its key re-formed
ARRIVE_NEAR = 6 * 32                  # an arrival this near its cluster's middle joins its key
GROUP_MIN = 4                         # a cluster of at least this many units gets a key
ASK_EVERY = 2 * 24


def _mid(units: list[dict]) -> tuple[float, float]:
    return sum(u["x"] for u in units) / len(units), sum(u["y"] for u in units) / len(units)


class ClusterGrouping:
    """The scripted grouping: keys follow the army's clusters, stably (see the module's docstring)."""

    name = "clusters"

    def __init__(self, g: Gary):
        self.g = g
        self.plan: dict[int, set[int]] = {}           # key -> the units that belong on it
        self.off_since: dict[int, int] = {}           # key -> since when its members are not one cluster
        self.next_ask = 0

    def step(self, clusters: list[list[dict]]) -> dict[int, set[int]] | None:
        """The plan, when it changed (None: as it was)."""
        f = self.g.hi.frame
        if f < self.next_ask:
            return None
        self.next_ask = f + ASK_EVERY
        clusters = [c for c in clusters if len(c) >= GROUP_MIN][:len(ARMY_KEYS)]
        alive = {u["tag"] for c in clusters for u in c}
        changed = False
        for key in list(self.plan):
            kept = self.plan[key] & alive
            if len(kept) < GROUP_MIN:
                del self.plan[key]
                changed = True
            elif kept != self.plan[key]:
                self.plan[key] = kept
        taken: set[int] = set()
        for c in clusters:
            tags = {u["tag"] for u in c}
            key = max((k for k in self.plan if k not in taken), key=lambda k: len(self.plan[k] & tags), default=None)
            if key is None or not self.plan[key] & tags:
                free = [k for k in ARMY_KEYS if k not in self.plan and k not in taken]
                if not free:
                    continue
                key = free[0]
                self.plan[key] = tags
                taken.add(key)
                changed = True
                continue
            taken.add(key)
            members = self.plan[key]
            if len(members & tags) < 0.7 * len(members) or len(members & tags) < 0.5 * len(tags):
                # (its members spread over other clusters, or this cluster mostly others: merged or split)
                since = self.off_since.setdefault(key, f)
                if f - since >= FOR_GOOD:
                    self.plan[key] = tags
                    self.off_since.pop(key, None)
                    changed = True
                continue
            self.off_since.pop(key, None)
            mx, my = _mid([u for u in c if u["tag"] in members] or c)
            arrived = {u["tag"] for u in c if u["tag"] not in members and math.dist((u["x"], u["y"]), (mx, my)) < ARRIVE_NEAR}
            if arrived:
                self.plan[key] = members | arrived
                changed = True
        return dict(self.plan) if changed else None


PROVIDERS = {"clusters": ClusterGrouping}
