"""Grouping, a provider slot (docs/CONTROL.md): which of Gary's army units belong together, and on which control-group
key. The motor carries a plan out (assigns, adds) and uses it: an order whose units are exactly a key's members is
one key and one click, an order to others is selected unit by unit as before. Asked when army units arrive or the
clusters merge or split for good, not every turn.

Providers:
  types     scripted (the default): the pros' habit, one unit type of one cluster on a key (LAYOUTS: Terran 1-4, Zerg 1-3),
            the biggest such groups by supply; a key re-assigned (not added to) when its group grew or moved on
  clusters  scripted: the army's clusters (up to 3, LearnedArmy's clustering) on keys 1-3, a cluster keeping the key
            holding most of its units; an arrival added to its cluster's key once it is there; a key is re-formed
            when its cluster has been merged or split for FOR_GOOD
  learned   (planned) imitated from the pros' own hotkey commands (ingest/hotkey_dataset.py: what they group, how
            often they re-form and recall)

Pros (ingest/hotkey_dataset.py --report, 200 modern TvZ games, after 5:00): army keys in use T 4, Z 3; assigns 6
(T) and 9.6 (Z) a minute, adds none, recalls 62 and 41; a group 5 (T) and 9 (Z) units at its assign, mostly one
type (Terran: marines 20%, vultures 15%, vessels 11%, medics 7%, tanks 6%, marines with medics 5%). Gary with
clusters (round GR40a): T 1 key, assigns 1.9, adds 3.8, recalls 0.1 a minute; Z 3 keys, recalls 6.8.
"""
from __future__ import annotations

import math
from pathlib import Path

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
    adds = True

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


# the keys with control groups on, by race: the army's, and the buildings' where they move (executor.py's default:
# town halls 4-5, production 6-9). The pros' (ingest/hotkey_dataset.py, assigns by what a key holds): Terran army on
# 1-5 (1-4 mostly), production on 6-7, other buildings on 8, 9 and 0; Zerg army on 1-3, hatcheries on 4-5. Terran:
# 4 army keys, its Command Centers on 5-6, production on 7-9 (one key fewer: a fourth building is clicked), the
# scout on 0 (motor.py). Zerg: the hatcheries keep 4-5, a raid 9 and lurkers 0 (army_rules.py, zerg_fight.py)
LAYOUTS = {"T": {"army": [1, 2, 3, 4], "hall": [5, 6], "production": [7, 8, 9]},
           "Z": {"army": [1, 2, 3]}, "P": {"army": [1, 2, 3]}}
TYPE_MIN_SUPPLY = 8                   # (supply x2) a type's group in a cluster: 4 marines or zerglings x2, 2 tanks
REFORM_AFTER = 10 * 24                # a key re-assigned at most this often
GROWN = 2                             # re-assigned when this many of its type joined its cluster


class TypeGrouping:
    """The pros' habit: a unit type of a cluster on a key, re-assigned when its group changed (module docstring)."""

    name = "types"
    adds = False                                      # (the motor re-assigns a key instead of adding to it)

    def __init__(self, g: Gary):
        self.g = g
        self.keys = LAYOUTS.get(g.race.letter, {}).get("army", ARMY_KEYS)
        self.plan: dict[int, set[int]] = {}           # key -> the units that belong on it
        self.kind: dict[int, int] = {}                # key -> its unit type
        self.formed: dict[int, int] = {}              # key -> when its plan was last set
        self.next_ask = 0

    def step(self, clusters: list[list[dict]]) -> dict[int, set[int]] | None:
        from gary.policy.army import supply_x2
        f = self.g.hi.frame
        if f < self.next_ask:
            return None
        self.next_ask = f + ASK_EVERY
        groups = self._groups(clusters)
        groups.sort(key=lambda x: -x[0])
        alive = {u["tag"] for c in clusters for u in c}
        changed = False
        for key in list(self.plan):
            self.plan[key] &= alive
            if not self.plan[key]:
                del self.plan[key], self.kind[key]
                changed = True
        taken: set[int] = set()
        left = []
        for _, t, tags in groups[:len(self.keys)]:    # the biggest groups keep (or get) keys
            key = max((k for k in self.plan if self.kind[k] == t and k not in taken),
                      key=lambda k: len(self.plan[k] & tags), default=None)
            if key is None or not self.plan[key] & tags:
                left.append((t, tags))
                continue
            taken.add(key)
            have = self.plan[key]
            moved = len(have & tags) < 0.7 * len(have)
            if (len(tags - have) >= GROWN or moved) and f - self.formed.get(key, -10 ** 9) >= REFORM_AFTER:
                self.plan[key], self.formed[key] = tags, f
                changed = True
        for t, tags in left:                          # new groups: a free key, else the key of the smallest group
            free = [k for k in self.keys if k not in self.plan]
            if not free:
                spare = [k for k in self.plan if k not in taken and f - self.formed.get(k, -10 ** 9) >= REFORM_AFTER]
                if not spare:
                    continue
                free = [min(spare, key=lambda k: len(self.plan[k]))]
            key = free[0]
            self.plan[key], self.kind[key], self.formed[key] = tags, t, f
            taken.add(key)
            changed = True
        return dict(self.plan) if changed else None

    def _groups(self, clusters: list[list[dict]]) -> list[tuple[int, int, set[int]]]:
        """(supply, type, tags): a type in a cluster, 12 at most."""
        from gary.policy.army import supply_x2
        groups = []
        for c in clusters:
            by: dict[int, list[dict]] = {}
            for u in c:
                by.setdefault(u["type"], []).append(u)
            for t, us in by.items():
                if sum(supply_x2(t) for _ in us) >= TYPE_MIN_SUPPLY:
                    mx, my = _mid(us)
                    us = sorted(us, key=lambda u: math.dist((u["x"], u["y"]), (mx, my)))[:12]
                    groups.append((sum(supply_x2(t) for _ in us), t, {u["tag"] for u in us}))
        return groups


LEARNED_JOIN = 0.4                    # the grouping model's partition: a unit joins a group at this mean chance
# (held-out pro fights, up to 4 keys: precision 0.529, recall 0.779 against the type rule's 0.530 and 0.706)


class LearnedGrouping(TypeGrouping):
    """The grouping model's groups (gary/policy/grouping_model.py: which units the pros keep on one key, from their
    keys in fights), on TypeGrouping's keys and stability; a group's kind (for keeping its key) is its commonest type."""

    name = "learned"

    def __init__(self, g: Gary):
        super().__init__(g)
        from gary.policy.grouping_model import GroupModel
        path = max(Path("runs/groups").glob("*/model.pt"), key=lambda q: q.parent.name)
        self.model = GroupModel.load(path)

    def _groups(self, clusters: list[list[dict]]) -> list[tuple[int, int, set[int]]]:
        import numpy as np
        from collections import Counter
        from gary.policy.army import supply_x2
        from gary.policy.grouping_model import partition
        g = self.g
        fx, fy = g.flip
        out = []
        for c in clusters:
            if len(c) < 2:
                continue
            c = c[:64]
            cx, cy = _mid(c)
            rows = np.zeros((len(c), 11), np.int16)
            for i, u in enumerate(c):
                rx, ry = u["x"] - cx, u["y"] - cy
                rows[i] = (u["type"], 1, -rx if fx else rx, -ry if fy else ry, min(u.get("hp") or 0, 32767),
                           u.get("shields") or 0, u.get("cooldown", 0), u.get("order", 0), -1, u.get("carrying", 0),
                           u.get("completed", 1))
            aff = self.model.affinity(rows, g.hi.frame * 42 / 1000)
            for grp in partition(aff, len(self.keys), join=LEARNED_JOIN):
                us = [c[i] for i in grp]
                sup = sum(supply_x2(u["type"]) for u in us)
                if sup >= TYPE_MIN_SUPPLY:
                    out.append((sup, Counter(u["type"] for u in us).most_common(1)[0][0], {u["tag"] for u in us}))
        return out


PROVIDERS = {"types": TypeGrouping, "learned": LearnedGrouping, "clusters": ClusterGrouping}
