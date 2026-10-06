"""Roles (layer L4, docs/CONTROL.md): what each of Gary's fighting units does, every half second.

A role per unit, as eval/micro_roles.py labels the pros' marines: fire, close in, pull back,
retreat (stand: nothing to do). Decided by hand rules (provider `rules/v1`) or by the role model
learned from the pros' labels (`learned/v1`, gary/policy/roles.py); each role's skill picks where
to go from the threats around (their positions and weapon ranges, the walkable ground), and the
motor layer's command skill clicks it. One role group per turn (12 units at most), the most
urgent. The hand rules:

  retreat    the zerg near Gary's group outweigh it (supply, SCVs fighting with it counted) 2.5 to
             1: to safety. Into a bunker of Gary's with room, within 15 tiles (the nearest marines,
             as many as fit); else, away from home, away from them to the edge of their weapon
             range plus a margin. At home (a Command Center within 14 tiles) there is no walking
             away: the SCVs would be left to the zerg, so the army fights with them
  pull_back  units in contact with an attacker that targets them, under half HP, with other units
             near them firing: each steps 1.5 tiles away from its attackers (a walkable direction)
  fire       units with an enemy in range that aren't attacking: attack-move at the enemy group
  close_in   units with no enemy in range, when Gary's group isn't weaker there: attack-move toward
             the nearest enemy group

Units another source holds by lease (a rule loading them into a bunker...) are left out.
"""

from __future__ import annotations

import math

from gary import terran as T
from gary.agent.intents import Intent
from gary.policy.army import is_building, is_worker, supply_x2

ATTACK = {10, 12, 14}
MOVE = {6}
# weapon ranges (pixels): what a unit can hit from how far, roughly, edges included
RANGE = {0: 160, 32: 48, 1: 224, 2: 192, 5: 224, 30: 224, 3: 192,             # marine, firebat, ghost, vulture, tanks, goliath
         37: 32, 38: 160, 103: 224, 39: 48, 43: 128, 44: 0, 46: 0, 47: 32, 50: 32}   # ling, hydra, lurker, ultra, muta...
NEAR = 10 * 32                       # units this close to the group's middle count for its strength
RETREAT_RATIO = 2.5
CLOSE_RATIO = 1.0                    # close in only when not weaker than this
CONTACT = 48
PULL_HP = 0.5
PULL_TILES = 1.5
RETREAT_MARGIN = 3 * 32
REPEAT = 24                          # the same role isn't given to a unit again within a second (but pull back)
SHELTER = 15 * 32                    # a bunker this close is where a retreat goes
HOME = 14 * 32                       # a Command Center this close: the group is defending home
BUNKER, MARINE = 125, 0
BUNKER_SLOTS = 4
MAX_GROUP = 12


def _mid(units) -> tuple[float, float]:
    return sum(u["x"] for u in units) / len(units), sum(u["y"] for u in units) / len(units)


class RoleRules:
    """rules/v1: hand rules for roles (see the module's docstring)."""

    def __init__(self, g):
        self.g = g
        self.told: dict[int, tuple[str, int]] = {}   # unit -> (role, frame)
        self._walk = None

    def decide(self, obs: dict, snap) -> Intent | None:
        g = self.g
        by_tag = snap[3]
        mine = [u for u in by_tag.values() if u["owner"] == g.slot and u["completed"] and supply_x2(u["type"]) > 0
                and not is_worker(u["type"]) and not is_building(u["type"])]
        enemy = [u for u in by_tag.values() if u["owner"] not in (g.slot, 11) and supply_x2(u["type"]) > 0
                 and not is_worker(u["type"]) and not is_building(u["type"])]
        mine = [u for u in mine if u["tag"] in set(g.leases.free([u["tag"] for u in mine], "roles"))]
        if not mine or not enemy:
            return None
        now = g.hi.frame
        mx, my = _mid(mine)
        workers = [u for u in by_tag.values() if u["owner"] == g.slot and u["type"] == T.SCV
                   and (u.get("order") in ATTACK or u["tag"] in g.pulled)]
        ours = sum(supply_x2(u["type"]) for u in mine + workers if math.dist((u["x"], u["y"]), (mx, my)) < NEAR)
        home = any(u["owner"] == g.slot and u["type"] == T.CC and math.dist((u["x"], u["y"]), (mx, my)) < HOME
                   for u in by_tag.values())
        theirs = sum(supply_x2(u["type"]) for u in enemy if math.dist((u["x"], u["y"]), (mx, my)) < NEAR)
        fresh = lambda u, role: not (self.told.get(u["tag"], ("", -10**9))[0] == role
                                     and now - self.told[u["tag"]][1] < REPEAT)
        in_range = lambda u: any(math.dist((u["x"], u["y"]), (e["x"], e["y"])) <= RANGE.get(u["type"], 160) + 32
                                 for e in enemy)

        # retreat: the group is outweighed; into a bunker if one is near with room, else away
        if theirs > RETREAT_RATIO * max(ours, 1):
            group = [u for u in mine if fresh(u, "RETREAT")]
            shelter = self._shelter(by_tag, (mx, my))
            if shelter is not None and group:
                bunker, room = shelter
                inside = [u for u in group if u["type"] == MARINE]
                inside = sorted(inside, key=lambda u: math.dist((u["x"], u["y"]), (bunker["x"], bunker["y"])))[:room]
                if inside:
                    return self._enter(inside, bunker)
            group = group[:MAX_GROUP]
            if group and not home:
                return self._intent("RETREAT", group, self._retreat_point(group, enemy), "move")
        # pull back: units under attack, hurt, with others firing near them
        targeted = {e.get("order_target") for e in enemy}
        firing = [u for u in mine if u.get("order") in ATTACK]
        pull = [u for u in mine if u["tag"] in targeted and u.get("hp_max") and u["hp"] < PULL_HP * u["hp_max"]
                and any(math.dist((u["x"], u["y"]), (e["x"], e["y"])) < CONTACT for e in enemy)
                and any(f["tag"] != u["tag"] and math.dist((u["x"], u["y"]), (f["x"], f["y"])) < 4 * 32 for f in firing)]
        if pull:
            group = pull[:MAX_GROUP]
            attackers = [e for e in enemy if e.get("order_target") in {u["tag"] for u in group}] or enemy
            point = self._step_away(_mid(group), _mid(attackers), PULL_TILES * 32)
            if point:
                return self._intent("PULL_BACK", group, point, "move")
        # fire: an enemy in range, not attacking
        idle = [u for u in mine if u.get("order") not in ATTACK and in_range(u) and fresh(u, "FIRE")]
        if idle:
            group = sorted(idle, key=lambda u: min(math.dist((u["x"], u["y"]), (e["x"], e["y"])) for e in enemy))[:MAX_GROUP]
            target = min(enemy, key=lambda e: math.dist((e["x"], e["y"]), _mid(group)))
            return self._intent("FIRE", group, (int(target["x"]), int(target["y"])), "attack_move")
        # close in: nothing in range, not weaker
        if ours >= CLOSE_RATIO * theirs:
            far = [u for u in mine if not in_range(u) and u.get("order") not in ATTACK and fresh(u, "CLOSE_IN")]
            if far:
                group = far[:MAX_GROUP]
                target = min(enemy, key=lambda e: math.dist((e["x"], e["y"]), _mid(group)))
                return self._intent("CLOSE_IN", group, (int(target["x"]), int(target["y"])), "attack_move")
        return None

    def _intent(self, role: str, group: list[dict], point, order: str) -> Intent | None:
        if point is None:
            return None
        now = self.g.hi.frame
        for u in group:
            self.told[u["tag"]] = (role, now)
        tags = [u["tag"] for u in group]
        key = (order, int(point[0]), int(point[1]))
        return Intent(role, tags, (key[1], key[2]), {"key": key, "tags": tags, "center": _mid(group)}, "roles")

    def _act(self, roles: dict[str, list[dict]], mine, enemy, by_tag, home: bool, fresh) -> Intent | None:
        """The most urgent role group as an intent: retreat, pull back, fire, close in."""
        group = [u for u in roles.get("RETREAT", []) if fresh(u, "RETREAT")]
        if group:
            shelter = self._shelter(by_tag, _mid(group))
            if shelter is not None:
                bunker, room = shelter
                inside = sorted((u for u in group if u["type"] == MARINE),
                                key=lambda u: math.dist((u["x"], u["y"]), (bunker["x"], bunker["y"])))[:room]
                if inside:
                    return self._enter(inside, bunker)
            if not home:
                return self._intent("RETREAT", group[:MAX_GROUP], self._retreat_point(group[:MAX_GROUP], enemy), "move")
        group = roles.get("PULL_BACK", [])[:MAX_GROUP]
        if group:
            targeted = [e for e in enemy if e.get("order_target") in {u["tag"] for u in group}]
            near = targeted or sorted(enemy, key=lambda e: math.dist((e["x"], e["y"]), _mid(group)))[:3]
            point = self._step_away(_mid(group), _mid(near), PULL_TILES * 32)
            if point:
                return self._intent("PULL_BACK", group, point, "move")
        group = [u for u in roles.get("FIRE", []) if u.get("order") not in ATTACK and fresh(u, "FIRE")]
        if group:
            group = sorted(group, key=lambda u: min(math.dist((u["x"], u["y"]), (e["x"], e["y"])) for e in enemy))[:MAX_GROUP]
            target = min(enemy, key=lambda e: math.dist((e["x"], e["y"]), _mid(group)))
            return self._intent("FIRE", group, (int(target["x"]), int(target["y"])), "attack_move")
        group = [u for u in roles.get("CLOSE_IN", []) if u.get("order") not in ATTACK and fresh(u, "CLOSE_IN")][:MAX_GROUP]
        if group:
            target = min(enemy, key=lambda e: math.dist((e["x"], e["y"]), _mid(group)))
            return self._intent("CLOSE_IN", group, (int(target["x"]), int(target["y"])), "attack_move")
        return None

    def _shelter(self, by_tag: dict, at):
        """A finished bunker of Gary's within SHELTER with room: (bunker, free slots), or None."""
        g = self.g
        tags = set(by_tag)
        for b in sorted((u for u in by_tag.values() if u["owner"] == g.slot and u["type"] == BUNKER and u["completed"]
                         and math.dist((u["x"], u["y"]), at) < SHELTER),
                        key=lambda u: math.dist((u["x"], u["y"]), at)):
            sent = {t for t in g.loaded.get(b["tag"], set()) if t not in tags}   # inside (unseen) or gone
            room = BUNKER_SLOTS - len(sent)
            if room > 0:
                return b, room
        return None

    def _enter(self, group: list[dict], bunker: dict) -> Intent:
        now = self.g.hi.frame
        for u in group:
            self.told[u["tag"]] = ("RETREAT", now)
        tags = [u["tag"] for u in group]
        key = ("own_unit", bunker["tag"])
        return Intent("RETREAT", tags, bunker["tag"], {"key": key, "tags": tags, "center": _mid(group),
                                                       "bunker": bunker["tag"]}, "roles")

    # --- where to go -----------------------------------------------------------------------------

    def _walkable(self, x: float, y: float) -> bool:
        if self._walk is None:
            terrain = getattr(self.g.hi.game, "terrain", None)
            self._walk = terrain()[0] if terrain is not None else False
        if self._walk is False:
            return True
        h, w = self._walk.shape
        mx, my = int(x // 8), int(y // 8)
        return 0 <= mx < w and 0 <= my < h and bool(self._walk[my, mx])

    def _step_away(self, frm, threat, dist: float):
        """A walkable point dist away from the threat, straight back or turned up to 60 degrees."""
        dx, dy = frm[0] - threat[0], frm[1] - threat[1]
        n = math.hypot(dx, dy) or 1
        base = math.atan2(dy / n, dx / n)
        for turn in (0, 0.5, -0.5, 1.0, -1.0):
            a = base + turn
            x, y = frm[0] + math.cos(a) * dist, frm[1] + math.sin(a) * dist
            if self._walkable(x, y):
                return x, y
        return None

    def _retreat_point(self, group: list[dict], enemy: list[dict]):
        """Away from the enemy group, to the edge of their reach plus a margin."""
        mid, them = _mid(group), _mid(enemy)
        reach = max(RANGE.get(e["type"], 32) for e in enemy)
        nearest = min(math.dist(mid, (e["x"], e["y"])) for e in enemy)
        dist = max(4 * 32, reach + RETREAT_MARGIN - nearest + 2 * 32)
        return self._step_away(mid, them, dist)


class RoleLearned(RoleRules):
    """learned/v1: the role model (gary/policy/roles.py) names each unit's role (its likeliest);
    the same skills as the hand rules carry the groups out. Reposition, stand and other: no command;
    nor "keep", for a model of decisions (train/roles.py --decisions)."""

    def __init__(self, g, model):
        super().__init__(g)
        self.model = model
        self.prev: dict[int, tuple[float, float]] = {}
        self.prev_frame = None

    def decide(self, obs: dict, snap) -> Intent | None:
        from gary.policy.roles import MAX_UNITS, extra_features
        names = self.model.config.get("roles")
        g = self.g
        now = g.hi.frame
        positions = {u["tag"]: (u["x"], u["y"]) for u in obs["units"]}
        prev, elapsed = self.prev, (now - self.prev_frame) if self.prev_frame is not None else 0
        self.prev, self.prev_frame = positions, now
        if snap is None:
            return None
        rows, tags, center, by_tag = snap
        if 0 < elapsed != 12:                  # motion per half second, as the model learned it
            prev = {t: (positions[t][0] - (positions[t][0] - p[0]) * 12 / elapsed,
                        positions[t][1] - (positions[t][1] - p[1]) * 12 / elapsed)
                    for t, p in prev.items() if t in positions}
        extra = extra_features(rows, tags, positions, prev if elapsed else {}, g.flip)
        probs = self.model.predict(rows, extra, obs["frame"] * 42 / 1000)
        free = set(g.leases.free(tags, "roles"))
        mine = [by_tag[t] for i, t in enumerate(tags[:MAX_UNITS]) if rows[i][1] == 1 and t in free
                and supply_x2(by_tag[t]["type"]) > 0 and not is_worker(by_tag[t]["type"])
                and not is_building(by_tag[t]["type"]) and by_tag[t]["completed"]]
        enemy = [u for u in by_tag.values() if u["owner"] not in (g.slot, 11) and supply_x2(u["type"]) > 0
                 and not is_worker(u["type"]) and not is_building(u["type"])]
        if not mine or not enemy:
            return None
        index = {t: i for i, t in enumerate(tags[:MAX_UNITS])}
        roles: dict[str, list[dict]] = {}
        for u in mine:
            roles.setdefault(names[int(probs[index[u["tag"]]].argmax())].upper(), []).append(u)
        mx, my = _mid(mine)
        home = any(u["owner"] == g.slot and u["type"] == T.CC and math.dist((u["x"], u["y"]), (mx, my)) < HOME
                   for u in by_tag.values())
        fresh = lambda u, role: not (self.told.get(u["tag"], ("", -10**9))[0] == role
                                     and now - self.told[u["tag"]][1] < REPEAT)
        return self._act(roles, mine, enemy, by_tag, home, fresh)
