"""How a fight went, as one number in value (minerals + gas), for scenarios, drills and RL rewards.

Every change is an event in time (frame, value; positive: good for the Terran), so a decision can
be credited with what happened after it (train/fight_cmd_rl.py):

  deaths     a unit dying: its value (Zerg +, Terran -), less the part its damage already counted
  damage     HP lost, as a share of the unit's HP, times its value, at partial weight (healing
             counts back): DAMAGE_WEIGHT for the Terran's units, less for the Zerg's (they
             regenerate), much less for buildings (they're repaired)
  energy     spellcasters' energy spent (medics healing...), at ENERGY_VALUE a point (it regenerates)
  threats    at the end, enemy fighters still near a Terran town hall: THREAT_WEIGHT of their value
             against the Terran (left there, they'd go on killing workers): backing off from a
             defense doesn't pay

T_lost / Z_lost and the workers lost count deaths only (for reading); "net" is the whole score.
"""

from __future__ import annotations

import math

from gary import terran as T

ZERG_COST = {37: (25, 0), 38: (75, 25), 39: (200, 200), 41: (50, 0), 42: (100, 0), 43: (100, 100),
             45: (100, 100), 46: (50, 150), 47: (12, 38), 103: (125, 125),
             131: (300, 0), 142: (200, 0), 149: (75, 0), 143: (125, 0), 146: (175, 0), 141: (200, 150),
             132: (450, 100), 135: (150, 0), 136: (150, 100), 137: (100, 100), 138: (200, 150)}
MEDIC = 34
WORKERS = {T.SCV, 41}
DAMAGE_WEIGHT = {"T": 0.5, "Z": 0.35}
BUILDING_DAMAGE = 0.2            # damage to buildings counts at this share of a unit's
ENERGY_USERS = {MEDIC, 1, 9, 45, 46}   # medic, ghost, science vessel, queen, defiler
ENERGY_VALUE = 0.6               # value per energy point (a medic heals ~2 HP a point; it regenerates)
THREAT_WEIGHT = 0.5
THREAT_PX = 12 * 32              # enemy fighters this close to a Terran town hall are a threat
TOWN_HALLS = {T.CC}


def value(unit_type: int) -> int:
    m, g = T.COST.get(unit_type) or ZERG_COST.get(unit_type) or (0, 0)
    if unit_type == MEDIC:
        m, g = 50, 25
    return m + g


def is_building(unit_type: int) -> bool:
    return 106 <= unit_type <= 175


class Score:
    def __init__(self, terran: int, zerg: int):
        self.side = {terran: "T", zerg: "Z"}
        self.terran, self.zerg = terran, zerg
        self.seen: dict[int, dict] = {}
        self.events: list[tuple[int, float]] = []
        self.parts = {"deaths": 0.0, "damage": 0.0, "energy": 0.0, "threats": 0.0}
        self.lost = {"T": 0, "Z": 0}
        self.workers_lost = {"T": 0, "Z": 0}
        self.lost_by_type: dict[str, dict[int, int]] = {"T": {}, "Z": {}}   # value lost per unit type

    def _add(self, frame: int, part: str, v: float) -> None:
        if v:
            self.events.append((frame, v))
            self.parts[part] += v

    def note(self, obs: dict, game) -> None:
        """The game now (the full state, not fogged); call every few frames."""
        frame = obs["frame"]
        here = set()
        for u in obs["units"]:
            side = self.side.get(u["owner"])
            if side is None:
                continue
            here.add(u["tag"])
            sign = 1 if side == "Z" else -1      # a Zerg loss is good for the Terran
            s = self.seen.get(u["tag"])
            hp = (u.get("hp") or 0) + (u.get("shields") or 0)
            if s is None:
                self.seen[u["tag"]] = {"side": side, "type": u["type"], "value": value(u["type"]), "hp": hp,
                                       "energy": u.get("energy", 0), "credited": 0.0, "dead": False}
                continue
            if u.get("hp_max") and hp != s["hp"]:
                w = DAMAGE_WEIGHT[side] * (BUILDING_DAMAGE if is_building(u["type"]) else 1.0)
                share = (s["hp"] - hp) / u["hp_max"] * s["value"] * w     # > 0: damaged
                s["credited"] += share
                self._add(frame, "damage", sign * share)
                s["hp"] = hp
            if u["type"] in ENERGY_USERS and u.get("energy", 0) != s["energy"]:
                self._add(frame, "energy", sign * (s["energy"] - u.get("energy", 0)) * ENERGY_VALUE)
                s["energy"] = u.get("energy", 0)
        for tag, s in self.seen.items():          # gone from view: dead, or inside a bunker / refinery
            if not s["dead"] and tag not in here and game.unit_type_of(tag) == -1:
                s["dead"] = True
                sign = 1 if s["side"] == "Z" else -1
                self._add(frame, "deaths", sign * (s["value"] - s["credited"]))
                self.lost[s["side"]] += s["value"]
                by = self.lost_by_type[s["side"]]
                by[s["type"]] = by.get(s["type"], 0) + s["value"]
                self.workers_lost[s["side"]] += s["type"] in WORKERS

    def finish(self, obs: dict, game, is_fighter) -> None:
        """The end of the window: the last changes, and the threats left at the Terran's bases."""
        self.note(obs, game)
        halls = [(u["x"], u["y"]) for u in obs["units"] if u["owner"] == self.terran and u["type"] in TOWN_HALLS]
        for u in obs["units"]:
            if u["owner"] == self.zerg and is_fighter(u["type"]) and \
                    any(math.dist((u["x"], u["y"]), h) < THREAT_PX for h in halls):
                self._add(obs["frame"], "threats", -THREAT_WEIGHT * value(u["type"]))

    def result(self) -> dict:
        return {"net": round(sum(v for _, v in self.events)), "T_lost": self.lost["T"], "Z_lost": self.lost["Z"],
                "T_workers_lost": self.workers_lost["T"], "Z_workers_lost": self.workers_lost["Z"],
                "T_lost_by_type": dict(self.lost_by_type["T"]), "Z_lost_by_type": dict(self.lost_by_type["Z"]),
                **{f"net_{k}": round(v) for k, v in self.parts.items()}, "events": list(self.events)}


def fighter(unit_type: int) -> bool:
    from gary.policy.army import is_worker, supply_x2
    return supply_x2(unit_type) > 0 and not is_worker(unit_type) and not is_building(unit_type)
