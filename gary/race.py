"""The race Gary plays, as one object (docs/REFACTOR.md Phase 0): the agent reads this race's facts
through `g.race`, never by importing a race's data module. The data modules (gary/terran.py,
gary/zerg.py, gary/protoss.py) keep their constants; a Race is the index over them, the same names
for every race. The enemy's side is gary/races.py (`enemy_of`)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from types import ModuleType

RACE_CODE = {0: "Z", 1: "T", 2: "P"}      # the game's race numbers (obs["players"][i]["race"])


def matchup_of(players: list[dict], slot: int, race: str) -> str | None:
    """The matchup as the models name it ("TvZ", "ZvZ", "PvT": the letters in order) for the player in
    this slot playing this race, from the game's players; None without an opponent."""
    other = next((RACE_CODE.get(p["race"]) for p in players if p["slot"] != slot and p.get("race") in RACE_CODE), None)
    return "v".join(sorted((race, other))) if other else None


@dataclass(frozen=True)
class Race:
    letter: str
    data: ModuleType            # the data module, for what only this race has (its executor reads it)
    worker: int
    town_halls: frozenset
    town_hall: int              # the one a worker builds (a Lair or Hive is morphed)
    supply_provider: int        # what a player builds for supply (Zerg: the Overlord, from a larva)
    gas_building: int
    production: frozenset       # the buildings that make its army (town halls included)
    producer: dict              # unit -> the building that makes it (Zerg: a town hall's larvae, or the unit morphed)
    costs: dict                 # unit -> (minerals, gas)
    sizes: dict                 # building -> footprint in tiles
    requires: dict              # unit -> what must exist (completed) first
    research: dict              # tech id -> (building, cost)
    upgrade: dict               # upgrade id -> (building, level-1 cost, levels)
    army: frozenset             # its fighting units (no workers)
    names: dict
    tech_names: dict
    upgrade_names: dict
    supply: dict                # unit -> supply it uses
    supply_provided: dict       # unit or building -> supply it gives
    minerals: dict = field(init=False)   # unit -> its mineral cost

    def __post_init__(self):
        object.__setattr__(self, "minerals", {u: m for u, (m, _gas) in self.costs.items()})

    @staticmethod
    @cache
    def of(letter: str) -> "Race":
        """Race.of("T") / ("Z") / ("P")."""
        from gary import protoss as P
        from gary import terran as T
        from gary import zerg as Z
        if letter == "T":
            return Race("T", T, T.SCV, frozenset(T.TOWN_HALLS), T.CC, T.DEPOT, T.REFINERY, frozenset(T.PRODUCTION),
                        T.PRODUCER, T.COST, T.SIZE, T.REQUIRES, T.RESEARCH, T.UPGRADE, frozenset(T.ARMY), T.NAMES,
                        T.TECH_NAMES, T.UPGRADE_NAMES, T.SUPPLY, T.SUPPLY_PROVIDED)
        if letter == "Z":
            return Race("Z", Z, Z.DRONE, frozenset(Z.TOWN_HALLS), Z.HATCHERY, Z.OVERLORD, Z.EXTRACTOR, frozenset(Z.PRODUCTION),
                        Z.PRODUCER, Z.COST, Z.SIZE, Z.REQUIRES, Z.RESEARCH, Z.UPGRADE, frozenset(Z.ARMY), Z.NAMES,
                        Z.TECH_NAMES, Z.UPGRADE_NAMES, Z.SUPPLY, Z.SUPPLY_PROVIDED)
        if letter == "P":
            return Race("P", P, P.PROBE, frozenset(P.TOWN_HALLS), P.NEXUS, P.PYLON, P.ASSIMILATOR, frozenset(P.PRODUCTION),
                        P.PRODUCER, P.COST, P.SIZE, P.REQUIRES, P.RESEARCH, P.UPGRADE, frozenset(P.ARMY), P.NAMES,
                        P.TECH_NAMES, P.UPGRADE_NAMES, P.SUPPLY, P.SUPPLY_PROVIDED)
        raise ValueError(f"no race {letter!r} (T, Z or P)")
