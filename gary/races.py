"""What a player knows about an enemy by race: the buildings that end a game, produce, or shoot
back, its fighters and their names. The matchup-independent rules (the press ranks the enemy's
buildings, the sweep avoids its static defense, the alarm counts fighters away from its town halls)
look the enemy up here instead of naming Terran buildings (docs/RACES.md)."""

from __future__ import annotations

from dataclasses import dataclass

from gary import terran as T
from gary import zerg as Z

# Protoss ids (gary/protoss.py when Gary plays it)
NEXUS, ROBOTICS, GATEWAY, FORGE, CANNON, STARGATE = 154, 155, 160, 165, 166, 167
CORSAIR, DARK_TEMPLAR, DARK_ARCHON, PROBE, ZEALOT, DRAGOON, HIGH_TEMPLAR, ARCHON = 60, 61, 63, 64, 65, 66, 67, 68
SHUTTLE, SCOUT, ARBITER, CARRIER, REAVER, OBSERVER = 69, 70, 71, 72, 83, 84


@dataclass(frozen=True)
class Enemy:
    race: str
    town_halls: frozenset       # what ends a game first
    production: frozenset       # what makes its army (town halls included where they do)
    static: frozenset           # static defense a group should not walk under
    army: frozenset             # its fighting units (no workers)
    names: dict


ENEMY = {
    "T": Enemy("T", frozenset({T.CC}), frozenset(T.PRODUCTION), frozenset({T.TURRET, T.BUNKER}), frozenset(T.ARMY),
               T.NAMES),
    "Z": Enemy("Z", frozenset(Z.TOWN_HALLS), frozenset(Z.TOWN_HALLS), frozenset({Z.SUNKEN, Z.SPORE}), frozenset(Z.ARMY),
               Z.NAMES),
    "P": Enemy("P", frozenset({NEXUS}), frozenset({NEXUS, GATEWAY, ROBOTICS, STARGATE}), frozenset({CANNON}),
               frozenset({CORSAIR, DARK_TEMPLAR, DARK_ARCHON, ZEALOT, DRAGOON, HIGH_TEMPLAR, ARCHON, SCOUT, ARBITER,
                          CARRIER, REAVER}),
               {NEXUS: "Nexus", ROBOTICS: "Robotics Facility", GATEWAY: "Gateway", FORGE: "Forge", CANNON: "Photon Cannon",
                STARGATE: "Stargate", CORSAIR: "Corsair", DARK_TEMPLAR: "Dark Templar", DARK_ARCHON: "Dark Archon",
                PROBE: "Probe", ZEALOT: "Zealot", DRAGOON: "Dragoon", HIGH_TEMPLAR: "High Templar", ARCHON: "Archon",
                SHUTTLE: "Shuttle", SCOUT: "Scout", ARBITER: "Arbiter", CARRIER: "Carrier", REAVER: "Reaver",
                OBSERVER: "Observer"}),
}


def enemy_of(matchup: str, race: str) -> Enemy:
    """The enemy in a matchup for a player of this race: "ZvZ", "Z" -> the Zerg (a mirror)."""
    races = matchup.split("v")
    if race not in races:
        raise ValueError(f"{race} does not play in {matchup}")
    return ENEMY[races[1] if races[0] == race else races[0]]
