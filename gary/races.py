"""What a player knows about an enemy by race: the buildings that end a game, produce, or shoot
back, its fighters and their names. The matchup-independent rules (the press ranks the enemy's
buildings, the sweep avoids its static defense, the alarm counts fighters away from its town halls)
look the enemy up here instead of naming Terran buildings (docs/RACES.md)."""

from __future__ import annotations

from dataclasses import dataclass

from gary import protoss as P
from gary import terran as T
from gary import zerg as Z


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
    "P": Enemy("P", frozenset(P.TOWN_HALLS), frozenset(P.PRODUCTION), frozenset({P.CANNON}), frozenset(P.ARMY),
               P.NAMES),
}


# flyers, of every race: what the ground army and workers can't catch or hit (the roles don't close in on a group of
# them alone away from home, the workers' pull doesn't count them)
FLYERS = frozenset({Z.MUTALISK, Z.SCOURGE, Z.GUARDIAN, Z.DEVOURER, Z.QUEEN, Z.OVERLORD, Z.COCOON,
                    T.WRAITH, T.VALKYRIE, T.BC, T.VESSEL, T.DROPSHIP,
                    P.CORSAIR, P.SCOUT, P.ARBITER, P.CARRIER, P.SHUTTLE, P.OBSERVER})


def enemy_of(matchup: str, race: str) -> Enemy:
    """The enemy in a matchup for a player of this race: "ZvZ", "Z" -> the Zerg (a mirror)."""
    races = matchup.split("v")
    if race not in races:
        raise ValueError(f"{race} does not play in {matchup}")
    return ENEMY[races[1] if races[0] == race else races[0]]
