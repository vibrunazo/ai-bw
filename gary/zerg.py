"""Zerg game facts Gary-Z needs (docs/SELF_PLAY.md): ids, costs, footprints, who makes what, and
what needs what. The Zerg counterpart of gary/terran.py, in the same form.

Unit and building ids are the game's (units.dat), tech and upgrade ids too. Costs and supply are a
unit's as the game charges them: a zergling or scourge order makes two (COST and SUPPLY are the
pair's). Upgrade costs are level-1 costs. Checked against units.dat by tests/test_game_data.py.
"""

from __future__ import annotations

# units
ZERGLING, HYDRALISK, ULTRALISK, DRONE, OVERLORD, MUTALISK, GUARDIAN, QUEEN = 37, 38, 39, 41, 42, 43, 44, 45
DEFILER, SCOURGE, DEVOURER, LURKER = 46, 47, 62, 103
LARVA, EGG, COCOON, LURKER_EGG = 35, 36, 59, 97
# buildings
HATCHERY, LAIR, HIVE, NYDUS, DEN, DEFILER_MOUND, GREATER_SPIRE, QUEENS_NEST = 131, 132, 133, 134, 135, 136, 137, 138
EVO, ULTRA_CAVERN, SPIRE, POOL, CREEP, SPORE, SUNKEN, EXTRACTOR = 139, 140, 141, 142, 143, 144, 146, 149
GEYSER = 188
TOWN_HALLS = {HATCHERY, LAIR, HIVE}

COST = {  # (minerals, gas)
    DRONE: (50, 0), ZERGLING: (50, 0), OVERLORD: (100, 0), HYDRALISK: (75, 25), MUTALISK: (100, 100),
    SCOURGE: (25, 75), QUEEN: (100, 100), ULTRALISK: (200, 200), DEFILER: (50, 150),
    LURKER: (50, 100), GUARDIAN: (50, 100), DEVOURER: (150, 50),
    HATCHERY: (300, 0), LAIR: (150, 100), HIVE: (200, 150), POOL: (200, 0), EXTRACTOR: (50, 0),
    DEN: (100, 50), SPIRE: (200, 150), GREATER_SPIRE: (100, 150), QUEENS_NEST: (150, 100),
    EVO: (75, 0), ULTRA_CAVERN: (150, 200), DEFILER_MOUND: (100, 100), CREEP: (75, 0),
    SUNKEN: (50, 0), SPORE: (50, 0), NYDUS: (150, 0),
}
SUPPLY = {  # whole supply, what one order costs (a zergling or scourge order makes two)
    DRONE: 1, ZERGLING: 1, HYDRALISK: 1, MUTALISK: 2, SCOURGE: 1, QUEEN: 2, ULTRALISK: 4, DEFILER: 2,
    LURKER: 2, GUARDIAN: 2, DEVOURER: 2, OVERLORD: 0,
}
PAIRS = {ZERGLING, SCOURGE}                # one larva makes two
SUPPLY_PROVIDED = {OVERLORD: 8, HATCHERY: 1, LAIR: 1, HIVE: 1}
SIZE = {  # footprint in tiles
    HATCHERY: (4, 3), LAIR: (4, 3), HIVE: (4, 3), POOL: (3, 2), EXTRACTOR: (4, 2), DEN: (3, 2),
    SPIRE: (2, 2), GREATER_SPIRE: (2, 2), QUEENS_NEST: (3, 2), EVO: (3, 2), ULTRA_CAVERN: (3, 2),
    DEFILER_MOUND: (4, 2), CREEP: (2, 2), SUNKEN: (2, 2), SPORE: (2, 2), NYDUS: (2, 2),
}

# who makes what: from a larva, by morphing a unit, or by morphing a building (the rest a drone builds)
FROM_LARVA = {DRONE, ZERGLING, OVERLORD, HYDRALISK, MUTALISK, SCOURGE, QUEEN, ULTRALISK, DEFILER}
UNIT_MORPH = {LURKER: HYDRALISK, GUARDIAN: MUTALISK, DEVOURER: MUTALISK}
BUILDING_MORPH = {LAIR: HATCHERY, HIVE: LAIR, GREATER_SPIRE: SPIRE, SUNKEN: CREEP, SPORE: CREEP}
DRONE_BUILDS = {HATCHERY, POOL, EXTRACTOR, DEN, SPIRE, QUEENS_NEST, EVO, ULTRA_CAVERN, DEFILER_MOUND,
                CREEP, NYDUS}

# what must exist (completed) before something can be made; a Lair or Hive counts as a Hatchery,
# a Hive as a Lair (HAS_TECH below), a Greater Spire as a Spire
REQUIRES = {
    POOL: [HATCHERY], EXTRACTOR: [], EVO: [HATCHERY], CREEP: [HATCHERY], DEN: [POOL], LAIR: [POOL],
    SUNKEN: [POOL], SPORE: [EVO], SPIRE: [LAIR], QUEENS_NEST: [LAIR], HIVE: [QUEENS_NEST],
    ULTRA_CAVERN: [HIVE], DEFILER_MOUND: [HIVE], GREATER_SPIRE: [HIVE], NYDUS: [HIVE],
    ZERGLING: [POOL], HYDRALISK: [DEN], MUTALISK: [SPIRE], SCOURGE: [SPIRE], QUEEN: [QUEENS_NEST],
    ULTRALISK: [ULTRA_CAVERN], DEFILER: [DEFILER_MOUND], LURKER: [DEN, LAIR],
    GUARDIAN: [GREATER_SPIRE], DEVOURER: [GREATER_SPIRE],
}
HAS_TECH = {HATCHERY: {HATCHERY, LAIR, HIVE}, LAIR: {LAIR, HIVE}, SPIRE: {SPIRE, GREATER_SPIRE}}

# research (tech id -> building, cost) and upgrades (upgrade id -> building, level-1 cost, levels)
LURKER_ASPECT = 32
RESEARCH = {11: (HATCHERY, (100, 100)),          # burrow
            13: (QUEENS_NEST, (100, 100)),       # spawn broodlings
            15: (DEFILER_MOUND, (200, 200)),     # plague
            16: (DEFILER_MOUND, (100, 100)),     # consume
            17: (QUEENS_NEST, (100, 100)),       # ensnare
            LURKER_ASPECT: (DEN, (200, 200))}
RESEARCH_NEEDS = {LURKER_ASPECT: [LAIR]}
UPGRADE = {3: (EVO, (150, 150), 3),              # carapace
           10: (EVO, (100, 100), 3),             # melee attacks
           11: (EVO, (100, 100), 3),             # missile attacks
           4: (SPIRE, (150, 150), 3),            # flyer carapace
           12: (SPIRE, (100, 100), 3),           # flyer attacks
           24: (LAIR, (200, 200), 1),            # ventral sacs
           25: (LAIR, (150, 150), 1),            # antennae
           26: (LAIR, (150, 150), 1),            # pneumatized carapace (overlord speed)
           27: (POOL, (100, 100), 1),            # metabolic boost (zergling speed)
           28: (POOL, (200, 200), 1),            # adrenal glands
           29: (DEN, (150, 150), 1),             # muscular augments
           30: (DEN, (150, 150), 1),             # grooved spines
           31: (QUEENS_NEST, (150, 150), 1),     # gamete meiosis
           32: (DEFILER_MOUND, (150, 150), 1),   # metasynaptic node
           52: (ULTRA_CAVERN, (200, 200), 1),    # anabolic synthesis
           53: (ULTRA_CAVERN, (150, 150), 1)}    # chitinous plating
UPGRADE_NEEDS = {28: [HIVE]}                     # (levels 2 and 3 of the Evolution Chamber's and the
UPGRADE_LEVEL_NEEDS = {2: LAIR, 3: HIVE}         # Spire's upgrades need a Lair, then a Hive)

ARMY = {ZERGLING, HYDRALISK, MUTALISK, SCOURGE, QUEEN, ULTRALISK, DEFILER, LURKER, GUARDIAN, DEVOURER}
NAMES = {ZERGLING: "Zergling", HYDRALISK: "Hydralisk", ULTRALISK: "Ultralisk", DRONE: "Drone",
         OVERLORD: "Overlord", MUTALISK: "Mutalisk", GUARDIAN: "Guardian", QUEEN: "Queen", DEFILER: "Defiler",
         SCOURGE: "Scourge", DEVOURER: "Devourer", LURKER: "Lurker", LARVA: "Larva", EGG: "Egg",
         HATCHERY: "Hatchery", LAIR: "Lair", HIVE: "Hive", NYDUS: "Nydus Canal", DEN: "Hydralisk Den",
         DEFILER_MOUND: "Defiler Mound", GREATER_SPIRE: "Greater Spire", QUEENS_NEST: "Queen's Nest",
         EVO: "Evolution Chamber", ULTRA_CAVERN: "Ultralisk Cavern", SPIRE: "Spire", POOL: "Spawning Pool",
         CREEP: "Creep Colony", SPORE: "Spore Colony", SUNKEN: "Sunken Colony", EXTRACTOR: "Extractor"}
TECH_NAMES = {11: "Burrow", 13: "Spawn Broodlings", 15: "Plague", 16: "Consume", 17: "Ensnare",
              LURKER_ASPECT: "Lurker Aspect"}
UPGRADE_NAMES = {3: "Carapace", 10: "Melee Attacks", 11: "Missile Attacks", 4: "Flyer Carapace",
                 12: "Flyer Attacks", 24: "Ventral Sacs", 25: "Antennae", 26: "Pneumatized Carapace",
                 27: "Metabolic Boost", 28: "Adrenal Glands", 29: "Muscular Augments", 30: "Grooved Spines",
                 31: "Gamete Meiosis", 32: "Metasynaptic Node", 52: "Anabolic Synthesis", 53: "Chitinous Plating"}


def has(done_types: set[int], need: int) -> bool:
    """A requirement met by what exists (completed): a Lair or Hive is a Hatchery, a Hive a Lair..."""
    return bool(HAS_TECH.get(need, {need}) & done_types)


def can_make(uid: int, done_types: set[int], researched: set[int] = frozenset()) -> bool:
    """Every requirement of this unit or building is met."""
    if uid == LURKER and LURKER_ASPECT not in researched:
        return False
    return all(has(done_types, r) for r in REQUIRES.get(uid, []))
