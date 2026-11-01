"""Protoss game facts: ids, costs, footprints, who makes what, and what needs what. The Protoss
counterpart of gary/terran.py and gary/zerg.py, in the same form (docs/REFACTOR.md: the race object
proven on three races; no Protoss executor yet).

Unit and building ids are the game's (units.dat), tech and upgrade ids too. Upgrade costs are
level-1 costs. Checked against units.dat by tests/test_game_data.py.
"""

from __future__ import annotations

# units
CORSAIR, DARK_TEMPLAR, DARK_ARCHON, PROBE, ZEALOT, DRAGOON, HIGH_TEMPLAR, ARCHON = 60, 61, 63, 64, 65, 66, 67, 68
SHUTTLE, SCOUT, ARBITER, CARRIER, INTERCEPTOR, REAVER, OBSERVER, SCARAB = 69, 70, 71, 72, 73, 83, 84, 85
# buildings
NEXUS, ROBOTICS, PYLON, ASSIMILATOR, OBSERVATORY, GATEWAY, CANNON = 154, 155, 156, 157, 159, 160, 162
CITADEL, CYBER, TEMPLAR_ARCHIVES, FORGE, STARGATE = 163, 164, 165, 166, 167
FLEET_BEACON, ARBITER_TRIBUNAL, SUPPORT_BAY, BATTERY = 169, 170, 171, 172
GEYSER = 188
TOWN_HALLS = {NEXUS}

COST = {  # (minerals, gas)
    PROBE: (50, 0), ZEALOT: (100, 0), DRAGOON: (125, 50), HIGH_TEMPLAR: (50, 150), DARK_TEMPLAR: (125, 100),
    SHUTTLE: (200, 0), REAVER: (200, 100), OBSERVER: (25, 75), SCOUT: (275, 125), CORSAIR: (150, 100),
    CARRIER: (350, 250), ARBITER: (100, 350),
    NEXUS: (400, 0), PYLON: (100, 0), ASSIMILATOR: (100, 0), GATEWAY: (150, 0), FORGE: (150, 0),
    CANNON: (150, 0), CYBER: (200, 0), BATTERY: (100, 0), CITADEL: (150, 100), ROBOTICS: (200, 200),
    STARGATE: (150, 150), TEMPLAR_ARCHIVES: (150, 200), OBSERVATORY: (50, 100), SUPPORT_BAY: (150, 100),
    FLEET_BEACON: (300, 200), ARBITER_TRIBUNAL: (200, 150),
}
SUPPLY = {  # whole supply
    PROBE: 1, ZEALOT: 2, DRAGOON: 2, HIGH_TEMPLAR: 2, DARK_TEMPLAR: 2, ARCHON: 4, DARK_ARCHON: 4,
    SHUTTLE: 2, REAVER: 4, OBSERVER: 1, SCOUT: 3, CORSAIR: 2, CARRIER: 6, ARBITER: 4,
}
SUPPLY_PROVIDED = {PYLON: 8, NEXUS: 9}
SIZE = {  # footprint in tiles
    NEXUS: (4, 3), PYLON: (2, 2), ASSIMILATOR: (4, 2), GATEWAY: (4, 3), FORGE: (3, 2), CANNON: (2, 2),
    CYBER: (3, 2), BATTERY: (3, 2), CITADEL: (3, 2), ROBOTICS: (3, 2), STARGATE: (4, 3),
    TEMPLAR_ARCHIVES: (3, 2), OBSERVATORY: (3, 2), SUPPORT_BAY: (3, 2), FLEET_BEACON: (3, 2),
    ARBITER_TRIBUNAL: (3, 2),
}

# who makes what: the building that warps a unit in, and the pair two units merge from
PRODUCER = {PROBE: NEXUS, ZEALOT: GATEWAY, DRAGOON: GATEWAY, HIGH_TEMPLAR: GATEWAY, DARK_TEMPLAR: GATEWAY,
            SHUTTLE: ROBOTICS, REAVER: ROBOTICS, OBSERVER: ROBOTICS,
            SCOUT: STARGATE, CORSAIR: STARGATE, CARRIER: STARGATE, ARBITER: STARGATE}
MERGE = {ARCHON: HIGH_TEMPLAR, DARK_ARCHON: DARK_TEMPLAR}
# what a unit fires that it builds itself (it is built empty): its ammunition and the minerals of one
AMMO = {REAVER: (SCARAB, 15), CARRIER: (INTERCEPTOR, 25)}

# what must exist (completed) before something can be made (buildings also need a pylon's power)
REQUIRES = {
    GATEWAY: [NEXUS], FORGE: [NEXUS], CYBER: [GATEWAY], CANNON: [FORGE], BATTERY: [GATEWAY],
    CITADEL: [CYBER], ROBOTICS: [CYBER], STARGATE: [CYBER], TEMPLAR_ARCHIVES: [CITADEL],
    OBSERVATORY: [ROBOTICS], SUPPORT_BAY: [ROBOTICS], FLEET_BEACON: [STARGATE],
    ARBITER_TRIBUNAL: [TEMPLAR_ARCHIVES, STARGATE],
    ZEALOT: [GATEWAY], DRAGOON: [CYBER], HIGH_TEMPLAR: [TEMPLAR_ARCHIVES], DARK_TEMPLAR: [TEMPLAR_ARCHIVES],
    SHUTTLE: [ROBOTICS], REAVER: [SUPPORT_BAY], OBSERVER: [OBSERVATORY], SCOUT: [STARGATE],
    CORSAIR: [STARGATE], CARRIER: [FLEET_BEACON], ARBITER: [ARBITER_TRIBUNAL],
}

# research (tech id -> building, cost) and upgrades (upgrade id -> building, level-1 cost, levels)
RESEARCH = {19: (TEMPLAR_ARCHIVES, (200, 200)),   # psionic storm
            20: (TEMPLAR_ARCHIVES, (150, 150)),   # hallucination
            27: (TEMPLAR_ARCHIVES, (200, 200)),   # mind control
            31: (TEMPLAR_ARCHIVES, (100, 100)),   # maelstrom
            21: (ARBITER_TRIBUNAL, (150, 150)),   # recall
            22: (ARBITER_TRIBUNAL, (150, 150)),   # stasis field
            25: (FLEET_BEACON, (200, 200))}       # disruption web
UPGRADE = {5: (FORGE, (100, 100), 3),             # ground armor
           13: (FORGE, (100, 100), 3),            # ground weapons
           15: (FORGE, (200, 200), 3),            # plasma shields
           6: (CYBER, (150, 150), 3),             # air armor
           14: (CYBER, (100, 100), 3),            # air weapons
           33: (CYBER, (150, 150), 1),            # singularity charge (dragoon range)
           34: (CITADEL, (150, 150), 1),          # leg enhancements (zealot speed)
           35: (SUPPORT_BAY, (200, 200), 1),      # scarab damage
           36: (SUPPORT_BAY, (200, 200), 1),      # reaver capacity
           37: (SUPPORT_BAY, (200, 200), 1),      # gravitic drive (shuttle speed)
           38: (OBSERVATORY, (100, 100), 1),      # sensor array
           39: (OBSERVATORY, (100, 100), 1),      # gravitic boosters
           40: (TEMPLAR_ARCHIVES, (150, 150), 1),   # khaydarin amulet
           49: (TEMPLAR_ARCHIVES, (150, 150), 1),   # argus talisman
           41: (FLEET_BEACON, (100, 100), 1),     # apial sensors
           42: (FLEET_BEACON, (200, 200), 1),     # gravitic thrusters
           43: (FLEET_BEACON, (100, 100), 1),     # carrier capacity
           47: (FLEET_BEACON, (100, 100), 1),     # argus jewel
           44: (ARBITER_TRIBUNAL, (150, 150), 1)}   # khaydarin core
UPGRADE_LEVEL_2_NEEDS = {5: TEMPLAR_ARCHIVES, 13: TEMPLAR_ARCHIVES, 15: CYBER, 6: FLEET_BEACON, 14: FLEET_BEACON}

ARMY = {ZEALOT, DRAGOON, HIGH_TEMPLAR, DARK_TEMPLAR, ARCHON, DARK_ARCHON, REAVER, SCOUT, CORSAIR, CARRIER,
        ARBITER}
PRODUCTION = {NEXUS, GATEWAY, ROBOTICS, STARGATE}
NAMES = {CORSAIR: "Corsair", DARK_TEMPLAR: "Dark Templar", DARK_ARCHON: "Dark Archon", PROBE: "Probe",
         ZEALOT: "Zealot", DRAGOON: "Dragoon", HIGH_TEMPLAR: "High Templar", ARCHON: "Archon",
         SHUTTLE: "Shuttle", SCOUT: "Scout", ARBITER: "Arbiter", CARRIER: "Carrier", INTERCEPTOR: "Interceptor",
         REAVER: "Reaver", OBSERVER: "Observer", SCARAB: "Scarab",
         NEXUS: "Nexus", ROBOTICS: "Robotics Facility", PYLON: "Pylon", ASSIMILATOR: "Assimilator",
         OBSERVATORY: "Observatory", GATEWAY: "Gateway", CANNON: "Photon Cannon", CITADEL: "Citadel of Adun",
         CYBER: "Cybernetics Core", TEMPLAR_ARCHIVES: "Templar Archives", FORGE: "Forge", STARGATE: "Stargate",
         FLEET_BEACON: "Fleet Beacon", ARBITER_TRIBUNAL: "Arbiter Tribunal", SUPPORT_BAY: "Robotics Support Bay",
         BATTERY: "Shield Battery"}
TECH_NAMES = {19: "Psionic Storm", 20: "Hallucination", 27: "Mind Control", 31: "Maelstrom", 21: "Recall",
              22: "Stasis Field", 25: "Disruption Web"}
UPGRADE_NAMES = {5: "Protoss Ground Armor", 13: "Protoss Ground Weapons", 15: "Protoss Plasma Shields",
                 6: "Protoss Air Armor", 14: "Protoss Air Weapons", 33: "Singularity Charge",
                 34: "Leg Enhancements", 35: "Scarab Damage", 36: "Reaver Capacity", 37: "Gravitic Drive",
                 38: "Sensor Array", 39: "Gravitic Boosters", 40: "Khaydarin Amulet", 49: "Argus Talisman",
                 41: "Apial Sensors", 42: "Gravitic Thrusters", 43: "Carrier Capacity", 47: "Argus Jewel",
                 44: "Khaydarin Core"}
