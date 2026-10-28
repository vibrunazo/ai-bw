"""Terran game facts Gary's bots need: ids, costs, footprints, who makes what, and what needs what.

Unit and building ids are the game's (units.dat); tech and upgrade ids are the game's too. Upgrade
costs are level-1 costs (each level costs more; the game refuses an order it can't pay for).
"""

from __future__ import annotations

# units
MARINE, GHOST, VULTURE, GOLIATH, TANK, SCV, WRAITH, VESSEL, DROPSHIP, BC = 0, 1, 2, 3, 5, 7, 8, 9, 11, 12
FIREBAT, MEDIC, VALKYRIE = 32, 34, 58
TANK_SIEGED = 30                       # a Siege Tank in Siege Mode (its own unit type in the game)
# buildings and add-ons
CC, COMSAT, NUKE_SILO, DEPOT, REFINERY, RAX, ACADEMY, FACTORY, STARPORT = 106, 107, 108, 109, 110, 111, 112, 113, 114
CONTROL_TOWER, SCIENCE_FACILITY, COVERT_OPS, PHYSICS_LAB, MACHINE_SHOP = 115, 116, 117, 118, 120
EBAY, ARMORY, TURRET, BUNKER = 122, 123, 124, 125
GEYSER = 188
TOWN_HALLS = {CC}

COST = {  # (minerals, gas)
    SCV: (50, 0), MARINE: (50, 0), FIREBAT: (50, 25), MEDIC: (50, 25), GHOST: (25, 75),
    VULTURE: (75, 0), TANK: (150, 100), GOLIATH: (100, 50),
    WRAITH: (150, 100), DROPSHIP: (100, 100), VESSEL: (100, 225), VALKYRIE: (250, 125), BC: (400, 300),
    CC: (400, 0), DEPOT: (100, 0), REFINERY: (100, 0), RAX: (150, 0), ACADEMY: (150, 0),
    FACTORY: (200, 100), STARPORT: (150, 100), SCIENCE_FACILITY: (100, 150), EBAY: (125, 0),
    ARMORY: (100, 50), TURRET: (75, 0), BUNKER: (100, 0),
    COMSAT: (50, 50), NUKE_SILO: (100, 100), MACHINE_SHOP: (50, 50), CONTROL_TOWER: (50, 50),
    PHYSICS_LAB: (50, 50), COVERT_OPS: (50, 50),
}
SIZE = {  # footprint in tiles
    CC: (4, 3), DEPOT: (3, 2), REFINERY: (4, 2), RAX: (4, 3), ACADEMY: (3, 2), FACTORY: (4, 3),
    STARPORT: (4, 3), SCIENCE_FACILITY: (4, 3), EBAY: (4, 3), ARMORY: (3, 2), TURRET: (2, 2),
    BUNKER: (3, 2), COMSAT: (2, 2), NUKE_SILO: (2, 2), MACHINE_SHOP: (2, 2), CONTROL_TOWER: (2, 2),
    PHYSICS_LAB: (2, 2), COVERT_OPS: (2, 2),
}
ADDON_PARENT = {COMSAT: CC, NUKE_SILO: CC, MACHINE_SHOP: FACTORY, CONTROL_TOWER: STARPORT,
                PHYSICS_LAB: SCIENCE_FACILITY, COVERT_OPS: SCIENCE_FACILITY}
ADDON_OFFSET = (4, 1)        # an add-on's top-left tile, from its parent's top-left tile

# what must exist (completed) before something can be made
REQUIRES = {
    RAX: [CC], ACADEMY: [RAX], FACTORY: [RAX], BUNKER: [RAX], EBAY: [CC], TURRET: [EBAY],
    STARPORT: [FACTORY], ARMORY: [FACTORY], SCIENCE_FACILITY: [STARPORT],
    COMSAT: [ACADEMY], NUKE_SILO: [COVERT_OPS],
    FIREBAT: [ACADEMY], MEDIC: [ACADEMY], GHOST: [ACADEMY, COVERT_OPS], GOLIATH: [ARMORY],
    VESSEL: [SCIENCE_FACILITY], VALKYRIE: [ARMORY],
}
# units: the building that trains them, and the add-on that building must have
PRODUCER = {SCV: CC, MARINE: RAX, FIREBAT: RAX, MEDIC: RAX, GHOST: RAX,
            VULTURE: FACTORY, TANK: FACTORY, GOLIATH: FACTORY,
            WRAITH: STARPORT, DROPSHIP: STARPORT, VESSEL: STARPORT, VALKYRIE: STARPORT, BC: STARPORT}
NEEDS_ADDON = {TANK: MACHINE_SHOP, DROPSHIP: CONTROL_TOWER, VESSEL: CONTROL_TOWER,
               VALKYRIE: CONTROL_TOWER, BC: CONTROL_TOWER}
REQUIRES[BC] = [PHYSICS_LAB]

# research (tech id -> building, cost) and upgrades (upgrade id -> building, level-1 cost, levels)
RESEARCH = {0: (ACADEMY, (100, 100)),            # stim packs
            24: (ACADEMY, (100, 100)),           # restoration
            3: (MACHINE_SHOP, (100, 100)),       # spider mines
            5: (MACHINE_SHOP, (150, 150)),       # siege mode
            2: (SCIENCE_FACILITY, (200, 200)),   # EMP
            7: (SCIENCE_FACILITY, (200, 200)),   # irradiate
            8: (PHYSICS_LAB, (100, 100)),        # yamato
            9: (CONTROL_TOWER, (150, 150))}      # cloaking field
UPGRADE = {0: (EBAY, (100, 100), 3),             # infantry armor
           7: (EBAY, (100, 100), 3),             # infantry weapons
           1: (ARMORY, (100, 100), 3),           # vehicle plating
           8: (ARMORY, (100, 100), 3),           # vehicle weapons
           2: (ARMORY, (150, 150), 3),           # ship plating
           9: (ARMORY, (100, 100), 3),           # ship weapons
           16: (ACADEMY, (150, 150), 1),         # U-238 shells
           17: (MACHINE_SHOP, (100, 100), 1),    # ion thrusters
           54: (MACHINE_SHOP, (100, 100), 1),    # charon boosters
           19: (SCIENCE_FACILITY, (150, 150), 1)}   # titan reactor
UPGRADE_LEVEL_2_NEEDS = SCIENCE_FACILITY           # levels 2 and 3 of weapons/armor

ARMY = {MARINE, FIREBAT, MEDIC, GHOST, VULTURE, TANK, GOLIATH, WRAITH, VESSEL, VALKYRIE, BC}
IN_MODE = {TANK_SIEGED: TANK}           # a unit type that is another in a mode: counted and moved as that one
# the fight's modes (gary/agent/roles.py, #60): a role -> the unit types it takes, the tech it needs and that tech's
# research frames, the order it starts, and the uses a unit has (None: any)
MODES = {"siege": ({TANK}, 5, 1200, 98, None), "unsiege": ({TANK_SIEGED}, 5, 1200, 99, None),
         "mine": ({VULTURE}, 3, 1200, 132, 3)}
PRODUCTION = {CC, RAX, FACTORY, STARPORT}
NAMES = {MARINE: "Marine", GHOST: "Ghost", VULTURE: "Vulture", GOLIATH: "Goliath", TANK: "Siege Tank",
         TANK_SIEGED: "Siege Tank (sieged)", SCV: "SCV", WRAITH: "Wraith", VESSEL: "Science Vessel", DROPSHIP: "Dropship", BC: "Battlecruiser",
         FIREBAT: "Firebat", MEDIC: "Medic", VALKYRIE: "Valkyrie", CC: "Command Center",
         COMSAT: "Comsat Station", NUKE_SILO: "Nuclear Silo", DEPOT: "Supply Depot", REFINERY: "Refinery",
         RAX: "Barracks", ACADEMY: "Academy", FACTORY: "Factory", STARPORT: "Starport",
         CONTROL_TOWER: "Control Tower", SCIENCE_FACILITY: "Science Facility", COVERT_OPS: "Covert Ops",
         PHYSICS_LAB: "Physics Lab", MACHINE_SHOP: "Machine Shop", EBAY: "Engineering Bay",
         ARMORY: "Armory", TURRET: "Missile Turret", BUNKER: "Bunker"}
TECH_NAMES = {0: "Stim Packs", 24: "Restoration", 3: "Spider Mines", 5: "Siege Mode", 2: "EMP Shockwave",
              7: "Irradiate", 8: "Yamato Gun", 9: "Cloaking Field"}
UPGRADE_NAMES = {0: "Infantry Armor", 7: "Infantry Weapons", 1: "Vehicle Plating", 8: "Vehicle Weapons",
                 2: "Ship Plating", 9: "Ship Weapons", 16: "U-238 Shells", 17: "Ion Thrusters",
                 54: "Charon Boosters", 19: "Titan Reactor"}

SUPPLY = {SCV: 1, MARINE: 1, FIREBAT: 1, MEDIC: 1, GHOST: 1, VULTURE: 2, TANK: 2, GOLIATH: 2,
          WRAITH: 2, DROPSHIP: 2, VESSEL: 2, VALKYRIE: 3, BC: 6}
SUPPLY_PROVIDED = {DEPOT: 8, CC: 10}
