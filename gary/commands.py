"""Brood War commands, encoded as in replays (the bytes after the player id).

These are what the game engine executes. For now Gary's code calls them directly; later the
human interface (docs/ARCHITECTURE.md §6.3) will be the only thing allowed to emit them, after
applying camera, selection, APM and mouse limits.
"""

from __future__ import annotations

import struct

# Unit type IDs (Brood War)
SCV, DRONE, PROBE = 7, 41, 64
COMMAND_CENTER, HATCHERY, NEXUS = 106, 131, 154
SUPPLY_DEPOT, BARRACKS, PYLON = 109, 111, 156
MINERAL_FIELDS = {176, 177, 178}
NO_UNIT = 0xE4  # "none" unit type

# Order IDs used by build commands
ORDER_PLACE_BUILDING = 0x1E      # Terran
ORDER_PLACE_PROTOSS_BUILDING = 0x1F
ORDER_DRONE_START_BUILD = 0x19


def select(tags: list[int]) -> bytes:
    if not 1 <= len(tags) <= 12:
        raise ValueError("a selection holds 1 to 12 units")
    return struct.pack("<BB", 0x09, len(tags)) + b"".join(struct.pack("<H", t) for t in tags)


def hotkey(action: int, group: int) -> bytes:
    """A control group (command 0x13): action 0 assigns the selection to the group (Ctrl+n), 1
    recalls it (n), 2 adds the selection to it (Shift+n); groups 0-9."""
    if action not in (0, 1, 2) or not 0 <= group <= 9:
        raise ValueError("hotkey action 0-2, group 0-9")
    return struct.pack("<BBB", 0x13, action, group)


def right_click(x: int, y: int, target_tag: int = 0, target_type: int = NO_UNIT, queued: bool = False) -> bytes:
    """Move to (x, y) in pixels, or act on a target unit (mine a mineral field, attack...)."""
    return struct.pack("<BhhHHB", 0x14, x, y, target_tag, target_type, int(queued))


def train(unit_type: int) -> bytes:
    """Train or morph from the selected building / larva."""
    return struct.pack("<BH", 0x1F, unit_type)


def build(unit_type: int, tile_x: int, tile_y: int, order: int = ORDER_PLACE_BUILDING) -> bytes:
    """The selected worker builds unit_type with its top-left corner at (tile_x, tile_y)."""
    return struct.pack("<BBHHH", 0x0C, order, tile_x, tile_y, unit_type)


def research(tech: int) -> bytes:
    """The selected building researches a tech (stim packs, siege mode...)."""
    return struct.pack("<BB", 0x30, tech)


def upgrade(upgrade_id: int) -> bytes:
    """The selected building starts an upgrade (weapons, armor, U-238 shells...)."""
    return struct.pack("<BB", 0x32, upgrade_id)


def stop() -> bytes:
    """The selected units stop."""
    return struct.pack("<BB", 0x1A, 0)


def hold_position() -> bytes:
    return struct.pack("<BB", 0x2B, 0)


def burrow() -> bytes:
    """Zerg: the selected units burrow (lurkers, and the rest once Burrow is researched)."""
    return struct.pack("<BB", 0x2C, 0)


def unburrow() -> bytes:
    return struct.pack("<BB", 0x2D, 0)


def stim() -> bytes:
    """Marines and firebats in the selection use stim packs."""
    return struct.pack("<B", 0x36)


def return_cargo() -> bytes:
    return struct.pack("<BB", 0x1E, 0)


def unload_all() -> bytes:
    """The selected bunker (or transport) lets everyone out."""
    return struct.pack("<BB", 0x28, 0)


def chat(sender_slot: int, text: str) -> bytes:
    """A chat message to everyone (as replays record it): up to 80 bytes of text."""
    return struct.pack("<BB", 0x5C, sender_slot) + text.encode("utf-8", "replace")[:79].ljust(80, b"\0")


def morph(unit_type: int) -> bytes:
    """Zerg: morph the selected larva (or units) into unit_type."""
    return struct.pack("<BH", 0x23, unit_type)


def morph_building(unit_type: int) -> bytes:
    """Zerg: morph the selected building into unit_type (a Hatchery into a Lair, a Lair into a Hive)."""
    return struct.pack("<BH", 0x35, unit_type)


LARVA = 35


ORDER_ATTACK_MOVE = 0x0E
ORDER_SCANNER_SWEEP = 0x8B           # a Comsat Station's scan (50 energy)
MARINE = 0
REFINERY = 110


def targeted_order(order: int, x: int, y: int, target_tag: int = 0, target_type: int = NO_UNIT,
                   queued: bool = False) -> bytes:
    """A specific order at a point or unit (attack-move, patrol, ...)."""
    return struct.pack("<BhhHHBB", 0x15, x, y, target_tag, target_type, order, int(queued))
