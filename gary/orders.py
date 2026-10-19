"""The game's order ids that mean the same for every race (what a unit or building is doing)."""

from __future__ import annotations

ORDER_RESEARCH, ORDER_UPGRADE, ORDER_BUILD_ADDON = 75, 76, 37
ORDER_PLACE_ADDON = 0x24
GAS_ORDERS = {81, 82, 83, 84}             # moving to / waiting for / harvesting / returning gas
