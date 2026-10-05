"""Intents and leases (docs/CONTROL.md): what the deciding layers ask for, and who commands which unit.

An Intent is a decision of the fight or army layer: a kind (what to do), the units, a target, the
data its skill needs, who decided it (source) and for how long it holds its units (ttl). The motor
layer (motor.py) carries it out with a skill.

A lease is the record of which intent last ordered a unit, until its ttl runs out. A unit ordered
by another source while still leased is a conflict, traced ("lease_conflict"): how often the rules
and the models fight over the same units. Leases are recorded, not enforced yet: enforcing them
changes what Gary does, so it comes with the skills that need it (the stutter step).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# how long an intent holds its units (frames), by kind; the fight model's commands as long as a
# unit isn't told the same thing again (fight.REPEAT_FRAMES)
TTL = {"PULL": 24 * 30, "UNPULL": 24 * 3, "LOAD": 24 * 3, "REPAIR": 24 * 3, "REINFORCE": 24 * 3,
       "ARMY_MOVE": 24 * 10, "ARMY_ATTACK": 24 * 10, "SCOUT": 24 * 600, "ATTACK_WAVE": 24 * 60}
COMMAND_TTL = 48


@dataclass
class Intent:
    kind: str                           # "MOVE", "ATTACK_UNIT", "PULL", "LOAD", "ARMY_MOVE", "SCOUT"...
    units: list[int]                    # the units it is for (the skill may pick them, e.g. PULL)
    target: tuple[int, int] | int | None    # a map point or a unit tag
    params: dict = field(default_factory=dict)   # what the skill needs (the task's data)
    source: str = ""                    # who decided: "fight", "rule:pull", "army", "scout"...
    issued: int = 0                     # frame
    ttl: int = 0

    def __post_init__(self):
        if not self.ttl:
            self.ttl = TTL.get(self.kind, COMMAND_TTL)


class Leases:
    def __init__(self, g):
        self.g = g
        self.held: dict[int, tuple[str, str, int]] = {}   # unit tag -> (source, intent kind, until frame)

    def take(self, tags, intent: Intent) -> None:
        """The skill just ordered these units for this intent."""
        now = self.g.hi.frame
        lost: dict[str, int] = {}
        for t in tags:
            held = self.held.get(t)
            if held and held[2] > now and held[0] != intent.source:
                key = f"{held[0]} {held[1]}"
                lost[key] = lost.get(key, 0) + 1
            self.held[t] = (intent.source, intent.kind, now + intent.ttl)
        if lost:
            self.g.event("lease_conflict", taker=intent.source, intent=intent.kind, held_by=lost)

    def release(self, tags) -> None:
        for t in tags:
            self.held.pop(t, None)

    def holder(self, tag: int) -> str | None:
        held = self.held.get(tag)
        return held[0] if held and held[2] > self.g.hi.frame else None
