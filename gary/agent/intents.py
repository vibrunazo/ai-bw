"""Intents and leases (docs/CONTROL.md): what the deciding layers ask for, and who commands which unit.

An Intent is a decision of the fight or army layer: a kind (what to do), the units, a target, the
data its skill needs, who decided it (source) and for how long it holds its units (ttl). The motor
layer (motor.py) carries it out with a skill.

A lease is the record of which intent last ordered a unit, until its ttl runs out. A unit ordered
by another source while still leased is a conflict, traced ("lease_conflict"): how often the rules
and the models fight over the same units. A manifest can enforce leases ("leases": "enforced"):
then a decision leaves out the units a source of higher priority still holds (the rules over the
fight's decisions, those over the army's), so a marine on its way into the bunker isn't sent
elsewhere by the fight model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# how long an intent holds its units (frames), by kind; the fight model's commands as long as a
# unit isn't told the same thing again (fight.REPEAT_FRAMES)
TTL = {"PULL": 24 * 30, "UNPULL": 24 * 3, "LOAD": 24 * 3, "REPAIR": 24 * 3, "REINFORCE": 24 * 3,
       "ARMY_MOVE": 24 * 10, "ARMY_ATTACK": 24 * 10, "SCOUT": 24 * 600, "ATTACK_WAVE": 24 * 60}
COMMAND_TTL = 48
ROLE_TTL = {"PULL_BACK": 12, "FIRE": 24, "CLOSE_IN": 24, "RETREAT": 48}   # roles.py: re-decided every half second


def priority(source: str) -> int:
    """Whose lease beats whose: the rules, then the fight's decisions, then the army."""
    return 3 if source.startswith("rule:") else 2 if source in ("fight", "roles") else 1


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
            self.ttl = TTL.get(self.kind, ROLE_TTL.get(self.kind, COMMAND_TTL))


class Leases:
    def __init__(self, g, enforced: bool = False):
        self.g = g
        self.enforced = enforced
        self.held: dict[int, tuple[str, str, int]] = {}   # unit tag -> (source, intent kind, until frame)

    def free(self, tags, source: str) -> list[int]:
        """The units a decision by this source may command: all of them when leases are only
        recorded; else the ones no source of higher priority holds."""
        if not self.enforced:
            return list(tags)
        now, p = self.g.hi.frame, priority(source)
        return [t for t in tags if not ((h := self.held.get(t)) and h[2] > now and priority(h[0]) > p)]

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
