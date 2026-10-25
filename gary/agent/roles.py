"""Roles (layer L4, docs/CONTROL.md): what each of Gary's fighting units does, every half second.

A role per unit, as eval/micro_roles.py labels the pros' marines: fire, close in, pull back,
retreat (stand: nothing to do). Decided by hand rules (provider `rules/v1`) or by the role model
learned from the pros' labels (`learned/v1`, gary/policy/roles.py); each role's skill picks where
to go from the threats around (their positions and weapon ranges, the walkable ground), and the
motor layer's command skill clicks it. One role group per turn (12 units at most), the most
urgent. The hand rules:

  retreat    the zerg near Gary's group outweigh it (supply, SCVs fighting with it counted) 2.5 to
             1: to safety. Into a bunker of Gary's with room, within 15 tiles (the nearest marines,
             as many as fit); else, away from home, away from them to the edge of their weapon
             range plus a margin. At home (a Command Center within 14 tiles) there is no walking
             away: the SCVs would be left to the zerg, so the army fights with them
  pull_back  units in contact with an attacker that targets them, under half HP, with other units
             near them firing: each steps 1.5 tiles away from its attackers (a walkable direction)
  fire       units with an enemy in range that aren't attacking: attack-move at the enemy group
  close_in   units with no enemy in range, when Gary's group isn't weaker there: attack-move toward
             the nearest enemy group
  brace      (an option) the zerg coming at the group (closer than half a second before): nobody
             closes in, let them come into the fire. What pros' marines do then: they stand and
             shift into place (17% of the time sideways), attack-moving at them half as much as
             v0.10, whose close in caused 72% of its marines' attack-moves while zerglings came.
             While none is within 7 tiles yet, units ahead of the pack (2+ tiles closer to the
             threat than its median) step back to it, and units attack-moving at it stop
  leash      (an option) at home, close in only on zerg that are near Gary's buildings (5 tiles) or
             attacking its units: the pros hold their base early, they don't hunt zerglings
             hovering outside it
  volley     (an option) close in by moving (not attack-moving: in BW the front marines stop to
             shoot one by one as they come in range, and the rest trickle in behind), to the edge
             of the marines' range from the nearest zerg; nobody of the group fires until 5 of it
             (or all, if fewer) are in range or one of it is attacked, then they fire together
  anchor     (an option) at home, the army stays by its anchors (Command Centers, bunkers): a unit
             farther than ANCHOR_OUT from all of them with zerg within ANCHOR_DANGER falls back to
             the nearest (moving: pulled out, not fighting its way); and the leash's prey are only
             the zerg by Gary's buildings, not those attacking a unit out there (in Gary's losses
             to Zerg v2's all-in, 41% of its army's deaths at 5:00-8:00 were out in the open, the
             units last told to fire or close in; the rest walked out after them)
  holds      (an option) units on hold position are left there by fire and close in (they shoot
             whatever comes in range): a pro's marines holding a gap in a wall or a choke stay in it
             (scenario 0c50ac60: the pro's 3 marines hold a gap against 17 zerglings for 45 s and
             live; v0.10 attack-moved them out at 5 s and lost them all)
  kite       (an option) vultures pull back from melee zerg (zerglings, ultralisks) within 2.5
             tiles at any HP, alone or not: hit and run (fire sends them back in). v0.10 attack-moved
             them into zergling packs like marines and lost 2.5x the pros' vulture value
  fire=      (an option) how fire fires: "enemy" (attack-move at the nearest enemy, as above: after
             each kill the units walk on toward where it was, off their spot), "here" (attack-move
             to the group's own middle: they shoot what comes in range and stay), or "hold" (hold
             position: they shoot what is in range, never chase). The pros' marines die on attack
             unit 52%, hold position 9%, a move 17%; v0.10's on attack unit 74%
  post       (an option) at home, the army holds a post by the Command Center, 3 tiles from it
             toward the zerg (or: "minerals", between it and its mineral line; "bunker", just behind
             the bunker by it nearest the zerg; "ramp", just inside the main's ramp, at the main): units with nothing in range more than 3 tiles from it go back to it
             (attack-moving), and only zerg attacking something of Gary's or standing by its
             buildings are closed in on (the leash). While zerg are 7-16 tiles from their marines
             at home, the pros' stand nearer the Command Center than v0.10's (within 6 tiles 46%
             of the time, v0.10's 37%), in bigger groups (7, v0.10's 5), and the standoff lasts:
             twice the marine-minutes of v0.10's, who close the distance
  vultures   (an option) vultures skirmish as a group, apart from the other roles (selected with a
             ctrl+click: every vulture on screen, no marines): any zerg within its reach + 2 tiles of
             a vulture, they move away 3.5 tiles (toward Gary's other fighters, not into a corner,
             not into the workers: the zerg follow);
             else, weapons ready and a zerg in range, they
             shoot the nearest; else, reloading with zerg near, they step away. They never close
             in. With zerg within 8 tiles the pros' vultures move 44% of the time, attack a unit
             33%, hold 8% (patrol 1%, mines rare this early), at 2.9 tiles/s; v0.10's attack 65%,
             move 7%, at 1.3 tiles/s, and lose twice as many (home defenses)
  gap_post   (an option, with post) where Gary's frontier (gary/walls.py) has a wall gap (an
             opening under 4 tiles that zerglings get through, with one of Gary's buildings a side
             of it) and the zerg are still outside the frontier, the post is just inside the gap
             nearest the zerg (ahead of closing in on zerg by Gary's buildings: they're at the wall), and units there hold position (fire and close in leave them on
             it). The pros' marines on hold stand at such an opening 23% of the hold time
  regroup    (an option) only the pack closes in: units with nothing in range that are away from
             the pack (the biggest bunch of Gary's fighters, linked within 4 tiles; a bunker and
             SCVs fighting count) rejoin it, attack-moving to its middle, instead of closing in
             alone. The pros' marines die among 3.8 friends (within 4 tiles), v0.10's among 2.5;
             48% of v0.10's die with at most one friend near, 29% of the pros'

Units another source holds by lease (a rule loading them into a bunker...) are left out.

With a decision log open (the fight layer's rl_log), each turn is recorded: the model's inputs
(the snapshot and the units' motion) and each of Gary's units' decision (its new role, or keep).
The hand rules' log is data to distill them into a model (ingest/role_distill.py); the learned
provider then draws roles from its probabilities instead of taking the likeliest, and its log is
what reinforcement learning trains on (train/roles_rl.py).
"""

from __future__ import annotations

import math

import numpy as np

from gary import terran as T
from gary import zerg as Z
from gary.agent.intents import Intent
from gary.policy.army import is_building, is_worker, supply_x2

# flyers: units the ground army can't catch. Away from home the rules don't close in on a group of them
# alone (human game 3: "1 Marine -> close in" on 3 mutalisks, across the map, dying alone, five times);
# in range they are fired at, at home they are closed in on over the mineral line as before
FLYERS = {Z.MUTALISK, Z.SCOURGE, Z.GUARDIAN, Z.DEVOURER, Z.QUEEN, Z.OVERLORD,
          T.WRAITH, T.VALKYRIE, T.BC, T.VESSEL, T.DROPSHIP}
FOLLOW = 6 * 32                      # medics this close to a group that fires or closes in go with it

ATTACK = {10, 12, 14}
MOVE = {6}
# weapon ranges (pixels): what a unit can hit from how far, roughly, edges included
RANGE = {0: 160, 32: 48, 1: 224, 2: 192, 5: 224, 30: 224, 3: 192,             # marine, firebat, ghost, vulture, tanks, goliath
         37: 32, 38: 160, 103: 224, 39: 48, 43: 128, 44: 0, 46: 0, 47: 32, 50: 32}   # ling, hydra, lurker, ultra, muta...
NEAR = 10 * 32                       # units this close to the group's middle count for its strength
RETREAT_RATIO = 2.5
CLOSE_RATIO = 1.0                    # close in only when not weaker than this
BRACE_FROM = 7 * 32                  # brace while no enemy fighter is this close to the group
AHEAD = 2 * 32                       # a unit this much closer to the threat than the pack's median is ahead
ATTACK_MOVE = 14
CONTACT = 48
PULL_HP = 0.5
PULL_TILES = 1.5
RETREAT_MARGIN = 3 * 32
REPEAT = 24                          # the same role isn't given to a unit again within a second (but pull back)
KEY_IN_FIGHT = 0.75                  # by_key: a key this much in the fight is ordered whole
# the again decision (a model with an offset head, train/roles.py --again): the unit's order given again to where the
# head points, not within AGAIN_GAP of its last order; units whose points are within AGAIN_NEAR share one order
AGAIN_GAP = 12
AGAIN_NEAR = 3 * 32
# option refresh (the skills keep their orders current): a unit whose move role stands (retreat, pull back,
# reposition) has its move given again where the role model moves it now, when that moved past REFRESH_PX since it was
# given, REFRESH_MIN after: the pros' fight commands are 49% (Terran) and 72% (Zerg) the same order to the same units
# with the target moved, a median 1.2-1.3 tiles (quartiles 0.5 / 1.2 / 3.3; first 10 minutes, 150 modern games), and
# 75-87% of them right-clicks on the ground, where "keep" gave Gary nothing to issue. Attack-moves of units fighting
# are not given again: re-aimed at the enemy nearest the group, or along the model's move, both cost net value on the
# 239 home defenses (-67 and -66, -52 without: re-acquiring targets mid-fight)
REFRESH_PX = 32
REFRESH_MIN = 12
SHELTER = 15 * 32                    # a bunker this close is where a retreat goes
HOME = 14 * 32                       # a Command Center this close: the group is defending home
BUNKER, MARINE = 125, 0
BUNKER_SLOTS = 4
MAX_GROUP = 12
WHERE_LEAD = 4                       # learned where: a destination this many of the pros' half-second moves ahead (2 s)
WHERE_MIN = 0.25 * 32                # px per half second: a predicted move shorter than this is no move
AHEAD_FROM = 7 * 32                  # ahead/v1: a unit with no seen enemy fighter this near is before contact
LONE_FRIEND = 4 * 32                 # option lone: a unit with no own fighter this near...
LONE_ENEMY = 6 * 32                  # ...and 2+ seen enemy fighters this near is alone and outnumbered
SIDEWAYS_PX = 8                      # closer by less than this isn't coming
LEASH = 5 * 32                       # leash: zerg this close to Gary's buildings may be closed in on
ANCHOR_OUT = 10 * 32                 # anchor: a unit this far from every Command Center and bunker is out...
ANCHOR_DANGER = 7 * 32               # ...and with zerg this close it falls back
VOLLEY = 5                           # volley: this many of a closing group in range fire together
HOLD = 107                           # the hold position order
ENTER = 92                           # the enter transport (or bunker) order
LOADING_BY = 3 * 32                  # units this close to one entering a bunker are left alone
VULTURE = 2
MELEE = {37, 39}                     # zergling, ultralisk
KITE_PX = 80                         # kite: melee zerg this close to a vulture
ZERG_REACH = {37: 16, 38: 128, 39: 32, 103: 192, 146: 224, 40: 16}   # ling, hydra, ultra, lurker, sunken, broodling
VULTURE_RANGE = 160
SKIRMISH_NEAR = 8 * 32               # vultures: zerg this close to the group are its business
TOO_CLOSE = 2 * 32                   # ...within reach + this: move away
AWAY = 3.5 * 32
EASE = 3 * 32                        # reloading, zerg within reach + this: step away
POST_FROM_CC = 3 * 32               # post: this far from the Command Center toward the zerg
POST_SLACK = 3 * 32                  # post: units farther than this from it go back
BEHIND_BUNKER = 2 * 32               # post="bunker": this far past the bunker, away from the zerg
AT_GAP = 48                          # ...units this close to it hold position
PACK_LINK = 4 * 32                   # regroup: units this close to each other are one bunch


def _known(role: str) -> str:
    """A role as the role model names it (it knows no brace or regroup: they reposition)."""
    from gary.policy.roles import DECISIONS
    return role if role in DECISIONS else "reposition"


def _mid(units) -> tuple[float, float]:
    return sum(u["x"] for u in units) / len(units), sum(u["y"] for u in units) / len(units)


class RoleRules:
    """rules/v1: hand rules for roles (see the module's docstring)."""

    def __init__(self, g, brace: bool = False, leash: bool = False, volley: bool = False, regroup: bool = False,
                 anchor: bool = False,
                 holds: bool = False, kite: bool = False, fire: str = "enemy", post: float | str | None = None,
                 vultures: bool = False, gap_post: bool = False, pull_hp: float = PULL_HP, surround: int = 0,
                 refresh: bool = False):
        self.g = g
        self.refresh = refresh                       # option: orders kept current as their targets move
        self.order_at: dict[int, tuple[int, int]] = {}   # unit -> the point of its last order (refresh)
        self.anchor = anchor                         # at home, units stay by the Command Centers and bunkers
        self.pull_hp = pull_hp                       # pull back: hurt below this share of max HP
        self.surround = surround                     # ...or this many melee zerg on it (0: off): to the pack
        self.post = post                             # post: tiles from the CC toward the zerg, or "minerals"
        self.vultures = vultures
        self.gap_post = gap_post
        self._gaps_for, self._gap_posts, self._frontier = None, [], None   # gap_post: (Gary's buildings) -> posts
        self.kite, self.fire = kite, fire
        self.brace, self.leash, self.volley, self.regroup, self.holds = brace, leash or post is not None, volley, regroup, holds
        self.closing: dict[int, frozenset] = {}      # volley: unit -> its closing group
        self.told: dict[int, tuple[str, int]] = {}   # unit -> (role, frame)
        self.more: list[Intent] = []                 # this turn's further orders (option batch; fight.py's queue)
        self._walk = None
        self._ground = None                          # the role model's terrain (gary/policy/roles.py Ground)
        self._ramp = None                            # post="ramp": the spot inside the main's ramp, or False
        self.first_seen: dict[int, int] = {}        # unit -> frame the roles first saw it (its age)
        self.prev: dict[int, tuple[float, float]] = {}
        self.prev_frame = None

    def _motion(self, obs: dict, snap):
        """The units' motion over the last half second (the role model's extra inputs), keeping
        track of where everyone was; None without a snapshot."""
        from gary.policy.roles import extra_features
        now = self.g.hi.frame
        positions = {u["tag"]: (u["x"], u["y"]) for u in obs["units"]}
        prev, elapsed = self.prev, (now - self.prev_frame) if self.prev_frame is not None else 0
        self.prev, self.prev_frame = positions, now
        if snap is None:
            return None
        rows, tags = snap[0], snap[1]
        if 0 < elapsed != 12:                  # motion per half second, as the model learned it
            prev = {t: (positions[t][0] - (positions[t][0] - p[0]) * 12 / elapsed,
                        positions[t][1] - (positions[t][1] - p[1]) * 12 / elapsed)
                    for t, p in prev.items() if t in positions}
        extra = extra_features(rows, tags, positions, prev if elapsed else {}, self.g.flip)
        self._terrain_extra(extra, tags, obs, positions)
        return self._own_extra(extra, tags, snap[3], now)

    def _terrain_extra(self, extra, tags: list[int], obs: dict, positions: dict) -> None:
        """The role model's terrain extras (gary/policy/roles.py terrain_features): the ground under
        each unit, and the way to Gary's Command Centers, bunkers and main ramp."""
        from gary.policy.roles import Ground, terrain_features
        g = self.g
        if self._ground is None:
            self._ground = False
            if getattr(g.hi.game, "terrain", None) is not None and g.map is not None and g.main is not None:
                try:
                    start = min(g.map.starts, key=lambda s: math.dist(s, g.main.center))
                    self._ground = Ground.of(g.hi.game, start)
                except Exception:                 # (a map the analysis can't read: no terrain inputs)
                    self._ground = False
        own = [u for u in obs["units"] if u["owner"] == g.slot]
        extra[:, 6:] = terrain_features(tags, positions, self._ground or None,
                                        [(u["x"], u["y"]) for u in own if u["type"] in g.race.town_halls],
                                        [(u["x"], u["y"]) for u in own if u["type"] == T.BUNKER and u["completed"]],
                                        g.flip)

    def _own_extra(self, extra, tags: list[int], by_tag: dict, now: int):
        """The role model's per-unit extras 3-5 for Gary's fighters: age, distance to its biggest
        bunch of fighters, that bunch's size (fresh units walking into a fight alone died alone)."""
        from gary.policy.roles import AGE_SCALE, PACK_DIST_SCALE, PACK_SIZE_SCALE
        g = self.g
        mine = [by_tag[t] for t in tags if t in by_tag and by_tag[t]["owner"] == g.slot and supply_x2(by_tag[t]["type"]) > 0
                and not is_worker(by_tag[t]["type"]) and not is_building(by_tag[t]["type"]) and by_tag[t]["completed"]]
        for u in mine:
            self.first_seen.setdefault(u["tag"], now)
        if not mine or not getattr(self, "own_extra", True):
            return extra                         # (option dataextras: 0, as in the pros' data the models learned from)
        pack = self._pack(mine, [], by_tag)
        size = sum(1 for u in mine if u["tag"] in pack["tags"])
        index = {t: i for i, t in enumerate(tags)}
        for u in mine:
            i = index[u["tag"]]
            if i < len(extra):
                extra[i, 3] = (now - self.first_seen[u["tag"]]) / 24 / AGE_SCALE
                extra[i, 4] = math.dist((u["x"], u["y"]), pack["mid"]) / PACK_DIST_SCALE
                extra[i, 5] = size / PACK_SIZE_SCALE
        return extra

    def _units(self, snap):
        """Gary's units the roles may command (in the snapshot, not held by a rule), and the enemy's."""
        from gary.policy.roles import MAX_UNITS
        g = self.g
        rows, tags, _, by_tag = snap
        free = set(g.leases.free(tags, "roles"))
        mine = [by_tag[t] for i, t in enumerate(tags[:MAX_UNITS]) if rows[i][1] == 1 and t in free
                and supply_x2(by_tag[t]["type"]) > 0 and not is_worker(by_tag[t]["type"])
                and not is_building(by_tag[t]["type"]) and by_tag[t]["completed"]]
        enemy = [u for u in by_tag.values() if u["owner"] not in (g.slot, 11) and supply_x2(u["type"]) > 0
                 and not is_worker(u["type"]) and not is_building(u["type"])]
        return mine, enemy

    def _log(self, obs: dict, snap, extra, mine: list[dict], decided: dict[int, str], intent) -> None:
        """One turn in the decision log: inputs, each unit's decision (a role, or keep)."""
        from gary.policy.roles import DECISIONS
        index = {t: i for i, t in enumerate(snap[1])}
        units = [index[u["tag"]] for u in mine]
        self.g.fight.rl_log.append({"rows": snap[0], "extra": extra, "time": obs["frame"] * 42 / 1000, "units": units,
                                    "roles": [DECISIONS.index(_known(decided.get(u["tag"], "keep"))) for u in mine],
                                    "executed": intent is not None})

    def decide(self, obs: dict, snap) -> Intent | None:
        g = self.g
        logging = g.fight.rl_log is not None
        self.before = self.prev                # where everyone was last turn (brace: who's coming)
        extra = self._motion(obs, snap) if logging or self.brace else None
        if snap is None:
            return None
        if not logging:                        # (as before the log existed)
            by_tag = snap[3]
            mine = [u for u in by_tag.values() if u["owner"] == g.slot and u["completed"] and supply_x2(u["type"]) > 0
                    and not is_worker(u["type"]) and not is_building(u["type"])]
            enemy = [u for u in by_tag.values() if u["owner"] not in (g.slot, 11) and supply_x2(u["type"]) > 0
                     and not is_worker(u["type"]) and not is_building(u["type"])]
            mine = [u for u in mine if u["tag"] in set(g.leases.free([u["tag"] for u in mine], "roles"))]
            return self._rules(snap[3], mine, enemy)
        mine, enemy = self._units(snap)
        intent = self._rules(snap[3], mine, enemy)
        if mine and enemy:
            decided = {t: intent.kind.lower() for t in intent.units} if intent is not None else {}
            self._log(obs, snap, extra, mine, decided, intent)
        return intent

    def _rules(self, by_tag: dict, mine: list[dict], enemy: list[dict]) -> Intent | None:
        g = self.g
        # units entering a bunker, and those by them, are left alone (a box around a group by them
        # would take them too, and its order would call the load off; motor.py checks its boxes too)
        entering = [(u["x"], u["y"]) for u in by_tag.values() if u["owner"] == g.slot and u.get("order") == ENTER]
        mine = [u for u in mine if not any(math.dist((u["x"], u["y"]), p) < LOADING_BY for p in entering)]
        if not mine or not enemy:
            return None
        vultures = []
        if self.vultures:
            vultures = [u for u in mine if u["type"] == VULTURE]
            mine = [u for u in mine if u["type"] != VULTURE]
            self.friends = _mid(mine) if mine else None
            self.scvs = [(u["x"], u["y"]) for u in by_tag.values() if u["owner"] == g.slot and u["type"] == g.race.worker]
            if not mine:
                return self._skirmish(vultures, enemy, urgent=False)
        now = g.hi.frame
        mx, my = _mid(mine)
        workers = [u for u in by_tag.values() if u["owner"] == g.slot and u["type"] == g.race.worker
                   and (u.get("order") in ATTACK or u["tag"] in g.pulled)]
        ours = sum(supply_x2(u["type"]) for u in mine + workers if math.dist((u["x"], u["y"]), (mx, my)) < NEAR)
        home = any(u["owner"] == g.slot and u["type"] in g.race.town_halls and math.dist((u["x"], u["y"]), (mx, my)) < HOME
                   for u in by_tag.values())
        theirs = sum(supply_x2(u["type"]) for u in enemy if math.dist((u["x"], u["y"]), (mx, my)) < NEAR)
        fresh = lambda u, role: not (self.told.get(u["tag"], ("", -10**9))[0] == role
                                     and now - self.told[u["tag"]][1] < REPEAT)
        in_range = lambda u: any(math.dist((u["x"], u["y"]), (e["x"], e["y"])) <= RANGE.get(u["type"], 160) + 32
                                 for e in enemy)

        # anchor: at home, units out in the open with zerg near fall back to the nearest anchor
        anchors = [(u["x"], u["y"]) for u in by_tag.values() if u["owner"] == g.slot
                   and (u["type"] in g.race.town_halls or u["type"] == T.BUNKER) and u["completed"]] if self.anchor and home else []
        if anchors:
            out = [u for u in mine if min(math.dist((u["x"], u["y"]), a) for a in anchors) > ANCHOR_OUT
                   and any(math.dist((u["x"], u["y"]), (e["x"], e["y"])) < ANCHOR_DANGER for e in enemy)
                   and fresh(u, "RETREAT")]
            if out:
                group = out[:MAX_GROUP]
                at = min(anchors, key=lambda a: math.dist(a, _mid(group)))
                return self._intent("RETREAT", group, (int(at[0]), int(at[1])), "move")
        # retreat: the group is outweighed; into a bunker if one is near with room, else away
        if theirs > RETREAT_RATIO * max(ours, 1):
            group = [u for u in mine if fresh(u, "RETREAT")]
            shelter = self._shelter(by_tag, (mx, my))
            if shelter is not None and group:
                bunker, room = shelter
                inside = [u for u in group if u["type"] == MARINE]
                inside = sorted(inside, key=lambda u: math.dist((u["x"], u["y"]), (bunker["x"], bunker["y"])))[:room]
                if inside:
                    return self._enter(inside, bunker)
            group = group[:MAX_GROUP]
            if group and not home:
                return self._intent("RETREAT", group, self._retreat_point(group, enemy), "move")
        # brace: the zerg coming
        coming = False
        if self.brace:
            braced = self._brace(mine, enemy, fresh)
            if braced is not False:
                return braced
            coming = self.coming
        # pull back: units under attack, hurt, with others firing near them
        targeted = {e.get("order_target") for e in enemy}
        firing = [u for u in mine if u.get("order") in ATTACK]
        pull = [u for u in mine if u["tag"] in targeted and u.get("hp_max") and u["hp"] < self.pull_hp * u["hp_max"]
                and any(math.dist((u["x"], u["y"]), (e["x"], e["y"])) < CONTACT for e in enemy)
                and any(f["tag"] != u["tag"] and math.dist((u["x"], u["y"]), (f["x"], f["y"])) < 4 * 32 for f in firing)]
        if self.surround:
            swarmed = self._swarmed(mine, enemy, targeted, pull, now)
            if swarmed is not None:
                return swarmed
        if self.kite:
            pull += [u for u in mine if u["type"] == VULTURE and u not in pull and any(
                e["type"] in MELEE and math.dist((u["x"], u["y"]), (e["x"], e["y"])) < KITE_PX for e in enemy)]
        if pull:
            group = pull[:MAX_GROUP]
            attackers = ([e for e in enemy if e.get("order_target") in {u["tag"] for u in group}]
                         or [e for e in enemy if any(math.dist((u["x"], u["y"]), (e["x"], e["y"])) < 128 for u in group)]
                         or enemy)
            point = self._step_away(_mid(group), _mid(attackers), PULL_TILES * 32)
            if point:
                return self._intent("PULL_BACK", group, point, "move")
        # fire: an enemy in range, not attacking
        held = {HOLD} if self.holds or self.gap_post else set()
        firing = held | ({HOLD} if self.fire == "hold" else set())
        # (medics don't fire or close in on their own: human game 3 had "1 Medic -> fire" at mutalisks, dying
        # alone; they go with the group that does, and retreat or pull back with everyone)
        medics = [u for u in mine if u["type"] == T.MEDIC]
        idle = [u for u in mine if u["type"] != T.MEDIC and u.get("order") not in ATTACK | firing and in_range(u)
                and fresh(u, "FIRE")]
        if self.volley:
            idle = self._volley_ready(idle, mine, enemy, in_range, targeted)
        if idle:
            group = sorted(idle, key=lambda u: min(math.dist((u["x"], u["y"]), (e["x"], e["y"])) for e in enemy))[:MAX_GROUP]
            if self.fire == "hold":
                for u in group:
                    self.told[u["tag"]] = ("FIRE", now)
                tags = [u["tag"] for u in group]
                return Intent("FIRE", tags, None, {"key": ("hold",), "tags": tags, "center": _mid(group)}, "roles")
            group = self._with_medics(group, medics, fresh)
            if self.fire == "here":
                mid = _mid(group)
                return self._intent("FIRE", group, (int(mid[0]), int(mid[1])), "attack_move")
            target = min(enemy, key=lambda e: math.dist((e["x"], e["y"]), _mid(group)))
            return self._intent("FIRE", group, (int(target["x"]), int(target["y"])), "attack_move")
        # vultures (after the others' fire: one group a turn): away from zerg too close
        if vultures:
            run = self._skirmish(vultures, enemy, urgent=True)
            if run is not None:
                return run
        # vultures: shoot, or ease off while reloading
        if vultures:
            shot = self._skirmish(vultures, enemy, urgent=False)
            if shot is not None:
                return shot
        # close in: nothing in range, not weaker, and they aren't coming at us
        if ours >= CLOSE_RATIO * theirs and not coming:
            prey = enemy
            if self.leash and home:
                mine_tags = {u["tag"] for u in by_tag.values() if u["owner"] == g.slot}
                built = [u for u in by_tag.values() if u["owner"] == g.slot and is_building(u["type"])]
                if anchors:                              # (not the zerg on a unit out in the open)
                    mine_tags = {u["tag"] for u in by_tag.values() if u["owner"] == g.slot
                                 and min(math.dist((u["x"], u["y"]), a) for a in anchors) <= ANCHOR_OUT}
                prey = [e for e in enemy if e.get("order_target") in mine_tags
                        or any(math.dist((e["x"], e["y"]), (b["x"], b["y"])) < LEASH for b in built)]
            if prey and not home and all(e["type"] in FLYERS for e in prey):
                prey = []                                 # (flyers out in the open: they pick the fight)
            far = [u for u in mine if u["type"] != T.MEDIC and not in_range(u) and u.get("order") not in ATTACK | held
                   and fresh(u, "CLOSE_IN")]
            if self.gap_post and home:                    # (zerg at the wall, outside it: hold the door)
                gap = self._gap(enemy)
                if gap is not None:
                    away = [u for u in mine if not in_range(u) and u.get("order") not in ATTACK | held
                            and math.dist((u["x"], u["y"]), gap) > AT_GAP and fresh(u, "POST")]
                    if away:
                        group = sorted(away, key=lambda u: math.dist((u["x"], u["y"]), gap))[:MAX_GROUP]
                        return self._intent("POST", group, (int(gap[0]), int(gap[1])), "attack_move")
                    there = [u for u in mine if u.get("order") not in ATTACK | held
                             and math.dist((u["x"], u["y"]), gap) <= AT_GAP and fresh(u, "POST")]
                    if there:
                        for u in there:
                            self.told[u["tag"]] = ("POST", now)
                        tags = [u["tag"] for u in there][:MAX_GROUP]
                        return Intent("POST", tags, None, {"key": ("hold",), "tags": tags, "center": gap}, "roles")
                    if mine and all(math.dist((u["x"], u["y"]), gap) <= AT_GAP or in_range(u) for u in mine):
                        return None
            if self.post is not None and home and not prey:
                post = self._post(by_tag, (mx, my), enemy)
                away = [u for u in mine if not in_range(u) and fresh(u, "POST") and u.get("order") not in ATTACK
                        and math.dist((u["x"], u["y"]), post) > POST_SLACK]
                if post and away:
                    group = sorted(away, key=lambda u: math.dist((u["x"], u["y"]), post))[:MAX_GROUP]
                    return self._intent("POST", group, (int(post[0]), int(post[1])), "attack_move")
            if far and prey and self.regroup:
                pack = self._pack(mine, workers, by_tag)
                loose = [u for u in far if u["tag"] not in pack["tags"] and fresh(u, "REGROUP")]
                if loose:
                    group = sorted(loose, key=lambda u: math.dist((u["x"], u["y"]), pack["mid"]))[:MAX_GROUP]
                    return self._intent("REGROUP", group, (int(pack["mid"][0]), int(pack["mid"][1])), "attack_move")
                far = [u for u in far if u["tag"] in pack["tags"]]
            if far and prey:
                group = far[:MAX_GROUP]
                target = min(prey, key=lambda e: math.dist((e["x"], e["y"]), _mid(group)))
                if not self.volley:
                    return self._intent("CLOSE_IN", self._with_medics(group, medics, fresh),
                                        (int(target["x"]), int(target["y"])), "attack_move")
                point = self._edge_of_range(group, target)
                intent = self._intent("CLOSE_IN", group, point, "move")
                if intent is not None:
                    members = frozenset(u["tag"] for u in group)
                    for t in members:
                        self.closing[t] = members
                return intent
        return None

    def _swarmed(self, mine: list[dict], enemy: list[dict], targeted: set, pull: list[dict], now: int):
        """Units with self.surround or more melee zerg on them walk to the rest of the pack (the
        zerg follow into its fire), else straight away from them."""
        melee = [e for e in enemy if e["type"] in MELEE]
        swarmed = [u for u in mine if u["tag"] in targeted and u not in pull and u["type"] != VULTURE
                   and sum(1 for e in melee if math.dist((u["x"], u["y"]), (e["x"], e["y"])) < CONTACT) >= self.surround
                   and not (self.told.get(u["tag"], ("", -10**9))[0] == "PULL_BACK" and now - self.told[u["tag"]][1] < REPEAT // 2)]
        if not swarmed:
            return None
        group = swarmed[:MAX_GROUP]
        tags = {u["tag"] for u in group}
        pack = [u for u in mine if u["tag"] not in tags
                and all(math.dist((u["x"], u["y"]), (e["x"], e["y"])) >= CONTACT for e in melee)]
        mid = _mid(group)
        if pack:
            friend = min(pack, key=lambda u: math.dist((u["x"], u["y"]), mid))
            if math.dist((friend["x"], friend["y"]), mid) > CONTACT:
                return self._intent("PULL_BACK", group, (int(friend["x"]), int(friend["y"])), "move")
        attackers = [e for e in melee if math.dist(mid, (e["x"], e["y"])) < 2 * CONTACT] or melee
        point = self._step_away(mid, _mid(attackers), PULL_TILES * 32)
        return self._intent("PULL_BACK", group, point, "move") if point else None

    def _escape(self, frm, threats: list[dict], dist: float, friends):
        """vultures: a walkable point dist away (12 directions) far from the threats and nearer
        Gary's other units (friends: a point, or None); None if every way is blocked."""
        best = None
        for k in range(12):
            a = k * math.pi / 6
            x, y = frm[0] + math.cos(a) * dist, frm[1] + math.sin(a) * dist
            if not all(self._walkable(frm[0] + (x - frm[0]) * f, frm[1] + (y - frm[1]) * f) for f in (0.33, 0.66, 1.0)):
                continue
            score = min(math.dist((x, y), (e["x"], e["y"])) for e in threats)
            if friends is not None:
                score -= 0.5 * math.dist((x, y), friends)
            if any(math.dist((x, y), s) < 5 * 32 for s in getattr(self, "scvs", ())):
                score -= 8 * 32                              # not into the workers: the zerg follow

            if best is None or score > best[0]:
                best = (score, (x, y))
        return best and best[1]

    def _skirmish(self, vultures: list[dict], enemy: list[dict], urgent: bool) -> Intent | None:
        """vultures: move away from zerg too close (urgent), else shoot or ease off (module doc)."""
        mid = _mid(vultures)
        near = [e for e in enemy if math.dist((e["x"], e["y"]), mid) < SKIRMISH_NEAR]
        if not near:
            return None
        gap = lambda e: min(math.dist((u["x"], u["y"]), (e["x"], e["y"])) for u in vultures) - ZERG_REACH.get(e["type"], 32)
        close = [e for e in near if gap(e) < TOO_CLOSE]
        now = self.g.hi.frame
        tags = [u["tag"] for u in vultures][:MAX_GROUP]
        params = lambda key: {"key": key, "tags": tags, "center": mid, "ctrl_click": True}
        if urgent:
            if not close:
                return None
            point = self._escape(mid, close, AWAY, self.friends)
            if point is None:
                return None
            for t in tags:
                self.told[t] = ("SKIRMISH", now)
            return Intent("SKIRMISH", tags, point, params(("move", int(point[0]), int(point[1]))), "roles")
        ready = [u for u in vultures if not u.get("cooldown")]
        target = min(near, key=lambda e: math.dist((e["x"], e["y"]), mid))
        if ready and math.dist((target["x"], target["y"]), mid) <= VULTURE_RANGE + 16 \
                and not all(u.get("order") == 10 and u.get("order_target") == target["tag"] for u in vultures):
            for t in tags:
                self.told[t] = ("SKIRMISH", now)
            return Intent("SKIRMISH", tags, target["tag"], params(("attack_unit", target["tag"])), "roles")
        easing = [e for e in near if gap(e) < EASE]
        if not ready and easing and any(u.get("order") != 6 for u in vultures):
            point = self._escape(mid, easing, AWAY / 2, self.friends)
            if point is not None:
                for t in tags:
                    self.told[t] = ("SKIRMISH", now)
                return Intent("SKIRMISH", tags, point, params(("move", int(point[0]), int(point[1]))), "roles")
        return None

    def _gap(self, enemy: list[dict]):
        """gap_post: the post inside the wall gap nearest the zerg (map pixels), or None (no wall gap,
        or the zerg are already inside the frontier)."""
        from gary.walls import frontier, outside, wall_gaps
        g = self.g
        obs = g.hi.observe()
        built = frozenset((u["tag"], u["completed"]) for u in obs["units"] if u["owner"] == g.slot and is_building(u["type"]))
        if built != self._gaps_for:
            fr = g.simcity._frontier_now(obs) if g.simcity is not None else None
            if fr is None:
                ccs = [(u["x"], u["y"]) for u in obs["units"] if u["owner"] == g.slot and u["type"] in g.race.town_halls]
                fr = frontier(g.hi.game, obs, ccs, list(g.enemy_starts)) if ccs and g.enemy_starts else None
            self._gaps_for, self._frontier = built, fr
            self._gap_posts = [w["post"] for w in wall_gaps(g.hi.game, obs, g.slot, fr)] if fr and "inside" in fr else []
        if not self._gap_posts:
            return None
        them = _mid(enemy)
        if not outside(self._frontier, them):
            return None                                   # inside already: past the wall's job
        return min(self._gap_posts, key=lambda p: math.dist(p, them))

    def _post(self, by_tag: dict, at, enemy: list[dict]):
        """post: by the Command Center nearest the group, POST_FROM_CC toward the zerg (walkable)."""
        ccs = [u for u in by_tag.values() if u["owner"] == self.g.slot and u["type"] in self.g.race.town_halls]
        if not ccs:
            return None
        cc = min(ccs, key=lambda u: math.dist((u["x"], u["y"]), at))
        if self.post == "bunker":                    # behind the bunker by that CC nearest the zerg
            spot = self._behind_bunker(by_tag, cc, enemy)
            if spot is not None:
                return spot
        if self.post == "ramp":                      # inside the main's ramp, when the fight is at the main
            spot = self._ramp_post(cc)
            if spot is not None:
                return spot
        if self.post == "minerals":                  # between the CC and its mineral line
            fields = [u for u in by_tag.values() if u["owner"] == 11 and u["type"] in (176, 177, 178)
                      and math.dist((u["x"], u["y"]), (cc["x"], cc["y"])) < 10 * 32]
            if fields:
                mx, my = _mid(fields)
                return (cc["x"] + mx) / 2, (cc["y"] + my) / 2
        zx, zy = _mid(enemy)
        d = math.dist((cc["x"], cc["y"]), (zx, zy)) or 1
        for f in (1.0, 0.75, 1.25, 0.5):
            tiles = self.post if isinstance(self.post, (int, float)) and self.post is not True else POST_FROM_CC / 32
            x = cc["x"] + (zx - cc["x"]) / d * tiles * 32 * f
            y = cc["y"] + (zy - cc["y"]) / d * tiles * 32 * f
            if self._walkable(x, y):
                return x, y
        return cc["x"], cc["y"]

    def _behind_bunker(self, by_tag: dict, cc: dict, enemy: list[dict]):
        """post="bunker": BEHIND_BUNKER past the bunker by this CC nearest the zerg, on the side away
        from them (where its marines fight beside it, out of the zerg's way); None without one."""
        zx, zy = _mid(enemy)
        bunkers = [u for u in by_tag.values() if u["owner"] == self.g.slot and u["type"] == BUNKER and u["completed"]
                   and math.dist((u["x"], u["y"]), (cc["x"], cc["y"])) < HOME]
        if not bunkers:
            return None
        b = min(bunkers, key=lambda u: math.dist((u["x"], u["y"]), (zx, zy)))
        d = math.dist((b["x"], b["y"]), (zx, zy)) or 1
        for f in (1.0, 0.5, 1.5):
            x = b["x"] + (b["x"] - zx) / d * BEHIND_BUNKER * f
            y = b["y"] + (b["y"] - zy) / d * BEHIND_BUNKER * f
            if self._walkable(x, y):
                return x, y
        return None

    def _ramp_post(self, cc: dict):
        """post="ramp": just inside the main's ramp (gary/terrain.py ramp_inside), when the CC is the
        main's and the map analysis is reliable; else None."""
        g = self.g
        if g.main is None or math.dist((cc["x"], cc["y"]), g.main.center) > 6 * 32:
            return None
        if self._ramp is None:
            self._ramp = False
            try:
                from gary.agent.block import _analysis
                a = _analysis(g)
                if a["reliable"]:
                    self._ramp = tuple(a["ramp_inside"])
            except Exception:                         # (no terrain in this game)
                pass
        return self._ramp or None

    def _pack(self, mine: list[dict], workers: list[dict], by_tag: dict) -> dict:
        """regroup: the biggest bunch (by supply) of Gary's fighters, SCVs fighting and bunkers,
        linked within PACK_LINK: {"tags", "mid"} (the middle of its fighters)."""
        bunkers = [u for u in by_tag.values() if u["owner"] == self.g.slot and u["type"] == BUNKER and u["completed"]]
        units = mine + workers + bunkers
        weight = lambda u: 8 if u["type"] == BUNKER else supply_x2(u["type"])
        left, best = set(range(len(units))), None
        while left:
            todo, bunch = [left.pop()], []
            while todo:
                i = todo.pop()
                bunch.append(units[i])
                near = [j for j in left if math.dist((units[i]["x"], units[i]["y"]), (units[j]["x"], units[j]["y"])) < PACK_LINK]
                left -= set(near)
                todo += near
            if best is None or sum(map(weight, bunch)) > sum(map(weight, best)):
                best = bunch
        fighters = [u for u in best if u in mine] or best
        return {"tags": {u["tag"] for u in best}, "mid": _mid(fighters)}

    def _volley_ready(self, idle: list[dict], mine: list[dict], enemy: list[dict], in_range, targeted) -> list[dict]:
        """volley: of the units that could fire, those of a closing group wait for the rest of it."""
        by = {u["tag"]: u for u in mine}
        ready = []
        for u in idle:
            members = self.closing.get(u["tag"])
            if members is None:
                ready.append(u)
                continue
            moving = [by[t] for t in members if t in by and by[t].get("order") in MOVE]
            if (u["tag"] in targeted or u.get("order") not in MOVE
                    or sum(in_range(m) for m in moving) >= min(VOLLEY, len(moving))
                    or any(m["tag"] in targeted for m in moving)):
                for t in members:                  # the group fires: it closes in no more
                    self.closing.pop(t, None)
                ready.append(u)
        return ready

    def _edge_of_range(self, group: list[dict], target: dict):
        """volley: where the group closes in to, just outside its range from the target (standing
        there, a unit fires at whatever comes in range; moving, it wouldn't)."""
        mid = _mid(group)
        reach = min(RANGE.get(u["type"], 160) for u in group)
        d = math.dist(mid, (target["x"], target["y"])) or 1
        if d <= reach:
            return int(target["x"]), int(target["y"])
        f = (reach + 24) / d
        return int(target["x"] + (mid[0] - target["x"]) * f), int(target["y"] + (mid[1] - target["y"]) * f)

    @staticmethod
    def _with_medics(group: list[dict], medics: list[dict], fresh) -> list[dict]:
        """The group plus the medics near it (an attack-move keeps them healing along), within MAX_GROUP."""
        if not medics or len(group) >= MAX_GROUP:
            return group
        mid = _mid(group)
        near = sorted((m for m in medics if math.dist((m["x"], m["y"]), mid) <= FOLLOW and fresh(m, "FIRE")),
                      key=lambda m: math.dist((m["x"], m["y"]), mid))
        return group + near[:MAX_GROUP - len(group)]

    def _intent(self, role: str, group: list[dict], point, order: str) -> Intent | None:
        if point is None:
            return None
        now = self.g.hi.frame
        for u in group:
            self.told[u["tag"]] = (role, now)
            self.order_at[u["tag"]] = (int(point[0]), int(point[1]))
        tags = [u["tag"] for u in group]
        key = (order, int(point[0]), int(point[1]))
        return Intent(role, tags, (key[1], key[2]), {"key": key, "tags": tags, "center": _mid(group)}, "roles")

    def _act(self, roles: dict[str, list[dict]], mine, enemy, by_tag, home: bool, fresh) -> Intent | None:
        """The most urgent role group as an intent: retreat, pull back, fire, close in."""
        group = [u for u in roles.get("RETREAT", []) if fresh(u, "RETREAT")]
        if group:
            shelter = self._shelter(by_tag, _mid(group))
            if shelter is not None:
                bunker, room = shelter
                inside = sorted((u for u in group if u["type"] == MARINE),
                                key=lambda u: math.dist((u["x"], u["y"]), (bunker["x"], bunker["y"])))[:room]
                if inside:
                    return self._enter(inside, bunker)
            if not home:
                return self._intent("RETREAT", group[:MAX_GROUP], self._retreat_point(group[:MAX_GROUP], enemy), "move")
        group = roles.get("PULL_BACK", [])[:MAX_GROUP]
        if group:
            targeted = [e for e in enemy if e.get("order_target") in {u["tag"] for u in group}]
            near = targeted or sorted(enemy, key=lambda e: math.dist((e["x"], e["y"]), _mid(group)))[:3]
            point = self._step_away(_mid(group), _mid(near), PULL_TILES * 32)
            if point:
                return self._intent("PULL_BACK", group, point, "move")
        group = [u for u in roles.get("FIRE", []) if u.get("order") not in ATTACK and fresh(u, "FIRE")]
        if group:
            group = sorted(group, key=lambda u: min(math.dist((u["x"], u["y"]), (e["x"], e["y"])) for e in enemy))[:MAX_GROUP]
            target = min(enemy, key=lambda e: math.dist((e["x"], e["y"]), _mid(group)))
            return self._intent("FIRE", group, (int(target["x"]), int(target["y"])), "attack_move")
        group = [u for u in roles.get("CLOSE_IN", []) if u.get("order") not in ATTACK and fresh(u, "CLOSE_IN")][:MAX_GROUP]
        if group:
            target = min(enemy, key=lambda e: math.dist((e["x"], e["y"]), _mid(group)))
            return self._intent("CLOSE_IN", group, (int(target["x"]), int(target["y"])), "attack_move")
        intent = self._join_or_hold(roles, mine, by_tag, fresh)
        if intent is not None:
            return intent
        group = [u for u in roles.get("POST", []) if u.get("order") not in ATTACK and fresh(u, "POST")]
        if group and home and self.post is not None:
            post = self._post(by_tag, _mid(mine), enemy)
            group = [u for u in group if post and math.dist((u["x"], u["y"]), post) > POST_SLACK][:MAX_GROUP]
            if group:
                return self._intent("POST", group, (int(post[0]), int(post[1])), "attack_move")
        return None

    def _stale(self, u: dict, role: str, point) -> bool:
        """refresh: the unit's last order was this role's, REFRESH_MIN ago or more, to a point REFRESH_PX or more from
        where the role's target is now."""
        told = self.told.get(u["tag"])
        at = self.order_at.get(u["tag"])
        return bool(told and told[0] == role and at and self.g.hi.frame - told[1] >= REFRESH_MIN
                    and math.dist(at, point) >= REFRESH_PX)


    def _join_or_hold(self, roles: dict, mine: list[dict], by_tag: dict, fresh) -> Intent | None:
        """join: to Gary's biggest bunch of fighters (attack-moving: shooting on the way), from farther
        than PACK_LINK; overrides an attack-move (the reinforce rule's, straight at the zerg). hold:
        hold position."""
        group = [u for u in roles.get("JOIN", []) if fresh(u, "REGROUP")]
        if group:
            pack = self._pack(mine, [], by_tag)
            group = [u for u in group if math.dist((u["x"], u["y"]), pack["mid"]) > PACK_LINK][:MAX_GROUP]
            if group:
                return self._intent("REGROUP", group, (int(pack["mid"][0]), int(pack["mid"][1])), "attack_move")
        group = [u for u in roles.get("HOLD", []) if u.get("order") != HOLD and fresh(u, "HOLD")][:MAX_GROUP]
        if group:
            now = self.g.hi.frame
            for u in group:
                self.told[u["tag"]] = ("HOLD", now)
            tags = [u["tag"] for u in group]
            return Intent("HOLD", tags, None, {"key": ("hold",), "tags": tags, "center": _mid(group)}, "roles")
        return None

    def _brace(self, mine: list[dict], enemy: list[dict], fresh):
        """While the zerg approach: the intent (step back to the pack, or stop charging), None (stand,
        let them come), or False (not approaching: the other rules decide)."""
        mid = _mid(mine)
        before = getattr(self, "before", {})
        coming = [e for e in enemy if e["tag"] in before
                  and math.dist((e["x"], e["y"]), mid) < math.dist(before[e["tag"]], mid) - SIDEWAYS_PX]
        self.coming = bool(coming)
        if not coming or min(math.dist((u["x"], u["y"]), (e["x"], e["y"])) for u in mine for e in enemy) < BRACE_FROM:
            return False
        tc = _mid(coming)
        dist = {u["tag"]: math.dist((u["x"], u["y"]), tc) for u in mine}
        median = sorted(dist.values())[len(dist) // 2]
        ahead = [u for u in mine if dist[u["tag"]] < median - AHEAD and fresh(u, "BRACE")]
        if ahead:
            back = self._step_away(mid, tc, 32) or mid
            return self._intent("BRACE", ahead[:MAX_GROUP], back, "move")
        charging = [u for u in mine if u.get("order") == ATTACK_MOVE and fresh(u, "BRACE")]
        if charging:
            now = self.g.hi.frame
            for u in charging[:MAX_GROUP]:
                self.told[u["tag"]] = ("BRACE", now)
            tags = [u["tag"] for u in charging[:MAX_GROUP]]
            return Intent("BRACE", tags, None, {"key": ("stop",), "tags": tags, "center": _mid(charging)}, "roles")
        return None

    def _shelter(self, by_tag: dict, at):
        """A finished bunker of Gary's within SHELTER with room: (bunker, free slots), or None."""
        g = self.g
        tags = set(by_tag)
        for b in sorted((u for u in by_tag.values() if u["owner"] == g.slot and u["type"] == BUNKER and u["completed"]
                         and math.dist((u["x"], u["y"]), at) < SHELTER),
                        key=lambda u: math.dist((u["x"], u["y"]), at)):
            sent = {t for t in g.loaded.get(b["tag"], set()) if t not in tags}   # inside (unseen) or gone
            room = BUNKER_SLOTS - len(sent)
            if room > 0:
                return b, room
        return None

    def _enter(self, group: list[dict], bunker: dict) -> Intent:
        now = self.g.hi.frame
        for u in group:
            self.told[u["tag"]] = ("RETREAT", now)
        tags = [u["tag"] for u in group]
        key = ("own_unit", bunker["tag"])
        return Intent("RETREAT", tags, bunker["tag"], {"key": key, "tags": tags, "center": _mid(group),
                                                       "bunker": bunker["tag"]}, "roles")

    # --- where to go -----------------------------------------------------------------------------

    def _walkable(self, x: float, y: float) -> bool:
        if self._walk is None:
            terrain = getattr(self.g.hi.game, "terrain", None)
            self._walk = terrain()[0] if terrain is not None else False
        if self._walk is False:
            return True
        h, w = self._walk.shape
        mx, my = int(x // 8), int(y // 8)
        return 0 <= mx < w and 0 <= my < h and bool(self._walk[my, mx])

    def _step_away(self, frm, threat, dist: float):
        """A walkable point dist away from the threat, straight back or turned up to 60 degrees."""
        dx, dy = frm[0] - threat[0], frm[1] - threat[1]
        n = math.hypot(dx, dy) or 1
        base = math.atan2(dy / n, dx / n)
        for turn in (0, 0.5, -0.5, 1.0, -1.0):
            a = base + turn
            x, y = frm[0] + math.cos(a) * dist, frm[1] + math.sin(a) * dist
            if self._walkable(x, y):
                return x, y
        return None

    def _retreat_point(self, group: list[dict], enemy: list[dict]):
        """Away from the enemy group, to the edge of their reach plus a margin."""
        mid, them = _mid(group), _mid(enemy)
        reach = max(RANGE.get(e["type"], 32) for e in enemy)
        nearest = min(math.dist(mid, (e["x"], e["y"])) for e in enemy)
        dist = max(4 * 32, reach + RETREAT_MARGIN - nearest + 2 * 32)
        return self._step_away(mid, them, dist)


class RoleLearned(RoleRules):
    """learned/v1: the role model (gary/policy/roles.py) names each unit's role (its likeliest);
    the same skills as the hand rules carry the groups out. Reposition, stand and other: no command;
    nor "keep", for a model of decisions (train/roles.py --decisions)."""

    def __init__(self, g, model, where: bool = True, draw: bool = False, lone: bool = False, batch: int = 1,
                 together: bool = False, by_key: bool = False, **options):
        super().__init__(g, **options)
        # option by_key: a control-group key whose units (all of them, all in this fight) drew one decision gets one
        # order for the whole key, as a pro recalls a key and orders it (71% of the pros' fight selections are
        # recalls); the per-unit filters (already attacking, ordered a moment ago) don't split it. With together,
        # key-mates draw alike as often as the pros' (eval/key_coherence.py)
        self.by_key = by_key
        self.key_told: dict[int, tuple[str, int]] = {}    # key -> (the role it was last given, when)
        # option together: the units on one control-group key draw their decisions with one shared random number, a
        # group deciding as one, as a pro decides for a key (eval/key_coherence.py, held-out pro fights: key-mates
        # take the same decision 80% of the time; drawn unit by unit the model's 30%, shared 77%, with no loss in
        # matching the pros' decisions); the units on no key draw alone
        self.together = together
        # option batch: up to this many orders a turn, a group per role by the same priority as the one order (the
        # rest given as the hands come free, fight.py's queue): one order a half-second turn, waiting on the hands,
        # capped fight commands whatever the model decided (train/roles_rl.py --pipe, docs/PROGRESS.md)
        self.batch = batch
        self.model = model
        self.use_where = where                  # option: move units where the model's where head says
        self.where: dict[int, tuple[float, float]] = {}   # unit -> where the model moves it (with a where head)
        self.draw = draw                        # option: each unit's decision drawn from the model's odds (the league)
        # (lone: RoleAhead's, a lone outnumbered unit the model's; here every unit is the model's already)

    def decide(self, obs: dict, snap) -> Intent | None:
        names = self.model.config.get("roles")
        g = self.g
        now = g.hi.frame
        extra = self._motion(obs, snap)
        if snap is None:
            return None
        rows, tags, center, by_tag = snap
        from gary.policy.roles import order_budget
        budget = order_budget(g.hi)
        keys = np.array(g.motor.keys_of(list(tags))) if self.model.config.get("keys") or self.together or self.by_key \
            else None
        shared: dict[int, float] = {}                    # key -> its units' shared uniform (together)
        probs, moves = self.model.predict_move(rows, extra, obs["frame"] * 42 / 1000, budget, keys)
        mine, enemy = self._units(snap)
        if not mine or not enemy:
            return None
        index = {t: i for i, t in enumerate(tags)}
        self.where = {}
        if moves is not None and self.use_where:     # (the model's moves are mirrored as its rows)
            fx, fy = (-1 if g.flip[0] else 1), (-1 if g.flip[1] else 1)
            for u in mine:
                dx, dy = moves[index[u["tag"]]] * 32
                if math.hypot(dx, dy) >= WHERE_MIN:
                    self.where[u["tag"]] = (u["x"] + fx * dx * WHERE_LEAD, u["y"] + fy * dy * WHERE_LEAD)
        logging = g.fight.rl_log is not None
        decided: dict[int, str] = {}
        drawn: dict[int, int] = {}
        roles: dict[str, list[dict]] = {}
        for u in mine:
            p = probs[index[u["tag"]]].astype(np.float64)
            if logging or self.draw:
                key = int(keys[index[u["tag"]]]) if self.together and keys is not None else -1
                r = shared.setdefault(key, g.fight.rng.random()) if key >= 0 else g.fight.rng.random()
                k = min(int(np.searchsorted(np.cumsum(p / p.sum()), r, side="right")), len(p) - 1)
            else:
                k = int(p.argmax())
            decided[u["tag"]], drawn[u["tag"]] = names[k], k
            role = names[k].upper()
            if self.refresh and role == "KEEP" and u["tag"] in self.where:
                told = self.told.get(u["tag"], ("", 0))[0]   # (a move kept current: sent on where the model moves it)
                if told in ("RETREAT", "PULL_BACK", "REPOSITION") and self._stale(u, told, self.where[u["tag"]]):
                    role = told
            roles.setdefault(role, []).append(u)
        mx, my = _mid(mine)
        home = any(u["owner"] == g.slot and u["type"] in g.race.town_halls and math.dist((u["x"], u["y"]), (mx, my)) < HOME
                   for u in by_tag.values())
        fresh = lambda u, role: not (self.told.get(u["tag"], ("", -10**9))[0] == role
                                     and now - self.told[u["tag"]][1] < REPEAT)
        again = roles.pop("AGAIN", [])
        keyed = self._by_key(roles, mine, enemy, keys, index) if self.by_key and keys is not None else []
        if keyed:
            used = {t for it in keyed for t in it.units}
            roles = {r: [u for u in us if u["tag"] not in used] for r, us in roles.items()}
        intent = self._act(roles, mine, enemy, by_tag, home, fresh)
        self.more = []
        if keyed:                               # (the keys' orders first, then the rest by the same priority)
            if intent is not None:
                keyed.append(intent)
            intent, self.more = keyed[0], keyed[1:self.batch]
        left = roles
        while intent is not None and len(self.more) + 1 < self.batch:
            used = {t for it in [intent] + self.more for t in it.units}
            left = {r: [u for u in us if u["tag"] not in used] for r, us in left.items()}
            nxt = self._act(left, mine, enemy, by_tag, home, fresh)
            if nxt is None or not set(nxt.units) - used:
                break
            self.more.append(nxt)
        if again:
            for it in self._again(obs, rows, extra, budget, index, again, keys):
                if intent is None:
                    intent = it
                elif len(self.more) + 1 < self.batch:
                    self.more.append(it)
        if logging:                             # (train/roles_rl.py: the drawn decisions, inputs as the model saw them)
            g.fight.rl_log.append({"rows": np.asarray(rows), "extra": self.model.inputs(extra, budget, keys),
                                   "time": obs["frame"] * 42 / 1000, "units": [index[u["tag"]] for u in mine],
                                   "roles": [drawn[u["tag"]] for u in mine], "executed": intent is not None})
        return intent

    KEY_ROLES = ("RETREAT", "PULL_BACK", "FIRE", "CLOSE_IN", "REPOSITION")

    def _by_key(self, roles: dict, mine: list[dict], enemy: list[dict], keys, index: dict) -> list[Intent]:
        """by_key: the orders for whole keys (see __init__), most urgent role first."""
        g, now = self.g, self.g.hi.frame
        role_of = {u["tag"]: r for r, us in roles.items() for u in us}
        here = {u["tag"]: u for u in mine}
        units = {u["tag"]: u for u in g.hi.observe()["units"] if u["owner"] == g.slot}
        out = []
        for key in sorted({int(keys[index[t]]) for t in here if index[t] < len(keys)} - {-1}):
            members = g.motor.members(key) & set(units)
            in_fight = [here[t] for t in members if t in here]
            if len(in_fight) < 2 or len(in_fight) < KEY_IN_FIGHT * len(members):
                continue
            # the members that decided something agree, the rest keep: the key's decision is theirs (a pro orders
            # the key: its kept members get the order again, its stragglers come along)
            decided = {role_of.get(u["tag"]) for u in in_fight} - {"KEEP", None}
            if len(decided) != 1:
                continue
            role = decided.pop()
            if role not in self.KEY_ROLES:
                continue
            group = [units[t] for t in g.leases.free(sorted(members), "roles")]
            if len(group) < len(members):        # (a member another layer holds: not the whole key)
                continue
            last = self.key_told.get(key, ("", -10 ** 9))
            if last[0] == role and now - last[1] < REPEAT:
                continue
            if role in ("FIRE", "CLOSE_IN"):
                target = min(enemy, key=lambda e: math.dist((e["x"], e["y"]), _mid(group)))
                point, order = (int(target["x"]), int(target["y"])), "attack_move"
            elif role == "RETREAT":
                point, order = self._retreat_point(group, enemy), "move"
            else:
                pts = [self.where[u["tag"]] for u in group if u["tag"] in self.where]
                if not pts:
                    continue
                point, order = (sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts)), "move"
            it = self._intent(role, group, point, order)
            if it is not None:
                self.key_told[key] = (role, now)
                out.append(it)
        return sorted(out, key=lambda it: self.KEY_ROLES.index(it.kind))

    def _again(self, obs: dict, rows, extra, budget, index: dict, units: list[dict], keys=None) -> list[Intent]:
        """The again decision: each unit's order given again, an attack-move if it is fighting (attacking, or told to
        fire or close in), else a move, to where the offset head points; units with nearby points share an order."""
        from gary.policy.roles import MAX_UNITS
        g, now = self.g, self.g.hi.frame
        offsets = self.model.predict_offset(rows, extra, obs["frame"] * 42 / 1000, budget, keys)
        if offsets is None:
            return []
        fx, fy = (-1 if g.flip[0] else 1), (-1 if g.flip[1] else 1)
        buckets: dict[tuple, list] = {}
        for u in units:
            told, at = self.told.get(u["tag"], ("", -10 ** 9))
            i = index[u["tag"]]
            if now - at < AGAIN_GAP or i >= MAX_UNITS:
                continue
            dx, dy = offsets[i] * 32
            point = (max(0.0, u["x"] + fx * dx), max(0.0, u["y"] + fy * dy))
            if u.get("order") in ATTACK or told in ("FIRE", "CLOSE_IN", "POST"):
                key = (told if told in ("FIRE", "CLOSE_IN") else "FIRE", "attack_move")
            else:
                key = (told if told in ("RETREAT", "PULL_BACK", "REPOSITION") else "REPOSITION", "move")
            buckets.setdefault(key, []).append((u, point))
        out = []
        for (role, order), items in buckets.items():
            while items:
                seed = items[0][1]
                group = [x for x in items if math.dist(x[1], seed) < AGAIN_NEAR][:MAX_GROUP]
                items = [x for x in items if x not in group]
                pts = [q for _, q in group]
                out.append(self._intent(role, [u for u, _ in group],
                                        (sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts)), order))
        return out

    def _act(self, roles, mine, enemy, by_tag, home: bool, fresh) -> Intent | None:
        """With a where head (ingest/response_dataset.py: the pros' units before contact as well as in
        it), retreat, pull back and reposition go where the model moves them (a lone marine to the
        buildings, marines into the mineral line as zerglings come: the user's examples), not by the
        hand skills (whose retreat never walks away at home); fire and close in as the rules do."""
        if not self.where:
            return super()._act(roles, mine, enemy, by_tag, home, fresh)
        for role in ("RETREAT", "PULL_BACK", "REPOSITION"):
            group = [u for u in roles.get(role, []) if u["tag"] in self.where and fresh(u, role)][:MAX_GROUP]
            if group:
                pts = [self.where[u["tag"]] for u in group]
                return self._intent(role, group, (sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts)),
                                    "move")
        return super()._act({k: v for k, v in roles.items() if k not in ("RETREAT", "PULL_BACK")},
                            mine, enemy, by_tag, home, fresh)


class RoleAhead(RoleRules):
    """ahead/v1: the hand rules in contact, the role model before it. The learned roles in place of the
    rules lost 52 net value a home defense (226 pro scenarios, v0.29 against v0.28), with or without
    their where; the rules' post, leash and holds are what the model doesn't know. Before contact the
    rules have nothing to say, and that is where Gary felt AFK to the user (marines moved into the
    mineral line as zerglings came, a lone marine running from 4 zerglings to the buildings): units
    with no seen enemy fighter within AHEAD_FROM that the model (with a where head) would reposition,
    pull back or retreat go where it moves them; the rules decide for the rest. With draw (the league),
    each such unit's decision is drawn from the model's odds instead of the likeliest, and logged."""

    def __init__(self, g, model, draw: bool = False, lone: bool = False, **options):
        super().__init__(g, **options)
        self.model = model
        self.draw = draw
        # option lone: a unit alone and outnumbered in contact is the model's too, not the rules' (theirs fight: such
        # units attacked 75-78% of the time and backed away 0-2%, the pros' 42-44% and 13-19%; human game 4 and 96
        # games against gz1, eval/humanlike.py lone_attack / lone_away)
        self.lone = lone

    @staticmethod
    def _alone_outnumbered(u: dict, mine: list[dict], enemy: list[dict]) -> bool:
        """No own fighter within LONE_FRIEND and 2+ seen enemy fighters within LONE_ENEMY (eval/humanlike.py's lone)."""
        near = lambda others, px: sum(1 for v in others if v["tag"] != u["tag"]
                                      and math.dist((u["x"], u["y"]), (v["x"], v["y"])) < px)
        return near(mine, LONE_FRIEND) == 0 and near(enemy, LONE_ENEMY) >= 2

    def decide(self, obs: dict, snap) -> Intent | None:
        g = self.g
        self.before = self.prev
        extra = self._motion(obs, snap)
        if snap is None:
            return None
        rows, tags, center, by_tag = snap
        mine, enemy = self._units(snap)
        if not mine or not enemy:
            return None
        from gary.policy.roles import order_budget
        budget = order_budget(g.hi)
        probs, moves = self.model.predict_move(rows, extra, obs["frame"] * 42 / 1000, budget)
        names = self.model.config.get("roles")
        index = {t: i for i, t in enumerate(tags)}
        now = g.hi.frame
        fresh = lambda u, role: not (self.told.get(u["tag"], ("", -10**9))[0] == role
                                     and now - self.told[u["tag"]][1] < REPEAT)
        groups: dict[str, list] = {}
        self.choice: dict[int, int] = {}
        if moves is not None:
            fx, fy = (-1 if g.flip[0] else 1), (-1 if g.flip[1] else 1)
            for u in mine:
                if min(math.dist((u["x"], u["y"]), (e["x"], e["y"])) for e in enemy) < AHEAD_FROM and not (
                        self.lone and self._alone_outnumbered(u, mine, enemy)):
                    continue
                i = index[u["tag"]]
                k = int(g.fight.rng.choice(len(probs[i]), p=probs[i] / probs[i].sum())) if self.draw else int(probs[i].argmax())
                self.choice[u["tag"]] = k
                role = names[k].upper()
                dx, dy = moves[i] * 32
                to = (u["x"] + fx * dx * WHERE_LEAD, u["y"] + fy * dy * WHERE_LEAD)
                if role in ("RETREAT", "PULL_BACK", "REPOSITION") and math.hypot(dx, dy) >= WHERE_MIN and fresh(u, role):
                    groups.setdefault(role, []).append((u, to))
                elif self.refresh and role == "KEEP" and math.hypot(dx, dy) >= WHERE_MIN:
                    told = self.told.get(u["tag"], ("", 0))[0]
                    if told in ("RETREAT", "PULL_BACK", "REPOSITION") and self._stale(u, told, to):
                        groups.setdefault(told, []).append((u, to))   # (its move kept current)
        from gary.decision_log import LOG, top
        from gary.policy.roles import MAX_UNITS
        if self.choice:                     # (the game's records: the roles chosen, the model's mean odds)
            from collections import Counter
            chosen = Counter(names[k] for k in self.choice.values())
            LOG.decision(g.hi.side, "roles", g.hi.frame, ", ".join(f"{r} {c}" for r, c in chosen.most_common()),
                         top=top(lambda k: names[k], probs[[index[t] for t in self.choice]].mean(0)),
                         inputs={"units": len(self.choice), "enemy": len(enemy)}, by="draw" if self.draw else "favorite",
                         then="orders: " + (", ".join(f"{r.lower()} {len(v)}" for r, v in groups.items()) or "none"))
        if g.fight.rl_log is not None and self.draw and self.choice:   # (train/roles_rl.py: the drawn decisions, their
            ids = [index[t] for t in self.choice]                       # inputs as the model saw them, budget too)
            self.g.fight.rl_log.append({"rows": np.asarray(rows), "extra": self.model._with_budget(extra, budget),
                                        "time": obs["frame"] * 42 / 1000, "units": ids,
                                        "roles": list(self.choice.values()), "executed": bool(groups)})
        if LOG.on and self.draw:            # (each before-contact unit's drawn decision and its log-probability)
            drawn = np.full(MAX_UNITS, -1, np.int8)
            logp = np.zeros(MAX_UNITS, np.float32)
            for u in mine:
                i = index[u["tag"]]
                k = self.choice.get(u["tag"])
                if i < MAX_UNITS and k is not None:
                    drawn[i], logp[i] = k, np.log(probs[i, k] + 1e-9)
            if (drawn >= 0).any():
                pad = lambda a: np.concatenate([a[:MAX_UNITS], np.zeros((max(0, MAX_UNITS - len(a)),) + a.shape[1:], a.dtype)])
                LOG.add("roles_t", frame=g.hi.frame, rows=pad(np.asarray(rows, np.int16)), extra=pad(np.asarray(extra, np.float16)),
                        count=min(len(rows), MAX_UNITS), time=obs["frame"] * 42 / 1000, drawn=drawn, logp=logp)
        for role in ("RETREAT", "PULL_BACK", "REPOSITION"):
            group = groups.get(role, [])[:MAX_GROUP]
            if group:
                pts = [p for _, p in group]
                return self._intent(role, [u for u, _ in group],
                                    (sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts)), "move")
        own = [u for u in by_tag.values() if u["owner"] == g.slot and u["completed"] and supply_x2(u["type"]) > 0
               and not is_worker(u["type"]) and not is_building(u["type"])]
        own = [u for u in own if u["tag"] in set(g.leases.free([u["tag"] for u in own], "roles"))]
        enemy_all = [u for u in by_tag.values() if u["owner"] not in (g.slot, 11) and supply_x2(u["type"]) > 0
                     and not is_worker(u["type"]) and not is_building(u["type"])]
        return self._rules(by_tag, own, enemy_all)


OVERRIDES = ("join", "hold")


class RoleOverride(RoleRules):
    """override/v1: the hand rules, with a learned override per unit (the role model's decisions
    join and hold, gary/policy/roles.py): every turn the model says, for each of Gary's units in the
    fight, join, hold, or anything else, which leaves the unit to the rules (its probability is
    the rest). Units it takes go first (their group's command), the rules decide for the others.
    Started from a model that almost never overrides, reinforcement learning (train/roles_rl.py
    --override) learns when to: fresh units walking into a fight alone died alone."""

    def __init__(self, g, model, **options):
        super().__init__(g, **options)
        self.model = model
        names = model.config.get("roles")
        self.over = [names.index(r) for r in OVERRIDES]

    def decide(self, obs: dict, snap) -> Intent | None:
        g = self.g
        logging = g.fight.rl_log is not None
        self.before = self.prev
        extra = self._motion(obs, snap)
        if snap is None:
            return None
        rows, tags, center, by_tag = snap
        mine, enemy = self._units(snap)
        if not mine or not enemy:
            return None
        from gary.policy.roles import order_budget
        probs = self.model.predict(rows, extra, obs["frame"] * 42 / 1000, order_budget(g.hi))
        index = {t: i for i, t in enumerate(tags)}
        decided, roles = {}, {}
        for u in mine:
            p = probs[index[u["tag"]]].astype(np.float64)
            three = np.array([max(0.0, 1 - p[self.over].sum()), *p[self.over]])
            k = int(g.fight.rng.choice(3, p=three / three.sum())) if logging else int(three.argmax())
            decided[u["tag"]] = "keep" if k == 0 else OVERRIDES[k - 1]
            if k:
                roles.setdefault(OVERRIDES[k - 1].upper(), []).append(u)
        now = g.hi.frame
        fresh = lambda u, role: not (self.told.get(u["tag"], ("", -10**9))[0] == role
                                     and now - self.told[u["tag"]][1] < REPEAT)
        intent = self._join_or_hold(roles, mine, by_tag, fresh)
        if intent is None:
            # the rules, on the units they take when deciding alone (all of Gary's free fighters in the
            # snapshot), less those overridden
            taken = {u["tag"] for us in roles.values() for u in us}
            own = [u for u in by_tag.values() if u["owner"] == g.slot and u["completed"] and supply_x2(u["type"]) > 0
                   and not is_worker(u["type"]) and not is_building(u["type"]) and u["tag"] not in taken]
            own = [u for u in own if u["tag"] in set(g.leases.free([u["tag"] for u in own], "roles"))]
            enemy_all = [u for u in by_tag.values() if u["owner"] not in (g.slot, 11) and supply_x2(u["type"]) > 0
                         and not is_worker(u["type"]) and not is_building(u["type"])]
            intent = self._rules(by_tag, own, enemy_all)
        if logging:
            self._log(obs, snap, extra, mine, decided, intent)
        return intent

