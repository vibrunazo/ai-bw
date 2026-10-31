"""The Zerg executor (docs/REFACTOR.md Phase 1): Zerg hands over the race-neutral Executor, from
Gary-Z's (its gary_z.py and zerg_v3.py, in git before the refactor's Phase 3).

  produce   a larva's unit: a tiny box on a fresh larva (one just morphed still shows as a larva a
            moment: not again), then its key; zergling and scourge orders make two
  morph     a unit's or a building's morph (a Lurker from a Hydralisk; a Lair, a Hive, a Greater
            Spire, a sunken or spore from a creep colony): one of them selected, one with nothing
            of Gary's on top first, another after each miss (an overlord over the hydralisks took
            every click), then its key
  build     a drone's building, by the shared build task with the drone's order: it walks there and
            becomes the building (on creep, which the game's check asks); creep colonies in front of
            the natural once Gary holds it
  supply    an overlord ahead of production (option depots): supply left at most 2 per hatchery
  gas       the shared refill, but none while the gas floats (over GAS_FLOAT and over the minerals: its
            drones mine minerals until it is under GAS_LOW), unless a Spire stands and the first flock
            isn't made yet
  rallies   each hatchery beside its base's minerals while droning, the army's meeting point from
            RALLY_FRONT_AT, the main while the meeting point is besieged
"""

from __future__ import annotations

import math
from collections import Counter

from gary import commands as C
from gary import orders as O
from gary import zerg as Z
from gary.agent.executor import PATH_CLEAR, Executor, _to_segment, field_click
from gary.agent.gary import Gary, Task
from gary.agent.macro import BMORPH, BUILD, EXPAND, MORPH, RESEARCH, UPGRADE, WAIT
from gary.decision_log import LOG
from gary.interface import screen_of

USED_LARVA = 2 * 24              # a larva just morphed shows as a larva a moment longer: not used again
LARVA_BOXES = 3                  # boxes on a larva that select something else too before the task lets go
CLEAR_PX = 16                    # a unit with another of Gary's this close may be under it
OVERLORD_FOR = 25 * 24           # an overlord ordered counts as coming this long (its morph takes 25 s)
GAS_FLOAT, GAS_LOW = 400, 300    # gas above this (and above the minerals): its drones mine minerals; below this, gas
MUTA_FLOCK_SAVE = 6              # no floating while a Spire stands and fewer mutalisks than this
RALLY_FRONT_AT = 5 * 60 * 24     # hatcheries rally to the army's meeting point from then on
RALLY_SIEGE = 14 * 32            # enemy fighters this near the meeting point: the hatcheries rally to the main...
RALLY_SIEGE_FOR = 20 * 24        # ...and keep to it this long after the last was seen there
CREEP_FRONT = 0.6                # a creep colony this far from the natural toward the meeting point
EGG_SLACK = 3 * 24               # an overlord's supply counted coming this long past its make time (the egg hatches
                                 # after it: a second overlord at 1:27 as the first hatched, 100 minerals of drones)
LORDS_EVERY = 3 * 24             # overlords over a mineral line looked for this often...
LORD_OVER = 2 * 32               # ...this near a field...
LORD_BEHIND = 4 * 32             # ...and sent this far behind their hatchery
LORD_CLEAR = 48                  # a rally's click on a field at least this far (pixels) from every overlord of Gary's
SPOT_RINGS = (4, 14)             # a building's spot: the first that fits in rings this far around the hatchery (tiles)


class ZergExecutor(Executor):
    build_order = C.ORDER_DRONE_START_BUILD
    # (larvae beside the hatchery: a mining order leaves them be; overlords over the drones go along, as Gary-Z's
    # did; drones just hatched, standing or walking to the rally, mine with the others. Refused, 40% of the mining
    # tasks ended without a gather order and the drones stood idle 206 s a game in the first 4 minutes, Gary-Z's 52;
    # with them, 113 s off work in all to Gary-Z's 125, 35 of them following an overlord that flew to the field.
    # A shift-click to take the overlord out of the selection doesn't, here: 330 s)
    box_extras = frozenset({Z.LARVA, Z.OVERLORD})
    box_orders = frozenset({2, 3, 6, 41})         # Guard, PlayerGuard, Move, ZergBirth
    army_select = "portrait"
    # (an expansion's drone killed on its way failed the build, and the macro decided it again only at its 1-5%
    # odds: gz1 held 3 bases from 12:00 with 700-860 minerals banked, round G4; Gary-Z's jobs sent another drone)
    builder_retries = 3
    in_flight = 24                                # frames a morph's money counts as spent before the game shows it
                                                  # (Gary-Z's: a larva's is taken at once; 2 s held a drone's 50 idle)

    def __init__(self, g: Gary):
        super().__init__(g)
        self.used_larva: dict[int, int] = {}             # larva tag -> frame it was morphed
        self.morphing: dict[int, int] = {}               # unit or building ordered to morph -> frame it's done
        self.morphs: list[tuple[int, int, int]] = []     # (frame, unit type, frames to make): eggs hide what they'll be
        self.build_frames: dict[int, int] = {}
        self.select_misses = 0
        self.rally_of: dict[int, str] = {}               # hatchery -> "minerals", "front" or "back"
        self.siege_until = 0
        self.gas_floating = False
        self.basic["hatch_rally"] = self._task_hatch_rally
        g.task_runners["hatch_rally"] = self._run_basic
        self.basic["lords_off"] = self._task_lords_off
        g.task_runners["lords_off"] = self._run_basic
        self.next_lords = 0
        self.commands["produce"] = self._task_produce
        self.commands["morph"] = self._task_morph
        for kind in ("produce", "morph"):
            g.task_runners[kind] = self._run_command

    # --- what a decision means for Zerg hands ------------------------------------------------------

    def carry_out(self, act: int, uid: int, obs: dict, mine: list[dict], done: list[dict], me: dict) -> dict | None:
        g = self.g
        have = {u["type"] for u in done}
        cost = self.cost_of(act, uid)
        if uid == Z.CREEP and self._colonies_capped(obs, mine):
            return self.no("the pros' cap of colonies for this minute")
        if act == MORPH and uid in Z.FROM_LARVA:
            if uid != Z.OVERLORD and me["supply_used"] + Z.SUPPLY.get(uid, 1) > me["supply_max"]:
                # supply blocked: wait if an overlord is on its way, else let one come next
                return self.wait("supply: an overlord is coming") if self.in_production(mine, EGG_SLACK)[Z.OVERLORD] else self.no("supply")
            if not self._fresh_larvae(mine):
                return self.wait("no larva") if any(u["type"] in Z.TOWN_HALLS for u in mine) else self.no("no hatchery")
            return {"kind": "produce", "act": act, "id": uid, "cost": cost}
        if act == MORPH and uid in Z.UNIT_MORPH:
            if uid == Z.LURKER and Z.LURKER_ASPECT not in g.researched:
                return self.no("no Lurker Aspect")
            units = [u for u in done if u["type"] == Z.UNIT_MORPH[uid]]
            if not units:
                return self.no(f"no {Z.NAMES[Z.UNIT_MORPH[uid]]} to morph")
            if me["supply_used"] + Z.SUPPLY.get(uid, 1) - Z.SUPPLY.get(Z.UNIT_MORPH[uid], 1) > me["supply_max"]:
                return self.no("supply")
            return {"kind": "morph", "act": act, "id": uid, "of": Z.UNIT_MORPH[uid], "cost": cost}
        if act == BMORPH and uid in Z.BUILDING_MORPH:
            parent = Z.BUILDING_MORPH[uid]
            parents = [u for u in done if u["type"] == parent]
            if not parents:
                return self.no(f"no {Z.NAMES[parent]} to morph")
            if not any(u.get("queue", 0) == 0 and u["order"] not in (O.ORDER_RESEARCH, O.ORDER_UPGRADE) for u in parents):
                return self.wait(f"its {Z.NAMES[parent]} is busy")
            return {"kind": "morph", "act": act, "id": uid, "of": parent, "cost": cost}
        if act == BUILD and uid in Z.DRONE_BUILDS:
            if any(ut == uid for ut, *_ in g.reserved) or g.building(uid):
                return self.no("one already on its way")     # (one of a kind on its way at a time)
            if uid == Z.EXTRACTOR:
                spot = self.free_geyser(obs, mine)
                return {"kind": "build", "where": spot[0], "data": {"spot": spot[1]}, "cost": cost} if spot \
                    else self.no("no free geyser")
            if uid == Z.CREEP and self._holds_natural(mine):
                nat, front = g.natural.center, g.rally
                near = (int(nat[0] + (front[0] - nat[0]) * CREEP_FRONT), int(nat[1] + (front[1] - nat[1]) * CREEP_FRONT))
                return {"kind": "build", "where": "natural", "data": {"near": near, "level_at": nat}, "cost": cost}
            return {"kind": "build", "where": "main", "cost": cost}
        if act == EXPAND:
            base = self.free_base(obs)
            return {"kind": "build", "where": "natural" if base is g.natural else base, "cost": cost} if base \
                else self.no("no free base")
        if act == RESEARCH and uid in Z.RESEARCH and uid not in g.researched:
            if not all(Z.has(have, r) for r in Z.RESEARCH_NEEDS.get(uid, [])):
                return self.no("what it needs isn't there")
            return self._at_building(obs, mine, done, Z.RESEARCH[uid][0], act, uid, cost)
        if act == UPGRADE and uid in Z.UPGRADE:
            where, _, levels = Z.UPGRADE[uid]
            level = g.upgraded[uid] + 1
            need = Z.UPGRADE_LEVEL_NEEDS.get(level) if levels > 1 else None
            if level > levels:
                return self.no("done already")
            if not all(Z.has(have, r) for r in Z.UPGRADE_NEEDS.get(uid, [])) or (need is not None and not Z.has(have, need)):
                return self.no("what its next level needs isn't there")
            cost = (cost[0] + 75 * (level - 1), cost[1] + 75 * (level - 1))
            return self._at_building(obs, mine, done, where, act, uid, cost)
        if act == RESEARCH and uid in g.researched:
            return self.no("done already")
        return self.no("not a decision Gary-Z carries out")

    def _at_building(self, obs, mine, done, where: int, act: int, uid: int, cost) -> dict | None:
        """A command to an idle building of this kind (a Lair or Hive counts as a Hatchery), WAIT if all are busy."""
        kinds = Z.HAS_TECH.get(where, {where})
        for kind in kinds:
            b = self.free_building(obs, mine, done, kind)
            if b is not None:
                return {"kind": "command", "act": act, "id": uid, "building": b, "cost": cost}
        name = Z.NAMES.get(where, str(where))
        return self.wait(f"its {name} is busy") if any(u["type"] in kinds for u in done) else self.no(f"no {name}")

    def _base_spot(self, unit_type: int, base, builder: int) -> tuple[int, int] | None:
        """The first spot in rings around the base's hatchery that fits (on its creep, which the game's check
        asks), off the drones' path to the minerals (Gary-Z's rings; the shared spot's gap around the building
        found no creep wide enough for a Spawning Pool by the first hatchery)."""
        g = self.g
        w, h = g.race.sizes[unit_type]
        cx, cy = base.center[0] // 32, base.center[1] // 32
        paths = [(base.center, m) for m in base.minerals]
        for r in range(*SPOT_RINGS):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    tx, ty = cx + dx, cy + dy
                    if any(abs(tx - rx) < g.race.sizes[rt][0] + 1 and abs(ty - ry) < g.race.sizes[rt][1] + 1
                           for rt, rx, ry in g.reserved) or self._bad(unit_type, tx, ty):
                        continue
                    mid = (tx * 32 + w * 16, ty * 32 + h * 16)
                    if any(_to_segment(mid, a, b) < w * 16 + PATH_CLEAR for a, b in paths):
                        continue
                    if self.fits(unit_type, tx, ty, builder):
                        return tx, ty
        return None

    def site_ok(self, unit_type: int, tx: int, ty: int) -> bool:
        """The game's check asks for creep where a Zerg building needs it."""
        return True

    def _holds_natural(self, mine: list[dict]) -> bool:
        return any(u["type"] in Z.TOWN_HALLS and math.dist((u["x"], u["y"]), self.g.natural.center) < 6 * 32 for u in mine)

    def _colonies_capped(self, obs: dict, mine: list[dict]) -> bool:
        """Creep colonies and what they became, at the pros' caps of all three for this minute."""
        model = self.g.macro.model
        caps = [model.cap(t, obs["frame"]) for t in (Z.CREEP, Z.SUNKEN, Z.SPORE)]
        if any(c is None for c in caps):
            return False
        n = sum(1 for u in mine if u["type"] in (Z.CREEP, Z.SUNKEN, Z.SPORE)) + \
            sum(1 for ut, *_ in self.g.reserved if ut == Z.CREEP) + self.g.building(Z.CREEP)
        return n >= sum(caps)

    def burst(self, act: int, uid: int) -> bool:
        return act == MORPH and uid in Z.FROM_LARVA

    def supply_due(self, obs: dict, mine: list[dict], done: list[dict], me: dict) -> tuple[int, str, bool] | None:
        """An overlord before a supply block (Gary-Z's): supply left at most 2 per hatchery, one coming at a
        time per finished hatchery."""
        halls = [u for u in mine if u["type"] in Z.TOWN_HALLS]
        coming = self.in_production(mine, EGG_SLACK)[Z.OVERLORD]
        left = me["supply_max"] - me["supply_used"]
        # (a larva awaited counts: the overlord takes the next one, else the drones took every larva and Gary
        # sat supply-blocked from 1:05 to 2:00 in the first mirror. With 100 minerals there, as Gary-Z's: without, the
        # goal held the money from 7/9 on, 6 s blocked to Gary-Z's 26 at 4:00, but the Extractor came at 3:03, not 1:26)
        if left <= 2 * len(halls) and me["minerals"] >= 100 and coming < max(1, sum(1 for h in halls if h["completed"])) \
                and self.g.macro.plan(MORPH, Z.OVERLORD, obs, mine, done, me, capped=False) is not None:
            # (past the pros' caps, which the dice keep to: Gary-Z's overlord rule ignored them)
            return MORPH, f"rule: an overlord ahead of production ({left} left, {len(halls)} hatcheries)", True
        return None

    # --- what Zerg has ----------------------------------------------------------------------------

    def in_production(self, mine: list[dict], slack: int = 0) -> Counter:
        """Units ordered and not out yet (eggs and cocoons hide what they'll be): from Gary's own orders,
        each for as long as it takes to make (and slack frames more); a zergling or scourge order makes two."""
        f = self.g.hi.frame
        self.morphs = [m for m in self.morphs if f - m[0] < m[2] + EGG_SLACK]
        out: Counter = Counter()
        for at, t, frames in self.morphs:
            if f - at < frames + slack:
                out[t] += 2 if t in Z.PAIRS else 1
        return out

    def count(self, mine: list[dict], unit_type: int) -> int:
        """As Gary-Z counted for the pros' caps: those standing (a Hatchery's Lairs and Hives too) and the
        buildings on their way, not the eggs (the pros' counts are of units out)."""
        g = self.g
        became = Z.TOWN_HALLS if unit_type == Z.HATCHERY else {unit_type}
        n = sum(1 for u in mine if u["type"] in became)
        n += sum(1 for ut, *_ in g.reserved if ut == unit_type)
        return n + (1 if g.building(unit_type) else 0)

    def _frames_to_make(self, uid: int) -> int:
        if uid not in self.build_frames:
            game = self.g.hi.game
            info = game.unit_type_info(uid) if hasattr(game, "unit_type_info") else None
            self.build_frames[uid] = int(info["build_frames"]) if info else 30 * 24
        return self.build_frames[uid]

    def _fresh_larvae(self, mine: list[dict]) -> list[dict]:
        f = self.g.hi.frame
        self.used_larva = {t: at for t, at in self.used_larva.items() if f - at < USED_LARVA}
        return [u for u in mine if u["type"] == Z.LARVA and u["tag"] not in self.used_larva]

    def _idle(self, u: dict, mine: list[dict]) -> bool:
        """Also a gas drone while the gas floats: to minerals; and a drone following an overlord of Gary's (a rally
        or mining click that hit one: on Odyssey every new drone followed the first overlord, 1,000-1,600
        drone-seconds of a game's first 5 minutes, round O3)."""
        if u["type"] == Z.DRONE and u["order"] == O.ORDER_FOLLOW and any(
                o["tag"] == u.get("order_target") and o["type"] == Z.OVERLORD for o in mine):
            return True
        return super()._idle(u, mine) or (self.gas_floating and u["type"] == Z.DRONE and u["order"] in (81, 82, 84))

    def trim_selection(self, obs: dict, mine: list[dict]) -> bool:
        """An overlord the box took with the drones: shift+click on its portrait (sent along, it flew to the field
        and the next clicks there hit it: 35 drone-seconds a game following one)."""
        sel = set(obs["selection"])
        lord = next((u for u in mine if u["tag"] in sel and u["type"] == Z.OVERLORD), None)
        if lord is None or len(sel) == 1:
            return False
        return bool(self.g.hi.shift_click_portrait(lord["tag"]).accepted)

    def click_blockers(self, mine: list[dict]) -> list[tuple[int, int]]:
        """Overlords: a right-click on a mineral field under one hits the overlord, and the drones follow it."""
        return [(u["x"], u["y"]) for u in mine if u["type"] == Z.OVERLORD]

    # --- tasks ------------------------------------------------------------------------------------

    def _task_produce(self, obs: dict, mine: list[dict], t: Task) -> None:
        """A larva selected (a tiny box), then the morph."""
        g, hi, uid = self.g, self.g.hi, t.data["id"]
        fresh = self._fresh_larvae(mine)
        sel = [u for u in mine if u["tag"] in obs["selection"]]
        fresh_sel = [u for u in sel if u["type"] == Z.LARVA and u["tag"] not in self.used_larva]
        if fresh_sel and len(fresh_sel) < len(sel) and not t.data.get("narrowed"):
            # the box took a drone or a zergling beside the larva too: a ctrl+click on the larva's portrait keeps
            # the larvae (boxed again, 13 tasks a game gave up after 3 boxes: round G1)
            t.data["narrowed"] = True
            if hi.ctrl_click_portrait(Z.LARVA).accepted:
                return
        if sel and all(u["type"] == Z.LARVA and u["tag"] not in self.used_larva for u in sel):
            if self.given(hi.morph(uid), "morph", t.data.get("act", -1), uid):
                for u in sel:
                    self.used_larva[u["tag"]] = hi.frame
                self.morphs.append((hi.frame, uid, self._frames_to_make(uid)))
            else:
                self._undecided(t)
            g.task = None
            return
        if not fresh:
            LOG.order(g.hi.side, hi.frame, g.race.names.get(uid, str(uid)), "dropped", "no larva free", by="executor")
            self._undecided(t)
            g.task = None
            return
        on = [lv for lv in fresh if screen_of(obs, lv["x"], lv["y"])]
        if not on:
            lv = fresh[(hi.frame // 48) % len(fresh)]
            hi.camera_minimap(lv["x"], lv["y"])
            return
        # boxes that took something else too (drones just hatched stand beside the larvae, and while this task
        # holds the hands nobody sends them mining: 35 s of boxes in the first mirror): the hands go back, and
        # the macro decides again
        t.data["boxes"] = t.data.get("boxes", 0) + 1
        if t.data["boxes"] > LARVA_BOXES:
            LOG.order(g.hi.side, hi.frame, g.race.names.get(uid, str(uid)), "dropped",
                      f"{LARVA_BOXES} boxes on a larva took something else too", by="executor")
            self._undecided(t)
            g.task = None
            return
        others = [u for u in mine if u["type"] != Z.LARVA or u["tag"] in self.used_larva]
        clear = [lv for lv in on if not any(abs(u["x"] - lv["x"]) < 12 and abs(u["y"] - lv["y"]) < 12 for u in others)]
        lv = (clear or on)[(hi.frame // 48) % len(clear or on)]
        x, y = screen_of(obs, lv["x"], lv["y"])
        hi.box(max(0, x - 3), max(0, y - 3), x + 3, y + 3)

    def _task_morph(self, obs: dict, mine: list[dict], t: Task) -> None:
        """One unit or building of the kind selected, then its morph."""
        g, hi, d = self.g, self.g.hi, t.data
        building = d["act"] == BMORPH
        f = hi.frame
        self.morphing = {t: until for t, until in self.morphing.items() if until > f}
        units = [u for u in mine if u["type"] == d["of"] and u["completed"] and u["tag"] not in self.morphing
                 and (not building or (u.get("queue", 0) == 0 and u["order"] not in (O.ORDER_RESEARCH, O.ORDER_UPGRADE)))]
        if not units:
            LOG.order(g.hi.side, hi.frame, g.race.names.get(d["id"], str(d["id"])), "dropped",
                      f"no {g.race.names.get(d['of'], d['of'])} free to morph", by="executor")
            self._undecided(t)
            g.task = None
            return
        if not self._select_one(obs, mine, units):
            return
        ok = self.given(hi.morph_building(d["id"]), "morph_building", d.get("act", -1), d["id"]) if building else             self.given(hi.morph(d["id"]), "morph", d.get("act", -1), d["id"])
        if not ok:
            LOG.order(g.hi.side, hi.frame, g.race.names.get(d["id"], str(d["id"])), "dropped", "the morph wasn't taken",
                      by="executor")
            self._undecided(t)
        if ok:
            # (a building morphing still shows as what it was: a colony becoming a sunken was ordered again and
            # again; its morph counts as in production until done)
            self.morphs.append((hi.frame, d["id"], self._frames_to_make(d["id"])))
            for tag in obs["selection"]:
                self.morphing[tag] = hi.frame + self._frames_to_make(d["id"])
        g.task = None

    def _undecided(self, t: Task) -> None:
        """The decision this task carried didn't happen: out of the model's "decisions so far" (noted when the task
        started; Gary-Z noted a decision once carried out, and the one Gary's history counted the larvae's that
        let go: more mutalisks and fewer zerglings than Gary-Z decided at 5-9 min, the parity diff)."""
        if "act" in t.data:
            self.undecided(t.data["act"], t.data["id"])

    def _select_one(self, obs: dict, mine: list[dict], units: list[dict]) -> bool:
        """One of these selected (the camera on it, then a click): one with nothing of Gary's on top of it,
        the nearest home first, another after each miss (an overlord hovering over the hydralisks took every
        click, 68 in 30 s, and held the macro for 30 s each time: eval/why.py, c11 seed 3001). True once it is."""
        g, hi = self.g, self.g.hi
        sel = set(obs["selection"])
        if len(sel) == 1 and any(u["tag"] in sel for u in units):
            return True
        tags = {u["tag"] for u in units}
        own = [o for o in mine if o["tag"] not in tags]
        clear = [u for u in units if not any(abs(o["x"] - u["x"]) < CLEAR_PX and abs(o["y"] - u["y"]) < CLEAR_PX for o in own)]
        home = g.main.center
        pool = sorted(clear or units, key=lambda u: math.dist((u["x"], u["y"]), home))
        u = pool[self.select_misses % len(pool)]
        p = screen_of(obs, u["x"], u["y"])
        if p is None:
            hi.camera_minimap(u["x"], u["y"])
            return False
        hi.click(*p)
        self.select_misses += 1
        return False

    # --- chores -----------------------------------------------------------------------------------

    def refill_gas(self, obs: dict, done: list[dict]) -> bool:
        """The shared refill, but none while the gas floats (its drones mine minerals meanwhile: the idle chore
        sends a drone off gas to minerals). A macro model with a gas head decides the drones on gas itself (#55)."""
        g = self.g
        if getattr(g.macro, "gas_target", None) is not None:
            self.gas_floating = False
            return super().refill_gas(obs, done)
        me = obs["me"]
        floating = self.gas_floating
        if me["gas"] > GAS_FLOAT and me["gas"] > me["minerals"]:
            self.gas_floating = True
        elif me["gas"] < GAS_LOW:
            self.gas_floating = False
        # (none while a Spire stands and the flock isn't made: pros save ~700 gas for the first mutalisks)
        if any(u["type"] == Z.SPIRE for u in done) and sum(1 for u in done if u["type"] == Z.MUTALISK) < MUTA_FLOCK_SAVE:
            self.gas_floating = False
        if self.gas_floating != floating:
            LOG.decision(g.hi.side, "gas", g.hi.frame, "gas drones to minerals" if self.gas_floating else "gas drones back",
                         inputs={"m": me["minerals"], "gas": me["gas"]},
                         by=f"rule: gas over {GAS_FLOAT} and over the minerals" if self.gas_floating else
                         f"rule: gas under {GAS_LOW}, or a Spire saving for the flock")
        return False if self.gas_floating else super().refill_gas(obs, done)

    def rerally(self, obs: dict, mine: list[dict], done: list[dict]) -> bool:
        """Each finished hatchery rallies beside its base's minerals while droning, to the army's meeting point
        (g.rally) from RALLY_FRONT_AT, to the main while enemy fighters are by the meeting point. Pros' hatchery
        rallies in 46 TvZ games: 78% toward the front, 21% by the minerals (a third of those before 6:00)."""
        g, hi = self.g, self.g.hi
        from gary.policy.army import is_building, is_worker, supply_x2
        front = hi.frame >= RALLY_FRONT_AT
        if any(u["owner"] not in (g.slot, 11) and supply_x2(u["type"]) > 0 and not is_worker(u["type"])
               and not is_building(u["type"]) and math.dist((u["x"], u["y"]), g.rally) < RALLY_SIEGE for u in obs["units"]):
            self.siege_until = hi.frame + RALLY_SIEGE_FOR
        mode = "back" if front and hi.frame < self.siege_until else "front" if front else "minerals"
        for h in done:
            if h["type"] not in Z.TOWN_HALLS or self.rally_of.get(h["tag"]) == mode:
                continue
            base = g.map.base_near((h["x"], h["y"]))
            if not base.minerals and mode == "minerals":
                self.rally_of[h["tag"]] = mode
                continue
            if mode == "back":
                spot = g.main.center
            elif mode == "front":
                spot = g.rally
            else:
                # on a mineral field of its base, as pros rally while droning: a drone rallied onto minerals mines by
                # itself (a ground spot beside them, Gary-Z's, left each new drone standing for the mining chore: 50
                # minerals a drone-minute to the pros' 56 by 5:00, round O1)
                spot = min(base.minerals, key=lambda m: math.dist(m, (h["x"], h["y"])))
            g.task = Task("hatch_rally", h["type"], data={"tag": h["tag"], "spot": spot, "mode": mode}, started=hi.frame)
            return True
        return self._lords_off_minerals(obs, mine, done)

    def _lords_off_minerals(self, obs: dict, mine: list[dict], done: list[dict]) -> bool:
        """An overlord parked over a mineral line of Gary's (the minerals rally brings each new one onto its field)
        is sent behind its hatchery, one at a time: the clicks on that field hit the overlord, and the drones
        followed it (on Odyssey, every new drone: 1,000-1,600 drone-seconds of a game's first 5 minutes, round O3)."""
        g, hi = self.g, self.g.hi
        if hi.frame < self.next_lords:
            return False
        self.next_lords = hi.frame + LORDS_EVERY
        lords = [u for u in done if u["type"] == Z.OVERLORD and u["order"] != O.ORDER_MOVE]
        for h in done:
            if h["type"] not in Z.TOWN_HALLS:
                continue
            base = g.map.base_near((h["x"], h["y"]))
            over = [u for u in lords if base.minerals and min(math.dist((u["x"], u["y"]), m) for m in base.minerals) < LORD_OVER]
            if over:
                mx = sum(x for x, _ in base.minerals) / len(base.minerals)
                my = sum(y for _, y in base.minerals) / len(base.minerals)
                d = math.dist((mx, my), (h["x"], h["y"])) or 1
                spot = (int(h["x"] + (h["x"] - mx) / d * LORD_BEHIND), int(h["y"] + (h["y"] - my) / d * LORD_BEHIND))
                w, ht = g.map.size
                spot = (min(max(spot[0], 16), w - 16), min(max(spot[1], 16), ht - 16))
                g.task = Task("lords_off", Z.OVERLORD, data={"tag": over[0]["tag"], "spot": spot}, started=hi.frame)
                return True
        return False

    def _task_lords_off(self, obs: dict, mine: list[dict], t: Task) -> None:
        """A click on the overlord, then a minimap move behind the hatchery."""
        g, hi, d = self.g, self.g.hi, t.data
        u = next((u for u in mine if u["tag"] == d["tag"]), None)
        if u is not None and obs["selection"] == [u["tag"]]:
            hi.minimap_right_click(*d["spot"])
            LOG.decision(hi.side, "executor", hi.frame, "overlord off the minerals", inputs={"at": d["spot"]},
                         by="rule: an overlord over a mineral line of Gary's")
            g.task = None
            return
        if u is None or t.data.get("clicks", 0) >= 2:
            g.task = None
            return
        t.data["clicks"] = t.data.get("clicks", 0) + 1
        p = screen_of(obs, u["x"], u["y"])
        if p is None:
            hi.camera_minimap(u["x"], u["y"])
            return
        hi.click(*p)

    def _clear_field(self, mine: list[dict], spot: tuple[int, int], h: dict) -> tuple[int, int]:
        """The field of the rally's base nearest the hatchery with no overlord of Gary's over it (a right-click on
        one under an overlord rallies onto the overlord: every new drone follows it), else spot."""
        lords = [(u["x"], u["y"]) for u in mine if u["type"] == Z.OVERLORD]
        base = self.g.map.base_near(spot)
        clear = [m for m in base.minerals if all(math.dist(m, o) >= LORD_CLEAR for o in lords)]
        return min(clear, key=lambda m: math.dist(m, (h["x"], h["y"]))) if clear else spot

    def _task_hatch_rally(self, obs: dict, mine: list[dict], t: Task) -> None:
        g, hi, d = self.g, self.g.hi, t.data
        h = next((u for u in mine if u["tag"] == d["tag"]), None)
        if h is None:
            g.task = None
            return
        if obs["selection"] == [h["tag"]]:
            if d["mode"] == "minerals":
                p = screen_of(obs, *d["spot"])
                if p is None:
                    hi.camera_minimap(*d["spot"])
                    return
                hi.right_click(*field_click(obs, self._clear_field(mine, d["spot"], h), (h["x"], h["y"])))
            else:
                hi.minimap_right_click(*d["spot"])
            self.rally_of[h["tag"]] = d["mode"]
            g.task = None
            return
        top = screen_of(obs, h["x"], h["y"] - 24)          # (its top: larvae sit at its foot)
        if top is None:
            hi.camera_minimap(h["x"], h["y"])
            return
        hi.click(*top)
