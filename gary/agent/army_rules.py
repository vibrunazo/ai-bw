"""Army rules a manifest turns on (docs/REFACTOR.md Phase 2), from Gary-Z's army; none Zerg-specific in
kind (a Terran could raid with wraiths, press with tanks, post a floating building):

  posts   watchers (Zerg: overlords) on the posts outside the enemy's natural, toward the map's middle and
          halfway there, once its home is known; before, one to each start it may be at (its scouting).
          One lost is replaced after POST_AGAIN.
  raid    the harassment flyers (Zerg: mutalisks) as one flock on a hotkey: formed from RAID_FLOCK by a
          town hall; out along the map's edge to a post beyond the mineral line with no static anti-air
          seen by it and the least anti-air; its workers picked off one at a time; backed off when the
          flock bleeds or anti-air comes within reach, again after a rest, another line after
          RAID_CYCLES; home under RAID_KEEP of the flock; at home, it defends the town halls.
  join    a cluster under JOIN_BELOW of the main one's size walks to it (each drawing its own way, they
          fought apart: Gary-Z's fighters died with 15% of its army there, the pros' 27%)
  sweep   the main body attacks an enemy packet in sight that it outweighs (theirs under SWEEP_ODDS of its
          own supply) within SWEEP_FAR, away from static defense
  press   far ahead (its army PRESS_RATIO times the enemy fighters seen in PRESS_MEMORY, PRESS_MIN at
          least) and nothing at home, the main body attacks the enemy building it knows of that ends a game
          first, the nearest within each rank; re-aimed every PRESS_AGAIN
  lords   an enemy overlord in sight with no fighters or anti-air of the enemy near it, within LORD_REACH of
          a free shooter of Gary's (Terran: a marine not in a fight or a bunker): the nearest LORD_HUNTERS
          hunt it, held by the rule (the fight and the army leave them be), re-aimed every LORD_REAIM, until
          it dies, gets away (LORD_LOST) or LORD_FOR is up. A pro Terran kills an exposed overlord with one or
          two marines (a supply block for the Zerg); Gary-T's marines shot one a few times and wandered off
          with their group (the user, after human game 4)

Each records its decisions (gary/decision_log.py) and works with Gary's hands: posts and the raid by their
own selections and minimap orders, the cluster rules by an army order to the motor.
"""

from __future__ import annotations

import math

from gary import commands as C
from gary.agent.gary import Gary
from gary.agent.intents import Intent, priority
from gary.decision_log import LOG
from gary.interface import screen_of
from gary.policy.army import is_building, is_worker, supply_x2

WORKERS = {7, 41, 64}
WATCHERS = {"Z": 42}                      # the race's posted watchers: the overlord
FLYERS = {"Z": {43}}                      # the race's raiding flyers: mutalisks
# what shoots air, by the enemy's race: unit -> worth in raiding flyers; the static ones (turrets, bunkers,
# spores, cannons) also stop a line
AIR_THREAT = {"T": {0: 1.0, 124: 3.0, 125: 4.0, 3: 2.0, 58: 2.0, 8: 1.5, 12: 4.0},
              "Z": {38: 1.0, 43: 1.0, 47: 1.5, 144: 3.0},
              "P": {66: 1.5, 60: 2.0, 70: 2.0, 67: 2.0, 162: 3.0}}
AIR_STATIC = {"T": {124, 125}, "Z": {144}, "P": {162}}
POST_OUT = 10 * 32
POST_AGAIN = 60 * 24
RAID_KEY = 9
RAID_MAX = 12                             # a flock is what a hotkey holds (more never matched the selection: the
                                          # hotkey pressed 70 times in 4 minutes, round G2)
RAID_FLOCK = 5
RAID_REST = 20 * 24
RAID_MEMORY = 90 * 24
RAID_POST = 7 * 32
RAID_TURRET = 8 * 32
RAID_SEE = 8 * 32
RAID_WAYPOINT = 14 * 32
RAID_MARINES = 3
RAID_HIT_HP = 30
RAID_HIT_S = 2
RAID_REST_S = 6
RAID_CYCLES = 3
RAID_SWITCH = 10 * 24
RAID_REAIM = 2 * 24
RAID_KEEP = 0.6
DEFEND_NEAR = 12 * 32
JOIN_BELOW = 1.0
SWEEP_FAR = 24 * 32
SWEEP_ODDS = 0.7
SWEEP_AGAIN = 8 * 24
PRESS_RATIO = 2.0
PRESS_MIN = 40
PRESS_MEMORY = 60 * 24
PRESS_AGAIN = 20 * 24
LORD_PREY = {"Z": {42}}                   # the enemy's flyers worth hunting alone, by its race: overlords
LORD_HUNTER = {"T": {0}, "Z": {38}, "P": {66}}   # Gary's shooters that hunt them, by its race: marines, hydralisks,
                                                 # dragoons (only Gary-T plays the rule yet)
LORD_REACH = 12 * 32
LORD_SAFE = 9 * 32                        # no enemy fighter, sunken or spore this near the overlord
LORD_HUNTERS = 2
LORD_REAIM = 3 * 24
LORD_FOR = 25 * 24
LORD_LOST = 16 * 32                       # the overlord this far from its hunters: let go
LORD_AGAIN = 5 * 24                       # the next hunt after one ends
LORD_BUNKER = 40                          # a marine this near a bunker is in it (or about to be)


def mid_of(units: list[dict]) -> tuple[float, float]:
    return sum(u["x"] for u in units) / len(units), sum(u["y"] for u in units) / len(units)


class EnemyMemory:
    """The enemy's buildings and fighters as seen (for the press, the sweep's odds, the raid's lines), and
    where its home is: the only start it may be at, or the start an enemy town hall was seen at."""

    def __init__(self, g: Gary):
        self.g = g
        self.buildings: dict[int, tuple[int, int, int]] = {}      # tag -> (x, y, type)
        self.fighters: dict[int, tuple[int, int, int, int]] = {}  # tag -> (frame, supply x2, x, y)
        self.air: dict[int, tuple[int, int, int, int]] = {}       # anti-air: tag -> (frame, x, y, type)
        self.seen_at = -1

    def look(self, obs: dict) -> None:
        g, f = self.g, self.g.hi.frame
        if f == self.seen_at:
            return
        self.seen_at = f
        threat = AIR_THREAT.get(g.enemy.race, {}) if g.enemy else {}
        for u in obs["units"]:
            if u["owner"] in (g.slot, 11):
                continue
            if u["type"] in threat:
                self.air[u["tag"]] = (f, u["x"], u["y"], u["type"])
            if is_building(u["type"]):
                self.buildings[u["tag"]] = (u["x"], u["y"], u["type"])
            elif supply_x2(u["type"]) > 0 and not is_worker(u["type"]):
                self.fighters[u["tag"]] = (f, supply_x2(u["type"]), u["x"], u["y"])
        self.fighters = {t: v for t, v in self.fighters.items() if f - v[0] <= PRESS_MEMORY}
        self.air = {t: v for t, v in self.air.items() if f - v[0] < RAID_MEMORY}

    def home(self):
        """The enemy's start (a Base), or None while unknown."""
        g = self.g
        starts = [g.map.base_near(s) for s in g.enemy_starts]
        if len(starts) == 1:
            return starts[0]
        halls = [(x, y) for x, y, t in self.buildings.values() if g.enemy and t in g.enemy.town_halls]
        for b in starts:
            if any(math.dist(b.center, h) < 8 * 32 for h in halls):
                return b
        return None

    def lines(self) -> list[tuple[float, float]]:
        """The enemy's mineral lines it can be raided at: its main's and natural's, and any base an enemy town
        hall was seen at."""
        g = self.g
        home = self.home()
        bases = [home, g.map.natural_of(home)] if home is not None else []
        halls = [(x, y) for x, y, t in self.buildings.values() if g.enemy and t in g.enemy.town_halls]
        bases += [b for b in g.map.bases if b.minerals and any(math.dist(b.center, h) < 6 * 32 for h in halls)]
        out = []
        for b in bases:
            if b is not None and b.minerals:
                line = mid_of([{"x": x, "y": y} for x, y in b.minerals])
                if all(math.dist(line, o) > 8 * 32 for o in out):
                    out.append(line)
        return out


class Posts:
    """Watchers on the posts outside the enemy's natural; before its home is known, one to each start."""

    def __init__(self, g: Gary, memory: EnemyMemory):
        self.g, self.memory = g, memory
        self.post_of: dict[int, int] = {}
        self.again: dict[int, int] = {}
        self.aimed = None
        self.points: list[tuple[int, int]] = []
        self.picked: int | None = None

    def _posts(self) -> list[tuple[int, int]]:
        g = self.g
        home = self.memory.home()
        key = home.center if home is not None else None
        if key != self.aimed or not self.points:
            self.aimed = key
            if home is None:
                self.points = [tuple(map(int, g.map.base_near(s).center)) for s in g.enemy_starts]
            else:
                nat = g.map.natural_of(home).center
                w, h = g.map.size
                d = math.dist(nat, (w / 2, h / 2)) or 1
                out = (int(nat[0] + (w / 2 - nat[0]) / d * POST_OUT), int(nat[1] + (h / 2 - nat[1]) / d * POST_OUT))
                mid = (int((g.natural.center[0] + out[0]) / 2), int((g.natural.center[1] + out[1]) / 2))
                self.points = [out, mid]
            self.post_of.clear()
        return self.points

    def step(self, obs: dict, mine: list[dict], done: list[dict]) -> str | None:
        g, hi = self.g, self.g.hi
        watcher = WATCHERS.get(g.race.letter)
        if watcher is None:
            return None
        lords = {u["tag"]: u for u in done if u["type"] == watcher}
        for k, post in enumerate(self._posts()):
            tag = self.post_of.get(k)
            if tag in lords or hi.frame < self.again.get(k, 0):
                continue
            if tag is not None:                        # (lost: another after a while)
                self.post_of.pop(k)
                self.again[k] = hi.frame + POST_AGAIN
                continue
            free = [u for u in lords.values() if u["tag"] not in self.post_of.values()]
            if not free:
                return None
            sel = set(obs["selection"])
            chosen = next((u for u in free if sel == {u["tag"]}), None)
            if chosen is None:
                u = min(free, key=lambda u: math.dist((u["x"], u["y"]), post))
                p = screen_of(obs, u["x"], u["y"])
                if p is None:
                    hi.camera_minimap(u["x"], u["y"])
                else:
                    hi.click(*p)
                return "army:posts"
            hi.minimap_right_click(*post)
            self.post_of[k] = chosen["tag"]
            what = "scouting the start" if self.aimed is None else "watch post"
            g.say(f"{g.race.names.get(watcher, 'watcher')} to {what} {k} at {post}")
            LOG.decision(g.hi.side, "army", hi.frame, f"{what} {k}", inputs={"at": post}, by="rule: posts")
            return "army:posts"
        return None


class Raid:
    """The flyers' raid on the enemy's mineral lines (Gary-Z's _mutas)."""

    def __init__(self, g: Gary, memory: EnemyMemory):
        self.g, self.memory = g, memory
        self.flock: set[int] = set()
        self.state, self.since, self.size = "home", 0, 0
        self.line = None
        self.cycles = 0
        self.hp: list[tuple[int, int]] = []
        self.raided_at: dict[tuple, int] = {}
        self.switched = -10 ** 6

    def step(self, obs: dict, mine: list[dict], done: list[dict]) -> str | None:
        g, hi = self.g, self.g.hi
        kinds = FLYERS.get(g.race.letter)
        if not kinds or g.enemy is None:
            return None
        flyers = {u["tag"]: u for u in done if u["type"] in kinds}
        self.flock &= set(flyers)
        flock = [flyers[t] for t in self.flock]
        enemy = [u for u in obs["units"] if u["owner"] not in (g.slot, 11)]
        threat = AIR_THREAT.get(g.enemy.race, {})
        static = AIR_STATIC.get(g.enemy.race, set())
        home = g.rally
        halls = [(u["x"], u["y"]) for u in mine if u["type"] in g.race.town_halls]
        raid = [u for u in enemy if not is_building(u["type"]) and not is_worker(u["type"]) and supply_x2(u["type"]) > 0
                and any(math.dist((u["x"], u["y"]), h) < DEFEND_NEAR for h in halls)]
        if raid and flock and self.state in ("home", "back"):
            return self._order("attack", mid_of(raid), "defend")
        if self.state == "defend":
            if not raid:
                self.state = "back"
            elif hi.frame - self.since >= 4 * 24:
                return self._order("attack", mid_of(raid), "defend")
            return None
        if self.state == "home":
            loose = [u for t, u in flyers.items() if t not in self.flock and any(math.dist((u["x"], u["y"]), h) < 16 * 32 for h in halls)]
            if loose and len(self.flock) < RAID_MAX and (self.flock or len(loose) >= RAID_FLOCK):
                return self._form(obs, flyers, loose)
            lines = self.memory.lines()
            if len(flock) >= RAID_FLOCK and hi.frame - self.since >= RAID_REST and lines:
                self.line = self._pick_line(lines)
                self.cycles, self.hp, self.size = 0, [], len(flock)
                return self._go(self.line)
            return None
        if not flock:
            self.state, self.since = "home", hi.frame
            return None
        mid = mid_of(flock)
        hp = sum(u["hp"] for u in flock)
        self.hp = [(f, h) for f, h in self.hp if hi.frame - f <= RAID_HIT_S * 24] + [(hi.frame, hp)]
        bleeding = self.hp[0][1] - hp >= RAID_HIT_HP
        static_near = any(v[3] in static and math.dist(v[1:3], mid) < RAID_TURRET for v in self.memory.air.values()
                          if hi.frame - v[0] < 5 * 24)
        shooters = sum(1 for u in enemy if u["type"] in threat and u["type"] not in static and math.dist((u["x"], u["y"]), mid) < RAID_SEE)
        if self.state in ("out", "hit", "rest"):
            if len(flock) < self.size * RAID_KEEP:
                g.say(f"raid home: {len(flock)} of {self.size} left")
                return self._order("move", home, "back")
            seen = {"flyers": len(flock), "shooters": shooters, "static": static_near, "hp_lost": self.hp[0][1] - hp,
                    "cycles": self.cycles}
            if self.state == "hit" and (bleeding or static_near or shooters >= RAID_MARINES):
                self.cycles += 1
                LOG.decision(g.hi.side, "raid", hi.frame, "back off", inputs=seen,
                             by="rule: the flock bleeds, or anti-air within reach")
                return self._order("move", self._post(self.line), "rest")
            if self.state == "out":
                turreted = any(v[3] in static and math.dist(v[1:3], self.line) < RAID_TURRET + 3 * 32 for v in self.memory.air.values())
                if turreted and hi.frame - self.switched >= RAID_SWITCH:
                    other = self._pick_line(self.memory.lines(), but=self.line)
                    self.switched = hi.frame
                    LOG.decision(g.hi.side, "raid", hi.frame, "another line" if other else "home", inputs=seen,
                                 by="rule: static anti-air seen by that line on the way in")
                    if other is None:
                        return self._order("move", home, "back")
                    self.line, self.cycles = other, 0
                    return self._go(self.line)
                if bleeding or shooters >= RAID_MARINES:
                    self.cycles += 1
                    LOG.decision(g.hi.side, "raid", hi.frame, "back off", inputs=seen, by="rule: shot at on the way in")
                    return self._order("move", self._post(self.line), "rest")
                if math.dist(mid, self._post(self.line)) < 5 * 32:
                    self.state, self.since = "hit", hi.frame
            if self.state == "rest" and hi.frame - self.since >= RAID_REST_S * 24:
                if self.cycles >= RAID_CYCLES:
                    other = self._pick_line(self.memory.lines(), but=self.line)
                    if other is None:
                        return self._order("move", home, "back")
                    self.line, self.cycles = other, 0
                    return self._go(self.line)
                self.state, self.since = "hit", hi.frame
            if self.state == "hit" and hi.frame - self.since >= RAID_REAIM:
                workers = [u for u in enemy if u["type"] in WORKERS and math.dist((u["x"], u["y"]), self.line) < 10 * 32]
                if workers:                                # (one worker at a time: a right-click on it)
                    w = min(workers, key=lambda u: ((u["hp"] or 0), math.dist((u["x"], u["y"]), mid)))
                    if not self._selected(obs):
                        hi.hotkey_recall(RAID_KEY)
                        return "army:raid"
                    p = screen_of(obs, w["x"], w["y"])
                    if p is None:
                        hi.camera_minimap(w["x"], w["y"])
                        return "army:raid"
                    hi.right_click(*p)
                    self.since = hi.frame
                    self.raided_at[tuple(self.line)] = hi.frame
                    return "army:raid"
                return self._order("attack", self.line, "hit")
            return None
        if self.state == "back" and math.dist(mid, home) < 6 * 32:
            self.state, self.since = "home", hi.frame
        return None

    def _form(self, obs: dict, flyers: dict, loose: list[dict]) -> str:
        """The flyers by a town hall onto the hotkey: a ctrl+click takes every one on the screen."""
        g, hi = self.g, self.g.hi
        sel = set(obs["selection"])
        if sel and sel <= set(flyers) and sel & {u["tag"] for u in loose}:
            (hi.hotkey_add if self.flock else hi.hotkey_set)(RAID_KEY)
            self.flock |= set(list(sel)[:RAID_MAX - len(self.flock)])
            self.size = len(self.flock)
            g.say(f"raid: a flock of {len(self.flock)}")
            return "army:raid"
        u = loose[0]
        p = screen_of(obs, u["x"], u["y"])
        if p is None:
            hi.camera_minimap(u["x"], u["y"])
        else:
            hi.ctrl_click(*p)
        return "army:raid"

    def _order(self, kind: str, point, then: str) -> str:
        """The flock selected (its hotkey), then a minimap order."""
        g, hi = self.g, self.g.hi
        obs = hi.observe()
        if not self._selected(obs):
            hi.hotkey_recall(RAID_KEY)
            return "army:raid"
        x, y = int(point[0]), int(point[1])
        if kind == "move":
            hi.minimap_right_click(x, y)
        else:
            hi.minimap_command(C.ORDER_ATTACK_MOVE, x, y)
        if then != self.state:
            g.say(f"raid ({len(self.flock)}): {then}")
        self.state, self.since = then, hi.frame
        return "army:raid"

    def _go(self, line) -> str:
        """Out to the line's post along the map's edge: a waypoint aside of the post on Gary's side, then the
        post, queued (straight from home, the flight crossed the enemy's base and its anti-air)."""
        g, hi = self.g, self.g.hi
        obs = hi.observe()
        if not self._selected(obs):
            hi.hotkey_recall(RAID_KEY)
            return "army:raid"
        post = self._post(line)
        w, h = g.map.size
        ox, oy = line[0] - w / 2, line[1] - h / 2
        d = math.hypot(ox, oy) or 1.0
        px, py = -oy / d, ox / d
        home = g.main.center
        side = 1 if (px * (home[0] - post[0]) + py * (home[1] - post[1])) >= 0 else -1
        wp = (int(min(w - 64, max(64, post[0] + side * px * RAID_WAYPOINT))),
              int(min(h - 64, max(64, post[1] + side * py * RAID_WAYPOINT))))
        hi.minimap_right_click(*wp)
        hi.minimap_right_click(*post, queued=True)
        if self.state != "out":
            g.say(f"raid ({len(self.flock)}): out")
            LOG.decision(g.hi.side, "raid", hi.frame, f"out to ({int(line[0]) // 32},{int(line[1]) // 32})",
                         inputs={"flyers": len(self.flock)}, by="rule: the line with the least anti-air seen")
        self.state, self.since = "out", hi.frame
        return "army:raid"

    def _selected(self, obs: dict) -> bool:
        """The flock (or what of it its hotkey brings up) is the selection."""
        sel = set(obs["selection"])
        return bool(sel) and sel <= self.flock

    def _pick_line(self, lines: list, but=None):
        """No static anti-air seen by it first, then the least anti-air near it, then the one raided longest ago."""
        lines = [l for l in lines if but is None or math.dist(l, but) > 2 * 32]
        if not lines:
            return None
        g = self.g
        threat = AIR_THREAT.get(g.enemy.race, {})
        static = AIR_STATIC.get(g.enemy.race, set())
        air = self.memory.air.values()

        def key(line):
            turrets = sum(1 for v in air if v[3] in static and math.dist(v[1:3], line) < RAID_TURRET + 3 * 32)
            shooters = sum(threat.get(v[3], 0) for v in air if v[3] not in static and math.dist(v[1:3], line) < 10 * 32)
            return turrets, shooters, self.raided_at.get(tuple(line), -10 ** 6)
        return min(lines, key=key)

    def _post(self, line) -> tuple[int, int]:
        """RAID_POST beyond the line, out from the map's middle (the minerals between it and the base)."""
        w, h = self.g.map.size
        dx, dy = line[0] - w / 2, line[1] - h / 2
        d = math.hypot(dx, dy) or 1.0
        return int(min(w - 64, max(64, line[0] + dx / d * RAID_POST))), int(min(h - 64, max(64, line[1] + dy / d * RAID_POST)))


class LordHunt:
    """One or two shooters on an exposed overlord (see the module's docstring). STOP-GAP (AGENTS.md): until the
    fight or army layer decides small detachments itself."""

    def __init__(self, g: Gary, memory: EnemyMemory):
        self.g, self.memory = g, memory
        self.prey: int | None = None
        self.hunters: list[int] = []
        self.since = self.last_aim = 0
        self.next_hunt = 0

    def _end(self, why: str) -> None:
        g = self.g
        g.leases.release(self.hunters)
        LOG.decision(g.hi.side, "army", g.hi.frame, "overlord hunt over", inputs={"hunters": len(self.hunters)}, by=why)
        self.prey, self.hunters = None, []
        self.next_hunt = g.hi.frame + LORD_AGAIN

    def step(self, obs: dict, mine: list[dict], done: list[dict]) -> str | None:
        g, hi = self.g, self.g.hi
        if g.enemy is None:
            return None
        prey_types = LORD_PREY.get(g.enemy.race, set())
        hunter_types = LORD_HUNTER.get(g.race.letter, set())
        if not prey_types or not hunter_types:
            return None
        enemy = [u for u in obs["units"] if u["owner"] not in (g.slot, 11)]
        guards = [(u["x"], u["y"]) for u in enemy if u["type"] in (144, 146) or (
            u["type"] not in prey_types and not is_building(u["type"]) and not is_worker(u["type"]) and supply_x2(u["type"]) > 0)]
        if self.prey is not None:
            prey = next((u for u in enemy if u["tag"] == self.prey), None)
            hunters = [u for u in done if u["tag"] in set(self.hunters)]
            if prey is None or not hunters:
                self._end("rule: the overlord is gone (killed or out of sight)" if hunters else "rule: its hunters are gone")
                return None
            if hi.frame - self.since > LORD_FOR or min(math.dist((u["x"], u["y"]), (prey["x"], prey["y"])) for u in hunters) > LORD_LOST \
                    or any(math.dist(q, (prey["x"], prey["y"])) < LORD_SAFE for q in guards):
                self._end("rule: the hunt took too long, the overlord got away, or the enemy came")
                return None
            if hi.frame - self.last_aim < LORD_REAIM:
                return None
            return self._aim(obs, hunters, prey)
        if hi.frame < self.next_hunt:
            return None
        bunkers = [(u["x"], u["y"]) for u in mine if u["type"] == 125]
        free = [u for u in done if u["type"] in hunter_types and priority(g.leases.holder(u["tag"]) or "") < 2
                and all(math.dist((u["x"], u["y"]), b) > LORD_BUNKER for b in bunkers)]
        if not free:
            return None
        for prey in (u for u in enemy if u["type"] in prey_types):
            at = (prey["x"], prey["y"])
            if any(math.dist(q, at) < LORD_SAFE for q in guards):
                continue
            near = sorted(free, key=lambda u: math.dist((u["x"], u["y"]), at))[:LORD_HUNTERS]
            if math.dist((near[0]["x"], near[0]["y"]), at) > LORD_REACH:
                continue
            self.prey, self.hunters, self.since, self.last_aim = prey["tag"], [u["tag"] for u in near], hi.frame, -10 ** 6
            g.say(f"{len(near)} {'marine' if len(near) == 1 else 'marines'} hunt an overlord at {tuple(map(int, at))}")
            LOG.decision(hi.side, "army", hi.frame, "hunt an overlord", inputs={"at": [int(at[0]), int(at[1])], "hunters": len(near)},
                         by="rule: an overlord in sight with no fighters or anti-air near it")
            return self._aim(obs, near, prey)
        return None

    def _aim(self, obs: dict, hunters: list[dict], prey: dict) -> str:
        """The hunters selected (a click on the first, a shift+click on the other), then an attack-move on the
        overlord's spot on the minimap; held by the rule till the next aim."""
        g, hi = self.g, self.g.hi
        tags = {u["tag"] for u in hunters}
        sel = set(obs["selection"])
        if sel != tags:
            u = next((u for u in hunters if u["tag"] not in sel), hunters[0]) if sel and sel <= tags else hunters[0]
            p = hi.aim(obs, u)
            if p is None:
                hi.camera_minimap(u["x"], u["y"])
            else:
                hi.click(*p, shift=bool(sel) and sel <= tags)
            return "army:lords"
        hi.minimap_command(C.ORDER_ATTACK_MOVE, int(prey["x"]), int(prey["y"]))
        g.leases.take(list(tags), Intent("HUNT", list(tags), prey["tag"], source="rule:lords", ttl=LORD_REAIM + 24))
        self.last_aim = hi.frame
        return "army:lords"


class ClusterRules:
    """join, sweep and press, asked for a cluster at its turn before the army model (army.py LearnedArmy)."""

    def __init__(self, g: Gary, memory: EnemyMemory, join: bool, sweep: bool, press: bool, cell_of, cell_dist):
        self.g, self.memory = g, memory
        self.join, self.sweep, self.press = join, sweep, press
        self.cell_of, self.cell_dist = cell_of, cell_dist
        self.sweep_after = self.press_after = 0

    def step(self, obs: dict, mine: list[dict], clusters: list[list[dict]], group: list[dict]) -> str | None:
        main = clusters[0]
        if group is not main:
            return self._join(group, main) if self.join else None
        if self.sweep and (who := self._sweep(obs, group)):
            return who
        if self.press and (who := self._press(obs, mine, group)):
            return who
        return None

    def _send(self, group: list[dict], point, attack: bool, p: float = 1.0) -> str:
        x, y = int(point[0]), int(point[1])
        self.g.motor.start(Intent("ARMY_ATTACK" if attack else "ARMY_MOVE", [u["tag"] for u in group], (x, y),
                                  {"tags": [u["tag"] for u in group], "x": x, "y": y, "attack": attack, "p": p,
                                   "fight": 0.0, "done": [], "boxes": 0}, "army"))
        return "army"

    def _join(self, group: list[dict], main: list[dict]) -> str | None:
        size = lambda c: sum(supply_x2(u["type"]) for u in c)
        if size(group) >= JOIN_BELOW * size(main):
            return None
        mid, to = mid_of(group), mid_of(main)
        if self.cell_dist(self.cell_of(*mid), self.cell_of(*to)) < 2:
            return None
        g = self.g
        LOG.decision(g.hi.side, "army", g.hi.frame, f"move to ({int(to[0]) // 32},{int(to[1]) // 32})",
                     inputs={"units": len(group), "main": len(main)},
                     by=f"rule: a cluster under {JOIN_BELOW:.0%} of the main one joins it", then="sent")
        return self._send(group, to, False)

    def _enemy_fighters(self, obs: dict) -> list[dict]:
        g = self.g
        return [u for u in obs["units"] if u["owner"] not in (g.slot, 11) and supply_x2(u["type"]) > 0
                and not is_worker(u["type"]) and not is_building(u["type"])]

    def _sweep(self, obs: dict, group: list[dict]) -> str | None:
        g, hi = self.g, self.g.hi
        if hi.frame < self.sweep_after or g.enemy is None:
            return None
        mid = mid_of(group)
        near = [e for e in self._enemy_fighters(obs) if math.dist((e["x"], e["y"]), mid) < SWEEP_FAR]
        if not near:
            return None
        theirs = sum(supply_x2(e["type"]) for e in near)
        ours = sum(supply_x2(u["type"]) for u in group)
        if theirs > SWEEP_ODDS * ours:
            return None
        at = mid_of(near)
        if any(u["owner"] not in (g.slot, 11) and u["type"] in g.enemy.static and math.dist((u["x"], u["y"]), at) < 9 * 32
               for u in obs["units"]):
            return None
        self.sweep_after = hi.frame + SWEEP_AGAIN
        LOG.decision(g.hi.side, "army", hi.frame, f"attack to ({int(at[0]) // 32},{int(at[1]) // 32})",
                     inputs={"units": len(group), "ours": ours / 2, "theirs": theirs / 2},
                     by="rule: the main body sweeps a packet it outweighs within reach", then="sent")
        return self._send(group, at, True)

    def _press(self, obs: dict, mine: list[dict], group: list[dict]) -> str | None:
        g, hi = self.g, self.g.hi
        mem = self.memory
        if hi.frame < self.press_after or not mem.buildings or g.enemy is None:
            return None
        own = sum(supply_x2(u["type"]) for u in mine if u["type"] in g.army_types and u["completed"])
        seen = sum(v[1] for v in mem.fighters.values())
        if own < PRESS_MIN or own < PRESS_RATIO * max(seen, 8):
            return None
        halls = [(u["x"], u["y"]) for u in mine if u["type"] in g.race.town_halls]
        if any(math.dist((e["x"], e["y"]), h) < 12 * 32 for e in self._enemy_fighters(obs) for h in halls):
            return None
        mid = mid_of(group)
        alive = {u["tag"] for u in obs["units"]}
        mem.buildings = {t: v for t, v in mem.buildings.items()
                         if t in alive or not any(math.dist((u["x"], u["y"]), v[:2]) < 6 * 32 for u in group)}
        if not mem.buildings:            # (a remembered building the body stands by and doesn't see: gone)
            return None
        rank = lambda t: 0 if t in g.enemy.town_halls else 1 if t in g.enemy.production else 2
        x, y, t = min(mem.buildings.values(), key=lambda v: (rank(v[2]), math.dist(v[:2], mid)))
        self.press_after = hi.frame + PRESS_AGAIN
        LOG.decision(g.hi.side, "army", hi.frame, f"attack to ({x // 32},{y // 32})",
                     inputs={"units": len(group), "ours": own / 2, "theirs_seen": seen / 2,
                             "target": g.enemy.names.get(t, str(t))},
                     by=f"rule: {PRESS_RATIO:g}x the enemy fighters seen, nothing at home: presses the nearest building",
                     then="sent")
        return self._send(group, (x, y), True)
