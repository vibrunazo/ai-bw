"""Zerg v3: a greedy Zerg that punishes a passive Terran (TODO.md item 4).

Every other opponent in the pool attacks early (the 9-pool, Zerg v2's 6:00 all-in, the replayed
pros' armies, which come with no economy behind them), so a Terran that over-defends is never made
to pay: a decision learned against them learns that defending is free. This one plays for economy
and makes Gary pay for staying home. On Zerg v2's code, through the same human interface:

  greedy     12 hatch, pool, extractor; a third hatchery at the base nearest its natural at THIRD_AT
             drones; drones to DRONES (three bases' worth); macro hatcheries when money piles up
             (v2's); a few zerglings (LINGS_EARLY) for its own safety
  lings/mutas the standard ZvT 3-hatch mutalisks: a Lair from LAIR_FROM, a second extractor with it,
             a Spire once the Lair is done, zergling speed; then mutalisks from every larva the
             drones don't take, zerglings with the minerals left over
  mutalisks  one flock (hotkey 9): from MUTA_FLOCK of them it flies at the mineral line of Gary's
             base with the least anti-air seen, shoots SCVs there (re-aimed every few seconds), and
             flies home when what shoots air near it (marines, turrets, bunkers, goliaths) outweighs
             it or it's lost half its mutalisks; MUTA_REST later it goes again, the other base first
  defense    from 6:00 a standing army of zerglings (LINGS_AT, LINGS_A_MINUTE more each minute)
             and SUNKENS sunken colonies in front of its natural, from SUNKENS_FROM
  home       its zerglings wait in front of its natural (not outside Gary's), and go home at once to
             any of Gary's fighters near its hatcheries (Zerg v1's defense); its groups attack-move
             everywhere but when running away (on a move order they walked through Gary's army
             without shooting back)
  attack     one big attack at ATTACK_AT, the zerglings and the flock, with no retreat (Zerg v2's
             all-in): its size is what its economy made
  poke       on seeds where seed % 3 == 0, its first zerglings go to the staging point outside
             Gary's natural at about 3:00, stay POKE_FOR, and come home without fighting: the
             moment a false early-pool alarm is tempting

Buildings, as a player places them through the interface (v2's way failed most of the time here:
a box around a mining drone takes its neighbors, a build order to a mining drone becomes a move,
the click's scatter or Gary's scout on the spot fails the placement, the money is spent while
the drone walks): a lone drone is clicked, walked to the spot by a minimap order, selected there
(alone, so a click takes it) and the building placed; if it doesn't appear, placed again, on a
free tile next to it if the spot is taken. Its cost is kept from the walk until it appears.

    python -m eval.vs_rush --versions v020 --opponent v3 --seeds 1 2 3 4
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from gary import commands as C
from gary.bots.zerg_rush import DRONE, HATCH, LING, MINING, OVERLORD, POOL
from gary.bots.zerg_v2 import EXTRACTOR, HATCHERIES, IDLE_ORDERS, ZergV2
from gary.interface import screen_of

LAIR, SPIRE, MUTA = 132, 141, 43
CREEP, SUNKEN = 143, 146
METABOLIC_BOOST = 27
DRONES = 44
HATCH_AT = 12
THIRD_AT = 16
LAIR_FROM = 200 * 24            # the Lair from 3:20 (the 3-hatch mutalisks' timing)
GAS_EACH = 3
GAS_UNSEEN = 10 * 24
LINGS_EARLY = 6
ATTACK_AT = 11 * 60 * 24
MUTAS_FIRST = 7 * 60 * 24       # from 7:00 on, mutalisks come before drones
MUTA_FLOCK = 7                  # mutalisks before the flock first goes out
MUTA_REST = 20 * 24             # home this long between raids
MUTA_REAIM = 4 * 24             # out: aimed again at the SCVs near this often
MUTA_SEE = 8 * 32               # anti-air this near the flock counts
MUTA_ODDS = 0.8                 # home when the anti-air near is worth this many mutalisks each
AIR_THREAT = {0: 1.0, 124: 3.0, 125: 4.0, 3: 2.0, 58: 2.0, 8: 1.5, 12: 4.0}   # marine, turret, bunker,
                                # goliath, valkyrie, wraith, battlecruiser: in mutalisks
MUTA_KEY = 9
LAIR_RETRY = 120 * 24           # a Lair ordered and not seen: ordered again after this (a morphing hatchery
                                # still shows as a Hatchery: sooner, it would morph a second one)
DEFEND_NEAR = 12 * 32           # Gary's fighters this near a hatchery: the flock defends
LINGS_FROM_MINERALS = 250       # with a Spire, zerglings from the minerals above this (mutalisks take the gas)
LINGS_FROM, LINGS_AT, LINGS_A_MINUTE = 6 * 60 * 24, 12, 4   # from 6:00 it keeps at least LINGS_AT zerglings,
                                # LINGS_A_MINUTE more each minute (a timing push meets an army)
SUNKENS, SUNKENS_FROM = 2, 330 * 24     # sunken colonies at the natural, from 5:30
POKE_FROM, POKE_UNTIL = 170 * 24, 270 * 24
POKE_LINGS = 4
POKE_FOR = 15 * 24
COST = {HATCH: 300, POOL: 200, EXTRACTOR: 50, SPIRE: 200, CREEP: 75}
GAS_COST = {SPIRE: 200}
SIZE = {HATCH: (4, 3), POOL: (3, 2), EXTRACTOR: (4, 2), SPIRE: (2, 2), CREEP: (2, 2)}   # tiles
AT_SPOT = 3 * 32                # the drone this near the spot: place the building
PLACED_WAIT = 4 * 24            # placed and not there after this, the drone standing: place again
PICK_TRIES = 4                  # clicks to get a drone alone before trying another
PLACE_CLICKS = 6                # clicks on the drone at its spot before starting over with another
JOB_TRIES = 8                   # placements before a building is given up


@dataclass
class Job:
    """A building on its way: pick a drone, walk it there, place, check."""
    step: str
    unit_type: int
    tile: tuple[int, int]       # top-left
    state: str = "pick"
    drone: int | None = None
    since: int = 0
    tries: int = 0
    picks: int = 0
    clicks: int = 0             # at the spot: clicks to select its drone so far


class ZergV3(ZergV2):
    sizes = {LING: 1.0}                  # (Zerg v1's groups: zerglings; the mutalisks are a flock of their own)

    def __init__(self, hi, mapinfo, seed: int = 0):
        super().__init__(hi, mapinfo)
        nat = self.nat_base.center
        others = [b for b in mapinfo.bases if b is not self.main_base and b is not self.nat_base and not b.start
                  and b.minerals]
        self.third_base = min(others, key=lambda b: math.dist(b.center, nat)) if others else None
        self.poke = seed % 3 == 0
        self.poke_group = None                 # (group number, frame it got there)
        self.poked = False
        self.jobs: dict[str, Job] = {}         # step -> the building on its way
        self.built: set[str] = set()           # steps whose building went up (or were given up)
        self.gas_of: dict[int, int] = {}       # gas drone -> its extractor
        self.gas_seen: dict[int, int] = {}     # drone -> frame last seen
        self.lair_at = None                    # frame the Lair was ordered
        self.flock: set[int] = set()           # the mutalisks on hotkey 9
        self.flock_state, self.flock_since, self.flock_size = "home", 0, 0
        self.flock_target = None               # where it's raiding
        self.raids = 0
        self.air_seen: dict[int, tuple] = {}   # Gary's anti-air seen: tag -> (frame, x, y, type)

    # --- where the army waits, and when it goes ---------------------------------------------------

    def _aim(self) -> None:
        super()._aim()
        self.gary_staging = self.staging       # (outside Gary's natural: for the poke and the attack)
        main = self.map.base_near(self.target)
        nat = self.map.natural_of(main)
        mean = lambda pts: (sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts))
        self.lines = [mean(b.minerals) for b in (main, nat) if b is not None and b.minerals]   # (raid targets)
        if not self.all_in:
            self.staging = self.gather         # its army waits at home

    def _order(self, n: int, obs: dict, kind: str, point, then: str) -> bool:
        """Zerg v1's group order, with every move an attack-move but a retreat or the poke: on a move
        order its units walked through Gary's army without shooting back."""
        if kind == "move" and then not in ("retreat", "poke") and not (then == "staging" and self.phase.get(n) == "poke"):
            kind = "attack"
        return super()._order(n, obs, kind, point, then)

    def _go(self, mine) -> bool:
        return self.hi.frame >= ATTACK_AT

    def _timing(self, mine) -> None:
        was = self.all_in
        super()._timing(mine)
        if self.all_in and not was and self._aimed_at is not None:
            self.staging = self.gary_staging   # muster outside Gary's natural: every group moves there
            for n in self.groups:
                if self.phase[n] in ("staged", "staging"):
                    self.phase[n] = "to_staging"

    def army(self, obs: dict, mine: list[dict]) -> bool:
        if self.poke and not self.poked and not self.all_in and self._poke(obs, mine):
            return True
        if self._mutas(obs, mine):
            return True
        return super().army(obs, mine)

    # --- the mutalisk flock ----------------------------------------------------------------------

    def _mutas(self, obs, mine) -> bool:
        """The flock: formed at home, raids a mineral line, home when outweighed (True: it acted)."""
        hi = self.hi
        if self._aimed_at != self.target:
            self._aim()
        mutas = {u["tag"]: u for u in mine if u["type"] == MUTA and u["completed"]}
        self.flock &= set(mutas)
        for u in obs["units"]:                     # (Gary's anti-air, remembered a minute)
            if u["owner"] not in (hi.slot, 11) and u["type"] in AIR_THREAT:
                self.air_seen[u["tag"]] = (hi.frame, u["x"], u["y"], u["type"])
        self.air_seen = {t: v for t, v in self.air_seen.items() if hi.frame - v[0] < 60 * 24}
        flock = [mutas[t] for t in self.flock]
        home = self.gather
        hatcheries = [(u["x"], u["y"]) for u in mine if u["type"] in HATCHERIES]
        raid = [u for u in obs["units"] if u["owner"] not in (hi.slot, 11) and u["type"] not in (7, 125, 124)
                and u["type"] in set(AIR_THREAT) | {32, 34, 1, 2, 5, 30}
                and any(math.dist((u["x"], u["y"]), h) < DEFEND_NEAR for h in hatcheries)]
        if raid and flock and self.flock_state in ("home", "back") and self.all_in != "go":
            at = (sum(u["x"] for u in raid) / len(raid), sum(u["y"] for u in raid) / len(raid))
            if self.flock_state != "defend" or hi.frame - self.flock_since >= MUTA_REAIM:
                return self._flock_order("attack", at, "defend")
        if self.flock_state == "defend" and not raid:
            self.flock_state = "back"
        if self.flock_state == "home":
            loose = [u for t, u in mutas.items() if t not in self.flock and math.dist((u["x"], u["y"]), home) < 12 * 32]
            if loose and (self.flock or len(loose) >= MUTA_FLOCK):
                return self._form_flock(obs, mutas, loose)
            ready = len(flock) >= MUTA_FLOCK and hi.frame - self.flock_since >= MUTA_REST
            if (ready or (self.all_in == "go" and flock)) and self.lines:
                self.flock_target = self.lines[0] if self.all_in == "go" else self._raid_target()
                return self._flock_order("attack", self.flock_target, "out")
            return False
        if not flock:
            self.flock_state, self.flock_since = "home", hi.frame
            return False
        mid = (sum(u["x"] for u in flock) / len(flock), sum(u["y"] for u in flock) / len(flock))
        if self.flock_state == "out":
            near = [v for v in self.air_seen.values() if hi.frame - v[0] < 5 * 24 and math.dist(v[1:3], mid) < MUTA_SEE]
            threat = sum(AIR_THREAT[v[3]] for v in near)
            lost = len(flock) < self.flock_size / 2
            if self.all_in != "go" and (threat > MUTA_ODDS * len(flock) or lost):
                self.log(f"mutalisks home: {len(flock)} against {threat:.0f} anti-air")
                return self._flock_order("move", home, "back")
            if hi.frame - self.flock_since >= MUTA_REAIM:
                scvs = [u for u in obs["units"] if u["owner"] not in (hi.slot, 11) and u["type"] == 7
                        and math.dist((u["x"], u["y"]), mid) < 10 * 32]
                at = (sum(u["x"] for u in scvs) / len(scvs), sum(u["y"] for u in scvs) / len(scvs)) if scvs else self.flock_target
                return self._flock_order("attack", at, "out")
            return False
        if self.flock_state == "defend":
            if hi.frame - self.flock_since >= MUTA_REAIM and raid:
                at = (sum(u["x"] for u in raid) / len(raid), sum(u["y"] for u in raid) / len(raid))
                return self._flock_order("attack", at, "defend")
            return False
        if self.flock_state == "back" and math.dist(mid, home) < 6 * 32:
            self.flock_state, self.flock_since = "home", hi.frame
        return False

    def _raid_target(self):
        """Gary's mineral line with the least anti-air seen lately (the one not raided last, on a tie)."""
        self.raids += 1

        def aa(line):
            return sum(AIR_THREAT[v[3]] for v in self.air_seen.values() if math.dist(v[1:3], line) < 10 * 32)
        return min(self.lines, key=lambda line: (aa(line), (self.lines.index(line) + self.raids) % len(self.lines)))

    def _form_flock(self, obs, mutas, loose) -> bool:
        """The mutalisks at home onto hotkey 9: a ctrl+click takes every one on the screen."""
        hi = self.hi
        sel = set(obs["selection"])
        if sel and sel <= set(mutas) and sel & {u["tag"] for u in loose}:
            if self.flock:
                hi.hotkey_add(MUTA_KEY)             # (new ones join the flock: shift+the hotkey)
            else:
                hi.hotkey_set(MUTA_KEY)
            self.flock |= sel
            self.flock_size = len(self.flock)
            self.log(f"mutalisks: a flock of {len(self.flock)}")
            return True
        u = loose[0]
        if self._look(obs, (u["x"], u["y"])):
            return True
        at = screen_of(obs, u["x"], u["y"])
        if at is None:
            return False
        hi.ctrl_click(*at)
        return True

    def _flock_order(self, kind: str, point, then: str) -> bool:
        """The flock selected (hotkey 9), then a minimap order."""
        hi = self.hi
        obs = hi.observe()
        if set(obs["selection"]) != self.flock & {u["tag"] for u in obs["units"]} or not obs["selection"]:
            hi.hotkey_recall(MUTA_KEY)
            return True
        x, y = int(point[0]), int(point[1])
        if kind == "move":
            hi.minimap_right_click(x, y)
        else:
            hi.minimap_command(C.ORDER_ATTACK_MOVE, x, y)
        if then != self.flock_state:
            self.log(f"mutalisks ({len(self.flock)}): {then}")
        self.flock_state, self.flock_since = then, hi.frame
        return True

    def _poke(self, obs, mine) -> bool:
        """Its first zerglings to outside Gary's natural for POKE_FOR, then home: a group of POKE_LINGS."""
        hi = self.hi
        if self._aimed_at != self.target:
            self._aim()
        if self.poke_group is None:
            if not POKE_FROM <= hi.frame <= POKE_UNTIL:
                if hi.frame > POKE_UNTIL:
                    self.poked = True
                return False
            if self.wave != POKE_LINGS:
                self.wave = POKE_LINGS             # (the next group formed is the poke)
            new = [n for n, ph in self.phase.items() if ph == "to_staging"]
            if new:
                self.poke_group = (new[0], None)
                self.wave = 12
                self.log(f"poke: group {new[0]} to Gary's natural")
                return self._order(new[0], obs, "move", self.gary_staging, "poke")
            return False
        n, there = self.poke_group
        if n not in self.groups:                   # (it died)
            self.poked = True
            return False
        members = [u for u in mine if u["tag"] in self.groups[n]]
        if there is None:
            if members and sum(math.dist((u["x"], u["y"]), self.gary_staging) < 6 * 32 for u in members) * 2 >= len(members):
                self.poke_group = (n, hi.frame)
            elif hi.frame - self.since[n] > 40 * 24:   # (it didn't get there: home anyway)
                self.poke_group = (n, hi.frame - POKE_FOR)
            return False
        if hi.frame - there >= POKE_FOR:
            self.poked = True
            self.log(f"poke: group {n} home")
            return self._order(n, obs, "move", self.gather, "staging")
        return False

    # --- the turn --------------------------------------------------------------------------------

    def act(self) -> None:
        hi = self.hi
        if hi.pending:
            return
        obs = hi.observe()
        me = obs["me"]
        mine = [u for u in obs["units"] if u["owner"] == hi.slot]
        self._timing(mine)
        if self.army(obs, mine):
            return
        hatches = [u for u in mine if u["type"] in HATCHERIES]
        pool = next((u for u in mine if u["type"] == POOL), None)
        drones = [u for u in mine if u["type"] == DRONE and u["completed"]]
        eggs_drones = len(drones) + sum(1 for u in mine if u["type"] == 36)
        if self.task:                                   # (v2's steps: a gas drone, a hatchery's rally)
            return self._continue(obs, mine, me)
        self._plan(mine, me, hatches, pool, drones)
        for job in list(self.jobs.values()):
            if self._work(job, obs, mine, me, drones):
                return
        if self._tech(obs, mine, me, hatches, pool):
            return
        if self._gas(obs, mine, me, drones):
            return
        if self._mine_idle(obs, mine, [d for d in drones if d["tag"] not in self._builders()], hatches):
            return
        if self._rallies(obs, hatches, pool):
            return
        if me["minerals"] - self._kept() < 50:          # (the money is kept for a building on its way)
            return
        want = self._want(me, mine, hatches, pool, eggs_drones)
        if want is None or me["supply_used"] + (0 if want == OVERLORD else 1) > me["supply_max"]:
            return
        self._larva(obs, mine, want)

    # --- buildings -------------------------------------------------------------------------------

    def _plan(self, mine, me, hatches, pool, drones) -> None:
        """New jobs: the buildings due now (each once)."""
        n = len(drones)
        exts = [u for u in mine if u["type"] == EXTRACTOR]

        def due(step, unit_type, spot, when):
            """spot: a function for the building's tile (asked only when it's due)."""
            if step in self.built or step in self.jobs or not when:
                return
            tile = spot()
            if tile is not None:
                self.jobs[step] = Job(step, unit_type, tuple(tile), since=self.hi.frame)
                self.log(f"builds {step} at {tile} ({n} drones)")
        due("hatch", HATCH, lambda: self.nat_base.tile, n >= HATCH_AT)
        due("pool", POOL, self._pool_spot, "hatch" in self.jobs or "hatch" in self.built)
        due("extractor", EXTRACTOR, lambda: self._geyser_tile(self.main_base), "pool" in self.jobs or "pool" in self.built)
        if self.third_base is not None:
            due("third", HATCH, lambda: self.third_base.tile, n >= THIRD_AT and "hatch" in self.built)
        lair = next((u for u in mine if u["type"] == LAIR), None)
        due("extractor2", EXTRACTOR, lambda: self._geyser_tile(self.nat_base), lair is not None or self.lair_at is not None)
        due("spire", SPIRE, lambda: self._spot_near(SPIRE, self.home), lair is not None and lair["completed"])
        nat = self.nat_base.center
        front = (int(nat[0] + (self.gather[0] - nat[0]) * 0.6), int(nat[1] + (self.gather[1] - nat[1]) * 0.6))
        for k in range(SUNKENS):                    # (one after the other: each on a free spot)
            due(f"creep{k}", CREEP, lambda: self._spot_near(CREEP, front, 0, 6),
                self.hi.frame >= SUNKENS_FROM and pool is not None and pool["completed"] and "hatch" in self.built
                and (k == 0 or f"creep{k - 1}" in self.built))
        # a macro hatchery: money piling up, no larva to spend it on (4 hatcheries at most)
        larvae = sum(1 for u in mine if u["type"] == C.LARVA)
        if 2 <= len(hatches) < 4 and me["minerals"] - self._kept() >= 350 and larvae == 0 \
                and not any(j.step.startswith("macro") for j in self.jobs.values()):
            due(f"macro{len(hatches)}_{self.hi.frame}", HATCH, lambda: self._spot_near(HATCH, self.home), True)

    def _tech(self, obs, mine, me, hatches, pool) -> bool:
        """The Lair from LAIR_FROM (100 gas), zergling speed once the Spire is started (True: it acted)."""
        hi = self.hi
        lair = any(u["type"] == LAIR for u in mine)
        if not lair and pool and pool["completed"] and hi.frame >= LAIR_FROM and me["gas"] >= 100 \
                and me["minerals"] - self._kept() >= 150 and (self.lair_at is None or hi.frame - self.lair_at > LAIR_RETRY):
            main = min((h for h in hatches if h["completed"]), key=lambda h: math.dist((h["x"], h["y"]), self.home), default=None)
            if main is None:
                return False
            if self._look(obs, (main["x"], main["y"])):
                return True
            if obs["selection"] != [main["tag"]]:
                top = screen_of(obs, main["x"], main["y"] - 24)
                if top is None:
                    return False
                hi.click(*top)
                return True
            if hi.morph_building(LAIR).accepted:
                self.lair_at = hi.frame
                self.log("Lair")
            return True
        creep = next((u for u in mine if u["type"] == CREEP and u["completed"]), None)
        if creep is not None and me["minerals"] - self._kept() >= 50:
            if self._look(obs, (creep["x"], creep["y"])):
                return True
            if obs["selection"] != [creep["tag"]]:
                at = screen_of(obs, creep["x"], creep["y"])
                if at is None:
                    return False
                hi.click(*at)
                return True
            hi.morph_building(SUNKEN)
            return True
        started = "spire" in self.built or "spire" in self.jobs
        if started and pool and pool["completed"] and "speed" not in self.ordered and me["gas"] >= 100 \
                and me["minerals"] - self._kept() >= 100:
            if self._look(obs, (pool["x"], pool["y"])):
                return True
            if obs["selection"] == [pool["tag"]]:
                if hi.upgrade(METABOLIC_BOOST).accepted:
                    self.ordered["speed"] = hi.frame
                    self.log("zergling speed")
                return True
            hi.click(*screen_of(obs, pool["x"], pool["y"]))
            return True
        return False

    @staticmethod
    def _geyser_tile(base):
        if not base.geysers:
            return None
        gx, gy = base.geysers[0]
        return (gx - 64) // 32, (gy - 32) // 32

    def _builders(self) -> set[int]:
        return {j.drone for j in self.jobs.values() if j.drone is not None}

    def _kept(self) -> int:
        """Minerals kept for buildings on their way (from the walk until they appear)."""
        return sum(COST[j.unit_type] for j in self.jobs.values() if j.state != "pick")

    def _up(self, job: Job, mine) -> bool:
        w, h = SIZE[job.unit_type]
        cx, cy = job.tile[0] * 32 + w * 16, job.tile[1] * 32 + h * 16
        return any(u["type"] == job.unit_type and math.dist((u["x"], u["y"]), (cx, cy)) < 3 * 32 for u in mine)

    def _spot(self, job: Job) -> tuple[int, int]:
        """The spot's middle, in pixels."""
        w, h = SIZE[job.unit_type]
        return job.tile[0] * 32 + w * 16, job.tile[1] * 32 + h * 16

    def _work(self, job: Job, obs, mine, me, drones) -> bool:
        """One step of a job (True: it used this turn's hands)."""
        hi = self.hi
        if self._up(job, mine):
            self.built.add(job.step)
            del self.jobs[job.step]
            return False
        if job.tries >= JOB_TRIES:
            self.log(f"{job.step} given up")
            self.built.add(job.step)
            del self.jobs[job.step]
            return False
        drone = next((u for u in drones if u["tag"] == job.drone), None)
        if job.drone is not None and drone is None:
            if job.state == "placed" and any(u["tag"] == job.drone for u in mine):
                return False                       # (it became the building: not counted as a drone)
            job.drone, job.state = None, "pick"
        spot = self._spot(job)
        if job.state == "pick":
            free = me["minerals"] - self._kept() >= COST[job.unit_type] - 100 and me["gas"] >= GAS_COST.get(job.unit_type, 0)
            if not free:
                return False
            return self._pick_drone(job, obs, mine, drones)
        if job.state == "walk":
            if math.dist((drone["x"], drone["y"]), spot) <= AT_SPOT:
                job.state = "place"
                return self._work(job, obs, mine, me, drones)
            if (drone["order"] in IDLE_ORDERS or drone["order"] == MINING) and hi.frame - job.since > 2 * 24:
                return self._walk(job, obs, drone)     # (it stopped short, or went mining: again)
            return False
        if job.state == "place":
            if me["minerals"] < COST[job.unit_type] or me["gas"] < GAS_COST.get(job.unit_type, 0):
                return False                       # (there, waiting for the money)
            if self._look(obs, spot):
                return True
            if obs["selection"] != [drone["tag"]]:
                at = screen_of(obs, drone["x"], drone["y"])
                job.clicks += 1
                if at is None or job.clicks > PLACE_CLICKS:   # (it won't be selected: another drone)
                    job.drone, job.state, job.clicks = None, "pick", 0
                    job.tries += 1
                    return False
                hi.click(*at)
                return True
            job.clicks = 0
            if not hi.game.can_place(hi.slot, job.unit_type, *job.tile, drone["tag"]):
                near = self._free_near(job, drone["tag"])
                if near is None:
                    job.tries += 1
                    return False
                job.tile = near
                spot = self._spot(job)
            tx, ty = job.tile
            p = screen_of(obs, tx * 32 + 16, ty * 32 + 16)
            if p is None:
                return False
            hi.build(job.unit_type, *p, order=C.ORDER_DRONE_START_BUILD)
            job.state, job.since = "placed", hi.frame
            job.tries += 1
            return True
        if job.state == "placed":
            if drone is not None and drone["order"] in IDLE_ORDERS and hi.frame - job.since > PLACED_WAIT:
                job.state = "place"                # (nothing went up: the click's scatter, or a unit on it)
            elif drone is not None and drone["order"] in (MINING,) + tuple(range(79, 92)):
                job.state = "walk"
            return False
        return False

    def _pick_drone(self, job: Job, obs, mine, drones) -> bool:
        """A drone for the job: clicked alone (a fresh one by a hatchery first, else one whose
        neighbors are farthest), then walked to the spot."""
        hi = self.hi
        sel = [u for u in drones if u["tag"] in obs["selection"]]
        if len(obs["selection"]) == 1 and sel and sel[0]["tag"] not in self.gas and sel[0]["tag"] not in self._builders():
            job.drone = sel[0]["tag"]
            return self._walk(job, obs, sel[0])
        busy = self.gas | self._builders()
        free = [d for d in drones if d["tag"] not in busy]
        if not free:
            return False
        others = [u for u in mine if u["type"] != C.LARVA]

        def room(d):
            return min((math.dist((d["x"], d["y"]), (o["x"], o["y"])) for o in others if o["tag"] != d["tag"]), default=999)
        idle = [d for d in free if d["order"] in IDLE_ORDERS]
        d = max(idle or free, key=lambda d: (room(d), -job.picks))
        if job.picks >= PICK_TRIES:                 # (it won't come alone: any other)
            d = free[(hi.frame // 24) % len(free)]
            job.picks = 0
        if self._look(obs, (d["x"], d["y"])):
            return True
        at = screen_of(obs, d["x"], d["y"])
        if at is None:
            return False
        job.picks += 1
        hi.click(*at)
        return True

    def _walk(self, job: Job, obs, drone) -> bool:
        """The selected drone walks to the spot (a minimap order: the camera stays)."""
        if obs["selection"] != [drone["tag"]]:
            job.state = "pick"
            return False
        self.hi.minimap_right_click(*self._spot(job))
        job.state, job.since, job.picks = "walk", self.hi.frame, 0
        return True

    def _free_near(self, job: Job, builder: int):
        """The free tile nearest the spot (within 2) the building fits, or None."""
        x0, y0 = job.tile
        for r in (1, 2):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) == r and self.hi.game.can_place(self.hi.slot, job.unit_type, x0 + dx, y0 + dy, builder):
                        return x0 + dx, y0 + dy
        return None

    # --- gas and larvae --------------------------------------------------------------------------

    def _gas_done(self) -> bool:
        return False                               # (mutalisks want gas all game)

    def _gas(self, obs, mine, me, drones) -> bool:
        """GAS_EACH drones on each finished extractor, from the base it stands at (v2's gas step)."""
        # (a drone inside an extractor isn't seen: a gas drone counts until unseen for GAS_UNSEEN)
        hi = self.hi
        for u in drones:
            self.gas_seen[u["tag"]] = hi.frame
        self.gas = {d for d in self.gas if hi.frame - self.gas_seen.get(d, hi.frame) <= GAS_UNSEEN}
        self.gas_of = {d: e for d, e in self.gas_of.items() if d in self.gas}
        for ext in (u for u in mine if u["type"] == EXTRACTOR and u["completed"]):
            if sum(1 for e in self.gas_of.values() if e == ext["tag"]) >= GAS_EACH:
                continue
            miners = [u for u in drones if u["order"] == MINING and u["tag"] not in self.gas
                      and u["tag"] not in self._builders() and math.dist((u["x"], u["y"]), (ext["x"], ext["y"])) < 12 * 32]
            if miners:
                self.task, self.drone_for, self.gas_target = "gas", self._pick(miners, obs)["tag"], ext["tag"]
                self._continue(obs, mine, me)
                return True
        return False

    def _continue(self, obs, mine, me) -> None:
        # a box around a mining drone takes its neighbors: whichever single drone it took goes (any
        # drone mines gas as well)
        sel = [u for u in obs["units"] if u["tag"] in obs["selection"]]
        if self.task == "gas" and len(sel) == 1 and sel[0]["owner"] == self.hi.slot and sel[0]["type"] == DRONE \
                and sel[0]["tag"] not in self.gas and sel[0]["tag"] not in self._builders():
            self.drone_for = sel[0]["tag"]
        gas = self.task == "gas" and self.drone_for
        super()._continue(obs, mine, me)
        if gas and gas in self.gas and gas not in self.gas_of:
            self.gas_of[gas] = self.gas_target

    def _want(self, me, mine, hatches, pool, eggs_drones):
        hi = self.hi
        count = lambda t: sum(1 for u in mine if u["type"] == t)
        self.overlords_at = [f for f in self.overlords_at if hi.frame - f < 24 * 25]
        spire = next((u for u in mine if u["type"] == SPIRE and u["completed"]), None)
        muta = spire is not None and me["minerals"] >= 100 and me["gas"] >= 100
        if self._overlord_due(me, hatches):
            return OVERLORD
        if pool and pool["completed"] and count(LING) < LINGS_EARLY and spire is None:
            return LING
        want_lings = LINGS_AT + LINGS_A_MINUTE * (hi.frame - LINGS_FROM) / 1440 if hi.frame >= LINGS_FROM else 0
        if pool and pool["completed"] and count(LING) < want_lings and not (muta and count(MUTA) < 3):
            return LING
        if muta and (eggs_drones >= DRONES or hi.frame >= MUTAS_FIRST):
            return MUTA
        if eggs_drones < DRONES and hi.frame < MUTAS_FIRST + 60 * 24:
            return DRONE
        if muta:
            return MUTA
        if spire is not None:                       # (the gas makes mutalisks; minerals above that, zerglings)
            return LING if me["minerals"] >= LINGS_FROM_MINERALS else None
        return LING if pool and pool["completed"] and me["minerals"] >= 150 else None
