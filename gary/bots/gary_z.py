"""Gary-Z v0: the learned Zerg (docs/SELF_PLAY.md), the opponent self-play needs.

Zerg v3's hands (building jobs, larva, gas, idle drones, rallies, its army, the mutalisk raid and
lurkers: gary/bots/zerg_v3.py) with the Zerg macro model's decisions in place of v3's scripted ones
(_plan, _tech, _want): what to make next, as the pro Zergs did (runs/macro/TvZ_Z_*, ingest and
train like the Terran's), steered by a pro Zerg plan rolled at the start (taxonomy/plans.py: the
opening, two or three hatcheries, spire or lurker first). The plan is a free, true label for
Gary-T's opponent model.

Each decision is carried out with the game's generic commands through the human interface: a larva
morphs (drones, zerglings, overlords, hydralisks, mutalisks, scourge, queens, ultralisks, defilers),
a unit morphs (lurkers, guardians, devourers), a building morphs (Lair, Hive, Greater Spire, sunken
and spore colonies), a drone builds (v3's jobs, every Zerg building: gary/zerg.py), a building
researches or upgrades. An overlord before a supply block is an execution guard, as v3 had; nothing
keeps larvae busy by habit (the model decides drones against army).

    python -m gary.bots.zerg_rush --vs v027 --opponent garyz --minutes 20 --seed 101 --save runs/x.rep
"""

from __future__ import annotations

import json
import math
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np

from gary import commands as C
from gary import terran as T
from gary import zerg as Z
from gary.bots import zerg_v3 as V3
from gary.decision_log import LOG, top
from gary.bots.zerg_v3 import Job, ZergV3
from gary.interface import screen_of
from gary.policy.army import ArmyModel, ArmyTracker
from gary.policy.macro import MacroModel, MacroTracker

MORPH, BMORPH, BUILD, RESEARCH, UPGRADE, EXPAND = 1, 2, 3, 4, 5, 6
MIN_P = 0.05                    # never a decision the model gives less than this (as the Terran's macro)
TECH_MIN_P = 0.10               # a building that opens a tech path: never on odds under this
TECH_PATH = {Z.DEN, Z.SPIRE, Z.QUEENS_NEST, Z.DEFILER_MOUND, Z.ULTRA_CAVERN}
DICE_FLOOR = 0.01               # the dice decoder: decisions under this odds aren't rolled
USED_LARVA = 2 * 24             # a larva just morphed shows as a larva a moment longer: not used again
IN_FLIGHT = 24                  # money just spent shows in the bank a moment later: counted as spent
GOAL_TIMEOUT = 30 * 24          # a decision not carried out in 30 s is dropped
QUERY_EVERY = 12
GAS_FLOAT, GAS_LOW = 400, 300   # gas above this (and above the minerals): its drones go back to minerals;
                                # below this, back to gas (150 trapped it: no gas mined from 4:00 to 8:00
                                # with ~160 banked, too little for mutalisks; pros bank 350-450 then)
RALLY_IN_LINE = 0.6              # the rally: this far from the hatchery toward its minerals' middle
POST_OUT = 10 * 32               # px: the outer watch post, this far outside Gary's natural toward the middle
POST_AGAIN = 60 * 24            # a post whose overlord was lost is manned again after this
ARMY_EVERY = 5 * 24              # a group's destination asked of the army model this often
HOLD_BELOW = -0.02              # an attack the fight estimate expects to lose more army share than this: held
# v1's groups take every Zerg ground fighter (v3's took zerglings only: hydralisks and ultralisks stood
# where they hatched), each by its supply in the game's own half-units (a zergling 1, a hydralisk 2: a
# group forms at v1's wave count, 12 = 6 supply; hand-picked weights before, the user's catch)
GROUND = {t: 2 * Z.SUPPLY[t] / (2 if t in Z.PAIRS else 1)
          for t in (Z.ZERGLING, Z.HYDRALISK, Z.ULTRALISK, Z.DEFILER, Z.QUEEN)}
RALLY_FRONT_AT = 5 * 60 * 24     # hatcheries rally to the army's meeting point from then on
PACE_MAX = 10 * 24               # the longest wait for the next decision
STANDING = {2, 3}               # Guard, PlayerGuard: a unit standing
STRAY = 15 * 32                 # a fighter in no group this far from the meeting point walks to it
STRAY_AGAIN = 10 * 24
TASK_TIMEOUT = 10 * 24          # a v2 task (a drone to gas or to a building) given up after this
SIDES = ("army", "macro", "helpers")   # who shares the hands, in turn
MINERAL_ORDERS = {85, 86, 87, 90}   # MoveToMinerals, WaitForMinerals, MiningMinerals, ReturnMinerals
LURK_STREAK = 8                 # v3's lurker routine: this many turns in a row at most...
LURK_BACKOFF = 3 * 24           # ...then it waits this long
MUTA_FLOCK_SAVE = 6             # gas saved (no floating) while a Spire stands and fewer mutalisks than this
JOIN_BELOW = 0.5                # a cluster under this share of the main one's size joins it
CLUSTERS = 3                    # the army by clusters: this many at most, each asked in turn
MAX_SELECT = 12                 # units a selection takes
STREAK = 3                      # actions in a row one side keeps the hands (a selection and its order)
PLANS_FILE = "TvZ_Z_plan.json"  # data/interim/taxonomy: the plans' names and how often pros play them

# v3's job code knows its own six buildings; Gary-Z builds any (only types v3 doesn't have are added,
# so Zerg v3 itself plays as before)
for _t, (_m, _g) in Z.COST.items():
    if _t in Z.DRONE_BUILDS:
        V3.COST.setdefault(_t, _m)
        V3.GAS_COST.setdefault(_t, _g)
for _t, _s in Z.SIZE.items():
    V3.SIZE.setdefault(_t, _s)


def _cell_dist(a: int, b: int) -> int:
    return max(abs(a // 16 - b // 16), abs(a % 16 - b % 16))


def latest_model(kind: str = "macro") -> Path:
    found = sorted(Path(f"runs/{kind}").glob("TvZ_Z_*/model.pt"))
    if not found:
        raise FileNotFoundError(f"no Zerg {kind} model in runs/{kind}: python -m train.{kind} --race Z")
    return found[-1]


def load_plans() -> tuple[list[str], list[float]] | None:
    """The plans' names and the pros' shares of them, if the taxonomy is here."""
    try:
        from ingest.inventory import data_root
        d = json.loads((data_root() / "interim" / "taxonomy" / PLANS_FILE).read_text(encoding="utf-8"))
    except Exception:
        return None
    names = d["names"]
    counts = [0] * len(names)
    for k in d["assignments"].values():
        counts[k] += 1
    return names, [c / max(1, sum(counts)) for c in counts]


class GaryZ(ZergV3):
    def __init__(self, hi, mapinfo, seed: int = 0, model: str | Path | None = None, plan: int | None = None):
        super().__init__(hi, mapinfo, seed)
        self.model = MacroModel.load(model or latest_model())
        plans = load_plans()
        rng = random.Random(seed)
        styles = self.model.spec.styles
        if plan is None and styles:
            weights = [plans[1][s] if plans and s < len(plans[1]) else 1.0 for s in styles]
            plan = rng.choices(styles, weights=weights)[0]
        self.style = plan
        self.plan_name = plans[0][plan] if plans and plan is not None and plan < len(plans[0]) else str(plan)
        # v3's army: the mutalisk flock always; lurkers burrow whenever there are lurkers (army below);
        # no scripted pressure (where the army goes is the army model's, later); v3's poke on, the first
        # zerglings outside Gary's natural at about 3:00 and home: the model decides from what it sees,
        # and without it Gary-Z had seen no Terran army by 6:00 in 8 games of 8 (pros: 4 units by 5:00,
        # only 4% none) and kept droning while pros switch to army (57% of larvae at 6:00)
        self.plan, self.poke, self.pressing = "mutas", True, False
        self.tracker = MacroTracker(self.model.spec, hi.slot, plan)
        # where the groups go: the Zerg army model (runs/army/TvZ_Z_*), as the Terran's army; v1's
        # stage-judge-go is replaced, its home defense and v3's raid and lurkers stay
        self.army_model = ArmyModel.load(latest_model("army"))
        self.army_tracker = ArmyTracker(self.army_model.spec, hi.slot, mapinfo.size, self.home)
        self.sizes = dict(GROUND)
        self.next_ask: dict[int, int] = {}       # group -> when its destination is drawn again
        self.army_goal: dict[int, tuple | None] = {}   # group -> its drawn destination
        self.sent_to: dict[int, tuple] = {}
        self.straggle_after, self.straggle_tries = 0, 0
        # the army by clusters, not v1's hotkey groups (GARY_Z_ARMY=groups for those): 24 games on the same
        # seeds, at home 47% at 5-7 min (groups 78%, pros 38%), in the field 47% (20%, 52%), Gary-Z's gap
        # score 0.34 (0.45), decided 38% (17%), lone units moving away 10% (3%, pros 18%)
        self.cluster_army = os.environ.get("GARY_Z_ARMY", "clusters") == "clusters"
        self.army_job: dict | None = None
        self.cluster_ask: dict[int, int] = {}
        self.cluster_turn = 0
        self.army_sent = self.army_dropped = 0   # (cluster orders given, and dropped unselected)
        self.lurk_streak, self.lurk_backoff = 0, 0
        self.task_key, self.task_since = None, 0      # group -> the destination it was last sent to
        self.rally_of: dict[int, bool] = {}     # hatchery -> rallied to the front (True) or its minerals
        self.goal: dict | None = None
        self.parked: dict | None = None        # a decision waiting for its money, kept for it meanwhile
        self.turn, self.streak = "macro", 0    # who goes first, and how many actions in a row it had
        self.gas_floating = False              # the gas piles up: its drones mine minerals meanwhile
        self.post_of: dict[int, int] = {}      # watch post -> its overlord
        self.post_again: dict[int, int] = {}   # watch post -> frame from which it's manned again
        self._post_points = None
        self.next_query = 0
        self.rng = random.Random(seed * 7919 + 1)          # the dice (deterministic per seed)
        # its fights: the Zerg role model (gary/bots/zerg_roles.py: the pros' Zerg units before contact and
        # in it), in place of v3's scripted micro; none trained yet, v3's
        from gary.bots.zerg_roles import ZergRoles, latest_zerg_model
        roles = latest_zerg_model()
        self.fight_roles = ZergRoles(self, roles) if roles is not None else None
        self.used_larva: dict[int, int] = {}                # larva tag -> frame it was morphed
        self.spent: list[tuple[int, int, int]] = []         # (frame, minerals, gas) just spent
        self.morphs: list[tuple[int, int, int]] = []       # (frame, unit type, frames to make) ordered
        self.researched: set[int] = set()
        self.upgraded: dict[int, int] = {}
        self.build_frames: dict[int, int] = {}
        self.log(f"Gary-Z v0: plan {self.plan_name} (macro model {Path(getattr(self.model, 'path', '')).parent.name})")

    # --- the loop: v3's hands, the model's decisions --------------------------------------------

    def act(self) -> None:
        hi = self.hi
        if hi.pending:
            return
        obs = hi.observe()
        me = obs["me"]
        mine = [u for u in obs["units"] if u["owner"] == hi.slot]
        self._timing(mine)
        # gas as it's spent: v3 put 3 drones on every extractor from the start whatever the plan, and the
        # gas piled up (1,204 at 6:00) while the minerals ran short
        floating = self.gas_floating
        if me["gas"] > GAS_FLOAT and me["gas"] > me["minerals"]:
            self.gas_floating = True
        elif me["gas"] < GAS_LOW:
            self.gas_floating = False
        # (none while a Spire stands and the flock isn't made: pros save ~700 gas for the first mutalisks,
        # and floating at 400 kept Gary-Z from ever having it: no mutalisks at 8-10 min, pros 6-8)
        if any(u["type"] == Z.SPIRE for u in mine) and sum(1 for u in mine if u["type"] == Z.MUTALISK) < MUTA_FLOCK_SAVE:
            self.gas_floating = False
        if self.gas_floating != floating:
            LOG.decision("Z", "gas", hi.frame, "gas drones to minerals" if self.gas_floating else "gas drones back",
                         inputs={"m": me["minerals"], "gas": me["gas"]},
                         by=f"rule: gas over {GAS_FLOAT} and over the minerals" if self.gas_floating else
                         f"rule: gas under {GAS_LOW}, or a Spire saving for the flock")
        hatches = [u for u in mine if u["type"] in Z.TOWN_HALLS]
        pool = next((u for u in mine if u["type"] == Z.POOL), None)
        drones = [u for u in mine if u["type"] == Z.DRONE and u["completed"]]
        # the army, the macro and v3's helpers share the hands: whoever acted keeps them (a look, a
        # selection, then its order: whoever came in between moved the camera or the selection, and the
        # sequence began again), STREAK actions at most, then the next goes first; one with nothing to do
        # hands over at once. (Each one alone took every turn at some point: v3's army in fights, nothing
        # decided from 7:30 to 10:30; three jobs at once from 1:00 to 2:00; idle drones 109 turns in 40 s.
        # The army every other turn moved the camera between a look and its selection: a drone to gas
        # held the rest for 2:30, a Hydralisk goal timed out twice with 9 larvae and 3,000 minerals)
        sides = list(SIDES)
        k = sides.index(self.turn)
        for name in sides[k:] + sides[:k]:
            LOG.owner["Z"] = name                               # (its orders stamped with it: decision_log.py)
            if getattr(self, "_side_" + name)(obs, mine, me, hatches, pool, drones):
                LOG.hands("Z", hi.frame, name)
                self.streak = self.streak + 1 if name == self.turn else 1
                self.turn = name
                if self.streak >= STREAK:
                    self.turn, self.streak = sides[(sides.index(name) + 1) % len(sides)], 0
                return

    def _side_army(self, obs, mine, me, hatches, pool, drones) -> bool:
        return self.army(obs, mine)

    def _side_macro(self, obs, mine, me, hatches, pool, drones) -> bool:
        return self._macro(obs, mine, me, hatches, pool, drones)

    def _side_helpers(self, obs, mine, me, hatches, pool, drones) -> bool:
        if self.task:                                   # (v2's task under way: a drone to gas, a building)
            self._continue(obs, mine, me)
            return True
        return self._helpers(obs, mine, me, hatches, pool, drones)

    def _continue(self, obs, mine, me) -> None:
        """v2's task (a drone to gas, a building), given up after TASK_TIMEOUT: the army moves the camera
        every other turn and the task looks back every time, so its own tries never ran out; it held
        every turn but the army's for 2:30, the larvae and 3,000 minerals unused (seed 108)."""
        key = (self.task, self.drone_for)
        if key != self.task_key:
            self.task_key, self.task_since = key, self.hi.frame
        if self.hi.frame - self.task_since > TASK_TIMEOUT:
            self.log(f"gives up on its {self.task.split(':')[0]} task")
            self.task, self.task_key = "", None
            return
        super()._continue(obs, mine, me)

    def _helpers(self, obs, mine, me, hatches, pool, drones) -> bool:
        """v3's hands: building jobs, gas, idle drones, rallies (True: one acted)."""
        for job in list(self.jobs.values()):
            if self._work(job, obs, mine, me, drones):
                return True
        if self._gas(obs, mine, me, drones):
            return True
        if self._mine_idle(obs, mine, [d for d in drones if d["tag"] not in self._builders()], hatches):
            return True
        return bool(self._rallies(obs, hatches, pool))

    def _macro(self, obs, mine, me, hatches, pool, drones) -> bool:
        """An overlord before a supply block, else the model's next decision carried out (True: acted)."""
        hi = self.hi
        # (v3 forgets an overlord 25 s after ordering it in its _want, which Gary-Z doesn't use: kept,
        # the list said one was always coming, and Gary-Z sat supply-blocked: 15/17 at 4:00)
        self.overlords_at = [f for f in self.overlords_at if hi.frame - f < 24 * 25]
        if self._overlord_due(me, hatches) and (self.goal is None or self.goal["id"] != Z.OVERLORD):
            if self._larva_morph(obs, mine, Z.OVERLORD):        # (an execution guard, as v3's)
                self.overlords_at.append(hi.frame)
                LOG.decision("Z", "macro", hi.frame, "Overlord", inputs=self._seen(obs, mine, me),
                             by="rule: an overlord before a supply block")
            return bool(hi.pending)
        self._decide(obs, mine, me)
        if self.goal is not None:
            self._carry_out(obs, mine, me)
        return bool(hi.pending)

    def _gas_done(self) -> bool:
        return self.gas_floating                   # (v2's idle routine sends gas drones to minerals then)

    def _gas(self, obs, mine, me, drones) -> bool:
        # (v3 forgets a gas drone only when it goes unseen; one sent back to minerals while the gas
        # floated is seen, so it stayed a gas drone and its extractor looked full all game: 0-3 drones
        # on gas from 6:00, gas 44 at 10:00 to the pros' 500, and the minerals unspent)
        back = {u["tag"] for u in drones if u["order"] in MINERAL_ORDERS}
        self.gas -= back
        self.gas_of = {d: e for d, e in self.gas_of.items() if d not in back}
        return False if self.gas_floating else super()._gas(obs, mine, me, drones)

    def _mining_bases(self, mine) -> list:
        """Every base Gary-Z holds (v2's took the main and the natural only: drones at a third base were
        walked back to the main; Argon's catch)."""
        return self._owned_bases(mine) or [self.main_base]

    def _rallies(self, obs, hatches, pool) -> bool:
        """Every finished hatchery rallies to the ground beside its own base's minerals (a minimap
        right-click there), not to the army's gathering point as v2's did: each drone walked to the
        choke and stood there until the idle routine fetched it (Argon's catch). In BW a rallied drone
        doesn't mine by itself (OpenBW: it stands by the field), so the idle routine still sends it,
        from next to the minerals. Pros' hatchery rallies in 46 TvZ games: 78% toward the front (the
        user's way: with the army, then the drones picked up with a ctrl+click), 21% by the minerals
        (a third of those before 6:00), onto a mineral field practically never. (True: it acted.)"""
        hi = self.hi
        # (the user's way: beside the minerals while droning, the army's meeting point from RALLY_FRONT_AT,
        # when pros switch to army; v1's groups form at the meeting point, and the army model orders groups)
        front = hi.frame >= RALLY_FRONT_AT
        for h in hatches:
            if not h["completed"] or self.rally_of.get(h["tag"]) == front:
                continue
            base = self.map.base_near((h["x"], h["y"]))
            if not base.minerals and not front:
                self.rally_of[h["tag"]] = front
                continue
            if front:
                spot = self.gather
            else:
                mx = sum(x for x, _ in base.minerals) / len(base.minerals)
                my = sum(y for _, y in base.minerals) / len(base.minerals)
                spot = (int(h["x"] + (mx - h["x"]) * RALLY_IN_LINE), int(h["y"] + (my - h["y"]) * RALLY_IN_LINE))
            if self._look(obs, (h["x"], h["y"])):
                return True
            if obs["selection"] == [h["tag"]]:
                hi.minimap_right_click(*spot)
                self.rally_of[h["tag"]] = front
                return True
            top = screen_of(obs, h["x"], h["y"] - 24)
            if top:
                hi.click(*top)
            return True
        return False

    def _posts(self) -> list[tuple[int, int]]:
        """Overlord watch posts: outside Gary's natural (toward the map's middle) and halfway there."""
        if self._post_points is None:
            nat = self.map.natural_of(self.map.base_near(self.target)).center
            w, h = self.map.size
            d = math.dist(nat, (w / 2, h / 2)) or 1
            out = (int(nat[0] + (w / 2 - nat[0]) / d * POST_OUT), int(nat[1] + (h / 2 - nat[1]) / d * POST_OUT))
            mid = (int((self.nat_base.center[0] + out[0]) / 2), int((self.nat_base.center[1] + out[1]) / 2))
            self._post_points = [out, mid]
        return self._post_points

    def _overlord_posts(self, obs: dict, mine: list[dict]) -> bool:
        """Overlords on the watch posts, as pros send theirs: the model decides from what Gary-Z has seen,
        and without them it had seen no Terran army by 8:00 in 8 games of 8 (pros: 4 units by 5:00, 4%
        none) and kept droning when pros switch to army (57% of larvae at 6:00). One lost is replaced
        after POST_AGAIN. (True: it acted.)"""
        hi = self.hi
        lords = {u["tag"]: u for u in mine if u["type"] == Z.OVERLORD and u["completed"]}
        for k, post in enumerate(self._posts()):
            tag = self.post_of.get(k)
            if tag in lords or hi.frame < self.post_again.get(k, 0):
                continue
            if tag is not None:                        # (lost: another after a while)
                self.post_of.pop(k)
                self.post_again[k] = hi.frame + POST_AGAIN
                continue
            free = [u for u in lords.values() if u["tag"] not in self.post_of.values()]
            if not free:
                return False
            sel = set(obs["selection"])
            chosen = next((u for u in free if sel == {u["tag"]}), None)
            if chosen is None:
                self._select_one(obs, free)
                return True
            hi.minimap_right_click(*post)
            self.post_of[k] = chosen["tag"]
            self.log(f"overlord to watch post {k} at {post}")
            return True
        return False

    @property
    def attack_after(self) -> int:
        """No earliest attack: the army model decides (eval/vs_rush.py sets 7:00 for the scripted rushes,
        and v1 formed no group before it: the army stood ungrouped and the model unasked until 7:00, in
        every Gary-Z game until now; Fable's catch, the army layer before fight roles)."""
        return 0

    @attack_after.setter
    def attack_after(self, value) -> None:
        pass

    def _go(self, mine) -> bool:
        return False                               # (no scripted all-in: the army model decides when to attack)

    def _grab_group(self, obs: dict, fresh: list[dict]) -> None:
        """A new group: a ctrl+click on the most common fighter type (every one of it on the screen),
        then a shift+click on each other fighter, 12 at most. v1's ctrl+click alone took one type (the
        hydralisks never joined; Argon's catch); a drag box took the drones at the meeting point too
        (they rally there from 5:00), and a group with a drone in it is refused."""
        mid = (sum(u["x"] for u in fresh) / len(fresh), sum(u["y"] for u in fresh) / len(fresh))
        near = sorted(fresh, key=lambda u: math.dist((u["x"], u["y"]), mid))[:12]
        kinds = Counter(u["type"] for u in near)
        main = kinds.most_common(1)[0][0]
        self.hi.ctrl_click(*screen_of(obs, *next((u["x"], u["y"]) for u in near if u["type"] == main)))
        room = 12 - sum(1 for u in fresh if u["type"] == main)
        for u in [u for u in near if u["type"] != main][:max(0, room)]:
            self.hi.click(*screen_of(obs, u["x"], u["y"]), shift=True)

    def _grab_loose(self, obs: dict, loose: list[dict]) -> None:
        """Loose fighters for a home defense, as a new group (a drag box at home takes the drones along,
        and v1 refuses a selection with a drone in it: they never defended)."""
        self._grab_group(obs, loose)

    def _stragglers(self, obs: dict, mine: list[dict]) -> bool:
        """Fighters standing away from the meeting point, in no group (hatched before the rallies moved
        to the front, or left after a defense), walk to it: a look, then a click on them and a right
        click on the minimap. (True: it acted.)"""
        hi = self.hi
        if hi.frame < RALLY_FRONT_AT or hi.frame < self.straggle_after:
            return False
        gather = self.gather_point()
        grouped = set().union(*self.groups.values()) if self.groups else set()
        far = [u for u in mine if u["type"] in self.sizes and u["completed"] and u["tag"] not in grouped
               and u["order"] in STANDING and math.dist((u["x"], u["y"]), gather) > STRAY]
        if not far:
            return False
        seen = [u for u in far if screen_of(obs, u["x"], u["y"])]
        if not seen:
            self.straggle_tries += 1
            if self.straggle_tries > 2:                     # (they don't come on the screen: later)
                self.straggle_after, self.straggle_tries = hi.frame + STRAY_AGAIN, 0
                return False
            hi.camera_minimap(far[0]["x"], far[0]["y"])
            return True
        self._grab_group(obs, seen)
        hi.minimap_right_click(int(gather[0]), int(gather[1]))
        self.straggle_after, self.straggle_tries = hi.frame + STRAY_AGAIN, 0
        return True

    def _grab_idle(self, obs, u) -> None:
        """An idle drone away from every hatchery (at the meeting point, among the army): a ctrl+click on
        it takes every drone on the screen and no army (the user's way: a box took zerglings with it and
        v2's routine gave up on the drone). By a hatchery, v2's small box."""
        homes = [(h["x"], h["y"]) for h in obs["units"] if h["owner"] == self.hi.slot and h["type"] in Z.TOWN_HALLS]
        if homes and min(math.dist((u["x"], u["y"]), h) for h in homes) > 8 * 32:
            self.hi.ctrl_click(*screen_of(obs, u["x"], u["y"]))
        else:
            super()._grab_idle(obs, u)

    def _army_goal(self, n: int, obs: dict, members: list[dict]):
        """Group n's destination, drawn every ARMY_EVERY from the army model's odds for that group (its
        cell and move or attack): (cell, attack), or None to stay (an attack the fight estimate expects to
        lose is held). Drawn, not the likeliest: the likeliest sent every group to one cell, nearly
        always home (the army bunched at home 79-89% of the time, pros 35-46%), and a drawn decision with
        its probability is what RL learns the army layer from (the league logs it)."""
        hi = self.hi
        if hi.frame < self.next_ask.get(n, 0):
            return self.army_goal.get(n)
        self.next_ask[n] = hi.frame + ARMY_EVERY
        grid, glob = self.army_tracker.inputs(obs, members)
        where, kind = self.army_model.predict(grid, glob)
        cell = self.rng.choices(range(where.shape[1]), weights=where[0].tolist())[0]
        attack = self.rng.random() < float(kind[0, 1])
        held = attack and self.army_model.fight(grid, glob, cell, True) < HOLD_BELOW
        self.army_goal[n] = None if held else (cell, attack)
        self._army_record(f"group {n}", members, where, kind, cell, attack, "held: the fight estimate" if held else None)
        LOG.add("army_z", frame=hi.frame, group=n, grid=grid[0].astype(np.float16), glob=glob[0].astype(np.float16),
                cell=cell, attack=attack, held=held,
                logp=float(np.log(where[0, cell] + 1e-9) + np.log(kind[0, int(attack)] + 1e-9)))
        return self.army_goal[n]

    def _group(self, n: int, obs: dict, lings: dict, enemy: list[dict]) -> bool:
        """Group n goes where its drawn goal is (_army_goal), unless it's there or on its way. (True: it acted.)"""
        if self.phase.get(n) == "to_staging":           # (a new group: to the staging point first, as v1's)
            return self._order(n, obs, "move", self.staging, "staging")
        members = [lings[t] for t in self.groups[n] if t in lings]
        if not members:
            return False
        goal = self._army_goal(n, obs, members)
        if goal is None:
            return False
        cell, attack = goal
        mid = (sum(u["x"] for u in members) / len(members), sum(u["y"] for u in members) / len(members))
        if self.army_tracker.cell(*mid) == cell or self.sent_to.get(n) == (cell, attack):
            return False
        acted = self._order(n, obs, "attack" if attack else "move", self.army_tracker.center(cell), "model")
        if self.recalled != n:              # (sent; a first call only recalls the group's hotkey: marked
            self.sent_to[n] = (cell, attack)   # sent then, the order itself never went: 12 asks for the
        return acted                           # field in a minute, none carried out)

    def army(self, obs: dict, mine: list[dict]) -> bool:
        self.army_tracker.remember(obs)
        if self._overlord_posts(obs, mine):
            return True
        if self.fight_roles is not None and self.fight_roles.step(obs, mine):
            return True
        if self.poke and not self.poked and not self.all_in and self._poke(obs, mine):
            return True
        if self._mutas(obs, mine):
            return True
        if any(u["type"] == Z.LURKER for u in mine) and self._lurkers(obs, mine):
            return True
        if self.cluster_army:
            return self._clusters_army(obs, mine)
        return super(ZergV3, self).army(obs, mine) or self._stragglers(obs, mine)

    def _lurkers(self, obs, mine) -> bool:
        """v3's lurker routine, with a limit: LURK_STREAK turns in a row, then it waits LURK_BACKOFF. It
        has no retry limit (a lurker it can't select, a burrow that doesn't take: pressed again every
        turn) and took every army turn of a probe game from 9:00 to 14:00, 2,100 of them: the rest of the
        army was never asked."""
        hi = self.hi
        if hi.frame < self.lurk_backoff:
            return False
        out = super()._lurkers(obs, mine)
        self.lurk_streak = self.lurk_streak + 1 if out else 0
        if self.lurk_streak >= LURK_STREAK:
            self.lurk_backoff, self.lurk_streak = hi.frame + LURK_BACKOFF, 0
        return out

    # --- the army by clusters (GARY_Z_ARMY=clusters) ---------------------------------------------

    def _clusters_army(self, obs: dict, mine: list[dict]) -> bool:
        """The whole ground army by the army model: its fighters clustered by where they stand (up to
        CLUSTERS, biggest first), each cluster in turn asked every ARMY_EVERY and sent where its drawn
        destination is, a dozen units at a time (a look, a click and shift+clicks, a minimap order). v1's
        hotkey groups left most of the army out (at 6-8 min 52 of 245 fighter-looks grouped in one probe
        game, none in another): units that hatched elsewhere, survived a defense or lost their group
        stood where they were, and the army model never moved them. (True: it acted.)"""
        hi = self.hi
        if self.army_job is not None:
            return self._army_job_step(obs, mine)
        fighters = [u for u in mine if u["type"] in self.sizes and u["completed"]]
        if not fighters:
            return False
        clusters = self._clusters(fighters)
        k = self.cluster_turn % len(clusters)
        if hi.frame < self.cluster_ask.get(k, 0):
            self.cluster_turn += 1
            return False
        self.cluster_ask[k] = hi.frame + ARMY_EVERY
        self.cluster_turn += 1
        members = clusters[k]
        # a cluster much smaller than the main one joins it rather than going its own way: each drawing
        # its own destination, they fought apart (Gary-Z's fighters died with 15% of its army there, the
        # pros' 27%; at odds of 1 to 3, the pros' about even: eval/zerg_losses.py)
        size = lambda c: sum(self.sizes.get(u["type"], 1) for u in c)
        main = clusters[0]
        if k > 0 and size(members) < JOIN_BELOW * size(main):
            mid = (sum(u["x"] for u in members) / len(members), sum(u["y"] for u in members) / len(members))
            to = (sum(u["x"] for u in main) / len(main), sum(u["y"] for u in main) / len(main))
            if _cell_dist(self.army_tracker.cell(*mid), self.army_tracker.cell(*to)) >= 2:
                self.army_job = {"tags": [u["tag"] for u in members], "point": (int(to[0]), int(to[1])),
                                 "kind": "move", "selecting": False, "tries": 0}
                LOG.decision("Z", "army", hi.frame, f"move to ({int(to[0]) // 32},{int(to[1]) // 32})",
                             inputs={"units": len(members), "main": len(main)},
                             by=f"rule: a cluster under {JOIN_BELOW:.0%} of the main one joins it")
                return self._army_job_step(obs, mine)
            return False
        grid, glob = self.army_tracker.inputs(obs, members)
        where, kind = self.army_model.predict(grid, glob)
        cell = self.rng.choices(range(where.shape[1]), weights=where[0].tolist())[0]
        attack = self.rng.random() < float(kind[0, 1])
        held = attack and self.army_model.fight(grid, glob, cell, True) < HOLD_BELOW
        LOG.add("army_z", frame=hi.frame, group=k, grid=grid[0].astype(np.float16), glob=glob[0].astype(np.float16),
                cell=cell, attack=attack, held=held,
                logp=float(np.log(where[0, cell] + 1e-9) + np.log(kind[0, int(attack)] + 1e-9)))
        mid = (sum(u["x"] for u in members) / len(members), sum(u["y"] for u in members) / len(members))
        stays = _cell_dist(cell, self.army_tracker.cell(*mid)) < 2
        self._army_record(f"cluster {k}", members, where, kind, cell, attack,
                          "held: the fight estimate" if held else "stays: the cell drawn is where it is" if stays else None)
        if held or stays:
            return False                                   # (held, or the drawn cell is where they are)
        self.army_job = {"tags": [u["tag"] for u in members], "point": self.army_tracker.center(cell),
                         "kind": "attack" if attack else "move", "selecting": False, "tries": 0}
        return self._army_job_step(obs, mine)

    def _army_record(self, who: str, members: list[dict], where, kind, cell: int, attack: bool, then) -> None:
        """An army decision in the game's records: the cells likeliest, the one drawn, move or attack."""
        def name(c):
            x, y = self.army_tracker.center(c)
            return f"({x // 32},{y // 32})"
        mid = (sum(u["x"] for u in members) / len(members), sum(u["y"] for u in members) / len(members))
        LOG.decision("Z", "army", self.hi.frame, f"{'attack' if attack else 'move'} to {name(cell)}",
                     float(where[0, cell]) * float(kind[0, int(attack)]), top(name, where[0], 3),
                     {"who": who, "units": len(members), "at": f"({int(mid[0]) // 32},{int(mid[1]) // 32})",
                      "p_attack": float(kind[0, 1])}, "draw", then=then or "sent")

    def _clusters(self, fighters: list[dict]) -> list[list[dict]]:
        """Fighters split by where they stand: up to CLUSTERS, biggest first; units within 2 grid cells
        of a cluster's busiest cell join it (as the Terran's army layer clusters)."""
        left, out = list(fighters), []
        while left and len(out) < CLUSTERS:
            cells = [self.army_tracker.cell(u["x"], u["y"]) for u in left]
            center = max(set(cells), key=cells.count)
            near = [u for u, c in zip(left, cells) if _cell_dist(c, center) <= 2]
            out.append(near)
            left = [u for u in left if u not in near]
        return out

    def _army_job_step(self, obs: dict, mine: list[dict]) -> bool:
        """The cluster order under way: the next dozen selected, then ordered, until all have it."""
        hi = self.hi
        job = self.army_job
        alive = {u["tag"]: u for u in mine}
        units = [alive[t] for t in job["tags"] if t in alive]
        if not units:
            self.army_job = None
            return False
        batch = units[:MAX_SELECT]
        sel = set(obs["selection"])
        fighters = {u["tag"] for u in mine if u["type"] in self.sizes}
        # a selection that is the army's own fighters and holds some of the order's units goes: a click on
        # one in a hundred bunched zerglings took the wrong one, and the order was dropped after 3 tries
        if job["selecting"] and sel and sel <= fighters and sel & set(job["tags"]):
            x, y = int(job["point"][0]), int(job["point"][1])
            if job["kind"] == "move":
                hi.minimap_right_click(x, y)
            else:
                hi.minimap_command(C.ORDER_ATTACK_MOVE, x, y)
            job["tags"] = [t for t in job["tags"] if t not in sel]
            job["selecting"], job["tries"] = False, 0
            self.army_sent += 1
            if not job["tags"]:
                self.army_job = None
            return True
        job["tries"] += 1
        if job["tries"] > 3:                               # (they won't be selected: the rest dropped)
            self.army_job = None
            self.army_dropped += 1
            return False
        mid = (sum(u["x"] for u in batch) / len(batch), sum(u["y"] for u in batch) / len(batch))
        if self._look(obs, mid):
            return True
        shown = [u for u in batch if screen_of(obs, u["x"], u["y"])]
        if not shown:
            job["tags"] = [t for t in job["tags"] if t not in {u["tag"] for u in batch}]
            return True
        # a ctrl+click on one of the batch's commonest type: every one of that type on the screen (12 at most)
        main = Counter(u["type"] for u in shown).most_common(1)[0][0]
        u = next(u for u in shown if u["type"] == main)
        hi.ctrl_click(*screen_of(obs, u["x"], u["y"]))
        job["selecting"] = True
        return True

    # --- deciding --------------------------------------------------------------------------------

    def _in_production(self, mine: list[dict]) -> dict[int, int]:
        """Units ordered and not out yet (eggs and cocoons hide what they'll be): from Gary-Z's own
        orders, each for as long as it takes to make."""
        f = self.hi.frame
        self.morphs = [m for m in self.morphs if f - m[0] < m[2]]
        out: dict[int, int] = {}
        for _, t, _ in self.morphs:
            out[t] = out.get(t, 0) + (2 if t in Z.PAIRS else 1)
        return out

    def _frames_to_make(self, uid: int) -> int:
        if uid not in self.build_frames:
            info = self.hi.game.unit_type_info(uid) if hasattr(self.hi.game, "unit_type_info") else None
            self.build_frames[uid] = int(info["build_frames"]) if info else 30 * 24
        return self.build_frames[uid]

    def _cost(self, act: int, uid: int) -> tuple[int, int]:
        if act == EXPAND:
            return Z.COST[Z.HATCHERY]
        if act == RESEARCH:
            return Z.RESEARCH[uid][1]
        if act == UPGRADE:
            return Z.UPGRADE[uid][1]
        return Z.COST.get(uid, (0, 0))

    def _free(self, me: dict, but: dict | None = None) -> tuple[int, int]:
        """The money free now: the bank, less what's just spent, the buildings on their way, and the
        parked decision's (unless it's the one asking)."""
        minerals, gas = self._bank(me)
        minerals -= self._kept()
        if self.parked is not None and self.parked is not but:
            pm, pg = self._cost(self.parked["act"], self.parked["id"])
            minerals, gas = minerals - pm, gas - pg
        # (none below zero: a Lurker Aspect parked for gas, 200 of a 158 bank, left -42 gas free and no
        # zergling (0 gas) was affordable: a minute with 9 larvae and 1,480 minerals unspent)
        return max(0, minerals), max(0, gas)

    def _affordable(self, g: dict, me: dict) -> bool:
        if g["act"] in (BUILD, EXPAND):                     # (a job keeps its own money, from the walk on)
            return True
        m, gas = self._cost(g["act"], g["id"])
        fm, fg = self._free(me, g)
        return fm >= m and fg >= gas

    def _decide(self, obs: dict, mine: list[dict], me: dict) -> None:
        """The next decision. One the money doesn't cover yet (a Lair or zergling speed waiting for gas,
        a Spire for minerals) is parked, its money kept, and the larvae go on meanwhile with what's left
        (a decision at a time held every larva: 8 drones at 2:00, 21 at 8:00, Zerg v3 11 and 41)."""
        hi = self.hi
        for which in ("goal", "parked"):
            g = getattr(self, which)
            if g is not None and hi.frame - g["since"] > GOAL_TIMEOUT:
                self.log(f"gives up on {g['name']}")
                LOG.order("Z", hi.frame, g["name"], "dropped", f"{which}: not carried out in {GOAL_TIMEOUT // 24} s")
                setattr(self, which, None)
        if self.goal is None and self.parked is not None and self._affordable(self.parked, me):
            self.goal, self.parked = self.parked, None          # (its money is there: it goes first)
            LOG.decision("Z", "macro", hi.frame, self.goal["name"], self.goal["p"], inputs=self._seen(obs, mine, me),
                         by="parked", then="unparked: its money is there")
            return
        if self.goal is not None or hi.frame < self.next_query:
            return
        self.next_query = hi.frame + QUERY_EVERY
        x = self.tracker.features(obs, self._in_production(mine))
        probs, secs = self.model.probs(x)
        # the pros' tempo: the model also predicts the seconds until their next decision; the next roll
        # comes then, not every half second (rolled every 0.5 s, a decision at a few percent came as
        # often as the drones: 100 creep colonies a minute across 8 games)
        self.next_query = hi.frame + int(min(PACE_MAX, max(QUERY_EVERY, float(secs[0]) * 24)))
        # the dice, as the Terran's macro: the model's odds rolled (a weighted shuffle), then the first
        # decision Gary-Z can carry out; the favorite alone is a drone nearly every time, and the Lair,
        # zergling speed or an army never came (the first game: 2 zerglings and no Lair by 8:00)
        order = [k for _, k in sorted(((self.rng.random() ** (1 / p), k) for k, p in enumerate(probs[0])
                                        if p >= DICE_FLOOR), reverse=True)]
        larva = any(u["type"] == Z.LARVA for u in mine)
        waiting = False
        # (the record: the model's top odds, what it saw, and each option drawn ahead of the choice and why
        # it was set aside; gary/decision_log.py)
        name = lambda k: self._name(*self.model.spec.decisions[k])
        over: list[list] = []

        def record(choice, p, then=None):
            LOG.decision("Z", "macro", hi.frame, choice, p, top(name, probs[0]), self._seen(obs, mine, me), "dice",
                         over[:8], then)
        for k in order:
            act, uid = self.model.spec.decisions[k]
            if not larva and act == MORPH and uid in Z.FROM_LARVA and Z.can_make(uid, self._done_types(mine), self.researched) \
                    and (uid == Z.OVERLORD or me["supply_used"] + Z.SUPPLY.get(uid, 1) <= me["supply_max"]):
                waiting = True  # (no larva free yet: that one waits for the next, as the Terran's macro waits for
                over.append([name(k), "no larva free: waits for one"])     # a busy building)
                continue
            if act == BUILD and uid in TECH_PATH and probs[0, k] < TECH_MIN_P:
                over.append([name(k), f"rule: a tech path under {TECH_MIN_P:.0%}"])
                continue        # (a tech path opened only on the model's real odds: drawn at 1-6% across dozens of
                                # rolls, a Hydralisk Den came in 21 of 24 spire-first games, then Lurker Aspect, and
                                # the model, reading that history as a lurker player's, gave mutalisks 0.8% where
                                # without the history it gave 16%: 12 of 24 games made none)
            if waiting and probs[0, k] < MIN_P:
                over.append([name(k), f"rule: under {MIN_P:.0%} while a larva is awaited"])
                continue        # (meanwhile only what the model really wants, MIN_P or more: taking the first thing
                                # possible made a creep colony of every larva shortage, 44-66% of the decisions
                                # from 3:00 to 8:00, rolled at 1-5%; waiting for every one held a macro hatchery
                                # back with 800 minerals floating: Argon's catch)
            if act < 0 or not self._can(act, uid, obs, mine, me):
                over.append([name(k), "can't now (what it needs, the pros' cap, nothing to make it from)"])
                continue
            g = {"act": act, "id": uid, "p": float(probs[0, k]), "since": hi.frame, "name": self._name(act, uid)}
            larva = act == MORPH and uid in Z.FROM_LARVA
            if self._affordable(g, me):
                self.goal = g
                self.log(f"next: {g['name']} ({g['p']:.0%})")
                record(g["name"], g["p"])
                if larva:                       # (pros spend larvae in bursts, a hatchery's at once: the timing
                    self.next_query = hi.frame + QUERY_EVERY    # head's gap is between bursts; paced by it,
                return                                          # the larvae piled up, 2,000 minerals at 10:00)
            if not larva and self.parked is None:
                self.parked = g
                self.log(f"next: {g['name']} ({g['p']:.0%}), parked for its money")
                record(g["name"], g["p"], "parked for its money")
                over.append([g["name"], "parked for its money"])
                continue                                     # (and a larva meanwhile, if one is affordable)
            if self.parked is not None and not larva:
                over.append([g["name"], "one already parked: meanwhile only larvae"])
                continue                                     # (one parked at a time; meanwhile only larvae)
            record(None, None, f"waits: {g['name']} needs money")   # (a larva the money doesn't cover yet: wait)
            return
        record(None, None, "nothing it can do now")

    def _seen(self, obs: dict, mine: list[dict], me: dict) -> dict:
        """What the macro saw, in brief, for its records."""
        def n(*types):
            return sum(1 for u in mine if u["type"] in types)
        return {"m": me["minerals"], "gas": me["gas"], "supply": f"{me['supply_used']}/{me['supply_max']}",
                "larvae": n(Z.LARVA), "drones": n(Z.DRONE), "hatcheries": n(*Z.TOWN_HALLS),
                "army": n(*Z.ARMY), "parked": self.parked["name"] if self.parked else None,
                "gas_floating": self.gas_floating}

    def _name(self, act: int, uid: int) -> str:
        if act == RESEARCH:
            return Z.TECH_NAMES.get(uid, f"tech {uid}")
        if act == UPGRADE:
            return Z.UPGRADE_NAMES.get(uid, f"upgrade {uid}")
        if act == EXPAND:
            return "expand"
        return Z.NAMES.get(uid, str(uid))

    def _done_types(self, mine: list[dict]) -> set[int]:
        return {u["type"] for u in mine if u["completed"]}

    def _can(self, act: int, uid: int, obs: dict, mine: list[dict], me: dict) -> bool:
        """Gary-Z could carry this out now, money aside: what it needs exists, there's something to
        make it from, and the pros' cap for this time isn't reached."""
        done = self._done_types(mine)
        cap = self.model.cap(uid, obs["frame"]) if act in (MORPH, BUILD, BMORPH, EXPAND) else None
        if cap is not None and self._count(mine, uid) >= cap:
            return False
        if uid == Z.CREEP:                              # (and all colonies within the pros' caps of all three)
            caps = [self.model.cap(t, obs["frame"]) for t in (Z.CREEP, Z.SUNKEN, Z.SPORE)]
            if all(c is not None for c in caps) and self._count(mine, Z.CREEP, every=True) >= sum(caps):
                return False
        if act == MORPH:
            if uid in Z.FROM_LARVA:
                if not any(u["type"] == Z.LARVA for u in mine):
                    return False
                if uid != Z.OVERLORD and me["supply_used"] + Z.SUPPLY.get(uid, 1) > me["supply_max"]:
                    return False
                return Z.can_make(uid, done, self.researched)
            parent = Z.UNIT_MORPH.get(uid)
            return parent is not None and any(u["type"] == parent and u["completed"] for u in mine) \
                and Z.can_make(uid, done, self.researched)
        if act == BMORPH:
            parent = Z.BUILDING_MORPH.get(uid)
            return parent is not None and Z.can_make(uid, done, self.researched) \
                and any(u["type"] == parent and u["completed"] and u.get("queue", 0) == 0 for u in mine)
        if act == BUILD:
            if uid not in Z.DRONE_BUILDS or not Z.can_make(uid, done, self.researched):
                return False
            if any(j.unit_type == uid for j in self.jobs.values()):
                return False                            # (one of a kind on its way at a time)
            if uid == Z.EXTRACTOR:
                return self._free_geyser(mine) is not None
            return any(u["type"] == Z.DRONE for u in mine)
        if act == EXPAND:
            return not any(j.step.startswith("expand") for j in self.jobs.values()) and self._next_base(obs, mine) is not None
        if act == RESEARCH:
            where, _ = Z.RESEARCH.get(uid, (None, None))
            return where is not None and uid not in self.researched \
                and all(Z.has(done, r) for r in Z.RESEARCH_NEEDS.get(uid, [])) \
                and self._idle_building(mine, where) is not None
        if act == UPGRADE:
            if uid not in Z.UPGRADE:
                return False
            where, _, levels = Z.UPGRADE[uid]
            level = self.upgraded.get(uid, 0) + 1
            need = Z.UPGRADE_LEVEL_NEEDS.get(level) if levels > 1 else None
            return level <= levels and all(Z.has(done, r) for r in Z.UPGRADE_NEEDS.get(uid, [])) \
                and (need is None or Z.has(done, need)) and self._idle_building(mine, where) is not None
        return False

    def _count(self, mine: list[dict], uid: int, every: bool = False) -> int:
        """How many Gary-Z has of uid for the pros' cap: those standing and those on their way (jobs);
        every: with what they became (a creep colony's sunken or spore; a hatchery's Lair or Hive, always).
        Without the jobs the cap never came: the model's few-percent creep colony, rolled every half
        second while larvae were short, was decided 100 times a minute across 8 games."""
        became = {Z.HATCHERY: Z.TOWN_HALLS}.get(uid, {Z.CREEP, Z.SUNKEN, Z.SPORE} if every and uid == Z.CREEP else {uid})
        return sum(1 for u in mine if u["type"] in became) + sum(1 for j in self.jobs.values() if j.unit_type == uid)

    def _idle_building(self, mine: list[dict], kind: int) -> dict | None:
        """A finished building of this kind (a Lair or Hive counts as a Hatchery...) not busy."""
        kinds = Z.HAS_TECH.get(kind, {kind})
        return next((u for u in mine if u["type"] in kinds and u["completed"] and u.get("queue", 0) == 0
                     and u.get("order") not in (T.ORDER_RESEARCH, T.ORDER_UPGRADE)), None)

    # --- carrying it out -------------------------------------------------------------------------

    def _carry_out(self, obs: dict, mine: list[dict], me: dict) -> None:
        g = self.goal
        act, uid = g["act"], g["id"]
        cost = Z.COST.get(uid) if act in (MORPH, BMORPH, BUILD, EXPAND) else \
            Z.RESEARCH[uid][1] if act == RESEARCH else Z.UPGRADE[uid][1]
        if act == EXPAND:
            cost = Z.COST[Z.HATCHERY]
        if act in (BUILD, EXPAND):                  # (a job keeps its money from the walk on)
            self._new_job(obs, mine, act, uid)
            return
        if not self._affordable(g, me):
            return
        if act == MORPH and uid in Z.FROM_LARVA:
            if self._larva_morph(obs, mine, uid):
                self._done(act, uid)
            return
        if act == MORPH:
            parent = Z.UNIT_MORPH[uid]
            units = [u for u in mine if u["type"] == parent and u["completed"]]
            if self._select_one(obs, units) and self.hi.morph(uid).accepted:
                self._done(act, uid)
            return
        if act == BMORPH:
            parent = Z.BUILDING_MORPH[uid]
            units = [u for u in mine if u["type"] == parent and u["completed"] and u.get("queue", 0) == 0]
            if self._select_one(obs, units) and self.hi.morph_building(uid).accepted:
                self._done(act, uid)
            return
        where = Z.RESEARCH[uid][0] if act == RESEARCH else Z.UPGRADE[uid][0]
        b = self._idle_building(mine, where)
        if b is None:
            self.goal = None
            return
        if self._select_one(obs, [b]):
            ok = self.hi.research(uid).accepted if act == RESEARCH else self.hi.upgrade(uid).accepted
            if ok:
                if act == RESEARCH:
                    self.researched.add(uid)
                    if uid == Z.LURKER_ASPECT:
                        self.aspect_at = self.hi.frame
                else:
                    self.upgraded[uid] = self.upgraded.get(uid, 0) + 1
                self._done(act, uid)

    def _bank(self, me: dict) -> tuple[int, int]:
        """The bank less what was just spent and doesn't show yet (the first game: 9 of 60 morphs went
        out short, the last one's cost not taken yet)."""
        f = self.hi.frame
        self.spent = [s for s in self.spent if f - s[0] < IN_FLIGHT]
        return me["minerals"] - sum(s[1] for s in self.spent), me["gas"] - sum(s[2] for s in self.spent)

    def _done(self, act: int, uid: int) -> None:
        if act not in (BUILD, EXPAND):              # (a job keeps its own money)
            cost = Z.COST.get(uid, (0, 0)) if act in (MORPH, BMORPH) else \
                Z.RESEARCH[uid][1] if act == RESEARCH else Z.UPGRADE[uid][1]
            self.spent.append((self.hi.frame, *cost))
        self.tracker.note_decision(act, uid, self.hi.frame)
        if act == MORPH:
            self.morphs.append((self.hi.frame, uid, self._frames_to_make(uid)))
            if uid == Z.OVERLORD:
                self.overlords_at.append(self.hi.frame)
        self.goal = None

    def _select_one(self, obs: dict, units: list[dict]) -> bool:
        """One of these units selected (the camera on it, then a click). True once it is."""
        if not units:
            self.goal = None
            return False
        sel = set(obs["selection"])
        if len(sel) == 1 and any(u["tag"] in sel for u in units):
            return True
        u = min(units, key=lambda u: math.dist((u["x"], u["y"]), self.home))
        if self._look(obs, (u["x"], u["y"])):
            return False
        p = screen_of(obs, u["x"], u["y"])
        if p is not None:
            self.hi.click(*p)
        return False

    def _larva_morph(self, obs: dict, mine: list[dict], uid: int) -> bool:
        """A larva selected (a tiny box), then the morph. True once the game took it. A larva just
        morphed still shows as one for a moment: not again (the first game: 29 of 60 morphs went to
        an egg)."""
        f = self.hi.frame
        self.used_larva = {t: at for t, at in self.used_larva.items() if f - at < USED_LARVA}
        fresh = [u for u in mine if u["type"] == Z.LARVA and u["tag"] not in self.used_larva]
        sel = [u for u in mine if u["tag"] in obs["selection"]]
        if sel and all(u["type"] == Z.LARVA and u["tag"] not in self.used_larva for u in sel):
            if self.hi.morph(uid).accepted:
                for u in sel:
                    self.used_larva[u["tag"]] = f
                return True
            return False
        if not fresh:
            return False
        others = [u for u in mine if u["type"] != Z.LARVA or u["tag"] in self.used_larva]
        self._pick_larva(obs, fresh, others)
        return False

    def _pick_larva(self, obs: dict, fresh: list[dict], others: list[dict]) -> None:
        """A tiny box on a fresh larva on the screen (one with nothing else on it), or the camera to one."""
        hi = self.hi
        on = [lv for lv in fresh if screen_of(obs, lv["x"], lv["y"])]
        if not on:
            lv = fresh[(hi.frame // 48) % len(fresh)]
            hi.camera_minimap(lv["x"], lv["y"])
            return
        clear = [lv for lv in on if not any(abs(u["x"] - lv["x"]) < 12 and abs(u["y"] - lv["y"]) < 12 for u in others)]
        lv = (clear or on)[(hi.frame // 48) % len(clear or on)]
        x, y = screen_of(obs, lv["x"], lv["y"])
        hi.box(max(0, x - 3), max(0, y - 3), x + 3, y + 3)

    def _new_job(self, obs: dict, mine: list[dict], act: int, uid: int) -> None:
        """A drone's building as one of v3's jobs (they pick the drone, walk it, place, retry)."""
        if act == EXPAND:
            base = self._next_base(obs, mine)
            tile = base.tile if base is not None else None
            unit, step = Z.HATCHERY, f"expand_{self.hi.frame}"
        else:
            unit, step = uid, f"{Z.NAMES.get(uid, uid)}_{self.hi.frame}"
            tile = self._building_spot(uid, mine)
        if tile is None:
            self.log(f"no spot for {self._name(act, uid)}")
            LOG.order("Z", self.hi.frame, self._name(act, uid), "dropped", "no spot for it", by="job")
            self.goal = None
            return
        self.jobs[step] = Job(step, unit, tuple(tile), since=self.hi.frame)
        self.log(f"builds {self._name(act, uid)} at {tuple(tile)}")
        self._done(act, uid)

    def _building_spot(self, uid: int, mine: list[dict]):
        """A spot for uid, off the spots of buildings on their way (two creep colonies went to one)."""
        tile = self._spot_for(uid, mine)
        if tile is None or uid == Z.EXTRACTOR:
            return tile
        w, h = Z.SIZE.get(uid, (2, 2))
        for j in self.jobs.values():
            jw, jh = Z.SIZE.get(j.unit_type, (2, 2))
            if tile[0] < j.tile[0] + jw and j.tile[0] < tile[0] + w and tile[1] < j.tile[1] + jh and j.tile[1] < tile[1] + h:
                return None
        return tile

    def _spot_for(self, uid: int, mine: list[dict]):
        if uid == Z.EXTRACTOR:
            return self._free_geyser(mine)
        if uid == Z.CREEP:                          # (colonies in front of the natural, as v3's sunkens)
            nat = self.nat_base.center
            front = (int(nat[0] + (self.gather[0] - nat[0]) * 0.6), int(nat[1] + (self.gather[1] - nat[1]) * 0.6))
            owned = any(u["type"] in Z.TOWN_HALLS and math.dist((u["x"], u["y"]), nat) < 6 * 32 for u in mine)
            return self._spot_near(Z.CREEP, front if owned else self.home, 0 if owned else 3, 8)
        return self._spot_near(uid, self.home)

    def _owned_bases(self, mine: list[dict]) -> list:
        halls = [(u["x"], u["y"]) for u in mine if u["type"] in Z.TOWN_HALLS]
        return [b for b in self.map.bases if any(math.dist(b.center, h) < 6 * 32 for h in halls)]

    def _free_geyser(self, mine: list[dict]):
        taken = [(u["x"], u["y"]) for u in mine if u["type"] == Z.EXTRACTOR]
        taken += [self._spot(j) for j in self.jobs.values() if j.unit_type == Z.EXTRACTOR]
        for base in self._owned_bases(mine):
            for gx, gy in base.geysers:
                if not any(math.dist((gx, gy), t) < 3 * 32 for t in taken):
                    return (gx - 64) // 32, (gy - 32) // 32
        return None

    def _next_base(self, obs: dict, mine: list[dict]):
        """The nearest base to the natural that nobody holds (no town hall of either side seen there)."""
        owned = self._owned_bases(mine)
        enemy = [(u["x"], u["y"]) for u in obs["units"] if u["owner"] not in (self.hi.slot, 11)
                 and 106 <= u["type"] <= 175]
        free = [b for b in self.map.bases if b not in owned and b.minerals
                and not any(math.dist(b.center, e) < 10 * 32 for e in enemy)
                and not any(j.unit_type == Z.HATCHERY and math.dist(self._spot(j), b.center) < 6 * 32 for j in self.jobs.values())]
        if self.nat_base not in owned and self.nat_base in free:
            return self.nat_base
        nat = self.nat_base.center
        return min(free, key=lambda b: math.dist(b.center, nat), default=None)
