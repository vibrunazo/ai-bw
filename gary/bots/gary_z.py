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
from gary import orders as O
from gary import zerg as Z
from gary.bots import zerg_v3 as V3
from gary.decision_log import LOG, top
from gary.bots.zerg_v3 import Job, ZergV3
from gary.interface import screen_of
from gary.policy.army import ArmyModel, ArmyTracker
from gary.policy.attribution import explain_army, explain_macro
from gary.policy.macro import MacroModel, MacroTracker, macro_model_path
from gary.races import enemy_of

MORPH, BMORPH, BUILD, RESEARCH, UPGRADE, EXPAND = 1, 2, 3, 4, 5, 6
MIN_P = 0.05                    # never a decision the model gives less than this (as the Terran's macro)
TECH_MIN_P = 0.10               # a building that opens a tech path: never on odds under this
TECH_PATH = {Z.DEN, Z.SPIRE, Z.QUEENS_NEST, Z.DEFILER_MOUND, Z.ULTRA_CAVERN}
DICE_FLOOR = 0.01               # the dice decoder: decisions under this odds aren't rolled
USED_LARVA = 2 * 24             # a larva just morphed shows as a larva a moment longer: not used again
IN_FLIGHT = 24                  # money just spent shows in the bank a moment later: counted as spent
GOAL_TIMEOUT = 30 * 24          # a decision not carried out in 30 s is dropped
COMMIT_TICK = 5 * 24            # commit mode: each cluster asked this often (ingest/army_dataset.py TICK)
CLEAR_PX = 16                   # _select_one: a unit with another of Gary-Z's this close may be under it
QUERY_EVERY = 12
GAS_FLOAT, GAS_LOW = 400, 300   # gas above this (and above the minerals): its drones go back to minerals;
                                # below this, back to gas (150 trapped it: no gas mined from 4:00 to 8:00
                                # with ~160 banked, too little for mutalisks; pros bank 350-450 then)
RALLY_IN_LINE = 0.6              # the rally: this far from the hatchery toward its minerals' middle
# the mutalisk raid (_mutas)
HATCH_MASS = float(os.environ.get("GARY_Z_HATCH_MASS", "0.5"))   # a hatchery or expansion at this much summed odds (one-offs: 1.0):
                                 # the summed odds reached 1.6 a game, one hatchery fewer than the pros from 8:00 on
PRESS_RATIO = 2.0                # the press: own fighter supply this many times the Terran fighters seen lately...
PRESS_MIN = 40                   # ...and at least this (x2: 20 supply)
PRESS_MEMORY = 60 * 24           # Terran fighters seen this long ago still count
PRESS_AGAIN = 20 * 24            # the press re-aimed this often
INTERCEPT_FAR = 24 * 32          # the main body sweeps a Terran packet in sight this near...
INTERCEPT_ODDS = 0.7             # ...with under this share of its own supply, away from turrets and bunkers
INTERCEPT_AGAIN = 8 * 24         # a sweep ordered at most this often
FLOCK_SAVE = os.environ.get("GARY_Z_FLOCK_SAVE", "0") == "1"   # the saving below (off: it made the flock, 8 in 30 s, and
                                 # changed no result: round g10 Gary-T won 10 lost 1 of 24 against g8's 8 and 1, the gap 0.33 against 0.31)
RALLY_SIEGE = 14 * 32            # enemy fighters this near the meeting point: the hatcheries rally to the main
RALLY_SIEGE_FOR = 20 * 24        # ...and keep to it this long after the last was seen there
HISTORY_FADE = 12                # minutes over which the pros' history fades to Gary-Z's own in the macro's inputs
ALARM_UNTIL = 12 * 60 * 24       # the alarm (_decide): until then...
ALARM_MARGIN = 16                # ...the enemy fighters seen in a minute outweigh the army by this (x2: 8 supply)
ALARM_SUNKENS = 3                # sunkens at the natural at most, one per 8 supply of threat
ALARM_HOME = 16 * 32             # Terran fighters this near a known Command Center of theirs are at home: no threat
ALARM_MIN_P = 0.30               # under the alarm, tech, expansions, research and upgrades need this much odds
SAVE_MIN_P = 0.20                # while saving for the flock, a decision that isn't an overlord or a colony needs this
                                 # much odds (colonies are the defense the push meets while the flock is saved for: without
                                 # them Gary-Z died by 10-15 min in 5 of 8 Heartbreak Ridge games, round g9)
RAID_FLOCK = 5                   # mutalisks at home before the flock forms and goes out (v3: 7)
RAID_MEMORY = 90 * 24            # anti-air seen is remembered this long
RAID_POST = 7 * 32               # px beyond a mineral line (out from the map's middle) where the flock waits
RAID_TURRET = 8 * 32             # a turret (range 7) this near the flock, or a line: a threat
RAID_SEE = 8 * 32                # marines this near the flock count
RAID_WAYPOINT = 14 * 32          # the approach: along the map's edge, this far aside of the post, then in
RAID_MARINES = 3                 # this many marines (or better) within reach: back off
RAID_HIT_HP = 30                 # the flock losing this much hp (a turret shot, five marine volleys)...
RAID_HIT_S = 2                   # ...in this many seconds: back off
RAID_REST_S = 6                  # seconds at the post before coming back
RAID_CYCLES = 3                  # back-offs at one line before another line (or home)
RAID_SWITCH = 10 * 24            # lines switched for a turret at most this often
RAID_REAIM = 2 * 24              # the SCV aimed at again this often
RAID_KEEP = 0.6                  # home with fewer than this share of the flock it left with
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
JOIN_BELOW = 1.0                # a cluster under this share of the main one's size joins it (0.5 until round g3: clusters
                                # of a median 11 still went out on their own and died in pieces, 118 fighters a game)
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


def latest_model(kind: str = "macro", matchup: str = "TvZ") -> Path:
    found = sorted(Path(f"runs/{kind}").glob(f"{matchup}_Z_*/model.pt"))
    if not found:
        raise FileNotFoundError(f"no Zerg {kind} model for {matchup} in runs/{kind}: python -m train.{kind} --matchup {matchup} --race Z")
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
    def __init__(self, hi, mapinfo, seed: int = 0, model: str | Path | None = None, plan: int | None = None,
                 matchup: str = "TvZ"):
        """matchup: whose models and pro tables (ZvZ for a mirror); the enemy's race is its other letter."""
        super().__init__(hi, mapinfo, seed)
        self.matchup = matchup
        self.enemy = enemy_of(matchup, "Z")              # (its buildings and fighters by race: gary/races.py)
        self.model = MacroModel.load(model or macro_model_path(matchup, "Z"))
        plans = load_plans() if matchup == "TvZ" else None
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
        self.tracker = MacroTracker(self.model.spec, hi.slot, plan, mapinfo.hash, matchup)
        # GARY_Z_HISTORY (default pro; own: this player's): the decision history the macro model sees (gary/policy/macro.py
        # MacroTracker.history): the pros' mean per minute, so its own drone-heavy past doesn't feed back on itself
        hist = os.environ.get("GARY_Z_HISTORY", "pro")      # pro (fading to its own over HISTORY_FADE minutes), pro_all
        if hist != "own":                                    # (the pros' all game), own, or a constant share (0.5)
            from gary.policy.macro import pro_history
            self.tracker.history = pro_history("Z", plan, matchup)      # (its own rolled style's pros)
            self.tracker.history_weight = 1.0 if hist in ("pro", "pro_all") else float(hist)
            self.tracker.history_until = HISTORY_FADE if hist == "pro" else 0
        # where the groups go: the Zerg army model (runs/army/TvZ_Z_*), as the Terran's army; v1's
        # stage-judge-go is replaced, its home defense and v3's raid and lurkers stay
        # GARY_Z_COMMIT ("latest" or a model's path): commitments (train/commit.py --race Z: keep / engage / move
        # every 5 s per cluster, given its commitment), the commit model also choosing destinations
        commit = os.environ.get("GARY_Z_COMMIT")
        self.army_commit = bool(commit)
        self.select_misses = 0                          # _select_one: clicks so far that selected something else
        self.army_model = ArmyModel.load(latest_model("commit", matchup) if commit == "latest" else commit) if commit \
            else ArmyModel.load(latest_model("army", matchup))
        self.commitments: list[tuple[int, int, int, int]] = []   # (frame, from cell, to cell, 1 engage / 2 move)
        self.army_tracker = ArmyTracker(self.army_model.spec, hi.slot, mapinfo.size, self.home)
        self.sizes = dict(GROUND)
        self.next_ask: dict[int, int] = {}       # group -> when its destination is drawn again
        self.army_goal: dict[int, tuple | None] = {}   # group -> its drawn destination
        self.sent_to: dict[int, tuple] = {}
        self.straggle_after, self.straggle_tries = 0, 0
        self.unreached: set[tuple[int, int]] = set()   # bases its drones never reached: not expanded to again
        # the mutalisk raid (_mutas): the line raided, back-offs at it, the flock's hp lately, when each line was
        # last hit, and Gary's Command Centers seen (their mineral lines are raid targets too)
        self.raid_line = None
        self.intercept_after = 0
        self.alarm_on = False
        self.press_after = 0
        self.enemy_buildings: dict[int, tuple[int, int, int]] = {}
        self.enemy_fighters: dict[int, tuple[int, int]] = {}
        self.raid_cycles = 0
        self.raid_switched = -10 ** 6
        self.raid_hp: list[tuple[int, int]] = []
        self.raided_at: dict[tuple, int] = {}
        self.seen_halls: dict[int, tuple[int, int]] = {}
        # the army by clusters, not v1's hotkey groups (GARY_Z_ARMY=groups for those): 24 games on the same
        # seeds, at home 47% at 5-7 min (groups 78%, pros 38%), in the field 47% (20%, 52%), Gary-Z's gap
        # score 0.34 (0.45), decided 38% (17%), lone units moving away 10% (3%, pros 18%)
        self.cluster_army = os.environ.get("GARY_Z_ARMY", "clusters") == "clusters"
        # GARY_Z_FLOOR (default 0: off): no keep / engage / move drawn under these odds, the destination among the
        # 5 likeliest cells. Off since L7 (48 games on the ladder maps against L6's 0.15): a cluster's engage odds
        # are 5.5% a tick (median) and the floor cut most engages (7% of draws at 8-10 minutes, pros 12%); off,
        # Gary-Z's gap score against the modern pros 0.31 (0.33), lower in every minute range, its army home 64%
        # of 8-10 minutes (72%; pros 19%), ahead at the limit in 11 games (3)
        self.commit_floor = float(os.environ.get("GARY_Z_FLOOR", "0")) or None
        # GARY_Z_MASS (default on; 0: off): every one-off decision (a hatchery or an expansion, a tech building,
        # Lair / Hive / Greater Spire, research, an upgrade) once its odds, summed over the decisions taken
        # while it was possible, reach one (the Terran macro's expand_mass and option mass). The dice roll
        # them at 1-5% a decision and seldom land: Gary-Z had one hatchery fewer than the pros from 8:00
        # (3/4, 4/5) with every larva used (capped 0%, pros 10%) and 540-650 minerals banked (pros 380-430),
        # its army half the pros' at 10:00
        self.mass = os.environ.get("GARY_Z_MASS", "1") != "0"
        self.one_off = [k for k, (a, u) in enumerate(self.model.spec.decisions)
                        if a in (RESEARCH, UPGRADE, EXPAND) or (a == BMORPH and u in (Z.LAIR, Z.HIVE, Z.GREATER_SPIRE))
                        or (a == BUILD and u in (Z.HATCHERY, Z.DEN, Z.SPIRE, Z.QUEENS_NEST, Z.DEFILER_MOUND, Z.ULTRA_CAVERN, Z.EVO))]
        self.odds: dict[int, float] = {}         # one-off decision -> its odds summed since it was possible
        from gary.agent.sightings import Sightings   # what Gary-Z saw and lost between decisions (its records)
        # (fighters only: a drone "dies" to the tracker when it morphs or enters an extractor)
        self.sightings = Sightings(hi, "Z", mapinfo, self.home, lambda u: u["type"] in Z.ARMY)
        self.army_job: dict | None = None
        self.cluster_ask: dict[int, int] = {}
        self.cluster_turn = 0
        self.army_sent = self.army_dropped = 0   # (cluster orders given, and dropped unselected)
        self.lurk_streak, self.lurk_backoff = 0, 0
        self.task_key, self.task_since = None, 0      # group -> the destination it was last sent to
        self.rally_of: dict[int, str] = {}      # hatchery -> where it rallies: "minerals", "front" or "back" (besieged)
        self.siege_until = 0                     # the meeting point had enemy fighters by it until then
        self.goal: dict | None = None
        self.parked: dict | None = None        # a decision waiting for its money, kept for it meanwhile
        self.turn, self.streak = "macro", 0    # who goes first, and how many actions in a row it had
        self.gas_floating = False              # the gas piles up: its drones mine minerals meanwhile
        self.post_of: dict[int, int] = {}      # watch post -> its overlord
        self.post_again: dict[int, int] = {}   # watch post -> frame from which it's manned again
        self._post_points = None
        self.next_query = 0
        self.rng = random.Random(seed * 7919 + 1)          # the dice (deterministic per seed)
        # its fights: the Zerg role model (gary/agent/zerg_roles.py: the pros' Zerg units before contact and
        # in it), in place of v3's scripted micro; none trained yet, v3's
        from gary.agent.zerg_roles import ZergRoles, latest_zerg_model
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
        self.sightings.look(obs)
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

    def gather_point(self):
        """Where fighters gather: the front, or the main while the front is besieged (_rallies)."""
        return self.home if self.hi.frame < self.siege_until else self.gather

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
        # besieged: enemy fighters by the meeting point (the natural's front): every hatchery rallies to the main
        # instead, for RALLY_SIEGE_FOR after they leave. On Python the push camped Gary-Z's natural from 7:05 and
        # every pair of zerglings hatched into it: 66 deaths in 6-10 min, 55 at the natural, 58 with two or fewer
        # of their own near (round g12 seed 5005). Pros re-rally to the main when the natural is contested
        from gary.policy.army import is_building, is_worker, supply_x2
        if any(u["owner"] not in (hi.slot, 11) and supply_x2(u["type"]) > 0 and not is_worker(u["type"]) and not is_building(u["type"])
               and math.dist((u["x"], u["y"]), self.gather) < RALLY_SIEGE for u in obs["units"]):
            self.siege_until = hi.frame + RALLY_SIEGE_FOR
        mode = "back" if front and hi.frame < self.siege_until else "front" if front else "minerals"
        for h in hatches:
            if not h["completed"] or self.rally_of.get(h["tag"]) == mode:
                continue
            base = self.map.base_near((h["x"], h["y"]))
            if not base.minerals and mode == "minerals":
                self.rally_of[h["tag"]] = mode
                continue
            if mode == "back":
                spot = self.home
            elif mode == "front":
                spot = self.gather
            else:
                mx = sum(x for x, _ in base.minerals) / len(base.minerals)
                my = sum(y for _, y in base.minerals) / len(base.minerals)
                spot = (int(h["x"] + (mx - h["x"]) * RALLY_IN_LINE), int(h["y"] + (my - h["y"]) * RALLY_IN_LINE))
            if self._look(obs, (h["x"], h["y"])):
                return True
            if obs["selection"] == [h["tag"]]:
                hi.minimap_right_click(*spot)
                self.rally_of[h["tag"]] = mode
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
        cell = self._draw_cell(where[0])
        attack = self.rng.random() < float(kind[0, 1])
        held = attack and self.army_model.fight(grid, glob, cell, True) < HOLD_BELOW
        self.army_goal[n] = None if held else (cell, attack)
        self._army_record(f"group {n}", members, where, kind, cell, attack, "held: the fight estimate" if held else None,
                          grid, glob)
        LOG.add("army_z", frame=hi.frame, group=n, grid=grid[0].astype(np.float16), glob=glob[0].astype(np.float16),
                cell=cell, attack=attack, held=held,
                logp=float(np.log(where[0, cell] + 1e-9) + np.log(kind[0, int(attack)] + 1e-9)))
        return self.army_goal[n]

    # --- the mutalisk raid ----------------------------------------------------------------------------

    def _mutas(self, obs, mine) -> bool:
        """v3's flock with a raid that gets through (v3's attack-moved seven at a mineral line and turned back
        once half were dead or the anti-air near outweighed them: 12 mutalisks made a game, 1.9 SCVs killed,
        pros 9). The mineral line with no turret seen by it and the fewest marines; a move (not an attack-move)
        to a point beyond the line, out from the map's middle, where the minerals screen it; then the SCVs
        picked off one at a time (a right-click on one, re-aimed every 2 s); back to that point the moment
        the flock bleeds (RAID_HIT_HP in RAID_HIT_S) or a turret or RAID_MARINES marines are within reach,
        again after RAID_REST_S if the threat left, else another line; home at 60% of the flock. (True: it
        acted.)"""
        hi = self.hi
        if self._aimed_at != self.target:
            self._aim()
        mutas = {u["tag"]: u for u in mine if u["type"] == Z.MUTALISK and u["completed"]}
        self.flock &= set(mutas)
        enemy = [u for u in obs["units"] if u["owner"] not in (hi.slot, 11)]
        for u in enemy:                                    # (anti-air, remembered: v3's air_seen, 90 s)
            if u["type"] in V3.AIR_THREAT:
                self.air_seen[u["tag"]] = (hi.frame, u["x"], u["y"], u["type"])
            elif u["type"] in self.enemy.town_halls:
                self.seen_halls[u["tag"]] = (u["x"], u["y"])
        self.air_seen = {t: v for t, v in self.air_seen.items() if hi.frame - v[0] < RAID_MEMORY}
        flock = [mutas[t] for t in self.flock]
        home = self.gather
        hatcheries = [(u["x"], u["y"]) for u in mine if u["type"] in Z.TOWN_HALLS]
        raid = [u for u in enemy if u["type"] in set(V3.AIR_THREAT) | {32, 34, 1, 2, 5, 30} and u["type"] not in (7, 124, 125)
                and any(math.dist((u["x"], u["y"]), h) < V3.DEFEND_NEAR for h in hatcheries)]
        if raid and flock and self.flock_state in ("home", "back") and self.all_in != "go":
            at = (sum(u["x"] for u in raid) / len(raid), sum(u["y"] for u in raid) / len(raid))
            if self.flock_state != "defend" or hi.frame - self.flock_since >= V3.MUTA_REAIM:
                return self._flock_order("attack", at, "defend")
        if self.flock_state == "defend" and not raid:
            self.flock_state = "back"
        if self.flock_state == "home":
            # (v3 formed the flock from mutalisks within 12 tiles of the gather point in front of the natural, 7 at
            # once: new mutalisks rally to the mineral lines, so in round g1 a flock formed in 13 of 24 games and
            # raided 20 times in all while 286 were made. Anywhere by a hatchery, RAID_FLOCK at once)
            loose = [u for t, u in mutas.items() if t not in self.flock
                     and any(math.dist((u["x"], u["y"]), h) < 16 * 32 for h in hatcheries)]
            if loose and (self.flock or len(loose) >= RAID_FLOCK):
                return self._form_flock(obs, mutas, loose)
            ready = len(flock) >= RAID_FLOCK and hi.frame - self.flock_since >= V3.MUTA_REST
            if (ready or (self.all_in == "go" and flock)) and self.lines:
                self.raid_line = self.lines[0] if self.all_in == "go" else self._raid_line(hi.frame)
                self.raid_cycles, self.raid_hp = 0, []
                self.flock_size = len(flock)
                return self._raid_go(self.raid_line)
            return False
        if not flock:
            self.flock_state, self.flock_since = "home", hi.frame
            return False
        mid = (sum(u["x"] for u in flock) / len(flock), sum(u["y"] for u in flock) / len(flock))
        hp = sum(u["hp"] for u in flock)
        self.raid_hp = [(f, h) for f, h in getattr(self, "raid_hp", []) if hi.frame - f <= RAID_HIT_S * 24] + [(hi.frame, hp)]
        bleeding = self.raid_hp[0][1] - hp >= RAID_HIT_HP
        turret_near = any(v[3] in (124, 125) and math.dist(v[1:3], mid) < RAID_TURRET for v in self.air_seen.values()
                          if hi.frame - v[0] < 5 * 24)
        marines_near = sum(1 for u in enemy if u["type"] in (0, 3, 58, 8) and math.dist((u["x"], u["y"]), mid) < RAID_SEE)
        lost = len(flock) < self.flock_size * RAID_KEEP
        if self.flock_state in ("out", "hit", "poke_back"):
            if lost and self.all_in != "go":
                self.log(f"mutalisks home: {len(flock)} of {self.flock_size} left")
                return self._flock_order("move", home, "back")
            threatened = bleeding or turret_near or marines_near >= RAID_MARINES
            if self.flock_state == "hit" and threatened and self.all_in != "go":
                self.raid_cycles += 1
                LOG.decision("Z", "raid", hi.frame, "back off", inputs={"mutas": len(flock), "marines": marines_near, "turret": turret_near,
                             "hp_lost": self.raid_hp[0][1] - hp, "cycles": self.raid_cycles},
                             by="rule: the flock bleeds, or a turret or marines within reach")
                return self._flock_order("move", self._raid_post(self.raid_line), "poke_back")
            if self.flock_state == "out" and self.all_in != "go":
                line_turret = any(v[3] in (124, 125) and math.dist(v[1:3], self.raid_line) < RAID_TURRET + 3 * 32
                                  for v in self.air_seen.values())
                if line_turret and hi.frame - self.raid_switched >= RAID_SWITCH:   # (the line itself is turreted)
                    other = self._raid_line(hi.frame, but=self.raid_line)
                    self.raid_switched = hi.frame
                    LOG.decision("Z", "raid", hi.frame, "another line" if other else "home", inputs={"mutas": len(flock)},
                                 by="rule: a turret seen by that line on the way in")
                    if other is None:
                        return self._flock_order("move", home, "back")
                    self.raid_line, self.raid_cycles = other, 0
                    return self._raid_go(self.raid_line)
                if bleeding or marines_near >= RAID_MARINES:          # (shot at in transit: back, then again)
                    self.raid_cycles += 1
                    LOG.decision("Z", "raid", hi.frame, "back off", inputs={"mutas": len(flock), "marines": marines_near,
                                 "turret": turret_near, "hp_lost": self.raid_hp[0][1] - hp, "cycles": self.raid_cycles},
                                 by="rule: shot at on the way in")
                    return self._flock_order("move", self._raid_post(self.raid_line), "poke_back")
            if self.flock_state == "out" and math.dist(mid, self._raid_post(self.raid_line)) < 5 * 32:
                self.flock_state, self.flock_since = "hit", hi.frame
            if self.flock_state == "poke_back" and hi.frame - self.flock_since >= RAID_REST_S * 24:
                if self.raid_cycles >= RAID_CYCLES:
                    other = self._raid_line(hi.frame, but=self.raid_line)
                    if other is None:
                        return self._flock_order("move", home, "back")
                    self.raid_line, self.raid_cycles = other, 0
                    return self._raid_go(self.raid_line)
                self.flock_state, self.flock_since = "hit", hi.frame
            if self.flock_state == "hit" and hi.frame - self.flock_since >= RAID_REAIM:
                scvs = [u for u in enemy if u["type"] == 7 and math.dist((u["x"], u["y"]), self.raid_line) < 10 * 32]
                if scvs:                                   # (one SCV at a time: a right-click on it, on screen)
                    scv = min(scvs, key=lambda u: (u["hp"], math.dist((u["x"], u["y"]), mid)))
                    if set(obs["selection"]) != self.flock & set(mutas):
                        hi.hotkey_recall(V3.MUTA_KEY)
                        return True
                    if self._look(obs, (scv["x"], scv["y"])):
                        return True
                    at = screen_of(obs, scv["x"], scv["y"])
                    if at is not None:
                        hi.right_click(*at)
                        self.flock_since = hi.frame
                        self.raided_at[tuple(self.raid_line)] = hi.frame
                        return True
                return self._flock_order("attack", self.raid_line, "hit")
            return False
        if self.flock_state == "defend":
            if hi.frame - self.flock_since >= V3.MUTA_REAIM and raid:
                at = (sum(u["x"] for u in raid) / len(raid), sum(u["y"] for u in raid) / len(raid))
                return self._flock_order("attack", at, "defend")
            return False
        if self.flock_state == "back" and math.dist(mid, home) < 6 * 32:
            self.flock_state, self.flock_since = "home", hi.frame
        return False

    def _raid_go(self, line) -> bool:
        """Out to a line's post along the map's edge: a waypoint RAID_WAYPOINT aside of the post (on Gary-Z's
        side), then the post, queued (a straight flight from home crossed the Terran base and its marines)."""
        hi = self.hi
        obs = hi.observe()
        mutas = {u["tag"] for u in obs["units"] if u["owner"] == hi.slot and u["type"] == Z.MUTALISK}
        if set(obs["selection"]) != self.flock & mutas or not obs["selection"]:
            hi.hotkey_recall(V3.MUTA_KEY)
            return True
        post = self._raid_post(line)
        w, h = self.map.size
        ox, oy = line[0] - w / 2, line[1] - h / 2           # out from the middle
        d = math.hypot(ox, oy) or 1.0
        px, py = -oy / d, ox / d                            # along the edge
        side = 1 if (px * (self.home[0] - post[0]) + py * (self.home[1] - post[1])) >= 0 else -1
        wp = (int(min(w - 64, max(64, post[0] + side * px * RAID_WAYPOINT))),
              int(min(h - 64, max(64, post[1] + side * py * RAID_WAYPOINT))))
        hi.minimap_right_click(*wp)
        hi.minimap_right_click(*post, queued=True)
        if self.flock_state != "out":
            self.log(f"mutalisks ({len(self.flock)}): out")
        self.flock_state, self.flock_since = "out", hi.frame
        return True

    def _raid_line(self, frame: int, but=None):
        """The mineral line to raid: no turret seen by it first, then the fewest marines seen near it, then the
        one raided longest ago; not `but`. Gary's main and natural (v3's lines) and any base whose Command
        Center Gary-Z has seen."""
        lines = list(self.lines)
        for v in self.seen_halls.values():
            if all(math.dist(v, l) > 8 * 32 for l in lines):
                lines.append(v)
        lines = [l for l in lines if but is None or math.dist(l, but) > 2 * 32]
        if not lines:
            return None
        recent = {t: v for t, v in self.air_seen.items() if frame - v[0] < RAID_MEMORY}

        def key(line):
            turrets = sum(1 for v in recent.values() if v[3] in (124, 125) and math.dist(v[1:3], line) < RAID_TURRET + 3 * 32)
            marines = sum(V3.AIR_THREAT[v[3]] for v in recent.values() if v[3] not in (124, 125) and math.dist(v[1:3], line) < 10 * 32)
            return (turrets, marines, self.raided_at.get(tuple(line), -10 ** 6))
        return min(lines, key=key)

    def _raid_post(self, line) -> tuple[int, int]:
        """Where the flock waits by a mineral line: RAID_POST beyond it, out from the map's middle (the minerals
        between it and the base), inside the map."""
        w, h = self.map.size
        dx, dy = line[0] - w / 2, line[1] - h / 2
        d = math.hypot(dx, dy) or 1.0
        x = min(w - 64, max(64, line[0] + dx / d * RAID_POST))
        y = min(h - 64, max(64, line[1] + dy / d * RAID_POST))
        return int(x), int(y)

    def _draw_cell(self, where) -> int:
        """A destination drawn from the model's odds over cells; with the floor (GARY_Z_FLOOR), among the 5
        likeliest only: drawn over every cell, two zerglings went across the map on 0.3% odds (seed 5001)."""
        p = np.asarray(where, dtype=np.float64)
        if self.commit_floor:
            keep = np.argsort(-p)[:5]
            pf = np.zeros_like(p)
            pf[keep] = p[keep]
            p = pf
        return self.rng.choices(range(len(p)), weights=(p / p.sum()).tolist())[0]

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
        self.cluster_ask[k] = hi.frame + (COMMIT_TICK if self.army_commit else ARMY_EVERY)
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
        self._remember_enemy(obs)
        if k == 0 and (self._intercept(obs, mine, members) or self._press(obs, mine, members)):
            return True
        if self.army_commit:
            return self._committed(obs, mine, members, k)
        grid, glob = self.army_tracker.inputs(obs, members)
        where, kind = self.army_model.predict(grid, glob)
        cell = self._draw_cell(where[0])
        attack = self.rng.random() < float(kind[0, 1])
        held = attack and self.army_model.fight(grid, glob, cell, True) < HOLD_BELOW
        LOG.add("army_z", frame=hi.frame, group=k, grid=grid[0].astype(np.float16), glob=glob[0].astype(np.float16),
                cell=cell, attack=attack, held=held,
                logp=float(np.log(where[0, cell] + 1e-9) + np.log(kind[0, int(attack)] + 1e-9)))
        mid = (sum(u["x"] for u in members) / len(members), sum(u["y"] for u in members) / len(members))
        stays = _cell_dist(cell, self.army_tracker.cell(*mid)) < 2
        self._army_record(f"cluster {k}", members, where, kind, cell, attack,
                          "held: the fight estimate" if held else "stays: the cell drawn is where it is" if stays else None,
                          grid, glob)
        if held or stays:
            return False                                   # (held, or the drawn cell is where they are)
        self.army_job = {"tags": [u["tag"] for u in members], "point": self.army_tracker.center(cell),
                         "kind": "attack" if attack else "move", "selecting": False, "tries": 0}
        return self._army_job_step(obs, mine)

    def _remember_enemy(self, obs: dict) -> None:
        """Terran fighters and buildings in sight, remembered (for the sweep's odds and the press's targets)."""
        from gary.policy.army import is_building, is_worker, supply_x2
        f = self.hi.frame
        for u in obs["units"]:
            if u["owner"] in (self.hi.slot, 11):
                continue
            if is_building(u["type"]):
                self.enemy_buildings[u["tag"]] = (u["x"], u["y"], u["type"])
            elif supply_x2(u["type"]) > 0 and not is_worker(u["type"]):
                self.enemy_fighters[u["tag"]] = (f, supply_x2(u["type"]), u["x"], u["y"])
        gone = {u["tag"] for u in obs["units"] if u["owner"] in (self.hi.slot, 11)}   # (own tags never collide)
        self.enemy_fighters = {t: v for t, v in self.enemy_fighters.items() if f - v[0] <= PRESS_MEMORY}

    def _press(self, obs: dict, mine: list[dict], members: list[dict]) -> bool:
        """Far ahead and unthreatened, the main body attacks the nearest Terran building it knows of: in round g6
        five of fifteen games ran to time with Gary-Z's army at 75 against 7, 78 against 11, 42 against 6, and it
        never went over. Own fighter supply at least PRESS_RATIO times the Terran fighters seen in PRESS_MEMORY
        (at least PRESS_MIN), no enemy fighter within 12 tiles of a hatchery; re-aimed every PRESS_AGAIN at the
        building nearest the body, so a cleared base leads to the next. (True: it acted.)"""
        hi = self.hi
        if hi.frame < self.press_after or not self.enemy_buildings:
            return False
        from gary.policy.army import is_building, is_worker, supply_x2
        own = sum(supply_x2(u["type"]) for u in mine if u["type"] in Z.ARMY and u["completed"])
        seen = sum(v[1] for v in self.enemy_fighters.values())
        if own < PRESS_MIN or own < PRESS_RATIO * max(seen, 8):
            return False
        halls = [(u["x"], u["y"]) for u in mine if u["type"] in Z.TOWN_HALLS]
        enemy = [u for u in obs["units"] if u["owner"] not in (hi.slot, 11) and supply_x2(u["type"]) > 0
                 and not is_worker(u["type"]) and not is_building(u["type"])]
        if any(math.dist((e["x"], e["y"]), h) < 12 * 32 for e in enemy for h in halls):
            return False
        mid = (sum(u["x"] for u in members) / len(members), sum(u["y"] for u in members) / len(members))
        alive = {u["tag"] for u in obs["units"]}
        self.enemy_buildings = {t: v for t, v in self.enemy_buildings.items()
                                if t in alive or not any(math.dist((u["x"], u["y"]), v[:2]) < 6 * 32 for u in members)}
        if not self.enemy_buildings:                 # (a remembered building the body stands by and doesn't see: gone)
            return False
        # (what ends a game first: Command Centers, then production, then the rest; the nearest within each. On
        # heartbreak seed 5002 the nearest-first press cleared the Terran's SCVs and army by 22:00 and left 18
        # buildings, turrets among them, standing at the 25-minute limit)
        rank = lambda t: 0 if t in self.enemy.town_halls else 1 if t in self.enemy.production else 2
        x, y, t = min(self.enemy_buildings.values(), key=lambda v: (rank(v[2]), math.dist(v[:2], mid)))
        self.press_after = hi.frame + PRESS_AGAIN
        LOG.decision("Z", "army", hi.frame, f"attack to ({x // 32},{y // 32})",
                     inputs={"units": len(members), "ours": own / 2, "theirs_seen": seen / 2, "target": self.enemy.names.get(t, str(t))},
                     by=f"rule: {PRESS_RATIO:g}x the enemy fighters seen, nothing at home: presses the nearest building", then="sent")
        self.army_job = {"tags": [u["tag"] for u in members], "point": (int(x), int(y)), "kind": "attack",
                         "selecting": False, "tries": 0}
        return self._army_job_step(obs, mine)

    def _intercept(self, obs: dict, mine: list[dict], members: list[dict]) -> bool:
        """The main body sweeps a Terran packet in sight that it outweighs (theirs under INTERCEPT_ODDS of its
        own supply) within INTERCEPT_FAR, away from Terran turrets and bunkers. In g5 seed 5007 the Terran fed
        units one by one across the map for ten minutes, 8-19 in sight at a time, while Gary-Z's 20-30 waited at
        home and fought each packet at its doorstep (357 fighters lost to 235). (True: it acted.)"""
        hi = self.hi
        if hi.frame < self.intercept_after:
            return False
        from gary.policy.army import is_building, is_worker, supply_x2
        enemy = [u for u in obs["units"] if u["owner"] not in (hi.slot, 11) and supply_x2(u["type"]) > 0
                 and not is_worker(u["type"]) and not is_building(u["type"])]
        if not enemy:
            return False
        mid = (sum(u["x"] for u in members) / len(members), sum(u["y"] for u in members) / len(members))
        near = [e for e in enemy if math.dist((e["x"], e["y"]), mid) < INTERCEPT_FAR]
        if not near:
            return False
        theirs = sum(supply_x2(e["type"]) for e in near)
        ours = sum(supply_x2(u["type"]) for u in members)
        if theirs > INTERCEPT_ODDS * ours:
            return False
        at = (sum(e["x"] for e in near) / len(near), sum(e["y"] for e in near) / len(near))
        static = [u for u in obs["units"] if u["owner"] not in (hi.slot, 11) and u["type"] in self.enemy.static]
        if any(math.dist((b["x"], b["y"]), at) < 9 * 32 for b in static):
            return False
        self.intercept_after = hi.frame + INTERCEPT_AGAIN
        LOG.decision("Z", "army", hi.frame, f"attack to ({int(at[0]) // 32},{int(at[1]) // 32})",
                     inputs={"units": len(members), "ours": ours / 2, "theirs": theirs / 2},
                     by="rule: the main body sweeps a packet it outweighs within reach", then="sent")
        self.army_job = {"tags": [u["tag"] for u in members], "point": (int(at[0]), int(at[1])), "kind": "attack",
                         "selecting": False, "tries": 0}
        return self._army_job_step(obs, mine)

    def _committed(self, obs: dict, mine: list[dict], members: list[dict], k: int) -> bool:
        """Commit mode: the cluster's next step drawn from the commit model's keep / engage / move given its
        commitment (its last order whose path passes within 2 cells, and its age); a new commitment's
        destination drawn among the cells 2+ away next to known enemy presence (engage) or not (move)."""
        from gary.policy.army import COMMIT_AGE_S, GRID, path_dist
        hi = self.hi
        mid = (sum(u["x"] for u in members) / len(members), sum(u["y"] for u in members) / len(members))
        here = self.army_tracker.cell(*mid)
        self.commitments = [c for c in self.commitments if hi.frame - c[0] <= COMMIT_AGE_S * 24]
        held, age = next(((c[3], (hi.frame - c[0]) / 24) for c in reversed(self.commitments)
                          if path_dist(here, c[1], c[2]) <= 2), (0, float(COMMIT_AGE_S)))
        grid, glob = self.army_tracker.inputs(obs, members, (held, age))
        where, kind = self.army_model.predict(grid, glob)
        q = kind[0].astype(np.float64)
        qd = q.copy()
        if self.commit_floor:                   # (no choice drawn under the floor; keep if none is above it)
            qd[qd < self.commit_floor] = 0.0
            if qd.sum() <= 0:
                qd[0] = 1.0
        qd = qd / qd.sum()
        choice = self.rng.choices(range(3), weights=qd.tolist())[0]
        names = ["keep", "engage", "move"]

        def name(c):
            x, y = self.army_tracker.center(c)
            return f"({x // 32},{y // 32})"
        seen = {"who": f"cluster {k}", "units": len(members), "at": name(here),
                "commitment": f"{names[held] if held else 'none'}, {age:.0f} s"}
        options = [[n, round(float(q[i]), 3)] for i, n in enumerate(names)]
        why = explain_army(self.army_model, grid, glob, lambda w, kd: kd[:, choice])
        LOG.add("commit_z", frame=hi.frame, group=k, grid=grid[0].astype(np.float16), glob=glob[0].astype(np.float16),
                choice=choice, logp=float(np.log(qd[choice] + 1e-9)))
        if choice == 0:
            LOG.decision("Z", "commit", hi.frame, "keep", q[0], options, seen, "draw", then="its commitment stands", why=why)
            return False
        m = np.asarray(grid[0]).reshape(-1, GRID, GRID)
        enemy = (m[3] > 0) | (m[6] > 0) | (m[7] > 0)
        near = enemy.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                near |= np.roll(np.roll(enemy, dy, 0), dx, 1)
        far = np.array([_cell_dist(c, here) >= 2 for c in range(GRID * GRID)])
        mask = far & (near.reshape(-1) if choice == 1 else ~near.reshape(-1))
        p = where[0] * mask
        if p.sum() <= 0:
            LOG.decision("Z", "commit", hi.frame, names[choice], q[choice], options, seen, "draw",
                         then="no cell on that side: its commitment stands", why=why)
            return False
        if self.commit_floor:                   # (among the 5 likeliest cells on that side)
            keep_cells = np.argsort(-p)[:5]
            pf = np.zeros_like(p)
            pf[keep_cells] = p[keep_cells]
            p = pf
        cell = self.rng.choices(range(len(p)), weights=(p / p.sum()).tolist())[0]
        self.commitments.append((hi.frame, here, cell, choice))
        LOG.decision("Z", "commit", hi.frame, f"{names[choice]} to {name(cell)}", q[choice], options, seen, "draw",
                     then="sent", why=why)
        self.army_job = {"tags": [u["tag"] for u in members], "point": self.army_tracker.center(cell),
                         "kind": "attack" if choice == 1 else "move", "selecting": False, "tries": 0}
        return self._army_job_step(obs, mine)

    def _army_record(self, who: str, members: list[dict], where, kind, cell: int, attack: bool, then, grid, glob) -> None:
        """An army decision in the game's records: the cells likeliest, the one drawn, move or attack."""
        def name(c):
            x, y = self.army_tracker.center(c)
            return f"({x // 32},{y // 32})"
        mid = (sum(u["x"] for u in members) / len(members), sum(u["y"] for u in members) / len(members))
        LOG.decision("Z", "army", self.hi.frame, f"{'attack' if attack else 'move'} to {name(cell)}",
                     float(where[0, cell]) * float(kind[0, int(attack)]), top(name, where[0], 3),
                     {"who": who, "units": len(members), "at": f"({int(mid[0]) // 32},{int(mid[1]) // 32})",
                      "p_attack": float(kind[0, 1])}, "draw", then=then or "sent",
                     why=explain_army(self.army_model, grid, glob, lambda w, kd: w[:, cell] * kd[:, int(attack)]))

    # (a minimap click's scatter can put a drone's move on the map's edge row, where it never arrives: walking
    # 4 minutes, swung between 35 and 14 tiles off, an expansion's drone kept the next ones from being decided
    # with 1,700 minerals banked: eval/why.py, c7 seed 3005 on Heartbreak Ridge)
    walk_timeout = 45 * 24

    def _walk_failed(self, job: Job) -> None:
        """Walked past walk_timeout: picked and sent again; an expansion whose drone never arrived is
        dropped and its base left for the game (on Heartbreak Ridge the bases at the map's top and bottom edge,
        decided 7 times in c7's 8 games there, were never reached)."""
        if job.step.startswith("expand") and job.tries >= 1:
            self.unreached.add(self.map.base_near(self._spot(job)).center)
            self._fate(job, "dropped", f"its drones never arrived ({job.tries} walks): that base left for this game")
            self.jobs.pop(job.step, None)
            return
        self._fate(job, "retried", f"still walking after {self.walk_timeout // 24} s: picked and sent again")

    def _up(self, job: Job, mine) -> bool:
        """v3's check, for a job that placed its building: one never placed isn't up because another of
        its kind stands within 3 tiles of its spot (three of four creep colonies decided in 20 s ended so,
        unbuilt: eval/why.py, seed 3005)."""
        return job.tries > 0 and super()._up(job, mine)

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
        # (not the jobs still picking a drone: keeping their money starved the larvae and lost games either way:
        # every picking job's for as long as it took, round g2_keep, Gary-T won 20 of 24 against 13; the oldest
        # hatchery job's for 90 s at most, round g4_keep90, 18 against 14, Gary-Z's gap 0.36 against 0.31)
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
        # larvae banked while the Spire morphs (as the gas is: pros flock 7-9 mutalisks the moment it stands; Gary-Z
        # had 0-3 larvae free then, no mutalisk in the next minute and 5 in two, round g8): no larva morph but an
        # overlord until the larvae are at their cap, where one more would be lost anyway
        n_larvae = sum(1 for u in mine if u["type"] == Z.LARVA)
        saving = (FLOCK_SAVE and any(u["type"] == Z.SPIRE and not u["completed"] for u in mine)
                  and sum(1 for u in mine if u["type"] == Z.MUTALISK) < MUTA_FLOCK_SAVE
                  and n_larvae < 3 * sum(1 for u in mine if u["type"] in Z.TOWN_HALLS and u["completed"]))
        waiting = False
        # the alarm: before ALARM_UNTIL, the Terran fighters seen in the last minute outweigh the army by ALARM_MARGIN.
        # The model kept droning at 60-70% odds with 20 marines in sight and 6 army at 6:00-7:30 (round g10, Python,
        # Gary-T won 7 of 8 by a push at 7:18-9:41 into 0-6 army): pros in that state make zerglings and sunkens,
        # and the pros' data holds few such states because they don't get into them. While it is on: no drones,
        # tech and expansions only at real odds, a sunken at the natural per 8 supply of threat (ALARM_SUNKENS at
        # most), and the larvae to the likeliest army unit the model gives
        from gary.policy.army import supply_x2 as _sx2
        self._remember_enemy(obs)
        # (fighters away from the Terran's known Command Centers only: the overlord outside its natural sees its home
        # army, which outweighs a Zerg's most of the game, and the alarm starved Gary-Z of drones for minutes: round
        # g11, Gary-T won 13 of 24 against 8, 5 on Fighting Spirit against 1. Pros react to an army moving out)
        ccs = [(x, y) for x, y, t in self.enemy_buildings.values() if t in self.enemy.town_halls]
        threat = sum(v[1] for v in self.enemy_fighters.values()
                     if not any(math.dist(v[2:4], c) < ALARM_HOME for c in ccs))
        own_army = sum(_sx2(u["type"]) for u in mine if u["type"] in Z.ARMY and u["completed"])
        alarm = hi.frame < ALARM_UNTIL and threat >= own_army + ALARM_MARGIN
        if alarm != self.alarm_on:
            self.alarm_on = alarm
            self.log(f"alarm {'on' if alarm else 'off'}: {threat / 2:.0f} enemy supply seen, {own_army / 2:.0f} army")
            LOG.decision("Z", "macro", hi.frame, "alarm on" if alarm else "alarm off", inputs={"threat": threat / 2, "army": own_army / 2},
                         by=f"rule: the enemy fighters seen in a minute outweigh the army by {ALARM_MARGIN // 2}")
        # (the record: the model's top odds, what it saw, and each option drawn ahead of the choice and why
        # it was set aside; gary/decision_log.py)
        name = lambda k: self._name(*self.model.spec.decisions[k])
        over: list[list] = []

        def record(choice, p, then=None, k=None, by="dice"):
            LOG.decision("Z", "macro", hi.frame, choice, p, top(name, probs[0]), self._seen(obs, mine, me), by,
                         over[:8], then, why=explain_macro(self.model, x, k) if k is not None else None)
        if alarm:
            want = min(ALARM_SUNKENS, threat // 16 + 1)
            sunkens = self._count(mine, Z.SUNKEN)
            creeps_done = [u for u in mine if u["type"] == Z.CREEP and u["completed"]]
            creep_pending = any(j.unit_type == Z.CREEP for j in self.jobs.values()) or any(
                u["type"] == Z.CREEP and not u["completed"] for u in mine)
            steps = []
            if creeps_done and sunkens < want:
                steps.append((BMORPH, Z.SUNKEN))
            if not creep_pending and sunkens + len(creeps_done) < want:
                steps.append((BUILD, Z.CREEP))
            army_ks = sorted((k for k, (a, u) in enumerate(self.model.spec.decisions)
                              if a == MORPH and u in (Z.ZERGLING, Z.HYDRALISK, Z.MUTALISK)), key=lambda k: -probs[0, k])
            steps += [self.model.spec.decisions[k] for k in army_ks]
            for act, uid in steps:
                if self._cant(act, uid, obs, mine, me):
                    continue
                k = self.model.spec.decisions.index((act, uid))
                g = {"act": act, "id": uid, "p": float(probs[0, k]), "since": hi.frame, "name": self._name(act, uid)}
                if self._affordable(g, me):
                    self.goal = g
                    self._summed(probs, k, obs, mine, me)
                    self.log(f"next: {g['name']} (alarm)")
                    record(g["name"], g["p"], k=k, by=f"rule: alarm ({threat / 2:.0f} enemy supply seen, {own_army / 2:.0f} army)")
                    return
                if act in (BUILD, BMORPH) and self.parked is None:
                    self.parked = g
                    record(g["name"], g["p"], "parked for its money", k=k, by="rule: alarm")
                    over.append([g["name"], "parked for its money"])
                    break
                record(None, None, f"waits: {g['name']} needs money (alarm)")
                return
        if self.mass:
            # a one-off whose summed odds reached one, the largest first: taken if possible and affordable, else
            # parked for its money (the larvae go on meanwhile, as below)
            due = ((self.odds[j], j) for j in self.one_off
                   if self.odds.get(j, 0.0) >= (HATCH_MASS if self.model.spec.decisions[j][1] == Z.HATCHERY else 1.0))
            for mass, j in sorted(due, reverse=True):
                act, uid = self.model.spec.decisions[j]
                if self._cant(act, uid, obs, mine, me):
                    continue
                g = {"act": act, "id": uid, "p": float(probs[0, j]), "since": hi.frame, "name": self._name(act, uid)}
                by = f"rule: its odds summed to {mass:.2f} since it was possible"
                if self._affordable(g, me):
                    self.goal = g
                    self._summed(probs, j, obs, mine, me)
                    self.log(f"next: {g['name']} ({mass:.2f} odds summed since it was possible)")
                    record(g["name"], g["p"], k=j, by=by)
                    return
                if self.parked is None:
                    self.parked = g
                    self._summed(probs, j, obs, mine, me)
                    self.log(f"next: {g['name']} ({mass:.2f} odds summed), parked for its money")
                    record(g["name"], g["p"], "parked for its money", k=j, by=by)
                    over.append([g["name"], "parked for its money"])
                    break
        for k in order:
            act, uid = self.model.spec.decisions[k]
            if not larva and act == MORPH and uid in Z.FROM_LARVA and Z.can_make(uid, self._done_types(mine), self.researched) \
                    and (uid == Z.OVERLORD or me["supply_used"] + Z.SUPPLY.get(uid, 1) <= me["supply_max"]):
                waiting = True  # (no larva free yet: that one waits for the next, as the Terran's macro waits for
                over.append([name(k), "no larva free: waits for one"])     # a busy building)
                continue
            if saving and not (act == MORPH and uid == Z.OVERLORD) and uid not in (Z.CREEP, Z.SUNKEN, Z.SPORE):
                if act == MORPH and uid in Z.FROM_LARVA:
                    over.append([name(k), "rule: larvae saved for the Spire's flock"])
                    continue
                if probs[0, k] < SAVE_MIN_P:          # (the minerals too: with the larvae held, the dice drained 600
                    over.append([name(k), f"rule: under {SAVE_MIN_P:.0%} while saving for the flock"])   # into six creep
                    continue                          # colonies and three sunkens at 8%, and the mutalisks waited for money)
            if alarm and act == MORPH and uid == Z.DRONE:
                over.append([name(k), "rule: alarm: no drones"])
                continue
            if alarm and (act == EXPAND or act in (RESEARCH, UPGRADE) or (act == BUILD and uid in TECH_PATH | {Z.HATCHERY, Z.EVO}))                     and probs[0, k] < ALARM_MIN_P:
                over.append([name(k), f"rule: alarm: under {ALARM_MIN_P:.0%}"])
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
            why = "not a decision" if act < 0 else self._cant(act, uid, obs, mine, me)
            if why:
                over.append([name(k), f"can't now: {why}"])
                continue
            g = {"act": act, "id": uid, "p": float(probs[0, k]), "since": hi.frame, "name": self._name(act, uid)}
            # (this decision's own: the larvae's flag above stays. Overwritten here once, a parked Lurker Aspect
            # left Gary-Z with no larva for the rest of the roll: drones set aside as waiting with 5 larvae
            # standing, the under-MIN_P rule on, creep colonies at 5% taken instead: eval/why.py, seed 3005)
            from_larva = act == MORPH and uid in Z.FROM_LARVA
            if self._affordable(g, me):
                self.goal = g
                self._summed(probs, k, obs, mine, me)
                self.log(f"next: {g['name']} ({g['p']:.0%})")
                record(g["name"], g["p"], k=k)
                if from_larva:                       # (pros spend larvae in bursts, a hatchery's at once: the timing
                    self.next_query = hi.frame + QUERY_EVERY    # head's gap is between bursts; paced by it,
                return                                          # the larvae piled up, 2,000 minerals at 10:00)
            if not from_larva and self.parked is None:
                self.parked = g
                self._summed(probs, k, obs, mine, me)
                self.log(f"next: {g['name']} ({g['p']:.0%}), parked for its money")
                record(g["name"], g["p"], "parked for its money", k=k)
                over.append([g["name"], "parked for its money"])
                continue                                     # (and a larva meanwhile, if one is affordable)
            if self.parked is not None and not from_larva:
                over.append([g["name"], "one already parked: meanwhile only larvae"])
                continue                                     # (one parked at a time; meanwhile only larvae)
            record(None, None, f"waits: {g['name']} needs money")   # (a larva the money doesn't cover yet: wait)
            return
        record(None, None, "nothing it can do now")

    def _summed(self, probs, k: int, obs: dict, mine: list[dict], me: dict) -> None:
        """A decision taken (k): every other one-off possible now adds its odds to its sum; k's own starts over."""
        if not self.mass:
            return
        for j in self.one_off:
            if j == k:
                self.odds[j] = 0.0
            elif probs[0, j] >= 0.01 and self._cant(*self.model.spec.decisions[j], obs, mine, me) is None:
                self.odds[j] = self.odds.get(j, 0.0) + float(probs[0, j])

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

    def _cant(self, act: int, uid: int, obs: dict, mine: list[dict], me: dict) -> str | None:
        """Why Gary-Z can't carry this out now, money aside (None: it can): what it needs is missing,
        there's nothing to make it from, or the pros' cap for this time is reached. In words, for its
        records."""
        done = self._done_types(mine)
        cap = self.model.cap(uid, obs["frame"]) if act in (MORPH, BUILD, BMORPH, EXPAND) else None
        if cap is not None and self._count(mine, uid) >= cap:
            return f"the pros' cap for this minute ({cap})"
        if uid == Z.CREEP:                              # (and all colonies within the pros' caps of all three)
            caps = [self.model.cap(t, obs["frame"]) for t in (Z.CREEP, Z.SUNKEN, Z.SPORE)]
            if all(c is not None for c in caps) and self._count(mine, Z.CREEP, every=True) >= sum(caps):
                return f"the pros' cap of colonies for this minute ({sum(caps)})"
        need = None if Z.can_make(uid, done, self.researched) else "what it needs isn't there"
        if act == MORPH:
            if uid in Z.FROM_LARVA:
                if not any(u["type"] == Z.LARVA for u in mine):
                    return "no larva"
                if uid != Z.OVERLORD and me["supply_used"] + Z.SUPPLY.get(uid, 1) > me["supply_max"]:
                    return "supply"
                return need
            parent = Z.UNIT_MORPH.get(uid)
            if parent is None or not any(u["type"] == parent and u["completed"] for u in mine):
                return "nothing to morph it from"
            return need
        if act == BMORPH:
            parent = Z.BUILDING_MORPH.get(uid)
            if parent is None or not any(u["type"] == parent and u["completed"] and u.get("queue", 0) == 0 for u in mine):
                return "no idle building to morph"
            return need
        if act == BUILD:
            if uid not in Z.DRONE_BUILDS or need:
                return need or "not a drone's building"
            if any(j.unit_type == uid for j in self.jobs.values()):
                return "one already on its way"         # (one of a kind on its way at a time)
            if uid == Z.EXTRACTOR:
                return None if self._free_geyser(mine) is not None else "no free geyser"
            return None if any(u["type"] == Z.DRONE for u in mine) else "no drone"
        if act == EXPAND:
            if any(j.step.startswith("expand") for j in self.jobs.values()):
                return "an expansion already on its way"
            return None if self._next_base(obs, mine) is not None else "no free base"
        if act == RESEARCH:
            where, _ = Z.RESEARCH.get(uid, (None, None))
            if where is None or uid in self.researched:
                return "done already"
            if not all(Z.has(done, r) for r in Z.RESEARCH_NEEDS.get(uid, [])):
                return "what it needs isn't there"
            return None if self._idle_building(mine, where) is not None else "no idle building for it"
        if act == UPGRADE:
            if uid not in Z.UPGRADE:
                return "unknown upgrade"
            where, _, levels = Z.UPGRADE[uid]
            level = self.upgraded.get(uid, 0) + 1
            need = Z.UPGRADE_LEVEL_NEEDS.get(level) if levels > 1 else None
            if level > levels:
                return "done already"
            if not all(Z.has(done, r) for r in Z.UPGRADE_NEEDS.get(uid, [])) or (need is not None and not Z.has(done, need)):
                return "what it needs isn't there"
            return None if self._idle_building(mine, where) is not None else "no idle building for it"
        return "not a decision Gary-Z carries out"

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
                     and u.get("order") not in (O.ORDER_RESEARCH, O.ORDER_UPGRADE)), None)

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
        # one with nothing of Gary-Z's on top of it, the nearest home first, another after each miss: an
        # overlord hovering over the hydralisks took every click, 68 in 30 s, and the Lurker decision held
        # the macro for 30 s each time with 6 larvae and 3,800 minerals idle (eval/why.py, c11 seed 3001)
        tags = {u["tag"] for u in units}
        own = [o for o in obs["units"] if o["owner"] == self.hi.slot and o["tag"] not in tags]
        clear = [u for u in units if not any(abs(o["x"] - u["x"]) < CLEAR_PX and abs(o["y"] - u["y"]) < CLEAR_PX for o in own)]
        pool = sorted(clear or units, key=lambda u: math.dist((u["x"], u["y"]), self.home))
        u = pool[self.select_misses % len(pool)]
        if self._look(obs, (u["x"], u["y"])):
            return False
        p = screen_of(obs, u["x"], u["y"])
        if p is not None:
            self.hi.click(*p)
            self.select_misses += 1
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
        free = [b for b in self.map.bases if b not in owned and b.minerals and b.center not in self.unreached
                and not any(math.dist(b.center, e) < 10 * 32 for e in enemy)
                and not any(j.unit_type == Z.HATCHERY and math.dist(self._spot(j), b.center) < 6 * 32 for j in self.jobs.values())]
        if self.nat_base not in owned and self.nat_base in free:
            return self.nat_base
        nat = self.nat_base.center
        return min(free, key=lambda b: math.dist(b.center, nat), default=None)
