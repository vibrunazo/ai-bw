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
import random
from pathlib import Path

from gary import commands as C
from gary import terran as T
from gary import zerg as Z
from gary.bots import zerg_v3 as V3
from gary.bots.zerg_v3 import Job, ZergV3
from gary.interface import screen_of
from gary.policy.macro import MacroModel, MacroTracker

MORPH, BMORPH, BUILD, RESEARCH, UPGRADE, EXPAND = 1, 2, 3, 4, 5, 6
MIN_P = 0.05                    # never a decision the model gives less than this (as the Terran's macro)
DICE_FLOOR = 0.01               # the dice decoder: decisions under this odds aren't rolled
USED_LARVA = 2 * 24             # a larva just morphed shows as a larva a moment longer: not used again
IN_FLIGHT = 24                  # money just spent shows in the bank a moment later: counted as spent
GOAL_TIMEOUT = 30 * 24          # a decision not carried out in 30 s is dropped
QUERY_EVERY = 12
GAS_FLOAT, GAS_LOW = 400, 150   # gas above this (and above the minerals): its drones go back to minerals;
                                # below the other, the extractors are filled again
RALLY_IN_LINE = 0.6              # the rally: this far from the hatchery toward its minerals' middle
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


def latest_model() -> Path:
    found = sorted(Path("runs/macro").glob("TvZ_Z_*/model.pt"))
    if not found:
        raise FileNotFoundError("no Zerg macro model in runs/macro: python -m train.macro --race Z")
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
        # no scripted poke or pressure (where the army goes is the army model's, later)
        self.plan, self.poke, self.pressing = "mutas", False, False
        self.tracker = MacroTracker(self.model.spec, hi.slot, plan)
        self.goal: dict | None = None
        self.parked: dict | None = None        # a decision waiting for its money, kept for it meanwhile
        self.turn, self.streak = "macro", 0    # who goes first, and how many actions in a row it had
        self.gas_floating = False              # the gas piles up: its drones mine minerals meanwhile
        self.army_turn_last = False            # the army took the last turn: the rest has this one
        self.next_query = 0
        self.rng = random.Random(seed * 7919 + 1)          # the dice (deterministic per seed)
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
        if me["gas"] > GAS_FLOAT and me["gas"] > me["minerals"]:
            self.gas_floating = True
        elif me["gas"] < GAS_LOW:
            self.gas_floating = False
        # the army and the rest take turns too: in fights v3's army took every turn, and nothing was
        # decided or morphed from 7:30 to 10:30 (1,593 minerals and 9 larvae unused at 10:00)
        if not self.army_turn_last and self.army(obs, mine):
            self.army_turn_last = True
            return
        self.army_turn_last = False
        hatches = [u for u in mine if u["type"] in Z.TOWN_HALLS]
        pool = next((u for u in mine if u["type"] == Z.POOL), None)
        drones = [u for u in mine if u["type"] == Z.DRONE and u["completed"]]
        if self.task:
            return self._continue(obs, mine, me)
        # the macro and v3's helpers share the hands: whoever acted keeps them (a selection, then its
        # order: alternating every action broke both, the other's selection in between), STREAK actions
        # at most, then the other goes first; one with nothing to do hands over at once. (Either could
        # take every turn: three jobs at once from 1:00 to 2:00; idle drones it couldn't send back,
        # 109 turns in 40 s from 10:20; and nothing was decided meanwhile)
        sides = [("macro", self._macro), ("helpers", self._helpers)]
        if self.turn == "helpers":
            sides.reverse()
        for name, side in sides:
            if side(obs, mine, me, hatches, pool, drones):
                self.streak = self.streak + 1 if name == self.turn else 1
                self.turn = name
                if self.streak >= STREAK:
                    self.turn, self.streak = ("helpers" if name == "macro" else "macro"), 0
                return

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
            return bool(hi.pending)
        self._decide(obs, mine, me)
        if self.goal is not None:
            self._carry_out(obs, mine, me)
        return bool(hi.pending)

    def _gas_done(self) -> bool:
        return self.gas_floating                   # (v2's idle routine sends gas drones to minerals then)

    def _gas(self, obs, mine, me, drones) -> bool:
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
        for h in hatches:
            if not h["completed"] or h["tag"] in self.rallied_at:
                continue
            base = self.map.base_near((h["x"], h["y"]))
            if not base.minerals:
                self.rallied_at.add(h["tag"])
                continue
            mx = sum(x for x, _ in base.minerals) / len(base.minerals)
            my = sum(y for _, y in base.minerals) / len(base.minerals)
            spot = (int(h["x"] + (mx - h["x"]) * RALLY_IN_LINE), int(h["y"] + (my - h["y"]) * RALLY_IN_LINE))
            if self._look(obs, (h["x"], h["y"])):
                return True
            if obs["selection"] == [h["tag"]]:
                hi.minimap_right_click(*spot)
                self.rallied_at.add(h["tag"])
                return True
            top = screen_of(obs, h["x"], h["y"] - 24)
            if top:
                hi.click(*top)
            return True
        return False

    def army(self, obs: dict, mine: list[dict]) -> bool:
        if self._mutas(obs, mine):
            return True
        if any(u["type"] == Z.LURKER for u in mine) and self._lurkers(obs, mine):
            return True
        return super(ZergV3, self).army(obs, mine)

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
        return minerals, gas

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
                setattr(self, which, None)
        if self.goal is None and self.parked is not None and self._affordable(self.parked, me):
            self.goal, self.parked = self.parked, None          # (its money is there: it goes first)
            return
        if self.goal is not None or hi.frame < self.next_query:
            return
        self.next_query = hi.frame + QUERY_EVERY
        x = self.tracker.features(obs, self._in_production(mine))
        probs, _ = self.model.probs(x)
        # the dice, as the Terran's macro: the model's odds rolled (a weighted shuffle), then the first
        # decision Gary-Z can carry out; the favorite alone is a drone nearly every time, and the Lair,
        # zergling speed or an army never came (the first game: 2 zerglings and no Lair by 8:00)
        order = [k for _, k in sorted(((self.rng.random() ** (1 / p), k) for k, p in enumerate(probs[0])
                                        if p >= DICE_FLOOR), reverse=True)]
        for k in order:
            act, uid = self.model.spec.decisions[k]
            if act < 0 or not self._can(act, uid, obs, mine, me):
                continue
            g = {"act": act, "id": uid, "p": float(probs[0, k]), "since": hi.frame, "name": self._name(act, uid)}
            larva = act == MORPH and uid in Z.FROM_LARVA
            if self._affordable(g, me):
                self.goal = g
                self.log(f"next: {g['name']} ({g['p']:.0%})")
                return
            if not larva and self.parked is None:
                self.parked = g
                self.log(f"next: {g['name']} ({g['p']:.0%}), parked for its money")
                continue                                     # (and a larva meanwhile, if one is affordable)
            if self.parked is not None and not larva:
                continue                                     # (one parked at a time; meanwhile only larvae)
            return                                           # (a larva the money doesn't cover yet: wait)

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
        if cap is not None and sum(1 for u in mine if u["type"] == uid) >= cap:
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
            self.goal = None
            return
        self.jobs[step] = Job(step, unit, tuple(tile), since=self.hi.frame)
        self.log(f"builds {self._name(act, uid)} at {tuple(tile)}")
        self._done(act, uid)

    def _building_spot(self, uid: int, mine: list[dict]):
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
