"""Gary v0.2, Terran: what to make, and in what order, comes from the learned macro model.

The model (gary/policy/macro.py) was trained on ~14,000 pro TvZ games. Whenever Gary is free to
decide, it looks at Gary's state the way a player knows it (own units, resources, what has been
scouted, what Gary has made so far) and ranks every possible next production decision. Gary takes
the most likely one it can actually do now (requirements built, a building free to make it, a base
free to expand to), waits for the money, and carries it out with human hands: select the
building (hotkey or click), press the hotkey, or walk a worker over and place the building.

Still scripted, from v0.1: mining, sending three workers to each refinery, hotkeys and rallies,
and the army (attack-move with every 16 army units). Not yet: defense, scouting, micro.

    python -m gary.bots.terran_v02 --map path/to/map.scx --minutes 10
    python -m gary.bots.terran_v02 --live              # in a Remastered game, via the bridge
    python -m gary.bots.terran_v02 --map ... --style 1 # steer toward a build style (taxonomy cluster)
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np

from gary import terran as T
from gary.bots import terran_v01 as v01
from gary.bots.terran_v01 import IDLE, Task, TerranGary
from gary.env import LIVE_COMMAND_DELAY
from gary.interface import HumanInterface, screen_of
from gary.mapinfo import Base, MapInfo
from gary.version import announcement
from gary.policy.macro import ACT_NAMES, MacroModel, MacroTracker

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN, BUILD, RESEARCH, UPGRADE, EXPAND = 0, 3, 4, 5, 6
TOWN_HALLS = {106, 131, 132, 133, 154}
MAX_SCVS = 60
GOAL_TIMEOUT = 24 * 30          # a decision Gary can't afford or do within 30 s is dropped
MIN_P = 0.05                    # never take a decision the model gives less than this: unlikely
                                # choices lead into states pros rarely reach, where it knows less
BUSY = {T.ORDER_RESEARCH, T.ORDER_UPGRADE, T.ORDER_BUILD_ADDON}
WAIT = {"kind": "wait"}         # _plan: possible, but not right now
RESET_HARVEST = 151             # the order of a worker stepping out of a refinery (or off a mineral)


def latest_model(matchup: str = "TvZ", race: str = "T") -> Path:
    found = sorted((REPO_ROOT / "runs" / "macro").glob(f"{matchup}_{race}_*/model.pt"))
    if not found:
        raise SystemExit(f"no trained macro model in runs/macro (python -m train.macro --race {race})")
    return found[-1]


def describe(act: int, uid: int) -> str:
    if act == TRAIN:
        return T.NAMES.get(uid, f"unit {uid}")
    if act == RESEARCH:
        return T.TECH_NAMES.get(uid, f"tech {uid}")
    if act == UPGRADE:
        return T.UPGRADE_NAMES.get(uid, f"upgrade {uid}")
    if act == EXPAND:
        return "expand"
    return T.NAMES.get(uid, f"{ACT_NAMES[act]} {uid}")


class TerranGaryV2(TerranGary):
    army_types = T.ARMY
    version = "v0.2"
    production_types = T.PRODUCTION

    def __init__(self, hi: HumanInterface, mapinfo: MapInfo, model: MacroModel,
                 style: int | None = None, verbose: bool = True):
        super().__init__(hi, mapinfo)
        self.model = model
        self.announce = announcement(self.version, style, {"macro": getattr(model, "path", "?")})
        self.tracker = MacroTracker(model.spec, self.slot, style)
        self.verbose = verbose
        self.goal: dict | None = None
        self.next_query = 0
        self.producing: dict[int, int] = {}       # building tag -> unit type it was last told to make
        self.researched: set[int] = set()
        self.upgraded: Counter = Counter()
        self.gas_workers: dict[int, set[int]] = {}  # refinery tag -> workers Gary sent there
        self.decisions: list[tuple[int, str, float]] = []   # (frame, decision, model probability)
        self.failed_until: dict[tuple[int, int], int] = {}  # decision -> frame it may be tried again

    # --- main loop ------------------------------------------------------------------------

    def act(self) -> None:
        hi = self.hi
        if hi.pending:
            return
        obs = hi.observe()
        if not self.setup_done:
            return self._setup(obs)
        me = obs["me"]
        mine = [u for u in obs["units"] if u["owner"] == self.slot]
        done = [u for u in mine if u["completed"]]
        self._release_reservations(mine)
        if self.task:
            return self._run_task(obs, mine)
        if self._resume_construction(mine):
            return
        idle = [u for u in done if u["type"] == T.SCV and u["order"] == IDLE and screen_of(obs, u["x"], u["y"])
                and self.ignore_idle_until.get(u["tag"], 0) <= hi.frame]
        if idle:
            self.task = Task("mine", data={"tags": [u["tag"] for u in idle]}, started=hi.frame)
            return
        if self._idle_sweep(obs, done):
            return
        for b in done:
            if b["type"] in self.production_types and b["tag"] not in self.hotkeyed:
                self.task = Task("hotkey_building", b["type"], data={"tag": b["tag"], "x": b["x"], "y": b["y"]},
                                 started=hi.frame)
                return
        for r in done:
            if r["type"] == T.REFINERY and self._on_gas(obs, r["tag"]) < 3 and                     self.ignore_idle_until.get(r["tag"], 0) <= hi.frame:
                self.task = Task("gas", data={"tag": r["tag"], "x": r["x"], "y": r["y"]}, started=hi.frame)
                return
        if self._army(obs, mine, done):
            return
        self._macro(obs, mine, done, me)

    def _army(self, obs: dict, mine: list[dict], done: list[dict]) -> bool:
        """Army control (v0.1's): attack-move with every 16 army units. True if Gary acted."""
        army = [u for u in done if u["type"] in self.army_types and u["tag"] not in self.army_sent]
        if len(army) >= v01.WAVE:
            self.task = Task("attack", started=self.hi.frame)
            return True
        return False

    def _run_task(self, obs: dict, mine: list[dict]) -> None:
        t = self.task
        if t.kind in ("command", "gas"):
            if self.hi.frame - t.started > 24 * 20:
                self.task = None
                return
            return self._task_command(obs, mine, t) if t.kind == "command" else self._task_gas(obs, mine, t)
        return super()._run_task(obs, mine)

    # --- deciding --------------------------------------------------------------------------

    def _macro(self, obs: dict, mine: list[dict], done: list[dict], me: dict) -> None:
        hi = self.hi
        if self.goal and hi.frame - self.goal["since"] > GOAL_TIMEOUT:
            self._say(f"gives up on {self.goal['name']}")
            self.goal = None
        if self.goal is None:
            if hi.frame < self.next_query:
                return
            self.next_query = hi.frame + 12
            # the most likely decision Gary can carry out now; if none can, wait for the most
            # likely one that only needs a busy building to free up
            probs, _ = self.model.probs(self.tracker.features(obs, self._in_production(mine)))
            pick = waiting = None
            for k in np.argsort(-probs[0]):
                if probs[0, k] < MIN_P:
                    break
                plan = self._plan(*self.model.spec.decisions[k], obs, mine, done, me)
                if plan is WAIT:
                    waiting = waiting if waiting is not None else k
                elif plan is not None:
                    # something doable now, unless the likelier decision is worth waiting for
                    # (a pro saves for the expansion rather than spend on whatever is possible)
                    if waiting is None or probs[0, k] >= 0.5 * probs[0, waiting] or me["minerals"] > 400:
                        pick = k
                    break
            k = pick if pick is not None else waiting
            if k is None:
                return
            act, uid = self.model.spec.decisions[k]
            self.goal = {"act": act, "id": uid, "p": float(probs[0, k]), "since": hi.frame,
                         "name": describe(act, uid)}
            self._say(f"next: {self.goal['name']} ({self.goal['p']:.0%})")
        g = self.goal
        plan = self._plan(g["act"], g["id"], obs, mine, done, me)   # still possible?
        if plan is None:
            self.goal = None
            return
        if plan is WAIT:                    # e.g. the building is still busy: wait for it
            return
        minerals, gas = plan["cost"]
        budget = me["minerals"] - sum(v01.COST[ut] for ut, *_ in self.reserved)
        walk_ahead = 40 if plan["kind"] == "build" else 0      # send the worker a little early
        if budget < minerals - walk_ahead or me["gas"] < gas:
            return
        if plan["kind"] == "build":
            data = {**plan.get("data", {}), "decision": (g["act"], g["id"])}
            self.task = Task("build", g["id"], plan["where"], data=data, started=hi.frame)
        else:
            self.task = Task("command", data=plan, started=hi.frame)
        self.tracker.note_decision(g["act"], g["id"], obs["frame"])
        self.decisions.append((hi.frame, g["name"], g["p"]))
        self.goal = None

    def _plan(self, act: int, uid: int, obs: dict, mine: list[dict], done: list[dict], me: dict) -> dict | None:
        """How to carry out this decision now; WAIT if Gary will be able to soon (the building that
        makes it is busy); None if Gary can't (a requirement is missing, nothing can make it, or
        v0.2 doesn't know how). Gary waits for a busy building rather than doing something less
        likely: the model's next decision is usually right, it just isn't time yet."""
        have = {u["type"] for u in done}
        if any(r not in have for r in T.REQUIRES.get(uid, [])):
            return None
        if self.failed_until.get((act, uid), 0) > self.hi.frame:
            return None                       # just failed (e.g. no room): not again right away
        cap = self.model.cap(uid, obs["frame"]) if act in (TRAIN, BUILD, EXPAND) else None
        if cap is not None and self._count(mine, uid) >= cap:
            return None                       # already as many as nine pros in ten ever had by now
        if act == TRAIN and uid in T.PRODUCER:
            if me["supply_used"] + T.SUPPLY[uid] > me["supply_max"]:
                # supply blocked: wait if a depot is on its way, else let a depot come next
                coming = any(u["type"] == T.DEPOT and not u["completed"] for u in mine) or                     any(ut == T.DEPOT for ut, *_ in self.reserved) or                     (self.task is not None and self.task.kind == "build" and self.task.unit_type == T.DEPOT)
                return WAIT if coming else None
            if uid == T.SCV and sum(1 for u in mine if u["type"] == T.SCV) >= MAX_SCVS:
                return None
            addon = T.NEEDS_ADDON.get(uid)
            if addon is not None and not any(u["type"] == addon for u in done):
                return None                       # e.g. tanks: no Machine Shop yet
            b = self._free_building(obs, mine, done, T.PRODUCER[uid], addon)
            return self._command_or_wait(b, done, T.PRODUCER[uid], act, uid, T.COST[uid])
        if act == BUILD and uid in T.ADDON_PARENT:
            bare = [b for b in done if b["type"] == T.ADDON_PARENT[uid] and self._addon_of(b, mine) is None]
            if not bare:
                return None
            free = [b for b in bare if b.get("queue", 0) == 0 and b["order"] not in BUSY]
            return {"kind": "command", "act": act, "id": uid, "building": free[0], "cost": T.COST[uid]}                 if free else WAIT
        if act == BUILD and uid == T.REFINERY:
            spot = self._free_geyser(obs, mine)
            return {"kind": "build", "where": spot[0], "data": {"spot": spot[1]}, "cost": T.COST[uid]}                 if spot else None
        if act == BUILD and uid in T.SIZE and uid != T.CC:
            return {"kind": "build", "where": "main", "cost": T.COST[uid]}
        if act == EXPAND and uid == T.CC:
            base = self._free_base(obs)
            return {"kind": "build", "where": "natural" if base is self.natural else base,
                    "cost": T.COST[T.CC]} if base else None
        if act == RESEARCH and uid in T.RESEARCH and uid not in self.researched:
            where, cost = T.RESEARCH[uid]
            b = self._free_building(obs, mine, done, where)
            return self._command_or_wait(b, done, where, act, uid, cost)
        if act == UPGRADE and uid in T.UPGRADE:
            where, cost, levels = T.UPGRADE[uid]
            level = self.upgraded[uid]
            if level >= levels or (level >= 1 and T.UPGRADE_LEVEL_2_NEEDS not in have):
                return None
            b = self._free_building(obs, mine, done, where)
            cost = (cost[0] + 75 * level, cost[1] + 75 * level)
            return self._command_or_wait(b, done, where, act, uid, cost)
        return None

    def _command_or_wait(self, b: dict | None, done: list[dict], kind: int, act: int, uid: int,
                         cost: tuple[int, int]) -> dict | None:
        if b is not None:
            return {"kind": "command", "act": act, "id": uid, "building": b, "cost": cost}
        return WAIT if any(u["type"] == kind for u in done) else None

    # --- what Gary has -----------------------------------------------------------------------

    def _in_production(self, mine: list[dict]) -> dict[int, int]:
        """Units being made right now (the game counts them, observations don't list them)."""
        out: Counter = Counter()
        for b in mine:
            if b["completed"] and b.get("queue", 0) > 0 and b["tag"] in self.producing:
                out[self.producing[b["tag"]]] += 1
        return out

    def _fail_build(self, t: Task) -> None:
        """A building Gary decided on couldn't be placed: it didn't happen, and it isn't
        retried for 10 s."""
        if "decision" in t.data:
            act, uid = t.data["decision"]
            self.tracker.unnote_decision(act, uid)
            self.failed_until[(act, uid)] = self.hi.frame + 24 * 10
            self._say(f"couldn't build {describe(act, uid)}")
        super()._fail_build(t)

    def _spot(self, unit_type: int, base: Base, builder: int) -> tuple[int, int] | None:
        """A spot at the base, or at another of Gary's bases when that one is full."""
        spot = super()._spot(unit_type, base, builder)
        if spot is None:
            for other in self._my_bases([u for u in self.hi.observe()["units"] if u["owner"] == self.slot]):
                if other is not base and (spot := super()._spot(unit_type, other, builder)):
                    break
        return spot

    def _count(self, mine: list[dict], unit_type: int) -> int:
        """Own units of a type, counting ones in production and buildings about to be placed."""
        n = sum(1 for u in mine if u["type"] == unit_type) + self._in_production(mine)[unit_type]
        n += sum(1 for ut, *_ in self.reserved if ut == unit_type)
        if self.task is not None and self.task.kind == "build" and self.task.unit_type == unit_type:
            n += 1
        return n

    def _addon_of(self, b: dict, mine: list[dict]) -> dict | None:
        """The add-on attached to a building (it sits just right of the building's lower half)."""
        ax, ay = b["x"] + 96, b["y"] + 16
        return next((u for u in mine if u["type"] in T.ADDON_PARENT and math.dist((u["x"], u["y"]), (ax, ay)) < 24),
                    None)

    def _free_building(self, obs: dict, mine: list[dict], done: list[dict], kind: int,
                       addon: int | None = None) -> dict | None:
        blind = round(self.hi.p.reaction_ms / 42) + self.hi.act_latency_frames + 4
        for b in done:
            if b["type"] != kind or b.get("queue", 0) > 0 or b["order"] in BUSY:
                continue
            if obs["now"] - self.ordered_at.get(b["tag"], -10**9) <= blind:
                continue
            if addon is not None:
                a = self._addon_of(b, mine)
                if a is None or a["type"] != addon or not a["completed"]:
                    continue
            return b
        return None

    def _free_geyser(self, obs: dict, mine: list[dict]) -> tuple[Base, tuple[int, int]] | None:
        refineries = [u for u in mine if u["type"] == T.REFINERY] + \
                     [{"x": tx * 32 + 64, "y": ty * 32 + 32} for ut, tx, ty in self.reserved if ut == T.REFINERY]
        for base in self._my_bases(mine):
            for gx, gy in base.geysers:
                if any(math.dist((r["x"], r["y"]), (gx, gy)) < 32 for r in refineries):
                    continue
                where = "main" if base is self.main else "natural" if base is self.natural else base
                return where, ((gx - 64) // 32, (gy - 32) // 32)
        return None

    def _free_base(self, obs: dict) -> Base | None:
        """The nearest base nobody has taken (the natural first), as far as Gary has seen."""
        halls = [u for u in obs["units"] if u["type"] in TOWN_HALLS]
        reserved = [(tx * 32 + 64, ty * 32 + 48) for ut, tx, ty in self.reserved if ut == T.CC]
        enemy = [tuple(s) for s in self.enemy_starts]
        free = [b for b in self.map.bases if b is not self.main
                and not any(math.dist(b.center, (h["x"], h["y"])) < 10 * 32 for h in halls)
                and not any(math.dist(b.center, r) < 10 * 32 for r in reserved)
                and not any(math.dist(b.center, s) < 10 * 32 for s in enemy)]
        if self.natural in free:
            return self.natural
        return min(free, key=lambda b: math.dist(b.center, self.main.center)) if free else None

    # --- doing ----------------------------------------------------------------------------

    def _select(self, obs: dict, b: dict) -> bool:
        """Get this building selected (hotkey, or look at it and click). True once it is."""
        hi = self.hi
        if obs["selection"] == [b["tag"]]:
            return True
        hk = self.hotkeyed.get(b["tag"], -1)
        if hk >= 0:
            hi.hotkey_recall(hk)
            return False
        p = screen_of(obs, b["x"], b["y"])
        if p:
            hi.click(*p)
        else:
            hi.camera_minimap(b["x"], b["y"])
        return False

    def _task_command(self, obs: dict, mine: list[dict], t: Task) -> None:
        hi, d = self.hi, t.data
        b = next((u for u in mine if u["tag"] == d["building"]["tag"]), None)
        if b is None:
            self.task = None
            return
        if not self._select(obs, b):
            return
        act, uid = d["act"], d["id"]
        if act == TRAIN:
            if hi.train(uid).accepted:
                self.producing[b["tag"]] = uid
                self.ordered_at[b["tag"]] = obs["now"]
        elif act == RESEARCH:
            if hi.research(uid).accepted:
                self.researched.add(uid)
        elif act == UPGRADE:
            if hi.upgrade(uid).accepted:
                self.upgraded[uid] += 1
        elif act == BUILD:                      # an add-on, next to the selected building
            tx, ty = (b["x"] - 64) // 32 + T.ADDON_OFFSET[0], (b["y"] - 48) // 32 + T.ADDON_OFFSET[1]
            p = screen_of(obs, tx * 32 + 16, ty * 32 + 16)
            if not p:
                hi.camera_minimap(b["x"], b["y"])
                return
            hi.build(uid, *p, order=T.ORDER_PLACE_ADDON)
        self.task = None

    def _task_gas(self, obs: dict, mine: list[dict], t: Task) -> None:
        """Send workers to a finished refinery, a few at a time, until three are on it."""
        hi, d = self.hi, t.data
        ref = next((u for u in mine if u["tag"] == d["tag"]), None)
        if ref is None or self._on_gas(obs, d["tag"]) >= 3:
            self.task = None
            return
        rp = screen_of(obs, ref["x"], ref["y"])
        if not rp:
            hi.camera_minimap(ref["x"], ref["y"])
            return
        if t.stage == "send":
            picked = [u["tag"] for u in mine if u["tag"] in obs["selection"] and u["type"] == T.SCV]
            if picked:
                hi.right_click(*rp)
                self.gas_workers.setdefault(d["tag"], set()).update(picked)
            t.stage = "start"
            d["tries"] = d.get("tries", 0) + 1
            if d["tries"] >= 5:                 # look again in 20 s rather than keep trying now
                self.ignore_idle_until[d["tag"]] = hi.frame + 24 * 20
                self.task = None
            return
        workers = [u for u in mine if self._free_worker(obs, u) and u["order"] not in T.GAS_ORDERS]
        if not workers:
            self.task = None
            return
        w = min(workers, key=lambda u: math.dist((u["x"], u["y"]), (ref["x"], ref["y"])))
        x, y = screen_of(obs, w["x"], w["y"])
        # a small drag box: a click on a mining worker can select its mineral field instead
        hi.box(max(0, x - 6), max(0, y - 6), min(hi.p.viewport[0] - 1, x + 6), min(hi.p.viewport[1] - 1, y + 6))
        t.stage = "send"

    def _on_gas(self, obs: dict, refinery: int) -> int:
        """Workers on this refinery: the ones Gary sent that are still on gas (on a gas order, just
        out of the refinery, or out of sight: a worker inside a refinery isn't visible), and any
        worker seen heading into it (the SCV that built it starts on gas by itself)."""
        seen = {u["tag"]: u for u in obs["units"] if u["owner"] == self.slot}
        alive = {w for w in self.gas_workers.get(refinery, set())
                 if w not in seen or seen[w]["order"] in T.GAS_ORDERS or seen[w]["order"] == RESET_HARVEST}
        alive |= {t for t, u in seen.items() if u["type"] == T.SCV and u["order"] in T.GAS_ORDERS
                  and u.get("order_target") == refinery}
        self.gas_workers[refinery] = alive
        return len(alive)

    def _say(self, text: str) -> None:
        if self.verbose:
            s = self.hi.frame // 24
            print(f"  {s // 60}:{s % 60:02d} Gary {text}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", help="a .scm/.scx map (or a pre-1.18 replay) for headless play")
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--save", default="gary_v02.rep")
    ap.add_argument("--command-delay", type=int, default=LIVE_COMMAND_DELAY)
    ap.add_argument("--model", help="macro model checkpoint (default: the latest TvZ Terran one in runs/macro)")
    ap.add_argument("--style", type=int, help="build-style cluster to steer toward (data/interim/taxonomy)")
    ap.add_argument("--quiet", action="store_true", help="don't print each decision")
    ap.add_argument("--live", action="store_true", help="play in a live SC:R client via the bridge")
    ap.add_argument("--pipe", default=r"\\.\pipe\gary_scr", help="bridge named pipe (with --live)")
    ap.add_argument("--profile", default="pro", choices=sorted(v01.PROFILES))
    ap.add_argument("--pov", help="where to save the point-of-view log (live)")
    args = ap.parse_args()
    model = MacroModel.load(args.model or latest_model())
    make = lambda hi, mapinfo: TerranGaryV2(hi, mapinfo, model, args.style, verbose=not args.quiet)
    if args.live:
        v01.play_live(args.pipe, args.minutes, args.profile, args.pov or "gary_v02_live.pov.jsonl", make_bot=make)
    else:
        if not args.map:
            ap.error("--map is required unless --live")
        v01.play(args.map, args.minutes, args.seed, args.save, command_delay=args.command_delay,
                 make_bot=make, name="Gary v0.2 (T)")


if __name__ == "__main__":
    main()
