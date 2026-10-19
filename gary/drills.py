"""Micro drills: small fights made up on real maps (#2; ARCHITECTURE.md §7.8, T0 micro curriculum).

A drill is a real map with a few units added for each side as preplaced units (a "use map
settings" game, as players build micro maps in the map editor), so saved drill replays play back
like any game. The Terran side is played by Gary's fight layer alone (no base, no macro) through
the human interface; the Zerg side by a script (zerglings attack-move at the Terran units). Drills
are drawn at random from a family (unit counts, place on the map, approach direction), so a model
can be trained on many variations of one problem and judged on real scenarios (eval/scenarios.py).

Families:
  mm_vs_lings    4-12 marines and 0-3 medics against 8-24 zerglings, on open ground
  home_defense   a Terran base at a real start location (Command Center, 8-14 SCVs mining,
                 2-8 marines, 0-2 medics) and 6-20 zerglings coming for its mineral line: the
                 real home defenses' situation; everything the Terran loses counts, so running
                 from the zerglings costs the workers

The zerglings chase: every third of a second they attack-move at the nearest Terran unit once
they're close, at their goal (the mineral line, or the Terran units) otherwise.

    python -m gary.drills --family mm_vs_lings --n 20 --controllers gary amove nothing
    python -m gary.drills --family mm_vs_lings --n 1 --save drill.rep     # a replay to watch
"""

from __future__ import annotations

import argparse
import json
import math
import random
import struct
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gary import commands as C
from gary import terran as T
from gary.env import LIVE_COMMAND_DELAY, Game, map_chk

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAP = REPO_ROOT / "tests" / "fixtures" / "replays" / "stardata_tvz_standard_ozp3w.rep"
TERRAN_SLOT, ZERG_SLOT = 0, 1
MEDIC, ZERGLING = 34, 37
MINERALS = {176, 177, 178}
CHASE_PX = 12 * 32               # zerglings this close to Terran units go for the nearest one
SPACING = 20                     # pixels between units in a starting clump
ZERG_EVERY = 8                   # the zerg script re-targets every third of a second


@dataclass
class Drill:
    family: str
    seed: int
    terran: list[tuple[int, int, int]]          # (unit type, x, y)
    zerg: list[tuple[int, int, int]]
    seconds: float = 30.0
    map_path: str = str(DEFAULT_MAP)
    info: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.family}_{self.seed}"


# --- making drills ----------------------------------------------------------------------------

_chk: dict[str, bytes] = {}
_probe: dict[str, Game] = {}


def _map(path: str) -> tuple[bytes, Game]:
    """The map's scenario data and a game on it to ask about terrain (cached per process)."""
    if path not in _chk:
        _chk[path] = map_chk(path)
        _probe[path] = Game.new(path, ["T", "Z"], seed=0)
    return _chk[path], _probe[path]


def _open(probe: Game, x: int, y: int, r: int) -> bool:
    """Ground units can stand everywhere within r pixels of (x, y)."""
    return all(probe.walkable(int(x + dx), int(y + dy))
               for dx in range(-r, r + 1, 16) for dy in range(-r, r + 1, 16) if dx * dx + dy * dy <= r * r)


def _clump(probe: Game, x: int, y: int, types: list[int], avoid=()) -> list[tuple[int, int, int]]:
    """Units in a clump around (x, y), each on walkable ground and clear of the avoid circles
    ((x, y, radius): minerals, buildings), spiralling out until all fit."""
    out, free = [], []
    for ring in range(0, 12):
        for k in range(max(1, 8 * ring)):
            a = 2 * math.pi * k / max(1, 8 * ring)
            px, py = int(x + ring * SPACING * math.cos(a)), int(y + ring * SPACING * math.sin(a))
            if probe.walkable(px, py) and all(math.dist((px, py), (ax, ay)) > r for ax, ay, r in avoid):
                free.append((px, py))
        if len(free) >= len(types):
            break
    return [(t, *p) for t, p in zip(types, free)]


def make_drill(family: str, seed: int, map_path: str = str(DEFAULT_MAP)) -> Drill:
    rng = random.Random(seed)
    _, probe = _map(map_path)
    w, h = probe.observe()["map"]["w"], probe.observe()["map"]["h"]
    if family == "home_defense":
        return _home_defense(seed, rng, probe, w, h, map_path)
    if family != "mm_vs_lings":
        raise ValueError(f"unknown drill family {family}")
    marines, medics, lings = rng.randint(4, 12), rng.randint(0, 3), rng.randint(8, 24)
    for _ in range(500):                         # open ground for both sides
        x, y = rng.randint(256, w - 256), rng.randint(256, h - 256)
        a = rng.uniform(0, 2 * math.pi)
        d = rng.randint(320, 480)
        zx, zy = int(x + d * math.cos(a)), int(y + d * math.sin(a))
        if _open(probe, x, y, 96) and 0 < zx < w and 0 < zy < h and _open(probe, zx, zy, 96):
            break
    else:
        raise RuntimeError("no open ground found")
    terran = _clump(probe, x, y, [T.MARINE] * marines + [MEDIC] * medics)
    zerg = _clump(probe, zx, zy, [ZERGLING] * lings)
    return Drill(family, seed, terran, zerg, info={"marines": marines, "medics": medics, "lings": lings})


def _home_defense(seed: int, rng: random.Random, probe: Game, w: int, h: int, map_path: str) -> Drill:
    start = rng.choice(probe.start_locations())
    sx, sy = start["x"], start["y"]
    minerals = [(u["x"], u["y"]) for u in probe.observe()["units"]
                if u["type"] in MINERALS and math.dist((u["x"], u["y"]), (sx, sy)) < 10 * 32]
    avoid = [(u["x"], u["y"], 48) for u in probe.observe()["units"] if u["owner"] == 11] + [(sx, sy, 88)]
    mx = sum(m[0] for m in minerals) / len(minerals)
    my = sum(m[1] for m in minerals) / len(minerals)
    scvs, marines, medics, lings = rng.randint(8, 14), rng.randint(2, 8), rng.randint(0, 2), rng.randint(6, 20)
    # SCVs between the Command Center and its minerals; the army on the other side of the CC
    worker_spot = (int((sx + mx) / 2), int((sy + my) / 2))
    ax, ay = sx + (sx - mx) * 0.8, sy + (sy - my) * 0.8
    terran = [(T.CC, sx, sy)] + _clump(probe, *worker_spot, [T.SCV] * scvs, avoid)
    terran += _clump(probe, int(ax), int(ay), [T.MARINE] * marines + [MEDIC] * medics, avoid)
    # zerglings from outside the base, toward the map's center
    toward = math.atan2(h / 2 - sy, w / 2 - sx)
    for _ in range(500):
        a = toward + rng.uniform(-0.7, 0.7)
        d = rng.randint(22 * 32, 32 * 32)
        zx, zy = int(sx + d * math.cos(a)), int(sy + d * math.sin(a))
        if 64 < zx < w - 64 and 64 < zy < h - 64 and _open(probe, zx, zy, 64):
            break
    else:
        raise RuntimeError("no open ground for the zerglings")
    zerg = _clump(probe, zx, zy, [ZERGLING] * lings)
    return Drill("home_defense", seed, terran, zerg, seconds=40.0, map_path=map_path,
                 info={"scvs": scvs, "marines": marines, "medics": medics, "lings": lings,
                       "goal": worker_spot, "minerals": [list(m) for m in minerals]})


def drill_chk(drill: Drill) -> bytes:
    """The map with the drill's units as preplaced units: a UNIT section put in front of the map's
    own (every UNIT section is read, in order, and junk at the end of protected maps stays out of
    the way)."""
    base, _ = _map(drill.map_path)
    entries = b""
    for serial, (t, x, y, owner) in enumerate([(*u, TERRAN_SLOT) for u in drill.terran] +
                                              [(*u, ZERG_SLOT) for u in drill.zerg]):
        # serial, x, y, type, relation, valid flags, valid properties, owner, hp %, shields %,
        # energy %, resources, units in hangar, state flags, unused, related unit
        entries += struct.pack("<IHHHHHHBBBBIHHII", 9000 + serial, x, y, t, 0, 0, 0, owner, 100, 100, 100, 0, 0, 0, 0, 0)
    return b"UNIT" + struct.pack("<I", len(entries)) + entries + base


# --- playing drills ---------------------------------------------------------------------------

def zerg_script(game: Game, obs: dict, goal: tuple[int, int] | None = None) -> None:
    """The zerglings, in groups of 12 (as a player selects them): attack-move at the nearest Terran
    unit once within CHASE_PX of one, else at the goal (or at the nearest Terran unit if none)."""
    lings = [u for u in obs["units"] if u["owner"] == ZERG_SLOT]
    terran = [u for u in obs["units"] if u["owner"] == TERRAN_SLOT and u["type"] != T.CC]
    if not lings or not terran:
        return
    for i in range(0, len(lings), 12):
        group = lings[i:i + 12]
        cx = sum(u["x"] for u in group) / len(group)
        cy = sum(u["y"] for u in group) / len(group)
        t = min(terran, key=lambda u: (u["x"] - cx) ** 2 + (u["y"] - cy) ** 2)
        x, y = (t["x"], t["y"]) if goal is None or math.dist((t["x"], t["y"]), (cx, cy)) < CHASE_PX else goal
        game.act(ZERG_SLOT, C.select([u["tag"] for u in group]))
        game.act(ZERG_SLOT, C.targeted_order(C.ORDER_ATTACK_MOVE, int(x), int(y)))


def start_mining(game: Game, obs: dict, drill: Drill) -> None:
    """Drill setup (not Gary's hands): every SCV to the mineral field nearest to it."""
    fields = [u for u in obs["units"] if u["type"] in MINERALS and u["owner"] == 11]
    for u in obs["units"]:
        if u["owner"] == TERRAN_SLOT and u["type"] == T.SCV and fields:
            m = min(fields, key=lambda f: (f["x"] - u["x"]) ** 2 + (f["y"] - u["y"]) ** 2)
            game.act(TERRAN_SLOT, C.select([u["tag"]]))
            game.act(TERRAN_SLOT, C.right_click(m["x"], m["y"], m["tag"], m["type"]))


def drill_gary(game: Game, drill: Drill, fight_model, seed: int, verbose: bool = False):
    """Gary's fight layer alone, with human hands and the camera on its units. The hands' APM
    budget varies from play to play (4-7 actions a second, bursts of 10-20), so a model can't
    learn to count on one."""
    import dataclasses
    from gary.agent.manifests import build_gary
    from gary.interface import PROFILES, HumanInterface
    r = random.Random(drill.seed * 1_000_003 + seed)
    profile = dataclasses.replace(PROFILES["pro"], apm_per_second=r.uniform(4.0, 7.0), apm_capacity=r.uniform(10.0, 20.0))
    hi = HumanInterface(game, TERRAN_SLOT, profile, seed=1)
    army = [u for u in drill.terran if u[0] not in (T.SCV, T.CC)] or drill.terran
    x = sum(u[1] for u in army) / len(army)
    y = sum(u[2] for u in army) / len(army)
    vw, vh = hi.p.viewport
    hi.camera = (max(0, int(x - vw / 2)), max(0, int(y - vh / 2)))
    w, h = hi.map_size
    bot = build_gary(hi, None, "drill", verbose=verbose, seed=seed, flip=(x > w / 2, y > h / 2), fight_model=fight_model)
    return bot, hi


def run_drill(drill: Drill, controller: str = "gary", fight_path: str | None = None, sample_seed: int = 0,
              save: str | None = None, verbose: bool = False, log: bool = False) -> dict:
    """Play a drill: Terran by "gary" (the fight layer), "amove" (one attack-move at the zerglings,
    then nothing: the baseline a player beats with micro) or "nothing" (units only fight back).
    Returns value lost by each side and the units left."""
    row = {"drill": drill.name, "controller": controller}
    try:
        game = Game.custom(drill_chk(drill), drill.name, [TERRAN_SLOT, ZERG_SLOT], ["T", "Z"],
                           ["Gary", "Zerglings"], seed=drill.seed, command_delay=LIVE_COMMAND_DELAY)
    except Exception as e:  # noqa: BLE001 - a drill that can't be set up is an error row
        row["error"] = f"{type(e).__name__}: {e}"[:200]
        return row
    try:
        bot = hi = None
        if controller == "gary":
            bot, hi = drill_gary(game, drill, _fight_model(fight_path), sample_seed, verbose)
            if log:
                bot.fight.rl_log = []
        obs = game.observe()
        start = {u["tag"]: (u["owner"], u["type"]) for u in obs["units"] if u["owner"] in (TERRAN_SLOT, ZERG_SLOT)}
        goal = tuple(drill.info["goal"]) if "goal" in drill.info else None
        if drill.family == "home_defense":
            start_mining(game, obs, drill)
        if controller == "amove":
            zerg = [u for u in obs["units"] if u["owner"] == ZERG_SLOT]
            mine = [u["tag"] for u in obs["units"] if u["owner"] == TERRAN_SLOT and u["type"] not in (T.SCV, T.CC)]
            zx, zy = sum(u["x"] for u in zerg) / len(zerg), sum(u["y"] for u in zerg) / len(zerg)
            for i in range(0, len(mine), 12):
                game.act(TERRAN_SLOT, C.select(mine[i:i + 12]))
                game.act(TERRAN_SLOT, C.targeted_order(C.ORDER_ATTACK_MOVE, int(zx), int(zy)))
        end = game.frame + int(drill.seconds * 1000 / 42)
        from eval.score import Score, fighter
        score = Score(TERRAN_SLOT, ZERG_SLOT)
        score.note(obs, game)
        while game.frame < end:
            if game.frame % ZERG_EVERY < 2:
                zerg_script(game, game.observe(), goal)
            if bot:
                bot.act()
                hi.step(2)
            else:
                game.step(2)
            if game.frame % 12 < 2:              # the score as it goes (credit for decisions: RL)
                score.note(game.observe(), game)
        score.finish(game.observe(), game, fighter)
        left = {TERRAN_SLOT: 0, ZERG_SLOT: 0}
        for tag, (owner, kind) in start.items():
            if game.unit_type_of(tag) != -1:
                left[owner] += 1
        res = score.result()
        row.update(res, T_left=left[TERRAN_SLOT], Z_left=left[ZERG_SLOT], SCVs_lost=res["T_workers_lost"],
                   **{k: v for k, v in drill.info.items() if k not in ("goal", "minerals")})
        if bot and log:
            row["log"] = bot.fight.rl_log
        if hi:
            row["acts"] = sum(1 for e in hi.pov if "act" in e)
        if save:
            game.save_replay(save)
            if hi:
                hi.save_pov(save[:-4] + ".pov.jsonl")
    except Exception as e:  # noqa: BLE001 - one broken drill must not stop a batch
        row["error"] = f"{type(e).__name__}: {e}"[:200]
    finally:
        game.close()
    return row


_models: dict = {}


def _fight_model(path: str | None):
    if path not in _models:
        import torch
        torch.set_num_threads(1)
        from gary.agent.manifests import model_for
        from gary.policy.fight_cmd import load_fight_policy
        _models.clear()
        _models[path] = load_fight_policy(path or model_for("fight_cmd_memory", "TvZ", "T"))
    return _models[path]


def behavior(rows: list[dict]) -> dict:
    """What the fight model did in logged plays: its command mix, and its moves (how far, and how
    often toward the zerglings): the check that a score isn't a trick (e.g. running away)."""
    from gary.policy.fight import ACTIONS
    from gary.policy.fight_cmd import point_of
    kinds, dist, toward = {}, [], []
    for r in rows:
        for d in r.get("log", []):
            a = ACTIONS[d["type"]]
            kinds[a] = kinds.get(a, 0) + 1
            if a in ("move", "attack_move") and d["cell"] >= 0 and d["select"]:
                rows_ = d["rows"]
                px, py = point_of(d["cell"])
                sel = rows_[d["select"]]
                mx, my = sel[:, 2].mean(), sel[:, 3].mean()
                dist.append(math.hypot(px - mx, py - my))
                en = rows_[rows_[:, 1] == 0]
                if len(en):
                    toward.append(np.hypot(en[:, 2] - px, en[:, 3] - py).min() < np.hypot(en[:, 2] - mx, en[:, 3] - my).min())
    n = sum(kinds.values()) or 1
    return {"mix": {k: round(v / n, 2) for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])[:5]},
            "move_px": round(float(np.median(dist)), 0) if dist else None,
            "moves_toward": round(float(np.mean(toward)), 2) if toward else None}


def _job(args: tuple) -> dict:
    family, seed, controller, fight_path, sample_seed = args
    return run_drill(make_drill(family, seed), controller, fight_path, sample_seed, log=controller == "gary")


def main() -> None:
    import os
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="mm_vs_lings")
    ap.add_argument("--n", type=int, default=20, help="drills (seeds 0..n-1)")
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--controllers", nargs="+", default=["gary", "amove", "nothing"])
    ap.add_argument("--fight-model", help="Gary's fight model (default: the latest command model with memory)")
    ap.add_argument("--parallel", type=int, default=max(1, (os.cpu_count() or 6) * 2 // 3))
    ap.add_argument("--save", help="with --n 1: save the drill's replay (and Gary's POV) here")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    seeds = range(args.first_seed, args.first_seed + args.n)
    if args.save:
        d = make_drill(args.family, seeds[0])
        print(json.dumps(d.info), len(d.terran), "terran,", len(d.zerg), "zerg")
        for c in args.controllers:
            path = args.save if len(args.controllers) == 1 else args.save.replace(".rep", f"_{c}.rep")
            r = run_drill(d, c, args.fight_model, save=path, verbose=args.verbose)
            print(c, {k: v for k, v in r.items() if k != "log"}, "->", path)
        return
    jobs = [(args.family, s, c, args.fight_model, 0) for s in seeds for c in args.controllers]
    with ProcessPoolExecutor(args.parallel) as pool:
        rows = list(pool.map(_job, jobs, chunksize=len(args.controllers)))
    print(f"{args.n} {args.family} drills, {make_drill(args.family, seeds[0]).seconds:g} s each; mean per drill:")
    for c in args.controllers:
        rs = [r for r in rows if r["controller"] == c and "error" not in r]
        errs = [r["error"] for r in rows if r["controller"] == c and "error" in r]
        if rs and c == "gary":
            print(f"  {'':8s} behavior: {behavior(rs)}")
        if rs:
            print(f"  {c:8s} net {np.mean([r['net'] for r in rs]):6.0f}  terran lost {np.mean([r['T_lost'] for r in rs]):5.0f}  "
                  f"zerg lost {np.mean([r['Z_lost'] for r in rs]):5.0f}  terran units left {np.mean([r['T_left'] for r in rs]):4.1f}"
                  + (f"  SCVs lost {np.mean([r['SCVs_lost'] for r in rs]):.1f}" if args.family == "home_defense" else "")
                  + (f"  ({len(errs)} errors: {errs[0]})" if errs else ""))
        else:
            print(f"  {c:8s} all {len(errs)} failed: {errs[:1]}")


if __name__ == "__main__":
    main()
