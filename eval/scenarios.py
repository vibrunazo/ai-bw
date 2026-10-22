"""Scenario tests from real pro positions (#2, step 4).

A held-out pro TvZ replay plays in OpenBW up to an early skirmish at the Terran's home (zerg
fighters attacking near a Terran town hall before 6:00; not the Terran's own attacks, like bunker
rushes). There the Terran is handed to a controller while the Zerg keeps replaying what the pro
Zerg did, and the next seconds are scored by the value (minerals + gas) each side loses.
Controllers:

  pro       the Terran pro's own commands keep playing (what really happened)
  gary      Gary (v0.4 by default) takes over the Terran with human hands
  nothing   nobody touches the Terran: units keep their last orders (the floor)

    python -m eval.scenarios --pick 2500 --screen     # find scenarios in (all) held-out games
    python -m eval.scenarios --run 239 --draws 8      # score them under each controller

--pick proposes up to 3 early fights per game; --screen plays each under the pro and under
nothing and keeps, per game, the first that is at the Terran's home (decided from the game at the
takeover: the fight is nearer a Terran town hall than a Zerg one, and no Terran bunker stands at a
Zerg base) and decisive: the pro's control was worth at least --min-gap (net value: zerg lost -
terran lost) over leaving the units alone. Elsewhere the zerglings weren't a threat and every
controller ties.

The Zerg replays commands, not intentions: once the Terran plays differently, a zerg command may
target a unit that isn't there any more, and units the Zerg makes after the takeover can get
different unit IDs than in the real game, so the Zerg's later commands to them may miss. Keep the
window short (the default is 45 s).

Worse, replayed zerg units don't react to a Terran that isn't where the pro's was: zerglings
moving to surround an army that isn't there walk on past it, and a gap the pro never left open is
never run through. So by default (--zerg reactive) the Zerg reacts once the game has diverged (a
Terran unit 4+ tiles from where the pro's was at that moment, or dead when the pro's wasn't; the
pro's positions are recorded from the pro's own play of the scenario, cached under
data/interim/scenarios/pro_tracks/). Then, around the Terran units that aren't where the pro's were
(4+ tiles off, or units the pro didn't have; Terran units where the pro's were keep facing the pro
Zerg's own choices, like marines holding a wall gap the pro Zerg chose not to attack), zerg
fighters within 6 tiles that stand, or move toward a spot no Terran unit is near any more, do what
pro zergs do: a group of up
to 12 moves past the nearest Terran unit (attack-moving lings trickle in one by one; pros move
them around the marines first, losing a few on the way), then attack-moves once half of it is
wrapped around Terran units (or after 2 s). Replayed moves that still fit (to within 4 tiles of a
Terran unit: the pro's own surround) or lead away (a retreat) stay the replay's, as does
everything else (the Zerg's macro, its later waves). One command group every third of a second,
the replay's own selection put back at once. The pro's own play never diverges: its score is as
before. --zerg replay: the replay only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from eval import priority
from gary.agent.manifests import manifest_name

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ingest"))

from inventory import data_root  # noqa: E402

from eval.score import Score, fighter  # noqa: E402
from gary import terran as T  # noqa: E402
from gary.policy.army import is_building  # noqa: E402

BEFORE_S = 360                  # skirmishes before 6:00
LEAD_FRAMES = 24                # take over one second before the skirmish snapshot
ZERG_FIGHTERS = {37, 38, 39, 43, 47, 50, 103}   # zergling, hydra, ultra, muta, scourge, infested, lurker
MIN_FIGHTERS = 3
CANDIDATES_PER_GAME = 3         # --pick: fights proposed per game...
CANDIDATE_GAP_S = 30            # ...at least this far apart
T_HALLS, Z_HALLS, BUNKER = {106}, {131, 132, 133}, 125
CONTROLLERS = ["pro", "gary", "nothing"]
DIVERGE_PX = 4 * 32             # a Terran unit this far from where the pro's was: the game has diverged
REACT_PX = 6 * 32               # (then) zerg fighters this close to Terran units, not attacking, react
REACT_EVERY = 8                 # frames between the reactive zerg's commands (one group each)
STILL_FITS_PX = 4 * 32          # a replayed move to within this of a Terran unit: the pro's surround, kept
SURROUND_PAST = 2 * 32          # the reactive surround moves this far past the nearest Terran unit
WRAPPED_PX = 48                 # a surrounding unit this close to a Terran unit: wrapped around it
SURROUND_FOR = 48               # frames: attack after this long even if not wrapped
REACTABLE = {1, 2, 3, 6}        # Stop, Guard, PlayerGuard, Move (not attacking, holding, burrowing...)
MOVE = 6
TRACK_EVERY = 12
ATTACK_ORDERS = {10, 11, 12, 14}


def scenario_id(sc: dict) -> str:
    """A scenario's name in rows and replays: its game's sha1 prefix (and, suite kinds, the kind's initial: a game
    gives one scenario per kind)."""
    return sc["sha1"][:8] + (sc["kind"][0] if sc.get("kind") else "")


def scenarios_path(split: str = "test", suite: str = "early") -> Path:
    """test: from games held out of training (for scoring); train: from training games (for
    reinforcement learning, train/fight_rl.py). suite: "early" (home defenses before 6:00) or "kinds"
    (eval/scenario_kinds.py: attacks, mixed fights, outnumbered units, harassment, from whole games)."""
    name = "tvz_early" if suite == "early" else f"tvz_{suite}"
    return data_root() / "interim" / "scenarios" / (f"{name}.jsonl" if split == "test" else f"{name}_train.jsonl")


# --- picking -----------------------------------------------------------------------------------

def pick(n: int, split: str = "test") -> list[dict]:
    """Early skirmishes, up to CANDIDATES_PER_GAME per TvZ game of the split (held-out games for
    "test"), at least CANDIDATE_GAP_S apart, from the fight dataset (ingest/fight_dataset.py);
    n candidates in all. Whether one is at the Terran's home is decided by --screen."""
    from train.macro import is_test
    fight_dir = data_root() / "interim" / "fight" / "v3"
    rows = {}
    with open(fight_dir / "index.jsonl", encoding="utf-8") as f:
        for r in map(json.loads, f):
            rows[r["sha1"]] = r
    out = []
    for r in sorted(rows.values(), key=lambda r: r["sha1"]):
        if not r.get("ok") or r["matchup"] != "TvZ" or is_test(r["sha1"]) != (split == "test"):
            continue
        ti = next((i for i, p in enumerate(r["players"]) if p["race"] == "T" and p["snapshots"] > 0), None)
        if ti is None:
            continue
        z = dict(np.load(fight_dir / r["sha1"][:2] / f"{r['sha1']}.npz"))   # each array decompressed once
        frames, players, starts, units = z["snap_frame"], z["snap_player"], z["snap_start"], z["units"]
        last = -10 ** 9
        taken = 0
        for i in range(len(frames)):
            if players[i] != ti or frames[i] * 42 / 1000 > BEFORE_S or frames[i] - last < CANDIDATE_GAP_S * 1000 / 42:
                continue
            u = units[starts[i]:starts[i + 1]]
            own_buildings = int(sum(1 for k, side in u[:, :2] if side == 1 and is_building(int(k))))
            fighters = int(sum(1 for k, side in u[:, :2] if side == 0 and int(k) in ZERG_FIGHTERS))
            if own_buildings and fighters >= MIN_FIGHTERS:
                out.append({"sha1": r["sha1"], "rel_path": r["rel_path"], "map": r.get("map"),
                            "terran_slot": r["players"][ti]["slot"], "terran": r["players"][ti]["name"],
                            "frame": max(0, int(frames[i]) - LEAD_FRAMES), "zerg_fighters": fighters,
                            "own_units": int((u[:, 1] == 1).sum()),
                            "cx": int(z["snap_cx"][i]), "cy": int(z["snap_cy"][i])})
                last, taken = int(frames[i]), taken + 1
                if taken == CANDIDATES_PER_GAME:
                    break
        if len(out) >= n:
            break
    return out


# --- running -----------------------------------------------------------------------------------

def stalled_buildings(obs: dict, slot: int) -> int:
    """The player's unfinished buildings that no SCV is building (e.g. its builder was killed)."""
    from gary.policy.army import is_building
    building = {u["order_target"] for u in obs["units"] if u["owner"] == slot and u["type"] == T.SCV}
    return sum(1 for u in obs["units"] if u["owner"] == slot and is_building(u["type"]) and not u["completed"]
               and u["tag"] not in building and u["type"] not in T.ADDON_PARENT)   # (add-ons build themselves)


def at_home(obs: dict, slot: int, center: tuple[int, int]) -> bool:
    """Whether a fight (its center) is at the Terran's (slot) home: nearer a Terran town hall than
    a Zerg one, and no Terran bunker at a Zerg base (a bunker rush)."""
    mine = [(u["x"], u["y"]) for u in obs["units"] if u["owner"] == slot and u["type"] in T_HALLS]
    theirs = [(u["x"], u["y"]) for u in obs["units"] if u["owner"] not in (slot, 11) and u["type"] in Z_HALLS]
    near = lambda halls, p: min((math.dist(p, h) for h in halls), default=1e9)
    rush = any(near(theirs, (u["x"], u["y"])) < 12 * 32 for u in obs["units"]
               if u["owner"] == slot and u["type"] == BUNKER)
    return near(mine, center) < near(theirs, center) and not rush


_games: OrderedDict = OrderedDict()      # per process: (sha1, frame) -> its game, saved at the takeover
CACHED_GAMES = 4


def scenario_game(sc: dict):
    """The scenario's game at the takeover frame: loaded and replayed up to it the first time,
    then restored from an exact copy (OpenBW's state copy) for every further play."""
    from gary.env import LIVE_COMMAND_DELAY, Game
    key = (sc["sha1"], sc["frame"])
    game = _games.pop(key, None)
    if game is None:
        game = Game.scenario(data_root() / "raw" / sc["rel_path"], command_delay=LIVE_COMMAND_DELAY)
        game.step(sc["frame"])
        game.save()
    else:
        game.restore()
    _games[key] = game
    while len(_games) > CACHED_GAMES:
        _games.popitem(last=False)[1].close()
    return game


_tracks: dict = {}                      # per process: (sha1, frame) -> the pro's Terran units over time


def pro_track(sc: dict) -> dict[int, dict[int, tuple[int, int]]]:
    """Where the pro's Terran units were, every TRACK_EVERY frames from the takeover, in the pro's own
    play of the scenario: {frame: {tag: (x, y)}}. Recorded once, cached on disk."""
    key = (sc["sha1"], sc["frame"])
    if key in _tracks:
        return _tracks[key]
    path = data_root() / "interim" / "scenarios" / "pro_tracks" / f"{sc['sha1'][:12]}_{sc['frame']}.json"
    if path.exists():
        track = {int(f): {int(t): tuple(p) for t, p in units.items()}
                 for f, units in json.loads(path.read_text(encoding="utf-8")).items()}
    else:
        track = {}

        def record(game, slot):
            if (game.frame - sc["frame"]) % TRACK_EVERY == 0:
                track[game.frame] = {u["tag"]: (u["x"], u["y"]) for u in game.observe()["units"] if u["owner"] == slot}
        run_one(sc, "pro", 60, "v010", None, on_step=record, zerg_play="replay")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({f: {t: list(p) for t, p in u.items()} for f, u in track.items()}), encoding="utf-8")
        tmp.replace(path)
    _tracks[key] = track
    return track


def diverged(obs: dict, slot: int, then: dict[int, tuple[int, int]]) -> bool:
    """Whether the Terran's units are no longer where the pro's were at this moment."""
    now = {u["tag"]: (u["x"], u["y"]) for u in obs["units"] if u["owner"] == slot}
    return any(t not in now or math.dist(now[t], p) > DIVERGE_PX for t, p in then.items())


class ReactiveZerg:
    """The scenario Zerg's fighters reacting to where the Terran's units really are (module doc)."""

    def __init__(self, slot: int, zerg: int, track: dict):
        self.slot, self.zerg, self.track = slot, zerg, track
        self.surrounding: list[tuple[list[int], int]] = []     # (tags, since frame)
        self.orders = 0

    def step(self, game) -> None:
        from gary import commands as C
        obs = game.observe()
        terran = [u for u in obs["units"] if u["owner"] == self.slot]
        terran = [u for u in terran if not is_building(u["type"])] or terran
        then = self.track[max((f for f in self.track if f <= game.frame), default=min(self.track))] if self.track else {}
        moved = [u for u in terran if u["tag"] not in then or math.dist((u["x"], u["y"]), then[u["tag"]]) > DIVERGE_PX]
        if not moved:
            return
        zerg = {u["tag"]: u for u in obs["units"] if u["owner"] == self.zerg}
        near = lambda u, d: any(math.dist((u["x"], u["y"]), (t["x"], t["y"])) < d for t in moved)
        for g in list(self.surrounding):                # wrapped around them: attack
            tags, since = g
            moving = [zerg[t] for t in tags if t in zerg and zerg[t]["order"] == MOVE]   # (else the replay took them)
            if not moving:
                self.surrounding.remove(g)
                continue
            if 2 * sum(near(u, WRAPPED_PX) for u in moving) >= len(moving) or game.frame - since >= SURROUND_FOR:
                self.surrounding.remove(g)
                self._order(game, moving, C.targeted_order(C.ORDER_ATTACK_MOVE, *self._nearest(moving, moved)))
                return
        busy = {t for tags, _ in self.surrounding for t in tags}
        idle = [u for u in zerg.values() if fighter(u["type"]) and u["type"] != 103 and u["tag"] not in busy
                and u["order"] in REACTABLE and near(u, REACT_PX) and self._stale(u, terran)]
        if not idle:
            return
        group = sorted(idle, key=lambda u: math.dist((u["x"], u["y"]), (idle[0]["x"], idle[0]["y"])))[:12]
        cx, cy = _center(group)
        tx, ty = self._nearest(group, moved)
        d = math.dist((cx, cy), (tx, ty)) or 1
        px, py = tx + (tx - cx) / d * SURROUND_PAST, ty + (ty - cy) / d * SURROUND_PAST
        self.surrounding.append(([u["tag"] for u in group], game.frame))
        self._order(game, group, C.right_click(max(0, int(px)), max(0, int(py))))

    @staticmethod
    def _stale(u: dict, terran: list[dict]) -> bool:
        """Standing, or moving toward a spot no Terran unit is near: the replay doesn't fit any more."""
        if u["order"] != MOVE:
            return True
        dest = (u["tx"], u["ty"])
        if any(math.dist(dest, (t["x"], t["y"])) < STILL_FITS_PX for t in terran):
            return False                                # the pro's own surround
        gap = lambda p: min(math.dist(p, (t["x"], t["y"])) for t in terran)
        return gap(dest) <= gap((u["x"], u["y"]))       # (moving away: the replay's retreat)

    @staticmethod
    def _nearest(group: list[dict], terran: list[dict]) -> tuple[int, int]:
        cx, cy = _center(group)
        t = min(terran, key=lambda t: math.dist((t["x"], t["y"]), (cx, cy)))
        return int(t["x"]), int(t["y"])

    def _order(self, game, group: list[dict], command: bytes) -> None:
        from gary import commands as C
        selection = game.selection(self.zerg)
        game._run(self.zerg, C.select([u["tag"] for u in group]))      # (now, between the replay's commands)
        game._run(self.zerg, command)
        if selection:
            game._run(self.zerg, C.select(selection))
        self.orders += 1


def _center(units: list[dict]) -> tuple[float, float]:
    return sum(u["x"] for u in units) / len(units), sum(u["y"] for u in units) / len(units)


def run_one(sc: dict, controller: str, seconds: float, version: str, style: int | None,
            save: str | None = None, verbose: bool = False, fight_path: str | None = None,
            sample_seed: int | None = None, on_step=None, zerg_play: str = "reactive",
            hands_seed: int = 1) -> dict:
    """One scenario under one controller. Gary: fight_path picks the fight model (default the
    latest trained); sample_seed seeds its sampled fight decisions and the row gets them
    ("log"). on_step(game, slot), if given, is called after every step (eval/micro_timing.py).
    hands_seed: Gary's hands (reaction delays, click spread). zerg_play: "reactive" (the Zerg reacts once the game diverges from the pro's) or "replay"."""
    row = {"sha1": scenario_id(sc), "controller": controller}
    try:
        track = pro_track(sc) if zerg_play == "reactive" and controller != "pro" else None
        game = scenario_game(sc)
        obs = game.observe()
        slot = sc["terran_slot"]
        zerg = next(p["slot"] for p in obs["players"] if p["slot"] != slot and p["race"] == 0)
        score = Score(slot, zerg)
        split_at, next_react = None, 0
        reactive = ReactiveZerg(slot, zerg, track)
        score.note(obs, game)
        if "cx" in sc:                         # (the fight's center, from the data)
            row["home"] = at_home(obs, slot, (sc["cx"], sc["cy"]))
        end = game.frame + int(seconds * 1000 / 42)
        bot, hi = None, None
        if controller != "pro":
            game.take_over(slot)
        if controller == "gary":
            bot, hi = make_gary(game, slot, version, style, verbose, fight_path, hands_seed)
            if sample_seed is not None:
                bot.fight.rng, bot.fight.rl_log = np.random.default_rng(sample_seed), []
        fighters0 = {u["tag"] for u in obs["units"] if u["owner"] == slot and fighter(u["type"])}
        reached, n_commands, last_cmd = set(), 0, -1      # (attention: units given a command, commands given)
        while game.frame < end:
            if bot:
                bot.act()
                hi.step(2)
                for c in hi.commands:
                    if c[0] > last_cmd:
                        reached.update(c[2])
                        n_commands += 1
                last_cmd = hi.commands[-1][0] if hi.commands else last_cmd
            else:
                game.step(2)
            if on_step:
                on_step(game, slot)
            if track is not None:              # the Zerg reacts once the game has diverged
                if split_at is None and game.frame in track and diverged(game.observe(), slot, track[game.frame]):
                    split_at = game.frame
                if split_at is not None and game.frame >= next_react:
                    next_react = game.frame + REACT_EVERY
                    reactive.step(game)
            if game.frame % 12 < 2:            # (the interface's own snapshot when it took one)
                score.note((hi and hi.latest_observation()) or game.observe(), game)
        score.finish(game.observe(), game, fighter)
        row.update(score.result())
        row["stalled"] = stalled_buildings(game.observe(), slot)
        if bot:
            row["commands_min"] = round(n_commands * 60 / seconds, 1)
            row["reached"] = round(len(reached & fighters0) / max(1, len(fighters0)), 3)
        if track is not None:
            row["diverged_s"] = round((split_at - sc["frame"]) * 42 / 1000, 1) if split_at is not None else None
            row["zerg_reacts"] = reactive.orders
        if bot and bot.fight is not None and bot.fight.rl_log is not None:
            row["log"] = bot.fight.rl_log
        if hi:
            acts = [e["act"] for e in hi.pov if "act" in e]
            row["acts"] = len(acts)
            from collections import Counter
            row["act_heads"] = dict(Counter(" ".join(a.split()[:2]) for a in acts))   # (what the hands did, by kind)
            # actions that are commands in a replay (camera moves aren't): comparable to EAPM
            row["command_acts"] = sum(1 for a in acts if "camera" not in a and not a.startswith(("F", "Shift+F", "arrow", "✗")))
            row["fight_orders"] = sum(1 for a in acts if a.startswith(("right-click", "A-click", "minimap right-click", "minimap attack", "S ", "H ", "T ", "C ")))
        if save:
            game.save_replay(save)
            if hi:
                hi.save_pov(save[:-4] + ".pov.jsonl")
                bot.save_trace(save[:-4] + ".trace.jsonl")
    except Exception as e:  # noqa: BLE001 - one broken scenario must not stop the table
        row["error"] = f"{type(e).__name__}: {e}"[:200]
    return row


_models: dict = {}                       # per process: loaded once


def _model(kind: str, path: str):
    if (kind, path) not in _models:
        import torch
        torch.set_num_threads(1)             # many scenarios run side by side
        from gary.policy.army import ArmyModel
        from gary.policy.fight_cmd import load_fight_policy
        from gary.policy.macro import MacroModel
        for k in [k for k in _models if k[0] == kind == "fight"]:
            del _models[k]                   # reinforcement learning: a new fight model each round
        _models[(kind, path)] = {"macro": MacroModel.load, "army": ArmyModel.load, "fight": load_fight_policy}[kind](path)
    return _models[(kind, path)]


def make_gary(game, slot: int, version: str, style: int | None, verbose: bool = False,
              fight_path: str | None = None, hands_seed: int = 1):
    """Gary (a manifest, gary/agent/manifests.py) taking over a game midway."""
    from gary.agent.manifests import build_gary
    from gary.interface import PROFILES, HumanInterface
    from gary.mapinfo import MapInfo
    hi = HumanInterface(game, slot, PROFILES["pro"], seed=hands_seed)
    bot = build_gary(hi, MapInfo.from_game(game), version, style, verbose=verbose, fight_path=fight_path,
                     load=lambda kind, path: _model(kind, path))
    bot.announce = []                # mid-game: no hello in the chat
    bot.adopt()                      # as if Gary had played the game so far
    return bot, hi


def _job(args: tuple) -> dict:
    return run_one(*args)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pick", type=int, help="find this many scenarios in held-out games (saved for --run)")
    ap.add_argument("--screen", action="store_true", help="keep only the picked scenarios where control mattered")
    ap.add_argument("--min-gap", type=int, default=100, help="--screen: net value the pro gained over nothing")
    ap.add_argument("--run", type=int, help="score the first N saved scenarios")
    ap.add_argument("--suite", default="early", choices=["early", "kinds"],
                    help="early: home defenses before 6:00; kinds: by state from whole games (eval/scenario_kinds.py)")
    ap.add_argument("--controllers", nargs="+", default=CONTROLLERS, choices=CONTROLLERS)
    ap.add_argument("--seconds", type=float, default=45)
    ap.add_argument("--version", default="v04", type=manifest_name,
                    help="which Gary takes over (v07free: v0.7 without its guard rail, for RL-trained fight models)")
    ap.add_argument("--style", type=int, default=1)
    ap.add_argument("--parallel", type=int, default=max(1, (os.cpu_count() or 6) * 2 // 3),
                    help="worker processes (default: 2/3 of the logical cores; more add nothing measurable)")
    ap.add_argument("--save", help="folder: save each scenario's replay (as played) there")
    ap.add_argument("--only", help="run just this scenario (sha1 prefix), Gary narrating what it does")
    ap.add_argument("--split", default="test", choices=["test", "train"],
                    help="scenarios from held-out games (test) or from training games (train, for RL)")
    ap.add_argument("--fight-model", help="Gary's fight model (default: the latest in runs/fight)")
    ap.add_argument("--samples", type=int, default=0,
                    help="Gary samples its fight decisions; score the mean over this many draws")
    ap.add_argument("--draws", type=int, default=1,
                    help="play each scenario this many times with Gary's hands seeded differently, score the mean "
                         "(fights at home often hang on a few frames: one draw is a coin flip)")
    ap.add_argument("--zerg", default="reactive", choices=["reactive", "replay"],
                    help="the Zerg reacts once the game diverges from the pro's (default), or only replays")
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    if args.pick:
        if args.suite == "kinds":
            from eval.scenario_kinds import pick as pick_kinds
            found = pick_kinds(args.pick, args.split, args.parallel)
        else:
            found = pick(args.pick, args.split)
        if args.screen:
            jobs = [(sc, c, args.seconds, args.version, args.style) for sc in found for c in ("pro", "nothing")]
            with ProcessPoolExecutor(args.parallel) as pool:
                res = list(pool.map(_job, jobs, chunksize=2))   # a scenario's plays on one worker
            net = lambda r: r["net"] if "error" not in r else None
            kept, games, away = [], set(), 0
            for sc, pro, nothing in zip(found, res[0::2], res[1::2]):
                key = (sc["sha1"], sc.get("kind"))
                if net(pro) is None or net(nothing) is None or key in games:
                    continue
                if args.suite == "early" and not pro["home"]:
                    away += 1
                    continue
                sc["gap"] = net(pro) - net(nothing)
                if sc["gap"] >= args.min_gap:
                    kept.append(sc)
                    games.add(key)
            print(f"screened {len(found)} fights: {away} not at the Terran's home; kept {len(kept)} where the pro's "
                  f"control was worth {args.min_gap}+ (one per game" + (" and kind)" if args.suite == "kinds" else ")"))
            found = kept
        path = scenarios_path(args.split, args.suite)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in found), encoding="utf-8")
        print(f"{len(found)} scenarios -> {path}")
    if args.run:
        scs = [json.loads(line) for line in scenarios_path(args.split, args.suite).read_text(encoding="utf-8").splitlines()][:args.run]
        if args.only:
            scs = [sc for sc in scs if sc["sha1"].startswith(args.only)]
            for c in args.controllers:
                print(c, run_one(scs[0], c, args.seconds, args.version, args.style, verbose=c == "gary",
                                 fight_path=args.fight_model, zerg_play=args.zerg))
            return
        if args.save:
            Path(args.save).mkdir(parents=True, exist_ok=True)
        jobs = []
        for sc in scs:
            for c in args.controllers:
                draws = args.samples if c == "gary" and args.samples else 0
                for k in range(max(1, draws)):
                    for h in range(args.draws if c == "gary" else 1):
                        name = f"{scenario_id(sc)}_{c}" + (f"_{k}" if k else "") + (f"_h{h}" if h else "")
                        save = str(Path(args.save) / f"{name}.rep") if args.save else None
                        jobs.append((sc, c, args.seconds, args.version, args.style, save, False, args.fight_model,
                                     k if draws else None, None, args.zerg, 1 + h))
        with ProcessPoolExecutor(args.parallel) as pool:
            per_scenario = len(jobs) // max(1, len(scs))       # a scenario's plays on one worker:
            rows = mean_rows(list(pool.map(_job, jobs, chunksize=per_scenario)))   # its game is reused
        report(scs, rows, args)


def mean_rows(rows: list[dict]) -> list[dict]:
    """Rows of the same scenario and controller (sampled draws) averaged into one."""
    by: dict[tuple, list[dict]] = {}
    for r in rows:
        by.setdefault((r["sha1"], r["controller"]), []).append(r)
    out = []
    for (sha, c), rs in by.items():
        ok = [r for r in rs if "error" not in r]
        if not ok:
            out.append(rs[0])
            continue
        out.append({"sha1": sha, "controller": c, **{k: int(round(np.mean([r[k] for r in ok])))
                                                      for k in ok[0] if k not in ("sha1", "controller", "log", "events")
                                                      and all(isinstance(r.get(k), (int, float)) for r in ok)}})
    return out


def report(scs: list[dict], rows: list[dict], args: argparse.Namespace) -> None:
    by = {(r["sha1"], r["controller"]): r for r in rows}
    cs = args.controllers
    print(f"{len(scs)} scenarios, {args.seconds:g} s after the takeover; value lost (minerals + gas) T / Z, "
          f"SCVs lost; Gary = {args.version} style {args.style}"
          + (f", fight decisions sampled, mean of {args.samples} draws" if args.samples else "")
          + (f", mean of {args.draws} draws of Gary's hands" if args.draws > 1 else "")
          + ("; the Zerg reacts once the game diverges" if args.zerg == "reactive" else "; the Zerg only replays"))
    print(f"{'game':9s} {'at':>5s} " + " ".join(f"{c:>18s}" for c in cs))
    tot = {c: [0, 0, 0, 0] for c in cs}
    nets = {c: [] for c in cs}
    for sc in scs:
        cells = []
        for c in cs:
            r = by[(scenario_id(sc), c)]
            if "error" in r:
                cells.append(f"{'error':>18s}")
                continue
            t = tot[c]
            t[0] += r["T_lost"]; t[1] += r["Z_lost"]; t[2] += r["T_workers_lost"]; t[3] += 1
            nets[c].append(r["net"])
            cells.append(f"{r['T_lost']:>6d} /{r['Z_lost']:>5d} ({r['T_workers_lost']:>2d})")
        s = sc["frame"] * 42 // 1000
        print(f"{scenario_id(sc):9s} {s // 60:>2d}:{s % 60:02d} " + " ".join(cells))
    print(f"{'mean':15s} " + " ".join(
        f"{t[0] / max(1, t[3]):>6.0f} /{t[1] / max(1, t[3]):>5.0f} ({t[2] / max(1, t[3]):>4.1f})" for t in tot.values()))
    print(f"{'net (score)':15s} " + " ".join(f"{np.mean(v) if v else 0:>18.0f}" for v in nets.values()))
    kinds = sorted({sc.get("kind") for sc in scs if sc.get("kind")})
    for k in kinds:                         # (suite kinds: net and orders by kind)
        for c in cs:
            ks = [by[(scenario_id(sc), c)] for sc in scs if sc.get("kind") == k and "error" not in by[(scenario_id(sc), c)]]
            if ks:
                orders = [r["fight_orders"] for r in ks if "fight_orders" in r]
                print(f"  {k:12s} {c}: {len(ks)} scenarios, net {np.mean([r['net'] for r in ks]):.0f}"
                      + (f", orders a minute {np.mean(orders) * 60 / args.seconds:.0f}" if orders else ""))
    for c in cs:
        split = [r for r in rows if r["controller"] == c and "zerg_reacts" in r]
        if split:
            ds = [r["diverged_s"] for r in split if r.get("diverged_s") is not None]
            print(f"{c}: diverged from the pro's play in {len(ds)} of {len(split)} scenarios"
                  + (f" (median {np.median(ds):.0f} s in)" if ds else "")
                  + f"; the Zerg reacted {np.mean([r['zerg_reacts'] for r in split]):.0f} times per scenario")
    errs = [r for r in rows if "error" in r]
    for r in errs[:5]:
        print("error", r["sha1"], r["controller"], r["error"])
    acts = [r for r in rows if "acts" in r]
    if acts:
        print(f"Gary's actions per scenario: {np.mean([r['acts'] for r in acts]):.0f} "
              f"(orders - right-clicks, A-clicks, minimap orders, S/H/T/C: {np.mean([r['fight_orders'] for r in acts]):.0f})")


if __name__ == "__main__":
    main()
