"""Wall fights: moments of Gary's own games where zerglings come at its wall, played out by variants.

The home defenses (eval/scenarios.py) start with the zerg already at the Terran's home, mostly past
what a wall is for, and full games against the rush dummy (eval/vs_rush.py) hang on so many other
things that 48 of them can't tell small changes apart. Here: Gary (by default v0.11 with the sim
city planner, gary/agent/simcity.py) plays the rush dummy, and the first moment that 4+ zerg
fighters come within 12 tiles of one of its wall gaps (gary/walls.py wall_gaps) from outside its
frontier is a scenario. Each is then played from that moment for 45 s by each variant (a fresh
Gary of that manifest taking over, as in the home defenses; the dummy goes on as it was), with
Gary's hands seeded several ways, scored as the home defenses are (eval/score.py).

    python -m eval.walls_bench --find 48                  # scenarios from 48 seeds (saved)
    python -m eval.walls_bench --run v011+simcity v011+simcity+gap_post --draws 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ingest"))

from inventory import data_root  # noqa: E402

SOURCE = "v011+simcity"
NEAR = 12 * 32
MIN_ZERG = 4
SECONDS = 45


def path() -> Path:
    return data_root() / "interim" / "walls" / "scenarios.jsonl"


class Match:
    """Gary against the rush dummy (or another of eval/vs_rush.py's Zergs), as eval/vs_rush.py plays it."""

    def __init__(self, seed: int, attack_after_s: float, version: str = SOURCE, map_path=None,
                 opponent: str = "dummy"):
        from gary import commands as C
        from gary.agent.manifests import build_gary
        from gary.bots.zerg_rush import HATCH, LIVE_COMMAND_DELAY, ZergRush
        from gary.env import Game
        from gary.interface import PROFILES, HumanInterface
        from gary.mapinfo import MapInfo
        from eval.vs_rush import DEFAULT_MAP
        self.game = Game.new(map_path or DEFAULT_MAP, ["T", "Z"], ["Gary", "Zerg rush"], seed=seed, command_delay=LIVE_COMMAND_DELAY)
        self.map = MapInfo.from_game(self.game)
        obs = self.game.observe()
        self.slot = {("T" if u["type"] == C.COMMAND_CENTER else "Z"): u["owner"] for u in obs["units"]
                     if u["type"] in (C.COMMAND_CENTER, HATCH)}
        self.his = {r: HumanInterface(self.game, s, PROFILES["pro" if r == "T" else "b_rank"], seed=s)
                    for r, s in self.slot.items()}
        self.gary = build_gary(self.his["T"], self.map, version, 1, verbose=False)
        if opponent == "replayed":
            from gary.bots.zerg_replayed import ZergReplayed
            self.zerg = ZergReplayed(self.his["Z"], self.map, seed)
        else:
            self.zerg = ZergRush(self.his["Z"], self.map)
        self.zerg.attack_after = int(attack_after_s * 24)
        self.zerg.target = next((u["x"], u["y"]) for u in obs["units"] if u["type"] == C.COMMAND_CENTER)
        self.done = False

    def over(self) -> bool:
        players = {p["slot"] for p in self.game.observe()["players"]}
        return not set(self.slot.values()) <= players

    def step(self) -> None:
        if self.done or (self.game.frame % 24 < 2 and self.over()):
            self.done = True                          # a side is gone: nothing more to play
            self.game.step(2)
            return
        try:
            self.gary.act()
            self.zerg.act()
        except StopIteration:                         # a player was just eliminated (its view has no "me")
            self.done = True
            return
        for i, hi in enumerate(self.his.values()):
            hi.step(2, advance_game=(i == 0))

    def take_over(self, version: str, hands_seed: int) -> None:
        """After a restore: fresh interfaces, a fresh Gary of this manifest (as if it had played the
        game so far), the dummy as it was."""
        from gary.agent.manifests import build_gary
        from gary.bots.zerg_rush import ZergRush
        from gary.interface import PROFILES, HumanInterface
        state = {k: v for k, v in vars(self.zerg).items() if k not in ("hi", "map")}
        self.his = {r: HumanInterface(self.game, s, PROFILES["pro" if r == "T" else "b_rank"],
                                      seed=hands_seed if r == "T" else s) for r, s in self.slot.items()}
        for hi in self.his.values():
            hi.frame = self.game.frame
        self.gary = build_gary(self.his["T"], self.map, version, 1, verbose=False)
        self.gary.announce = []
        self.gary.adopt()
        self.zerg = ZergRush(self.his["Z"], self.map)
        vars(self.zerg).update(state)


def trigger(m: Match):
    """The wall gap 4+ zerg fighters are coming at from outside the frontier, or None."""
    from eval.score import fighter
    from gary.walls import outside, wall_gaps
    sim = m.gary.simcity
    if sim is None or sim._frontier is None or "inside" not in sim._frontier:
        return None
    obs = m.game.observe()
    gaps = wall_gaps(m.game, obs, m.slot["T"], sim._frontier)
    zerg = [u for u in obs["units"] if u["owner"] == m.slot["Z"] and fighter(u["type"])]
    for gp in gaps:
        near = [z for z in zerg if math.dist((z["x"], z["y"]), gp["at"]) < NEAR]
        if len(near) >= MIN_ZERG and all(outside(sim._frontier, (z["x"], z["y"])) for z in near):
            return gp
    return None


def find_one(args) -> dict | None:
    seed, attack_after, minutes = args
    m = Match(seed, attack_after)
    end = int(minutes * 60 * 24)
    try:
        while m.game.frame < end and not (m.game.frame % 240 < 2 and m.over()):
            m.step()
            if m.game.frame % 24 < 2 and (gp := trigger(m)):
                return {"seed": seed, "attack_after": attack_after, "frame": m.game.frame, "gap": gp["at"],
                        "width": gp["width"]}
    except Exception:  # noqa: BLE001 - one broken game must not stop the search
        return None
    finally:
        m.game.close()
    return None


_cache: dict = {}


def play(args) -> dict:
    """One scenario under one variant and hands seed: net value, as the home defenses score it."""
    sc, version, hands = args
    from eval.score import Score, fighter
    key = (sc["seed"], sc["attack_after"], sc["frame"])
    m = _cache.get(key)
    if m is None:
        for old in list(_cache.values()):
            old.game.close()
        _cache.clear()
        m = Match(sc["seed"], sc["attack_after"])
        while m.game.frame < sc["frame"]:
            m.step()
        while m.game._in_flight:                      # (a save needs the commands sent to have landed)
            m.game.step(1)
        m.game.save()
        _cache[key] = m
    else:
        m.game.restore()
    m.take_over(version, hands)
    score = Score(m.slot["T"], m.slot["Z"])
    score.note(m.game.observe(), m.game)
    end = sc["frame"] + int(SECONDS * 1000 / 42)
    while m.game.frame < end and not (m.game.frame % 240 < 2 and m.over()):
        m.step()
        if m.game.frame % 12 < 2:
            score.note(m.game.observe(), m.game)
    score.finish(m.game.observe(), m.game, fighter)
    r = score.result()
    return {"key": key, "version": version, "hands": hands, "net": r["net"], "T_lost": r["T_lost"], "Z_lost": r["Z_lost"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--find", type=int, help="find scenarios in this many seeds' games")
    ap.add_argument("--attack-after", type=float, nargs="+", default=[0, 420])
    ap.add_argument("--minutes", type=float, default=12)
    ap.add_argument("--run", nargs="+", help="variants (manifests, with options) to score")
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--parallel", type=int, default=max(1, (os.cpu_count() or 6) * 2 // 3))
    args = ap.parse_args()
    if args.find:
        jobs = [(s, a, args.minutes) for a in args.attack_after for s in range(1, args.find + 1)]
        with ProcessPoolExecutor(args.parallel) as pool:
            found = [r for r in pool.map(find_one, jobs) if r]
        path().parent.mkdir(parents=True, exist_ok=True)
        path().write_text("".join(json.dumps(r) + "\n" for r in found), encoding="utf-8")
        print(f"{len(found)} wall fights in {len(jobs)} games -> {path()}")
    if args.run:
        scs = [json.loads(line) for line in path().read_text(encoding="utf-8").splitlines()]
        jobs = [(sc, v, h) for sc in scs for v in args.run for h in range(1, args.draws + 1)]
        with ProcessPoolExecutor(args.parallel) as pool:
            rows = list(pool.map(play, jobs, chunksize=len(args.run) * args.draws))
        per = {}
        for r in rows:
            per.setdefault(r["version"], {}).setdefault(tuple(r["key"]), []).append(r["net"])
        base = {k: st.mean(v) for k, v in per[args.run[0]].items()}
        print(f"{len(scs)} wall fights, {args.draws} draws each, {SECONDS} s; net value (zerg lost - terran lost):")
        for v in args.run:
            m = {k: st.mean(x) for k, x in per[v].items()}
            line = f"  {v:32s} {st.mean(m.values()):6.0f}"
            if v != args.run[0] and len(m) > 1:
                d = [m[k] - base[k] for k in m]
                line += f"   vs {args.run[0]}: {st.mean(d):+.0f} ± {st.stdev(d) / len(d) ** .5:.0f}, " \
                        f"better in {sum(x > 0 for x in d)}, worse in {sum(x < 0 for x in d)}"
            print(line)


if __name__ == "__main__":
    main()
