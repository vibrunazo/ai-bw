"""A headless game between two Garys (docs/REFACTOR.md Phase 3): each side a manifest (gary/agent/manifests.py),
its race the manifest's, on a map, from a seed, for some minutes. The one launcher: the league (eval/league.py),
mirrors (--a gz1 --b gz1), the tests and the tools that step a game (Match) all play through it.

Every 30 s a line with both sides' buildings, army and economy; at the end the result line ("A won", "B won" or
"time") and the replay, both sides' screens (.pov.jsonl for A, .bpov.jsonl for B), traces (.trace.jsonl,
.btrace.jsonl) and the shared decision records (.records.jsonl) beside it. Paired fight episodes (GARY_PAIRS:
eval/pairs.py) as before.

Usage:
  python -m eval.match --a gt1 --b gz1 --map <map> --seed 4004 --minutes 20
  python -m eval.match --a gz1 --b gz1 --map <map> --seed 6001 --minutes 20 --save runs/mirror/m.rep
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from gary.agent.manifests import build_gary, manifest_name, manifest_of  # noqa: E402
from gary.decision_log import LOG  # noqa: E402
from gary.env import LIVE_COMMAND_DELAY, Game  # noqa: E402
from gary.interface import PROFILES, HumanInterface  # noqa: E402
from gary.mapinfo import MapInfo  # noqa: E402
from gary.race import Race  # noqa: E402

FIGHTERS = {0, 32, 34, 37, 38, 43, 65, 66, 67, 68}   # the status line's army: marines, firebats, medics, zerglings,
                                                     # hydralisks, mutalisks, zealots, dragoons, templar, archons


def _slots(game, races: list[str]) -> list[int]:
    """Each side's slot: the owner of a town hall of its race (in a mirror, in slot order)."""
    obs = game.observe()
    halls = [u for u in obs["units"] if any(u["type"] in Race.of(r).town_halls for r in races)]
    out: list[int] = []
    for r in races:
        s = next(u["owner"] for u in sorted(halls, key=lambda u: u["owner"])
                 if u["type"] in Race.of(r).town_halls and u["owner"] not in out)
        out.append(s)
    return out


class Match:
    """A game to step by hand (the tools that branch, probe or record games): side A, Gary (a Terran manifest by
    default), against side B (gz1 by default), with pro hands. m.gary, m.zerg: the two Garys; m.slot, m.his:
    their slots and hands by race letter."""

    def __init__(self, seed: int, version: str, map_path, opponent: str = "gz1", style: int | None = None):
        races = [manifest_of(version).get("race", "T"), manifest_of(opponent).get("race", "T")]
        self.game = Game.new(map_path, races, ["Gary", "Gary B"], seed=seed, command_delay=LIVE_COMMAND_DELAY)
        self.map = MapInfo.from_game(self.game)
        slots = _slots(self.game, races)
        self.slot = dict(zip(races, slots))
        self.his = {r: HumanInterface(self.game, s, PROFILES["pro"], seed=s) for r, s in self.slot.items()}
        for r, hi in self.his.items():
            hi.side = r
        self.gary = build_gary(self.his[races[0]], self.map, version, style, verbose=False, seed=seed)
        self.zerg = build_gary(self.his[races[1]], self.map, opponent, seed=seed)
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


def play(map_path: str, minutes: float, seed: int, save: str, a: str, b: str, style: int | None = None) -> str:
    """One game, A against B; its result ("A won", "B won", "time")."""
    races = [manifest_of(a).get("race", "T"), manifest_of(b).get("race", "T")]
    mirror = races[0] == races[1]
    with Game.new(map_path, races, [f"Gary {a}", f"Gary {b}"], seed=seed, command_delay=LIVE_COMMAND_DELAY) as game:
        mapinfo = MapInfo.from_game(game)
        slots = _slots(game, races)
        obs = game.observe()
        # (the hands in the order the town halls come in the game's units, A's first on a tie, as the old
        # launcher made them: which hands step the game is part of a game's seed)
        first = {}
        for u in obs["units"]:
            for k in (0, 1):
                if u["owner"] == slots[k] and u["type"] in Race.of(races[k]).town_halls and slots[k] not in first:
                    first[slots[k]] = len(first)
        order = sorted(range(2), key=lambda k: first.get(slots[k], k))
        his = {k: HumanInterface(game, slots[k], PROFILES[manifest_of((a, b)[k]).get("hands", "pro")], seed=slots[k])
               for k in order}
        LOG.clear_records()
        for k, hi in his.items():
            hi.side = races[k]
        if mirror:
            LOG.slot = slots[0]
        garys = {0: build_gary(his[0], mapinfo, a, style, seed=seed)}
        if mirror:
            LOG.slot = slots[1]
        garys[1] = build_gary(his[1], mapinfo, b, seed=seed)
        LOG.slot = None
        end = int(minutes * 60 * 24)
        result = "time"
        from eval.pairs import Brancher                    # (paired fight episodes, when GARY_PAIRS is set)
        slot_of = {"T": slots[races.index("T")] if "T" in races else slots[0],
                   "Z": slots[races.index("Z")] if "Z" in races else slots[1]}
        brancher = Brancher(game, slot_of, seed or 0)
        ctx = {"gary": garys[0], "zerg": garys[1], "his": {races[k]: his[k] for k in his}, "slot_of": slot_of}
        lead = order[0]
        while his[lead].frame < end:
            players = {p["slot"] for p in game.observe()["players"]}
            if not set(slots) <= players:                     # someone was eliminated
                result = "A won" if slots[0] in players else "B won"
                break
            for k in (0, 1):
                if mirror:
                    LOG.slot = slots[k]                       # (the records say whose they are: eval/why.py --slot)
                garys[k].act()
            for i, k in enumerate(order):
                if mirror:
                    LOG.slot = slots[k]
                his[k].step(2, advance_game=(i == 0))
            LOG.slot = None
            if brancher.on:
                brancher.maybe(game, ctx)
            if his[lead].frame % (24 * 30) < 2:
                o = game.observe()
                t = his[lead].frame // 24
                parts = {}
                for k, label in ((0, "A" if mirror else races[0]), (1, "B" if mirror else races[1])):
                    mine = [u for u in o["units"] if u["owner"] == slots[k]]
                    race = Race.of(races[k])
                    p = next((q for q in o["players"] if q["slot"] == slots[k]), {"minerals": 0, "gas": 0})
                    parts[label] = (sum(1 for u in mine if 106 <= u["type"] <= 175),
                                    sum(1 for u in mine if u["type"] in FIGHTERS),
                                    sum(1 for u in mine if u["type"] == race.worker),
                                    sum(1 for u in mine if u["type"] in race.town_halls), p["minerals"], p["gas"])
                (la, pa), (lb, pb) = parts.items()
                print(f"{t // 60}:{t % 60:02d}  buildings {la} {pa[0]} {lb} {pb[0]}  army {la} {pa[1]} {lb} {pb[1]}  "
                      f"workers {la} {pa[2]} {lb} {pb[2]}  halls {la} {pa[3]} {lb} {pb[3]}  "
                      f"bank {la} {pa[4]}/{pa[5]} {lb} {pb[4]}/{pb[5]}", flush=True)
                if not pa[0] or not pb[0]:
                    result = "B won" if not pa[0] else "A won"
                    break
        game.save_replay(save)
        brancher.save()
        base = save[:-4] if save.endswith(".rep") else save
        his[0].save_pov(base + ".pov.jsonl")
        his[1].save_pov(base + ".bpov.jsonl")
        garys[0].save_trace(base + ".trace.jsonl")
        garys[1].save_trace(base + ".btrace.jsonl")
        LOG.save_records(base + ".records.jsonl")
        print(f"{result}; saved {save} (screens, traces and records beside it)")
        return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, type=manifest_name, help="side A's manifest")
    ap.add_argument("--b", required=True, type=manifest_name, help="side B's manifest")
    ap.add_argument("--map", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--style", type=int, help="A's build style (taxonomy cluster) to steer its macro toward")
    ap.add_argument("--save", default="match.rep")
    args = ap.parse_args()
    play(args.map, args.minutes, args.seed, args.save, args.a, args.b, args.style)


if __name__ == "__main__":
    main()
