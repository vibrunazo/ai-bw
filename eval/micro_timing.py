"""How marines move and shoot in home defenses: the pro vs Gary (#2).

Pros stutter-step: a short move, then shooting again a fraction of a second later. Gary's fight
model gives one command every half second to one group, so a move may wait long for its attack.
On the same scenarios as eval/scenarios.py, under each controller, for every marine fighting (a
zerg fighter within 7 tiles):

  time      the share of marine-time spent attacking (attacking a unit, attack-moving, attacking
            the ground), moving, standing (idle, hold, stop) or else (entering a bunker...)
  moves     move orders a fighting marine gets, per marine-minute of fighting
  resume    after each such move: how long until it attacks again (median), the share attacking
            again within 0.5 / 1 / 2 s, and the share that died first
  deaths    marines killed while on a move order vs while attacking

Orders, not commands, so the pro's play (replayed) and Gary's are measured the same way.

    python -m eval.micro_timing --run 114 --parallel 16
    python -m eval.micro_timing --run 114 --version v08 --controllers gary
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np

MARINE = 0
ATTACK = {10, 12, 14}                # AttackUnit, AttackTile, AttackMove
MOVE = {6}
STAND = {1, 2, 3, 107}               # Stop, Guard, PlayerGuard (idle), HoldPosition
ENTER = 92                           # EnterTransport (a bunker)
NEAR = 7 * 32                        # a zerg fighter this close: the marine is fighting
WINDOWS = (0.5, 1.0, 2.0)            # seconds
FPS = 1000 / 42


class Probe:
    """Follows every marine of the Terran through one scenario (on_step for run_one)."""

    def __init__(self):
        self.time = {"attack": 0, "move": 0, "stand": 0, "else": 0}   # frames
        self.moves: list[dict] = []           # one per move in a fight: frame, then resumed / died
        self.open: dict[int, dict] = {}       # marine tag -> its move waiting to be resolved
        self.last: dict[int, int] = {}        # marine tag -> its order last step
        self.deaths = {"move": 0, "attack": 0, "other": 0}
        self.prev_frame = None

    def __call__(self, game, slot: int) -> None:
        from gary.policy.army import is_building, is_worker, supply_x2
        obs = game.observe()
        f = obs["frame"]
        dt = f - self.prev_frame if self.prev_frame is not None else 0
        self.prev_frame = f
        marines = {u["tag"]: u for u in obs["units"] if u["owner"] == slot and u["type"] == MARINE and u["completed"]}
        zerg = [u for u in obs["units"] if u["owner"] not in (slot, 11) and supply_x2(u["type"]) > 0
                and not is_worker(u["type"]) and not is_building(u["type"])]
        for t, last in list(self.last.items()):
            if t in marines:
                continue
            del self.last[t]
            if last == ENTER:                 # into a bunker, not dead
                self.open.pop(t, None)
                continue
            self.deaths["move" if last in MOVE else "attack" if last in ATTACK else "other"] += 1
            if t in self.open:
                self.open.pop(t)["died"] = True
        for t, u in marines.items():
            o, before = u["order"], self.last.get(t)
            self.last[t] = o
            fighting = any(math.dist((u["x"], u["y"]), (z["x"], z["y"])) < NEAR for z in zerg)
            if fighting:
                kind = "attack" if o in ATTACK else "move" if o in MOVE else "stand" if o in STAND else "else"
                self.time[kind] += dt
            if t in self.open and o in ATTACK:
                m = self.open.pop(t)
                m["resumed"] = (f - m["frame"]) / FPS
            if o in MOVE and before is not None and before not in MOVE and fighting:
                m = {"frame": f}
                self.moves.append(m)
                self.open[t] = m

    def result(self) -> dict:
        return {"time": self.time, "moves": [{k: v for k, v in m.items() if k != "frame"} for m in self.moves],
                "deaths": self.deaths}


def _job(args: tuple) -> dict:
    from eval.scenarios import run_one
    sc, controller, version, style = args
    probe = Probe()
    row = run_one(sc, controller, 45, version, style, on_step=probe)
    return {"sha1": row["sha1"], "controller": controller, "error": row.get("error"), **probe.result()}


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r["error"]]
    time = {k: sum(r["time"][k] for r in ok) for k in ("attack", "move", "stand", "else")}
    total = sum(time.values()) or 1
    moves = [m for r in ok for m in r["moves"]]
    resumed = [m["resumed"] for m in moves if "resumed" in m]
    n = len(moves) or 1
    deaths = {k: sum(r["deaths"][k] for r in ok) for k in ("move", "attack", "other")}
    return {"scenarios": len(ok),
            "time": {k: round(v / total, 3) for k, v in time.items()},
            "fighting_minutes": round(total / FPS / 60, 1),
            "moves_per_marine_minute": round(len(moves) / (total / FPS / 60), 1),
            "resume_median_s": round(float(np.median(resumed)), 2) if resumed else None,
            "resumed_within": {f"{w}s": round(sum(1 for r in resumed if r <= w) / n, 3) for w in WINDOWS},
            "died_before_attacking": round(sum(1 for m in moves if m.get("died")) / n, 3),
            "deaths": deaths}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=int, default=114)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--controllers", nargs="+", default=["pro", "gary"], choices=["pro", "gary", "nothing"])
    ap.add_argument("--version", default="v09")
    ap.add_argument("--style", type=int, default=1)
    ap.add_argument("--parallel", type=int, default=16)
    ap.add_argument("--out", help="also save the per-scenario rows here (JSON lines)")
    args = ap.parse_args()
    from eval.scenarios import scenarios_path
    scs = [json.loads(line) for line in scenarios_path(args.split).read_text(encoding="utf-8").splitlines()][:args.run]
    jobs = [(sc, c, args.version, args.style) for sc in scs for c in args.controllers]
    with ProcessPoolExecutor(args.parallel) as pool:
        rows = list(pool.map(_job, jobs, chunksize=len(args.controllers)))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    for c in args.controllers:
        name = f"gary {args.version}" if c == "gary" else c
        print(name, json.dumps(summarize([r for r in rows if r["controller"] == c])))


if __name__ == "__main__":
    main()
