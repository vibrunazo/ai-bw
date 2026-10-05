"""What marines' moves in a fight are for: the pro vs Gary (#2, docs/CONTROL.md: the intent vocabulary).

On the home defenses of eval/scenarios.py, under each controller, every move order fighting
marines get (a zerg fighter within 7 tiles) is one event: the marines whose order turns to "move"
in the same quarter second (one command). After it plays out (2.5 s), it is labeled:

  approach   the moved marines go toward the zerg
  stutter    most of the fighting marines move (half or more), a short way (under 2.5 tiles), and
             shoot again within a second
  pull_back  a minority moves, mostly the ones being attacked or nearest the zerg, away from them,
             while most of the others keep shooting
  kite       away from the zerg, by units that moved away before in the last 2.5 s and shot in
             between (a repeated cycle)
  retreat    away, a long way (4+ tiles) or without shooting again within 2 s
  other      the rest

Orders, not commands, so the pro (replayed) and Gary are measured the same way. Labels are a
script's, a first provider for the intent_label slot (ARCHITECTURE.md 6.6): check them against
the replays before trusting them.

    python -m eval.micro_intents --run 114 --parallel 16
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

MARINE = 0
ATTACK = {10, 12, 14}
MOVE = {6}
NEAR = 7 * 32
SAME_COMMAND = 6                     # frames: units whose move starts this close are one command
SETTLE = 60                          # frames (2.5 s) after which an event is labeled
FPS = 1000 / 42


class Probe:
    def __init__(self):
        self.last: dict[int, int] = {}            # marine -> its order last step
        self.events: list[dict] = []              # open, then labeled
        self.pos: dict[int, list] = {}            # marine -> [(frame, x, y, order)]
        self.away_moves: dict[int, int] = {}      # marine -> frame of its last move away
        self.labels: Counter = Counter()
        self.shares: dict[str, list[float]] = {}

    def __call__(self, game, slot: int) -> None:
        from gary.policy.army import is_building, is_worker, supply_x2
        obs = game.observe()
        f = obs["frame"]
        marines = {u["tag"]: u for u in obs["units"] if u["owner"] == slot and u["type"] == MARINE and u["completed"]}
        zerg = [u for u in obs["units"] if u["owner"] not in (slot, 11) and supply_x2(u["type"]) > 0
                and not is_worker(u["type"]) and not is_building(u["type"])]
        targeted = {u.get("order_target") for u in zerg}
        near = {t: min((math.dist((u["x"], u["y"]), (z["x"], z["y"])) for z in zerg), default=1e9)
                for t, u in marines.items()}
        fighting = {t for t, d in near.items() if d < NEAR}
        for t, u in marines.items():
            self.pos.setdefault(t, []).append((f, u["x"], u["y"], u["order"]))
        started = [t for t, u in marines.items() if u["order"] in MOVE and self.last.get(t) not in MOVE
                   and self.last.get(t) is not None and t in fighting]
        for t, u in marines.items():
            self.last[t] = u["order"]
        if started:
            ev = self.events[-1] if self.events else None
            if ev and not ev.get("label") and f - ev["frame"] <= SAME_COMMAND:
                ev["moved"] |= set(started)
            else:
                dists = sorted(near[t] for t in fighting)
                front = {t for t in fighting if near[t] <= dists[max(0, len(dists) // 3 - 1)]} if dists else set()
                zx = np.mean([z["x"] for z in zerg]) if zerg else 0
                zy = np.mean([z["y"] for z in zerg]) if zerg else 0
                self.events.append({"frame": f, "moved": set(started), "fighting": set(fighting),
                                    "pressed": (targeted | front) & fighting, "zerg": (zx, zy),
                                    "start": {t: (marines[t]["x"], marines[t]["y"]) for t in fighting}})
        for ev in self.events:
            if "label" not in ev and f - ev["frame"] >= SETTLE:
                self._label(ev, marines)

    def finish(self) -> None:
        for ev in self.events:
            if "label" not in ev:
                self._label(ev, {})

    def _label(self, ev: dict, marines: dict) -> None:
        moved, fighting = ev["moved"], ev["fighting"]
        share = len(moved & fighting) / max(1, len(fighting))
        zx, zy = ev["zerg"]
        away, dist, resumed, others_shoot = [], [], [], []
        for t in fighting:
            track = [p for p in self.pos.get(t, []) if ev["frame"] <= p[0] <= ev["frame"] + SETTLE]
            if not track:
                continue
            sx, sy = ev["start"][t]
            if t in moved:
                # where it went while moving, and when it shot again
                end = next((p for p in track if p[3] not in MOVE), track[-1])
                d = math.dist((sx, sy), (end[1], end[2]))
                dist.append(d / 32)
                away.append(math.dist((end[1], end[2]), (zx, zy)) > math.dist((sx, sy), (zx, zy)))
                shot = next((p[0] for p in track if p[3] in ATTACK), None)
                resumed.append((shot - ev["frame"]) / FPS if shot is not None else None)
            else:
                others_shoot.append(sum(1 for p in track[:12] if p[3] in ATTACK) >= len(track[:12]) / 2)
        n = len(away)
        if not n:
            ev["label"] = "other"
            return
        away_share = sum(away) / n
        quick = [r for r in resumed if r is not None and r <= 1.0]
        median_dist = float(np.median(dist))
        pressed = len(moved & ev["pressed"]) / max(1, len(moved))
        cycled = sum(1 for t in moved if ev["frame"] - self.away_moves.get(t, -10**9) <= SETTLE) / max(1, len(moved))
        if away_share < 0.5:
            label = "approach"
        elif cycled >= 0.5:
            label = "kite"
        elif share >= 0.5 and median_dist < 2.5 and len(quick) >= n / 2:
            label = "stutter"
        elif share < 0.5 and pressed >= 0.5 and (not others_shoot or sum(others_shoot) >= len(others_shoot) / 2):
            label = "pull_back"
        elif median_dist >= 4 or sum(1 for r in resumed if r is not None and r <= 2.0) < n / 2:
            label = "retreat"
        else:
            label = "other"
        if away_share >= 0.5:
            for t in moved:
                self.away_moves[t] = ev["frame"]
        ev["label"] = label
        self.labels[label] += 1
        self.shares.setdefault(label, []).append(share)

    def result(self) -> dict:
        self.finish()
        return {"labels": dict(self.labels), "shares": {k: v for k, v in self.shares.items()}}


def _job(args: tuple) -> dict:
    from eval.scenarios import run_one
    sc, controller, version, style = args
    probe = Probe()
    row = run_one(sc, controller, 45, version, style, on_step=probe)
    return {"sha1": row["sha1"], "controller": controller, "error": row.get("error"), **probe.result()}


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r["error"]]
    labels: Counter = Counter()
    shares: dict[str, list[float]] = {}
    for r in ok:
        labels.update(r["labels"])
        for k, v in r["shares"].items():
            shares.setdefault(k, []).extend(v)
    total = sum(labels.values()) or 1
    return {"scenarios": len(ok), "moves": total,
            "labels": {k: {"n": n, "share": round(n / total, 3),
                           "group_moved": round(float(np.median(shares[k])), 2)}
                       for k, n in labels.most_common()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=int, default=114)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--controllers", nargs="+", default=["pro", "gary"], choices=["pro", "gary"])
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
        s = summarize([r for r in rows if r["controller"] == c])
        print(f"{name}: {s['moves']} move commands in {s['scenarios']} scenarios")
        for k, v in s["labels"].items():
            print(f"  {k:10s} {v['n']:5d} {v['share']:6.1%}   median share of fighting marines moved {v['group_moved']:.2f}")


if __name__ == "__main__":
    main()
