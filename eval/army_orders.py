"""What the army model's orders do: games played again (deterministic per seed) with every order the
army model gives (ARMY_MOVE / ARMY_ATTACK from the army layer; not the hunts) recorded, and the 45 s
after it read from the full game state (not Gary's view).

Per order: where it sends the group (Gary's main, its ramp, its natural, the field, the enemy's side:
eval/losses.py places), attack or move, the group's supply and its share of Gary's army, the model's
confidence and fight estimate, the units really ordered (an order can reach part of the group, or
none), and in the next WINDOW: the group's own supply lost and the Zerg army supply lost within NEAR of the group's units.
Summaries by destination, by share of the army, by fight estimate; and orders in lost games against
the rest.

    python -m eval.army_orders --games v2:standard_ozp3w:101-124 v3:python_gg13676:101-116
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics as st
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from eval import monitor, priority
from eval.losses import MAPS, places

REPO_ROOT = Path(__file__).resolve().parent.parent
VERSION = "v025"
WINDOW = 45 * 24
NEAR = 10 * 32                   # a zerg unit dying this close to the group's units: the group's doing
MINUTES = {"v3": 25}
ARMY = {0, 1, 2, 3, 5, 30, 32, 34, 8, 11, 12, 58}
ZERG_ARMY = {37, 38, 39, 43, 44, 45, 46, 47, 62, 103}


def play(job: tuple) -> dict:
    opponent, mp, seed = job
    from eval.walls_bench import Match
    from gary.policy.army import supply_x2
    with contextlib.redirect_stdout(io.StringIO()):
        m = Match(seed, 0, VERSION, MAPS[mp], opponent)
    where, _ = places(m)
    g = m.gary
    orders, open_ = [], []
    zerg_at: dict[int, tuple] = {}               # zerg army unit -> (x, y, supply x2) last seen
    start = g.motor.start

    def watched(intent):
        if intent.kind in ("ARMY_MOVE", "ARMY_ATTACK") and intent.source == "army" and not intent.params.get("hunt"):
            open_.append({"intent": intent, "frame": g.hi.frame})
        return start(intent)
    g.motor.start = watched
    end = MINUTES.get(opponent, 12) * 1440
    quiet = contextlib.redirect_stdout(io.StringIO())
    quiet.__enter__()
    while m.game.frame < end and not m.done:
        m.step()
        if not open_:
            zerg_at.clear()
            continue
        f = m.game.frame
        units = m.game.observe()["units"]
        alive = {u["tag"]: u for u in units}
        dead = [v for t, v in zerg_at.items() if t not in alive]
        zerg_at = {u["tag"]: (u["x"], u["y"], supply_x2(u["type"])) for u in units
                   if u["owner"] == m.slot["Z"] and u["type"] in ZERG_ARMY}
        for o in open_:
            if "t0" in o and dead:
                near = [alive[t] for t in o["group"] if t in alive]
                o["killed"] = o.get("killed", 0) + sum(sp for x, y, sp in dead
                                                       if any(abs(u["x"] - x) < NEAR and abs(u["y"] - y) < NEAR for u in near)) / 2
        for o in open_:
            if "t0" not in o:
                tags = set(o["intent"].params["tags"])
                army = [u for u in units if u["owner"] == m.slot["T"] and u["type"] in ARMY and u["completed"]]
                group = [u for u in army if u["tag"] in tags]
                o["t0"] = f
                o["group"] = {u["tag"]: supply_x2(u["type"]) for u in group}
                o["army_supply"] = sum(supply_x2(u["type"]) for u in army) / 2
                p = o["intent"].params
                o["row"] = {"minute": round(f / 1440, 2), "to": where(p["x"], p["y"]), "attack": bool(p["attack"]),
                            "group": sum(o["group"].values()) / 2, "share": sum(o["group"].values()) / 2 / max(1, o["army_supply"]),
                            "p": p.get("p"), "fight": p.get("fight"),
                            "from": where(sum(u["x"] for u in group) / max(1, len(group)), sum(u["y"] for u in group) / max(1, len(group)))}
        for o in [o for o in open_ if f - o["t0"] >= WINDOW]:
            lost = sum(s for t, s in o["group"].items() if t not in alive) / 2
            killed = o.get("killed", 0)
            r = o["row"]
            r.update(ordered=len(set(o["intent"].params.get("done", []))) / max(1, len(o["group"])),
                     lost=lost, lost_share=lost / max(0.5, r["group"]), killed=killed)
            orders.append(r)
            open_.remove(o)
    quiet.__exit__(None, None, None)
    players = {p["slot"] for p in m.game.observe()["players"]}
    result = "won" if m.slot["Z"] not in players else "lost" if m.slot["T"] not in players else "time"
    m.game.close()
    return {"game": f"{opponent}/{mp}/{seed}", "result": result, "orders": orders}


def report(games: list[dict]) -> None:
    rows = [dict(o, result=g["result"], opp=g["game"].split("/")[0]) for g in games for o in g["orders"]]
    print(f"{len(games)} games ({', '.join(f'{k} {v}' for k, v in sorted(_count(g['result'] for g in games).items()))}), "
          f"{len(rows)} army orders ({len(rows) / max(1, len(games)):.0f} a game)")

    def line(label, rs):
        if not rs:
            return
        print(f"  {label:34s} {len(rs):5d}  group {st.mean(r['group'] for r in rs):5.1f} sup ({st.mean(r['share'] for r in rs):4.0%}), "
              f"ordered {st.mean(r['ordered'] for r in rs):4.0%} | 45 s: lost {st.mean(r['lost'] for r in rs):4.1f} "
              f"({st.mean(r['lost_share'] for r in rs):4.0%} of the group), zerg lost by it {st.mean(r['killed'] for r in rs):4.1f}, "
              f"traded {sum(r['killed'] for r in rs) / max(0.5, sum(r['lost'] for r in rs)):4.2f}")
    print("                                     orders  (supply; the share of Gary's army)          (traded: zerg supply lost per Gary's)")
    for key, label in (("to", "to"), ("attack", "attack")):
        print(f" by {label}:")
        groups = defaultdict(list)
        for r in rows:
            groups[r[key]].append(r)
        for k, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            line(f"{k}", rs)
    print(" by the group's share of the army:")
    for lo, hi in ((0, 0.25), (0.25, 0.6), (0.6, 1.01)):
        line(f"{lo:.0%}-{min(hi, 1):.0%}", [r for r in rows if lo <= r["share"] < hi])
    print(" by the fight estimate:")
    for lo, hi in ((-9, -0.02), (-0.02, 0.02), (0.02, 9)):
        line(f"{lo:+.2f} to {hi:+.2f}", [r for r in rows if r["fight"] is not None and lo <= r["fight"] < hi])
    print(" to the field or the enemy's side, by the game's result:")
    for res in ("lost", "time", "won"):
        line(res, [r for r in rows if r["result"] == res and r["to"] in ("field", "enemy side")])
    print(" by opponent (to the field or the enemy's side):")
    for opp in sorted({r["opp"] for r in rows}):
        line(opp, [r for r in rows if r["opp"] == opp and r["to"] in ("field", "enemy side")])


def _count(xs):
    out = defaultdict(int)
    for x in xs:
        out[x] += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", nargs="+", required=True, help="opponent:map:first-last")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--out", default=str(REPO_ROOT / "runs" / "army_orders.jsonl"))
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args)
    jobs = []
    for spec in args.games:
        opp, mp, seeds = spec.split(":")
        a, b = (int(x) for x in seeds.split("-"))
        jobs += [(opp, mp, s) for s in range(a, b + 1)]
    games = []
    with ProcessPoolExecutor(args.parallel) as ex:
        futs = [ex.submit(play, j) for j in jobs]
        for fu in monitor.track(as_completed(futs), len(futs), "games"):
            games.append(fu.result())
            monitor.tally("result", games[-1]["result"])
    Path(args.out).write_text("\n".join(json.dumps(g) for g in games), encoding="utf-8")
    report(games)


if __name__ == "__main__":
    main()
