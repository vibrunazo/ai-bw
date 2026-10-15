"""Why the games against Zerg v3's lurker plan don't end: those games played again (deterministic per
seed) to 25 minutes, with the full game state (not Gary's view): every death of Gary's army with what
stood within LURKER_REACH of it (burrowed lurkers, lurkers up, other zerg fighters, nothing), the
lurkers alive and the lurkers killed, whether Gary saw the lurkers that stood by its deaths (a burrowed
lurker undetected isn't visible to it), the scans, and every minute both armies' supply, Gary's bank,
its army built and where its army stands.

    python -m eval.lurker_games --seeds 103 106 109 --maps standard_ozp3w python_gg13676
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

from eval import monitor, priority
from eval.losses import ARMY, MAPS, ZERG_FIGHTERS, places

VERSION = "v025"
MINUTES = 25                     # (Zerg v3 games run to 25 minutes from v0.25)
LURKER = 103
LURKER_REACH = 7 * 32            # a lurker's spines reach 6 tiles
EVERY = 12


def play(args: tuple) -> dict:
    mp, seed = args
    from eval.walls_bench import Match
    from gary.policy.army import supply_x2
    with contextlib.redirect_stdout(io.StringIO()):
        m = Match(seed, 0, VERSION, MAPS[mp], "v3")
    where, _ = places(m)
    m.gary.verbose = True                        # (its words, to count its scans)
    last: dict[int, dict] = {}
    lurkers_seen: dict[int, bool] = {}
    deaths, timeline, built = [], [], set()
    lurkers_dead = 0
    bit = 1 << m.slot["T"]
    said = io.StringIO()
    quiet = contextlib.redirect_stdout(said)
    quiet.__enter__()
    while m.game.frame < MINUTES * 1440 and not m.done:
        m.step()
        f = m.game.frame
        if f % EVERY >= 2:
            continue
        o = m.game.observe()
        mine = [u for u in o["units"] if u["owner"] == m.slot["T"]]
        army = [u for u in mine if u["type"] in ARMY and u["completed"]]
        zerg = [u for u in o["units"] if u["owner"] == m.slot["Z"] and u["type"] in ZERG_FIGHTERS]
        lurkers = [u for u in zerg if u["type"] == LURKER and u["completed"]]
        here = {u["tag"] for u in o["units"]}
        for tag in list(lurkers_seen):
            if tag not in here and m.game.unit_type_of(tag) == -1:
                lurkers_dead += 1
                del lurkers_seen[tag]
        for u in lurkers:
            lurkers_seen[u["tag"]] = True
        for tag, s in list(last.items()):
            if tag not in here:
                if m.game.unit_type_of(tag) == -1:
                    deaths.append({**s, "frame": f})
                del last[tag]
        for u in army:
            built.add(u["tag"])
            p = (u["x"], u["y"])
            near = [z for z in zerg if math.dist(p, (z["x"], z["y"])) < LURKER_REACH]
            lk = [z for z in near if z["type"] == LURKER]
            what = ("lurker seen" if any(z["visible_to"] & bit for z in lk) else "lurker unseen" if lk
                    else "other zerg" if near else "nothing")
            last[u["tag"]] = {"type": u["type"], "place": where(*p), "near": what}
        if f % 1440 < 2:
            me = next(p for p in o["players"] if p["slot"] == m.slot["T"])
            timeline.append({"minute": round(f / 1440),
                             "army": sum(supply_x2(u["type"]) for u in army) / 2,
                             "zerg": sum(supply_x2(u["type"]) for u in zerg) / 2,
                             "lurkers": len(lurkers), "unseen": sum(1 for u in lurkers if not u["visible_to"] & bit),
                             "built": len(built), "minerals": me["minerals"], "gas": me["gas"],
                             "bases": sum(1 for u in mine if u["type"] == 106),
                             "army_at": Counter(where(u["x"], u["y"]) for u in army)})
    quiet.__exit__(None, None, None)
    players = {p["slot"] for p in m.game.observe()["players"]}
    result = "won" if m.slot["Z"] not in players else "lost" if m.slot["T"] not in players else "time"
    m.game.close()
    return {"game": f"{mp}/{seed}", "result": result, "deaths": deaths, "timeline": timeline,
            "lurkers_dead": lurkers_dead, "scans": said.getvalue().count("Gary scans burrowed")}


def report(games: list[dict]) -> None:
    print(f"{len(games)} games: " + ", ".join(f"{k} {v}" for k, v in Counter(g["result"] for g in games).items()))
    deaths = [d for g in games for d in g["deaths"]]
    n = max(1, len(deaths))
    print(f"Gary's army deaths, {len(deaths) / len(games):.0f} a game; what stood within 7 tiles: " +
          ", ".join(f"{k} {v / n:.0%}" for k, v in Counter(d["near"] for d in deaths).most_common()))
    print("  by place: " + ", ".join(f"{k} {v / n:.0%}" for k, v in Counter(d["place"] for d in deaths).most_common()))
    print(f"lurkers killed, a game: {sum(g['lurkers_dead'] for g in games) / len(games):.1f}; "
          f"scans, a game: {sum(g['scans'] for g in games) / len(games):.1f}")
    print("medians    minute " + " ".join(f"{t:>5d}" for t in range(6, MINUTES + 1, 2)))
    for key in ("army", "zerg", "lurkers", "unseen", "built", "minerals", "gas", "bases"):
        cells = []
        for t in range(6, MINUTES + 1, 2):
            v = sorted(r[key] for g in games for r in g["timeline"] if r["minute"] == t)
            cells.append(f"{v[len(v) // 2]:>5.0f}" if v else "    -")
        print(f"  {key:16s} " + " ".join(cells))
    for t in (12, 16, 20):
        c = Counter()
        for g in games:
            for r in g["timeline"]:
                if r["minute"] == t:
                    tot = sum(r["army_at"].values()) or 1
                    c.update({k: v / tot / len(games) for k, v in r["army_at"].items()})
        print(f"  army at {t}:00: " + ", ".join(f"{k} {v:.0%}" for k, v in c.most_common()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(103, 149, 3)))
    ap.add_argument("--maps", nargs="+", default=["standard_ozp3w", "python_gg13676"])
    ap.add_argument("--parallel", type=int, default=4)
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args)
    jobs = [(mp, s) for mp in args.maps for s in args.seeds]
    games = []
    with ProcessPoolExecutor(args.parallel) as ex:
        futs = [ex.submit(play, j) for j in jobs]
        for fu in monitor.track(as_completed(futs), len(futs), "lurker games"):
            games.append(fu.result())
    report(games)


if __name__ == "__main__":
    main()
