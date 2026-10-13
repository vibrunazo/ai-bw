"""Why Gary loses: its lost games played again (deterministic per seed), every army death recorded
with where it happened, what the unit was doing and which of Gary's layers last ordered it; and the
armies over time. Games of the same sets it didn't lose are the contrast.

Per death: the place (Gary's main plateau, its main's ramp, its natural, the field between, the
enemy's side of the map), the unit's order, the last lease on it (gary/agent/intents.py Leases:
"army" moves, "roles"/"fight" decisions, "rule:..." reflexes, and the intent's kind), its own
fighters and zerg fighters within 6 tiles. Per game, every 30 s: both armies' supply alive, Gary's
army built so far, its bases, the posture's mode, and where Gary's army stands (the share of it
in each place). The full game state (not Gary's view): this explains, it isn't Gary's input.

    python -m eval.losses --games v2:python_gg13676:118 replayed:heartbreak_9rg84:104 ...
    python -m eval.losses --from-logs runs/sel                      # every game in vs_rush folders
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import math
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from eval import monitor, priority

REPO_ROOT = Path(__file__).resolve().parent.parent
MAPS = {m: REPO_ROOT / "tests" / "fixtures" / "replays" / f"stardata_tvz_{m}.rep"
        for m in ("python_gg13676", "standard_ozp3w", "heartbreak_9rg84")}
VERSION = "v019"
MINUTES = 12
ARMY = {0, 1, 2, 3, 5, 30, 32, 34, 8, 11, 12, 58}
ZERG_FIGHTERS = {37, 38, 39, 43, 103, 46, 47, 45, 44, 62}
NEAR = 6 * 32
EVERY = 12                       # frames between looks
TIMELINE = 30 * 24               # frames between timeline rows
NATURAL_R = 12 * 32
RAMP_R = 4 * 32


def places(m) -> tuple:
    """(where(x, y) -> place, main start) for Gary in this match."""
    from gary.agent.block import _analysis
    from gary.policy.roles import Ground
    g = m.gary
    start = min(g.map.starts, key=lambda s: math.dist(s, g.main.center))
    enemy = [s for s in g.map.starts if s != start]
    ground = Ground.of(m.game, start)
    a = _analysis(g)
    top, bottom, natural = tuple(a["ramp_top"]), tuple(a["ramp_bottom"]), tuple(a["natural"])

    def where(x, y) -> str:
        if min(math.dist((x, y), top), math.dist((x, y), bottom)) < RAMP_R:
            return "ramp"
        if ground.main[ground.cell(x, y)]:
            return "main"
        if math.dist((x, y), natural) < NATURAL_R:
            return "natural"
        if enemy and min(math.dist((x, y), e) for e in enemy) < math.dist((x, y), start):
            return "enemy side"
        return "field"
    return where, start


def play(args: tuple) -> dict:
    opponent, mp, seed, result_before = args
    from eval.walls_bench import Match
    from gary.policy.army import supply_x2
    with contextlib.redirect_stdout(io.StringIO()):
        m = Match(seed, 0, VERSION, MAPS[mp], opponent)
    try:
        where, start = places(m)
    except Exception as e:                       # (no analysis for this map)
        m.game.close()
        return {"game": f"{opponent}/{mp}/{seed}", "error": str(e)}
    g = m.gary
    last: dict[int, dict] = {}
    deaths, timeline, built = [], [], set()
    end = MINUTES * 60 * 24
    quiet = contextlib.redirect_stdout(io.StringIO())       # (the zerg bots' prints)
    quiet.__enter__()
    while m.game.frame < end and not m.done:
        m.step()
        f = m.game.frame
        if f % EVERY >= 2:
            continue
        o = m.game.observe()
        mine = [u for u in o["units"] if u["owner"] == m.slot["T"]]
        army = [u for u in mine if u["type"] in ARMY and u["completed"]]
        zerg = [u for u in o["units"] if u["owner"] == m.slot["Z"] and u["type"] in ZERG_FIGHTERS]
        here = {u["tag"] for u in o["units"]}
        for tag, s in list(last.items()):
            if tag not in here:
                if m.game.unit_type_of(tag) == -1:     # dead (not inside a bunker)
                    deaths.append({**s, "frame": f})
                del last[tag]
        held = g.leases.held if g.leases is not None else {}
        for u in army:
            built.add(u["tag"])
            p = (u["x"], u["y"])
            lease = held.get(u["tag"])
            last[u["tag"]] = {"type": u["type"], "place": where(*p), "order": u.get("order"),
                              "lease": f"{lease[0]} {lease[1]}" if lease else "none",
                              "own": sum(1 for v in army if math.dist(p, (v["x"], v["y"])) < NEAR) - 1,
                              "zerg": sum(1 for z in zerg if math.dist(p, (z["x"], z["y"])) < NEAR)}
        if f % TIMELINE < 2:
            spots = Counter(where(u["x"], u["y"]) for u in army)
            timeline.append({"minute": round(f / 1440, 1),
                             "army": sum(supply_x2(u["type"]) for u in army) / 2,
                             "zerg": sum(supply_x2(u["type"]) for u in zerg) / 2,
                             "built": sum(1 for _ in built), "bases": sum(1 for u in mine if u["type"] == 106),
                             "mode": getattr(g.posture, "mode", None) or ("rush" if getattr(g.posture, "rush", False)
                                                                           else "hold" if getattr(g.posture, "hold", False)
                                                                           else "normal"),
                             "army_at": {k: round(v / max(1, len(army)), 2) for k, v in spots.items()}})
    quiet.__exit__(None, None, None)
    players = {p["slot"] for p in m.game.observe()["players"]}
    result = "won" if m.slot["T"] in players and m.slot["Z"] not in players else \
        "lost" if m.slot["T"] not in players else "time"
    out = {"game": f"{opponent}/{mp}/{seed}", "result": result, "before": result_before,
           "end_minute": round(m.game.frame / 1440, 2), "deaths": deaths, "timeline": timeline}
    m.game.close()
    return out


def from_logs(folder: str) -> list[tuple]:
    """(opponent, map, seed, result) of every v018+selector / v019 game in vs_rush folders named
    <opponent>_<map> under folder."""
    out = []
    for d in sorted(Path(folder).iterdir()):
        mm = re.match(r"(dummy|v2|replayed)_(.+)$", d.name)
        if not d.is_dir() or not mm or mm.group(2) not in MAPS:
            continue
        for f in glob.glob(str(d / "*seed*.log")):
            if not re.search(r"(v019|v018\+selector)_seed", f):
                continue
            seed = int(re.search(r"seed(\d+)", f).group(1))
            txt = Path(f).read_text(encoding="utf-8", errors="replace")
            res = "won" if "Gary won" in txt else "lost" if "Gary lost" in txt else "time"
            out.append((mm.group(1), mm.group(2), seed, res))
    return out


def report(games: list[dict]) -> None:
    ok = [g for g in games if not g.get("error")]
    lost = [g for g in ok if g["result"] == "lost"]
    rest = [g for g in ok if g["result"] != "lost"]
    same = sum(1 for g in ok if g["before"] is None or g["result"] == g["before"])
    print(f"{len(ok)} games played again ({len(games) - len(ok)} errors): lost {len(lost)}, not lost {len(rest)}; "
          f"the same result as before in {same} of {len(ok)}")
    if lost:
        ends = sorted(g["end_minute"] for g in lost)
        print(f"  lost at minute: median {ends[len(ends) // 2]:.1f} ({', '.join(f'{e:.1f}' for e in ends)})")

    def share(gs, key, top=8):
        c = Counter(key(d) for g in gs for d in g["deaths"])
        n = sum(c.values()) or 1
        return n, ", ".join(f"{k} {v / n:.0%}" for k, v in c.most_common(top))
    for name, key in (("place", lambda d: d["place"]), ("last ordered by", lambda d: d["lease"]),
                      ("alone (no own fighter within 6 tiles)", lambda d: d["own"] == 0),
                      ("outnumbered 2:1+", lambda d: d["zerg"] >= 2 * (d["own"] + 1))):
        for label, gs in (("lost", lost), ("not lost", rest)):
            n, s = share(gs, key)
            print(f"  army deaths by {name} [{label}, {n / max(1, len(gs)):.1f} a game]: {s}")

    def at(gs, minute, key):
        v = [next((r[key] for r in g["timeline"] if r["minute"] >= minute), None) for g in gs]
        v = sorted(x for x in v if x is not None)
        return f"{v[len(v) // 2]:.0f}" if v else "-"
    print("  medians over time        " + "  ".join(f"{t}:00" for t in (3, 4, 5, 6, 7, 8, 10)))
    for label, gs in (("lost", lost), ("not lost", rest)):
        for key in ("army", "zerg", "built", "bases"):
            print(f"    {label:9s} {key:7s}       " + "  ".join(f"{at(gs, t, key):>4s}" for t in (3, 4, 5, 6, 7, 8, 10)))
    # where Gary's army stood, on average, in the minute before each loss
    if lost:
        c = Counter()
        for g in lost:
            rows = [r for r in g["timeline"] if g["end_minute"] - 2.5 <= r["minute"] <= g["end_minute"] - 0.5]
            for r in rows:
                for k, v in r["army_at"].items():
                    c[k] += v / max(1, len(rows))
        n = sum(c.values()) or 1
        print("  lost games, where Gary's army stood 2.5 to 0.5 minutes before the end: " + ", ".join(f"{k} {v / n:.0%}" for k, v in c.most_common()))
        modes = Counter(r["mode"] for g in lost for r in g["timeline"])
        print("  lost games, the posture's modes (30 s rows): " + ", ".join(f"{k} {v}" for k, v in modes.most_common()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", nargs="*", default=[], help="opponent:map:seed")
    ap.add_argument("--from-logs", help="every v019 game in the vs_rush folders under this one")
    ap.add_argument("--out", default=str(REPO_ROOT / "runs" / "losses.jsonl"))
    ap.add_argument("--parallel", type=int, default=14)
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    jobs = [(*(x := s.split(":"))[:2], int(x[2]), None) for s in args.games]
    if args.from_logs:
        jobs += from_logs(args.from_logs)
    with ProcessPoolExecutor(args.parallel) as pool:
        games = list(monitor.track(pool.map(play, jobs), len(jobs), "games"))
    Path(args.out).write_text("".join(json.dumps(g) + "\n" for g in games), encoding="utf-8")
    print(f"-> {args.out}")
    report(games)


if __name__ == "__main__":
    main()
