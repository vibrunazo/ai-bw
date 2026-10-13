"""Block or not: moments of Gary's own games, played on twice, without and with a choke block.

The data for the decision to close a choke with bodies (gary/agent/block.py): Gary (--version, with
fight.py BlockRule, which nothing asks yet) plays its usual opponents; every moment the enemy's
fighters come near its main's ramp (DECISION_EVERY apart, at most MAX_POINTS a game) is a decision
point, with what Gary sees then (block.features()). The game played on is the "no block" branch.
For each point the same game is played again from the start (games are deterministic per seed:
everything up to the point is the same, Gary's memory included) and asked to close the ramp there,
for BLOCK_FOR seconds. Both branches are scored over the HORIZON seconds after the point as the
home defenses are (eval/score.py: value killed minus lost), plus Gary's workers and buildings lost.

    python -m eval.block_branch --seeds 24                 # -> data/interim/blocks/branches.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from eval import monitor, priority

REPO_ROOT = Path(__file__).resolve().parent.parent
MAPS = {m: REPO_ROOT / "tests" / "fixtures" / "replays" / f"stardata_tvz_{m}.rep"
        for m in ("python_gg13676", "standard_ozp3w", "heartbreak_9rg84")}
OPPONENTS = ("dummy", "v2", "replayed")
VERSION = "v018+block"
MINUTES = 10
DECISION_EVERY = 10 * 24
MAX_POINTS = 4
HORIZON = 90 * 24
BLOCK_FOR = 60 * 24
NEAR = 12 * 32                        # enemy fighters this close to the ramp's top: a decision point


def out_path() -> Path:
    import sys
    sys.path.insert(0, str(REPO_ROOT / "ingest"))
    from inventory import data_root
    return data_root() / "interim" / "blocks" / "branches.jsonl"


def _match(opponent: str, mp: str, seed: int, version: str = VERSION):
    from eval.walls_bench import Match
    with contextlib.redirect_stdout(io.StringIO()):
        return Match(seed, 0, version, MAPS[mp], opponent)


class Watch:
    """The score as the game goes (eval/score.py), and Gary's workers and buildings, every 12 frames."""

    def __init__(self, m):
        from eval.score import Score
        self.m = m
        self.score = Score(m.slot["T"], m.slot["Z"])
        self.marks: list[tuple[int, int, int]] = []          # (frame, workers, buildings)

    def look(self) -> None:
        from gary.policy.army import is_building
        o = self.m.game.observe()
        self.score.note(o, self.m.game)
        mine = [u for u in o["units"] if u["owner"] == self.m.slot["T"]]
        self.marks.append((o["frame"], sum(1 for u in mine if u["type"] == 7), sum(1 for u in mine if is_building(u["type"]))))

    def window(self, start: int, end: int) -> dict:
        net = sum(v for f, v in self.score.events if start <= f < end)
        at = lambda fr: min(self.marks, key=lambda m: abs(m[0] - fr))
        a, b = at(start), at(end)
        return {"net": round(net), "workers_lost": a[1] - b[1], "buildings_lost": a[2] - b[2]}


def _step(m, w: Watch) -> bool:
    if m.game.frame % 240 < 2 and m.over():
        return False
    m.step()
    if m.done:
        return False
    if m.game.frame % 12 < 2:
        w.look()
    return True


def probe(args: tuple) -> list[dict]:
    """The game without blocks: its decision points, each with what Gary saw and the window after it."""
    from gary.agent import block as B
    opponent, mp, seed = args
    m = _match(opponent, mp, seed)
    choke = B.main_ramp(m.gary)
    if choke is None:
        m.game.close()
        return []
    w = Watch(m)
    w.look()
    points, last = [], -10 ** 9
    end = MINUTES * 60 * 24
    while m.game.frame < end and _step(m, w):
        f = m.game.frame
        if f % 24 < 2 and f - last >= DECISION_EVERY and len(points) < MAX_POINTS:
            obs = m.gary.hi.observe()                         # (what Gary sees: fogged, a reaction late)
            near = [u for u in obs["units"] if u["owner"] not in (m.slot["T"], 11) and u["type"] in (37, 38, 39, 43, 46, 47)
                    and math.dist((u["x"], u["y"]), choke.through[0]) < NEAR]
            if near:
                last = f
                feats = B.features(m.gary, obs, choke, B._analysis(m.gary)["ramp_width"])
                points.append({"opponent": opponent, "map": mp, "seed": seed, "frame": f, "features": feats})
    final = m.game.frame
    for p in points:
        p["base"] = w.window(p["frame"], min(final, p["frame"] + HORIZON))
    m.game.close()
    return points


def branch(point: dict) -> dict:
    """The same game to the point, then asked to close the main's ramp for BLOCK_FOR."""
    from gary.agent import block as B
    m = _match(point["opponent"], point["map"], point["seed"])
    w = Watch(m)
    w.look()
    f0 = point["frame"]
    while m.game.frame < f0 and _step(m, w):
        pass
    if m.game.frame != f0:
        m.game.close()
        return {**point, "error": f"diverged: at {m.game.frame}, not {f0}"}
    choke = B.main_ramp(m.gary)
    m.gary.blocks[choke.name] = choke
    while m.game.frame < f0 + HORIZON and _step(m, w):
        if m.game.frame >= f0 + BLOCK_FOR and m.gary.blocks.get(choke.name) is choke:
            del m.gary.blocks[choke.name]
    out = {**point, "block": w.window(f0, min(m.game.frame, f0 + HORIZON))}
    m.game.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--opponents", nargs="+", default=list(OPPONENTS))
    ap.add_argument("--maps", nargs="+", default=list(MAPS))
    ap.add_argument("--parallel", type=int, default=14)
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    jobs = [(o, mp, s) for o in args.opponents for mp in args.maps for s in range(1, args.seeds + 1)]
    with ProcessPoolExecutor(args.parallel) as pool:
        points = [p for ps in monitor.track(pool.map(probe, jobs), len(jobs), "games to find moments") for p in ps]
        print(f"{len(points)} decision points in {len(jobs)} games", flush=True)
        rows = list(monitor.track(pool.map(branch, points), len(points), "branches"))
    bad = [r for r in rows if r.get("error")]
    if bad:
        print(f"{len(bad)} branches diverged before their point (e.g. {bad[0]['error']})")
    rows = [r for r in rows if not r.get("error")]
    out_path().parent.mkdir(parents=True, exist_ok=True)
    out_path().write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    d = [r["block"]["net"] - r["base"]["net"] for r in rows]
    if d:
        better = sum(x > 0 for x in d)
        print(f"{len(rows)} branches -> {out_path()}: block better in {better}, worse in {sum(x < 0 for x in d)}, "
              f"same {len(d) - better - sum(x < 0 for x in d)}; mean net {sum(d) / len(d):+.0f}")


if __name__ == "__main__":
    main()
