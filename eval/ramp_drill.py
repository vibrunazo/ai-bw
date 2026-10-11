"""Ramp drill: zerglings run up Gary's main ramp at a small army in the main. Does the ramp get held?

A skill test, not a game: the fight at the ramp alone, the same for every version. The game starts
(Gary on one base); the army is created in the main between its Command Center and the ramp's top
(gary/terrain.py), the zerglings at the natural's choke (the way in); after LEAD (the time a player who sees them
coming has) they attack-move to the Command Center (again every 2 s). Gary is asked to hold the ramp (g.hold_ramp: the decision a higher layer would
make; versions without the skill, fight.py RampRule, ignore it) and plays everything else as in a game.

Per version: zerglings killed, Gary's army and SCVs lost, the most zerglings in the main at once
and zergling-seconds in it (past the ramp's top), and the share of drills where at most 2 ever got
in ("held"); versions paired with the first by drill.

    python -m eval.ramp_drill --versions v018 v018+ramp --seeds 1 2 3 4 5 6 7 8
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import statistics as st
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from eval import priority

REPO_ROOT = Path(__file__).resolve().parent.parent
MAPS = {m: REPO_ROOT / "tests" / "fixtures" / "replays" / f"stardata_tvz_{m}.rep"
        for m in ("python_gg13676", "standard_ozp3w", "heartbreak_9rg84")}
ARMIES = {                                   # unit type: count
    "marines": {0: 6},
    "bio": {0: 4, 32: 1, 34: 2},              # marines, a firebat, medics
    "few": {0: 1, 32: 1, 34: 3},              # what Gary had in the game that showed the ramp trick
}
LINGS = (8, 12, 16)
LING = 37
START = 4 * 24                               # the drill starts (the game settled)
LEAD = 6 * 24                                # the zerglings wait this long before they attack: Gary's time to get ready
SECONDS = 60
INSIDE = 2 * 32                              # a zergling this much nearer the main's center than the top is in
PLOT_EVERY = 4 * 24                          # --save: a picture of the ramp this often (frames)
PLOT_HALF = 14 * 32                          # ...this far around the ramp's top (pixels)
COLORS = {0: (60, 120, 255), 32: (255, 150, 30), 34: (255, 255, 255), 7: (0, 220, 220), 37: (230, 30, 30)}


def _picture(game, walk, height, center, units: list[dict], spots) -> "Image":
    """The ground around the ramp (darker: lower; black: unwalkable), the units as dots (marines
    blue, firebats orange, medics white, SCVs cyan, zerglings red), the hold spots as rings."""
    from PIL import Image, ImageDraw
    x0, y0 = int(center[0] - PLOT_HALF), int(center[1] - PLOT_HALF)
    size = 2 * PLOT_HALF // 8
    im = Image.new("RGB", (size, size))
    px = im.load()
    for j in range(size):
        for i in range(size):
            cy, cx = y0 // 8 + j, x0 // 8 + i
            inside = 0 <= cy < walk.shape[0] and 0 <= cx < walk.shape[1]
            px[i, j] = (60 + 50 * int(height[cy, cx]),) * 3 if inside and walk[cy, cx] else (0, 0, 0)
    im = im.resize((size * 3, size * 3), Image.NEAREST)
    d = ImageDraw.Draw(im)
    at = lambda x, y: ((x - x0) * 3 / 8, (y - y0) * 3 / 8)
    for sx, sy in spots:
        x, y = at(sx, sy)
        d.ellipse((x - 9, y - 9, x + 9, y + 9), outline=(255, 255, 0))
    for u in units:
        if u["type"] in COLORS:
            x, y = at(u["x"], u["y"])
            r = 4 if u["type"] == 37 else 5
            d.ellipse((x - r, y - r, x + r, y + r), fill=COLORS[u["type"]],
                      outline=(0, 0, 0) if u.get("order") != 107 else (255, 255, 0))
    d.text((4, 4), f"{game.frame / 24:.0f} s", fill=(255, 255, 0))
    return im


def _walkable_near(game, x: float, y: float) -> tuple[int, int]:
    for r in range(0, 20 * 32, 16):
        for k in range(16 if r else 1):
            px, py = int(x + r * math.cos(k * math.pi / 8)), int(y + r * math.sin(k * math.pi / 8))
            if 0 <= px < 8192 and 0 <= py < 8192 and game.walkable(px, py):
                return px, py
    return int(x), int(y)


def drill(args: tuple) -> dict:
    mp, seed, army, lings, version, save, lead = args
    from gary import commands as C
    from gary.terrain import map_analysis
    from eval.walls_bench import Match
    with contextlib.redirect_stdout(io.StringIO()):
        m = Match(seed, 10 ** 6, version, MAPS[mp])
    game, ts, zs = m.game, m.slot["T"], m.slot["Z"]
    a = map_analysis(game, m.map)
    main = m.gary.main.center
    r = a["starts"][min(a["starts"], key=lambda k: math.dist(a["starts"][k]["start"], main))]
    out = {"map": mp, "seed": seed, "army": army, "lings": lings, "version": version}
    if not r["reliable"]:
        game.close()
        return {**out, "error": "no reliable ramp"}
    top, bottom = r["ramp_top"], r["ramp_bottom"]
    d_top = math.dist(top, main)
    while game.frame < START:
        m.step()
    obs = game.observe()
    cc = next(u for u in obs["units"] if u["owner"] == ts and u["type"] == C.COMMAND_CENTER)
    mid = ((cc["x"] + top[0]) / 2, (cc["y"] + top[1]) / 2)
    for t, n in ARMIES[army].items():
        for _ in range(n):
            game.create_unit(ts, t, *_walkable_near(game, *mid))
    spawn = _walkable_near(game, *r["natural_choke"])
    for _ in range(lings):
        game.create_unit(zs, LING, *spawn)
    m.gary.hold_ramp = True                  # asked to hold the ramp (versions without fight.py RampRule ignore it)
    army_types = set(ARMIES[army])
    first = {u["tag"] for u in game.observe()["units"] if u["owner"] == ts and u["type"] in army_types}
    scvs0 = sum(1 for u in game.observe()["units"] if u["owner"] == ts and u["type"] == C.SCV)
    most_in, ling_frames = 0, 0
    pictures = []
    if save:
        walk, height, _ = game.terrain()
    end = game.frame + SECONDS * 24
    while game.frame < end and not m.done:
        f = game.frame
        if f >= START + lead and (f - START - lead) % 48 < 2:   # the zerglings: attack-move to the Command Center
            zl = [u["tag"] for u in game.observe()["units"] if u["owner"] == zs and u["type"] == LING]
            for i in range(0, len(zl), 12):
                game.act(zs, C.select(zl[i:i + 12]))
                game.act(zs, C.targeted_order(C.ORDER_ATTACK_MOVE, cc["x"], cc["y"]))
        try:
            m.gary.act()
        except StopIteration:                    # Gary was eliminated (the zerglings took its Command Center)
            out["eliminated"] = True
            break
        for i, hi in enumerate(m.his.values()):
            hi.step(2, advance_game=(i == 0))
        if save and (f - START) % PLOT_EVERY < 2 and len(pictures) < 8:
            pictures.append(_picture(game, walk, height, top, game.observe()["units"], [top, r["ramp_inside"]]))
        if f % 8 < 2:
            o = game.observe()
            inside = sum(1 for u in o["units"] if u["owner"] == zs and u["type"] == LING
                         and math.dist((u["x"], u["y"]), main) < d_top - INSIDE)
            most_in = max(most_in, inside)
            ling_frames += inside * 8
    o = game.observe()
    if out.get("eliminated"):
        o = {"units": [u for u in o["units"] if u["owner"] != ts]}
    alive = {u["tag"] for u in o["units"]}
    lings_left = sum(1 for u in o["units"] if u["owner"] == zs and u["type"] == LING)
    if save:
        game.save_replay(save)                   # (the units the drill made aren't in it: the pictures show the drill)
        m.his["T"].save_pov(save[:-4] + ".pov.jsonl")
        if pictures:
            from PIL import Image
            w, h = pictures[0].size
            strip = Image.new("RGB", (w * 4, h * ((len(pictures) + 3) // 4)))
            for k, im in enumerate(pictures):
                strip.paste(im, ((k % 4) * w, (k // 4) * h))
            strip.save(save[:-4] + ".png")
    lost = sum(1 for t in first if t not in alive and game.unit_type_of(t) == -1)
    scvs = sum(1 for u in o["units"] if u["owner"] == ts and u["type"] == C.SCV)
    game.close()
    return {**out, "killed": lings - lings_left, "army_lost": lost, "scvs_lost": max(0, scvs0 - scvs),
            "most_in": most_in, "ling_s_in": round(ling_frames / 24, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--versions", nargs="+", default=["v018", "v018+ramp"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 9)))
    ap.add_argument("--maps", nargs="+", default=list(MAPS))
    ap.add_argument("--armies", nargs="+", default=list(ARMIES))
    ap.add_argument("--lings", nargs="+", type=int, default=list(LINGS))
    ap.add_argument("--lead", type=float, default=LEAD / 24, help="seconds the zerglings wait before attacking")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--save", help="folder: each drill's pictures of the ramp every 4 s (map_seed_army_lings_version.png), "
                                   "replay and POV log")
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    if args.save:
        Path(args.save).mkdir(parents=True, exist_ok=True)
    save = lambda *k: str(Path(args.save) / ("_".join(map(str, k)).replace("+", "-") + ".rep")) if args.save else None
    jobs = [(mp, s, a, n, v, save(mp, s, a, n, v), int(args.lead * 24)) for mp in args.maps for s in args.seeds for a in args.armies
            for n in args.lings for v in args.versions]
    with ProcessPoolExecutor(args.parallel) as pool:
        rows = list(pool.map(drill, jobs))
    errors = [r for r in rows if r.get("error")]
    if errors:
        print(f"{len(errors)} drills skipped: {errors[0]['map']} {errors[0]['error']}")
    rows = [r for r in rows if not r.get("error")]
    key = lambda r: (r["map"], r["seed"], r["army"], r["lings"])
    base = {key(r): r for r in rows if r["version"] == args.versions[0]}
    print(f"{len(base)} drills: {len(args.maps)} maps, seeds {args.seeds[0]}-{args.seeds[-1]}, armies {args.armies}, zerglings {args.lings}")
    for v in args.versions:
        mine = [r for r in rows if r["version"] == v]
        line = (f"  {v:24s} held {sum(r['most_in'] <= 2 for r in mine) / len(mine):.0%}  zerglings in (most) "
                f"{st.mean(r['most_in'] for r in mine):.1f}  zergling-s in {st.mean(r['ling_s_in'] for r in mine):.0f}  "
                f"killed {st.mean(r['killed'] for r in mine):.1f}  army lost {st.mean(r['army_lost'] for r in mine):.1f}  "
                f"SCVs lost {st.mean(r['scvs_lost'] for r in mine):.1f}")
        if v != args.versions[0]:
            dk = [r["killed"] - r["army_lost"] - (base[key(r)]["killed"] - base[key(r)]["army_lost"]) for r in mine if key(r) in base]
            if len(dk) > 1:
                line += f"  (killed - lost) vs first {st.mean(dk):+.2f} ± {st.stdev(dk) / len(dk) ** .5:.2f}"
        print(line)
    for a in args.armies:
        for n in args.lings:
            print(f"    {a:8s} {n:2d} zerglings  " + "  ".join(
                f"{v}: held {sum(r['most_in'] <= 2 for r in mine) / len(mine):.0%} lost {st.mean(r['army_lost'] for r in mine):.1f}"
                for v in args.versions if (mine := [r for r in rows if r["version"] == v and r["army"] == a and r["lings"] == n])))
    for mp in args.maps:
        for v in args.versions:
            mine = [r for r in rows if r["version"] == v and r["map"] == mp]
            if mine:
                print(f"    {mp:18s} {v:24s} held {sum(r['most_in'] <= 2 for r in mine) / len(mine):.0%}  "
                      f"killed {st.mean(r['killed'] for r in mine):.1f}  army lost {st.mean(r['army_lost'] for r in mine):.1f}")


if __name__ == "__main__":
    main()
