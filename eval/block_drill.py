"""Choke drill: zerglings run at a small army that is asked to close a choke with its bodies (gary/agent/
block.py). Is the way closed when they come, and does it hold?

A skill test, not a game: one fight at one choke, the same for every version. The game starts (Gary
on one base) and settles; the army is created behind the choke, the zerglings outside it; Gary is
asked to close it (g.blocks: the decision a higher layer would make; versions without the skill,
fight.py BlockRule, ignore it) and plays everything else as in a game. After LEAD (the time a
player who sees them coming has) the zerglings attack-move to what the choke protects (again every
2 s). The chokes:

  ramp       the main's ramp (block.main_ramp(): SCVs from the mineral line may help); the army in
             the main, the zerglings at the natural's choke
  natural    the natural's choke (block.natural_choke()), with a Command Center created at the
             natural (as a base taken); the army at the natural, the zerglings well outside

Per version: closed when the first zergling reaches the choke ("sealed", exact: gary/walls.py),
held (at most 2 zerglings ever past it), zerglings past it (most at once, zergling-seconds),
killed, Gary's army and SCVs lost; versions paired with the first by drill.

    python -m eval.block_drill --versions v018 v018+block --chokes ramp natural --seeds 1 2 3 4
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import statistics as st
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from eval import monitor, priority

REPO_ROOT = Path(__file__).resolve().parent.parent
MAPS = {m: REPO_ROOT / "tests" / "fixtures" / "replays" / f"stardata_tvz_{m}.rep"
        for m in ("python_gg13676", "standard_ozp3w", "heartbreak_9rg84")}
ARMIES = {                                   # unit type: count
    "marines": {0: 6},
    "bio": {0: 4, 32: 1, 34: 2},              # marines, a firebat, medics
    "few": {0: 1, 32: 1, 34: 3},              # what Gary had in the game that showed the ramp trick
    "marines12": {0: 12},                     # (wider chokes: the naturals')
    "bio12": {0: 8, 32: 2, 34: 2},
}
CHOKE_ARMIES = {"ramp": ("marines", "bio", "few"), "natural": ("marines12", "bio12")}
OUTSIDE = 20 * 32                            # natural: the zerglings this far out past its choke
LINGS = (8, 12, 16)
LING = 37
START = 4 * 24                               # the drill starts (the game settled)
LEAD = 6 * 24                                # the zerglings wait this long before they attack: Gary's time to get ready
SECONDS = 60
CONTACT = 3 * 32                             # the first zergling this close to the choke: is the way closed?
INSIDE = 2 * 32                              # a zergling this much nearer what the choke protects (the Command Center) than the choke is: past it
PLOT_ZOOM = 6                                # ...picture pixels per minitile
PLOT_EVERY = 4 * 24                          # --save: a picture of the ramp this often (frames)
PLOT_HALF = 9 * 32                           # ...this far around the choke (pixels)
COLORS = {0: (60, 120, 255), 32: (255, 150, 30), 34: (255, 255, 255), 7: (0, 220, 220), 37: (230, 30, 30)}


def _picture(game, walk, height, center, units: list[dict], spots) -> "Image":
    """The ground around the choke (darker: lower; black: unwalkable), the units (marines
    blue, firebats orange, medics white, SCVs cyan, zerglings red: their collision boxes, outlined
    yellow on hold position), the block's planned spots as small rings."""
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
    im = im.resize((size * PLOT_ZOOM, size * PLOT_ZOOM), Image.NEAREST)
    d = ImageDraw.Draw(im)
    at = lambda x, y: ((x - x0) * PLOT_ZOOM / 8, (y - y0) * PLOT_ZOOM / 8)
    for sx, sy in spots:
        x, y = at(sx, sy)
        d.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(255, 255, 0))
    for u in units:
        if u["type"] in COLORS:
            i = game.unit_type_info(u["type"])
            ax, ay = at(u["x"] - i["left"], u["y"] - i["up"])
            bx, by = at(u["x"] + i["right"] + 1, u["y"] + i["down"] + 1)
            d.rectangle((ax, ay, bx, by), fill=COLORS[u["type"]],
                        outline=(255, 255, 0) if u.get("order") in (107, 177) else (0, 0, 0))
    d.text((4, 4), f"{game.frame / 24:.0f} s", fill=(255, 255, 0))
    return im


def _walkable_near(game, x: float, y: float) -> tuple[int, int]:
    for r in range(0, 20 * 32, 16):
        for k in range(16 if r else 1):
            px, py = int(x + r * math.cos(k * math.pi / 8)), int(y + r * math.sin(k * math.pi / 8))
            if 0 <= px < 8192 and 0 <= py < 8192 and game.walkable(px, py):
                return px, py
    return int(x), int(y)


def _block(m, name: str):
    """The skill holding this choke in Gary (fight.py BlockRule's ChokeBlock), or None."""
    rule = next((x for x in m.gary.fight.before if x.name == "block"), None)
    return rule.held.get(name) if rule is not None else None


def sealed(game, obs: dict, slot: int, outside, inside) -> bool:
    """Whether Gary's units (their collision boxes) and buildings close the way from outside to inside
    to a zergling right now (gary/walls.py, exact)."""
    from gary import walls as W
    units = [(u["type"], u["x"], u["y"]) for u in obs["units"] if u["owner"] == slot and u["type"] in W.WALKERS]
    return W.closed(game, obs, units, tuple(outside), tuple(inside))


def drill(args: tuple) -> dict:
    kind, mp, seed, army, lings, version, save, lead = args
    from gary import commands as C
    from gary.agent import block as B
    from eval.walls_bench import Match
    with contextlib.redirect_stdout(io.StringIO()):
        m = Match(seed, 10 ** 6, version, MAPS[mp])
    game, ts, zs = m.game, m.slot["T"], m.slot["Z"]
    out = {"choke": kind, "map": mp, "seed": seed, "army": army, "lings": lings, "version": version}
    while game.frame < START:
        m.step()
    obs = game.observe()
    cc = next(u for u in obs["units"] if u["owner"] == ts and u["type"] == C.COMMAND_CENTER)
    if kind == "ramp":
        choke = B.main_ramp(m.gary)
        if choke is None:
            game.close()
            return {**out, "error": "no reliable ramp"}
        at = choke.through[0]
        stand = ((cc["x"] + at[0]) / 2, (cc["y"] + at[1]) / 2)
        spawn = _walkable_near(game, *B._analysis(m.gary)["natural_choke"])
        target = (cc["x"], cc["y"])          # (what it protects: the main's Command Center)
    else:                                    # the natural: taken (a Command Center there), its choke closed
        choke = B.natural_choke(m.gary)
        nat, at = choke.inside, choke.through[0]
        game.create_unit(ts, C.COMMAND_CENTER, *nat)
        d = math.dist(nat, at) or 1
        stand = ((nat[0] + at[0]) / 2, (nat[1] + at[1]) / 2)
        spawn = _walkable_near(game, at[0] + (at[0] - nat[0]) / d * OUTSIDE, at[1] + (at[1] - nat[1]) / d * OUTSIDE)
        target = nat
    for t, n in ARMIES[army].items():
        for _ in range(n):
            game.create_unit(ts, t, *_walkable_near(game, *stand))
    for _ in range(lings):
        game.create_unit(zs, LING, *spawn)
    m.gary.blocks[choke.name] = choke        # asked to close it (versions without fight.py BlockRule ignore it)
    d_at = math.dist(at, target)
    army_types = set(ARMIES[army])
    first = {u["tag"] for u in game.observe()["units"] if u["owner"] == ts and u["type"] in army_types}
    scvs0 = sum(1 for u in game.observe()["units"] if u["owner"] == ts and u["type"] == C.SCV)
    most_in, ling_frames, contact = 0, 0, None
    pictures = []
    if save:
        walk, height, _ = game.terrain()
    end = game.frame + SECONDS * 24
    while game.frame < end and not m.done:
        f = game.frame
        if f >= START + lead and (f - START - lead) % 48 < 2:   # the zerglings: attack-move to what it protects
            zl = [u["tag"] for u in game.observe()["units"] if u["owner"] == zs and u["type"] == LING]
            for i in range(0, len(zl), 12):
                game.act(zs, C.select(zl[i:i + 12]))
                game.act(zs, C.targeted_order(C.ORDER_ATTACK_MOVE, int(target[0]), int(target[1])))
        try:
            m.gary.act()
        except StopIteration:                    # Gary was eliminated (the zerglings took its Command Center)
            out["eliminated"] = True
            break
        for i, hi in enumerate(m.his.values()):
            hi.step(2, advance_game=(i == 0))
        if save and (f - START) % PLOT_EVERY < 2 and len(pictures) < 8:
            b = _block(m, choke.name)
            spots = [(x, y) for _, x, y in b.plan["units"]] if b is not None and b.plan else [at]
            pictures.append(_picture(game, walk, height, at, game.observe()["units"], spots))
        if f % 8 < 2:
            o = game.observe()
            if contact is None and any(u["owner"] == zs and u["type"] == LING and math.dist((u["x"], u["y"]), at) < CONTACT
                                       for u in o["units"]):
                contact = sealed(game, o, ts, choke.outside, choke.inside)
                b = _block(m, choke.name)
                if b is not None and b.plan:             # the block then: its spots, and who stood on them holding
                    by = {u["tag"]: u for u in o["units"]}
                    spots = b.plan["units"]
                    on = [t for t, i in b.slot_of.items() if t in by and by[t]["order"] in (107, 177)
                          and math.dist((by[t]["x"], by[t]["y"]), spots[i][1:]) <= 4]
                    out["block"] = (len(on), len(spots), len(b.slot_of))
            inside = sum(1 for u in o["units"] if u["owner"] == zs and u["type"] == LING
                         and math.dist((u["x"], u["y"]), target) < d_at - INSIDE)
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
    return {**out, "came": contact is not None, "sealed": bool(contact), "killed": lings - lings_left, "army_lost": lost, "scvs_lost": max(0, scvs0 - scvs),
            "most_in": most_in, "ling_s_in": round(ling_frames / 24, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--versions", nargs="+", default=["v018", "v018+block"])
    ap.add_argument("--chokes", nargs="+", default=["ramp"], choices=list(CHOKE_ARMIES))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 9)))
    ap.add_argument("--maps", nargs="+", default=list(MAPS))
    ap.add_argument("--armies", nargs="+", help="default: the choke's (CHOKE_ARMIES)")
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
    jobs = [(k, mp, s, a, n, v, save(k, mp, s, a, n, v), int(args.lead * 24)) for k in args.chokes for mp in args.maps
            for s in args.seeds for a in (args.armies or CHOKE_ARMIES[k]) for n in args.lings for v in args.versions]
    with ProcessPoolExecutor(args.parallel) as pool:
        rows = list(monitor.track(pool.map(drill, jobs), len(jobs), "drills"))
    errors = [r for r in rows if r.get("error")]
    if errors:
        print(f"{len(errors)} drills skipped: {errors[0]['map']} {errors[0]['error']}")
    rows = [r for r in rows if not r.get("error")]
    for k in args.chokes:
        report(args, [r for r in rows if r["choke"] == k], k)


def report(args, rows: list[dict], kind: str) -> None:
    key = lambda r: (r["map"], r["seed"], r["army"], r["lings"])
    base = {key(r): r for r in rows if r["version"] == args.versions[0]}
    came = sum(r["came"] for r in rows) / max(1, len(rows))
    print(f"{kind}: {len(base)} drills: {len(args.maps)} maps, seeds {args.seeds[0]}-{args.seeds[-1]}, "
          f"zerglings {args.lings}; they reached the choke in {came:.0%}")
    rate = lambda rs, f: f"{sum(f(r) for r in rs) / len(rs):.0%}"
    for v in args.versions:
        mine = [r for r in rows if r["version"] == v]
        line = (f"  {v:24s} sealed {rate(mine, lambda r: r['sealed'])}  held {rate(mine, lambda r: r['most_in'] <= 2)}  "
                f"zerglings past (most) {st.mean(r['most_in'] for r in mine):.1f}  zergling-s past {st.mean(r['ling_s_in'] for r in mine):.0f}  "
                f"killed {st.mean(r['killed'] for r in mine):.1f}  army lost {st.mean(r['army_lost'] for r in mine):.1f}  "
                f"SCVs lost {st.mean(r['scvs_lost'] for r in mine):.1f}")
        if v != args.versions[0]:
            dk = [r["killed"] - r["army_lost"] - (base[key(r)]["killed"] - base[key(r)]["army_lost"]) for r in mine if key(r) in base]
            if len(dk) > 1:
                line += f"  (killed - lost) vs first {st.mean(dk):+.2f} ± {st.stdev(dk) / len(dk) ** .5:.2f}"
        print(line)
    for a in sorted({r["army"] for r in rows}, key=list(ARMIES).index):
        for n in args.lings:
            print(f"    {a:9s} {n:2d} zerglings  " + "  ".join(
                f"{v}: sealed {rate(mine, lambda r: r['sealed'])} held {rate(mine, lambda r: r['most_in'] <= 2)} "
                f"lost {st.mean(r['army_lost'] for r in mine):.1f}"
                for v in args.versions if (mine := [r for r in rows if r["version"] == v and r["army"] == a and r["lings"] == n])))
    for mp in args.maps:
        for v in args.versions:
            mine = [r for r in rows if r["version"] == v and r["map"] == mp]
            if mine:
                print(f"    {mp:18s} {v:24s} sealed {rate(mine, lambda r: r['sealed'])}  held {rate(mine, lambda r: r['most_in'] <= 2)}  "
                      f"killed {st.mean(r['killed'] for r in mine):.1f}  army lost {st.mean(r['army_lost'] for r in mine):.1f}")


if __name__ == "__main__":
    main()
