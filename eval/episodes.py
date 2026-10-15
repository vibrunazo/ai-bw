"""Defense episodes: the home-defense scenarios (eval/scenarios.py) played for longer, scored also by
how the Terran's army dies.

The scenarios' 45 s start mid-fight and end before a Barracks makes another marine, so they can't
show what lost Gary games: during a long fight at home its new marines went in one by one (on
Python, 48% of Gary's army deaths had no other of its fighters within 6 tiles; pros' 12%). Here
the same scenarios run --seconds (120 by default) with Gary playing the whole Terran, production
included, and every army death is recorded: alone or not, outnumbered or not, how old the unit was.

Per controller (pro, nothing, and each Gary version): net value (as eval/scenarios.py: zerg lost
minus terran lost, plus the threats left at the Terran's bases), SCVs lost, army deaths per episode,
the share of them alone (no other Terran fighter within 6 tiles) and outnumbered 2:1 or worse, the
median of zerg / Terran fighters around a death; Gary versions paired with the first.

    python -m eval.episodes --versions v023 v023+regroup --run 239 --draws 4
    python -m eval.episodes --split train --run 552 --versions v023      (the training episodes)
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from concurrent.futures import ProcessPoolExecutor

from eval import monitor, priority
from eval.scenarios import run_one, scenarios_path

ARMY = {0, 1, 2, 3, 5, 30, 32, 34}           # marine, ghost, vulture, goliath, tank (both modes), firebat, medic
ZERG_FIGHTERS = {37, 38, 39, 43, 103, 46, 47, 45, 44}
NEAR = 6 * 32                                # "around" a death
EVERY = 4                                    # frames between looks


class Deaths:
    """on_step hook for run_one: every army death of the controlled Terran, with its surroundings."""

    def __init__(self):
        self.last: dict[int, dict] = {}
        self.born: dict[int, int] = {}
        self.rows: list[dict] = []
        self.context: dict | None = None
        self.center = None                  # the scenario's fight (its own record)

    def __call__(self, game, slot: int) -> None:
        f = game.frame
        if f % EVERY:
            return
        o = game.observe()
        if self.context is None:
            self.context = situation(game, o, slot, self.center)
        here = {u["tag"] for u in o["units"]}
        for tag, s in list(self.last.items()):
            if tag not in here:
                if game.unit_type_of(tag) == -1:            # dead (not inside a bunker)
                    self.rows.append({**s, "died": f, "age": f - self.born.get(tag, f)})
                    del self.last[tag]
        army = [u for u in o["units"] if u["owner"] == slot and u["type"] in ARMY and u["completed"]]
        zerg = [u for u in o["units"] if u["owner"] not in (slot, 11) and u["type"] in ZERG_FIGHTERS]
        for u in army:
            self.born.setdefault(u["tag"], f)
            p = (u["x"], u["y"])
            self.last[u["tag"]] = {
                "zerg": sum(1 for z in zerg if math.dist(p, (z["x"], z["y"])) < NEAR),
                "own": sum(1 for v in army if math.dist(p, (v["x"], v["y"])) < NEAR),   # itself too
            }


def situation(game, o: dict, slot: int, center=None) -> dict:
    """The episode at its start as the Terran sees it: game minute, its Command Centers, finished
    bunkers, army (supply x2), the zerg fighters in sight (supply x2), and whether the fight (the
    scenario's center, else the zerg in sight) is nearest its main's Command Center (by a start
    location) or another."""
    from gary.policy.army import supply_x2
    mine = [u for u in o["units"] if u["owner"] == slot]
    zerg = [u for u in o["units"] if u["owner"] not in (slot, 11) and u["type"] in ZERG_FIGHTERS
            and (u.get("visible_to", 0) >> slot) & 1]
    ccs = [u for u in mine if u["type"] == 106]
    out = {"minute": round(o["frame"] / 1440, 2), "ccs": len(ccs),
           "bunkers": sum(1 for u in mine if u["type"] == 125 and u["completed"]),
           "army": sum(supply_x2(u["type"]) for u in mine if u["type"] in ARMY and u["completed"]),
           "zerg": sum(supply_x2(u["type"]) for u in zerg), "at_main": None}
    if ccs and (center or zerg):
        zx, zy = center or (sum(z["x"] for z in zerg) / len(zerg), sum(z["y"] for z in zerg) / len(zerg))
        cc = min(ccs, key=lambda u: math.dist((u["x"], u["y"]), (zx, zy)))
        starts = [(s["x"], s["y"]) for s in game.start_locations()]
        out["at_main"] = any(math.dist((cc["x"], cc["y"]), s) < 6 * 32 for s in starts)
    return out


def job(args: tuple) -> dict:
    sc, controller, seconds, version, hands, fight = args
    hook = Deaths()
    hook.center = (sc["cx"], sc["cy"]) if "cx" in sc else None
    r = run_one(sc, controller, seconds, version, 1, on_step=hook, hands_seed=hands, fight_path=fight)
    return {"sha1": sc["sha1"][:8], "who": controller if controller != "gary" else version, "hands": hands,
            "net": r.get("net"), "scvs": r.get("T_workers_lost"), "error": r.get("error"), "deaths": hook.rows,
            "context": hook.context}


def summarize(rows: list[dict], who: str, base: dict | None) -> dict:
    ok = [r for r in rows if r["who"] == who and not r["error"] and r["net"] is not None]
    per = {}
    for r in ok:
        per.setdefault(r["sha1"], []).append(r["net"])
    m = {k: st.mean(v) for k, v in per.items()}
    deaths = [d for r in ok for d in r["deaths"]]
    n = len(deaths) or 1
    out = {"who": who, "episodes": len(per), "net": round(st.mean(m.values()), 1) if m else None,
           "scvs_lost": round(st.mean(r["scvs"] or 0 for r in ok), 2) if ok else None,
           "army_deaths": round(len(deaths) / max(1, len(ok)), 2),
           "alone": round(sum(d["own"] <= 1 for d in deaths) / n, 3),
           "outnumbered": round(sum(d["zerg"] >= 2 * d["own"] for d in deaths) / n, 3),
           "zerg_around": st.median(d["zerg"] for d in deaths) if deaths else None,
           "own_around": st.median(d["own"] for d in deaths) if deaths else None,
           "alone_age_s": round(st.median(d["age"] / 24 for d in deaths if d["own"] <= 1), 1)
           if any(d["own"] <= 1 for d in deaths) else None}
    if base is not None:
        d = [m[k] - base[k] for k in m if k in base]
        if len(d) > 1:
            out["vs_first"] = f"{st.mean(d):+.1f} ± {st.stdev(d) / len(d) ** .5:.1f}"
    out["_per"] = m
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--versions", nargs="+", default=["v023"])
    ap.add_argument("--run", type=int, default=239, help="the first N saved scenarios")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--draws", type=int, default=4, help="Gary's hands, seeded differently")
    ap.add_argument("--no-pro", action="store_true", help="skip the pro and nothing controllers")
    ap.add_argument("--fight-model", nargs="+", default=[],
                    help="per version, its fight (role) model, \"-\" for the manifest's (e.g. - runs/roles_rl/<run>/model.pt)")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--out", help="write every row (with its deaths) to this JSON file")
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    scs = [json.loads(line) for line in scenarios_path(args.split).read_text(encoding="utf-8").splitlines()][:args.run]
    fights = dict(zip(args.versions, [None if f == "-" else f for f in args.fight_model]))
    jobs = []
    for sc in scs:
        if not args.no_pro:
            jobs += [(sc, "pro", args.seconds, args.versions[0], 1, None), (sc, "nothing", args.seconds, args.versions[0], 1, None)]
        jobs += [(sc, "gary", args.seconds, v, 1 + h, fights.get(v)) for v in args.versions for h in range(args.draws)]
    with ProcessPoolExecutor(args.parallel) as pool:
        per = max(1, len(jobs) // max(1, len(scs)))           # a scenario's plays on one worker (its game reused)
        rows = list(monitor.track(pool.map(job, jobs, chunksize=per), len(jobs), "episodes"))
    errors = [r for r in rows if r["error"]]
    if errors:
        print(f"{len(errors)} errors, e.g. {errors[0]['error']}")
    print(f"{len(scs)} {args.split} episodes of {args.seconds:.0f} s; Gary: {args.draws} draws of its hands")
    who = ([] if args.no_pro else ["pro", "nothing"]) + args.versions
    base = None
    for w in who:
        s = summarize(rows, w, base if w in args.versions[1:] else None)
        if w == args.versions[0]:
            base = s["_per"]
        s.pop("_per")
        print("  " + "  ".join(f"{k} {v}" for k, v in s.items()))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f)


if __name__ == "__main__":
    main()
