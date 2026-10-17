"""How much one army decision is worth against the game's chaos (Fable's measurement, before building a
critic for the army layer): at army decision moments in Gary-Z's league games, the game is saved and
played on twice per condition set, attack against hold, REPLAYS times each with both sides' hands
redrawn (their mouse scatter: the chaos), both bots live, for WINDOW.

  attack   Gary-Z's ground army attack-moves at the Terran base nearest it (its clusters rarely reach 8
           supply: the army together)
  hold     it stays where it is
  in both, Gary-Z's army layer is frozen for the window (no new cluster decisions), so the decision
  isn't overridden; the rest of both bots plays on

Score: the whole map's net value over the window, the Terran's units and buildings lost less the
Zerg's (/100). Per moment: the effect (attack's mean less hold's) and the spread within a condition
(the chaos). The macro test found a decision worth sd 130 against chaos's 512 over three minutes; if an
army decision's effect is comparable to its chaos, army RL on paired branches can start with the
score as the outcome and no critic; if buried, the critic is the bottleneck.

    python -m eval.army_effect --games 20 --first 7001
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from eval import monitor, priority

REPO = Path(__file__).resolve().parents[1]
SIDE = ["Z"]
WINDOW = 90 * 24
REPLAYS = 6
FROM = 6 * 60 * 24              # moments from 6:00
GAP = 2 * 60 * 24               # at least this far apart
MOMENTS = 4                     # per game at most
TERRAN_MIN_X2 = 32              # --side T: its army this big (16 supply) or more
MIN_SUPPLY_X2 = 12              # its ground army this big (6 supply) or more (its clusters rarely reach 8)
MAPS = {m: REPO / "tests" / "fixtures" / "replays" / f"stardata_tvz_{m}.rep"
        for m in ("python_gg13676", "standard_ozp3w", "heartbreak_9rg84")}


def _net(a: dict, b: dict, zslot: int) -> float:
    from eval.pairs import _value
    lost = lambda own: sum(_value(v[1]) for tag, v in a.items() if (v[0] == zslot) == own and tag not in b)
    return (lost(False) - lost(True)) / 100.0


class EffectProbe:
    """In one league game (GARY_EFFECT: the file, .npz): attack against hold at army decision moments."""

    def __init__(self, game, slot_of: dict, seed: int):
        self.path = os.environ.get("GARY_EFFECT")
        self.side = os.environ.get("GARY_EFFECT_SIDE", "Z")
        self.zslot, self.tslot = slot_of["Z"], slot_of["T"]
        self.rng = random.Random(seed * 7907 + 11)
        self.next = FROM
        self.moments: list[dict] = []

    @property
    def on(self) -> bool:
        return bool(self.path)

    def maybe(self, game, objs: dict) -> None:
        if self.side == "T":
            return self._maybe_terran(game, objs)
        from eval.pairs import _copy, _play, snap
        from gary.env import GameError
        f = game.frame
        if f < self.next or len(self.moments) >= MOMENTS:
            return
        zerg = objs["zerg"]
        if not getattr(zerg, "cluster_army", False) or zerg.army_job is not None:
            self.next = f + 48
            return
        o = game.observe()
        from gary.policy.army import supply_x2
        mine = [u for u in o["units"] if u["owner"] == self.zslot and u["type"] in zerg.sizes and u["completed"]]
        if not mine:
            self.next = f + 48
            return
        cluster = mine                                   # (the whole ground army: attack together, or hold)
        if sum(supply_x2(u["type"]) for u in cluster) < MIN_SUPPLY_X2:
            self.next = f + 48
            return
        ccs = [(u["x"], u["y"]) for u in o["units"] if u["owner"] == self.tslot and u["type"] == 106]
        if not ccs:
            return
        try:
            game.save()
        except GameError:
            self.next = f + 2
            return
        self.next = f + GAP
        mid = (sum(u["x"] for u in cluster) / len(cluster), sum(u["y"] for u in cluster) / len(cluster))
        target = min(ccs, key=lambda c: (c[0] - mid[0]) ** 2 + (c[1] - mid[1]) ** 2)
        start = snap(o)
        scores = {"attack": [], "hold": []}
        for cond in ("attack", "hold"):
            for r in range(REPLAYS):
                game.restore()
                c = _copy(objs)
                z = c["zerg"]
                for hi in c["his"].values():               # (the hands' scatter redrawn: the chaos)
                    hi.rng = random.Random(self.rng.random())
                z.cluster_ask = {k: f + WINDOW for k in range(10)}   # (the army layer frozen for the window)
                if cond == "attack":
                    z.army_job = {"tags": [u["tag"] for u in cluster], "point": target, "kind": "attack",
                                  "selecting": False, "tries": 0}
                _play(game, c, WINDOW)
                scores[cond].append(_net(start, snap(game.observe()), self.zslot))
        game.restore()
        self.moments.append({"frame": f, "supply": sum(supply_x2(u["type"]) for u in cluster) / 2, **scores})

    def _maybe_terran(self, game, objs: dict) -> None:
        """The same for Gary-T's army (its armies reach 20-50 supply; Gary-Z's rarely 6): attack the Zerg
        base nearest it, or hold, its army layer (and hunt) frozen for the window."""
        from eval.pairs import _copy, _play, snap
        from gary.agent.intents import Intent
        from gary.env import GameError
        from gary.policy.army import supply_x2
        f = game.frame
        if f < self.next or len(self.moments) >= MOMENTS:
            return
        gary = objs["gary"]
        army = getattr(gary, "army", None)
        o = game.observe()
        mine = [u for u in o["units"] if u["owner"] == self.tslot and u["type"] in gary.army_types and u["completed"]]
        if army is None or sum(supply_x2(u["type"]) for u in mine) < TERRAN_MIN_X2:
            self.next = f + 48
            return
        halls = [(u["x"], u["y"]) for u in o["units"] if u["owner"] == self.zslot and u["type"] in (131, 132, 133)]
        if not halls:
            return
        try:
            game.save()
        except GameError:
            self.next = f + 2
            return
        self.next = f + GAP
        mid = (sum(u["x"] for u in mine) / len(mine), sum(u["y"] for u in mine) / len(mine))
        x, y = min(halls, key=lambda c: (c[0] - mid[0]) ** 2 + (c[1] - mid[1]) ** 2)
        tags = [u["tag"] for u in mine]
        start = snap(o)
        scores = {"attack": [], "hold": []}
        for cond in ("attack", "hold"):
            for r in range(REPLAYS):
                game.restore()
                c = _copy(objs)
                for hi in c["his"].values():
                    hi.rng = random.Random(self.rng.random())
                g = c["gary"]
                g.army.next_ask = f + WINDOW               # (the army layer frozen for the window)
                g.army.hunt = False
                if cond == "attack":
                    g.motor.start(Intent("ARMY_ATTACK", tags, (int(x), int(y)),
                                         {"tags": tags, "x": int(x), "y": int(y), "attack": True, "p": 1.0, "fight": 0.0,
                                          "done": [], "boxes": 0}, "army"))
                _play(game, c, WINDOW)
                scores[cond].append(-_net(start, snap(game.observe()), self.zslot))   # (the Terran's side)
        game.restore()
        self.moments.append({"frame": f, "supply": sum(supply_x2(u["type"]) for u in mine) / 2, **scores})

    def save(self) -> None:
        if not self.path or not self.moments:
            return
        np.savez_compressed(self.path, frame=np.array([m["frame"] for m in self.moments]),
                            supply=np.array([m["supply"] for m in self.moments]),
                            attack=np.array([m["attack"] for m in self.moments]),
                            hold=np.array([m["hold"] for m in self.moments]))


def play(job: tuple) -> str:
    name, path, seed, minutes, out = job
    folder = out / name
    folder.mkdir(parents=True, exist_ok=True)
    rep = folder / f"seed{seed}.rep"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "GARY_EFFECT": str(rep.with_suffix(".effect.npz")),
           "GARY_EFFECT_SIDE": SIDE[0]}
    cmd = [sys.executable, "-m", "gary.bots.zerg_rush", "--vs", "v029+draw", "--opponent", "garyz", "--map", str(path),
           "--minutes", str(minutes), "--seed", str(seed), "--save", str(rep)]
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    rep.with_suffix(".log").write_text(p.stdout + p.stderr, encoding="utf-8")
    return str(rep)


def analyse(files: list[str]) -> dict:
    att, hold, sup = [], [], []
    for f in files:
        d = np.load(f)
        att.extend(d["attack"]); hold.extend(d["hold"]); sup.extend(d["supply"])
    att, hold = np.array(att), np.array(hold)
    if not len(att):
        return {"moments": 0}
    effect = att.mean(1) - hold.mean(1)
    within = np.sqrt((att.var(1, ddof=1) + hold.var(1, ddof=1)) / 2)          # the chaos, per moment
    se = within * np.sqrt(2 / REPLAYS)
    return {"moments": int(len(effect)), "replays": REPLAYS, "window_s": WINDOW // 24,
            "effect_mean": float(effect.mean()), "effect_sd": float(effect.std(ddof=1)),
            "effect_abs_median": float(np.median(np.abs(effect))),
            "chaos_sd_median": float(np.median(within)), "chaos_sd_mean": float(within.mean()),
            "moments_past_2se": float(np.mean(np.abs(effect) > 2 * se)),
            "ratio_effect_sd_to_chaos": float(effect.std(ddof=1) / max(1e-9, np.median(within))),
            "attack_better_share": float(np.mean(effect > 0)), "supply_median": float(np.median(sup))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--first", type=int, default=7001)
    ap.add_argument("--minutes", type=float, default=15)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--out", default="runs/army_effect")
    ap.add_argument("--analyse", action="store_true", help="only the analysis of what's there")
    ap.add_argument("--side", choices=("T", "Z"), default="Z", help="whose army decisions")
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    out = REPO / args.out
    SIDE[0] = args.side
    if not args.analyse:
        names = list(MAPS)
        jobs = [(names[k % 3], MAPS[names[k % 3]], args.first + k, args.minutes, out) for k in range(args.games)]
        with ThreadPoolExecutor(args.parallel) as pool:
            list(monitor.track(pool.map(play, jobs), len(jobs), "games"))
    report = analyse(sorted(glob.glob(str(out / "*" / "*.effect.npz"))))
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
