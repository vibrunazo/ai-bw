"""What each fighting marine is doing, every half second: the pro vs Gary (#2, docs/CONTROL.md).

A role per unit, not a label per command: in one moment the front marines step back while the
rest shoot, and that is two roles, not one messy intent. On the home defenses of
eval/scenarios.py, under each controller, every marine with a zerg fighter within 7 tiles gets,
for each half second:

  fire        attacking (most of the half second)
  engage      moving, closer to the nearest zerg fighter by the end
  pull_back   moving away from it, a short stretch (under 4 tiles), starting while threatened (a
              zerg fighter targeted it in the last second, or one is within 1.5 tiles)
  retreat     moving away, 4 tiles or more in one stretch
  reposition  moving sideways, or a short stretch away without a threat
  stand       idle, stopped or holding
  other       anything else (entering a bunker...)

And the response it's about: when a zerg fighter gets on a marine (targets it from within 1.5
tiles, after a calm second), how often the marine is pulled back within a second, and how fast.

Orders and positions, not commands, so the pro (replayed) and Gary are measured the same way.

    python -m eval.micro_roles --run 114 --parallel 16
    python -m eval.micro_roles --controllers pro --save runs/micro_roles --out roles.jsonl

With --out, the role stretches (marine, start and end frames, role, distance, threatened) and
the tracks of the marines in them are saved too; with --save, the replays, to watch the roles.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

MARINE = 0
ATTACK = {10, 12, 14}
MOVE = {6}
STAND = {1, 2, 3, 107}
NEAR = 7 * 32                        # a zerg fighter this close: the marine is fighting
CONTACT = 48                         # a zerg fighter this close (1.5 tiles) is on the marine
TARGETED_FOR = 24                    # frames: targeted this recently counts as threatened
SLICE = 12                           # frames: a role every half second
RETREAT_PX = 4 * 32
SIDEWAYS = 8                         # pixels: closer or farther by less than this is sideways
FPS = 1000 / 42


class Probe:
    def __init__(self):
        self.samples: dict[int, list] = {}   # marine -> [(frame, x, y, order, nearest zerg px, targeted frame)]
        self.targeted: dict[int, int] = {}   # marine -> last frame a zerg fighter targeted it
        self.start = None

    def __call__(self, game, slot: int) -> None:
        from gary.policy.army import is_building, is_worker, supply_x2
        obs = game.observe()
        f = obs["frame"]
        if self.start is None:
            self.start = f
        zerg = [u for u in obs["units"] if u["owner"] not in (slot, 11) and supply_x2(u["type"]) > 0
                and not is_worker(u["type"]) and not is_building(u["type"])]
        for z in zerg:
            if z.get("order_target"):
                self.targeted[z["order_target"]] = f
        for u in obs["units"]:
            if u["owner"] != slot or u["type"] != MARINE or not u["completed"]:
                continue
            near = min((math.dist((u["x"], u["y"]), (z["x"], z["y"])) for z in zerg), default=1e9)
            self.samples.setdefault(u["tag"], []).append(
                (f, u["x"], u["y"], u["order"], near, self.targeted.get(u["tag"], -10**9)))

    def result(self) -> dict:
        """Role stretches per marine, the response to contact, and the tracks of moving marines."""
        stretches, responses, tracks = [], [], {}
        slice_roles: Counter = Counter()
        for tag, s in self.samples.items():
            slices = []                                  # (frame0, frame1, kind, x0, y0, x1, y1, threatened)
            i = 0
            while i < len(s):
                j = i
                while j < len(s) and s[j][0] < s[i][0] + SLICE:
                    j += 1
                part = s[i:j]
                f0, x0, y0, _, near0, tgt0 = part[0]
                f1, x1, y1, _, near1, _ = part[-1]
                threatened = f0 - tgt0 <= TARGETED_FOR or near0 < CONTACT
                if near0 >= NEAR:
                    kind = None
                else:
                    orders = [p[3] for p in part]
                    if sum(o in ATTACK for o in orders) * 2 >= len(orders):
                        kind = "fire"
                    elif sum(o in MOVE for o in orders) * 2 >= len(orders):
                        kind = "engage" if near1 < near0 - SIDEWAYS else "away" if near1 > near0 + SIDEWAYS else "reposition"
                    elif sum(o in STAND for o in orders) * 2 >= len(orders):
                        kind = "stand"
                    else:
                        kind = "other"
                slices.append([f0, f1, kind, x0, y0, x1, y1, threatened])
                i = j
            # stretches away from the zerg: retreat, pull back or reposition, as a whole
            k = 0
            while k < len(slices):
                if slices[k][2] != "away":
                    k += 1
                    continue
                m = k
                while m < len(slices) and slices[m][2] == "away":
                    m += 1
                dist = math.dist(slices[k][3:5], slices[m - 1][5:7])
                role = "retreat" if dist >= RETREAT_PX else "pull_back" if slices[k][7] else "reposition"
                for q in range(k, m):
                    slices[q][2] = role
                k = m
            # merge runs of one role into stretches; count slices
            k = 0
            while k < len(slices):
                role = slices[k][2]
                m = k
                while m < len(slices) and slices[m][2] == role:
                    m += 1
                if role is not None:
                    slice_roles[role] += m - k
                    stretches.append({"marine": tag, "role": role, "f0": slices[k][0], "f1": slices[m - 1][1],
                                      "slices": m - k, "tiles": round(math.dist(slices[k][3:5], slices[m - 1][5:7]) / 32, 1),
                                      "threatened": slices[k][7]})
                k = m
            # contact: a zerg fighter on the marine after a calm second; pulled back within a second?
            calm = 0
            for k, sl in enumerate(slices):
                f0, kind = sl[0], sl[2]
                on = kind is not None and sl[7] and next(p for p in s if p[0] == f0)[4] < CONTACT
                if on and f0 - calm >= 24:
                    after = slices[k:k + 3]
                    back = next((a for a in after if a[2] in ("pull_back", "retreat")), None)
                    responses.append({"marine": tag, "frame": f0, "pulled": back is not None,
                                      "delay_s": round((back[0] - f0) / FPS, 2) if back else None})
                if not on:
                    calm = f0 if kind is None or not sl[7] else calm
                else:
                    calm = f0 + 24
            if any(st["marine"] == tag and st["role"] in ("pull_back", "retreat", "engage", "reposition") for st in stretches):
                tracks[tag] = [(p[0], p[1], p[2]) for p in s[::2]]
        return {"roles": dict(slice_roles), "stretches": stretches, "responses": responses, "tracks": tracks}


def _job(args: tuple) -> dict:
    from eval.scenarios import run_one
    sc, controller, version, style, save = args
    probe = Probe()
    path = f"{save}/{sc['sha1'][:8]}_{controller}.rep" if save else None
    row = run_one(sc, controller, 45, version, style, save=path, on_step=probe)
    return {"sha1": row["sha1"], "controller": controller, "slot": sc["terran_slot"], "error": row.get("error"),
            **probe.result()}


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r["error"]]
    roles: Counter = Counter()
    for r in ok:
        roles.update(r["roles"])
    total = sum(roles.values()) or 1
    minutes = total * SLICE / FPS / 60
    st = [s for r in ok for s in r["stretches"]]
    resp = [x for r in ok for x in r["responses"]]
    out = {"scenarios": len(ok), "marine_minutes": round(minutes, 1),
           "roles": {k: round(n / total, 3) for k, n in roles.most_common()},
           "contacts": len(resp), "pulled_within_1s": round(sum(x["pulled"] for x in resp) / max(1, len(resp)), 3),
           "pull_delay_median_s": float(np.median([x["delay_s"] for x in resp if x["pulled"]])) if any(x["pulled"] for x in resp) else None}
    for role in ("pull_back", "retreat", "engage"):
        s = [x for x in st if x["role"] == role]
        out[role] = {"per_marine_minute": round(len(s) / max(minutes, 1e-9), 1),
                     "median_s": round(float(np.median([x["slices"] * SLICE / FPS for x in s])), 2) if s else None,
                     "median_tiles": float(np.median([x["tiles"] for x in s])) if s else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=int, default=114)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--controllers", nargs="+", default=["pro", "gary"], choices=["pro", "gary"])
    ap.add_argument("--version", default="v09")
    ap.add_argument("--style", type=int, default=1)
    ap.add_argument("--parallel", type=int, default=16)
    ap.add_argument("--out", help="also save the per-scenario rows here (JSON lines)")
    ap.add_argument("--save", help="save each play's replay in this folder")
    args = ap.parse_args()
    from eval.scenarios import scenarios_path
    scs = [json.loads(line) for line in scenarios_path(args.split).read_text(encoding="utf-8").splitlines()][:args.run]
    if args.save:
        Path(args.save).mkdir(parents=True, exist_ok=True)
    jobs = [(sc, c, args.version, args.style, args.save) for sc in scs for c in args.controllers]
    with ProcessPoolExecutor(args.parallel) as pool:
        rows = list(pool.map(_job, jobs, chunksize=len(args.controllers)))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    for c in args.controllers:
        name = f"gary {args.version}" if c == "gary" else c
        s = summarize([r for r in rows if r["controller"] == c])
        print(f"{name}: {s['marine_minutes']} marine-minutes fighting in {s['scenarios']} scenarios")
        print("  time by role:", ", ".join(f"{k} {v:.1%}" for k, v in s["roles"].items()))
        print(f"  zerg on a marine: {s['contacts']} times; pulled back within 1 s: {s['pulled_within_1s']:.1%}"
              f" (median after {s['pull_delay_median_s']} s)")
        for role in ("pull_back", "retreat", "engage"):
            v = s[role]
            print(f"  {role:10s} {v['per_marine_minute']} per marine-minute, median {v['median_s']} s, {v['median_tiles']} tiles")


if __name__ == "__main__":
    main()
