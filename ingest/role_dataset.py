"""The role dataset: what each of the pros' fighting units did, every half second, and what the fight
looked like to them (for gary/policy/roles.py; docs/CONTROL.md step 5).

Plays the home-defense scenarios (eval/scenarios.py, the train split) under the pro's own commands
and, every half second, takes the fight as the Terran sees it: the snapshot the fight models use
(gary/policy/fight.py), enemy units only where the Terran sees them, enemy HP hidden as it is to
Gary (full), plus each unit's motion over the last half second and how many enemies target it.
Each of the Terran's army units in it is labeled with its role over the next half second, as
eval/micro_roles.py labels marines (fire, close in, pull back, retreat, reposition, stand, other;
toward and away measured group against group); units not fighting are unlabeled (-1).

    python -m ingest.role_dataset --parallel 16
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from eval.micro_roles import ATTACK, CONTACT, GROUP, MOVE, NEAR, RETREAT_PX, SIDEWAYS, STAND, TARGETED_FOR
from gary.agent.fight import MAX_HP
from gary.policy.roles import EXTRA, MAX_UNITS, ROLES, extra_features

SLICE = 12


class Probe:
    def __init__(self):
        self.start = None
        self.flip = None
        self.snaps = []                    # (frame, rows, extra, tags)
        self.samples: dict[int, list] = {}  # unit -> [(frame, x, y, order, nearest enemy, targeted frame, axis)]
        self.targeted: dict[int, int] = {}
        self.prev: dict[int, tuple] = {}

    def __call__(self, game, slot: int) -> None:
        from gary.policy.army import is_building, is_worker, supply_x2
        from gary.policy.fight import snapshot
        obs = game.observe()
        f = obs["frame"]
        if self.start is None:
            self.start = f
            mw, mh = obs["map"]["w"], obs["map"]["h"]
            cc = next((u for u in obs["units"] if u["owner"] == slot and u["type"] == 106), None)
            starts = game.start_locations()
            main = min(starts, key=lambda s: math.dist((s["x"], s["y"]), (cc["x"], cc["y"]))) if cc and starts else {"x": 0, "y": 0}
            self.flip = (main["x"] > mw / 2, main["y"] > mh / 2)
        army = [u for u in obs["units"] if u["owner"] == slot and u["completed"] and supply_x2(u["type"]) > 0
                and not is_worker(u["type"]) and not is_building(u["type"])]
        enemy = [u for u in obs["units"] if u["owner"] not in (slot, 11) and supply_x2(u["type"]) > 0
                 and not is_worker(u["type"]) and not is_building(u["type"])]
        for z in enemy:
            if z.get("order_target"):
                self.targeted[z["order_target"]] = f
        for u in army:
            near = min((math.dist((u["x"], u["y"]), (z["x"], z["y"])) for z in enemy), default=1e9)
            axis = None
            theirs = [z for z in enemy if math.dist((u["x"], u["y"]), (z["x"], z["y"])) < GROUP]
            if theirs:
                ours = [m for m in army if math.dist((u["x"], u["y"]), (m["x"], m["y"])) < GROUP]
                ox, oy = sum(m["x"] for m in ours) / len(ours), sum(m["y"] for m in ours) / len(ours)
                zx, zy = sum(z["x"] for z in theirs) / len(theirs), sum(z["y"] for z in theirs) / len(theirs)
                d = math.dist((ox, oy), (zx, zy))
                axis = ((zx - ox) / d, (zy - oy) / d) if d > 1 else None
            self.samples.setdefault(u["tag"], []).append(
                (f, u["x"], u["y"], u["order"], near, self.targeted.get(u["tag"], -10**9), axis))
        if (f - self.start) % SLICE >= 2:
            return
        # the fight as the Terran sees it: enemies it sees, their HP hidden
        seen = []
        for u in obs["units"]:
            if u["owner"] not in (slot, 11):
                if not (u.get("visible_to", 0) >> slot) & 1:
                    continue
                u = {**u, "hp": MAX_HP.get(u["type"], 100), "shields": 0}
            seen.append(u)
        snap = snapshot({**obs, "units": seen}, slot, self.flip, supply_x2, is_worker, is_building,
                        24 * 32, MAX_UNITS)
        positions = {u["tag"]: (u["x"], u["y"]) for u in seen}
        if snap is not None:
            rows, tags, _ = snap
            self.snaps.append((f, rows, extra_features(rows, tags, positions, self.prev, self.flip), tags))
        self.prev = positions

    def labels(self) -> dict[tuple[int, int], str]:
        """(unit, slice start frame) -> role over that half second."""
        out = {}
        for tag, s in self.samples.items():
            slices = []                    # [f0, kind, x0, y0, x1, y1, threatened]
            k = 0
            while k < len(s):
                f0 = s[k][0] - (s[k][0] - self.start) % SLICE
                j = k
                while j < len(s) and s[j][0] < f0 + SLICE:
                    j += 1
                part = s[k:j]
                _, x0, y0, _, near0, tgt0, axis = part[0]
                x1, y1 = part[-1][1:3]
                threatened = part[0][0] - tgt0 <= TARGETED_FOR or near0 < CONTACT
                kind = None
                if near0 < NEAR:
                    orders = [p[3] for p in part]
                    if sum(o in ATTACK for o in orders) * 2 >= len(orders):
                        kind = "fire"
                    elif sum(o in MOVE for o in orders) * 2 >= len(orders):
                        along = (x1 - x0) * axis[0] + (y1 - y0) * axis[1] if axis else 0.0
                        kind = "close_in" if along > SIDEWAYS else "away" if along < -SIDEWAYS else "reposition"
                    elif sum(o in STAND for o in orders) * 2 >= len(orders):
                        kind = "stand"
                    else:
                        kind = "other"
                slices.append([f0, kind, x0, y0, x1, y1, threatened])
                k = j
            k = 0
            while k < len(slices):
                if slices[k][1] != "away":
                    k += 1
                    continue
                m = k
                while m < len(slices) and slices[m][1] == "away":
                    m += 1
                dist = math.dist(slices[k][2:4], slices[m - 1][4:6])
                role = "retreat" if dist >= RETREAT_PX else "pull_back" if slices[k][6] else "reposition"
                for q in range(k, m):
                    slices[q][1] = role
                k = m
            for sl in slices:
                if sl[1] is not None:
                    out[(tag, sl[0])] = sl[1]
        return out


def _job(sc: dict) -> dict:
    from eval.scenarios import run_one
    probe = Probe()
    row = run_one(sc, "pro", 45, "v09", 1, on_step=probe)
    labels = probe.labels() if not row.get("error") else {}
    rows, extra, lab, times, units = [], [], [], [], []
    for f, r, e, tags in probe.snaps:
        y = np.array([ROLES.index(labels[(t, f)]) if (t, f) in labels and r[i][1] == 1 else -1
                      for i, t in enumerate(tags)], np.int8)
        if (y >= 0).any():
            rows.append(r)
            extra.append(e)
            lab.append(y)
            times.append(f * 42 / 1000)
            units.append(tags)
    return {"sha1": sc["sha1"], "error": row.get("error"), "rows": rows, "extra": extra, "labels": lab, "time": times,
            "tags": units}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--parallel", type=int, default=16)
    ap.add_argument("--limit", type=int, help="only this many scenarios (to try it)")
    ap.add_argument("--out", help="default: data/interim/roles/roles_<split>.npz")
    args = ap.parse_args()
    from eval.scenarios import scenarios_path
    from ingest.inventory import data_root
    scs = [json.loads(line) for line in scenarios_path(args.split).read_text(encoding="utf-8").splitlines()]
    if args.limit:
        scs = scs[:args.limit]
    with ProcessPoolExecutor(args.parallel) as pool:
        results = list(pool.map(_job, scs, chunksize=1))
    n = sum(len(r["rows"]) for r in results)
    rows = np.zeros((n, MAX_UNITS, 11), np.int16)
    extra = np.zeros((n, MAX_UNITS, EXTRA), np.float32)
    labels = np.full((n, MAX_UNITS), -1, np.int8)
    unit_tags = np.zeros((n, MAX_UNITS), np.int32)
    count = np.zeros(n, np.int16)
    time_s = np.zeros(n, np.float32)
    game = np.zeros(n, np.int32)
    i = 0
    for gi, r in enumerate(results):
        for rr, e, y, t, tg in zip(r["rows"], r["extra"], r["labels"], r["time"], r["tags"]):
            k = min(len(rr), MAX_UNITS)
            rows[i, :k], extra[i, :k], labels[i, :k], count[i], time_s[i], game[i] = rr[:k], e[:k], y[:k], k, t, gi
            unit_tags[i, :k] = tg[:k]
            i += 1
    out = Path(args.out) if args.out else data_root() / "interim" / "roles" / f"roles_{args.split}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, rows=rows, extra=extra, labels=labels, count=count, time=time_s, game=game, tags=unit_tags,
                        sha1=np.array([r["sha1"] for r in results]))
    y = labels[labels >= 0]
    print(f"{len(results)} scenarios ({sum(1 for r in results if r['error'])} errors), {n} snapshots, {len(y)} labeled units -> {out}")
    print("  " + ", ".join(f"{ROLES[k]} {np.mean(y == k):.1%}" for k in range(len(ROLES))))


if __name__ == "__main__":
    main()
