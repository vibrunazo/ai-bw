"""Training data for Gary's army model: where a player sends their army, and how.

From every clean, re-simulated game, each player's orders to army units (at least 4 supply
selected; at most one per 2 s, the last) become samples of:

  grid    (9, 16, 16)  the map as that player knew it, on a 16x16 grid (values capped at 255):
                       0-2 own army supply x2 / workers / buildings,
                       3-5 the same for enemy units visible right now,
                       6 enemy buildings ever seen (most at once), 7 enemy army seen in the last 30 s,
                       8 the ordered group (its supply x2, in its cell)
  units   (228,)       own units by type (all), seen (228,) enemy units visible now
  eco     (4,)         minerals, gas, supply used x2, supply max x2;  frame
  target  cell 0-255 where the group was sent;  kind 0 move, 1 attack (attack-move or a target)
  sup     supply x2 of the ordered group

Coordinates are mirrored so the player's own main is always in the top-left quadrant (maps and
start positions differ; the model learns "toward the enemy", not "toward the top-right").
Far moves (attacks, retreats, defending) and short ones (holding, fighting in place) are both
kept, so the model also learns when the army stays where it is.

Commitments (v2): orders alone say where a pro sends a group, not when they leave it be. So every
TICK seconds each of the player's army clusters (Gary's clustering: the densest cell of their army
and the cells within 2, up to 3 clusters, at least MIN_SUP) is a sample too, labeled by what the
player did with it in the next TICK seconds: keep (no order sending a group there 2+ cells: shorter
orders are fighting or standing in place), engage (an order sending it 2+ cells to a cell next to
enemy presence the player knew of: enemy army in sight or seen in 30 s, enemy buildings ever seen)
or move (any other order of 2+ cells). Each with the cluster's commitment:
its last order's label (engage or move, none if none in COMMIT_S), that order the last one whose
path (from where the group stood to where it was sent) passes within 2 cells of the cluster: the
commitment follows the group on its way. And that order's age. Pros re-decide on events, not a timer: the model learns the triggers from the state.

  tick_grid (9, 256)  as an order's grid, channel 8 the cluster's supply at its center
  tick_frame, tick_units, tick_seen, tick_eco  as for orders
  tick_label  0 keep, 1 engage, 2 move;  tick_target  the order's cell (-1 keep)
  tick_commit 0 none, 1 engage, 2 move;  tick_age  seconds since that order
  tick_loss (256,)    the player's own losses in the last LOSS_S seconds, where they died (a worker 2, a
                      fighter its supply x2, a building 4; capped at 255): what a pro reacts to, a raid on
                      the mineral line, that the grid of where units stand doesn't show (v3; c10: Gary-Z's
                      army kept its commitment while its drones went from 13 to 2)
  tick_static (256,)  the enemy's ground static defense the player knew of, where it stands (sunken, cannon,
                      bunker, each STATIC_SUPPLY: gary/policy/army.py; capped at 255), known from the first look
                      at its cell (an enemy building in sight there) until it dies: the other buildings' channel
                      counts a sunken as an Evolution Chamber, so a model couldn't learn when pros attack into
                      static defense (v4; the static guard, a stop-gap in AGENTS.md)
  tick_known (256,)   the enemy buildings the player knew of still standing, by cell: as tick_static, from the
                      first look at the cell after it was begun until it dies (channel 6 never forgets one, so
                      a model couldn't tell no building left: the scripted hunt's trigger)
  tick_unseen (256,)  seconds / UNSEEN_PER (gary/policy/army.py) since the player's own units were last in each cell,
                      capped at 255 (the bases not seen lately: the hunt's search) (v5)

Output: data/interim/army/v5/<sha1[:2]>/<sha1>.npz and data/interim/army/v5/index.jsonl. Needs a
gary_resim whose "gone" events carry the owner and the place (resim/gary_resim.cpp).

Usage:
  python ingest/army_dataset.py --matchup TvZ --limit 200
  python ingest/army_dataset.py --matchup TvZ
  python ingest/army_dataset.py --report
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # (the repo: gary's unit tables)
from inventory import data_root
from eval import monitor, priority
from macro_dataset import RACES, TOWN_HALLS, run_resim, select

VERSION = "v5"
GRID, CELLS, CHANNELS = 16, 256, 9
MIN_SUP = 8                  # at least 4 supply (x2) selected
WINDOW = 48                  # at most one sample per player per 2 s
RECENT = 30                  # channel 7 remembers enemy army for 30 s
UNSEEN_PER = 4               # tick_unseen's unit, seconds (gary/policy/army.py UNSEEN_PER)
KIND = {"move": 0, "amove": 1, "attack": 1, "patrol": 1}
TICK = 5                     # seconds (snapshots) between commitment samples
LOSS_S = 10                  # own losses counted this long (gary/policy/army.py LOSS_S)
COMMIT_S = 120               # an order older than this is no commitment any more
CLUSTERS = 3
NEAR, MOVED = 1, 2           # engage: the target this close to enemy presence, the group sent this far


def is_building(unit_type: int) -> bool:
    from gary.policy.army import is_building as building
    return building(unit_type)


def static_value(unit_type: int) -> int:
    """A static defense's weight on the static channel (gary/policy/army.py STATIC_SUPPLY; 0 if none)."""
    from gary.policy.army import STATIC_SUPPLY
    return STATIC_SUPPLY.get(unit_type, 0)


def loss_value(unit_type: int) -> int:
    """A loss's weight on the losses grid: a worker 2, a fighter its supply x2, a building 4 (else 0); as
    gary/policy/army.py's loss_value, which the army tracker uses in play."""
    from gary.policy.army import loss_value as value
    return value(unit_type)


def _add_static(static: np.ndarray, frames: np.ndarray, g: np.ndarray, f_done: int, f_gone: int, c: int,
                v: int) -> None:
    """static[t, c] += v from the first snapshot after f_done with an enemy building in sight in cell c (g's
    channel 5) to the last before f_gone."""
    t0 = int(np.searchsorted(frames, f_done))
    t1 = int(np.searchsorted(frames, f_gone))
    if t0 >= t1:
        return
    seen = np.nonzero(g[t0:t1, 5, c] > 0)[0]
    if len(seen):
        static[t0 + int(seen[0]):t1, c] += v


def cheb(a: int, b: int) -> int:
    return max(abs(a // GRID - b // GRID), abs(a % GRID - b % GRID))


def clusters(army: np.ndarray) -> list[tuple[int, int]]:
    """(center cell, supply x2) of up to CLUSTERS army clusters on one grid row (supply x2 per cell):
    the densest cell and every cell within 2 of it, then again with what's left (gary/agent/army.py)."""
    left = army.astype(np.int64).copy()
    out = []
    while len(out) < CLUSTERS and left.sum() >= MIN_SUP:
        c = int(left.argmax())
        cy, cx = divmod(c, GRID)
        y0, y1, x0, x1 = max(0, cy - 2), min(GRID, cy + 3), max(0, cx - 2), min(GRID, cx + 3)
        sq = left.reshape(GRID, GRID)
        sup = int(sq[y0:y1, x0:x1].sum())
        sq[y0:y1, x0:x1] = 0
        if sup >= MIN_SUP:
            out.append((c, sup))
    return out


def path_dist(c: int, a: int, b: int) -> float:
    """Cells (Chebyshev) from cell c to the straight path from cell a to cell b: a group ordered from a to b
    is somewhere on it."""
    cy, cx = divmod(c, GRID)
    ay, ax = divmod(a, GRID)
    by, bx = divmod(b, GRID)
    return min(max(abs(cy - (ay + (by - ay) * s)), abs(cx - (ax + (bx - ax) * s))) for s in np.linspace(0, 1, 9))


def engaging(grid: np.ndarray, target: int, group: int) -> bool:
    """An order into the enemy (train/engage.py's label): 2+ cells away, to a cell next to enemy army in
    sight or seen lately or enemy buildings ever seen."""
    if cheb(target, group) < MOVED:
        return False
    enemy = ((grid[3] > 0) | (grid[6] > 0) | (grid[7] > 0)).reshape(GRID, GRID)
    ty, tx = divmod(target, GRID)
    return bool(enemy[max(0, ty - NEAR):ty + NEAR + 1, max(0, tx - NEAR):tx + NEAR + 1].any())


def out_dir() -> Path:
    return data_root() / "interim" / "army" / VERSION


def flip_cells(a: np.ndarray, fx: bool, fy: bool) -> np.ndarray:
    """Mirror (..., 256) cell arrays so the player's main is top-left."""
    g = a.reshape(*a.shape[:-1], GRID, GRID)
    if fy:
        g = g[..., ::-1, :]
    if fx:
        g = g[..., :, ::-1]
    return np.ascontiguousarray(g).reshape(a.shape)


def cell_of(x: float, y: float, mw: int, mh: int, fx: bool, fy: bool) -> int:
    gx = min(GRID - 1, max(0, int(x * GRID // mw)))
    gy = min(GRID - 1, max(0, int(y * GRID // mh)))
    if fx:
        gx = GRID - 1 - gx
    if fy:
        gy = GRID - 1 - gy
    return gy * GRID + gx


def build_game(rec: dict) -> dict:
    row = {"sha1": rec["sha1"], "rel_path": rec["rel_path"], "matchup": rec["matchup"]}
    t0 = time.time()
    try:
        lines = run_resim(rec["rel_path"])
    except Exception as e:  # noqa: BLE001 - one bad replay must not stop the batch
        row.update(ok=False, error=str(e)[:300], interrupted=type(e).__name__ == "ResimInterrupted")
        return row
    header = next(r for r in lines if r["type"] == "header")
    mw, mh = header.get("map_w", 4096), header.get("map_h", 4096)
    snaps = [r for r in lines if r["type"] == "snapshot"]
    accepted: dict[int, int] = {}
    for s in snaps:
        for p in s["players"]:
            accepted[p["slot"]] = accepted.get(p["slot"], 0) + p["accepted"]
    slots = sorted(sorted(accepted, key=lambda s: -accepted[s])[:2])
    if len(slots) != 2 or not snaps:
        row.update(ok=False, error="not two players")
        return row
    starts = {}
    for r in lines:
        if r["type"] == "event" and r["frame"] <= 1 and r.get("unit") in TOWN_HALLS and r["slot"] in slots:
            starts.setdefault(r["slot"], (r["x"], r["y"]))
    if len(starts) != 2:
        row.update(ok=False, error="no start town halls")
        return row

    T = len(snaps)
    frames = np.array([s["frame"] for s in snaps], dtype=np.int32)
    per = {}
    for s in slots:
        per[s] = {"g": np.zeros((T, 6, CELLS), np.uint16), "units": np.zeros((T, 228), np.uint16),
                  "seen": np.zeros((T, 228), np.uint16), "eco": np.zeros((T, 4), np.int32)}
    for t, snap in enumerate(snaps):
        for p in snap["players"]:
            d = per.get(p["slot"])
            if d is None:
                continue
            g = p.get("g", [])
            flat = d["g"][t].reshape(-1)
            flat[np.array(g[0::2], dtype=np.int64)] = np.array(g[1::2], dtype=np.uint16)
            for k, (n_all, _) in p["units"].items():
                d["units"][t, int(k)] = n_all
            for k, n in p.get("seen", {}).items():
                d["seen"][t, int(k)] = n
            d["eco"][t] = (p["minerals"], p["gas"], round(p["supply_used"] * 2), round(p["supply_max"] * 2))

    out = {}
    counts = {}
    for i, s in enumerate(slots):
        fx, fy = starts[s][0] > mw / 2, starts[s][1] > mh / 2
        d = per[s]
        g = flip_cells(d["g"], fx, fy)
        seen_bld = np.maximum.accumulate(g[:, 5], axis=0)
        recent = np.zeros_like(g[:, 3])
        for t in range(T):           # enemy army seen in the last RECENT seconds (most at once)
            recent[t] = g[max(0, t - RECENT):t + 1, 3].max(0)
        orders = [r for r in lines if r["type"] == "army" and r["slot"] == s and r["sup"] >= MIN_SUP
                  and r["kind"] in KIND and 0 <= r["x"] < mw and 0 <= r["y"] < mh]
        picked = {}
        for r in orders:              # the last order in each 2-second window
            picked[r["frame"] // WINDOW] = r
        rows = sorted(picked.values(), key=lambda r: r["frame"])
        n = len(rows)
        grid = np.zeros((n, CHANNELS, CELLS), np.uint16)    # stored as uint8 (capped at 255)
        tidx = np.zeros(n, np.int32)
        target = np.zeros(n, np.int16)
        kind = np.zeros(n, np.int8)
        sup = np.zeros(n, np.int16)
        for j, r in enumerate(rows):
            t = max(0, min(T - 1, int(np.searchsorted(frames, r["frame"], side="right")) - 1))
            tidx[j] = t
            grid[j, :6] = g[t]
            grid[j, 6] = seen_bld[t]
            grid[j, 7] = recent[t]
            grid[j, 8, cell_of(r["cx"], r["cy"], mw, mh, fx, fy)] = r["sup"]
            target[j] = cell_of(r["x"], r["y"], mw, mh, fx, fy)
            kind[j] = KIND[r["kind"]]
            sup[j] = r["sup"]
        # commitments: every TICK s each army cluster, what the player did with it next (see the docstring)
        fx_ = lambda r: cell_of(r["x"], r["y"], mw, mh, fx, fy)
        gx_ = lambda r: cell_of(r["cx"], r["cy"], mw, mh, fx, fy)
        known = []                                   # (frame, group cell, label) of orders so far
        order_t = [max(0, min(T - 1, int(np.searchsorted(frames, r["frame"], side="right")) - 1)) for r in orders]
        labeled = []
        for r, t in zip(orders, order_t):
            base = np.concatenate([g[t], seen_bld[t][None], recent[t][None]])
            if cheb(fx_(r), gx_(r)) >= MOVED:          # (shorter: fighting or standing in place, a keep)
                labeled.append((r["frame"], gx_(r), fx_(r), 1 if engaging(base, fx_(r), gx_(r)) else 2))
        k = 0
        tk: dict[str, list] = {n: [] for n in ("grid", "t", "label", "target", "commit", "age")}
        for t in range(0, T, TICK):
            f0, f1 = frames[t], frames[t] + TICK * 24
            while k < len(labeled) and labeled[k][0] < f0:
                known.append(labeled[k])
                k += 1
            nxt = [o for o in labeled[k:] if o[0] < f1]
            for c, csup in clusters(g[t, 0]):      # (not `sup`: that is the orders' array, saved below)
                mine = [o for o in nxt if cheb(o[1], c) <= 2]
                past = [o for o in known if path_dist(c, o[1], o[2]) <= 2 and f0 - o[0] <= COMMIT_S * 24]
                at = np.zeros((CHANNELS, CELLS), np.uint16)
                at[:6] = g[t]
                at[6] = seen_bld[t]
                at[7] = recent[t]
                at[8, c] = csup
                tk["grid"].append(np.minimum(at, 255).astype(np.uint8))
                tk["t"].append(t)
                tk["label"].append(mine[0][3] if mine else 0)
                tk["target"].append(mine[0][2] if mine else -1)
                tk["commit"].append(past[-1][3] if past else 0)
                tk["age"].append((f0 - past[-1][0]) / 24 if past else COMMIT_S)
        # own losses: each death of this player's units in the LOSS_S before the tick, at its place
        deaths = [(r["frame"], cell_of(r["x"], r["y"], mw, mh, fx, fy), loss_value(r["unit"])) for r in lines
                  if r["type"] == "event" and r["ev"] == "gone" and r.get("slot") == s and "x" in r
                  and 0 <= r["x"] < mw and 0 <= r["y"] < mh]
        loss = np.zeros((len(tk["t"]), CELLS), np.int32)
        for j, t in enumerate(tk["t"]):
            for f, c, v in deaths:
                if frames[t] - LOSS_S * 24 < f <= frames[t]:
                    loss[j, c] += v
        out[f"p{i}_tick_loss"] = np.minimum(loss, 255).astype(np.uint8)
        out[f"p{i}_flip"] = np.array([fx, fy, mw, mh], np.int32)     # (the mirroring, to place other labels: v5)
        # the enemy's static defense: each from its "done" to its "gone" (by tag), known once the player sees an
        # enemy building in its cell after it is done
        static = np.zeros((T, CELLS), np.int32)
        built = {}
        for r in lines:
            if r["type"] != "event" or r.get("slot") in (s, None) or r["slot"] not in slots:
                continue
            if r["ev"] == "done" and static_value(r["unit"]) and "x" in r:
                built[r["tag"]] = (r["frame"], cell_of(r["x"], r["y"], mw, mh, fx, fy), static_value(r["unit"]))
            elif r["ev"] == "gone" and r["tag"] in built:
                f, c, v = built.pop(r["tag"])
                _add_static(static, frames, g, f, r["frame"], c, v)
        for f, c, v in built.values():
            _add_static(static, frames, g, f, frames[-1] + 1, c, v)
        out[f"p{i}_tick_static"] = np.minimum(static[np.array(tk["t"], np.int64)], 255).astype(np.uint8)
        # the enemy's buildings, each from its first "start" or "done" (a morph keeps its tag) to its "gone"
        known = np.zeros((T, CELLS), np.int32)
        begun = {}
        for r in lines:
            if r["type"] != "event" or r.get("slot") in (s, None) or r["slot"] not in slots:
                continue
            if r["ev"] in ("start", "done") and is_building(r["unit"]) and "x" in r and r["tag"] not in begun:
                begun[r["tag"]] = (r["frame"], cell_of(r["x"], r["y"], mw, mh, fx, fy), 1)
            elif r["ev"] == "gone" and r["tag"] in begun:
                f, c, v = begun.pop(r["tag"])
                _add_static(known, frames, g, f, r["frame"], c, v)
        for f, c, v in begun.values():
            _add_static(known, frames, g, f, frames[-1] + 1, c, v)
        out[f"p{i}_tick_known"] = np.minimum(known[np.array(tk["t"], np.int64)], 255).astype(np.uint8)
        # seconds since own units were last in each cell (snapshots a second apart)
        last = np.zeros(CELLS, np.int64)
        unseen = np.zeros((T, CELLS), np.int64)
        present = g[:, :3].sum(1) > 0
        for t in range(T):
            last[present[t]] = t
            unseen[t] = t - last
        out[f"p{i}_tick_unseen"] = np.minimum(unseen[np.array(tk["t"], np.int64)] // UNSEEN_PER, 255).astype(np.uint8)
        ti = np.array(tk["t"], np.int64)
        out[f"p{i}_tick_grid"] = np.array(tk["grid"], np.uint8).reshape(-1, CHANNELS, CELLS)
        out[f"p{i}_tick_frame"] = frames[ti] if len(ti) else np.zeros(0, np.int32)
        out[f"p{i}_tick_units"] = d["units"][ti]
        out[f"p{i}_tick_seen"] = d["seen"][ti]
        out[f"p{i}_tick_eco"] = d["eco"][ti]
        out[f"p{i}_tick_label"] = np.array(tk["label"], np.int8)
        out[f"p{i}_tick_target"] = np.array(tk["target"], np.int16)
        out[f"p{i}_tick_commit"] = np.array(tk["commit"], np.int8)
        out[f"p{i}_tick_age"] = np.array(tk["age"], np.float32)
        out[f"p{i}_grid"] = np.minimum(grid, 255).astype(np.uint8)
        out[f"p{i}_frame"] = frames[tidx]
        out[f"p{i}_units"] = d["units"][tidx]
        out[f"p{i}_seen"] = d["seen"][tidx]
        out[f"p{i}_eco"] = d["eco"][tidx]
        out[f"p{i}_target"] = target
        out[f"p{i}_kind"] = kind
        out[f"p{i}_sup"] = sup
        counts[i] = n

    path = out_dir() / rec["sha1"][:2] / f"{rec['sha1']}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, **out)
    path.write_bytes(buf.getvalue())
    names = {p["slot"]: p for p in header["players"]}
    players = [{"slot": s, "name": names.get(s, {}).get("name", ""),
                "race": RACES.get(names.get(s, {}).get("race"), "?"), "samples": counts[i],
                "ticks": int(len(out[f"p{i}_tick_label"]))}
               for i, s in enumerate(slots)]
    row.update(ok=True, map=header.get("map"), map_w=mw, map_h=mh, end_frame=header.get("end_frame"),
               players=players, bytes=path.stat().st_size, seconds=round(time.time() - t0, 2))
    return row


def load_index() -> dict[str, dict]:
    path = out_dir() / "index.jsonl"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["sha1"]: r for r in map(json.loads, f)}


def report() -> None:
    rows = list(load_index().values())
    ok = [r for r in rows if r.get("ok")]
    if not rows:
        print("no games yet")
        return
    print(f"games {len(ok)} ok, {len(rows) - len(ok)} failed")
    print(f"army-order samples {sum(p['samples'] for r in ok for p in r['players']):,}")
    print(f"disk {sum(r['bytes'] for r in ok) / 1e6:.0f} MB, resim {sum(r['seconds'] for r in ok) / max(1, len(ok)):.1f} s/game")
    for r in [r for r in rows if not r.get("ok")][:5]:
        print("  failed:", r["rel_path"], r.get("error"))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matchup", default="TvZ")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--source", help="only games from this raw source folder (e.g. cwal), e.g. with --force after they were re-simulated")
    ap.add_argument("--report", action="store_true")
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    if args.report:
        report()
        return
    done = {} if args.force else load_index()
    todo = [r for r in select(args.matchup, args.source) if r["sha1"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} games to build ({len(done)} already done)", flush=True)
    out_dir().mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out_dir() / "index.jsonl", "a", encoding="utf-8") as index, \
            ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(build_game, r) for r in todo]
        for n, fut in enumerate(monitor.track(as_completed(futures), len(futures), "games built"), 1):
            res = fut.result()
            if res.get("interrupted"):          # (killed, not failed: left out, so the next run builds it)
                continue
            index.write(json.dumps(res, ensure_ascii=False) + "\n")
            if n % 100 == 0 or n == len(todo):
                index.flush()
                print(f"{n}/{len(todo)}  {time.time() - t0:.0f} s", flush=True)
    report()


if __name__ == "__main__":
    main()
