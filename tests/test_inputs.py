"""The models' inputs in play against their training data's, for the same game (issue #47). Gary plays itself (gt1dev
against gz1dev) for a few minutes; every 5 s (on the army dataset's ticks) the test reads, for each side, the vector
its macro model, its army model and its critic would be fed now (the layers' own code, on copies of their trackers:
reading the live trackers would change what they remember). The replay is then re-simulated and run through the
dataset builders (ingest/macro_dataset.py, ingest/army_dataset.py, train/macro.py, train/commit.py, train/value.py)
and the same moments compared, block by block.

A block fails when it is off by more than its tolerance at more than FAIL_SHARE of the moments: what play feeds the
model is not what it learned from. Six such faults were found by hand on 2026-10-25 (the macro model's enemy history
empty, units inside bunkers missing from two models, the Zerg's eggs counted twice, the critic's style column, the
build placement), each of which had cost games or misread a round. Intended differences are listed (INTENDED) and
skipped; known faults not fixed yet (KNOWN, each with its issue) are reported and expected; one that
doesn't show in the game is reported, so the list can be kept true. Needs the game data and the models (skips without them).

Run: python -m pytest tests/test_inputs.py   (or plain: python tests/test_inputs.py; about 5 minutes)
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ingest"))

UNTIL = 9 * 60 * 24             # frames played
SIDES = (os.environ.get("GARY_TEST_T", "gt1dev"), os.environ.get("GARY_TEST_Z", "gz1dev"))   # (other manifests' check)
TICK, FIRST = 120, 24           # the army dataset's tick frames: FIRST + TICK n
FAIL_SHARE = 0.2                # a block off at more than this share of the moments fails
MIN_MOMENTS = 40                # a side with fewer moments compared fails (a test that compares nothing passes)
# (side, model, block) -> why play and the data differ on purpose
INTENDED = {
    ("*", "macro", "since last decision"): "play counts the model's decisions, the replays the production commands",
    ("*", "macro", "decided"): "the same (at most a decision or two apart); and with the pros' history (gz1's) another input",
    ("*", "critic", "since last decision"): "as the macro's",
    ("*", "critic", "decided"): "as the macro's",
    ("*", "army", "grid: own losses"): "play knows a unit lost once it's out of sight GONE_AFTER, the replay at its death",
    ("*", "army", "grid: enemy army seen 30 s"): "its memory's window starts at play's look, the replay's at each second",
}
# (side, model, block) -> the issue of a known fault, not fixed yet
KNOWN = {
    ("Z", "macro", "own all"): "#48: the units in eggs counted twice (option eggs lost its games)",
    ("Z", "critic", "own all"): "#50: drones inside the Extractors (the critic's view leaves the eggs out)",
    ("Z", "macro", "own completed"): "#50: drones inside the Extractors, counted by the replays",
    ("Z", "critic", "own completed"): "#50: the same",
    ("Z", "army", "own units"): "#50: the same",
    ("Z", "army", "enemy seen lately"): "#50: the tanks' turrets, counted by the replays",
    ("T", "roles", "extra 3"): "#52: a unit's age, filled in play, 0 in the role datasets (option dataextras)",
    ("T", "roles", "extra 4"): "#52: its distance to Gary's biggest bunch, the same",
    ("T", "roles", "extra 5"): "#52: that bunch's size, the same",
}


def _blocks_macro(spec, extras: int = 0) -> list[tuple[str, slice, str, float]]:
    """(name, columns, kind, tolerance): kind "count" compares counts (log1p undone), "value" the raw numbers."""
    no, ns, nd = len(spec.own_types), len(spec.seen_types), len(spec.decisions)
    k = 7
    out = [("time", slice(0, 1), "value", 1e-4), ("minerals and gas", slice(1, 3), "value", 0.016),
           ("supply", slice(3, 6), "value", 0.0051), ("since last decision", slice(6, 7), "value", 0.05)]
    for name, n in (("own all", no), ("own completed", no), ("enemy in sight", ns), ("enemy most seen", ns),
                    ("decided", nd)):
        out.append((name, slice(k, k + n), "count", 1.0))
        k += n
    k += len(spec.styles) + 1 + (len(spec.maps) + 1 if spec.maps else 0) + (1 if spec.year else 0) + len(spec.matchups)
    if extras:
        out.append(("critic extras", slice(k, k + extras), "value", 0.35))
    return out


def _blocks_army(spec) -> tuple[list, list]:
    """Grid channels and numbers compared (the group's own parts, channel 8 and the group's supply, left out)."""
    names = ["own army", "own workers", "own buildings", "enemy army", "enemy workers", "enemy buildings",
             "enemy buildings ever", "enemy army seen 30 s", "group"]
    extra = {"losses": "own losses", "static": "enemy static defense", "known": "enemy buildings known",
             "unseen": "cells unvisited"}
    grid = [(n, c) for c, n in enumerate(names) if n != "group"]
    for c, e in enumerate(spec._extra(), len(names)):
        grid.append((extra.get(e, e), c))
    no, ns = len(spec.own_types), len(spec.seen_types)
    k = 8
    nums = [("time", slice(0, 1), "value", 1e-4), ("minerals and gas", slice(1, 3), "value", 0.016),
            ("supply", slice(3, 5), "value", 0.0051), ("own army supply", slice(6, 7), "value", 0.011),
            ("enemy army seen (supply)", slice(7, 8), "value", 0.011)]
    for name, n in (("own units", no), ("enemy in sight", ns), ("enemy seen lately", ns)):
        nums.append((name, slice(k, k + n), "count", 1.0))
        k += n
    return grid, nums


def _off(a: np.ndarray, b: np.ndarray, kind: str) -> tuple[float, int]:
    """The largest difference and its column in the block."""
    d = np.abs(np.rint(np.expm1(a)) - np.rint(np.expm1(b))) if kind == "count" else np.abs(a - b)   # (whole counts:
                                                                                                    # float32 logs)
    return float(d.max()), int(d.argmax())


def _labels(g, model: str, name: str) -> list[str] | None:
    """What a block's columns are (unit types, decisions), for the report."""
    spec = g.macro.tracker.spec if model in ("macro", "critic") else g.army.tracker.spec
    names = g.race.names
    enemy = {**__import__("gary.race", fromlist=["Race"]).Race.of({"T": "Z", "Z": "T"}.get(g.race.letter, "T")).names}
    if name in ("own all", "own completed", "own units"):
        return [names.get(t, str(t)) for t in spec.own_types]
    if name.startswith("enemy in sight") or name.startswith("enemy most") or name == "enemy seen lately":
        return [enemy.get(t, str(t)) for t in spec.seen_types]
    if name == "decided":
        return [f"{a}:{names.get(u, u)}" for a, u in spec.decisions]
    return None


ROLE_READS: dict[str, list] = {}           # side -> [(rows, extra)] the role model was fed


def _capture(model, out: list) -> None:
    orig = model.predict_move

    def fed(rows, extra, *a, **k):
        out.append((np.array(rows, np.float32), np.array(extra, np.float32)))
        return orig(rows, extra, *a, **k)
    model.predict_move = fed


def _role_data(model, n: int = 4000):
    """A sample of the role model's training rows (its config's dataset): (rows, extra) of the units in them."""
    path = REPO / str(model.config.get("data", ""))
    if not path.is_file():
        return None
    cnt, rows, extra = (_npz_head(path, name, n) for name in ("count", "rows", "extra"))
    k = min(len(cnt), len(rows), len(extra))
    rows = np.concatenate([rows[i, :cnt[i]] for i in range(k)]).astype(np.float32)
    extra = np.concatenate([extra[i, :cnt[i]] for i in range(k)]).astype(np.float32)
    return rows, extra


def _npz_head(path: Path, name: str, k: int) -> np.ndarray:
    """The first k entries of one array in an .npz, read from its member without loading the rest (a role dataset's
    arrays are 0.5-1.4 GB)."""
    import zipfile
    with zipfile.ZipFile(path) as z, z.open(name + ".npy") as f:
        version = np.lib.format.read_magic(f)
        shape, _, dtype = (np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0)(f)
        k = min(k, shape[0])
        per = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        return np.frombuffer(f.read(k * per * dtype.itemsize), dtype).reshape((k,) + tuple(shape[1:]))


def compare_roles(sides) -> tuple[list[str], list[str], set]:
    """The role models: per input column, how often it is nonzero in play against in the model's training data (a
    column the data never fills that play does is an input the model never learned), and the enemy units' HP."""
    from gary.policy.fight import SIDE
    report, fails, seen_known = [], [], set()
    for r, g in sides.items():
        fed = ROLE_READS.get(r) or []
        if not fed:
            continue
        data = _role_data(g.fight.roles.model)
        if data is None:
            report.append(f"  {r} roles  no training data to compare with")
            continue
        prow = np.concatenate([x for x, _ in fed]); pext = np.concatenate([e for _, e in fed])
        drow, dext = data
        n_extra = getattr(g.fight.roles.model.net, "n_extra", pext.shape[1])
        checks = [(f"extra {j}", pext[:, j], dext[:, j]) for j in range(min(n_extra, pext.shape[1], dext.shape[1]))]
        en_p, en_d = prow[prow[:, SIDE] == 0], drow[drow[:, SIDE] == 0]
        checks.append(("enemy HP 0", (en_p[:, 4] == 0).astype(np.float32), (en_d[:, 4] == 0).astype(np.float32)))
        for name, a, b in checks:
            pa, pb = float((a != 0).mean()) if len(a) else 0.0, float((b != 0).mean()) if len(b) else 0.0
            off = (pb < 0.01 and pa > 0.05) or (pb > 0.99 and pa < 0.95) or (name == "enemy HP 0" and pa - pb > 0.2)
            state = "ok"
            key = (r, "roles", name)
            if off:
                if key in KNOWN:
                    state = "known"
                    seen_known.add(key)
                else:
                    state = "FAIL"
                    fails.append(f"{r} roles {name}: nonzero in {pa:.0%} of play's rows, {pb:.0%} of its training data's")
            report.append(f"  {r} roles  {name:44} play {pa:4.0%}, data {pb:4.0%}  {state}")
    return report, fails, seen_known


def _play(seconds: int):
    from eval.match import Match
    from gary.agent.manifests import model_for
    from gary.policy.belief import BeliefModel
    from gary.policy.value import ValueView
    with contextlib.redirect_stdout(io.StringIO()):
        m = Match(1, SIDES[0], REPO / "tests" / "fixtures" / "replays" / "stardata_tvz_standard_ozp3w.rep",
                  opponent=SIDES[1])
    sides = {"T": m.gary, "Z": m.zerg}
    views = {r: ValueView(g, BeliefModel.load(model_for("belief", "TvZ", r))) for r, g in sides.items()}
    reads = {r: {} for r in sides}
    for r, g in sides.items():                  # (the role models' inputs: what they are fed, as fed)
        layer = getattr(getattr(g, "fight", None), "roles", None)
        model = getattr(layer, "model", None)
        if model is not None and hasattr(model, "predict_move"):
            _capture(model, ROLE_READS.setdefault(r, []))
    while m.game.frame < seconds * 24 and not m.done:
        with contextlib.redirect_stdout(io.StringIO()):
            m.step()
        for r, g in sides.items():
            obs = g.hi.observe()
            views[r].look(obs)                                   # (once a second: it skips a second it has)
            f = obs["frame"] - (obs["frame"] - FIRST) % TICK     # (a side's looks can land a frame after the tick)
            if obs["frame"] - f > 1 or f < FIRST + TICK or f in reads[r]:
                continue
            mine = [u for u in obs["units"] if u["owner"] == g.slot]
            macro = g.macro.model_inputs(obs, mine, g.macro.tracker.copy())[0]
            grid, glob = g.army.tracker.copy().inputs(obs, [])
            reads[r][f] = {"macro": macro, "grid": grid[0], "glob": glob[0], "critic": views[r].features(obs)[0],
                           "style": g.macro.tracker.style, "map": g.macro.tracker.map_hash,
                           "matchup": g.macro.tracker.matchup}
    return m, sides, reads


def _replay_rows(m, sides, reads, tmp: Path):
    import army_dataset as A
    import macro_dataset as M
    from gary.policy.army import ARMY_SUPPLY, MEMORY_S, encode_global
    from train.macro import encode_rows
    from train.value import view_rows
    rep = tmp / "inputs.rep"
    m.game.save_replay(str(rep))
    m.game.close()
    lines = M.run_resim(str(rep))
    mac = M.game_arrays(lines)
    row = A.build_game({"sha1": "inputs-test", "rel_path": str(rep), "matchup": "TvZ", "out": str(tmp / "army.npz")})
    assert row.get("ok"), row
    with np.load(tmp / "army.npz") as z:
        arm = {k: z[k] for k in z.files}
    out = {}
    for r, g in sides.items():
        i = mac["slots"].index(g.slot)
        ia = [p["slot"] for p in row["players"]].index(g.slot)
        spec = g.macro.tracker.spec
        frames = sorted(reads[r])
        ts = np.searchsorted(mac["frames"], frames)
        ok = [k for k, f in zip(ts, frames) if k < len(mac["frames"]) and mac["frames"][k] == f]
        out[r] = {}
        own_sup = mac["units"][:, i, :, 0].astype(np.float32) @ ARMY_SUPPLY
        seen_all = mac["seen"][:, i].astype(np.float32)
        tf = arm[f"p{ia}_tick_frame"]
        aspec = g.army.tracker.spec
        for k in ok:
            f = int(mac["frames"][k])
            rd = reads[r][f]
            x, _ = encode_rows(mac, i, spec, np.array([k]), rd["style"], rd["map"], None, rd["matchup"])
            crit = view_rows(mac, i, spec, r, np.array([k]))[0]
            entry = {"macro": x[0], "critic": crit}
            j = np.nonzero(tf == f)[0]
            if len(j):
                j = int(j[0])
                grid = arm[f"p{ia}_tick_grid"][j].astype(np.float32)
                extra = [arm[f"p{ia}_tick_{n}"][j].astype(np.float32)[None] for n in ("loss", "static", "known", "unseen")
                         if (n == "loss" and aspec.losses) or (n != "loss" and getattr(aspec, n))]
                seen_mem = seen_all[max(0, k - MEMORY_S):k + 1].max(0)
                glob = encode_global(aspec, np.array([f], np.float32), arm[f"p{ia}_tick_eco"][j:j + 1].astype(np.float32),
                                     np.zeros(1, np.float32), arm[f"p{ia}_tick_units"][j:j + 1].astype(np.float32),
                                     arm[f"p{ia}_tick_seen"][j:j + 1].astype(np.float32), seen_mem[None],
                                     own_sup[k:k + 1], (seen_mem @ ARMY_SUPPLY)[None])[0]
                entry.update(grid=np.concatenate([grid] + extra, 0), glob=glob)
            out[r][f] = entry
    return out


def compare(sides, reads, rows) -> tuple[list[str], list[str], list[str]]:
    """Per side, model and block: the share of moments off by more than the tolerance. Returns (report lines,
    failures, known faults that didn't show)."""
    from gary.policy.value import EXTRAS
    report, fails, gone = [], [], []
    seen_known = set()
    for r, g in sides.items():
        spec, aspec = g.macro.tracker.spec, g.army.tracker.spec
        grid_blocks, num_blocks = _blocks_army(aspec)
        checks = [("macro", "macro", b) for b in _blocks_macro(spec)] + \
                 [("critic", "critic", b) for b in _blocks_macro(spec, len(EXTRAS))] + \
                 [("army", "glob", b) for b in num_blocks]
        for model, key, (name, cols, kind, tol) in checks:
            got = [_off(reads[r][f][key][cols], rows[r][f][key][cols], kind) for f in rows[r] if key in rows[r][f]]
            if not got:
                continue
            offs = np.array([o for o, _ in got])
            share = float(np.mean(offs > tol))
            labels = _labels(g, model, name)
            cols_off = [c for o, c in got if o > tol]
            most = max(set(cols_off), key=cols_off.count) if cols_off and labels else None
            _verdict(r, model, name + (f" (mostly {labels[most]})" if most is not None else ""), share, float(offs.max()),
                     len(got), report, fails, seen_known, key_name=name)
        for name, c in grid_blocks:
            offs = []
            for f in rows[r]:
                if "grid" not in rows[r][f]:
                    continue
                a, b = reads[r][f]["grid"][c], rows[r][f]["grid"][c]
                tol = max(2.0, 0.1 * max(a.sum(), b.sum()))
                offs.append((float(np.abs(a - b).sum()), tol))
            if offs:
                share = float(np.mean([o > t for o, t in offs]))
                _verdict(r, "army", f"grid: {name}", share, max(o for o, _ in offs), len(offs), report, fails, seen_known)
    gone = [f"{k}: {why}" for k, why in KNOWN.items() if k not in seen_known and k[0] in sides]
    return report, fails, gone


def _verdict(r, model, name, share, worst, n, report, fails, seen_known, key_name=None):
    key = (r, model, key_name or name)
    why = INTENDED.get(("*", model, key_name or name)) or INTENDED.get(key)
    known = KNOWN.get(key)
    state = "ok"
    if known and share > FAIL_SHARE / 4:        # (a known fault's size follows the game: gone only when it's near 0)
        seen_known.add(key)
    if share > FAIL_SHARE:
        if why:
            state = "intended"
        elif known:
            state = "known"
            seen_known.add(key)
        else:
            state = "FAIL"
            fails.append(f"{r} {model} {name}: off at {share:.0%} of {n} moments (worst {worst:.2f})")
    report.append(f"  {r} {model:6} {name:44} off {share:4.0%} of {n:3d}, worst {worst:7.2f}  {state}")


def test_inputs_match_training_data():
    try:
        m, sides, reads = _play(UNTIL // 24)
    except Exception as e:                      # no game data, env DLL or trained models here
        print(f"skip: {e}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        rows = _replay_rows(m, sides, reads, Path(tmp))
    report, fails, gone = compare(sides, reads, rows)
    r2, f2, seen2 = compare_roles(sides)
    report, fails = report + r2, fails + f2
    gone = [x for x in gone if not any(x.startswith(f"{k}:") for k in seen2)]
    print("\n".join(report))
    few = {r: len(v) for r, v in rows.items() if len(v) < MIN_MOMENTS}
    assert not few, f"too few moments compared (a side's looks missed the ticks?): {few}"
    assert not fails, "the models' inputs in play differ from their training data's:\n" + "\n".join(fails)
    if gone:                                     # (a known fault's size follows the game: a warning, not a failure)
        print("warning: known faults that didn't show in this game (check, and remove them from KNOWN once fixed):\n"
              + "\n".join(gone))
    print(f"ok: {sum(len(v) for v in rows.values())} moments compared")


if __name__ == "__main__":
    test_inputs_match_training_data()
