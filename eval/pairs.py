"""Paired fight episodes in league games (docs/SELF_PLAY.md "The loop": fight episodes paired on the same
positions, with Gary-Z the reacting side): the one RL target of the first league nights.

In a league game (gary/bots/zerg_rush.py play_match, GARY_PAIRS set), when Gary-Z's fighters see enemy
fighters near them (no more often than EVERY), the game is saved and two arms are played from that
moment for WINDOW, both bots live (each arm a deep copy of both bots; the models and the game shared,
the game restored before each arm): in each arm Gary-Z's fight roles draw their own decisions. Then the
game is restored and goes on as if nothing happened (the replay recorded is the main line's).

  training pair   both arms the incumbent role model, drawing with different dice: the difference of
                  their outcomes is the decisions' advantage, the position's luck cancelled
  A/B pair        the incumbent against a candidate (GARY_PAIRS_CANDIDATE), the same dice: the gate's
                  evidence (eval/league_loop.py keeps a candidate only if these favor it)

An arm's outcome (fight_outcome): the fight's net value over the window, every unit within FIGHT of the
fight's middle at the start counted, workers and buildings too (stakes: a pack that runs leaves them),
the enemy's lost less its own, wherever the units went. Diagnostics per arm, the independent check
against learned retreat (it was learned here before, from drills with nothing to defend): the share of
its fighters there that ended AWAY farther from the enemy, their mean move, and whether the fight was
in the field (none of its buildings near: there retreat costs nothing, and the share must not creep up
round over round without the paired outcome rising).
"""

from __future__ import annotations

import copy
import os
import random
from pathlib import Path

import numpy as np

WINDOW = 20 * 24                # frames an arm plays
EVERY = 30 * 24                 # frames between branch points at most
CHECK = 2 * 24                  # frames between looks for a fight
SIGHT = 12 * 32                 # px: Gary-Z's fighters seeing enemy fighters this near: a fight
FIGHT = 12 * 32                 # px: the fight's units, around its middle
AWAY = 4 * 32                   # px: a fighter this much farther from the enemy at the end: it went away
CHECK_EVERY = 4                 # every this many branch points, arm 0 is played again and must end the same
ARMY_WINDOW = 90 * 24           # army mode (GARY_PAIRS_LAYER=army_t): an arm plays this long...
ARMY_EVERY = 60 * 24            # ...at most this often...
ARMY_FROM = 5 * 60 * 24         # ...from then on...
ARMY_MIN_X2 = 32                # ...with the Terran's army this big (16 supply): its decision moves the game
                                # three times as much as the chaos over 90 s (eval/army_effect.py)


def _value(t: int) -> float:
    from gary import terran as T
    from gary import zerg as Z
    m, g = T.COST.get(t) or Z.COST.get(t) or (50, 0)
    return (m + g) / (2 if t in Z.PAIRS else 1)


def snap(o: dict) -> dict:
    """tag -> (owner, type, x, y), every unit but the neutrals'."""
    return {u["tag"]: (u["owner"], u["type"], u["x"], u["y"]) for u in o["units"] if u["owner"] != 11}


def fight_outcome(a: dict, b: dict, slot: int) -> tuple[float, dict] | None:
    """The fight around the slot's fighters at a, scored at b: (net value / 100, diagnostics), or None
    when there is no fight at a."""
    from gary.policy.army import is_building, is_worker, supply_x2
    fighter = lambda t: supply_x2(t) > 0 and not is_worker(t) and not is_building(t)
    mine = [(tag, x, y) for tag, (o, t, x, y) in a.items() if o == slot and fighter(t)]
    foes = [(x, y) for tag, (o, t, x, y) in a.items() if o != slot and fighter(t)]
    if not mine or not foes:
        return None
    m = np.array([(x, y) for _, x, y in mine], np.float32)
    e = np.array(foes, np.float32)
    d = np.sqrt(((m[:, None] - e[None]) ** 2).sum(-1)).min(1)
    if not (d < FIGHT).any():
        return None
    mid = m[d < FIGHT].mean(0)
    near = {tag: v for tag, v in a.items() if (v[2] - mid[0]) ** 2 + (v[3] - mid[1]) ** 2 < FIGHT ** 2}
    lost = lambda own: sum(_value(v[1]) for tag, v in near.items() if (v[0] == slot) == own and tag not in b)
    reward = (lost(False) - lost(True)) / 100.0
    ec = e[np.sqrt(((e - mid) ** 2).sum(1)) < FIGHT].mean(0) if (np.sqrt(((e - mid) ** 2).sum(1)) < FIGHT).any() else e.mean(0)
    ours = [(tag, x, y) for tag, x, y in mine if (x - mid[0]) ** 2 + (y - mid[1]) ** 2 < FIGHT ** 2]
    moved, away = [], []
    for tag, x, y in ours:
        if tag in b:
            x1, y1 = b[tag][2], b[tag][3]
            moved.append(np.hypot(x1 - x, y1 - y) / 32)
            away.append(np.hypot(x1 - ec[0], y1 - ec[1]) - np.hypot(x - ec[0], y - ec[1]) >= AWAY)
    field = not any(o == slot and is_building(t) for o, t, _, _ in near.values())
    return reward, {"away": float(np.mean(away)) if away else 0.0, "moved": float(np.mean(moved)) if moved else 0.0,
                    "field": field, "units": len(ours)}


class Brancher:
    """Paired episodes in one league game (GARY_PAIRS: the file they go to, .npz, written at the end)."""

    def __init__(self, game, slot_of: dict, seed: int):
        self.path = os.environ.get("GARY_PAIRS")
        self.layer = os.environ.get("GARY_PAIRS_LAYER", "roles_z")
        self.candidate_path = os.environ.get("GARY_PAIRS_CANDIDATE")
        self.slot_t = slot_of["T"]
        self.candidate = None
        self.slot_z = slot_of["Z"]
        self.rng = random.Random(seed * 104729 + 3)
        self.next = 0
        self.pairs: list[dict] = []
        self.kind_turn = 0
        # a nondeterministic crash in a deterministic system is memory corruption, and corruption that
        # doesn't crash gives silently wrong outcomes (Fable): a sample of branches replayed every game
        self.checks_ok = self.checks_bad = 0

    @property
    def on(self) -> bool:
        return bool(self.path)

    def maybe(self, game, objs: dict) -> None:
        """At a fight (no more often than EVERY), a pair of arms; objs: the bots and hands, by name."""
        if self.layer in ("army_t", "engage_t"):
            return self._maybe_army(game, objs)
        f = game.frame
        if f < self.next:
            return
        self.next = f + CHECK
        zerg = objs["zerg"]
        if getattr(zerg, "fight_roles", None) is None:
            return
        o = game.observe()
        z = self.slot_z
        seen = [(u["x"], u["y"]) for u in o["units"] if u["owner"] not in (z, 11) and (u.get("visible_to", 0) >> z) & 1
                and u["type"] in _FIGHTERS]
        mine = [(u["x"], u["y"]) for u in o["units"] if u["owner"] == z and u["type"] in _FIGHTERS and u["completed"]]
        if not seen or not mine:
            return
        a, b = np.array(mine, np.float32), np.array(seen, np.float32)
        if not (np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1)) < SIGHT).any():
            return
        from gary.env import GameError
        try:
            game.save()
        except GameError:
            return                                            # (commands in flight: a moment later)
        self.next = f + EVERY
        start = snap(o)
        if self.candidate_path and self.candidate is None:
            from gary.policy.roles import RoleModel
            self.candidate = RoleModel.load(self.candidate_path)
        ab = self.candidate is not None and self.kind_turn % 2 == 0
        self.kind_turn += 1
        dice = [self.rng.random() for _ in range(2)]
        if ab:
            dice = [dice[0], dice[0]]                         # (the same dice: the models differ)
        arms, end0 = [], None
        for arm in range(2):
            game.restore()
            c = _copy(objs)
            roles = c["zerg"].fight_roles
            if ab and arm == 1:
                roles.model = self.candidate
            roles.rng = random.Random(dice[arm])
            roles.record = []
            _play(game, c, WINDOW)
            roles._emit()                                     # (the arm's last snapshot, as carried out)
            end = snap(game.observe())
            end0 = end if arm == 0 else end0
            out = fight_outcome(start, end, z)
            arms.append({"reward": out[0] if out else 0.0, **(out[1] if out else {"away": 0.0, "moved": 0.0,
                         "field": True, "units": 0}), "model": str(getattr(roles.model, "path", "?")),
                         "decisions": roles.record})
        if self.kind_turn % CHECK_EVERY == 1:               # (arm 0 again: the same end, or the branches lie)
            game.restore()
            c = _copy(objs)
            c["zerg"].fight_roles.rng = random.Random(dice[0])
            _play(game, c, WINDOW)
            if snap(game.observe()) == end0:
                self.checks_ok += 1
            else:
                self.checks_bad += 1
        game.restore()
        self.pairs.append({"frame": f, "ab": ab, "arms": arms})

    def _maybe_army(self, game, objs: dict) -> None:
        """Army mode: at a moment the Terran's army is big enough, two arms of ARMY_WINDOW in which its army
        layer draws its decisions with different dice (training) or another model (A/B); the outcome is
        the whole map's net value on the Terran's side."""
        from gary.env import GameError
        from gary.policy.army import supply_x2
        f = game.frame
        if f < max(self.next, ARMY_FROM):
            return
        self.next = f + CHECK
        gary = objs["gary"]
        army = getattr(gary, "army", None)
        if army is None or not getattr(army, "draw", False) or (self.layer == "engage_t" and army.engage is None):
            return
        o = game.observe()
        mine = [u for u in o["units"] if u["owner"] == self.slot_t and u["type"] in gary.army_types and u["completed"]]
        if sum(supply_x2(u["type"]) for u in mine) < ARMY_MIN_X2:
            return
        try:
            game.save()
        except GameError:
            return
        self.next = f + ARMY_EVERY
        start = snap(o)
        if self.candidate_path and self.candidate is None:
            from gary.policy.army import ArmyModel
            self.candidate = ArmyModel.load(self.candidate_path)
            if self.layer == "engage_t":
                self.candidate.path = self.candidate_path
        ab = self.candidate is not None and self.kind_turn % 2 == 0
        self.kind_turn += 1
        # one decision branched: each arm's first army decision drawn with its own dice, the rest with
        # shared dice (a1 drew every decision of 90 s apart: ~+-5 a pair, the credit smeared over dozens);
        # only that first decision is the arm's to train on
        dice = [int(self.rng.random() * 2 ** 31) for _ in range(3)]
        if ab:
            dice = [dice[0], dice[0], dice[2]]
        arms, end0 = [], None
        for arm in range(2):
            game.restore()
            c = _copy(objs)
            layer = c["gary"].army
            if ab and arm == 1:
                if self.layer == "engage_t":
                    layer.engage = self.candidate
                else:
                    layer.model = self.candidate
            layer.rng = np.random.default_rng(dice[arm])
            layer.then_rng = np.random.default_rng(dice[2])
            if self.layer == "engage_t" and not ab:
                layer.force_engage = arm == 0              # (training: the first decision attack in one arm, hold in the other)
            layer.record = []
            _play(game, c, ARMY_WINDOW)
            end = snap(game.observe())
            end0 = end if arm == 0 else end0
            from eval.army_effect import _net
            arms.append({"reward": -_net(start, end, self.slot_z), "away": 0.0, "moved": 0.0, "field": False,
                         "units": len(mine),
                         "model": str(getattr(layer.engage if self.layer == "engage_t" else layer.model, "path", "?")),
                         "decisions": layer.record[:1]})
        if self.kind_turn % CHECK_EVERY == 1:
            game.restore()
            c = _copy(objs)
            c["gary"].army.rng = np.random.default_rng(dice[0])
            c["gary"].army.then_rng = np.random.default_rng(dice[2])
            if self.layer == "engage_t" and not ab:
                c["gary"].army.force_engage = True
            _play(game, c, ARMY_WINDOW)
            if snap(game.observe()) == end0:
                self.checks_ok += 1
            else:
                self.checks_bad += 1
        game.restore()
        self.pairs.append({"frame": f, "ab": ab, "arms": arms})

    def save(self) -> None:
        if not self.path or not self.pairs:
            return
        if self.layer in ("army_t", "engage_t"):
            return self._save_army()
        out = {"frame": [], "ab": [], "reward": [], "away": [], "moved": [], "field": [], "units": [], "model": []}
        dec = {"pair": [], "arm": [], "rows": [], "extra": [], "count": [], "time": [], "drawn": [], "logp": []}
        for i, p in enumerate(self.pairs):
            out["frame"].append(p["frame"])
            out["ab"].append(p["ab"])
            for k in ("reward", "away", "moved", "field", "units", "model"):
                out[k].append([arm[k] for arm in p["arms"]])
            for j, arm in enumerate(p["arms"]):
                for d in arm["decisions"]:
                    dec["pair"].append(i)
                    dec["arm"].append(j)
                    for k in ("rows", "extra", "count", "time", "drawn", "logp"):
                        dec[k].append(d[k])
        arrays = {k: np.array(v) for k, v in out.items()}
        arrays["checks"] = np.array([self.checks_ok, self.checks_bad])
        if dec["pair"]:
            arrays.update({f"dec/{k}": np.stack([np.asarray(x) for x in v]) for k, v in dec.items()})
        np.savez_compressed(self.path, **arrays)


    def _save_army(self) -> None:
        out = {"frame": [], "ab": [], "reward": [], "away": [], "moved": [], "field": [], "units": [], "model": []}
        keys = ("grid", "glob", "engage", "logp") if self.layer == "engage_t" else ("grid", "glob", "cell", "attack", "logp")
        dec = {"pair": [], "arm": [], **{k: [] for k in keys}}
        for i, p in enumerate(self.pairs):
            out["frame"].append(p["frame"])
            out["ab"].append(p["ab"])
            for k in ("reward", "away", "moved", "field", "units", "model"):
                out[k].append([arm[k] for arm in p["arms"]])
            for j, arm in enumerate(p["arms"]):
                for d in arm["decisions"]:
                    dec["pair"].append(i)
                    dec["arm"].append(j)
                    for k in keys:
                        dec[k].append(d[k])
        arrays = {k: np.array(v) for k, v in out.items()}
        arrays["checks"] = np.array([self.checks_ok, self.checks_bad])
        if dec["pair"]:
            arrays.update({f"dec/{k}": np.stack([np.asarray(x) for x in v]) for k, v in dec.items()})
        np.savez_compressed(self.path, **arrays)


def summarize(files: list[str], incumbent: str | None, candidate: str | None) -> dict:
    """A round's pairs: how many; the A/B evidence (the candidate's arm less the incumbent's, same
    position, same dice: mean and standard error); and per model the arms' retreat share in field
    fights and in fights with stakes, and the mean move (the check against learned retreat)."""
    ab, train, arms, checks = [], 0, {}, [0, 0]
    norm = lambda p: str(Path(p).resolve()) if p else None
    inc, cand = norm(incumbent), norm(candidate)
    for f in files:
        d = np.load(f)
        if "checks" in d.files:
            checks = [checks[0] + int(d["checks"][0]), checks[1] + int(d["checks"][1])]
        for i in range(len(d["frame"])):
            m0, m1 = norm(str(d["model"][i, 0])), norm(str(d["model"][i, 1]))
            r = d["reward"][i]
            if m0 == m1:
                train += 1
            elif cand and {m0, m1} == {inc, cand}:
                ab.append(r[1] - r[0] if m1 == cand else r[0] - r[1])
            for j, m in ((0, m0), (1, m1)):
                key = "candidate" if m == cand else "incumbent" if m == inc else "other"
                a = arms.setdefault(key, {"field_away": [], "stakes_away": [], "moved": [], "reward": []})
                (a["field_away"] if d["field"][i, j] else a["stakes_away"]).append(float(d["away"][i, j]))
                a["moved"].append(float(d["moved"][i, j]))
                a["reward"].append(float(r[j]))
    mean = lambda v: float(np.mean(v)) if v else None
    return {"pairs_training": train, "pairs_ab": len(ab), "checks_ok": checks[0], "checks_bad": checks[1],
            "ab_mean": mean(ab), "ab_se": float(np.std(ab, ddof=1) / np.sqrt(len(ab))) if len(ab) > 1 else None,
            "arms": {k: {kk: mean(vv) for kk, vv in v.items()} | {"n": len(v["reward"])} for k, v in arms.items()}}


def _fighters():
    from gary import terran as T
    from gary import zerg as Z
    return set(T.ARMY) | {30} | (set(Z.ARMY) - {Z.LURKER})


_FIGHTERS = _fighters()


def _copy(objs: dict) -> dict:
    """A deep copy of the bots and their hands; models (torch modules, sklearn estimators) and the game
    shared, not copied."""
    import torch
    from gary.env import Game
    patched = [(torch.nn.Module, getattr(torch.nn.Module, "__deepcopy__", None)),
               (Game, getattr(Game, "__deepcopy__", None))]
    try:
        from sklearn.base import BaseEstimator
        patched.append((BaseEstimator, getattr(BaseEstimator, "__deepcopy__", None)))
    except ImportError:
        pass
    for cls, _ in patched:
        cls.__deepcopy__ = lambda self, memo: self
    try:
        return copy.deepcopy(objs)
    finally:
        for cls, old in patched:
            if old is None:
                del cls.__deepcopy__
            else:
                cls.__deepcopy__ = old


def _play(game, objs: dict, frames: int) -> None:
    """The match loop's body (play_match) for this many frames, the decision log off (its arrays, and the
    game's records: an arm's decisions aren't the game's)."""
    import contextlib
    import io
    from gary.decision_log import LOG
    path, LOG.path = LOG.path, None
    kept = (len(LOG.records), dict(LOG.owner), dict(LOG.held))
    quiet = contextlib.redirect_stdout(io.StringIO())        # (the arms' own log lines aren't the game's)
    quiet.__enter__()
    try:
        his = objs["his"]
        end = his["T"].frame + frames
        slots = set(objs["slot_of"].values())
        while his["T"].frame < end:
            if not slots <= {p["slot"] for p in game.observe()["players"]}:
                break
            objs["gary"].act()
            objs["zerg"].act()
            for i, hi in enumerate(his.values()):
                hi.step(2, advance_game=(i == 0))
    finally:
        quiet.__exit__(None, None, None)
        LOG.path = path
        del LOG.records[kept[0]:]
        LOG.owner, LOG.held = kept[1], kept[2]
