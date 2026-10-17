"""Gary-Z's fights by the Zerg role model: what its units do in a fight and before one, learned from the
pros' Zerg units (ingest/response_dataset.py --race Z, train/roles.py --decisions --move), in place of
v3's scripted micro. The league's learner for the Zerg's fights: imitation first, RL sharpens it.

Every ASK_EVERY, with a seen enemy fighter within WATCH of its units, the fight as Gary-Z sees it (the
snapshot the Terran's fight models use, mirrored to its own corner) goes to the model, which says for
each unit keep, or its new role and where it moves over the next half second (drawn from its odds). The most urgent group
of units whose role changed is carried out with Gary-Z's hands, as a player does: a look, a click on
one and shift+clicks on the rest (12 at most), then the order.

  retreat, pull back, reposition   a move where the model moves them (WHERE_LEAD of its half-second
                                   moves ahead: a zergling pack backing off from marines that saw it)
  fire, close in                   an attack-move at the nearest enemy to them
  stand, other, keep               nothing (the unit goes on)
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np

from gary import commands as C
from gary import zerg as Z
from gary.decision_log import LOG
from gary.interface import screen_of

ASK_EVERY = 12                  # frames: a role every half second, as the model learned them
WATCH = 12 * 32                 # a seen enemy fighter this near its units: a fight (or one coming)
WHERE_LEAD = 4                  # a move's destination: this many of the model's half-second moves ahead
WHERE_MIN = 0.25 * 32           # px per half second: a shorter predicted move is no move
REPEAT = 24                     # the same role isn't given to a unit again within a second
TRIES = 3                       # looks and selections before a decision is dropped
MAX_GROUP = 12
ATTACKING = {10, 14}
URGENT = ("RETREAT", "PULL_BACK", "REPOSITION", "FIRE", "CLOSE_IN")
FIGHTERS = Z.ARMY - {Z.LURKER}  # (lurkers: v3's lurker routine burrows them)


def latest_zerg_model() -> Path | None:
    """The newest Zerg role model: updated by the league's RL (runs/roles_rl/Z_*), else the newest
    trained on the Zerg's response dataset (runs/roles/*/model.pt)."""
    from gary.policy.roles import RoleModel
    import os
    if os.environ.get("GARY_ZROLES_MODEL"):           # (the league loop's incumbent or candidate, pinned)
        return Path(os.environ["GARY_ZROLES_MODEL"])
    rl = sorted(Path("runs/roles_rl").glob("Z_*/model.pt"))
    if rl:
        return rl[-1]
    for p in sorted(Path("runs/roles").glob("*/model.pt"), reverse=True):
        try:
            if str(RoleModel.load(p).config.get("data", "")).endswith("response_Z.npz"):
                return p
        except Exception:
            continue
    return None


class ZergRoles:
    def __init__(self, bot, model_path: Path):
        from gary.policy.roles import RoleModel
        self.bot = bot
        self.model = RoleModel.load(model_path)
        self.names = [n.upper() for n in self.model.config["roles"]]
        self.flip = None
        self.ground = None
        self.prev: dict[int, tuple[float, float]] = {}
        self.next_ask = 0
        self.pending: dict | None = None
        self.told: dict[int, tuple[str, int]] = {}
        self.acted = 0                       # orders given (the log's count)
        # its own dice (deterministic per game): a paired episode draws its arms' fight decisions anew
        # without touching Gary-Z's other dice (eval/pairs.py)
        self.rng = random.Random(bot.rng.random())
        self.record: list | None = None      # a paired episode's arm: its decisions, kept here

    def _setup(self, obs: dict) -> None:
        from gary.policy.roles import Ground
        w, h = self.bot.hi.map_size           # (pixels)
        home = self.bot.home
        self.flip = (home[0] > w / 2, home[1] > h / 2)
        try:
            self.ground = Ground.of(self.bot.hi.game, home)
        except Exception:                    # (a map the analysis can't read: no terrain inputs)
            self.ground = None

    def step(self, obs: dict, mine: list[dict]) -> bool:
        """Gary-Z's fight turn (True: it acted)."""
        if self.pending is not None:
            return self._carry_out(obs)
        hi = self.bot.hi
        if hi.frame < self.next_ask:
            return False
        self.next_ask = hi.frame + ASK_EVERY
        if self.flip is None:
            self._setup(obs)
        from gary.policy.army import is_building, is_worker, supply_x2
        from gary.policy.fight import FIGHT_AROUND, snapshot
        from gary.policy.roles import MAX_UNITS, extra_features, terrain_features
        slot = hi.slot
        # (Gary-Z's observation is already its own view; a raw one carries visible_to)
        seen = [u for u in obs["units"] if u["owner"] in (slot, 11) or "visible_to" not in u
                or (u["visible_to"] >> slot) & 1]
        snap = snapshot({**obs, "units": seen}, slot, self.flip, supply_x2, is_worker, is_building,
                        FIGHT_AROUND, MAX_UNITS, near=WATCH)
        positions = {u["tag"]: (u["x"], u["y"]) for u in seen}
        if snap is None:
            self.prev = positions
            return False
        rows, tags, _ = snap
        extra = extra_features(rows, tags, positions, self.prev, self.flip)
        extra[:, 6:] = terrain_features(tags, positions, self.ground,
                                        [(u["x"], u["y"]) for u in mine if u["type"] in Z.TOWN_HALLS],
                                        [(u["x"], u["y"]) for u in mine if u["type"] == Z.SUNKEN and u["completed"]],
                                        self.flip)
        self.prev = positions
        probs, moves = self.model.predict_move(rows, extra, obs["frame"] * 42 / 1000)
        by_tag = {u["tag"]: u for u in seen}
        enemy = [u for u in seen if u["owner"] not in (slot, 11) and supply_x2(u["type"]) > 0
                 and not is_worker(u["type"]) and not is_building(u["type"])]
        if not enemy:
            return False
        fx, fy = (-1 if self.flip[0] else 1), (-1 if self.flip[1] else 1)
        groups: dict[str, list] = {}
        # each unit's decision drawn from the model's odds (keep, or a new role), not the likeliest: what
        # RL learns from is a drawn decision and its probability (the league logs every snapshot's)
        n = len(probs)
        drawn = np.full(MAX_UNITS, -1, np.int8)
        logp = np.zeros(MAX_UNITS, np.float32)
        for i, t in enumerate(tags[:n]):
            u = by_tag.get(t)
            if u is None or u["owner"] != slot or u["type"] not in FIGHTERS or not u["completed"]:
                continue
            k = self.rng.choices(range(probs.shape[1]), weights=probs[i].tolist())[0]
            drawn[i], logp[i] = k, np.log(probs[i, k] + 1e-9)
            role = self.names[k]
            if role not in URGENT:
                continue
            last = self.told.get(t)
            if last and last[0] == role and hi.frame - last[1] < REPEAT:
                continue
            if role in ("FIRE", "CLOSE_IN"):
                if u["order"] in ATTACKING:
                    continue
                groups.setdefault(role, []).append((u, None))
            elif moves is not None:
                dx, dy = moves[i] * 32
                if math.hypot(dx, dy) >= WHERE_MIN:
                    groups.setdefault(role, []).append((u, (u["x"] + fx * dx * WHERE_LEAD, u["y"] + fy * dy * WHERE_LEAD)))
        if (LOG.on or self.record is not None) and (drawn >= 0).any():
            pad = lambda a: np.concatenate([a[:MAX_UNITS], np.zeros((max(0, MAX_UNITS - len(a)),) + a.shape[1:], a.dtype)])
            item = dict(frame=hi.frame, rows=pad(rows.astype(np.int16)), extra=pad(extra.astype(np.float16)),
                        count=min(n, MAX_UNITS), time=obs["frame"] * 42 / 1000, drawn=drawn, logp=logp)
            if self.record is not None:
                self.record.append(item)
            else:
                LOG.add("roles_z", **item)
        for role in URGENT:
            group = groups.get(role, [])[:MAX_GROUP]
            if not group:
                continue
            units = [u for u, _ in group]
            mid = (sum(u["x"] for u in units) / len(units), sum(u["y"] for u in units) / len(units))
            if role in ("FIRE", "CLOSE_IN"):
                e = min(enemy, key=lambda e: math.dist((e["x"], e["y"]), mid))
                point, kind = (int(e["x"]), int(e["y"])), "attack"
            else:
                pts = [p for _, p in group]
                point, kind = (int(sum(x for x, _ in pts) / len(pts)), int(sum(y for _, y in pts) / len(pts))), "move"
            self.pending = {"role": role, "tags": [u["tag"] for u in units], "point": point, "kind": kind,
                            "tries": 0, "selecting": False}
            return self._carry_out(obs)
        return False

    def _carry_out(self, obs: dict) -> bool:
        p = self.pending
        hi = self.bot.hi
        alive = {u["tag"]: u for u in obs["units"] if u["owner"] == hi.slot}
        units = [alive[t] for t in p["tags"] if t in alive]
        if not units:
            self.pending = None
            return False
        sel = set(obs["selection"])
        if p["selecting"] and sel and sel <= set(p["tags"]):
            x, y = p["point"]
            on = screen_of(obs, x, y)
            if p["kind"] == "move":
                hi.right_click(*on) if on else hi.minimap_right_click(x, y)
            else:
                hi.order_click(C.ORDER_ATTACK_MOVE, *on) if on else hi.minimap_command(C.ORDER_ATTACK_MOVE, x, y)
            for t in sel:
                self.told[t] = (p["role"], hi.frame)
            self.acted += 1
            if self.acted % 25 == 1:
                self.bot.log(f"fight roles: {p['role'].lower()} ({len(sel)} units), {self.acted} orders so far")
            self.pending = None
            return True
        p["tries"] += 1
        if p["tries"] > TRIES:
            self.pending = None
            return False
        mid = (sum(u["x"] for u in units) / len(units), sum(u["y"] for u in units) / len(units))
        if self.bot._look(obs, mid):
            return True
        shown = [u for u in units if screen_of(obs, u["x"], u["y"])]
        if not shown:
            return True
        hi.click(*screen_of(obs, shown[0]["x"], shown[0]["y"]))
        for u in shown[1:MAX_GROUP]:
            hi.click(*screen_of(obs, u["x"], u["y"]), shift=True)
        p["selecting"] = True
        return True
