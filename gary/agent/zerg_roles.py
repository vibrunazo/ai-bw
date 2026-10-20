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
from collections import Counter
from pathlib import Path

import numpy as np

from gary import commands as C
from gary import zerg as Z
from gary.decision_log import LOG, top
from gary.interface import screen_of

ASK_EVERY = 12                  # frames: a role every half second, as the model learned them
WATCH = 12 * 32                 # a seen enemy fighter this near its units: a fight (or one coming)
WHERE_LEAD = 4                  # a move's destination: this many of the model's half-second moves ahead
WHERE_MIN = 0.25 * 32           # px per half second: a shorter predicted move is no move
REPEAT = 24                     # the same role isn't given to a unit again within a second
TRIES = 3                       # looks and selections before a decision is dropped
MAX_GROUP = 12
NEAR = 10 * 32                  # the retreat rule: fighters this near the group's middle count for its strength
RETREAT_RATIO = 1.5             # outweighed by this, away from home: back toward the hatchery
AT_HOME = 8 * 32                # a hatchery this near: home (its sunkens; no retreat)
RETREAT_STEP = 10 * 32          # px toward the hatchery per retreat order
RETREAT_AGAIN = 3 * 24          # a retreat ordered at most this often
ATTACKING = {10, 14}
URGENT = ("RETREAT", "PULL_BACK", "REPOSITION", "FIRE", "CLOSE_IN")
FIGHTERS = Z.ARMY - {Z.LURKER}  # (lurkers: v3's lurker routine burrows them)


def latest_zerg_model(data: str = "response_Z.npz") -> Path | None:
    """The newest Zerg role model: updated by the league's RL (runs/roles_rl/Z_*), else the newest
    trained on the Zerg's response dataset of this name (runs/roles/*/model.pt)."""
    from gary.policy.roles import latest_trained_on
    import os
    if os.environ.get("GARY_ZROLES_MODEL"):           # (the league loop's incumbent or candidate, pinned)
        return Path(os.environ["GARY_ZROLES_MODEL"])
    rl = sorted(Path("runs/roles_rl").glob("Z_*/model.pt"))
    if rl:
        return rl[-1]
    return latest_trained_on(data)


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
        self.queue: list[dict] = []          # this snapshot's other role groups, carried out in turn
        self.retreat_at = -10 ** 6           # when the retreat rule last ordered
        self.rec: dict | None = None         # this snapshot's decisions as carried out (logged once done)
        self.told: dict[int, tuple[str, int]] = {}
        self.acted = 0                       # orders given (the log's count)
        # its own dice (deterministic per game): a paired episode draws its arms' fight decisions anew
        # without touching Gary-Z's other dice (eval/pairs.py)
        self.rng = random.Random(bot.rng.random())
        self.record: list | None = None      # a paired episode's arm: its decisions, kept here
        self.shadow = None                   # an A/B pair's other model: its odds of the roles drawn, recorded

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
        if self.queue:                       # (every role group of the snapshot gets its order, one a turn)
            self.pending = self.queue.pop(0)
            return self._carry_out(obs)
        self._emit()
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
        # outweighed away from home: back toward the nearest hatchery, before the model's roles (its retreats
        # came 7% of the time; in round g3 Gary-Z lost 118 fighters a game, half of them with two or fewer of
        # their own within 6 tiles and 8 enemy fighters within 8: zerglings into bio in pieces)
        own = [u for u in seen if u["owner"] == slot and u["type"] in FIGHTERS and u["completed"]]
        halls = [(u["x"], u["y"]) for u in mine if u["type"] in Z.TOWN_HALLS]
        if own and halls and hi.frame - self.retreat_at >= RETREAT_AGAIN:
            mx, my = sum(u["x"] for u in own) / len(own), sum(u["y"] for u in own) / len(own)
            ours = sum(supply_x2(u["type"]) for u in own if math.dist((u["x"], u["y"]), (mx, my)) < NEAR)
            theirs = sum(supply_x2(u["type"]) for u in enemy if math.dist((u["x"], u["y"]), (mx, my)) < NEAR)
            home = min(halls, key=lambda h: math.dist(h, (mx, my)))
            if theirs > RETREAT_RATIO * max(ours, 1) and math.dist(home, (mx, my)) > AT_HOME:
                group = sorted(own, key=lambda u: math.dist((u["x"], u["y"]), (mx, my)))[:MAX_GROUP]
                d = math.dist(home, (mx, my))
                point = (int(mx + (home[0] - mx) / d * RETREAT_STEP), int(my + (home[1] - my) / d * RETREAT_STEP))
                self.retreat_at = hi.frame
                LOG.decision("Z", "roles", hi.frame, "retreat", inputs={"units": len(group), "ours": ours / 2, "theirs": theirs / 2},
                             by=f"rule: outweighed {RETREAT_RATIO:g} to 1 away from home, back toward the hatchery")
                self.pending = {"role": "RETREAT", "tags": [u["tag"] for u in group], "point": point, "kind": "move",
                                "tries": 0, "selecting": False}
                return self._carry_out(obs)
        fx, fy = (-1 if self.flip[0] else 1), (-1 if self.flip[1] else 1)
        groups: dict[str, list] = {}
        # each unit's decision drawn from the model's odds (keep, or a new role), not the likeliest: what
        # RL learns from is a drawn decision and its probability. Logged as carried out (Fable's catch):
        # a decision that needs no order (keep, stand, other; fire for a unit already attacking; the same
        # role again within REPEAT) at once, a role group's when its order goes; one never carried out
        # (a group whose selection failed, a move too short) masked
        n = len(probs)
        pad = lambda a: np.concatenate([a[:MAX_UNITS], np.zeros((max(0, MAX_UNITS - len(a)),) + a.shape[1:], a.dtype)])
        self.rec = {"frame": hi.frame, "rows": pad(rows.astype(np.int16)), "extra": pad(extra.astype(np.float16)),
                    "count": min(n, MAX_UNITS), "time": obs["frame"] * 42 / 1000,
                    "drawn": np.full(MAX_UNITS, -1, np.int8), "logp": np.zeros(MAX_UNITS, np.float32), "index": {}}
        for i, t in enumerate(tags[:min(n, MAX_UNITS)]):
            u = by_tag.get(t)
            if u is None or u["owner"] != slot or u["type"] not in FIGHTERS or not u["completed"]:
                continue
            k = self.rng.choices(range(probs.shape[1]), weights=probs[i].tolist())[0]
            lp = float(np.log(probs[i, k] + 1e-9))
            self.rec["index"][t] = (i, k, lp)
            role = self.names[k]
            last = self.told.get(t)
            if role not in URGENT or (last and last[0] == role and hi.frame - last[1] < REPEAT)                     or (role in ("FIRE", "CLOSE_IN") and u["order"] in ATTACKING):
                self.rec["drawn"][i], self.rec["logp"][i] = k, lp       # (carried out: nothing to order)
                continue
            if role in ("FIRE", "CLOSE_IN"):
                groups.setdefault(role, []).append((u, None))
            elif moves is not None:
                dx, dy = moves[i] * 32
                if math.hypot(dx, dy) >= WHERE_MIN:
                    groups.setdefault(role, []).append((u, (u["x"] + fx * dx * WHERE_LEAD, u["y"] + fy * dy * WHERE_LEAD)))
        # (the record: the roles drawn this snapshot, the model's mean odds over its units, the orders to give)
        drawn = Counter(self.names[k] for _, k, _ in self.rec["index"].values())
        if drawn:
            chosen = list(self.rec["index"].values())
            other = None
            if self.shadow is not None:               # (an A/B arm: the other model's odds of the same roles)
                sp = self.shadow.predict_move(rows, extra, obs["frame"] * 42 / 1000)[0]
                other = float(np.mean([sp[i, k] for i, k, _ in chosen]))
            self.rec["record"] = LOG.decision(
                hi.side, "roles", hi.frame, ", ".join(f"{r.lower()} {c}" for r, c in drawn.most_common()),
                float(np.mean([probs[i, k] for i, k, _ in chosen])),
                top(lambda k: self.names[k].lower(), probs[[i for i, _, _ in chosen]].mean(0)),
                {"units": len(chosen), "enemy": len(enemy)}, "draw",
                then="orders: " + (", ".join(f"{r.lower()} {len(g)}" for r, g in groups.items()) or "none"),
                p_other=other)
        # every non-empty group queued, most urgent first, one order a turn (the first alone went before:
        # one zergling drawing retreat cost the rest their orders, and draws were logged as done)
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
            self.queue.append({"role": role, "tags": [u["tag"] for u in units], "point": point, "kind": kind,
                               "tries": 0, "selecting": False})
        if self.queue:
            self.pending = self.queue.pop(0)
            return self._carry_out(obs)
        self._emit()
        return False

    def _emit(self) -> None:
        """The snapshot's decisions, as carried out, to the log (or a paired episode's arm)."""
        rec, self.rec = self.rec, None
        if rec is None or not (rec["drawn"] >= 0).any() or not (LOG.on or self.record is not None):
            return
        item = {k: v for k, v in rec.items() if k not in ("index", "record")}
        if self.record is not None:
            self.record.append(item)
        else:
            LOG.add("roles_z", **item)
        if rec.get("record") is not None:             # (its row among the RL decisions, in its record)
            rec["record"]["rl"] = len(self.record) - 1 if self.record is not None else \
                len(LOG.items["roles_z"]) - 1 if LOG.on else None

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
                if self.rec is not None and t in self.rec["index"]:      # (carried out: logged)
                    i, k, lp = self.rec["index"][t]
                    self.rec["drawn"][i], self.rec["logp"][i] = k, lp
            self.acted += 1
            if self.acted % 25 == 1:
                self.bot.log(f"fight roles: {p['role'].lower()} ({len(sel)} units), {self.acted} orders so far")
            self.pending = None
            return True
        p["tries"] += 1
        if p["tries"] > TRIES:
            LOG.order(hi.side, hi.frame, f"{p['role'].lower()} ({len(units)} units)", "dropped",
                      f"its units weren't selected in {TRIES} tries", by="roles")
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
