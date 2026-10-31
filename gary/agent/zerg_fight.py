"""The Zerg's fight layer (docs/REFACTOR.md Phase 2): a fight provider for the one Gary playing Zerg
(manifest fight.provider "zerg"), from Gary-Z's.

  roles     the Zerg role model's fights (gary/agent/zerg_roles.py: the pros' Zerg units before
            contact and in it, its retreat rule first), on the shared Gary's hands
  lurkers   the lurkers on a hotkey: burrowed at the army's meeting point, burrowing when enemies
            come near, up again to move on once it's calm (v3's routine, LURK_STREAK turns in a row at
            most, then a rest: alone it took every army turn of a probe game from 9:00 to 14:00)

  rules     the shared fight rules a manifest asks for (gary/agent/fight.py RULES: the workers' pull, the new
            fighters sent to a fight at home), asked first as in the other races' fight layer

The fight contract the arbiter reads: decide(obs, mine) -> who acted or None; block(obs, mine) for the
fight turns that put a task aside; next_fight, center, ctrl_click, rl_log, rng.
"""

from __future__ import annotations

import math
import random

from gary import zerg as Z
from gary.agent.gary import Gary
from gary.agent.zerg_roles import ZergRoles, latest_zerg_model
from gary.interface import screen_of

LURKER_KEY = 8                      # the lurkers' hotkey (0 was the scout's too, gary/agent/motor.py SCOUT_KEY: each set
                                    # overwrote the other, the recall took the scouts and every burrow was refused, about
                                    # 230 a game, so the lurkers never burrowed; the Zerg's keys: army 1-3, halls 4-5,
                                    # the raid 9)
BURROWED = {116, 117}               # orders: burrowing, burrowed (then a burrowed lurker guards, orders 2 and 3: the
                                    # order alone lost it, so self.down remembers; round ZP141a's lurkers were pressed
                                    # to burrow every 2 frames once down, about 250 refused a game, and never came up)
UNBURROWING = 118
MOVED_PX = 4                        # a lurker this far from where it burrowed is up (a burrowed unit can't move)
LURKER_NEAR = 7 * 32                # enemy units this near: the lurkers burrow
LURKER_CALM = 4 * 24                # ...and none near this long: up again (to move on)
LURK_STREAK = 8                     # turns in a row at most...
LURK_BACKOFF = 3 * 24               # ...then a rest this long
WORKERS = {7, 41, 64}               # SCV, drone, probe: no reason to burrow


class _Host:
    """What ZergRoles reads of its player (Gary-Z's attributes), on the shared Gary."""

    def __init__(self, g: Gary, seed: int):
        self.g = g
        self.hi = g.hi
        self.home = g.main.center
        self.rng = random.Random(seed * 7919 + 3)

    def log(self, text: str) -> None:
        self.g.say(text)

    def _look(self, obs: dict, p) -> bool:
        """The camera on p (True if it had to move: that was this turn's action)."""
        if screen_of(obs, int(p[0]), int(p[1])):
            return False
        self.hi.camera_minimap(int(p[0]), int(p[1]))
        return True


class ZergFight:
    center = None                   # (the army layer isn't kept off any units)
    ctrl_click = False
    rl_log = None

    def __init__(self, g: Gary, roles: bool = True, lurkers: bool = True, seed: int = 0,
                 roles_data: str = "response_Z.npz", before: list = (), in_fight: list = ()):
        self.g = g
        self.before, self.in_fight = list(before), list(in_fight)    # rules, as the FightLayer's
        self.rng = random.Random(seed)
        self.host = _Host(g, seed)
        model = latest_zerg_model(roles_data) if roles else None
        self.roles = ZergRoles(self.host, model) if model is not None else None
        self.model_path = str(model) if model is not None else None
        self.lurkers = lurkers
        self.lurk: set[int] = set()
        self.lurk_since = self.lurk_calm = 0
        self.down: dict[int, tuple[int, int]] = {}      # lurkers burrowed (as last seen): tag -> where
        self.lurk_streak, self.lurk_backoff = 0, 0

    @property
    def next_fight(self) -> int:
        """When the roles want a fight turn (one that may put a chore aside): now while an order is under way,
        else at their next ask. Without these turns the roles acted only when nothing else had the hands: a
        fifth of Gary-Z's fight decisions in round G1."""
        if self.roles is None:
            return 10 ** 9
        if self.roles.pending is not None or self.roles.queue:
            return 0
        return self.roles.next_ask

    def block(self, obs: dict, mine: list[dict]) -> str | None:
        """The fight turn: the rules, then the roles; after one, the macro's turn (production isn't starved by
        fights)."""
        for rule in self.before + self.in_fight:
            if rule.step(obs, mine):
                return f"rule:{rule.name}"
        if self.roles is not None and self.roles.step(obs, mine):
            self.g.macro_turn = True
            return "fight:roles"
        return None

    def decide(self, obs: dict, mine: list[dict]) -> str | None:
        for rule in self.in_fight:
            if rule.step(obs, mine):
                return f"rule:{rule.name}"
        if self.roles is not None and self.roles.step(obs, mine):
            return "fight:roles"
        if self.lurkers and any(u["type"] == Z.LURKER for u in mine) and self._lurkers_turn(obs, mine):
            return "fight:lurkers"
        return None

    # --- the lurkers ------------------------------------------------------------------------------

    def _lurkers_turn(self, obs: dict, mine: list[dict]) -> bool:
        hi = self.g.hi
        if hi.frame < self.lurk_backoff:
            return False
        out = self._lurkers(obs, mine)
        self.lurk_streak = self.lurk_streak + 1 if out else 0
        if self.lurk_streak >= LURK_STREAK:
            self.lurk_backoff, self.lurk_streak = hi.frame + LURK_BACKOFF, 0
        return out

    def _lurkers(self, obs: dict, mine: list[dict]) -> bool:
        g, hi = self.g, self.g.hi
        lurks = {u["tag"]: u for u in mine if u["type"] == Z.LURKER and u["completed"]}
        self.lurk &= set(lurks)
        for t, u in lurks.items():                      # (what is burrowed: seen going down, and not up or moved since)
            if u["order"] in BURROWED:
                self.down.setdefault(t, (u["x"], u["y"]))
            elif t in self.down and (u["order"] == UNBURROWING or math.dist(self.down[t], (u["x"], u["y"])) > MOVED_PX):
                del self.down[t]
        self.down = {t: xy for t, xy in self.down.items() if t in lurks}
        loose = [u for t, u in lurks.items() if t not in self.lurk]
        if loose:
            sel = set(obs["selection"])
            if sel and sel <= set(lurks) and sel & {u["tag"] for u in loose}:
                (hi.hotkey_add if self.lurk else hi.hotkey_set)(LURKER_KEY)
                self.lurk |= sel
                g.say(f"lurkers: {len(self.lurk)}")
                return True
            u = loose[0]
            if self.host._look(obs, (u["x"], u["y"])):
                return True
            at = screen_of(obs, u["x"], u["y"])
            if at is None:
                return False
            hi.ctrl_click(*at)
            return True
        group = [lurks[t] for t in self.lurk]
        if not group:
            return False
        mid = (sum(u["x"] for u in group) / len(group), sum(u["y"] for u in group) / len(group))
        burrowed = sum(1 for u in group if u["tag"] in self.down) * 2 >= len(group)
        enemy = [u for u in obs["units"] if u["owner"] not in (g.slot, 11) and u["type"] not in WORKERS
                 and math.dist((u["x"], u["y"]), mid) < LURKER_NEAR]
        if enemy:
            self.lurk_calm = hi.frame
        target = g.rally
        if enemy and not burrowed:
            return self._key(obs, "burrow")
        far = math.dist(mid, target) > 4 * 32
        if burrowed and far and not enemy and hi.frame - self.lurk_calm >= LURKER_CALM:
            return self._key(obs, "unburrow")
        if not burrowed and far and hi.frame - self.lurk_since >= 4 * 24:
            if not self._recalled(obs):
                return True
            hi.minimap_right_click(int(target[0]), int(target[1]))
            self.lurk_since = hi.frame
            return True
        if not burrowed and not far:
            return self._key(obs, "burrow")
        return False

    def _recalled(self, obs: dict) -> bool:
        """The lurkers selected (True), or their hotkey pressed now (False: the turn's action)."""
        if set(obs["selection"]) and set(obs["selection"]) <= self.lurk:
            return True
        self.g.hi.hotkey_recall(LURKER_KEY)
        return False

    def _key(self, obs: dict, key: str) -> bool:
        if not self._recalled(obs):
            return True
        (self.g.hi.burrow if key == "burrow" else self.g.hi.unburrow)()
        self.lurk_since = self.g.hi.frame
        return True
