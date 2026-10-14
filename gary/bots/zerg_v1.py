"""A better scripted Zerg sparring partner: the rush dummy's build (gary/bots/zerg_rush.py: 9 pool,
then zerglings), with an army that plays its zerglings the way a player would.

The dummy attack-moves every 12 zerglings straight at Gary's start: they trickle into whatever is
there and die. This one:

  gather     each 12 zerglings made are a control group (hotkeys 1-8), sent to a staging point
             outside Gary's natural's choke (its way in on a standard map: the main's ramp is
             behind it), where the groups wait for each other
  judge      it attacks when Gary's defense is weaker: its zerglings (0.45 a marine each) against
             the most of Gary's fighters it saw at once near Gary's bases in the last 3 min, or sees
             at the choke now if more (marines, firebats, medics, vultures, tanks, bunkers as 4 marines,
             fighting SCVs), 1.25 to 1; or, after waiting 50 s, 0.9 to 1. Else it waits and grows
  surround   then it moves the zerglings past the defenders (moving zerglings don't stop to fight
             the first marine they meet) and attack-moves once half of a group is wrapped around
             Gary's units, or after 2 s, at Gary's start
  retreat    a group facing more than it is (what it sees around it, 0.8 to 1), runs back to the
             staging point, and the zerglings there decide again
  defend     Gary's fighters within 10 tiles of the hatchery: every group, and the zerglings at home
             not in one yet, attack-move home (until none is left there)

It sees through the fog like Gary (what its units see), knows the map (where Gary's start and
natural are, like the dummy) and plays through the human interface: hotkeys, minimap orders.

    python -m eval.vs_rush --versions v022 --opponent v1 --seeds 1 2 3 4

Against Gary v0.11 (immediate attacks, 12-minute games, 48 seeds), Gary lost: to the dummy 5, to
v1 2, to v1a (always attacks, never retreats) 4, to v1d (v1a without the surround) 5. Hunting SCVs
(attacking toward the mineral line, re-aimed every 5 s at the SCVs near) made it 1: zerglings
chasing pulled SCVs ignore the marines shooting them. With a 9-pool and 9 drones, waiting and
retreating give Gary the time it needs: the smart parts don't make this build harder to hold.
"""

from __future__ import annotations

import math

from gary import commands as C
from gary.bots.zerg_rush import LING, WAVE, ZergRush
from gary.interface import screen_of

GROUPS = range(1, 9)
LING_POWER = 0.45
POWER = {0: 1.0, 32: 1.2, 34: 0.4, 2: 1.5, 5: 3.0, 30: 3.0, 3: 2.0, 1: 1.0, 125: 4.0, 7: 0.25}
ATTACK_RATIO, PATIENT_RATIO, PATIENCE = 1.25, 0.9, 50 * 24
STAGE_OUT = 6 * 32              # the staging point: this far outside the natural's choke
AT_STAGE = 5 * 32
DEFENSE_NEAR = 12 * 32          # Gary's units this near the choke are the defense
PAST = 3 * 32                   # surround: move this far past the defenders
WRAPPED = 48
SURROUND_FOR = 48
REISSUE = 5 * 24                # attacking: the attack-move again this often (zerglings that lost
                                # their target stand)
RETREAT_RATIO = 1.25            # a group retreats when what it sees around it is this much stronger
MEMORY = 180 * 24               # Gary's fighters seen near its bases count this long
GARY_NEAR = 25 * 32
HOME_NEAR = 10 * 32             # Gary's fighters this near the hatchery: everyone home


class ZergV1(ZergRush):
    attack_ratio, patient_ratio, retreat_ratio = ATTACK_RATIO, PATIENT_RATIO, RETREAT_RATIO
    surround = True
    wave = WAVE                     # zerglings in a new group
    sizes = {LING: 1.0}             # the army's unit types, each as so many zerglings (Zerg v3: hydralisks)
    hunt = False                    # (attacking: aim at the mineral line, and again every REISSUE at SCVs
                                    # near: zerglings chasing pulled SCVs ignore the marines shooting them;
                                    # Gary lost 1 game in 48 to that, 5 without it)

    def __init__(self, hi, mapinfo):
        super().__init__(hi, mapinfo)
        self._aimed_at = None
        self.seen: list[tuple[int, float]] = []          # (frame, Gary's defense seen near its bases then)
        self.groups: dict[int, set[int]] = {}
        self.phase: dict[int, str] = {}
        self.since: dict[int, int] = {}
        self.sent_with: dict[int, int] = {}
        self.staged_since: int | None = None
        self.form_after, self.form_tries = 0, 0
        self.recalled = None

    def gather_point(self):
        """Where new zerglings wait to be grouped: the hatchery (its rally is 5 tiles off)."""
        return self.home

    def _aim(self) -> None:
        """Where Gary is (self.target, set by whoever runs the match): its natural's choke, the
        staging point outside it, its main's mineral line."""
        from gary.terrain import map_analysis
        starts = map_analysis(self.hi.game, self.map)["starts"]
        gary = min(starts.values(), key=lambda s: math.dist(s["start"], self.target))
        nat, choke = gary["natural"], gary["natural_choke"]
        d = math.dist(nat, choke) or 1
        self.choke = tuple(choke)
        self.staging = (int(choke[0] + (choke[0] - nat[0]) / d * STAGE_OUT), int(choke[1] + (choke[1] - nat[1]) / d * STAGE_OUT))
        base = self.map.base_near(self.target)
        self.minerals = (sum(x for x, _ in base.minerals) / len(base.minerals), sum(y for _, y in base.minerals) / len(base.minerals))
        self._aimed_at = self.target

    def army(self, obs: dict, mine: list[dict]) -> bool:
        hi = self.hi
        if self._aimed_at != self.target:
            self._aim()
        lings = {u["tag"]: u for u in mine if u["type"] in self.sizes and u["completed"]}
        for n in list(self.groups):
            self.groups[n] &= set(lings)
            if not self.groups[n]:
                del self.groups[n], self.phase[n]
        grouped = set().union(*self.groups.values()) if self.groups else set()
        # a new group: 12 zerglings at the gathering point, not in a group, once attacks are allowed
        gather = self.gather_point()
        waiting = [u for t, u in lings.items() if t not in grouped and math.dist((u["x"], u["y"]), gather) < 10 * 32]
        fresh = [u for u in waiting if screen_of(obs, u["x"], u["y"])]
        if len(waiting) >= self.wave and len(fresh) < self.wave // 2 and hi.frame >= self.attack_after \
                and hi.frame >= self.form_after:
            hi.camera_minimap(*_mid(waiting))                  # look at them first
            self.form_tries += 1
            if self.form_tries >= 3:                           # (they keep walking off the screen)
                self.form_after, self.form_tries = hi.frame + 3 * 24, 0
            return True
        if len(fresh) >= self.wave and hi.frame >= self.attack_after and hi.frame >= self.form_after:
            n = next((k for k in GROUPS if k not in self.groups), None)
            if n is not None:
                sel = set(obs["selection"])
                if self.stage == "forming":
                    self.stage = ""
                    if sel and sel <= set(lings) and len(sel - grouped) >= self.wave // 2:
                        hi.hotkey_set(n)
                        self.groups[n], self.phase[n], self.since[n] = set(sel), "to_staging", hi.frame
                        for k in self.groups:              # (a zergling is in one group)
                            if k != n:
                                self.groups[k] -= sel
                        self.form_tries = 0
                        return True
                    self.form_tries += 1
                    if self.form_tries >= 2:               # leave the hands to the macro for a while
                        self.form_after, self.form_tries = hi.frame + 3 * 24, 0
                        return False
                # a ctrl+click on one: every zergling on the screen, nothing else (12 at most)
                u = min(fresh, key=lambda u: math.dist((u["x"], u["y"]), _mid(fresh)))
                hi.ctrl_click(*screen_of(obs, u["x"], u["y"]))
                self.stage = "forming"
                return True
        enemy = [u for u in obs["units"] if u["owner"] not in (hi.slot, 11) and u["type"] in POWER]
        # what it learns of Gary's defense: the most it saw at once near Gary's bases lately
        now_power = sum(POWER[e["type"]] for e in enemy if (e["type"] != 7 or e["order"] in (10, 14))
                        and (math.dist((e["x"], e["y"]), self.target) < GARY_NEAR
                             or math.dist((e["x"], e["y"]), self.choke) < GARY_NEAR))
        if now_power:
            self.seen.append((hi.frame, now_power))
        self.seen = [(f, p) for f, p in self.seen if hi.frame - f <= MEMORY]
        homes = [(u["x"], u["y"]) for u in obs["units"] if u["owner"] == hi.slot and u["type"] in (131, 132, 133)] or [self.home]
        raid = [e for e in enemy if e["type"] not in (7, 125) and min(math.dist((e["x"], e["y"]), h) for h in homes) < HOME_NEAR]
        if raid:
            at = _mid(raid)
            for n in sorted(self.groups):
                if self.phase[n] != "defend":
                    return self._order(n, obs, "attack", at, "defend")
            loose = [u for t, u in lings.items() if t not in grouped and screen_of(obs, u["x"], u["y"])
                     and u["order"] not in (10, 14)]
            if loose:
                sel = set(obs["selection"])
                if self.stage == "defending" and sel and sel <= set(lings):
                    hi.order_click(C.ORDER_ATTACK_MOVE, *screen_of(obs, int(at[0]), int(at[1])) or (0, 0)) \
                            if screen_of(obs, int(at[0]), int(at[1])) else hi.minimap_command(C.ORDER_ATTACK_MOVE, int(at[0]), int(at[1]))
                    self.stage = ""
                    return True
                self._box(obs, loose)
                self.stage = "defending"
                return True
            return False
        for n in sorted(self.groups):
            if self.phase[n] == "defend":                  # the raid is over: back to staging
                return self._order(n, obs, "move", self.staging, "staging")
            if self._group(n, obs, lings, enemy):
                return True
        return False

    def _group(self, n: int, obs: dict, lings: dict, enemy: list[dict]) -> bool:
        hi = self.hi
        members = [lings[t] for t in self.groups[n]]
        mid = _mid(members)
        phase = self.phase[n]
        if phase == "to_staging":
            return self._order(n, obs, "move", self.staging, "staging")
        if phase in ("staging", "retreat"):
            if sum(math.dist((u["x"], u["y"]), self.staging) < AT_STAGE for u in members) * 10 >= 7 * len(members):
                self.phase[n] = "staged"
            return False
        if phase == "staged":
            staged = [k for k, p in self.phase.items() if p == "staged"]
            if self.staged_since is None:
                self.staged_since = hi.frame
            ours = LING_POWER * sum(self._size(lings[t] for t in self.groups[k]) for k in staged)
            near = [e for e in enemy if math.dist((e["x"], e["y"]), self.choke) < DEFENSE_NEAR
                    and (e["type"] != 7 or e["order"] in (10, 14))
                    and (e["type"] != 125 or e["completed"])]
            remembered = max((p for _, p in self.seen), default=0.0)
            theirs = max(sum(POWER[e["type"]] for e in near), remembered)
            waited = hi.frame - self.staged_since
            if ours >= self.attack_ratio * theirs or (waited >= PATIENCE and ours >= self.patient_ratio * theirs):
                self.log(f"attacks: {ours:.1f} against {theirs:.1f} seen at the choke, after {waited / 24:.0f} s")
                for k in staged:                           # every staged group goes
                    self.phase[k] = "go"
                    self.sent_with[k] = len(self.groups[k])
                self.staged_since = None
                self.defenders = _mid(near) if near else None
            return False
        if phase == "go":
            if self.defenders is None or not self.surround:   # nothing seen to go around
                return self._order(n, obs, "attack", self.minerals if self.hunt else self.target, "attack")
            d = math.dist(self.defenders, self.minerals) or 1
            past = (self.defenders[0] + (self.minerals[0] - self.defenders[0]) / d * PAST,
                    self.defenders[1] + (self.minerals[1] - self.defenders[1]) / d * PAST)
            return self._order(n, obs, "move", past, "surround")
        if phase == "surround":
            wrapped = sum(any(math.dist((u["x"], u["y"]), (e["x"], e["y"])) < WRAPPED for e in enemy) for u in members)
            if 2 * wrapped >= len(members) or hi.frame - self.since[n] >= SURROUND_FOR:
                return self._order(n, obs, "attack", self.minerals if self.hunt else self.target, "attack")
            return False
        if phase == "attack":
            near = [e for e in enemy if math.dist((e["x"], e["y"]), mid) < 8 * 32
                    and (e["type"] != 7 or e["order"] in (10, 14))]
            if sum(POWER[e["type"]] for e in near) > self.retreat_ratio * LING_POWER * self._size(members):
                return self._order(n, obs, "move", self.staging, "retreat")
            if self.hunt and hi.frame - self.since[n] >= REISSUE:
                scvs = [e for e in enemy if e["type"] == 7 and math.dist((e["x"], e["y"]), mid) < 10 * 32]
                target = _mid(scvs) if scvs else self.minerals
                return self._order(n, obs, "attack", target, "attack")
        return False

    def _size(self, units) -> float:
        """The units as so many zerglings (a count, for zerglings alone)."""
        return sum(self.sizes[u["type"]] for u in units)

    def _order(self, n: int, obs: dict, kind: str, point, then: str) -> bool:
        """Group n selected (its hotkey), then a minimap order; the group's phase becomes then."""
        hi = self.hi
        sel = set(obs["selection"])
        if self.recalled == n and sel:                     # what the hotkey brings up is the group
            self.groups[n] = sel
        elif not sel or not sel <= self.groups[n]:
            hi.hotkey_recall(n)
            self.recalled = n
            return True
        self.recalled = None
        x, y = int(point[0]), int(point[1])
        if kind == "move":
            hi.minimap_right_click(x, y)
        else:
            hi.minimap_command(C.ORDER_ATTACK_MOVE, x, y)
        if then != self.phase[n]:
            self.log(f"group {n} ({len(self.groups[n])} zerglings): {then}")
        self.phase[n], self.since[n] = then, hi.frame
        return True

    def log(self, text: str) -> None:
        t = self.hi.frame // 24
        print(f"{t // 60:2d}:{t % 60:02d} Zerg {text}", flush=True)


class ZergV1Aggressive(ZergV1):
    """v1a: v1 that always attacks and never retreats (the dummy's relentlessness, v1's execution)."""
    attack_ratio, patient_ratio, retreat_ratio = 0.0, 0.0, float("inf")


class ZergV1Direct(ZergV1Aggressive):
    """v1d: v1a without the surround (attack-moves straight in)."""
    surround = False


def _mid(units) -> tuple[float, float]:
    return sum(u["x"] for u in units) / len(units), sum(u["y"] for u in units) / len(units)
