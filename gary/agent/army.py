"""The army (layer L3, docs/CONTROL.md): where groups go, and scouting.

  waves     every 16 army units attack-move to an enemy start (Gary v0.1-v0.2)
  learned   the army model (gary/policy/army.py, pro TvZ army orders; v0.3+): every 2 s it is asked
            about one of up to 3 clusters of the army; when it puts most of its belief on the
            cluster going 2+ grid cells away, says so twice, and its fight estimate doesn't expect
            going to cost army share, Gary drag-boxes the cluster and orders it on the minimap.
            Also the scout: one worker at 9 SCVs to the enemy start locations; where it finds the
            enemy, once around their main (7 tiles out: a Spawning Pool can stand out of sight
            of the hatchery), then home.
  hunt      (option) from 8:00, with 20+ army units and no enemy army seen for a minute, the army
            goes where enemy buildings were seen, then to the bases it hasn't passed lately: the
            model stays near what it knows, and a hatchery Gary never saw outlived the game.
            With scan, a Comsat Station with energy scans those bases first.
  watch     (option) the scout stays, as pros' first SCVs do (in pro TvZ the Terran has Zerg drones
            in sight 22 s of 3:00-6:00, any Zerg 104 s; Gary without this: 5 s and 44 s): after
            its circle it keeps touring the Zerg's main and natural (waypoints shift+clicked once:
            it walks them alone; a new tour from the point farthest from zerglings that come near)
            until they hunt it (STAY_LEAVE_LINGS near, or hurt) or STAY_UNTIL; then it holds a post outside Gary's natural toward theirs (where an
            attack gathers), backing off behind the natural while enemy fighters are near it and
            back after. A scout lost before RESCOUT_BY is replaced once.

Units in a fight are the fight layer's; the arbiter leaves them out (and the posture can keep the
army home altogether). Orders are intents (ARMY_MOVE, ARMY_ATTACK, SCOUT, ATTACK_WAVE) that the
motor layer (motor.py) carries out.
"""

from __future__ import annotations

import math

import numpy as np

from gary import terran as T
from gary.agent.gary import Gary
from gary.agent.intents import Intent
from gary.policy.army import GRID, ArmyModel, ArmyTracker, is_building, supply_x2
from gary.policy.belief import ARMY_SUPPLY

WAVE = 16
ASK_EVERY = 24 * 2                  # how often Gary asks the model about its army
MOVE_CELLS = 2                      # destinations closer than this are "stay"
MOVE_P = 0.5                        # move when the model puts more belief than this away from here
PRO_MOVES = {4: 1.0, 5: 2.2, 6: 3.2, 7: 3.5, 8: 4.0, 9: 4.3, 10: 4.4, 11: 5.0, 12: 5.4, 13: 5.3, 14: 6.3, 15: 6.0}
                                    # pro Terrans' army moves of 2+ cells per minute (400 TvZ games' orders, mean)
FIGHT_MIN = -0.02                   # don't go if the fight estimate says we'd lose 2+ points of army share
FIGHT_MARGIN = 0.01                 # ...or if going looks worse than staying by more than 1 point
REORDER_AFTER = 24 * 8              # don't repeat the same order to a group within 8 s
SCOUT_AT = 24 * 45                  # send a scouting worker from 0:45, at 9 SCVs (like pros)
SCOUT_WORKERS = 9
WAYPOINT_FOR = 8 * 24                # the scout gives up on a waypoint of its circle after this
CIRCLE = 7 * 32                     # the scout circles the enemy main this far from its start location
SCOUT_CHECK = 3 * 24                # the scout on its way: every 3 s it should be moving, or it's sent
SCOUT_RESENDS = 3                   # again (this many times, then another SCV)
SCOUT_HOME = 12 * 32                # px: a scout this close to the main, standing, is home (a worker again)
MOVE_ORDER = 6
HUNT_AFTER = 8 * 60 * 24            # hunt: from 8:00, with this many army units and no enemy army seen
HUNT_ARMY = 20                      # for QUIET, the army goes where enemy buildings were, then to
QUIET = 60 * 24                     # the bases it hasn't passed lately
HUNT_FOR = 60 * 24                  # ...each for this long at most
HOME_PX = 12 * 32                   # belief: destinations this close to a Command Center are home
SCAN_ENERGY = 50                    # hunt with scans: a Comsat Station with this much energy scans a base first
NOT_ARMY = {35, 36, 41, 42, 7, 64}  # larva, egg, drone, overlord, SCV, probe: not an enemy army
SCAN_STUCK = 20 * 24                # hunt: a remembered building's spot not settled after this: scanned
HUNT_BOXES = 8                      # hunt: drag boxes for the whole army (4 left 29 of 77 units standing)
RESCAN_SPOT = 60 * 24               # hunt: a remembered building's spot scanned again after this at most
GONE_PX = 7 * 32                    # an enemy building is gone when Gary's units are this close and don't see it
VISITED_FOR = 120 * 24              # a base Gary's units passed this recently isn't visited again
STAY_UNTIL = 5 * 60 * 24            # watch: the scout stays in the enemy's bases until then at most
STAY_LEAVE_LINGS = 3                # ...or until this many enemy fighters are near it
STAY_DANGER = 5 * 32                # ...near: this close
STAY_HURT = 0.6                     # ...or it's down to this share of its HP
STAY_RING = 6 * 32                  # ...going around their bases this far from each one's center
STAY_EVERY = 2 * 24                 # ...a new tour at most this often
POST_OUT = 10 * 32                  # watch post: this far out from Gary's natural's choke, toward their natural
POST_BACK = 7 * 32                  # ...enemy fighters this close: back behind the natural
POST_CALM = 10 * 24                 # ...back on the post after this long without them
RESCOUT_BY = 270 * 24               # a scout lost before 4:30 is replaced (once)


def cell_dist(a: int, b: int) -> int:
    return max(abs(a // GRID - b // GRID), abs(a % GRID - b % GRID))


class WaveArmy:
    remembers = False

    def __init__(self, g: Gary):
        self.g = g
        self.attacks_sent = 0
        self.army_sent: set[int] = set()

    def step(self, obs: dict, mine: list[dict], done: list[dict]) -> str | None:
        """Attack-move with every 16 army units. "army" if Gary acted."""
        g = self.g
        army = [u for u in done if u["type"] in g.army_types and u["tag"] not in self.army_sent]
        if len(army) >= WAVE:
            g.motor.start(Intent("ATTACK_WAVE", [], None, {}, "army"))
            return "army"
        return None



class LearnedArmy:
    remembers = True                # whatever Gary looks at, the tracker remembers

    def __init__(self, g: Gary, model: ArmyModel, pull=None, hunt: bool = False, scan: bool = False, belief=None,
                 belief_total: bool = False, watch: bool = False, home_guard: bool = False, belief_hold: bool = True,
                 hunt_belief: bool = False, hold_moves: bool = True, hold_margin: float | None = None,
                 draw: bool = False, engage=None):
        self.g = g
        # option: draw the destination and move/attack from the model's odds (the league's learner: RL
        # learns from drawn decisions and their probabilities, logged), instead of the likeliest cell
        # away when most of the odds lie away, said twice (Gary-Z's likeliest-cell army stood at home)
        self.draw = draw
        self.rng = np.random.default_rng(g.hi.slot * 7919 + int(getattr(g, "seed", 0) or 0))
        self.record: list | None = None        # a paired episode's arm: its drawn decisions, kept here
        self._seen: dict = {}                  # what the last question saw, in brief (the game's records)
        self.then_rng = None                   # a paired episode's arm: the shared dice after its first decision
        # option (with draw): the attack-or-hold model (train/engage.py: an army model whose kind head is
        # engage/hold, learned from the pros' orders, then RL on paired branches): it draws whether the
        # group goes into the enemy, and the destination is drawn from the army model's odds among the
        # cells next to known enemy presence, or among the others (the destinations stay imitation)
        self.engage = engage
        self.force_engage: bool | None = None  # a paired episode's arm: its first decision's engage forced
        self.move_tokens, self.tokens_at = 1.0, 0  # draw: moves paced at the pros' rate (PRO_MOVES)
        # option: whether the fight estimate may hold a plain move (positioning), and by how much going must
        # look worse than staying (and lose outright) to hold. v0.27's 1 and 2 points held about 90% of its
        # army's decisions at 5-6 min, 92% of them within 2 points: noise for an estimate correlating 0.40
        # (the army stood still 78% of 5-7 min, pros 37%: the humanlikeness table's tell)
        self.hold_moves = hold_moves
        self.fight_min = FIGHT_MIN if hold_margin is None else -hold_margin
        self.fight_margin = FIGHT_MARGIN if hold_margin is None else hold_margin
        self.belief_hold = belief_hold  # belief: groups going away hold when the hidden army outweighs them
        self.hunt_belief = hunt_belief  # option (with hunt and a belief model): hunt only with an army at
                                        # least the belief's 80th percentile of the zerg's
        self.home_guard = home_guard  # option: no army moves out while enemy fighters are at home
        self.watch = watch          # option: the scout stays in their bases, then watches outside the natural
        self.watch_stage = "stay"   # stay -> post
        self.dodged = False
        self.watch_next = 0
        self.watch_back_since = None
        self.rescouted = False
        self.post = None
        self.belief_total = belief_total   # belief: compare with all the enemy army it likely has, seen or not
        self.belief = belief        # option: the belief model (gary/policy/belief.py): hold groups the
        self.belief_mem = belief.memory() if belief is not None else None   # hidden enemy army outweighs
        self.scan = scan            # option (with hunt): bases are scanned before the army walks there
        self.hunt = hunt            # option: with no enemy building known, the army visits the bases
        self.enemy_buildings: dict[int, tuple[int, int]] = {}   # tag -> where it was seen
        self.enemy_army_seen = 0                                 # frame an enemy fighter was last seen
        self.hunted: dict[tuple[int, int], int] = {}             # hunt target -> frame first sent there
        self.visited: dict[int, int] = {}                       # base index -> frame Gary's units were there
        self.next_hunt = 0
        self.scanned: dict[tuple[int, int], int] = {}            # hunt: a remembered building's spot -> frame scanned
        self.model = model
        self.tracker = ArmyTracker(model.spec, g.slot, g.map.size, g.main.center)
        self.pull = pull            # a worker-pull rule run first (v0.3; later versions: the fight layer's)
        self.next_ask = 0
        self.ask_turn = 0
        self.proposed: dict[int, int] = {}            # cluster cell -> destination the model proposed
        self.ordered: dict[int, tuple[int, bool, int]] = {}   # cluster cell -> (cell, attack, frame)
        self.held: dict[int, int] = {}                # cluster cell -> destination Gary decided against
        self.scout_targets: list[tuple[int, int]] = []
        self.scout_done = False
        self.circle: list | None = None                 # the scout's way around the enemy main, once found
        self.next_circle_order = 0
        self.waypoint, self.waypoint_since = None, 0
        self.ordered_to = None
        self.scout_check = None     # (frame, (scout, target)) at the last check
        self.scout_resends = 0

    def remember(self, obs: dict) -> None:
        self.tracker.remember(obs)
        if self.belief_mem is not None:
            seen = np.zeros(228, np.float32)
            for u in obs["units"]:
                if u["owner"] not in (self.g.slot, 11) and 0 <= u["type"] < 228:
                    seen[u["type"]] += 1
            self.belief_mem.look(obs["frame"], seen)

    def _clusters(self, army: list[dict]) -> list[list[dict]]:
        """The army split by where it stands: up to 3 clusters (the main army, reinforcements on
        their way, ...), biggest first. Units within 2 grid cells of a cluster's center join it."""
        left, out = list(army), []
        while left and len(out) < 3:
            cells = [self.tracker.cell(u["x"], u["y"]) for u in left]
            center = max(set(cells), key=cells.count)
            near = [u for u, c in zip(left, cells) if cell_dist(c, center) <= 2]
            out.append(near)
            left = [u for u in left if u not in near]
        return out

    def step(self, obs: dict, mine: list[dict], done: list[dict]) -> str | None:
        """Who acted ("rule:pull", "scout", "army"), or None."""
        g = self.g
        hi = g.hi
        if self.pull is not None and self.pull.step(obs, mine):
            return "rule:pull"
        if self._scouting(obs, mine):
            return "scout"
        if self.watch and self.scout_done and self._watching(obs, mine):
            return "scout"
        hunt = self._hunt(obs, mine, done) if self.hunt else None
        if hunt is not None:                    # (the model isn't asked while the army hunts)
            return "army" if hunt else None
        if hi.frame < self.next_ask:
            return None
        if self.home_guard:
            from gary.agent.fight import enemy_fighters, near_home
            if near_home(mine, enemy_fighters(obs, g.slot)):
                self.next_ask = hi.frame + ASK_EVERY
                return None
        army = [u for u in done if u["type"] in g.army_types]
        clusters = self._clusters(army)
        if not clusters:
            return None
        self.next_ask = hi.frame + ASK_EVERY // len(clusters)       # one cluster per question
        group = clusters[self.ask_turn % len(clusters)]
        self.ask_turn += 1
        grid, glob = self.tracker.inputs(obs, group)
        where, kind = self.model.predict(grid, glob)
        cx = sum(u["x"] for u in group) / len(group)
        cy = sum(u["y"] for u in group) / len(group)
        here = self.tracker.cell(cx, cy)
        # move or stay: by how much of the model's belief lies away from here (spread over many
        # cells, "elsewhere" can be likelier than "here" even when "here" is the likeliest cell)
        away = np.array([cell_dist(c, here) >= MOVE_CELLS for c in range(GRID * GRID)])
        p_away = float(where[0, away].sum())
        seen = {"units": len(group), "at": self._cell_name(here), "p_away": p_away, "p_attack": float(kind[0, 1]),
                "tokens": self.move_tokens}
        self._seen = seen
        if self.draw:
            # (stay unless most of the odds lie away, as without draw: the model gives where a pro sends a
            # group when ordering it, not where it should be every 2 s; drawn every ask, the army roamed:
            # at home 25-29% of 8-10 min against the pros' 60%)
            if p_away < MOVE_P:
                self._record(where, "stay", None, f"rule: stays unless {MOVE_P:.0%} of the odds lie away")
                return None
            # (paced at the pros' rate of army moves of 2+ cells: they make 1-6 a minute, rising with the
            # game; Gary-T's army layer made 9-19, and its army roamed, at home 20-29% of 8-10 min against
            # the pros' 60%; a move spends a token, the tokens come at the pros' rate for the minute)
            m = min(15, hi.frame // 1440)
            self.move_tokens = min(2.0, self.move_tokens + (hi.frame - self.tokens_at) * PRO_MOVES.get(m, 6.0) / 1440)
            self.tokens_at = hi.frame
            if self.move_tokens < 1.0:
                self._record(where, "stay", None, f"rule: paced at the pros' {PRO_MOVES.get(m, 6.0):g} moves a minute")
                return None
            where = np.where(away[None], where, 0.0)
            return self._drawn(obs, mine, group, grid, glob, where, kind, here)
        if p_away < MOVE_P:
            self.proposed.pop(here, None)
            self._record(where, "stay", None, "favorite: most odds here")
            return None                                           # stay
        best = int(np.where(away, where[0], -1).argmax())          # where, if moving
        # move or attack-move as the model says, but never walk (rather than fight) into where
        # enemy buildings were seen. (Always attack-moving was tried: the model's plain moves are
        # often regrouping or pulling back, and stopping to fight there lost more games.)
        attack = int(kind[0].argmax()) == 1
        near_enemy = self.tracker.seen_buildings.reshape(GRID, GRID)[
            max(0, best // GRID - 2):best // GRID + 3, max(0, best % GRID - 2):best % GRID + 3].any()
        attack = attack or bool(near_enemy)
        proposed, self.proposed[here] = self.proposed.get(here), best
        if proposed is None or cell_dist(proposed, best) > 1:
            return None                                           # wait until it says so twice
        last = self.ordered.get(here)
        if last and cell_dist(last[0], best) <= 1 and last[1] == attack and hi.frame - last[2] < REORDER_AFTER:
            return None
        # will it go well? the model's fight estimate for going there vs staying put
        go = self.model.fight(grid, glob, best, attack)
        stay = self.model.fight(grid, glob, here, False)
        if go < max(self.fight_min, stay - self.fight_margin) and (attack or self.hold_moves):
            if self.held.get(here, -1) != best:
                g.say(f"army ({len(group)} units) holds: going to {self.tracker.center(best)} "
                      f"looks like {go:+.0%} army share, staying {stay:+.0%}")
            self.held[here] = best
            return None
        hidden = self._hidden_army(obs, mine, best)
        if hidden is not None and hidden > sum(supply_x2(u["type"]) for u in group):
            # the fight estimate knows only the enemy army seen lately; the belief model, what the
            # enemy likely has (its 80th percentile): out there unseen, more than this group
            if self.held.get(here, -1) != best:
                g.say(f"army ({len(group)} units) holds: going to {self.tracker.center(best)}, the zerg likely "
                      f"has {hidden / 2:.0f} supply more army than seen")
            self.held[here] = best
            return None
        self.held.pop(here, None)
        x, y = self.tracker.center(best)
        self.ordered[here] = (best, attack, hi.frame)
        g.motor.start(Intent("ARMY_ATTACK" if attack else "ARMY_MOVE", [u["tag"] for u in group], (x, y),
                             {"tags": [u["tag"] for u in group], "x": x, "y": y, "attack": attack, "p": p_away,
                              "fight": go, "done": [], "boxes": 0}, "army"))
        return "army"

    def _drawn(self, obs, mine, group, grid, glob, where, kind, here) -> str | None:
        """draw: a destination and move/attack drawn from the model's odds; a cell near here is staying.
        The hold (the fight estimate) and the repeat guard as without draw. Logged ("army_t")."""
        from gary.decision_log import LOG
        g, hi = self.g, self.g.hi
        p = where[0].astype(np.float64)
        if self.engage is not None:
            return self._engage_drawn(obs, mine, group, grid, glob, p, kind, here)
        best = int(self.rng.choice(len(p), p=p / p.sum()))
        drawn_attack = bool(self.rng.random() < float(kind[0, 1]))
        if self.then_rng is not None:          # (an arm's first decision drawn its own way, the rest shared)
            self.rng, self.then_rng = self.then_rng, None
        logp = float(np.log(p[best] + 1e-9) + np.log(kind[0, int(drawn_attack)] + 1e-9))
        def log(outcome):
            item = dict(frame=hi.frame, grid=grid[0].astype(np.float16), glob=glob[0].astype(np.float16), cell=best,
                        attack=drawn_attack, logp=logp, outcome=outcome)
            (self.record.append(item) if self.record is not None else LOG.add("army_t", **item))
            self._record(where, f"{'attack' if drawn_attack else 'move'} to {self._cell_name(best)}", float(np.exp(logp)),
                         "draw", {"stay": "stays: the cell drawn is near", "repeat": "not again: just ordered there",
                                  "held": "held: the fight estimate", "go": "sent"}[outcome])
        if cell_dist(best, here) < MOVE_CELLS:
            log("stay")
            return None
        self.move_tokens -= 1.0
        near_enemy = self.tracker.seen_buildings.reshape(GRID, GRID)[
            max(0, best // GRID - 2):best // GRID + 3, max(0, best % GRID - 2):best % GRID + 3].any()
        attack = drawn_attack or bool(near_enemy)
        last = self.ordered.get(here)
        if last and cell_dist(last[0], best) <= 1 and last[1] == attack and hi.frame - last[2] < REORDER_AFTER:
            log("repeat")
            return None
        go = self.model.fight(grid, glob, best, attack)
        stay = self.model.fight(grid, glob, here, False)
        if go < max(self.fight_min, stay - self.fight_margin) and (attack or self.hold_moves):
            log("held")
            return None
        log("go")
        x, y = self.tracker.center(best)
        self.ordered[here] = (best, attack, hi.frame)
        g.motor.start(Intent("ARMY_ATTACK" if attack else "ARMY_MOVE", [u["tag"] for u in group], (x, y),
                             {"tags": [u["tag"] for u in group], "x": x, "y": y, "attack": attack, "p": float(p[best]),
                              "fight": go, "done": [], "boxes": 0}, "army"))
        return "army"

    def _engage_drawn(self, obs, mine, group, grid, glob, p, kind, here) -> str | None:
        """The engage option: attack or hold drawn from the engage model (or forced, a paired arm's first
        decision), the destination from the army model's odds among the cells next to enemy presence the
        player knows (attack) or the others (hold: a move, or staying). No fight-estimate hold: whether to
        go in is this decision's. Logged ("engage_t": the inputs, engage, its log-probability)."""
        from gary.decision_log import LOG
        g, hi = self.g, self.g.hi
        _, ek = self.engage.predict(grid, glob)
        pe = float(ek[0, 1])
        forced = self.force_engage is not None
        if self.force_engage is not None:
            eng, self.force_engage = bool(self.force_engage), None
        else:
            eng = bool(self.rng.random() < pe)
        m = np.asarray(grid[0]).reshape(-1, GRID, GRID)
        enemy = (m[3] > 0) | (m[6] > 0) | (m[7] > 0)
        near = enemy.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                near |= np.roll(np.roll(enemy, dy, 0), dx, 1)
        mask = near.reshape(-1) if eng else ~near.reshape(-1)
        q = p * mask
        if q.sum() <= 0:
            q = p
        best = int(self.rng.choice(len(q), p=q / q.sum()))
        if self.then_rng is not None:
            self.rng, self.then_rng = self.then_rng, None
        item = dict(frame=hi.frame, grid=grid[0].astype(np.float16), glob=glob[0].astype(np.float16), engage=eng,
                    logp=float(np.log((pe if eng else 1 - pe) + 1e-9)))
        (self.record.append(item) if self.record is not None else LOG.add("engage_t", **item))
        LOG.decision("T", "engage", hi.frame, f"{'engage' if eng else 'hold'}: to {self._cell_name(best)}",
                     pe if eng else 1 - pe, [["engage", round(pe, 3)], ["hold", round(1 - pe, 3)]], self._seen,
                     "draw" if self.force_engage is None and not forced else "forced (a paired arm)",
                     then="stays: the cell drawn is near" if cell_dist(best, here) < MOVE_CELLS else "sent")
        if cell_dist(best, here) < MOVE_CELLS:
            return None
        self.move_tokens -= 1.0
        attack = eng or bool(self.rng.random() < float(kind[0, 1]))
        last = self.ordered.get(here)
        if last and cell_dist(last[0], best) <= 1 and last[1] == attack and hi.frame - last[2] < REORDER_AFTER:
            return None
        x, y = self.tracker.center(best)
        self.ordered[here] = (best, attack, hi.frame)
        self._record(where, f"{'attack' if attack else 'move'} to {self._cell_name(best)}", float(where[0, best]),
                     "favorite", "sent")
        g.motor.start(Intent("ARMY_ATTACK" if attack else "ARMY_MOVE", [u["tag"] for u in group], (x, y),
                             {"tags": [u["tag"] for u in group], "x": x, "y": y, "attack": attack, "p": float(p[best]),
                              "fight": 0.0, "done": [], "boxes": 0}, "army"))
        return "army"

    def _cell_name(self, cell: int) -> str:
        x, y = self.tracker.center(cell)
        return f"({x // 32},{y // 32})"

    def _record(self, where, choice: str, p: float | None, by: str, then: str | None = None) -> None:
        """An army decision in the game's records (gary/decision_log.py): the likeliest cells, what was
        chosen and by whom (the draw, a rule), what came of it."""
        from gary.decision_log import LOG, top
        LOG.decision("T", "army", self.g.hi.frame, choice, p, top(self._cell_name, where[0], 3), self._seen, by, then=then)

    def _hidden_army(self, obs: dict, mine: list[dict], dest: int) -> float | None:
        """With the belief model, for a destination away from Gary's bases: the enemy army supply x2
        Gary likely hasn't seen (the belief's 80th percentile less the most seen at once in the
        last 3 minutes); None otherwise."""
        if self.belief is None or not self.belief_hold:
            return None
        x, y = self.tracker.center(dest)
        if any(u["type"] == T.CC and math.dist((u["x"], u["y"]), (x, y)) < HOME_PX for u in mine):
            return None
        workers = sum(1 for u in mine if u["type"] == T.SCV)
        b = self.belief.predict(self.belief_mem.vector(obs["frame"], obs["me"]["supply_used"] * 2, workers))
        known = 0.0
        if self.tracker.seen_hist and not self.belief_total:
            most = np.max([a for _, a in self.tracker.seen_hist], axis=0)
            known = sum(most[t] * s for t, s in ARMY_SUPPLY[self.belief.enemy].items())
        return max(0.0, b["q80"] - known)

    def _hunt(self, obs: dict, mine: list[dict], done: list[dict]) -> bool | None:
        """The army model can sit far ahead with nothing left to fight (a hatchery at a base Gary
        never looked at outlives the game). Once no enemy army has been seen for a while, the army
        goes where enemy buildings were seen, nearest first, then to the bases its units haven't
        passed lately. True if Gary acted, False if the army is hunting but not ordered now, None if
        it isn't hunting."""
        g, f = self.g, self.g.hi.frame
        here = {u["tag"] for u in obs["units"]}
        for u in obs["units"]:
            if u["owner"] not in (g.slot, 11) and is_building(u["type"]):
                self.enemy_buildings[u["tag"]] = (u["x"], u["y"])
            elif u["tag"] in self.enemy_buildings:
                # seen, no longer theirs: an Extractor destroyed is a neutral geyser again, with its tag
                # (remembered as a building, the hunt went on to it, by Gary's own bases, for minutes)
                del self.enemy_buildings[u["tag"]]
            elif u["owner"] not in (g.slot, 11) and u["type"] not in NOT_ARMY:
                self.enemy_army_seen = f
        at = [(u["x"], u["y"]) for u in mine]
        for t, p in list(self.enemy_buildings.items()):
            if t not in here and any(math.dist(p, q) < GONE_PX for q in at):
                del self.enemy_buildings[t]
        for i, b in enumerate(g.map.bases):
            if any(math.dist(b.center, q) < GONE_PX for q in at):
                self.visited[i] = f
                self.hunted.pop(b.center, None)
        army = [u for u in done if u["type"] in g.army_types]
        if f < HUNT_AFTER or f - self.enemy_army_seen < QUIET or len(army) < HUNT_ARMY:
            return None
        if self.hunt_belief and self.belief is not None:
            # not seeing their army isn't their having none: the hunt only with an army at least what the
            # Zerg likely has (the belief's 80th percentile). At 8:30 against Zerg v3's zergling pressure,
            # 19 supply hunted into 16 (belief 17, q80 28) and was lost, then the game to the 11:00 all-in
            workers = sum(1 for u in mine if u["type"] == T.SCV)
            b = self.belief.predict(self.belief_mem.vector(obs["frame"], obs["me"]["supply_used"] * 2, workers))
            if sum(supply_x2(u["type"]) for u in army) < b["q80"]:
                return None
        cx, cy = sum(u["x"] for u in army) / len(army), sum(u["y"] for u in army) / len(army)
        # (a target the army hasn't reached in HUNT_FOR, out of its way or off its ground, is dropped)
        given_up = {p for p, since in self.hunted.items() if f - since > HUNT_FOR}
        self.enemy_buildings = {t: p for t, p in self.enemy_buildings.items() if p not in given_up}
        todo = list(self.enemy_buildings.values()) or \
                [b.center for i, b in enumerate(g.map.bases)
                 if f - self.visited.get(i, -VISITED_FOR) >= VISITED_FOR and b.center not in given_up]
        if not todo:
            return None
        if f < self.next_hunt:
            return False
        x, y = min(todo, key=lambda p: math.dist(p, (cx, cy)))
        self.next_hunt = f + REORDER_AFTER
        comsat = next((u for u in mine if u["type"] == T.COMSAT and u["completed"] and u.get("energy", 0) >= SCAN_ENERGY), None)
        if self.scan and comsat is not None and not self.enemy_buildings:
            # a base not seen lately: a scan first (as pros look for the last buildings), and the army
            # goes only where it finds something
            for i, b in enumerate(g.map.bases):
                if b.center == (x, y):
                    self.visited[i] = f
            g.motor.start(Intent("SCAN", [], (x, y), {"comsat": comsat["tag"], "x": x, "y": y}, "army"))
            return True
        if (self.scan and comsat is not None and (x, y) in self.enemy_buildings.values()
                and f - self.hunted.get((x, y), f) >= SCAN_STUCK
                and f - self.scanned.get((x, y), -RESCAN_SPOT) >= RESCAN_SPOT):
            # a building remembered that the army hasn't settled in SCAN_STUCK: a scan (one that died out
            # of sight is forgotten at once: the scan is Gary's unit by the spot. The army spent a minute
            # of the last two going to one by the map's edge its units couldn't come near; a scan before
            # every walk delayed the hunts that end games: 11 of 38 wins against the 9-pool undecided)
            self.scanned[(x, y)] = f
            g.motor.start(Intent("SCAN", [], (x, y), {"comsat": comsat["tag"], "x": x, "y": y}, "army"))
            return True
        self.hunted.setdefault((x, y), f)
        what = "where an enemy building was" if self.enemy_buildings else "to a base not seen lately"
        g.say(f"army ({len(army)} units) hunts, no enemy army seen for {(f - self.enemy_army_seen) // 24} s: "
              f"{what}, ({x}, {y})")
        tags = [u["tag"] for u in army]
        g.motor.start(Intent("ARMY_ATTACK", tags, (x, y), {"tags": tags, "x": x, "y": y, "attack": True, "p": 1.0,
                                                          "fight": 0.0, "done": [], "boxes": 0,
                                                          "max_boxes": HUNT_BOXES, "hunt": True}, "army"))
        return True

    # --- scouting (scripted for now) -----------------------------------------------------------

    def _scouting(self, obs: dict, mine: list[dict]) -> bool:
        """Like a pro, send a worker to find the enemy base early (the army model needs to know
        where the enemy is): the enemy start locations in turn, then home once it has seen an
        enemy building. True if Gary acted."""
        g = self.g
        if self.scout_done:
            # home: a worker again, for the mining routine (executor.py skips g.scout: sent home with a
            # move to the minerals, the scout stood idle there for the rest of the game, 5 minutes in
            # human game 2, 7 in a headless one)
            if g.scout is not None and not self.watch:
                s = next((u for u in mine if u["tag"] == g.scout), None)
                if s is None or (s["order"] == 3 and math.dist((s["x"], s["y"]), g.main.center) < SCOUT_HOME):
                    g.scout = None
            return False
        hi = g.hi
        if g.scout is None:
            workers = sum(1 for u in mine if u["type"] == T.SCV)
            if hi.frame >= SCOUT_AT and workers >= SCOUT_WORKERS:
                if not self.scout_targets:                 # (a new scout keeps the starts ruled out)
                    self.scout_targets = sorted(g.enemy_starts, key=lambda p: math.dist(p, g.main.center))
                g.motor.start(Intent("SCOUT", [], self.scout_targets[0], {"to": self.scout_targets[0]}, "scout"))
                return True
            return False
        scout = next((u for u in mine if u["tag"] == g.scout), None)
        if scout is None:                                  # the scout died
            self.scout_done = True
            return False
        if self.circle is None and any(u["owner"] not in (g.slot, 11) and is_building(u["type"]) for u in obs["units"]):
            # found them: around their main once (the pool can stand out of sight of the hatchery),
            # past their natural (a hatchery there by 1:50 is a hatch first, whatever the pool
            # says), then home
            c = self.scout_targets[0] if self.scout_targets else (scout["x"], scout["y"])
            ring = [(c[0] + math.cos(k * math.pi / 3) * CIRCLE, c[1] + math.sin(k * math.pi / 3) * CIRCLE) for k in range(6)]
            k0 = min(range(6), key=lambda k: math.dist(ring[k], (scout["x"], scout["y"])))
            w, h = g.map.size
            self.circle = [(int(min(max(x, 32), w - 32)), int(min(max(y, 32), h - 32))) for x, y in ring[k0:] + ring[:k0]]
            self.circle.append(g.map.natural_of(g.map.base_near(c)).center)
        if self.circle is not None:
            reached = bool(self.circle) and math.dist((scout["x"], scout["y"]), self.circle[0]) < 2 * 32
            if self.circle and self.circle[0] != self.waypoint:
                self.waypoint, self.waypoint_since = self.circle[0], g.hi.frame
            if reached or (self.circle and g.hi.frame - self.waypoint_since > WAYPOINT_FOR):
                self.circle.pop(0)              # (or it can't get there: one of Python's is off the ground)
                reached = True
            # an order per waypoint (each one takes Gary's hands), again only if the scout stopped
            if self.circle and (reached or self.ordered_to != self.circle[0] or scout["order"] in (2, 3)) \
                    and g.hi.frame >= self.next_circle_order:
                self.next_circle_order = g.hi.frame + 2 * 24
                self.ordered_to = self.circle[0]
                g.motor.start(Intent("SCOUT", [g.scout], self.circle[0], {"to": self.circle[0]}, "scout"))
                return True
            if not self.circle:
                self.circle = []
                self.scout_done = True
                if self.watch:                  # (it stays: _watching())
                    self.enemy_main = self.scout_targets[0] if self.scout_targets else (scout["x"], scout["y"])
                    return False
                g.motor.start(Intent("SCOUT", [g.scout], g.main.minerals[0], {"to": g.main.minerals[0], "home": True}, "scout"))
                return True
            return False
        if self.scout_targets and math.dist((scout["x"], scout["y"]), self.scout_targets[0]) < 6 * 32:
            self.scout_targets.pop(0)                      # nobody here: try the next start
            if self.scout_targets:
                g.motor.start(Intent("SCOUT", [g.scout], self.scout_targets[0], {"to": self.scout_targets[0]}, "scout"))
                return True
            self.scout_done = True
        elif self.scout_targets and self._scout_stuck(scout):
            # not moving (its box once took nothing and an SCV still selected became the scout: it
            # went on mining, and nothing seen by 2:20 called a hatch first an early pool)
            if self.scout_resends >= SCOUT_RESENDS:
                g.say("scout isn't going: another SCV")
                g.scout, self.scout_resends = None, 0
                return False
            self.scout_resends += 1
            g.motor.start(Intent("SCOUT", [g.scout], self.scout_targets[0], {"to": self.scout_targets[0]}, "scout"))
            return True
        return False

    def _watching(self, obs: dict, mine: list[dict]) -> bool:
        """watch: the scout in their bases, then on the post outside Gary's natural. True if Gary acted."""
        g = self.g
        f = g.hi.frame
        scout = next((u for u in mine if u["tag"] == g.scout), None) if g.scout is not None else None
        if scout is None:
            if f < RESCOUT_BY and not self.rescouted and self.watch_stage == "stay":
                self.rescouted = True            # lost early: one more, straight to their natural
                g.scout = None
                self.scout_done, self.circle = False, None
                return False
            return False
        enemy = [u for u in obs["units"] if u["owner"] not in (g.slot, 11) and supply_x2(u["type"]) > 0
                 and u["type"] not in NOT_ARMY and not is_building(u["type"])]
        near = [e for e in enemy if math.dist((e["x"], e["y"]), (scout["x"], scout["y"])) < STAY_DANGER]
        hp_max = 60
        if self.watch_stage == "stay":
            if f >= STAY_UNTIL or len(near) >= STAY_LEAVE_LINGS or (scout.get("hp") or hp_max) < STAY_HURT * hp_max:
                self.watch_stage = "post"
                self.watch_next = 0
            elif f >= self.watch_next and (scout["order"] != 6 or (near and not self.dodged)):
                # a tour of their main and natural, shift+clicked once (it walks it alone); again when
                # it ends, or from the point farthest from zerglings that come near (once per approach)
                main = getattr(self, "enemy_main", None) or (scout["x"], scout["y"])
                nat = g.map.natural_of(g.map.base_near(main)).center
                w, h = g.map.size
                clip = lambda x, y: (int(min(max(x, 32), w - 32)), int(min(max(y, 32), h - 32)))
                ring = [clip(c[0] + math.cos(k * math.pi / 3) * STAY_RING, c[1] + math.sin(k * math.pi / 3) * STAY_RING)
                        for c in (main, nat) for k in range(6)]
                start = max(range(len(ring)), key=lambda i: min((math.dist(ring[i], (e["x"], e["y"])) for e in near),
                                                                 default=-math.dist(ring[i], (scout["x"], scout["y"]))))
                tour = ring[start:] + ring[:start]
                self.dodged = bool(near)
                self.watch_next = f + STAY_EVERY
                g.motor.start(Intent("SCOUT", [g.scout], tour[0], {"to": tour[0], "then": tour[1:]}, "scout"))
                return True
            if not near:
                self.dodged = False
            return False
        # the post: outside Gary's natural's choke, toward their natural
        if self.post is None:
            from gary.agent.block import _analysis
            try:
                ch = tuple(_analysis(g)["natural_choke"])
            except Exception:
                ch = g.natural.center
            main = getattr(self, "enemy_main", None) or (g.enemy_starts[0] if g.enemy_starts else ch)
            their = g.map.natural_of(g.map.base_near(main)).center
            d = math.dist(ch, their) or 1
            self.post = (int(ch[0] + (their[0] - ch[0]) / d * POST_OUT), int(ch[1] + (their[1] - ch[1]) / d * POST_OUT))
        threat = [e for e in enemy if math.dist((e["x"], e["y"]), (scout["x"], scout["y"])) < POST_BACK]
        if threat:
            self.watch_back_since = f
            if math.dist((scout["x"], scout["y"]), g.natural.center) > 4 * 32 and f >= self.watch_next:
                self.watch_next = f + STAY_EVERY
                g.motor.start(Intent("SCOUT", [g.scout], g.natural.center, {"to": g.natural.center}, "scout"))
                return True
            return False
        if self.watch_back_since is not None and f - self.watch_back_since < POST_CALM:
            return False
        if math.dist((scout["x"], scout["y"]), self.post) > 2 * 32 and f >= self.watch_next and scout["order"] != 6:
            self.watch_next = f + STAY_EVERY * 2
            g.motor.start(Intent("SCOUT", [g.scout], self.post, {"to": self.post}, "scout"))
            return True
        return False

    def _scout_stuck(self, scout: dict) -> bool:
        """Every SCOUT_CHECK: True if the scout isn't on a move (it mines, builds, stands). (Not by
        distance: on Python its path bends away from the target for seconds at a time.)"""
        f = self.g.hi.frame
        key = (scout["tag"], self.scout_targets[0])
        if self.scout_check is None or self.scout_check[1] != key:
            self.scout_check = (f, key)
            return False
        if f - self.scout_check[0] < SCOUT_CHECK:
            return False
        self.scout_check = (f, key)
        return scout["order"] != MOVE_ORDER
