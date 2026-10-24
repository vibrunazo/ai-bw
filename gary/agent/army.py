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
from gary.policy.attribution import explain_army
from gary.policy.army import GRID, STATIC_SUPPLY, ArmyModel, ArmyTracker, is_building, supply_x2
from gary.policy.belief import ARMY_SUPPLY

WAVE = 16
ASK_EVERY = 24 * 2                  # how often Gary asks the model about its army
# a paired arm's forced commitment, locked: reversed at the pros' chance per 5 s by its age, from the first to past
# HAZARD_AGE (army dataset v3, 1,500 modern TvZ player-games a race: docs/PROGRESS.md, how long commitments stand)
HAZARD = {("T", 1): (0.105, 0.056), ("T", 2): (0.163, 0.049), ("Z", 1): (0.148, 0.048), ("Z", 2): (0.163, 0.038)}
HAZARD_AGE = 50.0
SCOUT_RETRY = 24 * 3               # a scout's worker not picked: asked again this much later
COMMIT_TICK = 24 * 5                # commit: each cluster asked this often (the pros' data: ingest/army_dataset.py TICK)
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
HUNT_SQUAD = 8                      # option hunt_split: bases unseen searched by squads of this many units at least...
HUNT_SQUADS = 4                     # ...at most this many at once
GATHER_ELSEWHERE = 0.25            # option gather: an engage with this share of the army's supply elsewhere gathers first...
GATHER_BACK = 12 * 32               # ...at the army's middle, at least this far from the target
GATHER_NEAR = 6 * 32                # ...the army there once this near it...
GATHER_READY = 0.75                 # ...with this share of its supply; or after GATHER_FOR
GATHER_FOR = 20 * 24
STATIC_NEAR = 8 * 32                # option static_odds: ground static defense (STATIC_SUPPLY) this near an engage's cell...
DEFENDERS_NEAR = 12 * 32            # ...with the enemy fighters seen in the last minute this near
HUNT_AHEAD_MIN = 80                 # option hunt_ahead: at least this much army (supply x2) to hunt while enemy fighters are seen
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
                 draw: bool = False, engage=None, commit: bool = False, commit_act: float | None = None,
                 scout: bool = True, types=None, rules=()):
        self.g = g
        # option rules (gary/agent/army_rules.py, from Gary-Z's army): posts, raid and lords every turn, before the clusters;
        # join, sweep and press at a cluster's turn, before the model
        from gary.agent.army_rules import ClusterRules, EnemyMemory, LordHunt, Posts, Raid
        self.memory = EnemyMemory(g) if rules else None
        turn = {"posts": Posts, "raid": Raid, "lords": LordHunt}
        self.turn_rules = [turn[r](g, self.memory) for r in rules if r in turn]
        self.cluster_rules = ClusterRules(g, self.memory, "join" in rules, "sweep" in rules, "press" in rules,
                                          lambda x, y: self.tracker.cell(x, y), cell_dist) \
            if any(r in rules for r in ("join", "sweep", "press")) else None
        self.hunt_ahead: float | None = None   # option: the hunt also this many times the enemy fighters seen (a manifest's)
        self.static_odds: float | None = None  # option: an engage only with this many times its spot's defense
        # option softguard: instead, a cell short of it keeps its odds times (the group / static_odds times its defense)
        # to this power: a nudge, never a veto (the model's engage always goes somewhere)
        self.static_beta: float | None = None
        self.gather_on = False                 # option: an engage gathers the army first (a manifest's)
        self.gathering: dict | None = None     # gather: the target, the staging point, the army's tags, since
        self.scout = scout          # option: a worker scouts the enemy's start (a Zerg's overlords do: its posture)
        # option: the unit types its clusters take (default the race's army; Zerg: its ground army, the flyers
        # and the lurkers having their own rules)
        self.types = set(types) if types is not None else None
        # option: draw the destination and move/attack from the model's odds (the league's learner: RL
        # learns from drawn decisions and their probabilities, logged), instead of the likeliest cell
        # away when most of the odds lie away, said twice (Gary-Z's likeliest-cell army stood at home)
        self.draw = draw
        self.rng = np.random.default_rng(g.hi.slot * 7919 + int(getattr(g, "seed", 0) or 0))
        self.record: list | None = None        # a paired episode's arm: its drawn decisions, kept here
        self._seen: dict = {}                  # what the last question saw, in brief (the game's records)
        self.shadow = None                     # an A/B pair's other model: its odds of each choice, recorded
        self.then_rng = None                   # a paired episode's arm: the shared dice after its first decision
        # option (with draw): the attack-or-hold model (train/engage.py: an army model whose kind head is
        # engage/hold, learned from the pros' orders, then RL on paired branches): it draws whether the
        # group goes into the enemy, and the destination is drawn from the army model's odds among the
        # cells next to known enemy presence, or among the others (the destinations stay imitation)
        self.engage = engage
        self.force_engage: bool | None = None  # a paired episode's arm: its first decision's engage forced
        # option: commitments (train/commit.py: the model's kind head is keep/engage/move, its numbers carry
        # the cluster's commitment). Each cluster asked every COMMIT_TICK: keep (its commitment stands),
        # engage or move (a new one, the destination drawn on that side). Replaces the pacing and the stay
        # rule: when to re-decide is the model's, learned from the pros, as are the triggers
        self.commit = commit
        self.commit_floor: float | None = None   # option floor (with draw): no choice drawn under these odds, and the
                                        # destination drawn among the 5 likeliest cells: the drawn rate and variety
                                        # without the 5% disasters (human game 3's two retreats)
        self.commit_act = commit_act    # option act: in play, a new commitment when the model's odds of one (engage
                                        # + move) reach this, the likelier of the two; None: the likeliest of the
                                        # three (which is keep most ticks: Gary-T won 7 of 24 against Gary-Z with 17
                                        # at time, drawn 13 and 11). 0.41 is where the deterministic rate matches
                                        # the drawn one (the pros') over human game 3 and round c10
        self.commitments: list[tuple[int, int, int, int]] = []   # (frame, from cell, to cell, 1 engage / 2 move)
        self.force_commit: int | None = None   # a paired episode's arm: its first decision forced
        self.lock_forced = False
        # option engagebias: the commit model's engage odds times exp(this) against move's (learned from the locked
        # commit pairs: engaging beat moving by +0.81 a pair at the moments the deployed model engaged 55% of, and
        # nothing in the state predicted where it didn't; issue #28)
        self.engage_bias = 0.0                # ...and kept (a training arm): reversed at the pros' hazard, not the model's
        self.locked: tuple | None = None        # (the forced commitment, the frame its lock was last rolled)
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
        self.hunt_advice = False    # option huntadvice (with the hunt off): what it would do, in the records only (labels)
        self.enemy_buildings: dict[int, tuple[int, int]] = {}   # tag -> where it was seen
        self.enemy_army_seen = 0                                 # frame an enemy fighter was last seen
        self.hunted: dict[tuple[int, int], int] = {}             # hunt target -> frame first sent there
        self.visited: dict[int, int] = {}                       # base index -> frame Gary's units were there
        self.next_hunt = 0
        self.hunt_split = False                 # option: unseen bases searched by squads at once (a manifest's)
        self.search: dict[tuple, list[int]] = {}   # hunt_split: base center -> its squad's tags
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
        self.next_scout_try = 0

    def remember(self, obs: dict) -> None:
        self.tracker.hidden = set().union(set(), *self.g.loaded.values())   # (marines in a bunker aren't lost)
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
        if self.memory is not None:
            self.memory.look(obs)
        for rule in self.turn_rules:
            if (who := rule.step(obs, mine, done)):
                return who
        if self.scout and self._scouting(obs, mine):
            return "scout"
        if self.watch and self.scout_done and self._watching(obs, mine):
            return "scout"
        if self.gathering is not None:          # (the model isn't asked while the army gathers for its engage)
            who = self._gather_step(obs, mine, done)
            if self.gathering is not None or who:
                return who
        hunt = self._hunt(obs, mine, done) if self.hunt else None
        if not self.hunt and self.hunt_advice:
            self._advise(obs, mine, done)
        if hunt is not None:                    # (the model isn't asked while the army hunts)
            return "army" if hunt else None
        if hi.frame < self.next_ask:
            return None
        if self.home_guard:
            from gary.agent.fight import enemy_fighters, near_home
            if near_home(mine, enemy_fighters(obs, g.slot)):
                self.next_ask = hi.frame + ASK_EVERY
                return None
        army = [u for u in done if u["type"] in (self.types if self.types is not None else g.army_types)]
        clusters = self._clusters(army)
        if not clusters:
            return None
        self.next_ask = hi.frame + (COMMIT_TICK if self.commit else ASK_EVERY) // len(clusters)   # one cluster a question
        group = clusters[self.ask_turn % len(clusters)]
        self.ask_turn += 1
        if self.cluster_rules is not None and (who := self.cluster_rules.step(obs, mine, clusters, group)):
            return who
        if self.commit:
            return self._committed(obs, mine, group)
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
        choice = f"{'attack' if attack else 'move'} to {self._cell_name(best)}"
        proposed, self.proposed[here] = self.proposed.get(here), best
        if proposed is None or cell_dist(proposed, best) > 1:
            self._record(where, choice, float(where[0, best]), "favorite", "waits: until the model says it twice")
            return None                                           # wait until it says so twice
        last = self.ordered.get(here)
        if last and cell_dist(last[0], best) <= 1 and last[1] == attack and hi.frame - last[2] < REORDER_AFTER:
            self._record(where, choice, float(where[0, best]), "favorite", "not again: just ordered there")
            return None
        # will it go well? the model's fight estimate for going there vs staying put
        go = self.model.fight(grid, glob, best, attack)
        stay = self.model.fight(grid, glob, here, False)
        if go < max(self.fight_min, stay - self.fight_margin) and (attack or self.hold_moves):
            if self.held.get(here, -1) != best:
                g.say(f"army ({len(group)} units) holds: going to {self.tracker.center(best)} "
                      f"looks like {go:+.0%} army share, staying {stay:+.0%}")
            self.held[here] = best
            self._record(where, choice, float(where[0, best]), "favorite",
                         f"held: the fight estimate ({go:+.0%} going, {stay:+.0%} staying)")
            return None
        hidden = self._hidden_army(obs, mine, best)
        if hidden is not None and hidden > sum(supply_x2(u["type"]) for u in group):
            # the fight estimate knows only the enemy army seen lately; the belief model, what the
            # enemy likely has (its 80th percentile): out there unseen, more than this group
            if self.held.get(here, -1) != best:
                g.say(f"army ({len(group)} units) holds: going to {self.tracker.center(best)}, the zerg likely "
                      f"has {hidden / 2:.0f} supply more army than seen")
            self.held[here] = best
            self._record(where, choice, float(where[0, best]), "favorite",
                         f"held: the belief model, {hidden / 2:.0f} supply more zerg army likely than seen")
            return None
        self.held.pop(here, None)
        self._record(where, choice, float(where[0, best]), "favorite", "sent",
                     why=explain_army(self.model, grid, glob, lambda w, k: w[:, best] * k[:, int(attack)]))
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
            other = None
            if self.shadow is not None:
                sw, sk = self.shadow.predict(grid, glob)
                other = float(sw[0, best]) * float(sk[0, int(drawn_attack)])
            self._record(where, f"{'attack' if drawn_attack else 'move'} to {self._cell_name(best)}", float(np.exp(logp)),
                         "draw", {"stay": "stays: the cell drawn is near", "repeat": "not again: just ordered there",
                                  "held": "held: the fight estimate", "go": "sent"}[outcome],
                         rl=self._rl_row("army_t"), p_other=other,
                         why=explain_army(self.model, grid, glob, lambda w, k: w[:, best] * k[:, int(drawn_attack)]))
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
        other = None
        if self.shadow is not None:
            other = float(self.shadow.predict(grid, glob)[1][0, int(eng)])
        LOG.decision(g.hi.side, "engage", hi.frame, f"{'engage' if eng else 'hold'}: to {self._cell_name(best)}",
                     pe if eng else 1 - pe, [["engage", round(pe, 3)], ["hold", round(1 - pe, 3)]], self._seen,
                     "forced (a paired arm)" if forced else "draw",
                     then="stays: the cell drawn is near" if cell_dist(best, here) < MOVE_CELLS else "sent",
                     rl=self._rl_row("engage_t"), p_other=other,
                     why=explain_army(self.engage, grid, glob, lambda w, k: k[:, int(eng)]))
        if cell_dist(best, here) < MOVE_CELLS:
            return None
        self.move_tokens -= 1.0
        attack = eng or bool(self.rng.random() < float(kind[0, 1]))
        last = self.ordered.get(here)
        if last and cell_dist(last[0], best) <= 1 and last[1] == attack and hi.frame - last[2] < REORDER_AFTER:
            return None
        x, y = self.tracker.center(best)
        self.ordered[here] = (best, attack, hi.frame)
        g.motor.start(Intent("ARMY_ATTACK" if attack else "ARMY_MOVE", [u["tag"] for u in group], (x, y),
                             {"tags": [u["tag"] for u in group], "x": x, "y": y, "attack": attack, "p": float(p[best]),
                              "fight": 0.0, "done": [], "boxes": 0}, "army"))
        return "army"

    def _commitment(self, here: int) -> tuple[int, float]:
        """The cluster's commitment: its last order whose path passes within 2 cells (1 engage, 2 move; 0
        none in COMMIT_AGE_S) and its age in seconds."""
        from gary.policy.army import COMMIT_AGE_S, path_dist
        f = self.g.hi.frame
        self.commitments = [c for c in self.commitments if f - c[0] <= COMMIT_AGE_S * 24]
        for at, a, b, k in reversed(self.commitments):
            if path_dist(here, a, b) <= 2:
                return k, (f - at) / 24
        return 0, float(COMMIT_AGE_S)

    def _committed(self, obs, mine, group) -> str | None:
        """The commit option: the cluster's next step drawn from the model's keep / engage / move, given its
        commitment; a new commitment's destination drawn from the model's odds among the cells 2+ away next
        to known enemy presence (engage) or not (move). Recorded, and logged for RL ("commit_t")."""
        from gary.decision_log import LOG
        g, hi = self.g, self.g.hi
        cx = sum(u["x"] for u in group) / len(group)
        cy = sum(u["y"] for u in group) / len(group)
        here = self.tracker.cell(cx, cy)
        held, age = self._commitment(here)
        if self.locked is not None and self.force_commit is None:
            c, rolled = self.locked
            from gary.policy.army import path_dist
            if c in self.commitments and path_dist(here, c[1], c[2]) <= 2:
                start, end = HAZARD.get((g.race.letter, c[3]), (0.12, 0.05))
                a = (hi.frame - c[0]) / 24
                per5 = start - (start - end) * min(1.0, a / HAZARD_AGE)
                p = 1.0 - (1.0 - per5) ** ((hi.frame - rolled) / (5 * 24))
                if self.rng.random() >= p:
                    self.locked = (c, hi.frame)
                    LOG.decision(g.hi.side, "commit", hi.frame, "keep", None, None, None, "locked (a paired arm)",
                                 then=f"its forced commitment stands ({a:.0f} s, the pros' reversal {per5:.0%} per 5 s)")
                    return None
                self.locked = None             # (reversed at the pros' chance: the model decides from now on)
            elif c not in self.commitments:
                self.locked = None
        grid, glob = self.tracker.inputs(obs, group, (held, age))
        where, kind = self.model.predict(grid, glob)
        q = kind[0].astype(np.float64)
        if self.engage_bias:                   # (between engage and move only: keep's odds, so the order rate, as they were)
            q = q.copy()
            act, e = q[1] + q[2], q[1] * np.exp(self.engage_bias)
            q[1], q[2] = act * e / (e + q[2]), act * q[2] / (e + q[2])
        if self.force_commit is not None:
            k, self.force_commit = int(self.force_commit), None
            by = "forced (a paired arm)"
        elif self.draw:                        # training (the league): sampled, logged with its odds
            qd = q.copy()
            if self.commit_floor is not None:
                qd[qd < self.commit_floor] = 0.0
                if qd.sum() <= 0:
                    qd[0] = 1.0
            k = int(self.rng.choice(3, p=qd / qd.sum()))
            by = "draw" if self.commit_floor is None else f"draw above {self.commit_floor:.0%}"
        elif self.commit_act is not None:     # play: acts once the odds of acting reach the threshold
            k = int(1 + q[1:].argmax()) if 1.0 - q[0] >= self.commit_act else 0
            by = f"favorite above {self.commit_act:.0%} odds of acting"
        else:                                  # play: the likeliest (human game 3 was played drawn: the two
            k = int(q.argmax())                # retreats that threw it were 4.8% and 5.1% draws)
            by = "favorite"
        if self.then_rng is not None:          # (an arm's first decision drawn its own way, the rest shared)
            self.rng, self.then_rng = self.then_rng, None
        names = ["keep", "engage", "move"]
        self._seen = {"units": len(group), "at": self._cell_name(here),
                      "commitment": f"{names[held] if held else 'none'}, {age:.0f} s"}
        options = [[n, round(float(q[i]), 3)] for i, n in enumerate(names)]
        why = explain_army(self.model, grid, glob, lambda w, kd: kd[:, k])
        item = dict(frame=hi.frame, grid=grid[0].astype(np.float16), glob=glob[0].astype(np.float16), choice=k,
                    logp=float(np.log(q[k] + 1e-9)))
        layer = f"commit_{g.race.letter.lower()}"     # (commit_t, commit_z: the league's RL reads them by race)
        (self.record.append(item) if self.record is not None else LOG.add(layer, **item))
        rl = self._rl_row(layer)
        if k == 0:
            LOG.decision(g.hi.side, "commit", hi.frame, "keep", q[0], options, self._seen, by, then="its commitment stands",
                         rl=rl, why=why)
            return None
        m = np.asarray(grid[0]).reshape(-1, GRID, GRID)
        enemy = (m[3] > 0) | (m[6] > 0) | (m[7] > 0)
        near = enemy.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                near |= np.roll(np.roll(enemy, dy, 0), dx, 1)
        far = np.array([cell_dist(c, here) >= MOVE_CELLS for c in range(GRID * GRID)])
        mask = far & (near.reshape(-1) if k == 1 else ~near.reshape(-1))
        weight = 1.0
        if k == 1 and self.static_odds is not None and self.memory is not None:
            ratio = self._outweighs(obs, mine, group)
            if self.static_beta is None:
                mask &= ratio >= 1.0
            else:
                weight = np.minimum(1.0, ratio) ** self.static_beta
        p = where[0] * mask * weight
        if p.sum() <= 0:
            LOG.decision(g.hi.side, "commit", hi.frame, names[k], q[k], options, self._seen, by,
                         then="no cell on that side: its commitment stands", rl=rl, why=why)
            return None
        if self.draw and self.commit_floor is not None:          # (among the 5 likeliest cells on that side)
            keep_cells = np.argsort(-p)[:5]
            pf = np.zeros_like(p)
            pf[keep_cells] = p[keep_cells]
            p = pf
        best = int(self.rng.choice(len(p), p=p / p.sum())) if self.draw else int(p.argmax())
        self.commitments.append((hi.frame, here, best, k))
        if by == "forced (a paired arm)" and self.lock_forced:
            self.locked = (self.commitments[-1], hi.frame)
        LOG.decision(g.hi.side, "commit", hi.frame, f"{names[k]} to {self._cell_name(best)}", q[k], options, self._seen, by,
                     then="sent", rl=rl, why=why)
        x, y = self.tracker.center(best)
        attack = k == 1
        self.ordered[here] = (best, attack, hi.frame)
        if attack and self.gather_on and self._gather_start(mine, group, (x, y)):
            return "army"
        g.motor.start(Intent("ARMY_ATTACK" if attack else "ARMY_MOVE", [u["tag"] for u in group], (x, y),
                             {"tags": [u["tag"] for u in group], "x": x, "y": y, "attack": attack, "p": float(q[k]),
                              "fight": 0.0, "done": [], "boxes": 0, "origin": f"commit {names[k]}"}, "army"))
        return "army"

    def _army_units(self, mine: list[dict]) -> list[dict]:
        types = self.types if self.types is not None else self.g.army_types
        return [u for u in mine if u["type"] in types and u["completed"]]

    def _gather_start(self, mine: list[dict], group: list[dict], target: tuple) -> bool:
        """gather, STOP-GAP (AGENTS.md): an engage while GATHER_ELSEWHERE of the army is away from the engaging group
        first brings the whole army to its own middle (at least GATHER_BACK from the target: a staging point toward
        the target drew the units into fights one by one on their way, 23 of 29 lost), then attacks with it
        (Gary-T's attacks had 4x the pros' late arrivals, two thirds of them units that already existed when the
        attack began: the other clusters, each deciding alone; eval/attacks.py). Until the commit model learns to
        gather before it engages. True if the army now gathers."""
        from gary.decision_log import LOG
        g, f = self.g, self.g.hi.frame
        army = self._army_units(mine)
        ours = {u["tag"] for u in group}
        cx = sum(u["x"] for u in group) / len(group)
        cy = sum(u["y"] for u in group) / len(group)
        total = sum(supply_x2(u["type"]) for u in army) or 1
        away = sum(supply_x2(u["type"]) for u in army
                   if u["tag"] not in ours and math.dist((u["x"], u["y"]), (cx, cy)) > GATHER_NEAR)
        if away < GATHER_ELSEWHERE * total:
            return False
        mx = sum(supply_x2(u["type"]) * u["x"] for u in army) / total
        my = sum(supply_x2(u["type"]) * u["y"] for u in army) / total
        d = math.dist((mx, my), target) or 1.0
        if d >= GATHER_BACK:
            stage = (int(mx), int(my))
        else:
            stage = (int(target[0] + (mx - target[0]) / d * GATHER_BACK), int(target[1] + (my - target[1]) / d * GATHER_BACK))
        tags = [u["tag"] for u in army]
        aim = (int(target[0]), int(target[1]))
        self.gathering = {"target": aim, "stage": stage, "tags": tags, "since": f, "again": f}
        g.say(f"army: gathers {len(army)} units at {stage} before engaging {aim} ({away / total:.0%} of it elsewhere)")
        LOG.decision(g.hi.side, "army", f, f"gather at ({stage[0] // 32},{stage[1] // 32})",
                     inputs={"units": len(army), "elsewhere": round(away / total, 2)},
                     by="rule: an engage with a quarter of the army elsewhere gathers it first", then="sent")
        self._send_all(tags, stage, True)
        return True

    def _gather_step(self, obs: dict, mine: list[dict], done: list[dict]) -> str | None:
        """gather: the attack once GATHER_READY of the army is at the staging point, or GATHER_FOR is up; late ones
        sent again. Enemy fighters at home don't call it off: whether the attack is still worth it is the commit
        model's call, already made with them in sight (moving out is often what ends a mutalisk harass: the flock
        goes home to defend; the user), and the units at home are the fight layer's."""
        from gary.decision_log import LOG
        g, f, gth = self.g, self.g.hi.frame, self.gathering
        alive = {u["tag"]: u for u in self._army_units(mine)}
        tags = [t for t in gth["tags"] if t in alive]
        why = "the army gone" if not tags else "out of time" if f - gth["since"] > 2 * GATHER_FOR else None
        if why:
            LOG.decision(g.hi.side, "army", f, "gather called off", inputs={"units": len(tags)}, by=f"rule: {why}")
            self.gathering = None
            return None
        total = sum(supply_x2(alive[t]["type"]) for t in tags) or 1
        there = {t for t in tags if math.dist((alive[t]["x"], alive[t]["y"]), gth["stage"]) <= GATHER_NEAR}
        share = sum(supply_x2(alive[t]["type"]) for t in there) / total
        if share >= GATHER_READY or f - gth["since"] >= GATHER_FOR:
            army = list(alive)                     # (with what came out meanwhile)
            aim = gth["target"]
            g.say(f"army: {share:.0%} gathered, {len(army)} units engage {aim}")
            LOG.decision(g.hi.side, "army", f, f"engage ({aim[0] // 32},{aim[1] // 32})",
                         inputs={"units": len(army), "gathered": round(share, 2), "waited_s": (f - gth["since"]) // 24},
                         by="rule: the army gathered (or the wait is up): the engage goes", then="sent")
            self.gathering = None
            self._send_all(army, aim, True)
            return "army"
        if f - gth["again"] >= REORDER_AFTER:
            late = [t for t in tags if t not in there]
            if late:
                gth["again"] = f
                self._send_all(late, gth["stage"], True)
                return "army"
        return None

    def _send_all(self, tags: list[int], point: tuple, attack: bool) -> None:
        x, y = int(point[0]), int(point[1])
        self.g.motor.start(Intent("ARMY_ATTACK" if attack else "ARMY_MOVE", list(tags), (x, y),
                                  {"tags": list(tags), "x": x, "y": y, "attack": attack, "p": 1.0, "fight": 0.0,
                                   "done": [], "boxes": 0, "max_boxes": HUNT_BOXES}, "army"))

    def _outweighs(self, obs: dict, mine: list[dict], group: list[dict]) -> np.ndarray:
        """static_odds: per cell, the group against static_odds times the ground static defense there (each
        STATIC_SUPPLY) and the enemy fighters seen near it in the last minute (inf where there is none): an engage may
        go where it is 1 or more, or (static_beta) is nudged away from the others (human game 4's three
        attacks all went into 2-4 sunkens and lost 24 supply for 1.5; against gz1, 38% of its attacks went into
        sunkens to the pros' 28%: eval/attacks.py). STOP-GAP (AGENTS.md): until the commit model weighs static
        defense itself."""
        ours = sum(supply_x2(u["type"]) for u in group)
        seen = {u["tag"] for u in obs["units"]}
        at = [(u["x"], u["y"]) for u in mine]
        # (one remembered where Gary's units now stand and don't see it is gone: the memory keeps every building seen)
        static = [(x, y, STATIC_SUPPLY[t]) for tag, (x, y, t) in self.memory.buildings.items() if t in STATIC_SUPPLY
                  and (tag in seen or not any(math.dist((x, y), q) < GONE_PX for q in at))]
        ratio = np.full(GRID * GRID, np.inf)
        if not static:
            return ratio
        fighters = list(self.memory.fighters.values())
        for c in range(GRID * GRID):
            cx, cy = self.tracker.center(c)
            walls = sum(w for x, y, w in static if math.dist((x, y), (cx, cy)) < STATIC_NEAR)
            if walls:
                guard = walls + sum(v[1] for v in fighters if math.dist((v[2], v[3]), (cx, cy)) < DEFENDERS_NEAR)
                ratio[c] = ours / (self.static_odds * guard)
        return ratio

    def _cell_name(self, cell: int) -> str:
        x, y = self.tracker.center(cell)
        return f"({x // 32},{y // 32})"

    def _record(self, where, choice: str, p: float | None, by: str, then: str | None = None, rl: int | None = None,
                p_other: float | None = None, why: list | None = None) -> None:
        """An army decision in the game's records (gary/decision_log.py): the likeliest cells, what was
        chosen and by whom (the draw, a rule), what came of it."""
        from gary.decision_log import LOG, top
        LOG.decision(self.g.hi.side, "army", self.g.hi.frame, choice, p, top(self._cell_name, where[0], 3), self._seen, by, then=then,
                     rl=rl, p_other=p_other, why=why)

    def _rl_row(self, layer: str) -> int | None:
        """The row of the decision just logged among the layer's RL decisions (a paired arm's, or the game's)."""
        from gary.decision_log import LOG
        if self.record is not None:
            return len(self.record) - 1
        return len(LOG.items[layer]) - 1 if LOG.on else None

    def _hidden_army(self, obs: dict, mine: list[dict], dest: int) -> float | None:
        """With the belief model, for a destination away from Gary's bases: the enemy army supply x2
        Gary likely hasn't seen (the belief's 80th percentile less the most seen at once in the
        last 3 minutes); None otherwise."""
        if self.belief is None or not self.belief_hold:
            return None
        x, y = self.tracker.center(dest)
        if any(u["type"] in self.g.race.town_halls and math.dist((u["x"], u["y"]), (x, y)) < HOME_PX for u in mine):
            return None
        workers = sum(1 for u in mine if u["type"] == self.g.race.worker)
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
        # (option hunt_ahead, STOP-GAP (AGENTS.md): also far ahead of the enemy fighters seen in the last minute, as a few new zerglings
        # kept "no enemy army seen" from ever holding: Gary-T led 18 of 48 games at 20:00 with 60-67 army to the
        # Zerg's 0-2 and its last 4 buildings standing, its clusters' engages drawn all over the map; round G13)
        ahead = self.hunt_ahead is not None and self.memory is not None and             sum(supply_x2(u["type"]) for u in army) >= max(HUNT_AHEAD_MIN, self.hunt_ahead * max(8, sum(v[1] for v in self.memory.fighters.values())))
        if f < HUNT_AFTER or (f - self.enemy_army_seen < QUIET and not ahead) or len(army) < HUNT_ARMY:
            return None
        if self.hunt_belief and self.belief is not None:
            # not seeing their army isn't their having none: the hunt only with an army at least what the
            # Zerg likely has (the belief's 80th percentile). At 8:30 against Zerg v3's zergling pressure,
            # 19 supply hunted into 16 (belief 17, q80 28) and was lost, then the game to the 11:00 all-in
            workers = sum(1 for u in mine if u["type"] == g.race.worker)
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
        if self.hunt_split and not self.enemy_buildings and len(army) >= 2 * HUNT_SQUAD:
            return self._search(army, todo, (cx, cy))
        self.search.clear()
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
        why = f"no enemy army seen for {(f - self.enemy_army_seen) // 24} s" if f - self.enemy_army_seen >= QUIET else             f"{self.hunt_ahead:g}x the enemy fighters seen"
        g.say(f"army ({len(army)} units) hunts, {why}: {what}, ({x}, {y})")
        tags = [u["tag"] for u in army]
        g.motor.start(Intent("ARMY_ATTACK", tags, (x, y), {"tags": tags, "x": x, "y": y, "attack": True, "p": 1.0,
                                                          "fight": 0.0, "done": [], "boxes": 0,
                                                          "max_boxes": HUNT_BOXES, "hunt": True}, "army"))
        return True

    def _advise(self, obs: dict, mine: list[dict], done: list[dict]) -> None:
        """hunt_advice: the hunt run as if on, its orders caught rather than given, and each written to the records as a
        "hunt advice" decision (its units and target): the rule's labels on the model's own states, for a commit model
        to imitate the hunt where the pros' replays end at GG before any game is finished (AGENTS.md, the hunt)."""
        from gary.decision_log import LOG
        g = self.g
        caught: list[Intent] = []
        start, say = g.motor.start, g.say
        g.motor.start, g.say = caught.append, (lambda *a, **k: None)
        try:
            self._hunt(obs, mine, done)
        finally:
            g.motor.start, g.say = start, say
        where = {u["tag"]: (u["x"], u["y"]) for u in mine}
        for it in caught:
            x, y = it.target
            at = [where[t] for t in it.units if t in where]
            cx, cy = (sum(p[0] for p in at) / len(at), sum(p[1] for p in at) / len(at)) if at else (-32, -32)
            LOG.decision(g.hi.side, "hunt advice", g.hi.frame, f"{'scan' if it.kind == 'SCAN' else 'engage'} to "
                         f"({x // 32},{y // 32})", by="rule: the hunt (advice)",
                         inputs={"tags": list(it.units), "to": [x // 32, y // 32], "at": [int(cx) // 32, int(cy) // 32],
                                 "kind": it.kind})

    def _search(self, army: list[dict], todo: list[tuple], at: tuple) -> bool:
        """hunt_split: no enemy building known, the bases not seen lately searched at once by squads (pros split up
        to find the last buildings; the whole army walking to one base at a time saw 6 in 4 minutes, and the Zerg's
        last hatchery outlived 4 minutes of hunting at 66-0 army: round Jhunt4). Each squad keeps its base until it
        has been seen; one squad ordered a turn. STOP-GAP (AGENTS.md): until searching is a learned skill."""
        from gary.decision_log import LOG
        g, f = self.g, self.g.hi.frame
        alive = {u["tag"] for u in army}
        todo_set = set(map(tuple, todo))
        self.search = {b: [t for t in tags if t in alive] for b, tags in self.search.items() if b in todo_set}
        self.search = {b: tags for b, tags in self.search.items() if tags}
        n = min(HUNT_SQUADS, len(army) // HUNT_SQUAD, len(todo))
        free = [u for u in army if not any(u["tag"] in tags for tags in self.search.values())]
        if len(self.search) < n and len(free) >= HUNT_SQUAD:
            base = min((b for b in map(tuple, todo) if b not in self.search), key=lambda b: math.dist(b, at))
            size = len(free) if len(self.search) == n - 1 else max(HUNT_SQUAD, len(army) // n)
            squad = sorted(free, key=lambda u: math.dist((u["x"], u["y"]), base))[:size]
            self.search[base] = [u["tag"] for u in squad]
            x, y = int(base[0]), int(base[1])
            self.hunted.setdefault((x, y), f)
            g.say(f"army: a squad of {len(squad)} searches the base at ({x}, {y}) ({len(self.search)} of {n} squads)")
            LOG.decision(g.hi.side, "army", f, f"search ({x // 32},{y // 32})", inputs={"squad": len(squad), "squads": n},
                         by="rule: no enemy building known, the bases not seen lately searched by squads", then="sent")
            tags = self.search[base]
            g.motor.start(Intent("ARMY_ATTACK", tags, (x, y), {"tags": tags, "x": x, "y": y, "attack": True, "p": 1.0,
                                                              "fight": 0.0, "done": [], "boxes": 0, "hunt": True}, "army"))
            return True
        return False

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
            workers = sum(1 for u in mine if u["type"] == g.race.worker)
            if hi.frame >= max(SCOUT_AT, self.next_scout_try) and workers >= SCOUT_WORKERS:
                self.next_scout_try = hi.frame + SCOUT_RETRY   # (a failed pick doesn't hold the hands every turn)
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
