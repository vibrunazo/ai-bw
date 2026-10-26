# Gary — Control layers

Who decides what, at which timescale, and how each layer becomes a swappable part. This note
applies [ARCHITECTURE.md](ARCHITECTURE.md) §6.4 (intents), §6.6 (decision slots and providers),
§6.7 (executors and the Attention Arbiter) and §9.1 (run manifests) to the code, written at v0.9, with
the plan from there. Status (2026-10-24): steps 1-5 are done (`gary/agent/`: the layers as providers, every
Gary a manifest; the arbiter with a trace; intents and the "direct" motor layer; leases **enforced** by
priority since v0.10, `intents.priority`: the rules, then the fight's decisions, then the army; roles per
unit, by hand rules and by a model learned from the pros); step 7 is under way (docs/SELF_PLAY.md). Not
yet: `Candidate` (the arbiter still hands the one task slot to the first layer that wants it, by fixed
priority; Gary-Z's turns are its `rotate` option). Today's manifests are `gt1`, `gt1dev`, `gz1` and
`gz1dev` (AGENTS.md); the v0.9 examples below show the contract.

## Why now

Gary v0.1–v0.9 are a chain of subclasses. Each version overrides methods of the one before, and
one `act()` runs a fixed priority list: a macro turn, then worker pull, bunker loading, the fight
model, then chores, gas and the army model. Each rule grabs units for a while by setting the one
`task`, or by hiding units from the next layer (v0.9 hides the army from the army model while it
holds). That worked for adding one behavior at a time. It stops working now, for three reasons:

- **We can't swap one part.** "Is a separate motor skill better than the fight model doing the
  timing itself?" needs two Garys identical except in that layer. With subclasses, a variant is a
  new class tangled with everything above it.
- **Rules pile up without an owner.** Pull, repair, load, reinforce and the fight model all
  command the same marines and SCVs. Who wins is the order of `if`s in `act()`, and nothing
  records why a unit got a command, or which command it didn't get.
- **The data points at a layer we don't have.** Pros' fighting marines get 8.4 move orders per
  minute of fighting and shoot again 0.84 s after a move; v0.9's get 1.5 and shoot again after
  1.6 s (`eval/micro_timing.py`). The fight model gives one command per half second to one
  group; a stutter step is a move and an attack a fraction of a second apart, on several groups.
  Either the model learns that timing, or a skill below it does the timing. We should test both.

## The layers

Slow layers decide what; fast layers decide how. Each layer has a contract (its inputs and
outputs) and any number of implementations ("providers", §6.6). A Gary is a choice of one
provider per layer.

| Layer | Timescale | Decides | Output (contract) | Providers today → planned |
|---|---|---|---|---|
| **L1 Posture** | minutes | The plan's mode and its constraints: normal, early-pool defense, ... | `Posture` | v0.9's rules → `supervisor_branch` slot: rules, stats table (§6.8), learned |
| **L2 Macro decisions** | seconds | What to build, train, research next | macro goal | learned macro model (v0.2+); scripted build per posture (v0.9's rush build) |
| **L2b Macro executor** | seconds | Placement, workers, gas, production clicks | candidate actions | scripted (v0.1+) |
| **L3 Army** | seconds | Where each group goes, attack or move, whether to take a fight | army orders (as intents) | learned army model (v0.3+), posture constraints (v0.9's hold) |
| **L4 Fight** | ~0.5 s | What each engaged group does: focus, kite, hold, load, repair, screen, pull, retreat | `Intent`s, or commands directly (see "two ways to fight") | rules (v0.7: pull, load, repair; v0.9: reinforce), command model (imitation + RL, v0.5+) → intent model |
| **L5 Motor skills** | frames | How an intent turns into clicks over time: stutter step, kite, load, repair, screen | candidate actions | "direct" (the intent is one command, as today) → scripted skills → learned motor model |
| **L6 Attention Arbiter** | per action | Whose action goes next on one pair of hands and one camera | one `HumanAction` | fixed priority (today's `if` order) → §6.7 rules with shares → learned |
| **L7 Human interface** | per action | Mouse travel, scatter, delays, APM | game commands | one implementation (unchanged) |

L1–L4 decide; L2b and L5 execute; L6 shares the hands; L7 is the hands. Nothing in L1–L5 sends
a command to the game: everything that clicks becomes a candidate action for the arbiter.

### Contracts

```python
@dataclass
class Posture:                      # L1 → L2, L3, L4
    name: str                       # "normal", "early_pool_defense", ...
    no_expand: bool = False
    macro: str | None = None        # a scripted build that replaces the learned macro, or None
    army_home: bool = False         # L3 keeps the army at home
    preload_bunkers: bool = False   # L4: marines into bunkers before the enemy is there
    reason: str = ""                # the facts that set it (for the trace)

@dataclass
class Intent:                       # L3/L4 → L5 (extends §6.4 with micro intents)
    kind: str                       # ATTACK_MOVE, MOVE, FOCUS, STUTTER, KITE, HOLD, LOAD, REPAIR,
                                    # SCREEN, PULL, RETREAT, ...
    units: list[int]                # the units it leases (see "ownership")
    target: tuple[int, int] | int | None   # a map point or a unit tag
    params: dict                    # e.g. {"from": tag} for KITE
    source: str                     # provider that issued it, with its version
    p: float | None                 # its probability, and the alternatives, when a model chose it
    alternatives: list[tuple[str, float]]
    ttl: int                        # frames until it lapses unless renewed

class Skill(Protocol):              # L5
    def start(self, intent: Intent, obs: dict) -> "SkillRun": ...

class SkillRun(Protocol):
    def step(self, obs: dict, hands: InterfaceState) -> Candidate | None:
        """The next action this intent needs now (None: nothing yet), until done()."""
    def done(self) -> bool: ...

@dataclass
class Candidate:                    # L2b, L5 → L6
    action: HumanAction             # select, click, key, camera... (§6.3)
    executor: str                   # "macro", "skill:stutter", "rule:pull", ...
    urgency: float                  # how soon it matters (a marine about to die vs a depot)
```

A provider sees the observation Gary sees (fog, reaction delay) and the interface state, never
the game's hidden state, as everywhere else.

### Ownership

Every unit has one owner at a time, by **lease**: an intent leases its units for its `ttl`, and
a lease is renewed by reissuing the intent. Defaults: workers belong to the macro executor, army
units to L3; an L4 intent takes units from L3 while a fight lasts; a skill holds its intent's
units while it runs. A higher layer can take units back (L1's posture can end a fight intent);
equal layers resolve by the intent's priority, and every takeover is a trace event (`arbiter`,
§6.5) with who lost what. This replaces today's single `task` and the trick of hiding units from a
layer.

### Two ways to fight: one model, or intent plus skill

Both are configurations of the same layers:

- **A. One model does it all.** L4 = the command model, emitting commands directly (move, attack,
  stim...; as v0.5–v0.9 do). L5 = "direct": each command is one click. To stutter-step, the model
  must itself issue the move and then the attack at the right moments, so it needs a faster
  cadence and the cooldown input it already has.
- **B. Intent plus motor skill.** L4 = an intent model (or rules) choosing STUTTER, KITE, FOCUS,
  HOLD... per group; L5 = scripted skills that time each move and attack to the weapon cooldown,
  through the same hands. The model then learns *when* to stutter, not the frame timing.
- **C. Later: intent plus learned motor.** The same L4 as B; L5 a small model trained on pro
  stutter steps (or by RL in drills).

The score for each: `eval/micro_timing.py` (pro-likeness of the timing), the 114 home defenses
(`eval/scenarios.py`), the rush benchmarks (`eval/vs_rush.py`), actions per minute spent (the same
APM budget for all). Same seeds and scenarios, only the manifest differs.

What B needs that A doesn't: intent labels for pro data, to train the intent model by imitation.
A script provider for the `intent_label` slot (§6.6) can find them in pro command streams: the
same units moved and then sent to attack within a second is a STUTTER; repeated moves away from
the nearest enemy with attacks in between is a KITE. As with every slot, other providers (a
model, a human) can label later, and the gold set keeps them honest.

### Manifests: a Gary is a choice per layer

v0.9, as it is in `gary/agent/manifests.py`:

```python
"v09": {"version": "v0.9",
        "posture": "early_pool/v1",
        "macro": {"provider": "learned", "model": "macro/latest", "builds": ["one_base_defense"]},
        "army": {"provider": "learned", "model": "army/latest"},
        "fight": {"provider": "command", "model": "fight_cmd_rl/v08",
                  "before": ["repair", "pull", "load"], "in_fight": ["reinforce"], "workers": False},
        "arbiter": "fixed_priority/v1"}
```

The executor is `scripted/v1` whenever there is a macro or an army; there is no motor layer yet
(step 3). Variant B for the experiment will change the fight layer and add a motor one. Every
version from v0.2 is a manifest (`v02`...`v09`, `v07free` for judging RL models, `drill` for
drills), built by `build_gary`; `python -m gary.play --gary v09` plays any of them, and the
benchmarks take the same names. v0.1's scripted build order was dropped (git history has it).
Options try one change on top of a manifest without a manifest of its own: `v010+post+suspend`
(`OPTIONS` in manifests.py, each with its measured result).

## Where today's code goes

| Today | Layer and provider |
|---|---|
| Rush detection, rush end (v0.9 `act`) | L1, `rules/early_pool_v1` |
| Rush build (`_rush_goal`), no expansion while holding | L2 scripted build for the posture; L1 constraint |
| Learned macro, `_plan`, goals | L2 `learned` |
| Build tasks, placement (`_spot`, `_spot_near`, `_fits`), gas, mining, idle sweep, hotkeys | L2b `scripted/v1` |
| Scouting | L3 (an info intent, SCOUT) for now; L1 decides when |
| Army model, its fight estimate | L3 `learned`; v0.9's hold becomes the posture's `army_home` |
| Worker pull, bunker load, repair, reinforce (v0.4, v0.7, v0.9 rules) | L4 rules issuing PULL, LOAD, REPAIR, ATTACK_MOVE intents; L5 skills that do the clicking (`_task_load` etc. move there) |
| Command model (v0.5+) and its RL | L4 provider for variant A |
| The `if` order in `act()`, `macro_turn`, `INTERRUPTIBLE` | L6 `fixed_priority/v1` |
| `HumanInterface` | L7, unchanged |

Since the one-Gary refactor (docs/REFACTOR.md, 2026-10-20) a race plugs in at L2b (gary/agent/executor.py's
`Executor`, with terran_executor.py and zerg_executor.py), and the Zerg's providers are posture `zerg`
(zerg_posture.py), the army options of army_rules.py (posts, raid, join, sweep, press) and fight `zerg`
(zerg_fight.py); the arbiter's `rotate` option takes the turns as Gary-Z did. Every game runs through
eval/match.py.

## Plan

Each step keeps Gary playable, and each step that only moves code must not change behavior: the
v0.9 manifest has to reproduce v0.9's games bit for bit (same replays for fixed seeds, as for the
earlier speedups).

1. **The layers as providers, versions as manifests.** Done: `gary/agent/` (gary, executor,
   posture, macro, army, fight, arbiter, manifests), `gary/play.py`; the class chain is gone.
   Parity: Gary's every action identical before and after in 110 games and scenarios (the rush
   benchmarks for v0.2-v0.9, 64 home defenses for v0.3, v0.4, v0.7free and v0.9, sampled RL
   plays) and the same drill results. Kept for parity, to change on its own later: the worker pull
   and v0.9's posture share one "zerglings last seen at home" clock (`Gary.calm_since`), so each
   one's sightings restart the other's calm.
2. **The arbiter as its own layer, with a trace.** Done: `fixed_priority/v1` names who got the
   hands for every task (`arbiter` events), how long each task held them and what a fight
   interrupted (`task_end`), and waits of 1 s or more (`wait`: macro with a decision taken, or a
   fight going on, while the hands were busy elsewhere), read from state the layers keep anyway.
   Saved next to every POV log (`*.trace.jsonl`); `python -m gary.agent.trace` sums traces up.
   Same actions in all 110 parity games and scenarios. A first look (v0.9, immediate 9-pool,
   seed 1, 8 min): the hands went 42% to macro, 19% to sending idle SCVs to mine, 9% to loading
   the bunker, 9% to the fight model.
3. **Intents and leases.** Done: the fight rules, the fight models and the army issue `Intent`s
   (`gary/agent/intents.py`: kind, units, target, the skill's data, source, ttl); the motor layer
   (`gary/agent/motor.py`, `direct/v1`) carries each out with a skill, today's clicks moved there.
   A skill's order takes its units' leases; an order to units another source still holds is
   traced (`lease_conflict`); enforced from v0.10 on (step 4). Same actions in all 110 parity games and
   scenarios. A first look (v0.9, immediate 9-pool, seed 1): the fight model and the bunker-load
   rule took marines from each other 9 times in 8 minutes.
4. **Roles and their skills; leases enforced.** Revised after watching pro fights: a label per
   command didn't hold up (most moments mix techniques: the front marines step back while the
   rest shoot), a role per unit did (`eval/micro_roles.py`: fire, close in, pull back, retreat,
   reposition, stand; toward and away measured group against group). Stutter isn't an intent:
   at the unit level it's a short pull back, then fire. So variant B is a role per unit every
   half second (`gary/agent/roles.py`, hand rules first: `rules/v1`), each role's skill picking
   where to go from the threats (weapon ranges, walkable ground), carried out by the command
   skill; leases enforced by priority (rules, then the fight's decisions, then the army).
   Manifests `v09lease` (leases alone) and `v09roles` against v0.9 on the three benchmarks. Done: every
   manifest since v0.10 enforces leases (`"leases": "enforced"`); gt1's roles in contact are hand rules
   (RoleAhead), gt1dev's the learned model.
5. **A learned role model:** imitation on the pros' role labels (per unit, every half second),
   then RL on the home defenses; variant B with a learned L4 instead of the hand rules. Done: the role
   model learned from the pros' units before contact (v0.29) and in it (gt1dev: attended units, the "again"
   decision, several orders a turn), each side; its first RL run that held (rl1, in gt1dev since gt0.6dev).
6. **The rest as needed:** KITE, SCREEN, LOAD as skills; §6.7's arbiter rules (fight priority
   with a minimum macro share); posture from more than rules.
7. **Every decision a policy, trained together.** Rules tuned one failure at a time traded one
   loss for another (against Zerg v2's all-in: a scout, a flood defense, an anchor, each neutral
   on 336 games): the decisions have to be learned, together, from whole games. What that needs:
   - **Each deciding layer a policy with probabilities**, that can sample in training and pick
     its likeliest in play, and logs each decision with its probability: the posture's mode
     (normal, early-pool defense, holding, zergling-flood defense) a small learned policy over
     symbolic facts, started as a copy of today's rules (distilled from them); the macro model
     sampling its next decision at a temperature; the army model; the roles (`RoleLearned`).
   - **A full-game trainer**: games against the opponent pool (the 9-pool, Zerg v1/v2, the
     replayed pros on three maps), each layer updated on what its decisions led to, kept near
     where it started (a KL penalty, as `train/roles_rl.py`). Games are chaotic (half a second
     of Gary's hands changes the next 90 s by +-425 net value, `eval/block_branch.py`), so
     decisions are judged against branches from the same moment of the same game (games are
     deterministic per seed: replayed to the moment, then a different draw), not against other
     games.
   - **Then the pieces the architecture lists**: the belief model (§6.9) and the opponent model
     (§7.5) as inputs to the posture and the army, the strategy selector's table (§6.8), and a
     Zerg Gary for self-play.
   Benchmarks and play keep each layer's likeliest decision; only training samples.
   First results: the plan policy starts as the posture's rules exactly (a strong vote for their
   mode plus a correction at zero: the same games, 6 seeds); the macro trained on 2,268 pairs of
   branches (`train/macro_rl.py`: at a moment of a game, its favorite against another of its
   options, the next 3 minutes) learned nothing: held out, it prefers the better one 50.0% of the
   time (before 50.6%). A pair's difference has a standard deviation of 848 net value (median
   112; 23% none at all) while a single production decision is worth tens: single decisions
   can't be credited in games this chaotic. So the learning goes where the effects are big and
   the counting is per game: the strategy selector (§6.8: which build style to play, from the
   map and what the scout saw, learned from game outcomes), then the fights on episodes (whose
   measures pair the same positions: +-10 on 239 home defenses).
   The value network (`train/value.py`: how good a moment is, from the macro model's view of
   it) was the missing piece for that credit: trained on 578k moments of 6,000 pro games and 88k
   of 648 Gary games, held out by game. Who wins: pros 0.64 AUC (from any moment, the early
   game included), Gary's decided games 0.92, but most of that is the lead already taken: before
   4:00 0.73, and against Zerg v2 a coin flip until 7:00 (0.49, 0.56). The net value of the next
   60 s: 52% of its variance explained (sd 477 -> 330), the next 180 s 30% (1136 -> 953). It
   reads who is ahead, not yet who will get ahead.
   As a judge of decisions (`eval/value_branches.py`: 288 moments of v0.19's games, each played
   on three times: as it was, with another of the macro's options, and with Gary skipping half a
   second; the same game replayed matched its first play in all 288): another macro decision
   changes the next 3 minutes at 77% of the moments, chaos alone at 39%; their spreads, sd 989
   against 529 (given any change, 1122 against 837). Judged by the value network at 30 or 60 s
   instead, both spreads shrink alike: the ratio of their variances stays 3.1-3.6 (the score's
   3.5), so the critic doesn't separate decisions from chaos any better than the score. It sees a
   little ahead: its read at 60 s correlates +0.34 with the net value of the 2 minutes after
   (the first 60 s of score: +0.27; at 30 s it sees nothing, +0.05; on placebo pairs, 0 as it
   should). Not yet the critic that credits single decisions: that needs inputs that foresee
   (the belief model's estimate of the enemy army, the fight in progress) and Gary's games of
   every style (its own were all style 1: `eval/walls_bench.py`'s Match forced it, now fixed).
   Value network v2 (`gary/policy/value.py`): those inputs (the belief model's estimate of the
   enemy army, hidden or not; both armies' last 30 s) and 1,080 Gary games of every style, plus a
   head learned by bootstrapping (the next 5 s of score plus its own value after them, discounted,
   half-life 60 s; it explains 42% of the discounted score to the game's end). Held out, against
   the same data without the new inputs: the next 60 s 30% explained (18%); the rest within noise
   (who wins against Zerg v2 before 4:00 rests on ~16 decided games: 0.44-0.56 from one training
   to the next). Judged once per branch (288 moments), the critic seemed worse than the score, but
   that test measured how much more a decision disturbs a game than a skipped half-second, not
   what it's worth. Measured right (`--repeats`: 90 moments, the macro's decision and another
   each replayed 6 times with both players' hands drawn anew from the moment): over 3 minutes,
   chaos spreads a pair by sd 512, a single macro decision's real effect is sd 130 (at 20 of the
   moments it changed nothing at all), and replays barely agree on which decision was better
   (+0.14, 95% interval -0.18 to +0.35). The critic at 30 s foretells the other replays' result as
   well as 3 minutes of score does (+0.23 against +0.14, equal within the noise): the same
   judgement from a sixth of the game. So single macro decisions are worth little next to the
   chaos (crediting one takes ~15 replays of it), not a fault of the critic; the learning stays
   per game (the strategy selector) and per fight (episodes paired on the same positions), and a
   macro learned from branches would need tens of thousands of them, judged by the critic at 30 s.
   Since (2026-10-24): the army's commit decision on paired branches with the forced arm locked at the
   pros' hazard (eval/pairs.py, commit_t), learned offline (train/commit_pairs.py); on 1,004 pairs it is at
   chance so far (docs/SELF_PLAY.md, "Learning beyond imitation").
   Value network v3 (2026-10-25): the critic had stayed on 217 features while the macro gave 421, unread. It now
   learns Gary's side from the league's own games, each replay re-simulated as the pros' are (one feature path; the
   net value from the units' deaths), a Terran and a Zerg critic (2,064 and 2,448 games since 2026-10-23): who wins on
   Gary's held-out games 0.88 AUC (0.56-0.72 before 5:00), the discounted head 33-36% of the score to the end, the
   net 180 s head nothing (the 30-minute league's late fights). Its view in play reads the moment as the training
   rows do (ValueView: its own look every second, the units inside a bunker, the player's own decisions, no style,
   map, year or matchup); tests/test_value.py guards it. Scoring the commit pairs' arms by it at their end (941 pairs)
   found no state where engaging pays that the deployed commit model doesn't already prefer. With early stopping (2026-10-26)
   its heads fit: net180 35% explained (28-37% in every window from 6:00), net60 33%, the discounted 42%.

Steps 1–3 are refactoring with parity checks, and they carry the risk of the plan: they touch
everything. They're worth it only if they stay small, so: no layer gets a provider it doesn't need
yet, and a manifest is a plain dict until we need more.

## Open questions

- **Learning across layers.** With a skill below it, the fight model acts every few seconds per
  group, and one intent becomes many clicks. RL over intents is a semi-Markov problem (options):
  the reward of an intent is everything until it ends. The trainer (`train/fight_cmd_rl.py`) has
  to handle variable durations.
- **Lease length.** Too short and units flicker between owners; too long and a skill keeps units
  after the fight moved on. Start with the intent model's own cadence plus a margin.
- **Who arbitrates between the arbiter's candidates and the posture?** A posture could also set
  the arbiter's shares (an early-pool defense gives fights priority). Not needed yet.
- **How far "direct" can go.** If variant A with a faster cadence matches B, the motor layer stays
  thin. The experiment decides, not this note.
