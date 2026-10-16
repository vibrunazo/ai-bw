# Self-play: the plan (draft for review)

Direction 3 in TODO.md: every piece of ARCHITECTURE.md present and connected, crude where it has to
be, training together. The first milestone is a learned Zerg Gary that plays full games against the
Terran Gary in OpenBW, from the same pro-replay pipeline. It's the opponent the rest needs.

## What exists

| Piece | State |
|---|---|
| Zerg macro model | trained 2026-10-04 on the Zerg side of the 14k TvZ games (runs/macro/TvZ_Z_*), never used |
| Zerg army model | `train/army.py --race Z` runs on the existing dataset (both players' orders are in it); not trained yet |
| Zerg execution | Zerg v3 (gary/bots/zerg_v3.py) through the same human interface: building jobs (any building, any spot, retried), larva production, tech, lurker morphs, groups and a mutalisk flock with their orders |
| Two players in one game | HumanInterface steps several players on one OpenBW game (one advances it) |
| Opponent model | the belief model (gary/policy/belief.py) over what the enemy has; not yet over its plan (z) |
| Trace | trace.jsonl (schema trace/v1): arbiter grants, tasks, intents |
| Critic | the value network that grades unfinished games (the selector's) |
| Parity | the bridge matches OpenBW (adapters/scr_bridge/tools/parity.py), so headless lessons hold live |

## The crudest Zerg Gary (v0)

1. **Macro:** Zerg v3's decisions (`_plan`, `_want`, `_tech`) replaced by the Zerg macro model's,
   as the Terran's are: each decision (a drone, an overlord, a hatchery, the pool, a Lair, a Spire,
   lurker aspect, ...) mapped onto v3's execution (a larva, a job, a morph). Caps from the pros'
   counts, as the Terran macro has.
2. **Army:** the Zerg army model (trained first) chooses where groups go, and v3's group and flock
   orders carry it out.
3. **Fights:** v3's scripted micro for v0. The first learned piece after that is Zerg fight roles
   from the pros' Zerg units (ingest/role_dataset.py labels marines; the same labels for zerglings
   and mutalisks: fire, close in, pull back, retreat). That's where kiting comes from: a Zerg that
   hits a lone SCV and pulls back, the thing the user beat Gary with.
4. **Done when:** 50 headless games Gary-T vs Gary-Z on the pro maps with no crash and no stall,
   and Gary-Z's economy within the pros' range (drones at 6:00 and 10:00, hatcheries, tech timings).

## Then the loop

- **League:** Gary-T and Gary-Z play each other in parallel headless games (low priority, nightly
  batches on this machine; full-game league training is rented compute). Each game's trace and
  outcome are kept.
- **One learned piece updated from outcomes:** a head of one model (a fight-role or army decision),
  trained by RL on the league's games with the critic grading unfinished ones and a KL term to the
  imitation policy, so it can't drift away from how pros play.
- **Anchors outside the loop:** the pros' distributions (TODO Direction 4) and human games. A league
  of two crude imitators can settle into passive, undecided games; the anchors catch that.
- **Anticipation (TODO):** a forecast of each enemy group's position 2-5 s ahead, learned from the
  pro games, an input to both sides' army and fight decisions.

## Order of work

1. Zerg army model trained; Gary-Z v0 (macro + army learned, v3's execution and micro).
2. 50-game Gary-T vs Gary-Z check; fix what stalls.
3. Zerg fight roles learned (kiting).
4. League harness: batches, traces, outcomes, the scoreboard.
5. First RL update of one piece from league outcomes, judged against the anchors and human games.

## Risks

- Two crude imitators may mostly produce passive games. Mitigation: the anchors, and time limits
  with the critic grading.
- Compute is shared: segments from mid-game states and nightly batches first.
- Zerg v3's execution was written for its own plans: some decisions the model makes (a third
  hatchery's spot, morphs at odd times) will need new execution.
