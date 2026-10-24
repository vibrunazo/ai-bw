# TODO: open work

What is left to do, newest first. Each section is dated; when an item is done it leaves this file (git history
and docs/PROGRESS.md keep what was done and how it measured). The GitHub issues are older and partly
overtaken: this file is the current list.

## Open threads (2026-10-24)

- **Commit pairs:** 1,004 pairs collected (CP110a, CP120a), the learner at chance and overfitting
  (train/commit_pairs.py); try the value network's read at the arm's end as the outcome (90 s of net value may be too
  short for a commitment) and the advantage's size rather than its sign, against a fixed rule ("engage at 10-15
  minutes") as the bar; the same lock and pairs for Gary-Z's commit decisions once something beats that bar.
- **Why v5's commit models trail the deployed one in games** (all of them, with or without the new channels, while
  scoring better offline): the era mix (weight the modern games as the macro's recipe does), ticks per game, how the
  deployed one was chosen; compare the deployed model's in-game behaviour with theirs (attack rate, commitment
  lengths, attacks into sunkens). Until then the deployed commit model stays.
- **The hunt's labels on the deployed commit model** (fine-tuned with a KL anchor to it, not retrained on v5), so the
  labels' effect is read apart from the data change.
- **gz1 with ownsupply and without overlords ahead,** against gt1: whether gz1 loses that stop-gap too.
- **TvT and ZvZ health rounds** on the dev manifests, read for absurd records (mirrors catch hidden race
  assumptions); the Protoss macro models' offline check on held-out pro states.
- **The gap score disagreed with results twice** (rethink, nosave: nearer the pros, worse games) and with the Zerg's
  counts once (ownsupply): judge decoder changes on results first; find what in the gap rewards it.
- **Runs:** a pre-flight test that plays one paired episode (the deep-copy bug broke every pair for days unseen); a
  way to stop a chain that also kills its workers and re-simulator calls (a stopped build marked 11,583 replays
  failed); requirements.txt pinned to the versions that work.

## Data and races (2026-10-19, updated 2026-10-24)

- **More modern replays.** The curve flattens around 3,000 clean games per matchup (docs/DATA_ERAS.md); the cwal.gg
  packs hold about 3,000 raw a matchup now (18,886), so roughly half again as many. cwal.gg has no API and its
  robots.txt disallows crawling: packs are downloaded by hand. Confirmed ladder bots are in ingest/inventory.py
  KNOWN_BOT_NAMES. The remaining desyncs (Attitude SE, Colorless Fate, TvT, PvT) are an SC:R-vs-1.16 unit-movement
  difference, traced with GARY_TRACE_UNIT / GARY_TRACE_ORDER (docs/DATA_ERAS.md); Aiolos fails with "unsupported map
  version 64".
- **TvT Gary-T.** TvT plays (tests/test_matchups.py); macro, army and commit models exist for it (2026-10-19); fight,
  belief, plan and engage are still TvZ's (hours of training: ask first).
- **Protoss** (docs/RACES.md step 4, docs/REFACTOR.md Phase 4): the executor (its data and macro, army and commit
  models exist); after TvZ self-play shows a measured gain. Its old replay packs aren't re-simulated yet.
- **Human game 4's list** (docs/PROGRESS.md): attacks (measured, partly done), SCV pulls and the early zerglings,
  single-unit micro, turrets, vision, the late game, building placement.

## Map mechanics on the current ladder maps (2026-10-18)

Colorless Fate has an egg wall and a half-mineral wall, Odyssey stacked minerals, Attitude SE critters; the 2026
pros' builds, placements and fights on these maps depend on them. Gary cannot see any of it: the re-simulation emits
no neutral units except minerals and geysers (the eggs are in no dataset) and the army grid skips neutral units. The
plan (docs/RACES.md principle: inputs rich enough for the per-map pro data to teach it, no rule per map):

1. Neutral units become first-class in the re-simulation, the macro features and the army grid; chokes and ramps
   recomputed from current walkability when a wall dies or is mined out.
2. ~~The map hash as a conditioning input~~: done for the macro (2026-10-19, with the year).
3. A per-map placement prior from the pros' building positions, replacing the hand-made walls table.
4. Executor abilities the behaviors need: attack a neutral unit, gather from a chosen patch, saturation counted by
   position (stacked minerals).
5. The league and self-play on the current ladder pool, extracted from the replays' map data; every new map gets a
   re-simulation check first.

## Direction (2026-10-15)

From the second outside review (after v0.25). It stands: rules for execution and reflexes, learned models for
decisions; the learning had failed for want of signal (a decision worth sd 130 against chaos's 512, scripted
opponents, short horizons), which self-play against a pro-derived opponent is meant to supply. Its steps since:
the v3 loop is frozen (the scripted Zergs are deleted; the league plays gz1); human games 1-4 were played through
the bridge (their failure lists are in docs/PROGRESS.md); the self-play skeleton is built (docs/SELF_PLAY.md); the
humanlikeness classifier and the gap to modern pros are on every round. Still open from it:

- **Anticipation as target inference** (human game 1: "like playing against someone AFK"): which asset each enemy
  group reaches in the next 15-20 s, learned from the pro games, judged as a classifier first, then as the trigger
  of a learned response.
- **The opponent model's belief over the enemy's plan (z)**, and a strategy layer conditioned on it.
- **How much to commit to defense, and acting first:** learned from outcomes in the league, not scripted (pros pull
  back just enough to survive).
- **Explainability:** a narrator over the decision records, once a trace schema has its first reader.

## The army model's fight estimate, from Gary's own fights (2026-10-13)

The fight estimate is trained on what the pro got: the change in army share in the 45 s after the pro's order
(`train/army.py`). Gary holds or attacks by what a pro would get from that fight, with hands that are weaker in
fights (1.5 move orders a fighting minute in v0.19's era, pros 8.4).

- Sample army-order moments from the pro data; let Gary take over from each (as `eval/scenarios.py` does) with the
  pro's order, several draws of its hands; label with Gary's own 45 s change.
- Train two heads on the same state, the pro's outcome and Gary's: their gap maps where Gary falls short of pros
  (compositions, terrain, sizes), a diagnostic on its own.
- Play with Gary's head; judge in the league against gz1.

## The strategy selector's table (2026-10-13, updated 2026-10-24)

The selector's table (v0.19) was counted against the scripted Zergs, now deleted, and against early aggression
only: rebuild it from league games against gz1, per map and style; then re-learn and re-test the early-defense
switch (`train/early_switch.py`) on that pool.

## Scattered fighters back into the main group (2026-10-13)

Two different problems; don't solve them with one rule.

- **Housekeeping, no fight on:** a fighter standing idle far from Gary's main group, no enemy near, walks back to
  it, as `home_scvs` sends stray SCVs back to mining. A chore, not a decision. Measure first (the stray tracker).
- **In a fight:** whether a new unit joins now or waits for the group. Pros lose 12% of their army alone, Gary ~48%
  in its losses; rules to hold new units back, a learned join/hold override and RL on the role choice all measured
  worse or equal, and the gather stop-gap (AGENTS.md) only partly. A real decision with exceptions: for the learned
  layer, once the critic and longer episodes value a fighter kept alive.

## An engine error at the map's edge (2026-10-13)

OpenBW stops a game ("attempt to get tile index for invalid position -7 263") when a unit's path leaves the map: seen
once in 40 games, an SCV in a main's corner behind its edge minerals sent to mine one of them. Such games count as
errors, not results. Find what puts SCVs in that corner (a depot spot by the edge?) and keep them out of it; the
engine itself is external code.

## The macro reads the scout (2026-10-13)

Only the early-pool rules and the answers (`gary/agent/answers.py`, a stop-gap: AGENTS.md) read what the scout sees;
the macro model has the seen counts as inputs. Judge whether it answers alone (the answers' register entry), then
whether a longer scouting look (`watch`) pays once something reads it.
