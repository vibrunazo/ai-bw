# Gary across races: where the code stands

Gary is one player for every race and every matchup ([ARCHITECTURE.md §1](ARCHITECTURE.md)). TvZ
with Gary as Terran was built first. This file is the audit of how far the code is from that
goal, as of 2026-10-18, and the order in which it gets there. Update it when a step lands.

## Data

| Matchup | Replays in the inventory | Re-simulated | Datasets (macro, army, fight) |
|---|---|---|---|
| PvZ | 18,924 | 192 | none |
| PvT | 18,187 | 138 | none |
| TvZ | 15,434 | 15,402 | all |
| PvP | 7,360 | 149 | none |
| ZvZ | 6,353 | 29, the rest running | none |
| TvT | 2,799 | 10, queued | none |

Everything but 598 games is from 1.16 and before (StarData, the TL packs): old maps, old metas.
The 598 Remastered-era games (version 1.21+) are the cwal.gg sample and a few TL games; a bulk
cwal.gg download is in progress. The inventory records era (`version`), source and bot flags per
replay; the trainers select on them (`train/select.py`: `--era old|modern`, `--source`, `--bots`),
so one set of datasets can train an old-era, a modern-era or a mixed model.

The pipeline is matchup-general already: `resim/run_batch.py`, the dataset builders and the
trainers take `--matchup`, and each race's unit types come from the data (`FeatureSpec.own_types`,
`seen_types`). The tech-cluster taxonomy (`taxonomy/discover.py`) runs per matchup and race; the
hand-named Zerg plans (`taxonomy/plans.py`) exist for TvZ only, and the macro dataset copes with a
matchup that has no taxonomy (style unset).

## Models

Per matchup and race: macro, army, commit, fight, belief, value. The belief tables (enemy types,
army supply, workers, town halls) cover all three races. The models are picked by
`manifests._newest(kind, matchup, race)`, whose defaults are TvZ and T: Gary-T's manifests do not
yet take the matchup from the game, so a Gary-T in TvT or TvP would load TvZ models.

## The Terran agent (gary/agent)

The layer contracts (arbiter, leases, roles, army, sightings, motor, trace, intents, selector) are
race-neutral. The Terran mechanics live in the providers. References to the Terran data module
(`gary/terran.py`) per file, as a measure of how bound each one is:

| File | Terran refs | What they are |
|---|---|---|
| macro.py | 145 | production buildings, depots, add-ons, bunkers, costs: the whole carry-out |
| executor.py | 29 | names, refinery, gas orders, add-on offsets, costs: the hands |
| answers.py | 23 | turrets, tanks, the CC: scripted answers |
| roles.py | 18 (+8 Zerg) | the CC as home, medics, SCVs, bunkers; mutas, lurkers, scourge as prey kinds |
| fight.py | 16 | SCV pulls, bunkers, the CC |
| posture.py, block.py, motor.py, simcity.py | 7-8 each | bunkers, marines, SCVs, building sizes |
| army.py, manifests.py, sightings.py, arbiter.py | 2-5 each | worker, town hall, comsat |
| gary.py, intents.py, selector.py, trace.py, answers' contracts | 0 | |

So: the macro carry-out and the hands are Terran; the fight, posture and answers rules are Terran
in their constants; everything above them is race-neutral. The order ids and map helpers that
`gary/terran.py` also exports are race-neutral and belong in a shared module.

## The Zerg player (gary/bots)

Gary playing Zerg is a separate stack: `zerg_rush.py` (319 lines) → `zerg_v2.py` (393) →
`zerg_v3.py` (823) → `gary_z.py` (1,623), plus `zerg_roles.py` (282). It shares the models
(`gary/policy`), the decision records and Sightings with the Terran agent, and reimplements the
rest: jobs instead of the executor, its own rallies and posts instead of posture, its own cluster
army, raid, sweep, press and alarm instead of the army and roles providers. Seven places assume
the enemy is Terran (the alarm's "away from their CCs", the press's building ranks, the sweep's
turrets and bunkers), so a ZvZ game is the first thing that breaks it.

## Order of work

1. **Mirrors from the data we have** (running): re-simulate ZvZ and TvT, build their datasets,
   train ZvZ Zerg and TvT Terran models. ZvZ Gary-Z vs Gary-Z exposes the Terran-opponent
   assumptions; TvT Gary-T vs Gary-T exposes the manifests' TvZ default and roles' Zerg prey
   kinds. Fix both with one table of the enemy per race (town halls, production, static defense,
   fighters, names) in a shared `gary/races.py`, looked up by the enemy race seen in the lobby.
2. **Modern replays** when the cwal.gg download lands: inventory, `run_batch.py --remastered`,
   datasets per matchup, then retrain `--era modern` against mixed and judge both on the league
   and on the humanlikeness gap measured against modern pros. The pro reference for economy and
   humanlikeness moves to the modern games.
3. **One Gary.** The Zerg stack moves onto the agent's arbiter: jobs become the Zerg executor
   (larva morphs, drone builds, creep); rallies and posts become posture; raid, sweep, press and
   alarm become army and roles providers that any race can use. A race is then a data module
   (`terran.py`, `zerg.py`, `protoss.py` with the same names) plus an executor and a macro
   carry-out. The gate: Gary-Z on the new stack matches its league record before the old stack is
   deleted (no compatibility layer).
4. **Protoss.** `gary/protoss.py`, the Protoss executor (probes place and leave, pylon power,
   gateway, robotics and stargate production, archon merges, shuttles) and the PvT, PvZ and PvP
   models from the data, old and modern. First test: Gary-P vs Gary-Z in the league.
