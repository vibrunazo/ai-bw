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
| ZvZ | 6,353 | 6,285 (6,103 clean) | all |
| TvT | 2,799 | 2,716 (2,486 clean) | all |

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

### Adding replays

Every stage is incremental, keyed by the replay's sha1: the inventory re-parses only new files,
`run_batch.py` re-simulates only replays missing from its index, the macro, army and fight
builders skip games already in theirs, and `build_orders.py` skips games already extracted and
keeps the rows of other matchups. Nothing is recomputed from scratch when replays are added;
`--force` redoes a stage on purpose. What does need redoing after new data:

- **Models** retrain from the whole dataset (hours, not days); there is no incremental training.
- **The tech-cluster taxonomy** (`taxonomy/discover.py`, HDBSCAN) can assign different cluster
  ids when re-run on more data, so the macro index's styles are rewritten
  (`macro_dataset.py --restyle`) and the macro model, which is conditioned on them, retrains. The
  Zerg plans (`taxonomy/plans.py`) are rules and stay stable.
- **The pros' decision history** (`train/macro_history.py`) is derived from the macro dataset and
  the macro model's decision index: re-run after either changes.

The inventory (`data/interim/inventory.jsonl`, one row per replay) is the index the scripts look
things up in: sha1, source and pack, era (`version`), matchup, map name and hash, duration, the
two players with race, APM, EAPM and bot flags, the winner, 1v1-human and duplicate flags. The
per-stage indexes (`resim_index.jsonl`, each dataset's `index.jsonl`, `build_orders.jsonl`) join
to it by sha1; `train/select.py` is that join for the trainers.

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

Since the refactor's Phase 0 (docs/REFACTOR.md, 2026-10-19) the agent reads its own race through
`g.race` (gary/race.py: worker, town halls, costs, sizes, names, producer, requires, research,
upgrades, supply, for T, Z and P), knows its matchup from the game's players (`g.matchup`) and its
enemy (`g.enemy`), and finds every model with `manifests.model_for(kind, matchup, race)`. The order
ids are in gary/orders.py. Every `T.` left in gary/agent is Terran mechanics by nature, for the
Terran executor in Phase 1: macro.py 96 (production buildings, depots, add-ons, bunkers), answers.py
19, roles.py 10, posture.py 8, executor.py and fight.py 5, the rest 1-3.

## The Zerg player (gary/bots)

Gary playing Zerg is a separate stack: `zerg_rush.py` (319 lines) → `zerg_v2.py` (393) →
`zerg_v3.py` (823) → `gary_z.py` (1,623), plus `zerg_roles.py` (282). It shares the models
(`gary/policy`), the decision records and Sightings with the Terran agent, and reimplements the
rest: jobs instead of the executor, its own rallies and posts instead of posture, its own cluster
army, raid, sweep, press and alarm instead of the army and roles providers. Seven places assume
the enemy is Terran (the alarm's "away from their CCs", the press's building ranks, the sweep's
turrets and bunkers), so a ZvZ game is the first thing that breaks it.

## Order of work

1. **Mirrors from the data we have.** Done for Zerg on 2026-10-18: ZvZ and TvT re-simulated and
   their datasets built; ZvZ Zerg macro (top-1 0.61) and army models trained; `gary/races.py`
   holds the enemy per race (town halls, production, static defense, fighters, names) and Gary-Z
   takes a matchup, so its press, sweep, raid memory and alarm read the table instead of naming
   Terran buildings. `eval/mirror.py` plays Gary-Z against Gary-Z; the first three games ran to a
   kill at 10:18 and 7:00 and one to the 20-minute limit, with mutalisk, zergling and scourge
   play. The records are tagged by player slot (a mirror cannot tell its players apart by race).
   Still open: Gary-Z reads the matchup from the launcher, not the lobby; the pro-history table
   and plans exist for ZvZ without build styles (no taxonomy yet); TvT Gary-T vs Gary-T has not
   been played (the manifests' TvZ default and roles' Zerg prey kinds will show there).
2. **Modern replays** (the first pack landed 2026-10-18: docs/DATA_ERAS.md has the comparison and the
   recipe) when more arrive: inventory, `run_batch.py --remastered`,
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
