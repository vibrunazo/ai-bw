# Gary across races: where the code stands

Gary is one player for every race and every matchup ([ARCHITECTURE.md §1](ARCHITECTURE.md)). TvZ
with Gary as Terran was built first. This file is the audit of how far the code is from that
goal, as of 2026-10-24, and the order in which it gets there. Update it when a step lands.

## Data

| Matchup | Replays in the inventory (Remastered era) | Re-simulated (ok) | Games in the datasets: macro / army / fight |
|---|---|---|---|
| PvZ | 22,694 (3,424) | 2,928 (2,872) | 2,820 / 2,665 / 2,665 |
| PvT | 22,456 (3,455) | 2,610 (2,562) | 2,273 / 2,166 / 2,166 |
| TvZ | 19,324 (3,329) | 18,066 (17,760) | 16,550 / 16,549 (army v5; v3 12,953) / 16,502 |
| PvP | 10,605 (3,158) | 2,781 (2,723) | 2,694 / 2,564 / 2,564 |
| ZvZ | 9,509 (3,041) | 9,216 (9,112) | 8,928 / 8,928 / 8,910 |
| TvT | 6,142 (3,048) | 5,367 (5,259) | 4,639 / 4,639 / 4,621 |

92,027 replays in all. The cwal.gg packs (2026) bring 18,886 Remastered-era games, about 3,000 per matchup;
the rest is 1.16 and before (StarData, the TL packs): old maps, old metas. Protoss's re-simulated games are
mostly the modern ones; its old packs aren't re-simulated yet. The inventory records era (`version`), source and bot flags per
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

Macro, army and commit models exist for all nine matchup-races (runs/macro, army, commit: PvT, PvZ, PvP,
TvT, ZvZ trained 2026-10-19; the shared per-race macro models, runs/macro_shared, take the matchup as an
input); fight, engage, plan and value for TvZ Terran; belief and value for TvZ both sides; the fight roles
from the response sets, TvZ both sides. The belief tables (enemy types, army supply, workers, town halls)
cover all three races. Every model is found by `manifests.model_for(kind, matchup, race)` from the
game's own matchup (Phase 0).

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

Since Phase 1 (2026-10-20) the hands are a race plug: gary/agent/executor.py (shared),
terran_executor.py and zerg_executor.py, and the macro's decision loop is shared, Gary-Z's rules as
its options. The manifest gz1 plays Zerg on the shared agent (its army and fights come from Gary-Z in
Phase 2); eval/mirror.py --sides gz1 garyz plays it against Gary-Z.

## The Zerg player (gary/agent, manifest gz1)

Since the refactor (docs/REFACTOR.md, 2026-10-20) Gary plays Zerg on the shared agent: the Zerg
executor (gary/agent/zerg_executor.py: larva morphs, unit and building morphs, drone builds through the
shared build task, overlords ahead, the gas float, hatchery rallies), the alarm as a posture
(zerg_posture.py), posts, raid, join, sweep and press as army options (army_rules.py), the role model's
fights and the lurkers as a fight provider (zerg_fight.py, zerg_roles.py), Gary-Z's turns as the
arbiter's rotate option. Gary-Z's separate stack (gary/bots: zerg_rush, v1-v3, gary_z, 3,400 lines) is
deleted, its league record matched first (round G6 against L9). The enemy is read by race
(gary/races.py, `g.enemy`), so ZvZ and ZvT play on the same code.

## Order of work

1. **Mirrors from the data we have.** Done for Zerg on 2026-10-18: ZvZ and TvT re-simulated and
   their datasets built; ZvZ Zerg macro (top-1 0.61) and army models trained; `gary/races.py`
   holds the enemy per race (town halls, production, static defense, fighters, names) and Gary-Z
   takes a matchup, so its press, sweep, raid memory and alarm read the table instead of naming
   Terran buildings. `eval/mirror.py` plays Gary-Z against Gary-Z; the first three games ran to a
   kill at 10:18 and 7:00 and one to the 20-minute limit, with mutalisk, zergling and scourge
   play. The records are tagged by player slot (a mirror cannot tell its players apart by race).
   Since: the matchup comes from the game's players (Phase 0) and tests/test_matchups.py plays TvZ, ZvZ
   and TvT through eval/match.py. Still open: the pro-history table and plans exist for ZvZ without build
   styles (no taxonomy yet); TvT has had no rounds or health check beyond the matchup test.
2. **Modern replays** (the first pack landed 2026-10-18, about 3,000 a matchup by 2026-10-24:
   docs/DATA_ERAS.md has the comparison and the recipe) when more arrive: inventory, `run_batch.py --remastered`,
   datasets per matchup, then retrain `--era modern` against mixed and judge both on the league
   and on the humanlikeness gap measured against modern pros. The pro reference for economy and
   humanlikeness moves to the modern games.
3. **One Gary.** Done on 2026-10-20 (docs/REFACTOR.md Phases 0-3). The Zerg stack moves onto the agent's arbiter: jobs become the Zerg executor
   (larva morphs, drone builds, creep); rallies and posts become posture; raid, sweep, press and
   alarm become army and roles providers that any race can use. A race is then a data module
   (`terran.py`, `zerg.py`, `protoss.py` with the same names) plus an executor and a macro
   carry-out. The gate: Gary-Z on the new stack matches its league record before the old stack is
   deleted (no compatibility layer).
4. **Protoss.** `gary/protoss.py`, the Protoss executor (probes place and leave, pylon power,
   gateway, robotics and stargate production, archon merges, shuttles) and the PvT, PvZ and PvP
   models from the data, old and modern. First test: Gary-P vs Gary-Z in the league. The data side is
   largely there (the table above; macro, army and commit models for PvT, PvZ, PvP since 2026-10-19); the
   executor is what's missing, and it waits until TvZ self-play shows a measured gain (TODO.md).
