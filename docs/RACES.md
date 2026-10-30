# Gary across races: where the code stands

Gary is one player for every race and every matchup ([ARCHITECTURE.md §1](ARCHITECTURE.md)). TvZ
with Gary as Terran was built first. This file is the audit of how far the code is from that
goal, as of 2026-10-30, and the order in which it gets there. Update it when a step lands.

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
`seen_types`). The tech-cluster taxonomy (`taxonomy/discover.py`) runs per matchup and race; every
matchup's build styles are named in the catalogue (`gary/styles.py`) and every game labelled by
`taxonomy/label_all.py`, both sides, so each race's macro model learns every matchup's styles.

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

Every race has the same set: one macro model per race over every matchup, the matchup an input and every
matchup's styles (`runs/macro_exp/{T,Z,P}_all_v2all`); army and commit models per matchup-race; a fight role
model per matchup-race from the pros' units in that matchup (`ingest/response_dataset.py`, `train/roles.py`),
mirrors included. Left uneven: the critic (`train/value.py`) and the belief model are TvZ only (#62 needs a
critic per matchup), and the Terran's TvZ role model alone has had RL (rl1). Every model is found by
`manifests.model_for(kind, matchup, race)` or given per matchup in the manifest (`roles_data` keyed by matchup).

## One agent, three race plugs

The layers (arbiter, leases, macro decision loop, army, fight, motor, grouping, trace) are shared; a race is a
data module (`gary/terran.py`, `zerg.py`, `protoss.py`, read through `g.race`), an executor (the hands:
`terran_executor.py`, `zerg_executor.py`, `protoss_executor.py`) and race-keyed tables (`gary/races.py` for the
enemy, the key layouts in `grouping.py`). Each race has a manifest that plays and a
dev build where learning is done and judged without the stop-gaps (AGENTS.md):

| | Terran | Zerg | Protoss |
|---|---|---|---|
| Plays | `gt1` gt0.10 | `gz1` gz0.3 | `gp1` gp0.4 (crude) |
| Dev build | `gt1dev` gt0.24dev | `gz1dev` gz0.12dev | `gp1dev` gp0.4dev (2026-10-30; 96 paired games a matchup against gp1: PvT -0.029, PvZ -0.068, PvP -0.076) |

## What still differs between the races

Differences of the game's mechanics are expected: each executor (add-ons, bunkers and SCVs building for the
Terran; larvae, morphs and eggs counted once for the Zerg; probes placing and leaving and pylon power for the
Protoss) and the Zerg's arbiter rotating its jobs. The rest is uneven work, to fold into shared code or race
tables over time. In the dev builds:

| Area | Terran (`gt1dev`) | Zerg (`gz1dev`) | Protoss (`gp1dev`) |
|---|---|---|---|
| Macro decoder | dice, summed odds for expansions, a queue of two in the building, built and unit counts | dice, its own extras: `pace`, `hold`, a tech floor (10%), a wait floor (5%), `hatch_mass`; no unit count | dice, summed odds, unit count |
| Scripted macro | the answers to lurkers and mutalisks (TvZ, AGENTS.md) | the early-pool alarm's builds (the Terran's are off since gt0.18dev) | none |
| First producer by a time (`rax_by`) | 78 s | 2:45 | none |
| Fight | TvZ: the RL'd role model and the fight rules (repair, pull, load, detect, stim, reinforce); PvT and TvT: the matchup's role model | its own provider (`zerg_fight.py`: the role model, lurkers, retreat when outweighed) | the matchup's role model, no rules: no archon merges, no shuttles |
| Army | the commit model, the hunt as advice, the static-defense nudge, scans | the commit model; overlord posts and mutalisk raid as rules (no model moves those units) | the commit model |
| Control groups | by type, keys 1-4 (the pros' layout) | by type, keys 1-3 | by type, keys 1-3 (not yet read from the Protoss pros' hotkeys) |
| Critic | TvZ | TvZ | none |

The playing builds differ more: `gt1` keeps every stop-gap, `gz1` the pros' decision history in its macro inputs,
`gp1` the hunt, a Pylon ahead of production and no control groups.

## Order of work

1. **A dev build per race.** Done: `gp1dev` (2026-10-30) completes the three.
2. **One macro decoder.** The Zerg's extras and the Protoss's Pylon ahead each measured off, or on for every race
   (gp1dev, without the Pylon ahead, decides fewer army units and more Pylons: docs/PROGRESS.md, 2026-10-30).
3. **The fight rules as race tables**: the Terran's hand rules and the Zerg's retreat, then the army rules (posts,
   raid) for any race's flyers.
4. **A critic per matchup** (#62), then the belief model per matchup.
5. **The Protoss's own mechanics** (#36): archon merges, shuttles, its early defense, its key layout from the pros'
   hotkeys.
