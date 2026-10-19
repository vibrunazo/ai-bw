# One Gary: the refactor plan (2026-10-19)

A plan for a session to build. The goal is RACES.md step 3: one agent that plays any race, where
a race is a plug and everything above the plug is shared. Gary-Z's separate stack (gary/bots/,
3,400 lines) is retired when the shared agent playing Zerg matches its league record. No
compatibility layers, no forwarding scripts: each step replaces what it supersedes.

Read first: RACES.md (the audit this plan executes), CONTROL.md (the layers as providers and the
manifest contract), ARCHITECTURE.md §6.4-6.7, HANDOFF.md (state of the data and models). The other
session is giving the macro model a matchup input (gary/policy/macro.py `FeatureSpec.matchups`,
uncommitted at the time of writing): that is the model side of sharing and is not part of this
plan; coordinate before touching gary/policy/macro.py or train/macro.py.

## What is shared and what is a plug

Shared, already race-neutral (gary/agent): the `Gary` state object, the `Task` contract and
`task_runners`, `FixedPriority` (the arbiter), leases, `Sightings`, the decision records, the
trace, `DirectMotor`, intents, the strategy selector, `LearnedMacro`'s decision loop (which
decision, when, with what odds), `LearnedArmy` (where groups go), the roles contract. The models
are keyed by matchup and race already.

A race plug is three things:

1. **A data module** with one fixed set of names: `gary/terran.py` and `gary/zerg.py` today,
   `gary/protoss.py` later. The agent reads them through `g.race`, never by importing a race.
2. **An executor**: the race's hands. How a worker is picked and sent to build, how a unit is
   produced (train from a building / morph from a larva or a drone / warp in from a gateway),
   add-ons, gas, placement rules, idle workers, rallies.
3. **The macro carry-out**: what a decision `(act, uid)` means mechanically for this race, in
   `LearnedMacro.plan` and `Task` kinds. The decision loop stays shared.

Fight and roles rules that are race-specific (SCV pulls, bunker loads, repair, stim; Gary-Z's
retreat rule) are providers chosen by the manifest, as they are now; the contract does not change.

## Phase 0: the race object (one day)

Make the agent read race facts from one place, with no behavior change. Gate: byte-identical
games (the seeds in tests/test_smoke_game.py and two league seeds) before and after.

- `gary/race.py`: a `Race` dataclass built from a data module: `letter`, `worker`, `town_halls`,
  `supply_provider`, `gas_building`, `production` (unit -> building), `costs`, `sizes`,
  `requires`, `research`, `upgrade`, `army`, `names`, `tech_names`, `upgrade_names`, `supply`.
  `Race.of("T")` / `Race.of("Z")`. The data modules keep their constants; the dataclass is the
  index over them. Give `gary/zerg.py` the names it lacks (`PRODUCER`, `SIZE` is there,
  `SUPPLY` is there, `GAS_ORDERS`, `NAMES` for every building) and move the race-neutral order
  ids out of `gary/terran.py` into `gary/orders.py` (`ORDER_RESEARCH`, `ORDER_UPGRADE`,
  `ORDER_BUILD_ADDON`, `ORDER_PLACE_ADDON`, `GAS_ORDERS`), fixing the imports in gary_z.py too.
- `Gary.__init__` takes `race: str`; `g.race = Race.of(race)`. `build_gary` passes the
  manifest's `"race"` (default `"T"` for every existing manifest). The main town hall lookup
  (type 106 today) becomes `g.race.town_halls`.
- Replace every `T.X` in gary/agent with `g.race.x` where the meaning is "this race's X":
  `T.SCV` -> `g.race.worker`, `T.CC` -> `g.race.town_halls`, `T.COST` -> `g.race.costs`,
  `T.PRODUCTION`, `T.REQUIRES`, `T.SUPPLY`, `T.NAMES`, `T.SIZE`, `T.ARMY`. The counts per file
  are in RACES.md (macro.py 145, executor.py 29, answers.py 23, roles.py 24, fight.py 16, then
  single digits). What stays `T.` is Terran mechanics by nature (add-ons, depots as supply, bunkers,
  turrets, Comsat) and moves into the Terran executor in Phase 1; mark each with a `# terran`
  comment now so Phase 1 is a grep.
- `g.enemy = enemy_of(matchup, race)` from `gary/races.py` (it exists: the enemy's town halls,
  production, static defense, fighters, names). The matchup comes from the game: the lobby's races
  are in `obs["players"]`; `manifests._newest(kind, matchup, race)` gets both from `g`, so a
  Gary-T in TvT loads TvT models and fails loudly if there are none.

## Phase 1: the executor as a race plug (two to three days)

Gate: byte-identical Terran games again; the Zerg executor passes the mirror smoke
(eval/mirror.py) with the same builds Gary-Z makes in its first four minutes.

- `gary/agent/executor.py`: split `ScriptedExecutor` into `Executor` (race-neutral: reservations,
  `pick_worker`, `select_worker`, `spot_near`, `fits`, `town_hall_spot`, `idle_mining`,
  `transfer`, `refill_gas`, `rerally`, `hotkey_new_buildings`, the `_run_basic` task runner's
  select-and-order skeleton) and `TerranExecutor(Executor)` (add-on spots and `on_addon_spot`,
  `resume_construction` of unfinished buildings, depots as supply, refinery orders, the
  Comsat). Every `T.` left in the base class after Phase 0 goes to the subclass.
- `gary/agent/zerg_executor.py`: `ZergExecutor(Executor)`, built from gary/bots/zerg_v3.py's
  jobs and gary_z.py's carry-out, as task runners under the shared `Task` contract:
  - `build` tasks: a drone picked (`_pick_drone`), walked (`_walk`, the walk timeout), the
    building placed on creep (`creep_coming`, `_spot_for`, `_building_spot`), retried
    (`JOB_TRIES`), with the job's fate in the records (`_fate`). The `Job` dataclass becomes the
    task's `data`.
  - `morph` tasks from larvae (`_larva_morph`, `_pick_larva`, `_select_one`) and from units
    (lurker aspect, guardian/devourer) and buildings (lair, hive, spire, sunken/spore).
  - gas (`_gas`, the floating rule from gary_z.py), idle drones (`_mine_idle`), overlord
    supply, rallies front/back/minerals with `siege_until` (`_rallies`, `gather_point`).
  - the "nothing of its own on top" unit pick from c13 (an overlord took every click).
- `LearnedMacro.plan` and `Task` kinds: `TRAIN` becomes "produce" and the executor decides
  whether that is a train order to a building or a larva morph; `BUILD`/`EXPAND` keep their kinds
  with the race's worker; `BMORPH` (building morph) is a new kind the Terran executor refuses.
  The decision index and the decoder stay shared. `in_production`, `count`, `free_building`,
  `free_geyser`, `free_base` read `g.race`.
- `build_gary`: `g.executor = {"T": TerranExecutor, "Z": ZergExecutor}[g.race.letter](g)`.
- Gary-Z's `_decide` (dice decoder, `_summed`, floors, hatchery mass, the alarm's step list) is
  already an option set of the shared decision loop in spirit; the dice decoder and the mass
  option exist in `LearnedMacro` as manifest options (`decoder`, `odds_mass`). Whatever Gary-Z's
  `_decide` does that `LearnedMacro` does not becomes a manifest option, not a Zerg special case.

## Phase 2: posture, army and roles providers from Gary-Z (two days)

Gate: the new Zerg manifest ("gz1") in the league against Gary-T v029 with c13's seeds and setup
scores at least Gary-Z's record (alive at 12 and 15 minutes, Zerg ahead at the limit, Gary-T
wins) and its gap score; a ZvZ mirror round plays to the end as the first mirrors did.

- Posture: `NormalPosture` for Zerg plus a `ZergPosture` carrying Gary-Z's posts and overlord
  posts (`_posts`, `_overlord_posts`) and the alarm (`threat` from `enemy_fighters` away from
  `g.enemy.town_halls`, `ALARM_*`). The alarm's effect on production (sunkens, creep, army
  units first) becomes `posture.allowed`/`posture.build`, the contract the Terran early-pool
  posture already uses.
- Army: `LearnedArmy` gains, as options, what Gary-Z's army has and the Terran's lacks: the raid
  (`_mutas`, `_raid_line`, `_raid_post`, `_raid_go`, RAID_*), the sweep (`_intercept`,
  INTERCEPT_*), the press (`_press`, PRESS_*), cluster grouping (`_clusters`, `_clusters_army`),
  commitments (`_committed`, the commit model), and `_remember_enemy` (the enemy memory the
  shared `Sightings` can carry instead). None of these is Zerg-specific: a Terran can raid with
  wraiths and press with tanks. They are options a manifest turns on.
- Roles: `gary/bots/zerg_roles.py`'s `ZergRoles` becomes a `RoleRules` subclass (the retreat rule
  before the model; the Zerg roles model), chosen by the manifest's `fight.provider`.
- Manifests: `"gz1"` = race Z, posture zerg, macro learned (the ZvZ or TvZ models by the game's
  matchup, the pro-history option as GARY_Z_HISTORY is now), army learned with raid/sweep/press,
  roles zerg. The env options of gary_z.py (GARY_Z_FLOOR, GARY_Z_MASS, GARY_Z_FLOCK_SAVE,
  GARY_Z_HISTORY, GARY_Z_COMMIT) become manifest options; eval/league.py and eval/mirror.py
  stop setting them.
- Records: `LOG.decision(side, ...)` and `hi.side` carry the race letter; both players of a
  mirror are told apart by the slot the launcher sets (eval/mirror.py does this already). Nothing
  else to change.

## Phase 3: one launcher, delete the old stack (one day)

- `eval/match.py`: a headless game between two manifests on a map: `--a v029 --b gz1 --map ...
  --seed ... --minutes ...`, races from the manifests, both POVs, traces and records saved, the
  result line the league parses. It replaces `gary/bots/zerg_rush.play_match`, `eval/mirror.py`
  and the `--opponent` zoo (v1, v1a, v1d, v2, v3, replayed are gone with the old stack;
  `zerg_replayed.py` is kept only if something still evaluates against replayed pros, otherwise it
  goes too). eval/league.py's `play()` calls it. gary/play.py (live games) builds through
  `build_gary` with the race from the lobby.
- Delete gary/bots/zerg_rush.py, zerg_v1.py, zerg_v2.py, zerg_v3.py, gary_z.py, zerg_roles.py,
  hello.py. Fix every import (eval/*.py, train/*.py, tests/). `git grep "gary.bots"` must come
  back empty but for zerg_replayed if kept.
- Docs: RACES.md step 3 done, the Zerg player paragraph rewritten; PROGRESS.md components table
  ("Matchups and races" row, "Macro executor" row); CONTROL.md "Where today's code goes";
  HANDOFF.md.

## Phase 4: Protoss (after, separately planned)

`gary/protoss.py` with the same names, `ProtossExecutor` (probes place and leave, pylon power as
the placement rule, gateway/robotics/stargate production, archon merges as a morph kind, shuttles
as transports), the PvT/PvZ/PvP models (the datasets exist for the modern pack). Not in this
plan's scope; the plan above must leave nothing that assumes two races (no `"T" if ... else "Z"`).

## Rules for the build

- One phase per pull request-sized commit series; each phase ends green on its gate before the
  next starts. The gates are the tests: add the byte-identical check as a test
  (tests/test_parity.py: a fixed manifest, map and seed, the POV log's hash before and after;
  the "before" hash recorded in the test when Phase 0 starts).
- Never edit the agent while a league round runs (the round imports the live code).
- Keep PROGRESS.md updated per phase (a line in the components table, a dated entry).
- Nothing new is learned in this refactor. If a behavior of Gary-Z cannot be reproduced through
  the shared contracts, write down which contract is missing rather than adding a Zerg branch.
- Models are not retrained for this; the matchup-input models the other session is building drop
  in through `_newest(kind, matchup, race)` when they exist.

## Order and effort

| Phase | Days | Gate |
|---|---|---|
| 0 Race object | 1 | byte-identical Terran games |
| 1 Executor plug | 2-3 | byte-identical Terran; Zerg mirror smoke builds like Gary-Z |
| 2 Providers from Gary-Z | 2 | gz1 >= Gary-Z in the league (c13 setup) and in ZvZ |
| 3 Launcher, delete | 1 | `git grep gary.bots` empty; league and mirror rounds run from eval/match.py |

About a week of one session. Phases 0 and 1 can start now; Phase 2's gate needs the league, so
coordinate the machine with whoever runs rounds.
