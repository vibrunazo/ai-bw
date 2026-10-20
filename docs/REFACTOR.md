# One Gary: the refactor plan (2026-10-19)

A plan for a session to build. The goal is RACES.md step 3: one agent that plays any race, where
a race is a plug and everything above the plug is shared. Gary-Z's separate stack (gary/bots/,
3,400 lines) is retired when the shared agent playing Zerg matches its league record. No
compatibility layers, no forwarding scripts: each step replaces what it supersedes.

Read first: RACES.md (the audit this plan executes), CONTROL.md (the layers as providers and the
manifest contract), ARCHITECTURE.md §6.4-6.7, DATA_ERAS.md (the data and models). The other
session is giving the macro model a matchup input (gary/policy/macro.py `FeatureSpec.matchups`,
landed 2026-10-19 as the shared per-race macro models, runs/macro_shared/<race>_all_*, chosen by
`gary.policy.macro.macro_model_path(matchup, race)` and on by default): that is the model side of
sharing and is not part of this plan. Models today: macro, army and commit for all nine
matchup-races; fight, belief, plan and engage for TvZ Terran only. A Gary of another race or
matchup therefore plays with the fight and roles providers that have models, or with rules.

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
  are in `obs["players"]`. One lookup for every model kind, `manifests.model_for(kind, g)`,
  built on `macro_model_path`'s rule (the race's shared model when it covers the matchup, else the
  matchup's own newest), so a Gary-T in TvT loads TvT models and fails loudly where there are
  none (fight, belief, plan, engage: TvZ Terran only today; the manifest then has to name a
  rules provider or the game does not start).
- `gary/protoss.py` with the same names as the other two data modules (unit and building ids,
  costs, sizes, requirements, production, research, upgrades, army, names, supply) and
  `Race.of("P")`, so the race object is proven on three races from the start. No Protoss
  executor yet: that is Phase 4. Protoss ids already in `gary/races.py` move here.

**Done 2026-10-19.** tests/test_parity.py holds the gate: three 8-minute games on the ladder maps
(KnockOut, Fighting Spirit, Odyssey; seeds 4004-4006) with the day's defaults, the hashes of both
screens, Gary-T's trace and the records, recorded before the first change. As built, against the
list above: `model_for(kind, matchup, race)` takes the matchup and race rather than the Gary (the
training scripts call it too), and kinds trained for TvZ Terran only (roles, macro_rl, fight_cmd_rl)
refuse any other matchup; no `# terran` comments, since every `T.` left in gary/agent is Terran
mechanics (a grep for `T\.` is Phase 1's list); the records' side is `hi.side` (what the interface's
own records use; it holds in a mirror, where a race letter doesn't). gary/races.py's Protoss ids
were wrong (165 and 166 are the Templar Archives and the Forge; the Photon Cannon is 162): fixed by
reading gary/protoss.py. Still a `"T"`: gary/play.py's live sides (Phase 3, the launcher).

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

**Done 2026-10-19/20.** As built, against the list above:

- `Executor` (gary/agent/executor.py) is the race-neutral hands; `TerranExecutor`
  (gary/agent/terran_executor.py) and `ZergExecutor` (gary/agent/zerg_executor.py) plug in through
  hooks rather than overriding the flow: `carry_out(act, uid, ...)` (what a decision means for the
  race: `LearnedMacro.plan` keeps the shared gates and asks it), `basic` / `commands` (the task kinds
  a race adds: Terran's resume, Zerg's produce, morph and hatchery rally), `site_ok` (placement rule),
  `command_build` (Terran add-ons), `build_order` (a drone's), `supply_due` (a depot or an overlord
  ahead of production, option depots), `in_production` and `count` (eggs hide what they'll be; the
  pros' caps count what stands), `burst` (a larva's decisions come in bursts), `box_extras` /
  `box_orders` / `click_blockers` (what a box around workers may take, and the overlords over the
  fields), `in_flight`, `rerally`, `refill_gas` (Zerg: the gas float).
- One build path for every race: a drone's building goes through the shared build task (pick, walk,
  place, the reservations), with its order (DroneStartBuild) and its orders counted as busy (DroneLand,
  70: a drone re-sent while landing never started). Gary-Z's jobs are not ported.
- `TRAIN` did not become "produce": the model's own acts stay (TRAIN for Terran, MORPH for a larva's
  unit or a unit's morph, BMORPH for a building's morph), and the executor's plan says which task
  kind carries one out; the macro makes a task of the plan's kind (Terran's are all "command").
- Gary-Z's `_decide` as `LearnedMacro` options (manifest gz1): `decoder` dice, `odds_mass` with the
  Zerg one-offs, `hatch_mass`, `park` (a decision the money doesn't cover is parked with its money
  kept; a larva's that waits for money or a larva is no goal, the dice again next look; a building's
  drone sets out 100 short), `pace` (the timing head's tempo, bursts), `tech_min_p`, `wait_min_p`,
  `supply_ahead`, `history` pro, `style` roll. The caps now apply to morphs too, and a rule's goal
  (the overlord) ignores them. Not yet: the alarm (Phase 2 with the army), the flock's saving.
- The gate: `python -m eval.zerg_opening` plays gz1 against Gary-Z in ZvZ mirrors on two ladder maps,
  each seed with the sides swapped, and compares the openings re-simulated. 32 games of 4 minutes,
  medians, gz1 / Gary-Z: drones at 4:00 12 / 12, Spawning Pool started 1:35 / 1:54, Extractor 1:17 /
  1:24, larva-seconds unused 290 / 283, drone-seconds off work 113 / 125. Still apart: the second
  hatchery by 4:00 in 14 games of 32 / 21 (the first hatchery decision at a median 3:38 / 1:55: Gary-Z
  rolls the dice more often, its macro not sharing the hands with the chores), and supply-blocked
  36 s / 26. tests/test_zerg_smoke.py holds one such opening; tests/test_parity.py byte-identical.

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

**Done 2026-10-20.** As built, against the list above:

- Posture: `ZergPosture` (gary/agent/zerg_posture.py) carries the alarm; its effect is the build "alarm"
  (`AlarmDefense`: sunkens at the natural, then army units by the model's odds) that hands back to the
  learned macro when it has nothing to do (`falls_through`), under the posture's `no_workers` and
  `one_off_min_p` filters. The overlord posts went to the army layer instead (option rule `posts`): they
  are its scouting, and the one Gary has to find the enemy itself (Gary-Z was handed its start by the
  launcher), before then an overlord to each start it may be at.
- Army: `LearnedArmy` with commitments was Gary-Z's cluster army already; gary/agent/army_rules.py adds
  the option rules `posts`, `raid` (the flyers' raid, per-race anti-air tables), `join`, `sweep`, `press`
  over a shared `EnemyMemory`; options `scout` and `types`; the hunt (the Terran's) on for gz1, so a far
  lead ends the game. The motor selects a Zerg army by a box and a ctrl+click on a portrait of the
  selection panel (gary/interface.py, the user's way of splitting units at the rally point).
- Roles: not a `RoleRules` subclass: the fight provider "zerg" (gary/agent/zerg_fight.py) hosts
  `ZergRoles` (moved to gary/agent/zerg_roles.py) and the lurkers' burrow routine, with fight turns
  that may put a chore aside, the macro's turn after each.
- Gary-Z's turns: the arbiter option `rotate` (army, macro and the chores in rotation, 3 actions at most
  each), found by the parity diff below.
- Env options: gz1's are manifest options; gary_z.py's own stay until Phase 3 deletes it.
- The gate: rounds G1-G6 (48 games each on the ladder maps against the default Gary-T, L9 the baseline).
  G1-G3 played gz1 with b_rank hands (the launcher gave pro hands only to the Zergs it listed); with pro
  hands G4 and G5 matched Gary-Z but for the lead at the limit (0 games to 5). A parity diff of the five
  L9 games Gary-Z led at 20 minutes against gz1's on the same seeds (eval/why.py, 5-13 minutes) named the
  rotation, the macro's history counting failed larva tasks, and a Pool rule firing when none could be
  built; ported, G6: Gary-T won 25 (30), gz1 1 (0), Zerg ahead at the limit 7 (5), alive at 12 / 15
  minutes 41 / 36 (42 / 25), gap score to modern pros 0.28 (0.33). A ZvZ round of gz1 against gz1 played
  to kills at 8:30, 11:00 and 19:00 and to clear leads at 20 minutes.
- Known differences: the commitment floor is off (as Gary-Z's default since L7); the flock's saving
  (`GARY_Z_FLOCK_SAVE`, off in Gary-Z too) isn't ported.

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
  the handoff notes (since folded into AGENTS.md and TODO.md).

**Done 2026-10-20.** eval/match.py is the one launcher (two manifests, races from them, both sides'
screens, traces and records, paired fight episodes): Gary-T against gz1 through it plays byte for byte the
games the old launcher played. eval/league.py, the tests and the tools that step a game (`Match`:
army_orders, block_branch, value_branches, losses, train/plan, value, early_switch, macro_rl) use it; their
opponent pools are gz1. gary/play.py plays the local player's race (`--terran`, `--zerg`). Deleted:
gary/bots (zerg_rush, zerg_v1-v3, gary_z, zerg_replayed, hello; zerg_roles moved to gary/agent),
eval/mirror.py, vs_rush.py, walls_bench.py, block_drill.py, lurker_games.py, army_effect.py,
garyz_accept.py, zerg_opening.py (each built on the scripted Zergs; in git). `git grep gary.bots` is
empty. tests/test_parity.py now plays Gary-T against gz1 (hashes recorded again: the Zerg changed);
the smoke games play gz1. A Gary's hands are its manifest's (`hands`, pro by default).

## Phase 4: Protoss (after, separately planned)

`gary/protoss.py` with the same names, `ProtossExecutor` (probes place and leave, pylon power as
the placement rule, gateway/robotics/stargate production, archon merges as a morph kind, shuttles
as transports), the PvT/PvZ/PvP models (the datasets exist for the modern pack). Not in this
plan's scope; the plan above must leave nothing that assumes two races (no `"T" if ... else "Z"`).

## Later: building placement (after the refactor and attack execution)

Human game 4 (docs/PROGRESS.md): Gary-T's buildings stood spread out around the main, it ran out
of room, and some builds likely failed. Today's placement (gary/agent/executor.py `spot`,
`_base_spot`, `spot_near`) takes the free tile closest to a point on the side away from the
minerals, with a one-tile gap all round; gary/agent/simcity.py only places the buildings of a
wall. Gary-Z has its own (`gary/bots/gary_z.py` `_building_spot`), which the refactor retires.

Placement belongs to the race plug (the executor, above), behind one shared interface:
`spot(kind, base, builder) -> tile | None`, where the shared part keeps the rules every race has
(clear of the mineral and gas paths, a lane kept from production to the base's way out, the
town-hall and add-on tiles reserved, the map edge, fogged creep) and the race hook adds its own:

- **Terran:** Supply Depots packed in blocks against the main's edges and cliffs; Barracks,
  Factories and Starports in columns with the two add-on tiles kept free to the right and a
  walkable row between columns; turrets and bunkers where the fight layer asks.
- **Zerg:** on creep only, off the drones' paths, the creep colonies at the front of the creep.
- **Protoss:** inside a pylon's power field, pylons placed to power the next buildings, the gaps
  a probe and a dragoon walk through.

The layouts are not to be invented: the pros' replays give each building's position relative to
its base (the town hall, the minerals, the way out) per map and start, so a placement prior can be
read or learned from them, as the walls were (gary/walls.py, PROGRESS.md "Which maps and starts
pros wall"). Measured before and after in the league like the rest: failed builds (`bad_spots`
entries), how far the buildings spread from their town hall against the pros', how many fit in
the main before it is full, and whether a ground path from each production building to the
base's way out stays open.

## Tests across matchups

**Done 2026-10-20.** tests/test_matchups.py plays each matchup 3 minutes through eval/match.py and prints the
table of what plays: TvZ, ZvZ and TvT ok, the Protoss matchups skipped ("no Protoss executor yet"); a Gary
that can't be built for a matchup (a model kind missing) is a skip with that reason. tests/test_parity.py
hashes a ZvZ and a TvT game besides the three TvZ ones. TvT was played for the first time: the default
Terran manifest is TvZ's, so manifests got a `vs` section (overrides by the enemy's race, applied once the
matchup is known from the lobby); against a Terran or a Protoss the default Gary-T plays the normal posture,
no scripted rush builds, no answers to a Zerg's plan, no natural bunker, no TvZ selector table, and the fight
layer's rules (the role model is TvZ's), with the TvT macro, army and commit models. In the first game both
sides had seen an early pool and built ramp bunkers.

tests/test_parity.py (Phase 0) and tests/test_smoke_game.py are parametrized over the matchups
that can be played: the game's races from the manifests, the map one of the ladder maps
(`eval/league.py ladder_maps`), a few minutes headless. In Phase 0 that is TvZ (Gary-T against
Gary-Z); after Phase 1, ZvZ (two Zerg executors) and TvT (two Terran executors with the TvT macro,
army and commit models and rules for the fight layer); after Phase 4, the Protoss matchups. A
matchup without a model for some kind is a skipped test with the missing kind in its reason, not a
failure, so the table of what plays is readable from the test output. A short league round per
matchup (eval/match.py, 8 games) is the manual gate for Phases 2 and 3, judged as the current
rounds are: paired games on the ladder maps, the gap score against modern pros, who is ahead at
the limit.

## Rules for the build

- One phase per pull request-sized commit series; each phase ends green on its gate before the
  next starts. The gates are the tests: add the byte-identical check as a test
  (tests/test_parity.py: a fixed manifest, map and seed, the POV log's hash before and after;
  the "before" hash recorded in the test when Phase 0 starts). The current defaults are the
  shared macro models, no bunker at the natural (option nonatbunker) and a depot ahead of
  production (option depots) for Gary-T, and Gary-Z with its commitment floor off: the parity
  games use exactly the default manifests of the day the phase starts.
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
