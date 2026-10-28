# TODO: open work

Open work is tracked in the [GitHub issues](https://github.com/vibrunazo/gary/issues); this file is their index,
grouped by area, newest thinking first (reconciled with the code on 2026-10-25). When an item is done its issue is
closed and its line leaves this file; docs/PROGRESS.md keeps what was done and how it measured.

## Learning (the self-play program, docs/SELF_PLAY.md)

- [#47](https://github.com/vibrunazo/gary/issues/47) Pipe checks: do the learned layers' decisions reach the game, and do
  their inputs in play match their training data? (placement fixed, widespot; the macro's enemy history and the
  bunkered units, seenall and armyseen; tests/test_inputs.py checks the inputs now; left: the decisions so far, off by
  up to 3 Barracks on the Terran)
- [#50](https://github.com/vibrunazo/gary/issues/50) Play and the replays count units differently: workers in gas buildings,
  turrets, the in-building queue's units, undetected cloaked units (the dataset's)
- [#28](https://github.com/vibrunazo/gary/issues/28) Commit decision: a learning signal from locked paired branches
  (the signal weak: scored by the critic too, nothing beats the deployed model's preference in 941 pairs; next, pairs
  again on gt0.14dev's fixed inputs, several dice per arm, or a decision with longer consequences)
- [#51](https://github.com/vibrunazo/gary/issues/51) Gary-Z's builds and burrows: the lurkers fixed (they never came up again);
  its builds held production (fixed, gz0.9dev); left: 99-233 burrows still refused a game, the late game of a
  losing Zerg (builds without drones)
- [#48](https://github.com/vibrunazo/gary/issues/48) The Zerg's macro model counts the units in its eggs twice; the fix
  (option eggs) lost 96 paired games: no rule leans on it; the right counts move the model off zerglings to
  Hydralisks and it slows (its counts a momentum cue): eggs with the macro trained on Gary-Z's own states next
- [#52](https://github.com/vibrunazo/gary/issues/52) The role models' inputs: Gary-Z's enemy HP fixed (gz0.5dev), the Terran's
  extras 3-5 in a rebuilt dataset and model (runs/roles/20261026-090057, not deployed: the next roles run's start)
- [#53](https://github.com/vibrunazo/gary/issues/53) No fight-scenario reward ranks the roles checkpoints as games do: a reward
  that counts what fighting costs the rest of the game, or 96 games
- [#54](https://github.com/vibrunazo/gary/issues/54) Build styles by name for each side, every game labelled (done: TvZ);
  the Terran and Zerg macro models trained with every game's style, judged per style (gt0.19dev, gz0.7dev); the Zerg's
  roll by the modern shares; other matchups
- [#57](https://github.com/vibrunazo/gary/issues/57) Gary-Z banks the gas it mines (920 at 10-15 minutes, pros 499)
- [#58](https://github.com/vibrunazo/gary/issues/58) Gary-Z's builds per style: style 2 played as itself 31%, the third Hatchery
- [#59](https://github.com/vibrunazo/gary/issues/59) Fold the dev manifests' execution fixes (parkmany, gaswaits, the gas head)
  into gt1, gt1dev and gz1
- [#55](https://github.com/vibrunazo/gary/issues/55) The opening economy: new workers sent at once, the first worker first,
  workers on gas the macro model's decision (gastarget, gt0.20dev: style 2's natural 3.6 minutes, pros 3.2; the Zerg's,
  gz0.7dev); left: about 100 minerals mined behind the pros by 2:00 (the split, doubled fields, idle builders; Gary-Z's
  first Overlord at 58 s, pros 53)
- [#29](https://github.com/vibrunazo/gary/issues/29) Why army dataset v5's commit models trail the deployed one in games
  (not the bunker counts: with armyseen v5 still trails, -0.039 over 96)
- [#3](https://github.com/vibrunazo/gary/issues/3) Close out won games (the hunt imitated: its labels on the deployed
  commit model next)
- [#11](https://github.com/vibrunazo/gary/issues/11) Learn from game outcomes
- [#12](https://github.com/vibrunazo/gary/issues/12) Macro model with game history (a sequence over recent events)
- [#39](https://github.com/vibrunazo/gary/issues/39) Anticipation: target inference for enemy groups
- [#40](https://github.com/vibrunazo/gary/issues/40) Opponent model: a belief over the enemy's plan
- [#42](https://github.com/vibrunazo/gary/issues/42) The army model's fight estimate from Gary's own fights
- [#45](https://github.com/vibrunazo/gary/issues/45) The macro answers what the scout sees, without the scripted answers
- [#35](https://github.com/vibrunazo/gary/issues/35) TvT Gary-T: its own fight, belief, plan and engage models
- [#34](https://github.com/vibrunazo/gary/issues/34) More modern replays

## Gameplay

- [#2](https://github.com/vibrunazo/gary/issues/2) Hold early zergling harass and all-ins (and how much to commit to defense)
- [#4](https://github.com/vibrunazo/gary/issues/4) Micro: spreading, sieging, spells as learned decisions
- [#26](https://github.com/vibrunazo/gary/issues/26) Strategy layer: pick among proven builds and follow the timeline
  (and rebuild the selector's table from league games)
- [#37](https://github.com/vibrunazo/gary/issues/37) Human game 4's failure list
- [#43](https://github.com/vibrunazo/gary/issues/43) Scattered fighters back into the main group
- [#38](https://github.com/vibrunazo/gary/issues/38) Map mechanics on the current ladder maps: neutral units and per-map priors
- [#16](https://github.com/vibrunazo/gary/issues/16) Learn building placement and scouting from replays
- [#25](https://github.com/vibrunazo/gary/issues/25) Walls: copy the pros' walls per map and start location
- [#23](https://github.com/vibrunazo/gary/issues/23) Map analysis: split maps at their chokes
- [#36](https://github.com/vibrunazo/gary/issues/36) Protoss: the executor (refactor Phase 4) begun, `gp1` plays every matchup
  with its role model and styles, plays its styles as the pros do (PV161a, PZ161a); next the dev builds' fights in PvT and
  PvZ by their matchups' role models (gt0.21dev, gz0.10dev); next PvP
- [#61](https://github.com/vibrunazo/gary/issues/61) The Terran's modes in PvT (gt0.22dev siege and lay mines): held as long as the
  pros hold them, the pros' mines outside fights, Spider Mines four minutes late
- [#62](https://github.com/vibrunazo/gary/issues/62) Type gg and leave when the critic's win odds stay under 20% (default;
  an option plays on; dev builds play to the end and record their would-gg moments); first a critic per matchup and its
  calibration
- Later, as supervised skills: [#21](https://github.com/vibrunazo/gary/issues/21) worker drilling,
  [#22](https://github.com/vibrunazo/gary/issues/22) bunker rush

## Evaluation and tools

- [#31](https://github.com/vibrunazo/gary/issues/31) Mirror health rounds and the Protoss macro models (rounds played, no errors;
  left: the death records' slot in mirrors; the Protoss PvT/PvZ misses are Zealot/Dragoon and reactive buildings)
- [#32](https://github.com/vibrunazo/gary/issues/32) The gap score disagrees with results
- [#44](https://github.com/vibrunazo/gary/issues/44) Engine error at the map's edge
- [#15](https://github.com/vibrunazo/gary/issues/15) Fit human-interface profiles from replays
- [#41](https://github.com/vibrunazo/gary/issues/41) Explainability: a narrator over the decision records
- [#17](https://github.com/vibrunazo/gary/issues/17) Viewer: the action overlay in the interactive window
- When full-game self-play needs the speed: [#19](https://github.com/vibrunazo/gary/issues/19) vectorized env,
  [#20](https://github.com/vibrunazo/gary/issues/20) a C++ per-frame core

## The Remastered bridge

- [#24](https://github.com/vibrunazo/gary/issues/24) Adapt to the layer refactor (`ScrGame.terrain`, `type_select`; a live
  game of the current gt1)
