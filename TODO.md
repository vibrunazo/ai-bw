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
  left: 99-233 burrows still refused a game, the late game of a losing Zerg (builds without drones), idle larvae
- [#48](https://github.com/vibrunazo/gary/issues/48) The Zerg's macro model counts the units in its eggs twice; the fix
  (option eggs) lost 96 paired games: no rule leans on it; the right counts move the model off zerglings to
  Hydralisks and it slows (its counts a momentum cue): eggs with the macro trained on Gary-Z's own states next
- [#49](https://github.com/vibrunazo/gary/issues/49) The critic's net-value heads rank moments well but are
  overconfident (180 s: correlation +0.37-0.49 from 6:00, R² near 0): early stopping, per-window report
- [#52](https://github.com/vibrunazo/gary/issues/52) The role models' inputs in play: Gary-Z saw every enemy at 0 HP; the
  Terran's extras 3-5 never trained (option dataextras)
- [#53](https://github.com/vibrunazo/gary/issues/53) No fight-scenario reward ranks the roles checkpoints as games do: a reward
  that counts what fighting costs the rest of the game, or 96 games
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
- [#36](https://github.com/vibrunazo/gary/issues/36) Protoss: the executor (refactor Phase 4), after TvZ self-play shows a gain
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
