# Gary — Progress

Where each Gary version stands against [ARCHITECTURE.md](ARCHITECTURE.md). The architecture
says what Gary should become; this file says what exists. Only the components that matter at the
current stage are listed; later ones (narrator, broadcaster, language models...) join when work
on them starts.

Legend: **—** not started · **○** scripted stand-in · **◐** partial · **●** working (for now)

## Roadmap phase

**Now: P4, the self-play skeleton** ([ARCHITECTURE.md §12](ARCHITECTURE.md#12-roadmap); the plan:
[SELF_PLAY.md](SELF_PLAY.md)), with P2b (first playable) reached early: Gary plays whole TvZ games
against people through the Remastered bridge, which re-simulates in OpenBW to the same state. The
phases didn't go strictly in order: the macro and army are learned by imitation (P3-style pieces)
rather than scripted, because the replay pipeline made that the quicker way to a playable Gary.
Since 2026-10-15 every piece of the self-play loop is being built crude before any is polished: a
learned Zerg opponent (Gary-Z), a humanlikeness classifier as the anchor the league isn't trained
on, the league itself, and RL on paired branches (a moment saved and played twice, both bots live).
Where it stands (2026-10-24): one Gary for every race (the refactor, 2026-10-20), playing Terran as `gt1`
(gt0.10) and Zerg as `gz1` (gz0.2); learning is trained and judged in `gt1dev` (gt0.11dev) and `gz1dev`
(gz0.3dev), which run without the stop-gaps over the layers being learned (AGENTS.md), and gt1dev against
gz1dev is the TvZ self-play pairing, balanced (each side wins or leads in 38% of 96 games). The one RL run
that held is on the Terran's fight roles (rl1, in gt1dev); the army's commit decision is the target now,
on paired branches with the forced arm locked at the pros' hazard, learned offline: at chance on 1,004
pairs so far (docs/SELF_PLAY.md).

| Phase | Status | What exists | What's missing for its exit criteria |
|---|---|---|---|
| P0 Infra | ◐ | OpenBW env, human interface v1, POV and decision logs, eval against scripted Zergs, replayed pros and pro scenarios; ~150k 45-second scenario plays per hour; full games 14 at a time (one torch thread each: 2.7x faster); games deterministic per seed, replayed to any moment to branch, and branched live with both bots (`eval/pairs.py`: checked every game that a sampled branch replays the same; commit pairs with the forced arm locked, 336 of 336 checks identical); decision records for every layer of both sides (`eval/why.py`); a run monitor page (`eval/monitor.py`); the Remastered bridge 1:1 with OpenBW (`adapters/scr_bridge/tools/parity.py`) | The formal trace schema; classic bots to play against; (a crash of the native env inside a branch, 1 game in 144: most likely memory starved by orphaned workers of a stopped job, since killed) |
| P1 Data | ◐ | 92k replays indexed (18.9k of the Remastered era, ~3,000 a matchup), resim with Remastered support, fog-of-war views; macro, army and fight sets for all six matchups (docs/RACES.md) | Camera logger, camera inference, fitting human profiles from data (select-to-order times fitted; click scatter and camera still placeholders) |
| P1b Taxonomy | ◐ | TvZ build clusters with readable rules (`taxonomy/discover.py`) | A reviewed `taxonomy/tvz/v1.yaml`, stability and gold-set reports |
| P1c Strategy stats | ◐ (started) | The strategy selector's table: each build style's results per map in Gary's own games (`eval/selector_table.py`, v0.19) | Win-rate tables from pro replays, the feature library, guide claims |
| P2 Micro | ◐ | Fight, command and role models from pro skirmishes; home-defense scenarios and 120 s episodes from pro replays; roles by hand rules in contact, a role model learned from the pros' units before contact (v0.29); stim learned from the pros (v0.27); RL on the roles four times, each ending at its start (three for the Terran, one for Gary-Z on paired episodes): roles stay at imitation and rules; choke blocks as a skill | Responses within the human band: the humanlikeness classifier's fight rows (lone outnumbered units never move away, pros 11-18%; marines rarely stimmed at the sight of mutalisks; no moving between shots) |
| P2b First playable | ◐ | Full games against people, offline, via the bridge (two human games so far, both lost by Gary; their failure lists shaped v0.26-v0.27 and the classifier); learned macro steered by build style, the style chosen per map (v0.19) | Build timings within tolerance (≥18/20 unharassed runs); supply blocks and idle production under harass within the human band |
| P2c Debug viewer | ● | POV viewer with cursor, clicks and every action in words; videos | Trace overlays beyond actions |
| P3 BC full game | ◐ | Separate imitation models for macro, army movement and fights, for the Terran and (Gary-Z) the Zerg; a role model learned before contact as well as in it, each side; belief models of the enemy army, each side (frozen; off by 11.7 supply on league games against 5.9 on pros: Gary's armies aren't the pros'); a value network for the Terran (`gary/policy/value.py`); the humanlikeness classifier and its pro-distance table (`eval/humanlike.py`) | One policy with intent and belief heads, the Opponent Model; Gary closer to the pros in the classifier's table (it tells them apart at AUC 0.99+) |
| **P4 RL** | **◐ (current)** | RL on fights (scenarios, episodes): the first run that held (rl1, the Terran's roles, in gt1dev); macro from branches (nothing: a decision is worth sd 130 against chaos's 512); per game: the strategy selector (v0.19: losses 57 → 30 of 333), the early-defense switch; the self-play skeleton: both sides on the one agent, the league (`eval/league.py`: paired 96-game rounds, the gap to modern pros with its spread), the dev manifests without stop-gaps, the TvZ pairing balanced (38% each side); the army's commit decision on locked paired branches, learned offline (`train/commit_pairs.py`: at chance on 1,004 pairs); a stop-gap imitated from its advice (the hunt: `huntadvice`, `--rule-labels`) | A learned layer beating its imitation start in 96 paired games on the commit decision; Gary-Z's commit pairs; Elo gain at equal humanlikeness |
| P5 Explain, P5b Broadcast | — | (The POV viewer's action overlay is a start) | |
| P6 Deploy | ◐ | Remastered bridge for offline games, 1:1 with OpenBW (a LAN replay re-simulates to the same state) | Pro blind-test sessions; the bridge's known bugs (a supply depot placed a tile off at expansions) |

## Components by version

| Component | What it is | Now (gt1 gt0.10, gt1dev gt0.11dev, gz1 gz0.2, gz1dev gz0.3dev) | v0.29 | v0.28 | v0.27 | v0.26 | v0.25 | v0.24 | v0.23 | v0.22 | v0.21 | v0.20 | v0.19 | v0.18 | v0.17 | v0.16 | v0.15 | v0.14 | v0.13 | v0.12 | v0.11 | v0.10 | v0.9 | v0.8 | v0.7 | v0.6 | v0.5 | v0.4 | v0.3 | v0.2 | v0.1 | v0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Game: headless | OpenBW game Gary plays in for development and tests (`env/`, `gary/env.py`) | Melee games on any map, network-style command delay, replays saved | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● |
| Game: live Remastered | Gary in a real StarCraft: Remastered client over LAN (`adapters/scr_bridge`) | Plays, 1:1 with OpenBW (a LAN game re-simulates to the same state); known gaps in the bridge's map checks and hotkeys (its README) | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | — |
| Human interface | The only path to the game: screen clicks, camera, fog, reaction delay, APM (§6.3) | Clicks, drag boxes, camera hotkeys, fog, reaction delay, APM budget; pro-speed hands fitted to pro replays' select-to-order times yet | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● |
| Perception and memory | What Gary knows: visible units, remembered enemy units (§6.9, §6.10) | Fog and hidden enemy HP; remembers enemy buildings and the enemy army seen in the last 3 min; map analysis (`gary/terrain.py`): main ramp, natural, natural choke; walls (`gary/walls.py`): which units get between buildings and cliffs, and where the openings are (not used in play yet) | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ |
| Macro decisions | What to build, train, research, and when | Learned from the pros (the shared Terran model over every matchup's games, both eras, the matchup, year and map as inputs); the model's units queued in the building two deep (gt0.9); stop-gaps on top in gt1 (the pros' caps, the SCV cap, the bank escape, the under-half rule, summed odds for one-offs and expansions, answers to lurkers and mutalisks, the early-pool posture: AGENTS.md's register); gt1dev without the caps, SCV cap and bank escape, drawing from the model's odds (dice) | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ○ | — |
| Macro executor | Carries out macro decisions: placement, production, gas, mining (§5) | Shared Executor with a Terran and a Zerg plug (refactor Phase 1): any Terran building, unit, add-on (placed where it fits, gt0.6), research, upgrade, expansion; trains into a busy building's queue (gt0.9); Zerg larva and building morphs, drone builds, overlords ahead (gz1) | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| Army decisions | Where armies go, move vs attack, whether a fight is worth taking | Clusters with commitments (keep, engage, move), learned from the pros, drawn from their odds; the static guard a nudge on engages into static defense (gt0.10); the hunt far ahead and squad search, the overlord hunt (stop-gaps, off in gt1dev, where the hunt runs in advice mode); control groups by type in gt1dev and gz1 | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ○ | ○ | — |
| Micro | Unit control in fights: stim, spread, siege, focus fire, spells | A role per unit every half second: by hand rules in contact in gt1 (RoleAhead), by the model learned from the pros' units in gt1dev (attended units, the again decision, several orders a turn, its RL run rl1); before contact the role model in both; stim learned from the pros; marines loaded into bunkers by rule | ◐ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | — | — | — | — |
| Scouting | Finding out what the opponent does | One worker at 9 SCVs to the enemy start locations | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | — | — | — |
| Reflexes | Quick reactions outside the plan (worker defense, repair, retreat) | Scripted worker pull, 3 workers per attacker, not against flyers or more than 8 zerg supply (v0.26); unfinished buildings get a new SCV; SCVs repair a bunker under attack, and 3 a threatened turret (v0.27); burrowed lurkers near Gary's fighters scanned (v0.22) | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | — | — | — |
| Attention arbiter | One camera and one APM budget shared by all tasks (§6.7) | One task at a time in a fixed priority order (Gary-Z's turns as its rotate option); leases enforced by priority since v0.10 | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| Strategy layer | Picks the build and switches plans from what it sees (§6.8) | The build style per map from counted game outcomes (`gary/agent/selector.py`, v0.19); `--style` overrides it; an early pool switches Gary to a scripted one-base defense until it's held; an alarm from nothing seen is called off until 4:00 (v0.18); the early-pool defense from 2:30 whatever is seen, unless the rules switched first (v0.22) | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | — | — |
| Opponent model | Beliefs about the opponent's build and army | A belief model of the enemy army from what the player saw, each side (`train/belief.py`: off by 5.9-5.3 supply on held-out pros); not used in play by default | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| Replay pipeline | Inventory, build orders, resim, desync checks (§7.1–7.2) | 92k replays indexed (18.9k Remastered era); TvZ, ZvZ, TvT re-simulated, Protoss mostly its modern games; fog-of-war views | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● |
| Build taxonomy | Build styles discovered from replays (§7.3) | TvZ clusters with readable rules; style labels for ~45% of players | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● |
| Training sets | Data for the learned models | Macro, army (v5 for TvZ: known enemy buildings, cells unseen, static defense) and fight sets for all six matchups; the response sets (`ingest/response_dataset.py`): the pros' units' roles before contact and in it, each side | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | — | — |
| Learning beyond imitation | Outcome-weighted learning, reinforcement learning | A value network (reads who is ahead); the roles RL (rl1, the first run that held, in gt1dev); the army's commit decision on locked paired branches learned offline (`train/commit_pairs.py`: at chance so far); a stop-gap's advice as labels (`--rule-labels`); the rules in docs/SELF_PLAY.md | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | — | — | — |
| POV viewer | Watch any game from Gary's screen (`viewer/`) | Cursor, clicks, every action in words; Remastered replays; videos | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | — |
| Trace | Gary's decisions and reasons, recorded (§6.5) | Decision log with model probabilities and fight estimates; POV action log; the learned layers' decisions per league game (`gary/decision_log.py`: inputs, the drawn decision, its log-probability) | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | — |
| Evaluation | Matches against opponents, many seeds, a scoreboard (§9) | `eval/league.py`: Terran against Zerg, paired 96-game rounds on the ladder maps; `eval/gap_windows.py`: the gap to modern pros per window with a bootstrap spread; `eval/attacks.py`: the attacks' trade and what began each; `eval/humanlike.py`; `eval/scenarios.py`; `eval/why.py` on every game's records | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ○ | ○ | — |
| Matchups and races | Gary plays every race and matchup by design (ARCHITECTURE.md §1); TvZ was the first | One Gary for every race (refactor, 2026-10-20): Terran (gt1, gt1dev) and Zerg (gz1, gz1dev) in TvZ; ZvZ and TvT play (tests/test_matchups.py); models for all nine matchup-races; Protoss data and models, no executor yet (Phase 4); the audit: docs/RACES.md | T (+Z) | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T |

Since the refactor (2026-10-20) a version is a manifest: `gt1` (gt0.x) plays Terran, `gt1dev` (gt0.xdev) is
where learning is judged, `gz1` (gz0.x) and `gz1dev` (gz0.xdev) the same for the Zerg. The per-version columns
above stop at v0.29; each later version's change and the rounds that adopted it are in the comment above its
definition in gary/agent/manifests.py and in the entries of "Next" below:

- **gt0.1-gt0.5** (2026-10-20/22): v0.29 with the options adopted round by round (the commit model, its draws and
  floor, summed odds, no natural bunker, depots ahead, the overlord hunt, the hunt far ahead and squad search,
  the static guard, the fight roles from whole games, lone units by the roles model). **gt0.6**: add-ons placed
  where they fit. **gt0.7**: keep_busy gone, the model's waiting units queued. **gt0.8**: depots ahead gone.
  **gt0.9**: the model's units trained into a busy building's queue. **gt0.10**: the static guard a nudge.
- **gt1dev**: gt1 with the role model in contact, control groups and rl1 (gt0.6dev), gt1's versions as they came,
  without the stop-gaps over its learned layers and the hunt in advice mode (gt0.10dev), drawing its macro from
  the model's odds (gt0.11dev).
- **gz0.1-gz0.2** (2026-10-20/22): the Zerg on the shared agent, then its army on control groups by type.
  **gz1dev**: gz1 without its stop-gaps (gz0.2dev), its overlord history its own (gz0.3dev).

## Versions

### Next: the self-play skeleton (started)

Every piece of the self-play loop, crude, before any of it is polished (docs/SELF_PLAY.md has the
plan and the full log).

- **The bridge, 1:1 with OpenBW:** a LAN game in Remastered re-simulates in OpenBW to the same
  state (`adapters/scr_bridge/tools/parity.py`), so games against people are as informative as
  headless ones.
- **Gary-Z v0, a learned Zerg opponent (`gary/bots/gary_z.py`):** Zerg v3's hands with the Zerg
  macro model (steered by a pro build plan rolled per game: 11 plans from the pros' openings) and
  the Zerg army model. Its execution leaks, found with probes and fixed one by one: decisions at the
  pros' tempo (a few-percent Creep Colony had come 100 times a minute), the hands shared by streaks
  (a camera tug-of-war froze the macro for minutes), gas drones re-staffed after a float, groups by
  supply and by type clicks, no earliest attack (its army model was never asked before 7:00).
  Acceptance (48 games, three maps, 20 minutes, against v0.27): Gary-T won 20, lost 1, 27 undecided;
  plans spread like the pros'; the same seed gives the same game. Its macro matches the pros' to
  8:00 (drones 24, three hatcheries); its army is half theirs, and it fights with v3's scripted micro.
- **The humanlikeness classifier (`eval/humanlike.py`):** one player's side, per minute, from the
  replays as that player saw them, against 300 pro games on the same maps: the economy, the army's
  composition and placement, and the moments before contact (what its units do when enemy
  fighters come into view: the user's sense of Gary as "AFK" was anticipation, not reaction time).
  It tells Gary from the pros at AUC 0.99+ in every minute, so the work list is its per-feature gap
  table, and a gap score that moves. Gary-T's biggest gaps: lone outnumbered units never move away
  (pros 11-18%), they attack (67-86%, pros 39-52%), marines rarely stimmed at the sight of mutalisks
  (14%, pros 40%), no moving between shots (pros 4-10%), bunkers and turrets the pros don't build.
  Gary-Z's: an army bunched, standing, at home. Games against the replayed-pro fixture can't be
  measured (it creates its units in the engine).
- **v0.28:** v0.27 whose army layer doesn't hold plain moves on the fight estimate's noise (it held
  ~90% of its decisions at 5-6 minutes, 92% of them within 2 points). Against the replayed pros (24
  games): won 5, lost 0 (v0.27: 2, 0).
- **v0.29:** v0.28 with a role model learned from 600 pro games' units before contact as well as in
  it, with a where head (each unit's move): the hand rules in contact, the model before it. On 226 pro
  home defenses: a tie with v0.28 (-2 +- 6 net value); its roles everywhere lost 52 (the rules were
  tuned on those scenarios for twenty versions). The neural role policy is the league's learner.
- **Gary-Z's fight roles:** the same role model, learned from 600 pro Zerg games (0.8M labels before
  contact), carried out with Gary-Z's hands (`gary/bots/zerg_roles.py`) in place of v3's scripted
  micro. Results and army rows unchanged (24 games): a few orders a minute in fights can't move an
  army that stands bunched at home; that was the army layer above.
- **The league (`eval/league.py`):** rounds of Gary-T against Gary-Z in parallel headless games; per
  game the result, both sides' gap scores and the learned layers' decision log (inputs, the drawn
  decision, its log-probability: what RL learns from). Gary-Z's army layer now draws each group's
  destination from the army model's odds; it had taken the likeliest cell for every group, nearly
  always home. Round r1 (v0.29, 24 games, 20 minutes): Gary-T won 6, lost 0, 18 undecided; Gary-Z's
  army at 8-10 minutes at home 56% (79% before; pros 35%), spread 15 tiles (6; 21), standing 43%
  (67%; 42%), 6.6 orders per fighter (4.2; 6.4). Fight episodes train the fight roles; the army layer
  learns from the league's outcomes.
- **The first RL update (`train/league_rl.py`):** both sides' learned layers log their decisions in
  league games (the Terran's army now draws too, option `draw`); each decision is scored from the
  replay (fight roles: the fight's net value over 45 s; army layers: the side's army-supply share 45 s
  later), and each layer takes one PPO step with a KL leash to its imitation model. Round r3 with the
  updated models against round r2, same seeds: no change past noise (Gary-T won 9 and 8, lost 0; mean
  outcomes and gap scores within their spread), as expected of one small step on 24 games. The loop
  runs end to end: play, log, score, update, play.
- **The first nights (`eval/league_loop.py`, `eval/pairs.py`):** one RL target, Gary-Z's fight roles,
  on paired fight episodes (a fight saved and played twice from the same moment, both bots live); the
  army layers frozen at imitation. Each round: fresh seeds, a candidate trained from the pairs (KL-leashed
  to imitation) and kept only if its A/B pairs beat the incumbent and its games aren't less pro-like.
  Checked before the runs: the same seed replays identically, an arm replays identically from a saved
  moment, the main game is unchanged by the branches. Twelve rounds, 288 games: no candidate kept (the
  A/B pairs within noise; bigger steps drifted from the pros, which the gap test caught); no learned
  retreat. So, Gary-Z's army layer: most of its ground army was never in a group the army model moved,
  and v3's lurker routine starved the rest for minutes. Fixed: the army model moves the whole army by
  clusters. Gary-Z's army now spends 47% of 5-7 minutes at home (78% before, pros 38%), its gap score
  falls to 0.34 there (0.45), the decided share rises to 38% (17%); it's more pro-like but not stronger
  (Gary-T won 9 of 24, 4 before). A Zerg belief model, frozen (pro games only).
- **Night n3, with clusters:** more fights, more pairs; the gate kept all five candidates (each A/B
  within noise, all five above zero, their games more pro-like each round). The final model against
  the imitation model on fresh seeds: +0.059 +- 0.054 net value a fight (not settled), Gary-Z's gap
  score 0.40-0.46 against 0.51-0.55 after 8 minutes. A second batch of fresh seeds: pooled over 373
  paired fights +0.020 +- 0.046, no gain; the retreat share equal. Three nights, 432 games: no
  measurable gain from fight-role RL, no learned retreat. The gate is now stricter. Gary-Z still wins
  no game.
- **Fixes from Fable's review:** the Zerg roles had ordered only the most urgent role group each half
  second and logged every unit's draw as done (the nights trained on roles that never happened): now
  every group is queued and only executed decisions are logged. Each league game replays a sample of
  its branches, and a round whose sample disagrees is voided (memory corruption gives wrong outcomes
  without crashing).
- **Why Gary-Z loses (`eval/zerg_losses.py`, against 60 pro Zergs):** its fighters die in the field
  at odds of 1 to 3 (pros about even), with 15% of its army there (pros 27%): clusters trickle in alone;
  its army is half the pros' and nearly all zerglings, no mutalisks at 8-10 minutes (pros 6-8, killing 9
  SCVs a game), and it barely attacks. The roles were never the bottleneck.
- **One army decision against the game's chaos (`eval/army_effect.py`):** at 77 moments of Gary-T's
  games, its army attacking against holding, branched with the hands redrawn: the decision's effect
  over 90 s varies three times as much as the chaos does (sd 650 against 210 value), past two standard
  errors at 62% of moments, attacking better at 40%. Army RL on paired branches can start with the
  score as its outcome, without a critic first.
- **Gary-Z's army and economy, three rounds on the same seeds (c1-c3):** its drones went back to gas
  only below 150 gas, so it mined none from 4:00 to 8:00 with ~160 banked: now below 300 (gas at 8-10
  minutes 250-400, was 80-130), and no floating while a Spire stands and its first mutalisks aren't made
  (pros save ~700 gas for them); a cluster under half the main one's size joins it. About half its
  games now make 10-20 mutalisks (none before), but they kill 0.3 SCVs a game against Gary-T's 9-11
  turrets (pros 9 SCVs); its fighters die at field odds of 0.38 (0.33; pros 0.71), with 16% of its army
  there (15%; pros 27%); Gary-T won 9, 7, 9 of 24. Gary-Z's army stays half the pros' (13 supply at
  10-12 minutes, pros 26-33), its minerals short when the Spire finishes: winning needs its macro, not
  more execution fixes.
- **RL on Gary-T's army layer, loop a1 (5 rounds, 120 games):** paired 90 s branches with every army
  decision of the window drawn apart: one candidate kept (A/B +0.50 +- 0.41), three rejected; about +-5
  per pair, the credit spread over dozens of decisions. Next loop: one decision branched, the rest with
  shared dice.
- **Loop a2, one army decision branched:** two candidates kept on a standard error, but the final model
  against the imitation one on two fresh batches: -0.18 +- 0.30 (262 pairs), no gain. At this scale
  (a few hundred decisions a round, +-5 noise a pair) the army layer doesn't learn either; the effect
  measurement's contrast (the whole army attacking or holding) is far bigger than the model's own
  alternatives.
- **The roles line closed:** the last fight-role run (decisions logged as carried out, a two-standard-
  error gate) kept nothing in six rounds; five tries at RL on per-unit roles have ended at the start.
- **Gary-Z's macro audit:** its decisions follow the macro model's odds (not the decoder); the model in
  Gary-Z's states gives mutalisks 3% at 6-8 minutes where pros produce 22%, and buildings 27-39% where
  pros put 17-18% (half creep colonies): Gary-Z's states look unlike the pros' to the model.
- **Attack or hold, learned (`train/engage.py`):** from 1.3M pro army orders, whether the group was sent
  into the enemy: held-out AUC 0.80. In play (option `engage`) it draws attack or hold and the army
  model picks the destination on that side; RL on paired branches forcing attack in one arm and hold in
  the other at one decision, the contrast measured at three times the chaos.
- **Placement check before the engage loop:** Gary-T's army is at home 20-29% of 8-10 minutes against the
  pros' 60% (older than drawing; partly hunting a passive Gary-Z). Calibrated, engage still goes in too
  often: its odds are per pro order, and Gary asks far more often than pros order; the army layer needs
  a when-to-order estimate. RL against a Gary-Z that never wins would learn to exploit it: the engage
  loop waits.
- **What drives Gary-Z's static defense:** the Terran army it sees. With it removed from the macro model's
  inputs, creep colonies fall from 7% to 2% of its odds at 8-10 minutes and drones rise from 22% to
  31%: the model answers visible pressure as pros do, and the pressure is Gary-T's army roaming (at home
  20-29% against the pros' 60%). The sides' humanlikeness is coupled.
- **Gary-T's army moves paced at the pros' rate** (1-6 moves of 2+ cells a minute; it made 9-19): its
  army stands as pros' do (0.40 of 8-10 minutes, pros 0.39), at home a little more (0.33, pros 0.60),
  its gap score lower at every minute range. Gary-Z, under less pressure, drones closer to the pros
  (22 at 8 minutes, pros 25) and has 22 army supply at 12 minutes (13 before, pros 33); still no win:
  its gas now banks unspent and minerals are short.
- **Gary-Z's mutalisks were a decoder after all:** in c5 the Spire finished in most games, yet 12 of 24
  decided no mutalisk with 600-800 gas banked. Drawing from the full odds at every decision, Gary-Z took a
  Hydralisk Den at 1-6% in 21 of 24 games, and the macro model, reading that history as a lurker
  player's, gave mutalisks 0.8% (16% with the history cleared). A building that opens a tech path now
  needs 10%+ odds. Same seeds (c6): Dens in 9 games (21), spire games deciding mutalisks 19 of 21 (11 of
  23), up to 59 a game; Gary-T won 11 of 24 (5), Gary-Z none.
- **Decision records (before the next RL loop):** every why-question so far (the sunkens, the roaming
  army, the dropped orders) cost a script re-simulating replays. Now every layer's decisions of a game,
  both sides, are saved next to its replay (`<game>.records.jsonl`): what each saw, its top alternatives,
  the choice, who chose and what intervened, each order's fate (sent, refused, retried, dropped, done),
  and who had the hands; `python -m eval.why <game> --minute M` prints a window in time order. The game
  is unchanged by them (seed 3005 replays to the same log). Watching Gary-Z's screen from 4 to 10
  minutes first, the first thing wrong was four Creep Colonies decided in 20 s at 3-6% odds while it had
  19 drones (pros 22 at 6 minutes). The records answered it in one query: after a parked decision (Lurker
  Aspect, waiting for money) Gary-Z treated its larvae as taken, so Drones were set aside as "no larva
  free" with 5 larvae standing and only options over 5% stayed; and three of the four colonies were
  never built, the job counting a colony 2 tiles away as its own.
- **Two rounds read from the records (c7, c8, c6's seeds):** with the larva bug fixed (c7) Gary-Z drones
  past the pros (27.5 at 8 minutes, pros 25; c5 22) and decides a third fewer creep colonies, but its
  army doesn't grow: at 11:00 one game had 1,758 minerals, 39 drones, 3 hatcheries and no larvae. Its
  records showed why in one query: the next expansion's drone had walked since 8:00 toward a point on
  the map's edge row it never reached, and the job never gave up, so no other expansion was decided (the
  Heartbreak edge bases: decided 7 times in 8 games, never built). Gary-Z's jobs now send a drone again
  after 45 s of walking, and drop an expansion whose drones twice never arrive (c8: 8 dropped). Minerals
  at 12 minutes 648 -> 378, army units 11 -> 14 (22 at 15), its gap score 0.34 -> 0.30; Gary-T won 9 and
  8 of 24, Gary-Z still none, 3 hatcheries at 12 minutes.
- **The records aren't complete for RL yet:** paired-branch arms are left out (and they are what RL
  trains on), records and the RL arrays share no decision id, a decision's record has no candidate-vs-
  incumbent odds and no input attribution (which inputs moved its odds), and some layers (Gary-T's fight
  rules, v3's mutalisk and lurker routines) leave only notes.
- **Arm records and input attribution:** each paired arm's records are kept (`eval/why.py --pairs` lists
  a game's pairs with the first decision where the arms differ, `--pair N` shows both arms), each learned
  decision linked to its row in the RL arrays and, in an A/B arm, with the other model's odds of the same
  choice. A learned decision also records what its odds rest on (each group of what it saw of the enemy,
  its money, history and plan zeroed in turn): Gary-Z's 7% creep colonies rest on the Terran army it has
  seen most (7% -> 4% without), as the offline probe found. A first engage game with pairs already shows
  the RL problem in plain sight: the one decision trained (the forced attack or hold) is redrawn by the
  arm itself 15 s later, and the arms end within a point of each other (two training pairs: a lead,
  not a result).
- **What Gary saw and lost between decisions (2026-10-18, `gary/agent/sightings.py`):** three record
  kinds for the human games' questions (does it react to an army approaching, to the first losses, or
  never; why was that marine there alone). `seen`: enemy fighters on contact, when 2+ supply of them is
  new, else every 10 s in sight, with where, by which base, how far from Gary's nearest building and
  whether they are closing on it (tiles gained in 5 s); an enemy building the first time it is seen.
  `belief`: once a minute with the belief model on (option `belief`), the Zerg army it expects against the
  most seen in 3 minutes. `death`: each own fighter gone, where, alone or with how many within 6 tiles,
  the enemy near it, the last source that ordered it (its lease) and who had the hands; marines sent into
  or standing by a bunker aren't deaths. From hi.observe() every half second, Gary-T only; the game is
  byte-identical with the looks on and off (one v2 game, replay and POV). `eval/why.py` shows them
  (`--layer seen`, `death`, `belief`). Live play now saves the POV log, trace and records every minute, so
  a client that dies keeps the game so far.
- **Human game 3 (2026-10-18, `runs/live/v029commit_human3`, v0.29 with `commit`; the user's Zerg won
  at 21:11; records, trace, POV and replay saved):** the user: "played MUCH better than last time;
  most of the AFK feeling is gone"; Gary "could have won at many points" (the natural broken at 6:30,
  the attacks at 9:00, 13:00 and 18:20 with the bigger army) "and always threw it": marines into the
  main's sunkens one at a time, no stim heard until 20:15, medics late, never six marines behind the
  mineral line; still bot-like: lone marines chasing 3 mutalisks across the map and dying, SCVs
  running main to natural and back, SCVs in the attacks, 12 marines shooting an egg. Overpool: the
  early zerglings again killed lone marines (1 against 6 by the bunker) and lone SCVs; the scouting
  SCV dodged the zerglings well; at 3:30 one zergling in the base drew no reaction (the rush posture
  keeps the marines in the bunker: nothing decided, by the records). What the records say:
  - **The two retreats that threw the win were draws.** The `commit` option plays with `draw=True`
    (the league's training mode), so the commitment layer samples: at 6:33 the 13 marines at the
    user's broken natural were sent home by "move (4.8%)", at 7:49 the 12 marines again by "move
    (5.1%)"; 19 commitments under 15% were drawn and sent in the game (5:22 move 4.2%, 12:28 move
    7.7% with 20 units, 15:23 move 7.3% with 17...). Play must take the likeliest decision; only
    training samples (CONTROL.md step 7). c9 and c10 were drawn too, so their play numbers are those
    of a sampling Gary.
  - **The fatal attacks were the hunt rule, not the commitment model:** at 8:32, "hunts, no enemy army
    seen for 73 s: where an enemy building was", 23 units attack-moved into the main at "100% sure",
    re-issued every 8 s; the marines arrived one by one into three sunkens (46 deaths in minute 18 the
    same way). The v0.16 hunt, adopted against Zerg v3, is a scripted attack decision on top of the
    learned one.
  - **Lone marines against mutalisks are the fight roles:** "fight: 1 Marine/Medic -> close in" and
    "1 Medic -> fire" against 3 mutalisks at 13:39-13:48 and 15:15-15:37, each dying alone (held by
    roles CLOSE_IN or FIRE); the roles send single units at flyers they can't catch.
  - **Macro:** Academy 7:39 (pros ~4:10), first Medic 8:39, Stim Packs 11:23 (6%), no third Command
    Center all game with 800-1,100 minerals banked at 11-13 min; turrets at 9:41 against mutalisks
    seen at 8:50, repaired (the user noticed).
  - **Deaths:** 176 fighters, 77 at the user's bases, 62 in the field, 37 at home; alone 50 (28%, pros
    12%); last held by army ARMY_ATTACK 70, roles CLOSE_IN 33, roles FIRE 12, reinforce 8.
  - Live note: the start command focused Gary's client, so the user saw Gary's spawn; focus the user's
    client after Alt+O.
- **The night after human game 3 (2026-10-18, runs/night, runs/league/f_*; every variant 24 games against
  Gary-Z on the three maps to 25 min, seeds 5001+, and 16 against Zerg v3):** the fixes, each an option
  judged on the same seeds (the roles' fixes in all: medics never fire or close in alone and follow the group
  that does; away from home no closing in on a group of flyers; mineral-line turrets at an end of the line
  toward the CC, no spot at the map's edge: 16 of 16 turret orders sent in a v3 mutalisk game, 13 before).
  | Gary-T against Gary-Z | won | lost | time | gap T 5-7 / 8-10 / 11-14 / 15-20 / all | v3 won-lost-time |
  |---|---|---|---|---|---|
  | commit+draw (as played) | 13 | 0 | 11 | .38 .47 .49 .55 / .38 | |
  | commit (the likeliest) | 7 | 0 | 17 | .47 .47 .58 .59 / .41 | 8-1-7 |
  | commit+nohunt | 0 | 0 | 24 | .47 .39 .58 .59 / .42 | 4-0-12 |
  | commit+nohunt+mass | 4 | 1 | 19 | .43 .35 .54 .44 / .34 | 3-0-13 |
  | commit+act (acts at 41% odds) | 6 | 0 | 18 | .46 .46 .56 .56 / .38 | |
  | commit+act+nohunt | 3 | 0 | 21 | .46 .44 .55 .54 / .37 | 4-0-12 |
  | commit+act+nohunt+mass | 3 | 0 | 21 | .43 .45 .48 .46 / .32 | 4-0-12 |
  | commit+draw+floor (no draw under 15%) | 8 | 1 | 15 | .42 .47 .52 .53 / .35 | |
  | commit+draw+floor+mass | 9 | 0 | 15 | .40 .45 .49 .50 / .33 | 10-1-5 |
  | commit+draw+floor+nohunt+mass | 6 | 0 | 18 | .40 .44 .49 .49 / .34 | |
  - **The decode:** the likeliest of keep / engage / move is keep most ticks, and Gary-T turned passive (7
    wins, worse late gap); acting at the pros' rate (41% odds of acting, where the deterministic rate matches
    the drawn one over 4,821 ticks) engages as often as drawing (37 against 41 commitments a game) but to the
    same cells (median 25 tiles, drawn 29) and finishes fewer games (6). Drawing above a floor of 15% keeps
    the drawn rate and variety without the 5% draws that threw the human game (all 19 low draws in it were
    under 14%): 8-9 wins, the gap as drawn or better. Play's decode is `floor`; `draw` alone stays the
    league's training mode.
  - **The hunt:** without it the likeliest decode never finishes a game (0 of 24; v3 4 won of 16) and the
    drawn decodes finish about three fewer: the engage choice doesn't yet go for the enemy's buildings on
    its own. The hunt stays on in play until it does; it is still the scripted attack that lost the human
    game's fights, so it is the next thing to learn (an engage toward remembered buildings once no army is
    seen) rather than keep.
  - **mass** (every one-off decision once its summed odds reach one, as expand_mass) is the one consistent
    gain in humanlikeness: the Terran gap 0.32-0.34 against 0.38-0.42 without it under every decode, mostly
    at 11-20 min; headless the early tech is unchanged (Refinery 4.6, Academy 5.3 min either way) and the late
    tech completes (a Starport in 23 of 24 games against 17, a Science Facility in 22 against 11, the Factory
    7.8 against 8.2 min). The human game's Academy at 7:39 was the rush posture's aftermath, not the decoder.
  - **Recommended for play:** `v029+commit+draw+floor+mass` (against Gary-Z won 9 lost 0 of 24, against v3 won
    10 lost 1 of 16, gap 0.33). Not yet a manifest of its own.
  - **Round e2 (`eval/pairs_read.py`): paired branches on the commitment layer, 241 training pairs, 70 of 70
    replay checks identical.** The commitment input didn't make the branched decision hold: the drawn layer
    sends 8.4 decisions a 90 s window, the forced engage is undone by a move in 200 of 236 pairs (median 17
    s), the forced move by an engage in 219 of 236 (14 s), and the arms differ by +0.01 +- 0.39 (sd 6.1; an
    arm's own 7.7), as e1. Single-decision army RL stays off until a commitment is held for a minimum span in
    the training arms, or the unit of RL is the whole window's policy.
  - **Still open from the game:** SCVs in the attacks (the records show transfers and turret builders, not
    yet why SCVs reached the user's base: a per-unit order trail is the next record); the model's own turrets
    still fail to start at some spots headless (3 of 16: the builder stops short); where the first marines
    stand against six zerglings (1 against 6 by the bunker).
- **Gary-Z, the morning after (2026-10-18, runs/league/g1-g4, each 24 games on seeds 5001+ against
  `v029+commit+draw+floor+mass`, 25 min; the baseline is f_floor_mass on the same seeds):** Gary-Z is what
  keeps the league one-sided (one win in about 240 games), so the Terran's fixes went to its side.
  | Gary-Z variant | Gary-T won / time | gap Z (all) | army 8/10/12 min | SCVs by mutas | attack min | T buildings lost |
  |---|---|---|---|---|---|---|
  | baseline | 9 / 15 | 0.30 | 10/14/12 | 1.9 | 0.3 | 0.6 |
  | + mass, floor, records (g1) | 13 / 11 | 0.32 | 8/11/15 | 2.0 | 0.2 | 0.4 |
  | + hatchery money kept while picking (g2) | 20 / 4 | 0.41 | | | | reverted |
  | + the new raid (g3) | 14 / 10 | 0.31 | 8/10/8 | 5.4 | 0.5 | 1.0 |
  | + money kept 90 s, oldest hatchery job (g4) | 18 / 6 | 0.36 | | | | reverted |
  | + every cluster joins the biggest; retreat at 1.5:1 away from home (g5) | 13 / 10, **lost 1** | 0.30 | 12/10/8 (15 min 22) | 6.8 | 0.6 | 0.4 |
  | + the main body sweeps packets it outweighs; hatcheries at 0.5 summed odds (g6) | 9 / 15 | 0.31 | | | | |
  | g6 with Gary-Z on commitments (g7) | 9 / 14, **lost 1** | 0.34 | | | | |
  | + the press: far ahead, the nearest Terran building, CCs first (g8) | 8 / 15, **lost 1** | 0.31 | | | | |
  | g8 at 30 minutes (g9; 17 of 24 games ran with the flock saving, edited mid-round) | 13 / 10, **lost 1** | 0.33 | | | | |
  | + larvae and minerals saved while the Spire morphs (g10) | 10 / 13, **lost 1** | 0.33 | | | | now an option |
  | + the alarm on the Terran fighters seen (g11) | 13 / 11 | 0.31 | | | | |
  | + the alarm on fighters away from the Terran's bases only (g12) | 10 / 14 | **0.29** | | | | |
  | + hatcheries rally to the main while the front is besieged (g13) | 8 / 15, **lost 1** (Python 5 / 3) | **0.29** | | | | |
  | + the macro sees the pros' mean history, not its own (g14) | 12 / 9, **lost 3** | 0.37 | | | | |
  | half its own history, half the pros' (g15) | 11 / 12, **lost 1** | 0.35 | | | | |
  | the pros' history of its own build style (g16) | 6 / 15, **lost 3** (Python 4 / 2 / 2) | 0.42 | | | | |
  | that history fading to its own over 12 minutes (g17, the default) | 8 / 15, **lost 1** (Python 3 / 4 / 1) | 0.33 | | | | |
  | pros (60 games) | | 0 | 18/26/32 | 9.0 | 1.4 | 6.1 |
  - **Its own records:** gary/agent/sightings.py serves both sides now; Gary-Z's act loop records what it saw and
    each fighter it lost, so `eval/why.py --side Z` answers its why-questions. Drones aren't counted: a drone
    "dies" to the tracker when it morphs or enters an extractor (182 false deaths in one game).
  - **Summed odds (GARY_Z_MASS) and a floor on its draws (GARY_Z_FLOOR):** the Terran's mass and floor on the Zerg
    side. Mass fired 2.0 times a game, mostly a Hydralisk Den and a Queen's Nest at 9-11 min; the floor stopped
    draws like "two zerglings to (84,36) at 0.3%". Neither moved the results.
  - **The hatchery gap is execution, not odds, and not money either.** Pros have 4 hatcheries at 8:00 and 5 at
    12:00, Gary-Z 3 and 4 with every larva used (capped 0%, pros 10%): larvae are its cap and the missing
    hatchery its missing army. Expansion and hatchery jobs waited in "pick" for money 30-120 s (42, 35, 20 and 8
    of them in g1; a hatchery's 90th-percentile time to stand 7 minutes), and 15 builder drones a game vanished
    on their walks (most in games already lost, with 26 marines in the main). Keeping the job's money while it
    picks lost games both ways (g2, g4): against this Terran, units now beat a hatchery later.
  - **The raid (gary_z.py `_mutas`, v3's left as it was):** v3's flock formed only from mutalisks within 12
    tiles of the gather point, where new ones never go (a flock in 13 of 24 games, 20 raids in all, 286
    mutalisks made), and attack-moved seven at a mineral line until half were dead. Now five anywhere at home
    form it; the line with no turret seen and the fewest marines; the approach along the map's edge to a post
    beyond the minerals; SCVs picked off one at a time; back to the post at 30 hp lost in 2 s or a turret or
    three marines within reach, in again after 6 s, another line after three back-offs; home at 60%. SCVs
    killed 2.0 -> 5.4 a game (pros 9.0), attack minutes 0.2 -> 0.5, Terran buildings 0.4 -> 1.0; the flock still
    loses 3-4 mutalisks a raid to marines in the mineral line it only sees once there.
  - **Joining and retreating (g5):** in g3 Gary-Z lost 118 fighters a game, 58% in the field, half of them with
    two or fewer of its own within 6 tiles and 8 enemy fighters within 8: zerglings into bio in pieces, under
    "move" orders to clusters of a median 11 (30 a game). Now every cluster but the biggest joins it
    (JOIN_BELOW 0.5 -> 1.0), and the Zerg roles retreat a group outweighed 1.5 to 1 away from a hatchery
    (zerg_roles.py, before the model's roles, which retreated 7% of the time). Round g5: Gary-Z's first league
    win of the day (seed 5004, Fighting Spirit), gap 0.30, army at 15 min 22 (17), SCVs by mutalisks 6.8; but
    130 deaths a game (118), the same share in pieces: the units die in transit between its bases and the
    main body, which the join rule sends them across the field to reach. The next piece is where new fighters
    gather (each hatchery rallies to its own mineral line; pros rally every hatchery to one point), so that
    the army forms before it moves, not on the way.
  - **Watching g5 seed 5007 (frames every 30 s from the re-simulation, both armies and Gary-Z's orders):**
    Gary-Z had no army from 6:00 to 7:30 (32 drone decisions and two zergling ones before 6:00, Lair 4:18,
    Spire 5:43) while the Terran had 13-18 supply; lost its first army at home at 9:00 to the Terran's first
    push of 15; then waited at its main while the Terran fed units one by one across the map for ten
    minutes, 8-19 in sight at a time, and fought each packet at its doorstep: 357 fighters lost to 235, the
    Terran at 94 supply to its 65 at 15:00 (42 SCVs to 39 drones, five Barracks against four larva-bound
    hatcheries). Not the composition (16 lurkers by 11:00, a Hive at 10:52, defilers from 15:00) and not the
    spread at home (two thirds of the ground army within 8 tiles of its middle when quiet).
  - **The sweep and the press (g6, g8):** the main body attack-moves onto enemy fighters in sight within 24
    tiles whose supply is under 70% of its own, away from turrets and bunkers (17.8 a game); and once its
    fighter supply is twice the Terran fighters seen in the last minute, at least 20 supply, with nothing within
    12 tiles of a hatchery, it attacks the nearest Terran building it knows of, Command Centers first, then
    production. A hatchery or expansion triggers at 0.5 summed odds (hatcheries by 12:00 beyond the first 3.0,
    was 2.5). Round g6: Gary-T won 9 (13), the trade even (168 Zerg deaths to 159 Terran; 130 to 113), Gary-Z's
    gap at 11-14 min 0.48 (0.61). Round g8 with the press: Gary-T won 8, lost 1, 15 at time; in the time games
    the Zerg army was ahead at the end in 3 of 15 (5 of 15 in g6: 75 to 7, 78 to 11, 42 to 6 without going over),
    and on heartbreak 5002 the press cleared the Terran's SCVs and army by 22:00 and still left 18 buildings at
    25:00. Gary-Z on commitments instead of the plain army model (g7): one win too, the gap worse (0.34).
    Today: 3 Gary-Z wins in 192 league games, none before.
  - **The flock saving (g10) and the alarm (g11, g12):** the Spire stood at 5.8 min with 0-3 larvae free and no
    mutalisk in the next minute (pros flock 7-9 at once). Larvae and minerals saved while it morphs gave 9 larvae,
    668 minerals and 766 gas when it stood and 8 mutalisks in 30 s, and changed no result (g10): off by default
    (`GARY_Z_FLOCK_SAVE=1`). On Python the Terran's push lands at 7:18-9:41 with 10-16 marines into 0-6 army, and
    Gary-Z saw it coming (11-20 fighters in sight by 6-7 min, the overlord posts work) while its macro model kept
    droning at 60-70% odds: pros in that state make zerglings and sunkens, and the pros' data holds few such states.
    The alarm: before 12:00, the Terran fighters seen in a minute outweigh the army by 8 supply: no drones, tech and
    expansions only at 30% odds, a sunken at the natural per 8 supply of threat (3 at most), the larvae to the
    likeliest army unit. Counting every fighter seen it fired falsely (the overlord outside the Terran natural sees
    its home army, which outweighs a Zerg's most of the game: g11, Gary-T won 13, 5 on Fighting Spirit against 1);
    counting those away from the Terran's known Command Centers (g12): Gary-T won 10, none lost, 14 at time, Gary-Z's
    gap 0.29, the day's best, 0.37 at 5-10 min. Python stays 7 of 8 to the Terran in every round of the day.
  - **The siege rally (g13):** on Python the push camps Gary-Z's natural from about 7:00 and every pair of
    zerglings hatched into it (g12 seed 5005: 66 deaths in 6-10 min, 55 at the natural, 58 with two or fewer of
    their own near). Now, with enemy fighters within 14 tiles of the meeting point, every hatchery rallies to the
    main and stragglers gather there, until 20 s after the last was seen. Round g13: Gary-T won 8, lost 1, 15 at
    time; Python 5 Terran wins and 3 at time, the first time all day it wasn't 7 or 8; Gary-Z's gap 0.29, its
    5-7 / 8-10 / 11-14 / 15-20 bands 0.36 / 0.37 / 0.46 / 0.43, the day's best in each.
  - **The decode's feedback loop (g14-g17):** the alarm was a hand-written version of what the macro model should
    know, so why didn't it react? The pro data: with 12+ bio units in sight, pros' next decisions move from 24% to
    17% drones and 8% to 17% zerglings, a mild signal; but pros at 5-9 min decide drones 17-24% of the time
    where Gary-Z's model gave 60-70% with the same drone count. The records' attribution: its drone odds rested on
    "its decisions so far" at +0.36. Its own dice-driven, drone-heavy past told the imitation model it was a
    greedy player, and the model obliged. MacroTracker.history (gary/policy/macro.py) can replace the player's own
    counts with the pros' mean per minute (data/interim/macro/v1/history_mean_<race>.npy; per build style,
    history_mean_<race>_styles.npy: an average over styles told a 3-hatch player it had made a 9-pool player's
    units). With it Gary-Z has 8 army at 6:00 instead of 0 and P(drone) 0.19 at 6-8 min; the rounds: the pros'
    history all game is the most decisive (g16: Gary-T won 6, Gary-Z 3, two on Python, the first ever there) and
    the least pro-like (gap 0.42: the army stands, sits at home and strays like a pro's, but the pros' typical
    past keeps the model droning past what it has, 53 workers at 15-20 min against 35); its own is the most
    pro-like (0.29) with the dead early game. The default is the pros' history of its own style fading to its
    own over 12 minutes (g17: Gary-T won 8, lost 1, gap 0.33, Python 3 / 4 / 1). The same loop exists on the
    Terran side (its history is its own too); untested there.
  - **The critic for games at time, and the Zerg value model:** `eval/league.py` grades every game at time by who
    is ahead at the end (supply used in the last minute, from the re-simulation the gap scores run; a lead of
    1.4x), in the round table and `--regrade` for past rounds: g13 regraded, Zerg ahead 3, Terran ahead 4, even 8
    of 15. `train/value.py --race Z` trains the Zerg value model's win head from the pros' Zerg side (held out
    AUC 0.643; the Terran's 0.64; runs/value/TvZ_Z_*); Gary-Z's own games for the net heads aren't played there
    yet. Both are the critic the plan said grades unfinished games; both sides of the league have one now.
  - **What keeps Gary-Z from winning now:** its army is a quarter of the pros' at 12 minutes with the drones and
    hatcheries nearly the pros' (39 drones at 12:00, pros 33): the larvae go to drones, and what army there is
    bleeds in the field at odds 0.45 with 18% of it present (pros 0.71, 27%). The next levers are the army's
    engagement (clusters joining before they fight, retreating when outweighed) and the expansion jobs'
    execution (which base, the drone's walk), not the decoder.
- **The attack-or-hold round read pair by pair (e1, 24 games, 188 training pairs, every replay check
  identical):** the forced decision (attack in one arm, hold in the other: the one RL trains on) was sent
  in every pair, 7.5 s after the branch (median). But each arm then decides 5.5 times more in its 90 s,
  engaging 37% and 36% of the time alike: the forced attack is reversed by a later hold in 94% of pairs
  (median 26 s), the forced hold in 85% (34 s). The arms' outcomes differ by -0.33 +- 0.43 (sd 5.65; an
  arm's own sd 7.74): no effect to learn from. eval/army_effect.py measured three times the chaos
  because there the whole army's choice held for the window. So the single branched decision isn't what
  decides a window; the cadence is (Gary-T asks every ~15 s, pros order 1-6 a minute). Before RL: the
  attack-or-hold choice held as a commitment until the pros would decide again (the keep class), then
  pairs on the commitment.
- **Commitments (keep / engage / move), learned from the pros:** every 5 s each pro army cluster labeled
  by what the pro did with it next (keep, engage, move), with its commitment (its last order, following
  the group on its way) and that order's age; 4,000 TvZ games (ingest/army_dataset.py v2,
  train/commit.py). Held-out: order vs keep AUC 0.76 (Zerg 0.78), re-decision rate per cluster-minute
  within 0.1-0.3 of the pros' at every minute, and the hysteresis learned (P(engage) already engaging /
  moving / none: 0.20 / 0.11 / 0.04, pros 0.17 / 0.09 / 0.03).
- **In play (c9: Gary-T on commitments; c10: both sides):** Gary-T's army stands like a pro's at 5-7
  minutes (0.33, pros 0.37; c8 0.65) and gives orders closer to their rate, but it isn't home more (8-10
  minutes 0.24, pros 0.63) and its gap score is unchanged (0.36). Gary-Z now attacks (352 engage
  commitments in c10, none before) and does worse: Gary-T won 13 of 24 (8), Gary-Z's drones at 8 minutes
  15 (27.5), its gap score 0.39 (0.30). Its worst game, read from the records: Gary-T's 8 units engaged
  its base at 6:06 while all of Gary-Z's army stood at the far end of the map on a minute-old move
  commitment, and every 5 s the model kept it at 88-92% while its drones went from 13 to 2. In the pro
  data, enemy army next to the player's buildings with the cluster away isn't a trigger (pros re-decide
  as often as anywhere, 0.22 / 0.20; the model matches), because the inputs have nothing for what a pro
  reacts to: losses. The grid shows where the workers are, not that they are dying.
- **Losses as an input (army dataset v3, the commit model's 10th channel):** gary_resim's deaths now carry
  the owner and place, and the army trackers count their own units gone (an SCV in a refinery, a marine
  in a bunker aren't). The pros' labels answer the question first: losing units away from a cluster,
  pros re-decide less, not more (Terran 17% of 5-s ticks against 22% with no losses; Zerg 15% against
  28%), and only ~7% of their orders go toward the losses. The model learns exactly that. So losses
  aren't the trigger: Gary-Z's c10 loss was its whole army far from home, a placement fault, not a
  timing one; Gary-T has the same tell (home 0.24 of 8-10 minutes, pros 0.63).
- **Who steers Gary-T's army away (c10, 8-20 minutes, from the records):** commitments 63% of the time,
  fights, home guard or no army 24%, the scripted hunt 13% (21 of 24 games; 38-65% of the time in four).
- **Year and map as macro inputs (2026-10-19):** the macro model's FeatureSpec takes the game's year and
  its map (the map data's SHA-1, as replays' MapDataHash; a slot for each of the 41 maps with 50+
  training games, else unknown); in play the year is 2026 and MapInfo hashes the loaded map (headless
  games; the live client has no map file: unknown). Retrained TvZ Terran and Zerg on both eras with the
  modern games weighted ten to one; on the modern held-out tenth (~19,800 samples each), against the
  same recipe without the inputs: Terran top-1 50.6% (51.2%), non-routine 43.2% (43.3%), Zerg 57.1%
  (56.2%), non-routine 49.0% (47.6%). The weighting already carried the modern meta; the inputs add a
  point for Zerg, nothing for Terran on next-decision accuracy. Both are the newest models (the
  defaults); the pros' history tables rebuilt for their decision index (train/macro_history.py).
- **The second modern pack (cwal 2026-10-19):** 12,752 new replays (6,100 more were the first pack's,
  byte for byte), August to October 2026, all 1v1 between humans on the 9-map ladder pool, none the
  same game as an earlier file. 13% flagged as bot games (the first pack 20%), leaving 11,048 usable,
  2.3 times the first pack's 4,875, and spread evenly over the six matchups (1,790-1,890 each; the
  first pack leaned ZvZ and was short of PvT). Longer games (median 14.5 minutes, as the 1.16 era's
  14.6; the first pack 10.5) and more players (3,991, the top ten 5% of games; the first 1,593 and 14%).
  Re-simulated (the 11,048 usable): 92.4% clean (the first pack 93.4%), 1.7% failed (mostly the "map
  version 64" maps), 5.9% desynced, the same matchups as before: ZvZ 99%, PvP 98%, PvZ 97%, TvZ 90%,
  PvT 87%, TvT 83% (the SC:R movement difference, docs/DATA_ERAS.md). Modern clean games now 2,150-2,830
  per matchup (both packs), near the ~3,000 where the curve flattens.
- **Where the commitments send the army (held-out pro ticks, army v3):** the commit model's destination
  draw goes home as often as the pros' orders do (move: Terran 26% against 29%, Zerg 17% against 19%;
  engage alike), with the cluster already home 44% (Zerg 30%) when the order comes. So on pro states the
  destinations are right too: Gary's home deficit (0.24 of 8-10 minutes against 0.63) comes from the
  states its games drift into and from the scripted layers (the hunt's 13%, fights), not from the
  model's choice of when or where.
- **The second pack in the datasets and the macro models (2026-10-19 night):** macro, army (v3) and
  fight datasets for all six matchups and the build orders now hold 2,160-2,850 modern games per
  matchup (TvZ 1,401 -> 2,463). Macro models trained for every matchup and race (both eras where the old
  one exists, modern x10, year and map inputs; candidates promoted only if at least as good on the
  modern held-out games): non-routine top-1 TvZ Terran 44.6% (the model before 43.7%), TvZ Zerg 47.4%
  (47.2%), ZvZ Zerg 63.5% (62.2%); first models for TvT Terran 36.4%, PvT Protoss 28.0% and Terran
  31.6%, PvZ Protoss 29.1% and Zerg 53.0%, PvP Protoss 46.6% (modern games only for the Protoss
  matchups). The pros' history tables rebuilt for all nine.
- **c11: the new macro models in play (c10's seeds and setup, the commit models pinned to c10's):**
  Gary-Z drones 24 at 8 minutes (c10 15, pros 25), 28 at 10, 44 at 15; army units 12 / 19 / 33.5 at 8 /
  12 / 15 minutes (5 / 7.5 / 12); alive at 15 minutes in 18 games of 24 (13); its gap score 0.33
  (0.39). Gary-T won 6 (13); at the 20-minute limit 10 games even, 6 Terran ahead, 2 Zerg ahead.
  Gary-Z still wins none outright and banks ~1,000 minerals at 12 minutes on 4 hatcheries (larvae).
  Its first ZvZ commit model: order vs keep AUC 0.80, rates within 0.3 of the pros', hysteresis
  0.26 / 0.12 / 0.06 (pros 0.25 / 0.11 / 0.05).
- **c12 (c11 with the commit models that see losses):** about c11: Gary-T won 7 (6), at the limit 8 even,
  7 Terran ahead, 2 Zerg ahead. No gain from the losses input, as the pro data said.
- **The first ZvZ mirror games (12, the new ZvZ macro and commit models):** they play to the end; 11 of 12
  reach 20 minutes undecided, both sides banking 1,000-5,000 minerals: the larva limit c11 showed
  against Terran, in Gary-Z's shared execution.
- **Gary-Z's banking, read from the records (c11's worst game, 3,800 minerals at 12 minutes):** not larvae
  there (6 free) but its macro making 11 decisions a minute: each Lurker decision held it 30 s and was
  given up, because an overlord over the hydralisks took every click (68 in 30 s, the screen log).
  Gary-Z now picks a unit with nothing of its own on top and another after each miss; an expansion
  whose drone never arrives is given up after one walk. c13 (c11's seeds and setup): alive at 12
  minutes in 23 games (21), at 15 in 21 (18), Zerg ahead at the limit 4 (2), Gary-T won 6 (6),
  minerals at 12 minutes 806 (1,083). The typical game is still larva-bound (2 free at 10-15 minutes).
- **The current ladder maps for headless games (ingest/ladder_maps.py):** every round so far played
  three 1.16 maps nobody plays now. A Remastered replay rewritten in the 1.16 format loads in OpenBW
  with its map (resim/scr_format.py to_legacy), and the original hash rides beside it for the macro
  model's map input: KnockOut 1.4, Octagon SE 2.0, Backrooms 1.1, Attitude SE 2.1, Radeon 1.2, Fighting
  Spirit 1.4, Colorless Fate 1.1, Odyssey-RE 2.0. eval/league.py --maps ladder. L1 is the first round
  on them.
- **L1, the first round on the ladder maps, and the scout:** Gary-T lost all three games on Odyssey-RE
  by never playing past 1:06: there the miners stand stacked (7 of 9 SCVs on one spot), the scout's
  10-pixel box caught several, the scout task was given up and asked for again every 6 frames ahead of
  the macro (the arbiter trace). Now a click picks the scout and a failed pick waits 3 s. L2 (L1 again):
  Gary-T won 10 of 24 (Odyssey 3 of 3), at the limit 9 even, 4 Terran ahead, 1 Zerg ahead.
- **The gap score against modern pros on the ladder maps:** it compared Gary with 1.16-era pros on the
  fixture maps. eval/humanlike.py --pros-era modern (cwal TvZ games on the current maps; league rounds
  on --maps ladder use them). L2 against them: Gary-T 0.30 (against the old pros 0.36), Gary-Z 0.39;
  Gary-Z's 2-4 minutes 0.46, not 0.70: most of its early "gap" was the old pros. Its 8-10 minutes (0.56)
  is now its clearest gap.
- **Model spec v2, one macro model per race (Fable's order: before any self-play training):** the
  matchup is an input (one-hot) besides era and map; train/macro.py --matchup all, judged per matchup
  against each matchup's own model (--compare-separate) on its modern held-out games. Non-routine top-1,
  shared / separate: Terran TvZ 44.8 / 44.6, TvT 37.5 / 36.4, PvT 37.3 / 31.6; Protoss PvT 32.1 /
  28.0, PvZ 31.3 / 29.1, PvP 47.7 / 46.6; Zerg TvZ 47.7 / 47.4, ZvZ 64.0 / 63.5, PvZ 55.2 / 53.0: better
  everywhere, most where a matchup has fewest games. In play behind GARY_MACRO_SHARED (L3, L2's seeds):
  Gary-T won 10 (10), at the limit 6 even, 8 Terran ahead, 0 Zerg ahead (9 / 4 / 1), gap scores 0.31 /
  0.41 (0.30 / 0.39): no clear difference on 24 games; a larger paired comparison decides.
- **Army and commit models for TvT, PvT, PvZ and PvP** (for the one-Gary refactor): commit order vs keep
  AUC 0.76-0.80.
- **The shared macro models are the default (L4, 48 games each, the same seeds and maps):** separate /
  shared: Gary-T won 13 / 12, Gary-Z 1 / 1 (its first wins against a Gary-T that plays), at the limit even
  14 / 20, Terran ahead 12 / 11, Zerg ahead 8 / 4; gap scores against modern pros Gary-T 0.30 / 0.30,
  Gary-Z 0.36 / 0.34 (2-4 minutes 0.48 / 0.41); Gary-Z's army at 12 and 15 minutes 17 and 28 / 23 and
  34.5. Even in results, closer to the pros for Gary-Z, better on held-out games in all nine
  matchup-races, and the layout later training builds on. GARY_MACRO_SHARED=0 plays each matchup's own.
- **What the modern pros show (L2 against them):** Gary-T's biggest tell is the natural bunker (2 at 5-10
  minutes; modern pros 0-1), then fewer refineries after 8 minutes (1, pros 2) and supply blocks at 5-7
  minutes (35% of the time, pros 0). Gary-Z's: its army at home (80% of 8-10 minutes, pros 19%), half the
  pros' drones on gas, no mutalisks by 10:00 in 19 of 48 games (a late Lair: the Spire's summed odds
  reach one at 10-14 minutes), 3 hatcheries to their 4.
- **Without the natural bunker (L5, 48 games each, the human-play default v029+commit+draw+floor+mass
  against it +nonatbunker):** Gary-T won 18 / 21 (lost 0 / 1), its gap score against modern pros 0.29 /
  0.25; Gary-Z ahead at the limit 3 / 6. Closer to the pros and stronger: the recommended version for
  human play is now v029+commit+draw+floor+mass+nonatbunker.
- **A depot ahead of production (L6, option depots, against L5b on the same 48 games):** the macro model
  left Gary-T supply-blocked at 4-8 minutes in 42 of 48 games: pros almost never are, so the model gives the
  depot under its floor once there, and 437 of its 749 decisions while blocked were "nothing it can do"
  (the records). With the scripted builds' rule (a depot while each producer has under 2 supply left):
  30 such decisions in 15 games; its supply cap at 8-10 and 11-14 minutes 76 and 116, the modern pros'
  exactly (68 and 108 before); workers 38 at 8-10 (pros 36, before 34). Gary-T won 21 / 21, lost 2 / 1;
  the overall gap score 0.27 / 0.25 against the same pros, within the noise of 48 games. Kept: the
  recommended version for human play is v029+commit+draw+floor+mass+nonatbunker+depots.
- **Gary-Z's commitment floor off (L7 against L6, 48 games):** Gary-Z drew no keep / engage / move under
  15% odds; a cluster's engage odds are 5.5% a tick (median), so it engaged at 7% of its draws at 8-10
  minutes (pros 12%) and moved 18% (pros 13%). Off: its gap score against modern pros 0.31 (0.33), lower
  in every minute range; army home 64% of 8-10 minutes (72%; pros 19%), in the field 27% (17%; pros
  69%), standing 63% of 5-7 minutes (85%; pros 19%); ahead at the limit in 11 games (3); Gary-T won 18
  (21). The default now.
- **Gary-T's floor (L8 against L7, 48 games):** off, its gap score against modern pros 0.29 either way:
  closer at 5-7 minutes (0.33 against 0.40: its army standing 39% of the time, pros 25%, with the floor
  80%) and further at 8-10 (0.45 against 0.38: home 31%, pros 53%, with the floor 38%); Gary-T won 22
  (18). No clearer humanlikeness, only strength: the floor stays.
- **Human game 4 (2026-10-19, Fighting Spirit 1.3, the user's Zerg overpool against v029+commit+draw+
  floor+mass+nonatbunker+depots; the user won at 18:30).** "Much better than last time, more attentive,
  fewer mistakes": the early zergling harass held, turrets well placed and repaired quickly, marines
  stimmed to chase mutalisks, the trades in its favor; a deliberate timing attack just before the
  mutalisks. What still read as a bot, the user's list:
  - the attack carried out as an attack-move from home: no medics, into sunkens, no regroup in front of
    the natural, no retreat when hit, not called off when outnumbered; plans that don't adapt;
  - no micro of single units in a group (APM after 5:00: 15% camera clicks, 24% selection boxes, 11% SCVs
    picked one by one; few orders to units, none per unit);
  - workers: idle SCVs all over its bases, mining SCVs walked to stand idle in the main, workers moved
    main -> natural when mutalisks hit the natural (8 at 6:38, 8 more by 11:14, 11 back by 12:57), SCVs
    pulled into fights (3 pulls at 7:47-7:53);
  - turrets overbuilt (16) at the army's cost; mutalisks hiding behind the natural's cliff never spotted
    (no scan or floating building for vision there); one Comsat, no energy for the lurker hold-position
    trick at 14:40 (it walked into lurkers it saw burrow); no Science Vessels;
  - late game: no expansion past the natural, no hunting the Zerg's expansions, no tech;
  - building placement: buildings spread out, no packed depots: by late game no room to build or to move.
  Also: the start must focus the human's client before the countdown ends (auto_game.py --start
  --focus-after PID).
- **Workers off their work (L9 against L6, 48 games, 2026-10-19).** A new humanlike feature,
  `workers_off`: the share of workers neither mining, building nor repairing while no enemy fighter is
  within 8 tiles (walking, standing, following or attacking all count; idle alone had read 9% and missed
  most of it), and `workers_off_3`, the share of minutes with 3 or more such workers. Three causes, found
  from the records and the replays:
  - a mining task whose minerals were off screen right-clicked them on the minimap: a move, and the
    workers stood idle by the field (41 of 188 mining tasks in human game 4); transfers did the same;
  - a selection box that caught miners standing stacked at the field was refused and retried (39);
  - a right-click on a field's middle hit an SCV standing there and became a follow order (a quarter of
    the off time in a test game).
  Now the camera goes to the base first and the field is right-clicked on its far side from the town hall
  (gary/agent/executor.py `_mine_gather`, `field_click`), a box with miners in it is accepted, and a base
  stays off-limits 60 s after zerg fighters were seen there (20 s let mutalisks that hid and came back
  draw the main's workers under them). Against modern pros (their workers_off 7% at 8-10 minutes, 9% at
  11-14, 14% at 15-20): Gary-T 18% / 23% / 30% before, 9% / 11% / 13% after; minutes with 3+ off at 8-10,
  100% before, 60% after (pros 43%). Gary-T won 30 (21), lost none (2); gap score 0.27 (0.29).
- **The one-Gary refactor, Phase 0 (2026-10-19): the race object.** Gary takes a race; the agent reads
  its race's facts through `g.race` (gary/race.py over gary/terran.py, gary/zerg.py and the new
  gary/protoss.py, all three checked against the game's unit data), its matchup and enemy from the
  game's players, and every model through one lookup by matchup and race. No behavior change:
  tests/test_parity.py, three games' logs hashed before and after, byte-identical.
- **The one-Gary refactor, Phase 1 (2026-10-20): the executor as a race plug.** The shared Executor with
  a Terran and a Zerg executor plugged in; the macro asks the executor what a decision means; Gary-Z's
  decision rules as macro options; the manifest gz1 is the one Gary playing Zerg (no army or fights of
  its own yet: Phase 2). Its openings against Gary-Z's in 32 ZvZ mirror games of 4 minutes
  (eval/zerg_opening.py): drones 12 / 12, the Pool 1:35 / 1:54, the Extractor 1:17 / 1:24, larvae and
  mining alike; the second hatchery later (by 4:00 in 14 games / 21) and supply blocks a little longer
  (36 s / 26). Terran games still byte-identical (tests/test_parity.py).
- **The one-Gary refactor, Phase 2 (2026-10-20): Gary-Z's army, posture and fights as shared providers.**
  The manifest gz1 is the shared agent playing Zerg in full: the alarm as a posture, Gary-Z's posts,
  raid, join, sweep and press as army options, its fight roles and lurkers as a fight provider, its
  turns as an arbiter option; portrait clicks in the selection panel (a box, then one type kept, or an
  overlord taken out). Against the default Gary-T, 48 games on the ladder maps (L9, Gary-Z's, in
  brackets): Gary-T won 25 (30), the Zerg ahead at the limit in 7 (5), alive at 15 minutes in 36 (25),
  gap score to modern pros 0.28 (0.33). On the way: the launcher had given gz1 b_rank hands (three
  rounds lost to it); a parity diff on the games Gary-Z led found what was missing.
- **The one-Gary refactor, Phase 3 (2026-10-20): one launcher, the old stack deleted.** eval/match.py plays
  any two manifests (the league, mirrors, tests and the tools that step a game through it), playing the
  same games byte for byte as the old launcher did; gary/play.py plays the local player's race (--terran,
  --zerg). gary/bots and the tools built on its scripted Zergs (vs_rush, walls_bench, block_drill,
  lurker_games, army_effect, garyz_accept, mirror, zerg_opening) are deleted; the tools that stepped
  games against them play gz1. The refactor is done: one Gary, a race a plug.
- **Every matchup that can play is tested (2026-10-20).** tests/test_matchups.py: TvZ, ZvZ and TvT play, the
  Protoss ones skip (no executor yet); TvT played for the first time, the Terran manifest's TvZ-only parts
  (early-pool posture, Zerg answers, the TvZ role model) switched off against a non-Zerg by its `vs` section.
- **Zerg's economy (2026-10-20).** By 5:00 gz1 had mined 2,492 minerals to modern pro Zergs' 3,232 (ZvT), with
  13 drones at 4:00 to their 16; its sunkens stand at the natural (111 of 112). The causes, in order of size:
  the macro took whatever was doable at 5% while the drone waited (a Spawning Pool during the 9/9 supply block,
  the natural at 2:06 to the pros' 1:38), now held (LearnedMacro.hold: a roll waiting for supply or money stays
  the roll); a second overlord ordered as the first hatched; the model's pro history read a minute late
  (all of minute 0 as "nothing decided"); and on Odyssey every new drone followed an overlord parked on the
  rally's mineral field (1,000-1,600 drone-seconds a game), now moved off. After: 2,752 mined, 14 drones at
  4:00 (16 in the version that held larva waits too, but its tech came late). 48 games against the default
  Gary-T (G11 against G9): Gary-T won 19 (24), the Zerg won 5 (0), the Zerg's gap to the pros 0.25 (0.28).
  Natural creep colonies (the user's suggestion after human game 4, option `colonies`) were mixed in G8:
  an option. Both races' workers double up on a field 3-4 times as often as pros', but a chore spreading
  them gained nothing per worker-minute (dropped).
- **One name for the Terran (2026-10-20): gt1.** The play version had grown to
  v029+commit+draw+floor+mass+nonatbunker+depots, each change an option appended once it won, while the league's
  default opponent was still the bare v029 (none of them). gt1 is that string plus lords, folded into one manifest
  (gary/agent/manifests.py), the default of gary/play.py, eval/league.py and the tests; changes are judged as
  gt1+<option> and an adopted one goes into it.
- **Attacks measured, and finishing games (gt1 = gt0.2, 2026-10-20).** eval/attacks.py: a Terran's attacks
  from the replays (3+ fighters by a Zerg building after 20 s with none), Gary's against modern pros'. Against
  gz1 (G13, 48 games; pros 48): army share there 0.50 (0.54), late arrivals per fighter there 0.26 (0.06),
  medics along 1.09 (1.0), into sunkens 38% (28%), lost / killed supply per attack 6.0 / 11.0 (2.5 / 6.0),
  still there at the end of a losing attack 0.12 (0.15). Its 18 unfinished leads: the Zerg down to a base or
  two with no army, the clusters' engages drawn all over the map (overlords count as enemy presence), the hunt
  held off by "no enemy army seen for a minute" while new zerglings appeared, then the whole army at one unseen
  base at a time. Now the hunt also runs at 4x the enemy fighters seen in the last minute, and with no
  building known squads of 8+ search the unseen bases at once: Gary-T won 24 (21), lost 2 (3) of 48. Not
  adopted: Gary-Z's join (won 20, the Zerg ahead at 20:00 in 8 to 2), press (won 22, the Zerg ahead in 7),
  press at 4x (22 won, 19 led: even), the hunt at 4x alone (23 won, 13 led).
- **Rounds to 30 minutes (2026-10-20).** Half of every 48-game round had ended at the 20:00 limit, graded by
  who was ahead, and nothing after 20:00 was measured. The league, match, humanlike and attack measures run
  30 minutes now (the gap score with a 21-30 bucket; 20% of modern TvZ pro games pass 20 minutes, 2% pass
  30). The new baseline, gt1 (gt0.3) against gz1, round B30: Gary-T won 34, the Zerg 4, 10 at the limit
  (Gary-T ahead 5, the Zerg 2, even 3); the same seeds at 20 minutes: 24, 2 and 22. Gap at 21-30 minutes:
  Gary-T 0.39, the Zerg 0.46 (its worst bucket). Still at 10-12 minutes: the fight, plan and value models'
  training data (to rebuild on whole games).
- **Fight roles from whole games (2026-10-20).** The role models that steer each fighter (fire, close in, pull
  back, retreat, reposition) were trained on the pros' units in the first 20 minutes of 600 games on the three
  1.16 maps (ingest/response_dataset.py). Now 30 minutes, 800 games per race, half old and half modern ladder
  games (`--era both`; Terran 3.5M labeled units, Zerg 2.8M), the same training (`train.roles --decisions
  --move`); a manifest picks its model by its data (`roles_data`, option `roles30` for either race). Against
  B30 (48 games, 30 minutes): gz1 with it, the Zerg won or led at 30:00 in 9 (6) and its gap at 21-30 minutes
  0.40 (0.46): in gz1. gt1 with it won 33 (34), the Zerg led at 30:00 in 6 (2), Gary-T's gap at 21-30 minutes
  0.45 (0.39) from one round with no spread: open, not rejected. The new model imitates the pros better offline
  (0.754 to 0.728) and the old one never saw a fight past 20 minutes or a modern map; Gary-T's fight options
  were tuned around the old one. Next: 96 paired games with the gap per 5-minute window and its spread
  (eval/gap_windows.py); if the late gap holds, re-tune the fight options to the new model, then the data
  ablation (old maps to 30 minutes against modern only). Done: over 96 paired games (B30 + B30b against R30T + R30Tb) the
  Terran's gap is within noise in every window (20-25 minutes -0.001, 25-30 +0.014, 90% intervals across
  zero); won 54 (57), lost 9 (7). In gt1 (gt0.4); R30T + R30Tb are its 96-game baseline. Also measured: two
  thirds of Gary-T's late arrivals at its attacks were units that existed when the attack began (other clusters,
  stragglers), a third new production (the pros' 37%): the trickle is the army not gathering first.
- **Gather before engaging (option `gather`, 2026-10-20; a stop-gap, AGENTS.md).** A commit engage with a quarter of
  the army elsewhere first brings the army to its own middle (12+ tiles from the target), then attacks at 75%
  there or after 20 s; it never vetoes the engage (a first version called it off with enemies at home: the
  user: moving out is often what ends a mutalisk harass). 96 paired games against gt1 (R40a + R30Tb against
  G40a + G40b): the gap at 10-15 minutes 0.352 (0.398, the interval below zero), lost 6 (9), but won 47 (54) and
  16 even at 30:00 (10); late arrivals at attacks 0.22 (0.26; pros 0.06), killed per attack 9.0 (7.5). Not
  adopted: the trickle mostly comes from attacks it doesn't cover (engages with under a quarter elsewhere, the
  hunt, launches on the timeout).
- **What the commit model sees, and a shared draw (2026-10-20).** It already sees the rest of the army: grid channel 0
  is the whole army's supply per cell, channel 8 the cluster asked, its supply and the army's in the global
  inputs, the dilated convolutions reach the whole map; only the other clusters' commitments are missing. A
  decoder giving every cluster the same random number per 5-second tick (each still drawn from its own odds),
  so that like clusters engage together: 96 paired games against gt1, won 44 (54), lost 14 (9), the gap
  unchanged in every window, late arrivals at attacks 0.25 (0.26). Deleted. Neither missing inputs nor
  independent draws explain the trickle; next: which decision sends the late arrivals.
- **What sends the late arrivals (2026-10-21).** The motor now records each army order's units and origin; an
  instrumented replay of R40a (D40a, the same games) traced every late arrival at Gary-T's attacks to its last
  army order: the hunt 38%, no army order in 2 minutes (fight-layer moves, the rally) 25%, an order elsewhere
  19%, the commit model's engage there 16%. A hunt that orders the main body and sends the rest to it first:
  96 paired games, won 46 (54), lost 10 (9), late arrivals 0.26 (0.26: the joiners still came late, now as
  "hunt join" 16%). Deleted. Three fixes aimed at the trickle (gather, a shared draw, the hunt's body) left it
  at 0.22-0.26 (pros 0.06) and none won more games: the trickle is how Gary-T's army spreads over the map
  (production, the hunt, the fight layer), not one rule to correct.
- **Human game 4's item 2 (2026-10-21): SCV pulls and the "AFK" early zerglings.** The pulls at 7:47-7:53 were the
  pull rule answering 5-6 zerglings at the natural with 1-2 marines left (the army lost at 6:20 into sunkens): 7 SCVs
  killed them all for 2. Against gz1 Gary-T answers harassment faster than pros (7 s, 11.7) and meets more of it;
  what is off is the lone unit: alone and outnumbered, it attacks 75-78% of the time (pros 42-44%) and backs
  away 0-2% (13-19%), in human game 4 and in 96 games against gz1 (eval/humanlike.py lone_attack / lone_away).
  In contact the hand rules decide (RoleAhead: the roles model only before contact). Option lone gives such
  units to the roles model: 96 paired games, won 57 (54), lost 8 (9), the gap within noise, its retreat orders
  doubled (401 to 184 in 48 games) but lone_away 0.02 (0.00): the model rarely chooses to back a unit off
  (retreat 0.4% of its decisions). In gt1 (gt0.5); the model's calibration is self-play's to fix.
- **The commit model with the other clusters' commitments (2026-10-21).** train/commit.py --others: four inputs per
  cluster sample, from the clusters sampled at the same tick (their supply committed to engage and to move, the
  nearest one's distance and whether it engages; gary/policy/army.py others_inputs). On the same 438,440 held-out
  pro ticks: the deployed model (10-17) order-vs-keep AUC 0.760, engage-vs-move 0.771, cell in the top 5 0.372;
  the same recipe retrained on today's data (16,496 player-games, both eras) 0.779 / 0.791 / 0.385; with the
  other clusters' commitments 0.779 / 0.791 / 0.383: nothing gained, not played. The retrain itself (runs/commit_exp/
  T_A, pinned with GARY_TCOMMIT_MODEL), 96 paired games against gt1: won 57 (57), lost 11 (8), the gap unchanged,
  fewer attacks (4.8 a game, 5.8), late arrivals 0.27 (0.24): better imitation per cluster, no better play. Not
  adopted: the army's coordination is past what per-cluster imitation teaches.
- **Attention in fights, and control groups (2026-10-21).** Where commands go during fights (first 10 minutes,
  the same measure for both: replays re-simulated for fight moments and unit commands, cmdcache for production):
  pro Terrans put 160 commands a fight-minute into the fight (67% of their commands then), Gary-T 45 (40%); pro
  Zergs 238 (77%), Gary-Z 16 (26%); macro during fights is at the pros' rate. Fight moments a command reached
  within 2 s: pros 71% / 67%, Gary-T 61%, Gary-Z 27%; army units reached: pros 60% / 58%, Gary-T 33%, Gary-Z 14%.
  The arbiter isn't starving the fight: Gary-T's hands sat free 44% of its fight moments. Imitating only attended
  pro units (ingest/response_dataset.py attended, train/roles.py --attended --budget) gave models no better on
  attended validation labels: the labels were not the bottleneck. Pros' army control groups: keys 1-3, assigned
  6-10 times a minute, almost never added to, recalled 41-71 a minute. Control groups in the motor (assign after a
  box of a cluster, recall for the orders after, a role's order to a group most of which got it): 96 paired games,
  won 47 (57), lost 14 (8), selections per order 1.56 (1.54), recalls 19% (19%): groups formed and seldom
  recalled, because three quarters of Gary-T's fight orders go to 1-4 units whose role just changed. Reverted. The
  gap is in how the roles layer orders (small groups, on role changes; pros re-order whole groups many times a
  minute): the learned layer's, for RL with the budget given.
- **Control groups as a slot, the pros' hotkeys, and re-issued orders (2026-10-21).** gary/agent/grouping.py: grouping
  is a provider slot (which units on which key, asked when units arrive or clusters merge or split for good); the
  scripted default, "clusters", and the motor's mechanics (assign, add; an order to exactly a key's members is one
  key and one click; no pooling), off by default. ingest/hotkey_dataset.py, 200 modern TvZ games a race, after 5:00:
  pro Terrans use 4 army keys, assign 6.0 a minute, add 0, recall 62, group 5 units, mostly by type (marines 20%,
  vultures 15%, vessels 11%, medics 7%, tanks 6%); pro Zergs 3 keys, assign 9.6, add 0, recall 41, group 9 units
  (mutalisks with an overlord 21%, zerglings 8%, lurkers 7%); selections per order 1.65 and 1.32. Re-issues: of the
  pros' attended fight commands (first 10 minutes, 150 games) 49% (Terran) and 72% (Zerg) re-issue the same order to
  the same units with the target moved (a median 1.2-1.3 tiles), 75-87% of them right-clicks on the ground. A skill
  keeping standing orders current (option refresh), on the 239 home defenses (8 draws; gt1 net -52, 22 orders a
  scenario): attack-moves re-aimed at the enemy nearest the group -67 (30 orders); along the role model's predicted
  move -66 (paired -12.9, 90% -22 to -3.6); moves only (retreat, pull back, reposition along the model's move) -55
  (paired -3.3, -7.5 to +0.7; 23 orders). Kept as the move-only option, the re-order mechanism for the roles RL.
- **A suite that sees re-orders: scenarios by state (2026-10-21).** eval/scenario_kinds.py (eval/scenarios.py --suite
  kinds): Terran fight snapshots from 4:00 of held-out modern TvZ games classified as mixed (a big fight, 2+ fighter
  types each side), attack (nearer a Zerg town hall), outnumbered (theirs 1.6x ours or more) and harass (from 6:00,
  Zerg fighters by the Terran's buildings, its army weaker there), screened as before (the pro's control worth 100+
  over doing nothing), away from home kept: 79 scenarios from 36 games (22 mixed, 19 attacks, 20 outnumbered,
  18 harassment; a median 9:07 in). 45 s each, 8 draws; net: pro +517, nothing -134, gt1 +105 (attacks +282,
  outnumbered +163, mixed +59, harassment -91; 34 orders a scenario). Paired (90% bootstrap over scenarios):
  gt1+refresh -5 (-27 to +16; mixed -45, -89 to -1): the move-only refresh doesn't help here either. The role
  model imitated from attended pro units with the order-rate input, told Gary's own rate (option att): +35 (-5 to
  +77); att+refresh against att -1 (-32 to +32). att+refresh (+141) is the RL's start: the re-order path is in
  it, whether to use it is the policy's to learn.
- **Control groups by type (2026-10-21).** Round GR40a (gt1+grouping against gz1+grouping, the clusters provider, 48
  games against L40a): Gary-T won 5 and lost 11 (L40a: 34 and 0), the gap +0.107 further from the pros (+0.074 to
  +0.143), 15-20 and 25-30 minutes most; both sides grouped, so which side it served is not known. Its keys, against
  the pros' (ingest/hotkey_dataset.py): T 1 key, assigns 1.9 a minute, adds 3.8, recalls 0.1 (pros 4 keys, 6, 0,
  62); Z 3 keys, recalls 6.8 (pros 41): an order's units were almost never exactly a key's. The default provider is
  now "types" (a unit type of a cluster on a key, re-assigned, not added to; a box narrowed to the type by its
  portrait) and the motor recalls a key whenever all its members get an order, key by key, the rest selected as
  before. In a 14-minute smoke game: T assigns 8.7 a minute, adds 0, recalls 42. Gary-T keeps 3 army keys (4-9
  are its buildings', 0 the scout's), the pros 4. To judge side by side, each side grouping alone. The hotkey
  statistics stay their own report (ingest/hotkey_dataset.py --report --gary), not in the humanlike gap:
  that measures decisions, and click habits would dominate it.
- **Grouping by type, each side alone (2026-10-22, 48 games each on L40a's seeds).** Gary-T grouping (GT40a,
  4 army keys): Gary-T won 23 and lost 8 (L40a 34 and 0), the gap +0.007 (-0.030 to +0.045), 15-20 and 25-30
  minutes further (+0.072, +0.059). Gary-Z grouping (GZ40a): Gary-T won 29 and lost 4, Gary-Z's gap -0.073, its
  15-20 minutes nearer (-0.076, -0.126 to -0.013). GR40a's collapse (both grouping) was the two at once. Not in
  gt1; for gz1 a second round (GZ40b, seeds 4010-4015 against L40b) decides on 96 paired games; Gary-T's losses
  are read in the records first (recalls overriding the roles, or units pulled off posts and bunkers). The fourth
  army key goes with Terran grouping.
- **What Gary-T's grouping losses were (2026-10-22, GT40a against L40a seed for seed):** the mechanics' cost, not the
  by-type default. The Terran's actions 5:00-14:00 rose from 198 to 244 a minute: camera 56.5 -> 66.0, drag boxes
  41.0 -> 51.8, portrait clicks 8.3 and assigns about 13 (each re-assign a task of its own: camera, box, portrait,
  Ctrl+key), for army recalls 1.5 -> 5.7; production orders 13.9 -> 13.4 and production keys' recalls unchanged
  (the keys' new layout wasn't it). The bank was higher from 6:00 (minerals 309 against 256; 1,525 against 1,154
  at 14:00), the army smaller from 14:00; in the 8 losses behind by 10:00 (army 15 against 23, workers 30 against
  38). A recall overriding the roles can't happen by construction (a key is recalled only when all its members get
  the order). The pros' assign costs one key, their units already selected for an order: the fix is assigning
  then, a dedicated selection only for a key long out of date.
- **Zerg grouping on 96 games (GZ40a + GZ40b against L40a + L40b):** Gary-Z's gap -0.054 (-0.089 to +0.034), Gary-T
  won 49 and lost 10 (57 and 8): not adopted; with cheaper assigns, both sides again.
- **How long commitments stand (2026-10-22; army dataset v3, 1,500 modern TvZ player-games a race).** The pros'
  reversal chance per 5 s tick falls with the commitment's age: Terran engages 10.5% at 0-5 s, 5.6% at 40-60 s
  (a quarter reversed by 20 s, half by 55 s, three quarters by 115 s); Terran moves 16.3% to 4.9% (10, 35, 110 s);
  Zerg engages 14.8% to 4.8% (15, 40, 120 s); Zerg moves 16.3% to 3.8% (10, 50, >120 s). Gary's own (L40a's
  records, per commit decision; 11 a minute for Terran, 6 for Zerg): Terran engages 8.5% at 0-10 s to 0.4% past
  60 s, moves 18.6% to 2.6%; Zerg engages 18.8% to 9.0%, moves 13.6% to 4.5%: the pros' falling shape, Gary-T
  holding engages longer and Gary-Z reversing them more. So e1 and e2's fast reversals (85-94% of forced arms
  undone within 14-34 s) were the arms being forced against the model, not the model's own hysteresis: the lock
  belongs on the training arms (a forced choice reversed at the pros' hazard, not the model's), the deployed
  policy needing little change.
- **The again decision (2026-10-22).** ingest/reissue_labels.py: in the attended response data (779 of 800 games
  read) 20.5% of labeled units get a command in the next half second; a keep with one is "again" (20.8% of the
  attended labels; keep 45.8%), its target a median 7 tiles from the unit. train/roles.py --again (runs/roles/
  20261022-021940): again recall 0.20 and precision 0.36, the offset head's error 4.9 tiles against 6.8 for the
  unit's own spot. Pipe check (10 held-out scenarios x 4, refresh, model in contact, batch): commands a minute
  79.6 at the start (63.7 without again), 86.0 with again pushed +2 (reach 71% -> 72%, net -122 -> -8), 76.0 with
  the moves pushed (reach 73%); with grouping the same (78.8 -> 86.0, 71% -> 73%). Orders decided a turn 2.14
  (pushed 2.65, 0.93 before again), but 113-132 queued orders dropped stale in 10 scenarios with the APM bucket
  full: the hands' time per order (cursor, camera, boxes) is the cap now. On the full suite the again start nets
  +63 (paired -38, -101 to +20, against the batch start without again; -85, -185 to +15, against RoleAhead's
  +141): more orders, not yet better ones.
- **What an order costs the hands, and the minimap (2026-10-22).** The pros, 300 modern TvZ games (fight dataset
  v3; replays don't say what was clicked on the minimap, so proxies): 92% of their in-fight point orders target
  within a screen of the units ordered (median 5.7 tiles away), and of orders within 8 frames of each other 1.3%
  are a screen apart: the fight stays on their screen, the minimap a small share. Their fight orders are small:
  Terran one unit 57%, 2-3 units 16%, 4+ 27% (Zerg 35%, 14%, 51%). Gary (10 held-out scenarios, the again model,
  several orders a turn): a role order takes the hands a median 6 frames (quartiles 4-12); per fight order 0.97
  drag boxes, 0.76 screen orders, 0.24 minimap orders (the motor gives them when the target is off screen), 0.67
  camera moves: camera, box around one or two units, order, over and over, the camera taken home by the macro and
  brought back group by group. Cheaper selection: a key set when an order's selection fits a planned group (one
  keypress), its own selection task only after 20 s out of date, a recall before any camera move, a turn's
  queued orders on screen first: camera moves 0.56 an order (0.61 with grouping), assigns 0.02, recalls no more
  in fights (0.16-0.24 an order, the production keys mostly): Gary's orders go to role groups of single units,
  seldom a whole key; recalls would need decisions per key, a learned layer's. Pipe check with grouping, again
  pushed: commands a minute 82.6 -> 89.4, reach 70% -> 72%, net -72 -> +68.
- **Grouping on 96 games again, with the cheaper keys (2026-10-22; rounds GT40c+d, GZ40c+d against L40a+b).**
  Gary-T grouping: won 45 and lost 17 (57 and 8), its gap +0.015 (-0.013 to +0.042), 10-15 minutes further. Not the
  key layout: on keys 1-3 with the buildings' keys as before (GT3a against L40a and GT40c, 48 games) it won 28 and
  lost 6 (27 and 7 with four keys; 34 and 0 without). Its cost is 25 actions a minute (army recalls 7.3 against 1.5,
  assigns 4.3) with macro decisions and production about the same; at 10:00 more workers (40 against 38) and less
  army (24 against 26, the Zerg's 14 against 11): where it loses is the next question, the fights' and army orders'.
  Gary-Z grouping: Gary-T won 48 and lost 13, Gary-Z's gap -0.046 (-0.082 to +0.028), 15-20 minutes nearer
  (-0.073, -0.102 to -0.017): with the first 96 (49 and 10), into gz1 (gz0.2).
- **Control groups in the role model (2026-10-22).** ingest/key_labels.py: 71% of the pros' labeled fighters are
  on a key at the moment (800 games). eval/key_coherence.py, held-out pro fights: key-mates take the same decision
  80.4% of the time (shuffled within the snapshot 68.2%), and a new decision is shared with a key-mate 73.0%
  (58.9%). The model's decisions drawn unit by unit: 30% and 25%, the same with the key as an input (train/roles.py
  --keys: no gain, shown or hidden). Drawn with one shared random number per key (option together): 77-78% and
  75-76%, matching the pros' decisions as often (32%). Pipe check with grouping and together, again pushed:
  commands a minute 76.5 -> 80.4, reach 67% -> 72% (with independent draws 70% -> 72%), recalls 0.24 a fight
  order (0.16): role groups still split keys (a fire group leaves out key-mates already attacking). The suite:
  the in-contact start with grouping +107, with together +110 (+32, -20 to +83, over +93), against RoleAhead's +141.
- **gt1dev (2026-10-22):** gt1+att+refresh+incontact+batch+grouping+together, the configuration learning is trained
  and judged in (AGENTS.md); gt1 stays what plays until gt1dev beats it in 96 paired games.
- **Why Gary-T's grouping lost (2026-10-22): a key's members taken from the wrong place.** Its attacks were
  unchanged (eval/attacks.py: the share of the army there 0.52-0.54 against 0.46, late arrivals 0.25 alike, trades
  73-75 / 78-83 supply a game against 75 / 83); its fighters died more, 82-85 a game by 16:00 against 74, close-in
  deaths 15 against 9.9 (the records' deaths). The motor took a key's members from the selection it saw before
  pressing Ctrl+key (a moment old) while the interface records what the key got when the press landed; where they
  differed, a recall gave its order to units it wasn't meant for. Read from the interface (b3025c0; that commit's
  message says the parity hashes changed: they didn't, the parity games are 8 minutes, too short for the keys),
  and no key's own selection during a fight: against gz0.2, 48 games (B50a, GT50a), Gary-T with grouping won 30 and
  lost 7 (29 and 5 without), its fighters dead 75.8 a game against 85.6, the gap -0.010 (-0.038 to +0.020); the
  Zerg's games unchanged (B50a played as GZ40c). The second 48 (B50b, GT50b) decides it.
- **Whole keys ordered, and the learned grouping provider (2026-10-22).** Option bykey (a key whose deciding
  members agree gets one order, 75% of it in the fight enough): of 599 keys in fights over 376 turns, 35% had
  members outside the fight, 27% were split, 19% all kept; keys ordered whole 48 -> 53, recalls 0.32 -> 0.43 a fight
  order, the pipe check's reach 70% -> 71%: diminishing. The pros' groups (after 5:00): Terran 3 / 5 / 9 units
  (quartiles), one type 74%; Zerg 5 / 9 / 12, one type 36%. Of the pairs of a pro's fighters 23.5% share a key; "the
  same type" names them with precision 41% and recall 92% (within 6 tiles 46% and 83%): the pros split a type over
  keys. The grouping model (train/groups.py; gary/policy/grouping_model.py: per-unit embeddings, a pair's chance on
  one key), held out, partitioned into up to 4 keys: precision 0.529, recall 0.779 (F1 0.630) against the type
  rule's 0.530 and 0.706 (0.607). As the provider (option grouplearned): groups of 5 at assign in a smoke game.
- **The first roles RL run that moved and improved (2026-10-22/23).** From gt1dev with the again model (its pipe
  check: again pushed, commands a minute 76.6 -> 80.5, reach 67% -> 72%; the suite +87): train/roles_rl.py, 60 rounds
  of 32 training scenarios x 8 plays, the KL anchor to the start, Gary's own order rate as the budget input.
  Validation (the training games' 35 scenarios) -209, -218, -214, -207, -177, -152 (round 50, kept), -193.
  - Gate 1, moved off its start: on the held-out suite KL 0.094 (median 0.067); fire 13.7% -> 16.0%, again 20.9% ->
    22.3%, reposition 12.7% -> 10.0%, retreat 2.6% -> 1.6%.
  - Gate 3, the held-out suite (79 scenarios from games held out of training, not the ones that chose the
    checkpoint; they did choose gt1dev's composition): +195 against the start's +87 (paired +105, +44 to +169; attacks
    +179, harassment +82, outnumbered +95, mixed +70), against RoleAhead's +141 +50 (-51 to +151).
  - Gate 4, 96 paired games against gz0.2 (R50a+b, option rl1, against D50a+b, gt1dev): won 54 lost 15 against 39 and
    16; score a game (won 1, ahead at 30:00 0.75, even 0.5, behind 0.25, lost 0) +0.089 +- 0.076 (better in 34, worse
    in 20); the Terran's gap -0.046 (-0.070 to -0.012), fight features -0.035 (-0.056 to +0.002; 0-20 minutes a
    little further, 20-25 nearer), macro -0.053 (-0.098 to -0.002); the Zerg's gap against it -0.094. Against gt1
    (B50a+b): +0.042 +- 0.075 (better in 28, worse in 18), the gap +0.008, fight -0.035 (-0.065 to +0.004), macro
    +0.035 (-0.009 to +0.070): not yet beating gt1; macro the side to watch.
  - Gate 2, attention (eval/attention.py, first 10 minutes; whole games in brackets): commands into the fight a fight
    minute, fight moments attended, army units reached: gt1 31, 63%, 33%; gt1dev's start 78-81, 91-92%, 60% (75-78,
    89-90%, 49%); rl1 75-79, 89-90%, 57-60% (74-75, 88-89%, 48%); pros 105, 70%, 58%. The RL learned better decisions,
    not more orders; the hands work (the model in contact, several orders a turn, again, control groups) is what
    moved attention, reach to the pros' level, and attended now passes the pros' (91% against 70%): a new tell.
  - Adopted into gt1dev (gt0.6dev); gt1 stays what plays.
- **The hands in fights don't explain the macro lean (2026-10-23).** eval/attention.py now splits a player's
  actions during fight moments (into the fight, other units, macro: Train, Build, Morph, Tech, Upgrade), Terran,
  whole games: gt1 (B50a+b) 70-72 actions a fight minute, 41-44% into the fight, 27-30% other units, 28-29% macro;
  gt1dev (R50a+b) 107-108, 69-70%, 13%, 17-18%; pros (100 games) 149, 63%, 23%, 14%. Macro actions a minute in fights
  gt1 20, gt1dev 18-19, pros 21; out of fights 12.7, 12.3, 24 (the pros' counts include repeated train clicks). The
  attended share (moments with a fight command in the last 2 s; gt1dev 89%, pros 67-70%) measures how evenly the
  fight is touched, not how the hands split: gt1dev added actions in fights rather than taking them from
  production, and Gary's macro gap is outside fights. Option attend (the share held at the pros', fight.py), 48
  games against gt1dev's R50a: won 28 lost 6 (31 and 8), attended 77%, commands into the fight 52 a fight minute
  (75), reach 37% (48%), macro in fights 19.9 (18), out of them 12.6 (12.4), the macro gap +0.049 (-0.008 to
  +0.100), the fight gap -0.019: not adopted. The gap table now carries the attention features (fight_attended,
  fight_reached; the modern pros' rows and rounds R50, B50, A50 augmented).
- **Gary-T's macro gap is its build style, and what the standard one exposes (2026-10-23).** The macro features
  furthest from the pros (gt1dev and gt1 alike): Factories 1.0 against 3.5 at 20-30 minutes, machine shops, armories,
  refineries and gas behind, no vultures or goliaths, mass marines; bases and supply behind late (supply 90 against
  119 at 15-20 minutes); never supply-blocked (0-2% against 9-19%). The strategy selector (gary/agent/selector.py)
  played style 4, two-Barracks marines (1.5% of pro Terran games: gas in 10%, no Factory), in all 96 games of R50a+b:
  its table was counted on 2026-10-13 in eval/vs_rush.py's short games against rushes on the 1.16 maps, and the
  ladder maps borrow it. The style rolled by the pros' shares instead (option styleroll: standard bio 82%), 48 games
  each: gt1dev won 26 lost 8 (R50a 31 and 8), the macro gap +0.025 (-0.042 to +0.091); gt1 won 23 lost 13 (B50a 29
  and 5), the macro gap -0.009. Not adopted. On the standard plan Gary mines gas and doesn't spend it (873 banked
  against the pros' 520 at 10-15 minutes, 1,681 against 978 at 15-20), its supply and army fall behind (76 and 30
  against 119 and 60 at 15-20): its marines and medics wait for busy Barracks (0.43 and 0.33 a macro decision) while
  the model almost never adds production (Barracks 0.8% of its odds, no Factory in its top). The rush style hid that
  by taking no gas. Next for macro: production added when money floats, as the pros do (the model's odds on its own
  banked states), before the style is let go. Also: the first producer's rule after the opening needs a buildable
  one for a Terran too (3c661c8; it set the goal up to 2,955 times a game in lost games).
- **Production idle, and the fill decoder (2026-10-23).** Production buildings at 10-20 minutes (sampled every 10 s,
  completed with nothing in training): gt1 (B50a, 24 games) Barracks 5.0 idle 44%, Factories 0.9 idle 79%, Starports
  1.5 idle 79%, Command Centers 2.0 idle 93%, minerals banked 2,354; pros (30 games) 4.9 idle 47%, 1.8 idle 46%, 1.4
  idle 54%, 2.8 idle 84%, 1,932. The Barracks are the pros'; the Factories and Starports are half as many and mostly
  idle (no tanks, vultures, goliaths, vessels: the high-supply units, the unspent gas). The model's next decision adds
  production as often as the pros' when money floats (10% of Gary's against 6%). Bases: the natural at 5 minutes
  (pros 3), the third at 15 (12), as many games reaching them; workers even to 12 minutes, 41 against 46 at 16. Option
  fill (keep_busy: an idle producer makes the model's likeliest of its own units when its last can't be made), 48
  games each: gt1 won 24 lost 10 (B50a 29 and 5), Factories still idle 78%, the gap +0.007; gt1dev won 22 lost 4
  (R50a 31 and 8), the gap +0.064 (+0.023 to +0.099). Not adopted (reverted on the branch); the units it picked were
  likely held by the pros' per-minute caps (macro.py plan, capped). Why an idle Factory isn't given a unit: at 10-20
  minutes (B50a) Siege Tanks and Science Vessels are in the model's top odds a fifth of the time (1,211 and 1,291 of
  5,736 decisions), and are passed over as "also waits: its Factory (Starport) is busy" 505 and 423 times, "the
  pros' cap for this minute" for vessels about 140, "no Machine Shop" 40. At the one "busy" moment replayed the
  Factory was training (queue 1), so the wait can be real; the idle 79% are the moments between, when keep_busy (on in
  gt1) doesn't train from it. keep_busy's own records (a probe on the branch, one 20-minute game of gt1): of 50
  records of an idle Factory or Starport with nothing trained, the Factory had no last unit in all 50 (it had never
  trained: a Siege Tank needs the Machine Shop it lacks, a Goliath the Armory, a Vulture is under the dice floor),
  the Starport in 20; the Science Vessels "waited for a busy Starport" while the idle one had no Control Tower. Two
  tries had no effect in that game (both discarded): the add-on built first when the chosen unit needs it (the tank's
  odds, about 1.5%, rarely win the dice) and the fill decoder above. Next: the add-on and the unit decided together
  (pros put a Machine Shop on their Factory at once), daytime work.
- **The second roles RL run (2026-10-23), from rl1 by the same recipe** (60 rounds, 32 x 8 plays, the KL anchor to
  its start, Gary's own order rate as input). Validation (the training games' scenarios) -152, -124, -161, -103,
  -131, -119, -53 (round 60, kept: runs/roles_rl/20261023_014740, option rl2). Gate 1: KL to rl1 0.082 (median 0.052);
  fire 16.2% -> 16.9%, close in 3.3% -> 4.1%, reposition 8.4% -> 6.5%, again 23.5% -> 24.1%. Gate 3, the held-out
  suite: +255 against rl1's +195 (paired +67, +3 to +134; mixed +142, +33 to +279; outnumbered +120, +5 to +258;
  attacks -22, harassment +10 within noise), against RoleAhead's +141 +117 (+12 to +231): the model in contact now
  beats the hand rules there. Commands a minute 72.5, fighters reached 86% (rl1's 74.9, 90%). Gate 4, 96 paired games
  (Q50a+b against R50a+b): won 49 lost 16 (rl1's 54 and 15), better in 21 and worse in 25 (-0.036 +- 0.069), the gap
  +0.044 (+0.014 to +0.068) further from the pros, macro +0.055 (+0.014 to +0.096), fight +0.028; against gt1 even on
  results (+0.005 +- 0.082), macro +0.091. Failed: not adopted (gt1dev stays on rl1; the run from rl2 begun meanwhile
  was stopped). Its hands are rl1's (105 actions a fight minute, 18-19% macro, macro 12.3-12.6 a minute out of
  fights); it fights more: the Terran's fighters dead by 16:00 64.6 -> 72.0 and 79.0 -> 90.6 a game, the Zerg's 92.4
  -> 101.7 and 107.1 -> 113.6, a slightly worse trade over the game, net value in 45-second scenarios. Credited with
  the 10 s after each decision (train/roles_rl.py HORIZON_S), the run learned to win the next exchange; the next run
  from rl1 credits 20 s.
- **Room for add-ons (2026-10-23).** A Machine Shop was passed over as "no Factory without an add-on, with room for
  one" in 134 of 192 of gt1's and gt1dev's games (33 of them 50 times or more), and at 16:00 of 16 gt1 games 10 of 25
  Starports had no Control Tower and no room for one (Factories 1 of 14, Command Centers 0 of 31): no Science
  Vessels, the Starports idle. Pros put a Machine Shop on their first Factory in 63% of games, a median 77 s after
  ordering it. Option addonroom (terran_executor.py addon_fits: a Factory, Starport or Science Facility placed only
  where its add-on fits), 48 games against B50a: won 29 lost 5 (the same), the gap -0.012 (-0.035 to +0.009), macro
  -0.012, fight -0.011; Control Towers 0.77 -> 1.25 at 12:00 and 0.82 -> 1.42 at 16:00, vessels 0.2 -> 0.5 and 1.8 ->
  2.9, Machine Shops 0.32 -> 0.39 and 0.39 -> 0.45, supply at 16:00 87 -> 96, Starports idle 79% -> 72%. A placement
  fix (the hands): into gt1 and gt1dev.
- **The third roles RL run (2026-10-23), from rl1 with each decision credited over 20 s** (--horizon 20; rl2's 10 s
  won the suite and failed in games). Validation -152, -115, -98, -104, -99, -107, -83 (round 60, kept:
  runs/roles_rl/20261023_043828, option rl3). Gate 1: KL to rl1 0.089; fire 16.1% -> 17.2%, keep 29.9% -> 32.1%, stand
  7.2% -> 6.0%, reposition 8.6% -> 7.2%, close in barely (3.4% -> 3.6%; rl2 3.3% -> 4.1%). Gate 3: the suite +268
  (paired +71 over rl1, +12 to +135; +122 over RoleAhead, +15 to +238). Gate 4 (T50a+b against R50a+b): won 47 lost 8
  (54 and 15), better in 22 and worse in 26 (-0.013 +- 0.076), the gap +0.069 (+0.038 to +0.096), macro +0.094, fight
  +0.031; the trade a little better than rl2's (the Terran's fighters dead +4-6 a game, the Zerg's +5-9). Both runs
  moved the macro side of the gap the same way; R50a+b were played before the first producer rule's tidy (3c661c8),
  Q50 and T50 after, so a fresh gt1dev baseline on the same code was played (D60a+b): its 96 games came out the same
  as R50a+b's to the last outcome and gap (the tidy only acts in lost games and changed none here). Both runs fail
  gate 4 against it: rl2 -0.036 +- 0.069, the gap +0.043 (macro +0.055); rl3 -0.013 +- 0.076, the gap +0.068 (macro
  +0.094). Not adopted; gt1dev stays on rl1. The lesson for the next run: the held-out suite rose twice (+67, +71,
  the checkpoint chosen on another split) and the games didn't follow; the 45-second net value is not the game's.
  Next: a 24-game check of a checkpoint in games before the suite is trusted, and the macro features the fights
  move (unit counts, bank) watched as part of the objective.
- **The macro check: the model has the pros' tech, Gary's states and keep_busy don't (2026-10-23).** The Terran macro
  model (runs/macro_shared/T_all_20261019-102013) at 10-20 minutes on held-out modern pro TvZ states (237 player-games,
  17,541 states): next decision against its odds, Siege Tank 7.7% / 8.9%, Goliath 5.8% / 4.8%, Vulture 3.3% / 3.3%,
  Factory 1.1% / 0.9%, Machine Shop 0.7% / 1.1%, Science Vessel 4.0% / 4.1%, Marine 33.9% / 31.6%, Medic 6.5% / 7.8%:
  calibrated, factory tech included. On gt1's own states (B60a+b re-simulated and encoded as it reads them), its odds
  turn from mech: Siege Tank 5.3%, Goliath 0.0%, Vulture 0.1%, Factory 0.5%, Marine 42.3%, Medic 10.2%; the style input
  hardly matters (style 1 or none: Marine 39-41%, Vulture 0.3-0.4%, Goliath 0.0%). Gary's next decisions: Marine
  71.2%, Medic 4.6%. At 10-20 minutes it trained 86 marines a game of which the macro's decisions chose 44: the rest
  were keep_busy (an idle Barracks repeats its last unit), spending the minerals the model's tech waits for and
  writing marines into the history it reads, so its states look a marine player's and the model, as a pro would
  there, asks for more. Options: nobusy (keep_busy off) and busymix (its unit drawn from the model's odds among the
  producer's own), rounds NB60a and BM60a against B60a.
- **gt1 gt0.7, gt1dev gt0.8dev: keep_busy removed, the model's decisions queued (2026-10-23).** keep_busy trained
  units the model didn't decide (an idle Barracks, Factory or Starport repeated its last unit, half the marines at
  10-20 minutes). In its place the model's units that wait for a busy producer are queued, up to one per producer of
  their kind, and the macro decides on meanwhile; nothing else is trained. Round Q60a against B60a (48 games, seeds
  4004-4009): won 30 lost 5 (29, 5), the gap 0.28 (0.29); the macro's decisions 13,203 to 19,347; the pros' caps
  refused 0.46 options a decision (0.63). With the pros' history as the macro's input as well (option prohist, round
  QP60a): won 21 lost 10, rejected. Folded into gt1 (gt0.7) and gt1dev (gt0.8dev); the old manifests that had it
  play without it now. The rounds against these are next: the macro rules registered in AGENTS.md off one at a
  time (options nocaps, noscvcap, nodepots).
- **Static defense as a commit-model input (2026-10-23; army dataset v4, branch static-input).** The commit model's
  map counted a sunken as any building, so the static guard (a veto) stood in for it. v4 adds the enemy's ground
  static defense the player knew of as a channel; two Terran commit models on the same 4,841 player-games (2,459
  modern, 2,500 old), with and without it: held-out order AUC 0.770 / 0.767, engage AUC 0.768 / 0.768. Per defended
  spot (the group's weight against the static defense within a cell), engages per 1,000 held-out ticks, pros /
  with / without: under 0.75x 1.66 / 1.09 / 1.02, 0.75-1.5x 2.67 / 2.15 / 2.02, 1.5-3x 4.45 / 3.40 / 3.19, 3-6x
  4.31 / 3.09 / 2.95; undefended spots 127 / 141 / 134. The pros don't avoid static defense much: 29% of their
  engages into it are under the guard's 1.5x, so the guard departs from them; both models follow the pros'
  pattern already, the channel adds ~5%, and both engage defended spots ~25% less than pros. So Gary's losses
  into sunkens (human game 4) aren't a decision the pros' data teaches against; what it attacks with and how it
  fights there is the likelier cause. In games, 48 each against Q60a (gt0.7): the guard off (NG70a) paired -0.052
  +- 0.064, the gap +0.011 (-0.010 to +0.038), lost / killed supply a game 70.9 / 92.6 (60.6 / 87.8), into sunkens
  0.49 of attacks (0.45): a close call, its second 48 on seeds 4010 (NG70b): over 96 paired games the guard off is
  worse, better 8 worse 23, -0.062 +- 0.044 (the gap +0.010, -0.012 to +0.032), so the guard stays. Why it helps
  while the commit model matches the pros offline: not what Gary knows (on its ticks it knows an undefended enemy
  building 99% of the time, pros 97%; it engages as often, 152 per 1,000 ticks to their 155), but where its attacks
  go: 0.21 of Gary's engages into defended cells, the pros' 0.11. By what began each attack (eval/attacks.py, Q60a):
  the commit model's engages 79, into sunkens 0.38 (0.45 with the guard off); the scripted hunt 62, into sunkens
  0.63; units sent elsewhere 46 (0.35); no recent order 32 (0.50). The scripted hunt (a stop-gap) goes into static
  defense most, and the guard doesn't screen it. With the static model (NGS70a) against
  its control (NGC70a, the same games without the channel): -0.073 +- 0.102, medics along 0.76 (1.05): not adopted.
- **The decoder's bank escape is load-bearing (2026-10-23).** Off (NB70a): paired -0.203 +- 0.123, the gap +0.126,
  macro +0.208; minerals banked at 5-15 minutes 1,124 (pros 511, gt1 387), fewer Barracks, Starports, medics. While
  the model's favorite waits for a busy building (its queue full) it stays the macro's goal: the model isn't asked
  again, so it never adds production with the bank. The escape covers for that. Two units queued per producer as
  well (QD70a): 675 banked, -0.078 +- 0.126, the gap +0.052 (+0.009 to +0.090). Pros queue in the building and pay
  up front; that is the next option.
- **The model's units queued in the building (option gamequeue, 2026-10-23).** A unit the model decides is trained
  into a busy building's own queue, two deep, and paid for, as pros queue; each building's queued units are
  counted as coming. 48 games against gt0.8's (ND70a): won 34 lost 6 (31, 8), paired +0.073 +- 0.145, the gap
  +0.025 (-0.008 to +0.046), macro +0.010; macro decisions 20.1 a minute (pros 20.3, without 16.0), at 5-15 minutes
  workers 35.8 (pros 34.9), medics 2.43 (2.68), the bank 425 (511). A close call: its second 48 (GQ80b against
  ND70b). With the bank escape off as well (GQN80a): -0.130 +- 0.152 against gamequeue alone, 801 banked, the gap
  +0.076: the escape still holds something up. Its second 48 (GQ80b against ND70b): +0.062 +- 0.144; over 96 paired
  games better in 34 and worse in 21, +0.068 +- 0.102, won 56 lost 12 (49, 18), the gap +0.018 (-0.010 to +0.037),
  macro -0.002: no harm, Gary's decisions nearer the pros', and the fault it replaces gone, so adopted: gt1 gt0.9.
- **gt1dev without the stop-gaps (gt0.10dev, 2026-10-23).** Stop-gaps stay in gt1 while they help, but the learning
  is judged where the models own the decisions: gt1dev drops the army's (the static guard, the hunt with its search
  and scans, the overlord hunt) and the macro's (the pros' caps, the SCV cap, the bank escape), option unscripted.
  Its new baseline (DV90a+b, 96 games): won 7 lost 29 (gt1 gt0.9: 56 and 12), paired -0.346 +- 0.095; the gap
  +0.128 (+0.088 to +0.153), the macro features +0.256 further from the pros, the fight features -0.060 nearer
  (-0.095 to -0.031). What the learned layers have to close; most of it is the macro's (the bank escape alone cost
  -0.203, the caps -0.109).
- **The hunt imitated, and the macro's decoder in gt1dev (2026-10-24).** Army dataset v5 (4,933 TvZ games: the
  enemy buildings known still standing, the cells own units haven't been in lately). The hunt's advice in 96 gt1dev
  games (HL100a+b, seeds 4016-4027: 1,126 pieces of advice, 677 labels joined to ticks) as engage labels, x4, on the
  same games as its control (option cmv5): on the pros' held-out ticks both as before (order AUC 0.768 / 0.766);
  where the hunt acts (held-out games, 33 labels) the model engages 0.80 of the time and has the hunt's target among
  its five likeliest cells 0.94, the control 0.16 and 0.12. In games: options cmhunt, cmv5 (HM100a, HC100a). The
  macro in gt1dev (no caps, SCV cap or bank escape) drawing from the model's odds (option dice, DC100a against DV90a):
  lost 6 (13), paired +0.080 +- 0.127, the gap -0.078 (-0.113 to -0.018), macro -0.123 (-0.176 to -0.035), the bank
  at 5-15 minutes 263 (pros 511, the favorite 843): nearer the pros, its second 48 next (DC100b). With rethink too (a
  unit whose queues are full not held as the goal, RT100a): macro -0.197, but bases 1.28 at 5-15 minutes (pros 2.10)
  and lost 17, -0.165 +- 0.128 against dice alone: everything spent on units, nothing saved for a Command Center; not
  adopted.
- **The v5 commit models in games, and a sharper dice (2026-10-24).** Army dataset v5 for all 16,549 TvZ games (the
  rest built in 1.5 h at 3.7 s a game); train/commit.py --max-ticks fits them (60 a player-game). On all of it, with
  the hunt's inputs: held-out order AUC 0.767, engage 0.778 (the deployed commit model: 0.756, 0.770, on its own
  test set); the hunt imitated as well, 0.765 / 0.776, following the hunt where it acts 0.81. In gt1dev (dice)
  against the deployed commit model (DC100a): the control (cmv5all, HB120a) -0.069 +- 0.096, the hunt model
  (cmhuntall, HA120a) -0.090 +- 0.098, the two -0.031 +- 0.093 apart; its commit engages into sunkens 0.45 (0.35).
  The third time v5 models trail the deployed one in games while scoring better offline: offline AUC doesn't
  predict play here. A v5 model with the deployed inputs (cmv5plain: 0.767 / 0.778 offline, HP120a): -0.176 +-
  0.099 against the deployed one, -0.104 +- 0.112 against the channels' control, lost 19: not the channels but the
  data or the recipe. Candidates for the review: the era mix (all of v5 is 85% old games; the macro's measured recipe
  weights the modern ones ten to one, docs/DATA_ERAS.md), 60 ticks a player-game over many games against 400 over
  fewer, and what the deployed model was selected on (games, not its offline score). dice07 (dice from the odds raised to 1/0.7; D07110a against DC100a): +0.011 +- 0.145,
  at 5-15 minutes workers 33.3 (27.8, pros 34.9), bases 2.01 (1.73, 2.10), army 21.2 (17.7, 25.1), bunkers 2.59
  (3.43, 0.55), the macro gap +0.073 (-0.004 to +0.148): a close call, its second 48 D07110b. Over 96 paired games
  (against DC100a+b): +0.037 +- 0.099, won 14 lost 24 (7, 18), the gap +0.031 (-0.018 to +0.072), macro +0.048: no
  measurable difference, more games decided either way; not adopted (gt1dev keeps dice).
- **Gary-Z's first overlord, and the pairing balanced (gz1dev gz0.3dev, 2026-10-24).** Why gz1dev supply-blocked:
  gz1's macro reads the pros' decision history early in the game (so its own drone-heavy past doesn't feed back), and
  that history holds their 9-supply overlord, so the model thought one was coming: at 60 s on 9/9 it gave Drone 0.99,
  Overlord under 0.01, and gz1dev (no overlords ahead) had supply 9 at 2:00 (pros 16.9), blocked 67% of that minute.
  Option ownsupply: the supply provider's history entry always Gary's own. Against gt1dev (ZO110a against ZD110a, 48
  paired games): the Terran's score -0.286 +- 0.117; the Zerg winning or ahead at 30:00 in 22 of 48 (4), the Terran
  in 17 (37); blocked at 3:00 1% (42%); drones at 10:00 30 (20, pros 28). Adopted, gz1dev gz0.3dev; the pairing's
  balance criterion (each side ~25%+) met on these 48, and over 96 (ZO110a+b): each side wins or leads in 36 games
  (38%), 24 even at 30:00. The pairing gt1dev gt0.11dev against gz1dev gz0.3dev is the TvZ self-play baseline. The Zerg's gap moved +0.208 further
  (to gz1's), with its counts much nearer the pros': the gap and the counts disagree for the Zerg as well.
- **The TvZ self-play pairing's baseline, and the hunt model under dice (2026-10-24).** gt1dev gt0.11dev against
  gz1dev gz0.2dev, 96 games (ZD110a+b): the Terran won 30 and lost 5, ahead at 30:00 in 39, the Zerg ahead in 12,
  even 10: the Zerg wins or leads 18% of games, the Terran 72%, short of the ~25% each side needs before self-play.
  gz1dev's gap is much nearer the pros than gz1's in the same pairing (0.316 against 0.509) but it is weaker:
  without overlords ahead it supply-blocks (8 drones at 2:00 on 9/9). The Zerg's strength is the next fix before
  self-play. The hunt model imitating only undefended targets (cmhuntu, HU110a) against its control (cmv5, HV110a),
  both gt1dev with dice: +0.068 +- 0.092, won 3 (1), but its commit engages into sunkens 0.46 (0.23) and lost /
  killed 32.0 / 53.9 a game (18.1 / 45.8): leaving the defended targets' labels out didn't stop it learning to
  attack bases, which have sunkens. Both v5 models are weaker than gt1dev's deployed commit model (-0.021 and
  -0.096 against DC100a); not the training set's size (the deployed one had 740,740 ticks, these 912,760).
- **gt1dev's early economy under dice, and two decoder tries (2026-10-24).** With dice gt1dev builds bunkers it
  can't use (2.7 at 5-10 minutes, pros 0.5: 8.9 decisions a game at 4.5% odds, the caps that refused them off in
  gt1dev) and fewer SCVs (18.8 at 5:00, pros 25.3, gt1 24.0): a drawn building the money doesn't cover holds the
  macro while it saves and the Command Center idles; the high bunker odds come in those states (11-17 workers at
  3-6 minutes, a pro behind and bracing). Gary-Z's park (a building waits for its money, other decisions go on)
  deadlocked a Terran at 6 SCVs (an SCV parked behind a building whose goal waited on it): not for Terran. nosave
  (a goal the money doesn't cover isn't held, the model asked again; NS110a against DC100a): lost 23 (6), -0.271 +-
  0.106, workers at 5-15 minutes 25.6 (27.8), army 12.1 (17.7), although the macro gap -0.075 nearer the pros: not
  adopted. Twice now (rethink, nosave) the macro gap moved toward the pros while the games got worse: a decoder
  change is judged on results first, the gap second.
- **Commit pairs locked (2026-10-24).** The pairs tool's training arms now hold their forced commitment, reversed
  at the pros' hazard (Terran engages 10.5% per 5 s falling to 5.6% past 40 s, moves 16.3% to 4.9%; Zerg 14.8% to
  4.8%, 16.3% to 3.8%), then the model decides (e1 and e2's arms were undone by the model within 14-34 s). Two bugs
  found on the way: every paired episode had failed since the refactor (the bots' deep copy met the race's data
  module; a race is now shared), and GARY_TCOMMIT_MODEL pinned gz1's commit model too. One 12-minute game: three
  pairs, locks held 0-78 s, the replay check identical. Pilot for the power check (CP110a, 24 games of 20 minutes,
  gt1dev against gz1dev): 233 training pairs, replay checks 70 of 70 identical (the arms differ only by the forced
  decision and what follows from it: the dice are shared after it); engage arm less move arm +0.7 a pair, sd 7.2
  (at 5-10 minutes -1.1, 10-15 +1.6, 15-20 +1.0). The minimum detectable effect written down before any gate (2
  standard errors): 1.0 at 200 pairs, 0.5 at 1,000, 0.3 at 2,000. The gate for a model learned from them: offline,
  it picks the better arm on held-out moments more often than chance (the interval above 0.5); then 96 paired
  games. gt1dev never lost to gz1dev in the pilot (7 won, 17 at time), so the collection plays gz1 (the Zerg won 29
  of 96 against gt1dev): moments against a side that concedes teach a model to beat only it (SELF_PLAY.md rule 3).
  First offline read (train/commit_pairs.py, CP110a and the first 52 games of CP120a: 626 labelled pairs, 78 held
  out, every replay check identical): the deployed commit model picks the better arm 0.50 of the time (train 0.495,
  held 0.51: chance); fine-tuned 20 epochs, train 0.55 and held 0.40 (0.29 to 0.51): it overfits, the gate not
  passed. Engage won 54% of the training pairs and 62% of the held-out ones: at these moments a constant "engage"
  beats the model, the one signal so far. Under the MDE (1,000-2,000 pairs for an effect of 0.3-0.5) nothing
  state-dependent is learnable yet; the collection goes on, and a 90 s net value may be too short a horizon for a
  commitment (a candidate: the value network's read at the end of the arm). The collection done (CP120a, 96 games of
  20 minutes against gz1: 905 training pairs, replay checks 266 of 266 identical): the engage arm less the move arm
  +0.82 +- 0.51 a pair (at 10-15 minutes +1.25 +- 0.92), but engage won only 49% of them: when engaging wins it wins
  bigger. On all 1,004 labelled pairs (CP110a+CP120a; 125 held out) the start model picks the better arm 0.50 (train)
  and 0.53 (held), at chance; fine-tuned, train 0.54 and held 0.46: it overfits again, the gate not passed. With
  these inputs and this outcome, which arm wins isn't predictable from the moment; next, the outcome's horizon (the
  value network at the arm's end) and the advantage's size rather than its sign.
  The advantage's size, offline (issue #28; 1,119 pairs from 117 games, cross-validated by game, judged as a
  policy's value: the mean of d where it engages): always engage +0.81 a pair (+0.42 to +1.24); engage at 10-15
  minutes +0.52; the deployed commit model's own preference +0.48 (it engages at 55% of the moments); ridge on the
  model's numbers +0.82 (engaging at 98%), gradient boosting on the numbers and grid summaries +0.84 (63%). Nothing
  in the state predicts where engaging pays (held-out correlation at best +0.09); engaging beats moving at nearly
  every stage (8-10 minutes the exception, -0.31 +- 0.91). The one-parameter improvement it supports: engage's odds
  over move's raised (option engagebias, doubled, keep's odds as they were), judged in 96 paired games (EB120a+b
  against DC100a+b): a 90 s gain may cost the game later, and only games show that.
- **Dice in gt1dev, the hunt model in games, gz1dev (2026-10-24).** Dice over 96 paired games (DC100a+b against DV90a+b):
  lost 18 (29), +0.058 +- 0.083, the gap -0.073 (-0.096 to -0.035), macro -0.105 (-0.143 to -0.049): adopted, gt1dev
  gt0.11dev. The hunt model (cmhunt, HM100a) against its control (cmv5, HC100a), both gt0.10dev: paired -0.005 +-
  0.101, but won 8 (3) with T ahead at 30:00 in 7 (10): of the games it led it finished 8 of 15, the control 3 of
  13; the gap -0.047 (-0.075 to -0.010) nearer the pros. It copied the hunt's attacks into sunkens too (0.51 of its
  commit engages, the control's 0.35, lost / killed 51.6 / 60.1 a game against 38.9 / 45.2), so a second model
  imitates only the hunt's undefended targets (cmhuntu, --rule-undefended: 401 of 644 labels; where they act it
  engages 0.50, the target in its five likeliest 0.95). gz1dev gz0.2dev: gz1 without join, sweep, press, the hunt,
  the caps, the bank escape and overlords ahead (unscripted for both races; its overlord posts and mutalisk raid
  kept, no model moves those units, and the decoders: without the tech floor dice drew a Lair and a Hydralisk Den on
  8 drones); one game: 14 drones at 7:00 (gz1 22).
- **The static guard as a nudge (gt1 gt0.10, 2026-10-23).** A spot the group doesn't outweigh 1.5 times keeps
  (the group / 1.5 times its defense) squared of its odds instead of none (option softguard): 96 paired games
  against the veto (SG80a+b against ND70a+b) better in 4, worse in 9, -0.016 +- 0.035, the gap +0.001: no
  measurable difference, so the veto goes. The hunt's labels: gt1dev runs it in advice mode (huntadvice; play
  unchanged, its decisions in the records), army dataset v5 has its inputs (enemy buildings known still standing,
  cells unseen), train/commit.py --rule-labels imitates it. A stop-gap is imitated (the rule labelling the model's own states) where the
  pros' data lacks the behaviour, the model has the rule's inputs and can express its action: the hunt first (pros
  leave at GG, so their replays hardly show a game finished), once the commit model sees which enemy buildings are
  gone and which bases weren't seen lately.
- **Three macro rules off, and why two of them still help (2026-10-23).** 48 games each against gt1 (Q60a: won 30
  lost 5, gap 0.28). The pros' caps off (NC70a): won 22 lost 10, paired -0.109 +- 0.152, the macro gap +0.070
  [+0.013, +0.125]. The SCV saturation cap off (NS70a): won 24 lost 4, -0.047 +- 0.102, the gap +0.041 [+0.007,
  +0.076]. Depots ahead off (ND70a): won 31 lost 8, -0.021 +- 0.129, the gap -0.029 [-0.060, +0.004], a close call:
  its second 48 on seeds 4010 (ND70b against Q70b): over 96 paired games no measurable difference (won 49 lost 18
  against 50 and 16, paired -0.018 +- 0.101, the gap -0.013, -0.040 to +0.012, macro -0.017, -0.057 to +0.017), blocked
  at 5-10 minutes 14% of the time (pros 9%, with the rule 2%), so the rule goes: gt1 gt0.8, gt1dev gt0.9dev (Gary-Z's
  overlords ahead stay until measured). The review of the two that help: with the caps off, turrets at
  10-20 minutes 16.4 (pros 7.1, gt1 9.2), medics 1.3 (4.4); with the SCV cap off, workers 48.6 (42.4, gt1 39.0). Gary
  takes 18 macro decisions a minute at 5-20 minutes, the pros' production commands 20 (200 modern games), so the
  rate isn't it. The decoder is: of the 1,174 turrets, 677 were picked while the model's favorite, a Marine, waited
  for Barracks already holding their one queued unit, 329 because they had half its odds and 341 only because the
  bank was over 400; of the 2,206 SCVs, 1,335 the same way. A pro whose next action is a Marine queues it in the
  building. The caps clean up after the decoder's substitutions, so the substitutions go first: options nobank
  (the bank escape off, now a registered stop-gap) and qdepth2 (two of the model's units queued per producer).
- **Baselines and the promotion check (2026-10-23 morning).** gt1 gt0.6 (B60a+b, seeds 4004-4009 and 4010-4015,
  against gz0.2): won 49 lost 10. gt1dev gt0.7dev (D70a+b): won 48 lost 13. Paired: gt1dev better in 21, worse in 24
  (-0.018 +- 0.063); the gap +0.010 (-0.016 to +0.040), the fight features nearer the pros (-0.039, -0.064 to -0.009),
  the macro further (+0.043, +0.003 to +0.088). gt1dev doesn't beat gt1: gt1 stays what plays. These four rounds are
  the baselines for the next changes to either.
- **Rules for learning beyond imitation (2026-10-22):** docs/SELF_PLAY.md, from an outside review of the RL
  results above checked against this file: the pipe check, the model owning the decision, branches locked by the
  pros' hold lengths, the minimum detectable effect before a gate, counterfactual learning offline; the league
  loop's PPO paused.
- **The roles RL: what success is, written before it starts (2026-10-21).** Reinforcement learning of the roles model
  (gt1's fight before contact) from the model that can re-order (option refresh), with Gary's own order budget from
  the arbiter during the episode as its budget input (not the pros' rate) and a drift penalty to the imitation
  policy, on the train split of the state-selected suite (eval/scenario_kinds.py: attacks, mixed fights,
  outnumbered units, harassment, from whole modern games). Its gates, in order; a gate failed ends the run's
  claim there:
  1. the policy moved off its starting model (its decisions on the held-out suite differ, by KL and by role
     shares);
  2. the attention numbers moved toward the pros' without passing them: commands into the fight per
     fight-minute (gt1 45, pros 160) and the share of units reached (33%, pros 60%);
  3. net value on the held-out state-selected suite, paired against the starting model;
  4. 96 paired games on the ladder maps (the gap per window with its spread), the macro side of the gap watched
     (more hands in fights can starve production).
  The stop rule: if it doesn't move off its start after the planned compute, the next step is a dedicated offset
  head imitated from the pros' re-issues (where a re-issued order goes), imitation again, not more RL.
  The planned compute: 60 rounds of 32 training scenarios x 8 plays (362 training scenarios, 35 kept out for
  validation). A first run started with RoleAhead (the hand rules in contact: the trained weights never reached
  most fight commands) and was stopped at round 37 (validation -223, -191, -227, -269; KL to the start 0.038);
  it doesn't count. The episodes and the 96 games play with the model deciding in contact too (option
  incontact: roles.py learned/v1 with draw, refresh and the RL log).
  Before a run, the pipe check (train/roles_rl.py --pipe 10: the start model against itself with +2 on the move
  roles' logits, refresh on, 10 held-out scenarios over the kinds, 4 plays each): move decisions 13.9% -> 54.2%,
  but commands a minute 60.0 -> 62.9, units reached 67% -> 61%, net -65 -> -344. The decisions reach the fight
  (which units move), not its volume: the fight layer takes a turn every 12 frames and gives one group order of
  one role a turn, and the next turn waits for the hands to carry it out; 42 of 90 possible turns in 45 s, 72%
  of them giving an order (pushed: 35 turns, 90%). The pipe is closed for commands and reach: the run waits.
  Opening it (2026-10-22), in order. (1) Cheaper orders, control groups by type with recalls (grouping on, the
  Terran's 4 army keys, keys adopted at the takeover): start 57.6 commands a minute, 66% reached, pushed 62.3 and
  61%; a probe of the half-second slots (10 scenarios): a fight turn 42%, no fight 29%, actions pending 12%, a
  build 8%, the last fight order still being given 5%. Recalls are rarer than first counted: 7.4 a minute in a
  14-minute game (the earlier 42 counted key 4, then also the Command Center's); an order seldom covers a whole
  key. (2) Several orders a turn (option batch: up to 4, a group per role by the same priority, given one after
  another as the hands come free, production's turn after the batch): start 63.7 and 68%, pushed 74.1 and 67%
  (with grouping; without it 65.2 / 70% and 74.0 / 67%): the push moves commands 16% (5% before), not reach.
  The turn is no longer the cap: 0.93 orders decided a turn (pushed 1.21) with room for 4, the APM bucket full at
  almost every order (none under one token), 14-20 queued orders dropped stale in 10 scenarios. What caps it now is
  what the decisions mean: keep gives no order, fire or close in to a unit already attacking gives none, and a
  re-issue (the pros' same order with the target moved, half their fight commands) exists only for moves, by
  refresh. On the full suite (79, 8 draws): the model deciding in contact (incontact) +47 against RoleAhead's +141
  (paired -95, -179 to -10); with batch +93 (+48 over incontact, +5 to +94). Reach didn't move: the run waits.
- **Tried the same night, not adopted (48 games each, against G11/G13):** gz1+floor (no commitment drawn under 15%,
  as Gary-T's): Gary-T won 24 (19); gz1+noalarm again on the new economy: the Zerg won 2 (5) but was ahead at
  the limit in 6 (3), a wash; for Gary-T, a decoder floor down to 1% while 1,500 minerals float (`spend`): the
  same results, the bank at 20:00 1,797 to 2,713. Gary-T floats from 15:00 because its favorite, Marine, is
  at the pros' cap for the minute and nothing else is over 5%; it leads 18-19 of 48 games at 20:00 without
  finishing them: the army's decisions, not money (human game 4's item 1). gz1 loses its zerglings two at a
  time to groups of seven (42 a game by 12:00, mostly at 9-11 minutes), with zergling speed at 3:48: a matter
  for the fight and army layers.
- **Next for Zerg:** the remaining opening gap (natural at 1:53 to the pros' 1:38, the first gas late at
  2:27 to their 2:00, zerglings 2.5x the pros' by 7:00 and mutalisks a third); finishing games it leads;
  late-game spending.
- **Next:** human game 4's list (above), starting with the attack's execution. After the refactor, human game 4's list
  (above), in this order; the workers' part is done (workers off their work, the main's miners drawn under
  mutalisks):
  1. the attack's execution: medics with the bio, no attack-move into sunkens, a regroup in front of the
     natural, a retreat when hit, called off when outnumbered; plans that adapt to what is seen. Measured
     (eval/attacks.py, 2026-10-20): against gz1 the medics come along and losing attacks are called off as
     the pros' are; the trickle (4x the pros') and attacks into sunkens (38% to 28%) remain. Finishing games:
     the hunt in gt0.2 (below);
  2. SCVs pulled into fights (3 pulls at 7:47-7:53) and SCVs and marines that felt AFK against the early
     zerglings;
  3. micro of single units in a group, not only group orders. Measured 2026-10-21: in fights Gary-T gives single-unit
     orders as often as pros (20-24 a fight-minute, pros 19.4; human game 4 too), orders per fighter per minute
     match theirs in every window and its units don't stand under fire: the hands aren't the gap ("none per unit"
     came from a whole-game APM breakdown). What is off is which unit does what, the roles model's (learned);
     one order a fight turn left 34-41% of its per-unit moves undone. Letting a turn give more orders whenever the
     hands were free again (most urgent first, dropped at the next turn): 96 paired games, won 55 (57), lost 10
     (8), the gap unchanged, fight commands a minute 52.8 (57.7), 2,400-3,800 orders a round left at the turn's
     end: the hands have no room in fights (camera, macro, the army layer share them). Deleted. Where the room
     goes: 1.54 selection actions per order in fights to the pros' 1.14 (first 10 minutes, 100 modern pro games
     against 48 of gt1); the pros' selections are 71% control-group recalls, Gary-T's 62% drag boxes and 14% clicks,
     19% recalls. Next for the hands: the army kept on control groups (the pros' habits, from their commands);
  4. turrets overbuilt (16) at the army's cost;
  5. vision: mutalisks hiding behind the natural's cliff (a scan or a floating building), Comsat energy
     kept for lurkers (the hold-position trick at 14:40), Science Vessels;
  6. the late game: expansions past the natural, hunting the Zerg's expansions, tech;
  6b. overlords: Gary-T leaves out-of-position overlords alone even when they are free kills (its marines shoot
     one a few times, then wander off); a Terran hunts them early with one or two marines, a supply block for
     the Zerg (the user, after human game 4). Done as option `lords` (army_rules.py LordHunt, 2026-10-20): the
     two nearest free marines hunt an overlord with nothing guarding it; overlords killed by 8:00 8 -> 21 in 16
     games with no more marines lost, and in 48 games against gz1 Gary-T won 21 (19), lost 3 (5). In the default,
     gt1 since (below);
  7. building placement (packed depots, room to build and move; a later phase in docs/REFACTOR.md).

### Between v0.16 and v0.22 (2026-10-10 to 10-14): the belief model and other experiments

- **What it is:** a learned estimate of the enemy's army from what the player has seen of it
  (`train/belief.py`; the first piece of the opponent model, ARCHITECTURE.md §6.9). Inputs: game
  time, each enemy type in sight now and the most at once over the last 30 s, 90 s, 3 min and
  ever, seconds since an enemy army unit was in sight, own supply and workers. Outputs: the enemy's
  counts of each army type, workers and town halls (unbiased expectations) and its army supply's
  median and 80th percentile. Trained on the macro training set (every second of 14,130 pro TvZ
  games: both players' real units and what each could see), Terran's point of view.
- **Why:** the army model's fight estimate knows only the enemy army seen lately, so it takes
  the fog for empty (on Python, 8 units walked out into 14 zerglings nobody had seen: +1%).
- **Results (held-out games, the Zerg's army supply):** off by 5.3 on average against 6.5 for
  "what was in sight over the last 3 minutes" (about what the fight estimate uses) and 7.1 for
  "the average for the game time"; the Zerg has 8+ supply more than believed in 11.9% of samples
  (15.9%), at 6-10 min 8.0% (15.2%); with the 80th percentile, 4.0% (6-10 min 1.9%), believing 5
  supply more than there is on average.
- **First use, not adopted:** groups the army model sends away from home hold when the belief's
  likely enemy army (80th percentile; less what was seen in 3 minutes, or all of it) outweighs
  them (`gary/policy/belief.py` keeps the same inputs during play; options `belief`,
  `belief_total`). Replayed pros on three maps lost 12 and 13 of 144 (v0.17 11), holding 637 and
  854 times; on Python Gary's army at 8:00 stays 2-5 alive of ~20 built either way: the excursions
  into unseen zerglings explained the games looked at, not the losses.
- **Where Gary's army dies (Python, 48 games, to 10:00):** 51% by a Command Center, 27% within 25
  tiles of one, 12% out on the map; 70% attacking. Against pro Terrans' deaths in fights there
  (451): alone (no other own fighter within 6 tiles) 48% (pros 12%), outnumbered 2:1 or worse 53%
  (29%), the fight around it 2 against 3 (6 against 7). Those dying alone are new: 13 s old
  (median), 62% under 20 s; last ordered by the fight roles (fire 39%, close in 23%) or the
  reinforce rule (18%): during a fight at home, each marine out of a Barracks goes in alone. Hand
  rules against it did worse (new fighters to the nearest bunch, `reinforce_pack`; loose units
  back to the pack, `regroup`: Zerg v2 lost 12-14 of 72, v0.17 9). This is the fight and army
  decisions' job, and the measure (share of deaths alone, pros 12%) is the yardstick for a learned
  one.
- **The learned fight decision, steps 1-2 (2026-10-10 night):** defense episodes
  (`eval/episodes.py`: the home defenses for 120 s, new units joining; v0.17 −49, deaths alone
  39%, pro 20%; the hand-rule variants paired with v0.17 as in games: regroup −12.6 ± 8.8,
  reinforce_pack −0.4 ± 3.4, deaths alone unchanged). The role model got post/join/hold and per-unit
  age, distance to Gary's biggest bunch, its size. v0.17's rules distilled into it (one draw of the
  552 training episodes): 88% of decisions, but in play −72 on the held-out episodes (−49) and −177
  on RL's validation episodes (rules −120); RL from there flat after 10 rounds (−174, stopped).
  A learned override of the rules instead (per unit join, hold, or the rules; RoleOverride,
  `v017override`, `train/roles_rl.py --override`): it starts as the rules (−127), and 60 rounds of
  RL ended where it started (−124); the drawn override probabilities went down (join 3.6% → 3.0%,
  hold 3.6% → 2.6%, also for new units far from the bunch): RL too found holding new units back
  doesn't pay, the third sign after the two rule variants. A closer copy of the rules (four draws
  per episode, 417k decisions, a bigger network: 88.9%; validation −148, rules −120), then 60
  rounds of RL on the whole role decision: −131 at rounds 10-20 (best), drifting to −166 and back
  to −138. Held out, the best against v0.17's rules: 239 episodes −54.5 (−48.7; −5.8 ± 14.8), deaths
  alone 38% (39%); replayed pros on three maps lost 12 of 144 (11), Zerg v2 10 of 72 (9), won 7
  (4). **The first learned fight decision on par with the hand rules over whole defenses, not
  beyond them**: the same ceiling as RL on the 45 s scenarios. Not adopted (v0.17 keeps its rules;
  the model: `v017roleml` with `runs/roles_rl/20261010_055912/model.pt`, not in git).
- **The role model given terrain (2026-10-13):** per unit, the ground under it and the way to its
  bases (gary/policy/roles.py `terrain_features`: ground height, the passage's width there, in the
  main or not, toward and how far the nearest Command Center, how far the nearest bunker and the
  main's ramp, from anywhere on the map), in the pros' data and in play. Imitating the pros' roles
  it adds nothing (held out 77.7% with it, 78.0% without; pull back and retreat within noise): a
  role is what a unit does for the next half second, and that follows the fight in front of it;
  where units stand is the skills' and the army's doing. v0.19's rules copied with it (four plays of
  the 552 training defenses, 120 s: 90.3% of decisions), then 60 rounds of RL as before: validation
  −149 at the start, best −124 (round 20), −135 at the end; v0.19's rules on the same defenses
  −118. The same ceiling as without terrain: three tries at learning the role choice now end at
  the rules. Not adopted (`v019roleml` with `runs/roles_rl/20261013_115621/model.pt`, not in git).
- **Where the army waits at home, per situation (2026-10-13; `eval/post_choice.py`):** the post's
  spot as a decision with six options (3 tiles from the Command Center toward the zerg, as now; at
  it; 6 tiles out; by the minerals; new: just inside the main's ramp, just behind the bunker), on
  every home defense with its situation at the start (`eval/episodes.py` now records it: minute,
  bases, bunkers, army, zerg in sight, the fight at the main or not). Learned on the 552 training
  defenses (2 draws; cells of the two situation features that did best across halves of them:
  bases and zerg in sight), judged once on the 239 held out (4 draws), paired with today's spot:
  the rule +2.0 ± 7.0; always 6 tiles out +13.3 ± 8.5 (training +7.0 ± 7.1; v0.10's −4 ± 8),
  by the minerals −4.8 ± 6.1 (training +13.0), at the ramp −6.2 ± 7.4, behind the bunker −5.7 ±
  3.8 (bunkers are rare this early: mostly the same as now). Where the army waits in a fight at
  home is worth less than the episodes can tell apart; a rule per situation picked from 22 found
  noise. Not adopted. With the role choice (three tries) and terrain, the home fight now looks
  near what the hand rules can get from these decisions; the losses in games come from elsewhere
  (where the army goes between fights, what is built).
- **Why v0.19 loses (2026-10-13; `eval/losses.py`):** its 336 games of the selector test played
  again (335 the same result: deterministic), every army death with its place, its order and the
  layer that last ordered it, both armies every 30 s; the 30 losses against the other 306. Not
  production: lost games built as many fighters (14 by 6:00, others 15) but kept none alive (1 at
  5:00, others 7); their deaths come alone twice as often (47%, 23%) and in the main (44%, 26%).
  Two kinds of loss. (1) 20 against the replayed pros, 18 of them hatch first with a pool at
  2:01-2:08 whose first zerglings (4+ supply) reach Gary at about 3:30: Gary's early-pool posture
  went on before them in 2 of 15 such losses (never in 11), against 19 of 34 games with the same
  early zerglings that it survived; its marines then die one by one in the main (71% alone). A
  pool at 2:02 after a hatchery isn't an early pool to the posture's rules, but its zerglings come
  as early. (2) 10 against Zerg v2 (7 on Python): Gary's army is even or ahead until about 6:00 and
  loses the big fight at 6-7 minutes (21 army deaths a game at 5:30-8:00, others 12), more of it
  in the field and the main, less at its natural.
- **The early-defense switch, learned (2026-10-13; `train/early_switch.py`, `gary/policy/switch.py`):**
  when to go into the early-pool defense, from paired games: each training game (seeds 1-48, the
  six opponent and map sets) played as the rules play it and again with the defense forced at
  2:00, 2:30, 3:00, 3:30 or 4:00 (deterministic: the switch the only difference); outcome won 1,
  lost 0, at the time limit the value network's chance. A 3-level tree over the posture's facts
  learned where switching pays (across seeds +0.044 a game, all 4 folds positive; its biggest leaf:
  a second hatchery and a pool seen, +0.12). Held out (432 paired games, seeds 101-148, and Zerg v2
  also 149-220): losses 40 → 39, wins 74 → 64, outcome −0.011 ± 0.027. The replayed pros' fast
  pools 20 → 0 losses, Zerg v2 20 → 39: it is on by 4:30 in 430 of 432 games, false alarms in 285
  of the 287 games without early zerglings (the rules: 154 of 285), SCVs at 6:00 25.2 → 24.1, army
  9.2 → 11.1. In effect "always defend early", which trades the losses one for one. Not adopted
  (posture option: `force_at`, `switch`). Every opponent in the pool attacks early and none
  punishes over-defending (TODO.md item 1): the decision is re-learned once a greedy Zerg is in the
  pool (item 4).
- **Zerg v3, a greedy Zerg (2026-10-13; `gary/bots/zerg_v3.py`, TODO.md item 4):** on Zerg v2's
  code, through the human interface: natural 2:24, third base 3:36, Hydralisk Den 6:06 (8 seeds on
  Fighting Spirit; Python 2:30, 4:12, 6:00), 23 drones at 6:00 and 42 at 8:00, about 10 hydralisks
  and a few zerglings at 9:00, when everything attacks with no retreat; its army waits at home and
  defends its hatcheries; on a third of the seeds its first zerglings poke at Gary's natural and go
  home. Its buildings are placed as a player places them (a lone drone clicked, walked there by a
  minimap order, the building placed, placed again if it doesn't appear): v2's way failed most of
  the time for buildings away from its main (a box around a mining drone takes its neighbors; a
  build order to a mining drone becomes a move; the click's scatter or Gary's scout on the spot;
  the money spent while the drone walks), and a drone inside an extractor isn't seen, so gas was
  oversaturated. Against v0.19 (48 seeds on each of two maps): lost 0, won 7, the rest undecided
  at 12:00; Gary's SCVs at 8:00 35-38. It punishes over-defending, as the pool didn't: the learned
  "always defend early" switch against the rules on it (48 paired games) won 2 (rules 8), outcome
  −0.18 ± 0.06, SCVs at 6:00 24.9 (27.1).
  With v3's games in the strategy selector's table (each style, 48 seeds on two maps; 12-minute
  games: no style lost, style 4 won the most), Python's pick moved from style 1 to 4 (all three
  maps now 4); on fresh seeds on Python, style 4 against style 1: Zerg v2 won 9 lost 4 (6, 7), the
  replayed pros won 4 lost 5 (2, 8), v3 won 11 lost 0 (13, 0). Played to 25 minutes (30 seeds on
  each of two maps), every game against v3 ends by 19:30 and Gary wins all 60 (median 15:30 on
  Fighting Spirit, 13:30 on Python): too weak for outcomes to say which decision was better. The
  early-defense switch re-learned with v3's 12-minute games: across seeds +0.021 (one fold of four
  negative), still "switch on a second hatchery and a pool seen"; not tested further until v3 is
  stronger and its games run long.
- **Zerg v3 with zerglings and mutalisks (2026-10-13):** hydralisks aren't what Zergs play against
  Terran (marines and medics beat them), so v3 now plays the standard 3-hatch mutalisks: a Lair
  from 3:20 (4:24 median), a Spire (6:00), mutalisks from 7:30 as one flock (hotkey 9) raiding the
  mineral line with the least anti-air seen, home when outweighed or half lost, defending its
  hatcheries; from 6:00 a standing zergling army (12, 4 more a minute), two sunken colonies at
  its natural, zergling speed; everything at 11:00. Its groups attack-move everywhere but when
  running away (on move orders its units walked through Gary's army without shooting), and it has
  pro-speed hands (a test opponent's hands are the benchmark's choice; the other Zergs keep
  theirs). Buildings that morph (a Lair, sunkens) go through the interface's new building morph.
  To 20 minutes (20 seeds on each of two maps): Fighting Spirit won 10, lost 0, Python won 11, lost
  2; most of the 17 undecided are decided in all but the last building (the Zerg down to one
  building against 67-79 of Gary's army, or Gary down to 2-3 buildings and no army), about 5
  close: Gary wins roughly 70-80%.
- **The scout, again (2026-10-13):** v0.19's scout leaves the Zerg's base once it has found it; the
  touring scout (option `watch`: shift+clicked waypoints around the Zerg's main and natural until
  hunted or 5:00, then outside Gary's natural) over the whole pool (fresh seeds, 432 games: the
  9-pool, Zerg v2 on two maps, the replayed pros on three, Zerg v3 to 20 minutes on two): lost 30
  (25), won 127 (133), against v3 won 54 (64). What it sees isn't used by any decision (the
  posture's rules look for an early pool only, the macro hardly reacts), and an SCV off minerals
  costs. Not adopted alone: it belongs with a decision that reads it (the early-defense switch).
- **Where the first marines gather (2026-10-13):** pros' marines at 3:00-4:00 stand at the top of
  the main's ramp (33%) and in the main (30%), at 5:00-6:00 at its bottom (30-34%) and the natural
  (17-19%), in the open 3-8% (60 pro games played again, every 30 s); Gary's at the natural (42%)
  and in the open (25%): its rally was fixed from the start 6 tiles in front of the natural, so
  every marine walked there alone from the main. Alone (no other fighter within 6 tiles) about
  the same (14%, 17%). Option `rally_ramp` (executor.py rerally): the rally inside the main's
  ramp, at its bottom from 4:45, in front of the natural from 7:30, production buildings
  re-rallied when it moves.
- **The early-defense switch with the touring scout (2026-10-14):** the switch re-learned on
  v0.19+watch's games (seeds 1-32, the whole pool with Zerg v3 to 20 minutes, 248 games, 803
  forced moments) with the scout's facts added (the most drones seen at once, an extractor seen,
  when the pool and the first zerglings were first seen; the posture remembers them, the switch
  reads them: `gary/policy/switch.py` facts). The tree uses them as a player would (more than 12
  drones seen: don't, -0.26; a hatchery but at most 4 drones seen: do, +0.15), but across seeds it
  gains nothing: -0.013 a game (4 folds -0.026 to +0.047); simpler trees at best +0.021 (12 folds,
  sd 0.065, 8 positive): within luck. In hindsight a switch per game would gain +0.21: there is
  something to learn, not at 248 games against these opponents. Parked until Zerg v3 plays
  several builds (then what the scout sees decides more).
- **Zerg v3 plays three plans (2026-10-14):** by seed: the 3-hatch mutalisks; lurkers (a Hydralisk
  Den with the Lair, Lurker Aspect at ~6:40, hydralisks morphed into 6-10 lurkers by 10:00 that sit
  burrowed in front of its natural and go with the 11:00 attack, burrowing when Gary's units come
  within 7 tiles and coming up to move on); and a 12-pool pressure (zergling speed first, 12
  zerglings outside Gary's natural attacking when its defense looks weaker until 4:45, then drones
  and mutalisks). The interface can burrow and unburrow now. Against v0.21 to 20 minutes (20
  seeds on each of two maps), by the end: the mutalisk plan Gary-favored in about 10 of 12, the
  pressure in about 10 of 14 (three Zerg-favored on Python), the lurkers the hardest (on Python 3 of
  7 Zerg-favored, two of them with 130-150 army against 0-18). Each plan a game, and they differ,
  so what the scout finds can matter now. One game stopped on an engine error (a unit squeezed
  between edge minerals and the map's edge; counted as an error, not a result).
- **Does Gary react to the Zerg's plan? (2026-10-14)** Its decisions against Zerg v3's three plans
  (40 games to 20 minutes): the same tech whatever it faces: Engineering Bay ~6:00, turrets from
  ~7:00 (15 decided a game against mutalisks, lurkers and the zergling pressure alike), Comsat
  ~7:00, Science Vessels late in a third of the games; and no scans at all against lurkers (scans
  only hunted buildings). The macro model plays a generic game; burrowed lurkers it can't see kill
  its marines. Option `detect` (fight.py DetectRule): a lurker that burrowed where Gary's units see
  it is remembered, and with Gary's fighters near it and a Comsat at 50 energy, scanned.
- **Do Gary's hands cost it games? (2026-10-14, TODO item 2)** The arbiter's wait events in the
  selector test's 666 games (87 lost): the fight layer waits for the hands about 0.3 s a minute,
  lost games no more than the rest (1 s+ fight waits 0.06 a minute in losses, 0.15 in the others).
  No case yet for giving the deciding layers inputs about the hands; slow or failed execution
  (missed selections) is another question.
- **The early-defense switch against Zerg v3's three plans (2026-10-14):** re-collected on
  v021+watch (seeds 1-40, the whole pool, v3 to 20 minutes: 312 games, 973 forced moments). The
  tree over what Gary knew (the scout's facts among them) gains +0.089 a game across seeds (12
  folds, ± 0.019, 11 positive), but so does the simplest rule, the defense at 2:00 or 2:30 in every
  game (+0.089): the best tree is one level deep. What it found is that the rules switch too late:
  forced at 2:30, the replayed pros +0.26, v3 on Python +0.10, Zerg v2 +0.02, v3 on Fighting
  Spirit 0, the 9-pool 0. Option `defend_early` (the defense from 2:30 unless the rules switched
  first), held out next.
- **A fix that moved every game (2026-10-14):** the minimap clamp that came with Zerg v3's plans (a
  minimap click's scatter kept on the map) changed v0.21's games against every opponent: a
  scout's click scattered past the map's edge used to be refused by the game (the scout skipped
  that waypoint), now it lands. Found by replaying one game at each commit since the baseline:
  the commit before replays the old game exactly, and so does that commit with the clamp undone.
  The old v0.21 numbers are no yardstick since; a new baseline at today's code, the same 432
  games: won 118, lost 30 (the replayed pros 18 of the losses, Heartbreak Ridge 10).
- **`defend_early` and `detect` on the same seeds (2026-10-14):** `defend_early` lost 15 (30) and
  won 115 (118): paired by seed, 26 losses became undecided and 9 undecided games losses (2 wins
  losses), while wins moved both ways about evenly (32 to undecided, 31 from undecided). The
  replayed pros 18 -> 3 losses; Zerg v2 on Fighting Spirit 5 -> 7, the one set where defending early
  costs. `detect` scanned in 18 of the 32 lurker games (2 to 17 scans each), and those games went
  differently after the first scan, but not one result changed in 432: against the lurker plan 29
  of 32 games are still undecided at 20 minutes with or without the scans (the mutalisk plan 17 won
  of 32, the zergling pressure 24 of 32). Gary doesn't end the lurker games; seeing the lurkers
  isn't what's missing.
- **Why the lurker games don't end (2026-10-14, `eval/lurker_games.py`):** played again with the
  full game state (v0.22, 8 lurker seeds on Fighting Spirit and Python, 20 minutes): Gary's army
  loses 141 units a game against the Zerg's 8 lurkers and kills 2.1 of them; 66% of its deaths
  with a lurker within 7 tiles, 67% on the enemy's side. The lurker stood in Gary's view every
  time: the engine wrapper's `visible_to` was the sprite's sight alone, so Gary saw burrowed
  lurkers it hadn't detected (fixed in `env/gary_env.cpp`: a cloaked or burrowed unit is visible
  only to those who detect it), and `detect`, waiting for a lurker to vanish, rarely scanned. It now
  also scans where Gary's units lose hp with no zerg in sight that could have hit them (as pros do
  when lurkers burrowed out of sight). Same 8 games: 10 scans a game (0), lurkers killed 4.0 (2.1),
  army deaths 125 (141), a third of them by lurkers undetected; won 2, lost 0 (2, 1). Games without
  burrowed units are unchanged (checked against Zerg v2 and the replayed pros). Still: Gary stays
  on 2 Command Centers all game with 1,000-2,000 gas unspent from 14:00, and its army keeps walking
  into the lurkers on the enemy's side.

### v0.29 (October 2026): the pros' moves before contact

- **What it is:** v0.28 with a role model learned from 600 pro games' Terran units whenever a seen
  enemy fighter was within 12 tiles, before contact as well as in it (`ingest/response_dataset.py`:
  1.3M labels before contact; decisions, keep or a new role, with a where head: each unit's move over
  the next half second); the fight layer watches 12 tiles. The hand rules decide in contact, the model
  before it (`roles.py` ahead/v1): units it would reposition, pull back or retreat go where it moves
  them (the user's examples: marines into the mineral line as zerglings come, a lone marine running to
  the buildings).
- **Why:** the user's "AFK" was anticipation, not reaction time; the classifier's fight rows (lone
  outnumbered units never moving away, pros 11-18%).
- **Results (226 pro home defenses not from its games, 8 draws):** a tie with v0.28 (-2 +- 6 net value a
  defense); the model's roles in place of the rules everywhere lost 52 +- 11, with or without its where
  (the rules were tuned on these scenarios over twenty versions); the 12-tile watch alone +4 +- 5.
  The neural role policy is the league's learner; v0.29 the version that carries it at no cost.

### v0.28 (October 2026): the army layer not held by noise

- **What it is:** v0.27 whose army layer never holds a plain move (positioning) on the fight
  estimate, and holds an attack only when it expects to lose 5+ points of army share or 5+ more than
  staying (v0.27: 2 and 1).
- **Why:** the classifier's tell: Gary-T's army stood still 78% of 5-7 minutes (pros 37%); at 5-6
  minutes it held ~90% of its decisions, 92% of them within 2 points of staying, noise for an estimate
  correlating 0.40 with outcomes.
- **Results:** against the replayed pros (3 maps x 8 seeds, 12 minutes) won 5, lost 0 (v0.27 2, 0);
  the full 72-seed judge against Zerg v2 not run.

### v0.27 (October 2026): the second human game's turrets and Comsat

- **What it is:** v0.26 with fixes from human game 2 (the user's Zerg won at ~17:00; parity
  clean). Against mutalisks the answer's turrets come first, two in each mineral line, the one
  nearest the zerg first (the user's mutas came through the natural; neither line had a turret when
  they got there: the model's own turrets, which go where any building does, away from the
  minerals, went up first). They stand just past the minerals, off the SCVs' paths (between the CC
  and the minerals, mining SCVs stood in the spot and the game refused it: seed 108 failed 11 of 12,
  now 0 of 4). SCVs repair a threatened turret, 3 of them (`repair_turrets`). Live placement keeps
  add-on spots free (the check was skipped without the terrain, which the live client lacks: turrets
  stood where both Comsats go, and no scan when lurkers came), and a pending add-on's building isn't
  kept busy (an SCV queued just before its Comsat order: the game refused the add-on, twice).
- **Idle SCVs (the user: "many idle SCVs all over"; 11-17 of 30-39 idle from 8:00 to 12:00):** the
  scout came home with a move and stood by the minerals for the rest of the game (the mining routine
  skips the scout; now released once home, shared code); a builder's next order is queued on a
  mineral field and the mining routine adds the other idle workers on the screen (`builder_back`);
  and the bunker-load rule sent the same marine every 2 s all game (a drag box loaded 5 marines into
  4 slots and the one left out was never counted: 694 loads in 8 games, now 63; shared code), taking
  Gary's hands from everything else. Headless idle SCV time 13% to 9%.
- **Headless (8 seeds, Zerg v3, 25 minutes):** won 6, lost 0 (v0.26: 7, 0); seed 105 nearly lost
  after the 8:00 hunt (v0.26 won it): no detectable harm on 8 seeds, which can't tell 6 from 7.
- **Stim, learned (the user: "never heard stim once"; the roles model has no stim):** from 400 pro TvZ
  games re-simulated in OpenBW (its stim timer, now in gary_env's observation), every half second
  each marine and firebat in a fight is a sample: what the Terran sees around it (HP, its stim,
  neighbors and medics, the zerg near it by kind, the fight starting: HP lost, the zerg closing in,
  more coming, neighbors stimmed) and whether it was stimmed in the next second (1.05M samples, 6.3%
  stimmed; ingest/stim_dataset.py). Gradient-boosted trees (train/stim.py): test AUC 0.76 on held-out
  games; at the pros' rate of stims, 24% of its picks are right (6.5% base). fight.py StimRule asks it
  every half second and stims the group around the units it picks, once Stim Packs is done (Gary
  remembers its own stims: the live client doesn't report them). On 8 seeds against Zerg v3 to 25
  minutes: won 7, lost 0, 5-8 marines a stim (without: 6, 0; seed 105, nearly lost, won at 13:00).
- **Live check (v0.27 vs the Remastered computer Zerg, 18 minutes, 2026-10-16):** 10 stims in the
  replay (groups of 2-10 marines, from 8:45), a Comsat that scanned burrowed lurkers at 14:01, 5
  bunker loads in the whole game, alive and growing at 18:00 (45 SCVs, 3 Command Centers).
- **Not addressed:** marines between main and natural picked off; the AFK feel.

### v0.26 (October 2026): the first human game's losses

- **What it is:** v0.25 with the fixes for what lost its first game against a human (the user's
  Zerg over LAN, 2026-10-16, Gary dead at 12:17; the bridge matched OpenBW over the whole replay, so
  every failure was Gary's): against mutalisks an Engineering Bay, an Academy and 2 turrets in every
  base's mineral line (answers.py; v0.24 had dropped the muta answer after it lost bot games, and
  Gary built no turret against the user's mutas); no worker pull against flyers or more than 8 zerg supply (x2) (SCVs pulled against 4 to 23 zerg,
  mutas among them, 26 SCVs to 0). Shared code: a parked build keeps its own SCV (executor.py).
- **Judged by:** the next games against humans. Headless only checks it breaks nothing: 6 seeds
  against Zerg v3 (14 minutes) won 2 (v0.25 2), lost 0 (0); the muta turrets fired where mutas came,
  no worker pulls where v0.25 pulled 12 and 14 times.
- **Not scripted, on purpose (the user's call):** leaving the rush defense (the model asked only
  marines, SCVs and depots from 4:45 to 8:13). How much to commit to defense is learned: pros pull
  back just enough to survive, and the nuances are too many for a rule (a catch-up rule, gas, Academy
  and the natural after the early pool, was written and removed).
- **Not addressed (the user's notes):** Gary plays like someone AFK: it waits for the opponent to
  act, and acts on what is, never on what is coming (lings running from marines turn toward the
  mineral line: the marines should go back then, not when SCVs are hit); a lone SCV walks out to
  build into the lings that killed the last one; kiting beats it. Learned pieces of the self-play
  skeleton (TODO, Direction 3).

### v0.25 (October 2026): hunts that finish games

- **What it is:** v0.24 whose hunts (the army going for the Zerg's last buildings once no zerg army
  has been seen for a minute) work: the order's camera on a unit (an army in two groups had its
  median point between them, where the camera found nobody, so the order timed out and the army
  stood for minutes by the last hatchery), up to 8 drag boxes (4 left 29 of 77 units standing), a
  scan of a remembered building the army hasn't settled in 20 s, and a remembered building seen no
  longer theirs forgotten (an Extractor destroyed is a neutral geyser with its tag: the hunt went on
  to geysers by Gary's own bases for minutes). From here Zerg v3's games run to 25 minutes.
- **Why:** of v0.24's 44 undecided games against Zerg v3 (20 minutes), 12 ended with the Zerg on
  one building, Gary's army 3x the Zerg's or more in 31.
- **Results (seeds 101+, 432 games, Zerg v3 to 25 minutes, against v0.24 on the same seeds):** won
  139 (132), lost 12 (10): 16 undecided won, 7 wins undecided, 2 wins lost (Zerg v3's zergling
  pressure, both after a hunt at 8:30-9:15 to "an enemy building" by Gary's own natural; not traced
  yet). Zerg v3: mutalisks won 26 (25), lurkers 20 (17), the pressure 27 lost 2 (29, 0). v0.24 itself
  wins 71 of 96 against Zerg v3 by 25 minutes, 51 by 20: many undecided games were slow wins.
- **Not adopted, and what it says about the army model:** the camera on a unit for every army
  order lost games (120 games: won 25 lost 7 against 28 and 4); 8 boxes for every order sent more
  units into fights (with a scan before every hunt: won 99 lost 15 of 432). The army model's
  orders to an army in two groups used to fail; carried out, they cost games. Gary far ahead
  without winning is the army model's decisions, not their execution.
- **Hunts gated by the belief model (2026-10-15, option `hunt_belief`):** v0.25's two new losses
  (Zerg v3's zergling pressure) came after a hunt at 8:30 with 19 supply into 16 unseen (the belief
  model: 17 expected, 28 at its 80th percentile; v0.24 lost its army there too and survived the 11:00
  all-in by chance). The hunt only with an army at least the belief's 80th percentile: won 125 (139),
  lost 19 (12) of 432. No pressure losses (31 won, 0 lost; 27, 2), but the lurker games won 10 (20):
  the belief, learned from pros' Zergs, expects 40-50 army supply at 15:00 where Zerg v3 has none,
  so late hunts wait; and Zerg v2 lost 14 (8) without the early hunts. Hunting on "no army seen"
  pays against these Zergs more often than not. Not adopted.
- **What the army model's orders do (2026-10-15, `eval/army_orders.py`):** 120 games of v0.25 played
  again (Zerg v2, the replayed pros, Zerg v3 to 25 minutes), 3,594 orders, each with the 45 s after
  it (the group's supply lost, the zerg's lost within 10 tiles of it). On average an order trades
  well (3 zerg supply lost per Gary's; against Zerg v3 2.3), but 6% of them, attacks that lose half a
  group of 6+ supply or more, hold 53% of Gary's army losses and trade 0.67. The model's fight
  estimate and confidence don't tell them apart (median +0.004 against +0.010 for all orders). Per
  game: Zerg v3's lurkers 9.2 such orders (71% of the army lost), its mutalisks 1.8 (45%), its
  zergling pressure 1.5 (45%), Zerg v2 1.0 (43%), the replayed pros 0.6 (27%). The army model attacks
  into lurkers it can't see: burrowed and undetected, they aren't in its inputs.

### v0.24 (October 2026): lurkers answered

- **What it is:** v0.23 that reads the Zerg's plan from what it has seen (`gary/agent/answers.py`:
  a lurker or its egg, or a Hydralisk Den with a Lair and no Spire, means lurkers; a Spire or a
  mutalisk, mutalisks; with nothing known by 6:30, a scan of the Zerg's main, again at 8:30) and
  answers lurkers as pros do, as goals to the macro: a Factory, a Machine Shop, Siege Mode, tanks,
  a Starport, a Control Tower, a Science Facility, two vessels, Irradiate; turrets one a base.
- **Why:** the model played the same game against every plan. Against Zerg v3's lurkers (by 10:00)
  Gary decided 11.7 turrets, never Siege Mode, a Science Facility at 11:24 in two thirds of the
  games; pro Terrans against lurkers (1,321 TvZ games of 12 minutes or more, Lurker Aspect by 9:00)
  2.4 turrets, Siege Mode at 8:42 in 96%, a Science Facility at 9:12 in 95%, Irradiate at 11:36.
  Against mutalisks the pros' turrets are 6.9 by 10:00, Gary's model's 10.9.
- **Results (seeds 101+, 432 games, against v0.23's games on the same seeds):** won 112 (109),
  lost 10 (11); against Zerg v3's lurkers won 11 of 32 (7), the mutalisks and the zergling pressure
  the same. An answer to mutalisks too (an Engineering Bay, turrets, a Starport, a Science Facility,
  vessels, Irradiate) won no more against them (15 of 32) and lost 3 more (13): the replayed pros on
  Heartbreak Ridge and the zergling pressure that adds mutalisks, its goals taking money and the
  macro's turn mid-fight; dropped. The tech scan rarely happens: the model's Comsat often comes
  after 10:00, so lurkers are read from the first lurker or egg seen (~8:30). 21 of 32 lurker games
  still undecided at 20 minutes, often with the Zerg down to its last buildings.
- **The touring scout, now that something reads it (2026-10-14):** v0.24 with `watch` against
  Zerg v3 (96 games to 20 minutes, against v0.24's own on the same seeds): the plan read earlier
  (median 7:00 against 8:36 for lurkers, 7:00 against 8:30 for mutalisks), won 53 (51) but lost 4
  (1): mutalisks won 18 lost 2 (15, 1), lurkers won 13 (11), the zergling pressure won 22 lost 2
  (25, 0), where the SCV off minerals costs as it did over the whole pool before. Not adopted;
  reading lurkers earlier is worth a little, a cheaper way to it (the tech scan with an earlier
  Comsat) is the next thing to try.

### v0.23 (October 2026): a third base

- **What it is:** v0.22 whose macro takes an expansion, from the third base on, once the model's
  odds of one, summed over Gary's decisions since its last Command Center (keep_busy's units
  too), reach one; production saves the 400 while it waits (`macro.py` expand_mass).
- **Why:** Gary stayed on two Command Centers all game against every opponent, gas banking to
  1,000-3,000. The model gives expanding 2-4% a decision after 12:00 (about 0 from 6:00 to 10:00),
  never the likeliest, and Gary takes the likeliest; pros take a third base in 86% of TvZ games of
  15 minutes or more (median 12.9 minutes). Forcing it showed three build bugs, fixed for every
  version: production spent every mineral, the 40 s build timeout counted a far base's walk, and a
  builder stopped on its way was never sent again.
- **Results (seeds 101+, 432 games, the whole pool, against a fresh v0.22 baseline at the same
  code: won 108, lost 11):** won 109, lost 11; only 3 games changed (Zerg v3: 2 undecided won, 1
  won undecided). Zerg v3 games at 16:00 on 3+ Command Centers 24 of 64 (3 of 65), at 20:00 31 of
  48 (5 of 49). Forcing the natural too lost 15 (won 113): odds that piled up while Gary held one
  base under pressure forced it at 4:00-6:00. The forced expansions still come at 15-16 minutes,
  later than pros', and on Python builders die walking to the third base through the zerg.

### v0.22 (October 2026): defending early, and scanning lurkers

- **What it is:** v0.21 with two options: `defend_early` (posture.py force_at: the early-pool
  defense from 2:30 unless the rules switched first) and `detect` (fight.py DetectRule: a lurker
  that burrowed where Gary's units see is scanned once Gary's fighters are near it and a Comsat
  has 50 energy).
- **Why:** the early-defense switch learned from 312 paired games gained no more than defending at
  2:30 in every game; and Gary had never scanned a lurker.
- **Results (seeds 101+, 432 games, the whole pool, Zerg v3 to 20 minutes, against the new v0.21
  baseline):** `defend_early` lost 15 (30), won 115 (118): paired by seed 26 losses became
  undecided and 11 games went the other way; wins moved both ways about evenly. The replayed pros
  lost 3 (18), Zerg v2 on Fighting Spirit 7 (5). It misses the bar on wins by 3, within noise;
  adopted with that said. `detect` changed no result (it scans in 18 of 32 lurker games, which end
  undecided either way); adopted as what pros do. The two together weren't run as such.

### v0.21 (October 2026): workers back to work, and where the work is

- **What it is:** v0.20 whose workers astray go back to mining (`executor.py` home_scvs): those
  away from every Command Center, idle or still attack-moving with nobody holding them, found off
  screen, sent by the minimap when their base's minerals are off screen, selected with one
  ctrl+click out in the field; left alone until 10 s after the last zerg at Gary's bases (a
  fight's defenders). They and every worker sent to mine go to the least saturated safe base Gary
  owns (balance), and a group of 3-8 moves from a base over 1.8 workers a patch to a safe one
  under 1.0 (a new Command Center).
- **Why:** in 16 games against Zerg v2/v3, 3.2 SCVs at a time stood away from every base doing
  nothing (9-21 at the worst moment, as few as 11 mining): a pull ended at the nearest base on the
  map, the mining routine gave no order when the minerals weren't on screen, and a box around
  them took the marines they stood with. Gary never moved workers to a new base. With the fix,
  1.1 at a time (2-14 at worst); transfers at 5:45-6:05 when the natural fills.
- **Results (fresh seeds 101+, 432 games, the whole pool, Zerg v3 to 20 minutes):** lost 23 (24),
  won 140 (135); Zerg v3 won 68 lost 0 (59, 3), Zerg v2 lost 5 (8), the replayed pros on
  Heartbreak Ridge lost 10 (6). (Before the minimap clamp; at today's code, Zerg v3 playing three
  plans: won 118, lost 30.)

### v0.20 (October 2026): new fighters gather where pros' do

- **What it is:** v0.19 whose rally moves as pros' marines do (`executor.py` rerally, option
  `rally_ramp`): inside the main's ramp, at its bottom from 4:45, in front of the natural from
  7:30; the Barracks are re-rallied when it moves.
- **Why:** pros' marines at 3:00-4:00 stand at the top of the main's ramp and in the main, at
  5:00-6:00 at its bottom and the natural, in the open 3-8%; Gary's rally was fixed from the
  start in front of the natural, so its marines walked there one by one from the main and a
  quarter stood in the open (Where the first marines gather, below).
- **Results (fresh seeds 101+, 432 games, the whole pool, Zerg v3 to 20 minutes):** lost 24 (25),
  won 135 (133); the replayed pros lost 13 (17), Zerg v3 won 59 (64). Within noise overall: kept
  because it does what pros do and isn't worse.

### v0.19 (October 2026): the build style chosen per map

- **What it is:** v0.18 whose build style is chosen by the strategy selector
  (`gary/agent/selector.py`, ARCHITECTURE.md §6.8): for each map, the style with the best expected
  score from games counted (won + half the undecided, a Beta posterior; a map with fewer than 20
  games of a style borrows from all maps). Options: the macro model unsteered, or steered toward
  one of the four TvZ styles it knows (1 standard bio, 2 Factory and vultures, 3 Factory and
  Starport, 4 two-Barracks marines). The table (`data/interim/selector/tvz_styles.json`, rebuilt
  by `eval/selector_table.py` from `eval/vs_rush.py` folders) counts 1,680 games of v0.18 (336
  per option: the 9-pool and the replayed pros on seeds 1-48, Zerg v2 on 1-72): it plays style 4
  on Fighting Spirit and Heartbreak Ridge, style 1 on Python. Training can draw from each style's posterior instead
  (option `selector_explore`, Thompson sampling) so styles still in doubt keep being tried.
- **Why:** single decisions can't be credited in games this chaotic (docs/CONTROL.md step 7);
  a build style is a decision per game whose effect is big enough to count. Styles 2 and 3 lose
  35% of their games against these opponents (the others 8-9%); two-Barracks marines lost the
  fewest (26 of 336, unsteered 31).
- **Results (fresh seeds 101+, 333 games, against v0.18):** lost 30 (57), won 64 (57). By
  opponent: Zerg v2 on Fighting Spirit lost 3 of 72 (22), won 11 (5); on Python lost 7 (10); the
  9-pool lost 0 of 48 (2), won 39 (36); the replayed pros on Heartbreak Ridge lost 5 of 48 (10),
  on Fighting Spirit 7 (5), on Python 8 (8) but won 2 (6). Most of the gain is one matchup: Zerg
  v2's all-in on Fighting Spirit, held by early marines. The table is fitted to this opponent
  pool; against people it is a prior until their games are counted.
- **Next for it:** grading the undecided games (70-90% end at the time limit) by the value
  network instead of a flat half; choosing again after the scout reports (the style for what was
  seen, not only for the map); the pros' win rates per style and situation (P1c) as the prior.

### v0.18 (October 2026): false early-pool alarms called off

- **What it is:** v0.17 whose posture (v5, `gary/agent/posture.py`) calls off an early-pool alarm
  raised only because nothing was seen, until 4:00, when no early pool shows up (no zerglings, no
  pool finished early); and once a rush is over, lets the marines out of the rush bunker by the
  main's Command Center (`fight.py` LoadRule: released bunkers are unloaded, loaded again only
  when threatened).
- **Why:** in a Python game against Zerg v2, Gary built the rush bunker from a false alarm and 4
  marines sat in it by the main's Command Center while the ramp fell.
- **Results:** replayed pros on three maps lost 11 of 144 (v0.17 9), won 9 (2); alarms called off
  22 times, bunkers let out 64 times; the 9-pool lost 1 of 48 (1), won 36 (27); Zerg v2 (72 seeds)
  lost 8 (10). Economy at 8:00 as before (SCVs on Fighting Spirit 32, Python 24 (25)).
- **Tried, not adopted (options):** holding the main's ramp on one base (`ramp`: medics and
  firebats on its top, marines just behind, hold position), SCVs that move past a bunker before
  repairing it (`repair_surround`), new fighters rallied to the mineral line while outweighed at
  home (`safe_rally`): lost 31 and 37 of 288 games (v0.18: 28), and the ramp hold won fewer against
  the 9-pool (29, 36); results by opponent in `gary/agent/manifests.py`. The main ramp is now found
  by walkable ground (`gary/terrain.py`: the narrowest stretch of the route out of the main), not by
  the tiles' height flags, which on tile-edited maps (Python) put its "top" partway down the slope.
- **Choke blocks, a reusable skill (`gary/agent/block.py`; fight.py BlockRule, option `block`):**
  units on hold position closing a choke with their bodies, as pros do (an SCV, a marine and an SCV
  on a ramp; late in a game, a few units closing an expansion's way in against harass). Which choke
  and when is a higher decision (`g.blocks`, name -> Choke; nothing in play asks yet): the main's
  ramp, the natural's choke, or the ways into any group of bases (`walls.frontier()`), each with
  who may close it (the fighters near it, SCVs from a mineral line) and what it keeps out (any unit
  type: its collision box decides). Where each unit stands is planned to the pixel from the units'
  collision boxes, the cliffs and the buildings (`gary/walls.py` block(): the narrowest
  cross-section that cuts the way, wall to wall, the fewest units, closed even with any one 8
  pixels off and against a zergling a pixel smaller each side): 2 SCVs on Python's and Fighting
  Spirit's ramps, 3-4 bodies on Heartbreak Ridge's; the ways into 12 of 14 bases on Fighting
  Spirit, 10 of 12 on Heartbreak Ridge and 6 of 12 on Python (the rest: open ground) close with 12
  marines or fewer, 2-3 at a narrow way. In the engine none of 8 zerglings gets through those
  blocks (9 ramps, 6 naturals); one unit fewer, most do. Each blocker is selected by clicks,
  placed with a precise right-click and holds position as it gets there; when all hold and the
  block still isn't closed (checked exactly) the farthest is placed again; the others wait further
  in until it stands, then hold behind it. Choke drill (`eval/block_drill.py`; 3 maps, 16 seeds,
  8-16 zerglings, 12 s to get ready): the main's ramp (6-7 units): closed when the first zergling
  arrives 85% (v0.18, not asked: 0%), held 85% (1%), killed 11.1 (7.3), army lost 1.2 (4.6); the
  natural's choke (12 units, 6-10 of them in a block 210-300 pixels wide): closed 42% (0%), held 69%
  (14%), but killed minus lost no better (-0.2 ± 0.2): placing that many one by one takes most of
  the time, and a thin line fights worse than a clump. Narrow chokes are worth closing with bodies;
  wide ones are for buildings (a choice for the decision that asks). Asked for by a simple rule in
  games (`ramp_auto`: on one base, zerg fighters seen at the ramp) it does worse: lost 43 of 336
  games (v0.18: 31), the 9-pool 4 of 48 (1), won 25 (36), where it takes the first marines the rush
  bunker at the ramp wants. When to block is a decision for the learned model.
- **Block or not, learned from branches (`eval/block_branch.py`, `train/block_decision.py`,
  option `block_learned`):** 1350 moments of Gary's games (432 games: the 9-pool, Zerg v2, the
  replayed pros on three maps) where enemy fighters came near the main's ramp, each played on twice
  from the same state (games are deterministic: the same game replayed to the moment, Gary's memory
  included), without and with the block, scored over the 90 s after. Blocking did better at 38%,
  worse at 59%, -54 on average (the 9-pool -79). A model of when it gains (gradient-boosted trees
  on what Gary sees: block.features()) found nothing on held-out games: -25 a moment (never
  blocking: 0; perfect hindsight +111). Placebo branches (Gary skipping half a second) measure
  chaos alone: it changes the next 90 s at 39% of the moments, by +-425 (Zerg v2: +-607), against
  the block's effect of about -65 against the 9-pool (chaos there +-116: a real effect) and -141
  against Zerg v2. Single branches average effects well but can't teach when: that needs several
  per moment. Blocking the main's ramp as it is now loses on average; not adopted.

### v0.17 (October 2026): no early pool from nothing seen at 2:20

- **What it is:** v0.16 whose posture (v4) calls nothing seen an early pool only at 2:50, not
  2:20 (`gary/agent/posture.py`).
- **Why:** pros don't see the Zerg by 2:20 on 4-player maps either: Terran pros, 300 TvZ replays,
  first see a Zerg building at 2:36 on Fighting Spirit (median; 2:59 at the 75th percentile), 2:10
  on Python, and send their scout at 1:18-1:36 (Gary at about 1:15). v0.16 played rush defense
  against 19 of 30 hatch firsts on Fighting Spirit and 21 of 29 on Python. Early pools are caught
  otherwise now: the pool remembered, a pool finished by 2:40, zerglings.
- **Results:** rush defense against hatch firsts 7 (19) on Fighting Spirit, 11 (21) on Python;
  replayed pros there lost 7 of 96 (4): one is the cost (a pool at 1:10 at the last start scouted,
  seen at 2:50), the others played out differently; the 9-pool 0 of 48 (0); economy as before or
  slightly better (SCVs at 8:00 on Fighting Spirit 32, 30). Zerg v2 (72 seeds): lost 9 (v0.16 before
  the click and delay fixes: 8), won 4 (2), SCVs at 8:00 35 (30), army built by 12:00 61 (57).

### v0.16 (October 2026): the army hunts

- **What it is:** v0.15 whose army, from 8:00, with 20+ army units and no enemy army seen for a
  minute, goes where enemy buildings were seen, nearest first, then to the bases its units
  haven't passed in 2 minutes, scanning them first when a Comsat Station has the energy
  (`gary/agent/army.py`, hunt and scan; a target not reached in a minute is dropped). The army
  model is asked again as soon as an enemy army shows up.
- **Why:** far ahead, the army model stays near what it knows: it goes where enemy buildings
  were ever seen (as it learned), so a hatchery at a base Gary never looked at outlived the game.
  Replayed pros on Fighting Spirit to 18:00: in 12 of 16 games Gary had razed all but 1-4 Zerg
  buildings with a 60-77-unit army, and won 2.
- **Results:** the same 16 games to 18:00: won 11 (2; 8 hunting without scans), lost 1 (2). To
  12:00 (hunting without scans, which only change games once the hunt is on), replayed pros on three
  maps won 4 of 144 (0) and lost 14 (14); Zerg v2 (72 seeds) won 2 (0), lost 8 (8): the same lost
  games, which end before the hunt can start.
- **The pool forgotten (shared code, the early-pool posture):** a pool counted only if in view
  once the scouting was done, but the scout sees it while circling the main and is done only
  past the natural, out of sight of it: an overpool (pool at 1:41) went unnoticed, Gary expanded
  at 2:48 and was dead by 6:00. Now the pool is remembered: replayed pros lost 12 of 144 (14), the
  dummy's 9-pool 0 of 48 (3) and won 29 (with the hunt; v0.15 won 5); economy as before.
- **The replayed fixture's buildings go up (test fixture):** the env created them finished, so
  every pool looked finished when placed. Now a drone appears at the spot and starts the
  building (the env can set a player's money: `Game.set_resources`), and it goes up for its real
  build time. Numbers on the fixture shift: v0.16 without the next item lost 15 of 144 (Heartbreak
  Ridge 9).
- **A pool first that takes its natural (posture v3):** on Heartbreak Ridge, pools at 1:36 with a
  hatchery at the natural too: the posture waited for one hatchery, and the zerglings came at 3:00
  to Gary's 1 marine. A pool finished by 2:40 is now an early pool whatever else (pro TvZ: 98% of
  pool firsts' pools, 0.7% of hatch firsts'; by 2:50 it would be 16% of hatch firsts): lost 13 of
  144 (15; Heartbreak Ridge 7), the same hatch firsts called early pools.
- **A scout that never went (shared code):** the scout's box once took nothing, and an SCV still
  selected from before became the scout; it went on mining, nothing was seen by 2:20, and a hatch
  first was played as an early pool. A scout not on a move order every 3 s is sent again (3 times,
  then another SCV takes over, keeping the starts ruled out). Same results otherwise (13 of 144;
  the 9-pool 0 of 48); a first version that judged by distance replaced every scout on Python,
  whose paths bend away from the target.
- **Clicks that missed (shared code):** Gary aimed at units where it saw them, 0.3 s late: a
  click on an SCV at its mineral field or by a Command Center selected nothing (or the Command
  Center), a box around a walking SCV or a moving group missed it, and the builder pick clicked
  again every 2 frames with no limit (bursts of up to 100 clicks, 150-235 a game). Now the
  interface aims where a unit will be when the action lands (`HumanInterface.aim`: its motion
  just before what Gary saw, carried forward; never past where it's going); builders and gas
  workers are picked standing if one is near (a box finds them), taken off their mineral field
  with a move first (mid-mining, an SCV let the build order go in 3 tries of 7), selected with a
  tiny box, or a click when the box took several SCVs (a build order to several is refused);
  another SCV after 3 misses; a parked build resumed looks at its spot again (it had failed when
  the spot was off screen). Missed selects 12 a game (239), failed builds 4.9-7.1 (8.1), actions
  to 8:00 6% fewer; replayed pros lost 10 of 144 (13). Army built by 8:00 2 lower on Python
  (2 more SCVs trained: the worker pulls' boxes now take the SCVs they ask for).
- **The reaction delay was longer than set (shared code):** the interface kept a snapshot every 4
  frames, so the pro profile's 200 ms was 210-340 ms (270 on average). Now a snapshot every frame
  (the last 64): 210 ms, the frame nearest 200. Home defenses −27 (−35); replayed pros lost 8 of
  144 (8); the 9-pool 0 (0); to 18:00 on Fighting Spirit won 13 of 16 (11). The 200 ms is still an
  estimate: it should be fitted from replays (first sight of an event to the first command on it).
- **Pros' reaction times, measured (`eval/reaction.py`):** replays don't say why a command was
  given, so only events with one clear answer count: one of the player's units is hit (and they
  hadn't commanded it for 2 s) -> their first command including it; an enemy unit comes into sight
  -> their first command targeting it; only near the player's last click (where the camera was),
  and over a control (the same search 3 s later: commands they'd have given anyway). Terran pros,
  200 TvZ games: hit -> command 10% within 420 ms, 25% 590, half by 1.1 s; sight -> command 10% by
  340 ms. That fits ~200 ms to see plus the hands. Gary (48 games against Zerg v2): hit -> command
  half by 590 ms, faster than the pros, but partly not reactions at all (its fight roles re-decide
  every half second); it never targets a unit it just saw (it attack-moves). Gary's own games are
  measurable only against opponents that play by commands (the replayed fixture creates units).
- **False alarms on 4-player maps (open):** with buildings going up, Gary plays rush defense
  against 18 of 30 hatch firsts on Fighting Spirit and 23 of 29 on Python (2 of 23 on Heartbreak
  Ridge): its scout hasn't found the Zerg by 2:20, and nothing seen is called an early pool. At
  2:20 the scout is still crossing to the second start (about 40 s a leg on Fighting Spirit), so
  waiting until 2:45 when the last start is known by elimination never applied (tried, no change);
  a 2:30 deadline changed nothing either (the alarms just came at 2:30). The lever is the scout:
  it leaves at about 1:15 (9 SCVs) and reaches the third start at about 2:45.
- **Most urgently missing:** a hunting army of 70+ crosses the map slowly (scans find the last
  hatcheries at 14-17 min); the army model's fight estimate still sends 8 units out into 14
  zerglings (Python, hatch-first floods at 4:30-5:30: 4 of the 12 losses).

### v0.15 (October 2026): bunkers kept manned

- **What it is:** v0.14 that keeps its bunkers loaded once they stand, not only once zerg come
  within 9 tiles (fight.py, the load rule's `always`), as pros man the natural's.
- **Why:** against a speedling all-in (Zerg v2, 20-30 zerglings at about 6:00) the zerglings
  waited out of sight 20 tiles from the natural, then crossed the 9 tiles in under 2 s: Gary's
  marines were still walking into the bunker, not shooting, and all 12 died in 10 s, killing 2.
- **Results:** Zerg v2, 72 seeds, 12 min: lost 8 (v0.14 15). Replayed pros on three maps lost 14
  of 144 (15), the 9-pool 3 of 48 (3); home defenses −49 (−52, +2 ± 5).
- **Zerg v2 had stalled (test fixture):** in 5 of 8 games it made nothing all game (a box around
  an idle drone took the overlords above it too, and it boxed again forever; drones following
  overlords after a right-click on a mineral field under one; a pool blocked by Gary's scout given
  up after 3 tries), so earlier results against it were too kind. Against the fixed v2, v0.14
  lost 15 of 72 within 12 min.
- **The gas chore froze Gary (shared code):** with no free SCV it ended at once and was asked
  again every 2 frames: nothing else got Gary's hands for the rest of the game (12 of 24 long
  games against v2, from 1 to 6 minutes before the end). Now the refinery waits 20 s.
- **Fight orders called bunker loads off (shared code):** the roles' next order went to a box
  around marines by the bunker, which took the marines walking in too. Units entering a bunker,
  and those by them, now get no role orders, and a fight order whose box would take them waits.
- **Games are deterministic but chaotic:** the same code and seed play the same game, but any
  change reshuffles every game after it, so A/B runs can't be paired by seed; 24 seeds of v2
  ("lost 4 against 7") were noise until 72 seeds.
- **Measured, not adopted (options in manifests.py):** pros' marines move 20% of the time a zerg
  is in range, Gary's 6%, but stepping back when 2-3 melee zerg are on a marine (−17, −6 on home
  defenses) or below 75% HP (−5) did worse; letting the main's bunker out to a fight at the natural
  lost 22 of 72 against v2 (15); 4 SCVs repairing a bunker (not 2) lost 15 (15).
- **Most urgently missing:** the all-in still breaks Gary in 1 game in 9, and Gary doesn't finish
  the games it holds (0 wins in 72 by 12:00): its army advances a grid cell or two per order
  ([#3](https://github.com/vibrunazo/gary/issues/3)).

### v0.14 (October 2026): production kept going

- **What it is:** v0.13 whose idle Barracks, Factories and Starports train the unit the model
  last chose for them again, with the money left after what the pending decision needs (macro.py
  `keep_busy`, manifest `v014`, the default).
- **Why:** with the economy measured (`eval/economy.py`), Gary mined like a pro (31 SCVs and 2
  Command Centers at 8:00, as the pros) but had half the army (8 supply against 19). Snapshots at
  4:00, 6:00, 8:00 next to 148 pro games: as many Barracks (3.1 against 2.9 at 8:00), but 1.2 of
  them producing (pros 2.1) with 300 minerals banked: the macro makes one decision at a time,
  every 4 s or so, and three Barracks and a Command Center need one every 3.5 s. Decisions were
  also wasted: a Comsat Station asked for every second for minutes (the natural's bunker stood
  on its spot: now no building goes on an add-on's spot, and add-ons go where they fit), and
  Missile Turrets re-ordered over and over (a second parked build replaced the first: one
  builder walks ahead at a time now). Both shared code.
- **Results:** army supply at 8:00 11-12 (v0.13 7-9; pros 19), at 12:00 32-44 (25-34; pros 36);
  SCVs as the pros'. Replayed pros lost 7 of 144 (10); the dummy's 9-pool 0 (0), won 10 (8).
- **False alarms (shared code):** against hatch-first pros Gary had played rush defense in 116 of
  144 games, its tech a minute late. Now a second hatchery seen by 3:00 with no zerg at home calls
  it off at once, zerglings count toward an early pool only while one hatchery has been seen, and
  the scout gives up on a waypoint it can't reach after 8 s (one of Python's circle points: it
  never got to the natural). Rush defense held on in 22 of 87 hatch-first games. Replayed pros
  lost 12 of 144 (7-10 before, the noise is about ±4); the dummy's 9-pool 1, its 7:00 rush 1
  (won 14). In hatch-first games: expansion 2:42 (pros 2:50), Academy 4:40 (4:08), Stim 5:53
  (5:09), Medics 6:23 (5:12), a Factory by 8:00 in 10% (70%), a Starport 2% (31%). Putting
  buildings, research and upgrades ahead of the units kept-busy producers make anyway changed
  none of it: the model decides them late.
- **The Refinery cost 75 minerals in Gary's table (shared code):** Brood War's is 100 (the only
  wrong cost against the game's data). Gary ordered it with 75-99 minerals and the game refused
  without a word (a new env call, `gary_env_debug_place`, shows which of the engine's checks a
  build order fails; this one passed them all and failed on the money in `place_building`); it
  was ordered again a minute later, in about 40% of the games. Now in hatch-first games: Refinery
  3:40 (pros 3:33), Academy 4:10 (4:08), Stim 5:15 (5:09), Medics 5:36 (5:12), Range 6:35 (6:14),
  a Factory by 8:00 in 20% (70%), a Starport 3% (31%). The price: replayed pros lost 17 of 144
  (12): with the money on time for gas, tech, the expansion and the natural's bunker, the first
  Barracks idles between 2:30 and 5:00 (the lost games had 0-3 army units at 5:00; pros have 3.4
  marines at 4:00, 7.2 at 6:00); and a 12-pool that also expands early (pool at 1:36-1:45) shows
  a second hatchery and calls rush defense off while its 11-17 zerglings come.
- **Army built, not only alive (`eval/economy.py`, every summary):** the replayed pros send every
  unit they made at Gary, so its army alive is about half the pros' at 8:00 (9-10 against 19), but
  it has built 22-24 supply to their 27 (pros' games: 70 by 12:00, Gary 48-53). Gary builds more
  SCVs than they do by 12:00 (42-51 against 39, on two bases both). Keeping Barracks busy while a
  building is saved for (not holding its money back) changed nothing measurable (replayed pros
  lost 19 of 144 against 17).
- **SCVs to saturation; an early pool that expands too (shared code):** Gary makes SCVs up to 2
  per mineral field and 3 per geyser of its bases (Command Centers finished or not) and 3
  builders, not to 60: 35-41 at 12:00 (pros 39; it had 42-51), army built 52-58 by 12:00 (48-53;
  pros 70). A second hatchery no longer calls rush defense off if a finished pool was seen by 2:50
  (started by 2:00: 50 s to build; hatch firsts' at 2:05) or zerglings by 3:00: rush defense
  called off in 3 pool-first games of 57 (9). Replayed pros lost 15 of 144 (19): 3 of 57 pool
  first, 12 of 87 hatch first (6 of 29 on Heartbreak Ridge); the dummy's 9-pool 1, its 7:00 rush 1
  (won 12).
- **Heartbreak Ridge's hatch-first losses (shared code):** zerglings at 4:00-4:30 against 0-3
  army units; one Barracks to 5:00, and at 3:15 an idle Barracks with 464 minerals: the build of
  the natural's Command Center held Gary's hands while its SCV walked there. Now a build at
  another base is parked while its SCV walks (the hands go back to production) and taken up again
  when it's there. Then that Command Center failed as its SCV arrived: an SCV and a marine had each
  been checked against the same 472 minerals (the money is taken a moment after the order): units
  and research ordered in the last 2 s now count as spent. Replayed pros lost 8 of 144 (15; on
  Heartbreak Ridge 4, its hatch firsts 2 of 29, were 6); the dummy's 9-pool 2. A second Barracks
  by 3:45 when the model hasn't (pros of the standard style: 3:31) lost 12: not kept.
- **The Engineering Bay is 4x3 tiles, not 3x2 (shared code):** found by a new test of Gary's cost,
  size and supply tables against the game's data (`tests/test_game_data.py`); its placement was
  checked on the wrong footprint and failed at least once in 24 of 144 games. Benchmark runs
  (`eval/vs_rush.py`, `eval/scenarios.py`, `eval/walls_bench.py`) now run below normal priority by
  default (`--priority normal` for full), and so do their games: other programs keep the CPU.
- **Why the Factory comes late: the decoder.** The macro model, on held-out pro games of the
  standard style, gives the Factory a median 4% at the moments the pro's next decision is one
  (ranked first in 1%; the Starport 5%, ranked first 15%): a rare next event beside the next marine.
  Gary takes the most likely decision (and never one under 5%), so on the pros' own states it would
  decide a Factory by 8:00 in 4% of games (they did in 62%); an Engineering Bay 31% (90%). Offline,
  rolling the odds matches the pros (Factory 63%, Starport 25%, Engineering Bay 78%); accumulating
  each decision's odds until one is due (smooth weighted round-robin) nearly (47%, 11%, 72%). In
  games, rolling cost SCVs (28 at 12:00 against 41: the SCV is the favorite more often than it's
  rolled) and changed no Factory timing (option `+dice`, not adopted).
- **Builds lost to Gary's own hands (shared code):** in 176 games' logs, Missile Turrets failed in
  125 games (lost orders, refused, re-ordered: ~1,000 times), Supply Depots in ~100, Barracks in
  ~50. Most were one race: a build's order takes a few frames to show in its SCV's state, and the
  next build's task picked that "idle" SCV and sent it walking to its own spot (the first build's
  SCV then had a move order: lost). Also the gas chore's drag box caught SCVs walking past to place
  a building and sent them to gas, and the worker pull took builders. Now an SCV just given a build,
  one placing or constructing, the parked builder and the scout are never picked; the gas chore
  selects again if its box caught a busy worker. In one game the failures to 11:00 went from 14 to 2.
- **After the bug hunt (shared code, 2026-10-09):** builds with no free SCV in the main take one
  from the natural (one game failed 13 Barracks in a row for it); a gas-chore edit had sent no SCV
  to gas (caught by a benchmark, fixed; `tests/test_smoke_game.py`, one game to 6:00, now checks
  it); keeping a pending add-on's or research's building and money from the habits cost more than
  it gave (the 9-pool lost 4 of 24 against 2, no natural taken) and was removed. Failed builds and
  goals per game 8.3 (about 13 before). Replayed pros lost 10 of 144 (8 before: within the noise),
  the 9-pool 3 (2); SCVs 31-35 at 8:00, army built 50-61 by 12:00, as before.
- **Most urgently missing:** the rest of the army at 8:00: pros have a Factory (1.0) and a
  Starport (0.6) by then, and more Medics (2.7 against Gary's 1.7); Gary's style-1 macro has
  none of the first two by 8:00.

### v0.13 (October 2026): the rush bunker by the Command Center

- **What it is:** v0.12 whose bunker against an early pool stands by its main's Command Center,
  3 tiles toward the main's way out (posture.py `early_pool/v2`, manifest `v013`, the default),
  not at the way out.
- **Why:** against the fastest pools Gary lost its SCVs: on Heartbreak Ridge it walked 10 of them
  out to its bunker at the main's way out (16-20 tiles from the mineral line), the bunker fell to
  10 zerglings with one marine in it, and the zerglings then took the mineral line. In the same
  games (the replayed Zerg's source games) the pros pulled 0-4 SCVs and lost none, or, against
  4-pools, pulled 10-11 and kept most. Against pools ordered before 1:20 pros build a bunker by
  3:30 in a quarter of the games, and on one base it stands by the Command Center (60 of 88; at
  the way out 18): the marines in it cover the mineral line, SCVs repair it from where they mine.
- **Results:** replayed pros on three maps, 48 seeds each: lost 3 of 144 (v0.12 19; Heartbreak
  Ridge 0, was 5). The dummy's 9-pool lost 0 (2), its 7:00 rush 3 (0): a wave of 26 zerglings at
  7:00 goes through the main's choke the bunker no longer holds (that test's Zerg sits home for
  7 minutes, the least realistic of them). Gary kills the dummy less often (won 3 and 3 against
  12 and 10). On the six fastest pools of Heartbreak Ridge, SCVs left a minute after the zerglings
  came: 16-18 in five (0-7 with the bunker at the way out). Home defenses (239, 8 draws): −49
  (v0.11 −55; the posture rarely matters there). Tried and not kept: a second bunker at
  the way out once the rush is held while Gary stays home (7:00 rush lost 1, but replayed pros 8
  of 144: it costs 100 more when the early pool was a false alarm, as when Gary hasn't seen the
  Zerg's base by 2:20 on a four-player map); pulling workers only for zerg in the mineral line
  (`+pull_line`: a few more SCVs kept, the bunker's spot mattered more).

- **The scout was taken (shared code, every version):** the executor took Gary's scouting SCV
  as a builder (and the mining chore sent it back to work) when it was on screen as it left: the
  Zerg's base was found in about a third of the games on four-player maps, and with nothing seen
  by 2:20 Gary played rush defense in nearly every game (and expanded at ~4:15). Now builders and
  idle mining leave the scout alone, the scout passes the Zerg's natural after circling its main,
  and a pool seen with one hatchery is an early pool if seen by 2:00 (hatch firsts order theirs
  at 2:05), otherwise once the natural has been looked at (always waiting for it, the dummy's
  9-pools were recognized at 3:04 instead of 1:55: 7 of 48 lost; a 2:45 blind alarm instead of
  2:20 lost 2 more). Gary now expands at 2:26-2:46 (pros 2:54) when it sees a hatch first. The
  price: replayed pros lost 11 of 144 (3 when Gary turtled every game), the dummy's 9-pool 1 (0),
  7:00 rush 3 (3); by 12:00, in the games not lost, about the same buildings and army (Heartbreak
  Ridge more: 36 and 35 against 30 and 29). Still false alarms: on four-player maps the scout has
  checked one start by 2:20 (18 of 29 hatch firsts on Fighting Spirit went into rush defense), and
  zerglings seen before 4:00 count as an early pool (a hatch first's arrive at 3:20-5:00).

- **Economy next to the pros' (`eval/economy.py`, in every `eval/vs_rush.py` summary):** losses
  alone reward turtling. Medians at 8:00 / 12:00, games still going: pros (300 TvZ games of
  12:30+) 31 / 39 SCVs, 2 / 2 Command Centers, 19 / 36 army supply. v0.13 against the replayed pros:
  SCVs 31-33 / 39-44 and 2 / 2 Command Centers, as the pros; army supply 6-9 / 20-36. Gary mines
  like a pro but has half the pros' army at 8:00.

### v0.12 (October 2026): a bunker at the natural

- **What it is:** v0.11 that builds a bunker at its natural once it takes it (a rule on the learned
  macro: macro.py, manifest `v012`, the default), where pros put theirs, and orders its first
  Barracks by 1:18 if the model hasn't (was 1:30: placed at 1:43, after the pros' ~1:30). Pro TvZ Terrans that
  expand by 5:00 build one within 2:30 of the Command Center in about half the games, on every
  open map (Fighting Spirit 52%, Python 59%, Heartbreak Ridge 50%...; Colosseum II, walled, 19%),
  a median 35-56 s after the CC, about 5 tiles from it: 45% of the way from the CC to the
  natural's way out (the frontier).
- **Results:** benchmarks on three maps now (Fighting Spirit, Python, Heartbreak Ridge:
  `tests/fixtures/replays/`). Against the replayed pros (as replayed games, below), 48 seeds each:
  lost 2 / 6 / 7 (v0.11 2 / 9 / 14), 15 of 144 against 25 (the natural bunker alone: 19); the
  hatch-first losses 3 against 13, the pool-first ones 12 either way. Pro-timed 9-pool on Fighting
  Spirit lost 2 (v0.11 2), won 14 (8).
- **The replayed pros were too strong** until now: the fixture spawned what the build orders
  list, and those are orders, not units (one order morphs several larvae; spammed orders make
  none): a Zerg with 24 zerglings at 3:00 by its orders had 4 in the game. Now the fixture plays
  the pro's game in OpenBW first (1-2 s) and spawns what that Zerg had, when it had it (a replay
  whose pool doesn't go down within 20 s of its order desynced: another is picked). Earlier
  replayed-pro numbers in this file (v0.11's "lost 7", the first v0.12 runs) were against it.
- **Shared code (every version):** the rush bunker goes 2 tiles inside the main's way out, found
  by the frontier on the main's plateau (`frontier(level=True)`), not by the ramp analysis, which
  misreads Heartbreak Ridge (one main's ramp 40 tiles off) and Colosseum II. Results about the
  same (v0.11's losses 32 -> 37 over the four tests, within the noise; Heartbreak Ridge 16 -> 17).
- **Builders walk ahead (shared code, every version):** the opening ran 5-7 s behind the pros'
  (first depot 1:05 against 0:58, first Barracks 1:33 against 1:28, Fighting Spirit, placed). The
  macro made one decision at a time, so a depot waited behind an SCV whose Command Center was
  busy (the model's next pick follows the last one): a unit whose producer is busy is now decided
  at once and trained when it frees up, and the next decision comes meanwhile (as pros queue the
  SCV and send the depot's builder). A building's SCV is sent 6 s of income before the money is
  there (it used to wait, selected, for the money, then walk), and the hands go back to
  production while it walks (the build is parked; its money is spoken for). Now: depot 0:56,
  Barracks 1:33 (the scout leaves at 1:16 from 9 SCVs), SCVs 10 / 12 / 14 / 16 at 1:30-3:00 as
  the pros'. Against the replayed pros: lost 19 of 144 (15 before), within the noise; the 9-pool
  lost 2. Also seen: against hatch-first Zergs Gary decides to expand at 4:15 (median), pros order
  their Command Center at 2:54.
- **Most urgently missing:** the fastest pool-first pros (6 of 19 on Heartbreak Ridge, its short
  rush distance): their 8-9 zerglings arrive at 2:55, when Gary's first marine is just out
  (2:51) and its bunker isn't finished. Pros order their Barracks at 1:30 there and their first
  marine at 2:23; Gary's Barracks goes down at 1:38 even when ordered at 1:18: its 150 minerals
  come at 1:34 (7 SCVs at 0:50, the first depot at 1:06: about 10 s behind a pro's opening).

### v0.11 (October 2026): more workers pulled, a post at home (#2)

- **What it is:** v0.10 that pulls 3 workers per attacking zerg unit instead of 2, and whose army
  at home holds a post by the Command Center (3 tiles from it toward the zerg; units with nothing
  in range farther than 3 tiles from it go back) and closes in only on zerg attacking something
  of Gary's or standing by its buildings (roles.py `post`; manifest `v011`, the default).
- **Results:** 239 home defenses, reactive Zerg, 8 draws: **−54** (v0.10 −69, +15; pro −5,
  nothing −305); SCVs lost 1.90 a defense (v0.10 2.06). Immediate 9-pool, 48 seeds: lost 4
  (v0.10 4); 7:00 rush, 16 seeds: lost 0, 37 buildings and 46 army supply at 12:00 (v0.10 the
  same). With the rush dummy on pro timing and the early-pool fixes of 2026-10-07 (shared
  code: every version has them): 9-pool lost 2 of 48, 7:00 rush 0, home defenses −55.
- **How we got here:** Gary pulled workers sooner, more often and longer than the pros (where
  the pros didn't pull at all in 44 of the 239, and 3.0 s after the takeover where they pulled
  at 7.6 s). But pulling like them scored worse: fewer workers a zerg (−13 ± 4), only after 4 s
  of threat (−19 ± 7), only against stronger zerg (−3 to −6). The other way paid: 3 workers a
  zerg +8 ± 2 (and fewer of them die: the hits are shared). Gary's workers make up for what its
  army doesn't do. The post had come out at +7 or +8 three times; on top of the pulls +7 ± 5.
  The 16-seed 9-pool check had rejected both (1-3 losses, v0.10 0), but on 32 more seeds v0.10
  lost 4 too: a few losses in 16 is noise.
- **Most urgently missing:** the gap to the pros (−54 vs −5) is still in what Gary loses
  (marines and SCVs); vulture micro (finer hands); positioning behind walls.
- **Walls, started:** while zerg are near, the pros' marines hold position 16% of the time
  (v0.11's 2%), and the 59 home defenses where the pro holds a lot carry 28% of the gap. Which
  units get between two buildings now comes from the game data (`gary/walls.py`: each building's
  collision box is smaller than its tiles by a different margin on each side, so a Supply Depot
  on top of another stops zerglings and side by side doesn't), checked against the engine
  (buildings placed in an OpenBW game, units sent through: the arithmetic agrees to the pixel in
  1,144 of 1,148 cases; an Academy above a Supply Depot is tighter than it says). In the pros'
  239 bases (at the takeovers), 38% of the touching building pairs stop zerglings: mostly Supply
  Depots stacked (125 times, all tight; side by side, 100 times, none), a Supply Depot above a
  Barracks or an Academy, and an Academy left of a Supply Depot (tight; right of it, not). Many
  touching pairs are clusters (depot fields, rows of Barracks), not walls. The gap finder
  (`gaps()` in gary/walls.py: pixel-exact, buildings and cliffs, who gets through and the
  openings narrowest first) shows the pros' TvZ sim city at 4-6 min is at the natural: its choke
  narrowed from a median 176 pixels to 74 by buildings (half or less in 95 of 114), closed to
  zerglings in only 5; the main ramp is never walled. Their marines on hold stand at an opening
  23% of the hold time (open ground 34%, by a building 16%, a bunker 13%, between buildings 11%).
  A post inside the wall gaps, holding (`+gap_post`, while the zerg are outside): −7 ± 4 on top
  of v0.11 (better in 26, worse in 42; the scenario that showed the pro's held gap, 0c50ac60,
  +104). These scenarios start with the zerg already at the Terran's home, mostly past a wall's
  job (keeping them out): walls are for full games, once Gary builds its own (a sim city
  planner). Which openings matter depends on the map: blocking the natural blocks the main where
  the only way in is through the natural, some maps need the natural blocked by buildings and
  the ramp by units, and some have the natural behind the main. So the defended frontier is
  what separates all of Gary's bases from the enemy's side, found over the whole map (coarse per
  tile, then exact around the openings), not the ramp and the natural each on its own.
  Done (`frontier()` in gary/walls.py, `python -m gary.walls --frontier <scenario>`): a minimum
  cut over the minitiles near the bases (max-flow), its pieces looked at exactly. In the pros'
  bases: with only the main, the ramp alone (61 of 123), another single way out (20), closed to
  zerglings altogether (13); with the natural taken, one opening (57 of 116), two (25), closed
  altogether in 2. (Neutral map units, like Othello's, collide too; the engine's pathing regions
  are exported for checks, Game.regions.)
  The sim city planner (`gary/agent/simcity.py`, option `+simcity`): when the executor places a
  Supply Depot, Barracks or Academy, a spot by a frontier opening that narrows it, growing the
  wall from one of its sides (zergling-tight against a cliff or wall building), keeping a door a
  32-pixel unit gets through, and not while zerg are near the opening. On Fighting Spirit it
  builds what pros do at the natural: a building against each cliff, one gap in the middle. Full
  games, 48 seeds each, on top of v0.11: 7:00 rush won 11, lost 0 (v0.11 8 / 0), the same
  buildings and army at 12:00; immediate 9-pool lost 7 (v0.11 4). Within the noise of 48
  games either way (which seeds are lost changes completely with any change): not adopted yet.
  The post in the wall gaps now takes its gaps from the frontier (`wall_gaps()`) and holds them
  while the zerg are outside the frontier, ahead of closing in. To judge it where walls matter,
  `eval/walls_bench.py` cuts wall fights out of Gary's own games (v0.11 + sim city against the
  rush dummy: the first moment 4+ zerg fighters come within 12 tiles of a wall gap from outside;
  78 in 96 games) and plays each from there by variants, 8 draws: the post −12 ± 12 (better in
  32, worse in 31; Gary nets +162 there either way: the dummy is weak). Not adopted.
- **A better scripted Zerg** (`gary/bots/zerg_v1.py`, `eval/vs_rush.py --opponent v1`): the
  dummy's 9-pool, with an army that forms control groups of 12, gathers outside Gary's natural,
  judges Gary's defense (what it sees, and the most it saw at once in the last 3 min), surrounds,
  retreats from losing fights and defends its hatchery (the dummy let Gary kill it with 36
  zerglings alive). Gary v0.11 lost (48 seeds, immediate): to the dummy 5, to v1 2, to v1
  always attacking 4, without the surround too 5; hunting SCVs made it 1 (zerglings chasing
  pulled SCVs ignore the marines). With a 9-pool on 9 drones, smart play gives Gary time: it
  isn't harder. Zerg v2 (`gary/bots/zerg_v2.py`, `--opponent v2`) puts v1's army on a pro-like
  zergling opening: 12 hatch at the natural, pool, extractor, speed, drones to 18, macro
  hatcheries (4) when larvae run out, the gas drones back to minerals after speed: 35-50
  zerglings by 8-10 min. Gary v0.11 still lost none of 48 games at either timing (and won 14-15):
  zerglings alone don't break bio once it's up, and the cautious judge lets the early window go.
  With a timing attack (groups wait at staging; once speed is done with 20 zerglings, or at 6:00,
  it musters everyone, then all attack at once, no retreat, reinforcements in groups of 6): Gary
  lost 2 of 48 (immediate) and 0 (from 7:00). The dummy's 9-pool, hitting before any bunker,
  is still the hardest scripted test (5 of 48).
- **The rush dummy was a minute late.** Pro TvZ (3,000 games): 3+ zerg fighters reach the
  Terran's buildings at 2:57 in the fastest 10% of games (median 4:24); pro early pools order the
  pool at 1:17 and the first zerglings at 2:02-2:10 (`data/interim/build_orders.jsonl`, 14,744
  Zergs; 30% go pool first). The dummy placed its pool at 1:46 and waited for 12 zerglings: it
  arrived at 4:03-4:31. Now its drone sets out at 150 minerals (pool at 1:27, zerglings at 2:37)
  and zerglings go as they hatch (the first 6 together): it arrives at 3:07-3:44 (median 3:22).
  Gary v0.11 against it, 48 seeds: **lost 27**, won 6 (against the late dummy: lost 5). Zerg v2
  is later still (hatchery 3:32 and pool 3:35, against the pros' 1:47 and 2:05).
- **Holding a pro-timed 9-pool** (2026-10-07): in the lost games Gary noticed the early pool at
  3:10 (the zerglings, at its base) and placed its bunker at 3:31, after they arrived at 3:21;
  in the held ones at 2:44, the bunker at 2:53. Fixes, each measured: the rush build keeps 100
  minerals for the bunker until it's coming and builds a Barracks first if there is none
  (it used to start a second Barracks and a depot first: the bunker at 3:14 though the pool was
  seen at 1:55); the scout circles the Zerg main once found (a pool can stand out of sight of
  the hatchery: it used to turn home at the first building); and with no Zerg building seen by
  2:20 Gary plays it as an early pool (on a 4-player map the scout reaches the second start at
  2:43). Lost 27 → 9 of 48 (won 9); 7:00 rush lost 2 (won 18). The price, against the
  hatch-first Zerg v2: won 7 instead of 16 (lost 2 either way), 6% fewer buildings and 8% less
  army at 12:00: Gary turtles when it can't see. Home defenses unchanged (−53). Left: Barracks
  late when the learned macro opens without one (4 of the 9), bunkers with no marines yet (3).
  Then: a Barracks by 1:30 whatever the learned macro says (pro TvZ Terrans order it at a median
  1:28, 95% by 1:39; 2% build a CC first), and build orders that were lost are ordered again
  after 6 s instead of 40 (a reserved spot with no SCV heading to it: the Barracks order of 1:26
  had gone to the SCV still constructing the depot, and nothing came until 2:11). Against the
  pro-timed 9-pool: **lost 2, won 16** of 48 (27 lost this morning); 7:00 rush lost 0, won 15;
  Zerg v2 lost 1, won 10; home defenses −55.
- **Builds that didn't start** (2026-10-07): in both remaining losses the bunker was ordered with
  the money there, and its SCV stopped short of the spot: OpenBW's pathfinder marks a unit that
  got stuck on the way as immovable, and the build order then fails on arrival (the game's own
  checks, read through a debug call). The SCV nearest the spot now builds (not the one nearest the
  main, across the mineral line), and a build whose SCV stopped without starting it is ordered
  again at once with the same SCV, somewhere else after a second failure there. Both seeds held
  (the bunker at 2:23, first try). 9-pool **lost 3, won 18** of 48 (the 3: the rush seen only by
  the 2:20 rule, the bunker down at 2:27 with 1-2 marines when the zerglings came); 7:00 rush lost
  0, won 11; Zerg v2 lost 0, won 10; home defenses −55. (Keeping the bunker's 100 minerals until
  it was placed, on top of its reservation, had held 200: 4 lost.)
- **Pro Zergs, replayed** (`gary/bots/zerg_replayed.py`, `--opponent replayed`): a test fixture,
  not a player. Each seed picks a pro Zerg from a TvZ game and the env creates its zerglings,
  hydralisks and mutalisks when that pro's were done, its buildings when ordered, and zergling
  speed when researched; everything goes at Gary as it hatches (the dummy's waves). No economy
  (killing its drones changes nothing): it tests holding pressure, never Gary's economy. 18 of
  48 seeds drew a pool first (from a 4-pool at 0:45 to a pool at 1:40); pool-first armies reach
  the Terran at 3:15-3:56, hatch-first at 4:35-6:16. Gary v0.11, 48 seeds: **lost 7**, won 0
  (4 of 18 pool first, 3 of 30 hatch first). The hatch-first losses: Gary expands with 1 Barracks
  and no bunker at the natural, and 15 zerglings at 5:00 run over 5 marines (pros' 1-rax expand
  puts a bunker there). Every unit the pro made attacks, so it's harsher than the pro was.
- **Which maps pros wall** (TvZ, 22 maps, 40 games each, at 5:00; per start location, since a
  natural's size can differ a lot between starts): the natural closed to zerglings or narrowed to
  half on Colosseum II (64%, every start), Neo Medusa (53%; 7 o'clock 8 of 11 closed, 10 o'clock 2
  of 10) and Longinus (33%; 7 o'clock 5 of 9, 10 o'clock with a wider natural 0 of 9); at most 13%
  elsewhere; Fighting Spirit, the benchmark's map, 0 of 29 (its natural is ~10 tiles wide: pros
  hold it with a bunker). Width alone doesn't decide it (Eye of the Storm's naturals are as narrow
  as the walled ones, and never walled). Funnel walls (from the Command Center to the cliff, the
  mineral line left open) aren't counted: they don't close the frontier. On Colosseum II
  (`tests/fixtures/replays/stardata_tvz_colosseum_pqt42.rep`): each natural's choke has creep built
  into the map, and a build ordered onto creep still in the fog fails when the SCV arrives (Gary's
  wall depots did); the executor now knows the map's creep from the start. The frontier can keep to
  buildable ground (`frontier(buildable=...)`): there it finds the pros' wall line (two Supply
  Depots and a Barracks just inside the natural's ramp). The sim city planner (`+simcity`, a door
  left; `+simcity_closed`) now builds on that line by 1:30-2:40, but one building at a time leaves
  6-26 px gaps zerglings get through. Parked: Gary holds rushes without walls now. Next for walls
  (copy the pros' walls per map and start, lifting the Barracks or leaving a gap as they do): a
  GitHub issue.

### v0.10 (October 2026): roles (#2, docs/CONTROL.md)

- **What it is:** v0.9 with Gary built from layers (`gary/agent/`, every version a manifest), and
  its fight decided as a role per unit every half second instead of by the fight command model:
  fire, close in, pull back (hurt units in contact step away from their attackers), retreat (into
  a bunker with room; away only when not at home), by hand rules (`gary/agent/roles.py`), each
  role's skill choosing where to go from the threats' weapon ranges and the walkable ground.
  Leases enforced: the rules' units (loading, repairing, pulled SCVs) aren't taken by the fight.
- **Results:** 114 home defenses **−57** with the reactive Zerg, mean of 8 draws of Gary's hands
  (v0.9 −70, pro +1; one draw had said −28 vs −77; see the next bullets for why the numbers
  moved); immediate 9-pool held 11, lost 0 (16 seeds; 4 of 48 later)
  (v0.9 13 / 1); 7:00 rush 16 / 0, 38 buildings at 12:00. Worst case: 1316133c (17 zerglings
  at the base, 15 SCVs lost; v0.9 3, the pro 8).
- **The benchmark, made fairer and steadier:** the scenario Zerg replayed the pro Zerg's
  commands even after Gary's play had moved away from the pro's, so its zerglings never reacted
  to where Gary's units really were. Now it reacts once the game diverges (`--zerg reactive`, the
  default; eval/scenarios.py's doc): stale moves become a surround past the nearest Terran unit,
  then an attack once half the group is wrapped around (lings dying while they move in is
  normal TvZ; attack-moving them in one by one would be the Zerg's mistake). And one draw per
  scenario was a coin flip: the same fight under slightly different hands differs by ~95 value
  (hold, or lose 10 SCVs), so `--draws 8` scores the mean over Gary's hands seeded 8 ways.
  The bias turned out small: only 0-1% of the Zerg value Gary or the pro killed was zerglings
  moving to a spot no Terran unit was near; the reactive Zerg is a little *easier* (v0.10 −54
  with the replay, −46 reactive). Versions, reactive Zerg, 8 draws, paired with v0.10 (± 1
  s.e.): learned roles after RL −43 (+3 ± 9), v0.10 with brace −54 (−8 ± 11), v0.9 −76 (−30 ±
  16), v0.8 −81 (−35 ± 14), v0.3 −146 (−100 ± 20).
  Two more fixes then: the Zerg reacts only around Terran units that aren't where the pro's were
  (marines holding the pro's wall gap keep facing the pro Zerg's own choice not to attack them),
  and a Gary taking over midway has done its early scouting (it used to send an SCV across the
  map at the takeover, which drew zerglings off: v0.10 −57, v0.9 −70, −13 ± 17 to v0.10).
- **Where the gap to the pro is** (home defenses, 8 draws): all in what Gary loses (53 more value
  per scenario; it kills as much), mostly marines (+22) and SCVs (+17; Gary's fight 40% more
  SCV-seconds, in more scenarios), then vultures (+8: attack-moved into zerglings; the pros lay
  mines and hold behind the wall). 10 scenarios make up 44 of the 56. Gary's marines die alone
  (48% with at most one friend within 4 tiles, the pros' 29%), alone already 3 s before; while
  the zerg approach at home the pros stand nearer the Command Center in bigger groups, and the
  standoff lasts twice as long. And Gary's fight waited for its hands 55 times (166 s), mostly
  behind a build that was only waiting for its worker or money.
- **Tried against it, none a clear gain** (options on top of a manifest, `--version v010+post`;
  gary/agent/manifests.py OPTIONS, each with its result): letting the zerg come (brace), closing
  in only on zerg attacking (leash), closing in by moving then firing together (volley: −38; a
  moving marine doesn't shoot, and Gary's half-second turns plus reaction time can't stop it in
  time), loose units rejoining the pack, staying on hold position, vultures kiting, firing in
  place or on hold, pulling workers only when attacked, fights interrupting waiting builds (the
  waits fell to 54 s, the score didn't move), and a post by the Command Center (+8 ± 9, the best).
  v0.10's rules look like a local optimum on this benchmark, whose noise (± 9 paired) hides
  changes under ~15. So the scenarios now come from all 1,411 held-out games (`--pick 2500
  --screen`: 1,982 early fights, 239 decisive home defenses, one per game; pro −5, nothing −305,
  v0.10 −69, v0.9 −98, −30 ± 12): there the post was +7 ± 5 and fights interrupting waiting
  builds +6 ± 4, but against the immediate 9-pool (16 seeds) the first lost 1 game and the second
  3, v0.10 none: the build a fight would interrupt is the bunker that holds the rush.
  RL on the roles against the reactive Zerg (train/roles_rl.py, from the rules distilled; 239,
  8 draws): 60 rounds +8 ± 10 to v0.10; 200 rounds with twice the data drifted worse after
  round 20 (validation on 4 draws each: −71 at the start, −84 at the end; its best, round 20,
  +3 ± 9). Not adopted.
- **Vultures** (239, zerg within 8 tiles): the pros' move 44% of the time, attack a unit 33%,
  hold 8% (patrol 1%, spider mines 14 in 53 games: rare this early), at 2.9 tiles/s; v0.10's
  attack a unit 65%, move 7%, at 1.3 tiles/s, and lose twice as many (51, the pros' 25). A
  skirmish rule (`+vultures`: away from zerg too close, toward the marines, not into the
  workers; shoot when ready; ease off while reloading) brought the orders near the pros' (move
  42%, attack 29%, 1.8 tiles/s) and lost 18% fewer vultures, but scored −14 ± 4: fewer zerg
  killed, more SCVs lost. Vulture micro (keeping them at speed while shooting: patrol, move
  shots) needs finer hands than half-second turns; later, with supervision or learning.
- **How we got here:** see v0.9's last bullets (CONTROL.md steps 1-5): the layer refactor with
  parity checks, per-command intent labels that didn't hold up when watched, roles per unit that
  did, then the rules beating both the learned fight model and a role model learned by
  imitation.
- **Most urgently missing:** a learned role decision that beats the hand rules (RL on the fairer
  benchmark); positioning before contact; a sharper benchmark (more scenarios); vulture play
  (mines).

### v0.9 (October 2026): holding early pools (#2)

- **What it is:** v0.8 that notices an early Spawning Pool (seen before 3:00 with one hatchery,
  zerglings before 4:00, or 6 at once before 5:00) and answers it like a player: a bunker on the
  main's high ground by the top of the ramp (from the new map analysis; the closest free spot to
  the ramp's top can be at its bottom, so only spots on the main's level count), the Barracks
  rallied behind it,
  marines into the bunker as soon as it's done, 2 SCVs repairing it under attack, the army model
  kept from walking the marines off, and a scripted one-base build (depots on time, marines and
  SCVs nonstop, a second and third Barracks on spare money) instead of the learned macro until
  it's held (from 5:00, a finished bunker and marines at least half the most zerglings seen, 6
  at least). Then the usual plan.
- **Map analysis** (`gary/terrain.py`, #23): the main plateau by ground height; routes that keep
  to the middle of passages find the main ramp (top, bottom, width), the natural and its choke.
  Mains on the same level as their natural (Heartbreak Ridge) are flagged unreliable; there the
  bunker goes between the main and the natural.
- **Results:** immediate 9-pool (`eval/vs_rush`, 16 seeds, 10 min): **held 11, lost 3** (v0.3:
  held 5, lost 5; v0.8: held 1, lost 9). Held = alive at 10:00 with 15+ buildings and an army at
  least the Zerg's. The 3 losses are the seeds where the scout found the pool last (3:27 or
  later) and the first zerglings arrived before the bunker. The 114 home defenses: **−77**
  (v0.8 −102, v0.7 −120, v0.3 −156, same code otherwise): the bunker play helps there too.
- **Bugs found on the way, fixed for every version:** the engine's placement check skips the
  terrain of fogged tiles (you may order a build into the fog), so Gary could pick a spot half on
  a ramp's edge; the command was "accepted", then silently dropped once a unit saw the spot, and
  the SCV kept mining. Gary now checks the map's buildable tiles too. Loading a bunker gave up
  when the marines and the bunker didn't fit on one screen; it now selects the marines, then
  moves the camera. Start locations ignored a base whose geyser had an Extractor on it, which
  crashed scouting in 5 of the 114 scenarios (their earlier scores left them out). Gas: Gary
  lost count of its gas workers (one stepping out of the refinery has a different order for a
  moment, and the SCV that built the refinery starts on gas by itself) and kept sending more, up
  to 10 on one refinery; it now counts every SCV it sees heading into the refinery.
- **Less over-reaction** (after watching it): the rush build (bunker, marines, no gas) now ends
  once the bunker stands with 4 marines and no zerglings are at home (gas between 3:17 and 5:20
  in most games, not never); only the army stays home, no expansion, until it's 1.5x the biggest
  zerg army seen. New marines out of a Barracks while zerglings are at the base get an
  attack-move to the fight (the rally walked them past it, not shooting). Depots start while each
  producer has 2 supply left. Home defenses: −77, unchanged.
- **The benchmark's Zerg was unfair** (seen in a side-by-side: one Gary got its first wave a minute
  later): the dummy box-selected a larva with a tiny box, new zerglings stood on the larvae, the
  box caught one, and the dummy reselected for up to 50 s, depending on where larvae spawned (the
  shared random numbers, so on how Gary played). It now rallies its hatchery away from the larvae
  and picks a clear larva: both Zergs now match to the second until the fighting. Its waves come a
  little earlier (first at 3:42, was 3:51). Rerun on it (earlier numbers aren't comparable):
  immediate 9-pool, held / lost: **v0.9 13 / 1**, v0.3 5 / 8, v0.8 2 / 8; 7:00 rush: 16 / 0 for all
  three, 37 buildings at 12:00 for v0.9 (v0.3 39, v0.8 38). Tried and dropped: holding until the
  Zerg army has been seen (or 6:00); against the 7:00 rush, whose zerglings stay home, it kept
  Gary on one base until 6:00 and two games nearly died (25 buildings on average).
- **Marine timing** (`eval/micro_timing.py`, the 114 home defenses, ~71 marine-minutes of
  fighting each): pros' marines get 8.4 move orders per minute of fighting and shoot again 0.84 s
  after a move (median; 46% within 1 s); v0.9's get 1.5 and shoot again after 1.6 s (15% within
  1 s); after a move 14% die before shooting again (pros 7%); 271 marines lost to the pros' 221.
  Gary doesn't move too much: it moves too rarely and too long, no stutter step.
- **Viewer:** with a POV log, videos show the player's whole 640x480 screen: the view, the console
  strip with the minimap where the player has it (minimap clicks show there; the POV log now
  records where they land), the actions on the right, and minerals, gas and supply top-right.
- **Control layers** ([CONTROL.md](CONTROL.md)): each layer (posture, macro, executor, army,
  fight, arbiter) is a swappable provider and every Gary version a manifest (`gary/agent/`; play
  any with `python -m gary.play --gary v09`); the v0.1-v0.9 class chain is gone. Every action of
  110 games and scenarios and every drill result identical before and after. The arbiter keeps a
  trace (who got Gary's hands, for how long, who waited; `python -m gary.agent.trace`), again
  with every action identical. Gary's hands can ctrl+click (every unit of a type on screen, up to
  12, by the game's own rule as OpenBW's client has it). Tried in the fight layer (`ctrl_click`:
  a group that is every unit of one type on screen gets a ctrl+click, not a drag box): rarely
  applies (52 ctrl+clicks to 3,147 boxes in 16 rush games), no gain (9-pool held 12 / lost 1,
  home defenses -88 vs v0.9's -77, within noise); not in any manifest.
- **Intents and leases** (CONTROL.md step 3): the fight rules, models and the army issue intents;
  a motor layer carries them out with today's clicks; leases record who last ordered each unit
  (traced conflicts: in one 9-pool game the fight model and the bunker-load rule took marines
  from each other 9 times in 8 minutes). Every action identical in the 110 parity games.
- **What marines' moves are for** (`eval/micro_intents.py`, a script's labels on the 114 home
  defenses): pros give 253 move commands to fighting marines, v0.9 72. Pros: approach 34%, tap
  (under half a tile: cut short or a nudge) 19%, retreat 17%, stutter 8%, kite 8%, pull back (a
  third of the marines, the ones under attack, while the rest shoot) 4%, other 10%. v0.9:
  retreat 40%, approach 28%, tap 8%, stutter 7%, kite 4%, pull back 1%. Gary moves rarely, and
  mostly away. Clips of pro examples per label, to check the labels by eye:
  `data/interim/videos/label_check/` (made from the saved replays and events).
  Watched: per-command labels don't hold up (most moments mix techniques: the front marines
  step back while the rest shoot); approach, retreat and pull back read clearly.
- **Roles per unit** (`eval/micro_roles.py`): every fighting marine, every half second: fire,
  close in (move toward the zerg), pull back (a short stretch away, started under threat), retreat (4+ tiles away),
  reposition, stand. Toward and away are group against group (along the line from the marines
  around to the zerg around): measured unit to unit, marines retreating through interlaced
  zerglings looked like they chased the ones ahead (watched). Pros: fire 58%, stand 21%,
  reposition 7.1%, close in 5.0%, retreat 4.0%, pull back 2.4%; v0.9: fire 67%, stand 12%, close in
  5.5%, retreat 4.4%, reposition 3.3%, pull back 1.1%. Zerglings got onto a marine 154 times
  against the pros, 246 against v0.9 (same scenarios); pulled back within a second 25% vs 18%. Clips of single pro marines per role:
  `data/interim/videos/role_check/`.
- **Roles instead of the fight model** (CONTROL.md step 4, `gary/agent/roles.py`, hand rules
  `rules/v1`: fire, close in, pull back, retreat into a bunker with room or out of the threat's
  reach; leases enforced). Manifests `v09lease` and `v09roles`, against v0.9:
  | | v0.9 | + leases | + leases + roles |
  |---|---|---|---|
  | 114 home defenses | -77 | -75 | **-44** (-37 on the first run) |
  | immediate 9-pool, held / lost | 13 / 1 | 12 / 1 | 8 / 2 |
  | 7:00 rush, held / lost | 16 / 0 | 16 / 0 | 16 / 0 |
  Leases alone change nothing measurable. The roles' hand rules beat the learned fight model in
  pro home defenses by a wide margin, and do somewhat worse against the 9-pool. Their worst home
  defense (1316133c: 15 SCVs lost vs 3) retreats the marines and leaves the SCVs to the lings.
  The first version retreated away from the bunker too (9-pool held 6, lost 5): now a retreat
  goes into a bunker with room first. Videos (pro | v0.9 | roles): `data/interim/videos/roles_vs/`.
  Then: no walking away at home (a Command Center within 14 tiles: only into a bunker, else the
  army fights with the SCVs), and SCVs fighting with the group count in its strength. Home
  defenses **-28**, immediate 9-pool held 11 / lost 0, 7:00 rush 16 / 0 (38 buildings): the
  roles' hand rules now beat v0.9 on the home defenses by 49 points and lose no rush game. Still
  bad: 1316133c (17 zerglings at the base, the roles hardly act, the pulled SCVs die: 15 lost,
  v0.9 3, the pro 8).
- **A learned role model** (CONTROL.md step 5: `ingest/role_dataset.py`, `gary/policy/roles.py`,
  `train/roles.py`): the pros' units in the 552 training home defenses, every half second, the
  fight as the Terran sees it plus each unit's motion and how many enemies target it, labeled
  with its role (17,522 snapshots, 48,344 units). Held out: 77% right (always "fire": 52%); recall
  / precision fire 0.87 / 0.92, close in 0.53 / 0.46, pull back 0.56 / 0.21, retreat 0.46 / 0.53,
  stand 0.92 / 0.91. In play (`v09roleml`, the same skills): home defenses **-98** (hand rules
  -28, v0.9 -77), immediate 9-pool held 9 / lost 3, 7:00 rush 16 / 0. It names "stand" for 28%
  of fighting marine time (no command), and fires less: it learned what the pros' units were
  doing, mostly the continuing effect of earlier commands, not when a player decides something.
  Then decisions (`train/roles.py --decisions`): a unit's role only where it changed from half a
  second before, else "keep" (70% of labels). Held out, recall / precision: fire 0.48 / 0.37,
  close in 0.47 / 0.25, pull back 0.40 / 0.24, retreat 0.15 / 0.10. In play: home defenses -70
  (states -98, hand rules -28, v0.9 -77), immediate 9-pool 11 / 2, 7:00 rush 16 / 0.
- **RL on roles** (`ingest/role_distill.py`, `train/roles_rl.py`): RL needs a model to start from
  and the best decisions were the hand rules, so they were distilled into the role network first
  (v0.10 playing the 552 training defenses, its decisions as labels: home defenses -31 unweighted,
  -71 with rare roles weighted up). Then 60 rounds on the training home defenses (32 positions x
  8 draws, roles drawn per unit, per-unit PPO, KL to the start): validation -97 -> -74 at round
  30, then drifting back (-89 at 60), as the command model's runs did. Held out, round 30: **-28**,
  the same as the hand rules. RL on roles is stable and reaches the rules, not beyond them yet.
- **Positioning before contact** (zerglings reach Gary's marines more often): while zerglings
  approach (nearest 7-16 tiles, closing), pros' marines stand 37% of the time, attack-move 30%,
  move across 17%, toward 8%; v0.10's attack-move 60% and stand 23%; contacts per marine-minute
  1.32 vs 1.63. 72% of those attack-moves were v0.10's own close-in role sending marines at
  zerglings that were coming anyway. A brace role (`roles.py`, option `brace`): no closing in
  while the zerg come; units ahead of the pack step back, units charging stop. With the fight
  starting at 16 tiles instead of 8 it was worse (home defenses -57, 9-pool 8 / 3: close in
  chased from farther); at 8 tiles (`v010brace`): attack-move 53%, stand 28%, contacts 1.58,
  home defenses -32, a tie with v0.10. Not adopted; the approach gap stays mostly open. Next: intents, and
  the first experiment (one model doing the fight's timing vs intents plus a stutter-step skill).
- **Most urgently missing:** stutter-stepping (a timing problem, see above); spotting the pool in
  time when the scout checks the Zerg's start last; terrain in the fight model; the screen (SCVs
  in front of marines); wall-offs.

### v0.8 (October 2026): fight model trained by reinforcement learning (in progress, #2)

- **What it is:** v0.7 with its command model trained further by reinforcement learning
  (`train/fight_cmd_rl.py`) on real home defenses from training games (498 pro positions, the pro
  Zerg's attack; Gary playing as v0.7 otherwise) and on `home_defense` drills, kept close to the
  imitation model; 54 validation defenses picked the checkpoint (round 70: −145 → −86). With it
  v0.8 carries out every command type again (no guard rail).
- **Results** (114 held-out home defenses, 8 draws): **−61**, 1.8 SCVs lost (v0.7 −100, 2.0;
  pro +29; nothing −260): the first learned fight model clearly better than the guard rail
  (about three times the noise). Its moves stay pro-like (about half toward the zerglings);
  pushed further, training drifted toward longer moves away from them, which validation caught.
  Immediate 9-pool: holds 1 of 8 (v0.7 2): still decided by macro and placement.
- **How we got here:** drills alone taught it to run away (nothing to defend) or overfit them;
  drills with something to defend helped some; training on real positions with real validation is
  what transferred. Replays of each stage: `eval/record.py`, and the trainer records at every test.
- **Fights share Gary's hands** (drills' Gary, now the `drill` manifest in `gary/agent/manifests.py`; `train/fight_cmd_rl.py`): drills got
  a macro chore (checking the Command Center between fight commands), camera interruptions and a
  random APM budget; the command model can take Gary's hand state as input (APM tokens left,
  production waiting, time since the last macro action, camera on the fight); each command costs a
  little in the reward. Retrained like v0.8's run (from imitation; run 6): the same score on the
  114 home defenses (−63 vs −61) with 9% fewer actions (77 vs 85 per play; "nothing" 54% of
  decisions vs 23%), reached in 40 rounds instead of 70. Not adopted (no gain in these scenarios,
  which ask little of macro). Continuing v0.8's own model instead (run 5) slid back at once.
- **A fuller score** (`eval/score.py`, used by scenarios, drills and RL rewards alike): deaths;
  damage as partial credit (HP lost times value, half weight, less for the regenerating Zerg and
  for repairable buildings; healing counts back); spellcasters' energy spent; and enemy fighters
  still near a Terran base at the end, against the Terran (so backing off a defense doesn't pay).
  Same ranking on the 114 home defenses: pro +1, v0.8 −91, v0.7 −130, v0.3 −161, nothing −302.
- **RL with the fuller score** (run 7, from imitation, with the shared-hands changes): validation
  −172 → −126 by round 40, then the drift toward backing off returned (slower than before);
  the 114 home defenses: −110 (v0.8 −91 on the same score). Seven RL runs in, the fight model
  peaks near v0.8's level after 40–70 rounds and then drifts; more reward tuning looks like
  diminishing returns. What it can't see is a likelier limit: terrain, recent damage, where the
  threats come from.
- **Most urgently missing:** holding early pools (macro, placement, walls; v0.9); terrain in the
  fight model; the screen (SCVs in front of marines).

### v0.7 (October 2026): defense fundamentals (#2)

- **What it is:** v0.6 with the basics a player does in a home defense, as rules, each found by
  watching Gary and the pro side by side in the same scenario and then counted on the suite:
  - **finish what it started:** an unfinished building nobody builds (its SCV was killed) gets
    another SCV (all versions; unfinished buildings left: 0.20 → 0.07 per scenario, pro 0.04)
  - **produce during fights:** after each fight command, production gets a turn, not chores like
    refilling gas (marines finished in 45 s: 1.5 → 2.4, pro 2.1)
  - **load the bunker** when zerglings come at it (pro ~1 load per scenario)
  - **workers pulled by v0.3's rule**, first in a fight, faster; the command model no longer
    commands SCVs
  - **no plain moves or stops from the model** for fighting units (a guard rail: they took
    marines out of fights, as watching Gary vs the pro showed; −129 with them, −117 without, the
    same as no model in fights at all), until reinforcement learning teaches when a move helps
- **Pro-speed hands** (all versions, live too): the `pro` profile is fitted to pro replays: from
  selecting units to the next targeted command pros take 168 ms (median, home defenses), Gary's
  old b_rank hands 504 ms. Gary now gives 107 commands a minute there (b_rank 80; the pros 185
  effective: the rest is Gary's decision loop, not its hands). With them: home defenses v0.7 −92,
  v0.3 −131, v0.6 −185; the immediate 9-pool: v0.3 holds 4 of 8, v0.7 2.
- **Results** (b_rank hands; 114 home defenses, 4 draws): **−117**, 2.1 SCVs lost: the first Gary better than
  v0.3's −157 (v0.6 with the first three fixes −195; pro +29; nothing −260). Against the
  immediate 9-pool it holds 2 of 8 seeds, like v0.3: that's decided by placement and macro
  (bunker, marines, walls in time), not by fight micro.
- **First RL on micro drills** (`gary/drills.py`: marines and medics vs zerglings, made-up fights
  on real maps as preplaced units, watchable replays; `train/fight_cmd_rl.py`): held-out drills
  −281 → +71 in 90 rounds (doing nothing +24), but it learned to **run away** (its moves go a
  median 860 px from the zerglings, never toward them): the drills have nothing to defend and the
  scripted zerglings re-target only once a second. On the 114 home defenses without the guard
  rail it barely helps (−126 → −121); in 07720d60, a marine fight, it helps a lot (−428 → +166),
  because the replayed Zerg can't punish a retreat. The guard rail stays; drills need something
  to defend and zerglings that chase.
- **Second RL run** with a `home_defense` drill family (a real base, SCVs mining, zerglings
  coming for the mineral line; everything lost counts) and zerglings that chase, trained together
  with `mm_vs_lings`: held-out `home_defense` +9 → +152 (attack-move +103, nothing −16) with
  healthy moves (~150 px, a third toward the zerglings); on open ground it drifted toward standing
  and backing off (moves 171 → 389 px), so the run stopped at round 40. On the 114 home defenses
  **without the guard rail: −81** (imitation model −126; v0.7 with its guard rail −92), moving
  like the imitation model (232 px, 43% toward the zerglings): the first learned fight model at
  least as good as the guard rail, within the noise (±14).
- **Third run**, `home_defense` only, from the second: drills +152 → +183, but the home defenses
  went back to −100 (moves longer, 281 px, fewer toward the zerglings): past a point, getting
  better at drills means getting better at drills. Neither RL model is clearly better than the
  guard rail; the drills miss too much of a real defense (the worker pull, bunkers, terrain,
  buildings, how the Zerg attacks).
- **Most urgently missing:** the pulled SCVs fight as a blob; the pros put them between the lings
  and the marines (a scripted "screen" next, then the ways for a model to learn it: unit features
  for the relation, synthetic micro drills with RL); macro against the immediate 9-pool.

### v0.6 (October 2026): the command model remembers (#2)

- **What it is:** v0.5 with a command model that also sees Gary's own commands of the last 8 s:
  per unit, what it was last told (kind, how long ago, where to), and the last 6 commands as
  extra tokens (`gary/policy/fight_memory.py`). In training these are the pro's own preceding
  commands (fight set v3 keeps unit IDs, fight centers and every command); in play, the human
  interface's log of what Gary's hands did.
- **Results:** offline, memory helps most with where to send units (destination within one cell
  54%, without memory 30%), then targets (57% vs 50%), selection IoU 0.70 vs 0.65, command type
  0.56 vs 0.49. Pro scenarios: −173 net, 2.7 SCVs lost (v0.5 −191, 3.2; pro +74; nothing −255).
- **What the pros do that Gary doesn't (counted in the data):** worker drilling. The pro
  right-clicks SCVs onto a far mineral patch so they pass through each other and the zerglings
  (mining workers don't collide), then attack-moves when the stack is on top of the lings. In the
  45 decisive scenarios the pro drilled in 22–24% (in the rest 7–10%), and those are the biggest
  swings (the pro's edge over doing nothing: 441–475 with a drill, ~290 without), about a third of
  the pro's total edge. A two-step, precisely timed trick like this isn't learned from clicks.
- **The suite was partly wrong:** 14 of those 45 scenarios were the Terran attacking (10 bunker
  rushes), which Gary's macro never plans. The picker now keeps only defenses at the Terran's home
  (114 games, one fight each). Re-scored there: pro +29, **v0.3's scripted worker pull −157**,
  v0.6 −209, v0.5 −211, v0.4 −216, nothing −260. On plain defense every learned fight model is
  worse than v0.3's simple rule. Drilling: 18% of these defenses, about a quarter of the pro's
  edge ([#21](https://github.com/vibrunazo/gary/issues/21); bunker rushes:
  [#22](https://github.com/vibrunazo/gary/issues/22); both are advanced techniques for later
  intents, not now).
- **Most urgently missing:** the fundamentals of defense, measured on this suite and on the
  immediate 9-pool (`eval/vs_rush`).

### v0.5 (October 2026): fight command model (#2)

- **What it is:** v0.4 with the fight model replaced by a command model: in a fight, every half
  second Gary draws the next command a pro would give (its type, which of Gary's units it goes to,
  up to 12, and its target unit or point), instead of an order per unit that Gary then regroups.
- **New:** `gary/policy/fight_cmd.py` and `train/fight_cmd.py` (commands rebuilt from the fight
  set: units told the same thing in the same half second; 0.7M training commands); its encoder
  starts from the fight model's; destinations as cells of a 32x32 grid around the fight.
- **Results:** offline (held-out games): selection IoU 0.74 (all own units: 0.55), target 52%
  (nearest unit: 27%), destination within one cell 32% (staying put: 19%); command type 59% when
  the pro commanded (most common: 65%, rare commands are weighted up). Pro scenarios (45, 4 draws
  each): −192 net, 2.8 SCVs lost (v0.4: −207, 3.3; pro +74; nothing −255). It now pulls SCVs,
  attack-moves, stims and sends workers back to mining, but in groups of 1–4 where the pro pulls
  12, and about one command every 1–2 s against the pro's ~1 per second.
- **Hands are not the bottleneck:** in fights pros select 3.9 units per command on average (5%
  of commands: 12); the model picks ~3. Of those, only 1.7 got the order; with the camera on the
  group, a wider drag box, reusing the selection and camera hotkeys, 2.2 do and Gary gives 24
  orders per scenario instead of 19, but the score stays (−193). With hands twice as fast (46
  orders, the pro's count) it's −210, and sampling closer to the model's likeliest commands
  (temperature 0.5, 0.25) gives −189: what Gary decides is what's missing, not how fast.
- **Wider view (fight set v2):** the fight snapshot now takes Gary's own units from 768 px (the
  whole mineral line; up to 64 units). Retrained on it (0.75M commands; selection IoU 0.65 vs
  0.43 for all own units, target 50%, destination 31%), Gary's commands grow to the pros' size
  (3.7 units chosen, the pros' real average is 3.9), but the score stays (−191).
- **Most urgently missing:** coherence over time. The model decides each command from the
  current snapshot alone, so Gary switches plans every second or two (move, attack-move, back to
  mining, move) where a pro holds one for several seconds (evacuate the mineral line, then come
  back): the command model needs memory of its recent commands, or a slower plan (intent) above it.

### v0.4 (October 2026): learned fight model (#2)

- **What it is:** v0.3 plus the fight model: whenever enemy fighters are near Gary's units, a
  transformer trained on 7.2M pro TvZ skirmish snapshots gives each unit near the fight an order
  (attack which unit, move where, attack-move, gather, repair, stop, hold, stim), carried out with
  human hands, one group per half second. Fights come first and may interrupt chores.
- **New:** fight data (resim `--fights`), fight model and training, env observes weapon cooldown,
  order target and carrying, chat announcement of the version and models, screencast-style action
  overlay in POV videos.
- **Results:** offline, the right action 55% of the time when the pro acted, the right target 35%
  (21% for "nearest unit"). In games, no better than v0.3 yet (8 seeds: immediate rush 5 lost; 7:00
  rush 1 won, 6 ahead). In pro scenarios (`eval/scenarios.py`: 45 early skirmishes from held-out
  games where the pro's control mattered; net value over 45 s, zerg lost minus terran lost): the
  pro +74, v0.3 −194, v0.4 −253, doing nothing −255. The fight model gives about 14 orders where
  the pro gives about 40, and rarely commits to what decided these fights: pulling a dozen SCVs
  with attack-move while marines kite. Sampling decisions from the model (instead of fixed
  thresholds) scores −207, and is now the default. Reinforcement learning from scenarios (`train/fight_rl.py`: 40 rounds
  of 512 plays, group-relative advantages, reward-to-go over 20 s, KL to the imitation model) left
  the held-out score where it started (−222 → −217); on 4 scenarios it trains and is scored on it
  gains about +90, so the loop learns but doesn't yet generalize from this much play.
- **Most urgently missing:** a fight model that acts like the pros in those scenarios (the
  scenarios are now the test); holding the immediate rush through the macro (bunker and marines
  in time).

### v0.3 (October 2026): learned macro and learned army

- **What it is:** v0.2's learned macro plus the army model: where each army cluster goes, move or
  attack-move, and a fight estimate that holds back attacks expected to lose army share. Scripted
  early scout and worker-defense reflex.
- **New:** army model with 3-minute enemy memory and fight estimate; armies as location clusters
  ordered by drag box; tile fallback for town halls; mining at Gary's own bases only; idle-worker
  sweep; actions recorded in words in the POV log.
- **Results** (Fighting Spirit, steered to standard bio with `--style 1`, 8 seeds,
  `eval/vs_rush.py`): against a zergling rush that attacks at 7:00, 1 win and 4 games far ahead at
  14:00, 3 nearly lost; against an immediate rush, 5 lost, 2 nearly lost, 1 holding.
- **Most urgently missing:**
  1. Holding early all-ins (bunker and marines timed to an early pool): [#2](https://github.com/vibrunazo/gary/issues/2).
  2. Closing out won games (armies advance one or two grid cells per order): [#3](https://github.com/vibrunazo/gary/issues/3).
  3. Micro: [#4](https://github.com/vibrunazo/gary/issues/4).
  4. Live play: the bridge's town-hall placement check is a tile off: [#5](https://github.com/vibrunazo/gary/issues/5).

### v0.2 (October 2026): learned macro

- **What it is:** production decided by the macro model, trained on ~14k pro TvZ games; v0.1's
  scripted army (attack-move with every 16 units).
- **New:** macro model and executor for the whole Terran tech tree; waits for busy buildings;
  never takes a decision under 5%; count caps learned from pros.
- **Results:** plays pro-like builds when steered (`--style 1`); unsteered, it can drift into tech
  paths without building the units. Against the 7:00 rush, 2 of 4 seeds won.

### v0.1 (October 2026): scripted Terran

- **What it is:** a fixed one-Barracks expand into marines, attack-move with every 16 marines.
- **Results:** loses to any early pressure; no defense at all.

### v0 (October 2026): hello world

- **What it is:** mines and makes workers through the human interface. Proves the loop works.

## Keeping this file

- **A new version:** add its column right after "Now" (newest first; older versions are a
  horizontal scroll away), update "Now", and add a section on top.
- **When the table gets too wide:** keep the latest versions as columns; older versions keep their
  sections below, and git history has the old tables.
- **Day-to-day tasks** (bugs, next steps, ideas) belong in [GitHub Issues](https://github.com/vibrunazo/gary/issues),
  not here: this file is the snapshot per version.
