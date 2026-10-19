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
What it has found so far: RL on per-unit fight roles ends at the starting model (four tries, the
last with paired episodes and a live opponent); an army decision moves the game three times as
much as the chaos over 90 s, so the army layer is the next RL target; and Gary-Z still loses every
game, for reasons in its army and economy rather than its micro.

| Phase | Status | What exists | What's missing for its exit criteria |
|---|---|---|---|
| P0 Infra | ◐ | OpenBW env, human interface v1, POV and decision logs, eval against scripted Zergs, replayed pros and pro scenarios; ~150k 45-second scenario plays per hour; full games 14 at a time (one torch thread each: 2.7x faster); games deterministic per seed, replayed to any moment to branch, and branched live with both bots (`eval/pairs.py`: checked every game that a sampled branch replays the same); a run monitor page (`eval/monitor.py`); the Remastered bridge 1:1 with OpenBW (`adapters/scr_bridge/tools/parity.py`) | The formal trace schema; classic bots to play against; (a crash of the native env inside a branch, 1 game in 144: most likely memory starved by orphaned workers of a stopped job, since killed) |
| P1 Data | ◐ | ~70k replays indexed, resim with Remastered support (go decided: it works), fog-of-war views | Camera logger, camera inference, fitting human profiles from data (select-to-order times fitted; click scatter and camera still placeholders) |
| P1b Taxonomy | ◐ | TvZ build clusters with readable rules (`taxonomy/discover.py`) | A reviewed `taxonomy/tvz/v1.yaml`, stability and gold-set reports |
| P1c Strategy stats | ◐ (started) | The strategy selector's table: each build style's results per map in Gary's own games (`eval/selector_table.py`, v0.19) | Win-rate tables from pro replays, the feature library, guide claims |
| P2 Micro | ◐ | Fight, command and role models from pro skirmishes; home-defense scenarios and 120 s episodes from pro replays; roles by hand rules in contact, a role model learned from the pros' units before contact (v0.29); stim learned from the pros (v0.27); RL on the roles four times, each ending at its start (three for the Terran, one for Gary-Z on paired episodes): roles stay at imitation and rules; choke blocks as a skill | Responses within the human band: the humanlikeness classifier's fight rows (lone outnumbered units never move away, pros 11-18%; marines rarely stimmed at the sight of mutalisks; no moving between shots) |
| P2b First playable | ◐ | Full games against people, offline, via the bridge (two human games so far, both lost by Gary; their failure lists shaped v0.26-v0.27 and the classifier); learned macro steered by build style, the style chosen per map (v0.19) | Build timings within tolerance (≥18/20 unharassed runs); supply blocks and idle production under harass within the human band |
| P2c Debug viewer | ● | POV viewer with cursor, clicks and every action in words; videos | Trace overlays beyond actions |
| P3 BC full game | ◐ | Separate imitation models for macro, army movement and fights, for the Terran and (Gary-Z) the Zerg; a role model learned before contact as well as in it, each side; belief models of the enemy army, each side (frozen; off by 11.7 supply on league games against 5.9 on pros: Gary's armies aren't the pros'); a value network for the Terran (`gary/policy/value.py`); the humanlikeness classifier and its pro-distance table (`eval/humanlike.py`) | One policy with intent and belief heads, the Opponent Model; Gary closer to the pros in the classifier's table (it tells them apart at AUC 0.99+) |
| **P4 RL** | **◐ (current)** | RL on fights (scenarios, episodes: at the rules' level); macro from branches (nothing: a decision is worth sd 130 against chaos's 512); per game: the strategy selector (v0.19: losses 57 → 30 of 333), the early-defense switch; the self-play skeleton: Gary-Z v0 as a learned opponent, the league (`eval/league.py`, `eval/league_loop.py`: rounds on fresh seeds, a candidate kept only by paired A/B evidence, a classifier gap no worse and no creeping retreat), RL on Gary-Z's fight roles on paired episodes (three nights, no gain), an army decision measured at three times the chaos (`eval/army_effect.py`), RL on Gary-T's army layer on paired 90 s branches (running) | A league where both sides win games (Gary-Z wins none yet); Elo gain at equal humanlikeness |
| P5 Explain, P5b Broadcast | — | (The POV viewer's action overlay is a start) | |
| P6 Deploy | ◐ | Remastered bridge for offline games, 1:1 with OpenBW (a LAN replay re-simulates to the same state) | Pro blind-test sessions; the bridge's known bugs (a supply depot placed a tile off at expansions) |

## Components by version

| Component | What it is | Now (v0.29) | v0.29 | v0.28 | v0.27 | v0.26 | v0.25 | v0.24 | v0.23 | v0.22 | v0.21 | v0.20 | v0.19 | v0.18 | v0.17 | v0.16 | v0.15 | v0.14 | v0.13 | v0.12 | v0.11 | v0.10 | v0.9 | v0.8 | v0.7 | v0.6 | v0.5 | v0.4 | v0.3 | v0.2 | v0.1 | v0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Game: headless | OpenBW game Gary plays in for development and tests (`env/`, `gary/env.py`) | Melee games on any map, network-style command delay, replays saved | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● |
| Game: live Remastered | Gary in a real StarCraft: Remastered client over LAN (`adapters/scr_bridge`) | Plays, 1:1 with OpenBW (a LAN game re-simulates to the same state); known gaps in the bridge's map checks and hotkeys (its README) | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | — |
| Human interface | The only path to the game: screen clicks, camera, fog, reaction delay, APM (§6.3) | Clicks, drag boxes, camera hotkeys, fog, reaction delay, APM budget; pro-speed hands fitted to pro replays' select-to-order times yet | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● |
| Perception and memory | What Gary knows: visible units, remembered enemy units (§6.9, §6.10) | Fog and hidden enemy HP; remembers enemy buildings and the enemy army seen in the last 3 min; map analysis (`gary/terrain.py`): main ramp, natural, natural choke; walls (`gary/walls.py`): which units get between buildings and cliffs, and where the openings are (not used in play yet) | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ |
| Macro decisions | What to build, train, research, and when | Learned from ~14k pro TvZ games; steerable by build style; rules on top: a Barracks by 1:30, a bunker at the natural when expanding; from the third base on, an expansion once the model's odds of one add up to one (v0.23); against lurkers seen (or a Hydralisk Den with a Lair), the pros' answer (v0.24); against mutalisks an Engineering Bay, an Academy and turrets in each mineral line, the nearest the zerg first (v0.26-v0.27) | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ○ | — |
| Macro executor | Carries out macro decisions: placement, production, gas, mining (§5) | Scripted: any Terran building, unit, add-on, research, upgrade, expansion | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| Army decisions | Where armies go, move vs attack, whether a fight is worth taking | Learned destinations and a learned fight estimate; an attack it expects to lose 5+ points of army share is held, plain moves never (v0.28: v0.27's 1-2 points held ~90% of its decisions at 5-6 min); hunts the Zerg's buildings when no army has been seen for a minute (v0.16, v0.25); option `draw`: decisions drawn from the model's odds and logged, the league's learner (RL on paired 90 s branches, running) | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ○ | ○ | — |
| Micro | Unit control in fights: stim, spread, siege, focus fire, spells | A role per unit every half second by hand rules in contact (fire, close in, pull back, retreat; a post by the Command Center at home); before contact, the role model learned from the pros' units moves units that would reposition, pull back or retreat where its where head says (v0.29); stim learned from the pros' stims (v0.27); marines loaded into bunkers by rule, kept in them (v0.15) | ◐ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | — | — | — | — |
| Scouting | Finding out what the opponent does | One worker at 9 SCVs to the enemy start locations | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | — | — | — |
| Reflexes | Quick reactions outside the plan (worker defense, repair, retreat) | Scripted worker pull, 3 workers per attacker, not against flyers or more than 8 zerg supply (v0.26); unfinished buildings get a new SCV; SCVs repair a bunker under attack, and 3 a threatened turret (v0.27); burrowed lurkers near Gary's fighters scanned (v0.22) | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | — | — | — |
| Attention arbiter | One camera and one APM budget shared by all tasks (§6.7) | One task at a time in a fixed priority order | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| Strategy layer | Picks the build and switches plans from what it sees (§6.8) | The build style per map from counted game outcomes (`gary/agent/selector.py`, v0.19); `--style` overrides it; an early pool switches Gary to a scripted one-base defense until it's held; an alarm from nothing seen is called off until 4:00 (v0.18); the early-pool defense from 2:30 whatever is seen, unless the rules switched first (v0.22) | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | — | — |
| Opponent model | Beliefs about the opponent's build and army | A belief model of the enemy army from what the player saw, each side (`train/belief.py`: off by 5.9-5.3 supply on held-out pros); not used in play by default | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| Replay pipeline | Inventory, build orders, resim, desync checks (§7.1–7.2) | ~70k replays indexed; TvZ re-simulated with fog-of-war views | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● |
| Build taxonomy | Build styles discovered from replays (§7.3) | TvZ clusters with readable rules; style labels for ~45% of players | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● |
| Training sets | Data for the learned models | Macro, army and fight sets for TvZ (fight v3: own units from twice as far, unit IDs, every command); the response sets (`ingest/response_dataset.py`): the pros' units' roles before contact and in it, from each side's vision, 600 games a side | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | — | — |
| Learning beyond imitation | Outcome-weighted learning, reinforcement learning | Fight estimate learned from outcomes; a value network for the Terran (who wins, the net value of the next 60 and 180 s; reads who is ahead); the league's RL (`train/league_rl.py`): PPO on paired branches with a KL leash to imitation, gated by paired A/B evidence and the classifier; no gain yet on fight roles, the army layer next | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | — | — | — |
| POV viewer | Watch any game from Gary's screen (`viewer/`) | Cursor, clicks, every action in words; Remastered replays; videos | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | — |
| Trace | Gary's decisions and reasons, recorded (§6.5) | Decision log with model probabilities and fight estimates; POV action log; the learned layers' decisions per league game (`gary/decision_log.py`: inputs, the drawn decision, its log-probability) | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | — |
| Evaluation | Matches against opponents, many seeds, a scoreboard (§9) | `eval/vs_rush.py`: several seeds against scripted Zergs, pro Zergs' armies spawned on their timings (not for economic or humanlikeness judgments: its replays don't re-simulate), or Gary-Z; `eval/scenarios.py`: 239 pro home defenses, several draws of Gary's hands; `eval/humanlike.py`: the humanlikeness classifier and the pro-distance table, either side; `eval/league.py`: Gary-T against Gary-Z; `eval/zerg_losses.py`, `eval/army_effect.py` | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ○ | ○ | — |
| Matchups and races | Gary plays every race and matchup by design (ARCHITECTURE.md §1); TvZ was the first | Terran and Zerg in TvZ, the Zerg (Gary-Z) as a second class tree with its own execution: the one-Gary-per-race refactor is owed; no Protoss; the first ZvZ mirror games played on 2026-10-18 (eval/mirror.py, ZvZ models, gary/races.py); TvT datasets built, not yet played; the Protoss matchups sit in the inventory; the trainers select by era, source and matchup (train/select.py); the audit and the order of work: docs/RACES.md | T (+Z) | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T |

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
- **Next:** Gary-Z's banking (drones made, minerals unspent: hatcheries and larvae); a human game with
  the current Gary-T.

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
