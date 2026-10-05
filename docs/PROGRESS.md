# Gary — Progress

Where each Gary version stands against [ARCHITECTURE.md](ARCHITECTURE.md). The architecture
says what Gary should become; this file says what exists. Only the components that matter at the
current stage are listed; later ones (narrator, broadcaster, language models...) join when work
on them starts.

Legend: **—** not started · **○** scripted stand-in · **◐** partial · **●** working (for now)

## Roadmap phase

**Now: P2 Micro** ([ARCHITECTURE.md §12](ARCHITECTURE.md#12-roadmap)), with a rough
P2b (first playable) reached early: Gary already plays whole TvZ games against people, through
the Remastered bridge. The phases didn't go strictly in order: the macro and army are learned
by imitation (P3-style pieces) rather than scripted, because the replay pipeline made that the
quicker way to a playable Gary.

| Phase | Status | What exists | What's missing for its exit criteria |
|---|---|---|---|
| P0 Infra | ◐ | OpenBW env, human interface v1, POV and decision logs, eval against a scripted rush and pro scenarios; ~150k 45-second scenario plays per hour on 16 parallel workers (3x faster than before profiling, same results bit for bit) | Full-game games/hour benchmark; the formal trace schema; classic bots to play against; the interface parity test |
| P1 Data | ◐ | ~70k replays indexed, resim with Remastered support (go decided: it works), fog-of-war views | Camera logger, camera inference, fitting human profiles from data (the interface's numbers are still placeholders) |
| P1b Taxonomy | ◐ | TvZ build clusters with readable rules (`taxonomy/discover.py`) | A reviewed `taxonomy/tvz/v1.yaml`, stability and gold-set reports |
| P1c Strategy stats | — | | Win-rate table, feature library, guide claims |
| **P2 Micro** | **◐ (current)** | Fight and command models learned from pro skirmishes; scenario suite from pro replays (`eval/scenarios.py`); first reinforcement learning on scenarios (`train/fight_rl.py`) | Responses within the human band: on home-defense scenarios v0.9 is the best Gary (fuller score: pro +1, v0.9 −77, v0.8 −102, v0.7 −120, v0.3 −156, nothing −302); it holds the immediate 9-pool in 11 of 16 games (v0.3 5, v0.8 1) |
| P2b First playable | ◐ | Full games against people, offline, via the bridge; learned macro steered by build style | Build timings within tolerance (≥18/20 unharassed runs); supply blocks and idle production under harass within the human band |
| P2c Debug viewer | ● | POV viewer with cursor, clicks and every action in words; videos | Trace overlays beyond actions |
| P3 BC full game | ◐ (early) | Separate imitation models for macro, army movement and fights | One policy with intent and belief heads, Opponent Model, value network |
| P4 RL | ◐ (started) | Segment RL (T2a) on 45 s fight scenarios: runs, no gain yet | Elo gain at equal humanlikeness |
| P5 Explain, P5b Broadcast | — | (The POV viewer's action overlay is a start) | |
| P6 Deploy | ◐ | Remastered bridge for offline games | Pro blind-test sessions |

## Components by version

| Component | What it is | Now (v0.9) | v0.9 | v0.8 | v0.7 | v0.6 | v0.5 | v0.4 | v0.3 | v0.2 | v0.1 | v0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Game: headless | OpenBW game Gary plays in for development and tests (`env/`, `gary/env.py`) | Melee games on any map, network-style command delay, replays saved | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● |
| Game: live Remastered | Gary in a real StarCraft: Remastered client over LAN (`adapters/scr_bridge`) | Plays; known gaps in the bridge's map checks and hotkeys (its README) | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | — |
| Human interface | The only path to the game: screen clicks, camera, fog, reaction delay, APM (§6.3) | Clicks, drag boxes, camera hotkeys, fog, reaction delay, APM budget; pro-speed hands fitted to pro replays' select-to-order times yet | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● |
| Perception and memory | What Gary knows: visible units, remembered enemy units (§6.9, §6.10) | Fog and hidden enemy HP; remembers enemy buildings and the enemy army seen in the last 3 min; map analysis (`gary/terrain.py`): main ramp, natural, natural choke | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ |
| Macro decisions | What to build, train, research, and when | Learned from ~14k pro TvZ games; steerable by build style | ● | ● | ● | ● | ● | ● | ● | ● | ○ | — |
| Macro executor | Carries out macro decisions: placement, production, gas, mining (§5) | Scripted: any Terran building, unit, add-on, research, upgrade, expansion | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| Army decisions | Where armies go, move vs attack, whether a fight is worth taking | Learned destinations and a learned fight estimate; holds attacks it expects to lose | ● | ● | ● | ● | ● | ● | ● | ○ | ○ | — |
| Micro | Unit control in fights: stim, spread, siege, focus fire, spells | Command model with memory, trained by imitation then reinforcement learning on real home defenses and drills (#2), for everything but workers; marines loaded into bunkers by rule (against an early pool, as soon as the bunker is done) | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | — | — | — | — |
| Scouting | Finding out what the opponent does | One worker at 9 SCVs to the enemy start locations | ○ | ○ | ○ | ○ | ○ | ○ | ○ | — | — | — |
| Reflexes | Quick reactions outside the plan (worker defense, repair, retreat) | Scripted worker pull (v0.3's rule; v0.4–v0.6 left it to the fight models, which did worse); unfinished buildings get a new SCV; SCVs repair a bunker under attack | ○ | ○ | ○ | ○ | ○ | ○ | ○ | — | — | — |
| Attention arbiter | One camera and one APM budget shared by all tasks (§6.7) | One task at a time in a fixed priority order | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| Strategy layer | Picks the build and switches plans from what it sees (§6.8) | You pick the build style by hand (`--style`); an early pool switches Gary to a scripted one-base defense until it's held | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | — | — |
| Opponent model | Beliefs about the opponent's build and army | Not started | — | — | — | — | — | — | — | — | — | — |
| Replay pipeline | Inventory, build orders, resim, desync checks (§7.1–7.2) | ~70k replays indexed; TvZ re-simulated with fog-of-war views | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● |
| Build taxonomy | Build styles discovered from replays (§7.3) | TvZ clusters with readable rules; style labels for ~45% of players | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● |
| Training sets | Data for the learned models | Macro, army and fight sets for TvZ (fight v3: own units from twice as far, unit IDs, every command) | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | — | — |
| Learning beyond imitation | Outcome-weighted learning, reinforcement learning | Fight estimate learned from outcomes; reinforcement learning on pro scenarios runs, no gain yet | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | — | — | — |
| POV viewer | Watch any game from Gary's screen (`viewer/`) | Cursor, clicks, every action in words; Remastered replays; videos | ● | ● | ● | ● | ● | ● | ● | ● | ● | — |
| Trace | Gary's decisions and reasons, recorded (§6.5) | Decision log with model probabilities and fight estimates; POV action log | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | — |
| Evaluation | Matches against opponents, many seeds, a scoreboard (§9) | `eval/vs_rush.py`: several seeds against one scripted Zerg rush; `eval/scenarios.py`: 114 pro home defenses where Gary takes over the Terran | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ○ | ○ | — |
| Matchups and races | | Terran in TvZ only | T | T | T | T | T | T | T | T | T | T |

## Versions

### v0.9 (October 2026): holding early pools (#2)

- **What it is:** v0.8 that notices an early Spawning Pool (seen before 3:00 with one hatchery,
  zerglings before 4:00, or 6 at once before 5:00) and answers it like a player: a bunker just
  inside the top of the main ramp (from the new map analysis), the Barracks rallied behind it,
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
  crashed scouting in 5 of the 114 scenarios (their earlier scores left them out).
- **Most urgently missing:** spotting the pool in time when the scout checks the Zerg's start
  last; terrain in the fight model; the screen (SCVs in front of marines); wall-offs.

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
- **Fights share Gary's hands** (`gary/bots/drill_gary.py`, `train/fight_cmd_rl.py`): drills got
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
