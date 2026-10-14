# Gary — Progress

Where each Gary version stands against [ARCHITECTURE.md](ARCHITECTURE.md). The architecture
says what Gary should become; this file says what exists. Only the components that matter at the
current stage are listed; later ones (narrator, broadcaster, language models...) join when work
on them starts.

Legend: **—** not started · **○** scripted stand-in · **◐** partial · **●** working (for now)

## Roadmap phase

**Now: P2 Micro going into P3/P4** ([ARCHITECTURE.md §12](ARCHITECTURE.md#12-roadmap)), with a
rough P2b (first playable) reached early: Gary already plays whole TvZ games against people,
through the Remastered bridge. The phases didn't go strictly in order: the macro and army are
learned by imitation (P3-style pieces) rather than scripted, because the replay pipeline made that
the quicker way to a playable Gary. Since v0.18 the work is docs/CONTROL.md step 7: every decision
a policy, learned from whole games. What it has found so far: single decisions are worth little
next to the games' chaos, so the learning goes where decisions are few and big (the build style,
learned per map in v0.19; when to go into the early-pool defense, being learned now).

| Phase | Status | What exists | What's missing for its exit criteria |
|---|---|---|---|
| P0 Infra | ◐ | OpenBW env, human interface v1, POV and decision logs, eval against scripted Zergs, replayed pros and pro scenarios; ~150k 45-second scenario plays per hour; full games 14 at a time (one torch thread each: 2.7x faster); games deterministic per seed, replayed to any moment to branch; a run monitor page (`eval/monitor.py`) | The formal trace schema; classic bots to play against; the interface parity test |
| P1 Data | ◐ | ~70k replays indexed, resim with Remastered support (go decided: it works), fog-of-war views | Camera logger, camera inference, fitting human profiles from data (select-to-order times fitted; click scatter and camera still placeholders) |
| P1b Taxonomy | ◐ | TvZ build clusters with readable rules (`taxonomy/discover.py`) | A reviewed `taxonomy/tvz/v1.yaml`, stability and gold-set reports |
| P1c Strategy stats | ◐ (started) | The strategy selector's table: each build style's results per map in Gary's own games (`eval/selector_table.py`, v0.19) | Win-rate tables from pro replays, the feature library, guide claims |
| **P2 Micro** | **◐ (current)** | Fight, command and role models from pro skirmishes; home-defense scenarios and 120 s episodes from pro replays; roles by hand rules, a learned role model (with terrain inputs) on par with them after RL, not beyond; choke blocks as a skill | Responses within the human band: on home-defense scenarios v0.10's hand rules are still the best Gary (pro +1, v0.10 −28); learned role choice and the post's spot found nothing past them |
| P2b First playable | ◐ | Full games against people, offline, via the bridge; learned macro steered by build style, the style chosen per map (v0.19) | Build timings within tolerance (≥18/20 unharassed runs); supply blocks and idle production under harass within the human band |
| P2c Debug viewer | ● | POV viewer with cursor, clicks and every action in words; videos | Trace overlays beyond actions |
| P3 BC full game | ◐ | Separate imitation models for macro, army movement and fights; a belief model of the enemy army; a value network (`gary/policy/value.py`: who wins, the score ahead; it judges at 30 s as 3 minutes of score do) | One policy with intent and belief heads, the Opponent Model |
| P4 RL | ◐ (started) | RL on fights (scenarios, episodes: at the rules' level); macro from branches (nothing: a decision is worth sd 130 against chaos's 512); per game: the strategy selector (v0.19: losses 57 → 30 of 333), the early-defense switch (in progress) | Elo gain at equal humanlikeness |
| P5 Explain, P5b Broadcast | — | (The POV viewer's action overlay is a start) | |
| P6 Deploy | ◐ | Remastered bridge for offline games | Pro blind-test sessions; the bridge's known bugs (a supply depot placed a tile off at expansions) |

## Components by version

| Component | What it is | Now (v0.19) | v0.19 | v0.18 | v0.17 | v0.16 | v0.15 | v0.14 | v0.13 | v0.12 | v0.11 | v0.10 | v0.9 | v0.8 | v0.7 | v0.6 | v0.5 | v0.4 | v0.3 | v0.2 | v0.1 | v0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Game: headless | OpenBW game Gary plays in for development and tests (`env/`, `gary/env.py`) | Melee games on any map, network-style command delay, replays saved | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● |
| Game: live Remastered | Gary in a real StarCraft: Remastered client over LAN (`adapters/scr_bridge`) | Plays; known gaps in the bridge's map checks and hotkeys (its README) | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | — |
| Human interface | The only path to the game: screen clicks, camera, fog, reaction delay, APM (§6.3) | Clicks, drag boxes, camera hotkeys, fog, reaction delay, APM budget; pro-speed hands fitted to pro replays' select-to-order times yet | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● |
| Perception and memory | What Gary knows: visible units, remembered enemy units (§6.9, §6.10) | Fog and hidden enemy HP; remembers enemy buildings and the enemy army seen in the last 3 min; map analysis (`gary/terrain.py`): main ramp, natural, natural choke; walls (`gary/walls.py`): which units get between buildings and cliffs, and where the openings are (not used in play yet) | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ |
| Macro decisions | What to build, train, research, and when | Learned from ~14k pro TvZ games; steerable by build style; rules on top: a Barracks by 1:30, a bunker at the natural when expanding | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ○ | — |
| Macro executor | Carries out macro decisions: placement, production, gas, mining (§5) | Scripted: any Terran building, unit, add-on, research, upgrade, expansion | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| Army decisions | Where armies go, move vs attack, whether a fight is worth taking | Learned destinations and a learned fight estimate; holds attacks it expects to lose; with no enemy army seen for a minute, hunts their buildings and the bases (v0.16) | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ○ | ○ | — |
| Micro | Unit control in fights: stim, spread, siege, focus fire, spells | A role per unit every half second (fire, close in, pull back, retreat) by hand rules; at home the army holds a post by the Command Center and closes in only on zerg attacking or by its buildings; each role's skill choosing where to go from the threats and the ground (docs/CONTROL.md); a learned role model and the command model (imitation, then RL) exist but play worse; marines loaded into bunkers by rule, kept in them (v0.15) | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | — | — | — | — |
| Scouting | Finding out what the opponent does | One worker at 9 SCVs to the enemy start locations | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | — | — | — |
| Reflexes | Quick reactions outside the plan (worker defense, repair, retreat) | Scripted worker pull, 3 workers per attacker (v0.3's rule; v0.4–v0.6 left it to the fight models, which did worse); unfinished buildings get a new SCV; SCVs repair a bunker under attack | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | — | — | — |
| Attention arbiter | One camera and one APM budget shared by all tasks (§6.7) | One task at a time in a fixed priority order | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| Strategy layer | Picks the build and switches plans from what it sees (§6.8) | The build style per map from counted game outcomes (`gary/agent/selector.py`, v0.19); `--style` overrides it; an early pool switches Gary to a scripted one-base defense until it's held; an alarm from nothing seen is called off until 4:00 (v0.18) | ◐ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | — | — |
| Opponent model | Beliefs about the opponent's build and army | Not started | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| Replay pipeline | Inventory, build orders, resim, desync checks (§7.1–7.2) | ~70k replays indexed; TvZ re-simulated with fog-of-war views | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● |
| Build taxonomy | Build styles discovered from replays (§7.3) | TvZ clusters with readable rules; style labels for ~45% of players | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● |
| Training sets | Data for the learned models | Macro, army and fight sets for TvZ (fight v3: own units from twice as far, unit IDs, every command) | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | — | — |
| Learning beyond imitation | Outcome-weighted learning, reinforcement learning | Fight estimate learned from outcomes; reinforcement learning on pro scenarios runs, no gain yet; a value network (who wins, the net value of the next 60 and 180 s) reads who is ahead, not yet who will get ahead | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | — | — | — |
| POV viewer | Watch any game from Gary's screen (`viewer/`) | Cursor, clicks, every action in words; Remastered replays; videos | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | ● | — |
| Trace | Gary's decisions and reasons, recorded (§6.5) | Decision log with model probabilities and fight estimates; POV action log | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | — |
| Evaluation | Matches against opponents, many seeds, a scoreboard (§9) | `eval/vs_rush.py`: several seeds against scripted Zergs, or pro Zergs' armies spawned on their timings; `eval/scenarios.py`: 114 pro home defenses where Gary takes over the Terran, the Zerg reacting once the game leaves the pro's, several draws of Gary's hands | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ○ | ○ | — |
| Matchups and races | | Terran in TvZ only | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T | T |

## Versions

### Next: the belief model (started)

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
