# Self-play: the plan (draft for review)

Direction 3 in TODO.md: every piece of ARCHITECTURE.md present and connected, crude where it has to
be, training together. The first milestone is a learned Zerg Gary that is a real opponent for the
Terran Gary in OpenBW, from the same pro-replay pipeline. It's the opponent the rest needs.
(Revised 2026-10-16 after outside reviews: the humanlikeness classifier replaced subjective ratings.)

## What exists

| Piece | State |
|---|---|
| Zerg macro model | trained 2026-10-04 on the Zerg side of the 14k TvZ games (runs/macro/TvZ_Z_*, 50% top-1), never used; predates later dataset changes |
| Zerg army model | `train/army.py --race Z` runs on the existing dataset (both players' orders are in it); not trained yet |
| Zerg execution | Zerg v3 (gary/bots/zerg_v3.py) through the same human interface: building jobs, larva production, tech, lurker morphs, groups, a mutalisk flock that raids the mineral line with the least anti-air |
| Zerg build styles | taxonomy/discover.py clusters pro builds per race (the Terran macro is steered by its own) |
| Two players in one game | HumanInterface steps several players on one OpenBW game (one advances it) |
| Opponent model | the belief model (gary/policy/belief.py) over what the enemy has; not yet over its plan (z) |
| Critic | the value network: reads who is ahead, not yet who will get ahead |
| Parity | the bridge matches OpenBW (adapters/scr_bridge/tools/parity.py), so headless lessons hold live |

## Gary-Z v0

1. **The vocabulary check (first, 30 minutes):** the Zerg macro model's most frequent decisions, each
   with an execution path in v3 (creep colonies and sunkens at chosen spots, scourge, queens,
   defilers, overlord speed, macro hatcheries, the third and fourth bases). Retrain the macro model
   with the current script before relying on it.
   **Done 2026-10-16** (the pro Zergs' decisions in 400 TvZ games of the macro dataset, per game):
   - covered by v3: drones 42, zerglings 32, hydralisks 20, overlords 11, mutalisks 8.8, lurkers
     5.8 (aspect, morph), creep colonies 5.7 and sunkens 3.8, expansions 3.9 and macro hatcheries
     1.3, extractors 3.5, the Pool, Lair 1.0, Spire 0.9, Hydralisk Den 0.9, zergling speed;
   - missing: ultralisks 3.6, scourge 3.5, defilers 2.0; Evolution Chamber 1.2 and its upgrades,
     Queen's Nest 0.6, Hive 0.6, Defiler Mound 0.5, Ultralisk Cavern 0.3, spores 0.3; consume,
     plague, burrow, hydralisk and overlord upgrades;
   - the commands are generic already (gary/commands.py morph, morph_building, build, research,
     upgrade for any type): the gap is a dispatcher from the model's (act, id) to v3's primitives
     (larva, building selection, jobs) and a Zerg facts module like gary/terran.py (costs, supply,
     requirements, producers, sizes, build times).
   - Zerg build styles exist (taxonomy TvZ_Z_tech, window 8:00) but are weak: one cluster of 24%,
     four of about 2%, two thirds of the games in none. Steering Gary-Z needs a better Zerg
     taxonomy first (another window or cluster size), its cards reviewed. The Zerg macro model
     (2026-10-04): top-1 50% (baseline 32%), timing error 2.5 s median.
   **Then (2026-10-16):** the Zerg macro model retrained on the current pipeline (top-1 50.3%, as
   before); Zerg plans (taxonomy/plans.py) label every pro game of 9:00 or more by opening (pool or
   hatchery first), hatcheries by 5:00 (2 or 3) and first tech (spire, lurker, hydralisks): 10 plans,
   the standard 3-hatch spire-first 44%, hatchery-first 2-hatch spire 12%, pool-first 3-hatch spire 11%,
   3-hatch lurker 9%, ... Then every game, short ones too, from what it showed by its end (Argon:
   "no style" meant "a short game", and early-pool rushes were in no plan): 11 plans with zergling
   pressure (10+ zerglings by 4:00, no tech, 1.3%); undecided short games and rare ones are "other"
   (5.6%); model TvZ_Z_20261016-101426, top-1 50.6%; the macro index restyled with them for the Zerg only (ingest/macro_dataset.py
   --restyle) and the model retrained steered by them (runs/macro/TvZ_Z_20261016-095424: top-1
   50.5%); gary/zerg.py, the Zerg facts, checked against units.dat (tests/test_game_data.py).
   **Gary-Z v0 runs (2026-10-16, gary/bots/gary_z.py, `--opponent garyz`):** v3's hands, the steered
   Zerg macro model's decisions (dice-rolled, as the Terran's), any Zerg unit, building, morph,
   research or upgrade. Fixed on the way: a larva just morphed was ordered again (29 of 60 morphs
   went to eggs), money just spent counted twice, v3's overlord list never cleared (supply-blocked).
   Then: a decision short of money is parked, its money kept, while larvae go on with the rest
   (Argon: a Lair waiting for gas held every larva); one pair of hands shared between the macro and
   v3's helpers (jobs, gas, idle drones, rallies) and the army: whoever acted keeps them for up to 3
   actions (a selection and its order), then the other goes first (each could take every turn: three
   jobs at once from 1:00 to 2:00; idle drones it couldn't send back, 109 turns in 40 s). Seed 101:
   13 drones at 4:00, 21 at 6:00, 22 and 3 hatcheries, a Lair and a Spire by 8:00 (Zerg v3 13, 24,
   41). Then (Argon's catches) hatcheries rally onto their own minerals, not the army's gathering
   point, and idle drones mine at any base Gary-Z holds; gas drones follow the bank (back to minerals
   above 400 gas). Over 8 seeds against v0.27 (eval/zerg_economy.py), medians at 4/6/8/10 min:
   drones 15/20/32/25 (Zerg v3: 12/20/29/28), hatcheries 2/3/3/2 (2/2/3/4), army supply 2/3/6/10
   (2/5/14/25), gas 392/142/274/194 (v3 piles 1,694 by 8:00). The economy is v3's; the army is half
   of it or less, and Gary-T wears it down by 12:00.
2. **Macro, steered by a build style:** at the start Gary-Z rolls a pro Zerg style (z) and its macro
   is conditioned on it, as the Terran's is: unsteered, a model blends 9-pool, 2-hatch mutalisk and
   3-hatch lurker into a soup, and Zerg tech is gas-gated. Each decision maps onto v3's execution (a
   larva, a job, a morph). Larvae aren't kept busy by habit (the Terran's keep_busy would spend
   them all on the last unit); an overlord before a supply block is an execution guard. Buildings
   placed by a general creep-spot search, not v3's fixed offsets. The z it rolled is a free, true
   label for Gary-T's opponent model.
3. **Army:** the Zerg army model (trained after the check) chooses where groups go; v3's group
   orders carry it out. The mutalisk raid stays a motor skill (the mineral line with the least
   anti-air, the retreat), started when the army model sends the flock at a base, not replaced by
   grid attack-moves.
4. **Fights:** v3's scripted micro.
5. **Accepted as an opponent, not as a bot that runs:** against v0.27 on three maps, 48 seeds:
   - no crash or stall, its economy in the pros' range (drones at 6:00 and 10:00, hatcheries, tech);
   - more games decided than the v3 pool's third;
   - its plans spread over time like the pros' do;
   - the same seed replays identically with two learned players in the game (branching and paired
     tests, the project's main instrument, depend on it).

## Before any RL

- **The humanlikeness classifier,** a prerequisite of the league (ARCHITECTURE's P3 exit
  criterion). Trained to tell a Gary game from a pro game, per minute, from one player's side only,
  against pros of the same race (Terran now: the Terran side of the 14k TvZ games; the same script on
  Gary-Z against the pro Zergs, part of its acceptance). Never the opponent's side, or it learns to
  spot Zerg v3.
  - Features at the decision level, not clicks (raw commands give it away on interface fingerprints):
    per minute supply, workers, army composition, where the army stands relative to the bases, units
    standing alone, orders per fighting minute, build timings, idle SCVs, and the time from an attack
    starting (the first enemy fighter seen near the player's base or units) to the first response.
    Several exist in eval/losses.py and eval/army_orders.py. Minutes compared only while the player
    is still in the game.
  - Gradient-boosted trees as train/stim.py, held out by game. The score is the AUC per game minute;
    the top features are the next work items (ARCHITECTURE section 7.6's feature-discovery loop).
  - Guardrails: every version is also scored by a frozen classifier (the first one's AUC: did the
    old tells go?) and shown as the raw pro-distance table (means and medians against the pros'),
    since a retrained classifier finds the next tell and hides progress. Pro games on the same maps
    as Gary's. The opponent leaks into a player's own state (against Zerg v2, Terran always fights an
    all-in at 6:00): once Gary-Z exists, Gary-T vs Gary-Z games are the ones scored.
  - A metric on every version's headless games and on the human games, on the monitor; not a reward
    in the league until it has been stable across several versions.
- **Anticipation as target inference:** which asset each enemy group reaches in the next 15-20 s
  (the main's minerals, the natural's, the army, a building, none), learned from the pro games.
  Judged first as a classifier on held-out pro games, then as the trigger of a learned response
  (marines back to the threatened mineral line; no builder walked into the lings that killed the
  last). The user's examples: zerglings turning toward the mineral line, mutalisks routing natural
  then main. (Human game 2: in the 3 s after a lone unit was hit, Gary's hands went to the fight
  more than usual, macro less; the bunker-load loop took 18%, fixed in v0.27. The arbiter isn't what
  made it feel AFK; the response is.)
- **Zerg fight roles,** the pros' zerglings and mutalisks labeled as the marines are (fire, close
  in, pull back, retreat). Kiting is a hypothesis, not a promise: Terran roles hit the rules'
  ceiling three times. Its drill: 4 zerglings against 2 SCVs and a marine, SCV kills counted, the
  pros' replays as the yardstick.

## The loop

- **League:** Gary-T and Gary-Z in parallel headless games (low priority, nightly batches here;
  full-game league training is rented compute), each game's outcome and classifier scores kept.
- **Every RL update keeps a KL term to the imitation policy it started from** (ARCHITECTURE's
  λ_KL, as AlphaStar): the leash on the learned parts while they train. It can't see the scripted
  parts, nor drift that compounds over a game; the classifier measures those after the fact.
- **The first RL target is where credit assignment already works:** fight episodes paired on the
  same positions, with Gary-Z the reacting side instead of the scripted reaction; or the strategy
  selector's per-game table over Gary-Z's plans. Segments from mid-game states (60-120 s branches
  from raids and pressure) before whole games. Army decisions learned from whole-game outcomes
  last, and only once the critic predicts more than who is ahead now: graded by today's critic,
  timeouts would teach hoarding the army at home. A segment's own reward (workers killed, bases
  kept, army traded) is a candidate to check against paired outcomes, not a given.
- **Anchors outside the loop:** the humanlikeness classifier and human games. A league of two crude imitators can
  settle into passive games; the anchors catch it.
- **Human games** stay a source of failure lists and replays (objective, scored by the classifier
  too). Whether Gary is worth playing against is P6's blind tests with target players, not ratings
  from us.
- **The trace schema** when its first reader (the narrator) needs it, not before.

## Order of work

1. The vocabulary check; the Zerg macro model retrained; Zerg style steering; the creep-spot search.
2. The Zerg army model; Gary-Z v0 (macro and army learned, v3's execution, raid skill, micro).
3. Acceptance: 48 seeds on three maps against v0.27 (decided share, plan spread, determinism).
4. The humanlikeness classifier (Terran first; Gary-Z's acceptance with it).
5. Target inference from the pro games, then its learned response, on both sides.
6. Zerg fight roles and the kiting drill.
7. League harness; the first RL update on paired fight episodes or the selector, judged by the
   anchors and human games.

## Risks

- Two crude imitators may mostly produce passive games: the acceptance test's decided share and
  the anchors (the classifier, human games).
- Compute is shared: segments and nightly batches first.
- v3's execution was written for its own plans: the vocabulary check finds the gaps before game 40.
