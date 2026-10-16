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
   41). Then (Argon's catches) hatcheries rally beside their own minerals, not the army's gathering
   point, and idle drones mine at any base Gary-Z holds (a rallied drone doesn't mine by itself in BW,
   the user's correction, checked in OpenBW: the idle routine sends it; pros' hatchery rallies in 46
   TvZ games: 78% toward the front, drones then picked up with the army's, 21% by the minerals; the
   front rally with a ctrl+click pickup is the pro way, for later. The user's practice: rally beside
   the hatchery while droning (the early game, and a new expansion until it's saturated), else to the
   army's meeting point, in front of the natural or mid map with map control); gas drones follow the bank (back to minerals
   above 400 gas). Over 8 seeds against v0.27 (eval/zerg_economy.py), medians at 4/6/8/10 min:
   drones 15/20/32/25 (Zerg v3: 12/20/29/28), hatcheries 2/3/3/2 (2/2/3/4), army supply 2/3/6/10
   (2/5/14/25), gas 392/142/274/194 (v3 piles 1,694 by 8:00). The economy is v3's; the army is half
   of it or less, and Gary-T wears it down by 12:00.
   The army, against the pros (300 of their games): they spend 70% of larvae on drones until 5:00,
   57-71% on army from 6:00 (drones 23/27/30, army supply 4/20/27 at 6/8/10). Gary-Z kept droning
   (71-82% at 6:00-7:59): it had seen no Terran army by 8:00 in 8 games of 8 (pros: 4 units by 5:00,
   4% none), so overlords now stand on watch posts outside Gary's natural and halfway (seen from
   4:00-5:00). Then every larva shortage became a creep colony (the first decision possible taken):
   it now waits for a larva, as the Terran's macro waits for a busy building, and no two buildings
   share a spot. 8 seeds: drones 16/24/20, army 4/10/12/16 at 6/8/10/12, 3-4 hatcheries. Half the
   pros' army at 8:00-10:00 still; Gary-T wears it down by 12:00. While larvae are short, other
   decisions the model gives 5% or more still go (Argon: waiting on every larva held a macro hatchery
   back). Against the pros at 6:00/8:00, Gary-Z's hatcheries (3/3, pros 3/3), free larvae (2/2, 3/4)
   and bank (244/202, 248/262) match: the macro holds to 8:00; the 10:00 float is a losing game (2
   hatcheries left). The army gap is the fights (v3's scripted army): next the acceptance test and the
   Zerg army model.
   **The Zerg army model** (runs/army/TvZ_Z_20261016-114756; held out: destination within one cell 62%,
   staying put 58%; fight estimate correlation 0.46, the Terran's 0.40) now chooses where Gary-Z's
   groups go, every 5 s, an attack it expects to lose held; v1's stage-judge-go and v2's scripted
   all-in are gone, v1's home defense, v3's raid and lurkers stay; groups take every ground fighter
   (v3's took zerglings only: hydralisks stood where they hatched). Hatcheries rally to the army's
   meeting point from 5:00 (the user's way; groups form there). 8 seeds, 12 min: noisy (army supply
   8/6/5 at 8/10/12, drones 27 at 12:00); judged by the acceptance test.
   **Acceptance (2026-10-16, eval/garyz_accept.py: 3 maps x 16 seeds, 20 minutes, against v0.27):**
   decided 65% (Gary-T won 31, lost 0, 17 undecided; Zerg v3's pool: about a third, Gary losing 3%);
   9 plans rolled at about the pros' shares (3-hatch spire first 18 of 48, 38%; pros 40%), the plan
   followed (lurker-first games research Lurker Aspect before the Spire); timings: Spire at 5:05-6:48
   (pros 5:09 in that plan), Lurker Aspect 5:14-6:33 (pros 4:40-5:35), the third Hatchery about two
   minutes late (pros 2:52); the same seed twice gives the same game. So it runs as an opponent and
   ends games, but it never wins: weaker than Zerg v3 (its fights are v3's scripted micro). Next: Zerg
   fight roles (and the kiting drill), the lever for its strength, and the humanlikeness classifier.
   **Argon's review (2026-10-16):** three execution bugs, fixed: a new group was a ctrl+click (one type:
   hydralisks never joined; now a drag box over every fighter); each group asked the army model alone
   (12 zerglings at a time into Gary's army; now one destination for the whole army); drones at the
   front meeting point were boxed with the army (now a ctrl+click on a drone away from the hatcheries).
   Those alone didn't help (8 seeds, army supply 9/4/0 at 8/10/12): the logs showed 93-103 Creep
   Colony decisions a minute across the 8 games. The decisions were rolled every half second, so one
   at a few percent came as often as the drones. Now the next roll waits the seconds the model
   predicts until the pros' next decision (its timing head, at most 10 s), and the cap counts pending
   jobs and creep, sunken and spore colonies together: 98 Creep Colony decisions in 8 games (from 411),
   drones 18/28 at 6/8 min (pros 23/27), army supply 4/10/5/6 at 6/8/10/12 (pros 4/20/27 at 6/8/10),
   Gary-T won 0 of 8 (2 before). The army model still gives 1-5 orders a game: groups seldom form
   before Gary-T's push (8:30-10:00), and the army dies defending at home. Fights are the lever.
   **Execution leaks, found with probes (2026-10-16):** groups form by zergling-equivalents (hydralisks
   counted as 1 each, a group waited for 12 units; Argon's catch); a group is a ctrl+click on its main
   type and shift+clicks on the rest (a drag box at the meeting point took the drones rallied there,
   and v1 refused it: groups took a minute to form, home defenses never formed); fighters left behind
   (hatched before the rallies moved, 4 of 12 at 8:00) walk to the meeting point; a parked decision's
   reserve no longer goes below zero (200 gas parked of 158 made every zergling unaffordable); larvae
   are spent in bursts, buildings at the timing head's pace (its gap is between pros' bursts); the
   army, the macro and the helpers share the hands by streaks (the army every other turn moved the
   camera between a look and its selection: a drone-to-gas task held every other turn for 2:30, and
   a Hydralisk goal timed out twice with 9 larvae and 3,000 minerals); v2's tasks time out. 8 seeds:
   drones 20/32/25/31 at 6/8/10/12 (pros 23/27/30 at 6/8/10), hatcheries 3/3/4 (pros 3/3/4), army
   supply 4/10/7/7 (pros 4/20/27), Gary-T won 0 of 8. Still open: the bank (560 at 8:00, 860 at 10:00;
   pros 250-360) with gas nearly gone (44 at 10:00), and the fights.
   **The larva cap and the gas (2026-10-16):** eval/zerg_economy.py now measures the share of
   hatchery-time at the 3-larva cap, next to 98 pro Zergs (TvZ games of 12:30+, cached): Gary-Z 0-1%
   from 6:00, pros 7-13%; its larvae are used as they come, so the floating bank wasn't slow hands
   (and not a missing macro hatchery, a learned decision). It was gas: v3 forgets a gas drone only
   when unseen, and one sent back to minerals while the gas floated stayed counted, its extractor
   full all game (0-3 drones on gas from 6:00). Fixed, and groups form by supply (the user's catch:
   the zergling-equivalent weights were hand-picked). 8 seeds, Gary-Z (pros): army supply 6/12/16/15
   at 6/8/10/12 (4/18/24/34), bank 68/234/486/494 (262/310/360/442), gas 156/82/128/213
   (400/422/500/510), drones 17/24/23/26 (21/25/28/32), Gary-T won 0 of 8.
   **The humanlikeness classifier, first version (2026-10-16, eval/humanlike.py):** one player's side,
   features per minute from the replays re-simulated (cut at each game's real end: OpenBW plays on
   past a replay's last command, and idle minutes looked like 8,000 minerals banked), against 300 pro
   games of 10 minutes or more on the three maps that re-simulate cleanly (a desynced pro replay had
   18 SCVs and no army at 10:00). Gary-T (112 games) and Gary-Z (16) are told apart at AUC 0.998-0.999
   in every minute, and still at 0.99 with the top 12 tells dropped: they differ in many ways at once,
   so the combined score saturates until Gary is closer, and the per-feature gap table (each feature
   alone, Gary against the pros per minute range) is the work list. Gary-T's biggest gaps: bunkers
   (2 by 5-7 min, pros 0), refineries (1, pros 2 from 8 min), the army standing (78% at 5-7 min, pros
   37%) with half the pros' orders per fighter, minerals banked (1,010 at 11-14 min, pros 423), turrets
   (11, pros 7), no vessels or tanks late. Gary-Z's: the army standing (87-95% to 7 min, pros 5-30%),
   all at home (pros: half in the field from 5 min), clumped (spread 2-5 tiles, pros 8-25), drones on
   gas (0-6%, pros 15%), a third of the pros' orders per fighter. The opponent leaks in (Gary is rarely
   attacked: Gary-Z seldom leaves home), as expected until both sides are Gary.
   **Fable's review: the army layer before fight roles (2026-10-16).** An army standing at home isn't
   engaged, so fight roles can't move that tell. Gary-Z's army layer, probed: eval/vs_rush.py sets the
   scripted rushes' earliest attack (7:00), and v1 formed no group before it, so until 7:00 the army
   stood ungrouped and the army model was never asked, in every Gary-Z game so far (now ignored:
   the model decides); and a group was marked sent when its hotkey was recalled, a turn before its
   order, so the order never went (12 asks for the field in a minute, none carried out). Now groups
   form from 5:00, and the army goes to the field and stays. Gary-T's army layer, from 112 games' logs:
   at 5-6 min it holds about 90% of its decisions (985 holds, 140 orders), and in 92% of the holds the
   fight estimate for going is within 2 points of staying, noise for an estimate correlating 0.40;
   the hold vetoes plain moves too (positioning, not attacks). Shared design, shared tell; a change
   to Gary-T is a new version (its loop is frozen). The classifier now labels Gary's games by opponent
   (league, replayed pros, human: the only real opponent) and reports a gap score that moves while the
   AUC saturates: the mean standardized gap per minute range (Gary-T in the league 0.38/0.39/0.43/0.49
   at 2-4/5-7/8-10/11-14 min; in the two human games 0.45/0.42/0.65/0.87) and the share of features
   inside the pros' interquartile range.
   **What felt AFK (the user, 2026-10-16):** not the army at home, and not damage: anticipation before
   contact and the right answer to the situation. A pro moves 6 marines into the mineral line as a dozen
   zerglings come and they back off, no blow traded; stims and gathers marines at the sight of mutalisks
   strafing at a distance; a lone marine near 4 zerglings runs to the buildings. Gary attacked the lings
   and died: a reaction-time script calls that fast. The classifier now looks every 8 frames in fights
   and at sightings, from the player's own vision. Gary-T against 300 pro Terrans (means, 5-7 / 8-10 /
   11-14 min): lone and outnumbered, moved away 0% / 1% / 0% (pros 18% / 18% / 11%), attacked 86% / 67% /
   67% (52% / 42% / 39%); marines stimmed after mutalisks come into view - / 14% / 14% (33% / 40% / 36%);
   moving while reloading 0% throughout (pros 4-10%); seconds to answer harassment 6.7 / 6.1 / 6.2 (pros
   11.8 / 9.1 / 11.5: faster, as the user said a script would find). Gathering at a sighting doesn't
   separate (pros spread as they move). Gap score in the league 0.44-0.48, in the human games 0.66-0.84.
   (Games against the replayed-pro fixture can't be measured: it creates its units in the engine with
   no commands, and its replays re-simulate with an idle Zerg. The classifier refuses them; the 1.0-1.2
   gaps first reported for them were Gary against an empty opponent.)
   Gary-Z baseline (48 games at e1a3e8e): Gary-T won 20, lost 1 (Gary-Z's first win), 27 undecided;
   army bunched (2-5 tiles, pros 5-29), standing 73-89% (5-65%), at home 84-97% (30-53%).
   **The response model (Terran, 2026-10-16):** the role model (fire, close in, pull back, retreat,
   reposition, stand) learned from 600 pro games' units whenever a seen enemy fighter was within 12
   tiles (ingest/response_dataset.py: 2.6M labels, 1.3M before contact), with a where head (each unit's
   move over the next half second: 0.6-0.75 tiles off, 0.94-1.25 if it guessed no move). Before
   contact it names the pros' new roles at 0.5-0.7 recall (retreat, rare, 0.2). In the 226 pro
   home-defense scenarios not from its games, its roles in place of the hand rules lose 52 net value
   (with or without its where); the 12-tile watch alone +4 +- 5; the rules in contact and the model
   before it (v0.29, roles.py ahead/v1) -2 +- 6: a tie. The rules were tuned on those scenarios over
   twenty versions; imitation on day one was never going to beat them there (Argon). The neural role
   policy is the league's learner for the Terran's fights (RL with the KL leash sharpens it); which
   version plays humans is a separate choice. Next: the same for Gary-Z (its fights are v3's script).
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
