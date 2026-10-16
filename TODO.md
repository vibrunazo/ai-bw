# TODO: from an outside review of the architecture (2026-10-13)

An outside review of ARCHITECTURE.md, CONTROL.md and PROGRESS.md, checked against the code. What
held up, in the order to do it. Pick these up once the early-switch work (`train/early_switch.py`)
is done and its runs have finished; none of them should run while it does (shared code, shared
CPU). Delete this file once each item is done or moved to an issue.

## Direction (2026-10-15)

A second outside review (2026-10-15, after v0.25) of the docs, the last 40 commits and the latest runs.
Its findings, kept here so they aren't lost:

- **The yardstick is wrong, the architecture right.** v0.16-v0.25 followed one loop: 432 games against
  Zerg v2, Zerg v3 and the replayed pros, read the losses, add a rule or option, keep it if the win
  count moves. It has run out of signal:
  - the opponents are self-authored: rules learned against them exploit their habits (hunting on "no
    army seen" was adopted because it "pays against these Zergs");
  - about two thirds of games end undecided, which can't rank options (and the bars were missed by
    noise: v0.22 adopted 3 wins short, v0.25 with 2 more losses);
  - nobody plays the sparring partner: no human game since about v0.11;
  - none of the goals is on the scoreboard: humanlikeness (G2) is measured ad hoc (deaths alone 48%
    against pros' 12%, 1.5 move orders a fighting minute against 8.4), steerability (G3) is a style
    flag, adjustable skill (G4) and explanations (G5) are untouched.
- **The learning program needs a decision.** CONTROL.md step 7 said decisions must be learned
  together, then roles RL, the value network, the early switch and hunt_belief all ended at or below
  the hand rules, and the work went back to tuning rules one failure at a time. Answer taken: rules
  for execution and reflexes, learned models for decisions; the learning failed for want of signal
  (a decision worth sd 130 against chaos's 512, scripted opponents, short horizons), which self-play
  against a pro-derived opponent is meant to supply.
- **Keep:** the layered design, manifests, deterministic seeds and branching, the run monitor, the
  honest "not adopted" records, the macro model from pro replays and the fight estimate, and the
  executor competence (hunts that finish, the camera on a unit, remembered buildings forgotten).

What to do, in order:

1. **Freeze the v3 loop.** v0.25 stays the default; no options adopted on Zerg v2/v3 win counts. When
   something must be ranked, weight the replayed pros (the least self-authored opponent).
2. **A human in the loop.** The Remastered bridge's known gaps fixed (adapters/scr_bridge README; done
   2026-10-15 offline, pinning live), then short sets against a human; that losses list sets the next
   version. This is the product; the rest is proxy.
   Gary's failures in the live games against the computer Zerg so far (runs/live, 2026-10-15):
   - marines and medics trickle up the enemy ramp one by one, and the 8:00 hunt throws the army
     away (games 3 and 5: 14 marines to 5);
   - the computer's early zergling rush lost game 2;
   - turrets: the game refused 13 turret builds at 13 spots in a row in game 5 (7:04-8:01, the
     Engineering Bay done, the money kept), and in game 4; the cause is unknown: log the builder,
     the spot and the game's error text when one fails;
   - minerals float on two bases (game 5: 1,207 at 14:00), and turrets only in the main (game 1).
3. **A self-play skeleton** (the user's priority): every piece of ARCHITECTURE.md present and
   connected, crude where it has to be, training together:
   - a Zerg Gary from the same pro-replay pipeline (Zerg macro and army imitation from the 14k TvZ
     games), on Zerg v3's execution code;
   - the opponent model's belief over the enemy's z, and a strategy layer conditioned on it;
   - a trace schema;
   - a league loop that updates at least one learned piece from outcomes, with the critic grading
     unfinished games and the pro data as an anchor (KL to the imitation policy).
   Risks: two crude imitation agents may mostly produce passive, undecided games; keep anchors outside
   the loop (pro distributions, human games). Compute is shared: segments from mid-game states (T2a)
   and nightly batches first; full-game league training is rented compute.
4. **Humanlikeness on the scoreboard.** The distances to pro distributions computed by hand in
   `eval/losses.py` and `eval/army_orders.py` become standing metrics in every run and on the
   monitor: deaths alone, orders per fighting minute, where the army stands each minute, build
   timings against pros. Use them alongside results, not instead of them (they can be gamed too).
   Folds into 3 (the KL and discriminator terms).
5. **Explainability.** A narrator over the existing records (decision log, trace, the analyses written
   by hand): cheap once the trace schema exists, one of the goals, and no other bot has it.

Items 3, 5 and 7 below wait for the skeleton's opponent; 6 (the engine edge error) stands.

## 3. The army model's fight estimate, from Gary's own fights

The fight estimate is trained on what the pro got: the change in army share in the 45 s after the
pro's order (`train/army.py`, lines 96-97). Gary holds or attacks by what a pro would get from that
fight, with hands that are weaker in fights (1.5 move orders a fighting minute, pros 8.4; v0.10's
vultures 1.3 tiles/s, pros 2.9, losing twice as many). It fits the second kind of v0.19 loss: the
6-7 minute fight against Zerg v2, deaths outnumbered 2:1 or worse 53% (pros 29%).

- Sample army-order moments from the pro data; let Gary take over from each (as `eval/scenarios.py`
  does) with the pro's order, several draws of its hands; label with Gary's own 45 s change.
- Train two heads on the same state, the pro's outcome and Gary's: their gap maps where Gary falls
  short of pros (compositions, terrain, sizes), a diagnostic on its own.
- Play with Gary's head; judge on Zerg v2 (losses, deaths outnumbered, deaths alone) and the
  replayed pros.
- Caveat: once a game leaves the pro's, the Zerg reacts by script, so the labels are "against a
  scripted reaction". 45 s is still noisy: several draws per state.

Not this: crediting mid-game decisions by full-game branches (CONTROL.md step 7 already found a
decision worth sd 130 against chaos's 512).

## 4. Zerg v3: a greedy Zerg that punishes passive play

The selector table (v0.19) counts only early-aggression opponents, and the replayed Zerg's own
docstring says never to judge Gary's economic choices with it, yet the build style is one. So
"style 4 on Fighting Spirit and Heartbreak Ridge" means "best against early aggression". Also check
whether styles 2 and 3 lose for the opponents or for Gary's vulture micro (`eval/losses.py` on their
games): the macro picks a pro's Factory build that Gary's hands can't play.

A Python bot (`gary/bots/zerg_v3.py`), not a BWAPI one yet: `env/gary_env.cpp` has no BWAPI, so
Steamhammer and the like need BWAPI built on OpenBW and a bot hosted in our loop (real C++ work),
and a foreign bot's own randomness threatens what branching and the paired tests rest on: games
deterministic per seed. A Python bot keeps that, plays through the human interface like Zerg v2
(human hands, unlike Steamhammer's micro), and runs in `eval/vs_rush.py` as is. Classic bots stay
on P0's list for later.

Keep it minimal: it has to punish passivity, not be strong (a scripted Zerg through the interface
lags pros by minutes; mutas and lurkers would be played badly). On Zerg v2's code:

- greedy: a third base early, drones to 40+, no 6:00 all-in;
- hydralisks (attack-move, no micro), one big attack at about 9:00 sized by its economy;
- zerglings back home when Gary's army moves out;
- some seeds open hatch first with a pool at ~2:00 and send a few early zerglings to poke without
  committing: the moment a false alarm is tempting (item 1).

Done: `gary/bots/zerg_v3.py` (PROGRESS.md: it punishes the always-defend switch, outcome −0.18 ± 0.06).
Then: add it to the selector table's opponent pool and rebuild the table; re-learn and re-test the
early-defense switch (`train/early_switch.py`; without it: always defends early, losses 40 -> 39 of
432, false alarms 285 of 287 quiet games).

## 5. Scattered fighters back into the main group

Two different problems; don't solve them with one rule.

- **Housekeeping, no fight on:** a fighter standing idle far from Gary's main group, no zerg near,
  walks (attack-moving) back to it, as `home_scvs` sends stray SCVs back to mining. A chore, not a
  decision. Small today: of fighters alone more than 15 tiles from the main group (0.65 a look
  every 10 s, 16 games against Zerg v2/v3), most were walking to the rally (shorter since
  `rally_ramp`) or chasing an attack or close-in order; idle and lost only ~4%. Measure first
  (the stray tracker), then a rule, judged over the whole pool.
- **In a fight:** whether a new unit joins now or waits for the group. Pros lose 12% of their army
  alone, Gary ~48% in its losses; rules to hold new units back (`reinforce_pack`, `regroup`,
  `safe_rally`), a learned join/hold override and RL on the role choice all measured worse or
  equal. A real decision with exceptions (one more marine saves a bunker; one more dies alone for
  nothing): for the learned layer, once the critic and longer episodes value a fighter kept alive
  (the 120 s episodes reward fighting now).

## 6. An engine error at the map's edge

OpenBW stops a game ("attempt to get tile index for invalid position -7 263") when a unit's path
leaves the map: seen once in 40 games, an SCV standing in a main's corner behind its edge minerals
sent to mine one of them (v0.21 against Zerg v3, Fighting Spirit seed 7). Such games count as errors
(eval/vs_rush.py), not results. Find what puts SCVs in that corner (a depot spot by the edge?) and
keep them out of it; the engine itself is external code.

## 7. The macro reads the scout, then the scout is worth testing

Gary builds the same tech whatever Zerg v3 plays (Engineering Bay ~6:00, turrets from ~7:00 against
mutalisks, lurkers and the zergling pressure alike; 40 games, PROGRESS.md 2026-10-14), and nothing
but the early-pool rules reads what the scout sees. The touring scout (`watch`) stays in the
Zerg's base no longer than the plain one (65 s against 64 s median, 858 games: it leaves when 3
zerglings come near, at ~3:00) and lost more without a reader (30 against 25 of 432, before v3's
three plans).

- First: answers to each plan as pros play them, triggered by what Gary has seen (the scout, later
  sightings, scans): turrets and an Engineering Bay against mutalisks, Comsat, Science Vessels and
  tanks against lurkers, bunkers and marines against the zergling pressure. Judge per Zerg v3 plan
  (seed % 3) and over the whole pool.
  Done for lurkers: v0.24 (`gary/agent/answers.py`), lurker games won 11 of 32 (7). Mutalisks:
  the model's turrets already do, an answer on top lost games. The tech scan seldom happens (the
  model's Comsat often after 10:00); 21 lurker games still end undecided.
  The touring scout on v0.24 against Zerg v3: plans read ~1.5 min earlier, won 53 (51) but lost 4 (1),
  the losses against the zergling pressure: not adopted (PROGRESS.md).
- Then: v0.22 with and without `watch` against Zerg v3, results per plan, to see whether a longer
  look pays once something reads it (and whether the scout should dodge zerglings to stay longer).
