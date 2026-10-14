# TODO: from an outside review of the architecture (2026-10-13)

An outside review of ARCHITECTURE.md, CONTROL.md and PROGRESS.md, checked against the code. What
held up, in the order to do it. Pick these up once the early-switch work (`train/early_switch.py`)
is done and its runs have finished; none of them should run while it does (shared code, shared
CPU). Delete this file once each item is done or moved to an issue.

## 2. Do Gary's hands cost it games? A cheap check first

No deciding model sees the state of the hands: `gary/policy/roles.py`, `army.py` and `macro.py` take
no APM, camera, selection or pending-action inputs, and ARCHITECTURE.md §3.1's masks were never
built. Whether that matters is unmeasured: `eval/losses.py` didn't keep arbiter traces.

Replay v0.19's 30 losses (`runs/losses.jsonl`; deterministic, 335 of 336 replayed the same) and ~30
wins with traces on, and compare the arbiter's `wait` events (≥1 s) during fights. Only if losses
stand out: give the deciding layers inputs about the hands (time since a group's last order went
through, the arbiter's queue). This decides only about those inputs; the deciders still assume
pro-level execution either way (item 3).

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
