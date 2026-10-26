# AGENTS.md

How to work in this repository, for coding agents and people alike. The design is in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); what each version did and how it measured is in
[docs/PROGRESS.md](docs/PROGRESS.md) (keep it updated after each milestone); open work is in the GitHub issues, indexed in [TODO.md](TODO.md) (a done item's issue is closed
and its line removed; a new one gets an issue and a line).

## Where things are

- `gary/`: the agent (`gary/agent`: arbiter, executor and its race plugs, macro, army, fight, rules), the
  race data (`gary/terran.py`, `zerg.py`, `protoss.py`, `race.py`), the learned models (`gary/policy`), the
  interface (human-like hands) and the game environment.
- `eval/`: the launcher (`match.py`), league rounds (`league.py`, which keeps each round's feature rows), measures
  against the pros (`humanlike.py`, `attacks.py`, `style_adherence.py`: the build per style), two versions compared game against game per time window with a
  bootstrap spread (`gap_windows.py`), the records' queries (`why.py`), the run monitor (`monitor.py`,
  `priority.py`).
- `train/`, `ingest/`, `resim/`: models, datasets from replays, the replay re-simulator.
- `adapters/scr_bridge`: playing StarCraft: Remastered. `tests/`: smoke, parity, matchups, game data.
- `docs/`: ARCHITECTURE (design), CONTROL (layers and manifests), RACES (the per-race audit), REFACTOR (one
  Gary for every race, done), SELF_PLAY (the self-play plan), DATA_ERAS (old and modern replays).

## The Garys and how to judge a change

- One Gary for every race: manifest `gt1` plays Terran, `gz1` Zerg (`gary/agent/manifests.py`). Every game
  runs through `python -m eval.match --a <manifest> --b <manifest>`; `python -m eval.league` plays Terran
  (`--vs`, default gt1) against Zerg (`--zerg`, default gz1); `gary/play.py --live` plays a human. A side's build
  style is told by name or id (`--style`, and `--style-b` for B in eval.match; `--list-styles` prints each race's per
  matchup from the catalogue `gary/styles.py`, every game labelled by `taxonomy/label_all.py`).
- Two Terran manifests: `gt1` is what plays (the strongest measured: human games, the tests' parity, the league's
  default) and `gt1dev` is what learning is trained and judged in: gt1 with the role model deciding in contact and
  the army on control groups, the configuration expected to win later, and without the stop-gaps over the layers
  being learned (option `unscripted`: the static guard, the hunt, the overlord hunt, the pros' caps, the SCV cap, the
  bank escape), and with the models' inputs in play counted as their training data counts them (`seenall`,
  `armyseen`; gt1 doesn't have them yet), so the models own those decisions there; a stop-gap stays in gt1 while it
  measurably helps, and its
  job in gt1dev is the learned layer's or nobody's. `gz1dev` is the same for the Zerg (gz1 with `unscripted`: join,
  sweep, press, the hunt, the caps, the bank escape and overlords ahead off; its overlord posts and mutalisk raid
  kept, as no model moves those units; decoders such as the tech floor are not stop-gaps and stay). gt1dev against
  gz1dev is the TvZ self-play pairing (balanced: each side wins or leads in 38% of 96 games; gt0.18dev against gz0.6dev
  35% and 42% of 48, docs/PROGRESS.md). A learned layer's work (data, training, the
  pipe check, RL) is done and judged in gt1dev; gt1dev replaces gt1 when it beats it in 96 paired games, with a
  version bump. The suite's numbers for both, and how far apart they are, are in docs/PROGRESS.md.
- A change is an option on them (`gt1+<option>`), judged in paired 48-game rounds on the ladder maps
  (`--maps ladder --seeds 6 --first 4004`, 30 minutes) and the gap score to modern pros; 24 games are too
  noisy for close calls, and a single round's gap score needs its spread (`eval/gap_windows.py`, 96 paired games
  for a close call). An adopted option is folded into the manifest with a version bump, so the name
  never grows into a chain of options. A close 48-game round gets its second 48 (`--first 4010`) at the first
  round's priority, without asking. A baseline is replayed whenever its code or the opponent's manifest changed (a
  baseline from a worktree against an older gz1 once misjudged an option). A stop-gap that makes no measurable
  difference in 96 paired games goes. A
  learned change that fails its 96 games is reviewed before the next try: a bug (the records, `eval.why`), inputs
  the model lacks for the decision, or the limit of what the data teaches; the finding goes in docs/PROGRESS.md.
- **Fidelity before wins.** Gary is a sparring partner meant to play like the pros, so a change that moves it toward
  the pros' play (the build the pros play, the inputs the models learned from, a decoder that samples the pros'
  timings) is judged first on whether Gary now does it (build timings and counts per style against the pros of that
  style on the same maps, eval/style_adherence.py, never overall against overall; the gap score,
  the pipe and input checks) and on wins second. Simple cheesy plans (style 4, two-Barracks gasless marines, 1.5% of
  pro Terran games) are expected to win more against weak opponents like gz1 and Gary itself, so a drop in wins
  when leaving one is not a verdict. If such a change loses games, it is not abandoned: look for what stops it
  working (bugs, execution leaks, inputs, the decoder) and fix those, keeping the change on in the dev manifest
  meanwhile. The mistake this rule exists for: on 2026-10-23 styleroll (the pros' styles by share in place of the
  selector's style 4) was dropped for winning 26 of 48 against 31, though its own review found the cause was
  execution (gas mined and not spent, Barracks busy, production never added), and gt1dev went on playing style 4
  through every macro fix built for that failure. Options that only bet on results (a stop-gap, a tuned threshold)
  are still judged on results.
- Before any learning run beyond imitation, the rules in docs/SELF_PLAY.md ("Learning beyond imitation"): the
  pipe check shows pushed logits move what Gary executes; the model owns the decision in play and in the judging
  games; branched decisions are locked for the window (the pros' hold lengths); the minimum detectable effect
  is written down before a gate; learning from counterfactuals is offline. No online PPO on small batches.
- Gary loads the newest `runs/<kind>/<matchup>_<race>_*` model; experiments go to `--out` so they stay out of
  that glob.
- Make fixes reusable: race-keyed tables and shared providers rather than race-specific code, so the other
  races and matchups get them too.

## Working rules

- Never edit agent code while a league round runs: each game is a new process importing the code.
- Stopping a job's shell leaves its pool workers running: stop a run with `python -m eval.stop <pid>` (or
  `--match <text>`), which kills its whole process tree. Killed re-simulator calls aren't indexed as failures, so
  a rerun builds them. Kill leftover `gary_resim.exe` before rebuilding the simulator (the link fails otherwise).
- After executor or macro edits run `tests/test_smoke_game.py`, and `tests/test_pairs.py` before any run with
  paired branches (the bots' deep copy and a branch's replay, in under a minute); `tests/test_value.py` fails when
  the critic can't read the current Gary (a change of the macro features: retrain it, `python -m train.value
  --since <date>`, from the league's replays); `tests/test_inputs.py` (about 5 minutes) holds the vectors the macro, army
  and critic are fed in play against their datasets' for the same game, after any change to a model's inputs, a
  tracker or a dataset builder (a new difference fails it; its INTENDED and KNOWN lists, with issues, say what may
  differ); a change of behaviour records the parity
  hashes again (`tests/test_parity.py --record`) in its commit, with the reason; `tests/test_matchups.py`
  plays every matchup that can play.
- "Why did Gary do that?" is answered from the game's records (`python -m eval.why`) before writing a
  re-simulation probe.
- Long scripts report to the run monitor (`eval.monitor`, `eval.priority`).
- The repository is public: no local paths, machine details or personal information in files; the
  StarCraft install is passed at runtime (`GARY_SC_INSTALL`). No Blizzard game data and no GPL code
  (CONTRIBUTING.md). Gary never plays on the Battle.net ladder.

## Stop-gaps

Gary's decisions are meant to be learned (imitation of the pros, then self-play). Some are scripted rules
today because the learned layer can't do them yet and the rule measurably helps now. Before adding a
scripted rule, weigh whether it should stay a rule or become learned behaviour. If it is a stop-gap, tag it
in the code with `STOP-GAP (AGENTS.md)` and give it an entry below: what it does, why it is scripted for now,
what should replace it, and the measure that says it can go. When a learned layer takes over its job, run
the measure with the rule off; if the learned layer holds, delete the rule and its entry.

Not stop-gaps: the hands (clicks, selection, mining, rallies, placement mechanics), decoders (how a model's
odds become a choice: dice, hold, floors), and fixes of plain bugs.

- **Overlord hunt** (option `lords`, in gt1; `army_rules.py` LordHunt). An enemy overlord in sight with no
  fighters, sunkens or spores near it: the two nearest free shooters hunt it (race tables: marines;
  hydralisks and dragoons listed, not yet played). Scripted because neither the army nor the fight layer
  decides small detachments. Replace with a learned detachment decision (split off N units for a target).
  Measure: overlords killed by 8:00 with no more marines lost (16 games: 8 to 21, 116 to 107 lost).
- **The hunt far ahead and squad search** (options `hunt4`, `split`, in gt1; `army.py` hunt_ahead,
  hunt_split). The hunt also runs at 4x the enemy fighters seen in the last minute; with no enemy building
  known, squads of 8+ search the unseen bases at once. Scripted because the commit model's engages scatter
  when the enemy is nearly dead, and it has no notion of finishing or searching. Replace with the commit or
  army model trained with a reward for ending games (self-play) and searching as a learned scouting skill.
  Measure: games won against gz1 in 48 (24 with it, 21 without) and leads unfinished at 20:00. The hunt begins a
  quarter of gt1's attacks and goes into sunkens most (0.63 of its attacks, the commit model's 0.38: eval/attacks.py,
  2026-10-23); the static guard doesn't screen it. Imitated (2026-10-24): gt1dev runs it in advice mode
  (`huntadvice`, play unchanged) and train/commit.py --rule-labels learns its advice; the models follow it where it
  acts (0.81) and finish more leads, but copy its attacks into sunkens, and on v5's data trail the deployed commit
  model in games, so not adopted yet (docs/PROGRESS.md). Tried on gz1 with the
  static guard (48 games against gt1): the Zerg won 1 (2), so off for Zerg.
- **Static-defense guard on engages** (options `static` and `softguard`, in gt1; `army.py` static_odds,
  static_beta). An engage's spots where the group doesn't outweigh 1.5 times the ground static defense there (6
  supply each) and the enemy fighters seen near it keep (the group / 1.5 times that) squared of their odds: a nudge
  since gt0.10 (96 paired games against the veto it was: -0.016 +- 0.035, no measurable difference). Scripted because the commit model doesn't weigh static defense (human game 4's
  three attacks all went into sunkens). Replace with static defense as a commit-model input and its outcomes
  in training. Measure: the attacks' trade (`eval/attacks.py`): supply killed / lost 78.8 / 44.8 a game with
  it, 71.9 / 52.6 without; 2026-10-23 (NG70a against Q60a, gt0.7): lost / killed 70.9 / 92.6 off, 60.6 / 87.8 on,
  over 96 paired games (NG70a+b against Q60a+Q70b) worse off, -0.062 +- 0.044: it stays. The pros attack where it forbids (29% of their engages into
  static defense are under its 1.5x), and a commit model that sees static defense learned nothing more (army dataset
  v4: docs/PROGRESS.md).
- **Gather before engaging** (option `gather`, not adopted; `army.py` `_gather_start`, `_gather_step`). A commit
  engage with a quarter of the army's supply elsewhere first brings the whole army to its own middle (at least
  12 tiles from the target), then attacks once 75% of it is there or 20 s pass. It only delays the engage, never
  vetoes it: enemies at home don't call it off (whether to attack anyway is the model's; moving out is often what
  ends a mutalisk harass). Scripted because each cluster's commit decision is taken alone: two thirds of Gary-T's late
  arrivals at its attacks were units that existed when the attack began. Replace with the commit model learning
  to gather before it engages (the pros' "move" commitments include regrouping), then self-play. Measure: late
  arrivals per fighter at attacks (`eval/attacks.py`: Gary-T 0.27, pros 0.06) and 96 paired games. Measured
  2026-10-20 (96 paired games against gt1): the 10-15 minute gap nearer the pros (-0.045, interval below zero),
  lost 6 (9), but won 47 (54), more games at 30:00; late arrivals 0.26 to 0.22 only.
- **The pros' caps per minute** (in gt1 and gz1; `macro.py` plan, `capped`; the model's `cap`). A unit, building or
  expansion is refused once Gary has more than nine pros in ten ever had by that minute. The commonest veto of
  the model's decisions: 8,293 options set aside in 48 games of gt1 (Marines 2,963, Medics 1,320, Science Vessels
  988, turrets 781, Firebats 539, Barracks 483, depots 470, Control Towers 303). Scripted because a decoded
  unlikely choice could run away (v0.14's era). Partly a consequence of keep_busy (removed in gt0.7: its marines
  pushed the counts over the cap): 0.63 options refused a decision with it, 0.46 without. Replace with the model's
  own odds (it knows the counts it has; calibrated on the pros' states, the macro check). Measure: option `nocaps`
  against gt1, results and the macro gap. Off (round NC70a against Q60a, 2026-10-23): lost 10 (5), the macro gap
  +0.070 [+0.013, +0.125], turrets at 10-20 minutes 16.4 (pros 7.1, with it 9.2), medics 1.3 (4.4): the caps clean
  up after the decoder (the bank escape and the under-half rule take the runner-up while the favorite waits), so
  they go after those.
- **SCVs capped at the bases' saturation** (in gt1; `terran_executor.py`, "SCVs enough for its bases", MAX_SCVS 60).
  The model's SCV decisions are refused beyond the bases' saturation (2,673 times in 48 games, most at 5-15
  minutes). Scripted because Gary over-made SCVs (2026-10-09: 42-51 at 12:00 against the pros' 39). Gary-T's
  Command Centers stand idle 93% of 10-20 minutes, the pros' 84%. Replace with the model's own SCV decisions.
  Measure: an option off: workers at 12-20 minutes against the pros', results and the macro gap. Off (round NS70a,
  2026-10-23): workers at 10-20 minutes 48.6 (pros 42.4, with it 39.0), the macro gap +0.077 [+0.031, +0.122]; 1,335
  of its 2,206 SCVs at 5-20 minutes were the decoder's runner-up, not the model's favorite.
- **Under half the waiting favorite's odds** (in gt1 and gz1; `macro.py`, the decoder). While the model's favorite
  waits for a busy building, alternatives under half its odds are passed over and the macro waits (2,104 times:
  depots 614, SCVs 458, Barracks 242, refineries 164, Engineering Bays 146, Siege Tanks 62). Scripted so a busy
  building doesn't hand the turn to unlikely choices. With the queue of waiting decisions (gt0.7) a favorite unit
  no longer holds the macro; measure how often this still fires there, and an option without it.
- **The bank escape** (in gt1 and gz1; `macro.py` the decoder, BANK_ESCAPE 400). While the model's favorite waits, with
  over 400 minerals anything Gary can do now is taken instead. With the caps off it made 341 of the 1,174 turrets at
  5-20 minutes, and 759 of the 2,206 SCVs with the SCV cap off. Scripted so money doesn't bank behind a busy
  building. Replace with the model's own odds (it reads its minerals: pros with a bank add production) and a deeper
  queue of its waiting units (option `qdepth2`). Measure: option `nobank`, the bank, results and the macro gap. Off
  (NB70a against Q60a, 2026-10-23): paired -0.203 +- 0.123, the macro gap +0.208, minerals banked at 5-15 minutes
  1,124 (pros 511, with it 387): while the favorite waits it stays the macro's goal and the model isn't asked again,
  so it never gets to add production; with qdepth2 too (QD70a) 675 banked, -0.078, the gap +0.052. It covers for
  the one goal at a time; next, the model's units queued in the building and paid for, as pros do (gamequeue,
  adopted in gt0.9; the escape still holds with it: off as well, -0.130 +- 0.152). In gt1dev (escape off) the macro
  draws from the model's odds (dice, gt0.11dev); rethink and nosave, which drop a waiting goal, were worse in games.
- **Odds summed into a decision** (`mass` for one-offs, in gt1: tech buildings, add-ons, research, upgrades;
  `expand_mass` for expansions; `macro.py` odds_mass, expand_odds). A one-off is decided once its odds, summed over
  the decisions taken while it was possible, reach one (278 times in 48 games), an expansion likewise (194).
  Decoders, but they decide on their own schedule rather than the model's odds of the moment. Scripted because
  the favorite (almost always a unit) never let a few-percent tech building come (a Factory by 8:00 in 4% of
  games against the pros' 62%). Measure: their decisions' timings against the pros' (build orders), and an option
  without them once the decoder draws (dice) everywhere.
- **Answers to the enemy's plan** (`answers.py`, in gt1). Lurkers or mutalisks read from what Gary saw force Siege
  Mode, tanks, vessels, Irradiate and turrets (443 forced decisions in 48 games) and veto turrets beyond one a base
  (387). Scripted because the macro model didn't answer what it had seen (human games 1-2). Replace with the seen
  enemy units as macro-model inputs (they are: the `seen` counts) and measure whether the model answers alone. In
  play its "most seen" inputs stayed 0 until option `seenall` (2026-10-25: the tracker looked only when the model was
  queried); gt1 doesn't have it yet, so the answers' measure is to be taken with it.
  Measure: an option off, against lurker and mutalisk plans: results and the time to the answer.
- **The early-pool posture's builds** (in gt1; off in gt1dev since gt0.18dev, option noalarm: it went to rush defense in
  every game, half of them blind, and held the opening's gas and natural back; rush defense from scouting: bunker at the ramp, marines first, no
  expansion). Forces 454 decisions and blocks expansion 145 times in 48 games. Scripted for 9-pool and 12-pool
  rushes (v0.11-v0.18, losses 27 of 48 to the pro-timed 9-pool before it). Replace with the posture as a learned
  read of the scouting (the posture's alarm a model input). Measure: losses to early pools (eval/vs_rush.py) and
  the gap at 0-10 minutes with it off.
- **Older rules, to review** (scripted before 2026-10-20; each needs a full entry when next worked on): Gary-Z's
  alarm and the natural colonies; the first producer by a time (`rax_by`); the v0.16 hunt and base scans; the fight
  layer's rules (repair, pull, load, reinforce, stim, detect; the Zerg roles' retreat when outweighed; RoleAhead's
  hand rules in contact, gt1's whole fight in contact); Gary-Z's army rules (posts, raid, join, sweep, press).

## When a new replay pack lands

```
python ingest/inventory.py --source cwal
python resim/run_batch.py --source cwal --remastered
python ingest/build_orders.py --source cwal
for m in TvZ PvT PvZ PvP ZvZ TvT: python ingest/{macro,army,fight}_dataset.py --matchup $m --source cwal
python -m train.macro --matchup TvZ --race Z --weight-modern 10      # judge with --test-era modern
```

Both eras with the modern games weighted ten to one is the measured recipe (docs/DATA_ERAS.md).
