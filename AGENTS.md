# AGENTS.md

How to work in this repository, for coding agents and people alike. The design is in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); what each version did and how it measured is in
[docs/PROGRESS.md](docs/PROGRESS.md) (keep it updated after each milestone); open work is in [TODO.md](TODO.md).

## Where things are

- `gary/`: the agent (`gary/agent`: arbiter, executor and its race plugs, macro, army, fight, rules), the
  race data (`gary/terran.py`, `zerg.py`, `protoss.py`, `race.py`), the learned models (`gary/policy`), the
  interface (human-like hands) and the game environment.
- `eval/`: the launcher (`match.py`), league rounds (`league.py`, which keeps each round's feature rows), measures
  against the pros (`humanlike.py`, `attacks.py`), two versions compared game against game per time window with a
  bootstrap spread (`gap_windows.py`), the records' queries (`why.py`), the run monitor (`monitor.py`,
  `priority.py`).
- `train/`, `ingest/`, `resim/`: models, datasets from replays, the replay re-simulator.
- `adapters/scr_bridge`: playing StarCraft: Remastered. `tests/`: smoke, parity, matchups, game data.
- `docs/`: ARCHITECTURE (design), CONTROL (layers and manifests), RACES (the per-race audit), REFACTOR (one
  Gary for every race, done), SELF_PLAY (the self-play plan), DATA_ERAS (old and modern replays).

## The Garys and how to judge a change

- One Gary for every race: manifest `gt1` plays Terran, `gz1` Zerg (`gary/agent/manifests.py`). Every game
  runs through `python -m eval.match --a <manifest> --b <manifest>`; `python -m eval.league` plays Terran
  (`--vs`, default gt1) against Zerg (`--zerg`, default gz1); `gary/play.py --live` plays a human.
- Two Terran manifests: `gt1` is what plays (the strongest measured: human games, the tests' parity, the league's
  default) and `gt1dev` is what learning is trained and judged in: gt1 with the role model deciding in contact and
  the army on control groups, the configuration expected to win later. A learned layer's work (data, training, the
  pipe check, RL) is done and judged in gt1dev; gt1dev replaces gt1 when it beats it in 96 paired games, with a
  version bump. The suite's numbers for both, and how far apart they are, are in docs/PROGRESS.md.
- A change is an option on them (`gt1+<option>`), judged in paired 48-game rounds on the ladder maps
  (`--maps ladder --seeds 6 --first 4004`, 30 minutes) and the gap score to modern pros; 24 games are too
  noisy for close calls, and a single round's gap score needs its spread (`eval/gap_windows.py`, 96 paired games
  for a close call). An adopted option is folded into the manifest with a version bump, so the name
  never grows into a chain of options.
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
- Stopping a job leaves its pool workers running: kill them by parent PID. Kill leftover `gary_resim.exe`
  before rebuilding the simulator (the link fails otherwise).
- After executor or macro edits run `tests/test_smoke_game.py`; a change of behaviour records the parity
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
  Measure: games won against gz1 in 48 (24 with it, 21 without) and leads unfinished at 20:00. Tried on gz1 with the
  static guard (48 games against gt1): the Zerg won 1 (2), so off for Zerg.
- **Static-defense guard on engages** (option `static`, in gt1; `army.py` static_odds). An engage only goes
  where the group outweighs 1.5 times the ground static defense there (6 supply each) and the enemy
  fighters seen near it. Scripted because the commit model doesn't weigh static defense (human game 4's
  three attacks all went into sunkens). Replace with static defense as a commit-model input and its outcomes
  in training. Measure: the attacks' trade (`eval/attacks.py`): supply killed / lost 78.8 / 44.8 a game with
  it, 71.9 / 52.6 without.
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
- **Older rules, to review** (scripted before 2026-10-20; each needs a full entry when next worked on): the
  early-pool posture and its builds (rush defense from scouting); Gary-Z's alarm and the natural colonies;
  the first producer by a time (`rax_by`); supply ahead of production and the pros' caps per minute; the
  answers to the enemy's tech (`answers.py`); the v0.16 hunt and base scans; the fight layer's rules
  (repair, pull, load, reinforce, stim, detect; the Zerg roles' retreat when outweighed); Gary-Z's army
  rules (posts, raid, join, sweep, press).

## When a new replay pack lands

```
python ingest/inventory.py --source cwal
python resim/run_batch.py --source cwal --remastered
python ingest/build_orders.py --source cwal
for m in TvZ PvT PvZ PvP ZvZ TvT: python ingest/{macro,army,fight}_dataset.py --matchup $m --source cwal
python -m train.macro --matchup TvZ --race Z --weight-modern 10      # judge with --test-era modern
```

Both eras with the modern games weighted ten to one is the measured recipe (docs/DATA_ERAS.md).
