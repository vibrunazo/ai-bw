# Stop-gaps: scripted rules standing in for learned behaviour

Gary's decisions are meant to be learned (imitation of the pros, then self-play: docs/SELF_PLAY.md). Some are
scripted rules today because the learned layer can't do them yet, and a rule measurably helps now. Each one
here is a stop-gap: it says what it does, why it is scripted for now, what should replace it, and the measure
that tells us it can go. In the code each carries the tag `STOP-GAP (docs/STOPGAPS.md)`.

Not stop-gaps: the hands (clicks, selection, mining, rallies, placement mechanics), decoders (how a model's odds
become a choice: dice, hold, floors), and fixes of plain bugs. Those stay.

When a learned layer takes over a rule's job, run the rule's measure with the rule off; if the learned layer
matches it, delete the rule and its entry.

## Added 2026-10-20

### Overlord hunt (`lords`, army_rules.py LordHunt)
- **Does:** an enemy overlord in sight with no fighters, sunkens or spores near it: the two nearest free
  shooters hunt it, held by the rule, until it dies, gets away, the enemy comes, or 25 s pass. Race tables:
  the prey by the enemy's race (overlords), the hunters by Gary's (Terran marines; Zerg hydralisks and Protoss
  dragoons listed, not yet played).
- **Why scripted:** the army model moves clusters and the fight roles engage fighters; neither has a "pick off
  a lone flyer with a few units" decision. The user's note after human game 4.
- **Replace with:** the fight or army layer deciding small detachments itself (a learned "split off N units to
  target X"), trained on the pros' detachments and then self-play.
- **Measure:** overlords killed by 8:00 with no extra marines lost (16 games: 8 to 21, 116 to 107 lost).

### The hunt when far ahead, and squad search (army.py `hunt_ahead`, `hunt_split`; in gt1)
- **Does:** the hunt (v0.16's: the whole army to the enemy's known buildings, then to bases unseen) also runs
  at 4x the enemy fighters seen in the last minute; with no enemy building known, squads of 8+ (4 at most)
  search the unseen bases at once.
- **Why scripted:** with the Zerg down to a base and no army, the commit model's engages scattered over every
  cell with an overlord seen; it doesn't know "finish the game" or "search".
- **Replace with:** the commit/army model with the game's state (the enemy's bases and army as believed) and
  a reward for ending games, in self-play; searching as a learned scouting skill.
- **Measure:** 48 games against gz1: games won (24 with it, 21 without) and leads unfinished at 20:00.

### Static-defense guard on engages (army.py `static_odds`, option `static`; in gt1)
- **Does:** an engage only goes where the group outweighs, 1.5 times, the ground static defense there (each
  sunken, cannon or bunker as 6 supply) plus the enemy fighters seen near it.
- **Why scripted:** human game 4's three attacks all went into 2-4 sunkens and lost; the commit model doesn't
  weigh static defense.
- **Replace with:** the commit model with static defense as an input and its outcomes in training (self-play
  will punish attacking into sunkens).
- **Measure:** the attacks' trade (eval/attacks.py): with it, supply killed / lost 78.8 / 44.8 a game against
  71.9 / 52.6, losing attacks 32% (38%); the share near sunkens is unchanged (43%, pros 28%), as big attacks
  still go in.

### Overlords moved off the mineral line (zerg_executor.py `_lords_off_minerals`)
- A hands chore, not a decision: kept (the minerals rally brings each new overlord onto the field it rallies
  to, and the clicks there hit it).

## Older rules (scripted before 2026-10-20), to review

One line each; each needs the fuller entry above when it is next worked on.

- **Early-pool posture and its builds** (posture early_pool/v5, one_base_defense, rush bunker): rush defense
  from scouting. Replace with: the macro and army models conditioned on what was scouted.
- **Gary-Z's alarm** (zerg_posture.py: drones forbidden and zerglings made while enemy fighters are seen) and
  **natural colonies** (option `colonies`). Replace with: the macro model reacting to threat inputs.
- **The first producer by a time** (macro.py rax_by: Barracks, Spawning Pool). Replace with: the macro model
  once it never skips it.
- **Supply ahead** (depots / overlords ahead of production) and **the pros' caps** per minute. Replace with: a
  macro model that doesn't block itself (the caps keep its odds near the pros' counts).
- **Answers** (answers.py: lurkers seen -> Siege Mode, tanks, vessels, turrets). Replace with: the macro model
  with the enemy's tech as an input.
- **The hunt** (army.py v0.16, when no enemy army was seen for a minute) and **scans** of bases. See the
  far-ahead entry.
- **Fight rules** (repair, pull, load, reinforce, stim, detect; Zerg roles' "retreat when outweighed"): the
  fight layer's hand rules around the learned roles. Replace with: the roles model and self-play.
- **Gary-Z's army rules** (posts, raid, join, sweep, press): overlord posts and the mutalisk raid are skills a
  learned layer would call; join, sweep and press are judgments to learn.
