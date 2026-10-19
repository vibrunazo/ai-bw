# Old and modern replays

Two eras of pro and high-ladder replays feed Gary's models: StarData and the TL packs (patch 1.16
and before, 2007-2016, mostly 2009-2013) and cwal.gg's Remastered-era ladder games (2026). This file
holds what was measured when the first modern pack arrived (2026-10-18), to decide how much the new
data is worth and how to weigh it. The trainers select on era with `--era` and weigh it with
`--weight-modern` (train/select.py, train/macro.py); the inventory records each replay's era as
`version` ("-1.16" or "1.21+").

## The first modern pack (data/raw/cwal/2026-10)

| | cwal 2026-10 | StarData |
|---|---|---|
| Replays | 6,098, about 1,015 per matchup | 65,354 |
| Flagged as bot games (inventory heuristic) | 1,187 (19%) | under 0.2% |
| Human 1v1 games | 4,907 | 65,288 |
| Re-simulated cleanly (no desync) | 4,586 (93% of the human games) | about 95% |
| Median game length | 11 min | 15 min |
| Median APM | 348 | 229 |
| Distinct player names | 1,770 | thousands |
| Maps | 9 current ladder maps | the maps of 2007-2016 |

Clean games per matchup: ZvZ 958 (98%), PvZ 829 (98%), PvP 807 (97%), TvZ 733 (92%), TvT 658
(85%), PvT 601 (88%). 111 games failed with "unsupported map version 64" (Aiolos).

The first re-simulation had 81% clean, with KnockOut at 36% desync. The cause was ours: a
Remastered replay numbers units in a 3,400-slot table and our translation to OpenBW's 1,700 slots
shifted every id, but both engines give the first unit created (the map's first UNIT entry) slot
0 and only allocate the rest from the top down. On KnockOut that first entry is a main's geyser,
so every command aimed at it was lost (resim/scr_replay.h, 2026-10-18). Of the remaining 4%
(321 games) most are TvT and PvT on Attitude SE (15%) and Colorless Fate (6%); other maps are at
1%. Ruled out for those: the unit-id translation (no misses before the flagged minute), the maps'
critters and extra text trigger (removing them from the map changes nothing), the active bug-fix
flags (identical in every replay), the Remastered add-to-control-group hotkey (OpenBW handles it),
and a systematic mining deficit (clean Attitude games refuse short-money trains no more often
than clean Octagon games). In the sample studied (Dapenda vs lllII!lllI!llIl, 2026-08-12), the
economy matched the real game to the mineral at 2:31 (224 before the Factory, 24 after, in
StarCraft and in the cwal.gg viewer). The first real divergence is at 2:53: Dapenda orders a
second Factory while one of their own SCVs stands on the site. OpenBW's build action drops such
an order silently (BW refuses placement over any unit except for add-ons), but the real client
sent it, which means the real SCV was not on the site at that frame: it had been ordered there
75 frames earlier and in the simulation had arrived and stopped 21 frames before. So the
remaining cause is a difference in unit movement (worker collision or pathing, SC:R versus
1.16), not in command decoding; it shows on maps and races where units are sent through busy
mineral lines to cramped build spots. Making own units never block placement was tried and
rejected: it fixed nothing and desynced 4 of 30 clean control games. The debug traces that
found this are GARY_TRACE_UNIT and GARY_TRACE_ORDER in resim/gary_resim.cpp.

The bot flag: 589 flagged players are at 800+ effective APM, which no human reaches. The other 899
sit at a 98%+ effective-to-total APM ratio between 300 and 800 APM, a rule calibrated on the old
data (human p99 was 0.97). Modern players spam less (median ratio 0.72 against 0.77), so some of
those may be humans; a few names appear flagged in some of their games and not others.

## The builds moved

From `ingest/build_orders.py` over both eras, bot games excluded (the scratch comparison that made
these tables is easy to redo: first timings of key buildings, and the first four buildings or
tech items by 7:00). Medians are game seconds; the share is of player-games having the item by
10:00.

**TvZ Zerg** (15,345 old, 852 new player-games): the 3-hatch openings are gone, 2-hatch Lair is
the game.

| Opening (first four) | Old | New |
|---|---|---|
| Hatchery, Pool, Hatchery, Extractor | 33.5% | 1.9% |
| Hatchery, Pool, Extractor, Lair | 11.5% | 62.7% |
| Pool, Hatchery, Hatchery, Extractor | 8.7% | 2.5% |

Lair 3:59 to 2:53, Spire 5:10 to 3:59 (78% to 88% of games), Hydralisk Den 77% to 65%, Lurker
Aspect 61% to 48%, Hive 24% to 39%, Defiler Mound 5% to 13%.

**TvZ Terran** (15,345 old, 852 new): the same tree, a minute earlier, more Armories.
Factory 6:10 to 4:59, Starport 7:43 to 6:52, Armory 19% to 38%, Machine Shop 68% to 50%. The
most common opening (Barracks, CC, Refinery, Engineering Bay) went from 7% to 22%.

**PvT Terran** (17,073 old, 680 new): Barracks, Refinery, Factory, Machine Shop fell from 47% to
6%; Barracks, Refinery, Factory, Vulture rose from 7% to 42%; Armory 48% to 73%; second CC 4:36 to
3:32.

**PvT Protoss** (17,075 old, 680 new): the gateway-core-dragoon opening held (50% to 53%), the
Nexus comes at 3:15 instead of 4:53, Robotics Support Bay 27% to 56%, Dragoon Range as the fourth
item went from 10% to under 1%.

**PvZ Protoss** (17,803 old, 848 new): the least stable matchup. Forge, Nexus, Cannon, Gateway
8% to 22%; Forge, Cannon, Nexus, Gateway 19% to 9%; gateway-first expands appeared (Gateway,
Nexus, Forge/Assimilator 1% to 17%); Stargate 72% to 88%, Robotics 49% to 36%. Only 55% of the
new openings are among the old top-20.

**PvZ Zerg** (17,801 old, 848 new): Pool, Hatchery, Extractor, Hatchery 18% to 37%; Lurker
Aspect 41% to 33%; Queen's Nest 11% to 2%.

**PvP** (13,816 old, 1,662 new): gateway-core-dragoon 40% to 62%, Citadel 48% to 24%, Robotics
Support Bay 39% to 57%; Nexus 6:22 to 5:01.

**TvT** (4,790 old, 1,542 new): Barracks, Refinery, Factory, CC 8% to 23%; double Factory 21% to
7%; Engineering Bay 45% to 22%; CC 4:11 to 3:19.

**ZvZ** (12,088 old, 1,960 new): the closest to unchanged; the same six openings in a different
order, Lair and Spire about 12 s earlier.

Everywhere: games are shorter, the tech tree comes earlier, expansions are earlier and more
frequent (town halls by 6:00 up in every row), and the sequences themselves are mostly not new
(80-94% of new openings are among the old top-20, PvZ Protoss 55%) but their frequencies are a
different distribution. A model trained on the old data alone would open 3-hatch against a 2026
Zerg player who has not seen it since the maps changed.

## What the models say

TvZ macro models (train/macro.py), evaluated on the held-out tenth of each era: 400 old
player-games and 70 modern ones (`--test-era`, `--weight-modern`, `--games`; runs/era_experiments).
Top-1 is the share of next-decision predictions that match the pro's.

| Training set | Player-games | Old top-1 | Modern top-1 (Z) | Modern top-1 (T) |
|---|---|---|---|---|
| Old era only | 14,087 | 0.521 | 0.534 | 0.498 |
| Modern only | 585 | 0.409 | 0.477 | 0.442 |
| Both, modern weight 1 | 13,258 | 0.507 | 0.560 | 0.506 |
| Both, modern weight 10 | 13,258 | 0.501 | 0.568 | 0.512 |

The headline accuracy hides the shift: modern games are more routine, so the old model scores
well on them while getting the decisions that changed wrong. Recall of those decisions on the
modern test, Zerg (old / modern-only / both x1 / both x10):

| Decision | Share of samples | Old | Modern only | Both x1 | Both x10 |
|---|---|---|---|---|---|
| Lair | 171 | 0.36 | 0.00 | 0.68 | 0.72 |
| Expansion hatchery | 647 | 0.29 | 0.24 | 0.44 | 0.51 |
| Spire | 151 | 0.25 | 0.00 | 0.33 | 0.46 |
| Extractor | 248 | 0.06 | 0.00 | 0.08 | 0.34 |
| Ultralisk | 153 | 0.54 | 0.22 | 0.65 | 0.71 |

Terran: Academy 0.13 / 0.00 / 0.18 / 0.45, Command Center 0.31 / 0.00 / 0.38 / 0.45, Valkyrie
0.14 / 0.00 / 0.17 / 0.29. The modern-only model, from 585 player-games, has zero recall on
everything rare: far too little data to stand alone. Weighting the modern games ten to one moves
the shifted decisions toward the modern meta at a cost of 0.6 points on the old era.

Learning curve on the old era (Zerg, old held-out top-1): 300 games 0.390, 1,000 0.444, 3,000
0.478, 10,000 0.503, 14,000 0.506. Each tripling adds less: 5, 3, 2 points. On the modern test,
585 modern games (0.477) are worth about 1,000 to 1,500 old ones (0.469 at 1,000): a game of the
right era counts roughly double.

## What to do with it

- **Get more modern replays**: about 3,000 clean human games per matchup is where the curve
  flattens, which at today's yield (81% human, 81% of those clean) means roughly 4,500 raw
  replays per matchup, four to five times the first pack. PvT and TvT, the lowest yields, need
  the desync fix first; ZvZ and PvP are already closest.
- **Train on both eras with the modern games weighted** (`--weight-modern 10` is the measured
  point; between 5 and 20 is the range to tune as the modern set grows). The old data keeps the
  rare situations and the reactions; the weight hands the openings to the modern games.
- **Era as an input** (planned): the model gets the game's year the way it gets the build style,
  and plays with it set to now. It learns what depends on era and what does not, so no weight
  has to be chosen. The same conditioning gives "play like 2012" for free.
- **Judge on the modern held-out set** from now on; the old held-out set measures the past.
