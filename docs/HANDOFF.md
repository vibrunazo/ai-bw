# Handoff (2026-10-19)

Where the project stands after the 2026-10-18 session, for whoever continues. Everything below is
committed and pushed; nothing runs in the background.

## State

- **Gary is every race, every matchup** (ARCHITECTURE.md §1). Gary-T (gary/agent) plays Terran in
  TvZ; Gary-Z (gary/bots/gary_z.py, its own stack) plays Zerg in TvZ and now ZvZ. docs/RACES.md is
  the audit of what is race-neutral and what is not, with the order of work; step 1 (Zerg mirror)
  is done, TvT and Protoss are not started.
- **Data.** StarData and TL packs (1.16 era) plus the first modern pack, data/raw/cwal/2026-10:
  6,098 ladder replays, 1,217 flagged as bot games, 4,586 re-simulating clean (93%). Macro, army and
  fight datasets exist for all six matchups (old era for TvZ, ZvZ, TvT; modern for all six);
  build orders for both eras. docs/DATA_ERAS.md has the comparison: the builds moved in every
  matchup; both eras with the modern games weighted ten to one is the measured recipe.
- **Pipeline.** Every stage is incremental by replay hash. Trainers take --matchup, --era, --source,
  --bots (train/select.py); the macro trainer also --test-era, --weight-modern, --games, --out.
  Dataset builders take --source (rebuild one pack with --force). train/macro_history.py writes the
  pros' decision-history tables, named history_mean_<matchup>_<race>.
- **Re-simulation.** Remastered unit ids: the bottom half of the 3,400-slot table maps one to one
  (resim/scr_replay.h), which took the pack from 81% to 93% clean. The remaining 4% (Attitude SE 15%,
  Colorless Fate 6%, TvT and PvT) is a unit-movement difference between SC:R and 1.16, traced with
  GARY_TRACE_UNIT / GARY_TRACE_ORDER; not fixable blind (DATA_ERAS.md). Aiolos fails with
  "unsupported map version 64" (111 games).
- **Models.** TvZ macro, army, commit, fight, belief, plan, engage, roles for Terran (runs/*/TvZ_T_*),
  TvZ and ZvZ macro and army for Zerg, a Zerg value model. The era experiments are in
  runs/era_experiments (not the newest models; Gary loads runs/<kind>/<matchup>_<race>_* newest).
- **Play versions.** Gary-T default for human play: v029+commit+draw+floor+mass (memory
  gary-bot-status). Gary-Z default GARY_Z_HISTORY=pro (per-style pro history fading over 12 min).
  eval/mirror.py plays Gary-Z against Gary-Z (ZvZ); eval/league.py Gary-T against Gary-Z.

## Open threads

1. **More modern replays.** The curve flattens around 3,000 clean games per matchup; at today's
   yield that is about 4,500 raw replays per matchup, four to five times the first pack. PvT and
   TvT have the lowest yield. The user downloads from cwal.gg by hand (no API, robots.txt disallows
   crawling; the creator is Dex, reachable through Patreon). Confirmed ladder bots: Namepen99,
   MOROKEI1-3, dreamdawn (ingest/inventory.py KNOWN_BOT_NAMES); the APM rule caught 226 of their
   258 games on its own.
2. **Era and map as model inputs.** One change: the macro FeatureSpec gets the game's year and map
   hash the way it has the build style; play with year = now. Then retrain TvZ T and Z on both eras
   with --weight-modern 10, judge on the modern held-out tenth, and make those the defaults. This is
   the first "plays the 2026 meta" Gary. TODO.md "Map mechanics" has the rest of the map plan.
3. **TvT Gary-T.** Needs manifests._newest to take the matchup from the game (defaults TvZ, T) and
   TvT models for macro, army, fight, belief, plan, engage, commit (hours of training; ask first).
   roles.py's Zerg prey kinds (mutas, lurkers, scourge) will show there too.
4. **One Gary** (RACES.md step 3): the Zerg stack onto the agent's arbiter; race = data module +
   executor + macro carry-out. Gate: Gary-Z on the new stack matches its league record.
5. **Protoss** (RACES.md step 4): gary/protoss.py, the executor, PvT/PvZ/PvP models.
6. **A human game** against the current Gary-T; none since human game 3 (2026-10-17). The LAN
   recipe is in memory (gary-live-sessions).
7. **Gary-T open items** from PROGRESS.md: the hunt rule replaced by learned engage, SCVs in
   attacks, turrets the model fails to start, Gary-Z finishing speed.

## How to resume the data work when a new pack lands

```
python ingest/inventory.py --source cwal
python resim/run_batch.py --source cwal --remastered
python ingest/build_orders.py --source cwal
for m in TvZ PvT PvZ PvP ZvZ TvT: python ingest/{macro,army,fight}_dataset.py --matchup $m --source cwal
python -m train.macro --matchup TvZ --race Z --weight-modern 10      # judge with --test-era modern
```

Rules learned the hard way: never edit gary_z.py while a league round runs; a training run with
--out keeps experiments out of the "newest model" glob; kill leftover gary_resim.exe before
rebuilding the simulator (the link fails otherwise).
