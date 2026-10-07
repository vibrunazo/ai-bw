# Replay fixtures

A small, committed set of replays for tests and for cloud/CI sessions that don't have the full
datasets. Bulk data lives in `data/` and is never committed.

## Rules for adding a fixture

- **Redistributable sources only.** STARDATA (BSD, with attribution), replays of games we play
  ourselves (e.g. vs. the computer or bot vs. bot), or packs whose license explicitly allows it.
  Nothing from sources marked `redistributable: unknown` or `no` in `data/sources.yaml`.
- **No ladder replays** from other players (account names, in-game chat).
- **Keep it small:** a handful of files, each well under 1 MB.
- **List every fixture below** with its source, license and why it's here.
- **Fixtures keep their source's license.** They are not covered by the repo's CC0/MIT-0 terms. STARDATA fixtures are BSD-licensed; keep the attribution.

## Fixtures

| File | Source | License | Version | Matchup | Why it's here |
|---|---|---|---|---|---|
| `stardata_tvz_standard_ozp3w.rep` | STARDATA `12/bwrep_ozp3w.rep` | BSD (StarData) | 1.16.1 | TvZ, Fighting Spirit, 13 min | Typical-length TvZ |
| `stardata_tvz_short_2yus7.rep` | STARDATA `0/bwrep_2yus7.rep` | BSD (StarData) | 1.16.1 | TvZ, Fighting Spirit, 4 min | Short game (early aggression / all-in) |
| `stardata_tvz_long_7i821.rep` | STARDATA `13/bwrep_7i821.rep` | BSD (StarData) | 1.16.1 | TvZ, Fighting Spirit, 30 min | Long late-game TvZ |
| `stardata_tvz_colosseum_pqt42.rep` | STARDATA `12/bwrep_pqt42.rep` | BSD (StarData) | 1.16.1 | TvZ, Colosseum II, 12 min | A map where pros wall their natural (and with creep built into it) |
| `stardata_pvz_standard_56jkk.rep` | STARDATA `1/bwrep_56jkk.rep` | BSD (StarData) | 1.16.1 | PvZ, Fighting Spirit, 13 min | A non-TvZ matchup |
| `vib_zvp_vs_computer_scr.rep` | Contributor's own game (`data/raw/vib/`) | CC0 | SC:R 1.21+ | ZvP vs. computer, Blood Bath, 13 min | Remastered replay format (OpenBW can't read it without conversion) |

STARDATA attribution: replays from the StarData dataset, Lin et al., "STARDATA: A StarCraft AI
Research Dataset" (2017), https://github.com/TorchCraft/StarData, BSD license. Selected with
`ingest/inventory.py`: 1v1 human games, winner known, no in-game chat.
