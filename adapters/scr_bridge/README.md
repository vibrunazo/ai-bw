# scr_bridge — Gary's StarCraft: Remastered adapter

Gary plays a **live Remastered client** through this adapter: it reads the game state the
client is simulating and injects Gary's commands as the local player's actions. That is what
lets Gary practice against real humans in custom games (`docs/ARCHITECTURE.md` "Game Adapter",
the "SC:R via bridge" backend).

**Status: v0 spike — skeleton validated live.** The build, hash pin, DLL injection, frame
watcher, pipe protocol and `gary.env.Game` surface all run against a real SC:R client on this
machine (`launch.py --smoke`: `ping: pong`, `hash_verified: true`, `frame_watcher: true`). Field offsets
marked `UNVERIFIED` in [`src/scr_profile.h`](src/scr_profile.h) must still pass the in-game probe
(below) before trusting observations. This document is the design record;
`docs/ARCHITECTURE.md` is deliberately *not* updated until the adapter is shown to play.

## The recipe: Shieldbattery's skeleton + Pluto's control + Gary's contract

Three projects cover the problem between them; we take one layer from each.

| Layer | Source | What we take |
|---|---|---|
| **Skeleton** (launch, inject, hook, frame pump, fail-safe) | [ShieldBattery](https://github.com/ShieldBattery/ShieldBattery) (`game/`, MIT) | In-process DLL architecture; "async side never touches game state, sync side lives in hooks"; verify-first startup; polite failure. |
| **Control** (issue orders as the player) | [Pluto-AI-Starcraft-Remaster](https://github.com/chan22222/Pluto-AI-Starcraft-Remaster) (**no license** — design only, no code copied) | Inject bot commands into Remastered's own turn queue via `send_command`; hold commands to match turn latency; JSONL log of every submitted packet; refuse to run on unknown binaries; drop commands on any fault but never kill the game. |
| **Contract** (what Gary sees and does) | `env/gary_env.cpp` + `gary/env.py` (this repo) | The `gary.env.Game` method surface and the exact `observe()` JSON schema, so `gary/interface.py` runs unchanged. |

We deliberately **skip** the Pluto bridge's BWAPI-4.4 / 1.16.1-memory-mirror layer. It exists
only because `pluto.dll` is binary-only and hard-codes 1.16.1 addresses. Gary speaks its own
contract over IPC, so we read SC:R structs directly and avoid that layer (and BWAPI's LGPL).

### License provenance (what may be copied, what may not)

| Source | License | Use in this folder |
|---|---|---|
| ShieldBattery `game/` | MIT | Architecture ideas; code copy allowed with notices |
| Pluto-AI-Starcraft-Remaster | **none declared** (all rights reserved) | **Design study + address facts only.** No code copied. Ask the author before copying |
| `external/screp` (command IDs/formats) | ISC-style (its repo) | Command table facts |
| `resim/gary_resim.cpp`, `env/gary_env.cpp` | this repo (MIT-0) | Translation logic (we already wrote the reverse direction), contract |

## How it works

```
launch.py ──finds SC:R (x86), checks SHA-256 against the profile pin
   │        writes gary_scr.json (pipe name, log path) next to gary_scr.dll
   │        scr_inject.exe starts StarCraft.exe **-launch** (the game exits immediately without
   │        this flag) suspended, loads gary_scr.dll (CreateRemoteThread + LoadLibraryW), resumes
   ▼
inside StarCraft.exe (32-bit):
   DllMain ──init thread──► verify exe hash ──► resolve profile RVAs against the module
   │                        start frame watcher thread: poll Game.frame_count (no game
   │                        patches and no hooks — see "Frame source" below)
   │                        start NDJSON named-pipe server (SEH-guarded dispatch)
   ▼
per completed simulation frame (the watcher):
   ├─ frame counter sync (Game.frame_count is ground truth)
   ├─ drain the act() outbox ──► translate legacy→1.21 commands ──► scr send_command()
   └─ (future: update an immutable snapshot; v0 reads memory at request time)
   ▼
gary/scr_env.py (ScrGame) ──same methods as gary.env.Game──► gary/interface.py ──► bots
```

Design notes, following the two reference projects:

- **Nothing in the game is patched.** A watcher thread polls `Game.frame_count` and flushes
  queued commands at frame boundaries; everything else runs on the pipe thread (Shieldbattery's
  async/sync split). Every request runs under an SEH guard: a bad memory read degrades to an
  error reply, never a dead connection or a crashed game.
- **Verify, then trust.** The executable's SHA-256 must match the pinned build (1.23.10.13515
  x86); `hash_verified` gates every state-reading request. A game patch fails closed with a
  clear message instead of reading garbage — the Pluto bridge's discipline. Later we can add
  `samase_scarf`-style runtime re-discovery (Shieldbattery's approach) to survive patches
  automatically.
- **Degrade, don't die.** If the frame watcher has not seen a frame yet (menu/loading), `act()`
  submits at request time and `step()` sleeps in real time, instead of refusing service. The
  watcher is an optimization of timing, not a correctness requirement.

### Frame source (v0 hook → v1 watcher)

v0 followed the Pluto bridge and hooked `GetTickCount` (loader import table + the game's cached
pointer at `kTimingTickPtr`, per [PL]). Live testing on this build showed neither indirection
point engages: the game imports no `GetTickCount` IAT slot (`iat_slots: 0` in
`build/logs/scr_bridge.log`) and the cached pointer never became patchable — the frame counter
stayed 0 and queued `act()` commands silently never flushed. v1 instead polls `Game.frame_count`
(10 ms) and treats each advance as the frame boundary: zero game patches, no call-site facts,
self-healing across input/timing differences. The turn queue tolerates the ≤10 ms jitter (turn
latency is ≥2 frames).

## Gary's contract (parity with `env/gary_env.cpp`)

`ScrGame` in [`gary/scr_env.py`](../../gary/scr_env.py) implements the same methods
[`gary/env.py`](../../gary/env.py)'s `Game` does, so `gary/interface.py` and the bots are
backend-agnostic:

| Method | Meaning | SC:R implementation |
|---|---|---|
| `observe()` | full state as JSON (schema below) | Game/Player/Unit/Sprite structs per `scr_profile.h` |
| `act(slot, bytes)` | one command, replay-format bytes (no player-id byte) | translate to 1.21 command, inject at the next frame boundary via `send_command` |
| `step(frames)` | advance N frames | wait until `Game.frame_count` advances by N (the game steps itself) |
| `unit_at(slot, x, y)` | tag a click would hit | clickable image-union + draw depth (gary_env's model; bbox fallback, see gaps) |
| `box_select(slot, ...)` | tags a drag box would select (≤12) | same rules as `gary_env_box_select`: own units, mobile before buildings |
| `unit_type_of(tag)` | unit type id | unit struct `unit_id` field |
| `can_place(...)` | green/red placement grid | buildable tiles + collision + depot distance (v1 approx, see gaps) |
| `depot_spot_ok(tx, ty)` | map knowledge for base finding | `gary_env_depot_spot_ok`'s algorithm over the game's tile flags |
| `start_locations()` | map knowledge | `Game.start_position` |
| `set_name` / `save_replay` | — | not applicable: the client writes its own replay (see gaps) |

`observe()` JSON — byte-for-byte the schema of `gary_env_observe`:

```json
{"frame": 1234,
 "map": {"w": 4096, "h": 3072},
 "players": [{"slot": 0, "name": "...", "race": 1, "minerals": 50, "gas": 0,
              "supply_used": 4, "supply_max": 10, "victory_state": 0}],
 "units": [{"tag": 1031, "owner": 0, "type": 7, "x": 111, "y": 222,
            "hp": 40, "shields": 0, "completed": 1, "visible_to": 255,
            "order": 2, "resources": 0, "queue": 0}]}
```

`tag` is a **16-bit opaque handle** exactly like `gary_env`'s (OpenBW `unit_id` layout:
`index+1 | generation<<11`), so `gary/commands.py`'s `<H>` fields work unchanged. Under the hood
the adapter translates tags ↔ SC:R's extended unit ids (`index1 | gen << (13 or 11)`; 3400-unit

## Commands: legacy → 1.21 translation (the Pluto control layer, reimplemented)

Gary emits legacy replay-format commands (`gary/commands.py`). Remastered 1.21+ has variants of
exactly **six** commands with 32-bit unit ids (an extra zero u16 after each id in the wire
format); everything else passes through unchanged. Facts from `resim/gary_resim.cpp` (which
translates the reverse direction) and `external/screp/rep/repcmd/types.go` (which names both
families):

| legacy (Gary) | 1.21 (SC:R queue) | change |
|---|---|---|
| `0x09` select | `0x63` | each unit tag u16 → SC:R id u32 |
| `0x0a` select-add | `0x64` | same |
| `0x0b` select-remove | `0x65` | same |
| `0x14` right-click | `0x60` | target tag u16 → u32 |
| `0x15` targeted order | `0x61` | target tag u16 → u32 |
| `0x29` unload | `0x62` | unit tag u16 → u32 |
| everything else | unchanged | e.g. `0x0c` build, `0x1f` train, `0x23` morph |

Every translated packet is appended to `logs/scr_commands.jsonl` as
`{"command_frame", "queued_frame", "packet_hex", "submitted"}` — a small piece of the Trace Bus
and the ground truth for the replay cross-check (the same commands must appear in the `.rep`
SC:R saves, parseable with `tools/bin/screp.exe`).

**Latency.** Commands enter the game's turn queue and land a few frames later. v0 submits at the
next completed frame and records `queued_frame`; the measured observation→effect delay is then
fed to `gary/interface.py`'s pending/reaction model. (The Pluto bridge measured 4 frames against
its training environment and held packets one frame to match; SC:R multiplayer estimates
`latencyFrames = ceil(24·(2 + user_delay) / turn_rate)` from `0x1240e58+44` / `0x1241288` —
both recorded in `scr_profile.h` as calibration inputs. Nothing is assumed; the adapter
measures and reports.)


## File map

| File | Role |
|---|---|
| `src/scr_profile.h` | Address/struct facts for build 1.23.10.13515 x86 + SHA-256 pins, with provenance and verification status |
| `src/commands.*` | legacy→1.21 translation + legacy record lengths (screp / `gary/commands.py`) |
| `src/handles.*` | 16-bit Gary tag ↔ SC:R unit pointer/id table (generation-tagged, stale-safe) |
| `src/observe.*` | struct reads → Gary's JSON; `unit_at`/`box_select`/`can_place`/`depot_spot_ok`/`start_locations` |
| `src/bridge.*` | DllMain, verify, frame watcher, outbox drain → `send_command` |
| `src/pipe_server.*` | named pipe, newline-delimited JSON request/response |
| `src/tests_main.cpp` | offline unit tests: translation golden vectors, handle table, framing |
| `tools/verify_profile.py` | offline: hash + PE sanity of `scr_profile.h` facts vs an installed exe |
| `tools/auto_game.py` | hands-free menu walk (posted keys) + lobby race picker; `--preset lan-create`/`lan-join`; screenshot-evidenced |
| `tools/close_mutex.py` | close SC:R's single-instance Event handle so a second client can run |
| `tools/lan_park.py` | start + inject Gary, park two clients keyboard-only: host at Create Game, guest at the LAN list |
| `tools/run_match.py`, `probe_live.py`, `end_match.py` | live probe pipeline: launch → melee → checklist → end match with replay |
| `launch.py` | find SC:R, verify, write config, inject DLL, optional smoke test |
| `gary/scr_env.py` | `ScrGame`: Gary's `Game` surface over the pipe |

## Build and run

Requirements: Visual Studio C++ tools (same as `env/`), Python 3. Nothing else — no Rust, no
MinHook, no BWAPI.

```bat
adapters\scr_bridge\build.bat              &rem builds build\gary_scr.dll + build\gary_scr_tests.exe (32-bit!)
adapters\scr_bridge\build\gary_scr_tests.exe                          &rem offline unit tests
python adapters\scr_bridge\tools\verify_profile.py "%GARY_SC_INSTALL%\x86\StarCraft.exe"
python adapters\scr_bridge\launch.py --smoke                          &rem inject, ping/status/observe, exit
```

The DLL is **32-bit** (`build.bat` uses `vcvars32`) because the pinned target is the x86 client
(`x86\StarCraft.exe`) that the Pluto bridge also targets. The 64-bit client is a later step
(needs its own profile; `samase_scarf` discovery is the patch-survivable route).

Playing: `python adapters\scr_bridge\launch.py` starts SC:R with the bridge; then run a bot with
`--scr` (planned) or attach later — Gary only ever acts as the local player. Starting a game is
hands-free: `tools/auto_game.py` types the menu hotkeys (`S,E,O,U,O`) as **posted background
keys** — no clicks, no focus steal (verified 2026-10-03: a melee match started end-to-end with
the game unfocused; `PrintWindow` state shots land in `build/auto_game/`).

Human-vs-Gary on one desktop is the same idea over LAN. `auto_game.py --preset lan-create
--pid <bot pid>` posts `M,E,Down,O,O,G,O` (Multiplayer, Expansion, LAN — Down from the
preselected Battle.net — Ok, Ok on the first registry account, G=Create, Ok on the preselected
map) and lands in the lobby; `--preset lan-join --pid <human pid>` posts `M,E,Down,O,O,O` on
the second client to join it (the create page defaults to Top vs Bottom on Bottleneck, a fine
1v1). Then `--race T --row home` (bot) and `--race Z --row away` (human) pick races — SC:R
widgets ignore posted mouse messages, so these two clicks use real input and briefly raise the
bot's window (harmless: multiplayer never pauses unfocused) — and `--start` posts Alt+O for the
5s countdown. **Note the lobby semantics:** the bridge reports `in_game=true` while sitting in
the lobby (map size + local player are already set there), so `frames > 0` is the "match
running" signal (`auto_game.match_started()`), not `in_game`.

**Two instances on one machine** (needed for local human-vs-Gary practice): SC:R enforces a
single instance via the named kernel Event `\Sessions\1\BaseNamedObjects\Starcraft Check For
Other Instances` — a second client exits within seconds of the first (the x86 and x64 clients
share the check). `tools/close_mutex.py` closes that handle inside the running instance — the
automated Process Explorer step (`NtQuerySystemInformation` + `DuplicateHandle(
DUPLICATE_CLOSE_SOURCE)`; `--list`/`--all` for diagnosis) — and works **without elevation**
when the client runs as the same user. Verified 2026-10-03: two clients up, a LAN game created
in one and joined from the other, both clients running unfocused (LAN matches do not pause).
With two instances, `auto_game.py --pid N` / `end_match.py --pid N` target one window.

**x86 client only.** The pinned build facts are for `x86\StarCraft.exe`; a 32-bit DLL cannot
load into the 64-bit client, and the address profile would differ anyway. The x86 client plays
the same game against the same opponents (including x64 clients over Battle.net).



## Verification plan (what makes this adapter trustworthy)

1. **Offline (no game needed):** `gary_scr_tests.exe` — command translation golden vectors (the
   inverse of `resim/gary_resim.cpp`'s tested translator), handle-table stale-generation
   behavior, record lengths. `tests/test_scr_bridge.py` — protocol framing, `ScrGame` method
   parity with `gary.env.Game`, SHA-256 check of the pin against the real
   `%GARY_SC_INSTALL%\x86\StarCraft.exe` (`GARY_SC_INSTALL`: your StarCraft folder).
2. **Binary-level (install only, no game run):** `verify_profile.py` — PE parse, hash pin,
   RVA→file-offset mapping, code-site sanity bytes.
3. **Live probe (one custom game vs computer):** `tools/run_match.py` runs the whole loop
   (launch → inject → menu-click a melee game → `tools/probe_live.py`). **Executed 2026-10-03
   against 1.23.10.13515 x86 — every check passes:** exactly 2 players in a 1v1, 4 workers + a
   main building, mineral fields read real amounts (`raw_resources` == `observe.resources` ==
   1500 at `kUnitResources = 0xd0`, which replaces the earlier 204 — a window-start off-by-4
   artifact), frames advance while playing, and an `act()` select lands in the saved replay as
   `Select` **ID 99 (0x63)** for our player with the exact unit tag we sent (`screp.exe -cmds`:
   Frame 3527, PlayerID 0, UnitTags [11525]; the replay's `Frames` equals the bridge's final
   count, 9793). The shields auto-hunt found **`kUnitShields = 0x60`** uniquely across 21
   protoss units; the check re-verifies on any game where the (random) computer race is Protoss.
   End the match with `tools/end_match.py` to have the client write the replay.
4. **Cross-backend (the real test):** run the same scripted bot on OpenBW (`gary/env.py`) and on
   SC:R and decision-diff the traces (the harness planned for §7.2 desync checks).

## Known gaps (v0)

- The unit fields added to the observation on 2026-10-15 so it matches `gary_env_observe` (energy,
  ground weapon cooldown, the order's target unit and point, carrying flags, players who detect it;
  `hp_max` comes from units.dat) sit where BW 1.16's unit struct has them (`scr_profile.h`, INFERRED:
  every pinned field so far matched 1.16, and OpenBW's unit_t keeps the same order). Energy moved
  from 168 to 162 and the queue ring start from 166 to 164. Pin them in a live game before its
  results count: energy decides Comsat scans (the TODO in `scr_profile.h` says how). `visible_to`
  now hides cloaked and burrowed units from players who don't detect them, as gary_env does.
- `can_place` is an approximation (buildable tiles + collision + depot distance): the fog rule
  ("unexplored tiles not allowed") and creep/pylon-power rules need the game's own placement
  function (resolve via `samase_scarf` later) to match `gary_env_can_place` exactly.
- `unit_at`/`box_select` resolve clicks through the sprite's image chain: the union of the
  clickable images' GRP frames (gary_env's model; frame geometry from the game's own GRP files via
  tools/gen_image_dat.py, struct offsets pinned by tools/hunt_click.py against live memory).
  Residual gaps: sprites whose body image hangs off differently-encoded links fall back to the
  sprite bounding box (`hunt_click.py --check` measures the split), and the flip bit uses OpenBW's
  flag value pending a facing-left sample. Click-model fidelity between training (OpenBW) and live
  (SC:R) must be measured before G2 numbers are trusted (`hunt_click.py --gold` prints gary_env's
  reference boxes per unit type).
- `set_name`/`save_replay` are not implemented (the client writes its own replay on match end;
  `tools/end_match.py` ends a match through the F10 menu and locates it: `Maps\Replays\LastReplay.rep`
  under the StarCraft folder in the user's Documents, wherever Documents points).
- Observation reads memory on demand (pipe thread) rather than a frame-stamped snapshot; v1
  moves state capture onto the frame watcher tick like Shieldbattery does.
- `act()` timing is "next frame boundary"; observed effect latency is logged and must be
  calibrated into `gary/interface.py` before human-likeness evaluation.
- Control groups are real 0x13 commands now (2026-10-15: the interface sends assign / recall / add,
  a headless replay shows them, and the game is the same as before), and `act()` passes them, with
  every other command Gary encodes that carries no unit tag (stop, hold, stim, return cargo, unload
  all, burrow, research, upgrade, building morph, chat: research and Stim had been refused). Verify
  in a live game that SC:R applies them (its replay should show `Hotkey` commands).
- Observation fields Gary v0.4's fight model reads per unit, now in `gary_env_observe`:
  `cooldown` (ground weapon cooldown), `order_target` (the order's target unit tag, 0 if none),
  `carrying` (worker carrying flags). Live they default to 0, which the fight model was not
  trained on; add them to the bridge's unit records. Also: shields for units without shields
  should be 0.
- `depot_spot_ok` disagreed with `gary_env_depot_spot_ok` by a tile (the natural's town hall at
  (37, 112) live, only (36, 112) placeable: "Gary v03 got owned 01", 2026-10-04). Fixed 2026-10-15:
  a resource blocks the spot when its box (units.dat dimensions), not its center, reaches the 3-tile
  margin, as OpenBW's `find_units_noexpand` finds it. On every terrain-valid spot within 6 tiles of
  every base of Fighting Spirit, Python and Heartbreak Ridge (5,927) the rule now agrees with
  gary_env; the center rule disagreed on 392. Confirm live on one natural.
- The headless env now delays every command by `gary.env.LIVE_COMMAND_DELAY` (3 frames: the
  reported turn latency 2 + 1 hand-off). Re-measure it once a live game sends real hotkeys, and
  keep the two in step if the latency setting changes.
- x86 client only; single local player only. Starting custom games hands-free is automated
  (`tools/auto_game.py --mode keys` types the menu hotkeys as posted background keys; the
  calibrated screenshot+click flow remains as `--mode clicks`); joining arbitrary lobbies is not.
  SC:R menu facts encoded in the tool: the client restores its last menu screen on launch, only
  text labels are buttons, the single-player menu's "Cancel" and ESC at the top-level menu both
  quit the game, and the window flips windowed/fullscreen while loading (only the click flow
  cares — it normalizes to fullscreen, the calibrated layout). Menu hotkey letters (S=ingle
  player, E=xpansion, O=K, cU=stom game) work via posted keys on an unfocused window (verified
  2026-10-03: `S,E,O,U,O` started a melee match end-to-end with the game in the background), so
  the walk needs no visible screen. The client pauses while unfocused in offline games;
  multiplayer (Local PC) matches cannot pause — confirm once the two-instance setup exists.

- **Play offline only — never the Battle.net ladder** (`CONTRIBUTING.md` ground rules).

Command packets are built in the command-table mode per `resim/gary_resim.cpp`'s inverse
(legacy→1.21 translation, golden-vector tested in `gary_scr_tests.exe`).

`act()` accepts only the **local player's** slot (a live client can only act as itself); other
slots return "rejected", matching `gary_env_act`'s semantics (1 accepted / 0 rejected / -1 error).

| `neivv/scarf`, `samase_scarf`, `bw_dat` | none declared upstream | Facts/offsets only; do not vendor without asking |
| Address + struct facts (RVAs, offsets) | facts about the 1.23.10.13515 binary | Recorded in `scr_profile.h` with provenance; verified by hash pin + probe |

Address/offset *facts* are interoperability data (like API signatures), not creative expression;
we record them with provenance and verify them mechanically against the user's own binary.
