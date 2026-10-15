#pragma once
// scr_profile: every address and struct-layout fact the adapter needs about one specific
// StarCraft: Remastered build. See ../README.md "License provenance": these are interoperability
// facts about the 1.23.10.13515 x86 binary, gathered from public sources and verified against
// the installed executable by tools/verify_profile.py and by runtime checks in bridge.cpp.
//
// Address convention (fact, from the analysis of this build): stored values are absolute VAs as
// seen when the module is loaded at kAnalyzedBase; RVA = va - kAnalyzedBase; at runtime
// addr(va) = module_base + (va - kAnalyzedBase).
//
// Provenance codes in comments:
//   [SB]    ShieldBattery (MIT) source
//   [PL]    Pluto-AI-Starcraft-Remaster source (no license; facts only)
//   [SCE]   external/screp (this repo)
//   [GARY]  this repo (resim/gary_resim.cpp, env/gary_env.cpp)
//   [BWD]   neivv/bw_dat struct definitions
//   [BWP]   BWAPI BWGame.h offset asserts (public)
// Status codes:
//   VERIFIED  >= 2 independent sources agree (and/or runtime-validated in bridge.cpp)
//   INFERRED  1 source or arithmetic from verified anchors
//   UNVERIFIED needs the live probe (README "Verification plan" step 3)

#include <cstdint>

namespace profile {

// --- build pin -------------------------------------------------------------------
constexpr char kBuildName[] = "StarCraft 1.23.10.13515 x86";
// SHA-256 of x86\StarCraft.exe for the pinned build (measured on the target machine).
constexpr char kExeSha256[] = "32DBBDD001DD381CB1B3A719B7AD1FC918A9D4BC99661C675E00254EFECCA827";
constexpr uint32_t kAnalyzedBase = 0x4a0000;  // [PL] base the VA facts below were observed at

// --- code and import table (VA) ---------------------------------------------------
constexpr uint32_t kSendCommand = 0x007da250;      // VERIFIED [PL]; void __cdecl(const uint8_t*, size_t)
constexpr uint32_t kPrintText = 0x007b3200;        // VERIFIED [PL]; void __cdecl(const char*, u32, u32)
constexpr uint32_t kFrameAfter = 0x0077d112;       // [PL] call site fact (unused by v1: the frame
                                                   // watcher needs no call sites)
constexpr uint32_t kFrameBefore = 0x0076e02f;      // [PL] call site fact (unused by v1)
constexpr uint32_t kTimingTickPtr = 0x00dd01c4;   // [PL]: the game's cached GetTickCount pointer
                                                   // (unused by v1; it is NOT a loader IAT slot)

// --- global pointers (VA of the pointer variable; value read at runtime) ------------
constexpr uint32_t kGameXorPtr = 0x00fa63cc;       // VERIFIED [PL]: game() = read32(this) ^ 0x10fae560
constexpr uint32_t kGameXorConst = 0x10fae560;     // VERIFIED [PL]
constexpr uint32_t kPlayersSubPtr = 0x0101f0e4;    // VERIFIED [PL]: players() =
constexpr uint32_t kPlayersXorPtr = 0x01240e38;    // VERIFIED [PL]:   (0xfb5969c2 - read32(sub)) ^ read32(this)
constexpr uint32_t kPlayersConst = 0xfb5969c2;     // VERIFIED [PL]
constexpr uint32_t kFirstActiveUnit = 0x010436a4;  // VERIFIED [PL] head of the active unit list
constexpr uint32_t kFirstHiddenUnit = 0x010436b4;  // VERIFIED [PL] head of the hidden unit list
constexpr uint32_t kUnitsBase = 0x010436f0;        // VERIFIED [PL] unit vector base pointer
constexpr uint32_t kUnitCount = 0x010436f4;        // VERIFIED [PL] unit vector length (1700 or 3400)
constexpr uint32_t kLocalPlayerId = 0x00fcacf4;    // VERIFIED [PL]
constexpr uint32_t kCommandUser = 0x00fcace8;      // VERIFIED [PL]
constexpr uint32_t kIsReplay = 0x01244580;         // VERIFIED [PL]
constexpr uint32_t kIsMultiplayer = 0x0123f7bc;    // VERIFIED [PL] (u8)
constexpr uint32_t kIsPaused = 0x0104a1cc;         // VERIFIED [PL]
constexpr uint32_t kScreenX = 0x01085b88;          // VERIFIED [PL] camera, for the camera logger later
constexpr uint32_t kScreenY = 0x01085b8c;          // VERIFIED [PL]
constexpr uint32_t kMapTileFlags = 0x01049478;     // VERIFIED [PL] tile flags array pointer
constexpr uint32_t kGameTypeTable = 0x01240e58;    // VERIFIED [PL] +40 game type u16, +44 turn rate u32
constexpr uint32_t kUserDelay = 0x01241288;        // VERIFIED [PL] multiplayer latency input

// --- Game struct (layout) ---------------------------------------------------------
// [BWD] bw_dat Game struct; [BWP] BWGame.h OFFSET_ASSERTs; [PL] scr_layout.h agree on all.
constexpr uint32_t kGameSize = 96000;              // VERIFIED [BWD][BWP]
constexpr uint32_t kGameMinerals = 0;              // VERIFIED s32[12]
constexpr uint32_t kGameGas = 48;                  // VERIFIED s32[12]
constexpr uint32_t kGameMapWidthTiles = 228;       // VERIFIED u16
constexpr uint32_t kGameMapHeightTiles = 230;      // VERIFIED u16
constexpr uint32_t kGameFrameCount = 332;          // VERIFIED u32
constexpr uint32_t kGameSupplies = 12372;          // VERIFIED Supplies[3] (race-major), each 144
constexpr uint32_t kGameVictoryState = 58896;      // VERIFIED u8[8]
constexpr uint32_t kGameStartPosition = 58928;     // VERIFIED [u16 x,y][8]; pixel/tile semantics UNVERIFIED
constexpr uint32_t kSuppliesSize = 144;            // VERIFIED: provided[12]@0, used[12]@48, max[12]@96
constexpr uint32_t kSuppliesProvided = 0;          // VERIFIED (supply available from depots -> gary_env "supply_max")
constexpr uint32_t kSuppliesUsed = 48;             // VERIFIED (u32; value is half-supply, like gary_env)
constexpr uint32_t kSuppliesMax = 96;              // VERIFIED (the 200-cap, not the "supply_max" of the JSON)

// --- Player struct (layout) -------------------------------------------------------
constexpr uint32_t kPlayerSize = 36;               // VERIFIED [PL][BWD]
constexpr uint32_t kPlayerType = 8;                // VERIFIED u8 (active-slot test is probe-verified)
constexpr uint32_t kPlayerRace = 9;                // VERIFIED u8 0=zerg 1=terran 2=protoss
constexpr uint32_t kPlayerName = 11;               // VERIFIED char[25]

// --- Unit struct (layout) ---------------------------------------------------------
constexpr uint32_t kUnitSize = 336;                // VERIFIED [PL][BWD] (0x150)
constexpr uint32_t kUnitNext = 4;                  // VERIFIED flingy.next (list walk) [BWD]
constexpr uint32_t kUnitHp = 8;                    // VERIFIED u32 24.8 fixed [BWD] flingy.hitpoints
constexpr uint32_t kUnitSprite = 12;               // VERIFIED ptr; 0 = dying (liveness test) [PL]
constexpr uint32_t kUnitTypeId = 100;              // INFERRED u16 (unit type < 228 test) [PL][BWD] naming
constexpr uint32_t kUnitPlayer = 76;               // VERIFIED u8 [PL][BWD]
constexpr uint32_t kUnitOrder = 77;                // VERIFIED u8 [PL][BWD]
constexpr uint32_t kUnitFlags = 220;               // VERIFIED u32 status flags [PL][BWD]
constexpr uint32_t kUnitGenIndex = 165;            // VERIFIED u8 generation (unit id identity) [PL][BWD]
constexpr uint32_t kUnitPosition = 40;             // VERIFIED u16 x, u16 y [BWD] flingy.position
// The following are INFERRED/UNVERIFIED; probe_unit reports raw values for them (README step 3).
constexpr uint32_t kUnitShields = 0x60;            // 96 VERIFIED (probe_live auto-hunt: the unique
                                                   // offset where u32>>8 == type max shields across
                                                   // 21 protoss units; matches [BWD] 0x60)
constexpr uint32_t kUnitBuildQueue = 152;          // VERIFIED u16[5] (hunt_queue.py: a train turns
                                                   // the empty marker 228 into the unit type id)
constexpr uint32_t kUnitBuildSlot = 164;           // UNVERIFIED u8 (raw display only: observe()'s
                                                   // queue count does not depend on it)
constexpr uint32_t kUnitEnergy = 162;              // INFERRED u16 8.8 fixed (0xA2: right after the five
                                                    // build-queue u16s at 152, BW 1.16's layout; was 168)
// TODO(pin): kUnitEnergy and kUnitBuildSlot now sit where BW 1.16 has them (0xa2 = 162, 0xa4 = 164;
// were 168/166), as do the other INFERRED fields: the verified ones around them (shields 0x60,
// build queue 152, resources 0xd0, status flags 0xdc) all match 1.16's unit struct. Energy is in
// the Gary contract now (Comsat scans need 50), so pin it before live games count. Pin like kUnitShields: probe_unit a unit
// whose energy is known and changing (a Comsat Station after a scan, a Medic or Science Vessel
// regenerating) and look for the u16 that tracks it (24.8 fixed: value >> 8 = energy); for the
// slot, train 2+ units and watch which u8 steps as the queue advances (tools/hunt_queue.py).
constexpr uint32_t kUnitGroundCooldown = 0x55;      // 85 INFERRED u8 (BW 1.16: after main order timer
                                                    // 0x54; OpenBW unit_t: owner, order, order state,
                                                    // order unit type, timer, ground weapon cooldown)
constexpr uint32_t kUnitOrderTargetPos = 0x58;      // 88 INFERRED u16 x, u16 y (the order's target point)
constexpr uint32_t kUnitOrderTargetUnit = 0x5c;     // 92 INFERRED ptr (the order's target unit, 0 if none)
constexpr uint32_t kUnitCarrying = 0xe0;            // 224 INFERRED u8 carrying flags (1 gas, 2 minerals:
                                                    // OpenBW carrying_flags, right after status flags 0xdc)
constexpr uint32_t kUnitDetected = 0xe4;            // 228 INFERRED u32 players who detect it (OpenBW
                                                    // detected_flags; 0x80000000 = not computed yet)
constexpr uint32_t kStatusRequiresDetector = 0x100;  // OpenBW game_types.h (burrowed units)
constexpr uint32_t kStatusCloaked = 0x200;
constexpr uint32_t kUnitResources = 0xd0;          // 208 VERIFIED u16: window dump of a 1500-mineral shows
                                                   // 0x05dc at 208 (iscript=20 at 210, OpenBW's
                                                   // resource { count, iscript } order). Earlier 204
                                                   // was a window-start off-by-4 artifact; 0 on a drone

// The raw window probe_unit dumps when hunting offsets: unit-struct bytes [start, start+len).
constexpr uint32_t kProbeWindowStart = 48;
constexpr uint32_t kProbeWindowLen = 192;

// status-flag bits: values from OpenBW's enum, which mirrors BW's bit values [GARY/openbw]
constexpr uint32_t kStatusCompleted = 0x1;
constexpr uint32_t kStatusGroundedBuilding = 0x2;
constexpr uint32_t kStatusDisabled = 0x400;

// --- Sprite struct (layout) -------------------------------------------------------
constexpr uint32_t kSpriteSize = 40;               // VERIFIED [PL][BWD] (0x28)
constexpr uint32_t kSpriteVisibility = 12;         // VERIFIED u8 mask (bit p = visible to p) [PL][GARY]
constexpr uint32_t kSpriteElevation = 13;          // VERIFIED u8 (draw depth) [PL]
constexpr uint32_t kSpriteWidth = 18;              // VERIFIED u8 (bbox approx, README gaps) [PL]
constexpr uint32_t kSpriteHeight = 19;             // VERIFIED u8 [PL]

// --- Image struct (layout; PINNED by measurement 2026-10-04) ----------------------
// tools/hunt_click.py --dump walked live sprite/image records and validated them against the
// GRP files' frame tables (gen_image_dat.py): the chain head is the sprite's first word, each
// image's type/frame pair parsed consistently, and the body/shadow records' raw bytes pinned
// x, y and the flags byte (bit values follow OpenBW image_t::flags, like the tile flags did).
constexpr uint32_t kSpriteImageHead = 0;           // VERIFIED ptr: first CImage (0 = none)
constexpr uint32_t kImageNext = 0;                 // VERIFIED ptr: next image in the sprite
constexpr uint32_t kImageType = 8;                 // VERIFIED u16 (images.dat row)
constexpr uint32_t kImageFrameIndex = 10;          // VERIFIED u16 (frame in the type's GRP)
constexpr uint32_t kImageX = 12;                   // VERIFIED s8 (offset from sprite position)
constexpr uint32_t kImageY = 13;                   // VERIFIED s8
constexpr uint32_t kImageFlags = 18;               // VERIFIED u8
constexpr uint8_t kImageFlagFlipped = 0x02;
constexpr uint8_t kImageFlagClickable = 0x20;

// --- command / unit-id facts (not addresses) ---------------------------------------
// SC:R extended unit id = index1 | (generation << shift); shift 13 when the unit vector is
// larger than 1700 (LMTS 3400-unit tables), else 11. [PL] unit_handle(); [GARY] resim inverse.
constexpr uint32_t kUnitIdShiftLarge = 13;
constexpr uint32_t kUnitIdShiftSmall = 11;
constexpr uint32_t kClassicUnitLimit = 1700;

}  // namespace profile

constexpr uint32_t kIsPaused = 0x0104a1cc;         // VERIFIED [PL]
constexpr uint32_t kScreenX = 0x01085b88;          // VERIFIED [PL] camera, for the camera logger later
constexpr uint32_t kScreenY = 0x01085b8c;          // VERIFIED [PL]
constexpr uint32_t kMapTileFlags = 0x01049478;     // VERIFIED [PL] tile flags array pointer
constexpr uint32_t kGameTypeTable = 0x01240e58;    // VERIFIED [PL] +40 game type u16, +44 turn rate u32
constexpr uint32_t kUserDelay = 0x01241288;        // VERIFIED [PL] multiplayer latency input
