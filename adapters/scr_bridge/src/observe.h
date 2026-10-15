#pragma once
// Reading the live game: Gary's observe() JSON and the click-model queries, straight from the
// SC:R structs laid out in scr_profile.h. All functions are called on the pipe thread; they only
// read memory (Shieldbattery's rule: never touch game state from outside the game thread — v1
// accepts the small race of reading mid-frame and reports the frame each observation was taken
// on; a frame-stamped snapshot in the hook is the v1 fix, README "Known gaps").
//
// The JSON schema is byte-for-byte gary_env_observe's (env/gary_env.cpp), so the JSON schema is byte-for-byte gary_env_observe's (env/gary_env.cpp), so
// gary/interface.py works unchanged.

#include <cstdint>
#include <string>
#include <vector>

#include "handles.h"

namespace garyscr {

struct World {  // resolved base addresses for one frame of reads (filled by bridge.cpp)
    uintptr_t module_base = 0;
    uint32_t game = 0;      // Game struct
    uint32_t players = 0;   // Player struct array
    uint32_t units_base = 0;
    uint32_t unit_count = 0;
    uint32_t local_player = 0;
    bool in_game = false;   // sane map size + local player
    std::string warning;    // profile warnings to surface in status
};

// Resolve the world from the profile (fills .warning with anything suspect). Returns false if
// the game is not in a state we can read (menu, loading).
bool world_read(World* w, std::string* error);

// One live unit's raw fields (also used by the probe).
struct UnitView {
    UnitRef ref;
    uint8_t owner = 0;
    uint16_t type = 0;
    int x = 0, y = 0;
    int hp = 0;
    int shields = 0;
    bool completed = false;
    uint8_t visible_to = 0;
    uint8_t order = 0;
    int resources = 0;
    int energy = 0;          // integer part
    int cooldown = 0;        // ground weapon cooldown
    uint32_t order_target_ptr = 0;
    int tx = 0, ty = 0;      // the order's target point
    int carrying = 0;
    uint32_t detected = 0;
    int queue = 0;
    uint8_t elevation = 0;
    int sprite = 0;         // sprite address, 0 = dying
    uint32_t flags = 0;
    // probe raw values for the UNVERIFIED offsets
    uint32_t raw_shields = 0;
    uint16_t raw_energy = 0;
    uint16_t raw_queue[5] = {};
    uint8_t raw_build_slot = 0;
    uint16_t raw_resources = 0;
    uint16_t raw_type_at_36 = 0;  // flingy_id, for the type-offset check
    uint8_t raw_window[192] = {};  // profile::kProbeWindowLen bytes from unit+kProbeWindowStart
};

// Collect every unit in the active + hidden lists (guarded against corrupt lists).
bool units_collect(const World& w, std::vector<UnitView>* out, std::string* error);

// resolve a Gary tag to a live unit; returns false when stale/absent.
bool unit_by_tag(const World& w, uint16_t tag, UnitView* out);

// Gary's observe() JSON (schema of gary_env_observe).
bool observe_json(const World& w, std::string* out, std::string* error);

// The click model (v1: sprite bounding box + draw depth, see README "Known gaps").
uint16_t unit_at(const World& w, int slot, int x, int y);
std::vector<uint16_t> box_select(const World& w, int slot, int x0, int y0, int x1, int y1);

// Placement checks (v1 approximation; README "Known gaps").
bool can_place(const World& w, int slot, uint16_t builder_tag, int unit_type, int tile_x,
               int tile_y);
bool depot_spot_ok(const World& w, int tile_x, int tile_y);
int tile_flags_raw(const World& w, int tile_x, int tile_y);
// Bounded raw-memory hex read for pinning struct offsets by measurement (hunt tools).
bool peek_hex(uint32_t addr, int len, std::string* out);
bool start_locations_json(const World& w, std::string* out, std::string* error);

// Raw values at the probe offsets for one unit (README verification step 3).
bool probe_unit_json(const World& w, uint16_t tag, std::string* out, std::string* error);

// Placement size in tiles for the buildings Gary places (tools/gen_unit_dat.py generates
// unit_dat.h from data/gamedata/scr/arr/units.dat; values are the game's own DAT).
bool placement_size(int unit_type, int* w_tiles, int* h_tiles);

}  // namespace garyscr
