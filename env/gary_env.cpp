// gary_env: a headless Brood War game (OpenBW) that Python code can play, through a small C API.
//
// Start a game on a map, or on a replay's map with that replay's players and races, or from a
// replay that keeps playing until someone takes over a side (a scenario); step it frame by frame, send each player's commands in the
// replay command format, read the full game state as JSON, and save the game as a replay.
//
// Players' commands use the same bytes as replay commands (select = 0x09, right click = 0x14,
// train = 0x1f, build = 0x0c, ...), which is also what Gary's human interface will emit.
//
// The state returned by gary_env_observe is the full, unfiltered game state. Fog of war is
// applied by the caller using each unit's "visible_to" bitmask (bit p = visible to player p).

#include <stdexcept>  // OpenBW's util.h uses std::runtime_error without including it

#include "bwgame.h"
#include "replay.h"
#include "replay_saver.h"
#include "replay_stream.h"
#include "scr_replay.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <algorithm>
#include <vector>

#ifdef _WIN32
#define GARY_API extern "C" __declspec(dllexport)
#else
#define GARY_API extern "C"
#endif

using namespace bwgame;

namespace {

bool file_exists(const std::string& path) {
	std::ifstream f(path, std::ios::binary);
	return f.good();
}

// Same loader as gary_resim: classic MPQs if present, else extracted loose files.
struct data_loader {
	std::optional<data_loading::data_files_loader<>> mpqs;
	std::string dir;
	explicit data_loader(std::string d) : dir(std::move(d)) {
		if (!dir.empty() && dir.back() != '/' && dir.back() != '\\') dir += '/';
		if (file_exists(dir + "StarDat.mpq")) mpqs.emplace(data_loading::data_files_directory(dir.c_str()));
	}
	void operator()(a_vector<uint8_t>& dst, a_string filename) {
		if (mpqs) {
			(*mpqs)(dst, std::move(filename));
			return;
		}
		std::string path = dir;
		for (char c : filename) path += (c == '\\') ? '/' : c;
		std::ifstream f(path, std::ios::binary | std::ios::ate);
		if (!f) error("data file not found: %s", path.c_str());
		dst.resize((size_t)f.tellg());
		f.seekg(0);
		f.read((char*)dst.data(), (std::streamsize)dst.size());
	}
};

// A saved moment of a game (gary_env_save / gary_env_restore): OpenBW's own exact state copy, as
// the viewer uses for seeking, plus where the replay's commands and the saved replay stand.
struct saved_game {
	state st;
	action_state action_st;
	size_t actions_data_position = 0;
	int next_action_frame = -1;
	replay_saver_state saver_st;
	uint32_t dropped_owners = 0;
	int dropped_as = 7;
};

struct env {
	game_player player;
	std::unique_ptr<saved_game> saved;
	action_state action_st;
	replay_state replay_st;
	std::optional<scr_replay<replay_functions>> funcs;  // (Remastered-era commands: scr_replay.h)
	std::vector<uint8_t> map_data;
	std::array<uint8_t, 633> header{};  // the source replay's game info, reused when saving
	replay_saver_state saver_st;
	std::string observation;
	std::string last_error;

	explicit env(const std::string& data_dir) : player(data_loader(data_dir)) {}
};

std::string json_escape(const a_string& s) {
	std::string r;
	for (unsigned char c : s) {
		if (c == '"' || c == '\\') { r += '\\'; r += (char)c; }
		else if (c < 0x20) { char buf[8]; snprintf(buf, sizeof buf, "\\u%04x", c); r += buf; }
		else r += (char)c;
	}
	return r;
}

}  // namespace

static std::string g_create_error;

namespace {

// Loads a replay's decoded stream (replay_stream.h). keep_commands: the replay's commands play
// as the game steps (a scenario); otherwise only its map, players and races are used.
std::unique_ptr<env> load_stream(const char* data_dir, std::vector<uint8_t> stream, bool keep_commands, int unit_limit) {
	auto e = std::make_unique<env>(data_dir);
	disable_cosmetic_trigger_actions(stream);
	e->funcs.emplace(e->player.st(), e->action_st, e->replay_st);
	e->funcs->unit_limit = unit_limit;
	e->funcs->load_replay(data_loading::data_reader_le(stream.data(), stream.data() + stream.size()));
	e->map_data = stream_map_data(stream);
	memcpy(e->header.data(), stream.data() + 4, e->header.size());
	if (!keep_commands) {
		e->replay_st.actions_data_buffer.clear();
		e->action_st.actions_data_position = 0;
		e->action_st.next_action_frame = -1;
	}
	e->replay_st.end_frame = 0x7fffffff;
	return e;
}

}  // namespace

// Creates a game on the map of the given (pre-1.18 format) replay file, with its players and
// races; the replay's own commands are not played. Returns null on error (see gary_env_error
// with a null handle for the message).
GARY_API void* gary_env_create(const char* data_dir, const char* replay_path) {
	try {
		std::ifstream in(replay_path, std::ios::binary);
		if (!in) error("can't open replay: %s", replay_path);
		std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
		return load_stream(data_dir, decode_legacy_replay(bytes), false, 1700).release();
	} catch (const std::exception& ex) {
		g_create_error = ex.what();
		return nullptr;
	}
}

// A scenario: the replay with its commands, which play as the game steps. Call
// gary_env_drop_commands for a player to take over that side from the current frame.
// data: the replay file (flat = 0), or a Remastered-era replay decoded by resim/scr_format.py's
// to_flat (flat = 1, with its unit_limit). Returns null on error.
GARY_API void* gary_env_create_scenario(const char* data_dir, const uint8_t* data, int size, int flat, int unit_limit) {
	try {
		std::vector<uint8_t> bytes(data, data + size);
		return load_stream(data_dir, flat ? std::move(bytes) : decode_legacy_replay(bytes), true, unit_limit).release();
	} catch (const std::exception& ex) {
		g_create_error = ex.what();
		return nullptr;
	}
}

// Remembers this moment of the game; gary_env_restore returns to it exactly (e.g. to play one
// scenario many times without reloading and replaying up to it). Returns false on error.
GARY_API bool gary_env_save(void* h) {
	auto* e = (env*)h;
	try {
		auto s = std::make_unique<saved_game>();
		s->st = copy_state(e->player.st());
		s->action_st = copy_state(e->action_st, e->player.st(), s->st);
		s->actions_data_position = e->action_st.actions_data_position;
		s->next_action_frame = e->action_st.next_action_frame;
		s->saver_st = e->saver_st;
		s->dropped_owners = e->funcs->dropped_owners;
		s->dropped_as = e->funcs->dropped_as;
		e->saved = std::move(s);
		return true;
	} catch (const std::exception& ex) {
		e->last_error = ex.what();
		return false;
	}
}

// Back to the moment gary_env_save remembered (it stays remembered). Returns false on error.
GARY_API bool gary_env_restore(void* h) {
	auto* e = (env*)h;
	try {
		if (!e->saved) error("gary_env_restore: nothing saved");
		const saved_game& s = *e->saved;
		e->player.st() = copy_state(s.st);
		e->action_st = copy_state(s.action_st, s.st, e->player.st());
		e->action_st.actions_data_position = s.actions_data_position;
		e->action_st.next_action_frame = s.next_action_frame;
		e->saver_st = s.saver_st;
		e->funcs->dropped_owners = s.dropped_owners;
		e->funcs->dropped_as = s.dropped_as;
		return true;
	} catch (const std::exception& ex) {
		e->last_error = ex.what();
		return false;
	}
}

// From now on, the replay's commands for this player are dropped (someone else plays that side).
GARY_API void gary_env_drop_commands(void* h, int slot) {
	auto* e = (env*)h;
	e->funcs->dropped_owners |= 1u << slot;
	const state& st = e->player.st();
	for (int i = 7; i >= 0; --i) {
		if (st.players[i].controller != player_t::controller_occupied) {
			e->funcs->dropped_as = i;
			return;
		}
	}
	// (no empty slot: impossible in 1v1; the default stays)
}

GARY_API const char* gary_env_error(void* h) {
	return h ? ((env*)h)->last_error.c_str() : g_create_error.c_str();
}

GARY_API void gary_env_destroy(void* h) { delete (env*)h; }

// Advances the game by n frames. Returns false on error.
GARY_API bool gary_env_step(void* h, int n) {
	auto* e = (env*)h;
	try {
		auto& f = *e->funcs;
		for (int i = 0; i != n; ++i) {
			// a scenario's recorded commands (none otherwise), saved with the game's own
			f.execute_actions_scr([&](int owner, int, const uint8_t*, const uint8_t*, bool) {
				replay_saver_functions(e->saver_st).add_action(e->player.st().current_frame, e->action_st.player_id[owner],
				                                               f.last_action.data() + 1, f.last_action.size() - 1);
			});
			f.state_functions::next_frame();
		}
		return true;
	} catch (const std::exception& ex) {
		e->last_error = ex.what();
		return false;
	}
}

// Executes one command for a player slot (bytes as in a replay, without the player id byte).
// Returns 1 if the engine accepted it, 0 if rejected, -1 on error.
GARY_API int gary_env_act(void* h, int slot, const uint8_t* data, int size) {
	auto* e = (env*)h;
	try {
		bool ok = e->funcs->read_action(slot, data, (size_t)size);
		// replays identify players by their player id, not their slot
		replay_saver_functions(e->saver_st).add_action(e->player.st().current_frame, e->action_st.player_id[slot], data, (size_t)size);
		return ok ? 1 : 0;
	} catch (const std::exception& ex) {
		e->last_error = ex.what();
		return -1;
	}
}

// The full game state as JSON (valid until the next call). Positions are in pixels.
GARY_API const char* gary_env_observe(void* h) {
	auto* e = (env*)h;
	const state& st = e->player.st();
	auto& f = *e->funcs;
	std::string& o = e->observation;
	o = "{\"frame\":" + std::to_string(st.current_frame) + ",\"map\":{\"w\":" +
	    std::to_string(st.game->map_tile_width * 32) + ",\"h\":" + std::to_string(st.game->map_tile_height * 32) +
	    "},\"players\":[";
	bool first = true;
	for (int p = 0; p != 8; ++p) {
		if (st.players[p].controller != player_t::controller_occupied) continue;
		if (!first) o += ',';
		first = false;
		int race = (int)st.players[p].race;
		int ri = race >= 0 && race < 3 ? race : 0;
		o += "{\"slot\":" + std::to_string(p) + ",\"name\":\"" + json_escape(e->replay_st.player_name[p]) +
		     "\",\"race\":" + std::to_string(race) + ",\"minerals\":" + std::to_string(st.current_minerals[p]) +
		     ",\"gas\":" + std::to_string(st.current_gas[p]) +
		     ",\"supply_used\":" + std::to_string(st.supply_used[p][ri].raw_value / 2.0) +
		     ",\"supply_max\":" + std::to_string(st.supply_available[p][ri].raw_value / 2.0) +
		     ",\"victory_state\":" + std::to_string(st.players[p].victory_state) + "}";
	}
	o += "],\"units\":[";
	first = true;
	for (const unit_t* u : ptr(st.visible_units)) {
		if (!first) o += ',';
		first = false;
		char buf[448];
		const unit_t* target = u->order_target.unit;
		snprintf(buf, sizeof buf,
		         "{\"tag\":%u,\"owner\":%d,\"type\":%d,\"x\":%d,\"y\":%d,\"hp\":%d,\"shields\":%d,"
		         "\"completed\":%d,\"visible_to\":%d,\"order\":%d,\"resources\":%d,\"queue\":%d,"
		         "\"cooldown\":%d,\"order_target\":%u,\"carrying\":%d,\"hp_max\":%d,\"energy\":%d}",
		         (unsigned)f.get_unit_id(u).raw_value, u->owner, (int)u->unit_type->id, u->sprite->position.x,
		         u->sprite->position.y, u->hp.integer_part(),
		         u->unit_type->has_shield ? u->shield_points.integer_part() : 0,
		         f.u_completed(u) ? 1 : 0, u->sprite->visibility_flags, (int)u->order_type->id,
		         f.ut_resource(u->unit_type) ? u->building.resource.resource_count : 0, (int)u->build_queue.size(),
		         (int)u->ground_weapon_cooldown, target ? (unsigned)f.get_unit_id(target).raw_value : 0u,
		         (int)u->carrying_flags, u->unit_type->hitpoints.integer_part(), u->energy.integer_part());
		o += buf;
	}
	o += "]}";
	return o.c_str();
}

// Saves everything played so far as a (pre-1.18 format) replay. Returns false on error.
GARY_API bool gary_env_save_replay(void* h, const char* path) {
	auto* e = (env*)h;
	try {
		// Same game setup as the source replay (players, races, starting units), with this game's
		// length and commands.
		std::array<uint8_t, 633> header = e->header;
		uint32_t frames = (uint32_t)e->player.st().current_frame;
		memcpy(header.data() + 1, &frames, 4);
		data_loading::file_writer<> w(path);
		auto rw = data_loading::make_replay_file_writer(w);
		rw.template put<uint32_t>(0x53526572);
		rw.put_bytes(header.data(), header.size());
		std::vector<uint8_t> commands;
		for (auto& v : e->saver_st.history) commands.insert(commands.end(), v.begin(), v.end());
		rw.template put<uint32_t>((uint32_t)commands.size());
		rw.put_bytes(commands.data(), commands.size());
		rw.template put<uint32_t>((uint32_t)e->map_data.size());
		rw.put_bytes(e->map_data.data(), e->map_data.size());
		return true;
	} catch (const std::exception& ex) {
		e->last_error = ex.what();
		return false;
	}
}

// --- what's under the mouse ---------------------------------------------------------------
// The human interface resolves clicks the way the game does: by what's drawn at that pixel,
// not by unit ID. v1 uses each sprite's clickable rectangle (the union of its clickable images'
// frames) rather than exact pixel shapes; overlapping units resolve by draw depth (elevation,
// then lower on screen = in front), then by smaller footprint.

namespace {

bool clickable_rect(const env* e, const unit_t* u, rect& out) {
	auto& f = *e->funcs;
	bool any = false;
	for (const image_t* image : ptr(u->sprite->images)) {
		if (!(image->flags & image_t::flag_clickable)) continue;
		xy pos = f.get_image_map_position(image);
		auto size = image->grp->frames.at(image->frame_index).size;
		xy to = pos + xy((int)size.x, (int)size.y);
		if (!any) { out = {pos, to}; any = true; continue; }
		out.from.x = std::min(out.from.x, pos.x); out.from.y = std::min(out.from.y, pos.y);
		out.to.x = std::max(out.to.x, to.x); out.to.y = std::max(out.to.y, to.y);
	}
	return any;
}

bool selectable(const unit_t* u) {
	switch (u->unit_type->id) {
	case UnitTypes::Terran_Nuclear_Missile: case UnitTypes::Protoss_Scarab: case UnitTypes::Spell_Disruption_Web:
	case UnitTypes::Spell_Dark_Swarm:
		return false;
	default:
		return true;
	}
}

uint32_t draw_depth(const unit_t* u) {
	const sprite_t* s = u->sprite;
	return ((uint32_t)s->elevation_level << 14) | (uint32_t)(s->elevation_level <= 4 ? s->position.y : 0);
}

bool seen_by(const unit_t* u, int slot) {
	return u->owner == slot || (u->sprite->visibility_flags & (1u << slot)) != 0;
}

}  // namespace

// The unit a player's click at map pixel (x, y) lands on, or 0. Only units that player can see.
GARY_API unsigned gary_env_unit_at(void* h, int slot, int x, int y) {
	auto* e = (env*)h;
	auto& f = *e->funcs;
	const unit_t* best = nullptr;
	for (const unit_t* u : ptr(e->player.st().visible_units)) {
		if (!selectable(u) || !seen_by(u, slot) || f.us_hidden(u)) continue;
		rect r;
		if (!clickable_rect(e, u, r) || x < r.from.x || y < r.from.y || x >= r.to.x || y >= r.to.y) continue;
		if (!best || draw_depth(u) > draw_depth(best) ||
		    (draw_depth(u) == draw_depth(best) &&
		     u->unit_type->placement_size.x * u->unit_type->placement_size.y <
		         best->unit_type->placement_size.x * best->unit_type->placement_size.y))
			best = u;
	}
	return best ? (unsigned)f.get_unit_id(best).raw_value : 0;
}

// What a drag box (map pixels) selects for a player, like the game: own units whose clickable
// area touches the box, mobile units before buildings, at most 12. Writes tags, returns count.
GARY_API int gary_env_box_select(void* h, int slot, int x0, int y0, int x1, int y1, unsigned* out, int max_out) {
	auto* e = (env*)h;
	auto& f = *e->funcs;
	if (x0 > x1) std::swap(x0, x1);
	if (y0 > y1) std::swap(y0, y1);
	std::vector<const unit_t*> units, buildings;
	for (const unit_t* u : ptr(e->player.st().visible_units)) {
		if (u->owner != slot || !selectable(u) || f.us_hidden(u)) continue;
		rect r;
		if (!clickable_rect(e, u, r) || r.to.x <= x0 || r.to.y <= y0 || r.from.x > x1 || r.from.y > y1) continue;
		(f.unit_can_be_multi_selected(u) ? units : buildings).push_back(u);
	}
	if (units.empty() && !buildings.empty()) units.push_back(buildings.front());
	int n = 0;
	for (const unit_t* u : units) {
		if (n == max_out || n == 12) break;
		out[n++] = f.get_unit_id(u).raw_value;
	}
	return n;
}

// What a ctrl+click (or a double-click) at map pixel (x, y) selects for a player: every unit of the
// clicked unit's owner and type in the view rectangle (map pixels), siege tanks in either mode as
// one type, at most 12. The game works this out in the player's own client and sends a plain
// selection; this follows OpenBW's client (ui.h, end_drag_select): the units in the view in the
// unit finder's order. 0 when the click hits no own unit that can be selected with others (a
// building: a plain click). Writes tags, returns count.
GARY_API int gary_env_type_select(void* h, int slot, int x, int y, int vx0, int vy0, int vx1, int vy1,
                                  unsigned* out, int max_out) {
	auto* e = (env*)h;
	auto& f = *e->funcs;
	unsigned tag = gary_env_unit_at(h, slot, x, y);
	const unit_t* clicked = tag ? f.get_unit(unit_id((uint16_t)tag)) : nullptr;
	if (!clicked || clicked->owner != slot || !f.unit_can_be_multi_selected(clicked)) return 0;
	auto is_tank = [&](const unit_t* u) {
		return f.unit_is(u, UnitTypes::Terran_Siege_Tank_Siege_Mode) || f.unit_is(u, UnitTypes::Terran_Siege_Tank_Tank_Mode);
	};
	int n = 0;
	for (const unit_t* u : f.find_units(rect{xy(vx0, vy0), xy(vx1, vy1)})) {
		if (n == max_out || n == 12) break;
		if (u->owner != slot || !selectable(u) || f.us_hidden(u)) continue;
		if (u->unit_type != clicked->unit_type && !(is_tank(u) && is_tank(clicked))) continue;
		out[n++] = f.get_unit_id(u).raw_value;
	}
	return n;
}

// Renames a player in this game and in replays saved from it (the name the source replay had
// is kept otherwise). Header layout: 12 player slots of 36 bytes from offset 161, name at +11.
GARY_API void gary_env_set_name(void* h, int slot, const char* name) {
	auto* e = (env*)h;
	if (slot < 0 || slot >= 12) return;
	uint8_t* field = e->header.data() + 161 + slot * 36 + 11;
	memset(field, 0, 25);
	strncpy((char*)field, name, 24);
	e->replay_st.player_name[slot] = name;
}

// Unit type of a live unit by tag, or -1.
GARY_API int gary_env_unit_type(void* h, unsigned tag) {
	auto* e = (env*)h;
	const unit_t* u = e->funcs->get_unit(unit_id((uint16_t)tag));
	return u ? (int)u->unit_type->id : -1;
}

// --- new games from a map --------------------------------------------------------------------

namespace {

void put16(uint8_t* p, uint16_t v) { memcpy(p, &v, 2); }
void put32(uint8_t* p, uint32_t v) { memcpy(p, &v, 4); }

// Remastered maps: version 206 -> 205, and an 'STR ' chunk built from 'STRx' if that's all there
// is (same fix as resim/scr_format.py).
std::vector<uint8_t> remastered_map_to_bw(std::vector<uint8_t> chk) {
	struct chunk { size_t pos; std::string name; int32_t size; };
	std::vector<chunk> chunks;
	for (size_t pos = 0; pos + 8 <= chk.size();) {
		int32_t size;
		memcpy(&size, chk.data() + pos + 4, 4);
		if (size < 0 || pos + 8 + (size_t)size > chk.size()) break;
		chunks.push_back({pos, std::string((const char*)chk.data() + pos, 4), size});
		pos += 8 + (size_t)size;
	}
	bool has_str = false, has_strx = false;
	for (auto& c : chunks) {
		if (c.name == "VER " && c.size >= 2 && chk[c.pos + 8] == 206 && chk[c.pos + 9] == 0) put16(chk.data() + c.pos + 8, 205);
		has_str |= c.name == "STR ";
		has_strx |= c.name == "STRx";
	}
	if (has_str || !has_strx) return chk;
	const chunk* strx = nullptr;
	for (auto& c : chunks) if (c.name == "STRx") strx = &c;
	const uint8_t* d = chk.data() + strx->pos + 8;
	uint32_t count;
	memcpy(&count, d, 4);
	if ((size_t)4 + 4 * (size_t)count > (size_t)strx->size) return chk;
	size_t base = 2 + 2 * (size_t)count;
	std::vector<uint8_t> body{0};
	std::vector<uint16_t> offsets;
	for (uint32_t i = 0; i != count; ++i) {
		uint32_t off;
		memcpy(&off, d + 4 + 4 * i, 4);
		std::string str;
		for (uint32_t j = off; j < (uint32_t)strx->size && d[j]; ++j) str += (char)d[j];
		if (!str.empty() && base + body.size() + str.size() + 1 < 0x10000) {
			offsets.push_back((uint16_t)(base + body.size()));
			body.insert(body.end(), str.begin(), str.end());
			body.push_back(0);
		} else {
			offsets.push_back((uint16_t)base);
		}
	}
	std::vector<uint8_t> str_chunk(2 + 2 * offsets.size());
	put16(str_chunk.data(), (uint16_t)count);
	for (size_t i = 0; i != offsets.size(); ++i) put16(str_chunk.data() + 2 + 2 * i, offsets[i]);
	str_chunk.insert(str_chunk.end(), body.begin(), body.end());
	chk.insert(chk.end(), {'S', 'T', 'R', ' '});
	uint8_t len[4];
	put32(len, (uint32_t)str_chunk.size());
	chk.insert(chk.end(), len, len + 4);
	chk.insert(chk.end(), str_chunk.begin(), str_chunk.end());
	return chk;
}

// Map data (CHK) from a map file (.scm/.scx) or from the map embedded in a pre-1.18 replay.
std::vector<uint8_t> read_map(const std::string& path) {
	std::string lower;
	for (char c : path) lower += (char)tolower((unsigned char)c);
	auto ends = [&](const char* x) { size_t n = strlen(x); return lower.size() >= n && lower.compare(lower.size() - n, n, x) == 0; };
	if (ends(".rep")) {
		std::ifstream in(path, std::ios::binary);
		if (!in) error("can't open replay: %s", path.c_str());
		std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
		data_loading::data_reader_le raw(bytes.data(), bytes.data() + bytes.size());
		auto r = data_loading::make_replay_file_reader(raw);
		r.template get<uint32_t>();
		std::vector<uint8_t> skip(633);
		r.get_bytes(skip.data(), skip.size());
		skip.resize(r.template get<uint32_t>());
		r.get_bytes(skip.data(), skip.size());
		std::vector<uint8_t> map(r.template get<uint32_t>());
		r.get_bytes(map.data(), map.size());
		return map;
	}
	data_loading::mpq_file<> mpq(path.c_str());
	a_vector<uint8_t> chk;
	mpq(chk, "staredit\\scenario.chk");
	return remastered_map_to_bw(std::vector<uint8_t>(chk.begin(), chk.end()));
}

// Player slots that have a start location on the map (UNIT chunk: start location = unit 214).
std::vector<int> start_location_slots(const std::vector<uint8_t>& chk) {
	std::vector<int> slots;
	for (size_t pos = 0; pos + 8 <= chk.size();) {
		int32_t size;
		memcpy(&size, chk.data() + pos + 4, 4);
		if (size < 0 || pos + 8 + (size_t)size > chk.size()) break;
		if (!memcmp(chk.data() + pos, "UNIT", 4)) {
			for (size_t u = 0; u + 36 <= (size_t)size; u += 36) {
				const uint8_t* d = chk.data() + pos + 8 + u;
				uint16_t type;
				memcpy(&type, d + 8, 2);
				int owner = d[16];
				if (type == 214 && owner < 8 && std::find(slots.begin(), slots.end(), owner) == slots.end()) slots.push_back(owner);
			}
		}
		pos += 8 + (size_t)size;
	}
	std::sort(slots.begin(), slots.end());
	return slots;
}

}  // namespace

namespace {

// A new game on map data (a scenario.chk) with n players in the given slots. Melee: the map's
// start locations and starting units (workers and a main building). Use map settings (ums): the
// map's own preplaced units for those slots, and no melee victory rules (drills, made by adding
// units to a map: gary/drills.py).
env* new_game(const char* data_dir, std::vector<uint8_t> map_data, const std::string& map_name, int n,
              const std::vector<int>& slots, const int* races, const char* const* names, uint32_t seed, bool ums) {
	auto e = std::make_unique<env>(data_dir);
	e->map_data = std::move(map_data);
	std::array<int, 12> race_of{}, controller{}, player_id{};
	player_id.fill(-1);
	for (int i = 0; i != n; ++i) {
		int slot = slots[i];
		controller[slot] = player_t::controller_occupied;
		race_of[slot] = races[i];
		player_id[slot] = i;
		e->replay_st.player_name[slot] = names[i];
	}
	e->funcs.emplace(e->player.st(), e->action_st, e->replay_st);
	for (size_t i = 0; i != 12; ++i) e->action_st.player_id[i] = player_id[i];
	state& st = e->player.st();
	game_load_functions load(st);
	load.load_map_data(e->map_data.data(), e->map_data.size(), [&]() {
		load.setup_info.victory_condition = ums ? 0 : 1;  // melee
		load.setup_info.starting_units = ums ? 0 : 2;     // workers and a main building
		load.setup_info.tournament_mode = 0;
		load.setup_info.resource_type = 1;
		load.setup_info.starting_minerals = 50;
		for (size_t i = 0; i != 12; ++i) {
			st.players[i].controller = controller[i];
			st.players[i].race = (race_t)race_of[i];
			st.players[i].force = 0;
			if (ums && i < 8) load.setup_info.create_melee_units_for_player[i] = false;
		}
		st.lcg_rand_state = seed;
	});
	e->replay_st.end_frame = 0x7fffffff;
	e->replay_st.map_name = map_name.c_str();

	// Game info for saved replays, laid out like a replay saved by the game itself.
	uint8_t* h = e->header.data();
	memset(h, 0, e->header.size());
	h[0] = 1;  // Brood War
	h[7] = 72;
	put32(h + 8, seed);
	memset(h + 12, 8, 8);
	strncpy((char*)h + 24, "Gary", 23);
	put16(h + 52, (uint16_t)st.game->map_tile_width);
	put16(h + 54, (uint16_t)st.game->map_tile_height);
	h[56] = (uint8_t)n;  // active players
	h[57] = (uint8_t)n;  // slots
	h[58] = 6;           // fastest
	put16(h + 60, ums ? 10 : 2);    // game type: use map settings / melee
	put16(h + 62, 1);
	put16(h + 68, (uint16_t)st.game->tileset_index);
	strncpy((char*)h + 72, "Gary", 24);
	strncpy((char*)h + 97, map_name.c_str(), 31);
	put16(h + 129, ums ? 10 : 2);
	put16(h + 131, 1);
	const uint8_t melee[11] = {1, 1, 1, 2, 2, 0, 1, 1, 0, 1, 0};  // victory .. tournament
	const uint8_t map_settings[11] = {0, 1, 1, 2, 0, 0, 1, 1, 0, 1, 0};
	memcpy(h + 137, ums ? map_settings : melee, 11);
	put32(h + 152, 50);
	for (int i = 0; i != 12; ++i) {
		uint8_t* slot = h + 161 + i * 36;
		put32(slot, (uint32_t)i);
		put32(slot + 4, (uint32_t)player_id[i]);
		slot[8] = (uint8_t)controller[i];
		slot[9] = (uint8_t)(controller[i] ? race_of[i] : 6);
		slot[10] = 0;
		strncpy((char*)slot + 11, e->replay_st.player_name[i].c_str(), 24);
	}
	for (int i = 0; i != 8; ++i) put32(h + 161 + 12 * 36 + i * 4, (uint32_t)i);
	for (int i = 0; i != 8; ++i) h[161 + 12 * 36 + 32 + i] = (!ums && controller[i]) ? 1 : 0;  // melee units
	return e.release();
}

std::string map_stem(const std::string& path) {
	std::string stem = path;  // the map's file name, shown as the replay's map name
	size_t slash = stem.find_last_of("/\\");
	if (slash != std::string::npos) stem = stem.substr(slash + 1);
	size_t dot = stem.find_last_of('.');
	if (dot != std::string::npos) stem = stem.substr(0, dot);
	return stem;
}

}  // namespace

// Starts a melee game on a map file (.scm/.scx, or a pre-1.18 replay's embedded map) with
// n players. races: 0 zerg, 1 terran, 2 protoss. names: n strings. Players get random start
// locations (from seed). Returns null on error (gary_env_error(null) has the message).
GARY_API void* gary_env_create_game(const char* data_dir, const char* map_path, int n, const int* races,
                                    const char* const* names, uint32_t seed) {
	try {
		auto map = read_map(map_path);
		auto slots = start_location_slots(map);
		if ((int)slots.size() < n) error("map has %d start locations, %d players requested", (int)slots.size(), n);
		uint32_t rng = seed * 2654435761u + 1;
		for (size_t i = slots.size(); i > 1; --i) {  // shuffle start locations
			rng = rng * 1103515245u + 12345u;
			std::swap(slots[i - 1], slots[(rng >> 16) % i]);
		}
		return new_game(data_dir, std::move(map), map_stem(map_path), n, slots, races, names, seed, false);
	} catch (const std::exception& ex) {
		g_create_error = ex.what();
		return nullptr;
	}
}

// Starts a "use map settings" game from map data (a scenario.chk, e.g. a map with drill units
// added: gary/drills.py): n players in the given slots (0-7) with the map's preplaced units for
// them, and no melee rules. Returns null on error.
GARY_API void* gary_env_create_custom(const char* data_dir, const uint8_t* chk, int size, const char* map_name,
                                      int n, const int* slots, const int* races, const char* const* names, uint32_t seed) {
	try {
		return new_game(data_dir, std::vector<uint8_t>(chk, chk + size), map_name, n, std::vector<int>(slots, slots + n),
		                races, names, seed, true);
	} catch (const std::exception& ex) {
		g_create_error = ex.what();
		return nullptr;
	}
}

// A map file's scenario data (scenario.chk), into out (capacity bytes). Returns its size (copied
// only if it fits; call with capacity 0 to ask), or -1 on error.
GARY_API int gary_env_map_chk(const char* map_path, uint8_t* out, int capacity) {
	try {
		auto map = read_map(map_path);
		if ((int)map.size() <= capacity) memcpy(out, map.data(), map.size());
		return (int)map.size();
	} catch (const std::exception& ex) {
		g_create_error = ex.what();
		return -1;
	}
}

// --- building placement and map knowledge ------------------------------------------------------

// The map's terrain in bulk (map knowledge: gary/terrain.py): per 8x8-pixel minitile (row by row,
// map tile width * 4 per row), whether ground units can stand there and its ground height (0 low,
// 1 middle, 2 high); per 32x32 tile, whether it's buildable. Returns the minitile count, or -1.
GARY_API int gary_env_terrain(void* h, uint8_t* walk, uint8_t* height, uint8_t* build) {
	auto* e = (env*)h;
	try {
		auto& f = *e->funcs;
		const state& st = e->player.st();
		int tw = (int)st.game->map_tile_width, th = (int)st.game->map_tile_height;
		int mw = tw * 4, mh = th * 4;
		for (int my = 0; my != mh; ++my) {
			for (int mx = 0; mx != mw; ++mx) {
				xy p(mx * 8 + 4, my * 8 + 4);
				walk[my * mw + mx] = f.is_walkable(p) ? 1 : 0;
				height[my * mw + mx] = (uint8_t)f.get_ground_height_at(p);
			}
		}
		for (int i = 0; i != tw * th; ++i) build[i] = (st.tiles[i].flags & tile_t::flag_unbuildable) ? 0 : 1;
		return mw * mh;
	} catch (const std::exception& ex) {
		e->last_error = ex.what();
		return -1;
	}
}

// Whether ground units can stand at map pixel (x, y) (the terrain's walkable minitiles; units and
// buildings aside). For placing drill units (gary/drills.py).
GARY_API bool gary_env_walkable(void* h, int x, int y) {
	auto& f = *((env*)h)->funcs;
	if (x < 0 || y < 0 || x >= (int)f.game_st.map_width || y >= (int)f.game_st.map_height) return false;
	return f.is_walkable(xy(x, y));
}

// Whether a player could place unit_type with its top-left tile at (tile_x, tile_y) right now,
// using the game's own check (the green/red placement grid): terrain, creep / pylon power,
// units in the way, the resource-depot distance rule, and unexplored tiles (not allowed).
// builder_tag: the worker that would build it (0 if none).
GARY_API bool gary_env_can_place(void* h, int slot, unsigned builder_tag, int unit_type, int tile_x, int tile_y) {
	auto* e = (env*)h;
	auto& f = *e->funcs;
	const unit_type_t* ut = f.get_unit_type((UnitTypes)unit_type);
	const unit_t* builder = builder_tag ? f.get_unit(unit_id((uint16_t)builder_tag)) : nullptr;
	xy pos(32 * tile_x + ut->placement_size.x / 2, 32 * tile_y + ut->placement_size.y / 2);
	return f.can_place_building(builder, slot, ut, pos, false, true);
}

// Static map knowledge players have before the game starts (they know the maps): whether a
// resource depot (Command Center / Nexus / Hatchery, 4x3 tiles) fits at a top-left tile by terrain
// and the 3-tile distance from resources, ignoring units and fog. Used to find base locations.
GARY_API bool gary_env_depot_spot_ok(void* h, int tile_x, int tile_y) {
	auto* e = (env*)h;
	auto& f = *e->funcs;
	const state& st = e->player.st();
	int w = (int)st.game->map_tile_width, hgt = (int)st.game->map_tile_height;
	if (tile_x < 0 || tile_y < 0 || tile_x + 4 > w || tile_y + 3 > hgt) return false;
	for (int y = tile_y; y != tile_y + 3; ++y) {
		for (int x = tile_x; x != tile_x + 4; ++x) {
			auto& tile = st.tiles[y * w + x];
			if (tile.flags & (tile_t::flag_partially_walkable | tile_t::flag_unbuildable)) return false;
		}
	}
	rect bb{xy(32 * (tile_x - 3), 32 * (tile_y - 3)), xy(32 * (tile_x + 4 + 3), 32 * (tile_y + 3 + 3))};
	for (const unit_t* n : f.find_units_noexpand(bb)) {
		if (f.ut_resource(n->unit_type)) return false;
	}
	return true;
}

// Start locations of the map (pixel centers), from its UNIT chunk: map knowledge.
GARY_API const char* gary_env_start_locations(void* h) {
	auto* e = (env*)h;
	std::string& o = e->observation;
	o = "[";
	const auto& chk = e->map_data;
	bool first = true;
	for (size_t pos = 0; pos + 8 <= chk.size();) {
		int32_t size;
		memcpy(&size, chk.data() + pos + 4, 4);
		if (size < 0 || pos + 8 + (size_t)size > chk.size()) break;
		if (!memcmp(chk.data() + pos, "UNIT", 4)) {
			for (size_t u = 0; u + 36 <= (size_t)size; u += 36) {
				const uint8_t* d = chk.data() + pos + 8 + u;
				uint16_t x, y, type;
				memcpy(&x, d + 4, 2);
				memcpy(&y, d + 6, 2);
				memcpy(&type, d + 8, 2);
				if (type != 214) continue;
				if (!first) o += ',';
				first = false;
				o += "{\"x\":" + std::to_string(x) + ",\"y\":" + std::to_string(y) + ",\"slot\":" + std::to_string(d[16]) + "}";
			}
		}
		pos += 8 + (size_t)size;
	}
	o += "]";
	return o.c_str();
}
