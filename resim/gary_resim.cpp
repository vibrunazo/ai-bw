// gary_resim: re-simulates a Brood War replay in OpenBW (headless) and prints periodic state
// snapshots as JSON lines on stdout.
//
// Usage:
//   gary_resim --data <dir> --replay <file.rep | -> [--every <frames>] [--flat]    ("-" = read stdin)
//
// --flat: the input is a Remastered-era replay already decoded by resim/scr_format.py.
// --unit-limit N: the replay's unit table size (Remastered LMTS section); IDs are translated
//                 when it's larger than 1700.
//
// --data is either a folder with the three 1.16.1/1.18 MPQs (StarDat.mpq, BrooDat.mpq,
// Patch_rt.mpq) or a folder of loose files extracted from a newer install (e.g. SC:R's CASC
// storage), laid out with their in-archive paths (arr/units.dat, scripts/iscript.bin, ...).
//
// Output, one JSON object per line:
//   {"type":"header", ...}                      replay info
//   {"type":"snapshot","frame":F,"players":[...]} every --every frames (default 24 = 1 s; 0 = off)
//   {"type":"event","frame":F,"slot":P,"ev":"start|done|gone","unit":ID,"tag":T,...}  as they happen
//   {"type":"cmd","frame":F,"slot":P,"act":"train|build|...","id":N}  accepted production commands
//   {"type":"end", ...}                          totals
//
// Each snapshot reports, per player: minerals, gas, supply (in BW's displayed units), unit counts
// by unit type ID (all / completed), what it can see of other players' units right now ("seen":
// type -> count, i.e. through the fog of war), and how many replay actions the engine accepted or
// rejected since the previous snapshot. "g" places units on a 16x16 grid over the map, as flat
// pairs (channel * 256 + cell, value): channels 0-2 own army supply x2 / workers / buildings,
// 3-5 the same for other players' units this player can see; cell = row * 16 + column.
// {"type":"army",...} lines are accepted orders given to army units (print_army_command).
// Rejected actions have a low baseline from spam; a sustained spike is the main desync signal
// (the replay's commands stop making sense in the simulated state).

#include <stdexcept>  // OpenBW's util.h uses std::runtime_error without including it

#include "bwgame.h"
#include "replay.h"
#include "replay_stream.h"
#include "scr_replay.h"

#include <cstdio>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif
#include <cstring>
#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <set>
#include <unordered_map>
#include <vector>

using namespace bwgame;

namespace {

bool file_exists(const std::string& path) {
	std::ifstream f(path, std::ios::binary);
	return f.good();
}

// Game data from the classic MPQs if present, else from a folder of extracted files.
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

std::string json_escape(const a_string& s) {
	std::string r;
	for (unsigned char c : s) {
		if (c == '"' || c == '\\') { r += '\\'; r += (char)c; }
		else if (c < 0x20) { char buf[8]; snprintf(buf, sizeof buf, "\\u%04x", c); r += buf; }
		else r += (char)c;  // replay strings are raw bytes (often CP949); consumers decode leniently
	}
	return r;
}

// Same as replay_functions::next_frame, but counts accepted / rejected actions per player
// (Remastered command handling: scr_replay.h).
bool is_player(const state& st, int i);

struct counting_replay_functions : scr_replay<replay_functions> {
	std::array<int, 12> accepted{};
	std::array<int, 12> rejected{};
	std::array<int, 12> total_rejected{};

	using scr_replay<replay_functions>::scr_replay;

	void execute_actions_counted() {
		execute_actions_scr([&](int owner, int action_id, const uint8_t* cmd, const uint8_t* end, bool ok) {
			if (ok) print_production_command(owner, action_id, cmd, end);
			if (ok) print_army_command(owner, action_id, cmd, end);
			if (ok && fights_until > 0 && st.current_frame <= fights_until) print_unit_command(owner, action_id, cmd, end);
			if (!ok && debug_rejects) fprintf(stderr, "reject frame %d owner %d action 0x%02x\n", st.current_frame, owner, action_id);
			if (owner >= 0 && owner < 12) {
				if (ok) ++accepted[owner];
				else { ++rejected[owner]; ++total_rejected[owner]; }
			}
		});
	}

	// Accepted production decisions, for training data: what the player chose to make, when.
	//   {"type":"cmd","frame":F,"slot":P,"act":"train|morph|build|bmorph|research|upgrade","id":N[,"x":X,"y":Y]}
	// (build: tile position, so a new town hall's location tells an expansion from a macro hatch)
	void print_production_command(int owner, int action_id, const uint8_t* cmd, const uint8_t* end) {
		auto u16 = [&](int at) { return cmd + at + 1 < end ? (int)(cmd[at] | cmd[at + 1] << 8) : -1; };
		auto u8 = [&](int at) { return cmd + at < end ? (int)cmd[at] : -1; };
		const char* act = nullptr;
		int id = -1, x = -1, y = -1;
		switch (action_id) {  // cmd[0] = player id, cmd[1] = action id
		case 0x1f: act = "train"; id = u16(2); break;
		case 0x23: act = "morph"; id = u16(2); break;
		case 0x35: act = "bmorph"; id = u16(2); break;
		case 0x0c: act = "build"; x = u16(3); y = u16(5); id = u16(7); break;
		case 0x30: act = "research"; id = u8(2); break;
		case 0x32: act = "upgrade"; id = u8(2); break;
		default: return;
		}
		if (x >= 0) printf("{\"type\":\"cmd\",\"frame\":%d,\"slot\":%d,\"act\":\"%s\",\"id\":%d,\"x\":%d,\"y\":%d}\n",
		                   st.current_frame, owner, act, id, x, y);
		else printf("{\"type\":\"cmd\",\"frame\":%d,\"slot\":%d,\"act\":\"%s\",\"id\":%d}\n", st.current_frame, owner, act, id);
	}

	// Accepted orders given to army units (the selection holds fighting units: not workers,
	// buildings, overlords or larvae), for training data: where the player sent the army, how.
	//   {"type":"army","frame":F,"slot":P,"kind":"move|amove|attack|patrol|other","x":X,"y":Y,
	//    "n":units,"sup":supply x2,"cx":CX,"cy":CY}   (x, y: target; cx, cy: the selection's center)
	void print_army_command(int owner, int action_id, const uint8_t* cmd, const uint8_t* end) {
		auto u16 = [&](int at) { return cmd + at + 1 < end ? (int)(cmd[at] | cmd[at + 1] << 8) : -1; };
		auto u8 = [&](int at) { return cmd + at < end ? (int)cmd[at] : -1; };
		int x, y, target, order = -1;
		switch (action_id) {  // cmd[0] = player id, cmd[1] = action id
		case 0x14: x = u16(2); y = u16(4); target = u16(6); break;                      // right click
		case 0x15: x = u16(2); y = u16(4); target = u16(6); order = u8(10); break;      // targeted order
		case 0x60: x = u16(2); y = u16(4); target = translate_unit_id((uint16_t)u16(6)); break;
		case 0x61: x = u16(2); y = u16(4); target = translate_unit_id((uint16_t)u16(6)); order = u8(12); break;
		default: return;
		}
		int n = 0, sup = 0;
		long sx = 0, sy = 0;
		for (unit_t* u : action_st.selection.at(owner)) {
			if (!u || u->owner != owner || ut_building(u->unit_type) || ut_worker(u->unit_type)) continue;
			if (u->unit_type->supply_required.raw_value <= 0) continue;
			++n;
			sup += u->unit_type->supply_required.raw_value;
			sx += u->sprite->position.x;
			sy += u->sprite->position.y;
		}
		if (!n) return;
		const char* kind = "other";
		if (order < 0) {
			unit_t* t = target > 0 ? get_unit(unit_id((uint16_t)target)) : nullptr;
			kind = t && t->owner != owner && t->owner < 8 ? "attack" : "move";
		} else if (order == 14) kind = "amove";
		else if (order == 10 || order == 11) kind = "attack";
		else if (order == 6) kind = "move";
		else if (order == 152) kind = "patrol";
		printf("{\"type\":\"army\",\"frame\":%d,\"slot\":%d,\"kind\":\"%s\",\"x\":%d,\"y\":%d,\"n\":%d,\"sup\":%d,\"cx\":%ld,\"cy\":%ld}\n",
		       st.current_frame, owner, kind, x, y, n, sup, sx / n, sy / n);
	}

	// --- fights (--fights SECONDS), for training data on unit control ---------------------------
	// Every command a player gives to units, with the units selected when it ran:
	//   {"type":"ucmd","frame":F,"slot":P,"cmd":"rclick|order|stop|hold|stim|siege|unsiege|
	//    return|burrow|unburrow","units":[tags],"x":X,"y":Y,"target":tag,"order":id}
	// and, every FIGHT_EVERY frames while a player's units have visible enemy fighters within
	// FIGHT_NEAR pixels, the units around the fight as that player could see them:
	//   {"type":"fight","frame":F,"slot":P,"cx":CX,"cy":CY,"u":[[tag,type,own,x,y,hp,shields,
	//    cooldown,order,order_target,carrying,completed],...]}   own: 1 the player, 0 enemy, 2 neutral
	//    (mineral fields and geysers, so "back to mining" can be learned)
	// (tags are OpenBW's unit IDs in both, so commands and snapshots line up).
	// For reaction times (eval/reaction.py), as they happen:
	//   {"type":"sight","frame":F,"slot":P,"tag":T,"unit":ID,"x":X,"y":Y}  an enemy unit comes into
	//    P's sight after REACT_GAP frames out of it (or for the first time)
	//   {"type":"hit","frame":F,"slot":P,"tag":T,"unit":ID,"x":X,"y":Y}    P's unit loses hit points
	//    or shields after REACT_GAP frames without
	int fights_until = 0;          // frame; 0 = off
	// a fight snapshot: everyone within FIGHT_AROUND of the fight, and the player's own units
	// within FIGHT_OWN (the whole mineral line a player might pull or evacuate)
	static constexpr int FIGHT_EVERY = 12, FIGHT_NEAR = 8 * 32, FIGHT_AROUND = 12 * 32, FIGHT_OWN = 24 * 32;

	static bool fighter(const unit_t* u, const counting_replay_functions& f) {
		return u->unit_type->supply_required.raw_value > 0 && !f.ut_worker(u->unit_type) && !f.ut_building(u->unit_type);
	}

	void print_unit_command(int owner, int action_id, const uint8_t* cmd, const uint8_t* end) {
		auto u16 = [&](int at) { return cmd + at + 1 < end ? (int)(cmd[at] | cmd[at + 1] << 8) : -1; };
		auto u8 = [&](int at) { return cmd + at < end ? (int)cmd[at] : -1; };
		const char* name = nullptr;
		int x = -1, y = -1, target = 0, order = -1;
		switch (action_id) {  // cmd[0] = player id, cmd[1] = action id
		case 0x14: name = "rclick"; x = u16(2); y = u16(4); target = u16(6); break;
		case 0x15: name = "order"; x = u16(2); y = u16(4); target = u16(6); order = u8(10); break;
		case 0x60: name = "rclick"; x = u16(2); y = u16(4); target = translate_unit_id((uint16_t)u16(6)); break;
		case 0x61: name = "order"; x = u16(2); y = u16(4); target = translate_unit_id((uint16_t)u16(6)); order = u8(12); break;
		case 26: name = "stop"; break;
		case 30: name = "return"; break;
		case 37: name = "unsiege"; break;
		case 38: name = "siege"; break;
		case 43: name = "hold"; break;
		case 44: name = "burrow"; break;
		case 45: name = "unburrow"; break;
		case 54: name = "stim"; break;
		default: return;
		}
		std::string units;
		for (unit_t* u : action_st.selection.at(owner)) {
			if (!u || u->owner != owner || ut_building(u->unit_type)) continue;
			if (!units.empty()) units += ',';
			units += std::to_string(get_unit_id(u).raw_value);
		}
		if (units.empty()) return;
		if (target == 0x7ff) target = 0;
		printf("{\"type\":\"ucmd\",\"frame\":%d,\"slot\":%d,\"cmd\":\"%s\",\"units\":[%s],\"x\":%d,\"y\":%d,\"target\":%d,\"order\":%d}\n",
		       st.current_frame, owner, name, units.c_str(), x, y, target, order);
	}

	void print_fights() {
		if (!fights_until || st.current_frame > fights_until || st.current_frame % FIGHT_EVERY) return;
		for (int p = 0; p != 8; ++p) {
			if (!is_player(st, p)) continue;
			// visible enemy fighters close to any of p's units
			std::vector<const unit_t*> engaged;
			for (int o = 0; o != 8; ++o) {
				if (o == p || !is_player(st, o)) continue;
				for (const unit_t* e : ptr(st.player_units[o])) {
					if (!e->sprite || us_hidden(e) || !fighter(e, *this) || !(e->sprite->visibility_flags & (1 << p))) continue;
					for (const unit_t* m : ptr(st.player_units[p])) {
						if (!m->sprite || us_hidden(m)) continue;
						int dx = e->sprite->position.x - m->sprite->position.x, dy = e->sprite->position.y - m->sprite->position.y;
						if (dx * dx + dy * dy <= FIGHT_NEAR * FIGHT_NEAR) { engaged.push_back(e); break; }
					}
				}
			}
			if (engaged.empty()) continue;
			long cx = 0, cy = 0;
			for (auto* e : engaged) { cx += e->sprite->position.x; cy += e->sprite->position.y; }
			cx /= (long)engaged.size();
			cy /= (long)engaged.size();
			std::string out;
			for (int o : {0, 1, 2, 3, 4, 5, 6, 7, 11}) {
				if (o != 11 && !is_player(st, o)) continue;
				for (const unit_t* u : ptr(st.player_units[o])) {
					if (!u->sprite || us_hidden(u)) continue;
					if (o == 11 && !ut_resource(u)) continue;      // neutral: only minerals and geysers
					if (o != p && !(u->sprite->visibility_flags & (1 << p))) continue;
					long dx = u->sprite->position.x - cx, dy = u->sprite->position.y - cy;
					long r = o == p ? FIGHT_OWN : FIGHT_AROUND;
					if (dx * dx + dy * dy > r * r) continue;
					const unit_t* t = u->order_target.unit;
					char buf[160];
					snprintf(buf, sizeof buf, "%s[%u,%d,%d,%d,%d,%d,%d,%d,%d,%u,%d,%d]", out.empty() ? "" : ",",
					         (unsigned)get_unit_id(u).raw_value, (int)u->unit_type->id, o == p ? 1 : o == 11 ? 2 : 0,
					         u->sprite->position.x, u->sprite->position.y, u->hp.integer_part(),
					         u->unit_type->has_shield ? u->shield_points.integer_part() : 0,
					         (int)u->ground_weapon_cooldown, (int)u->order_type->id,
					         t ? (unsigned)get_unit_id(t).raw_value : 0u, (int)u->carrying_flags, u_completed(u) ? 1 : 0);
					out += buf;
				}
			}
			printf("{\"type\":\"fight\",\"frame\":%d,\"slot\":%d,\"cx\":%ld,\"cy\":%ld,\"u\":[%s]}\n",
			       st.current_frame, p, cx, cy, out.c_str());
		}
	}

	static constexpr int REACT_GAP = 48;
	std::unordered_map<uint32_t, int> last_seen;      // (player << 16 | tag) -> frame it was last in sight
	struct hp_state { int hp; int frame; };
	std::unordered_map<uint16_t, hp_state> last_hit;  // tag -> hit points + shields, frame last hit

	void print_reactions() {
		if (!fights_until || st.current_frame > fights_until) return;
		int now = st.current_frame;
		for (int o = 0; o != 8; ++o) {
			if (!is_player(st, o)) continue;
			for (const unit_t* u : ptr(st.player_units[o])) {
				if (!u->sprite || us_hidden(u)) continue;
				uint16_t tag = (uint16_t)get_unit_id(u).raw_value;
				for (int p = 0; p != 8; ++p) {
					if (p == o || !is_player(st, p) || !(u->sprite->visibility_flags & (1 << p))) continue;
					auto [it, fresh] = last_seen.try_emplace((uint32_t)p << 16 | tag, now);
					if (fresh || now - it->second > REACT_GAP)
						printf("{\"type\":\"sight\",\"frame\":%d,\"slot\":%d,\"tag\":%u,\"unit\":%d,\"x\":%d,\"y\":%d}\n",
						       now, p, (unsigned)tag, (int)u->unit_type->id, u->sprite->position.x, u->sprite->position.y);
					it->second = now;
				}
				int hp = u->hp.integer_part() + (u->unit_type->has_shield ? u->shield_points.integer_part() : 0);
				auto [h, first] = last_hit.try_emplace(tag, hp_state{hp, -REACT_GAP - 1});
				if (!first && hp < h->second.hp) {
					if (now - h->second.frame > REACT_GAP && u_completed(u))
						printf("{\"type\":\"hit\",\"frame\":%d,\"slot\":%d,\"tag\":%u,\"unit\":%d,\"x\":%d,\"y\":%d}\n",
						       now, o, (unsigned)tag, (int)u->unit_type->id, u->sprite->position.x, u->sprite->position.y);
					h->second.frame = now;
				}
				h->second.hp = hp;
			}
		}
	}

	// Debug: GARY_TRACE_UNIT=<raw unit id>[,from_frame[,to_frame]] prints that unit's position, order
	// and order state every frame (stderr), to follow one unit through a suspected divergence.
	void trace_unit() {
		static const char* env = getenv("GARY_TRACE_UNIT");
		if (!env) return;
		static unsigned raw = 0; static int from = 0, to = 1 << 30; static bool parsed = false;
		if (!parsed) { parsed = true; sscanf(env, "%u,%d,%d", &raw, &from, &to); }
		if (st.current_frame < from || st.current_frame > to) return;
		static const char* order_env = getenv("GARY_TRACE_ORDER");   // GARY_TRACE_ORDER=<order id>: every unit holding it
		if (order_env) {
			int want = atoi(order_env);
			for (unit_t* v : ptr(st.visible_units)) {
				if ((int)v->order_type->id == want)
					fprintf(stderr, "trace frame %d unit %u owner %d type %d at %d,%d order %d state %d target %d,%d\n",
					        st.current_frame, (unsigned)get_unit_id(v).raw_value, v->owner, (int)v->unit_type->id,
					        v->sprite->position.x, v->sprite->position.y, want, v->order_state, v->order_target.pos.x, v->order_target.pos.y);
			}
			return;
		}
		unit_t* u = get_unit(unit_id((uint16_t)raw));
		if (!u) { fprintf(stderr, "trace frame %d unit %u: none\n", st.current_frame, raw); return; }
		fprintf(stderr, "trace frame %d unit %u type %d at %d,%d order %d state %d timer %d target %d,%d queue %d\n",
		        st.current_frame, raw, (int)u->unit_type->id, u->sprite->position.x, u->sprite->position.y,
		        (int)u->order_type->id, u->order_state, u->main_order_timer, u->order_target.pos.x, u->order_target.pos.y,
		        (int)(u->order_queue.empty() ? 0 : 1));
	}

	void next_frame_counted() {
		if (st.current_frame == replay_st.end_frame) error("replay: attempt to play past end");
		execute_actions_counted();
		state_functions::next_frame();
		trace_unit();
		track_units();
		print_fights();
		print_reactions();
		if (debug_rejects) track_object_usage();
	}

	// Peak number of live engine objects, to see whether a game gets near OpenBW's (1.16.1)
	// limits: units 1700, bullets 100, sprites 2500, images 5000, orders 2000.
	struct usage { size_t peak = 0; int frame = 0; };
	std::array<usage, 5> object_usage{};

	template<typename C>
	static size_t live(const C& c) {
		size_t free_n = 0;
		for (auto it = c.free_list.begin(); it != c.free_list.end(); ++it) ++free_n;
		return c.size - free_n;
	}

	void track_object_usage() {
		if (st.current_frame % 8) return;
		size_t now[5] = {live(st.units_container), live(st.bullets_container), live(st.sprites_container),
		                 live(st.images_container), live(st.orders_container)};
		for (int i = 0; i != 5; ++i) {
			if (now[i] > object_usage[i].peak) object_usage[i] = {now[i], st.current_frame};
		}
	}

	void print_object_usage() {
		const char* names[5] = {"units", "bullets", "sprites", "images", "orders"};
		for (int i = 0; i != 5; ++i)
			fprintf(stderr, "peak %s %zu at frame %d" "\n", names[i], object_usage[i].peak, object_usage[i].frame);
	}

	// --- unit events -------------------------------------------------------------------------
	// Diffing each player's unit list frame by frame, keyed by unit ID (the same IDs replays use
	// in their Select commands), gives the true build order:
	//   start  a building began (placed, or a drone/building morphed into it; "from" = old type)
	//   done   a building finished, or a unit was produced (eggs, larva, etc. are not reported)
	//   gone   a building or unit died (or was removed when its owner left): its owner and where it
	//          was last seen alive (for losses by place: ingest/army_dataset.py)
	struct tracked { int type; bool completed; bool seen; int slot; int x; int y; };
	std::unordered_map<uint16_t, tracked> known;

	bool transient(int type) const {
		switch ((UnitTypes)type) {
		case UnitTypes::Zerg_Larva: case UnitTypes::Zerg_Egg: case UnitTypes::Zerg_Cocoon:
		case UnitTypes::Zerg_Lurker_Egg: case UnitTypes::Protoss_Interceptor: case UnitTypes::Protoss_Scarab:
		case UnitTypes::Terran_Vulture_Spider_Mine:
			return true;
		default:
			return false;
		}
	}

	void emit(const char* ev, int slot, int type, uint16_t tag, int from = -1, const unit_t* u = nullptr,
	          int x = -1, int y = -1) {
		char buf[200];
		int n = snprintf(buf, sizeof buf, "{\"type\":\"event\",\"frame\":%d,\"slot\":%d,\"ev\":\"%s\",\"unit\":%d,\"tag\":%u",
		                 st.current_frame, slot, ev, type, (unsigned)tag);
		if (from >= 0) n += snprintf(buf + n, sizeof buf - n, ",\"from\":%d", from);
		if (u && u->sprite) n += snprintf(buf + n, sizeof buf - n, ",\"x\":%d,\"y\":%d", u->sprite->position.x, u->sprite->position.y);
		else if (x >= 0) n += snprintf(buf + n, sizeof buf - n, ",\"x\":%d,\"y\":%d", x, y);
		snprintf(buf + n, sizeof buf - n, "}\n");
		fputs(buf, stdout);
	}

	void track_units() {
		for (auto& kv : known) kv.second.seen = false;
		for (int p = 0; p != 8; ++p) {
			for (unit_t* u : ptr(st.player_units[p])) {
				uint16_t tag = get_unit_id(u).raw_value;
				int type = (int)u->unit_type->id;
				bool done = u_completed(u);
				bool building = ut_building(u->unit_type);
				int ux = u->sprite ? u->sprite->position.x : -1, uy = u->sprite ? u->sprite->position.y : -1;
				auto it = known.find(tag);
				if (it == known.end()) {
					if (building) emit(done ? "done" : "start", p, type, tag, -1, u);  // done: starting buildings
					else if (done && !transient(type)) emit("done", p, type, tag, -1, u);
					known[tag] = {type, done, true, p, ux, uy};
					continue;
				}
				tracked& t = it->second;
				t.seen = true;
				t.slot = p;
				if (ux >= 0) { t.x = ux; t.y = uy; }
				if (t.type != type) {
					if (building) emit("start", p, type, tag, t.type, u);           // drone -> building, hatchery -> lair
					else if (done && transient(t.type) && !transient(type)) emit("done", p, type, tag, t.type, u);
					t.type = type;
					t.completed = done;
					if (building && done) emit("done", p, type, tag, -1, u);
					continue;
				}
				if (done && !t.completed && !transient(type)) emit("done", p, type, tag, -1, u);
				t.completed = done;
			}
		}
		for (auto it = known.begin(); it != known.end();) {
			if (!it->second.seen) {
				if (!transient(it->second.type)) emit("gone", it->second.slot, it->second.type, it->first, -1, nullptr,
				                                      it->second.x, it->second.y);
				it = known.erase(it);
			} else ++it;
		}
	}
};

bool is_player(const state& st, int i) {
	return st.players[i].controller == player_t::controller_occupied ||
	       st.players[i].controller == player_t::controller_user_left ||
	       st.players[i].controller == player_t::controller_computer_game;
}

constexpr int GRID = 16, GRID_CELLS = GRID * GRID, GRID_CHANNELS = 6;

void print_snapshot(const state& st, const replay_state& rst, counting_replay_functions& f) {
	// What each player can see of everyone else's units right now (fog of war).
	std::map<int, int> seen[8];
	for (int o = 0; o != 8; ++o) {
		for (const unit_t* u : ptr(st.player_units[o])) {
			if (!u->sprite || f.us_hidden(u)) continue;
			for (int p = 0; p != 8; ++p) {
				if (p != o && (u->sprite->visibility_flags & (1 << p))) ++seen[p][(int)u->unit_type->id];
			}
		}
	}
	// Where things are, per player, on a 16x16 grid over the map: own army supply (x2), workers,
	// buildings; then the same for other players' units this player can see.
	std::vector<int> grid(8 * GRID_CHANNELS * GRID_CELLS, 0);
	int mw = std::max(1, (int)f.game_st.map_width), mh = std::max(1, (int)f.game_st.map_height);
	for (int o = 0; o != 8; ++o) {
		for (const unit_t* u : ptr(st.player_units[o])) {
			if (!u->sprite || f.us_hidden(u)) continue;
			int cat, val = 1;
			if (f.ut_building(u->unit_type)) cat = 2;
			else if (f.ut_worker(u->unit_type)) cat = 1;
			else if (u->unit_type->supply_required.raw_value > 0) { cat = 0; val = u->unit_type->supply_required.raw_value; }
			else continue;  // overlords, larvae, eggs, ...
			int gx = std::min(GRID - 1, std::max(0, u->sprite->position.x * GRID / mw));
			int gy = std::min(GRID - 1, std::max(0, u->sprite->position.y * GRID / mh));
			int cell = gy * GRID + gx;
			grid[(o * GRID_CHANNELS + cat) * GRID_CELLS + cell] += val;
			for (int p = 0; p != 8; ++p) {
				if (p != o && (u->sprite->visibility_flags & (1 << p)))
					grid[(p * GRID_CHANNELS + 3 + cat) * GRID_CELLS + cell] += val;
			}
		}
	}
	std::string out = "{\"type\":\"snapshot\",\"frame\":" + std::to_string(st.current_frame) + ",\"players\":[";
	bool first_player = true;
	for (int p = 0; p != 8; ++p) {
		if (!is_player(st, p)) continue;
		if (!first_player) out += ',';
		first_player = false;
		int race = (int)st.players[p].race;
		int ri = race >= 0 && race < 3 ? race : 0;
		out += "{\"slot\":" + std::to_string(p) +
		       ",\"minerals\":" + std::to_string(st.current_minerals[p]) +
		       ",\"gas\":" + std::to_string(st.current_gas[p]) +
		       ",\"mined\":" + std::to_string(st.total_minerals_gathered[p]) +
		       ",\"mined_gas\":" + std::to_string(st.total_gas_gathered[p]) +
		       // supply is stored in half units (a zergling is 1)
		       ",\"supply_used\":" + std::to_string(st.supply_used[p][ri].raw_value / 2.0) +
		       ",\"supply_max\":" + std::to_string(st.supply_available[p][ri].raw_value / 2.0) +
		       ",\"accepted\":" + std::to_string(f.accepted[p]) +
		       ",\"rejected\":" + std::to_string(f.rejected[p]) + ",\"units\":{";
		bool first_unit = true;
		for (int id = 0; id != 228; ++id) {
			int all = st.unit_counts[p][(UnitTypes)id];
			if (!all) continue;
			int done = st.completed_unit_counts[p][(UnitTypes)id];
			if (!first_unit) out += ',';
			first_unit = false;
			out += "\"" + std::to_string(id) + "\":[" + std::to_string(all) + "," + std::to_string(done) + "]";
		}
		out += "},\"g\":[";
		bool first_cell = true;
		for (int i = 0; i != GRID_CHANNELS * GRID_CELLS; ++i) {
			int v = grid[p * GRID_CHANNELS * GRID_CELLS + i];
			if (!v) continue;
			if (!first_cell) out += ',';
			first_cell = false;
			out += std::to_string(i) + "," + std::to_string(v);
		}
		out += "],\"seen\":{";
		bool first_seen = true;
		for (auto& kv : seen[p]) {
			if (!first_seen) out += ',';
			first_seen = false;
			out += "\"" + std::to_string(kv.first) + "\":" + std::to_string(kv.second);
		}
		out += "}}";
		f.accepted[p] = f.rejected[p] = 0;
	}
	out += "]}\n";
	fputs(out.c_str(), stdout);
}


}  // namespace

int main(int argc, char** argv) {
	std::string data_dir, replay_file;
	bool flat = false;
	int unit_limit = 1700;
	double fights_s = 0;           // --fights SECONDS: fight snapshots and unit commands up to then
	int every = 24;
	for (int i = 1; i < argc; ++i) {
		if (!strcmp(argv[i], "--data") && i + 1 < argc) data_dir = argv[++i];
		else if (!strcmp(argv[i], "--replay") && i + 1 < argc) replay_file = argv[++i];
		else if (!strcmp(argv[i], "--every") && i + 1 < argc) every = std::max(0, atoi(argv[++i]));
		else if (!strcmp(argv[i], "--flat")) flat = true;
		else if (!strcmp(argv[i], "--unit-limit") && i + 1 < argc) unit_limit = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--fights") && i + 1 < argc) fights_s = atof(argv[++i]);
		else {
			fprintf(stderr, "usage: gary_resim --data <dir> --replay <file.rep> [--every <frames>]\n");
			return 2;
		}
	}
	if (data_dir.empty() || replay_file.empty()) {
		fprintf(stderr, "usage: gary_resim --data <dir> --replay <file.rep> [--every <frames>]\n");
		return 2;
	}

	try {
		game_player player{data_loader(data_dir)};
		action_state action_st;
		replay_state replay_st;
		counting_replay_functions f(player.st(), action_st, replay_st);
		f.unit_limit = unit_limit;
		f.fights_until = (int)(fights_s * 1000 / 42);
		// "--replay -" reads the replay from stdin. Callers use it for paths the Windows ANSI
		// command line can't carry (e.g. Korean map names in file names).
		std::vector<uint8_t> replay_bytes;
		if (replay_file == "-") {
#ifdef _WIN32
			_setmode(_fileno(stdin), _O_BINARY);
#endif
			char chunk[65536];
			size_t n;
			while ((n = fread(chunk, 1, sizeof chunk, stdin)) > 0) replay_bytes.insert(replay_bytes.end(), chunk, chunk + n);
		} else {
			std::ifstream in(replay_file, std::ios::binary);
			if (!in) error("can't open replay: %s", replay_file.c_str());
			replay_bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
		}
		// Bring every replay to the decoded stream ("reRS" | header | commands | map), so the map
		// can be adjusted before OpenBW loads it.
		std::vector<uint8_t> stream = flat ? replay_bytes : decode_legacy_replay(replay_bytes);
		disable_cosmetic_trigger_actions(stream);
		if (getenv("GARY_PRINT_HEADER")) {  // debug: the decoded 633-byte game info, hex
			for (size_t i = 4; i != 4 + 633; ++i) fprintf(stderr, "%02x", stream[i]);
			fputc('\n', stderr);
		}
		data_loading::data_reader_le r(stream.data(), stream.data() + stream.size());
		f.load_replay(r);
		const state& st = player.st();

		std::string header = "{\"type\":\"header\",\"end_frame\":" + std::to_string(replay_st.end_frame) +
		                     ",\"map\":\"" + json_escape(replay_st.map_name) + "\"" +
		                     ",\"map_w\":" + std::to_string(f.game_st.map_width) +
		                     ",\"map_h\":" + std::to_string(f.game_st.map_height) + ",\"players\":[";
		bool first = true;
		for (int p = 0; p != 8; ++p) {
			if (!is_player(st, p)) continue;
			if (!first) header += ',';
			first = false;
			header += "{\"slot\":" + std::to_string(p) + ",\"name\":\"" + json_escape(replay_st.player_name[p]) +
			          "\",\"race\":" + std::to_string((int)st.players[p].race) + "}";
		}
		header += "]}\n";
		fputs(header.c_str(), stdout);

		while (!f.is_done()) {
			f.next_frame_counted();
			if (every > 0 && st.current_frame % every == 0) print_snapshot(st, replay_st, f);
		}
		print_snapshot(st, replay_st, f);
		if (f.debug_rejects) f.print_object_usage();

		std::string end = "{\"type\":\"end\",\"frame\":" + std::to_string(st.current_frame) + ",\"total_rejected\":{";
		first = true;
		for (int p = 0; p != 8; ++p) {
			if (!is_player(st, p)) continue;
			if (!first) end += ',';
			first = false;
			end += "\"" + std::to_string(p) + "\":" + std::to_string(f.total_rejected[p]);
		}
		end += "}}\n";
		fputs(end.c_str(), stdout);
	} catch (const std::exception& e) {
		fprintf(stderr, "error: %s\n", e.what());
		return 1;
	}
	return 0;
}
