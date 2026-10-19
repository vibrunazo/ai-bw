// Playing Remastered-era replays in OpenBW, shared by gary_resim and gary_view.
//
// The replay must first be brought to OpenBW's decoded stream (resim/scr_format.py: to_flat), which
// handles the file format and the map. What's left is in the commands, handled here:
//   - 1.21 added variants of six commands with an extra (always zero) 16-bit field after each unit
//     ID; they're translated to the legacy commands OpenBW implements.
//   - Recent games use a 3400-unit table (the replay's LMTS section): unit IDs then have a 13-bit
//     index, and the same unit's index is (limit - 1700) higher than in OpenBW's 1700-unit table.
//   - Observers have player IDs OpenBW doesn't know (128+); their commands are skipped or run as
//     the neutral player so the command stream stays aligned.
//
// Use as a base: struct my_functions : scr_replay<replay_functions> { ... }, then call
// scr_next_frame() (or execute_actions_scr() yourself) instead of replay_functions::next_frame().

#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

template<typename Base>
struct scr_replay : Base {
	using Base::Base;

	int unit_limit = 1700;
	bool debug_rejects = getenv("GARY_DEBUG_REJECTS") != nullptr;
	// Players (bit per slot) whose recorded commands are dropped, e.g. because someone else took
	// over their side mid-replay. To keep the stream aligned they still run, as dropped_as: an
	// empty slot, which owns no units, so selecting and ordering do nothing.
	uint32_t dropped_owners = 0;
	int dropped_as = 7;
	// The last executed command in legacy form (player id byte first), e.g. to save it again.
	std::vector<uint8_t> last_action;

	uint16_t translate_unit_id(uint16_t raw) {
		if (unit_limit <= 1700) return raw;
		int index1 = raw & 0x1fff;  // index + 1; 0 = no unit
		int generation = raw >> 13;
		if (!index1) return 0;
		// Both engines hand the first unit created (the map's first UNIT entry) slot 0 and then
		// allocate from the top of the table down, so an index in the bottom half is the same in
		// both; the top half is shifted by the extra slots. (KnockOut's first entry is a main's
		// geyser: every command aimed at it was lost, and 29 of 29 desyncs there followed.)
		int shift = unit_limit - 1700;
		int sim_index1 = index1 <= shift ? index1 : index1 - shift;
		bwgame::unit_t* u = this->get_unit((size_t)(sim_index1 - 1));
		if (!u || (int)(u->unit_id_generation % 8) != generation) {
			if (debug_rejects)
				fprintf(stderr, "unit id miss frame %d raw %u index1 %d gen %d sim %s type %d gen %d" "\n",
				        this->st.current_frame, (unsigned)raw, index1, generation, u ? "unit" : "none",
				        u ? (int)u->unit_type->id : -1, u ? (int)(u->unit_id_generation % 8) : -1);
			return 0x7ff;  // no such unit
		}
		return this->get_unit_id(u).raw_value;
	}

	// Advances past a 1.21 command without executing it.
	void skip_action_121(bwgame::data_loading::data_reader_le& r, const uint8_t* end) {
		auto need = [&](size_t n) { if (r.ptr + n > end) bwgame::error("truncated 1.21 action"); };
		need(2);
		r.get<uint8_t>();
		int id = r.get<uint8_t>();
		size_t n = id == 0x60 ? 11 : id == 0x61 ? 12 : id == 0x62 ? 4 : 0;
		if (id >= 0x63) { need(1); n = (size_t)r.get<uint8_t>() * 4; }
		need(n);
		r.ptr += n;
	}

	bool read_action_121(bwgame::data_loading::data_reader_le& r, const uint8_t* end) {
		auto u8 = [&]() { if (r.ptr + 1 > end) bwgame::error("truncated 1.21 action"); return r.get<uint8_t>(); };
		auto u16 = [&]() { if (r.ptr + 2 > end) bwgame::error("truncated 1.21 action"); return r.get<uint16_t>(); };
		std::vector<uint8_t> out;
		auto put8 = [&](int v) { out.push_back((uint8_t)v); };
		auto put16 = [&](int v) { out.push_back((uint8_t)(v & 0xff)); out.push_back((uint8_t)((v >> 8) & 0xff)); };
		put8(u8());  // player id
		int id = u8();
		switch (id) {
		case 0x60:  // right click: x, y, target, (0), unit type, queued
		case 0x61: {  // targeted order: x, y, target, (0), unit type, order, queued
			put8(id == 0x60 ? 0x14 : 0x15);
			put16(u16()); put16(u16()); put16(translate_unit_id(u16())); u16(); put16(u16());
			if (id == 0x61) put8(u8());
			put8(u8());
			break;
		}
		case 0x62:  // unload: unit, (0)
			put8(0x29); put16(translate_unit_id(u16())); u16();
			break;
		default: {  // 0x63 select, 0x64 select add, 0x65 select remove: count, (unit, (0)) * count
			put8(id == 0x63 ? 0x09 : id == 0x64 ? 0x0a : 0x0b);
			int n = u8();
			put8(n);
			for (int i = 0; i != n; ++i) { put16(translate_unit_id(u16())); u16(); }
		}
		}
		last_action = out;
		return this->read_action(out.data(), out.size());
	}

	// Runs this frame's replay commands. on_action(owner, action_id, command bytes, end, accepted)
	// is called for every player's command (not observers' or dropped players'), after it ran;
	// last_action then holds it in legacy form.
	template<typename F>
	void execute_actions_scr(F&& on_action) {
		auto& st = this->st;
		auto& action_st = this->action_st;
		uint8_t* begin = this->replay_st.actions_data_buffer.data();
		uint8_t* end_all = begin + this->replay_st.actions_data_buffer.size();
		if (st.current_frame != action_st.next_action_frame) return;
		while (action_st.actions_data_position != size_t(end_all - begin)) {
			bwgame::data_loading::data_reader_le r(begin + action_st.actions_data_position, end_all);
			int frame = r.get<int32_t>();
			if (frame != st.current_frame) {
				action_st.next_action_frame = frame;
				return;
			}
			size_t actions_size = r.get<uint8_t>();
			const uint8_t* ptr = r.get_n(actions_size);
			const uint8_t* end = ptr + actions_size;
			bwgame::data_loading::data_reader_le r2(ptr, end);
			while (r2.ptr != end) {
				int player_id = r2.ptr[0];
				auto i = std::find(action_st.player_id.begin(), action_st.player_id.end(), player_id);
				int owner = i == action_st.player_id.end() ? -1 : (int)(i - action_st.player_id.begin());
				bool dropped = owner >= 0 && (dropped_owners >> owner) & 1;
				if (owner < 0 || dropped) {
					if (r2.ptr + 1 < end && r2.ptr[1] >= 0x60 && r2.ptr[1] <= 0x65) {
						skip_action_121(r2, end);
					} else {
						r2.get<uint8_t>();
						this->read_action(dropped ? dropped_as : 11, r2);
					}
					continue;
				}
				int action_id = r2.ptr + 1 < end ? r2.ptr[1] : -1;
				const uint8_t* cmd = r2.ptr;
				bool ok;
				if (action_id >= 0x60 && action_id <= 0x65) {
					ok = read_action_121(r2, end);
				} else {
					ok = this->read_action(r2);
					last_action.assign(cmd, r2.ptr);
				}
				on_action(owner, action_id, cmd, end, ok);
			}
			action_st.actions_data_position = end - begin;
		}
	}

	void scr_next_frame() {
		if (this->st.current_frame == this->replay_st.end_frame) bwgame::error("replay: attempt to play past end");
		execute_actions_scr([](int, int, const uint8_t*, const uint8_t*, bool) {});
		this->state_functions::next_frame();
	}
};
