// gary_view: watch a replay in OpenBW's renderer, optionally from a player's point of view.
//
//   gary_view --data <dir> --replay <file.rep> [--pov <file.pov.jsonl>] [--size 640x400]
//             [--scale N] [--record out.mp4 [--from SECONDS] [--to SECONDS] [--speed N]]
//             [--flat --unit-limit N] [--subtitles file.ass]
//
// Remastered replays: run it through viewer/watch.py, which decodes the replay first (--flat
// takes the decoded stream, --unit-limit the game's unit table size).
//
// The game is drawn at --size (with --pov: the player's screen size from the log) and
// stretched to fit the window, keeping its shape; --scale sets the starting window size
// (default 2x). Maximize or resize the window freely.
//
// --record renders the game (as fast as possible, not in real time) into a video through
// ffmpeg, which must be on PATH. --speed N plays N game frames per video frame (default 1:
// real time at Fastest; 4 = a 4x time-lapse).
//
// With --pov (written by gary.interface.HumanInterface.save_pov), the view follows that
// player's camera frame by frame and draws their mouse: a cross for the cursor, and a ring
// that opens out where a click landed, green for a left click and red for a right click.
// (v1: no fog of war; you see everything inside the camera area.) With a POV log the picture is
// the player's whole screen: the game view, and below it a console strip with the minimap where
// the player's minimap is (clicks on it show there); the player's minerals, gas and supply are
// top-right. Without one, the minimap is bottom-left.
//
// Keys: space pauses, left/right seek 10 s (with shift: 60 s), up/down change speed,
// WASD moves the view when there's no POV log, Esc quits. The title bar shows the time.

#include <stdexcept>  // OpenBW's util.h uses std::runtime_error without including it

#include "ui.h"
#include "common.h"
#include "bwgame.h"
#include "replay.h"
#include "scr_replay.h"   // resim/: Remastered commands

#include "SDL.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace bwgame;

namespace bwgame {
namespace ui {
void log_str(a_string str) {
	fwrite(str.data(), str.size(), 1, stdout);
	fflush(stdout);
}
void fatal_error_str(a_string str) {
	fprintf(stderr, "fatal error: %s\n", str.c_str());
	std::terminate();
}
}  // namespace ui
}  // namespace bwgame

namespace {

bool file_exists(const std::string& path) {
	std::ifstream f(path, std::ios::binary);
	return f.good();
}

struct data_loader {
	std::shared_ptr<data_loading::data_files_loader<>> mpqs;
	std::string dir;
	explicit data_loader(std::string d) : dir(std::move(d)) {
		if (!dir.empty() && dir.back() != '/' && dir.back() != '\\') dir += '/';
		if (file_exists(dir + "StarDat.mpq"))
			mpqs = std::make_shared<data_loading::data_files_loader<>>(data_loading::data_files_directory(dir.c_str()));
	}
	void operator()(a_vector<uint8_t>& dst, a_string filename) {
		if (mpqs) {
			(*mpqs)(dst, std::move(filename));
			return;
		}
		std::string path = dir;
		for (char c : filename) path += (c == '\\') ? '/' : c;
		if (path.compare(dir.size(), 6, "sound/") == 0) {
			dst.clear();  // the viewer is silent: sound files aren't extracted
			return;
		}
		std::ifstream f(path, std::ios::binary | std::ios::ate);
		if (!f) error("data file not found: %s", path.c_str());
		dst.resize((size_t)f.tellg());
		f.seekg(0);
		f.read((char*)dst.data(), (std::streamsize)dst.size());
	}
};

// One line of a POV log: {"frame": F, "camera": [x, y], "cursor": [x, y], "click": "left"|"right"|null,
// "at": [x, y] (a minimap click: where on the screen), "sel": [tags] (the selection, when it changed),
// "cmd": {"kind": "move"|"attack"|"amove"|"gather"|"hold"|"stop"|..., "x": X, "y": Y, "target": tag}}
struct pov_entry {
	int frame;
	int cam_x, cam_y, cur_x, cur_y;
	int click;  // 0 none, 1 left, 2 right
	int at_x = -1, at_y = -1;
	bool has_sel = false;
	std::vector<int> sel;
	std::string cmd;  // the command's kind ("" none)
	int cmd_x = -1, cmd_y = -1, cmd_target = 0;
};

// The integers of the JSON array after "key" ([1, 2, 3]).
std::vector<int> json_ints_after(const std::string& line, const char* key) {
	std::vector<int> out;
	size_t p = line.find(key);
	if (p == std::string::npos) return out;
	p = line.find('[', p);
	size_t e = line.find(']', p);
	if (p == std::string::npos || e == std::string::npos) return out;
	for (size_t i = p + 1; i < e;) {
		while (i < e && (line[i] == ' ' || line[i] == ',')) ++i;
		if (i >= e) break;
		out.push_back(atoi(line.c_str() + i));
		while (i < e && line[i] != ',') ++i;
	}
	return out;
}

int json_int_after(const std::string& line, const char* key, int index = 0) {
	size_t p = line.find(key);
	if (p == std::string::npos) return 0;
	p += strlen(key);
	for (int i = 0; i != index; ++i) p = line.find(',', p) + 1;
	while (p < line.size() && (line[p] == ' ' || line[p] == '[' || line[p] == ':')) ++p;
	return atoi(line.c_str() + p);
}

// The string value of "key" ("" for null or missing), whatever the spacing.
std::string json_str_after(const std::string& line, const char* key) {
	size_t p = line.find(key);
	if (p == std::string::npos) return "";
	p = line.find_first_not_of(" :", p + strlen(key));
	if (p == std::string::npos || line[p] != '"') return "";
	size_t e = line.find('"', p + 1);
	return e == std::string::npos ? "" : line.substr(p + 1, e - p - 1);
}

struct pov_log {
	std::vector<pov_entry> entries;
	int view_w = 0, view_h = 0;
	int slot = -1;  // whose minerals, gas and supply to show
	int minimap_x = -1, minimap_y = -1;  // where the player's minimap is on the screen
};

pov_log load_pov(const std::string& path) {
	pov_log out;
	std::ifstream in(path);
	if (!in) error("can't open %s", path.c_str());
	std::string line;
	while (std::getline(in, line)) {
		if (line.find("\"viewport\"") != std::string::npos) {
			out.view_w = json_int_after(line, "\"viewport\"", 0);
			out.view_h = json_int_after(line, "\"viewport\"", 1);
			if (line.find("\"slot\"") != std::string::npos) out.slot = json_int_after(line, "\"slot\"");
			if (line.find("\"minimap\"") != std::string::npos) {
				out.minimap_x = json_int_after(line, "\"minimap\"", 0);
				out.minimap_y = json_int_after(line, "\"minimap\"", 1);
			}
		}
		if (line.find("\"frame\"") == std::string::npos) continue;
		pov_entry e{};
		e.frame = json_int_after(line, "\"frame\"");
		e.cam_x = json_int_after(line, "\"camera\"", 0);
		e.cam_y = json_int_after(line, "\"camera\"", 1);
		e.cur_x = json_int_after(line, "\"cursor\"", 0);
		e.cur_y = json_int_after(line, "\"cursor\"", 1);
		std::string click = json_str_after(line, "\"click\"");
		e.click = click == "left" ? 1 : click == "right" ? 2 : 0;
		if (line.find("\"at\"") != std::string::npos) {
			e.at_x = json_int_after(line, "\"at\"", 0);
			e.at_y = json_int_after(line, "\"at\"", 1);
		}
		if (line.find("\"sel\"") != std::string::npos) {
			e.has_sel = true;
			e.sel = json_ints_after(line, "\"sel\"");
		}
		size_t c = line.find("\"cmd\"");
		if (c != std::string::npos) {
			std::string rest = line.substr(c);
			e.cmd = json_str_after(rest, "\"kind\"");
			e.cmd_x = json_int_after(rest, "\"x\"");
			e.cmd_y = json_int_after(rest, "\"y\"");
			e.cmd_target = json_int_after(rest, "\"target\"");
		}
		out.entries.push_back(e);
	}
	return out;
}

constexpr int click_mark_frames = 24;  // a click stays marked for a second of game time
constexpr int command_frames = 36;     // a command's line to its target stays a second and a half
constexpr int console_h = 80;          // the console strip below a 640x400 view (a 640x480 screen)

// A 3x5 pixel font for the resource counts: digits and '/', one row per byte (3 low bits).
const uint8_t* glyph(char c) {
	static const uint8_t digits[10][5] = {
	    {7, 5, 5, 5, 7}, {2, 6, 2, 2, 7}, {7, 1, 7, 4, 7}, {7, 1, 7, 1, 7}, {5, 5, 7, 1, 1},
	    {7, 4, 7, 1, 7}, {7, 4, 7, 5, 7}, {7, 1, 1, 1, 1}, {7, 5, 7, 5, 7}, {7, 5, 7, 1, 7}};
	static const uint8_t slash[5] = {1, 1, 2, 4, 4};
	if (c >= '0' && c <= '9') return digits[c - '0'];
	return c == '/' ? slash : nullptr;
}

struct pov_ui : scr_replay<ui_functions> {
	std::vector<pov_entry> pov;
	// The cursor in screen pixels; recent clicks in map pixels, so they stay on the right spot
	// whatever the view (minimap clicks in screen pixels: on_screen).
	int cursor_x = -1, cursor_y = -1;
	int cursor_click = 0;
	struct mark { int x, y, kind, age; bool on_screen; };
	std::vector<mark> marks;
	int view_h = 0;                         // the game view's height; below it, the console strip
	int minimap_x = -1, minimap_y = -1;
	std::vector<int> selection;             // the player's selected units (tags), as the POV log says
	struct order_mark { std::string kind; int x, y, target, age; std::vector<int> units; };
	std::vector<order_mark> orders;         // recent commands to the selection

	using scr_replay<ui_functions>::scr_replay;

	// Show the latest POV entry at or before the current frame (works after seeking too).
	void follow() {
		if (pov.empty()) return;
		int now = st.current_frame;
		auto it = std::upper_bound(pov.begin(), pov.end(), now, [](int f, const pov_entry& e) { return f < e.frame; });
		if (it == pov.begin()) return;
		size_t i = (it - pov.begin()) - 1;
		const auto& e = pov[i];
		screen_pos = xy(e.cam_x, e.cam_y);
		cursor_x = e.cur_x;
		cursor_y = e.cur_y;
		cursor_click = 0;
		marks.clear();
		selection.clear();
		for (size_t j = i + 1; j-- > 0;) {
			if (pov[j].has_sel) {
				selection = pov[j].sel;
				break;
			}
		}
		orders.clear();
		{
			std::vector<int> sel_then = selection;  // the selection as it was at each command
			for (size_t j = i + 1; j-- > 0 && pov[j].frame > now - command_frames;) {
				if (!pov[j].cmd.empty()) orders.push_back({pov[j].cmd, pov[j].cmd_x, pov[j].cmd_y, pov[j].cmd_target,
				                                            now - pov[j].frame, sel_then});
				if (pov[j].has_sel) {
					for (size_t k = j; k-- > 0;) {
						if (pov[k].has_sel) { sel_then = pov[k].sel; break; }
						if (k == 0) sel_then.clear();
					}
				}
			}
		}
		for (size_t j = i + 1; j-- > 0 && pov[j].frame > now - click_mark_frames;) {
			const auto& c = pov[j];
			if (!c.click) continue;
			bool minimap = c.at_x >= 0;
			marks.push_back({minimap ? c.at_x : c.cam_x + c.cur_x, minimap ? c.at_y : c.cam_y + c.cur_y, c.click,
			                 now - c.frame, minimap});
			if (now - c.frame < 6 && !cursor_click) {
				cursor_click = c.click;
				if (minimap) cursor_x = c.at_x, cursor_y = c.at_y;   // the hand is on the minimap
			}
		}
	}

	uint8_t nearest_color(int r, int g, int b) {
		const auto& wpe = tileset_img.wpe;
		int best = 0, best_score = 1 << 30;
		for (int i = 0; i != 256; ++i) {
			int dr = r - wpe[4 * i], dg = g - wpe[4 * i + 1], db = b - wpe[4 * i + 2];
			int score = dr * dr + dg * dg + db * db;
			if (score < best_score) best = i, best_score = score;
		}
		return (uint8_t)best;
	}

	int slot = -1;

	// The POV player's minerals, gas and supply, top-right: a colored square, then the number.
	void draw_resources(uint8_t* data, size_t pitch) {
		if (slot < 0 || slot >= 8) return;
		auto put = [&](int x, int y, uint8_t c) {
			if (x >= 0 && y >= 0 && x < (int)screen_width && y < (int)screen_height) data[y * pitch + x] = c;
		};
		int race = (int)st.players[slot].race;
		int ri = race >= 0 && race < 3 ? race : 0;
		int used = (st.supply_used[slot][ri].raw_value + 1) / 2;
		int avail = std::min(200, st.supply_available[slot][ri].raw_value / 2);
		struct item { uint8_t color; std::string text; };
		const item items[] = {{nearest_color(80, 200, 255), std::to_string(st.current_minerals[slot])},
		                      {nearest_color(40, 220, 80), std::to_string(st.current_gas[slot])},
		                      {nearest_color(255, 255, 255), std::to_string(used) + "/" + std::to_string(avail)}};
		const int k = 2;  // font scale
		int width = 0;
		for (auto& it : items) width += 6 * k + 4 + (int)it.text.size() * 4 * k + 8;
		int x = (int)screen_width - width - 4, y = 4;
		const uint8_t black = nearest_color(0, 0, 0), white = nearest_color(255, 255, 255);
		for (int yy = y - 2; yy != y + 5 * k + 2; ++yy)
			for (int xx = x - 4; xx != (int)screen_width - 2; ++xx) put(xx, yy, black);
		for (auto& it : items) {
			for (int yy = 0; yy != 5 * k; ++yy)
				for (int xx = 0; xx != 5 * k; ++xx) put(x + xx, y + yy, it.color);
			x += 6 * k + 4;
			for (char c : it.text) {
				if (const uint8_t* g = glyph(c)) {
					for (int r = 0; r != 5; ++r)
						for (int b = 0; b != 3; ++b)
							if (g[r] >> (2 - b) & 1)
								for (int dy = 0; dy != k; ++dy)
									for (int dx = 0; dx != k; ++dx) put(x + b * k + dx, y + r * k + dy, white);
				}
				x += 4 * k;
			}
			x += 8;
		}
	}

	// The console strip below the game view, with the minimap where the player has it.
	void draw_console(uint8_t* data, size_t pitch) {
		auto area = get_minimap_area();
		if (view_h <= 0 || view_h >= (int)screen_height) {
			draw_minimap(data, pitch);              // no POV: OpenBW's place, bottom-left
			return;
		}
		const uint8_t dark = nearest_color(24, 24, 32);
		for (int y = view_h; y != (int)screen_height; ++y) memset(data + y * pitch, dark, screen_width);
		int dx = minimap_x >= 0 ? minimap_x - area.from.x : 0, dy = minimap_y >= 0 ? minimap_y - area.from.y : 0;
		if (area.to.x + dx > (int)screen_width || area.to.y + dy > (int)screen_height || area.from.x + dx < 1 ||
		    area.from.y + dy < 1) dx = dy = 0;
		draw_minimap(data + dy * (ptrdiff_t)pitch + dx, pitch);
	}

	void draw_callback(uint8_t* data, size_t pitch) override {
		draw_console(data, pitch);
		draw_resources(data, pitch);
		if (pov.empty() || cursor_x < 0) return;
		auto put = [&](int x, int y, uint8_t c) {
			if (x >= 0 && y >= 0 && x < (int)screen_width && y < (int)screen_height) data[y * pitch + x] = c;
		};
		const uint8_t white = nearest_color(255, 255, 255), black = nearest_color(0, 0, 0);
		const uint8_t green = nearest_color(40, 255, 40), red = nearest_color(255, 40, 40);
		const uint8_t orange = nearest_color(255, 150, 30), yellow = nearest_color(255, 230, 60);
		auto ellipse = [&](int cx, int cy, int rx, int ry, uint8_t c) {
			for (int a = 0; a != 120; ++a) {
				double t = a * 3.14159265 / 60;
				put(cx + (int)lround(rx * cos(t)), cy + (int)lround(ry * sin(t)), c);
			}
		};
		auto unit_screen = [&](int tag, int& sx, int& sy, int& r) {
			unit_t* u = get_unit(unit_id((uint16_t)tag));
			if (!u || !u->sprite || us_hidden(u)) return false;
			sx = u->sprite->position.x - screen_pos.x;
			sy = u->sprite->position.y - screen_pos.y;
			auto& d = u->unit_type->dimensions;
			r = std::max(6, (d.from.x + d.to.x) / 2 + 3);
			return true;
		};
		// The selection: a green ring under each selected unit, as in the game.
		for (int tag : selection) {
			int sx, sy, r;
			if (unit_screen(tag, sx, sy, r)) ellipse(sx, sy + r / 3, r, r * 2 / 3, green);
		}
		// Recent commands: a line from the units commanded to where they were sent (green move, red
		// attack a unit, orange attack-move, yellow gather), a ring on a target unit; hold/stop: a mark.
		for (auto& o : orders) {
			uint8_t c = o.kind == "move" ? green : o.kind == "attack" ? red : o.kind == "amove" ? orange :
			            o.kind == "gather" ? yellow : white;
			long ux = 0, uy = 0;
			int n = 0;
			for (int tag : o.units) {
				int sx, sy, r;
				if (unit_screen(tag, sx, sy, r)) ux += sx, uy += sy, ++n;
			}
			if (o.kind == "hold" || o.kind == "stop") {
				if (n) {
					int x = (int)(ux / n), y = (int)(uy / n) - 14;
					for (int d = -4; d <= 4; ++d) put(x + d, y - 4, c), put(x + d, y + 4, c), put(x - 4, y + d, c), put(x + 4, y + d, c);
				}
				continue;
			}
			int tx = o.x - screen_pos.x, ty = o.y - screen_pos.y, tr = 0;
			if (o.target) {
				int sx, sy;
				if (unit_screen(o.target, sx, sy, tr)) tx = sx, ty = sy;
				ellipse(tx, ty, tr + 2, tr + 2, c);
			} else {
				for (int d = -3; d <= 3; ++d) put(tx + d, ty + d, c), put(tx + d, ty - d, c);
			}
			if (n) {
				int x0 = (int)(ux / n), y0 = (int)(uy / n);
				int steps = std::max(std::abs(tx - x0), std::abs(ty - y0));
				for (int k = 0; k <= steps; k += 1)
					if ((k / 4) % 2 == 0) put(x0 + (tx - x0) * k / std::max(1, steps), y0 + (ty - y0) * k / std::max(1, steps), c);
			}
		}
		// Clicks: a ring that opens from 4 to 20 pixels over the mark's life, 2 pixels thick.
		for (auto& m : marks) {
			int sx = m.on_screen ? m.x : m.x - screen_pos.x, sy = m.on_screen ? m.y : m.y - screen_pos.y;
			double r = 4 + 16.0 * m.age / click_mark_frames;
			uint8_t c = m.kind == 1 ? green : red;
			for (int a = 0; a != 96; ++a) {
				double t = a * 3.14159265 / 48;
				for (double rr : {r, r + 1}) put(sx + (int)lround(rr * cos(t)), sy + (int)lround(rr * sin(t)), c);
			}
		}
		// Cursor: a cross with a dark outline, in the click color right after a click.
		int cx = cursor_x, cy = cursor_y;
		uint8_t c = cursor_click == 1 ? green : cursor_click == 2 ? red : white;
		for (int d = -9; d <= 9; ++d) {
			for (int o : {-1, 1}) {
				put(cx + d, cy + o, black);
				put(cx + o, cy + d, black);
			}
		}
		for (int d = -8; d <= 8; ++d) {
			put(cx + d, cy, c);
			put(cx, cy + d, c);
		}
	}
};

// Seeking backwards replays from the nearest snapshot (one every 10 s of game time).
struct snapshots {
	struct saved {
		state st;
		action_state action_st;
	};
	std::map<int, std::unique_ptr<saved>> by_frame;
	static constexpr int interval = 10 * 1000 / 42;

	void maybe_save(pov_ui& ui) {
		int f = ui.st.current_frame;
		if (f % interval || by_frame.count(f)) return;
		auto v = std::make_unique<saved>();
		v->st = copy_state(ui.st);
		v->action_st = copy_state(ui.action_st, ui.st, v->st);
		by_frame[f] = std::move(v);
	}

	void advance(pov_ui& ui) {
		maybe_save(ui);
		ui.scr_next_frame();
	}

	void seek(pov_ui& ui, int target) {
		target = std::max(0, std::min(target, (int)ui.replay_st.end_frame));
		auto i = by_frame.upper_bound(target);
		if (i != by_frame.begin()) {
			--i;
			if (target < ui.st.current_frame || i->first > ui.st.current_frame) {
				ui.st = copy_state(i->second->st);
				ui.action_st = copy_state(i->second->action_st, i->second->st, ui.st);
			}
		}
		while (ui.st.current_frame < target && !ui.is_done()) advance(ui);
	}
};

std::string clock_str(int frame) {
	int s = frame * 42 / 1000;
	char buf[16];
	snprintf(buf, sizeof buf, "%d:%02d", s / 60, s % 60);
	return buf;
}

}  // namespace

int run(int argc, char** argv);

int main(int argc, char** argv) {
	try {
		return run(argc, argv);
	} catch (const std::exception& e) {
		fprintf(stderr, "error: %s\n", e.what());
		return 1;
	}
}

int run(int argc, char** argv) {
	std::string data_dir, replay_file, pov_file, record_file, subtitles_file;
	int width = 0, height = 0, scale = 2;
	double from_s = 0, to_s = 1e9;
	int speed = 1;
	bool flat = false;          // the replay file is already OpenBW's decoded stream (viewer/watch.py)
	int unit_limit = 1700;
	for (int i = 1; i < argc; ++i) {
		if (!strcmp(argv[i], "--data") && i + 1 < argc) data_dir = argv[++i];
		else if (!strcmp(argv[i], "--replay") && i + 1 < argc) replay_file = argv[++i];
		else if (!strcmp(argv[i], "--pov") && i + 1 < argc) pov_file = argv[++i];
		else if (!strcmp(argv[i], "--size") && i + 1 < argc) sscanf(argv[++i], "%dx%d", &width, &height);
		else if (!strcmp(argv[i], "--scale") && i + 1 < argc) scale = std::max(1, atoi(argv[++i]));
		else if (!strcmp(argv[i], "--record") && i + 1 < argc) record_file = argv[++i];
		else if (!strcmp(argv[i], "--subtitles") && i + 1 < argc) subtitles_file = argv[++i];
		else if (!strcmp(argv[i], "--from") && i + 1 < argc) from_s = atof(argv[++i]);
		else if (!strcmp(argv[i], "--to") && i + 1 < argc) to_s = atof(argv[++i]);
		else if (!strcmp(argv[i], "--speed") && i + 1 < argc) speed = std::max(1, atoi(argv[++i]));
		else if (!strcmp(argv[i], "--flat")) flat = true;
		else if (!strcmp(argv[i], "--unit-limit") && i + 1 < argc) unit_limit = atoi(argv[++i]);
	}
	if (data_dir.empty() || replay_file.empty()) {
		fprintf(stderr, "usage: gary_view --data <dir> --replay <file.rep> [--pov <file.pov.jsonl>] [--size WxH] [--scale N]\n");
		return 2;
	}
	pov_log pov;
	if (!pov_file.empty()) pov = load_pov(pov_file);
	if (width <= 0 || height <= 0) {
		width = pov.view_w > 0 ? pov.view_w : 640;
		height = pov.view_h > 0 ? pov.view_h + console_h : 400;   // the player's screen: view and console
	}

	data_loader loader(data_dir);
	game_player player(loader);
	pov_ui ui(std::move(player));
	ui.create_window = false;  // OpenBW draws off-screen; we scale it into our own window
	ui.draw_ui_elements = false;
	ui.load_all_image_data(loader);
	ui.load_data_file = [&](a_vector<uint8_t>& data, a_string filename) { loader(data, std::move(filename)); };
	ui.init();
	ui.unit_limit = unit_limit;
	if (flat) {
		std::ifstream in(replay_file, std::ios::binary);
		std::vector<uint8_t> stream((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
		if (stream.empty()) error("can't read %s", replay_file.c_str());
		ui.load_replay(data_loading::data_reader_le(stream.data(), stream.data() + stream.size()));
	} else {
		ui.load_replay_file(replay_file.c_str());
	}
	ui.pov = std::move(pov.entries);
	ui.slot = pov.slot;
	ui.view_h = pov.view_h;
	ui.minimap_x = pov.minimap_x;
	ui.minimap_y = pov.minimap_y;
	ui.resize(width, height);
	ui.screen_pos = {0, 0};
	ui.set_image_data();

	if (!record_file.empty()) {
		// Pipe raw frames into ffmpeg: H.264, ~24 frames per second like the game at Fastest.
		// Scaled up 2x with crisp pixels; --subtitles burns in a subtitle file on top (viewer/watch.py
		// writes one with Gary's actions, like a screencast-keys overlay).
		std::string vf = "scale=iw*2:ih*2:flags=neighbor";
		if (!subtitles_file.empty()) {
			std::string path;
			for (char c : subtitles_file) {
				if (c == '\\') path += '/';
				else if (c == ':') path += "\\:";
				else if (c == '\'') path += "\\'";
				else path += c;
			}
			vf += ",subtitles='" + path + "'";
		}
		char cmd[2048];
		snprintf(cmd, sizeof cmd,
		         "ffmpeg -loglevel error -y -f rawvideo -pix_fmt rgba -s %dx%d -r 24 -i - "
		         "-vf \"%s\" -c:v libx264 -pix_fmt yuv420p -crf 20 \"%s\"", width, height, vf.c_str(),
		         record_file.c_str());
#ifdef _WIN32
		FILE* out = _popen(cmd, "wb");
#else
		FILE* out = popen(cmd, "w");
#endif
		if (!out) {
			fprintf(stderr, "can't start ffmpeg\n");
			return 1;
		}
		int first = (int)(from_s * 1000 / 42), last = (int)(to_s * 1000 / 42);
		while (!ui.is_done() && ui.st.current_frame < last) {
			for (int i = 0; i != speed && !ui.is_done(); ++i) ui.scr_next_frame();
			ui.replay_frame = ui.st.current_frame;
			if (ui.st.current_frame < first) continue;
			ui.follow();
			ui.update();
			const uint8_t* px = (const uint8_t*)ui.rgba_surface->lock();
			for (int y = 0; y != height; ++y) fwrite(px + (size_t)y * ui.rgba_surface->pitch, 4, width, out);
			ui.rgba_surface->unlock();
		}
#ifdef _WIN32
		_pclose(out);
#else
		pclose(out);
#endif
		printf("wrote %s\n", record_file.c_str());
		return 0;
	}

	if (SDL_Init(SDL_INIT_VIDEO) != 0) error("SDL_Init failed: %s", SDL_GetError());
	SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "nearest");  // crisp pixels when scaled up
	SDL_Window* window = SDL_CreateWindow("Gary viewer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
	                                      width * scale, height * scale, SDL_WINDOW_RESIZABLE);
	if (!window) error("SDL_CreateWindow failed: %s", SDL_GetError());
	SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_PRESENTVSYNC);
	if (!renderer) renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_SOFTWARE);
	if (!renderer) error("SDL_CreateRenderer failed: %s", SDL_GetError());
	SDL_RenderSetLogicalSize(renderer, width, height);  // letterboxed, keeps the shape
	SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, width, height);

	snapshots snaps;
	bool paused = false, quit = false;
	int game_speed = 1;  // game frames per 42 ms tick
	const auto tick = std::chrono::milliseconds(42);  // Fastest
	auto last_tick = std::chrono::steady_clock::now();
	std::string last_title;
	while (!quit) {
		SDL_Event ev;
		while (SDL_PollEvent(&ev)) {
			if (ev.type == SDL_QUIT) quit = true;
			if (ev.type != SDL_KEYDOWN) continue;
			bool shift = (ev.key.keysym.mod & KMOD_SHIFT) != 0;
			int step = (shift ? 60 : 10) * 1000 / 42;
			switch (ev.key.keysym.sym) {
			case SDLK_ESCAPE: quit = true; break;
			case SDLK_SPACE: paused = !paused; break;
			case SDLK_LEFT: snaps.seek(ui, ui.st.current_frame - step); break;
			case SDLK_RIGHT: snaps.seek(ui, ui.st.current_frame + step); break;
			case SDLK_UP: game_speed = std::min(16, game_speed * 2); break;
			case SDLK_DOWN: game_speed = std::max(1, game_speed / 2); break;
			}
		}
		if (ui.pov.empty()) {
			const Uint8* keys = SDL_GetKeyboardState(nullptr);
			int d = 12;
			if (keys[SDL_SCANCODE_A]) ui.screen_pos.x -= d;
			if (keys[SDL_SCANCODE_D]) ui.screen_pos.x += d;
			if (keys[SDL_SCANCODE_W]) ui.screen_pos.y -= d;
			if (keys[SDL_SCANCODE_S]) ui.screen_pos.y += d;
		}

		auto now = std::chrono::steady_clock::now();
		if (!paused && !ui.is_done()) {
			int n = 0;
			while (now - last_tick >= tick && n++ < 8 && !ui.is_done()) {
				for (int i = 0; i != game_speed && !ui.is_done(); ++i) snaps.advance(ui);
				last_tick += tick;
			}
			if (n >= 8) last_tick = now;
		} else {
			last_tick = now;
		}
		ui.replay_frame = ui.st.current_frame;
		ui.follow();
		ui.update();

		void* px = ui.rgba_surface->lock();
		SDL_UpdateTexture(texture, nullptr, px, ui.rgba_surface->pitch);
		ui.rgba_surface->unlock();
		SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
		SDL_RenderClear(renderer);
		SDL_RenderCopy(renderer, texture, nullptr, nullptr);
		SDL_RenderPresent(renderer);

		std::string title = "Gary viewer  " + clock_str(ui.st.current_frame) + " / " + clock_str(ui.replay_st.end_frame) +
		                    (game_speed > 1 ? "  x" + std::to_string(game_speed) : "") +
		                    (paused ? "  [paused]" : ui.is_done() ? "  [end]" : "");
		if (title != last_title) {
			SDL_SetWindowTitle(window, title.c_str());
			last_title = title;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
	}
	SDL_DestroyTexture(texture);
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();
	return 0;
}
