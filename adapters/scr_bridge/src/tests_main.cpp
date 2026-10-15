// Offline unit tests for the parts of gary_scr that do not need the game: command translation
// (the inverse of resim/gary_resim.cpp's translator), the handle table, and the framing/JSON
// helpers. Run: build\gary_scr_tests.exe   (exit code 0 = all pass)

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "commands.h"
#include "handles.h"
#include "json_min.h"
#include "pipe_server.h"

using namespace garyscr;

static int failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);           \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

static bool eq(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) { return a == b; }

static void test_handles() {
    // classic table: the handle equals Gary's tag (resim/gary_resim.cpp: limit <= 1700)
    UnitRef u;
    u.ptr = 0x1000;
    u.index1 = 5;
    u.generation = 3;
    CHECK(tag_of(u, 1700) == (uint16_t)(5 | (3 << 11)));
    CHECK(scr_id_of(u, 1700) == tag_of(u, 1700));
    // 3400-unit table: 13-bit index, 3-bit generation (fits u16; id == tag here too)
    u.index1 = 3000;
    u.generation = 11;
    CHECK(scr_id_of(u, 3400) == (uint16_t)(3000u | ((11u % 8) << 13)));
    CHECK(scr_id_of(u, 3400) == tag_of(u, 3400));
    CHECK(scr_id_of(u, 3400) <= 0xffff);
    uint32_t index1;
    uint8_t gen;
    CHECK(handle_split(tag_of(u, 3400), 3400, &index1, &gen));
    CHECK(index1 == 3000 && gen == 11 % 8);
    CHECK(handle_split(tag_of(u, 1700), 1700, &index1, &gen));
    CHECK(index1 == (3000 & 0x7ff) && gen == 11 % 32);
    CHECK(!handle_split(0, 1700, &index1, &gen));
}

static void test_lengths() {
    size_t len = 0;
    const uint8_t sel[] = {0x09, 0x02, 0x01, 0x00, 0x02, 0x00};
    CHECK(legacy_record_length(sel, sizeof sel, &len) && len == 6);
    const uint8_t build[] = {0x0c, 0x1e, 0x04, 0x00, 0x05, 0x00, 0x6d, 0x00};
    CHECK(legacy_record_length(build, sizeof build, &len) && len == 8);
    const uint8_t train[] = {0x1f, 0x00, 0x00};
    CHECK(legacy_record_length(train, sizeof train, &len) && len == 3);
    // every command gary/commands.py encodes that has no unit tag passes as is (the lengths of its
    // encoders: research, upgrade, stop, hold, burrow, unburrow, return cargo, unload all: 2; stim 1;
    // building morph and control groups 3; chat 82)
    for (uint8_t id : {0x30, 0x32, 0x1a, 0x2b, 0x2c, 0x2d, 0x1e, 0x28}) {
        const uint8_t two[] = {id, 0x00};
        CHECK(legacy_record_length(two, sizeof two, &len) && len == 2);
    }
    const uint8_t stim[] = {0x36};
    CHECK(legacy_record_length(stim, sizeof stim, &len) && len == 1);
    const uint8_t morph_building[] = {0x35, 0x84, 0x00};
    CHECK(legacy_record_length(morph_building, sizeof morph_building, &len) && len == 3);
    const uint8_t hotkey[] = {0x13, 0x01, 0x04};
    CHECK(legacy_record_length(hotkey, sizeof hotkey, &len) && len == 3);
    uint8_t chat[82] = {0x5c, 0x00};
    CHECK(legacy_record_length(chat, sizeof chat, &len) && len == 82);
    const uint8_t bad[] = {0x44};
    CHECK(!legacy_record_length(bad, sizeof bad, &len));
    const uint8_t truncated[] = {0x0c, 0x1e};
    CHECK(!legacy_record_length(truncated, sizeof truncated, &len));
}

// resolver for tests: tag -> fixed SC:R id; unknown tags are stale
static uint32_t resolve_test(uint16_t tag) {
    static const std::map<uint16_t, uint32_t> table = {{1, 0x11}, {2, 0x22}, {3, 0x1234}};
    auto it = table.find(tag);
    return it == table.end() ? 0 : it->second;
}

static void test_translation() {
    std::vector<std::vector<uint8_t>> out;
    std::string err;

    // select 2 units -> 0x63 with u32 ids (resim/gary_resim.cpp read_action_121, reversed)
    const uint8_t sel[] = {0x09, 0x02, 0x01, 0x00, 0x03, 0x00};
    CHECK(translate(sel, sizeof sel, resolve_test, &out, &err));
    CHECK(out.size() == 1);
    CHECK(eq(out[0], {0x63, 0x02, 0x11, 0x00, 0x00, 0x00, 0x34, 0x12, 0x00, 0x00}));

    // select-add -> 0x64
    out.clear();
    const uint8_t add[] = {0x0a, 0x01, 0x02, 0x00};
    CHECK(translate(add, sizeof add, resolve_test, &out, &err));
    CHECK(eq(out[0], {0x64, 0x01, 0x22, 0x00, 0x00, 0x00}));

    // right click -> 0x60 (target tag 2 -> u32 0x22; type 0xe4; queued 0)
    out.clear();
    const uint8_t rc[] = {0x14, 0x10, 0x00, 0x20, 0x00, 0x02, 0x00, 0xe4, 0x00, 0x00};
    CHECK(translate(rc, sizeof rc, resolve_test, &out, &err));
    CHECK(out[0].size() == 12 && out[0][0] == 0x60);
    CHECK(out[0][5] == 0x22 && out[0][6] == 0x00 && out[0][7] == 0x00 && out[0][8] == 0x00);
    CHECK(out[0][9] == 0xe4 && out[0][10] == 0x00 && out[0][11] == 0x00);

    // targeted order -> 0x61 (order 0x0e kept, queued 1 kept)
    out.clear();
    const uint8_t to[] = {0x15, 0x10, 0x00, 0x20, 0x00, 0x00, 0x00, 0xe4, 0x00, 0x0e, 0x01};
    CHECK(translate(to, sizeof to, resolve_test, &out, &err));
    CHECK(out[0].size() == 13 && out[0][0] == 0x61);
    CHECK(out[0][9] == 0xe4 && out[0][10] == 0x00 && out[0][11] == 0x0e && out[0][12] == 0x01);

    // unload -> 0x62
    out.clear();
    const uint8_t un[] = {0x29, 0x01, 0x00};
    CHECK(translate(un, sizeof un, resolve_test, &out, &err));
    CHECK(eq(out[0], {0x62, 0x11, 0x00, 0x00, 0x00}));

    // everything else passes through unchanged
    out.clear();
    const uint8_t train[] = {0x1f, 0x00, 0x00};
    CHECK(translate(train, sizeof train, resolve_test, &out, &err));
    CHECK(eq(out[0], {0x1f, 0x00, 0x00}));

    // stale handle -> error, nothing emitted
    out.clear();
    const uint8_t stale[] = {0x09, 0x01, 0x09, 0x00};
    CHECK(!translate(stale, sizeof stale, resolve_test, &out, &err));
    CHECK(err.find("expired") != std::string::npos);

    // a multi-record buffer splits correctly (select then train)
    out.clear();
    const uint8_t multi[] = {0x09, 0x01, 0x01, 0x00, 0x1f, 0x02, 0x00};
    CHECK(translate(multi, sizeof multi, resolve_test, &out, &err));
    CHECK(out.size() == 2 && out[0][0] == 0x63 && out[1][0] == 0x1f);
}

static void test_framing_and_json() {
    LineSplitter s;
    std::vector<std::string> lines;
    s.feed("{\"a\":1}\n{\"b\":", 13, &lines);
    CHECK(lines.size() == 1 && lines[0] == "{\"a\":1}");
    s.feed("2}\r\n{\"c\":3}\n", 12, &lines);
    CHECK(lines.size() == 3 && lines[1] == "{\"b\":2}" && lines[2] == "{\"c\":3}");

    std::string v;
    int64_t n = 0;
    CHECK(json_get_string("{\"id\":7,\"method\":\"act\",\"hex\":\"0901\"}", "method", &v));
    CHECK(v == "act");
    CHECK(json_get_string("{\"hex\":\"a\\\"b\"}", "hex", &v) && v == "a\"b");
    CHECK(json_get_int("{\"id\":42,\"slot\":3}", "id", &n) && n == 42);
    CHECK(json_get_int("{\"id\":-7}", "id", &n) && n == -7);
    CHECK(!json_get_int("{\"id\":x}", "id", &n));
}

int main() {
    test_handles();
    test_lengths();
    test_translation();
    test_framing_and_json();
    if (failures) {
        printf("%d failure(s)\n", failures);
        return 1;
    }
    printf("all tests passed\n");
    return 0;
}
