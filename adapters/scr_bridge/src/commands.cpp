#include "commands.h"

#include <cstring>
#include <string>

namespace garyscr {
namespace {

void put16(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back((uint8_t)(x & 0xff));
    v.push_back((uint8_t)((x >> 8) & 0xff));
}

void put32(std::vector<uint8_t>& v, uint32_t x) {
    put16(v, x & 0xffff);
    put16(v, (x >> 16) & 0xffff);
}

uint16_t get16(const uint8_t* p) { return (uint16_t)(p[0] | ((uint16_t)p[1] << 8)); }

}  // namespace

// Record lengths of the legacy (pre-1.21) commands Gary's interface emits, plus the save/load
// and sync records. Sources: replay format as parsed by external/screp/repparser, the encoders
// in gary/commands.py, and resim/gary_resim.cpp (which decodes these same records). Any other
// id is rejected rather than guessed at, so a command can never be misparsed.
bool legacy_record_length(const uint8_t* p, size_t n, size_t* out_len) {
    if (n < 1) return false;
    switch (p[0]) {
    case 0x05:  // keep-alive
    case 0x08:  // restart game
        *out_len = 1;
        return true;
    case 0x06:  // save game: id, 4 unknown bytes, NUL-terminated name (SB bw/commands.rs)
    case 0x07:  // load game: same shape
        for (size_t i = 5; i < n; ++i) {
            if (p[i] == 0) {
                *out_len = i + 1;
                return true;
            }
        }
        return false;
    case 0x09:  // select
    case 0x0a:  // select add
    case 0x0b:  // select remove
        if (n < 2 || (size_t)p[1] > 12) return false;
        *out_len = 2 + (size_t)p[1] * 2;
        return n >= *out_len;
    case 0x0c:  // build: id, order, tile x, tile y, unit type (gary/commands.py)
        *out_len = 8;
        return n >= *out_len;
    case 0x14:  // right click: id, x, y, target tag, target type, queued
        *out_len = 10;
        return n >= *out_len;
    case 0x15:  // targeted order: id, x, y, target tag, target type, order, queued
        *out_len = 11;
        return n >= *out_len;
    case 0x1f:  // train: id, unit type
    case 0x20:  // cancel train: id, queue slot (gary_resim read_action_cancel_build_queue)
    case 0x23:  // unit morph: id, unit type
    case 0x29:  // unload: id, unit tag
    case 0x35:  // building morph: id, unit type (a Lair, a Sunken Colony)
    case 0x13:  // control group: id, action (0 assign, 1 recall, 2 add), group 0-9 (OpenBW actions.h)
        *out_len = 3;
        return n >= *out_len;
    case 0x1a:  // stop: id, queued
    case 0x1e:  // return cargo: id, queued
    case 0x28:  // unload all: id, queued
    case 0x2b:  // hold position: id, queued
    case 0x2c:  // burrow: id, queued
    case 0x2d:  // unburrow: id, queued
    case 0x30:  // research: id, tech
    case 0x32:  // upgrade: id, upgrade
        *out_len = 2;
        return n >= *out_len;
    case 0x36:  // stim packs: id alone
        *out_len = 1;
        return n >= *out_len;
    case 0x5c:  // chat (replay record): id, sender slot, 80-byte NUL-padded text (gary/commands.py)
        *out_len = 82;
        return n >= *out_len;
    case 0x37:  // sync (SB bw/commands.rs id::SYNC)
        *out_len = 2;
        return n >= *out_len;
    default:
        return false;
    }
}

bool translate_one(const uint8_t* p, size_t n,
                   const std::function<uint32_t(uint16_t)>& resolve_tag,
                   std::vector<std::vector<uint8_t>>* out, std::string* error) {
    size_t len = 0;
    if (!legacy_record_length(p, n, &len)) {
        char buf[96];
        snprintf(buf, sizeof buf, "unsupported or truncated legacy command 0x%02x (%zu bytes)",
                 n ? p[0] : 0, n);
        *error = buf;
        return false;
    }
    const uint8_t id = p[0];
    // resolve a unit handle; 0 handle in = 0 out (a missing target), stale handle = error.
    auto unit = [&](uint16_t tag, uint32_t* id32) -> bool {
        if (!tag) {
            *id32 = 0;
            return true;
        }
        *id32 = resolve_tag(tag);
        if (!*id32) {
            *error = "command references an expired unit handle";
            return false;
        }
        return true;
    };

    if (id >= 0x09 && id <= 0x0b) {
        // select family: 0x09/0x0a/0x0b -> 0x63/0x64/0x65, unit tags u16 -> SC:R ids u32.
        std::vector<uint8_t> rec{(uint8_t)(0x63 + (id - 0x09)), p[1]};
        for (unsigned i = 0; i < p[1]; ++i) {
            uint32_t id32;
            if (!unit(get16(p + 2 + i * 2), &id32)) return false;
            put32(rec, id32);
        }
        out->push_back(std::move(rec));
        return true;
    }
    if (id == 0x14 || id == 0x15) {
        // right click / targeted order -> 0x60 / 0x61, target tag u16 -> u32.
        std::vector<uint8_t> rec{id == 0x14 ? (uint8_t)0x60 : (uint8_t)0x61};
        uint32_t id32;
        if (!unit(get16(p + 5), &id32)) return false;
        put16(rec, get16(p + 1));  // x
        put16(rec, get16(p + 3));  // y
        put32(rec, id32);          // target (u16 id + u16 zero field in the 1.21 wire format)
        put16(rec, get16(p + 7));  // target unit type
        if (id == 0x15) rec.push_back(p[9]);  // order
        rec.push_back(id == 0x15 ? p[10] : p[9]);  // queued
        out->push_back(std::move(rec));
        return true;
    }
    if (id == 0x29) {
        // unload -> 0x62, unit tag u16 -> u32.
        std::vector<uint8_t> rec{0x62};
        uint32_t id32;
        if (!unit(get16(p + 1), &id32)) return false;
        put32(rec, id32);
        out->push_back(std::move(rec));
        return true;
    }
    // Everything else (build, train, morph, cancel, sync, ...) is identical in 1.21: pass the
    // record through byte for byte (resim/gary_resim.cpp: "1.21 added variants of six commands").
    out->emplace_back(p, p + len);
    return true;
}

bool translate(const uint8_t* data, size_t n,
               const std::function<uint32_t(uint16_t)>& resolve_tag,
               std::vector<std::vector<uint8_t>>* out, std::string* error) {
    for (size_t off = 0; off < n;) {
        size_t len = 0;
        if (!legacy_record_length(data + off, n - off, &len)) {
            char buf[96];
            snprintf(buf, sizeof buf, "unsupported or truncated legacy command 0x%02x at %zu",
                     data[off], off);
            *error = buf;
            return false;
        }
        if (!translate_one(data + off, len, resolve_tag, out, error)) return false;
        off += len;
    }
    return true;
}

}  // namespace garyscr
