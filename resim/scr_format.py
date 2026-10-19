"""Decode Remastered-era replays (1.18+) for OpenBW.

OpenBW reads only the legacy replay layout (pre-1.18, PKWare-compressed sections). StarCraft
1.18+ writes the same sections compressed with zlib, and 1.21+ also adds an extra length field
and new sections (SKIN, LMTS, BFIX, CCLR, GCFG). This module decodes those files into the
uncompressed stream OpenBW's replay loader consumes:

    "reRS" | header (0x279 bytes) | u32 len | commands | u32 len | map data (CHK)

gary_resim reads that stream with `--flat`. The 1.21 command variants (0x60–0x65: right click,
targeted order, unload, select, select add, select remove) are translated to their legacy
equivalents by gary_resim itself while it simulates.

Remastered maps may also keep their strings in an 'STRx' chunk, which OpenBW doesn't read; an
equivalent 'STR ' chunk is added. And 1.21+ unit IDs use a larger unit table: gary_resim
translates them with `--scr-unit-ids`.

What is lost: the Remastered-only sections. Most are cosmetic (skins, colors), but LMTS (object
limits) and BFIX (active bug fixes) describe engine behavior that OpenBW's 1.16.1 engine doesn't
have, so a game that depends on them can drift. The desync detector measures how often.

The format follows screp's decoder (github.com/icza/screp, repparser/repdecoder).
"""

from __future__ import annotations

import struct
import zlib

HEADER_SIZE = 0x279


class NotModern(Exception):
    """The file uses the legacy layout, which OpenBW reads directly."""


def replay_format(data: bytes) -> str:
    """'legacy' (pre-1.18), 'modern' (1.18–1.20) or 'modern121' (1.21+)."""
    if len(data) < 30:
        raise ValueError("file too short to be a replay")
    if data[12:13] == b"s":
        return "modern121"
    return "legacy" if data[28] != 0x78 else "modern"


class _Reader:
    def __init__(self, data: bytes):
        self.data, self.pos = data, 0

    def int32(self) -> int:
        if self.pos + 4 > len(self.data):
            raise ValueError("unexpected end of replay")
        (n,) = struct.unpack_from("<i", self.data, self.pos)
        self.pos += 4
        return n

    def take(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.data):
            raise ValueError("unexpected end of replay")
        b = self.data[self.pos:self.pos + n]
        self.pos += n
        return b

    def section(self, size: int) -> bytes:
        """One section: checksum, chunk count, then chunks (each zlib-compressed or stored)."""
        if size == 0:
            return b""
        self.int32()  # checksum, not verified
        count = self.int32()
        out = bytearray()
        for _ in range(count):
            chunk = self.take(self.int32())
            if len(chunk) > 4 and chunk[0] == 0x78 and ((chunk[0] << 8) | chunk[1]) % 31 == 0:
                out += zlib.decompress(chunk)
            else:
                out += chunk
        return bytes(out[:size])


def remastered_map_to_bw(chk: bytes) -> bytes:
    """Make a Remastered map (CHK) readable by OpenBW: map version 206 becomes 205, and strings
    stored in an 'STRx' chunk (32-bit offsets) get an equivalent 'STR ' chunk (16-bit)."""
    chunks = []
    pos = 0
    while pos + 8 <= len(chk):
        name = chk[pos:pos + 4]
        (size,) = struct.unpack_from("<i", chk, pos + 4)
        if size < 0 or pos + 8 + size > len(chk):
            break  # protected/odd maps: leave the rest untouched
        chunks.append((name, chk[pos + 8:pos + 8 + size]))
        pos += 8 + size
    names = {n for n, _ in chunks}
    # VER 206 is the Remastered map format: the 205 (Brood War) layout plus STRx and color
    # chunks. OpenBW only knows up to 205; with STR provided below, 205 reads it correctly.
    ver = [i for i, (n, d) in enumerate(chunks) if n == b"VER " and d[:2] == struct.pack("<H", 206)]
    if ver:
        at = 0
        for i, (n, d) in enumerate(chunks):
            if i in ver:
                chk = chk[:at + 8] + struct.pack("<H", 205) + chk[at + 10:]
            at += 8 + len(d)
    if b"STR " in names or b"STRx" not in names:
        return chk
    strx = [d for n, d in chunks if n == b"STRx"][-1]
    (count,) = struct.unpack_from("<I", strx, 0)
    strings = []
    for i in range(count):
        (off,) = struct.unpack_from("<I", strx, 4 + 4 * i)
        end = strx.find(b"\0", off) if off < len(strx) else -1
        strings.append(strx[off:end] if end >= 0 else b"")
    # rebuild with 16-bit offsets; strings that don't fit in 64 KB become empty
    base = 2 + 2 * count
    body, offsets = bytearray(b"\0"), []
    for s in strings:
        if s and base + len(body) + len(s) + 1 < 0x10000:
            offsets.append(base + len(body))
            body += s + b"\0"
        else:
            offsets.append(base)  # points at the shared empty string
    str_chunk = struct.pack("<H", count) + b"".join(struct.pack("<H", o) for o in offsets) + bytes(body)
    return chk + b"STR " + struct.pack("<i", len(str_chunk)) + str_chunk


def remastered_sections(data: bytes) -> dict[str, bytes]:
    """The Remastered-only sections after the classic ones (e.g. 'LMTS', 'BFIX'), decoded."""
    fmt = replay_format(data)
    if fmt == "legacy":
        return {}
    r = _Reader(data)
    r.section(4)
    if fmt == "modern121":
        r.int32()
    r.section(HEADER_SIZE)
    r.section(struct.unpack("<I", r.section(4))[0])
    r.section(struct.unpack("<I", r.section(4))[0])
    r.section(0x300)  # player names
    sizes = {b"SKIN": 0x15E0, b"LMTS": 0x1C, b"BFIX": 0x08, b"CCLR": 0xC0, b"GCFG": 0x19}
    out = {}
    while r.pos + 8 <= len(data):
        sid = r.take(4)
        raw = r.int32()
        start = r.pos
        if sid in sizes:
            out[sid.decode()] = r.section(sizes[sid])
        r.pos = start + raw
    return out


def unit_limit(data: bytes) -> int:
    """Size of the game's unit table (1700 classic; recent Remastered games use 3400)."""
    lmts = remastered_sections(data).get("LMTS", b"")
    return struct.unpack_from("<I", lmts, 0x0C)[0] if len(lmts) >= 0x10 else 1700


def _sections(data: bytes) -> tuple[bytes, bytes, bytes]:
    """A 1.18+ replay's header, commands and map data (the map as the replay holds it)."""
    fmt = replay_format(data)
    if fmt == "legacy":
        raise NotModern("legacy replay: OpenBW reads it directly")
    r = _Reader(data)
    replay_id = r.section(4)
    if replay_id not in (b"reRS", b"seRS"):
        raise ValueError(f"not a replay (id {replay_id!r})")
    if fmt == "modern121":
        r.int32()  # encoded length between the first sections (1.21+)
    header = r.section(HEADER_SIZE)
    commands = r.section(struct.unpack("<I", r.section(4))[0])
    map_data = r.section(struct.unpack("<I", r.section(4))[0])
    return header, commands, map_data


def map_data(data: bytes) -> bytes:
    """A 1.18+ replay's map (CHK) as recorded: its SHA-1 is the replay's MapDataHash (ingest/inventory.py)."""
    return _sections(data)[2]


def to_legacy(data: bytes) -> bytes:
    """A 1.18+ replay rewritten in the 1.16 replay format, which OpenBW (gary/env.py Game.new) loads with its
    map: each section stored uncompressed (a segment of its full size is read raw), the map made readable
    (remastered_map_to_bw). For games on the current ladder maps (ingest/ladder_maps.py)."""
    import zlib

    def section(b: bytes) -> bytes:   # (OpenBW's checksum: CRC-32 without the final inversion)
        segs = [b[i:i + 8192] for i in range(0, len(b), 8192)]
        return struct.pack("<II", zlib.crc32(b) ^ 0xFFFFFFFF, len(segs)) + b"".join(struct.pack("<I", len(s)) + s for s in segs)
    header, commands, chk = _sections(data)
    chk = remastered_map_to_bw(chk)
    return b"".join(section(x) for x in (b"reRS", header, struct.pack("<I", len(commands)), commands,
                                         struct.pack("<I", len(chk)), chk))


def to_flat(data: bytes) -> bytes:
    """Decode a 1.18+ replay into the flat stream gary_resim reads with --flat."""
    header, commands, map_data = _sections(data)
    map_data = remastered_map_to_bw(map_data)
    if len(header) != HEADER_SIZE:
        raise ValueError("truncated header")
    return (b"reRS" + header + struct.pack("<I", len(commands)) + commands
            + struct.pack("<I", len(map_data)) + map_data)
