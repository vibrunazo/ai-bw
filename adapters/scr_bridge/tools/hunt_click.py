#!/usr/bin/env python3
"""Pin the sprite/image/GRP offsets the click model needs by measurement, then check them.

--dump follows each unit's sprite to candidate image heads and grp headers and scores the
BWAPI/Pluto layout hypothesis (CSprite.image at +0; CImage {next,prev,grp,frameSet,frameIndex,
x,y,flags}; GRP = u16 frames,w,h + per-frame u8 x,y,w,h per external/openbw read_grp). Adjust
the constants below until everything scores, then record them in src/scr_profile.h.

--check runs the acceptance test: for every own unit on screen with nothing drawn over it,
unit_at(its centre) must return that unit ("Gary playing fine" does not prove this).

    python tools/hunt_click.py --dump    # raw evidence (live game required)
    python tools/hunt_click.py --check   # behavioural acceptance (rebuilt DLL required)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from gary import commands as C  # noqa: E402
from gary.scr_env import ScrGame  # noqa: E402

# layout: bw_dat (neivv/aise) structs.rs, the SCR Sprite and Image (src/scr_profile.h). The 2026-10-04
# "pinned" layout read the previous sprite as the first image; corrected 2026-10-15. Flag bits are
# OpenBW image_t::flags (0x02 flipped, 0x20 clickable, 0x40 hidden). Frame sizes come from the GRP
# files via gen_image_dat.py (memory holds the image type + frame index).
SPR_IMAGE = 0x20        # Sprite -> first image (bw_dat SCR layout; see src/scr_profile.h)
IMG_NEXT = 0x04         # Image -> next image
IMG_PREV = 0x00
IMG_TYPE = 0x08         # u16 image type (images.dat row)
IMG_FRAMEINDEX = 0x1A   # u16 frame within the grp
IMG_X = 0x0E            # s8 offset from sprite position
IMG_Y = 0x0F            # s8
IMG_FLAGS = 0x0C        # u16 (low byte holds the bits used here)
IMG_PARENT = 0x3C       # the sprite that owns the image
FLAG_CLICKABLE = 0x20
FLAG_FLIPPED = 0x02
FLAG_Y_FROZEN = 0x04


def u16(b: bytes, o: int) -> int:
    return int.from_bytes(b[o:o + 2], "little")


def s8(b: bytes, o: int) -> int:
    v = b[o]
    return v - 256 if v >= 128 else v


def u32(b: bytes, o: int) -> int:
    return int.from_bytes(b[o:o + 4], "little")


def heap_ptr(v: int) -> bool:
    return 0x10000 <= v < 0x80000000 and v % 2 == 0


def peek(game: ScrGame, addr: int, length: int) -> bytes | None:
    """game.peek that treats an unmapped address as 'not a pointer' (the bridge fails safe)."""
    try:
        return game.peek(addr, length)
    except Exception:
        return None


def grp_ok(game: ScrGame, addr: int):
    """(frames, w, h, frame table bytes) if addr points at a plausible GRP header."""
    if not addr:
        return None
    head = peek(game, addr, 14)
    if not head or len(head) < 14:
        return None
    frames, w, h = u16(head, 0), u16(head, 2), u16(head, 4)
    if not (1 <= frames <= 400 and 1 <= w <= 300 and 1 <= h <= 300):
        return None
    table = peek(game, addr + 6, frames * 8)
    return (frames, w, h, table) if table else None


def image_chain(game: ScrGame, sprite: int, head_off: int = SPR_IMAGE):
    """The sprite's own images, first to last (bw_dat's SCR layout: first image at SPR_IMAGE, last at
    SPR_IMAGE + 4, each image owned by the sprite at IMG_PARENT). A link that leaves the sprite ends it."""
    out = []
    head = peek(game, sprite, head_off + 8)
    if not head:
        return out
    addr, last = u32(head, head_off), u32(head, head_off + 4)
    for _ in range(16):
        if not heap_ptr(addr):
            break
        data = peek(game, addr, 0x40)
        if not data or u32(data, IMG_PARENT) != sprite:
            break
        out.append((addr, data))
        if addr == last:
            break
        addr = u32(data, IMG_NEXT)
    return out


def image_rect(game: ScrGame, sprite_pos, img_addr: int, data: bytes):
    """gary_env's clickable rect for one image: map position + grp frame size (files)."""
    v = valid_image(data, IMG_TYPE, IMG_FRAMEINDEX)
    if v is None:
        return None
    t, f = v
    flags = data[IMG_FLAGS]
    # gary_env gates on the runtime flag_clickable bit (images.dat is_clickable is copied into
    # the flags at creation; overlays can be file-clickable but hidden at runtime).
    if not flags & FLAG_CLICKABLE:
        return None
    fx, fy, fw, fh = FRAMES[t][f]
    gw, gh = CANVAS[t]
    px, py = sprite_pos[0] + s8(data, IMG_X), sprite_pos[1] + s8(data, IMG_Y)
    if flags & FLAG_FLIPPED:
        px += gw // 2 - (fx + fw)
    else:
        px += fx - gw // 2
    py += fy - gh // 2
    return (px, py, px + fw, py + fh)


def unit_rect(game: ScrGame, u: dict):
    probe = game.probe_unit(u["tag"])
    sprite = int(probe.get("sprite", 0))
    if not sprite:
        return None
    rects = [r for r in (image_rect(game, (u["x"], u["y"]), a, d)
                         for a, d in image_chain(game, sprite)) if r]
    if not rects:
        # fallback for sprites whose body image link isn't resolved yet: the sprite bbox
        sd = peek(game, sprite, 48)
        if not sd:
            return None
        bw, bh = sd[0x12], sd[0x13]
        return (u["x"] - bw // 2, u["y"] - bh // 2, u["x"] - bw // 2 + bw, u["y"] - bh // 2 + bh)
    return (min(r[0] for r in rects), min(r[1] for r in rects),
            max(r[2] for r in rects), max(r[3] for r in rects))


def load_tables():
    """(is_clickable[999], grp canvas [999], frames [999][(x,y,w,h)]) from the game files."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import gen_image_dat as G
    cols = G.read_columns((G.SCR / "arr" / "images.dat").read_bytes())
    names = G.read_tbl((G.SCR / "arr" / "images.tbl").read_bytes())
    clickable = cols["is_clickable"]
    canvas = [(0, 0)] * 999
    frames: list[list[tuple[int, int, int, int]]] = [[] for _ in range(999)]
    for t in range(999):
        gi = cols["grp_filename_index"][t]
        if gi < len(names):
            p = G.SCR / "unit" / names[gi].replace("\\", "/")
            if p.exists() and p.suffix.lower() == ".grp":
                canvas[t], frames[t] = G.read_grp(p.read_bytes())[:2], G.read_grp(p.read_bytes())[2]
    return clickable, canvas, frames


CLICKABLE, CANVAS, FRAMES = load_tables()


def valid_image(data: bytes, toff: int, foff: int):
    """(type, frame) if that offset pair consistently parses as an image."""
    t = u16(data, toff)
    f = u16(data, foff)
    if t >= 999 or not FRAMES[t] or f >= len(FRAMES[t]):
        return None
    return t, f


def chain_records(game: ScrGame, sprite: int, head_off: int, toff: int, foff: int):
    base = peek(game, sprite, head_off + 4)
    if not base:
        return []
    addr = u32(base, head_off)
    out = []
    seen = set()
    for _ in range(8):
        if not heap_ptr(addr) or addr in seen:
            break
        seen.add(addr)
        data = peek(game, addr, 0x28)
        if not data or not valid_image(data, toff, foff):
            break
        out.append((addr, data))
        addr = u32(data, 0)  # next
    return out


def rect_of(rec: bytes, toff: int, foff: int, xo: int, yo: int, sprite_pos):
    v = valid_image(rec, toff, foff)
    if v is None:
        return None
    t, f = v
    if not CLICKABLE[t]:
        return None
    fx, fy, fw, fh = FRAMES[t][f]
    gw, gh = CANVAS[t]
    px = sprite_pos[0] + s8(rec, xo) + fx - gw // 2
    py = sprite_pos[1] + s8(rec, yo) + fy - gh // 2
    return (px, py, px + fw, py + fh)


def dump(game: ScrGame) -> None:
    """Score the layout hypothesis; the combo with total wins is the pin."""
    obs = game.observe()
    lp = game.status().get("local_player", -1)
    units = [u for u in obs["units"] if u["owner"] == lp][:4]
    samples = []
    for u in units:
        probe = game.probe_unit(u["tag"])
        sprite = int(probe.get("sprite", 0))
        words = {o: u32(peek(game, sprite, 48), o) for o in range(0, 44, 4)} if sprite else {}
        samples.append((u, sprite, words))
        print(f"unit tag {u['tag']} type {u['type']} at ({u['x']},{u['y']}) sprite {sprite:#x}")
    # trace the first unit's chains so the raw image records can be read off directly
    u, sprite, words = samples[0]
    for head_off in (0x00, 0x04, 0x28):
        recs = chain_records(game, sprite, head_off, 0x08, 0x0A)
        print(f"  chain via head@{head_off:#x}: {len(recs)} records")
        for a, d in recs:
            print(f"    {a:08x}: {' '.join(f'{b:02x}' for b in d[:0x18])} "
                  f"t={u16(d, 8)} f={u16(d, 10)}")
    # score (type_off, frame_off, head_off, x_off, y_off)
    best = {}
    for toff in (0x08, 0x0A, 0x0C):
        for foff in (0x08, 0x0A, 0x0C, 0x0E):
            if foff == toff:
                continue
            for head_off in (0x00, 0x04, 0x24, 0x28):
                for xo in (0x0C, 0x0E, 0x10, 0x13, 0x14):
                    for yo in (0x0D, 0x0F, 0x11, 0x15, 0x16):
                        wins = 0
                        for u, sprite, words in samples:
                            recs = chain_records(game, sprite, head_off, toff, foff)
                            rects = [r for r in (rect_of(d, toff, foff, xo, yo, (u["x"], u["y"]))
                                                 for _, d in recs) if r]
                            if rects and any(r[0] <= u["x"] < r[2] and r[1] <= u["y"] < r[3]
                                             for r in rects):
                                wins += 1
                        key = (toff, foff, head_off, xo, yo)
                        best[key] = wins
    for key, wins in sorted(best.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {wins}/{len(samples)} wins: type@{key[0]:#x} frame@{key[1]:#x} "
              f"head@{key[2]:#x} x@{key[3]:#x} y@{key[4]:#x}")


def check(game: ScrGame) -> int:
    obs = game.observe()
    lp = game.status().get("local_player", -1)
    # re-read each unit's position together with its sprite (probe) -- moving units drift
    # several pixels between the observe() snapshot and the peeks below
    fresh = []
    for u in obs["units"]:
        if u["owner"] != lp:
            continue
        p = game.probe_unit(u["tag"])
        if p:
            fresh.append({"tag": u["tag"], "type": u["type"], "x": p["x"], "y": p["y"]})
    rects = {u["tag"]: unit_rect(game, u) for u in fresh}
    # brute-force the remaining fields against gary_env's reference boxes (--gold, unclipped):
    # which record + formula variant + (x,y) byte pair reproduces them exactly?
    gold = {35: (-10, -4, 7, 9), 41: (-12, -10, 21, 19), 7: (-20, -18, 11, 13),
            42: (-22, -34, 29, 33), 64: (-6, -10, 15, 15), 154: (-48, -48, 49, 49)}
    votes: dict[tuple, set] = {}
    nearest: dict[int, tuple] = {}
    for u in fresh:
        g = gold.get(u["type"])
        if not g:
            continue
        sprite = int(game.probe_unit(u["tag"]).get("sprite", 0))
        for _, d in image_chain(game, sprite):
            for f2off in (0, 1, -1):
                v = valid_image(d, IMG_TYPE, IMG_FRAMEINDEX)
                if not v:
                    continue
                t, f = v
                if not FRAMES[t] or not (0 <= f + f2off < len(FRAMES[t])):
                    continue
                fx, fy, fw, fh = FRAMES[t][f + f2off]
                gw, gh = CANVAS[t]
                for xo in range(0x08, 0x18):
                    for yo in range(0x08, 0x18):
                        for flipped in (False, True):
                            px = s8(d, xo) + (gw // 2 - (fx + fw) if flipped else fx - gw // 2)
                            py = s8(d, yo) + fy - gh // 2
                            r = (px, py, px + fw, py + fh)
                            if r == g:
                                key = (xo, yo, flipped, f2off)
                                votes.setdefault(key, set()).add(u["type"])
                            dist = sum(abs(a - b) for a, b in zip(r, g))
                            if u["type"] not in nearest or dist < nearest[u["type"]][0]:
                                nearest[u["type"]] = (dist, r, t, f + f2off, flipped)
    print("  golden votes (x,y,flipped,frame_adj) -> types:",
          {k: sorted(s) for k, s in votes.items()} or "NONE")
    for typ, (dist, r, t, f, fl) in nearest.items():
        print(f"  nearest type {typ}: dist={dist} rel={r} (image {t} frame {f} flip={fl}) "
              f"gold={gold[typ]}")
    ok = skipped = bad = 0
    for u in fresh:
        r = rects[u["tag"]]
        if not r or not (r[0] <= u["x"] < r[2] and r[1] <= u["y"] < r[3]):
            recs = image_chain(game, int(game.probe_unit(u["tag"]).get("sprite", 0)))
            info = [(u16(d, IMG_TYPE), u16(d, IMG_FRAMEINDEX), hex(d[IMG_FLAGS]),
                     len(FRAMES[u16(d, IMG_TYPE)]) if u16(d, IMG_TYPE) < 999 else -1)
                    for _, d in recs]
            print(f"  BAD  tag {u['tag']} type {u['type']}: rect {r} vs centre ({u['x']},{u['y']}) "
                  f"chain={info}")
            bad += 1
            continue
        over = [t for t, rr in rects.items() if t != u["tag"] and rr
                and rr[0] <= u["x"] < rr[2] and rr[1] <= u["y"] < rr[3]]
        hit = game.unit_at(lp, u["x"], u["y"])
        if over:
            skipped += 1
        elif hit == u["tag"]:
            ok += 1
        else:
            print(f"  BAD  tag {u['tag']} type {u['type']}: unit_at(centre) -> {hit}")
            bad += 1
    print(f"centre round-trip: {ok} ok, {skipped} skipped (overlapping sprites), {bad} bad")
    return 1 if bad else 0


def gold() -> int:
    """gary_env's own hit map on OpenBW: the exact clickable box each type's model produces,
    relative to the unit position -- the reference for the live offsets."""
    from gary.env import Game
    map_path = REPO / "tests" / "fixtures" / "replays" / "stardata_tvz_standard_ozp3w.rep"
    for races in (["T", "Z"], ["P", "T"], ["Z", "P"]):
        with Game.new(map_path, races, ["gold", "x"], seed=1) as g:
            obs = g.observe()
            slot = next((u["owner"] for u in obs["units"] if u["type"] in (106, 131, 154)), 0)
            shown = set()
            for u in obs["units"]:
                if u["type"] in shown or u["type"] in (176, 177, 178):
                    continue
                pts = []
                for dy in range(-48, 49, 2):
                    for dx in range(-48, 49, 2):
                        if g.unit_at(u["owner"], u["x"] + dx, u["y"] + dy) == u["tag"]:
                            pts.append((dx, dy))
                if pts:
                    shown.add(u["type"])
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    print(f"  type {u['type']:3d}: rel rect ({min(xs):4d},{min(ys):4d})-"
                          f"({max(xs) + 1:4d},{max(ys) + 1:4d})  size {max(xs) + 1 - min(xs)}x"
                          f"{max(ys) + 1 - min(ys)}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--gold", action="store_true", help="gary_env reference boxes on OpenBW")
    args = ap.parse_args()
    if args.gold:
        raise SystemExit(gold())
    game = ScrGame.connect()
    try:
        if args.check:
            raise SystemExit(check(game))
        dump(game)
    finally:
        game.close()


if __name__ == "__main__":
    main()
