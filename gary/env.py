"""Python access to a headless Brood War game (env/gary_env.dll, built on OpenBW).

    from gary.env import Game
    game = Game.new("path/to/map.scx", races=["T", "Z"], names=["Gary", "Gary"], seed=1)
    obs = game.observe()                  # full state as a dict
    game.act(slot, commands.select([tag]))
    game.step(8)
    game.save_replay("out.rep")

This is the raw game: full information, no limits. Gary goes through gary.interface.HumanInterface,
which adds fog of war, the camera, mouse travel and the APM budget (docs/ARCHITECTURE.md §6.3).

command_delay (frames) makes commands run that many frames after they're sent, like a networked
game, where every command waits for the next network turn. LIVE_COMMAND_DELAY is what Gary saw in
a LAN game against a human on Remastered; train and test with it so Gary doesn't learn to expect
its orders to land instantly.
"""

from __future__ import annotations

import ctypes
import json
import os
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# From a LAN game on Remastered (2026-10-04): the live client reports a turn latency of 2 frames,
# plus 1 frame to hand the command over, and with 3 here Gary v0.1 behaves as it did live (it
# resent each train order in bursts of 6-7 there, 7-8 here; 5 with no delay, from its own reaction
# time alone).
LIVE_COMMAND_DELAY = 3


def _load_dll() -> ctypes.CDLL:
    path = REPO_ROOT / "env" / "build" / ("gary_env.dll" if os.name == "nt" else "libgary_env.so")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run env/build.bat")
    dll = ctypes.CDLL(str(path))
    dll.gary_env_create.restype = ctypes.c_void_p
    dll.gary_env_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    dll.gary_env_create_custom.restype = ctypes.c_void_p
    dll.gary_env_create_custom.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                                           ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                                           ctypes.POINTER(ctypes.c_char_p), ctypes.c_uint32]
    dll.gary_env_terrain.restype = ctypes.c_int
    dll.gary_env_terrain.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
    dll.gary_env_walkable.restype = ctypes.c_bool
    dll.gary_env_walkable.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    dll.gary_env_map_chk.restype = ctypes.c_int
    dll.gary_env_map_chk.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    dll.gary_env_create_scenario.restype = ctypes.c_void_p
    dll.gary_env_create_scenario.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    dll.gary_env_drop_commands.argtypes = [ctypes.c_void_p, ctypes.c_int]
    dll.gary_env_save.restype = ctypes.c_bool
    dll.gary_env_save.argtypes = [ctypes.c_void_p]
    dll.gary_env_restore.restype = ctypes.c_bool
    dll.gary_env_restore.argtypes = [ctypes.c_void_p]
    dll.gary_env_error.restype = ctypes.c_char_p
    dll.gary_env_error.argtypes = [ctypes.c_void_p]
    dll.gary_env_destroy.argtypes = [ctypes.c_void_p]
    dll.gary_env_step.restype = ctypes.c_bool
    dll.gary_env_step.argtypes = [ctypes.c_void_p, ctypes.c_int]
    dll.gary_env_act.restype = ctypes.c_int
    dll.gary_env_act.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    dll.gary_env_observe.restype = ctypes.c_char_p
    dll.gary_env_observe.argtypes = [ctypes.c_void_p]
    dll.gary_env_save_replay.restype = ctypes.c_bool
    dll.gary_env_save_replay.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    dll.gary_env_unit_at.restype = ctypes.c_uint
    dll.gary_env_unit_at.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    dll.gary_env_selection.restype = ctypes.c_int
    dll.gary_env_selection.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint), ctypes.c_int]
    dll.gary_env_type_select.restype = ctypes.c_int
    dll.gary_env_type_select.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                         ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint),
                                         ctypes.c_int]
    dll.gary_env_box_select.restype = ctypes.c_int
    dll.gary_env_box_select.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint), ctypes.c_int]
    dll.gary_env_unit_type_info.restype = ctypes.c_bool
    dll.gary_env_unit_type_info.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    dll.gary_env_create_unit.restype = ctypes.c_uint
    dll.gary_env_create_unit.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    dll.gary_env_creep.restype = ctypes.c_int
    dll.gary_env_creep.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    dll.gary_env_set_upgrade.restype = None
    dll.gary_env_set_upgrade.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    dll.gary_env_set_resources.restype = None
    dll.gary_env_set_resources.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    dll.gary_env_regions.restype = ctypes.c_int
    dll.gary_env_regions.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                                     ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                                     ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    dll.gary_env_debug_builder.restype = ctypes.c_int
    dll.gary_env_debug_place.restype = ctypes.c_int
    dll.gary_env_debug_place.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    dll.gary_env_debug_builder.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    dll.gary_env_unit_type.restype = ctypes.c_int
    dll.gary_env_unit_type.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    dll.gary_env_set_name.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p]
    dll.gary_env_can_place.restype = ctypes.c_bool
    dll.gary_env_can_place.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_int]
    dll.gary_env_depot_spot_ok.restype = ctypes.c_bool
    dll.gary_env_depot_spot_ok.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    dll.gary_env_start_locations.restype = ctypes.c_char_p
    dll.gary_env_start_locations.argtypes = [ctypes.c_void_p]
    dll.gary_env_create_game.restype = ctypes.c_void_p
    dll.gary_env_create_game.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
                                         ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_char_p),
                                         ctypes.c_uint32]
    return dll


_dll: ctypes.CDLL | None = None


def _gamedata_dir() -> Path:
    return Path(os.environ.get("GARY_DATA", REPO_ROOT / "data")) / "gamedata" / "scr"


def decode_observation(raw: bytes) -> dict:
    return json.loads(raw.decode("utf-8", errors="replace"))


def map_chk(map_path: str | Path) -> bytes:
    """A map file's scenario data (scenario.chk; a .scm/.scx map or a pre-1.18 replay's map)."""
    global _dll
    _dll = _dll or _load_dll()
    n = _dll.gary_env_map_chk(str(map_path).encode(), None, 0)
    if n < 0:
        raise GameError(_dll.gary_env_error(None).decode(errors="replace"))
    buf = ctypes.create_string_buffer(n)
    _dll.gary_env_map_chk(str(map_path).encode(), buf, n)
    return buf.raw


class GameError(RuntimeError):
    pass


class Game:
    def __init__(self, handle: int, command_delay: int = 0):
        self._h = handle
        self.frame = 0
        self.command_delay = command_delay
        self._in_flight: list[tuple[int, int, bytes]] = []  # (run at frame, slot, command)
        # slot -> fn(command, accepted): told the result of each delayed command when it runs
        self.on_result: dict = {}

    @classmethod
    def new(cls, map_path: str | Path, races: list[str], names: list[str] | None = None,
            seed: int | None = None, gamedata: str | Path | None = None, command_delay: int = 0) -> "Game":
        """A new melee game. map_path: a .scm/.scx map, or a pre-1.18 replay (its embedded map).
        races: e.g. ["T", "Z"]. Players get random start locations (from seed). The seed is also
        the replay's start time, as in the real game; it defaults to the current time."""
        global _dll
        _dll = _dll or _load_dll()
        codes = {"Z": 0, "T": 1, "P": 2}
        n = len(races)
        seed = int(time.time()) if seed is None else seed
        names = names or [f"Gary {i + 1}" for i in range(n)]
        race_arr = (ctypes.c_int * n)(*[codes[r.upper()[0]] for r in races])
        name_arr = (ctypes.c_char_p * n)(*[nm.encode()[:24] for nm in names])
        h = _dll.gary_env_create_game(str(gamedata or _gamedata_dir()).encode(), str(map_path).encode(),
                                      n, race_arr, name_arr, seed)
        if not h:
            raise GameError(_dll.gary_env_error(None).decode(errors="replace"))
        game = cls(h, command_delay)
        game.map_path = str(map_path)        # (the map's data hash: MapInfo.from_game)
        return game

    @classmethod
    def custom(cls, chk: bytes, map_name: str, slots: list[int], races: list[str], names: list[str],
               seed: int = 0, gamedata: str | Path | None = None, command_delay: int = 0) -> "Game":
        """A "use map settings" game on map data (a scenario.chk, see map_chk): the given slots get
        the map's preplaced units and no melee rules apply (drills: gary/drills.py). Saved replays
        carry the map, so they play back like any game."""
        global _dll
        _dll = _dll or _load_dll()
        codes = {"Z": 0, "T": 1, "P": 2}
        n = len(slots)
        h = _dll.gary_env_create_custom(str(gamedata or _gamedata_dir()).encode(), chk, len(chk), map_name.encode()[:31], n,
                                        (ctypes.c_int * n)(*slots), (ctypes.c_int * n)(*[codes[r.upper()[0]] for r in races]),
                                        (ctypes.c_char_p * n)(*[nm.encode()[:24] for nm in names]), seed)
        if not h:
            raise GameError(_dll.gary_env_error(None).decode(errors="replace"))
        return cls(h, command_delay)

    @classmethod
    def from_replay_map(cls, replay: str | Path, gamedata: str | Path | None = None,
                        command_delay: int = 0) -> "Game":
        """A new game on a (pre-1.18 format) replay's map, with its players and races.
        The replay's own commands are not played."""
        global _dll
        _dll = _dll or _load_dll()
        h = _dll.gary_env_create(str(gamedata or _gamedata_dir()).encode(), str(replay).encode())
        if not h:
            raise GameError(_dll.gary_env_error(None).decode(errors="replace"))
        return cls(h, command_delay)

    @classmethod
    def scenario(cls, replay: str | Path, gamedata: str | Path | None = None, command_delay: int = 0) -> "Game":
        """A replay (any format) that plays its own commands as the game steps: step to the moment
        of interest, then take_over(slot) to play that side from there. The other players keep
        replaying what they did in the real game."""
        global _dll
        _dll = _dll or _load_dll()
        import sys
        sys.path.insert(0, str(REPO_ROOT / "resim"))
        import scr_format
        data = Path(replay).read_bytes()
        flat, limit = 0, 1700
        if scr_format.replay_format(data) != "legacy":
            flat, limit = 1, scr_format.unit_limit(data)
            data = scr_format.to_flat(data)
        h = _dll.gary_env_create_scenario(str(gamedata or _gamedata_dir()).encode(), data, len(data), flat, limit)
        if not h:
            raise GameError(_dll.gary_env_error(None).decode(errors="replace"))
        return cls(h, command_delay)

    def save(self) -> None:
        """Remember this moment exactly (no command may be in flight); restore() returns to it."""
        if self._in_flight:
            raise GameError("save with commands in flight")
        self._check(_dll.gary_env_save(self._h))
        self._saved_frame = self.frame

    def restore(self) -> None:
        """Back to the moment save() remembered: the game, its replay commands, the replay being
        recorded, who was taken over. Callbacks (on_result) are cleared."""
        self._check(_dll.gary_env_restore(self._h))
        self.frame = self._saved_frame
        self._in_flight = []
        self.on_result = {}

    def take_over(self, slot: int) -> None:
        """From now on the replay's commands for this player are dropped: act() plays that side."""
        _dll.gary_env_drop_commands(self._h, slot)

    def _check(self, ok: bool) -> None:
        if not ok:
            raise GameError(_dll.gary_env_error(self._h).decode(errors="replace"))

    def step(self, frames: int = 1) -> None:
        end = self.frame + frames
        while self.frame < end:
            due = [c for c in self._in_flight if c[0] <= self.frame]
            if due:
                self._in_flight = [c for c in self._in_flight if c[0] > self.frame]
                for _, slot, command in due:
                    ok = self._run(slot, command)
                    if (report := self.on_result.get(slot)):
                        report(command, ok)
            nxt = min((c[0] for c in self._in_flight), default=end)
            n = max(1, min(end, nxt) - self.frame)
            self._check(_dll.gary_env_step(self._h, n))
            self.frame += n

    def act(self, slot: int, command: bytes) -> bool | None:
        """Send one command for a player slot. Without a command delay it runs now and returns
        whether the engine accepted it. With one, it returns None and runs command_delay frames
        later; on_result[slot] then hears whether it was accepted."""
        if self.command_delay <= 0:
            return self._run(slot, command)
        self._in_flight.append((self.frame + self.command_delay, slot, bytes(command)))
        return None

    def _run(self, slot: int, command: bytes) -> bool:
        r = _dll.gary_env_act(self._h, slot, command, len(command))
        self._check(r >= 0)
        return r == 1

    def observe(self) -> dict:
        return decode_observation(self.observe_raw())

    def observe_raw(self) -> bytes:
        """The observation undecoded (decode_observation turns it into observe()'s dict): for
        keeping many snapshots of which only a few are ever read."""
        return _dll.gary_env_observe(self._h)

    def unit_at(self, slot: int, x: int, y: int) -> int:
        """Tag of the unit a click by this player at map pixel (x, y) would hit, or 0."""
        return _dll.gary_env_unit_at(self._h, slot, x, y)

    def box_select(self, slot: int, x0: int, y0: int, x1: int, y1: int) -> list[int]:
        """Tags a drag box (map pixels) would select for this player (up to 12)."""
        out = (ctypes.c_uint * 12)()
        n = _dll.gary_env_box_select(self._h, slot, x0, y0, x1, y1, out, 12)
        return list(out[:n])

    def unit_type_info(self, unit_type: int) -> dict | None:
        """A unit type's numbers from the game data: placement size (w, h, pixels), collision box
        (left, up, right, down: pixels from its position, inclusive), cost, build time, hit points."""
        out = (ctypes.c_int * 11)()
        if not _dll.gary_env_unit_type_info(self._h, unit_type, out):
            return None
        return {"w": out[0], "h": out[1], "left": out[2], "up": out[3], "right": out[4], "down": out[5],
                "minerals": out[6], "gas": out[7], "build_frames": out[8], "hp": out[9], "supply_x2": out[10]}

    def create_unit(self, owner: int, unit_type: int, x: int, y: int) -> int:
        """Test scenarios only: a finished unit at a map pixel (a building exactly there). Its tag, or 0."""
        return _dll.gary_env_create_unit(self._h, owner, unit_type, x, y)

    def creep(self):
        """Creep now, per 32x32 tile (numpy bool array, map tiles high x wide)."""
        import numpy as np
        obs = self.observe()
        tw, th = obs["map"]["w"] // 32, obs["map"]["h"] // 32
        out = np.zeros(tw * th, dtype=np.uint8)
        _dll.gary_env_creep(self._h, out.ctypes.data)
        return out.reshape(th, tw).astype(bool)

    def debug_builder(self, tag: int) -> int:
        """Debugging: why an SCV's build order isn't starting (bits): 1 not at its move target, 2 more
        than 128 px from the spot, 4 not enough resources, 8 can't place, 16 empty build queue,
        32 immovable (stuck)."""
        return _dll.gary_env_debug_builder(self._h, tag)

    def debug_place(self, slot: int, unit_type: int, tile_x: int, tile_y: int) -> int:
        """Debugging: why the engine would refuse this build order from the player's selected unit
        (bits): 1 not one unit selected, 2 the order isn't valid for it, 4 can't place, 8 can't reach
        the spot, 16 a tile isn't explored. 0: it would take it (unless the money is short)."""
        return _dll.gary_env_debug_place(self._h, slot, unit_type, tile_x, tile_y)

    def set_upgrade(self, owner: int, upgrade: int, level: int = 1) -> None:
        """Test fixtures only: a player's upgrade level, on its units at once."""
        _dll.gary_env_set_upgrade(self._h, owner, upgrade, level)

    def set_resources(self, owner: int, minerals: int, gas: int) -> None:
        """Test fixtures only: a player's minerals and gas, set outright."""
        _dll.gary_env_set_resources(self._h, owner, minerals, gas)

    def selection(self, slot: int) -> list[int]:
        """The units this player has selected right now (tags)."""
        out = (ctypes.c_uint * 12)()
        n = _dll.gary_env_selection(self._h, slot, out, 12)
        return list(out[:n])

    def type_select(self, slot: int, x: int, y: int, view: tuple[int, int, int, int]) -> list[int]:
        """Tags a ctrl+click (or double-click) at map pixel (x, y) would select for this player: the
        clicked unit's type in the view rectangle (map pixels x0, y0, x1, y1), up to 12; empty if
        the click hits no own unit that can be selected with others."""
        out = (ctypes.c_uint * 12)()
        n = _dll.gary_env_type_select(self._h, slot, x, y, *view, out, 12)
        return list(out[:n])

    def unit_type_of(self, tag: int) -> int:
        return _dll.gary_env_unit_type(self._h, tag)

    def set_name(self, slot: int, name: str) -> None:
        """Rename a player (also in replays saved from this game)."""
        _dll.gary_env_set_name(self._h, slot, name.encode()[:24])

    def can_place(self, slot: int, unit_type: int, tile_x: int, tile_y: int, builder_tag: int = 0) -> bool:
        """The game's own placement check (what the green/red grid shows the player)."""
        return _dll.gary_env_can_place(self._h, slot, builder_tag, unit_type, tile_x, tile_y)

    def depot_spot_ok(self, tile_x: int, tile_y: int) -> bool:
        """Map knowledge: a resource depot fits here by terrain and resource distance."""
        return _dll.gary_env_depot_spot_ok(self._h, tile_x, tile_y)

    def terrain(self):
        """The map's terrain (numpy arrays): walkable and ground height (0 low, 1 middle, 2 high)
        per 8x8-pixel minitile, shape (map tiles high * 4, wide * 4); buildable per 32x32 tile."""
        import numpy as np
        obs = self.observe()
        tw, th = obs["map"]["w"] // 32, obs["map"]["h"] // 32
        walk, height, build = (ctypes.create_string_buffer(tw * th * 16), ctypes.create_string_buffer(tw * th * 16),
                               ctypes.create_string_buffer(tw * th))
        if _dll.gary_env_terrain(self._h, walk, height, build) < 0:
            raise GameError(_dll.gary_env_error(self._h).decode(errors="replace"))
        grid = lambda b, h_, w_: np.frombuffer(b.raw, np.uint8).reshape(h_, w_).copy()
        return grid(walk, th * 4, tw * 4).astype(bool), grid(height, th * 4, tw * 4), grid(build, th, tw).astype(bool)

    def regions(self) -> dict:
        """The game's own ground pathing regions: "region" per minitile (numpy, -1 where none),
        per region its walkable "group" (ground units get only within one) and whether it is
        "walkable", and the "neighbors" (pairs of regions a ground unit walks between)."""
        import numpy as np
        obs = self.observe()
        mw, mh = obs["map"]["w"] // 8, obs["map"]["h"] // 8
        region = (ctypes.c_int * (mw * mh))()
        cap, pcap = 5000, 40000
        group, walk = (ctypes.c_int * cap)(), (ctypes.c_uint8 * cap)()
        pairs, npairs = (ctypes.c_int * (2 * pcap))(), ctypes.c_int()
        n = _dll.gary_env_regions(self._h, region, group, walk, cap, pairs, pcap, ctypes.byref(npairs))
        if n < 0:
            raise GameError(_dll.gary_env_error(self._h).decode(errors="replace"))
        return {"region": np.frombuffer(region, np.int32).reshape(mh, mw).copy(),
                "group": np.frombuffer(group, np.int32)[:n].copy(), "walkable": np.frombuffer(walk, np.uint8)[:n].astype(bool),
                "neighbors": [(pairs[2 * i], pairs[2 * i + 1]) for i in range(npairs.value)]}

    def walkable(self, x: int, y: int) -> bool:
        """Whether ground units can stand at map pixel (x, y) (terrain only)."""
        return _dll.gary_env_walkable(self._h, x, y)

    def start_locations(self) -> list[dict]:
        """Map knowledge: the start locations' pixel centers."""
        return json.loads(_dll.gary_env_start_locations(self._h).decode())

    def save_replay(self, path: str | Path) -> None:
        self._check(_dll.gary_env_save_replay(self._h, str(path).encode()))

    def close(self) -> None:
        if self._h:
            _dll.gary_env_destroy(self._h)
            self._h = None

    def __enter__(self) -> "Game":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
