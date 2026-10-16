#!/usr/bin/env python3
"""Drive the SC:R menus to start a custom melee game vs one computer — no human clicks.

    python adapters/scr_bridge/tools/auto_game.py               # click through to a melee game
    python adapters/scr_bridge/tools/auto_game.py --calibrate   # screenshot the game window only
    python adapters/scr_bridge/tools/auto_game.py --click 0.5,0.3   # one click at (fx, fy)
    python adapters/scr_bridge/tools/auto_game.py --seq tvz         # a named click sequence (SEQUENCES)
    python adapters/scr_bridge/tools/auto_game.py --target ok       # one named click (CLICKS)

Clicks are (fx, fy) fractions of the game window rect, so the map is resolution-independent.
Every run saves build/auto_game/step_NN_before.png screenshots; the STEPS table is calibrated
from those (that loop needs no human either: run --calibrate, read the shot, add a step, repeat).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "adapters" / "scr_bridge"))

import ctypes  # noqa: E402
from ctypes import wintypes  # noqa: E402

from PIL import Image, ImageGrab  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "build" / "auto_game"
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# Calibrated click map: label, (fx, fy) of the game window (fullscreen layout: screen-sized
# window with the 4:3 game content letterboxed inside). Calibrated 2026-10-03 against
# 1.23.10.13515. Only TEXT labels are buttons in BW menus — clicking art just cancels dialogs.
# SC:R restores its last menu screen on the next launch (observed: the single-player menu), so
# early steps are no-ops unless their screen is up. NEVER click (0.755, 0.894): that is the
# single-player menu's "Cancel", which quits the game (ESC at the top-level menu does too).
STEPS: list[tuple[str, float, float]] = [
    ("single_player", 0.371, 0.256),  # main menu -> "Select Game Type" dialog (else no-op)
    ("expansion", 0.557, 0.619),      # dialog: "Expansion" text -> character pick (else no-op)
    ("registry_ok", 0.730, 0.822),    # character pick ("kuia") -> Ok -> single-player menu
    ("play_custom", 0.512, 0.861),    # single-player menu -> Create game screen
    ("create_ok", 0.730, 0.820),      # Create game (map preset, Melee, one Computer) -> Ok
]

# Multiplayer lobby race picker, fractions of the window rect (fullscreen layout), calibrated
# 2026-10-03 against 1.23.10.13515 from lobby shots. Each player row is [player][race][color];
# the COLOR combo (right) shows "Random" by default and is NOT the race picker. The race list
# pops below the combo: Zerg, Terran, Protoss, Random. The home-team row is at fy 0.113; the
# away-team row (the joined player's own row) is LOBBY_ROW_FY_AWAY lower. SC:R polls the real
# cursor for these widgets (posted clicks are ignored), so pick_race uses real-input clicks:
# fine in the lobby because multiplayer never pauses and the swap takes <1s.
LOBBY_RACE_DROPDOWN: tuple[float, float] = (0.395, 0.113)  # the "Select Race" combo, home row
LOBBY_RACE_ITEMS: dict[str, tuple[float, float]] = {
    "Z": (0.395, 0.142),
    "T": (0.395, 0.160),
    "P": (0.395, 0.178),
    "R": (0.395, 0.196),
}
LOBBY_ROW_FY_AWAY = 0.066  # add for the away-team row (--row away); measured 95px at 1440p

# Single-player clicks (real input, fullscreen layout), calibrated 2026-10-15 at 2560x1440 against
# 1.23.10. The Create screen (Play Custom) remembers the map and folder, not the races or slots: a
# fresh one has the local player Random-raced and three Computer slots. Its race and slot widgets are
# dropdowns: click the combo, then the item in the list that pops below it.
CLICKS: dict[str, tuple[float, float]] = {
    # single-player menu
    "play_custom": (0.510, 0.864),
    # Create screen: the map list and folders (a folder opens with a select click plus Ok)
    "map_row_1": (0.249, 0.293),          # first row of the map list ([ladder] in the maps root)
    "map_scroll_up": (0.523, 0.289),      # the list's up arrow
    "season_11_row": (0.270, 0.373),      # ladder folder scrolled to the top: Season 11
    "fighting_spirit_row": (0.315, 0.492),  # Season 11: [4]Fighting_Spirit_1.3 (Season 10 has it too)
    "ok": (0.730, 0.820),                 # Ok: opens a selected folder, or starts the game
    # Create screen: player rows (the local player first, then slots 2-4)
    "my_race": (0.438, 0.672),
    "my_race_terran": (0.375, 0.736),
    "slot2_race": (0.438, 0.712),         # the first Computer's race
    "slot2_race_zerg": (0.369, 0.742),
    "slot3": (0.252, 0.751),              # slot 3's player combo (Computer / Closed / ...)
    "slot3_closed": (0.187, 0.782),
    "slot4": (0.326, 0.790),
    "slot4_closed": (0.187, 0.822),
    # LAN Create screen (lan_park.py's host stops there): the same map list, plus a game type
    "game_type": (0.522, 0.560),          # its dropdown (Top vs Bottom by default)
    "game_type_melee": (0.358, 0.624),
    "season_10_fighting_spirit_row": (0.315, 0.412),  # Season 10 scrolled to the top
    # LAN lobby, Melee: player rows 0.040 apart (the host's at 0.113: LOBBY_RACE_DROPDOWN, --race)
    "lobby_row2_race": (0.389, 0.148),    # the joined player's race, on its own client
    "lobby_row2_race_zerg": (0.362, 0.175),
    # in a match: the game menu (the console's Menu button, or F10)
    "game_menu": (0.651, 0.828),
    "end_mission": (0.494, 0.524),
    "quit_mission": (0.494, 0.312),       # pressed twice: the second confirms
    "minimap_corner": (0.137, 0.889),     # in a match: the minimap, to move the real camera away
}
# Named sequences for --seq: (target, seconds to settle after the click)
SEQUENCES: dict[str, list[tuple[str, float]]] = {
    # the Create screen at the maps root -> Fighting Spirit selected (skip if it's already selected:
    # the screen remembers the folder and map)
    "fighting-spirit": [("map_row_1", 2), ("ok", 3), ("map_scroll_up", 1), ("map_scroll_up", 1),
                        ("season_11_row", 2), ("ok", 3), ("fighting_spirit_row", 2)],
    # the Create screen -> local player Terran vs one Zerg computer (Gary's live-test setup)
    "tvz": [("my_race", 1), ("my_race_terran", 1), ("slot3", 1), ("slot3_closed", 1),
            ("slot4", 1), ("slot4_closed", 1), ("slot2_race", 1), ("slot2_race_zerg", 2)],
    # LAN guest in the lobby (Melee, second row) -> Zerg
    "lan-guest-zerg": [("lobby_row2_race", 1), ("lobby_row2_race_zerg", 1)],
    # in a match -> back at the menus; the client writes Maps/Replays/LastReplay.rep
    "end": [("game_menu", 2), ("end_mission", 2), ("quit_mission", 3), ("quit_mission", 4)],
}

# Menu key flows for --preset (posted background keys). lan-create/lan-join end in the
# multiplayer LOBBY (not a running game), so they skip the in_game() success check.
# lan-create: M=Multiplayer, E=xpansion, Down=LAN (Battle.net starts selected), O=K,
# O=K on the Gary account (first, already selected), G=Create game, O=K (Bottleneck
# preset map is preselected). lan-join: same until the Games screen, O=Ok joins the
# selected (only) listed game.
PRESETS: dict[str, tuple[str, bool]] = {
    "melee": ("S,E,O,U,O", True),
    "lan-create": ("M,E,Down,O,O,G,O", False),
    "lan-join": ("M,E,Down,O,O,O", False),
}


class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", ctypes.c_wchar * 260),
    ]


def find_process(name: str) -> int:
    """PID of the first running process whose image name matches (\"StarCraft\" -> StarCraft.exe)."""
    want = name if name.lower().endswith(".exe") else name + ".exe"
    snap = kernel32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    if snap in (0, -1, 0xFFFFFFFF):
        return 0
    try:
        pe = _PROCESSENTRY32W()
        pe.dwSize = ctypes.sizeof(_PROCESSENTRY32W)
        ok = kernel32.Process32FirstW(snap, ctypes.byref(pe))
        while ok:
            if pe.szExeFile.lower() == want.lower():
                return int(pe.th32ProcessID)
            ok = kernel32.Process32NextW(snap, ctypes.byref(pe))
        return 0
    finally:
        kernel32.CloseHandle(snap)


TARGET_PID = 0  # set from --pid: when nonzero, only that StarCraft.exe window is considered


def find_game_window(pid: int = 0) -> int:
    target = pid or TARGET_PID or find_process("StarCraft")
    if not target:
        return 0
    best = {"hwnd": 0, "area": -1}

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, lparam):
        wpid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(wpid))
        if wpid.value == lparam and user32.IsWindowVisible(hwnd):
            r = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(r))
            area = max(0, r.right - r.left) * max(0, r.bottom - r.top)
            if area > best["area"]:
                best["hwnd"] = hwnd
                best["area"] = area
        return True  # visit every window and keep the largest (the game view)

    user32.EnumWindows(cb, target)
    return best["hwnd"]


def window_rect(hwnd: int) -> tuple[int, int, int, int]:
    r = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(r)):
        raise RuntimeError(f"GetWindowRect failed (hwnd={hwnd} — game window gone?)")
    if r.right <= r.left or r.bottom <= r.top:
        raise RuntimeError(f"degenerate game window rect {(r.left, r.top, r.right, r.bottom)}")
    return r.left, r.top, r.right, r.bottom


def ensure_fullscreen(hwnd: int) -> int:
    """The STEPS table is calibrated for the fullscreen layout (screen-sized window; the 4:3
    game content is letterboxed inside it). SC:R can start windowed or flip modes, so toggle
    with Alt+Enter until the window covers the screen. Returns the (possibly new) hwnd."""
    sw, sh = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    for _ in range(3):
        hwnd = find_game_window() or hwnd
        try:
            if window_rect(hwnd) == (0, 0, sw, sh):
                return hwnd
        except RuntimeError:
            pass  # window mid-recreation (SC:R does that while switching modes)
        raise_window(hwnd)
        user32.keybd_event(0x12, 0, 0, 0)          # VK_MENU down
        user32.keybd_event(0x0D, 0, 0, 0)          # VK_RETURN down
        user32.keybd_event(0x0D, 0, 0x0002, 0)     # VK_RETURN up
        user32.keybd_event(0x12, 0, 0x0002, 0)     # VK_MENU up
        time.sleep(3.0)
    raise SystemExit(f"game window stuck at {window_rect(hwnd)}, not fullscreen "
                     f"{(0, 0, sw, sh)} — the calibrated STEPS need the fullscreen layout")


def raise_window(hwnd: int) -> None:
    """Bring the game to the foreground. Plain SetForegroundWindow is refused when our
    process does not own the foreground (Windows lock), so attach to the current
    foreground thread's input queue first — the standard bypass."""
    user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    cur = kernel32.GetCurrentThreadId()
    fg_tid = user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), None)
    attached = bool(fg_tid and fg_tid != cur)
    if attached:
        user32.AttachThreadInput(cur, fg_tid, True)
    try:
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
    finally:
        if attached:
            user32.AttachThreadInput(cur, fg_tid, False)
    time.sleep(0.4)


def shot(hwnd: int, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    # ImageGrab captures screen pixels at the window rect, so the game must be on top
    # (a fullscreen-sized background window would capture whatever covers it).
    raise_window(hwnd)
    for attempt in range(3):
        try:
            ImageGrab.grab(bbox=window_rect(hwnd)).save(path)
            return path
        except (RuntimeError, ValueError):
            time.sleep(1.5)  # SC:R recreates its window while switching display modes
            hwnd = find_game_window() or hwnd
    raise SystemExit(f"could not screenshot the game window (hwnd={hwnd})")


def click(hwnd: int, fx: float, fy: float) -> tuple[int, int]:
    hwnd = find_game_window() or hwnd
    l, t, r, b = window_rect(hwnd)
    x = l + int((r - l) * fx)
    y = t + int((b - t) * fy)
    raise_window(hwnd)
    user32.SetCursorPos(x, y)
    time.sleep(0.15)
    user32.mouse_event(0x0002, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTDOWN
    time.sleep(0.06)
    user32.mouse_event(0x0004, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTUP
    return x, y


def post_click(hwnd: int, fx: float, fy: float) -> tuple[int, int]:
    """Background click: posted mouse messages, no real cursor, no focus steal.

    NOTE: SC:R's own widgets ignore these (they poll the real cursor) — lobby/menus need
    click() (real input, raises the window first). post_click remains for plain Win32
    windows and for diagnosing which path a widget honors. Same (fx, fy) convention.
    """
    l, t, r, b = window_rect(hwnd)
    x = l + int((r - l) * fx)
    y = t + int((b - t) * fy)
    lp = ((y - t) << 16) | ((x - l) & 0xFFFF)
    user32.PostMessageW(hwnd, 0x0200, 0, lp)        # WM_MOUSEMOVE
    user32.PostMessageW(hwnd, 0x0201, 0x0001, lp)   # WM_LBUTTONDOWN (MK_LBUTTON)
    user32.PostMessageW(hwnd, 0x0202, 0, lp)        # WM_LBUTTONUP
    print(f"-> posted click at ({x}, {y})  fx={fx:.3f} fy={fy:.3f}")
    return x, y


def pick_race(hwnd: int, race: str, row: str = "home") -> None:
    """Multiplayer lobby: open the race dropdown and pick the race.

    Real-input clicks (click()) because SC:R widgets ignore posted mouse messages; each click
    briefly raises Gary's window first (never paused: multiplayer runs unfocused). row="away"
    targets the away-team row — the joined player's own row (the host owns the home row).
    """
    race = race.upper()
    if race not in LOBBY_RACE_ITEMS or tuple(LOBBY_RACE_DROPDOWN) == (0.0, 0.0):
        raise SystemExit(
            "lobby race picker not calibrated yet — run --preset lan-create, read the lobby "
            "shot, and fill LOBBY_RACE_DROPDOWN / LOBBY_RACE_ITEMS in tools/auto_game.py")
    dy = LOBBY_ROW_FY_AWAY if row == "away" else 0.0
    bg_shot(hwnd, "race_00_before_dropdown.png")
    click(hwnd, LOBBY_RACE_DROPDOWN[0], LOBBY_RACE_DROPDOWN[1] + dy)
    time.sleep(1.0)
    bg_shot(hwnd, "race_01_dropdown_open.png")
    fx, fy = LOBBY_RACE_ITEMS[race]
    click(hwnd, fx, fy + dy)
    time.sleep(1.0)
    bg_shot(hwnd, f"race_02_after_{race}.png")
    print(f"race '{race}' picked ({row} row) — verify race_02_after_{race}.png")


def in_game() -> bool:
    try:
        from gary.scr_env import ScrGame

        return bool(ScrGame.connect().status().get("in_game"))
    except Exception:
        return False


def match_started() -> bool:
    """True only while the simulation is ticking. The bridge counts the multiplayer LOBBY as
    in_game=true (sane map size + local player already set there), so frames > 0 is the signal
    that distinguishes "in the lobby" from "match running"."""
    try:
        from gary.scr_env import ScrGame

        return (ScrGame.connect().status().get("frames", 0) or 0) > 0
    except Exception:
        return False


# ---- keyboard menu navigation (default): no visible clicks, works on an occluded window ----
# SC:R menus respond to hotkey letters (S=ingle player, E=xpansion, O=K, cU=stom, ...), so the
# whole menu walk is one typed sequence with settle delays. Keys are POSTED to the window
# (WM_KEYDOWN/CHAR/KEYUP) so the game needs neither focus nor visibility — the user's screen
# stays theirs. If a menu ever ignores posted input, --foreground-keys uses the real input path.

PW_RENDERFULLCONTENT = 2
WM_KEYDOWN, WM_KEYUP, WM_CHAR = 0x0100, 0x0101, 0x0102


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD), ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG), ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD),
    ]


def bg_shot(hwnd: int, name: str) -> Path:
    """PrintWindow capture — unlike shot(), works while the window is occluded or behind other
    windows, so step evidence never requires the user's visible screen."""
    gdi32 = ctypes.windll.gdi32
    l, t, r, b = window_rect(hwnd)
    w, h = r - l, b - t
    hdc = user32.GetWindowDC(hwnd)
    memdc = gdi32.CreateCompatibleDC(hdc)
    bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
    old = gdi32.SelectObject(memdc, bmp)
    ok = user32.PrintWindow(hwnd, memdc, PW_RENDERFULLCONTENT)
    bmi = BITMAPINFOHEADER()
    bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.biWidth, bmi.biHeight = w, -h  # top-down
    bmi.biPlanes, bmi.biBitCount = 1, 32
    buf = ctypes.create_string_buffer(w * h * 4)
    gdi32.GetDIBits(memdc, bmp, 0, h, buf, ctypes.byref(bmi), 0)
    gdi32.SelectObject(memdc, old)
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(memdc)
    user32.ReleaseDC(hwnd, hdc)
    img = Image.frombytes("RGB", (w, h), buf.raw, "raw", "BGRX")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    img.save(path)
    if not ok:
        print(f"  note: PrintWindow returned 0 for {name} — image may be blank")
    return path


# Named keys for --keys tokens: name -> (virtual key, scan code, extended-key flag)
NAMED_KEYS = {
    "UP": (0x26, 0x48, True), "DOWN": (0x28, 0x50, True),
    "LEFT": (0x25, 0x4B, True), "RIGHT": (0x27, 0x4D, True),
    "ENTER": (0x0D, 0x1C, False), "RETURN": (0x0D, 0x1C, False),
    "ESC": (0x1B, 0x01, False), "ESCAPE": (0x1B, 0x01, False),
    "TAB": (0x09, 0x0F, False), "SPACE": (0x20, 0x39, False),
    "BACKSPACE": (0x08, 0x0E, False),
}


def _lparam(scan: int, down: bool, extended: bool = False, alt: bool = False) -> int:
    lp = 1 | ((scan & 0xFF) << 16)
    if extended:
        lp |= 1 << 24
    if alt:
        lp |= 1 << 29
    if not down:
        lp |= (1 << 30) | (1 << 31)
    return lp


def send_key(hwnd: int, key: str, background: bool = True) -> None:
    """Send one key: a letter/digit, a named key (Down/Enter/Esc/...), or Alt+X (accelerator).

    background=True posts the messages directly (no focus change at all);
    background=False focuses the window and uses the real input path.
    """
    key = key.strip().upper()
    ext = False
    if key.startswith("ALT+") and len(key) == 5:
        ch = key[4]
        vk = ord(ch)
        scan = user32.MapVirtualKeyW(vk, 0) & 0xFF
    elif key in NAMED_KEYS:
        vk, scan, ext = NAMED_KEYS[key]
    elif len(key) == 1:
        vk = ord(key)
        scan = user32.MapVirtualKeyW(vk, 0) & 0xFF
    else:
        raise ValueError(f"unknown key token: {key!r}")

    if background:
        if key.startswith("ALT+"):  # menu accelerator: Alt down, X down/char/up, Alt up
            WM_SYSKEYDOWN, WM_SYSKEYUP, WM_SYSCHAR = 0x0104, 0x0105, 0x0106
            user32.PostMessageW(hwnd, WM_SYSKEYDOWN, 0x12, _lparam(0x38, True))
            user32.PostMessageW(hwnd, WM_SYSKEYDOWN, vk, _lparam(scan, True, alt=True))
            user32.PostMessageW(hwnd, WM_SYSCHAR, vk, _lparam(scan, True, alt=True))
            user32.PostMessageW(hwnd, WM_SYSKEYUP, vk, _lparam(scan, False, alt=True))
            user32.PostMessageW(hwnd, WM_SYSKEYUP, 0x12, _lparam(0x38, False))
        else:
            user32.PostMessageW(hwnd, WM_KEYDOWN, vk, _lparam(scan, True, ext))
            if len(key) == 1 and not ext:
                user32.PostMessageW(hwnd, WM_CHAR, vk, _lparam(scan, True))
            user32.PostMessageW(hwnd, WM_KEYUP, vk, _lparam(scan, False, ext))
    else:
        raise_window(hwnd)
        if key.startswith("ALT+"):
            user32.keybd_event(0x12, 0x38, 0, 0)
            user32.keybd_event(vk, scan, 0, 0)
            user32.keybd_event(vk, scan, 0x0002, 0)
            user32.keybd_event(0x12, 0x38, 0x0002, 0)
        else:
            user32.keybd_event(vk, scan, 1 if ext else 0, 0)
            user32.keybd_event(vk, scan, 0x0002 | (1 if ext else 0), 0)


def main() -> None:
    global TARGET_PID
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true", help="screenshot the game window, click nothing")
    ap.add_argument("--click", metavar="FX,FY", help="perform exactly one click, then screenshot")
    ap.add_argument("--seq", choices=sorted(SEQUENCES),
                    help="run a named click sequence (CLICKS/SEQUENCES), then screenshot: "
                         "fighting-spirit, tvz (Terran vs Zerg computer), end (end the match)")
    ap.add_argument("--target", choices=sorted(CLICKS), help="one click on a named target, then screenshot")
    ap.add_argument("--settle", type=float, default=4.0,
                    help="seconds to wait after each click (SC:R textures load late)")
    ap.add_argument("--mode", choices=("keys", "clicks"), default="keys",
                    help="keys: hotkey navigation, no visible interaction (default); "
                         "clicks: the calibrated screenshot+click flow")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="",
                    help="named key flows: melee (S,E,O,U,O) / lan-create (host) / lan-join")
    ap.add_argument("--keys", default="",
                    help="comma-separated key tokens (default: from --preset, else S,E,O,U,O): "
                         "letters, Down/Up/Left/Right, Enter, Esc, Tab, Space, Alt+X")
    ap.add_argument("--esc", type=int, default=0,
                    help="send N ESC keys first (backs out of a restored submenu; NEVER at the "
                         "top-level main menu, that quits the game)")
    ap.add_argument("--pid", type=int, default=0,
                    help="drive this StarCraft.exe pid (needed when two clients run)")
    ap.add_argument("--bg-click", metavar="FX,FY",
                    help="one background (posted) click at (fx, fy), then screenshot")
    ap.add_argument("--race", metavar="T|Z|P|R",
                    help="lobby: pick Terran/Zerg/Protoss/Random on the race dropdown")
    ap.add_argument("--row", choices=("home", "away"), default="home",
                    help="which lobby row --race targets: home (host) or away (joined player)")
    ap.add_argument("--start", action="store_true",
                    help="lobby (host): post Alt+O to start the game (5s countdown)")
    ap.add_argument("--delay", type=float, default=2.0, help="seconds to wait after each key")
    ap.add_argument("--foreground-keys", action="store_true",
                    help="focus the game and use real input instead of posted messages")
    args = ap.parse_args()
    TARGET_PID = args.pid

    hwnd = find_game_window()
    if not hwnd:
        raise SystemExit("game window not found — launch.py first")
    print(f"game window: hwnd={hwnd} rect={window_rect(hwnd)}")

    if args.calibrate:
        p = shot(hwnd, f"calib_{int(time.time())}.png")
        print(f"calibration screenshot: {p}")
        return

    if args.seq:
        for name, settle in SEQUENCES[args.seq]:
            pos = click(hwnd, *CLICKS[name])
            print(f"{name}: clicked at {pos}")
            time.sleep(settle)
        print(f"screenshot: {shot(hwnd, f'after_{args.seq}.png')}")
        return

    if args.click or args.target:
        fx, fy = CLICKS[args.target] if args.target else (float(v) for v in args.click.split(","))
        pos = click(hwnd, fx, fy)
        print(f"clicked at {pos}")
        shot(hwnd, "after_click.png")
        return

    if args.bg_click:
        fx, fy = (float(v) for v in args.bg_click.split(","))
        pos = post_click(hwnd, fx, fy)
        print(f"posted click at {pos}")
        bg_shot(hwnd, "after_bg_click.png")
        return

    if args.mode == "keys":
        preset_keys, preset_expect_game = PRESETS.get(args.preset, ("", True))
        keys_str = args.keys or preset_keys or ("S,E,O,U,O" if not (args.race or args.start) else "")
        keys = ["ESC"] * args.esc + [k.strip().upper() for k in keys_str.split(",") if k.strip()]
        expect_game = preset_expect_game and not args.race and not args.start
        mode = "foreground" if args.foreground_keys else "background"
        print(f"keyboard navigation ({mode} keys, {args.delay}s apart): {' '.join(keys)}")
        for i, k in enumerate(keys):
            hwnd = find_game_window() or hwnd
            p = bg_shot(hwnd, f"key_{i:02d}_before_{k}.png")
            print(f"key {i} '{k}': state shot {p.name}")
            if expect_game and match_started():
                print("match already running — stopping")
                break
            send_key(hwnd, k, background=not args.foreground_keys)
            time.sleep(args.delay)
        hwnd = find_game_window() or hwnd
        bg_shot(hwnd, "key_final.png")

        if args.race:
            pick_race(hwnd, args.race, args.row)
        if args.start:
            send_key(hwnd, "Alt+O", background=not args.foreground_keys)
            print("Alt+O posted (host start, 5s countdown) — waiting for the simulation...")
            deadline = time.time() + 25
            while time.time() < deadline:
                if match_started():
                    print("SUCCESS: countdown over, simulation ticking (frames > 0)")
                    return
                time.sleep(1)
            raise SystemExit("Alt+O sent but the game did not start — check build/auto_game/*.png")
        if not expect_game:
            print("lobby flow done — verify key_final.png (and race_*.png if --race); start "
                  "later with --start once the other client has joined and picked its race")
            return

        deadline = time.time() + 15
        while time.time() < deadline:
            if match_started():
                print("SUCCESS: match running (frames > 0)")
                return
            time.sleep(1)
        raise SystemExit(
            "keys sent but the game did not start — read build/auto_game/key_*.png "
            "(captured invisibly), adjust --keys, or retry with --foreground-keys / --mode clicks")

    if not STEPS:
        raise SystemExit(
            "STEPS is not calibrated yet. Run with --calibrate, read build/auto_game/*.png, "
            "and fill the STEPS table (label, fx, fy) in tools/auto_game.py.")

    hwnd = ensure_fullscreen(hwnd)
    print(f"fullscreen layout ready: hwnd={hwnd} rect={window_rect(hwnd)}")

    for i, (label, fx, fy) in enumerate(STEPS):
        hwnd = find_game_window() or hwnd
        p = shot(hwnd, f"step_{i:02d}_before_{label}.png")
        print(f"step {i} {label}: shot {p.name}")
        if match_started():
            print("match already running — stopping")
            break
        pos = click(hwnd, fx, fy)
        print(f"  clicked {pos}")
        time.sleep(args.settle)

    deadline = time.time() + 30
    while time.time() < deadline:
        if match_started():
            print("SUCCESS: match running (frames > 0)")
            return
        time.sleep(1)
    shot(hwnd, "final_state.png")
    raise SystemExit(
        "game did not start within 30s — read build/auto_game/step_*.png and final_state.png "
        "and correct the STEPS map")


if __name__ == "__main__":
    main()