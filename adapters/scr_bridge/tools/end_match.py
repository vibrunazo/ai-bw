#!/usr/bin/env python3
"""End the current custom game through the in-game menu so SC:R saves LastReplay.rep.

Vision-free: the F10 game-menu dialog is found by diffing screenshots, its buttons by their
bright frames, and each click is verified against the bridge's in_game flag and the replay
file — "Return to Game"/other buttons just reopen the menu and the next candidate is tried.
ESC (safe in-game; only the top-level main menu quits on ESC) cancels any sub-dialog.

    python adapters/scr_bridge/tools/end_match.py [--pid N]
"""
from __future__ import annotations

import os
import ctypes
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image, ImageChops  # noqa: E402

from auto_game import click, find_game_window, raise_window, shot, window_rect  # noqa: E402
from gary.scr_env import ScrGame  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "build" / "auto_game"
REPLAY_DIRS = [
    Path.home() / "Documents" / "StarCraft" / "Maps" / "Replays",
    *([Path(os.environ["GARY_SC_INSTALL"]) / "Maps" / "Replays", Path(os.environ["GARY_SC_INSTALL"]) / "Maps" / "Save"]
      if os.environ.get("GARY_SC_INSTALL") else []),     # (GARY_SC_INSTALL: your StarCraft folder)
]
user32 = ctypes.windll.user32


def find_replay() -> Path | None:
    for d in REPLAY_DIRS:
        p = d / "LastReplay.rep"
        if p.exists():
            return p
    return None


def match_running() -> bool:
    """Strict: a connection failure raises — a silent False here once faked a match end.
    frames > 0, because the bridge counts the multiplayer LOBBY as in_game=true (sane map
    size + local player are already set there)."""
    st = ScrGame.connect().status()
    return (st.get("frames", 0) or 0) > 0


def press(keys: int) -> None:
    user32.keybd_event(keys, 0, 0, 0)
    user32.keybd_event(keys, 0, 0x0002, 0)


def diff_bbox(a: Path, b: Path) -> tuple[int, int, int, int] | None:
    ia, ib = Image.open(a).convert("L"), Image.open(b).convert("L")
    d = ImageChops.difference(ia, ib).point(lambda v: 255 if v > 24 else 0)
    return d.getbbox()


def button_bands(menu_shot: Path, bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    """Button rows in the dialog: contiguous row bands of bright pixels in the center strip."""
    im = Image.open(menu_shot).convert("L").crop(bbox)
    w, h = im.size
    x0, x1 = int(w * 0.35), int(w * 0.65)
    px = im.load()
    rows = [sum(1 for x in range(x0, x1) if px[x, y] > 150) / max(1, x1 - x0) for y in range(h)]
    bands: list[tuple[int, int]] = []
    start = None
    for y, frac in enumerate(rows):
        if frac > 0.25 and start is None:
            start = y
        elif frac <= 0.25 and start is not None:
            if y - start >= 12 and start > 6 and y < h - 6:  # skip dialog frame edges
                bands.append((start, y))
            start = None
    return bands


def done(rep: Path | None) -> bool:
    return rep is not None or not match_running()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pid", type=int, default=0,
                    help="StarCraft.exe pid whose window to drive (needed when two clients run)")
    args = ap.parse_args()
    hwnd = find_game_window(args.pid)
    if not hwnd:
        raise SystemExit("game window not found")
    if not match_running():
        raise SystemExit("bridge says no match is running — start one first")
    rep = find_replay()
    if rep:
        print(f"replay already exists: {rep}")

    # Reference shot with no menu, then get the menu up and localize its dialog.
    ref = shot(hwnd, "end_ref.png")
    press(0x75)  # F10
    time.sleep(1.5)
    menu = shot(hwnd, "end_menu.png")
    bbox = diff_bbox(ref, menu)
    if not bbox or (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < 200 * 200:
        raise SystemExit("no game menu appeared after F10 — run from inside a match")
    print(f"menu dialog bbox={bbox}")
    bands = button_bands(menu, bbox)
    print(f"button bands (top..bottom): {bands}")
    if not bands:  # fall back to evenly spaced candidates inside the dialog
        bands = [(int((bbox[3] - bbox[1]) * f) + bbox[1], int((bbox[3] - bbox[1]) * f) + bbox[1] + 20)
                 for f in (0.60, 0.70, 0.80, 0.90)]
        print(f"fallback bands: {bands}")

    l, t, r, b = window_rect(hwnd)
    for idx in range(len(bands) - 1, -1, -1):  # bottom-up: "Quit Mission" sits low
        y0, y1 = bands[idx]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (y0 + y1) / 2 + (bbox[1] if y0 < bbox[3] - bbox[1] else 0)
        fx, fy = (cx - l) / (r - l), (cy - t) / (b - t)
        print(f"clicking button band {idx} at image=({cx:.0f},{cy:.0f}) fractions=({fx:.3f},{fy:.3f})")
        click(hwnd, fx, fy)
        time.sleep(2.5)
        rep = find_replay()
        if done(rep):
            print("match ended." + (f" replay: {rep}" if rep else " (waiting for replay file)"))
            return
        press(0x1B)  # cancel a possible sub-dialog (Options/Save); closes the menu otherwise
        time.sleep(1.0)
        now = shot(hwnd, "end_try.png")
        if diff_bbox(ref, now) is None:  # menu gone — reopen for the next candidate
            press(0x75)
            time.sleep(1.0)

    raise SystemExit("no button ended the match — end it manually (F10 -> Quit Mission)")


if __name__ == "__main__":
    main()
