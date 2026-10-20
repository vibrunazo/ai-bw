#!/usr/bin/env python3
"""Park two SC:R clients at the multiplayer screens: host at Create Game, guest at the LAN list.

    python adapters/scr_bridge/tools/lan_park.py

Keyboard-only menu walks (posted keys, like auto_game.py --mode keys — no visible clicks):
  1. launch the first client and inject the Gary bridge (launch.py)
  2. host:  M,E,Down,O,O,G -> Multiplayer, Expansion, LAN (Down from the preselected
     Battle.net), Ok, Ok on the Gary account (first, preselected), G=Create game: STOP in
     the create-game dialog (the game is NOT created)
  3. close the host's single-instance Event (close_mutex.py) so a second client can start
  4. launch a second plain client (StarCraft.exe -launch, no bridge)
  5. guest: M,E,Down,O,Down,O -> same menus, but Down picks the SECOND account and Ok logs
     it in: STOP at the LAN games list

Closed loop: SC:R renders its menu ~5s before it reads input, so blindly timed keys drift (a
run landed O on the main menu's Options). Each key is posted, then watched — one that did not
change the screen is resent (arrows only move a highlight, so they get a soft check instead of
resends). Evidence for every step: build/auto_game/lan_park_{host,guest}_*.png (PrintWindow
capture, works while the windows are occluded — the keys are posted, the screen stays yours).

The host flow is the calibrated auto_game.py lan-create preset minus its final Ok (which would
create the game); the guest is lan-join's prefix with the account Down and minus its final Ok
(which would join the listed game). Finish by hand: pick map/race in the host dialog and Ok to
create; the game appears in the guest's games list — join it. Gary plays the HOST client (the
one with the bridge), so its lobby row must be Terran. Once the match runs:

    python -m gary.play --live
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import auto_game as ag  # noqa: E402  (shared: find_game_window, send_key, bg_shot, find_process)
from PIL import Image  # noqa: E402

HOST_KEYS = "M,E,Down,O,O,G"      # M=Multiplayer, E=xpansion, Down=LAN (Battle.net preselected),
                                  # O=K, O=K on the Gary account (first), G=Create: STOP in the
                                  # create-game dialog (lan-create's final O would create it)
GUEST_KEYS = "M,E,Down,O,Down,O"  # same menus; at the account list Down picks the SECOND account
                                  # and O=K logs it in: STOP at the games list (lan-join's final
                                  # O would join the listed game)

# ---- closed-loop key walking: confirm every key landed, resend swallowed ones ----
# SC:R renders its menu a few seconds before it starts reading input (measured ~5s; keys posted
# earlier are silently dropped — one run drifted so far that O opened the main menu's Options).
# Each dialog key is posted, then watched: opening/closing a dialog changes much of the frame, so
# a key that changed nothing is resent. Arrows only move a highlight — under the noise floor of
# the animated menu background — so they skip the resend loop and get a soft check instead.

SIG_SIZE = (160, 90)   # coarse frame signature; enough to see dialogs appear/disappear
SIG_DELTA = 60         # per-sample |ΔR|+|ΔG|+|ΔB| above which a sample counts as changed
BIG_CHANGE = 0.04      # changed-sample fraction that means a dialog opened or closed
CHANGE_TIMEOUT = 8.0   # seconds a key gets to visibly change the screen
KEY_TRIES = 3          # sends of a swallowed key before giving up
ARM_SETTLE = 6.0       # after the menu renders, let the input path come up (measured ~5s)
NO_VERIFY = {"UP", "DOWN", "LEFT", "RIGHT"}   # selection moves: soft-checked, never resent


def run(cmd: list[str]) -> str:
    """Run a helper script, print its output, return it."""
    res = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    out = (res.stdout or "") + (res.stderr or "")
    for line in out.splitlines():
        if line.strip():
            print(f"  [{Path(cmd[-1]).name}] {line}")
    return out


def launch_gary() -> tuple[int, Path]:
    """launch.py: start the client, inject the bridge. Returns (pid, exe path)."""
    out = run([sys.executable, str(HERE.parent / "launch.py")])
    pid_m = re.search(r"client running as pid (\d+)", out)
    exe_m = re.search(r"launching (.+?\.exe)", out)
    if not pid_m or not exe_m:
        raise SystemExit("launch.py did not report a pid/exe — read its output above")
    return int(pid_m.group(1)), Path(exe_m.group(1))


def close_single_instance(pid: int) -> None:
    """close_mutex.py: close the named Event inside the client so a second one may start."""
    out = run([sys.executable, str(HERE / "close_mutex.py"), "--pid", str(pid)])
    if "closed" not in out:
        raise SystemExit(
            f"could not close the single-instance handle in pid {pid} — a second client "
            f"would just bounce to the first window (see close_mutex.py output above)")


def wait_window(pid: int, timeout: float = 45.0) -> int:
    """The pid's game window, polling while the client boots to its menu."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        hwnd = ag.find_game_window(pid)
        if hwnd:
            print(f"  pid {pid}: game window hwnd={hwnd} rect={ag.window_rect(hwnd)}")
            return hwnd
        time.sleep(1.0)
    raise SystemExit(f"pid {pid}: no game window within {timeout:.0f}s")


def frame_sig(path: Path) -> bytes:
    """Coarse RGB signature of a saved screenshot, for change detection."""
    return Image.open(path).convert("RGB").resize(SIG_SIZE).tobytes()


def sig_delta(a: bytes, b: bytes) -> float:
    """Fraction of signature samples that visibly changed between two frames."""
    changed = 0
    for i in range(0, len(a), 3):
        if abs(a[i] - b[i]) + abs(a[i + 1] - b[i + 1]) + abs(a[i + 2] - b[i + 2]) > SIG_DELTA:
            changed += 1
    return changed / (len(a) / 3)


def lit_fraction(path: Path) -> float:
    """Fraction of samples that are not near-black — the boot screens are black, menus are not."""
    d = Image.open(path).convert("RGB").resize(SIG_SIZE).tobytes()
    lit = sum(1 for i in range(0, len(d), 3) if d[i] + d[i + 1] + d[i + 2] > 48)
    return lit / (len(d) / 3)


def wait_menu(pid: int, hwnd: int, tag: str, timeout: float = 120.0) -> int:
    """Wait until the client renders a menu (not the black boot screen), then arm its input."""
    deadline = time.time() + timeout
    lit = 0
    while time.time() < deadline:
        hwnd = ag.find_game_window(pid) or hwnd
        p = ag.bg_shot(hwnd, f"lan_park_{tag}_boot.png")
        if lit_fraction(p) >= 0.10:
            lit += 1
            if lit >= 2:
                print(f"  {tag}: menu rendered ({p.name}) — letting input arm {ARM_SETTLE:.0f}s")
                time.sleep(ARM_SETTLE)
                return hwnd
        else:
            lit = 0
        time.sleep(0.5)
    raise SystemExit(f"{tag}: no menu on screen after {timeout:.0f}s — see lan_park_{tag}_boot.png")


def press(hwnd: int, pid: int, key: str, before: Path, tag: str, i: int, delay: float) -> str:
    """Post one key, confirm it landed (resending swallowed keys), settle `delay`. Status note.

    Dialog keys change much of the frame; if nothing changed, the key was eaten (SC:R drops
    input around startup) and is resent up to KEY_TRIES times. Each retry first re-checks the
    pre-key shot, so a key that landed late is never sent twice.
    """
    hwnd = ag.find_game_window(pid) or hwnd
    if key in NO_VERIFY:
        ag.send_key(hwnd, key, background=True)
        time.sleep(delay)
        p = ag.bg_shot(hwnd, f"lan_park_{tag}_{i:02d}_after_{key}.png")
        d = sig_delta(frame_sig(before), frame_sig(p))
        warn = "" if d >= 0.001 else "  <- frame unchanged, likely swallowed — check the shots"
        return f"selection move, frame delta {d:.2%}{warn}"
    base = frame_sig(before)
    for attempt in range(1, KEY_TRIES + 1):
        if attempt > 1:
            p = ag.bg_shot(hwnd, f"lan_park_{tag}_{i:02d}_retry{attempt}_{key}.png")
            d = sig_delta(base, frame_sig(p))
            if d >= BIG_CHANGE:
                return f"landed late ({p.name}, {d:.1%} changed)"
            print(f"    '{key}' swallowed (screen unchanged) — resend {attempt}/{KEY_TRIES}")
        ag.send_key(hwnd, key, background=True)
        t0 = time.time()
        while time.time() - t0 < CHANGE_TIMEOUT:
            time.sleep(0.7)
            hwnd = ag.find_game_window(pid) or hwnd
            p = ag.bg_shot(hwnd, f"lan_park_{tag}_{i:02d}_after_{key}.png")
            d = sig_delta(base, frame_sig(p))
            if d >= BIG_CHANGE:
                time.sleep(delay)
                return f"changed {d:.1%} in {time.time() - t0:.1f}s ({p.name})"
    raise SystemExit(
        f"{tag} key {i} '{key}': never changed the screen after {KEY_TRIES} tries — "
        f"see lan_park_{tag}_{i:02d}_after_{key}.png")


def walk(pid: int, keys: list[str], tag: str, delay: float) -> None:
    """Post the key flow to the pid's window, screenshotting before each key and at the end."""
    hwnd = wait_window(pid)
    hwnd = wait_menu(pid, hwnd, tag)
    for i, k in enumerate(keys):
        hwnd = ag.find_game_window(pid) or hwnd
        before = ag.bg_shot(hwnd, f"lan_park_{tag}_{i:02d}_before_{k}.png")
        note = press(hwnd, pid, k, before, tag, i, delay)
        print(f"  {tag} key {i} '{k}': {note} (state shot {before.name})")
    hwnd = ag.find_game_window(pid) or hwnd
    p = ag.bg_shot(hwnd, f"lan_park_{tag}_final.png")
    print(f"  {tag} parked: verify {p}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delay", type=float, default=2.5,
                    help="seconds to wait after each posted key (SC:R textures load late)")
    ap.add_argument("--host-keys", default=HOST_KEYS,
                    help=f"override the host key flow (default: {HOST_KEYS})")
    ap.add_argument("--guest-keys", default=GUEST_KEYS,
                    help=f"override the guest key flow (default: {GUEST_KEYS})")
    args = ap.parse_args()

    running = ag.find_process("StarCraft")
    if running:
        raise SystemExit(f"StarCraft.exe already running (pid {running}) — close it first; "
                         f"lan_park.py starts both clients itself")

    print("[1/5] launching the host client and injecting the Gary bridge (launch.py)")
    host_pid, exe = launch_gary()
    print(f"      host pid={host_pid}  exe={exe}")

    print(f"[2/5] host menus ({args.host_keys}) -> Create Game dialog, NOT creating")
    walk(host_pid, [k.strip().upper() for k in args.host_keys.split(",") if k.strip()],
         "host", args.delay)

    print(f"[3/5] closing the host's single-instance handle (close_mutex.py)")
    close_single_instance(host_pid)

    print("[4/5] launching your client (plain, no bridge)")
    guest = subprocess.Popen([str(exe), "-launch"], cwd=str(exe.parent))
    print(f"      guest pid={guest.pid}")

    print(f"[5/5] guest menus ({args.guest_keys}) -> LAN games list")
    walk(guest.pid, [k.strip().upper() for k in args.guest_keys.split(",") if k.strip()],
         "guest", args.delay)

    print(f"""
DONE — both clients parked (evidence in {ag.OUT_DIR}\\lan_park_*.png):
  host  pid {host_pid}  (Gary's client, bridge injected): Create Game dialog open
  guest pid {guest.pid}  (yours): LAN games list

Finish by hand:
  1. host window:  pick map + game type in the dialog, Ok to create the game
  2. guest window: the game shows up in the list — join it, pick your race
  3. host lobby:   set Gary's row race to Terran (the bot is Terran-only)
  4. host starts the game; once the match runs:  python -m gary.play --live""")


if __name__ == "__main__":
    main()