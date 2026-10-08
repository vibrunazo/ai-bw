"""Start StarCraft: Remastered (x86) with the Gary bridge injected, then talk to it.

    python adapters/scr_bridge/launch.py --smoke          # launch, inject, ping/status/observe
    python adapters/scr_bridge/launch.py                  # launch and leave it running
    python adapters/scr_bridge/launch.py --attach 1234    # inject into a running client

The launcher refuses to start a build whose SHA-256 does not match the pinned profile (pass
--allow-unknown to override for experiments). Custom games and offline only — never the
Battle.net ladder (CONTRIBUTING.md ground rules).
"""
from __future__ import annotations

import os
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from profile_facts import read_profile, sha256_file  # noqa: E402

KNOWN_EXES = [                                   # (GARY_SC_INSTALL: your StarCraft folder)
    *([str(Path(os.environ["GARY_SC_INSTALL"]) / "x86" / "StarCraft.exe")] if os.environ.get("GARY_SC_INSTALL") else []),
    r"C:\Program Files (x86)\StarCraft\x86\StarCraft.exe",
    r"C:\StarCraft\x86\StarCraft.exe",
]

PROBE_CHECKLIST = """\
Live probe checklist (start a custom melee game vs one computer, then run this with --smoke):
  1. status: hash_verified=true, frame_watcher=true, in_game=true, latency_frames present
  2. observe: exactly 2 players in a 1v1; names/races/minerals sane
  3. observe: 4 workers + 1 main building per player; marine hp=40 (probe_unit on one)
  4. probe_unit on a mineral field: raw_resources equals observe's resources and is a real
     amount (1500/2500 fresh, lower once mined) — an all-zero read means the offset is wrong
  5. probe_unit on a protoss unit: raw_shields>>8 equals its shield maximum (probe 20,
     zealot 60); tools/probe_live.py's shields auto-hunt pins the offset as kUnitShields
  6. probe_unit on a training building: raw_queue[0] is its unit type (<228)
  7. act() a select command for the local player; the client's saved replay must contain 0x63
     for that player (check with tools/bin/screp.exe) — proves the command path end to end
  8. frame counter advances while playing (status frames grows)
"""


def find_exe(explicit: str | None) -> Path:
    candidates = [explicit] if explicit else KNOWN_EXES
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    raise SystemExit(
        "StarCraft x86 client not found; pass --exe <path>\\StarCraft.exe (the x86 one)")



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", help="path to the x86 StarCraft.exe")
    ap.add_argument("--attach", type=int, metavar="PID",
                    help="inject into a running client instead of launching")
    ap.add_argument("--pipe", default=r"\\.\pipe\gary_scr", help="named pipe to serve")
    ap.add_argument("--allow-unknown", action="store_true",
                    help="skip the SHA-256 pin check (experiments only)")
    ap.add_argument("--smoke", action="store_true",
                    help="connect, ping/status/observe, print, and exit")
    ap.add_argument("--probe", type=int, metavar="TAG", help="also probe_unit(TAG)")
    ap.add_argument("--kill", action="store_true", help="kill the launched game after --smoke")
    ap.add_argument("--wait", type=float, default=8.0,
                    help="seconds to wait for the client to load before connecting")
    args = ap.parse_args()

    profile = read_profile()
    dll = HERE / "build" / "gary_scr.dll"
    injector = HERE / "build" / "scr_inject.exe"
    for f in (dll, injector):
        if not f.exists():
            raise SystemExit(f"{f} not built: run adapters\\scr_bridge\\build.bat")

    if args.attach:
        pid = args.attach
        out = subprocess.run([str(injector), "attach", str(pid), str(dll)],
                             capture_output=True, text=True)
        if out.returncode != 0:
            raise SystemExit(f"scr_inject attach failed: {out.stdout}{out.stderr}")
        print(f"injected {dll.name} into pid {pid}")
    else:
        exe = find_exe(args.exe)
        exe_hash = sha256_file(exe)
        pin = profile["ExeSha256"]
        if exe_hash != pin and not args.allow_unknown:
            raise SystemExit(
                f"{exe} has SHA-256 {exe_hash}\n"
                f"but the profile pins {pin} ({profile.get('BuildName', '?')}).\n"
                "A game patch would make every address wrong: refusing. "
                "(--allow-unknown to override; then re-verify with tools/verify_profile.py)")
        write_config(dll.parent, args.pipe, pin if not args.allow_unknown else exe_hash)
        print(f"launching {exe} -launch")
        # SC:R exits immediately unless started with -launch (the standalone-start flag the
        # reference launchers use); the game window may open on the desktop.
        out = subprocess.run([str(injector), str(exe), str(dll), str(exe.parent), "-launch"],
                             capture_output=True, text=True)
        if out.returncode != 0:
            raise SystemExit(f"scr_inject failed: {out.stdout}{out.stderr}")
        pid = int(out.stdout.split()[1])
        print(f"injected {dll.name} into pid {pid}")

    if not args.smoke:
        print(f"client running as pid {pid}; attach a bot with gary.scr_env.ScrGame.connect()")
        return

    time.sleep(args.wait)
    sys.path.insert(0, str(REPO))
    from gary.env import GameError
    from gary.scr_env import ScrGame

    try:
        game = ScrGame.connect(args.pipe)
    except GameError as e:
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
        raise SystemExit(f"{e}\nCheck build\\logs\\scr_bridge.log next to the DLL.")
    try:
        print("ping:", game.ping())
        print("status:", json.dumps(game.status(), indent=2))
        try:
            obs = game.observe()
            print(f"observe: frame={obs['frame']} map={obs['map']} "
                  f"players={len(obs['players'])} units={len(obs['units'])}")
        except GameError as e:
            print("observe:", f"(not readable yet: {e})")
        if args.probe is not None:
            print("probe_unit:", json.dumps(game.probe_unit(args.probe), indent=2))
    except Exception:
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
        raise
    finally:
        game.close()
    print()
    print(PROBE_CHECKLIST)
    if args.kill:
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
        print(f"killed pid {pid}")


def write_config(dll_dir: Path, pipe: str, expect_sha256: str) -> Path:
    config = {"pipe": pipe, "log_dir": "logs", "expect_sha256": expect_sha256}
    path = dll_dir / "gary_scr.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path



if __name__ == "__main__":
    main()
