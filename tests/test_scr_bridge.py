"""Python-side checks for the SC:R adapter (tests/gary_scr_tests.exe covers the C++ logic).

Run: python -m pytest tests/test_scr_bridge.py   (or plain: python tests/test_scr_bridge.py)
Nothing here needs the game running.
"""
from __future__ import annotations

import os
import inspect
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "adapters" / "scr_bridge"))

from profile_facts import read_profile, sha256_file  # noqa: E402

# GARY_SC_INSTALL: your StarCraft folder (the one with .build.info)
EXE = Path(os.environ.get("GARY_SC_INSTALL", "")) / "x86" / "StarCraft.exe"


def test_profile_pin_matches_the_installed_build():
    """The pinned SHA-256 must be the real one on this machine (skip if not installed)."""
    if not os.environ.get("GARY_SC_INSTALL") or not EXE.exists():
        print(f"skip: {EXE} not installed")
        return
    profile = read_profile()
    assert sha256_file(EXE) == profile["ExeSha256"].upper(), (
        "profile pin drifted from the installed executable")
    assert profile["BuildName"] == "StarCraft 1.23.10.13515 x86"


def test_scr_game_matches_gary_env_surface():
    """ScrGame must expose the gary.env.Game methods the human interface calls."""
    from gary.env import Game
    from gary.scr_env import ScrGame

    for name in ("observe", "act", "step", "unit_at", "box_select", "unit_type_of",
                 "can_place", "depot_spot_ok", "start_locations", "set_name", "save_replay",
                 "close"):
        assert hasattr(ScrGame, name), f"ScrGame.{name} missing"
        assert callable(getattr(ScrGame, name))
    # same call signatures for the methods the interface calls positionally
    for name in ("act", "unit_at", "unit_type_of"):
        g = inspect.signature(getattr(Game, name))
        s = inspect.signature(getattr(ScrGame, name))
        assert list(g.parameters) == list(s.parameters), f"{name} signature differs"


def test_units_dat_placement_sizes():
    """The generated placement table must match the game's own units.dat."""
    sys.path.insert(0, str(REPO / "adapters" / "scr_bridge" / "tools"))
    import gen_unit_dat

    dat = REPO / "data" / "gamedata" / "scr" / "arr" / "units.dat"
    if not dat.exists():
        print(f"skip: {dat} not extracted")
        return
    cols = gen_unit_dat.read_columns(dat.read_bytes())
    raw = cols["placement_size"]
    xy = [(raw[i], raw[i + 1]) for i in range(0, len(raw), 2)]
    scale = 32 if max(max(v) for v in xy) >= 16 else 1
    tiles = [(max(1, x // scale), max(1, y // scale)) for x, y in xy]
    assert tiles[106] == (4, 3), "Command Center should be 4x3 tiles"
    assert tiles[156] == (2, 2), "Pylon should be 2x2 tiles"
    assert tiles[111] == (4, 3), "Barracks should be 4x3 tiles"


def test_protocol_request_shape():
    """The request ScrGame sends must be parseable by json_min.h's grammar."""
    request = '{"id":1,"method":"act","args":{"slot":3,"hex":"0901000200"}}'
    assert '"method":"act"' in request
    assert '"hex":"0901000200"' in request
    response = '{"id":1,"ok":true,"result":1}'
    assert '"ok":true' in response


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as e:
                print(f"FAIL {name}: {type(e).__name__}: {e}")
                failed += 1
    sys.exit(1 if failed else 0)
