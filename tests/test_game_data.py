"""Gary's Terran and Zerg tables (gary/terran.py, gary/zerg.py) against the game's own data: costs,
footprints, supply.

A wrong cost is silent: the game refuses a build it can't pay for without a word, and Gary waits
and orders it again (the Refinery said 75 minerals, the game's is 100: in about 40% of games gas
came a minute late). Needs the game data (skips without it).

Run: python -m pytest tests/test_game_data.py   (or plain: python tests/test_game_data.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from gary import terran as T  # noqa: E402
from gary import zerg as Z  # noqa: E402

MAP = REPO / "tests" / "fixtures" / "replays" / "stardata_tvz_standard_ozp3w.rep"


def _game():
    try:
        from gary.env import Game
        return Game.new(MAP, ["T", "Z"], ["a", "b"], seed=1)
    except Exception as e:                      # no game data or env DLL here
        print(f"skip: {e}")
        return None


def test_tables_match_the_game():
    game = _game()
    if game is None:
        return
    with game:
        wrong = []
        for R in (T, Z):
            pairs = getattr(R, "PAIRS", set())          # (Zerg: one order makes two, the table has the pair's)
            for ut, cost in R.COST.items():
                info = game.unit_type_info(ut)
                if info and (info["minerals"], info["gas"]) != tuple(cost):
                    wrong.append(f"cost {R.NAMES.get(ut, ut)}: {cost}, game {(info['minerals'], info['gas'])}")
            for ut, (w, h) in R.SIZE.items():
                info = game.unit_type_info(ut)
                if info and (info["w"] // 32, info["h"] // 32) != (w, h):
                    wrong.append(f"size {R.NAMES.get(ut, ut)}: {(w, h)}, game {(info['w'] // 32, info['h'] // 32)}")
            for ut, supply in R.SUPPLY.items():
                info = game.unit_type_info(ut)
                want = supply * 2 // (2 if ut in pairs else 1)
                if info and info["supply_x2"] != want:
                    wrong.append(f"supply {R.NAMES.get(ut, ut)}: {supply}, game {info['supply_x2'] / 2}")
    assert not wrong, "; ".join(wrong)


if __name__ == "__main__":
    test_tables_match_the_game()
    print("ok")
