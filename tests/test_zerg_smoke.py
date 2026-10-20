"""The one Gary playing Zerg (manifest gz1) in a 4-minute ZvZ mirror on a ladder map (eval/match.py): its
opening's basics still work (drones made and mining, a Spawning Pool, an Extractor, overlords ahead of the
supply), side A's, from the replay re-simulated. Needs the game data and the ladder maps (skips without
them); about a minute.

Run: python -m pytest tests/test_zerg_smoke.py   (or plain: python tests/test_zerg_smoke.py)
"""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MAP = "KnockOut 1.4"
SEED = 5001
MINUTES = 4


def test_zerg_opening_basics():
    try:
        from eval.league import ladder_maps
        path = ladder_maps([MAP])[MAP]
    except Exception as e:                      # no game data or ladder maps here
        print(f"skip: {e}")
        return
    from eval.match import play
    from gary import zerg as Z
    from gary.env import Game
    with tempfile.TemporaryDirectory() as tmp:
        rep = str(Path(tmp) / "zvz.rep")
        with contextlib.redirect_stdout(io.StringIO()):
            play(str(path), MINUTES, SEED, rep, "gz1", "gz1")
        g = Game.scenario(rep)
        slot, pool, extractor, blocked, drones = None, False, False, 0, 0
        for sec in range(1, MINUTES * 60 + 1):
            while g.frame < sec * 24:
                g.step(24 - g.frame % 24 if g.frame % 24 else 24)
            o = g.observe()
            slot = slot if slot is not None else min(p["slot"] for p in o["players"])
            mine = [u for u in o["units"] if u["owner"] == slot]
            me = next(p for p in o["players"] if p["slot"] == slot)
            pool = pool or any(u["type"] == Z.POOL for u in mine)
            extractor = extractor or any(u["type"] == Z.EXTRACTOR for u in mine)
            blocked += me["supply_max"] < 200 and me["supply_used"] >= me["supply_max"]
            drones = sum(1 for u in mine if u["type"] == Z.DRONE)
    assert drones >= 9, f"{drones} drones at {MINUTES}:00"
    assert pool, f"no Spawning Pool by {MINUTES}:00"
    assert extractor, f"no Extractor by {MINUTES}:00"
    assert blocked <= 90, f"supply-blocked {blocked} s of {MINUTES} minutes"


if __name__ == "__main__":
    test_zerg_opening_basics()
    print("ok")
