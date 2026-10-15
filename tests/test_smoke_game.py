"""One game of the default Gary against the rush dummy (held to 7:00) to 6:00: the basics of its
macro still work. A broken chore can ruin every game quietly (a guard in the gas chore once sent no
SCV to gas: Stim Packs never came, and only a 192-game benchmark showed it); this takes a minute.
Needs the game data (skips without it).

Run: python -m pytest tests/test_smoke_game.py   (or plain: python tests/test_smoke_game.py)
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

UNTIL = 6 * 60 * 24
FAILURES = ("didn't start", "order was lost", "couldn't build", "gives up")


def test_default_gary_macro_basics():
    try:
        from eval.walls_bench import Match
        with contextlib.redirect_stdout(io.StringIO()):
            m = Match(1, 420, "v023")
    except Exception as e:                      # no game data or env DLL here
        print(f"skip: {e}")
        return
    from gary import terran as T
    said = []
    m.gary.say = said.append
    gas_seen = 0
    while m.game.frame < UNTIL and not m.done:
        m.step()
        if m.game.frame % 24 < 2:
            o = m.game.observe()
            gas_seen = max(gas_seen, sum(1 for u in o["units"] if u["owner"] == m.slot["T"] and u["type"] == T.SCV
                                         and u["order"] in T.GAS_ORDERS))
    o = m.game.observe()
    mine = [u for u in o["units"] if u["owner"] == m.slot["T"]]
    count = lambda t: sum(1 for u in mine if u["type"] == t)
    failures = [s for s in said if any(f in s for f in FAILURES)]
    m.game.close()
    assert count(T.SCV) >= 20, f"{count(T.SCV)} SCVs at 6:00"
    assert count(T.RAX) >= 1 and count(T.REFINERY) >= 1, "no Barracks or no Refinery by 6:00"
    assert gas_seen >= 2, f"at most {gas_seen} SCVs on gas"
    assert len(failures) <= 6, f"{len(failures)} failed builds or goals: {failures[:6]}"


if __name__ == "__main__":
    test_default_gary_macro_basics()
    print("ok")
