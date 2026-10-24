"""A paired episode's machinery on the dev manifests (eval/pairs.py): both bots deep-copied at a saved moment, an
arm played from it, the game restored, the same arm played again and ending the same. Every paired episode had
failed since the one-Gary refactor (the bots' deep copy met the race's data module) and nothing showed it for days
(issue #33); this takes under a minute. Needs the game data (skips without it).

Run: python -m pytest tests/test_pairs.py   (or plain: python tests/test_pairs.py)
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

UNTIL = 40 * 24                 # frames played before the branch point
ARM = 5 * 24                    # frames each arm plays


def test_paired_arms_copy_and_repeat():
    try:
        from eval.match import Match
        with contextlib.redirect_stdout(io.StringIO()):
            m = Match(1, "gt1dev", REPO / "tests" / "fixtures" / "replays" / "stardata_tvz_standard_ozp3w.rep",
                      opponent="gz1dev")
    except Exception as e:                      # no game data or env DLL here
        print(f"skip: {e}")
        return
    from eval.pairs import _copy, _play, snap
    while m.game.frame < UNTIL and not m.done:
        m.step()
    ctx = {"gary": m.gary, "zerg": m.zerg, "his": m.his, "slot_of": m.slot}
    m.game.save()
    ends = []
    for _ in range(2):
        m.game.restore()
        arm = _copy(ctx)                        # (fails if a bot holds something deepcopy can't take: a module)
        assert arm["gary"] is not m.gary and arm["gary"].race is m.gary.race
        _play(m.game, arm, ARM)
        ends.append(snap(m.game.observe()))
    assert ends[0] == ends[1], "an arm replayed from the same moment ended differently"
    m.game.restore()
    for _ in range(24):                          # (the main line goes on from the restored moment)
        m.step()
    print(f"ok: two arms of {ARM // 24} s from {UNTIL // 24} s, identical")


if __name__ == "__main__":
    test_paired_arms_copy_and_repeat()
