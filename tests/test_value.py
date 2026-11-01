"""The critic reads the current Gary (issue #46): the value network's input width is the width of its view of a game in
play (gary/policy/value.py ValueView) and of the replays' rows it learns from (train/value.py view_rows). The macro
features grew from 217 to 421 and the critic, trained on the old ones, failed on every read for days with nothing to
show it. A feature change now fails here until the critic is trained again (python -m train.value). Every matchup and race
with a critic (runs/value/<matchup>_<race>_*) is read by its dev build. Needs the game data and the trained models
(skips without them).

Run: python -m pytest tests/test_value.py   (or plain: python tests/test_value.py)
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

UNTIL = 60 * 24                 # frames played before the critic reads the game


DEV = {"T": "gt1dev", "Z": "gz1dev", "P": "gp1dev"}


def critics() -> list[tuple[str, str]]:
    """(matchup, race) of every critic trained."""
    return sorted({tuple(p.name.split("_")[:2]) for p in (REPO / "runs" / "value").glob("*_*_*") if (p / "model.pt").exists()})


def read_critic(matchup: str, race: str) -> None:
    try:
        from eval.match import Match
        from gary.agent.manifests import model_for
        from gary.policy.belief import BeliefModel
        from gary.policy.value import EXTRAS, ValueModel, ValueView
        from gary.races import enemy_of
        with contextlib.redirect_stdout(io.StringIO()):
            m = Match(1, DEV[race], REPO / "tests" / "fixtures" / "replays" / "stardata_tvz_standard_ozp3w.rep",
                      opponent=DEV[enemy_of(matchup, race).race])
        critic = ValueModel.load(model_for("value", matchup, race))
        view = ValueView(m.gary, BeliefModel.load(model_for("belief", matchup, race)))
    except Exception as e:                      # no game data, env DLL or trained models here
        print(f"skip {matchup} {race}: {e}")
        return
    assert m.gary.matchup == matchup, (m.gary.matchup, matchup)
    while m.game.frame < UNTIL and not m.done:
        with contextlib.redirect_stdout(io.StringIO()):
            m.step()
        if m.game.frame % 24 < 2:
            view.look(m.gary.hi.observe())
    obs = m.gary.hi.observe()
    view.look(obs)
    x = view.features(obs)
    width = m.gary.macro.tracker.spec.size + len(EXTRAS)
    assert x.shape == (1, width), (x.shape, width)
    assert critic.ckpt["n_in"] == width, (f"the critic ({critic.path}) reads {critic.ckpt['n_in']} features, Gary's "
                                          f"view has {width}: train it again (python -m train.value --matchup "
                                          f"{matchup} --race {race})")
    assert critic.ckpt.get("spec") == m.gary.macro.tracker.spec.to_dict(), "the critic was trained on another macro spec"
    p = critic.predict(x)
    print(f"ok {matchup} {race}: the critic reads {width} features; at {UNTIL // 24} s win logit {p['win'][0]:+.2f}, "
          f"net180 {p['net180'][0]:+.0f}")


def test_critic_reads_current_gary():
    for matchup, race in critics():
        read_critic(matchup, race)


if __name__ == "__main__":
    test_critic_reads_current_gary()
