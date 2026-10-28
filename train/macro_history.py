"""The pros' typical decision history: mean counts of each macro decision taken so far, per minute.

gary/policy/macro.py's MacroTracker can show the macro model the pros' typical past instead of the
player's own (Gary-Z's own history fed back on itself: PROGRESS.md, round g12). This writes what
it reads, from the macro dataset and the newest macro model's decision index:

  data/interim/macro/v1/history_mean_<matchup>_<race>.npy        (31, decisions): the race's mean, minutes 0-30
                                                          (..._<catalogue>.npy with --styles: a manifest's
                                                          macro history_styles reads them)
  data/interim/macro/v1/history_mean_<matchup>_<race>_styles.npy (style ids, 31, decisions): per build style id, at each
                                                          minute the style's own mean while MIN_STYLE of its
                                                          games are still going, the race's mean after

Re-run after the macro model is retrained (its decision index is the column order) or when games
are added. Takes the same --era/--source filter as the trainers.

Usage:
  python -m train.macro_history --race Z
  python -m train.macro_history --matchup TvZ --race T --era modern
  python -m train.macro_history --matchup PvZ --race Z --styles v2   # the styles from the catalogue the model learned
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from gary.agent.manifests import _newest  # noqa: E402
from gary.policy.macro import MacroModel  # noqa: E402
from train.macro import macro_dir, player_games  # noqa: E402
from train.select import GameFilter  # noqa: E402

MINUTES = 31
MIN_STYLE = 50


def counts_by_minute(path: str, player: int, spec) -> np.ndarray | None:
    """(MINUTES, decisions): this player's decisions taken before each minute mark."""
    d = np.load(path)
    c = d["cmds"][d["cmds"][:, 1] == player]
    if len(c) == 0:
        return None
    cls = np.array([spec.decision_index(a, u) for a, u in c[:, 2:4].tolist()], dtype=np.int64)
    out = np.zeros((MINUTES, len(spec.decisions)), np.float32)
    end = int(d["frames"][-1])
    for m in range(MINUTES):
        f = m * 24 * 60
        if f > end:
            out[m] = np.nan                 # the game was over: no observation of this minute
            continue
        k = np.searchsorted(c[:, 0], f, side="left")
        np.add.at(out[m], cls[:k], 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matchup", default="TvZ")
    ap.add_argument("--race", required=True, choices=["T", "Z", "P"])
    ap.add_argument("--styles", help="each player's style from this catalogue (gary/styles.py), as train/macro.py --styles")
    ap.add_argument("--model", help="the macro model whose decision index the tables follow (default: the race's shared)")
    GameFilter.add_args(ap)
    args = ap.parse_args()
    keep = GameFilter.from_args(args)
    import train.macro as tm
    tm.STYLE_VERSION = args.styles
    from gary.policy.macro import macro_model_path, shared_macro
    spec = MacroModel.load(str(args.model or macro_model_path(args.matchup, args.race))).spec   # (GARY_MACRO_SHARED: the shared model's)
    jobs = player_games(args.matchup, args.race, keep)
    print(f"{len(jobs)} player-games ({keep.describe()}); {len(spec.decisions)} decisions", flush=True)
    tables, styles = [], []
    for path, i, style, _ in jobs:
        t = counts_by_minute(path, i, spec)
        if t is not None:
            tables.append(t)
            styles.append(-1 if style is None else int(style))
    tables, styles = np.stack(tables), np.array(styles)
    mean = np.nanmean(tables, axis=0).astype(np.float32)
    out = macro_dir()
    tag = f"{args.matchup}_{args.race}" + ("_shared" if shared_macro() else "") + (f"_{args.styles}" if args.styles else "")
    np.save(out / f"history_mean_{tag}.npy", mean)
    print(f"wrote history_mean_{tag}.npy {mean.shape}; {mean[10].sum():.0f} decisions taken by 10:00 on average")
    if len(spec.styles):                    # (by style id, as gary/policy/macro.py pro_history reads it: a model of every
        per = np.repeat(mean[None], max(spec.styles) + 1, axis=0)       # matchup has ids 0-, 20- and 40-)
        for s in spec.styles:
            rows = tables[styles == s]
            alive = (~np.isnan(rows[:, :, 0])).sum(axis=0) if len(rows) else np.zeros(MINUTES, int)
            enough = alive >= MIN_STYLE                     # minutes this style still has MIN_STYLE games in
            if enough.any():
                per[s, enough] = np.nanmean(rows[:, enough], axis=0)
            print(f"  style {s}: {len(rows)} players, its own mean to minute {int(enough.sum()) - 1}, the race's after")
        np.save(out / f"history_mean_{tag}_styles.npy", per.astype(np.float32))
        print(f"wrote history_mean_{tag}_styles.npy {per.shape}")


if __name__ == "__main__":
    main()
