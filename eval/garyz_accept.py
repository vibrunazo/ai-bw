"""Gary-Z's acceptance as an opponent (docs/SELF_PLAY.md): not a bot that runs, an opponent.

Against a Terran Gary on the three pool maps (Python, Fighting Spirit, Heartbreak Ridge), seeds 101+:

  decided    the share of games won or lost by the time limit (Zerg v3's pool decides about a third:
             an opponent self-play learns from must end games)
  plans      the plan each seed rolled, and its key timings (the first Spire, Lurker Aspect, the
             third Hatchery) next to the pros' medians for that plan (taxonomy plans' cards)
  repeat     the same seed played twice gives the same game, with two learned players in it
             (branching and paired tests depend on it)

    python -m eval.garyz_accept --seeds 16 --minutes 20          # 48 games, the full test
    python -m eval.garyz_accept --seeds 2 --minutes 12           # a trial
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAPS = {m: REPO / "tests" / "fixtures" / "replays" / f"stardata_tvz_{m}.rep"
        for m in ("python_gg13676", "standard_ozp3w", "heartbreak_9rg84")}
KEY = {"spire": 141, "lurker_aspect": None, "hatch3": 131}


def run_map(name: str, path: Path, versions: str, seeds: list[int], minutes: float, out: Path, parallel: int) -> None:
    subprocess.run([sys.executable, "-m", "eval.vs_rush", "--versions", versions, "--opponent", "garyz",
                    "--seeds", *map(str, seeds), "--minutes", str(minutes), "--map", str(path),
                    "--parallel", str(parallel), "--out", str(out / name)], cwd=REPO, check=False,
                   stdout=subprocess.DEVNULL)


def result_of(log: str) -> str:
    for line in reversed(log.splitlines()):
        if line.startswith(("Gary won", "Gary lost", "time")):
            return line.split(";")[0].replace("Gary ", "")
    return "error"


def timings(log: str) -> dict[str, str]:
    """Gary-Z's first Spire and Lurker Aspect decisions and its third Hatchery (from its log)."""
    out = {}
    m = re.search(r"^\s*(\d+:\d\d) Zerg builds Spire", log, re.M)
    out["spire"] = m.group(1) if m else "-"
    m = re.search(r"^\s*(\d+:\d\d) Zerg next: Lurker Aspect", log, re.M)
    out["lurker"] = m.group(1) if m else "-"
    halls = re.findall(r"^\s*(\d+:\d\d) Zerg builds (?:expand|Hatchery)", log, re.M)
    out["hatch3"] = halls[1] if len(halls) > 1 else "-"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vs", default="v027", help="the Terran Gary")
    ap.add_argument("--seeds", type=int, default=16, help="seeds per map, from 101")
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--out", default="runs/garyz_accept")
    ap.add_argument("--no-repeat", action="store_true", help="skip the same-seed-twice check")
    args = ap.parse_args()
    out = Path(args.out)
    seeds = list(range(101, 101 + args.seeds))
    for name, path in MAPS.items():
        run_map(name, path, args.vs, seeds, args.minutes, out, args.parallel)
    results, plans = Counter(), Counter()
    print(f"{'map':16s} {'seed':>4s}  {'result':8s} {'plan':45s} {'spire':>6s} {'lurker':>6s} {'hatch3':>6s}")
    for name in MAPS:
        for s in seeds:
            f = out / name / f"{args.vs}_seed{s}.log"
            if not f.exists():
                continue
            log = f.read_text(encoding="utf-8", errors="replace")
            r = result_of(log)
            plan = (re.search(r"Gary-Z v0: plan (.+?) \(", log) or [None, "?"])[1]
            t = timings(log)
            results[r] += 1
            plans[plan] += 1
            print(f"{name:16s} {s:>4d}  {r:8s} {plan:45s} {t['spire']:>6s} {t['lurker']:>6s} {t['hatch3']:>6s}")
    n = sum(results.values())
    decided = results["won"] + results["lost"]
    print(f"\n{n} games: Gary-T won {results['won']}, lost {results['lost']}, undecided {results['time']}, "
          f"errors {results['error']}; decided {decided / max(1, n):.0%} (Zerg v3's pool: about a third)")
    print("plans rolled:", dict(plans.most_common()))
    if args.no_repeat:
        return
    # the same seed twice: identical logs (but for the save paths)
    logs = []
    for k in range(2):
        rep = out / f"repeat_{k}.rep"
        p = subprocess.run([sys.executable, "-m", "gary.bots.zerg_rush", "--vs", args.vs, "--opponent", "garyz",
                            "--map", str(MAPS["standard_ozp3w"]), "--minutes", "8", "--seed", "101", "--save", str(rep)],
                           cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace")
        logs.append([l for l in p.stdout.splitlines() if "saved" not in l])
    same = logs[0] == logs[1]
    print(f"repeat: the same seed twice (8 minutes) gives {'the same game' if same else 'DIFFERENT games'}"
          + ("" if same else f" (first difference at line {next(i for i, (a, b) in enumerate(zip(*logs)) if a != b)})"))


if __name__ == "__main__":
    main()
