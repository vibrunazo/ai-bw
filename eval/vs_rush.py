"""Gary versions against the scripted Zerg rush (gary/bots/zerg_rush.py), over several seeds.

One game decides little (a different start position or one early decision changes the whole
game), so changes to Gary are judged here: each version plays the same seeds, in parallel, and
the table shows wins, losses and games still running at the time limit.

    python -m eval.vs_rush --versions v02 v03 --seeds 1 2 3 4 --attack-after 420
    python -m eval.vs_rush --versions v03 --seeds 1 2 3 4 5 6 --attack-after 0 --minutes 10
    python -m eval.vs_rush --versions v018 --opponent replayed --seeds 1 2 3 4 --attack-after 0

--opponent replayed is a test fixture (gary/bots/zerg_replayed.py): each seed's Zerg army appears
when a pro Zerg's did, in a real TvZ game.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import statistics
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from eval import priority  # noqa: E402
from eval.economy import AT, KEYS, line, pro_reference  # noqa: E402
DEFAULT_MAP = REPO_ROOT / "tests" / "fixtures" / "replays" / "stardata_tvz_standard_ozp3w.rep"


def play(version: str, seed: int, args: argparse.Namespace, out: Path) -> dict:
    rep = out / f"{version}_seed{seed}.rep"
    cmd = [sys.executable, "-m", "gary.bots.zerg_rush", "--vs", version, "--map", str(args.map),
           "--minutes", str(args.minutes), "--seed", str(seed), "--attack-after", str(args.attack_after),
           "--save", str(rep), "--opponent", args.opponent] + (["--style", str(args.style)] if args.style is not None else [])
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log = proc.stdout + proc.stderr
    (out / f"{version}_seed{seed}.log").write_text(log, encoding="utf-8")
    result = "error"
    for line in reversed(log.splitlines()):
        if line.startswith(("Gary won", "Gary lost", "time")):
            result = line.split(";")[0].replace("Gary ", "")
            break
    states = re.findall(r"^\s*(\d+:\d\d)\s+buildings T (\d+) Z (\d+)\s+army T (\d+) Z (\d+)", log, re.M)
    last = states[-1] if states else ("?", "?", "?", "?", "?")
    economy = {}
    for t, scvs, ccs, army, built in re.findall(r"^\s*(\d+):00\s.*economy SCVs (\d+) CCs (\d+) army supply ([\d.]+) built ([\d.]+)",
                                                log, re.M):
        if int(t) * 60 in AT:
            economy[int(t) * 60] = {"scvs": int(scvs), "ccs": int(ccs), "army": float(army), "built": float(built)}
    return {"version": version, "seed": seed, "result": result, "at": last[0],
            "buildings": f"{last[1]}-{last[2]}", "army": f"{last[3]}-{last[4]}", "economy": economy}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--versions", nargs="+", default=["v02", "v03"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--attack-after", type=float, default=420, help="the rush attacks no earlier (seconds)")
    ap.add_argument("--minutes", type=float, default=14)
    ap.add_argument("--map", default=str(DEFAULT_MAP))
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--style", type=int, help="steer Gary's macro toward this build style")
    ap.add_argument("--out", help="folder for replays and logs (default: a temporary folder)")
    ap.add_argument("--opponent", default="dummy", choices=["dummy", "v1", "v1a", "v1d", "v2", "replayed"], help="the Zerg (gary/bots/): replayed spawns a pro's army on time")
    priority.add_argument(ap)
    args = ap.parse_args()
    priority.apply(args.priority)
    out = Path(args.out or tempfile.mkdtemp(prefix="gary_eval_"))
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(v, s) for v in args.versions for s in args.seeds]
    with ThreadPoolExecutor(args.parallel) as pool:
        rows = list(pool.map(lambda j: play(j[0], j[1], args, out), jobs))
    print(f"rush attacks after {args.attack_after:.0f} s, {args.minutes:g} min games, style {args.style}; "
          f"replays and logs in {out}")
    print(f"{'version':8} {'seed':>4}  {'result':6} {'at':>6}  buildings T-Z  army T-Z")
    for r in rows:
        print(f"{r['version']:8} {r['seed']:>4}  {r['result']:6} {r['at']:>6}  {r['buildings']:>13}  {r['army']:>8}")
    print("economy (median of the games Gary is still in at that time):")
    try:
        print("  " + line("pros", pro_reference()))
    except Exception as e:                       # (no replay data here)
        print(f"  pros: not available ({e})")
    for v in args.versions:
        mine = [r for r in rows if r["version"] == v]
        at = {t: {k: statistics.median(r["economy"][t][k] for r in mine if t in r["economy"]) for k in KEYS}
              for t in AT if any(t in r["economy"] for r in mine)}
        print("  " + line(v, at))
    for v in args.versions:
        mine = [r for r in rows if r["version"] == v]
        print(f"{v}: won {sum(r['result'] == 'won' for r in mine)}, lost {sum(r['result'] == 'lost' for r in mine)}, "
              f"undecided {sum(r['result'] == 'time' for r in mine)} of {len(mine)}")


if __name__ == "__main__":
    main()
