"""Sum up a Gary trace (gary/agent/arbiter.py): who had Gary's hands, for how long, and who waited.

    python -m gary.agent.trace path/to/game.trace.jsonl [more traces...]

  hands     per requester (macro, army, scout, fight, rule:pull, executor:gas...): tasks, seconds
            holding the hands, share of all task time
  fights    tasks a fight turn interrupted, by kind
  waits     per layer: how often and how long it waited for the hands (macro with a decision
            taken, the fight with one going on), the longest, and what held the hands
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

FPS = 1000 / 42


def summarize(paths: list[Path]) -> dict:
    hands: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
    interrupted: Counter = Counter()
    waits: dict[str, list] = defaultdict(list)
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines()[1:]:
            e = json.loads(line)
            if e["kind"] == "task_end":
                h = hands[e["by"]]
                h[0] += 1
                h[1] += e["frames"] / FPS
                if e["interrupted"]:
                    interrupted[e["task"]] += 1
            elif e["kind"] == "wait":
                waits[e["layer"]].append(e)
    total = sum(h[1] for h in hands.values()) or 1
    return {
        "hands": {who: {"tasks": n, "seconds": round(s, 1), "share": round(s / total, 3)}
                  for who, (n, s) in sorted(hands.items(), key=lambda kv: -kv[1][1])},
        "interrupted": dict(interrupted.most_common()),
        "waits": {layer: {"count": len(ws), "seconds": round(sum(w["frames"] for w in ws) / FPS, 1),
                          "longest_s": round(max(w["frames"] for w in ws) / FPS, 1),
                          "hands_on": dict(Counter(t for w in ws for t in w["hands_on"]).most_common(5))}
                  for layer, ws in waits.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+", type=Path)
    args = ap.parse_args()
    s = summarize(args.traces)
    print(f"{len(args.traces)} trace(s)")
    print("hands (who asked: tasks, seconds, share of task time)")
    for who, h in s["hands"].items():
        print(f"  {who:18s} {h['tasks']:5d} {h['seconds']:8.1f} s  {h['share']:6.1%}")
    print("interrupted by a fight:", s["interrupted"] or "nothing")
    for layer, w in s["waits"].items():
        print(f"{layer} waited {w['count']} times, {w['seconds']} s in all (longest {w['longest_s']} s); "
              f"hands on: {w['hands_on']}")


if __name__ == "__main__":
    main()
