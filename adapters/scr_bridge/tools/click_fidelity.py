#!/usr/bin/env python3
"""How faithful the bridge's live click model is: a live game's replay re-simulated in OpenBW, and every
click Gary made there asked again of gary_env (the training game's click model).

Gary's POV log records each click (camera + cursor = the map point) and what the bridge answered (the
unit it selected, or nothing). The replay records the commands the game received; every selection Gary
sent is in it as a Select with the same units, which ties the POV log's frames to the game's. At each
click's frame (less the command latency) the re-simulated game is asked what unit_at gives there.

    python adapters/scr_bridge/tools/click_fidelity.py runs/live/v025_live1.pov.jsonl runs/live/v025_live1.rep
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from bisect import bisect_left
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from gary.env import Game  # noqa: E402

LATENCY = 2                      # frames from a click to its command in the replay (the bridge's latency_frames)


def replay_selects(rep: Path) -> list[tuple[int, tuple]]:
    out = subprocess.run([str(REPO / "tools" / "bin" / "screp.exe"), "-cmds", str(rep)], capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stdout
    d = json.loads(out)
    return [(c["Frame"], tuple(c.get("UnitTags") or [])) for c in d["Commands"]["Cmds"]
            if c["Type"]["Name"] == "Select"], d


def main() -> None:
    pov_path, rep = Path(sys.argv[1]), Path(sys.argv[2])
    rows = [json.loads(l) for l in pov_path.read_text(encoding="utf-8").splitlines()][1:]
    selects, d = replay_selects(rep)
    # tie POV frames to game frames: each POV row that sent a selection, in order, to the replay's Select
    # with the same units (a few may be missing: the game refused them, or they weren't sent)
    anchors, j = [], 0
    for r in rows:
        if r.get("sel"):
            tags = tuple(r["sel"])
            k = next((k for k in range(j, min(j + 8, len(selects))) if selects[k][1] == tags), None)
            if k is not None:
                anchors.append((r["frame"], selects[k][0]))
                j = k + 1
    print(f"{len(anchors)} of {sum(1 for r in rows if r.get('sel'))} selections found in the replay (frame anchors)")
    pf = [a for a, _ in anchors]

    def game_frame(pov_frame: int) -> int:
        i = min(max(1, bisect_left(pf, pov_frame)), len(anchors) - 1)
        (p0, g0), (p1, g1) = anchors[i - 1], anchors[i]
        return round(g0 + (pov_frame - p0) * (g1 - g0) / max(1, p1 - p0))

    clicks = []
    for r in rows:
        act = r.get("act") or ""
        if r.get("click") == "left" and act.startswith("click:"):
            x, y = r["camera"][0] + r["cursor"][0], r["camera"][1] + r["cursor"][1]
            live = r["sel"][0] if r.get("sel") else 0
            clicks.append((game_frame(r["frame"]) - LATENCY, x, y, live, act))
    clicks.sort()
    g = Game.scenario(rep)
    obs = g.observe()
    slot = next(u["owner"] for u in obs["units"] if u["type"] == 106)
    agree, kinds, examples = 0, Counter(), []
    names = {}
    for f, x, y, live, act in clicks:
        if f > g.frame:
            g.step(f - g.frame)
        want = g.unit_at(slot, x, y)
        units = {u["tag"]: u for u in g.observe()["units"]}
        name = lambda t: "nothing" if not t else {7: "SCV", 176: "Mineral Field", 177: "Mineral Field", 178: "Mineral Field",
                                                  0: "Marine", 125: "Bunker", 124: "Missile Turret", 109: "Supply Depot",
                                                  106: "Command Center", 111: "Barracks"}.get(units.get(t, {}).get("type"), f"type {units.get(t, {}).get('type')}")
        if want == live:
            agree += 1
        else:
            kinds[(name(want), name(live))] += 1
            if len(examples) < 8:
                u = units.get(want) or units.get(live) or {}
                examples.append((f, (x, y), f"openbw {name(want)}", f"bridge {name(live)}",
                                 f"unit at ({u.get('x')}, {u.get('y')}), {math.dist((x, y), (u.get('x', x), u.get('y', y))):.0f} px away"))
    print(f"{len(clicks)} clicks: the bridge agreed with OpenBW on {agree} ({agree / max(1, len(clicks)):.0%})")
    print("disagreements (OpenBW's answer, the bridge's):")
    for k, v in kinds.most_common(12):
        print(f"  OpenBW {k[0]:16s} bridge {k[1]:16s} {v}")
    for e in examples:
        print("  ", e)


if __name__ == "__main__":
    main()
