"""Why did Gary do that? Every decision of one game in a window, across layers, in time order, with who
had the hands: a query over the game's records (gary/decision_log.py), not a re-simulation.

    python -m eval.why runs/league/c6 --seed 3005 --minute 5            (minute 5:00-6:00)
    python -m eval.why runs/watch/gz3005.rep --minute 5 --to 5.5 --side Z
    python -m eval.why runs/watch/gz3005.records.jsonl --minute 5 --layer macro --orders all

The game: a .rep or .records.jsonl, or a folder searched (recursively, the newest first) for the seed's
records (<anything>seed<seed>.records.jsonl or <anything><seed>.records.jsonl).

Each line: the time, the side and who had its hands, then
  a decision   layer: choice (p) by who; what it saw; the top alternatives; what was set aside and why; then
  an order     order: what, its fate (sent, refused, retried, dropped, done) and why, given by whom
  a note       what a layer said in the game's log (--notes off to hide)
--orders: "fates" (the default: every order that wasn't simply sent, and the production orders sent),
"all" (every order, repeats folded), "none". Last, the window's counts per layer and per order fate.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PRODUCTION = ("train ", "morph ", "place ", "research ", "upgrade ")


def find(where: str, seed: int | None) -> Path:
    p = Path(where)
    if p.is_file():
        if p.name.endswith(".records.jsonl"):
            return p
        for base in (p.with_suffix(""), Path(str(p)[:-len(".pov.jsonl")]) if p.name.endswith(".pov.jsonl") else None):
            if base is not None and (r := base.with_name(base.name + ".records.jsonl")).exists():
                return r
        sys.exit(f"no records next to {p} (games played before the records existed have none)")
    if seed is None:
        sys.exit("a folder needs --seed")
    found = [r for r in p.rglob("*.records.jsonl")
             if r.name.endswith((f"seed{seed}.records.jsonl", f"_{seed}.records.jsonl", f"{seed}.records.jsonl"))]
    if not found:
        sys.exit(f"no records for seed {seed} under {p}")
    return max(found, key=lambda r: r.stat().st_mtime)


def clock(f: int) -> str:
    s = f / 24
    return f"{int(s // 60)}:{s % 60:04.1f}"


def pct(p) -> str:
    return f"{p:.0%}" if p is not None else ""


def line(r: dict) -> str:
    if r["kind"] == "decision":
        out = f"{r['layer']}: {r['choice'] if r['choice'] is not None else '(none)'}"
        if "p" in r:
            out += f" ({pct(r['p'])})"
        out += f" by {r['by']}"
        if r.get("then"):
            out += f" -> {r['then']}"
        if r.get("in"):
            out += "\n" + " " * 26 + "saw " + ", ".join(f"{k} {v}" for k, v in r["in"].items() if v is not None)
        if r.get("top"):
            out += "\n" + " " * 26 + "top " + ", ".join(f"{n} {pct(p)}" for n, p in r["top"])
        for n, why in r.get("over", []):
            out += "\n" + " " * 26 + f"set aside {n}: {why}"
        return out
    if r["kind"] == "order":
        out = f"order: {r['what']} {r['fate'].upper()}"
        if r.get("why"):
            out += f": {r['why']}"
        return out + f" (by {r['by']})"
    return f"note: {r['text']}"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("game", help="a .rep or .records.jsonl, or a folder (with --seed)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--minute", type=float, required=True, help="from this minute (5 or 5.5)")
    ap.add_argument("--to", type=float, help="to this minute (default: a minute on)")
    ap.add_argument("--side", choices=["T", "Z"])
    ap.add_argument("--layer", action="append", help="only these layers' decisions (macro, army, engage, roles, "
                                                     "posture, gas); repeatable")
    ap.add_argument("--orders", choices=["fates", "all", "none"], default="fates")
    ap.add_argument("--notes", choices=["on", "off"], default="on")
    args = ap.parse_args()
    path = find(args.game, args.seed)
    rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()[1:]]
    f0, f1 = int(args.minute * 1440), int((args.to if args.to is not None else args.minute + 1) * 1440)
    print(f"{path}: {clock(f0)}-{clock(f1)}")
    hands: dict[str, str] = {}
    counts, fates = Counter(), Counter()
    last, repeat = None, 0

    def flush():
        nonlocal last, repeat
        if last is not None:
            print(last + (f"  (x{repeat})" if repeat > 1 else ""))
        last, repeat = None, 0
    for r in rows:
        if r["f"] >= f1:
            break
        if r["kind"] == "hands":
            hands[r["side"]] = r["owner"]
            continue
        if r["f"] < f0 or (args.side and r["side"] != args.side):
            continue
        if r["kind"] == "decision":
            counts[f"{r['side']} {r['layer']}"] += 1
            if args.layer and r["layer"] not in args.layer:
                continue
        elif r["kind"] == "order":
            fates[f"{r['side']} {r['fate']}"] += 1
            if args.orders == "none" or args.layer:
                continue
            if args.orders == "fates" and r["fate"] == "sent" and not r["what"].startswith(PRODUCTION):
                continue
        elif r["kind"] == "note" and (args.notes == "off" or args.layer):
            continue
        head = f"{clock(r['f']):>7} {r['side']} [{hands.get(r['side'], '?')[:16]:<16}] "
        text = head + line(r)
        if r["kind"] == "order" and last is not None and last.split("] ", 1)[-1] == text.split("] ", 1)[-1]:
            repeat += 1                                    # (the same order again: folded)
            continue
        flush()
        last, repeat = text, 1
    flush()
    print("\ndecisions: " + ", ".join(f"{k} {n}" for k, n in sorted(counts.items())))
    print("orders: " + ", ".join(f"{k} {n}" for k, n in sorted(fates.items())))


if __name__ == "__main__":
    main()
