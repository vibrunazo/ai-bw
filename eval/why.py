"""Why did Gary do that? Every decision of one game in a window, across layers, in time order, with who
had the hands: a query over the game's records (gary/decision_log.py), not a re-simulation.

    python -m eval.why runs/league/c6 --seed 3005 --minute 5            (minute 5:00-6:00)
    python -m eval.why runs/watch/gz3005.rep --minute 5 --to 5.5 --side Z
    python -m eval.why runs/watch/gz3005.records.jsonl --minute 5 --layer macro --orders all
    python -m eval.why runs/league_loop/e1/r0 --seed 3005 --pairs        (the game's paired branches)
    python -m eval.why runs/league_loop/e1/r0 --seed 3005 --pair 2       (one pair's arms side by side)

The game: a .rep or .records.jsonl, or a folder searched (recursively, the newest first) for the seed's
records (<anything>seed<seed>.records.jsonl or <anything><seed>.records.jsonl).

Each line: the time, the side and who had its hands, then
  a decision   layer: choice (p) by who; what it saw; the top alternatives; for a learned layer, the input
               groups its odds rest on (each zeroed in turn: the odds without it); what was set aside and
               why; then
  an order     order: what, its fate (sent, refused, retried, dropped, waiting, done) and why, given by whom
  a note       what a layer said in the game's log (--notes off to hide)
  seen         enemy fighters coming into sight, new ones, or every 10 s while in sight (how many, where, by
               which base, closing on Gary's buildings or not); an enemy building first seen
  belief       once a minute: the enemy army the belief model expects against the most seen in 3 minutes
  death        one of Gary's fighters gone: where, alone or not, the enemy near it, who held it, the hands
               (gary/agent/sightings.py; --layer seen, belief or death shows those alone)
--orders: "fates" (the default: every order that wasn't simply sent, and the production orders sent),
"all" (every order, repeats folded), "none". Last, the window's counts per layer and per order fate.

Paired branches (eval/pairs.py, the RL's own episodes; <game>.pairs.records.jsonl): --pairs lists them
(when, the arms' models and outcomes, the first decision where the arms differ); --pair N shows both
arms' records interleaved (A0, A1), each decision with its row among the RL arrays ([rl #n]) and, in an
A/B pair, the other model's odds of the same choice.
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
    found = [r for r in p.rglob("*.records.jsonl") if not r.name.endswith(".pairs.records.jsonl")
             and r.name.endswith((f"seed{seed}.records.jsonl", f"_{seed}.records.jsonl", f"{seed}.records.jsonl"))]
    if not found:
        sys.exit(f"no records for seed {seed} under {p}")
    if len({r.parent for r in found}) > 1:              # (a round plays each seed on every map)
        sys.exit(f"seed {seed} is in several games; name one:\n  " + "\n  ".join(str(r) for r in sorted(found)))
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
            out += f" ({pct(r['p'])}" + (f", other model {pct(r['p_other'])}" if "p_other" in r else "") + ")"
        if r.get("rl") is not None:
            out += f" [rl #{r['rl']}]"
        out += f" by {r['by']}"
        if r.get("then"):
            out += f" -> {r['then']}"
        if r.get("in"):
            out += "\n" + " " * 26 + "saw " + ", ".join(f"{k} {v}" for k, v in r["in"].items() if v is not None)
        if r.get("top"):
            out += "\n" + " " * 26 + "top " + ", ".join(f"{n} {pct(p)}" for n, p in r["top"])
        if r.get("why"):
            out += "\n" + " " * 26 + "rests on " + ", ".join(f"{g} ({pct(r['p'])} -> {pct(q)} without)" for g, q in r["why"])
        for n, why in r.get("over", []):
            out += "\n" + " " * 26 + f"set aside {n}: {why}"
        return out
    if r["kind"] == "order":
        out = f"order: {r['what']} {r['fate'].upper()}"
        if r.get("why"):
            out += f": {r['why']}"
        return out + f" (by {r['by']})"
    if r["kind"] == "seen":
        where = f"at {r['at'][0]},{r['at'][1]} ({r['near']}" + (f", {r['dist']} tiles from a building)" if r.get("dist") is not None else ")")
        if r["what"] == "building":
            return f"seen: {r['type']}{'' if r.get('done', True) else ' (unfinished)'} {where}"
        out = f"seen: {r['what']}, {r['n']} units, {r['supply']:g} supply: " \
              + ", ".join(f"{n} {k}" for k, n in sorted(r["types"].items(), key=lambda kv: -kv[1])) + f" {where}"
        if r.get("closing") is not None:
            out += f", closing {r['closing']:+g} tiles in 5 s"
        if r.get("new"):
            out += f", {r['new']:g} supply new"
        return out
    if r["kind"] == "belief":
        out = f"belief: zerg army {r['supply']:g} supply expected"
        if "q80" in r:
            out += f" (80th percentile {r['q80']:g})"
        out += f", most seen at once in 3 min {r['seen_3min']:g}"
        if "drones" in r:
            out += f", drones {r['drones']:g}"
        return out
    if r["kind"] == "death":
        out = f"death: {r['type']} at {r['at'][0]},{r['at'][1]} ({r['near']}), "
        out += "alone" if r["alone"] else f"{r['with']} own fighters within 6 tiles (nearest {r['nearest']})"
        out += f", {r['enemy_near']} enemy fighters within 8"
        if r.get("held_by"):
            out += f", held by {r['held_by']}"
        return out + f" (hands: {r['hands']})"
    return f"note: {r['text']}"


def show(rows: list[dict], f0: int, f1: int, args, tag=lambda r: r["side"] + (str(r["slot"]) if "slot" in r else "")) -> None:
    """The records in [f0, f1) in time order, filtered by args; tag(r) names whose line it is."""
    hands: dict[tuple, str] = {}
    counts, fates = Counter(), Counter()
    last, last_key, repeat = None, None, 0

    def flush():
        nonlocal last, last_key, repeat
        if last is not None:
            print(last + (f"  (x{repeat})" if repeat > 1 else ""))
        last, last_key, repeat = None, None, 0
    for r in rows:
        if r["f"] >= f1:
            break
        who = (r.get("arm"), r["side"])
        if r["kind"] == "hands":
            hands[who] = r["owner"]
            continue
        if r["f"] < f0 or (args.side and r["side"] != args.side) or (args.slot is not None and r.get("slot") != args.slot):
            continue
        if r["kind"] == "decision":
            counts[f"{tag(r)} {r['layer']}"] += 1
            if args.layer and r["layer"] not in args.layer:
                continue
        elif r["kind"] == "order":
            fates[f"{tag(r)} {r['fate']}"] += 1
            if args.orders == "none" or args.layer:
                continue
            if args.orders == "fates" and r["fate"] == "sent" and not r["what"].startswith(PRODUCTION):
                continue
        elif r["kind"] == "note":
            if args.notes == "off" or args.layer:
                continue
        elif r["kind"] in ("seen", "belief", "death"):
            counts[f"{tag(r)} {r['kind']}"] += 1
            if args.layer and r["kind"] not in args.layer:
                continue
        else:
            continue
        key = (tag(r), line(r))
        if r["kind"] == "order" and key == last_key:
            repeat += 1                                    # (the same order again: folded)
            continue
        flush()
        last = f"{clock(r['f']):>7} {tag(r)} [{hands.get(who, '?')[:16]:<16}] " + line(r)
        last_key, repeat = key, 1
    flush()
    print("\ndecisions: " + ", ".join(f"{k} {n}" for k, n in sorted(counts.items())))
    print("orders: " + ", ".join(f"{k} {n}" for k, n in sorted(fates.items())))


def diverge(arms: dict[int, list[dict]]) -> tuple | None:
    """The first decision where the arms differ (layer and choice, in order): (index, A0's, A1's)."""
    a = [r for r in arms.get(0, []) if r["kind"] == "decision"]
    b = [r for r in arms.get(1, []) if r["kind"] == "decision"]
    for i in range(min(len(a), len(b))):
        if (a[i]["layer"], a[i]["choice"]) != (b[i]["layer"], b[i]["choice"]):
            return i, a[i], b[i]
    if len(a) != len(b):
        i = min(len(a), len(b))
        return i, a[i] if i < len(a) else None, b[i] if i < len(b) else None
    return None


def pairs(path: Path, args) -> None:
    """The game's paired branches: listed (--pairs), or one pair's arms side by side (--pair N)."""
    p = path.with_name(path.name.replace(".records.jsonl", ".pairs.records.jsonl"))
    if not p.exists():
        sys.exit(f"no pair records at {p} (a game without paired branches, or played before they were kept)")
    rows = [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines()[1:]]
    heads = {r["pair"]: r for r in rows if r["kind"] == "pair"}
    arms: dict[int, dict[int, list]] = {}
    for r in rows:
        if r["kind"] != "pair":
            arms.setdefault(r["pair"], {}).setdefault(r["arm"], []).append(r)
    short = lambda m: Path(m).parent.name or m

    def choice(j, x):
        return f"A{j} {x['choice']} ({pct(x.get('p'))})" if x else f"A{j} (none)"
    if args.pair is None:
        print(f"{p}: {len(heads)} pairs")
        for i, h in heads.items():
            d = diverge(arms.get(i, {}))
            kind = "A/B " + " vs ".join(short(m) for m in h["models"]) if h["ab"] else "training"
            out = f"#{i} {clock(h['f'])} {h['layer']} {kind}: outcomes {h['rewards'][0]:+.0f} / {h['rewards'][1]:+.0f}"
            if d is None:
                out += "; the arms' decisions never differ"
            else:
                at = d[1] or d[2]
                out += f"; differ first at {clock(at['f'])} ({at['layer']}): {choice(0, d[1])} | {choice(1, d[2])}"
            print(out)
        return
    h = heads.get(args.pair)
    if h is None:
        sys.exit(f"no pair #{args.pair}; --pairs lists them")
    print(f"{p} pair #{args.pair}: {h['layer']}, {'A/B' if h['ab'] else 'training'}, from {clock(h['f'])}")
    for j, m in enumerate(h["models"]):
        print(f"  A{j}: {m}  outcome {h['rewards'][j]:+.1f}")
    d = diverge(arms.get(args.pair, {}))
    if d is not None:
        at = d[1] or d[2]
        print(f"  first difference: decision {d[0]}, {clock(at['f'])}: {choice(0, d[1])} | {choice(1, d[2])}")
    merged = sorted((r for a in arms.get(args.pair, {}).values() for r in a), key=lambda r: (r["f"], r["arm"]))
    end = max((r["f"] for r in merged), default=h["f"]) + 1
    print()
    show(merged, h["f"], end, args, tag=lambda r: f"A{r['arm']} {r['side']}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("game", help="a .rep or .records.jsonl, or a folder (with --seed)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--minute", type=float, help="from this minute (5 or 5.5)")
    ap.add_argument("--to", type=float, help="to this minute (default: a minute on)")
    ap.add_argument("--pairs", action="store_true", help="the game's paired branches, listed")
    ap.add_argument("--pair", type=int, help="one paired branch's arms, side by side")
    ap.add_argument("--side", choices=["T", "Z", "P"])
    ap.add_argument("--slot", type=int, help="one player's records in a mirror (eval/mirror.py tags them by slot)")
    ap.add_argument("--layer", action="append", help="only these layers' decisions (macro, army, engage, roles, "
                                                     "posture, gas) or records (seen, belief, death); repeatable")
    ap.add_argument("--orders", choices=["fates", "all", "none"], default="fates")
    ap.add_argument("--notes", choices=["on", "off"], default="on")
    args = ap.parse_args()
    path = find(args.game, args.seed)
    if args.pairs or args.pair is not None:
        return pairs(path, args)
    if args.minute is None:
        sys.exit("--minute (or --pairs, --pair N)")
    rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()[1:]]
    f0, f1 = int(args.minute * 1440), int((args.to if args.to is not None else args.minute + 1) * 1440)
    print(f"{path}: {clock(f0)}-{clock(f1)}")
    show(rows, f0, f1, args)


if __name__ == "__main__":
    main()
