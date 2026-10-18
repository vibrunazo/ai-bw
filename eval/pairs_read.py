"""A round's paired branches read pair by pair: does the branched decision hold, and do the arms' outcomes
differ beyond the noise? (What round e1 was read for by hand: the forced attack was reversed by a later hold
in 94% of pairs, median 26 s in, and the arms' outcomes differed by -0.33 +- 0.43.)

    python -m eval.pairs_read runs/league/e2_fixed                 (the commit layer: keep / engage / move)
    python -m eval.pairs_read runs/league/e1 --layer engage

Per training pair (two arms from one moment, the first decision forced apart): the forced choice in each arm,
the decisions of that layer sent later in the window per arm, whether and when the forced choice was undone
(an engage arm that later moved home, a hold arm that later engaged), and the outcome difference (arm 0 less
arm 1). Then the means: reversal rates, time to the reversal, mean difference with its standard error against
the arms' own spread.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

ACT = {"commit": {"engage": "engage", "move": "move", "keep": "hold"},
       "engage": {"attack": "engage", "hold": "hold"}}


def read(folder: Path, layer: str) -> list[dict]:
    pairs = []
    for p in sorted(folder.rglob("*.pairs.records.jsonl")):
        rows = [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines()[1:]]
        heads = {r["pair"]: r for r in rows if r["kind"] == "pair"}
        arms: dict[int, dict[int, list]] = {}
        for r in rows:
            if r["kind"] == "decision" and r.get("layer") == layer:
                arms.setdefault(r["pair"], {}).setdefault(r["arm"], []).append(r)
        for i, h in heads.items():
            if h["ab"]:
                continue
            item = {"game": p.name, "pair": i, "f": h["f"], "rewards": h["rewards"], "arms": {}}
            for arm in (0, 1):
                decs = arms.get(i, {}).get(arm, [])
                forced = next((r for r in decs if str(r.get("by", "")).startswith("forced")), None)
                later = [r for r in decs if r is not forced and r.get("then") == "sent"]
                def kind(r):
                    word = (re.match(r"[a-z_]+", str(r["choice"]).lower()) or re.match("", "")).group(0)
                    return ACT.get(layer, {}).get(word, word)
                fk = kind(forced) if forced else None
                undone = None
                for r in later:
                    k = kind(r)
                    if (fk == "engage" and k in ("move", "hold")) or (fk in ("hold", "move") and k == "engage"):
                        undone = (r["f"] - h["f"]) / 24
                        break
                item["arms"][arm] = {"forced": fk, "later": len(later), "later_engage": sum(1 for r in later if kind(r) == "engage"),
                                     "undone_s": undone}
            pairs.append(item)
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("round", help="the round's folder (runs/league/<round>)")
    ap.add_argument("--layer", default="commit", help="the layer's name in the records (commit, engage, army)")
    args = ap.parse_args()
    pairs = read(Path(args.round), args.layer)
    if not pairs:
        raise SystemExit("no training pairs with that layer's decisions")
    diffs = [p["rewards"][0] - p["rewards"][1] for p in pairs]
    sd = st.pstdev(diffs) if len(diffs) > 1 else 0.0
    se = sd / max(1, len(diffs)) ** 0.5
    arm_sd = st.pstdev([r for p in pairs for r in p["rewards"]])
    print(f"{len(pairs)} training pairs in {args.round} ({args.layer})")
    print(f"outcome difference (arm 0 - arm 1): {st.mean(diffs):+.2f} +- {se:.2f} (sd {sd:.2f}; an arm's own sd {arm_sd:.2f})")
    for arm in (0, 1):
        a = [p["arms"][arm] for p in pairs if p["arms"].get(arm)]
        forced = [x["forced"] for x in a if x["forced"]]
        kinds = {k: forced.count(k) for k in sorted(set(forced))}
        later = st.mean([x["later"] for x in a]) if a else 0
        eng = st.mean([x["later_engage"] for x in a]) if a else 0
        und = [x["undone_s"] for x in a if x["undone_s"] is not None]
        with_f = [x for x in a if x["forced"]]
        print(f"arm {arm}: forced {kinds}; later decisions sent {later:.1f} a window ({eng:.1f} engage); "
              f"forced choice undone in {len(und)} of {len(with_f)}"
              + (f", median {st.median(und):.0f} s in" if und else ""))
    both = [p for p in pairs if p["arms"].get(0, {}).get("forced") and p["arms"].get(1, {}).get("forced")]
    kept = [p for p in both if p["arms"][0]["undone_s"] is None and p["arms"][1]["undone_s"] is None]
    if kept:
        d = [p["rewards"][0] - p["rewards"][1] for p in kept]
        print(f"pairs where neither arm undid its choice: {len(kept)}; their difference {st.mean(d):+.2f} "
              f"+- {st.pstdev(d) / max(1, len(d)) ** 0.5:.2f}")


if __name__ == "__main__":
    main()
