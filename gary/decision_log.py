"""Every layer's decisions in one game: for the league's RL (arrays) and for reading a game (records).

Arrays (GARY_DECISION_LOG names the .npz; off without it): the learned layers' decisions as RL trains
on them, each decision's inputs, what was drawn and its log-probability, saved when the game's process
ends. Layers, by side (both sides of a headless game share the process): "army_z" (gary/bots/gary_z.py:
where a group goes, move or attack), "roles_z" (gary/agent/zerg_roles.py: each unit's role in a fight
snapshot), "army_t" (gary/agent/army.py, with draw), "engage_t" (its engage option), "roles_t"
(gary/agent/roles.py ahead/v1: its units' roles before contact). Stacked per layer: "<layer>/<field>".

Records (always kept; saved next to the replay as <game>.records.jsonl, read with eval/why.py): the
same decisions and every other layer's, in one readable shape, so that a why-question about a game is
a query over it rather than a script that re-simulates it. One JSON line each:

  decision  {f, side, layer, choice, p, top: [[option, p]...], in: {what it saw}, by, over: [[option,
            why passed over]...], then, rl, p_other}: by is who chose (the model, a draw, a rule, the
            posture); over, what intervened: the likelier or drawn options set aside before the choice, and
            why (a rule, the money, a busy building, the posture); then, what came of it right away (parked,
            held...); rl, its row among the layer's RL decisions (the arrays above, or a paired arm's in
            eval/pairs.py); p_other, the same choice's odds under the other model of an A/B pair; why,
            for a learned layer, the input groups that move the choice's odds most, each as [group, the
            odds with it zeroed] (gary/policy/attribution.py)
  order     {f, side, what, fate, why, by}: an order's fate: sent (to the game), refused (by the
            interface: apm, off screen; or by the game), retried, dropped, done; by, who had the hands
  hands     {f, side, owner}: who has the hands from now (Gary-T's arbiter; Gary-Z's turns)
  note      {f, side, text}: what a layer said in the game's log (a rule firing, a group sent)
  seen, belief, death   what Gary saw and lost between decisions (gary/agent/sightings.py): enemy
            fighters in sight and whether they close on its buildings, enemy buildings first seen, the
            belief model's estimate once a minute, each own fighter lost (alone or not, who held it)

A paired branch's arms (eval/pairs.py) keep their own records, saved beside the pairs as
<game>.pairs.records.jsonl: a "pair" line ({pair, f, layer, ab, rewards, models}) then each arm's
records, tagged {pair, arm}. eval/why.py --pair N shows both arms side by side.
"""

from __future__ import annotations

import atexit
import json
import os
from collections import defaultdict

import numpy as np

TOP = 5                               # alternatives kept per decision


def top(names, probs, k: int = TOP) -> list[list]:
    """The k likeliest options as [name, p], for a record."""
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    idx = np.argsort(-p)[:k]
    return [[names(int(i)) if callable(names) else names[int(i)], round(float(p[i]), 3)] for i in idx if p[i] > 0]


class DecisionLog:
    def __init__(self):
        self.slot: int | None = None            # the player whose records are being written (a mirror launcher sets it)
        self.path = os.environ.get("GARY_DECISION_LOG")
        self.items: dict[str, list[dict]] = defaultdict(list)
        self.records: list[dict] = []
        self.owner: dict[str, str] = {}   # side -> who has the hands (stamped on its orders)
        self.held: dict[str, str] = {}    # side -> the hands' owner last recorded
        if self.path:
            atexit.register(self.save)

    @property
    def on(self) -> bool:
        return bool(self.path)

    def add(self, layer: str, **fields) -> None:
        if self.path:
            self.items[layer].append(fields)

    def save(self) -> None:
        out = {}
        for layer, items in self.items.items():
            for k in items[0]:
                out[f"{layer}/{k}"] = np.stack([np.asarray(it[k]) for it in items])
        if out:
            np.savez_compressed(self.path, **out)

    # --- records ---------------------------------------------------------------------------------

    def _who(self, side: str) -> str:
        """The key of a player's hands state: the side, plus the slot when a game has set one (a mirror)."""
        return side if self.slot is None else f"{side}{self.slot}"

    def _put(self, r: dict) -> None:
        """A record; with the player's slot when a game has set one (a mirror: both sides are the same race,
        so "side" alone does not tell the players apart; eval/why.py --slot)."""
        if self.slot is not None:
            r["slot"] = self.slot
        self.records.append(r)

    def decision(self, side: str, layer: str, frame: int, choice, p: float | None = None, top: list | None = None,
                 inputs: dict | None = None, by: str = "model", over: list | None = None, then: str | None = None,
                 rl: int | None = None, p_other: float | None = None, why: list | None = None) -> dict:
        r = {"f": int(frame), "kind": "decision", "side": side, "layer": layer, "choice": choice, "by": by}
        if p is not None:
            r["p"] = round(float(p), 3)
        if p_other is not None:
            r["p_other"] = round(float(p_other), 3)
        if rl is not None:
            r["rl"] = int(rl)
        if top:
            r["top"] = top
        if inputs:
            r["in"] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in inputs.items()}
        if over:
            r["over"] = over
        if then:
            r["then"] = then
        if why:
            r["why"] = why
        self._put(r)
        return r

    def order(self, side: str, frame: int, what: str, fate: str, why: str = "", by: str | None = None) -> None:
        r = {"f": int(frame), "kind": "order", "side": side, "what": what, "fate": fate}
        if why:
            r["why"] = why
        r["by"] = by if by is not None else self.owner.get(self._who(side), "?")
        self._put(r)

    def hands(self, side: str, frame: int, owner: str) -> None:
        who = self._who(side)
        self.owner[who] = owner
        if self.held.get(who) != owner:
            self.held[who] = owner
            self._put({"f": int(frame), "kind": "hands", "side": side, "owner": owner})

    def note(self, side: str, frame: int, text: str) -> None:
        self._put({"f": int(frame), "kind": "note", "side": side, "text": text})

    def save_records(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"schema": "records/v1"}) + "\n")
            for r in sorted(self.records, key=lambda r: r["f"]):      # (stable: same-frame order kept)
                f.write(json.dumps(r) + "\n")

    def clear_records(self) -> None:
        self.records.clear()
        self.owner.clear()
        self.held.clear()


LOG = DecisionLog()
