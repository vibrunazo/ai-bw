"""When and where to cast a spell: learned from the pros' casts (ingest/cast_dataset.py, train/cast.py).

A spell is an entry of SPELLS (Psionic Storm first; the other races' area spells take the same shape). Two models
per spell, trained on the pros' games:

  when   every half second in a fight, each caster with the energy for it is a sample: what the player sees around
         it (CAST_NAMES), and whether the player cast it with that unit in the next second
  where  at each cast, the candidate points: the visible enemy units within reach (a point on a unit, as a player
         clicks); each is a sample (SPOT_NAMES) and the one nearest where the pro cast is the label

Gary asks the same about its own casters (fight.py CastRule) and casts at the spot the second model ranks first.
The player's own casts of the last few seconds are inputs of both (pros don't cast twice on one spot): in the pros'
games the casts found in the data, in Gary's its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gary.policy.army import is_building, is_worker, supply_x2
from gary.races import FLYERS


@dataclass(frozen=True)
class Spell:
    name: str
    caster: int                 # unit type
    order: int                  # the cast's order id (OpenBW Orders)
    tech: int                   # its research (tech id)
    research: int               # frames to research it
    energy: int
    radius: int                 # pixels: the area it hits, from the point (about)
    reach: int                  # pixels: candidate points this far from the caster (its range and a short walk)


SPELLS = {"storm": Spell("storm", 67, 142, 19, 1800, 75, 48, 11 * 32)}
SAMPLE_EVERY = 12               # frames: a sample every half second
CAST_WITHIN = 24                # the when model's label: a cast within this many frames
FIGHT_PX = 12 * 32              # a caster with a visible enemy unit this close is in a fight (sampled)
RECENT = 4 * 24                 # the player's own casts this recent count as on the spot
CAST_AGAIN = 3 * 24             # Gary: a caster it just ordered left alone this long (the walk and the cast)
RINGS = (4 * 32, 7 * 32, 10 * 32)
SPOT_NAMES = ["dist", "enemies", "enemy_supply", "enemy_workers", "enemy_air", "own", "own_supply",
              "enemies_96", "recent_cast", "moving"]
CAST_NAMES = (["energy", "hp_frac", "minute", "casters_near", "own_supply_8", "nearest_enemy", "targeting_me",
               "hp_lost", "spots"]
              + [f"{k}_{r // 32}" for r in RINGS for k in ("enemies", "enemy_supply", "workers", "air")]
              + [f"best_{n}" for n in SPOT_NAMES] + [f"second_{n}" for n in SPOT_NAMES])
STILL_ORDERS = {0, 1, 2, 3, 107}     # (die, stop, guard, player guard, hold position...) not moving


def enemy_units(units: list[dict], slot: int) -> list[dict]:
    """The enemy units the player sees, buildings left out (a full observation says who sees each unit; Gary's own
    has only what it sees, without visible_to)."""
    return [u for u in units if u["owner"] not in (slot, 11) and not is_building(u["type"]) and u.get("completed", 1)
            and ("visible_to" not in u or (u["visible_to"] >> slot) & 1)]


def in_fight(u: dict, enemies: list[dict]) -> bool:
    return any(math.dist((u["x"], u["y"]), (e["x"], e["y"])) < FIGHT_PX for e in enemies)


def _hp(u: dict) -> int:
    return u["hp"] + u.get("shields", 0)


def spots(spell: Spell, caster: dict, own: list[dict], enemies: list[dict],
          recent: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], np.ndarray]:
    """The candidate points for a cast (on the enemy units within reach) and each one's view (SPOT_NAMES)."""
    cx, cy = caster["x"], caster["y"]
    pts, rows = [], []
    r = spell.radius
    for e in enemies:
        d = math.dist((cx, cy), (e["x"], e["y"]))
        if d > spell.reach:
            continue
        x, y = e["x"], e["y"]
        hit = [q for q in enemies if math.dist((x, y), (q["x"], q["y"])) < r]
        mine = [q for q in own if math.dist((x, y), (q["x"], q["y"])) < r + 16]
        rows.append([d / 32, len(hit), sum(supply_x2(q["type"]) for q in hit),
                     sum(1 for q in hit if is_worker(q["type"])), sum(1 for q in hit if q["type"] in FLYERS), len(mine), sum(supply_x2(q["type"]) for q in mine),
                     sum(1 for q in enemies if math.dist((x, y), (q["x"], q["y"])) < 2 * r),
                     sum(1 for p in recent if math.dist((x, y), p) < 2 * r),
                     0.0 if e.get("order") in STILL_ORDERS else 1.0])
        pts.append((x, y))
    return pts, np.array(rows, np.float32).reshape(-1, len(SPOT_NAMES))


def spot_value(rows: np.ndarray) -> np.ndarray:
    """A plain ranking of the spots for the cast model's best and second (enemy supply hit less own supply hit)."""
    if not len(rows):
        return np.zeros(0, np.float32)
    col = SPOT_NAMES.index
    return rows[:, col("enemy_supply")] - 2 * rows[:, col("own_supply")] - 4 * rows[:, col("recent_cast")]


def cast_features(spell: Spell, u: dict, own: list[dict], enemies: list[dict], frame: int,
                  recent: list[tuple[int, int]], hp_before: int | None = None) -> np.ndarray:
    """One caster's view of the fight (CAST_NAMES). recent: the player's own casts of the last RECENT frames."""
    x, y = u["x"], u["y"]
    d = [(math.dist((x, y), (e["x"], e["y"])), e) for e in enemies]
    rings = []
    for r in RINGS:
        near = [e for dist, e in d if dist < r]
        rings += [len(near), sum(supply_x2(e["type"]) for e in near), sum(1 for e in near if is_worker(e["type"])),
                  sum(1 for e in near if e["type"] in FLYERS)]
    casters = sum(1 for o in own if o is not u and o["type"] == spell.caster and (o.get("energy") or 0) >= spell.energy
                  and math.dist((x, y), (o["x"], o["y"])) < 8 * 32)
    own_sup = sum(supply_x2(o["type"]) for o in own if math.dist((x, y), (o["x"], o["y"])) < 8 * 32)
    hp_max = max(1, (u.get("hp_max") or 40) + 40)
    pts, rows = spots(spell, u, own, enemies, recent)
    order = np.argsort(-spot_value(rows)) if len(rows) else []
    best = rows[order[0]] if len(rows) else np.zeros(len(SPOT_NAMES), np.float32)
    second = rows[order[1]] if len(rows) > 1 else np.zeros(len(SPOT_NAMES), np.float32)
    head = [u.get("energy") or 0, _hp(u) / hp_max, frame / 1440, casters, own_sup,
            min((dist for dist, _ in d), default=FIGHT_PX) / 32,
            sum(1 for _, e in d if e.get("order_target") == u["tag"]),
            (hp_before - _hp(u)) if hp_before is not None else 0, len(rows)]
    return np.concatenate([np.array(head + rings, np.float32), best, second])


class SpotScorer:
    """The where model's network (train/cast.py train_where), in numpy: a score for each spot."""

    def __init__(self, mu: np.ndarray, sd: np.ndarray, layers: list[tuple[np.ndarray, np.ndarray]]):
        self.mu, self.sd, self.layers = mu, sd, layers

    def scores(self, rows: np.ndarray) -> np.ndarray:
        h = (rows - self.mu) / self.sd
        for i, (w, b) in enumerate(self.layers):
            h = h @ w + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0)
        return h[:, 0]


class CastModel:
    """The trained models (train/cast.py): P(the pros cast with this unit in the next second), and the spot."""

    def __init__(self, path):
        import pickle
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.when, self.where, self.path = d["when"], d["where"], str(path)
        self.spell = SPELLS[d["spell"]]

    def wants(self, X: np.ndarray) -> np.ndarray:
        return self.when.predict_proba(X)[:, 1]

    def spot(self, rows: np.ndarray) -> int:
        return int(np.argmax(self.where.scores(rows)))
