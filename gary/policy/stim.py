"""When to stim: learned from the pros' stims (ingest/stim_dataset.py, train/stim.py).

Every half second in a fight, each of the Terran's marines and firebats with a zerg fighter within
10 tiles is a sample: what the player sees around it (below), and whether the player stimmed it in
the next half second. Gary asks the model the same about its own units and stims the ones it says
to (fight.py StimRule). The stim a unit has left is the game's own in the pros' games (OpenBW's
stim timer) and, in Gary's, what Gary remembers of its own stims (the live client doesn't report
it): STIM_TIME timer units (8 frames each) from the press.
"""

from __future__ import annotations

import math

import numpy as np

MARINE, FIREBAT, MEDIC = 0, 32, 34
BIO = (MARINE, FIREBAT)
STIM_TIME = 37                   # the stim timer right after a stim (OpenBW: counts down every 8 frames)
FIGHT_PX = 10 * 32               # a unit with a zerg fighter this close is in a fight (sampled)
SAMPLE_EVERY = 12                # frames: a sample every half second
# zerg fighters by kind (BW unit ids): zergling, hydralisk, lurker, mutalisk, ultralisk, scourge,
# guardian, defiler, sunken colony
KINDS = {37: 0, 38: 1, 103: 2, 43: 3, 39: 4, 47: 5, 44: 6, 46: 7, 146: 8}
RINGS = (4 * 32, 7 * 32, 10 * 32)
SUPPLY_X2 = {37: 1, 38: 2, 103: 4, 43: 4, 39: 8, 47: 1, 44: 4, 46: 4, 146: 4,
             MARINE: 2, FIREBAT: 2, MEDIC: 2, 5: 4, 30: 4, 3: 4, 2: 4, 58: 4, 9: 4}
NAMES = (["hp", "hp_frac", "stimmed", "stim_left", "own_bio_4", "own_bio_8", "medics_6", "targeting_me",
          "nearest_enemy", "cooldown", "attacking", "own_supply_8", "enemy_supply_7", "ratio", "minute",
          "hp_lost", "closing", "enemy_supply_7_growth", "stimmed_near"]
         + [f"{k}_{r // 32}" for r in RINGS for k in ("ling", "hydra", "lurker", "muta", "ultra", "scourge",
                                                       "guardian", "defiler", "sunken")])
ATTACK_ORDERS = {10, 14}         # attack unit, attack move


def enemy_fighters(units: list[dict], slot: int) -> list[dict]:
    """The zerg fighters the player sees (a game's full observation says who sees each unit; Gary's
    own has only what it sees, without visible_to)."""
    return [u for u in units if u["owner"] not in (slot, 11) and u["type"] in KINDS
            and ("visible_to" not in u or (u["visible_to"] >> slot) & 1)]


def features(u: dict, own: list[dict], enemies: list[dict], stim_left: int, frame: int,
             before: tuple[float, float, float] | None = None, stim_of=None) -> np.ndarray:
    """One bio unit's view of the fight (NAMES). before: its (hp, nearest enemy in tiles, enemy supply
    within 7 tiles) half a second ago (the fight starting: hit, the zerg closing in, more coming);
    stim_of: tag -> stim left, for its neighbors (pros stim a group together)."""
    x, y = u["x"], u["y"]
    d = [(math.dist((x, y), (e["x"], e["y"])), e) for e in enemies]
    rings = np.zeros((len(RINGS), 9), np.float32)
    for dist, e in d:
        for i, r in enumerate(RINGS):
            if dist < r:
                rings[i, KINDS[e["type"]]] += 1
    near_own = [(math.dist((x, y), (o["x"], o["y"])), o) for o in own if o is not u]
    bio4 = sum(1 for dist, o in near_own if dist < 4 * 32 and o["type"] in BIO)
    bio8 = sum(1 for dist, o in near_own if dist < 8 * 32 and o["type"] in BIO)
    medics = sum(1 for dist, o in near_own if dist < 6 * 32 and o["type"] == MEDIC)
    own_sup = sum(SUPPLY_X2.get(o["type"], 0) for dist, o in near_own if dist < 8 * 32) + 2
    enemy_sup = sum(SUPPLY_X2.get(e["type"], 0) for dist, e in d if dist < 7 * 32)
    hp_max = max(1, u.get("hp_max") or 50)
    head = [u["hp"], u["hp"] / hp_max, 1.0 if stim_left > 0 else 0.0, stim_left / STIM_TIME, bio4, bio8, medics,
            sum(1 for _, e in d if e.get("order_target") == u["tag"]), min((dist for dist, _ in d), default=FIGHT_PX) / 32,
            1.0 if u.get("cooldown", 0) > 0 else 0.0, 1.0 if u.get("order") in ATTACK_ORDERS else 0.0,
            own_sup, enemy_sup, enemy_sup / own_sup, frame / 1440]
    nearest = head[8]
    hp0, near0, sup0 = before if before is not None else (u["hp"], nearest, enemy_sup)
    bio_near = [o for dist, o in near_own if dist < 5 * 32 and o["type"] in BIO]
    stimmed_near = (sum(1 for o in bio_near if stim_of(o["tag"]) > 0) / len(bio_near)) if bio_near and stim_of else 0.0
    head += [hp0 - u["hp"], near0 - nearest, enemy_sup - sup0, stimmed_near]
    return np.concatenate([np.array(head, np.float32), rings.reshape(-1)])


def recall(u: dict, enemies: list[dict]) -> tuple[float, float, float]:
    """What features() needs half a second later as `before`."""
    d = [math.dist((u["x"], u["y"]), (e["x"], e["y"])) for e in enemies]
    sup = sum(SUPPLY_X2.get(e["type"], 0) for e, dist in zip(enemies, d) if dist < 7 * 32)
    return (u["hp"], min(d, default=FIGHT_PX) / 32, sup)


def in_fight(u: dict, enemies: list[dict]) -> bool:
    return any(math.dist((u["x"], u["y"]), (e["x"], e["y"])) < FIGHT_PX for e in enemies)


class StimModel:
    """The trained model (train/stim.py): P(the pros stim this unit in the next half second)."""

    def __init__(self, path):
        import pickle
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.clf, self.threshold, self.path = d["model"], d["threshold"], str(path)

    def wants(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1] >= self.threshold
