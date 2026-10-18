"""Which inputs moved a learned decision: each group of what the deciding model saw zeroed in turn (as if
not there: no Terran army seen, no money, no history), the chosen option's odds again. The groups that
move them most go in the decision's record (gary/decision_log.py "why"), so that a record says what the
choice rested on, not only what was logged next to it (ARCHITECTURE section 8: counterfactual consistency
by construction). One batched forward pass a decision: a row per group.

Only groups whose absence is a state the player could be in are zeroed: what it saw of the enemy, its
money, its own history and plan. Its own units, supply and the time aren't: with no drones the model wants
a drone (43% -> 96%), true and useless, and such answers crowded out the reasons.

  macro  (gary/policy/macro.py):  minerals, gas, enemy army / workers / buildings in sight now and the most
         seen, its decision history, its build style
  army   (gary/policy/army.py):   the enemy on the map (army and workers in sight, buildings in sight and ever
         seen, army seen lately), enemy units in sight by type, enemy army remembered, money
"""

from __future__ import annotations

import numpy as np

MIN_DELTA = 0.01                # odds moved less than this: not listed
TOP = 3


def _kinds(types: list[int]) -> dict[str, list[int]]:
    """Positions in a list of unit types, by workers / army / buildings."""
    from gary.policy.army import is_building, is_worker, supply_x2
    out = {"workers": [], "army": [], "buildings": [], "other units": []}
    for i, t in enumerate(types):
        out["workers" if is_worker(t) else "buildings" if is_building(t) else "army" if supply_x2(t) > 0
            else "other units"].append(i)
    return out


def macro_groups(spec) -> list[tuple[str, np.ndarray]]:
    """The macro features (encode's layout) that are zeroed, in named groups of column indices."""
    n_own, n_seen, n_dec = len(spec.own_types), len(spec.seen_types), len(spec.decisions)
    seen = _kinds(spec.seen_types)
    o = 7 + 2 * n_own
    groups = [("minerals", [1]), ("gas", [2])]
    for k in ("army", "workers", "buildings"):
        groups.append((f"enemy {k} in sight", [o + i for i in seen[k]]))
    for k in ("army", "workers", "buildings"):
        groups.append((f"enemy {k} most seen", [o + n_seen + i for i in seen[k]]))
    o += 2 * n_seen
    groups.append(("its decisions so far", list(range(o, o + n_dec))))
    o += n_dec
    groups.append(("its build style", list(range(o, spec.size))))
    return [(name, np.array(idx, int)) for name, idx in groups if idx]


def army_groups(spec) -> list[tuple[str, int, np.ndarray]]:
    """The army model's inputs that are zeroed, in named groups: (name, 0 = grid channel / 1 = numbers, indices)."""
    n_own, n_seen = len(spec.own_types), len(spec.seen_types)
    o = 8
    return ([("own losses lately", 0, np.array([9]))] if getattr(spec, "losses", False) else []) + [
            ("enemy army in sight (map)", 0, np.array([3])), ("enemy workers in sight (map)", 0, np.array([4])),
            ("enemy buildings in sight (map)", 0, np.array([5])), ("enemy buildings ever seen", 0, np.array([6])),
            ("enemy army seen lately", 0, np.array([7])), ("money", 1, np.array([1, 2])),
            ("enemy units in sight by type", 1, np.arange(o + n_own, o + n_own + n_seen)),
            ("enemy army remembered", 1, np.array([7] + list(range(o + n_own + n_seen, o + n_own + 2 * n_seen))))]


def _top(p0: float, ps: np.ndarray, names: list[str]) -> list[list]:
    order = np.argsort(-np.abs(ps - p0))
    return [[names[i], round(float(ps[i]), 3)] for i in order[:TOP] if abs(ps[i] - p0) >= MIN_DELTA]


def explain_macro(model, x: np.ndarray, k: int) -> list[list]:
    """[[group, the chosen option's odds without it]...], the groups that move them most."""
    groups = macro_groups(model.spec)
    xs = np.repeat(x[:1], len(groups) + 1, axis=0)
    for i, (_, idx) in enumerate(groups):
        xs[i + 1, idx] = 0.0
    probs, _ = model.probs(xs)
    return _top(float(probs[0, k]), probs[1:, k], [g[0] for g in groups])


def explain_army(model, grid: np.ndarray, glob: np.ndarray, choose) -> list[list]:
    """choose(where, kind) -> the chosen option's odds per row (where: (n, 256), kind: (n, 2))."""
    groups = army_groups(model.spec)
    g = np.repeat(np.asarray(grid, np.float32).reshape(1, -1)[:1], len(groups) + 1, axis=0)
    g = g.reshape(len(groups) + 1, -1, 256)
    b = np.repeat(np.asarray(glob, np.float32)[:1], len(groups) + 1, axis=0)
    for i, (_, part, idx) in enumerate(groups):
        if part == 0:
            g[i + 1, idx] = 0.0
        else:
            b[i + 1, idx] = 0.0
    where, kind = model.predict(g, b)
    ps = np.asarray(choose(where, kind), np.float64)
    return _top(float(ps[0]), ps[1:], [x[0] for x in groups])
