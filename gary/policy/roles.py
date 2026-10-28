"""Gary's role model: what each of its fighting units does for the next half second (docs/CONTROL.md).

Learned from the pros' units in home defenses (ingest/role_dataset.py, trained by train/roles.py):
the role each unit had (eval/micro_roles.py's labels: fire, close in, pull back, retreat,
reposition, stand, other). Input: the fight as the player sees it, the same unit rows as the fight
model (gary/policy/fight.py: type, side, position relative to the fight mirrored so the main is
top-left, HP, shields, cooldown, order and its target, carrying, completed), plus per unit what a
player sees move: its motion over the last half second, and how many enemies target it. A small
transformer lets every unit see every other; out, per own unit, a role.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from gary.policy.fight import N_ORDERS, N_TYPES, ORDER, OTARGET, SIDE, TYPE, numeric

ROLES = ["fire", "close_in", "pull_back", "retreat", "reposition", "stand", "other", "siege", "unsiege", "mine"]
# (siege, unsiege, mine: the modes, #60; a dataset built before them has none, and a model names its own roles)
# the decision model: a new role when the unit's changes, else keep; and roles the hand rules use or
# the pros' units have no label for: post (the hand rules' post by the Command Center), join (to
# Gary's biggest bunch of fighters), hold (hold position)
DECISIONS = ROLES[:7] + ["keep", "post", "join", "hold"] + ROLES[7:]    # (the modes after: older logs keep their indices)
# again (train/roles.py --again): the unit's order given again, to where the offset head points (the pros' re-issues,
# ingest/reissue_labels.py: a keep with a command of the player's in the next half second)
AGAIN = "again"
# the control-group key as an input (train/roles.py --keys): per unit a one-hot of its key (0-9) and a column for
# none; the key numbers shuffled per snapshot in training, so the model reads "these units share a key", not what
# a number means (ingest/key_labels.py: the pros' key-mates take the same decision 80% of the time, 72% shuffled)
KEY_DIMS = 11


def key_onehot(keys: np.ndarray) -> np.ndarray:
    """(..., KEY_DIMS) from keys (...): 0-9 a key, anything else none."""
    keys = np.asarray(keys)
    out = np.zeros(keys.shape + (KEY_DIMS,), np.float32)
    on = (keys >= 0) & (keys <= 9)
    out[on, keys[on].astype(int)] = 1.0
    out[~on, KEY_DIMS - 1] = 1.0
    return out
MAX_UNITS = 64
TERRAIN = 8                          # the ground under each unit and the player's own bases (Ground)
EXTRA = 6 + TERRAIN                  # motion x, y over the last half second (tiles), enemies targeting it;
                                     # the unit's age, its distance to Gary's biggest bunch of fighters,
                                     # that bunch's size (gary/agent/roles.py); TERRAIN (models before
                                     # have 3 or 6: they read the first n_extra)
AGE_SCALE = 60.0                     # seconds
PACK_DIST_SCALE = 320.0              # pixels
PACK_SIZE_SCALE = 12.0
PACK_LINK = 4 * 32                   # units this close to each other are one bunch
SHELTERS = {125}                     # bunkers: in the bunch, weighing 8 (supply x2)
RACE_SHELTERS = {"T": {125}, "Z": {146}, "P": {162, 172}}   # where each race's units shelter (the terrain inputs):
                                     # bunkers; sunken colonies; photon cannons and shield batteries


def biggest_bunch(units: list[dict]) -> list[dict]:
    """The biggest bunch (by supply x2; a bunker 8) of these units, linked within PACK_LINK."""
    import math
    from gary.policy.army import supply_x2
    weight = lambda u: 8 if u["type"] in SHELTERS else supply_x2(u["type"])
    left, best = set(range(len(units))), None
    while left:
        todo, bunch = [left.pop()], []
        while todo:
            i = todo.pop()
            bunch.append(units[i])
            near = [j for j in left if math.dist((units[i]["x"], units[i]["y"]), (units[j]["x"], units[j]["y"])) < PACK_LINK]
            left -= set(near)
            todo += near
        if best is None or sum(map(weight, bunch)) > sum(map(weight, best)):
            best = bunch
    return best or []


def own_extras(extra: np.ndarray, tags: list[int], by_tag: dict, slot: int, first_seen: dict, now: int,
               fill: bool = True) -> np.ndarray:
    """Extras 3-5 for the player's fighters in a snapshot (the same in play, gary/agent/roles.py, and in the role
    datasets, ingest/role_dataset.py): its age since it was first in a fight snapshot (first_seen: tag -> frame, kept
    here), its distance to the player's biggest bunch of fighters (with its bunkers), that bunch's size."""
    import math
    from gary.policy.army import is_building, is_worker, supply_x2
    mine = [by_tag[t] for t in tags if t in by_tag and by_tag[t]["owner"] == slot and supply_x2(by_tag[t]["type"]) > 0
            and not is_worker(by_tag[t]["type"]) and not is_building(by_tag[t]["type"]) and by_tag[t]["completed"]]
    for u in mine:
        first_seen.setdefault(u["tag"], now)
    if not mine or not fill:
        return extra
    bunkers = [u for u in by_tag.values() if u["owner"] == slot and u["type"] in SHELTERS and u["completed"]]
    best = biggest_bunch(mine + bunkers)
    fighters = [u for u in best if u in mine] or best
    mid = (sum(u["x"] for u in fighters) / len(fighters), sum(u["y"] for u in fighters) / len(fighters))
    in_best = {u["tag"] for u in best}
    size = sum(1 for u in mine if u["tag"] in in_best)
    index = {t: i for i, t in enumerate(tags)}
    for u in mine:
        i = index[u["tag"]]
        if i < len(extra):
            extra[i, 3] = (now - first_seen[u["tag"]]) / 24 / AGE_SCALE
            extra[i, 4] = math.dist((u["x"], u["y"]), mid) / PACK_DIST_SCALE
            extra[i, 5] = size / PACK_SIZE_SCALE
    return extra
MOTION_SCALE = 32.0


def extra_features(rows: np.ndarray, tags: list[int], positions: dict[int, tuple[float, float]],
                   prev: dict[int, tuple[float, float]], flip: tuple[bool, bool]) -> np.ndarray:
    """(n, EXTRA): each unit's motion since the previous snapshot (mirrored like the rows; 0 if it
    wasn't seen then), and how many units in the snapshot target it."""
    n = len(rows)
    out = np.zeros((n, EXTRA), np.float32)
    for i, t in enumerate(tags):
        if t in prev and t in positions:
            dx, dy = positions[t][0] - prev[t][0], positions[t][1] - prev[t][1]
            out[i, 0] = (-dx if flip[0] else dx) / MOTION_SCALE
            out[i, 1] = (-dy if flip[1] else dy) / MOTION_SCALE
    targets = rows[:, OTARGET]
    for j in targets[(targets >= 0) & (targets < n)]:
        out[j, 2] += 1
    return out


GROUND_SCALE = 1024.0                # pixels
WIDTH_SCALE = 256.0                  # pixels: a passage's width


class Ground:
    """The map as the role model sees it under each unit (map knowledge, as a player studies the map
    before playing: Game.terrain() and gary/terrain.py's analysis): ground height, how wide the
    passage is there (narrow: a ramp or a choke), and the player's main plateau and ramp."""

    def __init__(self, walk: np.ndarray, height: np.ndarray, start: tuple[float, float], ramp_top):
        from scipy import ndimage
        self.walk, self.height = walk, height
        self.alt = ndimage.distance_transform_edt(walk)
        ys, xs = np.nonzero(walk)
        k = int(np.argmin((ys * 8 + 4 - start[1]) ** 2 + (xs * 8 + 4 - start[0]) ** 2))
        cell = (int(ys[k]), int(xs[k]))
        labels, _ = ndimage.label(walk & (height == height[cell]))
        self.main = labels == labels[cell]
        self.ramp_top = ramp_top

    @classmethod
    def of(cls, game, start: tuple[float, float]) -> "Ground":
        """For the player whose main is at start (pixels)."""
        from gary.mapinfo import MapInfo
        from gary.terrain import map_analysis
        walk, height, _ = game.terrain()
        a = map_analysis(game, MapInfo.from_game(game))["starts"]
        mine = a[min(a, key=lambda k: (a[k]["start"][0] - start[0]) ** 2 + (a[k]["start"][1] - start[1]) ** 2)]
        return cls(walk, height, start, tuple(mine["ramp_top"]) if mine["reliable"] else None)

    def cell(self, x: float, y: float) -> tuple[int, int]:
        h, w = self.walk.shape
        return min(h - 1, max(0, int(y // 8))), min(w - 1, max(0, int(x // 8)))


def terrain_features(tags: list[int], positions: dict[int, tuple[float, float]], ground: "Ground | None",
                     ccs: list[tuple[float, float]], bunkers: list[tuple[float, float]],
                     flip: tuple[bool, bool]) -> np.ndarray:
    """(n, TERRAIN) per unit: its ground's height (0 low to 1 high), the passage's width there, in
    the player's main (0/1); toward the player's nearest Command Center (x, y mirrored like the rows)
    and how far; how far the nearest bunker and the main's ramp are (anywhere on the map: the
    player knows where its own bases are). Zeros without a map (drills)."""
    out = np.zeros((len(tags), TERRAIN), np.float32)
    if ground is None:
        return out
    far = 2.0
    for i, t in enumerate(tags):
        if t not in positions:
            continue
        x, y = positions[t]
        c = ground.cell(x, y)
        out[i, 0] = ground.height[c] / 2.0
        out[i, 1] = min(2 * 8 * float(ground.alt[c]), 2 * WIDTH_SCALE) / WIDTH_SCALE
        out[i, 2] = float(ground.main[c])
        if ccs:
            cx, cy = min(ccs, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
            dx, dy = cx - x, cy - y
            out[i, 3] = (-dx if flip[0] else dx) / GROUND_SCALE
            out[i, 4] = (-dy if flip[1] else dy) / GROUND_SCALE
            out[i, 5] = min(np.hypot(dx, dy) / GROUND_SCALE, far)
        else:
            out[i, 5] = far
        out[i, 6] = min(min((np.hypot(bx - x, by - y) for bx, by in bunkers), default=far * GROUND_SCALE) / GROUND_SCALE, far)
        ramp = ground.ramp_top
        out[i, 7] = min(np.hypot(ramp[0] - x, ramp[1] - y) / GROUND_SCALE, far) if ramp is not None else far
    return out


class RoleNet(nn.Module):
    def __init__(self, d: int = 128, layers: int = 3, heads: int = 4, n_roles: int = len(ROLES), n_extra: int = 3,
                 move: bool = False, offset: bool = False):
        super().__init__()
        self.n_extra = n_extra
        self.type_emb = nn.Embedding(N_TYPES, 48)
        self.side_emb = nn.Embedding(3, 8)
        self.order_emb = nn.Embedding(N_ORDERS, 16)
        self.inp = nn.Linear(48 + 8 + 16 + 9 + n_extra, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.role = nn.Linear(d, n_roles)
        # where: each unit's move over the next half second (tiles, mirrored as the rows), as the pros
        # moved theirs (ingest/response_dataset.py): a role names what, this where (a reposition among
        # the mineral line's workers, a retreat toward the buildings)
        self.move = nn.Linear(d, 2) if move else None
        # where an again order points (tiles from the unit, mirrored as the rows; the pros' commands' targets)
        self.offset = nn.Linear(d, 2) if offset else None

    def forward(self, kind, side, order, num, extra, pad, with_move: bool = False, with_offset: bool = False):
        """kind, side, order (B, N) long; num (B, N, 9); extra (B, N, EXTRA); pad (B, N) True where
        there is no unit. Returns role logits (B, N, R); with_move, also the moves (B, N, 2); with_offset, also
        the again orders' offsets (B, N, 2), after the moves when both."""
        h = self.inp(torch.cat([self.type_emb(kind), self.side_emb(side), self.order_emb(order), num,
                                extra[..., :self.n_extra]], -1))
        h = self.body(h, src_key_padding_mask=pad)
        out = (self.role(h),) + ((self.move(h),) if with_move else ()) + ((self.offset(h),) if with_offset else ())
        return out if len(out) > 1 else out[0]


def tensors(units: np.ndarray, extra: np.ndarray, pad: np.ndarray, time_s: np.ndarray, device):
    u = torch.as_tensor(units.astype(np.int64), device=device)
    return (u[..., TYPE].clamp(0, N_TYPES - 1), u[..., SIDE].clamp(0, 2), u[..., ORDER].clamp(0, N_ORDERS - 1),
            torch.as_tensor(numeric(units, time_s), device=device), torch.as_tensor(extra, device=device),
            torch.as_tensor(pad, device=device))


class RoleModel:
    def __init__(self, net: RoleNet, config: dict):
        self.net, self.config = net.eval(), config

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "RoleModel":
        ck = torch.load(path, map_location=device, weights_only=True)
        net = RoleNet(**ck["config"]["net"])
        net.load_state_dict(ck["state"])
        model = cls(net.to(device), ck["config"])
        model.path = str(path)
        return model

    def save(self, path: str | Path) -> None:
        torch.save({"state": self.net.state_dict(), "config": self.config}, path)

    def inputs(self, extra: np.ndarray, budget: float | None = None, keys: np.ndarray | None = None) -> np.ndarray:
        """The per-unit extras as this model reads them: the budget column and the key one-hot when it was trained
        with them (without keys given: none for every unit)."""
        extra = self._with_budget(extra, budget)
        if self.config.get("keys"):
            k = np.full(extra.shape[:-1], -1) if keys is None else np.asarray(keys)[:extra.shape[0]]
            extra = np.concatenate([extra, key_onehot(k).astype(extra.dtype)], -1)
        return extra

    def _with_budget(self, extra: np.ndarray, budget: float | None) -> np.ndarray:
        """A model trained with the player's order rate as an input (train/roles.py --budget) gets it as one more
        column (/ 100); without a value, BUDGET_DEFAULT."""
        if not self.config.get("budget"):
            return extra
        b = (budget if budget is not None else BUDGET_DEFAULT) / 100.0
        return np.concatenate([extra, np.full(extra.shape[:-1] + (1,), b, extra.dtype)], -1)

    @torch.no_grad()
    def predict(self, rows: np.ndarray, extra: np.ndarray, time_s: float, budget: float | None = None,
                keys: np.ndarray | None = None) -> np.ndarray:
        """One snapshot: rows (n, 11), extra (n, EXTRA). Role probabilities per unit (n, R)."""
        dev = next(self.net.parameters()).device
        n = min(len(rows), MAX_UNITS)
        extra = self.inputs(extra, budget, keys)
        logits = self.net(*tensors(rows[None, :n], extra[None, :n], np.zeros((1, n), bool),
                                   np.array([time_s], np.float32), dev))
        return torch.softmax(logits[0], -1).cpu().numpy()

    @torch.no_grad()
    def predict_move(self, rows: np.ndarray, extra: np.ndarray, time_s: float,
                     budget: float | None = None, keys: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Role probabilities (n, R) and moves (n, 2: tiles over the next half second, mirrored as the
        rows; unmirror with the snapshot's flip). A model without the where head: moves None."""
        dev = next(self.net.parameters()).device
        n = min(len(rows), MAX_UNITS)
        extra = self.inputs(extra, budget, keys)
        inputs = tensors(rows[None, :n], extra[None, :n], np.zeros((1, n), bool), np.array([time_s], np.float32), dev)
        if self.net.move is None:
            return torch.softmax(self.net(*inputs)[0], -1).cpu().numpy(), None
        logits, move = self.net(*inputs, with_move=True)
        return torch.softmax(logits[0], -1).cpu().numpy(), move[0].cpu().numpy()

    @torch.no_grad()
    def predict_offset(self, rows: np.ndarray, extra: np.ndarray, time_s: float, budget: float | None = None,
                       keys: np.ndarray | None = None):
        """The offset head's again targets (n, 2: tiles from each unit, mirrored as the rows), or None without one."""
        if getattr(self.net, "offset", None) is None:
            return None
        dev = next(self.net.parameters()).device
        n = min(len(rows), MAX_UNITS)
        extra = self.inputs(extra, budget, keys)
        inputs = tensors(rows[None, :n], extra[None, :n], np.zeros((1, n), bool), np.array([time_s], np.float32), dev)
        return self.net(*inputs, with_offset=True)[1][0].cpu().numpy()


BUDGET_DEFAULT = 100.0           # unit commands a minute: the pros' median (ingest/response_dataset.py budget)
BUDGET_S = 30


def order_budget(hi) -> float:
    """The player's unit commands a minute over the last BUDGET_S (the interface's own record, hi.commands): what a
    --budget role model is told it is spending."""
    since = hi.frame - BUDGET_S * 24
    return sum(1 for c in hi.commands if c[0] >= since) * 60.0 / BUDGET_S


def latest_trained_on(data_name: str) -> Path | None:
    """The newest role model (runs/roles/*/model.pt) trained on a dataset file of this name (e.g. response_T.npz,
    response_Z_30min.npz), or None."""
    for p in sorted(Path("runs/roles").glob("*/model.pt"), reverse=True):
        try:
            ck = torch.load(p, map_location="cpu", weights_only=True)
        except Exception:
            continue
        if str(ck.get("config", {}).get("data", "")).replace("\\", "/").endswith("/" + data_name) or                 str(ck.get("config", {}).get("data", "")) == data_name:
            return p
    return None

