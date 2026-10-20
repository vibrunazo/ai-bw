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

ROLES = ["fire", "close_in", "pull_back", "retreat", "reposition", "stand", "other"]
# the decision model: a new role when the unit's changes, else keep; and roles the hand rules use or
# the pros' units have no label for: post (the hand rules' post by the Command Center), join (to
# Gary's biggest bunch of fighters), hold (hold position)
DECISIONS = ROLES + ["keep", "post", "join", "hold"]
MAX_UNITS = 64
TERRAIN = 8                          # the ground under each unit and the player's own bases (Ground)
EXTRA = 6 + TERRAIN                  # motion x, y over the last half second (tiles), enemies targeting it;
                                     # the unit's age, its distance to Gary's biggest bunch of fighters,
                                     # that bunch's size (gary/agent/roles.py); TERRAIN (models before
                                     # have 3 or 6: they read the first n_extra)
AGE_SCALE = 60.0                     # seconds
PACK_DIST_SCALE = 320.0              # pixels
PACK_SIZE_SCALE = 12.0
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
                 move: bool = False):
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

    def forward(self, kind, side, order, num, extra, pad, with_move: bool = False):
        """kind, side, order (B, N) long; num (B, N, 9); extra (B, N, EXTRA); pad (B, N) True where
        there is no unit. Returns role logits (B, N, R); with_move, also the moves (B, N, 2)."""
        h = self.inp(torch.cat([self.type_emb(kind), self.side_emb(side), self.order_emb(order), num,
                                extra[..., :self.n_extra]], -1))
        h = self.body(h, src_key_padding_mask=pad)
        if with_move:
            return self.role(h), self.move(h)
        return self.role(h)


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

    @torch.no_grad()
    def predict(self, rows: np.ndarray, extra: np.ndarray, time_s: float) -> np.ndarray:
        """One snapshot: rows (n, 11), extra (n, EXTRA). Role probabilities per unit (n, R)."""
        dev = next(self.net.parameters()).device
        n = min(len(rows), MAX_UNITS)
        logits = self.net(*tensors(rows[None, :n], extra[None, :n], np.zeros((1, n), bool),
                                   np.array([time_s], np.float32), dev))
        return torch.softmax(logits[0], -1).cpu().numpy()

    @torch.no_grad()
    def predict_move(self, rows: np.ndarray, extra: np.ndarray, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Role probabilities (n, R) and moves (n, 2: tiles over the next half second, mirrored as the
        rows; unmirror with the snapshot's flip). A model without the where head: moves None."""
        dev = next(self.net.parameters()).device
        n = min(len(rows), MAX_UNITS)
        inputs = tensors(rows[None, :n], extra[None, :n], np.zeros((1, n), bool), np.array([time_s], np.float32), dev)
        if self.net.move is None:
            return torch.softmax(self.net(*inputs)[0], -1).cpu().numpy(), None
        logits, move = self.net(*inputs, with_move=True)
        return torch.softmax(logits[0], -1).cpu().numpy(), move[0].cpu().numpy()


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

