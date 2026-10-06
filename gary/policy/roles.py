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
MAX_UNITS = 64
EXTRA = 3                            # motion x, y over the last half second (tiles), enemies targeting it
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


class RoleNet(nn.Module):
    def __init__(self, d: int = 128, layers: int = 3, heads: int = 4):
        super().__init__()
        self.type_emb = nn.Embedding(N_TYPES, 48)
        self.side_emb = nn.Embedding(3, 8)
        self.order_emb = nn.Embedding(N_ORDERS, 16)
        self.inp = nn.Linear(48 + 8 + 16 + 9 + EXTRA, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.role = nn.Linear(d, len(ROLES))

    def forward(self, kind, side, order, num, extra, pad):
        """kind, side, order (B, N) long; num (B, N, 9); extra (B, N, EXTRA); pad (B, N) True where
        there is no unit. Returns role logits (B, N, R)."""
        h = self.inp(torch.cat([self.type_emb(kind), self.side_emb(side), self.order_emb(order), num, extra], -1))
        h = self.body(h, src_key_padding_mask=pad)
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
