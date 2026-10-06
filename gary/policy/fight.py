"""Gary's fight model: what each of its units does in a skirmish (the architecture's Policy).

Learned from pro replays (ingest/fight_dataset.py, trained by train/fight.py). Input: the units
around a fight as the player sees them (own, visible enemy, mineral fields): type, side,
position relative to the fight (mirrored so the player's main is top-left), HP, shields, weapon
cooldown, current order and its target, carrying, completed. A small transformer lets every unit
see every other. Output, per own unit: an action (ACTIONS: nothing, move, attack-move, attack a
unit, gather, right-click an own unit, stop, hold, stim, return cargo, other), the unit it
targets (a pointer into the set) and, for moves, where to (relative to the unit).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ACTIONS = ["none", "move", "attack_move", "attack_unit", "gather", "own_unit", "stop", "hold",
           "stim", "return_cargo", "other"]
MAX_UNITS = 48
N_TYPES, N_ORDERS = 256, 256
POS_SCALE = 512.0               # pixels
DEST_SCALE = 512.0
# unit row columns (ingest/fight_dataset.py): type, side, x, y, hp, shields, cooldown, order,
# order target row, carrying, completed
TYPE, SIDE, X, Y, HP, SH, CD, ORDER, OTARGET, CARRY, DONE = range(11)


def numeric(units: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    """(..., 9) float features from (..., 11) unit rows and the game time per snapshot (...)."""
    u = units.astype(np.float32)
    t = np.broadcast_to(time_s[..., None], u.shape[:-1])
    return np.stack([u[..., X] / POS_SCALE, u[..., Y] / POS_SCALE, np.minimum(u[..., HP], 2000) / 200.0,
                     np.minimum(u[..., SH], 1000) / 200.0, u[..., CD] / 30.0, (u[..., OTARGET] >= 0).astype(np.float32),
                     (u[..., CARRY] > 0).astype(np.float32), u[..., DONE], t / 600.0], axis=-1)


class FightNet(nn.Module):
    def __init__(self, d: int = 128, layers: int = 3, heads: int = 4):
        super().__init__()
        self.type_emb = nn.Embedding(N_TYPES, 48)
        self.side_emb = nn.Embedding(3, 8)
        self.order_emb = nn.Embedding(N_ORDERS, 16)
        self.inp = nn.Linear(48 + 8 + 16 + 9, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.act = nn.Linear(d, len(ACTIONS))
        self.query = nn.Linear(d, d)                # pointer to the target unit
        self.key = nn.Linear(d, d)
        self.dest = nn.Linear(d, 2)                 # where to move, relative to the unit

    def forward(self, kind: torch.Tensor, side: torch.Tensor, order: torch.Tensor, num: torch.Tensor,
                pad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """kind, side, order (B, N) long; num (B, N, 9); pad (B, N) True where there is no unit.
        Returns action logits (B, N, A), target logits (B, N, N), destinations (B, N, 2)."""
        h = self.inp(torch.cat([self.type_emb(kind), self.side_emb(side), self.order_emb(order), num], -1))
        h = self.body(h, src_key_padding_mask=pad)
        target = torch.einsum("bnd,bmd->bnm", self.query(h), self.key(h)) / h.shape[-1] ** 0.5
        target = target.masked_fill(pad[:, None, :], -1e9)
        return self.act(h), target, self.dest(h)


def tensors(units: np.ndarray, pad: np.ndarray, time_s: np.ndarray, device) -> tuple[torch.Tensor, ...]:
    """Model inputs from padded unit rows (B, N, 11), the padding mask (B, N) and time (B,)."""
    u = torch.as_tensor(units.astype(np.int64), device=device)
    return (u[..., TYPE].clamp(0, N_TYPES - 1), u[..., SIDE].clamp(0, 2), u[..., ORDER].clamp(0, N_ORDERS - 1),
            torch.as_tensor(numeric(units, time_s), device=device), torch.as_tensor(pad, device=device))


class FightModel:
    def __init__(self, net: FightNet, config: dict):
        self.net, self.config = net.eval(), config

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "FightModel":
        ck = torch.load(path, map_location=device, weights_only=True)
        net = FightNet(**ck["config"]["net"])
        net.load_state_dict(ck["state"])
        model = cls(net.to(device), ck["config"])
        model.path = str(path)
        return model

    def save(self, path: str | Path) -> None:
        torch.save({"state": self.net.state_dict(), "config": self.config}, path)

    @torch.no_grad()
    def predict(self, units: np.ndarray, time_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One snapshot: units (n, 11). Per unit: action probabilities (n, A), target probabilities
        (n, n), destination (n, 2) in pixels."""
        dev = next(self.net.parameters()).device
        n = min(len(units), MAX_UNITS)
        u = units[None, :n]
        pad = np.zeros((1, n), bool)
        act, target, dest = self.net(*tensors(u, pad, np.array([time_s], np.float32), dev))
        return (torch.softmax(act[0], -1).cpu().numpy(), torch.softmax(target[0], -1).cpu().numpy(),
                (dest[0] * DEST_SCALE).cpu().numpy())


FIGHT_NEAR = 8 * 32             # enemy fighters this close to own units make a fight
FIGHT_AROUND = 12 * 32          # units this close to the fight's center are in the snapshot
FIGHT_OWN = 24 * 32             # ...and the player's own units this close (fight set v2)


def snapshot(obs: dict, slot: int, flip: tuple[bool, bool], supply_x2, is_worker, is_building,
             own_radius: int = FIGHT_AROUND, max_units: int = MAX_UNITS, near: int = None):
    """The fight around the player's units right now, built from an observation the way
    resim --fights builds it from replays: (unit rows (n, 11), their tags, the center (x, y)),
    or None when no visible enemy fighter is near. Rows are mirrored by flip (the player's main
    top-left) and sorted nearest first. own_radius: how far the player's own units are taken
    (FIGHT_AROUND as in fight set v1, FIGHT_OWN as in v2)."""
    mine = [u for u in obs["units"] if u["owner"] == slot]
    enemy = [u for u in obs["units"] if u["owner"] not in (slot, 11)]
    fighters = [e for e in enemy if supply_x2(e["type"]) > 0 and not is_worker(e["type"]) and not is_building(e["type"])]
    near = near or FIGHT_NEAR                # (wider for a fight layer that watches threats coming)
    engaged = [e for e in fighters
               if any((e["x"] - m["x"]) ** 2 + (e["y"] - m["y"]) ** 2 <= near ** 2 for m in mine)]
    if not engaged:
        return None
    cx = sum(e["x"] for e in engaged) / len(engaged)
    cy = sum(e["y"] for e in engaged) / len(engaged)
    around = [u for u in obs["units"]
              if (u["x"] - cx) ** 2 + (u["y"] - cy) ** 2 <= (own_radius if u["owner"] == slot else max(FIGHT_AROUND, near)) ** 2
              and (u["owner"] != 11 or u["type"] in (176, 177, 178, 188))]
    around.sort(key=lambda u: (u["x"] - cx) ** 2 + (u["y"] - cy) ** 2)
    around = around[:max_units]
    if not any(u["owner"] == slot for u in around):
        return None                              # the engaged enemies are spread out: no one fight
    row = {u["tag"]: i for i, u in enumerate(around)}
    fx, fy = flip
    rows = np.zeros((len(around), 11), np.int16)
    for i, u in enumerate(around):
        rx, ry = u["x"] - cx, u["y"] - cy
        side = 1 if u["owner"] == slot else 2 if u["owner"] == 11 else 0
        rows[i] = (u["type"], side, -rx if fx else rx, -ry if fy else ry, min(u.get("hp") or 0, 32767),
                   u.get("shields") or 0, u.get("cooldown", 0), u.get("order", 0),
                   row.get(u.get("order_target", 0), -1), u.get("carrying", 0), u.get("completed", 1))
    return rows, [u["tag"] for u in around], (cx, cy)
