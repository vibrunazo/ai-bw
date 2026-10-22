"""Which of a player's units belong on one control-group key, learned from the pros (train/groups.py; the learned
provider of gary/agent/grouping.py). Input: the fight as the role model sees it (gary/policy/fight.py's unit rows:
type, side, position relative to the fight mirrored, HP, shields, cooldown, order, carrying, completed); a small
transformer gives each unit an embedding, and a pair of the player's own units belongs on one key by their
embeddings' scaled dot product (a logit). Labels: ingest/key_labels.py (the key each pro unit was on).

The pros split one unit type over several keys: of the pairs of a pro's fighters 23.5% share a key, and "the same
type" names them with a precision of 41% (recall 92%; within 6 tiles 46% and 83%): the rule the scripted provider
groups by.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from gary.policy.fight import N_ORDERS, N_TYPES, ORDER, SIDE, TYPE, numeric


class GroupNet(nn.Module):
    def __init__(self, d: int = 96, layers: int = 2, heads: int = 4, emb: int = 48):
        super().__init__()
        self.type_emb = nn.Embedding(N_TYPES, 32)
        self.side_emb = nn.Embedding(3, 8)
        self.order_emb = nn.Embedding(N_ORDERS, 8)
        self.inp = nn.Linear(32 + 8 + 8 + 9, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d, emb)
        self.bias = nn.Parameter(torch.zeros(()))
        self.emb = emb

    def forward(self, kind, side, order, num, pad):
        """(B, N) long, num (B, N, 9), pad (B, N). Pair logits (B, N, N): unit i and j on one key."""
        h = self.inp(torch.cat([self.type_emb(kind), self.side_emb(side), self.order_emb(order), num], -1))
        e = self.out(self.body(h, src_key_padding_mask=pad))
        return e @ e.transpose(1, 2) / self.emb ** 0.5 + self.bias


def tensors(units: np.ndarray, pad: np.ndarray, time_s: np.ndarray, device):
    u = torch.as_tensor(units.astype(np.int64), device=device)
    return (u[..., TYPE].clamp(0, N_TYPES - 1), u[..., SIDE].clamp(0, 2), u[..., ORDER].clamp(0, N_ORDERS - 1),
            torch.as_tensor(numeric(units, time_s), device=device), torch.as_tensor(pad, device=device))


class GroupModel:
    def __init__(self, net: GroupNet, config: dict):
        self.net, self.config = net.eval(), config

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "GroupModel":
        ck = torch.load(path, map_location=device, weights_only=True)
        net = GroupNet(**ck["config"]["net"])
        net.load_state_dict(ck["state"])
        model = cls(net.to(device), ck["config"])
        model.path = str(path)
        return model

    def save(self, path: str | Path) -> None:
        torch.save({"state": self.net.state_dict(), "config": self.config}, path)

    @torch.no_grad()
    def affinity(self, rows: np.ndarray, time_s: float) -> np.ndarray:
        """(n, n): the chance two units are on one key, for one snapshot's rows."""
        dev = next(self.net.parameters()).device
        n = len(rows)
        logits = self.net(*tensors(rows[None], np.zeros((1, n), bool), np.array([time_s], np.float32), dev))
        return torch.sigmoid(logits[0]).cpu().numpy()


def partition(aff: np.ndarray, keys: int, at_least: int = 2, join: float = 0.5) -> list[list[int]]:
    """Units into at most keys groups by their affinity: each unit, the strongest first, joins the group it is most
    likely on one key with (its mean affinity to the group's units over join), or starts one while keys allow;
    groups smaller than at_least dropped."""
    n = len(aff)
    order = np.argsort(-(aff.sum(1) - aff.diagonal()))
    groups: list[list[int]] = []
    for i in order:
        best, score = None, join
        for gi, g in enumerate(groups):
            s = float(np.mean(aff[i, g]))
            if s > score and len(g) < 12:
                best, score = gi, s
        if best is not None:
            groups[best].append(int(i))
        elif len(groups) < keys:
            groups.append([int(i)])
    return [g for g in groups if len(g) >= at_least]
