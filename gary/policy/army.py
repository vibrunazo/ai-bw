"""Gary's army model: where to send an army group, and whether to move or attack.

Learned from pro replays (ingest/army_dataset.py, trained by train/army.py). Input: the map as the
player knows it, on a 16x16 grid mirrored so the player's main is top-left (own army, workers and
buildings; enemy ones visible now; enemy buildings ever seen; enemy army seen in the last 30 s;
the group being ordered), plus numbers (game time, resources, supply, the group's size, own army
composition, enemy units in view, and memory: the most of each enemy unit type seen at once in
the last 3 minutes, and the enemy army supply that adds up to). Output: a probability for each of
the 256 cells as the group's destination, move vs attack, and a fight estimate: for a given
destination and order, how the player's share of all army supply on the map changes over the next
45 s (positive: the order goes well). Gary uses it to hold back from attacks it expects to lose.

Pros keep fighting armies where they are about as often as they send them somewhere new, so the
model also says "stay": its best cell is then the one the group is already in.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

GRID, CELLS, CHANNELS = 16, 256, 9
RECENT_S = 30
MEMORY_S = 180                # how long enemy units count as "seen" in the numbers
VALUE_HORIZON_S = 45          # the fight estimate looks this far ahead
VALUE_SCALE = 10.0            # the value head predicts 10x the change in army share


@dataclass
class ArmySpec:
    race: str
    own_types: list[int]          # own unit types counted in the numbers
    seen_types: list[int]         # enemy unit types counted
    commit: bool = False          # the group's commitment in the numbers too (train/commit.py)
    losses: bool = False          # a 10th map channel: own losses in the last LOSS_S, where they died
    others: bool = False          # the other clusters' commitments in the numbers (others_inputs)
    static: bool = False          # a map channel after the losses': the enemy's ground static defense known (STATIC_SUPPLY)
    known: bool = False           # then: the enemy buildings known still standing (channel 6 never forgets one)
    unseen: bool = False          # then: seconds / UNSEEN_PER since own units were last in each cell (bases not seen lately)

    def _extra(self) -> list[str]:
        return [n for n in ("losses", "static", "known", "unseen") if getattr(self, n)]

    @property
    def channels(self) -> int:
        return CHANNELS + len(self._extra())

    def channel(self, name: str) -> int:
        """The index of an optional map channel (losses, static, known, unseen), in that order after the 9."""
        return CHANNELS + self._extra().index(name)

    @property
    def static_channel(self) -> int:
        return self.channel("static")

    @property
    def n_global(self) -> int:
        return 8 + len(self.own_types) + 2 * len(self.seen_types) + (COMMIT_INPUTS if self.commit else 0) + \
            (OTHER_INPUTS if self.others else 0)

    def to_dict(self) -> dict:
        return asdict(self)


LOSS_S = 10                       # own losses counted this long (ingest/army_dataset.py v3)
GONE_AFTER = 48                   # an own unit out of sight this long is lost (an SCV in a refinery, a drone
                                  # in an extractor, comes back sooner; a loaded bunker's marines are skipped)
COMMIT_INPUTS = 3                 # the group's commitment: engaging, moving (one-hot; none: both 0), its age
OTHER_INPUTS = 4                  # the other clusters: supply shares committed to engage and to move, the nearest's
                                  # distance and whether it engages (others_inputs)
COMMIT_AGE_S = 120                # an order older than this is no commitment (ingest/army_dataset.py COMMIT_S)
STATIC_SUPPLY = {146: 12, 162: 12, 125: 12}   # ground static defense, each worth this (supply x2): sunken, cannon, bunker
STATIC_GONE_PX = 7 * 32           # a known static defense (or building) is gone when own units are this close and don't see it
UNSEEN_PER = 4                    # the unseen channel's unit: seconds (255: 17 minutes or more)


def loss_value(unit_type: int) -> int:
    """A loss's weight on the losses channel: a worker 2, a fighter its supply x2, a building 4 (else 0)."""
    if is_worker(unit_type):
        return 2
    if is_building(unit_type):
        return 4
    return supply_x2(unit_type)


def path_dist(c: int, a: int, b: int) -> float:
    """Cells (Chebyshev) from cell c to the straight path from cell a to cell b: a group ordered from a to b
    is somewhere on it (its commitment follows it there; as ingest/army_dataset.py)."""
    cy, cx = divmod(c, GRID)
    ay, ax = divmod(a, GRID)
    by, bx = divmod(b, GRID)
    return min(max(abs(cy - (ay + (by - ay) * s)), abs(cx - (ax + (bx - ax) * s))) for s in np.linspace(0, 1, 9))


def commit_inputs(commit: np.ndarray, age_s: np.ndarray) -> np.ndarray:
    """(n, COMMIT_INPUTS): commit 0 none, 1 engage, 2 move; age in seconds."""
    return np.stack([(commit == 1).astype(np.float32), (commit == 2).astype(np.float32),
                     np.minimum(age_s, COMMIT_AGE_S).astype(np.float32) / 60.0], 1)


def others_inputs(centers: list[int], sups: list[float], commits: list[int], army: float) -> np.ndarray:
    """(n, OTHER_INPUTS) for n clusters seen at one moment (cells, supply x2, commitment 0 none / 1 engage / 2 move;
    army: the whole army's supply x2), each from the others: their supply committed to engage and to move, as
    shares of the army; the nearest one's distance (cells / GRID; 1 when there is none) and whether it engages.
    What a cluster's own view lacked to wait for the rest of the army (docs/PROGRESS.md, 2026-10-21)."""
    out = np.zeros((len(centers), OTHER_INPUTS), np.float32)
    for i, c in enumerate(centers):
        rest = [j for j in range(len(centers)) if j != i]
        out[i, 0] = sum(sups[j] for j in rest if commits[j] == 1) / max(army, 1.0)
        out[i, 1] = sum(sups[j] for j in rest if commits[j] == 2) / max(army, 1.0)
        out[i, 2] = 1.0
        if rest:
            cy, cx = divmod(c, GRID)
            dist = lambda j: max(abs(divmod(centers[j], GRID)[0] - cy), abs(divmod(centers[j], GRID)[1] - cx))
            j = min(rest, key=dist)
            out[i, 2] = dist(j) / GRID
            out[i, 3] = float(commits[j] == 1)
    return out


def encode_global(spec: ArmySpec, frame: np.ndarray, eco: np.ndarray, group_sup: np.ndarray,
                  units: np.ndarray, seen: np.ndarray, seen_mem: np.ndarray, own_army: np.ndarray,
                  enemy_army_mem: np.ndarray, commit: np.ndarray | None = None,
                  others: np.ndarray | None = None) -> np.ndarray:
    """(n, n_global) numbers. eco: minerals, gas, supply used x2, supply max x2; group_sup x2;
    units / seen / seen_mem: (n, 228) counts (seen_mem: the most seen at once in MEMORY_S);
    own_army, enemy_army_mem: army supply x2 (enemy: of seen_mem); commit: (n, COMMIT_INPUTS) when
    spec.commit (commit_inputs)."""
    parts = [(frame / (24 * 60 * 20.0))[:, None], np.minimum(eco[:, 0] / 1000.0, 5)[:, None],
             np.minimum(eco[:, 1] / 1000.0, 5)[:, None], (eco[:, 2] / 400.0)[:, None],
             (eco[:, 3] / 400.0)[:, None], (group_sup / 400.0)[:, None],
             (own_army / 400.0)[:, None], (enemy_army_mem / 400.0)[:, None],
             np.log1p(units[:, spec.own_types]), np.log1p(seen[:, spec.seen_types]),
             np.log1p(seen_mem[:, spec.seen_types])]
    if spec.commit:
        parts.append(commit if commit is not None else np.zeros((len(frame), COMMIT_INPUTS), np.float32))
    if spec.others:
        parts.append(others if others is not None else np.tile(np.array([[0, 0, 1, 0]], np.float32), (len(frame), 1)))
    return np.concatenate([p.astype(np.float32) for p in parts], axis=1)


class ArmyNet(nn.Module):
    def __init__(self, n_global: int, ch: int = 64, n_kind: int = 2, channels: int = CHANNELS):
        super().__init__()
        self.inp = nn.Conv2d(channels, ch, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(n_global, ch), nn.GELU(), nn.Linear(ch, ch))
        # dilated convolutions: every cell sees the whole 16x16 map after four layers
        self.convs = nn.ModuleList(nn.Conv2d(ch, ch, 3, padding=d, dilation=d) for d in (1, 2, 4, 8))
        self.mix = nn.Linear(2 * ch, ch)            # whole-map summary, fed back to every cell
        self.where = nn.Conv2d(ch, 1, 1)
        self.kind = nn.Linear(2 * ch, n_kind)        # move/attack; or keep/engage/move (train/commit.py)
        # fight estimate for a given destination and order: the whole map + the destination cell
        self.value = nn.Sequential(nn.Linear(3 * ch + 2, ch), nn.GELU(), nn.Linear(ch, 1))

    def forward(self, grid: torch.Tensor, glob: torch.Tensor, target: torch.Tensor | None = None,
                attack: torch.Tensor | None = None):
        """Destination logits (n, 256), move/attack logits (n, 2) and, given a target cell and
        attack (0/1) per row, the fight estimate (n,) (in VALUE_SCALE units)."""
        h = F.gelu(self.inp(torch.log1p(grid)) + self.glob(glob)[:, :, None, None])
        for conv in self.convs:
            h = h + F.gelu(conv(h))
        pooled = torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)
        h = h + self.mix(pooled)[:, :, None, None]
        pooled = torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)
        where, kind = self.where(h).flatten(1), self.kind(pooled)
        if target is None:
            return where, kind, None
        at = h.flatten(2)[torch.arange(len(target), device=h.device), :, target]
        order = F.one_hot(attack.long(), 2).float()
        return where, kind, self.value(torch.cat([pooled, at, order], 1)).squeeze(-1)


class ArmyModel:
    def __init__(self, spec: ArmySpec, net: ArmyNet, config: dict):
        self.spec, self.net, self.config = spec, net.eval(), config

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "ArmyModel":
        ck = torch.load(path, map_location=device, weights_only=True)
        spec = ArmySpec(**ck["spec"])
        net = ArmyNet(spec.n_global, **ck["config"]["net"])
        net.load_state_dict(ck["state"])
        model = cls(spec, net.to(device), ck["config"])
        model.path = str(path)
        return model

    def save(self, path: str | Path) -> None:
        torch.save({"spec": self.spec.to_dict(), "state": self.net.state_dict(), "config": self.config}, path)

    @torch.no_grad()
    def predict(self, grid: np.ndarray, glob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(n, 256) destination probabilities and (n, 2) move/attack probabilities."""
        dev = next(self.net.parameters()).device
        where, kind, _ = self.net(torch.as_tensor(grid, dtype=torch.float32, device=dev).view(-1, self.spec.channels, GRID, GRID),
                                  torch.as_tensor(glob, dtype=torch.float32, device=dev))
        return torch.softmax(where, -1).cpu().numpy(), torch.softmax(kind, -1).cpu().numpy()

    @torch.no_grad()
    def fight(self, grid: np.ndarray, glob: np.ndarray, target: int, attack: bool) -> float:
        """Expected change in the player's share of all army supply over the next 45 s if the
        group is sent to `target` (moving, or attack-moving)."""
        dev = next(self.net.parameters()).device
        _, _, v = self.net(torch.as_tensor(grid, dtype=torch.float32, device=dev).view(-1, self.spec.channels, GRID, GRID),
                           torch.as_tensor(glob, dtype=torch.float32, device=dev),
                           torch.tensor([target], device=dev), torch.tensor([int(attack)], device=dev))
        return float(v[0]) / VALUE_SCALE


class ArmyTracker:
    """Builds the army model's inputs during play from gary.interface observations, the way
    ingest/army_dataset.py built them from replays (same grid, same mirroring, same memory)."""

    def __init__(self, spec: ArmySpec, slot: int, map_size: tuple[int, int], main: tuple[int, int]):
        self.spec, self.slot = spec, slot
        self.mw, self.mh = map_size
        self.fx, self.fy = main[0] > self.mw / 2, main[1] > self.mh / 2
        self.seen_buildings = np.zeros(CELLS, np.float32)
        self.recent: list[tuple[int, np.ndarray]] = []      # (frame, enemy army grid)
        self.seen_hist: list[tuple[int, np.ndarray]] = []   # (frame, enemy unit counts by type)
        self.own_last: dict[int, tuple[int, int]] = {}         # spec.losses: own tag -> (type, cell) last seen
        self.missing: dict[int, tuple[int, int, int]] = {}     # own tags out of sight: (frame, cell, value)
        self.lost: list[tuple[int, int, int]] = []             # (frame, cell, value) of own losses
        self.hidden: set[int] = set()                          # own tags not lost when unseen (in a bunker)
        self.statics: dict[int, tuple[int, int, int]] = {}     # spec.static: enemy static defense tag -> (x, y, type)
        self.buildings: dict[int, tuple[int, int]] = {}        # spec.known: enemy building tag -> (x, y), still standing
        self.visited = np.zeros(CELLS, np.int64)               # spec.unseen: the frame own units were last in each cell

    def cell(self, x: float, y: float) -> int:
        gx = min(GRID - 1, max(0, int(x * GRID // self.mw)))
        gy = min(GRID - 1, max(0, int(y * GRID // self.mh)))
        gx = GRID - 1 - gx if self.fx else gx
        gy = GRID - 1 - gy if self.fy else gy
        return gy * GRID + gx

    def center(self, cell: int) -> tuple[int, int]:
        """Map pixel at the middle of a (mirrored) cell."""
        gy, gx = divmod(cell, GRID)
        gx = GRID - 1 - gx if self.fx else gx
        gy = GRID - 1 - gy if self.fy else gy
        return int((gx + 0.5) * self.mw / GRID), int((gy + 0.5) * self.mh / GRID)

    def _now(self, obs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Channels 0-5 (own / visible enemy army, workers, buildings), own and visible enemy
        unit counts by type."""
        g = np.zeros((self.spec.channels, CELLS), np.float32)
        units = np.zeros(228, np.float32)
        seen = np.zeros(228, np.float32)
        for u in obs["units"]:
            t = u["type"]
            if u["owner"] == 11 or not 0 <= t < 228:
                continue
            if is_building(t):
                cat, val = 2, 1
            elif is_worker(t):
                cat, val = 1, 1
            elif supply_x2(t) > 0:
                cat, val = 0, supply_x2(t)
            else:
                cat = None
            mine = u["owner"] == self.slot
            if mine:
                units[t] += 1
            else:
                seen[t] += 1
            if cat is not None:
                g[cat + (0 if mine else 3), self.cell(u["x"], u["y"])] += val
        return g, units, seen

    def remember(self, obs: dict) -> None:
        """Update the memory (enemy buildings ever seen, enemy army seen lately). Call it every
        time the player looks, not only when asking the model: a scout's glimpse counts."""
        if self.recent and self.recent[-1][0] == obs["frame"]:
            return
        g, _, seen = self._now(obs)
        f0 = obs["frame"]
        if self.spec.losses:
            self._losses(obs)
        if self.spec.static or self.spec.known:
            self._statics(obs)
        if self.spec.unseen:
            for u in obs["units"]:
                if u["owner"] == self.slot:
                    self.visited[self.cell(u["x"], u["y"])] = f0
        self.seen_buildings = np.maximum(self.seen_buildings, g[5])
        self.recent = [(f, a) for f, a in self.recent if f0 - f <= RECENT_S * 24] + [(f0, g[3])]
        self.seen_hist = [(f, a) for f, a in self.seen_hist if f0 - f <= MEMORY_S * 24] + [(f0, seen)]

    def _losses(self, obs: dict) -> None:
        """Own units gone since the last look: lost where last seen, once out of sight GONE_AFTER (an SCV
        in a refinery comes back); units in self.hidden (loaded in a bunker) aren't."""
        f0 = obs["frame"]
        now = {u["tag"]: (u["type"], self.cell(u["x"], u["y"])) for u in obs["units"] if u["owner"] == self.slot}
        for tag, (t, c) in self.own_last.items():
            if tag not in now and tag not in self.hidden and tag not in self.missing:
                self.missing[tag] = (f0, c, loss_value(t))
        for tag in [t for t in self.missing if t in now]:
            del self.missing[tag]                      # (back: it was only out of sight)
        for tag, (f, c, v) in list(self.missing.items()):
            if f0 - f >= GONE_AFTER:
                del self.missing[tag]
                if v:
                    self.lost.append((f, c, v))
        self.lost = [x for x in self.lost if f0 - x[0] <= LOSS_S * 24 + GONE_AFTER]
        self.own_last = {**{t: x for t, x in self.own_last.items() if t in self.missing}, **now}

    def _statics(self, obs: dict) -> None:
        """Enemy buildings seen (and the completed ground static defense among them), remembered until own units stand
        near the place and don't see it (the dataset: from the first look at its cell until it dies)."""
        now = set()
        for u in obs["units"]:
            if u["owner"] in (self.slot, 11) or not is_building(u["type"]):
                continue
            now.add(u["tag"])
            self.buildings[u["tag"]] = (u["x"], u["y"])
            if u["type"] in STATIC_SUPPLY and u.get("completed", True):
                self.statics[u["tag"]] = (u["x"], u["y"], u["type"])
        at = [(u["x"], u["y"]) for u in obs["units"] if u["owner"] == self.slot]
        for memory in (self.statics, self.buildings):
            for tag, (x, y, *_) in list(memory.items()):
                if tag not in now and any(math.dist((x, y), q) < STATIC_GONE_PX for q in at):
                    del memory[tag]

    def inputs(self, obs: dict, group: list[dict], commit: tuple[int, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """grid (9, 256) and numbers for ordering `group` (own units) now; commit: the group's commitment
        (0 none, 1 engage, 2 move) and its age in seconds, for a model with spec.commit."""
        self.remember(obs)
        g, units, seen = self._now(obs)
        g[6] = self.seen_buildings
        g[7] = np.max([a for _, a in self.recent], axis=0)
        if self.spec.losses:                   # (where its units died in the last LOSS_S, as the pros' data)
            for f, c, v in self.lost:
                if obs["frame"] - f <= LOSS_S * 24 + GONE_AFTER:
                    g[self.spec.channel("losses"), c] += v
        if self.spec.static:
            for x, y, t in self.statics.values():
                g[self.spec.static_channel, self.cell(x, y)] += STATIC_SUPPLY[t]
        if self.spec.known:
            for x, y in self.buildings.values():
                g[self.spec.channel("known"), self.cell(x, y)] += 1
        if self.spec.unseen:
            g[self.spec.channel("unseen")] = (obs["frame"] - self.visited) / (24 * UNSEEN_PER)
        group_sup = 0
        for u in group:
            s = supply_x2(u["type"])
            g[8, self.cell(u["x"], u["y"])] += s
            group_sup += s
        g = np.minimum(g, 255)
        me = obs["me"]
        eco = np.array([[me["minerals"], me["gas"], me["supply_used"] * 2, me["supply_max"] * 2]], np.float32)
        seen_mem = np.max([a for _, a in self.seen_hist], axis=0)
        glob = encode_global(self.spec, np.array([obs["frame"]], np.float32), eco,
                             np.array([group_sup], np.float32), units[None], seen[None], seen_mem[None],
                             np.array([units @ ARMY_SUPPLY], np.float32),
                             np.array([seen_mem @ ARMY_SUPPLY], np.float32),
                             commit_inputs(np.array([commit[0] if commit else 0]),
                                           np.array([commit[1] if commit else COMMIT_AGE_S])) if self.spec.commit else None)
        return g[None], glob


# Supply x2 of fighting units for all races (what the replay data's grid weighs units by), and
# which unit types are workers or buildings.
SUPPLY_X2 = {
    0: 2, 1: 2, 2: 4, 3: 4, 5: 4, 30: 4, 8: 4, 9: 4, 11: 4, 12: 12, 32: 2, 34: 2, 58: 6,       # Terran
    37: 1, 38: 2, 39: 8, 43: 4, 44: 4, 45: 4, 46: 4, 47: 1, 50: 2, 62: 4, 103: 4,             # Zerg
    60: 4, 61: 4, 63: 8, 65: 4, 66: 4, 67: 4, 68: 8, 69: 4, 70: 6, 71: 8, 72: 12, 83: 8, 84: 2,  # Protoss
}
WORKERS = {7, 41, 64}


def supply_x2(unit_type: int) -> int:
    return SUPPLY_X2.get(unit_type, 0)


def is_worker(unit_type: int) -> bool:
    return unit_type in WORKERS


def is_building(unit_type: int) -> bool:
    return 106 <= unit_type <= 175


ARMY_SUPPLY = np.zeros(228, np.float32)      # supply x2 of fighting units by type (workers: 0)
for _t, _s in SUPPLY_X2.items():
    ARMY_SUPPLY[_t] = _s
