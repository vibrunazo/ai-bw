"""Gary's macro model: what to produce next, and how soon.

Learned from pro replays (ingest/macro_dataset.py, trained by train/macro.py). Given a player's
state at a moment, as that player knew it, it predicts their next production decision (train,
morph, build, expand, research, upgrade, with the unit/tech/upgrade id) and the seconds until it.

Inputs (all from the player's own point of view):
  game time; minerals, gas, supply used and max; own units by type (all / completed);
  enemy units visible now, and the most of each type seen so far (what was scouted);
  every production decision made so far (counts per decision type), seconds since the last one;
  the player's build style (taxonomy cluster), when known, for steering.

One model per race. The feature layout (FeatureSpec) is saved with the weights, so a checkpoint
encodes states the same way in training and in play.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

ACT_NAMES = ["train", "morph", "bmorph", "build", "research", "upgrade", "expand"]
OTHER = (-1, -1)                 # decisions too rare to have their own class


@dataclass
class FeatureSpec:
    race: str
    own_types: list[int]                     # unit type ids this race uses
    seen_types: list[int]                    # enemy unit type ids that get seen
    decisions: list[tuple[int, int]]         # class vocabulary: (act, id); the last is OTHER
    styles: list[int] = field(default_factory=list)   # build-style clusters (unknown = extra slot)

    def __post_init__(self):
        self.decisions = [tuple(d) for d in self.decisions]
        self._index = {d: i for i, d in enumerate(self.decisions)}

    @property
    def size(self) -> int:
        return (7 + 2 * len(self.own_types) + 2 * len(self.seen_types) + len(self.decisions)
                + len(self.styles) + 1)

    def decision_index(self, act: int, uid: int) -> int:
        return self._index.get((act, uid), self._index[OTHER])

    def to_dict(self) -> dict:
        d = asdict(self)
        d["decisions"] = [list(x) for x in self.decisions]
        return d


def encode(spec: FeatureSpec, frame: np.ndarray, eco: np.ndarray, units: np.ndarray,
           seen_now: np.ndarray, seen_max: np.ndarray, decided: np.ndarray,
           since_last_s: np.ndarray, style: int | None) -> np.ndarray:
    """Features for n moments of one player. Shapes: frame (n,), eco (n, 4) [minerals, gas,
    supply used x2, supply max x2], units (n, 228, 2), seen_now / seen_max (n, 228),
    decided (n, len(spec.decisions)) counts so far, since_last_s (n,)."""
    n = len(frame)
    used, cap = eco[:, 2] / 400.0, eco[:, 3] / 400.0
    style_onehot = np.zeros((n, len(spec.styles) + 1), dtype=np.float32)
    style_onehot[:, spec.styles.index(style) if style in spec.styles else len(spec.styles)] = 1
    parts = [
        (frame / (24 * 60 * 20.0))[:, None],
        np.minimum(eco[:, 0] / 1000.0, 5)[:, None],
        np.minimum(eco[:, 1] / 1000.0, 5)[:, None],
        used[:, None], cap[:, None], (cap - used)[:, None],
        np.minimum(since_last_s / 60.0, 3)[:, None],
        np.log1p(units[:, spec.own_types, 0]), np.log1p(units[:, spec.own_types, 1]),
        np.log1p(seen_now[:, spec.seen_types]), np.log1p(seen_max[:, spec.seen_types]),
        np.log1p(decided),
        style_onehot,
    ]
    x = np.concatenate([p.astype(np.float32) for p in parts], axis=1)
    assert x.shape[1] == spec.size, (x.shape, spec.size)
    return x


class MacroNet(nn.Module):
    def __init__(self, n_in: int, n_classes: int, hidden: int = 512, layers: int = 3, dropout: float = 0.1):
        super().__init__()
        blocks, d = [], n_in
        for _ in range(layers):
            blocks += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        self.body = nn.Sequential(*blocks)
        self.what = nn.Linear(hidden, n_classes)     # which decision comes next
        self.when = nn.Linear(hidden, 1)             # log(1 + seconds until it)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.what(h), self.when(h).squeeze(-1)


class MacroModel:
    """A trained macro model for one race: encode a state, predict the next decision."""

    def __init__(self, spec: FeatureSpec, net: MacroNet, config: dict, caps: dict | None = None):
        self.spec, self.net, self.config = spec, net.eval(), config
        self.caps = caps or {}      # unit type -> most pros had, per game minute (caps.json)

    def cap(self, unit_type: int, frame: int) -> int | None:
        """How many of this unit type pros had by this point of the game (90th percentile), or
        None if unknown."""
        c = self.caps.get(unit_type)
        return None if c is None else c[min(frame // (24 * 60), len(c) - 1)]

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "MacroModel":
        ck = torch.load(path, map_location=device, weights_only=True)
        spec = FeatureSpec(**ck["spec"])
        net = MacroNet(spec.size, len(spec.decisions), **ck["config"]["net"])
        net.load_state_dict(ck["state"])
        caps_path = Path(path).with_name("caps.json")
        caps = {int(k): v for k, v in json.loads(caps_path.read_text()).items()} if caps_path.exists() else None
        model = cls(spec, net.to(device), ck["config"], caps)
        model.path = str(path)
        return model

    def save(self, path: str | Path) -> None:
        torch.save({"spec": self.spec.to_dict(), "state": self.net.state_dict(), "config": self.config}, path)

    @torch.no_grad()
    def probs(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Probability of each decision class (spec.decisions order) and seconds until, per row."""
        dev = next(self.net.parameters()).device
        logits, when = self.net(torch.as_tensor(x, dtype=torch.float32, device=dev))
        return torch.softmax(logits, -1).cpu().numpy(), torch.expm1(when).clamp(min=0).cpu().numpy()

    @torch.no_grad()
    def predict(self, x: np.ndarray, top: int = 3) -> list[dict]:
        """For each row of features: the top decisions with probabilities, and seconds until."""
        dev = next(self.net.parameters()).device
        logits, when = self.net(torch.as_tensor(x, dtype=torch.float32, device=dev))
        probs = torch.softmax(logits, -1)
        p, idx = probs.topk(top, -1)
        secs = torch.expm1(when).clamp(min=0)
        out = []
        for row in range(len(x)):
            out.append({"next": [(*self.spec.decisions[int(i)], float(q)) for q, i in zip(p[row], idx[row])],
                        "seconds": float(secs[row])})
        return out



def pro_history(race: str, style: int | None = None) -> np.ndarray | None:
    """The pros' mean decision counts per minute for a race (data/interim/macro/v1/history_mean_<race>.npy), or
    for one of its build styles (history_mean_<race>_styles.npy, the race's mean where a style has too few
    players: an average over styles told a 3-hatch player it had made a 9-pool player's units). None if not
    computed."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ingest"))
    try:
        from inventory import data_root
        root = data_root() / "interim" / "macro" / "v1"
        if style is not None and (root / f"history_mean_{race}_styles.npy").exists():
            t = np.load(root / f"history_mean_{race}_styles.npy")
            if 0 <= style < len(t):
                return t[style]
        return np.load(root / f"history_mean_{race}.npy")
    except Exception:
        return None


class MacroTracker:
    """Builds the model's features during play, from what the player sees (gary.interface
    observations), the way ingest/macro_dataset.py built them from replays.

    Units still in production count as "all" but not "completed" (the game counts a unit once its
    production starts); observations don't list them, so the bot reports them (in_production).
    Enemy units count only while visible; the most of each type ever seen at once is kept, which
    is what the replay data's seen_max holds."""

    def __init__(self, spec: FeatureSpec, slot: int, style: int | None = None):
        self.spec, self.slot, self.style = spec, slot, style
        self.seen_max = np.zeros(228, dtype=np.float32)
        self.decided = np.zeros(len(spec.decisions), dtype=np.float32)
        self.last_decision_frame = 0
        # history: None, or the pros' mean decision counts per minute ((31, decisions): data/interim/macro/
        # history_mean_<race>.npy): the model then sees a pro's typical past, not this player's own. Gary-Z's own
        # history fed back on itself: its drone odds rested on "its decisions so far" at +0.36 (the records'
        # attribution, round g12), 60-70% drones at 6:00-7:30 where pros decide drones 17-24% of the time
        self.history: np.ndarray | None = None
        self.history_weight = 1.0        # the pros' share in the history the model sees (1: theirs alone; 0.5: half its own)
        self.history_until = 0           # minutes over which that share fades to 0 (0: constant). With the pros' history
                                         # all game Gary-Z over-droned late (53 workers at 15-20 min, pros 35: the pros'
                                         # typical past kept telling the model to drone past what it had), with its
                                         # own early (the loop) it had no army at 6:00; round g16 against g13

    def note_decision(self, act: int, uid: int, frame: int) -> None:
        self.decided[self.spec.decision_index(act, uid)] += 1
        self.last_decision_frame = frame

    def unnote_decision(self, act: int, uid: int) -> None:
        """A decision that didn't happen after all (e.g. no room to place the building)."""
        i = self.spec.decision_index(act, uid)
        self.decided[i] = max(0, self.decided[i] - 1)

    def features(self, obs: dict, in_production: dict[int, int]) -> np.ndarray:
        units = np.zeros((1, 228, 2), dtype=np.float32)
        seen = np.zeros(228, dtype=np.float32)
        for u in obs["units"]:
            t = u["type"]
            if not 0 <= t < 228:
                continue
            if u["owner"] == self.slot:
                units[0, t, 0] += 1
                units[0, t, 1] += 1 if u["completed"] else 0
            elif u["owner"] != 11:
                seen[t] += 1
        for t, n in in_production.items():
            units[0, t, 0] += n
        self.seen_max = np.maximum(self.seen_max, seen)
        me = obs["me"]
        eco = np.array([[me["minerals"], me["gas"], me["supply_used"] * 2, me["supply_max"] * 2]], dtype=np.float32)
        frame = np.array([obs["frame"]], dtype=np.float32)
        decided = self.decided
        if self.history is not None:
            pro = self.history[min(obs["frame"] // 1440, len(self.history) - 1)]
            w = self.history_weight * (max(0.0, 1.0 - obs["frame"] / (self.history_until * 1440)) if self.history_until else 1.0)
            decided = w * pro + (1.0 - w) * self.decided
        return encode(self.spec, frame, eco, units, seen[None], self.seen_max[None], decided[None],
                      np.array([(obs["frame"] - self.last_decision_frame) / 24.0], dtype=np.float32), self.style)
