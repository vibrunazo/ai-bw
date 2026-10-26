"""Gary's macro model: what to produce next, and how soon.

Learned from pro replays (ingest/macro_dataset.py, trained by train/macro.py). Given a player's
state at a moment, as that player knew it, it predicts their next production decision (train,
morph, build, expand, research, upgrade, with the unit/tech/upgrade id) and the seconds until it.

Inputs (all from the player's own point of view):
  game time; minerals, gas, supply used and max; own units by type (all / completed);
  enemy units visible now, and the most of each type seen so far (what was scouted);
  every production decision made so far (counts per decision type), seconds since the last one;
  the player's build style (taxonomy cluster), when known, for steering;
  the game's year and map (the map's data hash, one of the maps with enough games, else unknown), when
  the spec has them: the meta moved between the 1.16 era and Remastered (docs/DATA_ERAS.md), and
  builds differ by map. In play the year is now and the map is the game's (MapInfo.hash).

One model per race. The feature layout (FeatureSpec) is saved with the weights, so a checkpoint
encodes states the same way in training and in play.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

ACT_NAMES = ["train", "morph", "bmorph", "build", "research", "upgrade", "expand"]
OTHER = (-1, -1)                 # decisions too rare to have their own class
BUILD_ACT, EXPAND_ACT = 3, 6     # (ingest/macro_dataset.py ACTS["build"], ACT_EXPAND)
PENDING_FOR = 60 * 24            # builtcount: a decided building not started this long after didn't happen
HALL_APART = 10 * 32             # builtcount: a town hall this far from all of the player's others is an expansion


@dataclass
class FeatureSpec:
    race: str
    own_types: list[int]                     # unit type ids this race uses
    seen_types: list[int]                    # enemy unit type ids that get seen
    decisions: list[tuple[int, int]]         # class vocabulary: (act, id); the last is OTHER
    styles: list[int] = field(default_factory=list)   # build-style clusters (unknown = extra slot)
    maps: list[str] = field(default_factory=list)     # map data hashes with their own slot (unknown = extra)
    year: bool = False                                # the game's year as an input
    matchups: list[str] = field(default_factory=list)  # one model for the race's matchups: which one (one-hot)

    def __post_init__(self):
        self.decisions = [tuple(d) for d in self.decisions]
        self._index = {d: i for i, d in enumerate(self.decisions)}

    @property
    def size(self) -> int:
        return (7 + 2 * len(self.own_types) + 2 * len(self.seen_types) + len(self.decisions)
                + len(self.styles) + 1 + (len(self.maps) + 1 if self.maps else 0) + (1 if self.year else 0)
                + len(self.matchups))

    def decision_index(self, act: int, uid: int) -> int:
        return self._index.get((act, uid), self._index[OTHER])

    def to_dict(self) -> dict:
        d = asdict(self)
        d["decisions"] = [list(x) for x in self.decisions]
        return d


def encode(spec: FeatureSpec, frame: np.ndarray, eco: np.ndarray, units: np.ndarray,
           seen_now: np.ndarray, seen_max: np.ndarray, decided: np.ndarray,
           since_last_s: np.ndarray, style: int | None, map_hash: str | None = None,
           year: float | np.ndarray | None = None, matchup: str | None = None) -> np.ndarray:
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
    if spec.maps:
        map_onehot = np.zeros((n, len(spec.maps) + 1), dtype=np.float32)
        map_onehot[:, spec.maps.index(map_hash) if map_hash in spec.maps else len(spec.maps)] = 1
        parts.append(map_onehot)
    if spec.year:
        parts.append(np.broadcast_to(np.asarray((PLAY_YEAR if year is None else year) - 2000.0, np.float32) / 30.0,
                                     (n,))[:, None])
    if spec.matchups:
        mu_onehot = np.zeros((n, len(spec.matchups)), dtype=np.float32)
        if matchup in spec.matchups:
            mu_onehot[:, spec.matchups.index(matchup)] = 1
        parts.append(mu_onehot)
    x = np.concatenate([p.astype(np.float32) for p in parts], axis=1)
    assert x.shape[1] == spec.size, (x.shape, spec.size)
    return x


PLAY_YEAR = 2026.0               # the year in play (the meta Gary plays)


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



REPO = Path(__file__).resolve().parents[2]


def shared_macro() -> bool:
    """The race's shared macro model (one for all its matchups, the matchup an input: train/macro.py --matchup all;
    runs/macro_shared) and its history tables (..._shared): the default since L4 (48 games each on the ladder
    maps: results even, Gary-Z's gap score 0.34 against 0.36, its army at 12 minutes 23 against 17; better on the
    held-out games in all nine matchup-races). GARY_MACRO_SHARED=0: each matchup's own model."""
    return os.environ.get("GARY_MACRO_SHARED", "1") != "0"


def macro_model_path(matchup: str, race: str) -> Path:
    """The macro model to play matchup as race: the race's shared model when shared_macro() and it covers the
    matchup, else the matchup's own newest (runs/macro/<matchup>_<race>_*)."""
    if shared_macro():
        for p in sorted((REPO / "runs" / "macro_shared").glob(f"{race}_all_*/model.pt"), reverse=True):
            if matchup in json.loads((p.parent / "report.json").read_text(encoding="utf-8")).get("matchups", []):
                return p
    found = sorted((REPO / "runs" / "macro").glob(f"{matchup}_{race}_*/model.pt"))
    if not found:
        raise FileNotFoundError(f"no {race} macro model for {matchup}: python -m train.macro --matchup {matchup} --race {race}")
    return found[-1]


def pro_history(race: str, style: int | None = None, matchup: str = "TvZ") -> np.ndarray | None:
    """The pros' mean decision counts per minute for a race in a matchup (data/interim/macro/v1/
    history_mean_<matchup>_<race>.npy, train/macro_history.py), or for one of its build styles (..._styles.npy, the race's mean where a style has too few
    players: an average over styles told a 3-hatch player it had made a 9-pool player's units). None if not
    computed."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ingest"))
    try:
        from inventory import data_root
        root = data_root() / "interim" / "macro" / "v1"
        tag = f"{matchup}_{race}" + ("_shared" if shared_macro() else "")   # (the model's decision index)
        if style is not None and (root / f"history_mean_{tag}_styles.npy").exists():
            t = np.load(root / f"history_mean_{tag}_styles.npy")
            if 0 <= style < len(t):
                return t[style]
        return np.load(root / f"history_mean_{tag}.npy")
    except Exception:
        return None


class MacroTracker:
    """Builds the model's features during play, from what the player sees (gary.interface
    observations), the way ingest/macro_dataset.py built them from replays.

    Units still in production count as "all" but not "completed" (the game counts a unit once its
    production starts); observations don't list them, so the bot reports them (in_production).
    Enemy units count only while visible; the most of each type ever seen at once is kept, which
    is what the replay data's seen_max holds."""

    def __init__(self, spec: FeatureSpec, slot: int, style: int | None = None, map_hash: str | None = None,
                 matchup: str | None = None):
        self.spec, self.slot, self.style, self.map_hash, self.matchup = spec, slot, style, map_hash, matchup
        self.seen_max = np.zeros(228, dtype=np.float32)
        self.decided = np.zeros(len(spec.decisions), dtype=np.float32)
        self.last_decision_frame = 0
        # history: None, or the pros' mean decision counts per minute ((31, decisions): data/interim/macro/
        # history_mean_<matchup>_<race>.npy): the model then sees a pro's typical past, not this player's own. Gary-Z's own
        # history fed back on itself: its drone odds rested on "its decisions so far" at +0.36 (the records'
        # attribution, round g12), 60-70% drones at 6:00-7:30 where pros decide drones 17-24% of the time
        self.history: np.ndarray | None = None
        self.history_weight = 1.0        # the pros' share in the history the model sees (1: theirs alone; 0.5: half its own)
        self.own_keys: list[int] = []    # decisions whose history is always its own (option ownsupply: the supply provider)
        self.history_until = 0           # minutes over which that share fades to 0 (0: constant). With the pros' history
                                         # all game Gary-Z over-droned late (53 workers at 15-20 min, pros 35: the pros'
                                         # typical past kept telling the model to drone past what it had), with its
                                         # own early (the loop) it had no army at 6:00; round g16 against g13
        # option seenall: what is in sight looked at every look_every frames (look()), not only when the model is
        # queried: the replays' seen_max counts every second, and with the queue and dice (16 queries in 200 s) the
        # model in play saw an empty enemy history (15 enemies in sight, seen_max all 0, 2026-10-25)
        self.look_every = 0
        self.last_look = -1 << 30
        self.tag_types: dict[int, int] = {}   # own units' types by tag (one inside a bunker is out of sight)
        # option builtcount: the buildings among the decisions so far counted as the replays count them, each building
        # that started (a decision of the model's or a rule's), at its decision's time if it had one: decided Barracks
        # never built stayed counted (3 at 2:06 where the replay had 1), rules' bunkers never were (0 where it had 1)
        self.reconcile = False
        self.pending: list[list] = []         # building decisions noted and not seen started yet: [frame, act, uid]
        self.buildings_seen: set = set()      # (tag, type) of the player's buildings seen
        self.building_tags: set = set()       # tags seen as a building (a building's morph keeps its tag)
        self.halls_at: list = []              # the player's town halls' positions (an expansion: away from them)
        self.reconciled = False

    def look(self, obs: dict) -> None:
        """What is in sight now, between the model's queries: the most of each enemy type seen at once, and the
        player's own units' types by tag."""
        self.last_look = obs["frame"]
        seen = np.zeros(228, dtype=np.float32)
        for u in obs["units"]:
            t = u["type"]
            if not 0 <= t < 228:
                continue
            if u["owner"] == self.slot:
                self.tag_types[u["tag"]] = t
            elif u["owner"] != 11:
                seen[t] += 1
        self.seen_max = np.maximum(self.seen_max, seen)
        if self.reconcile:
            self._reconcile(obs)

    def _reconcile(self, obs: dict) -> None:
        """builtcount: a building of the player's seen for the first time consumes a pending decision for its type (or,
        none pending, a rule's building, counts as one: the replays count every building that started); a pending
        decision not started within PENDING_FOR is taken back; a building morphing from another is left alone (the
        replays count the morph's command)."""
        import math
        from gary.policy.army import is_building
        from gary.race import Race
        halls = Race.of(self.spec.race).town_halls
        own = [u for u in obs["units"] if u["owner"] == self.slot and is_building(u["type"])]
        new = [u for u in own if (u["tag"], u["type"]) not in self.buildings_seen]
        for u in new:
            self.buildings_seen.add((u["tag"], u["type"]))
        first = not self.reconciled
        self.reconciled = True
        for u in new:
            morph = u["tag"] in self.building_tags
            self.building_tags.add(u["tag"])
            if first or morph:                   # (the start's buildings; a Lair, a Sunken: not a build)
                if u["type"] in halls and first:
                    self.halls_at.append((u["x"], u["y"]))
                continue
            t = u["type"]
            match = next((p for p in self.pending if p[2] == t), None)
            if match is not None:
                self.pending.remove(match)
            else:
                away = t in halls and all(math.dist((u["x"], u["y"]), h) > HALL_APART for h in self.halls_at)
                self.decided[self.spec.decision_index(EXPAND_ACT if away else BUILD_ACT, t)] += 1
            if t in halls:
                self.halls_at.append((u["x"], u["y"]))
        f = obs["frame"]
        for p in [p for p in self.pending if f - p[0] > PENDING_FOR]:
            self.pending.remove(p)
            i = self.spec.decision_index(p[1], p[2])
            self.decided[i] = max(0, self.decided[i] - 1)

    def copy(self) -> "MacroTracker":
        """A copy to read the model's inputs with (features() updates what was seen): reading the live tracker would
        change it, and a tracker that updates only when read was the fault seenall fixed."""
        c = copy.copy(self)
        c.seen_max, c.decided, c.tag_types = self.seen_max.copy(), self.decided.copy(), dict(self.tag_types)
        return c

    def note_decision(self, act: int, uid: int, frame: int) -> None:
        self.decided[self.spec.decision_index(act, uid)] += 1
        self.last_decision_frame = frame
        if self.reconcile and act in (BUILD_ACT, EXPAND_ACT):
            self.pending.append([frame, act, uid])

    def unnote_decision(self, act: int, uid: int) -> None:
        """A decision that didn't happen after all (e.g. no room to place the building)."""
        i = self.spec.decision_index(act, uid)
        self.decided[i] = max(0, self.decided[i] - 1)
        p = next((p for p in reversed(self.pending) if p[1] == act and p[2] == uid), None)
        if p is not None:
            self.pending.remove(p)

    def features(self, obs: dict, in_production: dict[int, int], hidden=()) -> np.ndarray:
        """hidden: tags of the player's own units out of sight (inside a bunker), counted as the replays count
        them, by the type look() last saw."""
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
        for tag in hidden:
            if (t := self.tag_types.get(tag)) is not None:
                units[0, t] += 1
        self.seen_max = np.maximum(self.seen_max, seen)
        me = obs["me"]
        eco = np.array([[me["minerals"], me["gas"], me["supply_used"] * 2, me["supply_max"] * 2]], dtype=np.float32)
        frame = np.array([obs["frame"]], dtype=np.float32)
        decided = self.decided
        if self.history is not None:
            # the pros' counts at this moment: between the minute marks the table holds (decisions taken before each),
            # not the last mark's (all of minute 0 told the model nothing had been decided while it had 5 drones, and
            # it asked for a Spawning Pool at 97% from 0:20; the pros' cap refused it to 1:00: 42 s without a drone)
            t = obs["frame"] / 1440
            m, top = min(int(t), len(self.history) - 1), len(self.history) - 1
            a = t - int(t) if m < top else 0.0
            pro = (1.0 - a) * self.history[m] + a * self.history[min(m + 1, top)]
            w = self.history_weight * (max(0.0, 1.0 - obs["frame"] / (self.history_until * 1440)) if self.history_until else 1.0)
            decided = w * pro + (1.0 - w) * self.decided
            if self.own_keys:            # (the pros' 9-supply overlord in it: the model thought one was coming, and at 9/9
                decided = decided.copy()     # gave Overlord under 1% while Gary-Z had none: blocked 67% of 2:00, gz1dev)
                decided[self.own_keys] = self.decided[self.own_keys]
        return encode(self.spec, frame, eco, units, seen[None], self.seen_max[None], decided[None],
                      np.array([(obs["frame"] - self.last_decision_frame) / 24.0], dtype=np.float32), self.style,
                      self.map_hash, None, self.matchup)
