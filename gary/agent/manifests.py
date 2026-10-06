"""Gary versions as manifests: one provider per layer (docs/CONTROL.md), and building a Gary from one.

A version is a choice per layer, nothing more. To try a variant, copy a manifest and change the
layer you're testing; everything else stays the same Gary.

    from gary.agent.manifests import build_gary
    gary = build_gary(hi, mapinfo, "v09", style=1)
"""

from __future__ import annotations

import copy
from pathlib import Path

from gary import terran as T
from gary.agent.army import LearnedArmy, WaveArmy
from gary.agent.arbiter import DrillArbiter, FixedPriority
from gary.agent.executor import ScriptedExecutor
from gary.agent.fight import RULES, FightLayer, PullRule
from gary.agent.gary import Gary
from gary.agent.macro import LearnedMacro, OneBaseDefense
from gary.agent.motor import DirectMotor
from gary.agent.posture import EarlyPoolPosture, NormalPosture
from gary.version import announcement

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# the fight model v0.8 was judged with (RL run 4, round 70: 114 home defenses -61); later runs don't
# replace it unless they beat it and this is updated
V08_FIGHT_MODEL = REPO_ROOT / "runs" / "fight_cmd_rl" / "home_defense_20261005_091044" / "model.pt"

_LEARNED = {"provider": "learned", "model": "macro/latest"}
_ARMY = {"provider": "learned", "model": "army/latest"}
_V09_FIGHT = {"provider": "command", "model": "fight_cmd_rl/v08", "before": ["repair", "pull", "load"],
              "in_fight": ["reinforce"], "workers": False}

MANIFESTS: dict[str, dict] = {
    # learned macro, army in waves of 16
    "v02": {"version": "v0.2", "posture": "normal", "macro": _LEARNED, "army": {"provider": "waves"}, "fight": None,
            "arbiter": "fixed_priority/v1"},
    # learned army, scouting, a worker-pull rule
    "v03": {"version": "v0.3", "posture": "normal", "macro": _LEARNED, "army": {**_ARMY, "pull": True}, "fight": None,
            "arbiter": "fixed_priority/v1"},
    # a learned fight model per unit; bunkers loaded by rule
    "v04": {"version": "v0.4", "posture": "normal", "macro": _LEARNED, "army": _ARMY,
            "fight": {"provider": "per_unit", "model": "fight/latest", "before": ["load"]},
            "arbiter": "fixed_priority/v1"},
    # the fight command model: one command at a time to a selection
    "v05": {"version": "v0.5", "posture": "normal", "macro": _LEARNED, "army": _ARMY,
            "fight": {"provider": "command", "model": "fight_cmd/latest", "before": ["load"]},
            "arbiter": "fixed_priority/v1"},
    # ...that remembers its recent commands
    "v06": {"version": "v0.6", "posture": "normal", "macro": _LEARNED, "army": _ARMY,
            "fight": {"provider": "command", "model": "fight_cmd/latest_memory", "before": ["load"]},
            "arbiter": "fixed_priority/v1"},
    # workers pulled by rule, not the model's; a guard rail on its moves and stops
    "v07": {"version": "v0.7", "posture": "normal", "macro": _LEARNED, "army": _ARMY,
            "fight": {"provider": "command", "model": "fight_cmd/latest_memory", "before": ["pull", "load"],
                      "guard_rail": True, "workers": False},
            "arbiter": "fixed_priority/v1"},
    # v0.7 without the guard rail: for judging fight models trained by RL (train/fight_cmd_rl.py)
    "v07free": {"version": "v0.7", "posture": "normal", "macro": _LEARNED, "army": _ARMY,
                "fight": {"provider": "command", "model": "fight_cmd/latest_memory", "before": ["pull", "load"],
                          "workers": False},
                "arbiter": "fixed_priority/v1"},
    # the fight model trained by RL, no guard rail
    "v08": {"version": "v0.8", "posture": "normal", "macro": _LEARNED, "army": _ARMY,
            "fight": {"provider": "command", "model": "fight_cmd_rl/v08", "before": ["pull", "load"], "workers": False},
            "arbiter": "fixed_priority/v1"},
    # holds early pools: the early-pool posture, its one-base build, repair and reinforce rules
    "v09": {"version": "v0.9", "posture": "early_pool/v1",
            "macro": {**_LEARNED, "builds": ["one_base_defense"]}, "army": _ARMY, "fight": _V09_FIGHT,
            "arbiter": "fixed_priority/v1"},
    # step 4 (docs/CONTROL.md): v0.9 with leases enforced; and with roles instead of the fight model
    "v09lease": {"version": "v0.9", "posture": "early_pool/v1", "leases": "enforced",
                 "macro": {**_LEARNED, "builds": ["one_base_defense"]}, "army": _ARMY, "fight": _V09_FIGHT,
                 "arbiter": "fixed_priority/v1"},
    "v09roles": {"version": "v0.9", "posture": "early_pool/v1", "leases": "enforced",
                 "macro": {**_LEARNED, "builds": ["one_base_defense"]}, "army": _ARMY,
                 "fight": {**_V09_FIGHT, "provider": "roles", "model": None}, "arbiter": "fixed_priority/v1"},
    # step 5: the roles named by a model learned from the pros' role labels
    "v09roleml": {"version": "v0.9", "posture": "early_pool/v1", "leases": "enforced",
                  "macro": {**_LEARNED, "builds": ["one_base_defense"]}, "army": _ARMY,
                  "fight": {**_V09_FIGHT, "provider": "roles", "model": "roles/latest"}, "arbiter": "fixed_priority/v1"},
    # the fight layer alone, for micro drills (gary/drills.py)
    "drill": {"version": "drill", "posture": "normal", "macro": None, "army": None,
              "fight": {"provider": "command", "model": "fight_cmd/latest_memory", "workers": False},
              "arbiter": "drill/v1"},
}


# --- models ------------------------------------------------------------------------------------

def _newest(kind: str, matchup: str = "TvZ", race: str = "T") -> Path:
    found = sorted((REPO_ROOT / "runs" / kind).glob(f"{matchup}_{race}_*/model.pt"))
    if not found:
        raise SystemExit(f"no trained {kind} model in runs/{kind} (python -m train.{kind} --race {race})")
    return found[-1]


def latest_macro_model() -> Path:
    return _newest("macro")


def latest_army_model() -> Path:
    return _newest("army")


def latest_fight_model() -> Path:
    """The newest per-unit fight model (gary/policy/fight.py)."""
    return _newest("fight")


def latest_command_model(memory: bool = False) -> Path:
    """The newest fight command model, with or without memory of its own recent commands."""
    import torch
    for path in sorted((REPO_ROOT / "runs" / "fight_cmd").glob("TvZ_T_*/model.pt"), reverse=True):
        if torch.load(path, map_location="cpu", weights_only=True)["config"]["net"].get("memory", False) == memory:
            return path
    raise SystemExit(f"no trained command model{' with memory' if memory else ''} in runs/fight_cmd "
                     f"(python -m train.fight_cmd --race T{' --memory --data v3' if memory else ''})")


def latest_rl_fight_model() -> Path:
    """v0.8's fight model, or (on a machine without it) the newest one trained by reinforcement
    learning (runs/fight_cmd_rl)."""
    if V08_FIGHT_MODEL.exists():
        return V08_FIGHT_MODEL
    found = sorted((REPO_ROOT / "runs" / "fight_cmd_rl").glob("*/model.pt"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise SystemExit("no RL-trained fight model in runs/fight_cmd_rl (python -m train.fight_cmd_rl)")
    return found[-1]


def latest_role_model() -> Path:
    """The newest role model (train/roles.py)."""
    found = sorted((REPO_ROOT / "runs" / "roles").glob("*/model.pt"))
    if not found:
        raise SystemExit("no trained role model in runs/roles (python -m ingest.role_dataset, python -m train.roles)")
    return found[-1]


MODEL_PATHS = {"roles/latest": latest_role_model, "macro/latest": latest_macro_model, "army/latest": latest_army_model,
               "fight/latest": latest_fight_model, "fight_cmd/latest": lambda: latest_command_model(False),
               "fight_cmd/latest_memory": lambda: latest_command_model(True), "fight_cmd_rl/v08": latest_rl_fight_model}


def model_path(ref: str) -> Path:
    """A manifest's model reference ("macro/latest", ...) or a checkpoint path."""
    return MODEL_PATHS[ref]() if ref in MODEL_PATHS else Path(ref)


def load_model(kind: str, path: str | Path):
    """kind: "macro", "army" or "fight" (the fight model or the command model, whichever the
    checkpoint holds)."""
    from gary.policy.army import ArmyModel
    from gary.policy.fight_cmd import load_fight_policy
    from gary.policy.macro import MacroModel
    return {"macro": MacroModel.load, "army": ArmyModel.load, "fight": load_fight_policy}[kind](path)


# --- building a Gary ---------------------------------------------------------------------------

def manifest_of(name: str | dict) -> dict:
    return copy.deepcopy(MANIFESTS[name] if isinstance(name, str) else name)


def build_gary(hi, mapinfo, manifest: str | dict, style: int | None = None, verbose: bool = True,
               fight_path: str | Path | None = None, load=load_model, seed: int = 0,
               flip: tuple[bool, bool] | None = None, interruptions: bool = True, fight_model=None) -> Gary:
    """A Gary from a manifest (a name in MANIFESTS, or a dict like them). fight_path (or an already
    loaded fight_model) replaces the manifest's fight model; load(kind, path) loads models (callers
    that cache pass their own); seed seeds the fight layer's sampling; flip and interruptions are
    for drills (no map)."""
    m = manifest_of(manifest)
    g = Gary(hi, mapinfo, m["version"], verbose)
    g.army_types = set(T.ARMY)
    g.production_types = T.PRODUCTION if mapinfo is not None else (T.CC,)
    if flip is not None:
        g.flip = flip
    models: dict[str, str] = {}
    g.motor = DirectMotor(g)
    g.leases.enforced = m.get("leases") == "enforced"
    if m["macro"] is not None or m["army"] is not None:
        g.executor = ScriptedExecutor(g)
    g.posture = {"normal": NormalPosture, "early_pool/v1": EarlyPoolPosture}[m["posture"]](g)
    if m["macro"] is not None:
        path = model_path(m["macro"]["model"])
        macro = load("macro", str(path))
        models["macro"] = getattr(macro, "path", "?")
        builds = {"one_base_defense": OneBaseDefense(g)} if "one_base_defense" in m["macro"].get("builds", []) else {}
        g.macro = LearnedMacro(g, macro, style, builds)
    if m["army"] is not None:
        if m["army"]["provider"] == "waves":
            g.army = WaveArmy(g)
        else:
            army = load("army", str(model_path(m["army"]["model"])))
            models["army"] = getattr(army, "path", "?")
            g.army = LearnedArmy(g, army, PullRule(g) if m["army"].get("pull") else None)
    if m["fight"] is not None:
        f = m["fight"]
        if f["provider"] == "roles" and f.get("model") is None:
            fight = None
            models["fight"] = "roles/rules_v1"   # (the chat says so: as many lines as with a model)
        elif f["provider"] == "roles":
            from gary.policy.roles import RoleModel
            fight = RoleModel.load(model_path(f["model"]))
            models["fight"] = fight.path
        else:
            fight = fight_model or load("fight", str(fight_path or model_path(f["model"])))
            models["fight"] = getattr(fight, "path", "?")
        g.fight = FightLayer(g, fight, f["provider"], [RULES[r](g) for r in f.get("before", [])],
                             [RULES[r](g) for r in f.get("in_fight", [])], f.get("guard_rail", False),
                             f.get("workers", True), seed, f.get("ctrl_click", False))
    if m["arbiter"] == "drill/v1":
        g.arbiter = DrillArbiter(g, seed, interruptions)
    else:
        g.arbiter = FixedPriority(g)
        g.announce = announcement(g.version, style, models)
    return g
