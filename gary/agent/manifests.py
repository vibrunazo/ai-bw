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
from gary.agent.macro import LearnedMacro, LingFloodDefense, OneBaseDefense
from gary.agent.motor import DirectMotor
from gary.agent.posture import (EarlyPoolPosture, EarlyPoolPostureV2, EarlyPoolPostureV3, EarlyPoolPostureV4,
                                EarlyPoolPostureV5, EarlyPoolPostureV6, NormalPosture)
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
    # v0.10: roles instead of the fight model (hand rules, roles.py), leases enforced
    "v010": {"version": "v0.10", "posture": "early_pool/v1", "leases": "enforced",
             "macro": {**_LEARNED, "builds": ["one_base_defense"]}, "army": _ARMY,
             "fight": {**_V09_FIGHT, "provider": "roles", "model": None}, "arbiter": "fixed_priority/v1"},
    # v0.11: v0.10 pulling 3 workers per zerg (not 2), and holding a post by the Command Center
    "v011": {"version": "v0.11", "posture": "early_pool/v1", "leases": "enforced",
             "macro": {**_LEARNED, "builds": ["one_base_defense"]}, "army": _ARMY,
             "fight": {**_V09_FIGHT, "provider": "roles", "model": None, "roles": {"post": 3},
                       "rule_options": {"pull": {"per_enemy": 3}}},
             "arbiter": "fixed_priority/v1"},
    # v0.12: v0.11 with a bunker at the natural once it takes it, and its first Barracks ordered by
    # 1:18 if the model hasn't (macro.py)
    "v012": {"version": "v0.12", "posture": "early_pool/v1", "leases": "enforced",
             "macro": {**_LEARNED, "builds": ["one_base_defense"], "natural_bunker": True, "rax_by": 78 * 24},
             "army": _ARMY,
             "fight": {**_V09_FIGHT, "provider": "roles", "model": None, "roles": {"post": 3},
                       "rule_options": {"pull": {"per_enemy": 3}}},
             "arbiter": "fixed_priority/v1"},
    # v0.13: v0.12 with the rush bunker by the main's Command Center, not its way out (posture.py v2)
    "v013": {"version": "v0.13", "posture": "early_pool/v2", "leases": "enforced",
             "macro": {**_LEARNED, "builds": ["one_base_defense"], "natural_bunker": True, "rax_by": 78 * 24},
             "army": _ARMY,
             "fight": {**_V09_FIGHT, "provider": "roles", "model": None, "roles": {"post": 3},
                       "rule_options": {"pull": {"per_enemy": 3}}},
             "arbiter": "fixed_priority/v1"},
    # v0.14: v0.13 whose idle Barracks, Factories and Starports make the model's last unit for them
    # again (macro.py keep_busy)
    "v014": {"version": "v0.14", "posture": "early_pool/v2", "leases": "enforced",
             "macro": {**_LEARNED, "builds": ["one_base_defense"], "natural_bunker": True, "rax_by": 78 * 24,
                       "keep_busy": True},
             "army": _ARMY,
             "fight": {**_V09_FIGHT, "provider": "roles", "model": None, "roles": {"post": 3},
                       "rule_options": {"pull": {"per_enemy": 3}}},
             "arbiter": "fixed_priority/v1"},
    # v0.15: v0.14 that keeps its bunkers loaded, not only once zerg are near (fight.py, load rule):
    # speedlings cross the 9 tiles in under 2 s, and marines walking in don't shoot
    "v015": {"version": "v0.15", "posture": "early_pool/v2", "leases": "enforced",
             "macro": {**_LEARNED, "builds": ["one_base_defense"], "natural_bunker": True, "rax_by": 78 * 24,
                       "keep_busy": True},
             "army": _ARMY,
             "fight": {**_V09_FIGHT, "provider": "roles", "model": None, "roles": {"post": 3},
                       "rule_options": {"pull": {"per_enemy": 3}, "load": {"always": True}}},
             "arbiter": "fixed_priority/v1"},
    # v0.16: v0.15 whose army hunts once no enemy army has been seen for a minute, scanning the bases
    # (army.py hunt, scan), and whose posture calls a pool finished by 2:40 an early pool (v3)
    "v016": {"version": "v0.16", "posture": "early_pool/v3", "leases": "enforced",
             "macro": {**_LEARNED, "builds": ["one_base_defense"], "natural_bunker": True, "rax_by": 78 * 24,
                       "keep_busy": True},
             "army": {**_ARMY, "hunt": True, "scan": True},
             "fight": {**_V09_FIGHT, "provider": "roles", "model": None, "roles": {"post": 3},
                       "rule_options": {"pull": {"per_enemy": 3}, "load": {"always": True}}},
             "arbiter": "fixed_priority/v1"},
    # v0.17: v0.16 that calls nothing seen an early pool only at 2:50 (posture v4)
    "v017": {"version": "v0.17", "posture": "early_pool/v4", "leases": "enforced",
             "macro": {**_LEARNED, "builds": ["one_base_defense"], "natural_bunker": True, "rax_by": 78 * 24,
                       "keep_busy": True},
             "army": {**_ARMY, "hunt": True, "scan": True},
             "fight": {**_V09_FIGHT, "provider": "roles", "model": None, "roles": {"post": 3},
                       "rule_options": {"pull": {"per_enemy": 3}, "load": {"always": True}}},
             "arbiter": "fixed_priority/v1"},
    # v0.18: v0.17 whose posture (v5) calls off an alarm from nothing seen until 4:00, and lets the
    # main's rush bunker out once the rush is over
    "v018": {"version": "v0.18", "posture": "early_pool/v5", "leases": "enforced",
             "macro": {**_LEARNED, "builds": ["one_base_defense"], "natural_bunker": True, "rax_by": 78 * 24,
                       "keep_busy": True},
             "army": {**_ARMY, "hunt": True, "scan": True},
             "fight": {**_V09_FIGHT, "provider": "roles", "model": None, "roles": {"post": 3},
                       "rule_options": {"pull": {"per_enemy": 3}, "load": {"always": True}}},
             "arbiter": "fixed_priority/v1"},
    # v0.17 with learned roles (gary/policy/roles.py; the model: --fight-model / fight_path, default
    # the latest in runs/roles): the same rules around them, roles' options passed on (the post)
    "v017roleml": {"version": "v0.17", "posture": "early_pool/v4", "leases": "enforced",
                   "macro": {**_LEARNED, "builds": ["one_base_defense"], "natural_bunker": True, "rax_by": 78 * 24,
                             "keep_busy": True},
                   "army": {**_ARMY, "hunt": True, "scan": True},
                   "fight": {**_V09_FIGHT, "provider": "roles", "model": "roles/latest", "roles": {"post": 3},
                             "rule_options": {"pull": {"per_enemy": 3}, "load": {"always": True}}},
                   "arbiter": "fixed_priority/v1"},
    # v0.17 with a learned override of its roles' rules (gary/agent/roles.py RoleOverride: per unit
    # join, hold, or the rules; the model: --fight-model / fight_path, default the latest in runs/roles)
    "v017override": {"version": "v0.17", "posture": "early_pool/v4", "leases": "enforced",
                     "macro": {**_LEARNED, "builds": ["one_base_defense"], "natural_bunker": True,
                               "rax_by": 78 * 24, "keep_busy": True},
                     "army": {**_ARMY, "hunt": True, "scan": True},
                     "fight": {**_V09_FIGHT, "provider": "role_override", "model": "roles/latest",
                               "roles": {"post": 3}, "rule_options": {"pull": {"per_enemy": 3}, "load": {"always": True}}},
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
               "fight_cmd/latest_memory": lambda: latest_command_model(True), "fight_cmd_rl/v08": latest_rl_fight_model,
               "belief/latest": lambda: _newest("belief"), "plan/latest": lambda: _newest("plan")}


def _learned_posture(g, m: dict, models: dict):
    """The plan policy deciding the posture's mode (gary/policy/plan.py; docs/CONTROL.md step 7)."""
    from gary.agent.posture import LearnedPosture
    from gary.policy.plan import PlanModel
    model = PlanModel.load(model_path(m.get("plan_model", "plan/latest")))
    g.plan_model = model.path            # (not in the start-of-game chat: one more line shifts the game)
    return LearnedPosture(g, model, m.get("plan_temperature", 0.0))


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

def _roles(**options):
    return lambda m: m["fight"].setdefault("roles", {}).update(options)


def _pull(**options):
    return lambda m: m["fight"].setdefault("rule_options", {}).setdefault("pull", {}).update(options)


# options to try on top of a manifest, "v010+post+suspend" (docs/CONTROL.md: every layer swappable);
# home defenses (114; [239]: the 239 of all held-out games), reactive Zerg, 8 draws, vs v0.10 (+ is
# better, ± 1 s.e.), and immediate 9-pools lost of 16 (v0.10: 0 of these 16, but 4 of 48: a
# loss or three in 16 is noise); 2026-10-06
OPTIONS = {
    "brace": _roles(brace=True),            # roles.py: let the zerg come, nobody closes in     -8 ± 11
    "leash": _roles(leash=True),            # close in only on zerg attacking or by buildings   -9 ± 9
    "volley": _roles(volley=True),          # close in by moving, then fire together           -38 ± 10
    "regroup": _roles(regroup=True),        # loose units rejoin the pack, the pack closes in  -13 ± 10
    # roles.py anchor (units out in the open fall back to the Command Centers and bunkers; the leash
    # ignores zerg on units out there) and army.py home_guard (no army moves out while enemy fighters
    # are at home): Gary's army split in its losses to Zerg v2's all-in (41% of its army's deaths at
    # 5:00-8:00 out in the open; with anchor 36%). On v0.18, lost (v0.18): Zerg v2 (72) Fighting Spirit
    # 10 (8), Python 7 (11); replayed pros (48) Python 4 (3), Fighting Spirit 2 (5), Heartbreak Ridge 1
    # (3); the 9-pool 1 (1), won 31 (36): 25 of 336 (31). Home defenses [239] -7 ± 10. Not adopted
    "anchor": lambda m: (m["fight"].setdefault("roles", {}).update(anchor=True), m["army"].update(home_guard=True)),
    "holds": _roles(holds=True),            # units on hold position stay on it                 -4 ± 9
    "kite": _roles(kite=True),              # vultures pull back from zerglings                  -5 ± 2
    "fire_here": _roles(fire="here"),       # fire: attack-move to the group's own middle       -8 ± 5
    "fire_hold": _roles(fire="hold"),       # fire: hold position                              -23 ± 10
    "post": _roles(post=3),                 # hold a post by the Command Center (and leash)     +8 ± 9, [239] +7 ± 5,
                                            # and +7 ± 5 on top of pull_three: in v0.11
    "post0": _roles(post=0),                # ...at the Command Center                          +6 ± 10, [239] +4 ± 5
    "post6": _roles(post=6),                # ...6 tiles from it                                -4 ± 8
    # vultures skirmish (move away, shoot, ease off; roles.py): [239] −14 ± 4, worse in 36 of the
    # 47 with vultures: 18% fewer vultures lost, but fewer zerg killed and more SCVs lost
    "vultures": _roles(vultures=True),
    # posts inside the wall gaps (gary/walls.py), holding: on v0.11 [239] -7 ± 4 (better in 26, worse
    # in 42; 0c50ac60 +104): the home defenses start with the zerg at home, past the wall's job;
    # on top of sim city, in 78 wall fights (eval/walls_bench.py) -12 ± 12
    "gap_post": _roles(gap_post=True),
    # depots, barracks, academies wall the frontier, a door left (simcity.py)
    # the first Barracks ordered by 1:18 (not 1:30): on v0.12 without it, replayed pros on three maps
    # lost 15 of 144 (19), the 9-pool 2 either way. Placed at ~1:38: Gary's money comes ~10 s late
    "rax118": lambda m: m["macro"].update(rax_by=78 * 24),
    # a bunker at the natural once Gary takes it, where pros put theirs (macro.py): on v0.11, against
    # the replayed pros (48 seeds), lost 28 of 192 on three maps (v0.11 37); 9-pool lost 2 either way
    "natbunker": lambda m: m["macro"].update(natural_bunker=True),
    "simcity": lambda m: m.update(simcity=True),
    "simcity_closed": lambda m: m.update(simcity=True, simcity_door=0),   # ...closed
    # roles.py, pros' marines move 20% of the time a zerg is in range, Gary's 6%: on v0.14 [239]
    "surround2": _roles(surround=2),        # units with 2+ melee zerg on them walk to the pack -17 ± 6
    "surround3": _roles(surround=3),        # ...3+                                              -6 ± 3
    "pull75": _roles(pull_hp=0.75),         # pull back below 75% HP (not 50%)                  -5 ± 2
    # fight.py: a loaded bunker with no zerg near lets its marines out to the fight elsewhere (the
    # main's rush bunker while the natural is attacked). On v0.14 against Zerg v2 (72 seeds, 12 min)
    # lost 22 (15). Not adopted
    "unload": lambda m: m["fight"].update(before=m["fight"]["before"] + ["unload"]),
    # fight.py: 4 SCVs repair a bunker (not 2): against Zerg v2 (72 seeds) lost 15 (15)
    "repair4": lambda m: m["fight"].setdefault("rule_options", {}).update(repair={"n": 4}),
    # fight.py: 4 SCVs move past the bunker, then repair it (spread around it); and new fighters rally
    # to the main's mineral line while the zerg at home outweigh Gary's fighters there. Both on top of
    # v0.17+posture5+ramp, lost (ramp alone): Zerg v2 on Fighting Spirit 16 (8) and Python 13 (12) of
    # 72; replayed pros (48) Fighting Spirit 2 (5), Python 5 (5); the 9-pool 1 (1). Not adopted
    "repair_surround": lambda m: m["fight"].setdefault("rule_options", {}).update(repair={"n": 4, "surround": True}),
    "safe_rally": lambda m: m["fight"].setdefault("rule_options", {}).update(reinforce={"safe": True}),
    # fight.py: bunkers kept loaded. On v0.14: Zerg v2 (72 seeds, 12 min) lost 8 (15); replayed pros on
    # three maps lost 14 of 144 (15), the 9-pool 3 (3); home defenses [239] +2 ± 5. In v0.15
    "preload": lambda m: m["fight"].setdefault("rule_options", {}).update(load={"always": True}),
    # army.py: from 8:00, 20+ army units and no enemy army seen for a minute, the army goes where enemy
    # buildings were, then to the bases. On v0.15: replayed pros on Fighting Spirit to 18:00 (16 seeds)
    # won 8 (2), lost 2 (2); to 12:00 on three maps won 4 of 144 (0), lost 14 (14); Zerg v2 (72 seeds)
    # won 2 (0), lost 8 (8). In v0.16
    "hunt": lambda m: m["army"].update(hunt=True),
    # army.py: groups going away from home hold when the belief model's likely unseen enemy army
    # (gary/policy/belief.py, 80th percentile less what was seen in 3 min) outweighs them. On v0.17,
    # replayed pros on three maps lost 12 of 144 (11); it held 637 times. Not adopted
    "belief": lambda m: m["army"].update(belief="belief/latest"),
    # ...against all the enemy army it likely has, not only what wasn't seen (what was seen may be
    # dead): lost 13 of 144 (11), held 854 times. Not adopted
    "belief_total": lambda m: m["army"].update(belief="belief/latest", belief_total=True),
    # army.py: the hunt scans bases before walking there: on v0.16 without it, replayed pros on Fighting
    # Spirit to 18:00 (16 seeds) won 11 (8). In v0.16
    "scan": lambda m: m["army"].update(scan=True),
    # army.py watch: the scout stays in the Zerg's bases (moving, away from zerglings) until hunted or
    # 5:00, then watches outside Gary's natural; replaced once if lost by 4:30. Zerg in sight 3:00-6:00:
    # 78 s (44; pros 104). On v0.18 against Zerg v2 (72 seeds) Fighting Spirit lost 11 (8), won 9 (3);
    # Python lost 10 (11), won 3 (1). The macro sees the zerglings now but builds no more Barracks.
    # With the waypoint tour (in sight 103 s): Fighting Spirit lost 10 (8), won 10 (3); Python 8 (11), 3 (1)
    "watch": lambda m: m["army"].update(watch=True),
    # posture.py v3: a pool finished by 2:40 is an early pool, with a natural or not. On v0.16 (v2),
    # replayed pros (buildings going up) lost 13 of 144 (15; Heartbreak Ridge 7, 9), the same hatch
    # firsts called early pools. In v0.16
    "pool_sure": lambda m: m.update(posture="early_pool/v3"),
    # fight.py: new fighters join Gary's nearest bunch of fighters, not the zerg directly. On v0.17:
    # Python (48 seeds) lost 5 (5); Zerg v2 (72) lost 13 (9); with regroup too 5 and 14 (regroup
    # alone 5 and 12). Not adopted
    "reinforce_pack": lambda m: m["fight"].setdefault("rule_options", {}).update(reinforce={"to_pack": True}),
    # posture.py v5: an alarm from nothing seen called off until 4:00; the main's bunker let out after.
    # On v0.17: replayed pros on three maps lost 11 of 144 (9), won 9 (2); the 9-pool lost 1 (1), won
    # 36 (27); Zerg v2 (72) lost 8 (10). In v0.18
    "posture5": lambda m: m.update(posture="early_pool/v5"),
    # posture.py v6: a zergling mass seen (6+ at once, 1.5x Gary's army, 3:30-8:00): the
    # ling_flood_defense build (macro.py), no expansion, the army home, bunkers manned. With watch
    # (the scout that sees it coming), on v0.18, lost (v0.18): Zerg v2 (72) Fighting Spirit 8 (8),
    # Python 7 (11); replayed pros (48) Python 8 (3), Fighting Spirit 6 (5), Heartbreak Ridge 4 (3);
    # the 9-pool 1 (1), won 32 (36): 34 of 336 (31). It switches in 40% of the replayed pros' games
    # (false alarms), and in the losses only once the zerglings attack (6:07, 6:47); Gary floats
    # 400-700 minerals at 6:00 on one Barracks either way. Not adopted
    # the posture's mode from the plan policy (gary/policy/plan.py, the newest run: distilled from
    # v6's rules, then trained) with v6's builds
    "plan": lambda m: (m.update(posture="plan/v1"),
                       m["macro"].update(builds=m["macro"]["builds"] + ["ling_flood_defense"])),
    "flood": lambda m: (m.update(posture="early_pool/v6"),
                        m["macro"].update(builds=m["macro"]["builds"] + ["ling_flood_defense"])),
    # fight.py BlockRule: the chokes asked for (g.blocks; nothing in play asks yet: eval/block_drill.py
    # does) closed with units' bodies (gary/agent/block.py). ramp_auto: it asks for the main's ramp
    # itself on one base when zerg fighters are seen there. With the body block, on v0.18, lost
    # (without): Zerg v2 on Fighting Spirit 12 (8) and Python 10 (11) of 72; replayed pros (48)
    # Python 7 (3), Fighting Spirit 5 (5), Heartbreak Ridge 5 (3); the 9-pool 4 (1), won 25 (36): 43
    # of 336 (31). Against the 9-pool it takes the first marines the rush bunker at the ramp wants.
    # Not adopted: when to block is for a learned decision
    "block": lambda m: m["fight"].update(before=m["fight"]["before"] + ["block"]),
    # the decision learned from branches (train/block_decision.py, the newest run): on held-out games
    # -25 a moment against never blocking's 0 (docs/PROGRESS.md). Not adopted
    "block_learned": lambda m: (m["fight"].update(before=m["fight"]["before"] + ["block"]),
                                m["fight"].setdefault("rule_options", {}).update(block={"decide": "latest"})),
    "ramp_auto": lambda m: (m["fight"].update(before=m["fight"]["before"] + ["block"]),
                            m["fight"].setdefault("rule_options", {}).update(block={"auto_ramp": True})),
    "post_minerals": _roles(post="minerals"),   # ...between it and the minerals                +2 ± 8
    # posture.py v2: the rush bunker by the main's CC. On v0.12: replayed pros on three maps lost 3 of
    # 144 (19); the dummy's 9-pool 0 (2), its 7:00 rush 3 (0); won 3 and 3 (12 and 10)
    "bunker_cc": lambda m: m.update(posture="early_pool/v2"),
    # macro.py: idle Barracks, Factories, Starports make the model's last unit for them again. On v0.13:
    # army supply at 8:00 11-12 (7-9; pros 19), at 12:00 32-44 (25-34; pros 36); replayed pros lost 7 of
    # 144 (10), the 9-pool 0 (0)
    "busy": lambda m: m["macro"].update(keep_busy=True),
    # macro.py: roll the model's odds (a weighted shuffle) instead of taking the most likely decision.
    # Offline, on pros' states, it matched their Factory timing (63% by 8:00, theirs 62%; the
    # favorite 4%). In games on v0.14 (Fighting Spirit, replayed pros, 32 seeds): fewer SCVs (28 at
    # 12:00 against 41: the SCV is rolled less often than it's the favorite), army built 40 by 12:00
    # (57), a Factory by 8:00 in 9% (19%), lost 3 (2). Not adopted
    "dice": lambda m: m["macro"].update(decoder="dice"),
    # training (docs/CONTROL.md step 7): the macro draws its next decision at temperature 1 (or
    # sample_t05: 0.5), each draw logged (g.macro.draws)
    "sample": lambda m: m["macro"].update(decoder="sample", temperature=1.0),
    "sample_t05": lambda m: m["macro"].update(decoder="sample", temperature=0.5),
    # fight.py: pull workers only for zerg in the mineral line (not at a bunker further out). On the
    # fast pools that beat v0.12 on Heartbreak Ridge it kept a few more SCVs; the bunker's spot mattered
    "pull_line": _pull(mineral_line=True),
    "pull_attacked": _pull(attacked_only=True),   # fight.py: pull workers only when zerg attack  +2 ± 3, [239] +3 ± 3
    "pull_ratio15": _pull(ratio=1.5),       # ...when the zerg outweigh the army 1.5 to 1      [239] -3 ± 4
    "pull_ratio2": _pull(ratio=2.0),        #                                                   [239] -6 ± 5
    "pull_delay2": _pull(delay_s=2),        # ...after 2 s of threat without a break           [239] -4 ± 4
    "pull_delay4": _pull(delay_s=4),        #                                                   [239] -19 ± 7
    "pull_one": _pull(per_enemy=1),         # one worker per enemy unit (and one)               [239] -13 ± 4
    "pull_three": _pull(per_enemy=3),       # [239] +8 ± 2 (4: +8 ± 3, 5: +9 ± 3); 9-pool 6 of 48 (v0.10 4): in v0.11
    "pull_three_half": _pull(per_enemy=3, max_share=0.5),   # ...at most half the base's workers (9-pool: no change)
    "pull_three_60": _pull(per_enemy=3, max_share=0.6),
    "pull_four": _pull(per_enemy=4),
    "pull_five": _pull(per_enemy=5),
    "pull_ratio07": _pull(ratio=0.7),       # ...already when the zerg are 0.7 of the army      [239] +1 ± 4
    # a fight interrupts waiting builds: +3 ± 6, [239] +6 ± 4; but 9-pool lost 3 (the bunker comes late)
    "suspend": lambda m: m.update(arbiter="fixed_priority/v2"),
}


def manifest_of(name: str | dict) -> dict:
    """A manifest by name, with options: "v010+post+suspend" is v010 with those OPTIONS."""
    if not isinstance(name, str):
        return copy.deepcopy(name)
    base, *options = name.split("+")
    m = copy.deepcopy(MANIFESTS[base])
    for o in options:
        OPTIONS[o](m)
    return m


def manifest_name(name: str) -> str:
    """For argparse (type=): a manifest name, with options, that exists."""
    base, *options = name.split("+")
    if base not in MANIFESTS or base == "drill" or any(o not in OPTIONS for o in options):
        raise ValueError(name)
    return name


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
        if m.get("simcity"):
            from gary.agent.simcity import SimCity
            g.simcity = SimCity(g, m.get("simcity_door", 32))
    g.posture = {"normal": NormalPosture, "early_pool/v1": EarlyPoolPosture,
                 "early_pool/v2": EarlyPoolPostureV2, "early_pool/v3": EarlyPoolPostureV3,
                 "early_pool/v4": EarlyPoolPostureV4, "early_pool/v5": EarlyPoolPostureV5,
                 "early_pool/v6": EarlyPoolPostureV6,
                 "plan/v1": lambda g: _learned_posture(g, m, models)}[m["posture"]](g)
    if m["macro"] is not None:
        path = model_path(m["macro"]["model"])
        macro = load("macro", str(path))
        models["macro"] = getattr(macro, "path", "?")
        builds = {k: c(g) for k, c in (("one_base_defense", OneBaseDefense), ("ling_flood_defense", LingFloodDefense))
                  if k in m["macro"].get("builds", [])}
        g.macro = LearnedMacro(g, macro, style, builds, m["macro"].get("natural_bunker", False),
                               m["macro"].get("rax_by", 90 * 24), m["macro"].get("keep_busy", False),
                               m["macro"].get("decoder", "favorite"), seed, m["macro"].get("temperature", 1.0))
    if m["army"] is not None:
        if m["army"]["provider"] == "waves":
            g.army = WaveArmy(g)
        else:
            army = load("army", str(model_path(m["army"]["model"])))
            models["army"] = getattr(army, "path", "?")
            belief = None
            if m["army"].get("belief"):
                from gary.policy.belief import BeliefModel
                belief = BeliefModel.load(model_path(m["army"]["belief"]))
                models["belief"] = belief.path
            g.army = LearnedArmy(g, army, PullRule(g) if m["army"].get("pull") else None, hunt=m["army"].get("hunt", False),
                                 scan=m["army"].get("scan", False), belief=belief,
                                 belief_total=m["army"].get("belief_total", False), watch=m["army"].get("watch", False),
                                 home_guard=m["army"].get("home_guard", False))
    if m["fight"] is not None:
        f = m["fight"]
        if f["provider"] == "roles" and f.get("model") is None:
            fight = None
            models["fight"] = "roles/rules_v1"   # (the chat says so: as many lines as with a model)
        elif f["provider"] in ("roles", "role_override"):
            from gary.policy.roles import RoleModel
            fight = fight_model or RoleModel.load(fight_path or model_path(f["model"]))
            models["fight"] = fight.path
        else:
            fight = fight_model or load("fight", str(fight_path or model_path(f["model"])))
            models["fight"] = getattr(fight, "path", "?")
        rule = lambda r: RULES[r](g, **f.get("rule_options", {}).get(r, {}))
        g.fight = FightLayer(g, fight, f["provider"], [rule(r) for r in f.get("before", [])],
                             [rule(r) for r in f.get("in_fight", [])], f.get("guard_rail", False),
                             f.get("workers", True), seed, f.get("ctrl_click", False),
                             f.get("watch_tiles", 0) * 32, f.get("roles", {}))
    if m["arbiter"] == "drill/v1":
        g.arbiter = DrillArbiter(g, seed, interruptions)
    else:
        g.arbiter = FixedPriority(g, suspend_builds=m["arbiter"] == "fixed_priority/v2")
        g.announce = announcement(g.version, style, models)
    return g
