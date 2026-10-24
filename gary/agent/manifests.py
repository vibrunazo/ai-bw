"""Gary versions as manifests: one provider per layer (docs/CONTROL.md), and building a Gary from one.

A version is a choice per layer, nothing more. To try a variant, copy a manifest and change the
layer you're testing; everything else stays the same Gary.

    from gary.agent.manifests import build_gary
    gary = build_gary(hi, mapinfo, "v09", style=1)
"""

from __future__ import annotations

import os

import copy
from pathlib import Path

from gary.agent.army import LearnedArmy, WaveArmy
from gary.agent.arbiter import DrillArbiter, FixedPriority
from gary.agent.terran_executor import TerranExecutor
from gary.agent.zerg_executor import ZergExecutor
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
# what a Terran manifest built for TvZ plays against a non-Zerg (its "vs" section, by the enemy's race): the
# early-pool posture, the scripted rush builds, the answers to the Zerg's plan, the natural bunker and the
# strategy selector's TvZ table are the Zerg's; the role model is trained on TvZ only, so the fight layer's
# rules play (the first TvT game, Gary-T against Gary-T: both sides saw an early pool and built ramp bunkers)
_VS_NON_ZERG = {"posture": "normal", "macro": {"builds": [], "answers": False, "natural_bunker": False, "selector": None},
                "fight": {"provider": "roles", "model": None, "roles": {"post": 3}}}

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
    # again (macro.py keep_busy; removed in gt0.7, so v0.14 to gt0.6 now play without it)
    "v014": {"version": "v0.14", "posture": "early_pool/v2", "leases": "enforced",
             "macro": {**_LEARNED, "builds": ["one_base_defense"], "natural_bunker": True, "rax_by": 78 * 24},
             "army": _ARMY,
             "fight": {**_V09_FIGHT, "provider": "roles", "model": None, "roles": {"post": 3},
                       "rule_options": {"pull": {"per_enemy": 3}}},
             "arbiter": "fixed_priority/v1"},
    # v0.15: v0.14 that keeps its bunkers loaded, not only once zerg are near (fight.py, load rule):
    # speedlings cross the 9 tiles in under 2 s, and marines walking in don't shoot
    "v015": {"version": "v0.15", "posture": "early_pool/v2", "leases": "enforced",
             "macro": {**_LEARNED, "builds": ["one_base_defense"], "natural_bunker": True, "rax_by": 78 * 24},
             "army": _ARMY,
             "fight": {**_V09_FIGHT, "provider": "roles", "model": None, "roles": {"post": 3},
                       "rule_options": {"pull": {"per_enemy": 3}, "load": {"always": True}}},
             "arbiter": "fixed_priority/v1"},
    # v0.16: v0.15 whose army hunts once no enemy army has been seen for a minute, scanning the bases
    # (army.py hunt, scan), and whose posture calls a pool finished by 2:40 an early pool (v3)
    "v016": {"version": "v0.16", "posture": "early_pool/v3", "leases": "enforced",
             "macro": {**_LEARNED, "builds": ["one_base_defense"], "natural_bunker": True, "rax_by": 78 * 24},
             "army": {**_ARMY, "hunt": True, "scan": True},
             "fight": {**_V09_FIGHT, "provider": "roles", "model": None, "roles": {"post": 3},
                       "rule_options": {"pull": {"per_enemy": 3}, "load": {"always": True}}},
             "arbiter": "fixed_priority/v1"},
    # v0.17: v0.16 that calls nothing seen an early pool only at 2:50 (posture v4)
    "v017": {"version": "v0.17", "posture": "early_pool/v4", "leases": "enforced",
             "macro": {**_LEARNED, "builds": ["one_base_defense"], "natural_bunker": True, "rax_by": 78 * 24},
             "army": {**_ARMY, "hunt": True, "scan": True},
             "fight": {**_V09_FIGHT, "provider": "roles", "model": None, "roles": {"post": 3},
                       "rule_options": {"pull": {"per_enemy": 3}, "load": {"always": True}}},
             "arbiter": "fixed_priority/v1"},
    # v0.18: v0.17 whose posture (v5) calls off an alarm from nothing seen until 4:00, and lets the
    # main's rush bunker out once the rush is over
    "v018": {"version": "v0.18", "posture": "early_pool/v5", "leases": "enforced",
             "macro": {**_LEARNED, "builds": ["one_base_defense"], "natural_bunker": True, "rax_by": 78 * 24},
             "army": {**_ARMY, "hunt": True, "scan": True},
             "fight": {**_V09_FIGHT, "provider": "roles", "model": None, "roles": {"post": 3},
                       "rule_options": {"pull": {"per_enemy": 3}, "load": {"always": True}}},
             "arbiter": "fixed_priority/v1"},
    # v0.19: v0.18 whose build style is chosen per map by the strategy selector (gary/agent/selector.py)
    "v019": {"version": "v0.19", "posture": "early_pool/v5", "leases": "enforced",
             "macro": {**_LEARNED, "builds": ["one_base_defense"], "natural_bunker": True, "rax_by": 78 * 24, "selector": "play"},
             "army": {**_ARMY, "hunt": True, "scan": True},
             "fight": {**_V09_FIGHT, "provider": "roles", "model": None, "roles": {"post": 3},
                       "rule_options": {"pull": {"per_enemy": 3}, "load": {"always": True}}},
             "arbiter": "fixed_priority/v1"},
    # v0.20: v0.19 whose new fighters gather as pros' marines do (executor.py rerally: inside the main's ramp,
    # at its bottom from 4:45, in front of the natural from 7:30)
    "v020": {"version": "v0.20", "posture": "early_pool/v5", "leases": "enforced", "rally": "ramp",
             "macro": {**_LEARNED, "builds": ["one_base_defense"], "natural_bunker": True, "rax_by": 78 * 24, "selector": "play"},
             "army": {**_ARMY, "hunt": True, "scan": True},
             "fight": {**_V09_FIGHT, "provider": "roles", "model": None, "roles": {"post": 3},
                       "rule_options": {"pull": {"per_enemy": 3}, "load": {"always": True}}},
             "arbiter": "fixed_priority/v1"},
    # v0.21: v0.20 whose workers astray go back to mining, at the least saturated safe base, with transfers
    # to a new base (executor.py home_scvs, balance)
    "v021": {"version": "v0.21", "posture": "early_pool/v5", "leases": "enforced", "rally": "ramp",
             "home_scvs": True, "balance": True,
             "macro": {**_LEARNED, "builds": ["one_base_defense"], "natural_bunker": True, "rax_by": 78 * 24, "selector": "play"},
             "army": {**_ARMY, "hunt": True, "scan": True},
             "fight": {**_V09_FIGHT, "provider": "roles", "model": None, "roles": {"post": 3},
                       "rule_options": {"pull": {"per_enemy": 3}, "load": {"always": True}}},
             "arbiter": "fixed_priority/v1"},
    # v0.22: v0.21 that defends against an early pool from 2:30 unless the rules switched first (posture.py
    # force_at) and scans burrowed lurkers near its fighters (fight.py DetectRule)
    "v022": {"version": "v0.22", "posture": "early_pool/v5", "leases": "enforced", "rally": "ramp",
             "home_scvs": True, "balance": True, "defend_early": 150 * 24,
             "macro": {**_LEARNED, "builds": ["one_base_defense"], "natural_bunker": True, "rax_by": 78 * 24, "selector": "play"},
             "army": {**_ARMY, "hunt": True, "scan": True},
             "fight": {**_V09_FIGHT, "provider": "roles", "model": None, "roles": {"post": 3},
                       "before": _V09_FIGHT["before"] + ["detect"],
                       "rule_options": {"pull": {"per_enemy": 3}, "load": {"always": True}}},
             "arbiter": "fixed_priority/v1"},
    # v0.23: v0.22 that takes a third base (and more) once the macro model's odds of expanding add up
    # to one (macro.py expand_mass)
    "v023": {"version": "v0.23", "posture": "early_pool/v5", "leases": "enforced", "rally": "ramp",
             "home_scvs": True, "balance": True, "defend_early": 150 * 24,
             "macro": {**_LEARNED, "builds": ["one_base_defense"], "natural_bunker": True, "rax_by": 78 * 24, "selector": "play", "expand_mass": 1.0},
             "army": {**_ARMY, "hunt": True, "scan": True},
             "fight": {**_V09_FIGHT, "provider": "roles", "model": None, "roles": {"post": 3},
                       "before": _V09_FIGHT["before"] + ["detect"],
                       "rule_options": {"pull": {"per_enemy": 3}, "load": {"always": True}}},
             "arbiter": "fixed_priority/v1"},
    # v0.24: v0.23 that reads the zerg's plan and answers lurkers as pros do (answers.py: Siege Mode, tanks,
    # a Science Facility, vessels, Irradiate, turrets one a base; a tech scan of their main at 6:30)
    "v024": {"version": "v0.24", "posture": "early_pool/v5", "leases": "enforced", "rally": "ramp",
             "home_scvs": True, "balance": True, "defend_early": 150 * 24,
             "macro": {**_LEARNED, "builds": ["one_base_defense"], "natural_bunker": True, "rax_by": 78 * 24, "selector": "play", "expand_mass": 1.0, "answers": True},
             "army": {**_ARMY, "hunt": True, "scan": True},
             "fight": {**_V09_FIGHT, "provider": "roles", "model": None, "roles": {"post": 3},
                       "before": _V09_FIGHT["before"] + ["detect"],
                       "rule_options": {"pull": {"per_enemy": 3}, "load": {"always": True}}},
             "arbiter": "fixed_priority/v1"},
    # v0.25: v0.24 whose hunts finish games (army.py, motor.py: a hunt's camera on a unit, 8 boxes, a scan of a
    # remembered building the army hasn't settled in 20 s, an Extractor destroyed forgotten). Shared code: the
    # manifest is v0.24's; the version marks where the fixes came in
    "v025": {"version": "v0.25", "posture": "early_pool/v5", "leases": "enforced", "rally": "ramp",
             "home_scvs": True, "balance": True, "defend_early": 150 * 24,
             "macro": {**_LEARNED, "builds": ["one_base_defense"], "natural_bunker": True, "rax_by": 78 * 24, "selector": "play", "expand_mass": 1.0, "answers": True},
             "army": {**_ARMY, "hunt": True, "scan": True},
             "fight": {**_V09_FIGHT, "provider": "roles", "model": None, "roles": {"post": 3},
                       "before": _V09_FIGHT["before"] + ["detect"],
                       "rule_options": {"pull": {"per_enemy": 3}, "load": {"always": True}}},
             "arbiter": "fixed_priority/v1"},
    # v0.26: v0.25 with the fixes from human game 1 (OPTIONS muta_answer, pull_cap), and the
    # parked build that keeps its SCV (executor.py, shared code)
    "v026": {"version": "v0.26", "posture": "early_pool/v5", "leases": "enforced", "rally": "ramp",
             "home_scvs": True, "balance": True, "defend_early": 150 * 24,
             "macro": {**_LEARNED, "builds": ["one_base_defense"], "natural_bunker": True, "rax_by": 78 * 24, "selector": "play", "expand_mass": 1.0, "answers": True,
                       "answer_options": {"mutas": True}},
             "army": {**_ARMY, "hunt": True, "scan": True},
             "fight": {**_V09_FIGHT, "provider": "roles", "model": None, "roles": {"post": 3},
                       "before": _V09_FIGHT["before"] + ["detect"],
                       "rule_options": {"pull": {"per_enemy": 3, "ground_only": True, "max_enemy": 8},
                                        "load": {"always": True}}},
             "arbiter": "fixed_priority/v1"},
    # v0.27: v0.26 with the fixes from human game 2: SCVs repair a mineral-line turret under attack (3, as pros
    # send several; OPTION repair_turrets); stim, learned from the pros' stims (OPTION stim); builders queued back
    # to the minerals (OPTION builder_back). Shared code: against mutalisks the answer's turrets go up first,
    # in the mineral line nearest the zerg first (the model's own turrets, built away from the minerals, wait
    # until then); live placement keeps add-on spots free (turrets had taken both Comsats' places); a pending
    # add-on's building isn't kept busy (an SCV queued just before its Comsat: the game refused it)
    "v027": {"version": "v0.27", "posture": "early_pool/v5", "leases": "enforced", "rally": "ramp",
             "home_scvs": True, "balance": True, "defend_early": 150 * 24, "builder_back": True,
             "macro": {**_LEARNED, "builds": ["one_base_defense"], "natural_bunker": True, "rax_by": 78 * 24, "selector": "play", "expand_mass": 1.0, "answers": True,
                       "answer_options": {"mutas": True}},
             "army": {**_ARMY, "hunt": True, "scan": True},
             "fight": {**_V09_FIGHT, "provider": "roles", "model": None, "roles": {"post": 3},
                       "before": _V09_FIGHT["before"] + ["detect", "stim"],
                       "rule_options": {"pull": {"per_enemy": 3, "ground_only": True, "max_enemy": 8},
                                        "load": {"always": True}, "repair": {"turrets": 3}}},
             "arbiter": "fixed_priority/v1"},
    # v0.28: v0.27 whose army isn't held by noise (the humanlikeness classifier's tell: the army stood still 78% of
    # 5-7 min, pros 37%): plain moves (positioning) always go; an attack is held only when the fight estimate says
    # it loses 5+ points of army share, or 5+ worse than staying (v0.27: 2 and 1, held ~90% of decisions at 5-6 min)
    "v028": {"version": "v0.28", "posture": "early_pool/v5", "leases": "enforced", "rally": "ramp",
             "home_scvs": True, "balance": True, "defend_early": 150 * 24, "builder_back": True,
             "macro": {**_LEARNED, "builds": ["one_base_defense"], "natural_bunker": True, "rax_by": 78 * 24, "selector": "play", "expand_mass": 1.0, "answers": True,
                       "answer_options": {"mutas": True}},
             "army": {**_ARMY, "hunt": True, "scan": True, "hold_moves": False, "hold_margin": 0.05},
             "fight": {**_V09_FIGHT, "provider": "roles", "model": None, "roles": {"post": 3},
                       "before": _V09_FIGHT["before"] + ["detect", "stim"],
                       "rule_options": {"pull": {"per_enemy": 3, "ground_only": True, "max_enemy": 8},
                                        "load": {"always": True}, "repair": {"turrets": 3}}},
             "arbiter": "fixed_priority/v1"},
    # v0.29: v0.28 whose fighters answer what they see before contact as the pros do: the role model (decisions:
    # keep or a new role, with a where head) learned from 600 pro games' units whenever a seen enemy fighter was
    # within 12 tiles (ingest/response_dataset.py: 1.3M labels before contact); the fight layer watches 12 tiles.
    # The hand rules in contact, the model before it (roles.py ahead/v1): its roles for every unit instead of
    # the rules lost 52 net value a home defense (226 pro scenarios), with or without its where; the watch
    # alone +4 +- 5
    "v029": {"version": "v0.29", "posture": "early_pool/v5", "leases": "enforced", "rally": "ramp",
             "home_scvs": True, "balance": True, "defend_early": 150 * 24, "builder_back": True,
             "macro": {**_LEARNED, "builds": ["one_base_defense"], "natural_bunker": True, "rax_by": 78 * 24, "selector": "play", "expand_mass": 1.0, "answers": True,
                       "answer_options": {"mutas": True}},
             "army": {**_ARMY, "hunt": True, "scan": True, "hold_moves": False, "hold_margin": 0.05},
             "fight": {**_V09_FIGHT, "provider": "role_ahead", "model": "runs/roles/20261016-195206/model.pt",
                       "roles": {"post": 3}, "watch_tiles": 12,
                       "before": _V09_FIGHT["before"] + ["detect", "stim"],
                       "rule_options": {"pull": {"per_enemy": 3, "ground_only": True, "max_enemy": 8},
                                        "load": {"always": True}, "repair": {"turrets": 3}}},
             "vs": {"T": _VS_NON_ZERG, "P": _VS_NON_ZERG},
             "arbiter": "fixed_priority/v1"},
    # v0.19 with learned roles (gary/policy/roles.py, with the terrain inputs; the model: --fight-model /
    # fight_path, default the latest in runs/roles): the same rules around them
    "v019roleml": {"version": "v0.19", "posture": "early_pool/v5", "leases": "enforced",
                   "macro": {**_LEARNED, "builds": ["one_base_defense"], "natural_bunker": True, "rax_by": 78 * 24, "selector": "play"},
                   "army": {**_ARMY, "hunt": True, "scan": True},
                   "fight": {**_V09_FIGHT, "provider": "roles", "model": "roles/latest", "roles": {"post": 3},
                             "rule_options": {"pull": {"per_enemy": 3}, "load": {"always": True}}},
                   "arbiter": "fixed_priority/v1"},
    # v0.17 with learned roles (gary/policy/roles.py; the model: --fight-model / fight_path, default
    # the latest in runs/roles): the same rules around them, roles' options passed on (the post)
    "v017roleml": {"version": "v0.17", "posture": "early_pool/v4", "leases": "enforced",
                   "macro": {**_LEARNED, "builds": ["one_base_defense"], "natural_bunker": True, "rax_by": 78 * 24},
                   "army": {**_ARMY, "hunt": True, "scan": True},
                   "fight": {**_V09_FIGHT, "provider": "roles", "model": "roles/latest", "roles": {"post": 3},
                             "rule_options": {"pull": {"per_enemy": 3}, "load": {"always": True}}},
                   "arbiter": "fixed_priority/v1"},
    # v0.17 with a learned override of its roles' rules (gary/agent/roles.py RoleOverride: per unit
    # join, hold, or the rules; the model: --fight-model / fight_path, default the latest in runs/roles)
    "v017override": {"version": "v0.17", "posture": "early_pool/v4", "leases": "enforced",
                     "macro": {**_LEARNED, "builds": ["one_base_defense"], "natural_bunker": True,
                               "rax_by": 78 * 24},
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
    # the one Gary playing Zerg (docs/REFACTOR.md Phase 1): the shared agent with the Zerg executor and Gary-Z's
    # macro options (the dice; one-offs and hatcheries by summed odds; a decision parked for its money while
    # the larvae go on; the model's tempo; no tech path under 10%; an overlord ahead; the pros' history; a rolled
    # style; a roll waiting for supply or money held, 2026-10-20: LearnedMacro.hold, round G11 against G9). Its army
    # and fights from Gary-Z (Phase 2)
    # gz0.2: its army on control groups by type (gary/agent/grouping.py; 96 paired games twice against gt1, rounds
    # GZ40a+b and GZ40c+d against L40a+b: Gary-T won 49 and 48 (57), lost 10 and 13 (8); Gary-Z's gap at 15-20
    # minutes nearer the pros both times, -0.073 (-0.102 to -0.017) the second)
    "gz1": {"version": "gz0.2", "race": "Z", "posture": "zerg", "leases": "enforced", "balance": True, "rotate": True,
            "grouping": "types",
            "macro": {**_LEARNED, "builds": ["alarm"], "decoder": "dice", "odds_mass": 1.0, "hatch_mass": 0.5, "park": True, "pace": True,
                      "tech_min_p": 0.10, "wait_min_p": 0.05, "supply_ahead": True, "rax_by": 165 * 24, "history": "pro", "style": "roll",
                      "hold": True},
            # the army: clusters with commitments, drawn (Gary-Z's GARY_Z_COMMIT=latest), its ground army (the flyers
            # and lurkers have their own rules); no worker scout (its overlords)
            "army": {"provider": "learned", "model": "commit/latest", "commit": True, "draw": True, "scout": False,
                     "types": [37, 38, 39, 45, 46], "rules": ["posts", "raid", "join", "sweep", "press"], "hunt": True},
            # its fight roles from the pros' Zerg units in whole games to 30 minutes, old and modern (round R30Z against
            # B30: the Zerg ahead or winning at 30:00 in 9 of 48 games to 6, its gap at 21-30 minutes 0.40 to 0.46)
            "fight": {"provider": "zerg", "roles_data": "response_Z_30min.npz"}, "arbiter": "fixed_priority/v1"},
    # the fight layer alone, for micro drills (gary/drills.py)
    "drill": {"version": "drill", "posture": "normal", "macro": None, "army": None,
              "fight": {"provider": "command", "model": "fight_cmd/latest_memory", "workers": False},
              "arbiter": "drill/v1"},
}


# --- models ------------------------------------------------------------------------------------

def _newest(kind: str, matchup: str, race: str) -> Path:
    found = sorted((REPO_ROOT / "runs" / kind).glob(f"{matchup}_{race}_*/model.pt"))
    if not found:
        raise SystemExit(f"no trained {kind} model for {race} in {matchup} in runs/{kind} "
                         f"(python -m train.{kind} --matchup {matchup} --race {race})")
    return found[-1]


def _command_model(matchup: str, race: str, memory: bool) -> Path:
    """The newest fight command model, with or without memory of its own recent commands."""
    import torch
    for path in sorted((REPO_ROOT / "runs" / "fight_cmd").glob(f"{matchup}_{race}_*/model.pt"), reverse=True):
        if torch.load(path, map_location="cpu", weights_only=True)["config"]["net"].get("memory", False) == memory:
            return path
    raise SystemExit(f"no trained command model{' with memory' if memory else ''} for {race} in {matchup} in "
                     f"runs/fight_cmd (python -m train.fight_cmd --matchup {matchup} --race {race}"
                     f"{' --memory --data v3' if memory else ''})")


def _tvz_terran(kind: str, matchup: str, race: str, find) -> Path:
    """A kind trained for Gary-T in TvZ only (its runs aren't named by matchup)."""
    if (matchup, race) != ("TvZ", "T"):
        raise SystemExit(f"no {kind} model for {race} in {matchup}: there are only TvZ Terran ones (runs/{kind})")
    found = find()
    if not found:
        raise SystemExit(f"no trained {kind} model in runs/{kind}")
    return found[-1]


def _runs(kind: str, key=None) -> list[Path]:
    return sorted((REPO_ROOT / "runs" / kind).glob("*/model.pt"), key=key)


def model_for(kind: str, matchup: str, race: str) -> Path:
    """The model of this kind a player of this race uses in this matchup: the race's shared model where
    one covers the matchup (macro: gary.policy.macro.macro_model_path), else the matchup's own newest.
    None for the matchup: SystemExit (a manifest then has to name a rules provider). Kinds: macro,
    army, commit, engage, belief, plan, fight, army_rl, fight_cmd and fight_cmd_memory (the command model
    without and with memory), and TvZ Terran's own roles, macro_rl and fight_cmd_rl (v0.8's fight
    model, or else the newest trained by reinforcement learning)."""
    if kind == "macro":
        from gary.policy.macro import macro_model_path
        return macro_model_path(matchup, race)
    if kind in ("fight_cmd", "fight_cmd_memory"):
        return _command_model(matchup, race, kind == "fight_cmd_memory")
    if kind == "roles":
        return _tvz_terran(kind, matchup, race, lambda: _runs("roles"))
    if kind == "macro_rl":
        return _tvz_terran(kind, matchup, race, lambda: _runs("macro_rl"))
    if kind == "fight_cmd_rl":
        return _tvz_terran(kind, matchup, race, lambda: [V08_FIGHT_MODEL] if V08_FIGHT_MODEL.exists() else
                           _runs("fight_cmd_rl", key=lambda p: p.stat().st_mtime))
    return _newest(kind, matchup, race)


# a manifest's model references: "<kind>/latest", and these
REFS = {"fight_cmd/latest_memory": "fight_cmd_memory", "fight_cmd_rl/v08": "fight_cmd_rl"}


def _engage_model(g, m: dict):
    """The attack-or-hold model for the army layer (option engage; GARY_TENGAGE_MODEL pins it: the
    league loop's incumbent or candidate)."""
    ref = m["army"].get("engage")
    if not ref:
        return None
    from gary.policy.army import ArmyModel
    return ArmyModel.load(os.environ.get("GARY_TENGAGE_MODEL") or model_path(ref, g.matchup, g.race.letter))


def _learned_posture(g, m: dict, models: dict):
    """The plan policy deciding the posture's mode (gary/policy/plan.py; docs/CONTROL.md step 7)."""
    from gary.agent.posture import LearnedPosture
    from gary.policy.plan import PlanModel
    model = PlanModel.load(model_path(m.get("plan_model", "plan/latest"), g.matchup, g.race.letter))
    g.plan_model = model.path            # (not in the start-of-game chat: one more line shifts the game)
    return LearnedPosture(g, model, m.get("plan_temperature", 0.0))


HISTORY_FADE = 12                    # option history: minutes over which the pros' history fades to the player's own


def rolled_style(model, matchup: str, race: str, seed: int) -> int | None:
    """A build style rolled among the model's, by the pros' shares of them where the taxonomy is here
    (data/interim/taxonomy/<matchup>_<race>_plan.json, Gary-Z's plan roll; else the tech clusters', <matchup>_<race>
    _tech.json: Terran TvZ 1 standard bio 81%, 2 Factory and vultures 13%, 3 Factory and Starport 4%, 4 two-Barracks
    marines 2%), else evenly."""
    import json
    import random
    styles = model.spec.styles
    if not styles:
        return None
    shares = None
    try:
        from ingest.inventory import data_root
        d = json.loads((data_root() / "interim" / "taxonomy" / f"{matchup}_{race}_plan.json").read_text(encoding="utf-8"))
        counts = [0] * len(d["names"])
        for k in d["assignments"].values():
            counts[k] += 1
        shares = [c / max(1, sum(counts)) for c in counts]
    except Exception:                         # (no plan taxonomy here: the tech clusters', by style number)
        try:
            from ingest.inventory import data_root
            d = json.loads((data_root() / "interim" / "taxonomy" / f"{matchup}_{race}_tech.json").read_text(encoding="utf-8"))
            vals = list(d["assignments"].values())
            shares = [vals.count(k) / max(1, len(vals)) for k in range(max(vals) + 1)]
        except Exception:
            pass
    weights = [shares[s] if shares and s < len(shares) else 1.0 for s in styles]
    return random.Random(seed).choices(styles, weights=weights)[0]


def model_path(ref: str, matchup: str, race: str) -> Path:
    """A manifest's model reference ("macro/latest", ...) for this matchup and race, or a checkpoint path."""
    if ref in REFS:
        return model_for(REFS[ref], matchup, race)
    if ref.endswith("/latest"):
        return model_for(ref[:-len("/latest")], matchup, race)
    return Path(ref)


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
    # roles.py (with a role model that has a where head): retreat, pull back and reposition by the rules'
    # skills, not where the model moves them (ablation of v0.29)
    "nowhere": lambda m: m["fight"].setdefault("roles", {}).update(where=False),
    # army.py: destinations and move/attack drawn from the army model's odds, logged for the league's RL
    # (Gary-Z's likeliest-cell army stood at home 79% at 8-10 min; drawn 56%, pros 35%)
    "draw": lambda m: (m["army"].update(draw=True),
                       m["fight"].setdefault("roles", {}).update(draw=True) if m["fight"].get("provider") == "role_ahead" else None),
    # the army model updated by the league's RL (train/league_rl.py --layer army_t), drawing as it learned
    "army_rl": lambda m: m["army"].update(model="army_rl/latest", draw=True),
    # army.py: attack or hold drawn by the engage model (train/engage.py, from the pros' orders: held-out AUC
    # 0.80), destinations from the army model on that side; the league's learner for the army's initiative
    "engage": lambda m: m["army"].update(draw=True, engage="engage/latest"),
    # army.py: commitments (train/commit.py: keep / engage / move every 5 s per cluster, given its commitment,
    # learned from the pros' armies between orders too); the army model replaced by the commit model, whose
    # destination head was trained along (GARY_TCOMMIT_MODEL pins it)
    # In play the likeliest of keep / engage / move and the likeliest cell; drawn only with +draw (the league's
    # training): human game 3 was played drawn, and the two retreats that threw it were 4.8% and 5.1% draws
    "commit": lambda m: m["army"].update(commit=True, model="commit/latest"),
    # army.py: no hunt (v0.16's scripted attack once no enemy army was seen for a minute): with commitments the
    # engage choice decides when the army goes for the enemy's buildings. In human game 3 the hunt sent 23
    # marines into three sunkens at "100% sure", one by one, twice
    "nohunt": lambda m: m["army"].update(hunt=False),
    # army.py commit_act: with commit, a new commitment once the model's odds of one reach 41% (where the
    # deterministic rate matches the drawn one, the pros'), the likelier of engage and move; the plain likeliest
    # is keep most ticks and Gary-T turned passive (won 7 of 24 against Gary-Z, 17 at time; drawn 13 and 11)
    "act": lambda m: m["army"].update(commit_act=0.41),
    # army.py commit_floor (with commit and draw): the drawn decode without its disasters: no keep / engage / move
    # drawn under 15% (human game 3's 19 low draws were all under 14%), the destination among the 5 likeliest
    # cells. The deterministic decodes won fewer games against Gary-Z than drawing (6-7 of 24 against 13)
    "floor": lambda m: m["army"].update(commit_floor=0.15),
    # army_rules.py LordHunt: one or two marines hunt an exposed overlord (the user, after human game 4)
    "lords": lambda m: m["army"].update(rules=[*m["army"].get("rules", ()), "lords"]),
    # army_rules.py ClusterRules (Gary-Z's, shared): join, a cluster under the main one's size walks to it rather than
    # deciding alone (Gary-T's attacks: 0.26 late arrivals per fighter there, the pros' 0.06, eval/attacks.py); sweep,
    # the main body attacks a packet it outweighs; press, far ahead and nothing at home, it attacks the buildings
    # that end the game. Not adopted for gt1 (48 games against G13: +join won 20 (21), the Zerg ahead at 20:00 in 8
    # (2); +press won 22 but the Zerg ahead in 7; +join+press won 18)
    **{r: (lambda r: lambda m: m["army"].update(rules=[*m["army"].get("rules", ()), r]))(r) for r in ("join", "sweep", "press")},
    # army.py (STOP-GAP, AGENTS.md; in gt1, shared by every race's learned army): hunt_ahead, the hunt also at 4x
    # the enemy fighters seen in the last minute; hunt_split, unseen bases searched by squads; static_odds, an engage
    # only where the group outweighs the ground static defense there (6 supply each) and the enemy fighters seen
    # near it 1.5 times
    "hunt4": lambda m: m["army"].update(hunt_ahead=4.0),
    "split": lambda m: m["army"].update(hunt_split=True),
    "static": lambda m: m["army"].update(static_odds=1.5),
    # army.py gather (STOP-GAP, AGENTS.md): a commit engage with a quarter of the army elsewhere gathers it at a staging
    # point short of the target first, then attacks with it (any race's learned army)
    "gather": lambda m: m["army"].update(gather=True),
    # the fight roles model trained on whole games to 30 minutes, old and modern (ingest/response_dataset.py --era
    # both; until 2026-10-20 the first 20 minutes on the three 1.16 maps): either race's. In gz1 and gt1
    "roles30": lambda m: m["fight"].update(roles_data=f"response_{m.get('race', 'T')}_30min.npz"),
    # the role model imitated from attended pro units only, told the player's order rate (train/roles.py --attended
    # --budget): Gary's own budget from its hands (order_budget); the roles RL's start (docs/PROGRESS.md)
    # the role model decides in contact too (roles.py learned/v1, not RoleAhead's hand rules there), the same skills
    # carrying its groups out: what the roles RL trains reaches every fight command
    "incontact": lambda m: m["fight"].update(provider="roles"),
    # fight.py attend: the fight's share of the hands held at the pros' attended share of fight moments
    "attend": lambda m: m["fight"].update(attend=True),
    # the first roles RL run that moved and improved (train/roles_rl.py from gt1dev, round 50 of 60 by validation on
    # the training games' scenarios): the held-out state-selected suite +195 against its start's +87 (paired +105,
    # +44 to +169), against RoleAhead's +141 +50 (-51 to +151)
    "rl1": lambda m: m["fight"].update(roles_model="runs/roles_rl/20261022_185355/model.pt"),
    # the third roles RL run, from rl1 with each decision credited over 20 s (--horizon 20; rl2's 10 s won the suite
    # and lost in games): the held-out suite +268 against rl1's +195 (paired +71, +12 to +135), against RoleAhead +122
    "rl3": lambda m: m["fight"].update(roles_model="runs/roles_rl/20261023_043828/model.pt"),
    # the second roles RL run, from rl1 (60 rounds, its best validation at round 60, -53 against rl1's -152): the
    # held-out suite +255 against rl1's +195 (paired +67, +3 to +134), against RoleAhead's +141 +117 (+12 to +231)
    "rl2": lambda m: m["fight"].update(roles_model="runs/roles_rl/20261023_014740/model.pt"),
    # roles.py learned/v1 batch: up to 4 orders a fight turn, a group per role, given as the hands come free
    "batch": lambda m: m["fight"].setdefault("roles", {}).update(batch=4),
    # the attended model with the again decision and its offset head (train/roles.py --again: the pros' re-issues)
    # roles.py learned/v1 together: a key's units draw their decisions with one shared random number (a group decides
    # as one, as the pros' keys do; with grouping)
    "together": lambda m: m["fight"].setdefault("roles", {}).update(together=True),
    # roles.py learned/v1 by_key: a key whose units all drew one decision gets one order, the whole key
    "bykey": lambda m: m["fight"].setdefault("roles", {}).update(by_key=True),
    "again": lambda m: m["fight"].update(roles_data=f"response_{m.get('race', 'T')}_att_again.npz"),
    "att": lambda m: m["fight"].update(roles_data=f"response_{m.get('race', 'T')}_att.npz"),
    # roles.py RoleAhead lone: a unit alone and outnumbered in contact decided by the roles model, not the hand rules
    "lone": lambda m: m["fight"].setdefault("roles", {}).update(lone=True),
    # roles.py refresh: the skills keep their orders current (a standing role's order given again when its target moved
    # a tile: half to three quarters of the pros' fight commands)
    "refresh": lambda m: m["fight"].setdefault("roles", {}).update(refresh=True),
    # gary/agent/grouping.py: the army on control groups, a unit type of a cluster on a key, re-assigned (the pros'
    # habit); the motor recalls a key when all its members get an order, key by key
    "grouping": lambda m: m.update(grouping="types"),
    # the learned provider: the grouping model's groups (gary/policy/grouping_model.py, train/groups.py)
    "grouplearned": lambda m: m.update(grouping="learned"),
    # the same on keys 1-3 with the buildings' keys left where they are (executor.py): is the Terran's cost the layout
    # (production on 3 keys instead of 4) or the grouping? (rounds GT40c/d against L40a/b)
    "grouping3": lambda m: m.update(grouping="types", grouping_layout=False),
    # fight.py: enemy fighters within 12 tiles make a fight (the model's 8 by default; ablation of v0.29)
    "watch12": lambda m: m["fight"].update(watch_tiles=12),
    # army.py: groups going away from home hold when the belief model's likely unseen enemy army
    # (gary/policy/belief.py, 80th percentile less what was seen in 3 min) outweighs them. On v0.17,
    # replayed pros on three maps lost 12 of 144 (11); it held 637 times. Not adopted
    "belief": lambda m: m["army"].update(belief="belief/latest"),
    # army.py hunt_belief: the hunt (no enemy army seen for a minute) only with an army at least the belief
    # model's 80th percentile of the zerg's (the belief loaded for this alone: no holds). On v0.25 (432 games,
    # Zerg v3 to 25 min): won 125 (139), lost 19 (12): no pressure losses (27 won lost 2 -> 31, 0), but lurker
    # games won 10 (20) (the belief, learned from pros' Zergs, expects 40-50 supply at 15:00 where v3 has
    # none) and Zerg v2 lost 14 (8): early hunts pay against these Zergs. Not adopted
    "hunt_belief": lambda m: m["army"].update(hunt_belief=True),
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
    # On v0.19 over the whole pool (fresh seeds, 432 games, Zerg v3 to 20 min): lost 30 (25), won 127 (133);
    # against v3 won 54 (64), lost 4 (1): an SCV off minerals for minutes, for information no decision uses
    "watch": lambda m: m["army"].update(watch=True),
    # executor.py rerally: new fighters gather inside the main's ramp, at its bottom from 4:45, in front
    # of the natural from 7:30, as the pros' marines (the rally in front of the natural from the start
    # put a quarter of Gary's marines in the open at 3:00-6:00, the pros' 3-8%)
    # On v0.19 over the whole pool (fresh seeds, 432 games, Zerg v3 to 20 min): lost 24 (25), won 135 (133);
    # replayed pros lost 13 (17), Zerg v3 won 59 (64). In v0.20
    "rally_ramp": lambda m: m.update(rally="ramp"),
    # executor.py _idle, motor.py unpull: workers away from every Command Center, idle or still attack-moving
    # with nobody holding them, go back to mining (the sweep finds them off screen, pulled or not), and a
    # pull's workers go back to the nearest base Gary owns (before: the nearest base on the map)
    "home_scvs": lambda m: m.update(home_scvs=True),
    # executor.py: a builder's next order queued on a mineral field, and the mining task's other idle workers
    # on the screen added before sending (human game 2: 11-17 of 30-39 SCVs idle from 8:00 to 12:00). In v0.27
    "builder_back": lambda m: m.update(builder_back=True),
    # home_scvs, and executor.py balance: workers sent to mine go to the least saturated safe base (not the
    # nearest), a group of workers moves from a base over 1.8 a patch to a safe one under 1.0 (a new
    # Command Center), and workers out in the field are selected with one ctrl+click
    # On v0.20 over the whole pool (fresh seeds, 432 games, Zerg v3 to 20 min): lost 23 (24), won 140 (135);
    # Zerg v3 won 68 (59) lost 0 (3), Zerg v2 lost 5 (8), the replayed pros on Heartbreak Ridge lost 10 (6).
    # home_scvs alone, before its defenders were left alone in fights: lost 26, won 139. In v0.21
    "balance": lambda m: m.update(home_scvs=True, balance=True),
    # fight.py DetectRule: a lurker that burrowed near Gary's fighters is scanned (a Comsat with 50 energy).
    # On v0.21: scans in 18 of 32 lurker games, no result changed in 432 (the lurker games end undecided). In v0.22
    "detect": lambda m: m["fight"].update(before=list(m["fight"].get("before", [])) + ["detect"]),
    # posture.py force_at: the early-pool defense from 2:30 whatever the rules see (unless they switched first).
    # Learned from paired games (train/early_switch.py, v021+watch, 312 games with Zerg v3's three plans):
    # a tree over what the scout saw gains no more than this (+0.089 a game across seeds, both). On v0.21
    # (432 games, the new baseline): lost 15 (30), won 115 (118). In v0.22
    "defend_early": lambda m: m.update(defend_early=150 * 24),
    # macro.py expand_mass: the expansion once the model's odds of it, summed over Gary's decisions since
    # the last, reach one (the favorite never picks a third base). On v0.22 (432 games): won 109 (108), lost
    # 11 (11); Zerg v3 games at 20:00 on 3+ Command Centers 31 of 48 (5 of 49). Forcing the natural too
    # (at 4:00-6:00, under pressure) lost 15. In v0.23
    "expand": lambda m: m["macro"].update(expand_mass=1.0),
    # macro.py odds_mass: every one-off decision (a tech building, an add-on, research, an upgrade) once its odds,
    # summed over the decisions taken while it was possible, reach one (as expand_mass for expansions)
    "mass": lambda m: m["macro"].update(odds_mass=1.0),
    # macro.py: no bunker at the natural (the 2010s TvZ habit: against the modern pros on the ladder maps Gary-T
    # had 2 bunkers at 5-10 minutes where they have 0-1, its biggest tell there; eval/humanlike.py --pros-era modern)
    "nonatbunker": lambda m: m["macro"].update(natural_bunker=False),
    # macro.py: a depot ahead of production (each producer under 2 supply left), as the scripted builds: the model
    # leaves Gary-T supply-blocked (42 of 48 games at 4-8 minutes, L5b; pros almost never are)
    "depots": lambda m: m["macro"].update(supply_ahead=True),
    # zerg_posture.py: no alarm (the normal posture, no alarm build): while on, the alarm forbade drones, 46% of 5-12
    # minutes in round G6, and gz1 ended there with 22 drones to the pros' 27-33
    "noalarm": lambda m: (m.update(posture="normal"), m["macro"].update(builds=[])),
    # zerg_posture.py NaturalColonies: the pros' natural colonies (a sunken from 3:45, a spare colony from 5:00 morphed
    # when a push is seen), the alarm's drones not forbidden
    "colonies": lambda m: (m.update(colonies=True, alarm_drones=False), m["macro"].update(builds=["alarm", "colonies"])),
    # answers.py: the zerg's plan read from what Gary saw (a tech scan of its main at 6:30 if nothing is
    # known), and the pros' answer to it as macro goals (against lurkers: Siege Mode, tanks, vessels,
    # Irradiate, turrets one a base; against mutalisks nothing: the model's own turrets do). On v0.23 (432
    # games): won 112 (109), lost 10 (11); against Zerg v3's lurkers won 11 of 32 (7). In v0.24
    "answers": lambda m: m["macro"].update(answers=True),
    # Human game 1 (v0.25 vs the user's Zerg, 2026-10-16, lost at 12:17): answers.py against mutalisks
    # (Engineering Bay, Academy, 2 turrets in every mineral line); fight.py's pull not against flyers nor
    # more than 8 zerg supply (x2). (Not a rule to end the rush defense: how much to commit to defense is
    # learned, the user's call: pros pull back just enough to survive, and the nuances are too many to script)
    # Judged by the next games against humans; the headless pool only checks they break nothing. In v0.26
    "muta_answer": lambda m: m["macro"].setdefault("answer_options", {}).update(mutas=True),
    # fight.py StimRule: stims when the pros would (gary/policy/stim.py, learned from 400 of their TvZ games;
    # human game 2: stim researched, never used)
    "stim": lambda m: m["fight"].__setitem__("before", m["fight"]["before"] + ["stim"]),
    "repair_turrets": lambda m: m["fight"].setdefault("rule_options", {}).setdefault("repair", {}).update(turrets=3),
    "pull_cap": lambda m: m["fight"].setdefault("rule_options", {}).setdefault("pull", {}).update(ground_only=True, max_enemy=8),
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
    "post_bunker": _roles(post="bunker"),       # ...just behind the bunker by it nearest the zerg
    "post_ramp": _roles(post="ramp"),           # ...just inside the main's ramp (at the main)
    # posture.py v2: the rush bunker by the main's CC. On v0.12: replayed pros on three maps lost 3 of
    # 144 (19); the dummy's 9-pool 0 (2), its 7:00 rush 3 (0); won 3 and 3 (12 and 10)
    "bunker_cc": lambda m: m.update(posture="early_pool/v2"),
    # macro.py: roll the model's odds (a weighted shuffle) instead of taking the most likely decision.
    # Offline, on pros' states, it matched their Factory timing (63% by 8:00, theirs 62%; the
    # favorite 4%). In games on v0.14 (Fighting Spirit, replayed pros, 32 seeds): fewer SCVs (28 at
    # 12:00 against 41: the SCV is rolled less often than it's the favorite), army built 40 by 12:00
    # (57), a Factory by 8:00 in 9% (19%), lost 3 (2). Not adopted
    "dice": lambda m: m["macro"].update(decoder="dice"),
    # macro.py dice_t 0.7: dice from the odds raised to 1/0.7 (a 4.5% option about 1%, a 30% one about 30%): gt1dev's
    # dice drew 8.9 bunkers a game at 4.5% (pros build 0.5-1 by 10:00)
    "dice07": lambda m: m["macro"].update(dice_t=0.7),
    # training (docs/CONTROL.md step 7): the macro draws its next decision at temperature 1 (or
    # sample_t05: 0.5), each draw logged (g.macro.draws)
    "sample": lambda m: m["macro"].update(decoder="sample", temperature=1.0),
    "sample_t05": lambda m: m["macro"].update(decoder="sample", temperature=0.5),
    # the macro model trained from branches (train/macro_rl.py, the newest run)
    "macro_rl": lambda m: m["macro"].update(model="macro_rl/latest"),
    # the strategy selector (gary/agent/selector.py; its table: eval/selector_table.py): the build style
    # per map with the best expected score (selector_explore: a draw from each, Thompson sampling).
    # On v0.18 (fresh seeds 101+, 333 games): lost 30 (57), won 64 (57); Zerg v2 on Fighting Spirit
    # lost 3 of 72 (22). In v0.19
    "selector": lambda m: m["macro"].update(selector="play"),
    # the build style rolled per game by the pros' shares (rolled_style), not the selector's table: that was counted
    # in eval/vs_rush.py's short games against rushes on the 1.16 maps (2026-10-13), where two-Barracks marines (style
    # 4, 1.5% of pro Terrans: gas in 10% of them, no Factory) won most, and every 30-minute game on the ladder maps
    # borrowed it: Gary-T played style 4 in all 96 of rounds R50a+b (its macro gap: no gas, no Factory, mass marines)
    # terran_executor.py addon_room: a Factory, Starport or Science Facility placed only where its add-on fits
    "addonroom": lambda m: m.update(addon_room=True),
    # the pros' decision history in the macro model's inputs, fading to its own over HISTORY_FADE minutes (as gz1's): on
    # gt1's own marine-heavy states the model, calibrated on the pros', asked for Marines 42% and no mech (the macro
    # check); its own past fed back on itself
    "prohist": lambda m: m["macro"].update(history="pro"),
    # gary/policy/macro.py own_keys: with the pros' history, the supply provider's entry its own (the pros' early
    # overlord in it told Gary-Z's model one was coming: at 9/9 it gave Overlord under 1%, blocked 67% of 2:00)
    "ownsupply": lambda m: m["macro"].update(own_supply=True),
    # three registered rules off (AGENTS.md): the pros' caps per minute; SCVs capped at the bases' saturation (MAX_SCVS
    # stays); depots ahead of production
    "nocaps": lambda m: m["macro"].update(caps=False),
    "noscvcap": lambda m: m.update(scv_saturation=False),
    "nodepots": lambda m: m["macro"].update(supply_ahead=False),
    # macro.py bank_escape off: the runner-up isn't taken just because the bank is over 400 while the favorite waits
    # (with the caps off it made 341 of 1,174 turrets at 5-20 minutes, and 759 of 2,206 SCVs with the SCV cap off)
    "nobank": lambda m: m["macro"].update(bank_escape=False),
    # macro.py queue_depth 2: two of the model's waiting units per producer (pros queue in the building; one each,
    # the favorite blocked, the decoder took the runner-up: 58% of the turrets with the caps off)
    "qdepth2": lambda m: m["macro"].update(queue_depth=2),
    # terran_executor.py train_depth 2: the model's unit trained into a busy building's own queue (two deep) and paid
    # for, as pros queue; else it waits for the building and, its queue full, holds the macro (the bank escape covers
    # for that: off, 1,124 minerals banked at 5-15 minutes, pros 511)
    "gamequeue": lambda m: m.update(train_depth=2),
    # army.py hunt_advice: with the hunt off, what it would do written to the records (the rule's labels on the
    # model's own states, to imitate it)
    "huntadvice": lambda m: m["army"].update(hunt_advice=True),
    # macro.py rethink: a unit whose producers' queues are full isn't held as the goal; the model is asked again
    "rethink": lambda m: m["macro"].update(rethink=True),
    # macro.py nosave: a goal the money doesn't cover yet isn't held; the model is asked again at its next look
    "nosave": lambda m: m["macro"].update(nosave=True),
    # the stop-gaps over the layers being learned off (AGENTS.md: gt1dev and gz1dev run without them, so the models
    # own those decisions and are judged without the rules): over the army the commit model moves, the static guard,
    # the hunt (its search and scans), the overlord hunt, Gary-Z's join, sweep and press; over the macro, the pros'
    # caps, the SCV saturation cap, the bank escape, supply ahead of production. Kept: the rules that alone move units
    # no model decides (Gary-Z's overlord posts and mutalisk raid), and the decoders (the tech path's floor: without it
    # dice drew Gary-Z a Lair and a Hydralisk Den on 8 drones)
    "unscripted": lambda m: (m["army"].update(static_odds=None, hunt=False,
                                              rules=[r for r in m["army"].get("rules", ()) if r in ("posts", "raid")]),
                             m["macro"].update(caps=False, bank_escape=False, supply_ahead=False),
                             m.update(scv_saturation=False)),
    # army.py static_odds off: the static guard's veto gone (pros attack spots it forbids: 29% of their engages into
    # static defense are under 1.5 times it, held-out modern TvZ)
    "noguard": lambda m: m["army"].update(static_odds=None),
    # army.py static_beta 2: the static guard as a nudge, not a veto: a spot the group doesn't outweigh 1.5 times keeps
    # its odds times (the group / 1.5 times its defense) squared (a quarter at half of it)
    "softguard": lambda m: m["army"].update(static_beta=2.0),
    # Terran commit models on army dataset v4's 4,841 player-games (train/commit.py --out runs/commit_exp/...): with the
    # enemy's static defense as a map channel, and its control without it (the same games)
    "cmstatic": lambda m: m["army"].update(model="runs/commit_exp/TvZ_T_v4_static/model.pt"),
    "cmnostatic": lambda m: m["army"].update(model="runs/commit_exp/TvZ_T_v4_nostatic/model.pt"),
    # Terran commit models on army dataset v5 (the hunt's inputs: enemy buildings known still standing, cells unseen):
    # imitating the scripted hunt (its advice in 96 gt1dev games as engage labels, x4: where it acts, held-out, engage
    # 0.80 and its target among the five likeliest cells 0.94, the control's 0.16 and 0.12), and the control
    "cmhunt": lambda m: m["army"].update(model="runs/commit_exp/TvZ_T_v5_hunt/model.pt"),
    "cmv5": lambda m: m["army"].update(model="runs/commit_exp/TvZ_T_v5_inputs/model.pt"),
    # the same on all of army dataset v5 (16,549 TvZ games, 60 ticks a player-game): the control and the hunt imitated
    # (held-out order AUC 0.767 / 0.765, engage 0.778 / 0.776; where the hunt acts 0.81 engage, its target in the
    # five likeliest 0.91)
    "cmv5all": lambda m: m["army"].update(model="runs/commit_exp/TvZ_T_v5all_inputs/model.pt"),
    "cmhuntall": lambda m: m["army"].update(model="runs/commit_exp/TvZ_T_v5all_hunt/model.pt"),
    # all of v5 with the deployed model's inputs (no new channels): whether the v5 models trail it for the channels
    # army.py engage_bias ln 2: the commit model's engage odds doubled against move's (issue #28: at the locked pairs'
    # moments engaging beat moving by +0.81 a pair, +0.42 to +1.24, the deployed model engaging 55% of them)
    "engagebias": lambda m: m["army"].update(engage_bias=0.693),
    # army.py hold_target: a commitment decided again keeps its destination while the model still rates it among the
    # five likeliest cells (a decoder; the redraw redirected the army twice as fast as pros redirect theirs)
    "holdtarget": lambda m: m["army"].update(hold_target=True),
    # army.py unit_commits: a cluster's order leaves out units on another cluster's fresh commitment (the pipe, issue #47)
    "unitcommits": lambda m: m["army"].update(unit_commits=True),
    "cmv5plain": lambda m: m["army"].update(model="runs/commit_exp/TvZ_T_v5all_plain/model.pt"),
    # the hunt imitated only where its target has no known static defense (401 of its 644 labels; the first copied its
    # attacks into sunkens, 0.51 of its engages)
    "cmhuntu": lambda m: m["army"].update(model="runs/commit_exp/TvZ_T_v5_hunt_undef/model.pt"),
    "styleroll": lambda m: m["macro"].update(selector=None, style="roll"),
    "selector_explore": lambda m: m["macro"].update(selector="explore"),
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


# the one Gary playing Terran (2026-10-20): v0.29 with the options adopted round by round since (commit, draw,
# floor, mass, nonatbunker, depots: docs/PROGRESS.md L5-L6; lords: round G13), folded into one name; and the hunt
# (army.py) also when 4x ahead of the enemy fighters seen in the last minute, searching unseen bases with squads
# (round Jhs against G13: won 24 (21), lost 2 (3)); engages only where the group outweighs the static defense
# (round Jstatic against Jhs: the same 24 won, led at 20:00 15 (12), supply killed / lost 78.8 / 44.8 a game to
# 71.9 / 52.6); the fight roles from whole games to 30 minutes, old and modern (96 paired games against gt0.3,
# rounds B30 + B30b against R30T + R30Tb: won 54 (57), lost 9 (7), the gap per 5-minute window within noise in
# every window; adopted as the model that has seen late-game and modern fights); a unit alone and outnumbered in
# contact decided by the roles model, not the hand rules (96 paired games against gt0.4: won 57 (54), lost 8 (9),
# gap within noise; retreats doubled, though the model still rarely backs a lone unit off: its calibration is the
# next step, self-play's). Changes are judged as options against it, and an adopted one goes in here
# gt0.6: Factories, Starports and Science Facilities placed where their add-on fits (option addonroom; 96 paired games
# against gt0.5, rounds AR50a+b against B50a+b: won 49 lost 10 against 48 and 13, better in 23 and worse in 13, the
# gap even; Control Towers 0.82 -> 1.42 and Science Vessels 1.8 -> 2.9 at 16:00)
# gt0.7: keep_busy gone (it trained idle producers' last unit, half the marines at 10-20 minutes), the model's waiting
# decisions queued up to one per producer instead (48 games against gt0.6, Q60a against B60a: won 30 lost 5 against 29
# and 5, the gap -0.002; macro decisions 13,203 -> 19,347, the caps fired 0.63 -> 0.46 a decision)
# gt0.8: depots ahead of production gone, the model's own depots (96 paired games against gt0.7, ND70a+b against
# Q60a+b: won 49 lost 18 against 50 and 16, paired -0.018 +- 0.101, the gap -0.013, -0.040 to +0.012; blocked at 5-10
# minutes 14% of the time, pros 9%, gt0.7 2%; at 10-20 minutes 2%, pros 14%, gt0.7 0.2%)
# gt0.9: the model's units trained into a busy building's own queue, two deep (gamequeue; 96 paired games against
# gt0.8, GQ80a+b against ND70a+b: won 56 lost 12 against 49 and 18, better in 34 and worse in 21, +0.068 +- 0.102, the
# gap +0.018, -0.010 to +0.037; macro decisions 20.1 a minute, the pros' 20.3)
# gt0.10: the static guard a nudge, not a veto (softguard; 96 paired games against the hard guard, SG80a+b against
# ND70a+b: better in 4, worse in 9, -0.016 +- 0.035, the gap +0.001: no measurable difference, so the veto goes)
MANIFESTS["gt1"] = {**manifest_of("v029+commit+draw+floor+mass+nonatbunker+lords+hunt4+split+static+roles30+lone"
                                  "+addonroom+gamequeue+softguard"), "version": "gt0.10"}
# gt1dev: the Terran Gary that learning is trained and judged in (AGENTS.md): gt1 with the role model deciding in
# contact as well (attended pro units with their re-issues as a decision, again, Gary's own order rate as input, its
# moves kept current, several orders a turn) and the army on control groups by type, a key's units drawing their
# decisions together (again in it since its pipe check moves reach: 67% -> 72% pushed, without it none). Weaker than gt1 today
# (the state-selected suite: +110 against +141; Terran grouping 45 won 17 lost against 57 and 8 in 96 games); it
# becomes gt1 when it beats it in 96 paired games
# gt0.6dev: its role model from the first roles RL run (option rl1; 96 paired games against gt0.5dev, rounds R50a+b
# against D50a+b: won 54 lost 15 against 39 and 16, better in 34 games and worse in 20, the gap -0.046, -0.070 to
# -0.012, macro features -0.053; against gt1 better in 28 and worse in 20, the gap even, macro +0.035, -0.009 to +0.070)
# gt0.7dev: gt1's add-on room (gt0.6); gt0.8dev: gt1's queue without keep_busy (gt0.7); gt0.9dev: gt1's own depots
# (gt0.8); gt0.10dev: gt1's in-building queue (gt0.9), without the army's and the macro's stop-gaps (unscripted), the
# hunt in advice mode (huntadvice: its labels in the records, play unchanged; baseline DV90a+b); gt0.11dev: the macro
# draws from the model's odds (dice; 96 paired games against gt0.10dev, DC100a+b against DV90a+b: lost 18 against 29,
# +0.058 +- 0.083, the gap -0.073, -0.096 to -0.035, macro -0.105; the bank at 5-15 minutes 263, pros 511, 843 before)
MANIFESTS["gt1dev"] = {**manifest_of("gt1+again+refresh+incontact+batch+grouping+together+rl1+unscripted+huntadvice"
                                     "+dice"), "version": "gt0.11dev"}

# gz0.2dev: gz1 without the stop-gaps over its learned layers (unscripted): the Zerg side of TvZ self-play;
# gz0.3dev: the overlord's history its own (ownsupply: the pros' history in its inputs held their 9-supply overlord,
# so the model thought one was coming; 48 paired games against gt1dev, ZO110a against ZD110a: the Terran's score
# -0.286 +- 0.117, the Zerg winning or ahead at 30:00 in 22 of 48 (4), blocked at 3:00 1% (42%), drones at 10:00 30
# (20, pros 28))
MANIFESTS["gz1dev"] = {**manifest_of("gz1+unscripted+ownsupply"), "version": "gz0.3dev"}


def _against(m: dict, vs: dict) -> dict:
    """A manifest with its overrides for the enemy's race (its "vs" section): a layer's dict merged key by key, the
    rest replaced."""
    out = copy.deepcopy(m)
    for k, v in vs.items():
        out[k] = {**(out.get(k) or {}), **v} if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


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
    import torch
    torch.set_num_threads(1)       # its models are small: games run side by side (14 at once: 2.7x faster)
    m = manifest_of(manifest)
    g = Gary(hi, mapinfo, m["version"], verbose, m.get("race", "T"))
    if g.enemy is not None and g.enemy.race in m.get("vs", {}):
        m = _against(m, m["vs"][g.enemy.race])
    g.army_types = set(g.race.army)
    g.production_types = g.race.production if mapinfo is not None else tuple(g.race.town_halls)
    if flip is not None:
        g.flip = flip
    models: dict[str, str] = {}
    g.motor = DirectMotor(g)
    if m.get("grouping"):                     # (gary/agent/grouping.py: the control-group provider; none: no keys)
        from gary.agent.grouping import LAYOUTS, PROVIDERS as GROUPINGS
        g.motor.grouping = GROUPINGS[m["grouping"]](g)
        layout = LAYOUTS.get(g.race.letter, {})
        if m.get("grouping_layout", True) is False:      # (the army on 1-3, the buildings' keys as they were)
            from gary.agent.grouping import ARMY_KEYS
            g.motor.grouping.keys = list(ARMY_KEYS)
        elif "hall" in layout:                # (the buildings' keys moved for the army's)
            g.key_pools = (layout["hall"], layout["production"])
    g.leases.enforced = m.get("leases") == "enforced"
    if m["macro"] is not None or m["army"] is not None:
        g.executor = {"T": TerranExecutor, "Z": ZergExecutor}[g.race.letter](g)
        g.rally_plan = m.get("rally")
        g.home_scvs = m.get("home_scvs", False)
        g.builder_back = m.get("builder_back", False)
        g.balance = m.get("balance", False)
        if m.get("simcity"):
            from gary.agent.simcity import SimCity
            g.simcity = SimCity(g, m.get("simcity_door", 32))
    from gary.agent.zerg_posture import AlarmDefense, NaturalColonies, ZergPosture
    g.posture = {"normal": NormalPosture, "zerg": lambda g: ZergPosture(g, m.get("colonies", False), m.get("alarm_drones", True)),
                 "early_pool/v1": EarlyPoolPosture,
                 "early_pool/v2": EarlyPoolPostureV2, "early_pool/v3": EarlyPoolPostureV3,
                 "early_pool/v4": EarlyPoolPostureV4, "early_pool/v5": EarlyPoolPostureV5,
                 "early_pool/v6": EarlyPoolPostureV6,
                 "plan/v1": lambda g: _learned_posture(g, m, models)}[m["posture"]](g)
    if m.get("defend_early") and hasattr(g.posture, "force_at"):
        g.posture.force_at = m["defend_early"]          # (option defend_early: the rush defense from then)
    if m["macro"] is not None:
        path = model_path(m["macro"]["model"], g.matchup, g.race.letter)
        macro = load("macro", str(path))
        models["macro"] = getattr(macro, "path", "?")
        builds = {k: c(g) for k, c in (("one_base_defense", OneBaseDefense), ("ling_flood_defense", LingFloodDefense),
                                        ("alarm", AlarmDefense), ("colonies", NaturalColonies))
                  if k in m["macro"].get("builds", [])}
        if style is None and m["macro"].get("selector") and mapinfo is not None:
            # the strategy selector (gary/agent/selector.py): the build style for this map, from games counted
            import random
            from gary.agent.selector import StrategySelector, map_key
            explore = m["macro"]["selector"] == "explore"
            style = StrategySelector().choose(map_key(mapinfo), explore, random.Random(seed * 7919 + int(map_key(mapinfo), 16) % 1000))
            g.selected_style = style
        if style is None and m["macro"].get("style") == "roll":
            style = rolled_style(macro, g.matchup, g.race.letter, seed)
            g.selected_style = style
        g.macro = LearnedMacro(g, macro, style, builds, m["macro"].get("natural_bunker", False),
                               m["macro"].get("rax_by", 90 * 24),
                               m["macro"].get("decoder", "favorite"), seed, m["macro"].get("temperature", 1.0),
                               m["macro"].get("park", False), m["macro"].get("pace", False), m["macro"].get("tech_min_p"),
                               m["macro"].get("hatch_mass"), m["macro"].get("wait_min_p"))
        g.macro.expand_mass = m["macro"].get("expand_mass")
        g.macro.caps_on = m["macro"].get("caps", True)
        g.macro.bank_escape = m["macro"].get("bank_escape", True)
        g.macro.queue_depth = m["macro"].get("queue_depth", 1)
        g.macro.rethink = m["macro"].get("rethink", False)
        g.macro.nosave = m["macro"].get("nosave", False)
        g.macro.dice_t = m["macro"].get("dice_t", 1.0)
        if hasattr(g.executor, "scv_saturation"):
            g.executor.scv_saturation = m.get("scv_saturation", True)
            g.executor.train_depth = m.get("train_depth", 1)
        if hasattr(g.executor, "addon_room"):
            g.executor.addon_room = m.get("addon_room", False)
        g.macro.odds_mass = m["macro"].get("odds_mass")
        g.macro.supply_ahead = m["macro"].get("supply_ahead", False)
        g.macro.hold = m["macro"].get("hold", False)
        if m["macro"].get("history") == "pro":
            # (Gary-Z's: the pros' decision history in the model's inputs, fading to its own over HISTORY_FADE
            # minutes, so its own drone-heavy past doesn't feed back on itself)
            from gary.policy.macro import pro_history
            g.macro.tracker.history = pro_history(g.race.letter, style, g.matchup)
            g.macro.tracker.history_weight = 1.0
            g.macro.tracker.history_until = HISTORY_FADE
            if m["macro"].get("own_supply"):
                spec = g.macro.model.spec
                g.macro.tracker.own_keys = [k for k, (act, uid) in enumerate(spec.decisions) if uid == g.race.supply_provider]
        if m["macro"].get("answers"):
            from gary.agent.answers import Answers
            g.macro.answers = Answers(g, **m["macro"].get("answer_options", {}))
    if m["army"] is not None:
        if m["army"]["provider"] == "waves":
            g.army = WaveArmy(g)
        else:
            pinned = os.environ.get("GARY_TCOMMIT_MODEL" if m["army"].get("commit") else "GARY_TARMY_MODEL")                 if m["army"].get("draw") and g.race.letter == "T" else None                      # (the league loop's; the Terran's)
            army = load("army", pinned or str(model_path(m["army"]["model"], g.matchup, g.race.letter)))
            models["army"] = getattr(army, "path", "?")
            belief = None
            if m["army"].get("belief") or m["army"].get("hunt_belief"):
                from gary.policy.belief import BeliefModel
                belief = BeliefModel.load(model_path(m["army"].get("belief") or "belief/latest", g.matchup, g.race.letter))
                models["belief"] = belief.path
            g.army = LearnedArmy(g, army, PullRule(g) if m["army"].get("pull") else None, hunt=m["army"].get("hunt", False),
                                 scan=m["army"].get("scan", False), belief=belief,
                                 belief_total=m["army"].get("belief_total", False), watch=m["army"].get("watch", False),
                                 home_guard=m["army"].get("home_guard", False),
                                 belief_hold=bool(m["army"].get("belief")), hunt_belief=m["army"].get("hunt_belief", False),
                                 hold_moves=m["army"].get("hold_moves", True), hold_margin=m["army"].get("hold_margin"),
                                 draw=m["army"].get("draw", False), engage=_engage_model(g, m),
                                 commit=m["army"].get("commit", False), commit_act=m["army"].get("commit_act"),
                                 scout=m["army"].get("scout", True), types=m["army"].get("types"),
                                 rules=m["army"].get("rules", ()))
            g.army.commit_floor = m["army"].get("commit_floor")
            g.army.hunt_ahead = m["army"].get("hunt_ahead")
            g.army.hunt_split = m["army"].get("hunt_split", False)
            g.army.static_odds = m["army"].get("static_odds")
            g.army.static_beta = m["army"].get("static_beta")
            g.army.hunt_advice = m["army"].get("hunt_advice", False)
            g.army.engage_bias = m["army"].get("engage_bias", 0.0)
            g.army.hold_target = m["army"].get("hold_target", False)
            g.army.unit_commits_on = m["army"].get("unit_commits", False)
            g.army.gather_on = m["army"].get("gather", False)
    if m["fight"] is not None and m["fight"]["provider"] == "zerg":
        from gary.agent.zerg_fight import ZergFight
        g.fight = ZergFight(g, m["fight"].get("roles", True), m["fight"].get("lurkers", True), seed,
                            m["fight"].get("roles_data", "response_Z.npz"))
        models["fight"] = g.fight.model_path or "none"
    elif m["fight"] is not None:
        f = m["fight"]
        if f["provider"] == "roles" and f.get("model") is None:
            fight = None
            models["fight"] = "roles/rules_v1"   # (the chat says so: as many lines as with a model)
        elif f["provider"] in ("roles", "role_override", "role_ahead"):
            from gary.policy.roles import RoleModel
            if f.get("roles_model") and fight_path is None and fight_model is None:
                fight_path = f["roles_model"]                # (a pinned checkpoint: an RL run's, say)
            if f.get("roles_data") and fight_path is None and fight_model is None:
                from gary.policy.roles import latest_trained_on
                fight_path = latest_trained_on(f["roles_data"])
                if fight_path is None:
                    raise FileNotFoundError(f"no role model trained on {f['roles_data']}")
            fight = fight_model or RoleModel.load(fight_path or model_path(f["model"], g.matchup, g.race.letter))
            models["fight"] = fight.path
        else:
            fight = fight_model or load("fight", str(fight_path or model_path(f["model"], g.matchup, g.race.letter)))
            models["fight"] = getattr(fight, "path", "?")
        rule = lambda r: RULES[r](g, **f.get("rule_options", {}).get(r, {}))
        g.fight = FightLayer(g, fight, f["provider"], [rule(r) for r in f.get("before", [])],
                             [rule(r) for r in f.get("in_fight", [])], f.get("guard_rail", False),
                             f.get("workers", True), seed, f.get("ctrl_click", False),
                             f.get("watch_tiles", 0) * 32, f.get("roles", {}), f.get("attend", False))
    if m["arbiter"] == "drill/v1":
        g.arbiter = DrillArbiter(g, seed, interruptions)
    else:
        g.arbiter = FixedPriority(g, suspend_builds=m["arbiter"] == "fixed_priority/v2", rotate=m.get("rotate", False))
        g.announce = announcement(g.version, style, models)
    return g
