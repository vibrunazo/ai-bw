"""Build orders from replay commands, with a confidence for each item.

Replays record what players *ordered*, not what happened: a build can fail (blocked spot, no
money), be retried, or be cancelled. Without re-simulating the game, each item is classified by
the evidence available in the command stream:

  confirmed   a later command could only have been issued if this item existed
              (training zerglings proves a spawning pool; morphing a lair proves the pool too)
  ordered     ordered, not cancelled, but no later command proves it
  cancelled?  a cancel command was issued shortly after and is attributed to this item (heuristic)

Repeated orders of the same building at the same spot within a few seconds are collapsed into
one item (the last attempt), with `attempts` counting them.

Re-simulation is the ground truth; this extractor's error rate gets measured against it once
resim exists (docs/ARCHITECTURE.md §7.3).

Output: data/interim/build_orders.jsonl, one record per player per game.

Usage:
  python ingest/build_orders.py --matchup TvZ           # all TvZ games in the inventory
  python ingest/build_orders.py --matchup TvZ --limit 200
  python ingest/build_orders.py --report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from cmdcache import F, G, H, N, O, P, POS, T, TAGS, load_game
from inventory import FRAME_MS, data_root, find_screp, load_inventory

RETRY_FRAMES = 24 * 10        # same building re-ordered within 10 s ...
RETRY_TILES = 6               # ... within 6 tiles of the previous spot = a retry
CANCEL_WINDOW_FRAMES = 24 * 90
UNITS_UNTIL_S = 8 * 60        # unit production is kept for the opening only (volume)

# Buildings every player starts with: requirements on them carry no evidence.
STARTING = {"Command Center", "Hatchery", "Nexus"}

# What must exist before a command for each item can be issued (direct requirements only).
# Names follow screp's output.
REQUIRES: dict[str, list[str]] = {
    # --- Terran units
    "Marine": ["Barracks"], "Firebat": ["Barracks", "Academy"], "Medic": ["Barracks", "Academy"],
    "Ghost": ["Barracks", "Covert Ops"], "Vulture": ["Factory"],
    "Siege Tank (Tank Mode)": ["Factory", "Machine Shop"], "Goliath": ["Factory", "Armory"],
    "Wraith": ["Starport"], "Dropship": ["Starport", "Control Tower"],
    "Science Vessel": ["Starport", "Control Tower", "Science Facility"],
    "Battlecruiser": ["Starport", "Control Tower", "Physics Lab"],
    "Valkyrie": ["Starport", "Control Tower", "Armory"], "Nuclear Missile": ["Nuclear Silo"],
    # --- Terran buildings and add-ons
    "Barracks": [], "Academy": ["Barracks"], "Bunker": ["Barracks"], "Factory": ["Barracks"],
    "Starport": ["Factory"], "Armory": ["Factory"], "Science Facility": ["Starport"],
    "Missile Turret": ["Engineering Bay"], "ComSat": ["Academy"], "Machine Shop": ["Factory"],
    "Control Tower": ["Starport"], "Covert Ops": ["Science Facility"],
    "Physics Lab": ["Science Facility"], "Nuclear Silo": ["Covert Ops"],
    # --- Terran tech and upgrades
    "Stim Packs": ["Academy"], "Restoration": ["Academy"], "Optical Flare": ["Academy"],
    "U-238 Shells (Marine Range)": ["Academy"], "Tank Siege Mode": ["Machine Shop"],
    "Spider Mines": ["Machine Shop"], "Ion Thrusters (Vulture Speed)": ["Machine Shop"],
    "Charon Boosters (Goliath Range)": ["Machine Shop", "Armory"],
    "Terran Infantry Weapons": ["Engineering Bay"], "Terran Infantry Armor": ["Engineering Bay"],
    "Terran Vehicle Weapons": ["Armory"], "Terran Vehicle Plating": ["Armory"],
    "Terran Ship Weapons": ["Armory"], "Terran Ship Plating": ["Armory"],
    "Cloaking Field": ["Control Tower"], "Apollo Reactor (Wraith Energy)": ["Control Tower"],
    "Irradiate": ["Science Facility"], "EMP Shockwave": ["Science Facility"],
    "Titan Reactor (Science Vessel Energy)": ["Science Facility"],
    "Personnel Cloaking": ["Covert Ops"], "Lockdown": ["Covert Ops"], "Yamato Gun": ["Physics Lab"],
    # --- Zerg units (larva and unit morphs)
    "Zergling": ["Spawning Pool"], "Hydralisk": ["Hydralisk Den"], "Mutalisk": ["Spire"],
    "Scourge": ["Spire"], "Queen": ["Queens Nest"], "Ultralisk": ["Ultralisk Cavern"],
    "Defiler": ["Defiler Mound"], "Lurker": ["Lurker Aspect"],
    "Guardian": ["Greater Spire"], "Devourer": ["Greater Spire"],
    # --- Zerg buildings and building morphs
    "Hydralisk Den": ["Spawning Pool"], "Spire": ["Lair"], "Queens Nest": ["Lair"],
    "Ultralisk Cavern": ["Hive"], "Defiler Mound": ["Hive"], "Nydus Canal": ["Hive"],
    "Lair": ["Spawning Pool"], "Hive": ["Lair", "Queens Nest"],
    "Sunken Colony": ["Creep Colony", "Spawning Pool"],
    "Spore Colony": ["Creep Colony", "Evolution Chamber"], "Greater Spire": ["Spire", "Hive"],
    # --- Zerg tech and upgrades
    "Lurker Aspect": ["Hydralisk Den", "Lair"], "Ensnare": ["Queens Nest"],
    "Spawn Broodlings": ["Queens Nest"], "Plague": ["Defiler Mound"], "Consume": ["Defiler Mound"],
    "Metabolic Boost (Zergling Speed)": ["Spawning Pool"],
    "Adrenal Glands (Zergling Attack)": ["Spawning Pool", "Hive"],
    "Muscular Augments (Hydralisk Speed)": ["Hydralisk Den", "Lair"],
    "Grooved Spines (Hydralisk Range)": ["Hydralisk Den"],
    "Pneumatized Carapace (Overlord Speed)": ["Lair"], "Antennae (Overlord Sight)": ["Lair"],
    "Ventral Sacs (Overlord Transport)": ["Lair"],
    "Zerg Melee Attacks": ["Evolution Chamber"], "Zerg Missile Attacks": ["Evolution Chamber"],
    "Zerg Carapace": ["Evolution Chamber"], "Zerg Flyer Attacks": ["Spire"],
    "Zerg Flyer Carapace": ["Spire"], "Anabolic Synthesis (Ultralisk Speed)": ["Ultralisk Cavern"],
    "Chitinous Plating (Ultralisk Armor)": ["Ultralisk Cavern"],
    "Gamete Meiosis (Queen Energy)": ["Queens Nest"], "Defiler Energy": ["Defiler Mound"],
    # --- Protoss units
    "Zealot": ["Gateway"], "Dragoon": ["Gateway", "Cybernetics Core"],
    "High Templar": ["Gateway", "Templar Archives"], "Dark Templar": ["Gateway", "Templar Archives"],
    "Shuttle": ["Robotics Facility"], "Reaver": ["Robotics Facility", "Robotics Support Bay"],
    "Observer": ["Robotics Facility", "Observatory"], "Scout": ["Stargate"], "Corsair": ["Stargate"],
    "Carrier": ["Stargate", "Fleet Beacon"], "Arbiter": ["Stargate", "Arbiter Tribunal"],
    # --- Protoss buildings
    "Gateway": ["Pylon"], "Forge": ["Pylon"], "Cybernetics Core": ["Gateway"],
    "Photon Cannon": ["Forge"], "Shield Battery": ["Gateway"],
    "Robotics Facility": ["Cybernetics Core"], "Stargate": ["Cybernetics Core"],
    "Citadel of Adun": ["Cybernetics Core"], "Templar Archives": ["Citadel of Adun"],
    "Observatory": ["Robotics Facility"], "Robotics Support Bay": ["Robotics Facility"],
    "Fleet Beacon": ["Stargate"], "Arbiter Tribunal": ["Templar Archives", "Stargate"],
    # --- Protoss tech and upgrades
    "Psionic Storm": ["Templar Archives"], "Hallucination": ["Templar Archives"],
    "Maelstrom": ["Templar Archives"], "Mind Control": ["Templar Archives"],
    "Recall": ["Arbiter Tribunal"], "Stasis Field": ["Arbiter Tribunal"],
    "Disruption Web": ["Fleet Beacon"],
    "Singularity Charge (Dragoon Range)": ["Cybernetics Core"],
    "Leg Enhancement (Zealot Speed)": ["Citadel of Adun"],
    "Protoss Ground Weapons": ["Forge"], "Protoss Ground Armor": ["Forge"],
    "Protoss Plasma Shields": ["Forge"], "Protoss Air Weapons": ["Cybernetics Core"],
    "Protoss Air Armor": ["Cybernetics Core"], "Scarab Damage": ["Robotics Support Bay"],
    "Reaver Capacity": ["Robotics Support Bay"],
    "Gravitic Drive (Shuttle Speed)": ["Robotics Support Bay"],
    "Sensor Array (Observer Sight)": ["Observatory"],
    "Gravitic Booster (Observer Speed)": ["Observatory"],
    "Khaydarin Amulet (Templar Energy)": ["Templar Archives"],
    "Argus Talisman (Dark Archon Energy)": ["Templar Archives"],
    "Carrier Capacity": ["Fleet Beacon"], "Gravitic Thrusters (Scout Speed)": ["Fleet Beacon"],
    "Khaydarin Core (Arbiter Energy)": ["Arbiter Tribunal"],
}

# Using an ability proves its research finished. Command type or targeted-order name -> tech/building.
USE_EVIDENCE = {
    "Stim": "Stim Packs", "Siege": "Tank Siege Mode",
    "PlaceMine": "Spider Mines", "CastScannerSweep": "ComSat", "CastIrradiate": "Irradiate",
    "CastEMPShockwave": "EMP Shockwave", "CastLockdown": "Lockdown",
    "CastPsionicStorm": "Psionic Storm", "CastMaelstrom": "Maelstrom",
    "CastMindControl": "Mind Control", "CastRecall": "Recall", "CastStasisField": "Stasis Field",
    "CastDisruptionWeb": "Disruption Web", "CastConsume": "Consume", "CastPlague": "Plague",
    "CastEnsnare": "Ensnare", "CastSpawnBroodlings": "Spawn Broodlings",
}

# Gas starts at 0, and the client refuses orders it can't afford, so ordering anything that costs
# gas proves a gas building existed. Everything not listed here costs gas.
GAS_BUILDING = {"T": "Refinery", "Z": "Extractor", "P": "Assimilator"}
GAS_FREE = {
    "SCV", "Marine", "Vulture", "Supply Depot", "Barracks", "Refinery", "Engineering Bay", "Bunker",
    "Missile Turret", "Command Center", "Academy",
    "Drone", "Overlord", "Zergling", "Hatchery", "Extractor", "Spawning Pool", "Evolution Chamber",
    "Creep Colony", "Sunken Colony", "Spore Colony",
    "Probe", "Zealot", "Pylon", "Gateway", "Forge", "Nexus", "Assimilator", "Photon Cannon",
    "Cybernetics Core", "Shield Battery",
}

# The building a command is issued from. When a unit is trained, the selected building's ID is in
# the preceding Select command; each new building ID of a type is one more instance of that type.
# This is the only command-level evidence for expansions and extra production buildings.
PRODUCER = {
    "SCV": "Command Center", "ComSat": "Command Center", "Nuclear Silo": "Command Center",
    "Marine": "Barracks", "Firebat": "Barracks", "Medic": "Barracks", "Ghost": "Barracks",
    "Vulture": "Factory", "Siege Tank (Tank Mode)": "Factory", "Goliath": "Factory",
    "Machine Shop": "Factory", "Wraith": "Starport", "Dropship": "Starport",
    "Science Vessel": "Starport", "Battlecruiser": "Starport", "Valkyrie": "Starport",
    "Control Tower": "Starport", "Covert Ops": "Science Facility", "Physics Lab": "Science Facility",
    "Probe": "Nexus", "Zealot": "Gateway", "Dragoon": "Gateway", "High Templar": "Gateway",
    "Dark Templar": "Gateway", "Shuttle": "Robotics Facility", "Reaver": "Robotics Facility",
    "Observer": "Robotics Facility", "Scout": "Stargate", "Corsair": "Stargate",
    "Carrier": "Stargate", "Arbiter": "Stargate",
    "Sunken Colony": "Creep Colony", "Spore Colony": "Creep Colony",
    # Zerg: larva hide which hatchery they came from, but these are issued from the hatchery itself
    "Lair": "Hatchery", "Burrowing": "Hatchery", "Pneumatized Carapace (Overlord Speed)": "Hatchery",
    "Antennae (Overlord Sight)": "Hatchery", "Ventral Sacs (Overlord Transport)": "Hatchery",
}
# Zerg rally points can only be set from hatcheries (lairs and hives keep the hatchery's ID).
ZERG_RALLY_ORDERS = {"RallyPointTile", "RallyPointUnit"}
# Commands only a mobile unit accepts. A drone that gets one after being sent to build is still a
# drone, so that building didn't happen. (Right Click is excluded: on a hatchery it sets the rally.)
UNIT_ONLY_ORDERS = {"Move", "AttackMove", "Attack1", "Harvest1", "Patrol", "Follow", "HoldPosition"}
UNIT_ONLY_CMDS = {"Return Cargo", "Burrow", "Hold Position"}
SELECT_CMDS = {"Select", "Select Add", "Select Remove"}

KIND_OF_CMD = {
    "Build": "build", "Building Morph": "building_morph", "Unit Morph": "unit", "Train": "unit",
    "Tech": "tech", "Upgrade": "upgrade",
}
STRUCTURE_KINDS = {"build", "building_morph"}


def secs(frame: int) -> int:
    return round(frame * FRAME_MS / 1000)


def unique12(tags: list[int]) -> list[int]:
    """BW selections and hotkey groups hold at most 12 distinct units."""
    return list(dict.fromkeys(tags))[:12]


def requirements(name: str, race: str | None) -> list[str]:
    reqs = list(REQUIRES.get(name, []))
    if race in GAS_BUILDING and name not in GAS_FREE and name in REQUIRES:
        reqs.append(GAS_BUILDING[race])
    return reqs


def mmss(frame: int) -> str:
    return f"{secs(frame) // 60}:{secs(frame) % 60:02d}"


def extract_player(cmds: list[tuple], race: str | None) -> tuple[list[dict], dict]:
    """Build-order items for one player's commands, plus counters for the summary."""
    items: list[dict] = []
    stats = Counter()
    selection: list[int] = []
    hotkeys: dict[int, list[int]] = {}
    last_of: dict[tuple[str, str], dict] = {}     # (kind, name) -> latest stored item
    by_builder: dict[int, dict] = {}              # Zerg drone ID -> the building it was sent to make
    by_name: dict[str, list[dict]] = defaultdict(list)
    structures: list[dict] = []
    producer_first_use: dict[str, dict[int, int]] = defaultdict(dict)  # type -> tag -> first frame
    larva_source: int | None = None   # a building selected right before switching to its larva
    last_single: tuple[int, int] | None = None
    for c in cmds:
        t, frame, order = c[T], c[F], c[O]
        if t == "Select":
            tags = list(c[TAGS] or [])
            # Zerg: selecting a hatchery and pressing S sends a one-unit Select, then a Select of its
            # larva. If a morph follows, that one unit was a hatchery (a drone has no larva).
            larva_source = (last_single[0] if race == "Z" and last_single and frame - last_single[1] <= 12
                            and last_single[0] not in tags else None)
            last_single = (tags[0], frame) if len(tags) == 1 else None
            selection = tags
        elif t == "Select Add":
            selection = unique12(selection + list(c[TAGS] or []))
        elif t == "Select Remove":
            selection = [u for u in selection if u not in set(c[TAGS] or [])]
        elif t == "Hotkey":
            group, op = c[G], c[H]
            if op == "Assign":
                hotkeys[group] = list(selection)
            elif op == "Add":
                hotkeys[group] = unique12(hotkeys.get(group, []) + selection)
            elif op == "Select":
                selection = list(hotkeys.get(group, []))
        if t == "Unit Morph" and larva_source is not None:
            producer_first_use["Hatchery"].setdefault(larva_source, frame)
            stats["larva_source_seen"] += 1
        if t not in ("Select", "Unit Morph"):
            larva_source = None
        if race == "Z" and t == "Targeted Order" and order in ZERG_RALLY_ORDERS:
            for tag in selection:
                producer_first_use["Hatchery"].setdefault(tag, frame)
        if race == "Z" and by_builder and (t in UNIT_ONLY_CMDS or (
                t == "Targeted Order" and order in UNIT_ONLY_ORDERS)):
            for tag in selection:
                pending = by_builder.get(tag)
                if pending and pending["conf"] == "ordered":
                    pending["conf"] = "not built"
                    pending["evidence"] = f"drone given a unit order at {mmss(frame)}"
                    stats["zerg_drone_redirected"] += 1
        kind = KIND_OF_CMD.get(t)
        if t == "Build" and order == "BuildNydusExit":
            # placing the exit is done from the canal itself: proof the canal exists, not a new build
            items.append({"frame": frame, "kind": "use", "name": "use:Nydus Canal",
                          "_evidence_only": True, "_proves": "Nydus Canal"})
            continue
        if kind:
            name = c[N] or "?"
            ptype = PRODUCER.get(name)
            if ptype and (kind != "build" or order == "PlaceAddon"):
                for tag in selection:
                    producer_first_use[ptype].setdefault(tag, frame)
            if kind == "unit" and secs(frame) > UNITS_UNTIL_S:
                # still counts as evidence for requirements, but isn't stored
                items.append({"frame": frame, "kind": kind, "name": name, "_evidence_only": True})
                continue
            pos = c[POS]
            prev = last_of.get((kind, name))
            is_retry = (
                prev is not None and kind != "unit" and frame - prev["frame"] <= RETRY_FRAMES
                and (pos is None or prev.get("pos") is None
                     or max(abs(pos[0] - prev["pos"][0]), abs(pos[1] - prev["pos"][1])) <= RETRY_TILES)
            )
            if is_retry:
                prev.update(frame=frame, pos=pos, attempts=prev.get("attempts", 1) + 1)
                stats["retries_collapsed"] += 1
                continue
            item = {"frame": frame, "kind": kind, "name": name, "conf": "ordered"}
            if pos:
                item["pos"] = pos
            if race == "Z" and kind == "build" and len(selection) == 1:
                # A drone becomes the building and keeps its ID. If the same drone is later sent
                # to build something else, this building never happened (failed or cancelled).
                drone = selection[0]
                earlier = by_builder.get(drone)
                if earlier and earlier["conf"] != "confirmed":
                    earlier["conf"] = "not built"
                    earlier["evidence"] = f"drone reused at {mmss(frame)}"
                    stats["zerg_drone_reused"] += 1
                item["_builder"] = drone
                by_builder[drone] = item
            items.append(item)
            last_of[(kind, name)] = item
            by_name[name].append(item)
            if kind in STRUCTURE_KINDS:
                structures.append(item)
        elif (used := USE_EVIDENCE.get(t) or USE_EVIDENCE.get(order or "")):
            # a pseudo-event: proves `used` exists, never stored
            items.append({"frame": frame, "kind": "use", "name": f"use:{used}",
                          "_evidence_only": True, "_proves": used})
        elif t in ("Cancel Build", "Cancel Morph"):
            # Attribute to the latest unconfirmed structure (or, for morphs, building morph) in the window.
            target = None
            for i in reversed(structures):
                if frame - i["frame"] > CANCEL_WINDOW_FRAMES:
                    break
                if i["conf"] == "ordered":
                    target = i
                    break
            if target:
                target["conf"] = "cancelled?"
                stats["cancels_attributed"] += 1
            else:
                stats["cancels_unattributed"] += 1

    # Requirement evidence: any command for X proves X's requirements existed at that time.
    # Events are in time order, so once a requirement is confirmed, later events add nothing.
    confirmed_req: set[str] = set()
    for ev in items:
        reqs = [ev["_proves"]] if "_proves" in ev else requirements(ev["name"], race)
        for req in reqs:
            if req in STARTING or req in confirmed_req:
                continue
            candidates = [i for i in by_name.get(req, []) if i["frame"] < ev["frame"]]
            if not candidates:
                stats["requirement_without_order"] += 1  # e.g. ordered before the replay saw it
                continue
            confirmed_req.add(req)
            if any(i["conf"] == "confirmed" for i in candidates):
                continue
            # Prefer the earliest non-cancelled order; evidence overrides a heuristic cancel.
            target = next((i for i in candidates if i["conf"] == "ordered"),
                          next((i for i in candidates if i["conf"] == "cancelled?"), candidates[0]))
            if target["conf"] == "cancelled?":
                stats["cancel_overridden_by_evidence"] += 1
            target["conf"] = "confirmed"
            target["evidence"] = f"{ev['name']} at {mmss(ev['frame'])}"

    # Producer evidence: the k-th distinct building ID used to produce from proves the k-th order of
    # that building type (the first ID is the starting building for Command Center / Nexus).
    for ptype, tags in producer_first_use.items():
        orders = list(by_name.get(ptype, []))
        # Zerg: the building has its drone's ID, so match it exactly. If the same drone was sent
        # more than once, the last order before the ID acts as a building is the one that happened.
        for builder in {o.get("_builder") for o in orders} - {None}:
            used_at = tags.get(builder)
            if used_at is None:
                continue
            before = [o for o in orders if o.get("_builder") == builder and o["frame"] < used_at]
            if before:
                o = before[-1]
                del tags[builder]
                if o["conf"] in ("not built", "cancelled?"):
                    stats["negative_overridden_by_evidence"] += 1
                o["conf"] = "confirmed"
                o["evidence"] = f"used as {ptype} at {mmss(used_at)}"
        orders = [o for o in orders if o["conf"] not in ("confirmed", "not built")]
        uses = sorted(tags.values())
        if ptype in STARTING:
            uses = uses[1:]
        for k, used_at in enumerate(uses):
            if k >= len(orders):
                stats["producer_without_order"] += 1
                break
            target = orders[k]
            if target["frame"] >= used_at or target["conf"] == "confirmed":
                continue
            if target["conf"] == "cancelled?":
                stats["cancel_overridden_by_evidence"] += 1
            target["conf"] = "confirmed"
            target["evidence"] = f"produced from at {mmss(used_at)}"

    out = []
    for i in items:
        if i.get("_evidence_only"):
            continue
        i["t"] = secs(i["frame"])
        i.pop("_builder", None)
        out.append(i)
    return out, stats


def extract_game(screp: str, raw_root: Path, cache_root: Path, rec: dict) -> tuple[list[dict], Counter]:
    game = load_game(screp, raw_root / rec["rel_path"], rec["sha1"], cache_root)
    players = [p for p in game["players"] if not p["observer"] and p["type"] == "Human"]
    by_player = defaultdict(list)
    for c in game["cmds"]:
        by_player[c[P]].append(c)
    inv_players = {p["name"]: p for p in rec["players"]}
    out, total = [], Counter()
    for p in players:
        ip = inv_players.get(p["name"], {})
        items, stats = extract_player(by_player.get(p["id"], []), ip.get("race"))
        total += stats
        opp = next((q for q in rec["players"] if q["name"] != p["name"]), {})
        out.append({
            "sha1": rec["sha1"], "rel_path": rec["rel_path"], "source": rec["source"],
            "matchup": rec["matchup"], "map": rec["map"], "map_hash": rec["map_hash"],
            "duration_s": rec["duration_s"], "player": p["name"], "race": ip.get("race"),
            "opponent_race": opp.get("race"), "start_clock": ip.get("start_clock"),
            "opponent_start_clock": opp.get("start_clock"),
            "won": (rec["winner_team"] == p["team"]) if rec.get("winner_team") else None,
            "items": items,
        })
    return out, total


def select_games(args: argparse.Namespace) -> list[dict]:
    recs = load_inventory(data_root() / "interim" / "inventory.jsonl").values()
    games = [r for r in recs if r.get("ok") and not r.get("dup_of") and r.get("is_1v1_human")
             and not r.get("has_bot_suspect") and not r.get("labeled_bot_game")]
    if args.matchup:
        games = [r for r in games if r["matchup"] == args.matchup]
    if args.source:
        games = [r for r in games if r["source"] == args.source]
    games.sort(key=lambda r: r["rel_path"])
    return games[: args.limit] if args.limit else games


def build(args: argparse.Namespace) -> None:
    screp = find_screp(args.screp)
    raw_root = data_root() / "raw"
    cache_root = data_root() / "interim" / "cmdcache"
    games = select_games(args)
    out = data_root() / "interim" / "build_orders.jsonl"
    # Incremental: games already extracted are skipped (--force redoes them), and the rows of games
    # outside this run's selection (other matchups, other sources) are kept.
    kept, done, selected = [], set(), {g["sha1"] for g in games}
    if out.exists():
        for r in map(json.loads, out.open(encoding="utf-8")):
            done.add(r["sha1"])
            if r["sha1"] not in selected or not args.force:
                kept.append(r)
    if not args.force:
        games = [g for g in games if g["sha1"] not in done]
    print(f"{len(games)} games selected ({len(done)} already extracted)", flush=True)
    rows, totals, failed = [], Counter(), 0
    # Processes, not threads: the evidence pass is pure Python and would serialize on the GIL.
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_game, screp, raw_root, cache_root, g): g for g in games}
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                r, s = fut.result()
                rows += r
                totals += s
            except Exception as e:
                failed += 1
                print(f"  failed: {futures[fut]['rel_path']}: {e!r}"[:200])
            if n % 2000 == 0:
                print(f"  {n}/{len(games)}", flush=True)
    rows = kept + rows
    rows.sort(key=lambda r: (r["rel_path"], r["player"] or ""))
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(rows)} player-games, {failed} games failed)")
    print("  " + ", ".join(f"{k}: {v}" for k, v in sorted(totals.items())))


def report(args: argparse.Namespace) -> None:
    path = data_root() / "interim" / "build_orders.jsonl"
    if not path.exists():
        sys.exit("no build orders yet: run without --report first")
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    print(f"{len(rows)} player-games")
    # First instance of each structure/tech per player-game: what a build order is about.
    by = defaultdict(Counter)
    for r in rows:
        seen = set()
        for i in r["items"]:
            if i["kind"] in STRUCTURE_KINDS | {"tech", "upgrade"} and i["name"] not in seen:
                seen.add(i["name"])
                by[(r["race"], i["name"])][i["conf"]] += 1
    print("first instance of each item per player-game")
    print(f"\n{'race':<4} {'item':<40} {'n':>7} {'confirmed':>10} {'ordered':>8} {'cancelled?':>10} {'not built':>9}")
    for (race, name), c in sorted(by.items(), key=lambda kv: (kv[0][0] or "", -sum(kv[1].values()))):
        n = sum(c.values())
        if n < args.min_count:
            continue
        print(f"{race or '?':<4} {name:<40} {n:>7} {c['confirmed'] / n:>9.0%} "
              f"{c['ordered'] / n:>8.0%} {c['cancelled?'] / n:>10.0%} {c['not built'] / n:>9.0%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--matchup", help="e.g. TvZ")
    ap.add_argument("--source", help="e.g. stardata")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--min-count", type=int, default=50, help="report: hide rarer items")
    ap.add_argument("--force", action="store_true", help="re-extract games already in build_orders.jsonl")
    ap.add_argument("--screp")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    report(args) if args.report else build(args)


if __name__ == "__main__":
    main()
