"""Play Gary: headless in OpenBW (fast, deterministic, saves a replay and Gary's POV log), or live in a
StarCraft: Remastered client through the bridge. Any version (a manifest, gary/agent/manifests.py).

    python -m gary.play --gary v027 --map path/to/map.scx --minutes 12
    python -m gary.play --gary v027 --live
"""

from __future__ import annotations

import argparse
import json
import time

from gary import commands as C
from gary.agent.manifests import build_gary, manifest_name, manifest_of
from gary.env import LIVE_COMMAND_DELAY, Game, GameError
from gary.decision_log import LOG
from gary.interface import PROFILES, HumanInterface
from gary.mapinfo import MapInfo

CC, SCV, MARINE, RAX, DEPOT = C.COMMAND_CENTER, C.SCV, C.MARINE, C.BARRACKS, C.SUPPLY_DEPOT


DEFAULTS = {"T": "gt1", "Z": "gz1"}   # the Gary each race plays
HALLS = {C.COMMAND_CENTER: "T", C.HATCHERY: "Z", C.NEXUS: "P"}


def slot_of(game, race: str) -> int:
    """The slot whose start town hall is this race's. (Slot order is the engine's, not the order of the races
    list, so bind by what's actually on the map.)"""
    obs = game.observe()
    slots = {p["slot"] for p in obs["players"]}
    for u in obs["units"]:
        if HALLS.get(u["type"]) == race and u["owner"] in slots:
            return u["owner"]
    raise SystemExit(f"no {race} player in this game")


def report(hi: HumanInterface, game) -> None:
    obs = game.observe()
    me = next(p for p in obs["players"] if p["slot"] == hi.slot)
    count = lambda t: sum(1 for u in obs["units"] if u["owner"] == hi.slot and u["type"] == t)
    print(f"{hi.frame // 1440}:00  supply {me['supply_used']:.0f}/{me['supply_max']:.0f}  "
          f"minerals {me['minerals']}  SCVs {count(SCV)}  marines {count(MARINE)}  "
          f"CC {count(CC)}  rax {count(RAX)}  depots {count(DEPOT)}  "
          f"APM {round(hi.stats['actions'] / max(1, hi.frame / 1440))}", flush=True)


def play(map_path: str, minutes: float, seed: int | None, save: str, make_bot, name: str,
         command_delay: int = LIVE_COMMAND_DELAY) -> None:
    """Play against an idle opponent on a headless OpenBW game (gary/env.py). command_delay: frames
    from sending a command to it running, as in a networked game. make_bot(hi, mapinfo, race) builds Gary."""
    race = make_bot.race
    with Game.new(map_path, [race, "Z" if race != "Z" else "T"], [name, "Idle"], seed=seed, command_delay=command_delay) as game:
        mapinfo = MapInfo.from_game(game)
        hi = HumanInterface(game, slot_of(game, race), PROFILES["pro"], seed=1)
        hi.side = race
        gary = make_bot(hi, mapinfo, race)
        end = int(minutes * 60 * 24)
        while hi.frame < end:
            gary.act()
            hi.step(2)
            if hi.frame % (24 * 60) < 2:
                report(hi, game)
        game.save_replay(save)
        pov = save[:-4] + ".pov.jsonl" if save.endswith(".rep") else save + ".pov.jsonl"
        hi.save_pov(pov)
        gary.save_trace(pov[:-len(".pov.jsonl")] + ".trace.jsonl")
        LOG.save_records(pov[:-len(".pov.jsonl")] + ".records.jsonl")
        print(f"saved {save} and {pov} (and Gary's trace and records)")


def play_live(pipe: str, minutes: float, profile: str, pov: str | None, make_bot,
              snapshots: str | None = None) -> None:
    """Play in a live StarCraft: Remastered client through the bridge (adapters/scr_bridge).

    Gary plays the client's local player (the bridge only accepts that slot), as its race's Gary
    (Protoss: none yet, waited out). act() latency is calibrated from the adapter's reported latency_frames so the
    human interface lands actions when they really take effect. The client writes its own
    replay on match end. snapshots: a file for what the bridge showed Gary each second.
    """
    from gary.scr_env import ScrGame
    game = ScrGame.connect(pipe)
    hi: HumanInterface | None = None
    gary = None
    snaps = None
    try:
        # wait for a game with Gary's Command Center, polling the bridge's status 4 times a second, so
        # Gary starts within a few frames of the match (step() at the menus waited up to 120 s, its
        # frame target left over from the last match); a match that isn't Terran is waited out
        said = None
        while True:
            st = game.status()
            slot = st.get("local_player", -1)
            state = "menus"
            if st.get("in_game") and slot >= 0:
                obs = game.observe()
                mains = [u["type"] for u in obs["units"] if u["owner"] == slot
                         and u["type"] in (CC, C.NEXUS, C.HATCHERY)]
                race = HALLS.get(mains[0]) if mains else None
                if race in make_bot.races:
                    break
                if mains:
                    state = "Protoss"
            if state != said:
                said = state
                print("waiting for a live game (start one)..." if state == "menus" else
                      f"the match running has the local player as {state}: no Gary plays it yet; waiting for the next match",
                      flush=True)
            time.sleep(0.25)
        st = game.status()
        mapinfo = MapInfo.from_game(game)
        hi = HumanInterface(game, slot, PROFILES[profile], seed=1)
        hi.side = race
        print(f"playing slot {slot} | act() latency calibrated to {hi.act_latency_frames} frames "
              f"(adapter reported latency_frames={st.get('latency_frames')})", flush=True)
        gary = make_bot(hi, mapinfo, race)
        end = hi.frame + int(minutes * 60 * 24)
        minute = hi.frame // 1440
        snaps = open(snapshots, "w", encoding="utf-8") if snapshots else None
        next_snap = 0
        while hi.frame < end:
            gary.act()
            hi.step(2)                       # waits for the live game to advance (its clock can jump)
            if snaps and hi.frame >= next_snap:
                # what the bridge showed Gary, once a second, for parity with OpenBW re-simulating the
                # replay (adapters/scr_bridge/tools/parity.py)
                obs = hi.history[-1][1]
                snaps.write(json.dumps({"frame": obs["frame"], "players": obs["players"], "units": obs["units"]},
                                       separators=(",", ":")) + "\n")
                next_snap = hi.frame + 24
            if hi.frame // 1440 > minute:
                minute = hi.frame // 1440
                report(hi, game)
                if pov:                      # saved every minute: a client or console that dies keeps the game so far
                    _save_live(hi, gary, pov)
    except GameError as e:
        print(f"game over / bridge lost: {e}")
    finally:
        if snaps:
            snaps.close()
        if pov and hi is not None:
            try:
                _save_live(hi, gary, pov)
                print(f"saved {pov}")
            except Exception as e:
                print(f"could not save pov: {e}")
        game.close()


def _save_live(hi: HumanInterface, gary, pov: str) -> None:
    """The live game's POV log, and Gary's trace and records next to it (gary/decision_log.py)."""
    hi.save_pov(pov)
    if gary is not None and pov.endswith(".pov.jsonl"):
        gary.save_trace(pov[:-len(".pov.jsonl")] + ".trace.jsonl")
        LOG.save_records(pov[:-len(".pov.jsonl")] + ".records.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--terran", default=DEFAULTS["T"], type=manifest_name, help="the Gary that plays Terran (a manifest)")
    ap.add_argument("--zerg", default=DEFAULTS["Z"], type=manifest_name, help="the Gary that plays Zerg (a manifest)")
    ap.add_argument("--race", default="T", choices=["T", "Z"], help="headless: the race Gary plays")
    ap.add_argument("--map", help="a .scm/.scx map (or a pre-1.18 replay) for headless play")
    ap.add_argument("--minutes", type=float, default=12)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--save", help="replay path (default gary_<version>.rep)")
    ap.add_argument("--command-delay", type=int, default=LIVE_COMMAND_DELAY,
                    help=f"headless: frames before a command runs (default {LIVE_COMMAND_DELAY}, as measured live)")
    ap.add_argument("--style", help="Gary's build style, by id or name (gary/styles.py), over its manifest's choice; "
                    "--list-styles prints them")
    ap.add_argument("--list-styles", action="store_true", help="print the build styles Gary's race has, per matchup, and exit")
    ap.add_argument("--fight-model", help="replace the version's fight model with this checkpoint")
    ap.add_argument("--quiet", action="store_true", help="don't print each decision")
    ap.add_argument("--live", action="store_true", help="play in a live SC:R client via the bridge")
    ap.add_argument("--pipe", default=r"\\.\pipe\gary_scr", help="bridge named pipe (with --live)")
    ap.add_argument("--profile", default="pro", choices=sorted(PROFILES))
    ap.add_argument("--pov", help="where to save the point-of-view log (live)")
    ap.add_argument("--snapshots", help="live: save what the bridge showed Gary each second (for tools/parity.py)")
    args = ap.parse_args()
    manifests = {"T": args.terran, "Z": args.zerg}
    if args.list_styles:
        from gary.styles import describe
        for mu in ("TvZ", "TvT", "ZvZ", "PvT", "PvZ", "PvP"):
            if args.race in mu:
                print(describe(mu, args.race))
        return

    def make(hi, mapinfo, race):
        return build_gary(hi, mapinfo, manifests[race], args.style, verbose=not args.quiet, fight_path=args.fight_model)
    make.races, make.race = set(manifests), args.race
    if args.live:
        play_live(args.pipe, args.minutes, args.profile, args.pov or "gary_live.pov.jsonl", make, args.snapshots)
    else:
        if not args.map:
            ap.error("--map is required unless --live")
        version = manifest_of(manifests[args.race])["version"]
        play(args.map, args.minutes, args.seed, args.save or f"gary_{manifests[args.race]}.rep", make,
             f"Gary {version} ({args.race})", command_delay=args.command_delay)


if __name__ == "__main__":
    main()
