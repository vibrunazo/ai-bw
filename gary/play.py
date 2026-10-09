"""Play Gary: headless in OpenBW (fast, deterministic, saves a replay and Gary's POV log), or live in a
StarCraft: Remastered client through the bridge. Any version (a manifest, gary/agent/manifests.py).

    python -m gary.play --gary v016 --map path/to/map.scx --minutes 12 --style 1
    python -m gary.play --gary v016 --live --style 1
"""

from __future__ import annotations

import argparse

from gary import commands as C
from gary.agent.manifests import build_gary, manifest_name, manifest_of
from gary.env import LIVE_COMMAND_DELAY, Game, GameError
from gary.interface import PROFILES, HumanInterface
from gary.mapinfo import MapInfo

CC, SCV, MARINE, RAX, DEPOT = C.COMMAND_CENTER, C.SCV, C.MARINE, C.BARRACKS, C.SUPPLY_DEPOT


def terran_slot(game) -> int:
    """The slot that owns a Command Center: Gary plays Terran. (Slot order is the engine's, not
    the order of the races list, so bind by what's actually on the map.)"""
    obs = game.observe()
    slots = {p["slot"] for p in obs["players"]}
    for u in obs["units"]:
        if u["type"] == CC and u["owner"] in slots:
            return u["owner"]
    raise SystemExit("no Terran (Command Center) player in this game")


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
    """Play against an idle Zerg on a headless OpenBW game (gary/env.py). command_delay: frames
    from sending a command to it running, as in a networked game. make_bot(hi, mapinfo) builds Gary."""
    with Game.new(map_path, ["T", "Z"], [name, "Idle (Z)"], seed=seed, command_delay=command_delay) as game:
        mapinfo = MapInfo.from_game(game)
        hi = HumanInterface(game, terran_slot(game), PROFILES["pro"], seed=1)
        gary = make_bot(hi, mapinfo)
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
        print(f"saved {save} and {pov} (and Gary's trace)")


def play_live(pipe: str, minutes: float, profile: str, pov: str | None, make_bot) -> None:
    """Play in a live StarCraft: Remastered client through the bridge (adapters/scr_bridge).

    Gary plays the client's local player (the bridge only accepts that slot), so it must be
    Terran. act() latency is calibrated from the adapter's reported latency_frames so the
    human interface lands actions when they really take effect. The client writes its own
    replay on match end.
    """
    from gary.scr_env import ScrGame
    game = ScrGame.connect(pipe)
    hi: HumanInterface | None = None
    gary = None
    try:
        # wait for a game to start and for Gary's Command Center to exist
        while True:
            st = game.status()
            slot = st.get("local_player", -1)
            if st.get("in_game") and slot >= 0:
                obs = game.observe()
                mains = [u["type"] for u in obs["units"] if u["owner"] == slot
                         and u["type"] in (CC, C.NEXUS, C.HATCHERY)]
                if CC in mains:
                    break
                if mains:   # a game is running, but Gary isn't Terran in it
                    raise SystemExit(
                        f"the local player is {'Protoss' if mains[0] == C.NEXUS else 'Zerg'}: "
                        f"Gary plays Terran. Start a Terran game (melee assigns a random race: "
                        f"end this one and try again, or use auto_game.py --preset lan-create --race T)")
            print("waiting for a live game with Gary's Command Center (start one)...")
            game.step(24)
        st = game.status()
        mapinfo = MapInfo.from_game(game)
        hi = HumanInterface(game, slot, PROFILES[profile], seed=1)
        print(f"playing slot {slot} | act() latency calibrated to {hi.act_latency_frames} frames "
              f"(adapter reported latency_frames={st.get('latency_frames')})", flush=True)
        gary = make_bot(hi, mapinfo)
        end = hi.frame + int(minutes * 60 * 24)
        while hi.frame < end:
            gary.act()
            hi.step(2)                       # waits for the live game to advance
            if hi.frame % (24 * 60) < 2:
                report(hi, game)
    except GameError as e:
        print(f"game over / bridge lost: {e}")
    finally:
        if pov and hi is not None:
            try:
                hi.save_pov(pov)
                if gary is not None and pov.endswith(".pov.jsonl"):
                    gary.save_trace(pov[:-len(".pov.jsonl")] + ".trace.jsonl")
                print(f"saved {pov}")
            except Exception as e:
                print(f"could not save pov: {e}")
        game.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gary", default="v016", type=manifest_name, help="which Gary (a manifest, options with +: v010+post)")
    ap.add_argument("--map", help="a .scm/.scx map (or a pre-1.18 replay) for headless play")
    ap.add_argument("--minutes", type=float, default=12)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--save", help="replay path (default gary_<version>.rep)")
    ap.add_argument("--command-delay", type=int, default=LIVE_COMMAND_DELAY,
                    help=f"headless: frames before a command runs (default {LIVE_COMMAND_DELAY}, as measured live)")
    ap.add_argument("--style", type=int, help="build-style cluster to steer the macro toward")
    ap.add_argument("--fight-model", help="replace the version's fight model with this checkpoint")
    ap.add_argument("--quiet", action="store_true", help="don't print each decision")
    ap.add_argument("--live", action="store_true", help="play in a live SC:R client via the bridge")
    ap.add_argument("--pipe", default=r"\\.\pipe\gary_scr", help="bridge named pipe (with --live)")
    ap.add_argument("--profile", default="pro", choices=sorted(PROFILES))
    ap.add_argument("--pov", help="where to save the point-of-view log (live)")
    args = ap.parse_args()
    make = lambda hi, mapinfo: build_gary(hi, mapinfo, args.gary, args.style, verbose=not args.quiet,
                                          fight_path=args.fight_model)
    version = manifest_of(args.gary)["version"]
    if args.live:
        play_live(args.pipe, args.minutes, args.profile, args.pov or f"gary_{args.gary}_live.pov.jsonl", make)
    else:
        if not args.map:
            ap.error("--map is required unless --live")
        play(args.map, args.minutes, args.seed, args.save or f"gary_{args.gary}.rep", make, f"Gary {version} (T)",
             command_delay=args.command_delay)


if __name__ == "__main__":
    main()
