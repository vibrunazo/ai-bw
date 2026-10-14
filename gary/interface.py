"""The human interface: the only way Gary touches the game (docs/ARCHITECTURE.md §3, §6.2, §6.3).

Gary acts like a person at a keyboard and mouse, and sees what a person would see:

- **Camera.** Gary has a screen-sized view of the map. Clicks must land inside it (minimap
  commands are the exception, with much worse precision). The view moves the ways a player moves
  it: clicking the minimap (mouse travel, minimap-pixel precision), arrow-key scrolling (at a
  scroll speed), location hotkeys (F2-F4) and double-tapping a group hotkey. Camera moves cost
  time and a hand, not APM: the game doesn't count them as actions either.
- **Mouse.** Clicks are screen pixels, never unit IDs. The game decides what's under the
  pixel, so in a stack of units the one on top gets clicked. The cursor has to travel: each
  click lands after a Fitts's-law delay from where the cursor was, with scatter that grows
  when Gary rushes.
- **Selection.** At most 12 units, built by clicking, drag boxes and hotkeys.
- **APM budget.** A token bucket: every action costs one token; tokens refill at a steady
  rate up to a burst capacity. No tokens, no action.
- **Seeing.** Fog of war: enemy units only when visible. Enemy HP and shields only for an
  enemy unit you have selected. Everything arrives late by the profile's reaction time.

v1 limits (deliberately simple; each is a parameter to calibrate from replay and camera-logger
data later): hit-testing uses each unit's clickable rectangle, not its exact pixels; box
selection uses the game's own rules approximately; no memory of fogged units yet (§6.10).

Usage:
    hi = HumanInterface(game, slot=3, profile=PROFILES["pro"])
    obs = hi.observe()
    hi.camera_center(x, y); hi.click(sx, sy); hi.right_click(sx, sy); hi.train(SCV)
    hi.step(8)   # advances the game; queued clicks land when their mouse travel finishes
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field, replace

from gary import commands as C
from gary import terran as T
from gary.env import Game, decode_observation
from gary.policy.fight_memory import command_kind

FRAME_MS = 42  # one frame at Fastest speed
HISTORY = 64   # frames of snapshots kept: the reaction delay and aim's look back fit easily
LEAD_SPAN = 8  # aim: a unit's motion is judged over this many frames before what Gary saw...
LEAD_MAX = 64  # ...and carried forward this far at most (pixels)
NO_UNIT = 0


@dataclass(frozen=True)
class Profile:
    """Human limits (docs/ARCHITECTURE.md §6.2). Numbers are placeholders until fitted from data."""
    name: str = "b_rank"
    viewport: tuple[int, int] = (640, 400)   # playable screen area in map pixels
    apm_capacity: float = 10.0               # burst size (tokens)
    apm_per_second: float = 3.5              # sustained rate (~210 APM)
    fitts_a_ms: float = 90.0                 # Fitts's law: T = a + b * log2(D / W + 1)
    fitts_b_ms: float = 110.0
    scatter_px: float = 3.0                  # click scatter (std dev) at a relaxed pace
    scatter_rushed_px: float = 6.0           # extra scatter when the APM bucket is nearly empty
    minimap_scatter_px: float = 48.0         # map pixels
    reaction_ms: float = 300.0               # observation delay
    # camera (placeholders until measured from camera-logger data)
    minimap_rect: tuple[int, int, int, int] = (6, 348, 128, 128)  # screen x, y, w, h (in the HUD)
    scroll_px_per_frame: float = 24.0        # arrow-key scrolling speed
    key_ms: float = 80.0                     # time between two key presses


PROFILES = {
    # Gary's default: as fast as the pros in early TvZ defenses. Fitted to the time from selecting
    # units to the next targeted command in pro replays (median 168 ms in the home-defense
    # scenarios; b_rank's mouse took 504 ms); reaction and key speed are estimates until the camera
    # logger measures them (§7.4). eval/reaction.py, Terran pros in 200 TvZ replays, from an event
    # to their first command on it (hands included): a unit of theirs hit, 10% within 420 ms, half
    # by 1.1 s; an enemy unit coming into sight, 10% by 340 ms: about 200 ms to see plus the hands
    "pro": Profile(name="pro", fitts_a_ms=20, fitts_b_ms=28, reaction_ms=200.0, key_ms=40.0,
                   apm_per_second=7.0, apm_capacity=20.0),
    "b_rank": Profile(),
    "b_rank_widescreen": Profile(name="b_rank_widescreen", viewport=(854, 400)),
    "c_rank": Profile(name="c_rank", apm_capacity=7.0, apm_per_second=2.2, scatter_px=4.0,
                      scatter_rushed_px=8.0, reaction_ms=400.0),
}


@dataclass
class ActionResult:
    accepted: bool            # did the interface take the action (APM, camera, selection limits)
    reason: str = ""          # why not, if not
    send_frame: int = -1      # when the command is emitted to the game
    land_frame: int = -1      # when its effect reaches the game (send_frame + act() latency)


@dataclass
class _Pending:
    send_at: int              # frame on which the hand/key finishes and the command is emitted
    kind: str
    args: dict = field(default_factory=dict)


class HumanInterface:
    def __init__(self, game: Game, slot: int, profile: Profile = PROFILES["pro"], seed: int = 0):
        self.game = game
        self.slot = slot
        self.p = profile
        self.rng = random.Random(seed)
        obs = game.observe()
        self.frame = obs["frame"]
        self.map_size = (obs["map"]["w"], obs["map"]["h"])
        start = next((u for u in obs["units"] if u["owner"] == slot and u["type"] in
                      (C.COMMAND_CENTER, C.HATCHERY, C.NEXUS)), None)
        vw, vh = self.p.viewport
        cx, cy = (start["x"], start["y"]) if start else (vw // 2, vh // 2)
        self.camera = (max(0, cx - vw // 2), max(0, cy - vh // 2))   # top-left, map pixels
        self.cursor = (vw // 2, vh // 2)                              # screen pixels
        self.hand_free_at = self.frame                                # when the mouse is free
        self.keys_free_at = self.frame                                # when the keyboard hand is free
        self.camera_locations: dict[int, tuple[int, int]] = {}        # F2-F4 screen locations
        self.scroll: tuple[float, float, int] | None = None           # (dx, dy per frame, frames left)
        self.tokens = self.p.apm_capacity
        self.selection: list[int] = []
        # every unit command Gary's hands issued, as a replay records it: (frame, kind, unit tags, x, y,
        # target tag), kinds as in gary/policy/fight_memory.py (the command model's memory)
        self.commands: list[tuple] = []
        self.last_macro_frame = -10 ** 6      # last train / morph / research / upgrade / build sent
        self.hotkeys: dict[int, list[int]] = {}
        self.pending: list[_Pending] = []
        # a snapshot every frame, [frame, observation], for the last HISTORY frames; kept undecoded
        # (bytes) until read, since Gary only ever looks at the one from its reaction time ago (and
        # aim at the few before it). (Every 4 frames, a 200 ms delay was 210-340 ms, 270 on average)
        self.history: deque[list] = deque([[self.frame, obs]], maxlen=HISTORY)
        self._observe_raw = getattr(game, "observe_raw", None)   # the live bridge has no raw form
        self.stats = {"actions": 0, "rejected_by_interface": 0, "rejected_by_game": 0}
        # act() latency: frames from emitting a command (game.act) to its effect in the game.
        # OpenBW applies commands in the same step (0). A live SC:R client queues them into its
        # turn queue and lands them latency_frames later (the bridge reports this in status()),
        # plus the frame it takes to hand the packet to the queue. The live smoke
        # (adapters/scr_bridge/tools/smoke_act.py) measures the real value and calibrate_act_latency
        # stores it here so the pending/land model predicts when actions actually take effect.
        self.act_latency_frames = 0
        self._auto_calibrate_act_latency()
        results = getattr(self.game, "on_result", None)   # delayed commands report back when they run
        if results is not None:
            results[self.slot] = self._record_result
        # point-of-view log (camera, cursor, clicks per frame) for the viewer: viewer/gary_view
        self.pov: list[dict] = []
        self._mouse_moves: list[tuple[int, int, tuple[int, int], tuple[int, int]]] = []
        self._last_pov: tuple | None = None
        self._last_pov_sel: list[int] = []

    # --- act() latency calibration ---------------------------------------------------------

    def _auto_calibrate_act_latency(self) -> None:
        """Seed act_latency_frames from a live backend's status() when one is available.

        gary.scr_env.ScrGame reports the game's own turn latency as `latency_frames`
        (e.g. 2). The bridge hands a packet to the turn queue on the next frame boundary, so
        the send->effect delay is latency_frames + the one-frame hand-off. This is a prior;
        the live smoke measures the true delay and calibrate_act_latency() records it.
        """
        delay = getattr(self.game, "command_delay", None)   # gary.env.Game's simulated network delay
        if delay is not None:
            self.act_latency_frames = int(delay)
        status = getattr(self.game, "status", None)
        if not callable(status):
            return
        try:
            st = status()
        except Exception:
            return
        latency = st.get("latency_frames")
        if latency is not None:
            self.act_latency_frames = int(latency) + 1   # +1 frame to hand the packet to the queue

    def calibrate_act_latency(self, frames: float | None = None) -> float:
        """Set the measured send->effect latency (in frames). With frames=None, re-read the
        backend's reported latency_frames (the prior). Returns the value now in use.

        Feed the value measured by adapters/scr_bridge/tools/smoke_act.py so the interface's
        pending/land model (and ActionResult.land_frame) tell the truth about when an action
        takes effect on a live client.
        """
        if frames is None:
            self._auto_calibrate_act_latency()
        else:
            self.act_latency_frames = int(round(frames))
        return self.act_latency_frames

    # --- what Gary sees ------------------------------------------------------------------

    def observe(self) -> dict:
        """The game as this player saw it reaction_ms ago, fogged, plus the interface state."""
        delay = round(self.p.reaction_ms / FRAME_MS)
        seen = self.history[0]
        for entry in self.history:
            if entry[0] <= self.frame - delay:
                seen = entry
        seen = self._snapshot(seen)
        me = next(p for p in seen["players"] if p["slot"] == self.slot)
        bit = 1 << self.slot
        inspected = set(self.selection) if len(self.selection) == 1 else set()
        units = []
        for u in seen["units"]:
            mine = u["owner"] == self.slot
            if not mine and not (u["visible_to"] & bit):
                continue
            v = dict(u)
            v.pop("visible_to", None)
            if not mine:
                v.pop("tx", None)                    # (where an enemy unit is going: not on screen)
                v.pop("ty", None)
            if not mine and u["owner"] != 11 and u["tag"] not in inspected:
                v["hp"] = v["shields"] = None        # the UI doesn't show enemy HP unless selected
                v.pop("energy", None)                # (nor energy)
            units.append(v)
        return {
            "frame": seen["frame"], "now": self.frame, "me": me, "units": units,
            "camera": {"x": self.camera[0], "y": self.camera[1],
                       "w": self.p.viewport[0], "h": self.p.viewport[1]},
            "cursor": self.cursor, "selection": list(self.selection),
            "hotkeys": {k: list(v) for k, v in self.hotkeys.items()},
            "apm_tokens": round(self.tokens, 2),
        }

    def aim(self, obs: dict, u: dict) -> tuple[int, int] | None:
        """Where on screen to click or box a unit seen in obs: where it will be when an action
        decided now lands, its motion over the frames before what Gary saw carried forward, as a
        player leads a moving unit (from what Gary saw only: what it sees is reaction_ms old, and a
        click where an SCV was found nothing). None if off screen."""
        seen = obs["frame"]
        prev = None
        for entry in self.history:
            if entry[0] <= seen - LEAD_SPAN:
                prev = entry
        before = prev and next((v for v in self._snapshot(prev)["units"] if v["tag"] == u["tag"]), None)
        if not before or seen <= prev[0]:
            return screen_of(obs, u["x"], u["y"])
        land = max(self.frame, self.hand_free_at) + max(1, round(self.p.fitts_a_ms / FRAME_MS))
        k = (land - seen) / (seen - prev[0])
        dx, dy = (u["x"] - before["x"]) * k, (u["y"] - before["y"]) * k
        d = math.hypot(dx, dy)
        # never past where it's going (an SCV turns at its mineral field and at the Command Center)
        tx, ty = u.get("tx", 0), u.get("ty", 0)
        most = min(LEAD_MAX, math.dist((u["x"], u["y"]), (tx, ty))) if (tx or ty) else LEAD_MAX
        if d > most:
            dx, dy = dx * most / d, dy * most / d
        return screen_of(obs, round(u["x"] + dx), round(u["y"] + dy)) or screen_of(obs, u["x"], u["y"])

    @staticmethod
    def _snapshot(entry: list) -> dict:
        """A history entry's observation, decoded on first use."""
        if isinstance(entry[1], bytes):
            entry[1] = decode_observation(entry[1])
        return entry[1]

    def latest_observation(self) -> dict | None:
        """The full (unfogged) game state at the current frame if the interface took one this frame,
        else None. For tools around Gary (scoring), never for Gary itself."""
        last = self.history[-1]
        return self._snapshot(last) if last[0] == self.frame else None

    # --- helpers -------------------------------------------------------------------------

    def _spend(self) -> ActionResult | None:
        if self.tokens < 1:
            self.stats["rejected_by_interface"] += 1
            return ActionResult(False, "apm")
        self.tokens -= 1
        self.stats["actions"] += 1
        return None

    def _game_result(self, send_frame: int) -> ActionResult:
        """A command that reaches the game (game.act): its effect lands act_latency_frames later."""
        return ActionResult(True, send_frame=send_frame,
                            land_frame=send_frame + self.act_latency_frames)

    def _instant_result(self, send_frame: int) -> ActionResult:
        """A camera / key move: no game command, so it takes effect immediately."""
        return ActionResult(True, send_frame=send_frame, land_frame=send_frame)

    def _on_screen(self, sx: int, sy: int) -> bool:
        return 0 <= sx < self.p.viewport[0] and 0 <= sy < self.p.viewport[1]

    def _schedule_click(self, sx: int, sy: int, target_w: float, kind: str, **args) -> ActionResult:
        if not self._on_screen(sx, sy):
            self.stats["rejected_by_interface"] += 1
            return ActionResult(False, "offscreen")
        if (r := self._spend()):
            return r
        dist = math.dist(self.cursor, (sx, sy))
        travel_ms = self.p.fitts_a_ms + self.p.fitts_b_ms * math.log2(dist / max(target_w, 1) + 1)
        start = max(self.frame, self.hand_free_at)
        land = start + max(1, round(travel_ms / FRAME_MS))
        rushed = 1 - self.tokens / self.p.apm_capacity
        sigma = self.p.scatter_px + self.p.scatter_rushed_px * rushed
        lx = round(sx + self.rng.gauss(0, sigma))
        ly = round(sy + self.rng.gauss(0, sigma))
        self.hand_free_at = land
        self._mouse_moves.append((start, land, self.cursor, (lx, ly)))
        self.cursor = (sx, sy)
        self.pending.append(_Pending(land, kind, {"sx": lx, "sy": ly, **args}))
        return self._game_result(land)

    def _schedule_key(self, kind: str, **args) -> ActionResult:
        if (r := self._spend()):
            return r
        land = self.frame + 1
        self.pending.append(_Pending(land, kind, args))
        return self._game_result(land)

    def _remember(self, cmd: str, order: int, x: int, y: int, target: int) -> None:
        """Log a unit command given to the current selection (see self.commands)."""
        self.commands.append((self.frame, command_kind(cmd, order, target), tuple(self.selection), x, y, target or 0))
        del self.commands[:-64]

    def _send(self, command: bytes) -> bool:
        """Send a command. False only if the game rejected it on the spot; a delayed command
        counts as sent (like a player, the interface assumes its click worked) and its result
        arrives later through _record_result."""
        ok = self.game.act(self.slot, command)
        if ok is False:
            self._record_result(command, False)
        return ok is not False

    def _record_result(self, command: bytes, ok: bool) -> None:
        if not ok:
            self._log_pov(act=f"\u2717 rejected: {COMMAND_NAMES.get(command[0], hex(command[0]))}")
            self.stats["rejected_by_game"] += 1
            key = f"rejected_by_game_{command[0]:#04x}"
            self.stats[key] = self.stats.get(key, 0) + 1

    def _move_camera(self, x: float, y: float) -> None:
        vw, vh = self.p.viewport
        map_w, map_h = self.map_size
        self.camera = (round(min(max(0, x), map_w - vw)), round(min(max(0, y), map_h - vh)))

    def _center_camera(self, x: int, y: int) -> None:
        vw, vh = self.p.viewport
        self._move_camera(x - vw // 2, y - vh // 2)

    def _to_map(self, sx: int, sy: int) -> tuple[int, int]:
        return self.camera[0] + sx, self.camera[1] + sy

    # --- actions (each costs one APM token) ----------------------------------------------

    # --- camera ---------------------------------------------------------------------------
    # Camera moves aren't game commands (replays don't record them and APM doesn't count them),
    # so they cost no APM tokens. They cost time and the hand that does them instead: the
    # minimap needs the mouse, scrolling and location hotkeys need the keyboard.

    def camera_minimap(self, map_x: int, map_y: int) -> ActionResult:
        """Left click on the minimap: the mouse travels to the minimap, and the view jumps there.
        Precise to about one minimap pixel (a 128-tile map is 32 map pixels per minimap pixel)."""
        mx, my, mw, mh = self.p.minimap_rect
        map_w, map_h = self.map_size
        target = (mx + map_x * mw / map_w, my + map_y * mh / map_h)
        dist = math.dist(self.cursor, target)
        travel_ms = self.p.fitts_a_ms + self.p.fitts_b_ms * math.log2(dist / 4 + 1)
        start = max(self.frame, self.hand_free_at)
        land = start + max(1, round(travel_ms / FRAME_MS))
        self.hand_free_at = land
        self._mouse_moves.append((start, land, self.cursor, (round(target[0]), round(target[1]))))
        self.cursor = (round(target[0]), round(target[1]))
        px = map_w / mw
        x = map_x + self.rng.uniform(-px / 2, px / 2)
        y = map_y + self.rng.uniform(-px / 2, px / 2)
        self.pending.append(_Pending(land, "camera", {"x": round(x), "y": round(y)}))
        return self._instant_result(land)

    def camera_scroll(self, dx: int, dy: int) -> ActionResult:
        """Arrow-key scrolling by (dx, dy) map pixels, at the profile's scroll speed."""
        frames = max(1, math.ceil(math.hypot(dx, dy) / self.p.scroll_px_per_frame))
        start = max(self.frame, self.keys_free_at)
        self.keys_free_at = start + frames
        self.pending.append(_Pending(start, "scroll", {"dx": dx / frames, "dy": dy / frames, "frames": frames}))
        return self._instant_result(start + frames)

    def camera_location_set(self, n: int) -> ActionResult:
        """Shift+F2..F4: remember the current view."""
        return self._key("camera_location_set", n=n)

    def camera_location(self, n: int) -> ActionResult:
        """F2..F4: jump to a remembered view."""
        if n not in self.camera_locations:
            return ActionResult(False, "no such location")
        return self._key("camera_location", n=n)

    def camera_to_group(self, n: int) -> ActionResult:
        """Double-tap a group hotkey: select it (one APM token) and center the view on it."""
        r = self.hotkey_recall(n)
        if r.accepted:
            self._key("camera_to_group", n=n)
        return r

    def _key(self, kind: str, **args) -> ActionResult:
        """A key press that isn't a game command (no APM token)."""
        land = max(self.frame, self.keys_free_at) + max(1, round(self.p.key_ms / FRAME_MS))
        self.keys_free_at = land
        self.pending.append(_Pending(land, kind, args))
        return self._instant_result(land)

    def click(self, sx: int, sy: int, shift: bool = False) -> ActionResult:
        """Left click on the screen: select whatever is drawn there."""
        return self._schedule_click(sx, sy, 16, "click", shift=shift)

    def ctrl_click(self, sx: int, sy: int) -> ActionResult:
        """Ctrl+click on the screen: every unit on screen of the clicked one's type (up to 12), as
        the game selects them (a plain click where that isn't a unit of Gary's that can be selected
        with others, or where the env can't tell)."""
        return self._schedule_click(sx, sy, 16, "ctrl_click")

    def box(self, sx0: int, sy0: int, sx1: int, sy1: int) -> ActionResult:
        """Drag-select a screen rectangle."""
        if not (self._on_screen(sx0, sy0) and self._on_screen(sx1, sy1)):
            self.stats["rejected_by_interface"] += 1
            return ActionResult(False, "offscreen")
        w = max(abs(sx1 - sx0), abs(sy1 - sy0), 16)
        return self._schedule_click(sx1, sy1, w, "box", x0=sx0, y0=sy0)

    def right_click(self, sx: int, sy: int, queued: bool = False) -> ActionResult:
        """Smart command at a screen point: move, attack, gather, follow."""
        return self._schedule_click(sx, sy, 16, "right_click", queued=queued)

    def minimap_right_click(self, map_x: int, map_y: int, queued: bool = False) -> ActionResult:
        """Right click on the minimap: anywhere on the map, but imprecise (queued: shift held, after
        the selection's current orders: a waypoint)."""
        if (r := self._spend()):
            return r
        land = max(self.frame, self.hand_free_at) + round(self.p.fitts_a_ms / FRAME_MS) + 1
        s = self.p.minimap_scatter_px
        self.hand_free_at = land
        self.pending.append(_Pending(land, "minimap_right_click", {
            "x": round(map_x + self.rng.gauss(0, s)), "y": round(map_y + self.rng.gauss(0, s)), "queued": queued}))
        return self._game_result(land)

    def minimap_command(self, order: int, map_x: int, map_y: int) -> ActionResult:
        """A targeted order (e.g. attack-move: A, then click) on the minimap."""
        if (r := self._spend()):
            return r
        land = max(self.frame, self.hand_free_at) + round((self.p.fitts_a_ms + self.p.key_ms) / FRAME_MS) + 1
        s = self.p.minimap_scatter_px
        self.hand_free_at = land
        self.pending.append(_Pending(land, "minimap_order", {
            "order": order, "x": round(map_x + self.rng.gauss(0, s)), "y": round(map_y + self.rng.gauss(0, s))}))
        return self._game_result(land)

    def train(self, unit_type: int) -> ActionResult:
        """Hotkey: train from the selected building."""
        return self._schedule_key("train", unit_type=unit_type)

    def order_click(self, order: int, sx: int, sy: int) -> ActionResult:
        """A hotkey order then a click on the screen: e.g. A + click = attack-move there, or
        attack the unit under the cursor."""
        return self._schedule_click(sx, sy, 16, "order_click", order=order)

    def stop(self) -> ActionResult:
        return self._schedule_key("stop")

    def hold(self) -> ActionResult:
        return self._schedule_key("hold")

    def stim(self) -> ActionResult:
        return self._schedule_key("stim")

    def unload_all(self) -> ActionResult:
        return self._schedule_key("unload_all")

    def return_cargo(self) -> ActionResult:
        return self._schedule_key("return_cargo")

    def chat(self, text: str) -> ActionResult:
        """Enter, type a message, Enter: a chat line to everyone."""
        return self._schedule_key("chat", text=text)

    def research(self, tech: int) -> ActionResult:
        """Hotkey: research a tech in the selected building."""
        return self._schedule_key("research", tech=tech)

    def upgrade(self, upgrade_id: int) -> ActionResult:
        """Hotkey: start an upgrade in the selected building."""
        return self._schedule_key("upgrade", upgrade_id=upgrade_id)

    def morph_building(self, unit_type: int) -> ActionResult:
        """Zerg: the selected building becomes unit_type (its hotkey on the command card)."""
        return self._schedule_key("morph_building", unit_type=unit_type)

    def morph(self, unit_type: int) -> ActionResult:
        """Hotkey: morph the selected larva."""
        return self._schedule_key("morph", unit_type=unit_type)

    def build(self, unit_type: int, sx: int, sy: int, order: int = C.ORDER_PLACE_BUILDING) -> ActionResult:
        """Place a building with the selected worker at a screen point (its top-left tile)."""
        return self._schedule_click(sx, sy, 32, "build", unit_type=unit_type, order=order)

    def hotkey_set(self, n: int) -> ActionResult:
        return self._schedule_key("hotkey_set", n=n)

    def hotkey_add(self, n: int) -> ActionResult:
        return self._schedule_key("hotkey_add", n=n)

    def hotkey_recall(self, n: int) -> ActionResult:
        return self._schedule_key("hotkey_recall", n=n)

    # --- time --------------------------------------------------------------------------

    def step(self, frames: int, advance_game: bool = True) -> None:
        """Advance the game, landing queued actions on their frames. When several players share
        one game, every interface steps, but only one of them advances the game itself."""
        end = self.frame + frames
        while self.frame < end:
            due = sorted((a for a in self.pending if a.send_at <= self.frame), key=lambda a: a.send_at)
            for a in due:
                self.pending.remove(a)
                self._land(a)
            if self.scroll:
                dx, dy, left = self.scroll
                self._move_camera(self.camera[0] + dx, self.camera[1] + dy)
                self.scroll = (dx, dy, left - 1) if left > 1 else None
            if advance_game:
                self.game.step(1)
            self._log_pov()
            self.frame += 1
            self.tokens = min(self.p.apm_capacity, self.tokens + self.p.apm_per_second * FRAME_MS / 1000)
            self.history.append([self.frame, self._observe_raw() if self._observe_raw else self.game.observe()])

    def _displayed_cursor(self) -> tuple[int, int]:
        """Where the mouse is drawn this frame: moving along its path while a click travels."""
        pos = None
        for start, land, src, dst in self._mouse_moves:
            if start <= self.frame <= land:
                t = (self.frame - start) / max(1, land - start)
                pos = (round(src[0] + (dst[0] - src[0]) * t), round(src[1] + (dst[1] - src[1]) * t))
            elif land < self.frame:
                pos = dst
        self._mouse_moves = [m for m in self._mouse_moves if m[1] >= self.frame - 1] or self._mouse_moves[-1:]
        return pos if pos is not None else self.cursor

    def _log_pov(self, click: str | None = None, act: str | None = None, at: tuple[int, int] | None = None,
                 cmd: dict | None = None) -> None:
        """One POV entry: camera, cursor, the click if any, and what Gary did, in words ("act");
        at: where on the screen a minimap click landed (the cursor model doesn't move for those);
        sel: the selection, when it changed; cmd: a command to the selection ({"kind": move, attack,
        amove, gather, hold, stop..., "x", "y": map point, "target": unit tag}), for the viewer."""
        cur = self._displayed_cursor()
        key = (self.camera, cur)
        sel_changed = self.selection != self._last_pov_sel
        if click is None and act is None and not sel_changed and key == self._last_pov and self.frame % 24:
            return
        self._last_pov = key
        entry = {"frame": self.frame, "camera": list(self.camera), "cursor": list(cur), "click": click}
        if act:
            entry["act"] = act
        if at:
            entry["at"] = [round(at[0]), round(at[1])]
        if sel_changed:
            entry["sel"] = list(self.selection)
            self._last_pov_sel = list(self.selection)
        if cmd:
            entry["cmd"] = cmd
        self.pov.append(entry)

    def save_pov(self, path) -> None:
        """Write the point-of-view log (one JSON object per line) for viewer/gary_view --pov."""
        import json
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"schema": "pov/v1", "slot": self.slot, "viewport": list(self.p.viewport),
                                "minimap": list(self.p.minimap_rect)}) + "\n")
            for e in self.pov:
                f.write(json.dumps(e) + "\n")

    def _land(self, a: _Pending) -> None:
        g = self.game
        click = {"click": "left", "ctrl_click": "left", "box": "left", "build": "left", "right_click": "right",
                 "minimap_right_click": "right", "minimap_order": "left"}.get(a.kind)
        act = None                       # what happened, in words, for the POV log
        cmd = None                       # a command to the selection: kind, map point, target (POV log)
        if a.kind == "camera":
            self._center_camera(a.args["x"], a.args["y"])
            act = "minimap click (camera)"
        elif a.kind == "scroll":
            self.scroll = (a.args["dx"], a.args["dy"], a.args["frames"])
            act = "arrow keys (scroll)"
        elif a.kind == "camera_location_set":
            self.camera_locations[a.args["n"]] = self.camera
            act = f"Shift+F{a.args['n']} (save screen)"
        elif a.kind == "camera_location":
            self.camera = self.camera_locations[a.args["n"]]
            act = f"F{a.args['n']} (jump to screen)"
        elif a.kind == "camera_to_group":
            tags = set(self.hotkeys.get(a.args["n"], []))
            units = [u for u in self._snapshot(self.history[-1])["units"] if u["tag"] in tags]
            if units:
                self._center_camera(sum(u["x"] for u in units) // len(units), sum(u["y"] for u in units) // len(units))
            act = f"{a.args['n']} {a.args['n']} (center on group)"
        elif a.kind == "click":
            tag = g.unit_at(self.slot, *self._to_map(a.args["sx"], a.args["sy"]))
            act = "click: nothing there"
            if tag:
                name = unit_name(g.unit_type_of(tag))
                if a.args["shift"] and self.selection:
                    act = f"shift-click: add {name}"
                    if self._send(bytes([0x0A, 1]) + tag.to_bytes(2, "little")):
                        self.selection = (self.selection + [tag])[:12]
                else:
                    act = f"click: select {name}"
                    if self._send(C.select([tag])):
                        self.selection = [tag]
        elif a.kind == "ctrl_click":
            x, y = self._to_map(a.args["sx"], a.args["sy"])
            cx, cy = self.camera
            vw, vh = self.p.viewport
            tags = g.type_select(self.slot, x, y, (cx, cy, cx + vw, cy + vh)) if hasattr(g, "type_select") else []
            tag = g.unit_at(self.slot, x, y)
            if tags:
                act = f"ctrl+click: select {len(tags)} {unit_name(g.unit_type_of(tags[0]))}"
                if self._send(C.select(tags)):
                    self.selection = tags
            elif tag:
                act = f"click: select {unit_name(g.unit_type_of(tag))}"
                if self._send(C.select([tag])):
                    self.selection = [tag]
            else:
                act = "click: nothing there"
        elif a.kind == "box":
            x0, y0 = self._to_map(a.args["x0"], a.args["y0"])
            x1, y1 = self._to_map(a.args["sx"], a.args["sy"])
            tags = g.box_select(self.slot, x0, y0, x1, y1)
            act = f"drag box: {len(tags)} unit{'s' * (len(tags) != 1)}" if tags else "drag box: nothing"
            if tags and self._send(C.select(tags)):
                self.selection = tags
        elif a.kind == "right_click":
            x, y = self._to_map(a.args["sx"], a.args["sy"])
            tag = g.unit_at(self.slot, x, y)
            kind = g.unit_type_of(tag) if tag else None
            if kind is None:
                act = "right-click ground (move)"
            elif kind in RESOURCES:
                act = f"right-click {unit_name(kind)} (gather)"
            else:
                act = f"right-click {unit_name(kind)}"
            enemy = bool(tag) and kind not in RESOURCES and tag not in self._own_tags()
            cmd = {"kind": "gather" if kind in RESOURCES else "attack" if enemy else "move", "x": x, "y": y,
                   "target": tag or 0}
            self._send(C.right_click(x, y, tag, C.NO_UNIT if not tag else kind, a.args["queued"]))
            self._remember("rclick", -1, x, y, tag)
        elif a.kind == "minimap_right_click":
            queued = a.args.get("queued", False)
            act = "shift+minimap right-click (waypoint)" if queued else "minimap right-click (move)"
            cmd = {"kind": "move", "x": a.args["x"], "y": a.args["y"], "target": 0}
            self._send(C.right_click(a.args["x"], a.args["y"], queued=queued))
            self._remember("rclick", -1, a.args["x"], a.args["y"], 0)
        elif a.kind == "minimap_order":
            act = "minimap attack-move" if a.args["order"] == C.ORDER_ATTACK_MOVE else f"minimap order {a.args['order']}"
            cmd = {"kind": "amove" if a.args["order"] == C.ORDER_ATTACK_MOVE else "order", "x": a.args["x"],
                   "y": a.args["y"], "target": 0}
            self._send(C.targeted_order(a.args["order"], a.args["x"], a.args["y"]))
            self._remember("order", a.args["order"], a.args["x"], a.args["y"], 0)
        elif a.kind == "train":
            act = f"train {unit_name(a.args['unit_type'])}"
            self._send(C.train(a.args["unit_type"]))
            self.last_macro_frame = self.frame
        elif a.kind == "morph":
            act = f"morph {unit_name(a.args['unit_type'])}"
            self._send(C.morph(a.args["unit_type"]))
            self.last_macro_frame = self.frame
        elif a.kind == "morph_building":
            act = f"morph {unit_name(a.args['unit_type'])}"
            self._send(C.morph_building(a.args["unit_type"]))
            self.last_macro_frame = self.frame
        elif a.kind == "research":
            act = f"research {T.TECH_NAMES.get(a.args['tech'], a.args['tech'])}"
            self._send(C.research(a.args["tech"]))
            self.last_macro_frame = self.frame
        elif a.kind == "upgrade":
            act = f"upgrade {T.UPGRADE_NAMES.get(a.args['upgrade_id'], a.args['upgrade_id'])}"
            self._send(C.upgrade(a.args["upgrade_id"]))
            self.last_macro_frame = self.frame
        elif a.kind == "build":
            x, y = self._to_map(a.args["sx"], a.args["sy"])
            act = f"place {unit_name(a.args['unit_type'])}"
            self._send(C.build(a.args["unit_type"], x // 32, y // 32, a.args["order"]))
            self.last_macro_frame = self.frame
        elif a.kind == "order_click":
            x, y = self._to_map(a.args["sx"], a.args["sy"])
            tag = g.unit_at(self.slot, x, y)
            kind = g.unit_type_of(tag) if tag else None
            what = "attack-move" if a.args["order"] == C.ORDER_ATTACK_MOVE else f"order {a.args['order']}"
            act = f"A-click {unit_name(kind)} ({what})" if kind is not None else f"A-click ground ({what})"
            cmd = {"kind": ("attack" if tag else "amove") if a.args["order"] == C.ORDER_ATTACK_MOVE else "order",
                   "x": x, "y": y, "target": tag or 0}
            self._send(C.targeted_order(a.args["order"], x, y, tag, kind if kind is not None else C.NO_UNIT))
            self._remember("order", a.args["order"], x, y, tag)
        elif a.kind in ("stop", "hold", "stim", "return_cargo", "unload_all"):
            act = {"stop": "S (stop)", "hold": "H (hold position)", "stim": "T (stim packs)",
                   "return_cargo": "C (return cargo)", "unload_all": "U (unload all)"}[a.kind]
            if a.kind in ("stop", "hold"):
                cmd = {"kind": a.kind, "x": -1, "y": -1, "target": 0}
            self._send({"stop": C.stop, "hold": C.hold_position, "stim": C.stim, "return_cargo": C.return_cargo,
                        "unload_all": C.unload_all}[a.kind]())
            self._remember({"return_cargo": "return"}.get(a.kind, a.kind), -1, 0, 0, 0)
        elif a.kind == "chat":
            act = f"chat: {a.args['text']}"
            self._send(C.chat(self.slot, a.args["text"]))
        elif a.kind == "hotkey_set":
            self.hotkeys[a.args["n"]] = list(self.selection)
            act = f"Ctrl+{a.args['n']} (make group)"
        elif a.kind == "hotkey_add":
            self.hotkeys[a.args["n"]] = list(dict.fromkeys(self.hotkeys.get(a.args["n"], []) + self.selection))[:12]
            act = f"Shift+{a.args['n']} (add to group)"
        elif a.kind == "hotkey_recall":
            tags = self.hotkeys.get(a.args["n"], [])
            act = f"{a.args['n']} (select group)"
            if tags and self._send(C.select(tags)):
                self.selection = list(tags)
        at = None
        if a.kind in ("minimap_right_click", "minimap_order"):
            mx, my, mw, mh = self.p.minimap_rect
            at = (mx + a.args["x"] * mw / self.map_size[0], my + a.args["y"] * mh / self.map_size[1])
        self._log_pov(click, act, at, cmd)

    def _own_tags(self) -> set[int]:
        last = self._snapshot(self.history[-1])
        return {u["tag"] for u in last["units"] if u["owner"] == self.slot}

RESOURCES = {176, 177, 178, 188, 110, 157, 149}      # mineral fields, geyser, refineries / extractor
OTHER_NAMES = {176: "Mineral Field", 177: "Mineral Field", 178: "Mineral Field", 188: "Vespene Geyser",
               35: "Larva", 36: "Egg", 37: "Zergling", 41: "Drone", 42: "Overlord", 131: "Hatchery"}
COMMAND_NAMES = {0x09: "select", 0x0A: "shift-select", 0x0C: "build", 0x14: "right-click", 0x15: "order",
                 0x1F: "train", 0x23: "morph", 0x35: "morph building", 0x30: "research", 0x32: "upgrade", 0x5C: "chat",
                 0x1A: "stop", 0x2B: "hold", 0x36: "stim", 0x1E: "return cargo", 0x28: "unload all"}


def unit_name(unit_type: int) -> str:
    return T.NAMES.get(unit_type) or OTHER_NAMES.get(unit_type) or f"unit {unit_type}"


def screen_of(obs: dict, x: int, y: int) -> tuple[int, int] | None:
    """Screen position of a map point in an observation, or None if it's outside the view."""
    cam = obs["camera"]
    sx, sy = x - cam["x"], y - cam["y"]
    return (sx, sy) if 0 <= sx < cam["w"] and 0 <= sy < cam["h"] else None
