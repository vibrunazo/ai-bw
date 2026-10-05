"""Watch any replay in gary_view, Remastered ones included, optionally from Gary's point of view.

Remastered replays (1.18+) are decoded first (resim/scr_format.py), then gary_view plays the
decoded stream. Every other option goes straight to gary_view (see viewer/gary_view.cpp).

When recording a video with a POV log, Gary's actions ("click: select SCV", "Ctrl+4 (make
group)", "train SCV", "right-click Mineral Field (gather)", "minimap attack-move", and in red
"rejected: ..." when the game refused a command) are burned into the bottom-right of the view, like a
screencast-keys overlay.

    python viewer/watch.py "data/raw/vib/Gary v3 got owned.rep" --pov gary_v01_live.pov.jsonl
    python viewer/watch.py some_game.rep --pov gary.pov.jsonl --record gary.mp4 --to 300
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "resim"))
import scr_format  # noqa: E402

GAME_FPS = 1000 / 42           # game frames per second at Fastest
VIDEO_FPS = 24                 # gary_view records one frame per --speed game frames at 24 fps
SHOW_S = 2.5                   # each action stays this long (game time)
MAX_LINES = 6
CONSOLE_H = 80                 # gary_view's console strip under the view (viewer/gary_view.cpp)

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Keys,Arial,24,&H00FFFFFF,&H00FFFFFF,&H80000000,&H80000000,1,0,0,0,100,100,0,0,3,4,0,3,16,16,{margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ass_time(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def actions_subtitles(pov_path: str, out_path: str, from_s: float, speed: int) -> int:
    """Write an .ass subtitle file with the POV log's actions, timed to the recorded video: bottom-right
    of the game view, above the console strip (gary_view draws the player's 640x480 screen, 2x)."""
    first = int(from_s * 1000 / 42)
    acts = []
    vw, vh = 640, 400
    with open(pov_path, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if "viewport" in e:
                vw, vh = e["viewport"]
            if e.get("act") and e.get("frame", -1) >= first:
                acts.append((e["frame"], e["act"]))
    header = ASS_HEADER.replace("{w}", str(vw * 2)).replace("{h}", str((vh + CONSOLE_H) * 2)) \
        .replace("{margin}", str(CONSOLE_H * 2 + 16))
    to_video = lambda frame: (frame - first) / speed / VIDEO_FPS
    lines = []
    for i, (frame, text) in enumerate(acts):
        start = to_video(frame)
        end = max(start + 1.0, to_video(frame + SHOW_S * GAME_FPS))
        if i + MAX_LINES < len(acts):                  # never more than MAX_LINES on screen
            end = min(end, to_video(acts[i + MAX_LINES][0]))
        text = text.replace("\\", "/").replace("{", "(").replace("}", ")")
        if text.startswith("✗"):
            text = "{\\c&H5A5AFF&}" + text
        lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(max(end, start + 0.05))},Keys,,0,0,0,,{text}")
    Path(out_path).write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("replay")
    ap.add_argument("--data", default=str(REPO_ROOT / "data" / "gamedata" / "scr"), help="extracted game data")
    ap.add_argument("--pov")
    ap.add_argument("--record")
    ap.add_argument("--from", dest="from_s", type=float, default=0)
    ap.add_argument("--speed", type=int, default=1)
    args, rest = ap.parse_known_args()
    exe = REPO_ROOT / "viewer" / "build" / ("gary_view.exe" if os.name == "nt" else "gary_view")
    if not exe.exists():
        sys.exit("gary_view not built: run viewer/build.bat (see viewer/README.md)")
    data = Path(args.replay).read_bytes()
    cmd = [str(exe), "--data", args.data]
    temp = []
    if scr_format.replay_format(data) != "legacy":
        tmp = tempfile.NamedTemporaryFile(suffix=".flat", delete=False)
        tmp.write(scr_format.to_flat(data))
        tmp.close()
        temp.append(tmp.name)
        cmd += ["--replay", tmp.name, "--flat", "--unit-limit", str(scr_format.unit_limit(data))]
    else:
        cmd += ["--replay", args.replay]
    if args.pov:
        cmd += ["--pov", args.pov]
    if args.record:
        cmd += ["--record", args.record]
        if args.pov:
            subs = tempfile.NamedTemporaryFile(suffix=".ass", delete=False)
            subs.close()
            temp.append(subs.name)
            n = actions_subtitles(args.pov, subs.name, args.from_s, args.speed)
            if n:
                cmd += ["--subtitles", subs.name]
    cmd += ["--from", str(args.from_s), "--speed", str(args.speed)]
    try:
        sys.exit(subprocess.call(cmd + rest))
    finally:
        for t in temp:
            os.unlink(t)


if __name__ == "__main__":
    main()
