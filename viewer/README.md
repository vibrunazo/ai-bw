# viewer

`gary_view` plays a replay in OpenBW's game window. With a POV log it shows the game from
one player's point of view, which is how we watch Gary play: what its camera showed, where its
mouse went and where it clicked.

**Status (v1):**

- Plays any replay `gary_resim` can play, including the ones `gary_env` saves and Remastered
  ones (via `viewer/watch.py`), e.g. a live game against Gary with its POV log.
- `--pov <file.pov.jsonl>` follows that player's camera frame by frame and draws their mouse:
  a cross for the cursor, and a ring that opens out where a click landed (green for left, red
  for right). The view is exactly the player's screen, scaled up to fit the window (start
  size `--scale`, default 2x; resize or maximize freely). There's no fog of war yet: you see
  everything in the camera area.
- `--record out.mp4` renders straight to a video through `ffmpeg` (must be on `PATH`), faster
  than real time. `--from`/`--to` (seconds) pick a stretch, `--speed N` makes an N× time-lapse.
- Silent: sound files aren't extracted.

The POV log comes from the human interface
([`gary/interface.py`](../gary/interface.py), `HumanInterface.save_pov`); Gary's bots write
it next to the replay when run with `--save`.

## Build

Same requirements as `env`, plus the SDL2 development package for Visual C++
(`SDL2-devel-2.32.10-VC.zip` from the [SDL releases](https://github.com/libsdl-org/SDL/releases))
unpacked into `external/sdl2`. Run `viewer\build.bat`; it produces `viewer\build\gary_view.exe`.

## Try it

```
python -m gary.play --gary v09 --map tests\fixtures\replays\stardata_tvz_standard_ozp3w.rep --minutes 10 --save data\interim\gary_games\gary_v09.rep
viewer\build\gary_view.exe --data data\gamedata\scr --replay data\interim\gary_games\gary_v09.rep --pov data\interim\gary_games\gary_v09.pov.jsonl
vieweruild\gary_view.exe --data data\gamedata\scr --replay data\interim\gary_games\gary_v09.rep --pov data\interim\gary_games\gary_v09.pov.jsonl
python viewer\watch.py data\interim\gary_games\gary_v09.rep --pov data\interim\gary_games\gary_v09.pov.jsonl --record gary_v09.mp4 --to 240
```

Remastered replays (and any other) through the wrapper, which decodes them first:

```
python viewer/watch.py "data/raw/vib/Gary v3 got owned.rep" --pov gary_v01_live.pov.jsonl
```

In the window, space pauses, left/right seek 10 s (shift: 60 s), up/down change the speed and
Esc quits. Without `--pov`, WASD moves the view. The title bar shows the game time.
