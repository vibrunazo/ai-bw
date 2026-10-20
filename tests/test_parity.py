"""The refactor's gate (docs/REFACTOR.md): fixed games with the day's default manifests play byte-identically
before and after. Each game's logs (both sides' screens and traces, and every layer's records) are hashed,
the git hash in Gary's greeting masked, and compared with the hashes recorded in tests/fixtures/parity.json.
A change of behavior on purpose records them again (--record), in its own commit, with the reason.

Needs the game data and the ladder maps (skips without them); a few minutes on parallel processes.

Run: python tests/test_parity.py            (check)
     python tests/test_parity.py --record   (write the hashes)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
FIXTURE = REPO / "tests" / "fixtures" / "parity.json"

TERRAN = "v029+commit+draw+floor+mass+nonatbunker+depots"   # the defaults of 2026-10-19 (docs/REFACTOR.md)
# (map, seed, side A, side B): TvZ on three maps, and a mirror of each race that plays (docs/REFACTOR.md
# "Tests across matchups")
GAMES = [("KnockOut 1.4", 4004, "T", "Z"), ("Fighting Spirit 1.4", 4005, "T", "Z"), ("Odyssey-RE 2.0", 4006, "T", "Z"),
         ("KnockOut 1.4", 6001, "Z", "Z"), ("Radeon 1.2", 7001, "T", "T")]
MINUTES = 8
LOGS = (".pov.jsonl", ".bpov.jsonl", ".trace.jsonl", ".btrace.jsonl", ".records.jsonl")
ZERG = "gz1"
BY_RACE = {"T": TERRAN, "Z": ZERG}
GIT = re.compile(r"git [0-9a-f]{7,40}\+?")


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, encoding="utf-8") as f:
        next(f, None)                                   # (the header: schema, slot)
        for line in f:
            h.update(GIT.sub("git -", line).encode("utf-8"))
    return h.hexdigest()


def play(map_name: str, seed: int, a: str, b: str, out: Path) -> dict[str, str]:
    """One game's log hashes."""
    sys.path.insert(0, str(REPO / "ingest"))
    from inventory import data_root
    rep = out / f"{re.sub(r'[^A-Za-z0-9]', '', map_name)}_{seed}_{a}{b}.rep"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    env.pop("GARY_RUN_NOTE", None)
    cmd = [sys.executable, "-m", "eval.match", "--a", BY_RACE[a], "--b", BY_RACE[b], "--seed", str(seed),
           "--minutes", str(MINUTES), "--map", str(data_root() / "maps" / "ladder" / f"{map_name}.rep"),
           "--save", str(rep)]
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    if p.returncode:
        raise RuntimeError(f"{map_name} seed {seed}: {p.stderr[-2000:]}")
    return {kind: _digest(Path(str(rep)[:-len(".rep")] + kind)) for kind in LOGS}


def hashes() -> dict[str, dict[str, str]]:
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(len(GAMES)) as pool:
        got = pool.map(lambda g: (f"{g[0]} {g[1]}" + ("" if g[2:] == ("T", "Z") else f" {g[2]}v{g[3]}"), play(*g, Path(tmp))),
                       GAMES)
        return dict(got)


def _available() -> str | None:
    try:
        sys.path.insert(0, str(REPO / "ingest"))
        from inventory import data_root
        missing = [g[0] for g in GAMES if not (data_root() / "maps" / "ladder" / f"{g[0]}.rep").exists()]
        return f"no ladder map {missing[0]}" if missing else None
    except Exception as e:                              # no game data here
        return str(e)


def test_parity():
    why = _available()
    if why:
        print(f"skip: {why}")
        return
    want = json.loads(FIXTURE.read_text(encoding="utf-8"))
    got = hashes()
    diff = [f"{game} {kind}" for game in want for kind in want[game] if got[game][kind] != want[game][kind]]
    assert not diff, f"games differ from the recorded ones: {diff}"


if __name__ == "__main__":
    if "--record" in sys.argv:
        FIXTURE.write_text(json.dumps(hashes(), indent=1) + "\n", encoding="utf-8")
        print(f"recorded {FIXTURE}")
    else:
        test_parity()
        print("ok")
