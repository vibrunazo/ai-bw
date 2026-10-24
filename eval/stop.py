"""Stop a run and everything it started: its pool workers, games and re-simulator calls (issue #33).

Stopping a job's shell leaves its multiprocessing workers running, and they go on starving the machine or, cut
off from their parent, write their work up as failures (a stopped dataset build marked 11,583 replays failed).
This kills a whole process tree, deepest first.

    python -m eval.stop 1234                  # the process 1234 and all its descendants
    python -m eval.stop --match army_dataset  # every Python process whose command line has this, and theirs
    python -m eval.stop --match CP120a --dry  # list what would be stopped
"""
from __future__ import annotations

import argparse
import os

import psutil


def tree(p: psutil.Process) -> list[psutil.Process]:
    """The process and its descendants, deepest first."""
    try:
        kids = p.children(recursive=True)
    except psutil.Error:
        kids = []
    return kids[::-1] + [p]


def matching(pattern: str) -> list[psutil.Process]:
    """Python processes whose command line holds the pattern (not this one), outermost only."""
    me = os.getpid()
    found = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
            if p.info["pid"] != me and "python" in (p.info["name"] or "").lower() and pattern in cmd \
                    and "eval.stop" not in cmd:
                found.append(p)
        except psutil.Error:
            continue
    ids = {p.pid for p in found}
    return [p for p in found if not any(a.pid in ids for a in _ancestors(p))]


def _ancestors(p: psutil.Process) -> list[psutil.Process]:
    try:
        return p.parents()
    except psutil.Error:
        return []


def stop(roots: list[psutil.Process], dry: bool = False) -> int:
    victims = []
    for r in roots:
        victims += [p for p in tree(r) if p not in victims]
    for p in victims:
        try:
            print(f"{'would stop' if dry else 'stopping'} {p.pid} {' '.join(p.cmdline())[:120]}")
        except psutil.Error:
            pass
    if dry:
        return len(victims)
    for p in victims:
        try:
            p.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(victims, timeout=10)
    return len(victims)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pids", nargs="*", type=int, help="processes to stop with their descendants")
    ap.add_argument("--match", help="stop the Python processes whose command line holds this (and theirs)")
    ap.add_argument("--dry", action="store_true", help="list, stop nothing")
    args = ap.parse_args()
    roots = []
    for pid in args.pids:
        try:
            roots.append(psutil.Process(pid))
        except psutil.NoSuchProcess:
            print(f"no process {pid}")
    if args.match:
        roots += matching(args.match)
    if not roots:
        print("nothing to stop")
        return
    n = stop(roots, args.dry)
    print(f"{n} process{'es' if n != 1 else ''} {'found' if args.dry else 'stopped'}")


if __name__ == "__main__":
    main()
