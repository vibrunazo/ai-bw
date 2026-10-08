"""Benchmark runs share the machine: they run below normal priority by default, so other programs
keep the CPU they need and the games take what's left (child processes, the games themselves,
inherit the priority).

    python -m eval.vs_rush ... --priority normal      # full priority (nothing else running)
"""

from __future__ import annotations

import os
import sys

LEVELS = ("low", "normal")


def add_argument(ap) -> None:
    ap.add_argument("--priority", choices=LEVELS, default="low",
                    help="process priority of the run and its games (default: below normal)")


def apply(level: str) -> None:
    """Lower this process's priority (and so its children's), if asked."""
    if level != "low":
        return
    if sys.platform == "win32":
        import ctypes
        BELOW_NORMAL_PRIORITY_CLASS = 0x4000
        k = ctypes.windll.kernel32
        k.SetPriorityClass(k.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)
    else:
        os.nice(10)
