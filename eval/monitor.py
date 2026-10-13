"""Gary's run monitor: what's running, how far along it is, and what finished, on a web page.

Every benchmark and training run registers itself as it starts (eval/priority.py apply(), which they
all call): a status file in runs/monitor/ with its command, the lines it prints, and its progress
where the script reports it:

    for r in monitor.track(pool.map(play, jobs), len(jobs), "games"):   # a progress bar, an ETA
    monitor.tally("v019", "won")                                         # running counts

The page reads those files, and lists other Gary processes running without one (started before
this existed). Each run shows what its script does (its docstring) and why it runs, if said
(GARY_RUN_NOTE when it starts, or --note later). GARY_MONITOR=off keeps a run off it.

    python -m eval.monitor                    # http://127.0.0.1:8766
    python -m eval.monitor --host 0.0.0.0     # also from other devices on the LAN
    python -m eval.monitor --note 1234 "why"  # say why the run with pid 1234 runs
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FOLDER = REPO_ROOT / "runs" / "monitor"
PAGE = Path(__file__).with_name("monitor.html")
KEEP = 100                             # status files kept (the newest)
LINES = 400                            # printed lines kept per run
SAVE_EVERY = 2.0                       # seconds between writes of a status file
PORT = 8766                            # (8765: another project's monitor)

# --- the reporter: in the run's own process ----------------------------------------------------

_run = None


class _Run:
    def __init__(self, argv: list[str]):
        FOLDER.mkdir(parents=True, exist_ok=True)
        script = Path(argv[0])
        try:
            name = ".".join(script.resolve().relative_to(REPO_ROOT).with_suffix("").parts)
        except ValueError:
            name = script.stem
        self.started = time.time()
        self.path = FOLDER / f"{time.strftime('%Y%m%d-%H%M%S')}_{os.getpid()}_{name}.json"
        self.data = {"name": name, "args": argv[1:], "pid": os.getpid(), "note": os.environ.get("GARY_RUN_NOTE"),
                     "started": self.started, "ended": None,
                     "state": "running", "error": None, "phases": [], "tally": {}, "lines": []}
        self.partial = ""
        self.saved = 0.0
        self.lock = threading.Lock()
        self.save(force=True)
        files = sorted(FOLDER.glob("*.json"))
        for old in files[:-KEEP]:
            old.unlink(missing_ok=True)
            note_path(int(old.stem.split("_")[1])).unlink(missing_ok=True)

    def save(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.saved < SAVE_EVERY:
            return
        self.saved = now
        with self.lock:
            text = json.dumps({**self.data, "lines": self.data["lines"] + ([self.partial] if self.partial else []),
                               "saved": now})
        tmp = self.path.with_suffix(".tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:                                   # (the page reading it: next time)
            pass

    def out(self, s: str) -> None:
        with self.lock:
            for ch in s.replace("\r\n", "\n"):
                if ch == "\n":
                    self.data["lines"].append(self.partial)
                    self.partial = ""
                elif ch == "\r":
                    self.partial = ""
                else:
                    self.partial += ch
            del self.data["lines"][:-LINES]
        self.save()

    def finish(self, state: str, error: str | None = None) -> None:
        if self.data["state"] != "running":
            return
        if self.partial:
            self.out("\n")
        self.data.update(state=state, error=error, ended=time.time())
        for p in self.data["phases"]:
            p["ended"] = p["ended"] or self.data["ended"]
        self.save(force=True)


class _Tee:
    """The run's stdout, copied into its status file as it prints."""

    def __init__(self, stream, run: _Run):
        self.stream, self.run = stream, run

    def write(self, s):
        n = self.stream.write(s)
        self.run.out(s)
        return n

    def flush(self):
        self.stream.flush()

    def __getattr__(self, k):
        return getattr(self.stream, k)


def start(argv: list[str] | None = None) -> None:
    """Register this process's run (once; eval/priority.py apply() calls it)."""
    global _run
    if _run is not None or os.environ.get("GARY_MONITOR", "").lower() == "off":
        return
    _run = _Run(list(argv or sys.argv))
    sys.stdout = _Tee(sys.stdout, _run)
    hook = sys.excepthook

    def failed(kind, value, tb):
        _run.finish("stopped" if issubclass(kind, KeyboardInterrupt) else "failed", f"{kind.__name__}: {value}")
        hook(kind, value, tb)
    sys.excepthook = failed
    atexit.register(lambda: _run.finish("done"))


def track(items, total: int, phase: str):
    """Yield items (results as they come), counting them as a phase of the run: done of total."""
    p = {"phase": phase, "total": int(total), "done": 0, "started": time.time(), "ended": None}
    if _run is not None:
        for q in _run.data["phases"]:
            q["ended"] = q["ended"] or p["started"]
        _run.data["phases"].append(p)
        _run.save(force=True)
    for item in items:
        p["done"] += 1
        if _run is not None:
            _run.save()
        yield item
    p["ended"] = time.time()
    if _run is not None:
        _run.save(force=True)


def tally(group: str, key: str, n: int = 1) -> None:
    """Count one more `key` in `group` (a version's games won, say)."""
    if _run is None:
        return
    g = _run.data["tally"].setdefault(group, {})
    g[key] = g.get(key, 0) + n
    _run.save()


# --- the page: its own process -----------------------------------------------------------------

_docs: dict[str, tuple[str, str]] = {}


def describe(name: str) -> tuple[str, str]:
    """(first paragraph, the rest) of a run's script's docstring: "eval.vs_rush" -> eval/vs_rush.py."""
    if name not in _docs:
        import ast
        path = REPO_ROOT.joinpath(*name.split(".")).with_suffix(".py")
        try:
            doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
        except (OSError, SyntaxError, ValueError):
            doc = ""
        first, _, rest = doc.strip().partition("\n\n")
        _docs[name] = (" ".join(first.split()), rest.strip())
    return _docs[name]


def note_path(pid: int) -> Path:
    return FOLDER / f"{pid}.note"


def _with_text(r: dict) -> dict:
    """A run with its description and its note (a note written later, by --note, wins)."""
    r["summary"], r["details"] = describe(r["name"])
    try:
        r["note"] = note_path(r["pid"]).read_text(encoding="utf-8").strip() or r.get("note")
    except OSError:
        pass
    return r


def _alive(pid: int) -> bool:
    import psutil
    return psutil.pid_exists(pid)


class _Machine:
    """CPU use of the machine and of each run (its process and its workers), between two looks."""

    def __init__(self):
        import psutil
        self.psutil = psutil
        self.procs: dict[int, object] = {}
        self.last = psutil.cpu_times()          # (psutil's own cpu_percent keeps its last look per thread)

    def cpu(self) -> float:
        now = self.psutil.cpu_times()
        a, b = self.last, now
        self.last = now
        total = sum(b) - sum(a)
        return 100.0 * (1 - (b.idle - a.idle) / total) if total > 0 else 0.0

    def cores(self, pid: int) -> tuple[float, int]:
        """(cores busy, worker processes) for a run's process tree."""
        ps = self.psutil
        try:
            root = ps.Process(pid)
            tree = [root] + root.children(recursive=True)
        except ps.Error:
            return 0.0, 0
        busy = 0.0
        for p in tree:
            q = self.procs.setdefault(p.pid, p)
            try:
                busy += q.cpu_percent(None) / 100
            except ps.Error:
                pass
        return busy, len(tree) - 1

    def others(self, known: set[int]) -> list[dict]:
        """Gary's runs not registered: python -m eval.*, train.*, gary.* (not a run's workers)."""
        ps = self.psutil
        found, workers = [], set()
        for pid in known:
            try:
                workers |= {c.pid for c in ps.Process(pid).children(recursive=True)}
            except ps.Error:
                pass
        for p in ps.process_iter(["pid", "cmdline", "create_time"]):
            cmd = p.info["cmdline"] or []
            if p.info["pid"] in known or p.info["pid"] in workers or "-m" not in cmd:
                continue
            mod = cmd[cmd.index("-m") + 1] if cmd.index("-m") + 1 < len(cmd) else ""
            if not mod.startswith(("eval.", "train.", "gary.")) or mod == "eval.monitor":
                continue
            busy, n = self.cores(p.info["pid"])
            found.append({"name": mod, "args": cmd[cmd.index("-m") + 2:], "pid": p.info["pid"],
                          "started": p.info["create_time"], "cores": round(busy, 1), "workers": n})
        return [_with_text(o) for o in found]

    def status(self) -> dict:
        runs = []
        for f in sorted(FOLDER.glob("*.json"), reverse=True)[:40]:
            try:
                r = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if r["state"] == "running":
                if _alive(r["pid"]):
                    busy, n = self.cores(r["pid"])
                    r.update(cores=round(busy, 1), workers=n)
                else:
                    r["state"] = "died"
            runs.append(_with_text(r))
        known = {r["pid"] for r in runs if r["state"] == "running"}
        vm = self.psutil.virtual_memory()
        return {"now": time.time(), "cpu": round(self.cpu(), 1), "cpus": self.psutil.cpu_count(),
                "memory": vm.percent, "runs": runs, "others": self.others(known)}


def serve(host: str, port: int) -> None:
    machine = _Machine()
    lock = threading.Lock()
    cache = {"at": 0.0, "body": b""}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.split("?")[0] == "/api/status":
                with lock:
                    if time.time() - cache["at"] > 2:
                        cache["body"] = json.dumps(machine.status()).encode()
                        cache["at"] = time.time()
                    body, kind = cache["body"], "application/json"
            elif self.path in ("/", "/index.html"):
                body, kind = PAGE.read_bytes(), "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    print(f"Gary's run monitor: http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}/", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0: also from other devices on the LAN")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--note", nargs=2, metavar=("PID", "TEXT"), help="say why a run (by its pid) is running, and quit")
    args = ap.parse_args()
    FOLDER.mkdir(parents=True, exist_ok=True)
    if args.note:
        note_path(int(args.note[0])).write_text(args.note[1], encoding="utf-8")
        return
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
