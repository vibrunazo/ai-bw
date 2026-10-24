#!/usr/bin/env python3
"""Close the SC:R single-instance object inside a running client (automated procexp step).

SC:R refuses a second client: at startup it creates the named kernel Event
\\Sessions\\<n>\\BaseNamedObjects\\Starcraft Check For Other Instances and exits within
seconds if it already exists. Closing that handle inside the running instance (Process
Explorer: Handles -> Close Handle) lets a second client run. This tool does the same from
the command line:

    python adapters/scr_bridge/tools/close_mutex.py --list   # show matching handles only
    python adapters/scr_bridge/tools/close_mutex.py          # close them (all SC:R pids)
    python adapters/scr_bridge/tools/close_mutex.py --pid N  # only that process

Runs without elevation when the client runs as the same user (OpenProcess only needs
PROCESS_DUP_HANDLE; SeDebugPrivilege is enabled best-effort). If Windows refuses with
ACCESS_DENIED, rerun from an elevated prompt — that is the one case that still needs admin.

Safety: only handles whose object leaf name matches (default: the single-instance Event
above) are touched; File handles are never name-queried (some block). Everything else in
the process is left alone.
"""
from __future__ import annotations

import argparse
import ctypes
import sys
from ctypes import wintypes

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
ntdll = ctypes.WinDLL("ntdll")
advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

SystemExtendedHandleInformation = 64
ObjectNameInformation = 1
ObjectTypeInformation = 2
STATUS_INFO_LENGTH_MISMATCH = 0xC0000004
MAX_HANDLE_TABLE = 1 << 30      # bytes: a stray process with ~2.7 million handles passed 64 MB (issue #8)
PROCESS_DUP_HANDLE = 0x0040
DUPLICATE_SAME_ACCESS = 0x2
DUPLICATE_CLOSE_SOURCE = 0x1
TOKEN_ADJUST_PRIVILEGES = 0x20
TOKEN_QUERY = 0x8
SE_PRIVILEGE_ENABLED = 0x2

ntdll.NtQuerySystemInformation.argtypes = [ctypes.c_int, ctypes.c_void_p, wintypes.ULONG,
                                           ctypes.POINTER(wintypes.ULONG)]
ntdll.NtQuerySystemInformation.restype = wintypes.LONG
ntdll.NtQueryObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.ULONG,
                                ctypes.POINTER(wintypes.ULONG)]
ntdll.NtQueryObject.restype = wintypes.LONG

kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.DuplicateHandle.argtypes = [wintypes.HANDLE, wintypes.HANDLE, wintypes.HANDLE,
                                     ctypes.POINTER(wintypes.HANDLE), wintypes.DWORD,
                                     wintypes.BOOL, wintypes.DWORD]
kernel32.DuplicateHandle.restype = wintypes.BOOL
kernel32.GetCurrentProcess.restype = wintypes.HANDLE
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.CreateMutexW.restype = wintypes.HANDLE

advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                      ctypes.POINTER(wintypes.HANDLE)]
advapi32.OpenProcessToken.restype = wintypes.BOOL
advapi32.LookupPrivilegeValueW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, ctypes.c_void_p]
advapi32.LookupPrivilegeValueW.restype = wintypes.BOOL
advapi32.AdjustTokenPrivileges.argtypes = [wintypes.HANDLE, wintypes.BOOL, ctypes.c_void_p,
                                           wintypes.DWORD, ctypes.c_void_p, ctypes.c_void_p]
advapi32.AdjustTokenPrivileges.restype = wintypes.BOOL


class _UNICODE_STRING(ctypes.Structure):
    _fields_ = [("Length", wintypes.USHORT), ("MaximumLength", wintypes.USHORT),
                ("Buffer", ctypes.c_void_p)]


class _SYSTEM_HANDLE_ENTRY(ctypes.Structure):
    _fields_ = [("Object", ctypes.c_void_p),
                ("UniqueProcessId", ctypes.c_size_t),
                ("HandleValue", ctypes.c_size_t),
                ("GrantedAccess", wintypes.ULONG),
                ("CreatorBackTraceIndex", wintypes.USHORT),
                ("ObjectTypeIndex", wintypes.USHORT),
                ("HandleAttributes", wintypes.ULONG),
                ("Reserved", wintypes.ULONG)]


class _SYSTEM_HANDLE_INFORMATION_EX(ctypes.Structure):
    _fields_ = [("NumberOfHandles", ctypes.c_size_t),
                ("Reserved", ctypes.c_size_t),
                ("Handles", _SYSTEM_HANDLE_ENTRY * 1)]


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]


class _LUID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Luid", _LUID), ("Attributes", wintypes.DWORD)]


class _TOKEN_PRIVILEGES(ctypes.Structure):
    _fields_ = [("PrivilegeCount", wintypes.DWORD),
                ("Privileges", _LUID_AND_ATTRIBUTES * 1)]


class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_wchar * 260),
    ]


def find_pids(name: str) -> list[int]:
    want = name if name.lower().endswith(".exe") else name + ".exe"
    snap = kernel32.CreateToolhelp32Snapshot(0x2, 0)
    if snap in (0, -1, 0xFFFFFFFF):
        return []
    out: list[int] = []
    try:
        pe = _PROCESSENTRY32W()
        pe.dwSize = ctypes.sizeof(_PROCESSENTRY32W)
        ok = kernel32.Process32FirstW(snap, ctypes.byref(pe))
        while ok:
            if pe.szExeFile.lower() == want.lower():
                out.append(int(pe.th32ProcessID))
            ok = kernel32.Process32NextW(snap, ctypes.byref(pe))
        return out
    finally:
        kernel32.CloseHandle(snap)


def enable_debug_privilege() -> bool:
    tok = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(),
                                     TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, ctypes.byref(tok)):
        return False
    try:
        luid = _LUID()
        if not advapi32.LookupPrivilegeValueW(None, "SeDebugPrivilege", ctypes.byref(luid)):
            return False
        tp = _TOKEN_PRIVILEGES(1, (_LUID_AND_ATTRIBUTES(luid, SE_PRIVILEGE_ENABLED),))
        return bool(advapi32.AdjustTokenPrivileges(tok, False, ctypes.byref(tp),
                                                   ctypes.sizeof(tp), None, None))
    finally:
        kernel32.CloseHandle(tok)


def system_handles() -> list[_SYSTEM_HANDLE_ENTRY]:
    size = 1 << 20
    while size <= MAX_HANDLE_TABLE:
        buf = ctypes.create_string_buffer(size)
        ret = wintypes.ULONG(0)
        status = ntdll.NtQuerySystemInformation(SystemExtendedHandleInformation, buf, size,
                                                ctypes.byref(ret)) & 0xFFFFFFFF
        if status == STATUS_INFO_LENGTH_MISMATCH:
            # (the table grows between calls: what it asked for, with room, or twice the size)
            size = max(size * 2, ret.value + (1 << 20))
            continue
        if status != 0:
            raise OSError(f"NtQuerySystemInformation failed: 0x{status & 0xFFFFFFFF:08X}")
        info = ctypes.cast(buf, ctypes.POINTER(_SYSTEM_HANDLE_INFORMATION_EX)).contents
        base = ctypes.addressof(info.Handles)
        return [ctypes.cast(base + i * ctypes.sizeof(_SYSTEM_HANDLE_ENTRY),
                            ctypes.POINTER(_SYSTEM_HANDLE_ENTRY)).contents
                for i in range(info.NumberOfHandles)]
    raise OSError(f"handle table larger than {MAX_HANDLE_TABLE >> 20} MB — giving up")


def query_object_name(handle: int) -> str:
    buf = ctypes.create_string_buffer(1024)
    ret = wintypes.ULONG(0)
    status = ntdll.NtQueryObject(handle, ObjectNameInformation, buf, len(buf), ctypes.byref(ret))
    if status != 0:
        return ""
    us = ctypes.cast(buf, ctypes.POINTER(_UNICODE_STRING)).contents
    if not us.Buffer or not us.Length:
        return ""
    return ctypes.wstring_at(us.Buffer, us.Length // 2)


def query_object_type(handle: int) -> str:
    buf = ctypes.create_string_buffer(256)
    ret = wintypes.ULONG(0)
    status = ntdll.NtQueryObject(handle, ObjectTypeInformation, buf, len(buf),
                                 ctypes.byref(ret)) & 0xFFFFFFFF
    if status != 0:
        return ""
    us = ctypes.cast(buf, ctypes.POINTER(_UNICODE_STRING)).contents
    if not us.Buffer or not us.Length:
        return ""
    return ctypes.wstring_at(us.Buffer, us.Length // 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="show matching handles, close nothing")
    ap.add_argument("--pid", type=int, default=0,
                    help="only this process (default: every StarCraft.exe)")
    ap.add_argument("--name", default="Starcraft Check For Other Instances",
                    help="object leaf name to match (default: SC:R's single-instance Event)")
    ap.add_argument("--all", action="store_true",
                    help="diagnostic: list every named Mutant handle in the target pids")
    args = ap.parse_args()

    pids = [args.pid] if args.pid else find_pids("StarCraft")
    if not pids:
        print("no StarCraft.exe running — nothing to close")
        return 0

    have_debug = enable_debug_privilege()
    print(f"target pids: {pids}; SeDebugPrivilege "
          f"{'enabled' if have_debug else 'not available (usually fine)'}")

    want_leaf = args.name.lower()
    closed = 0
    matched = 0
    for pid in pids:
        proc = kernel32.OpenProcess(PROCESS_DUP_HANDLE, False, pid)
        if not proc:
            err = ctypes.get_last_error()
            print(f"pid {pid}: OpenProcess failed (winerror {err}) — "
                  + ("rerun from an elevated prompt" if err == 5 else "skipping"),
                  file=sys.stderr)
            continue
        try:
            if args.all:
                hist: dict[int, int] = {}
                for e in system_handles():
                    if e.UniqueProcessId == pid:
                        hist[e.ObjectTypeIndex] = hist.get(e.ObjectTypeIndex, 0) + 1
                print(f"pid {pid}: {sum(hist.values())} handles, by type index: "
                      + ", ".join(f"{k}:{v}" for k, v in sorted(hist.items())))
            for e in system_handles():
                if e.UniqueProcessId != pid:
                    continue
                dup = wintypes.HANDLE()
                if not kernel32.DuplicateHandle(proc, e.HandleValue, kernel32.GetCurrentProcess(),
                                                ctypes.byref(dup), 0, False, DUPLICATE_SAME_ACCESS):
                    continue
                try:
                    tname = query_object_type(dup)
                    if tname == "File":
                        continue  # name queries on file handles can block — skip
                    name = query_object_name(dup)
                finally:
                    kernel32.CloseHandle(dup)
                leaf = name.rsplit("\\", 1)[-1].lower() if name else ""
                if args.all:
                    if name:
                        print(f"pid {pid} handle=0x{e.HandleValue:04X} type={tname} name={name!r}")
                    continue
                if leaf != want_leaf:
                    continue
                matched += 1
                print(f"pid {pid} handle=0x{e.HandleValue:04X} name={name}")
                if args.list:
                    continue
                out = wintypes.HANDLE()
                if kernel32.DuplicateHandle(proc, e.HandleValue, kernel32.GetCurrentProcess(),
                                            ctypes.byref(out), 0, False, DUPLICATE_CLOSE_SOURCE):
                    kernel32.CloseHandle(out)
                    closed += 1
                    print("  closed")
                else:
                    print(f"  close failed (winerror {ctypes.get_last_error()})", file=sys.stderr)
        finally:
            kernel32.CloseHandle(proc)

    if not matched:
        print("no matching handle found (client may have died, or never held one)")
    elif not args.list:
        print(f"closed {closed} of {matched} matching handle(s) — a second client may now start")
    return 0


if __name__ == "__main__":
    sys.exit(main())