"""Dev server launcher: `uvicorn app:app --reload`, without orphaned processes on Windows.

Windows has no "child dies with its parent" rule, so killing the top process of
`uv run uvicorn --reload` leaves the reloader and its server child alive, still holding the port.
This launcher puts itself in a Job Object flagged KILL_ON_JOB_CLOSE (every child it spawns, the
reload worker included, joins the job), so the whole tree is reaped when the launcher dies. It
also exits when its own parent (e.g. `uv run`) disappears.

    cd backend
    uv run python serve.py [--port 8080] [--no-reload]
"""
import argparse
import os
import sys
import threading

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    _JobObjectExtendedLimitInformation = 9
    _SYNCHRONIZE = 0x00100000
    _INFINITE = 0xFFFFFFFF

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _BASIC_LIMITS(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                    ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _EXTENDED_LIMITS(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BASIC_LIMITS),
                    ("IoInfo", _IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    _k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    _k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _k32.GetCurrentProcess.restype = wintypes.HANDLE
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]

    _job_handle = None  # must stay referenced: closing it (process exit) kills the whole job

    def _kill_tree_when_i_die() -> None:
        global _job_handle
        job = _k32.CreateJobObjectW(None, None)
        limits = _EXTENDED_LIMITS()
        limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not (job
                and _k32.SetInformationJobObject(job, _JobObjectExtendedLimitInformation,
                                                 ctypes.byref(limits), ctypes.sizeof(limits))
                and _k32.AssignProcessToJobObject(job, _k32.GetCurrentProcess())):
            print(f"warning: could not create job object (error {ctypes.get_last_error()}); "
                  "child processes may outlive this launcher", file=sys.stderr)
            return
        _job_handle = job

    class _PROCESSENTRY32(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                    ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
                    ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG),
                    ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_wchar * 260)]

    _k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    _k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    _k32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32)]
    _k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32)]
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]

    def _parent_of(pid: int):
        """(parent pid, exe name) of `pid`, or None if it is gone."""
        snap = _k32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
        entry = _PROCESSENTRY32()
        entry.dwSize = ctypes.sizeof(entry)
        try:
            ok = _k32.Process32FirstW(snap, ctypes.byref(entry))
            while ok:
                if entry.th32ProcessID == pid:
                    return entry.th32ParentProcessID, entry.szExeFile
                ok = _k32.Process32NextW(snap, ctypes.byref(entry))
        finally:
            _k32.CloseHandle(snap)
        return None

    def _exit_when_parent_dies() -> None:
        # A venv's python.exe is only a launcher that spawns the real interpreter, so our direct
        # parent is that launcher (which survives its own parent being killed). Watch the
        # process above it too (typically `uv run`).
        watch = [os.getppid()]
        parent = _parent_of(watch[0])
        if parent and parent[1].lower().startswith("python"):
            watch.append(parent[0])

        handles = [h for h in (_k32.OpenProcess(_SYNCHRONIZE, False, pid) for pid in watch) if h]
        if not handles:
            return

        def wait(handle):
            _k32.WaitForSingleObject(handle, _INFINITE)
            os._exit(0)  # job handle closes -> reload worker is killed too

        for h in handles:
            threading.Thread(target=wait, args=(h,), daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8080, help="default 8080 (8000 is blocked here)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-reload", action="store_true")
    args = parser.parse_args()

    if sys.platform == "win32":
        _kill_tree_when_i_die()
        _exit_when_parent_dies()

    import uvicorn
    uvicorn.run("app:app", host=args.host, port=args.port, reload=not args.no_reload,
                reload_excludes=["tests/*"])


# The reload worker re-imports this module as __mp_main__, so nothing may run on import.
if __name__ == "__main__":
    main()
