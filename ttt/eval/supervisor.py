"""Sandbox supervisor `supervisor.py STATUS_FD TIMEOUT_S GRACE_S REAP_S CMD...`: run CMD, kill leftovers."""
# Self-contained like wrapper.py: it runs in the scrubbed child environment, where ttt is not importable.
import contextlib
import ctypes
import json
import os
import signal
import subprocess
import sys
import time

PR_SET_CHILD_SUBREAPER = 36


def become_subreaper():
    """Make orphaned descendants reparent to this process (Linux prctl); False where that is unavailable."""
    try:
        prctl = ctypes.CDLL(None, use_errno=True).prctl
    except (OSError, AttributeError):
        return False
    return prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) == 0


def descendants(root):
    """Pids of every process below `root`, read from /proc; empty where /proc is unavailable."""
    try:
        names = [name for name in os.listdir("/proc") if name.isdigit()]
    except OSError:
        return []
    children = {}
    for name in names:
        try:
            with open(f"/proc/{name}/stat", "rb") as f:
                ppid = int(f.read().rsplit(b")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue  # the process exited while we were scanning
        children.setdefault(ppid, []).append(int(name))
    found, frontier = [], [root]
    while frontier:
        frontier = [child for pid in frontier for child in children.get(pid, [])]
        found.extend(frontier)
    return found


def signal_all(pids, sig):
    """Send `sig` to every pid in `pids` that still exists."""
    for pid in pids:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, sig)


def wait_exited(pid, timeout_s):
    """Wait for a child to exit without reaping it, so its pid cannot be reused while we signal it."""
    deadline = time.monotonic() + timeout_s
    delay = 0.002
    while os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(delay, remaining))
        delay = min(delay * 2, 0.1)
    return True


def kill_leftovers(reap_s):
    """SIGKILL and reap every remaining descendant; returns (how many were signalled, all gone?)."""
    deadline = time.monotonic() + reap_s
    killed = set()
    while True:
        try:
            while os.waitpid(-1, os.WNOHANG)[0]:
                pass
        except ChildProcessError:
            return len(killed), True
        if time.monotonic() >= deadline:
            return len(killed), False
        pids = descendants(os.getpid())
        signal_all(pids, signal.SIGKILL)
        killed.update(pids)
        time.sleep(0.01)


def supervise(proc, timeout_s, grace_s, reap_s):
    """Wait for the command, stop it at the timeout, then clear out every descendant it left behind."""
    finished = wait_exited(proc.pid, timeout_s)
    if not finished:
        signal_all({proc.pid, *descendants(os.getpid())}, signal.SIGTERM)
        if not wait_exited(proc.pid, grace_s):
            proc.kill()
    returncode = proc.wait()
    killed, clean = kill_leftovers(reap_s)
    return {"returncode": returncode, "timed_out": not finished, "killed": killed, "clean": clean}


def main(argv):
    """Start the command, then write the supervision outcome as JSON lines to STATUS_FD."""
    status_fd = int(argv[1])
    timeout_s, grace_s, reap_s = (float(x) for x in argv[2:5])
    with os.fdopen(status_fd, "w", encoding="utf-8") as status:
        report = {"subreaper": become_subreaper()}
        try:
            proc = subprocess.Popen(argv[5:], stdin=subprocess.DEVNULL)
        except OSError as e:
            report["spawn_error"] = f"{type(e).__name__}: {e}"
            status.write(json.dumps(report) + "\n")
            return 1
        status.write(json.dumps({"started": proc.pid}) + "\n")
        status.flush()
        report.update(supervise(proc, timeout_s, grace_s, reap_s))
        status.write(json.dumps(report) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
