"""Run one assembled program under a supervisor: child Python, scrubbed environment, hard timeout."""
from __future__ import annotations

import errno
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from functools import lru_cache
from importlib import resources
from typing import BinaryIO, Optional

from ttt.log import warn

STDOUT_TAIL_CHARS = 20000
RESULT_LIMIT_BYTES = 64 * 1024 * 1024
ERROR_LIMIT_BYTES = 64 * 1024
STATUS_LIMIT_BYTES = 64 * 1024
KILL_GRACE_S = 5.0
REAP_S = 10.0
# The supervisor needs at most timeout + KILL_GRACE_S + REAP_S plus its own start-up; past that it is stuck.
SUPERVISOR_SLACK_S = 30.0
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
_FILES = ("supervisor.py", "wrapper.py", "program.py", "result.json", "error.txt")


class SandboxError(RuntimeError):
    """The sandbox could not run the program at all: an infrastructure failure, not the candidate's."""


@lru_cache(maxsize=None)
def _script_source(name: str) -> str:
    return resources.files(__package__).joinpath(name).read_text(encoding="utf-8")


def resolve_python(python: Optional[str]) -> str:
    """Absolute path of the interpreter that runs candidates (default: the current one)."""
    if not python:
        return sys.executable
    found = shutil.which(python)
    if found is None:
        raise ValueError(f"python interpreter not found: {python!r}")
    return os.path.abspath(found)


def child_env(root: str, cpus: int, python: str, extra: Optional[dict] = None) -> dict:
    """Fresh environment for a sandboxed child: PATH, private HOME/TMPDIR, unbuffered output, thread caps."""
    home, tmp = os.path.join(root, "home"), os.path.join(root, "tmp")
    os.makedirs(home, exist_ok=True)
    os.makedirs(tmp, exist_ok=True)
    # Unbuffered because stdout is a file: a child killed at the timeout would lose buffered output.
    env = {"PATH": os.pathsep.join([os.path.dirname(python), os.defpath]),
           "HOME": home, "TMPDIR": tmp, "PYTHONUNBUFFERED": "1"}
    env.update({var: str(cpus) for var in THREAD_VARS})
    env.update(extra or {})
    return env


def wait_exited(pid: int, timeout_s: float) -> bool:
    """Wait for a child to exit without reaping it, so its process-group id cannot be reused yet."""
    deadline = time.monotonic() + timeout_s
    delay = 0.002
    while os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(delay, remaining))
        delay = min(delay * 2, 0.1)
    return True


def signal_group(pgid: int, sig: int) -> bool:
    """Send `sig` to a process group; False when the group has no members left."""
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return False
    return True


def read_tail(f: BinaryIO, limit_chars: int) -> str:
    """The last `limit_chars` characters of an open UTF-8 text file, undecodable bytes replaced."""
    f.seek(0, os.SEEK_END)
    f.seek(max(0, f.tell() - 4 * limit_chars))
    return f.read().decode("utf-8", errors="replace")[-limit_chars:]


def read_candidate_file(path: str, limit_bytes: int) -> tuple[Optional[bytes], Optional[str]]:
    """Up to `limit_bytes` + 1 bytes of a candidate-written file: (data, None if absent; refusal reason)."""
    name = os.path.basename(path)
    try:
        # O_NONBLOCK so a FIFO planted under this name cannot block the open.
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None, None
    except OSError as e:
        if e.errno == errno.ELOOP:
            return None, f"{name} is a symlink; refused."
        return None, f"{name} is unreadable: {e.strerror}."
    with os.fdopen(fd, "rb") as f:
        if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
            return None, f"{name} is not a regular file; refused."
        return f.read(limit_bytes + 1), None


def remove_tree(path: str) -> None:
    """Delete a scratch directory, warning instead of raising if the candidate made it undeletable."""
    try:
        shutil.rmtree(path)
    except OSError as e:
        warn("sandbox_cleanup_failed", path=path, error=str(e))


def describe_exit(returncode: int) -> str:
    """'exit code N' or 'killed by signal N (NAME)', flagging SIGKILL as a likely out-of-memory kill."""
    if returncode >= 0:
        return f"exit code {returncode}"
    number = -returncode
    try:
        name = signal.Signals(number).name
    except ValueError:
        name = "unknown signal"
    hint = ", out of memory?" if number == signal.SIGKILL else ""
    return f"killed by signal {number} ({name}{hint})"


@lru_cache(maxsize=1)
def _warn_no_subreaper() -> None:
    """Warn once per process: without a subreaper, descendants that leave the process group can survive."""
    warn("sandbox_no_subreaper", detail="descendants in new sessions can outlive their candidate")


def _write_inputs(root: str, program: str) -> None:
    # surrogatepass: lone surrogates then fail to compile, as the candidate's own error.
    with open(os.path.join(root, "program.py"), "w", encoding="utf-8", errors="surrogatepass") as f:
        f.write(program)
    for name in ("supervisor.py", "wrapper.py"):
        with open(os.path.join(root, name), "w", encoding="utf-8") as f:
            f.write(_script_source(name))


def _start(python: str, root: str, entrypoint: str, timeout_s: float, cpus: int,
           out: BinaryIO) -> tuple[subprocess.Popen, int]:
    """Launch supervisor + wrapper in a new session; returns the process and the status pipe's read end."""
    p = {name: os.path.join(root, name) for name in _FILES}
    read_fd, write_fd = os.pipe()
    argv = [python, p["supervisor.py"], str(write_fd), repr(float(timeout_s)), repr(KILL_GRACE_S),
            repr(REAP_S), python, p["wrapper.py"], p["program.py"], entrypoint, p["result.json"],
            p["error.txt"], str(cpus)]
    try:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                cwd=root, env=child_env(root, cpus, python), start_new_session=True,
                                pass_fds=(write_fd,))
    except BaseException:
        os.close(read_fd)
        raise
    finally:
        os.close(write_fd)
    return proc, read_fd


def _read_status(fd: int) -> dict:
    """Merge the JSON lines the supervisor wrote; non-blocking, so a lingering writer cannot stall us."""
    os.set_blocking(fd, False)
    data = b""
    while len(data) <= STATUS_LIMIT_BYTES:
        try:
            chunk = os.read(fd, 4096)
        except BlockingIOError:
            break
        if not chunk:
            break
        data += chunk
    status: dict = {}
    for line in data.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            warn("sandbox_status_unreadable", line=line[:200].decode("utf-8", errors="replace"))
            continue
        if isinstance(record, dict):
            status.update(record)
    return status


def _finish(proc: subprocess.Popen, status_fd: int, timeout_s: float) -> dict:
    """Wait for the supervisor (with a backstop), then return its status plus how it ended itself."""
    try:
        try:
            overran = not wait_exited(proc.pid, timeout_s + SUPERVISOR_SLACK_S)
        finally:
            # Backstop: anything still in the supervisor's own group goes too (normally nothing is).
            signal_group(proc.pid, signal.SIGKILL)
            proc.wait()
        status = _read_status(status_fd)
    finally:
        os.close(status_fd)
    if overran:
        warn("sandbox_supervisor_overran", pid=proc.pid, timeout_s=timeout_s)
    status.update(supervisor_returncode=proc.returncode, overran=overran)
    return status


def _read_result(root: str) -> tuple[Optional[str], Optional[str]]:
    """The candidate's result.json as text, or why it is refused; (None, None) when there is none."""
    data, refused = read_candidate_file(os.path.join(root, "result.json"), RESULT_LIMIT_BYTES)
    if refused or data is None:
        return None, refused
    if len(data) > RESULT_LIMIT_BYTES:
        return None, f"result.json exceeds the {RESULT_LIMIT_BYTES}-byte limit; refused."
    return data.decode("utf-8", errors="replace"), None


def _read_error(root: str) -> Optional[str]:
    """The candidate's error.txt (truncated), why it is refused, or None when there is none."""
    data, refused = read_candidate_file(os.path.join(root, "error.txt"), ERROR_LIMIT_BYTES)
    if refused or data is None:
        return refused
    text = data[:ERROR_LIMIT_BYTES].decode("utf-8", errors="replace")
    return text + " [truncated]" if len(data) > ERROR_LIMIT_BYTES else text


def _outcome(root: str, status: dict, timeout_s: float) -> tuple[Optional[str], Optional[str]]:
    """(result JSON, error) from the files the candidate left and how its processes ended."""
    if "started" not in status:
        why = status.get("spawn_error") or ("it overran" if status["overran"] else
                                            describe_exit(status["supervisor_returncode"]))
        raise SandboxError(f"sandbox supervisor did not start the program: {why}")
    if status.get("subreaper") is False:
        _warn_no_subreaper()
    if status.get("clean") is False:
        warn("sandbox_descendants_survived", killed=status.get("killed"))
    if status.get("timed_out") or status["overran"]:
        return None, f"Evaluation timed out after {timeout_s:g}s."
    result, refused = _read_result(root)
    if result is not None or refused:
        return result, refused
    error = _read_error(root)
    if error is not None:
        return None, error
    how = (describe_exit(status["returncode"]) if "returncode" in status else
           f"sandbox supervisor {describe_exit(status['supervisor_returncode'])}")
    return None, f"Program produced no result file; {how}."


def run_program(program: str, entrypoint: str, timeout_s: float, cpus: int,
                python: Optional[str] = None) -> tuple[Optional[str], str, Optional[str]]:
    """(result JSON or None, stdout tail, error) of `entrypoint()`; raises SandboxError if it cannot start."""
    try:
        python = resolve_python(python)
    except ValueError as e:
        raise SandboxError(str(e)) from e
    cpus = max(1, int(cpus))
    try:
        root = tempfile.mkdtemp(prefix="ttt_sbx_")
    except OSError as e:
        raise SandboxError(f"cannot create a sandbox directory: {e}") from e
    try:
        try:
            _write_inputs(root, program)
            out = open(os.path.join(root, "stdout.txt"), "w+b")
        except OSError as e:
            raise SandboxError(f"cannot write the sandbox files: {e}") from e
        with out:
            try:
                proc, status_fd = _start(python, root, entrypoint, timeout_s, cpus, out)
            except OSError as e:
                raise SandboxError(f"cannot start the sandbox supervisor: {e}") from e
            status = _finish(proc, status_fd, timeout_s)
            # Read through our own descriptor: the candidate may have unlinked or replaced the file.
            stdout = read_tail(out, STDOUT_TAIL_CHARS)
        result, error = _outcome(root, status, timeout_s)
        return result, stdout, error
    finally:
        remove_tree(root)
