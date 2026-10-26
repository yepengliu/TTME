"""Sandbox child `wrapper.py PROGRAM ENTRYPOINT RESULT_JSON ERROR_TXT CPUS`: run the entrypoint, write JSON."""
# Self-contained on purpose: the child runs in the scrubbed environment, where ttt is not importable.
import builtins
import functools
import importlib.util
import json
import multiprocessing
import os
import sys
import traceback
from concurrent import futures
from concurrent.futures import process

_BLOCKED_OS_CALLS = ("remove", "unlink", "rename", "replace", "rmdir", "mkdir", "makedirs",
                     "chmod", "chown", "link", "symlink")
_ORIGINAL_POOL = process.ProcessPoolExecutor
_CPU_CAP = [1]


def _write_blocked(*args, **kwargs):
    raise PermissionError("Filesystem mutation disabled in sandboxed workers")


def _sandbox_worker():
    """Disable file writes and filesystem mutation inside pool worker processes."""
    original_open = builtins.open

    def read_only_open(file, mode="r", *args, **kwargs):
        if any(ch in mode for ch in ("w", "a", "+", "x")):
            raise PermissionError("File writes are disabled in sandboxed workers")
        return original_open(file, mode, *args, **kwargs)

    builtins.open = read_only_open
    for name in _BLOCKED_OS_CALLS:
        if hasattr(os, name):
            setattr(os, name, _write_blocked)


def install_limits(cpus):
    """Force spawn and cap every ProcessPoolExecutor created in this process at `cpus` workers."""
    _CPU_CAP[0] = max(1, int(cpus))
    multiprocessing.set_start_method("spawn", force=True)
    process.ProcessPoolExecutor = CappedProcessPool
    futures.ProcessPoolExecutor = CappedProcessPool


def _run_initializers(cpus, user_initializer, *initargs):
    """Apply the limits inside a pool worker, run the program's own initializer, then lock down."""
    install_limits(cpus)
    try:
        if user_initializer is not None:
            user_initializer(*initargs)
    finally:
        _sandbox_worker()


class CappedProcessPool(_ORIGINAL_POOL):
    """ProcessPoolExecutor that never exceeds the CPU budget and always uses spawn."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(),
                 **kwargs):
        requested = (os.cpu_count() or 1) if max_workers is None else max_workers
        workers = max(1, min(int(requested), _CPU_CAP[0]))
        super().__init__(max_workers=workers,
                         mp_context=mp_context or multiprocessing.get_context("spawn"),
                         initializer=functools.partial(_run_initializers, _CPU_CAP[0], initializer),
                         initargs=tuple(initargs), **kwargs)


def to_jsonable(value):
    """Convert numpy arrays and scalars to Python lists and numbers, and tuples to lists."""
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    numpy = sys.modules.get("numpy")
    if numpy is not None and isinstance(value, (numpy.ndarray, numpy.generic)):
        return to_jsonable(value.tolist())
    return value


def type_signature(value):
    """Type facts the JSON conversion erases: qualified top-level type, list-ness, plain-number elements."""
    plain = isinstance(value, (list, tuple)) and all(
        not isinstance(x, bool) and isinstance(x, (int, float)) for x in value)
    kind = type(value)
    return {"type_name": f"{kind.__module__}.{kind.__qualname__}",
            "is_list": isinstance(value, list), "plain_numbers": plain}


def encode_result(value):
    """JSON envelope {"value", "signature"} for a returned value; raises if it is not serialisable."""
    return json.dumps({"value": to_jsonable(value), "signature": type_signature(value)})


def _write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _call_entrypoint(program_path, entrypoint):
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    sys.modules["program"] = program
    spec.loader.exec_module(program)
    return getattr(program, entrypoint)()


def main(argv):
    """Run the candidate and leave either result.json or error.txt behind."""
    program_path, entrypoint, result_path, error_path, cpus = argv[1:6]
    install_limits(cpus)
    try:
        value = _call_entrypoint(program_path, entrypoint)
    except Exception as e:  # the candidate's own failure: recorded and shown back to the model
        _write(error_path, f"Program execution failed: {e}")
        traceback.print_exc()
        return
    try:
        text = encode_result(value)
    except (TypeError, ValueError, RecursionError) as e:
        _write(error_path, f"Result is not JSON-serialisable: {e}")
        return
    _write(result_path, text)


if __name__ == "__main__":
    # The guard matters: spawned pool workers re-import this file as __mp_main__.
    main(sys.argv)
