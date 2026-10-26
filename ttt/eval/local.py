"""Evaluate candidates as local subprocesses."""
from __future__ import annotations

import dataclasses
import threading
from typing import Optional

from ttt import tasks
from ttt.eval.sandbox import SandboxError, resolve_python, run_program
from ttt.eval.score import failed, score_result
from ttt.log import warn
from ttt.types import EvalResult


def evaluate_program(task: str, program: str, timeout_s: float, cpus: int,
                     python: Optional[str] = None) -> EvalResult:
    """Run an assembled program in the sandbox, then score its result in a fresh process."""
    entrypoint = tasks.get(task).entrypoint
    try:
        result_json, stdout, error = run_program(program, entrypoint, timeout_s, cpus, python)
    except SandboxError as e:
        warn("eval_sandbox_error", task=task, error=str(e)[:500])
        return failed(f"eval_sandbox_error: {e}", infra_error=True)
    if result_json is None:
        return failed(error or "Program produced no result file.", stdout)
    return dataclasses.replace(score_result(task, result_json, cpus=cpus), stdout=stdout)


class LocalEvaluator:
    """Evaluator that runs at most `workers` sandboxed candidates at a time on this machine."""

    def __init__(self, workers: int, python: Optional[str] = None):
        if workers < 1:
            raise ValueError("LocalEvaluator needs at least one worker")
        self.capacity = workers
        self._python = resolve_python(python)
        self._slots = threading.Semaphore(workers)

    def evaluate(self, task: str, program: str, timeout_s: float, cpus: int) -> EvalResult:
        """Block until a slot is free, then run and score one program."""
        with self._slots:
            return evaluate_program(task, program, timeout_s, cpus, self._python)
