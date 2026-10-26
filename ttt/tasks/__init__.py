"""Task registry: prompts, program assembly, scoring and archive gates for ac1, ac2 and erdos."""
from __future__ import annotations

from typing import Sequence

from ttt.tasks import ac, erdos
from ttt.tasks.common import Returned
from ttt.types import Task

_TASKS: dict[str, Task] = {t.name: t for t in (ac.AC1, ac.AC2, erdos.ERDOS)}
TASK_NAMES: tuple[str, ...] = tuple(_TASKS)
__all__ = ["TASK_NAMES", "Returned", "get", "score_sequence"]


def get(name: str) -> Task:
    """Return the task called `name`."""
    if name not in _TASKS:
        raise ValueError(f"unknown task {name!r}; expected one of {TASK_NAMES}")
    return _TASKS[name]


def score_sequence(task: str, sequence: Sequence[float]) -> float:
    """Raw score of a stored construction: the AC sequence itself, or the Erdős h after normalisation."""
    get(task)
    if task == "erdos":
        return erdos.stored_c5(sequence)
    value = ac.sequence_value(task, list(sequence))
    if value is None:
        raise ValueError(f"not a valid {task} construction")
    return value
