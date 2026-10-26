"""Records and interfaces shared by the loop, tasks, evaluation, LLM and memory."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence

import numpy as np


@dataclass
class State:
    """One archived program and its construction; `value` is signed so higher is better."""

    id: str
    timestep: int
    value: Optional[float]
    code: str
    construction: Optional[list[float]]
    parent_values: list[float] = field(default_factory=list)
    parents: list[dict] = field(default_factory=list)
    observation: str = ""
    construction_hash: Optional[str] = None


@dataclass
class EvalResult:
    """Outcome of running and scoring one candidate program."""

    ok: bool
    raw_score: Optional[float]
    construction: Optional[list[float]]
    stdout: str
    error: Optional[str]
    infra_error: bool = False


@dataclass
class ChildOutcome:
    """What the memory layers learn about one rollout of a parent."""

    ok: bool
    raw_score: Optional[float]
    admitted: bool
    beats_parent: bool
    docstring: Optional[str]
    code: Optional[str]
    code_offset: int
    error: Optional[str]
    stdout: str
    child: Optional[State] = None


@dataclass
class Assignment:
    """Direction text appended to one child's question, plus row tags for the log."""

    arm: str
    text: str
    tags: dict = field(default_factory=dict)


class Task(Protocol):
    """A math-construction task: prompt, harness assembly, scoring and archive gates."""

    name: str
    maximize: bool
    entrypoint: str
    length_window: tuple[int, int]

    def initial_state(self, rng: np.random.Generator) -> State: ...

    def refresh_initial(self, state: State, rng: np.random.Generator) -> None: ...

    def question(self, parent: State, memory_context: Optional[str]) -> str: ...

    def assemble(self, code: str, parent: State) -> str: ...

    def line_offset(self, code: str, parent: State) -> int: ...

    def score(self, result: Any) -> EvalResult: ...

    def admissible(self, construction: Sequence[float]) -> bool: ...


class Evaluator(Protocol):
    """Runs an assembled program in a sandbox and scores it in a fresh process."""

    capacity: int

    def evaluate(self, task: str, program: str, timeout_s: float, cpus: int) -> EvalResult: ...


class Client(Protocol):
    """Completion endpoint of one served model."""

    def generate(self, prompt: str, *, max_tokens: int, temperature: float = 1.0,
                 top_p: float = 1.0, top_k: int = -1, stop: Optional[list[str]] = None,
                 extra: Optional[dict] = None,
                 timeout_s: Optional[float] = None) -> tuple[str, str]: ...

    def count_tokens(self, text: str) -> int: ...


class Format(Protocol):
    """Prompt rendering, completion protocol and code extraction for one model family."""

    name: str

    def render(self, question: str) -> str: ...

    def complete(self, prompt: str, client: Client) -> str: ...

    def extract_code(self, text: str) -> Optional[str]: ...

    def aux_call(self, client: Client, prompt: str, *, effort: str, max_tokens: int,
                 temperature: float = 1.0,
                 timeout_s: Optional[float] = None) -> tuple[str, str]:
        """Run a memory/memo call; returns (final answer text or "", finish_reason)."""
        ...


class Memory(Protocol):
    """Lineage recipe and Guideline Memory injected into each parent's question."""

    def register(self, state: State, raw_score: Optional[float]) -> None: ...

    def context_for(self, parent: State) -> Optional[str]: ...

    def record_child(self, parent: State, outcome: ChildOutcome) -> None: ...

    def end_of_step(self, step: int, best_raw: Optional[float]) -> None: ...

    def state_dict(self) -> dict: ...

    def load_state_dict(self, state: dict) -> None: ...


class Memo(Protocol):
    """Per-parent method directions chosen from a library mined at boundaries."""

    def plan(self, step: int, parents: list[State],
             memory: Optional[Memory]) -> Optional[list[list[Assignment]]]: ...

    def record(self, step: int, group: int, parent: State, assignment: Assignment,
               outcome: ChildOutcome) -> None: ...

    def end_of_step(self, step: int) -> None: ...

    def boundary(self, step: int, archive_ids: frozenset[str]) -> None:
        """Mine and merge when due; only transitions whose states are still archived are mined."""
        ...

    def state_dict(self) -> dict: ...

    def load_state_dict(self, state: dict) -> None: ...


def raw_score(value: Optional[float], maximize: bool) -> Optional[float]:
    """Convert a signed archive value back to the task's raw score."""
    if value is None:
        return None
    return value if maximize else -value


def signed_value(raw: Optional[float], maximize: bool) -> Optional[float]:
    """Convert a raw task score into the archive's higher-is-better value."""
    if raw is None:
        return None
    return raw if maximize else -raw
