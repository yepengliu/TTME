"""Autocorrelation inequalities: AC1 minimises an upper bound, AC2 maximises a lower bound."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Optional, Sequence

import numpy as np

from ttt.prompts import load
from ttt.tasks.common import (Returned, assemble_program, assembly_line_offset, new_state_id,
                              state_context, with_memory)
from ttt.types import EvalResult, State, signed_value

PROMPT_BUDGET_S = 1000
LENGTH_LIMITS = (1000, 100000)
INITIAL_LENGTH_RANGE = (1000, 8000)
INJECT_VAR = "height_sequence_1"
INVALID = "Invalid solution."


@lru_cache(maxsize=None)
def verifier(name: str) -> Callable[[Any], float]:
    """Return the task's evaluate_sequence, compiled from the text injected into every program."""
    # Scoring runs the same text the candidate calls, so the two can never drift apart.
    namespace: dict = {"np": np}
    exec(load(f"tasks/{name}/verifier.txt"), namespace)
    return namespace["evaluate_sequence"]


def sequence_value(name: str, sequence: Any) -> Optional[float]:
    """Upstream verify-then-score: the raw score, or None when evaluate_sequence raises or returns inf."""
    try:
        value = verifier(name)(sequence)
    except Exception:  # upstream counts any exception raised by the verifier as an invalid solution
        return None
    if value == np.inf:
        return None
    return float(value)


def _constant_vector(rng: np.random.Generator) -> list[float]:
    height = float(rng.random())
    return [height] * int(rng.integers(*INITIAL_LENGTH_RANGE))


@dataclass(frozen=True)
class AutocorrelationTask:
    """A program returns a list of non-negative step heights, scored by the task's evaluate_sequence."""

    name: str
    maximize: bool
    entrypoint: str
    metric_name: str
    target: float
    length_window: tuple[int, int] = LENGTH_LIMITS

    def initial_state(self, rng: np.random.Generator) -> State:
        """Constant-height vector of random length, seeded with the upstream initial program."""
        construction = _constant_vector(rng)
        value = signed_value(sequence_value(self.name, construction), self.maximize)
        code = "```python\n" + load(f"tasks/{self.name}/initial_program.txt") + "\n```"
        return State(id=new_state_id(rng), timestep=-1, value=value, code=code,
                     construction=construction)

    def refresh_initial(self, state: State, rng: np.random.Generator) -> None:
        """Redraw a picked initial state's constant vector, as upstream does for AC tasks."""
        state.construction = _constant_vector(rng)
        state.value = signed_value(sequence_value(self.name, state.construction), self.maximize)
        state.construction_hash = None

    def question(self, parent: State, memory_context: Optional[str]) -> str:
        """The upstream prompt for this parent, with the memory block after the state context."""
        ctx = with_memory(state_context(parent, self.target, self.metric_name, self.maximize),
                          memory_context)
        if parent.construction:
            ctx += f"\nLength of the construction: {len(parent.construction)}"
        return load(f"tasks/{self.name}/question.txt").format(
            shown_verifier=load(f"tasks/{self.name}/shown_verifier.txt"),
            literature=load(f"tasks/{self.name}/literature.txt"),
            budget_s=PROMPT_BUDGET_S, state_ctx=ctx)

    def assemble(self, code: str, parent: State) -> str:
        """Program actually executed: numpy, verifier, `height_sequence_1`, then the model's code."""
        return assemble_program(code, parent, load(f"tasks/{self.name}/verifier.txt"), INJECT_VAR)

    def line_offset(self, code: str, parent: State) -> int:
        """Harness lines before the model's code, to map sandbox line numbers back to the model's."""
        return assembly_line_offset(self.assemble(code, parent), code)

    def score(self, result: Any) -> EvalResult:
        """Score a sandbox Returned record or a live return value; the construction is the returned list."""
        if isinstance(result, Returned):
            # evaluate_sequence rejects non-lists and non-int/float or bool elements; JSON erased
            # those types, so the sandbox's record of them decides validity first.
            if not (result.is_list and result.plain_numbers):
                return EvalResult(ok=False, raw_score=None, construction=None, stdout="",
                                  error=INVALID)
            result = result.value
        value = sequence_value(self.name, result)
        if value is None:
            return EvalResult(ok=False, raw_score=None, construction=None, stdout="", error=INVALID)
        return EvalResult(ok=True, raw_score=value, construction=list(result), stdout="", error=None)

    def admissible(self, construction: Sequence[float]) -> bool:
        """Archive gate on construction length."""
        return self.length_window[0] <= len(construction) <= self.length_window[1]


AC1 = AutocorrelationTask(name="ac1", maximize=False, entrypoint="propose_candidate",
                          metric_name="upper bound", target=1.5030)
AC2 = AutocorrelationTask(name="ac2", maximize=True, entrypoint="construct_function",
                          metric_name="lower bound", target=0.97)
