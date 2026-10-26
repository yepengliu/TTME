"""Erdős minimum overlap: find a step function h on [0, 2] that minimises the C5 bound."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from ttt.prompts import load
from ttt.tasks.common import (Returned, assemble_program, assembly_line_offset, new_state_id,
                              state_context, with_memory)
from ttt.types import EvalResult, State

TARGET = 0.3808
METRIC_NAME = "C₅ bound"
MAX_LENGTH = 1000
INITIAL_POINTS_RANGE = (40, 100)
INJECT_VAR = "initial_h_values"
INVALID = "Invalid solution."
FRESH_CODE_SECTION = "Write code to optimize this construction."
STORED_SUM_RTOL = 1e-12


def normalised_h(h_values: Any, n_points: Any, sum_rtol: float = 0.0) -> np.ndarray:
    """Apply the upstream checks on (h, n) and rescale h to sum n/2; raises ValueError when invalid."""
    try:
        h = np.array(h_values, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Cannot convert h_values to numpy array: {e}") from e
    if h.ndim != 1:
        raise ValueError(f"h_values must be 1D array, got shape {h.shape}")
    if h.shape[0] != n_points:
        raise ValueError(f"Expected h shape ({n_points},), got {h.shape}")
    if h.shape[0] == 0:
        raise ValueError("h_values is empty")
    if not np.all(np.isfinite(h)):
        raise ValueError("h_values contain NaN or inf values")
    if np.any(h < 0) or np.any(h > 1):
        raise ValueError(f"h(x) is not in [0, 1]. Range: [{h.min()}, {h.max()}]")
    target_sum = n_points / 2.0
    current_sum = np.sum(h)
    if abs(current_sum - target_sum) > sum_rtol * target_sum:
        if current_sum <= 0:
            raise ValueError("h_values sum to zero")
        h = h * (target_sum / current_sum)
        if np.any(h < 0) or np.any(h > 1):
            raise ValueError(f"After normalization, h(x) is not in [0, 1]. "
                             f"Range: [{h.min()}, {h.max()}]")
    return h


def c5_value(h: np.ndarray) -> float:
    """C5 = max(correlate(h, 1 - h) * dx) with dx = 2 / len(h); raises ValueError if not positive."""
    dx = 2.0 / h.shape[0]
    c5 = float(np.max(np.correlate(h, 1.0 - h, mode="full") * dx))
    if not np.isfinite(c5) or c5 <= 0:
        raise ValueError(f"Computed C5 is not a positive finite number: {c5}")
    return c5


def stored_c5(sequence: Sequence[float]) -> float:
    """C5 of a stored construction under the same checks as score(); raises ValueError when invalid."""
    # A stored h was normalised once already; rescaling it again by a factor within rounding of 1
    # would only perturb the last bits of its score.
    return c5_value(normalised_h(sequence, len(sequence), STORED_SUM_RTOL))


@dataclass(frozen=True)
class ErdosTask:
    """A program's run() returns (h_values, c5_bound, n_points); the score is C5 recomputed from h."""

    name: str = "erdos"
    maximize: bool = False
    entrypoint: str = "run"
    length_window: tuple[int, int] = (1, MAX_LENGTH)

    def initial_state(self, rng: np.random.Generator) -> State:
        """Upstream random start: h = 0.5 plus a zero-mean perturbation, 40-99 samples, no code."""
        n_points = int(rng.integers(*INITIAL_POINTS_RANGE))
        construction = np.ones(n_points) * 0.5
        perturbation = rng.uniform(-0.4, 0.4, n_points)
        perturbation = perturbation - np.mean(perturbation)
        construction = construction + perturbation
        dx = 2.0 / n_points
        correlation = np.correlate(construction, 1 - construction, mode="full") * dx
        c5_bound = float(np.max(correlation))
        return State(id=new_state_id(rng), timestep=-1, value=-c5_bound, code="",
                     construction=construction.tolist())

    def refresh_initial(self, state: State, rng: np.random.Generator) -> None:
        """Upstream keeps the Erdős initial construction when it is picked again."""
        return None

    def question(self, parent: State, memory_context: Optional[str]) -> str:
        """The upstream prompt for this parent, with the memory block after the state context."""
        ctx = with_memory(state_context(parent, TARGET, METRIC_NAME, self.maximize), memory_context)
        construction_section = ""
        if parent.construction is not None and len(parent.construction) > 0:
            construction_section = load("tasks/erdos/construction_section.txt").format(
                n=len(parent.construction))
        if parent.code and parent.code.strip():
            code_section = load("tasks/erdos/code_section.txt")
        else:
            code_section = FRESH_CODE_SECTION
        return load("tasks/erdos/question.txt").format(
            state_ctx=ctx, construction_section=construction_section, code_section=code_section)

    def assemble(self, code: str, parent: State) -> str:
        """Program actually executed: numpy, verify_c5_solution, `initial_h_values`, the model's code."""
        return assemble_program(code, parent, load("tasks/erdos/verifier.txt"), INJECT_VAR)

    def line_offset(self, code: str, parent: State) -> int:
        """Harness lines before the model's code, to map sandbox line numbers back to the model's."""
        return assembly_line_offset(self.assemble(code, parent), code)

    def score(self, result: Any) -> EvalResult:
        """Validate (h, c5, n) as upstream does and score C5 recomputed from the normalised h."""
        # The self-reported c5 is ignored: the upstream check accepts any report within
        # atol=1e-4 of the true value, which programs can exploit by under-reporting.
        if isinstance(result, Returned):
            result = result.value
        try:
            h_values, _reported_c5, n_points = result
            h = normalised_h(h_values, n_points)
            c5 = c5_value(h)
        except (TypeError, ValueError):
            return EvalResult(ok=False, raw_score=None, construction=None, stdout="", error=INVALID)
        return EvalResult(ok=True, raw_score=c5, construction=h.tolist(), stdout="", error=None)

    def admissible(self, construction: Sequence[float]) -> bool:
        """Archive gate on construction length."""
        return self.length_window[0] <= len(construction) <= self.length_window[1]


ERDOS = ErdosTask()
