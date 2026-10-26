"""Prompt state context and program assembly shared by every task."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ttt.types import State

OBSERVATION_TAIL_CHARS = 500


@dataclass(frozen=True)
class Returned:
    """A sandboxed program's return value after JSON conversion, with the type facts the conversion erased."""

    value: Any
    type_name: str
    is_list: bool
    plain_numbers: bool


def state_context(state: State, target: float, metric_name: str, maximize: bool) -> str:
    """Render the parent's code, value and stdout block like upstream State.to_prompt, in the task's direction."""
    ctx = f"You are iteratively optimizing {metric_name}."
    direction = "higher" if maximize else "lower"
    if state.code and state.code.strip():
        ctx += f"\nHere is the last code we ran:\n{state.code}"
    else:
        ctx += "\nNo previous code available."
    if state.parent_values and state.value is not None and state.construction:
        before = state.parent_values[0] if maximize else -state.parent_values[0]
        after = state.value if maximize else -state.value
        gap = target - after if maximize else after - target
        ctx += (f"\nHere is the {metric_name} before and after running the code above "
                f"({direction} is better): {before:.6f} -> {after:.6f}")
        ctx += (f"\nTarget: {target}. Current gap: {gap:.6f}. "
                "Further improvements will also be generously rewarded.")
    elif state.value is not None:
        after = state.value if maximize else -state.value
        gap = target - after if maximize else after - target
        # Upstream hard-codes "higher" here even for minimised tasks.
        ctx += f"\nCurrent {metric_name} ({direction} is better): {after:.6f}"
        ctx += (f"\nTarget: {target}. Current gap: {gap:.6f}. "
                "Further improvements will also be generously rewarded.")
    else:
        ctx += f"\nTarget {metric_name}: {target}"
    if state.observation and state.observation.strip():
        stdout = state.observation.strip()
        if len(stdout) > OBSERVATION_TAIL_CHARS:
            stdout = "\n\n\t\t ...(TRUNCATED)...\n" + stdout[-OBSERVATION_TAIL_CHARS:]
        ctx += f"\n\n--- Previous Program Output ---\n{stdout}\n--- End Output ---"
    return ctx


def with_memory(ctx: str, memory_context: Optional[str]) -> str:
    """Append the memory block directly after the state context."""
    return ctx + "\n\n" + memory_context if memory_context else ctx


def _compiles(source: str) -> bool:
    try:
        compile(source, "<model>", "exec")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False
    return True


def hoist_future(code: str) -> tuple[str, str]:
    """Split `from __future__` lines off a program that compiles on its own; returns (hoisted, rest)."""
    # The harness prepends code, so a model's `from __future__` would otherwise be a SyntaxError.
    if "from __future__" not in code or not _compiles(code):
        return "", code
    hoisted, kept = [], []
    for line in code.splitlines(keepends=True):
        if line.strip().startswith("from __future__ import"):
            hoisted.append(line if line.endswith("\n") else line + "\n")
        else:
            kept.append(line)
    rest = "".join(kept)
    if not hoisted or not _compiles("".join(hoisted) + rest):
        return "", code
    return "".join(hoisted), rest


def assemble_program(code: str, parent: State, verifier_src: str, inject_var: str) -> str:
    """Prefix the model's code with numpy, the task verifier and the parent's construction."""
    if parent is None:
        raise ValueError("a parent state is required to assemble a program")
    hoisted, body = hoist_future(code)
    base = hoisted + "import numpy as np\n\n" + verifier_src + "\n\n"
    if parent.construction is not None:
        values = [x.item() if isinstance(x, np.generic) else x for x in parent.construction]
        base += f"{inject_var} = np.array({values!r})\n\n"
    return base + body


def assembly_line_offset(assembled: str, code: str) -> int:
    """Number of harness lines that precede the model's own code in an assembled program."""
    _, body = hoist_future(code or "")
    if not body:
        return assembled.count("\n")
    idx = assembled.rfind(body)
    if idx < 0:
        return assembled.count("\n") - body.count("\n")
    return assembled.count("\n", 0, idx)


def new_state_id(rng: np.random.Generator) -> str:
    """A uuid4 string drawn from `rng`, so seeded runs get reproducible ids."""
    return str(uuid.UUID(bytes=rng.bytes(16), version=4))
