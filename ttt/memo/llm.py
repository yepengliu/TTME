"""Model calls and deadline-limited thread pools shared by the memo stages."""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from ttt.log import warn
from ttt.types import Client, Format


@dataclass
class LLM:
    """The format and client every memo call goes through, and each call's timeout."""

    fmt: Format
    client: Client
    timeout_s: Optional[float]

    def call(self, question: str, *, effort: str, max_tokens: int) -> tuple[str, str]:
        """One auxiliary call capped at timeout_s; returns (final answer text, finish reason)."""
        text, finish = self.fmt.aux_call(self.client, question, effort=effort,
                                         max_tokens=max_tokens, temperature=1.0,
                                         timeout_s=self.timeout_s)
        return text or "", finish

    def count(self, question: str) -> int:
        """Prompt tokens of `question` as rendered for the model."""
        return self.client.count_tokens(self.fmt.render(question))


@dataclass
class PoolResult:
    """Per-item outcome of one pool run, keyed by item index."""

    done: dict = field(default_factory=dict)
    failed: dict = field(default_factory=dict)
    skipped: list = field(default_factory=list)
    uncollected: list = field(default_factory=list)


_SKIP = object()


def _worker(fn: Callable, items: Sequence, todo: queue.SimpleQueue, out: queue.SimpleQueue,
            stop: threading.Event, deadline: float) -> None:
    while not stop.is_set():
        try:
            i = todo.get_nowait()
        except queue.Empty:
            return
        if time.monotonic() >= deadline:
            out.put((i, _SKIP, None))
            continue
        try:
            out.put((i, fn(items[i]), None))
        except Exception as exc:  # handed to the collecting thread, which reports it
            out.put((i, None, exc))


def run_pool(fn: Callable[[Any], Any], items: Sequence[Any], *, workers: int,
             deadline: float, kind: str) -> PoolResult:
    """Run `fn` over `items` on daemon threads until the monotonic `deadline`."""
    res = PoolResult()
    if not items:
        return res
    todo: queue.SimpleQueue = queue.SimpleQueue()
    out: queue.SimpleQueue = queue.SimpleQueue()
    for i in range(len(items)):
        todo.put(i)
    stop = threading.Event()
    # Daemon threads, and nobody joins them: a call still in flight at the deadline is
    # abandoned, and must hold neither the step nor the process at exit.
    for n in range(min(max(1, workers), len(items))):
        threading.Thread(target=_worker, args=(fn, items, todo, out, stop, deadline),
                         name="%s-%d" % (kind, n), daemon=True).start()
    until = max(deadline, time.monotonic() + 1.0)
    finished: set = set()
    try:
        while len(finished) < len(items):
            try:
                i, value, exc = out.get(timeout=max(0.0, until - time.monotonic()))
            except queue.Empty:
                break
            finished.add(i)
            if exc is not None:
                res.failed[i] = f"{type(exc).__name__}: {exc}"[:300]
                warn(f"{kind}_call_failed", item=i, error=res.failed[i])
            elif value is _SKIP:
                res.skipped.append(i)
            else:
                res.done[i] = value
    finally:
        stop.set()
    if len(finished) < len(items):
        res.uncollected = [i for i in range(len(items)) if i not in finished]
        warn(f"{kind}_budget_spent", finished=len(finished), total=len(items))
    if res.skipped:
        warn(f"{kind}_budget_skipped", skipped=len(res.skipped), total=len(items))
    return res
