"""In-process stand-in for a served model, plus format-correct completions for tests."""
from __future__ import annotations

import threading
from typing import Callable, Optional, Union

from ttt.llm.formats import FINAL_MARKER, HARMONY_FINAL_TEXT, THINK_CLOSE

Reply = Union[str, tuple[str, str]]
Responder = Callable[[str, dict], Reply]


class FakeClient:
    """Client whose completions come from responder(prompt, params); records every call."""

    def __init__(self, responder: Responder) -> None:
        self.responder = responder
        self.calls: list[tuple[str, dict]] = []
        self._lock = threading.Lock()

    def generate(self, prompt: str, *, max_tokens: int, temperature: float = 1.0,
                 top_p: float = 1.0, top_k: int = -1, stop: Optional[list[str]] = None,
                 extra: Optional[dict] = None,
                 timeout_s: Optional[float] = None) -> tuple[str, str]:
        """Return the responder's reply; a bare string finishes with "stop"."""
        params = {"max_tokens": max_tokens, "temperature": temperature, "top_p": top_p,
                  "top_k": top_k, "stop": list(stop) if stop else None,
                  "extra": dict(extra) if extra else None, "timeout_s": timeout_s}
        with self._lock:
            self.calls.append((prompt, params))
        reply = self.responder(prompt, params)
        return reply if isinstance(reply, tuple) else (reply, "stop")

    def count_tokens(self, text: str) -> int:
        """Approximate token count: four characters per token."""
        return len(text) // 4

    def wait_ready(self, timeout_s: float = 0.0, interval_s: float = 0.0) -> None:
        """Always ready."""


def harmony_completion(program: str, analysis: str = "Plan the construction.") -> str:
    """A harmony completion (special tokens kept) whose final message is the program."""
    return (f"<|channel|>analysis<|message|>{analysis}{FINAL_MARKER}"
            f"```python\n{program}\n```")


def chatml_completion(program: str, thinking: str = "Plan the construction.") -> str:
    """A chatml completion that closes its think block and then gives the program."""
    return f"{thinking}\n{THINK_CLOSE}\n\n```python\n{program}\n```"


def completion_for(fmt_name: str, program: str) -> str:
    """A generation completion wrapping program in the given format."""
    if fmt_name == "harmony_two_phase":
        return harmony_completion(program)
    if fmt_name == "chatml_think":
        return chatml_completion(program)
    raise ValueError(f"unknown format {fmt_name!r}")


def aux_reply_for(fmt_name: str, answer: str) -> str:
    """An auxiliary-call completion (special tokens skipped) whose final answer is answer."""
    if fmt_name == "harmony_two_phase":
        return f"analysisThinking about it.{HARMONY_FINAL_TEXT}{answer}"
    if fmt_name == "chatml_think":
        return f"Thinking about it.\n{THINK_CLOSE}\n{answer}"
    raise ValueError(f"unknown format {fmt_name!r}")


def is_generation_call(params: dict) -> bool:
    """True for rollout generation calls, which keep special tokens; aux calls do not."""
    return bool(params.get("extra")) and params["extra"].get("skip_special_tokens") is False


def program_responder(fmt_name: str, program: Union[str, Callable[[str], str]],
                      aux_answer: str = "") -> Responder:
    """Responder that answers generation calls with program and aux calls with aux_answer."""
    def respond(prompt: str, params: dict) -> str:
        if not is_generation_call(params):
            return aux_reply_for(fmt_name, aux_answer)
        code = program(prompt) if callable(program) else program
        return completion_for(fmt_name, code)
    return respond
