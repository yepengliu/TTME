"""Model prompt formats: rendering, completion protocol, code extraction and auxiliary calls."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ttt import log
from ttt.config import Model, Sampling
from ttt.prompts import load
from ttt.types import Client, Format

CONTEXT_BUFFER = 50

HARMONY_SYSTEM = load("llm/harmony_system.txt")
RETURN = "<|return|>"
END = "<|end|>"
FINAL_CHANNEL = "<|channel|>final<|message|>"
FINAL_OPEN = "<|start|>assistant" + FINAL_CHANNEL
FINAL_MARKER = END + FINAL_OPEN
PHASE2_PREFILL = load("llm/phase2_prefill.txt")
HARMONY_FINAL_TEXT = "assistantfinal"

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
THINK_OPEN = "<think>\n"
THINK_CLOSE = "</think>"

# Channel and turn markers are special tokens; vLLM drops them unless told otherwise.
KEEP_SPECIAL_TOKENS = {"skip_special_tokens": False, "spaces_between_special_tokens": False}

_LAST_BLOCK = re.compile(r"```(python)\n(?!```)(.*?)(?:\n```)?(?=\n```|$)", re.DOTALL)
_FIRST_BLOCK = re.compile(r"```python\s+([\s\S]*?)\s*```")


def harmony_prompt(question: str, date: Optional[str], effort: str) -> str:
    """Single-turn harmony prompt ending in the bare assistant header."""
    day = date if date is not None else datetime.now().strftime("%Y-%m-%d")
    system = HARMONY_SYSTEM.format(date=day, effort=effort)
    return f"{system}<|start|>user<|message|>{question}{END}<|start|>assistant"


def chatml_prompt(question: str) -> str:
    """Single-turn chatml prompt with the assistant turn opened inside a think block."""
    return f"{IM_START}user\n{question}{IM_END}\n{IM_START}assistant\n{THINK_OPEN}"


def last_python_block(text: str) -> Optional[str]:
    """Code of the last ```python block (closing fence optional), or None if absent or empty."""
    matches = list(_LAST_BLOCK.finditer(text))
    if not matches:
        return None
    code = matches[-1].group(2).rstrip()
    if not code.strip():
        return None
    m = _FIRST_BLOCK.search(f"```python\n{code}\n```")
    return m.group(1).strip() if m is not None else None


def extract_harmony_code(text: str) -> Optional[str]:
    """Cut the completion at <|return|> and return the code of its last python block."""
    count = text.count(RETURN)
    if count > 1:
        log.warn("completion_has_multiple_returns", count=count)
        return None
    if count == 1:
        text = text[: text.index(RETURN)]
    return last_python_block(text)


def after_marker(text: str, marker: str, *, last: bool) -> str:
    """Stripped text after the first (or last) occurrence of marker, or "" without one."""
    parts = text.rsplit(marker, 1) if last else text.split(marker, 1)
    return parts[1].strip() if len(parts) > 1 else ""


def _sampling_args(sampling: Sampling, stop: str) -> dict:
    return {"temperature": sampling.temperature, "top_p": sampling.top_p,
            "top_k": sampling.top_k, "stop": [stop], "extra": dict(KEEP_SPECIAL_TOKENS)}


def two_phase_complete(prompt: str, client: Client, *, phase1_max_tokens: int,
                       sampling: Sampling) -> str:
    """Think up to phase1_max_tokens of context, then force or continue the final channel."""
    args = _sampling_args(sampling, RETURN)
    prompt_len = client.count_tokens(prompt)
    budget = phase1_max_tokens - prompt_len
    if budget <= 0:
        raise ValueError(f"prompt length {prompt_len} exceeds phase1_max_tokens {phase1_max_tokens}")
    text1, finish = client.generate(prompt, max_tokens=budget, **args)
    if finish != "length":
        return text1
    used = client.count_tokens(text1)
    if FINAL_CHANNEL in text1:
        budget = sampling.context_window - prompt_len - used - CONTEXT_BUFFER
        if budget <= 0:
            return text1
        text2, _ = client.generate(prompt + text1, max_tokens=budget, **args)
        return text1 + text2
    prefill = PHASE2_PREFILL + (FINAL_OPEN if text1.endswith(END) else FINAL_MARKER)
    budget = (sampling.context_window - prompt_len - used - client.count_tokens(prefill)
              - CONTEXT_BUFFER)
    if budget <= 0:
        return text1 + prefill
    text2, _ = client.generate(prompt + text1 + prefill, max_tokens=budget, **args)
    return text1 + prefill + text2


@dataclass(frozen=True)
class HarmonyTwoPhase:
    """gpt-oss harmony prompts with the two-phase analysis/final completion."""

    model: Model
    sampling: Sampling
    name: str = "harmony_two_phase"

    def render(self, question: str) -> str:
        """Render a generation prompt at the model's reasoning effort."""
        return harmony_prompt(question, self.model.date, self.model.reasoning_effort)

    def complete(self, prompt: str, client: Client) -> str:
        """Run the two-phase completion of a rendered prompt."""
        return two_phase_complete(prompt, client, phase1_max_tokens=self.model.phase1_max_tokens,
                                  sampling=self.sampling)

    def extract_code(self, text: str) -> Optional[str]:
        """Program text of a completion, or None."""
        return extract_harmony_code(text)

    def aux_call(self, client: Client, prompt: str, *, effort: str, max_tokens: int,
                 temperature: float = 1.0,
                 timeout_s: Optional[float] = None) -> tuple[str, str]:
        """Answer a memory/memo question; returns (final-channel text or "", finish_reason)."""
        text, finish = client.generate(harmony_prompt(prompt, self.model.date, effort),
                                       max_tokens=max_tokens, temperature=temperature,
                                       top_p=1.0, top_k=-1, stop=[RETURN], timeout_s=timeout_s)
        return after_marker(text, HARMONY_FINAL_TEXT, last=False), finish


@dataclass(frozen=True)
class ChatmlThink:
    """Chatml prompts with an opened think block and a single completion call."""

    model: Model
    sampling: Sampling
    name: str = "chatml_think"

    def render(self, question: str) -> str:
        """Render a generation prompt."""
        return chatml_prompt(question)

    def complete(self, prompt: str, client: Client) -> str:
        """Complete a rendered prompt with the rest of the context window."""
        prompt_len = client.count_tokens(prompt)
        budget = self.sampling.context_window - prompt_len - CONTEXT_BUFFER
        if budget <= 0:
            raise ValueError(f"prompt length {prompt_len} leaves no generation budget in "
                             f"context_window {self.sampling.context_window}")
        text, _ = client.generate(prompt, max_tokens=budget, **_sampling_args(self.sampling, IM_END))
        return text

    def extract_code(self, text: str) -> Optional[str]:
        """Program text of a completion, or None."""
        return last_python_block(text)

    def aux_call(self, client: Client, prompt: str, *, effort: str, max_tokens: int,
                 temperature: float = 1.0,
                 timeout_s: Optional[float] = None) -> tuple[str, str]:
        """Answer a memory/memo question (effort is ignored: chatml has no reasoning level)."""
        text, finish = client.generate(chatml_prompt(prompt), max_tokens=max_tokens,
                                       temperature=temperature, top_p=1.0, top_k=-1,
                                       stop=[IM_END], timeout_s=timeout_s)
        return after_marker(text, THINK_CLOSE, last=True), finish


def build_format(model: Model, sampling: Sampling) -> Format:
    """Return the prompt format named by model.format."""
    if model.format == "harmony_two_phase":
        if not model.phase1_max_tokens:
            raise ValueError("harmony_two_phase needs model.phase1_max_tokens")
        return HarmonyTwoPhase(model, sampling)
    if model.format == "chatml_think":
        return ChatmlThink(model, sampling)
    raise ValueError(f"unknown model format {model.format!r}")
