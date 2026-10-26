"""Docstring extraction and condensing for generated programs."""
from __future__ import annotations

import ast
import difflib
import re
from typing import Optional, Sequence

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_SHARED_SENTENCE_RATIO = 0.6
_SHARED_SENTENCE_SCAN = 3
_MAX_SHARED_DROPPED = 3
# What parsing hostile model text raises: NUL bytes give ValueError, deep nesting the other two.
_PARSE_ERRORS = (SyntaxError, ValueError, MemoryError, RecursionError)


def unfence(code: str) -> str:
    """Strip a surrounding ```python fence from stored program text."""
    if not code:
        return ""
    text = code.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        text = text[first_nl + 1:] if first_nl != -1 else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text


def extract_docstring(code: str, entrypoint: Optional[str]) -> Optional[str]:
    """Module docstring of a program, else the entry point's or a top-level function's docstring."""
    if not code:
        return None
    try:
        tree = ast.parse(code)
    except _PARSE_ERRORS:
        return None
    doc = ast.get_docstring(tree)
    if doc:
        return " ".join(doc.split())
    funcs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if entrypoint:
        # Helpers come first in these programs, so the entry point must win the fallback.
        funcs.sort(key=lambda n: n.name != entrypoint)
    for node in funcs:
        d = ast.get_docstring(node)
        if d:
            return " ".join(d.split())
    return None


def _normalize_sentence(text: str) -> str:
    return " ".join(re.sub(r"[^0-9a-z]+", " ", text.lower()).split())


def _is_shared_sentence(sentence: str, others: list[list[str]]) -> bool:
    """True when at least half of the other docs open with a near-identical sentence."""
    a = _normalize_sentence(sentence)
    if not a:
        return True
    hits = 0
    for other in others:
        for candidate in other[:_SHARED_SENTENCE_SCAN]:
            b = _normalize_sentence(candidate)
            if b and difflib.SequenceMatcher(None, a, b).ratio() >= _SHARED_SENTENCE_RATIO:
                hits += 1
                break
    return hits * 2 >= len(others)


def truncate_on_word(text: str, cap: int) -> str:
    """Cut text to at most `cap` characters at a word boundary, marking the cut with " ..."."""
    text = text.strip()
    if len(text) <= cap:
        return text
    cut = text[:cap]
    space = cut.rfind(" ")
    if space > cap // 2:
        cut = cut[:space]
    return cut.rstrip(" ,;:-") + " ..."


def condense_docs(docs: Sequence[Optional[str]], cap: int) -> list[str]:
    """Drop the leading sentences the docs share, then truncate each on a word boundary."""
    parts = [_SENTENCE_SPLIT.split(d.strip()) if d else [] for d in docs]
    if len(parts) > 1:
        # Compare against a snapshot so the result does not depend on input order.
        original = [list(p) for p in parts]
        for i, part in enumerate(parts):
            others = [p for j, p in enumerate(original) if j != i and p]
            dropped = 0
            while (len(part) > 1 and others and dropped < _MAX_SHARED_DROPPED
                   and _is_shared_sentence(part[0], others)):
                part.pop(0)
                dropped += 1
    return [truncate_on_word(" ".join(p), cap) for p in parts]
