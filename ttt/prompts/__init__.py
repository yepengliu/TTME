"""Prompt templates shipped as text files under ttt/prompts/."""
from __future__ import annotations

from functools import lru_cache
from importlib import resources


@lru_cache(maxsize=None)
def load(name: str) -> str:
    """Return the template at ttt/prompts/<name> (e.g. "memo/select.txt") verbatim."""
    return resources.files(__package__).joinpath(name).read_text(encoding="utf-8")
