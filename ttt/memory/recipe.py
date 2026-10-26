"""Lineage recipe: how a parent's construction was built from its admitted ancestors."""
from __future__ import annotations

from typing import Optional

from ttt.memory.docstring import condense_docs
from ttt.prompts import load

MAX_RECIPE_STEPS = 6
RECIPE_DOC_CHARS = 2000


def render_recipe(chain: list[str], registry: dict[str, dict], maximize: bool) -> Optional[str]:
    """Render up to six registered chain states, oldest change first, as score transitions plus docstrings."""
    chain = [key for key in chain if key in registry][:MAX_RECIPE_STEPS]
    if len(chain) < 2:
        return None
    steps = list(zip(chain, chain[1:]))[::-1]
    docs = condense_docs([registry[newer]["doc"] or "(no algorithm summary)" for newer, _ in steps],
                         cap=RECIPE_DOC_CHARS)
    lines = [load("memory/recipe_heading.txt")]
    for (newer, older), doc in zip(steps, docs):
        n, o = registry[newer], registry[older]
        if n["raw"] is None or o["raw"] is None:
            lines.append(f"  {doc}")
            continue
        delta = n["raw"] - o["raw"]
        gain = delta if maximize else -delta
        lines.append(f"  {o['raw']:.6f} -> {n['raw']:.6f} (gain {gain:+.2e}): {doc}")
    return "\n".join(lines)
