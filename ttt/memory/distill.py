"""Per-step rollout records, the step summary built from them and the distiller question."""
from __future__ import annotations

from typing import Optional

from ttt.memory.docstring import condense_docs
from ttt.memory.failures import is_timeout_signature, summarize_failures
from ttt.memory.guidelines import distiller_view
from ttt.prompts import load
from ttt.types import ChildOutcome

TOP_ROWS_PER_STEP = 5
DISTILLER_DOC_CHARS = 2000
TIMEOUT_STORE_CHARS = 20480
CODE_STORE_CHARS = 32768
_INSTRUCTION_PARTS = ("guideline_update", "crash_attribution_guide", "crash_code_guide",
                      "microdistill_synthesis_guide")


def step_record(parent_key: str, outcome: ChildOutcome) -> dict:
    """The JSON row kept for one rollout until end of step; evidence is stored for failures only."""
    ok = bool(outcome.ok)
    reason = "" if ok else (outcome.error or "")
    return {
        "parent": parent_key,
        "doc": outcome.docstring,
        "raw": None if outcome.raw_score is None else float(outcome.raw_score),
        "ok": ok,
        "better": bool(outcome.admitted and outcome.beats_parent),
        "reason": reason,
        "stdout": (outcome.stdout or "")[-TIMEOUT_STORE_CHARS:] if is_timeout_signature(reason) else "",
        "code": "" if ok else (outcome.code or "")[:CODE_STORE_CHARS],
        "code_offset": 0 if ok else int(outcome.code_offset or 0),
    }


def _success_rows(ok: list[dict], maximize: bool) -> list[str]:
    """The best scored rollout of each distinct parent, top five, best first."""
    per_parent: dict[str, list[dict]] = {}
    for r in ok:
        if r["raw"] is not None:
            per_parent.setdefault(r["parent"], []).append(r)
    pick = max if maximize else min
    best = [(pick(rows, key=lambda r: r["raw"]), len(rows)) for rows in per_parent.values()]
    best.sort(key=lambda pair: pair[0]["raw"], reverse=maximize)
    best = best[:TOP_ROWS_PER_STEP]
    docs = condense_docs([r["doc"] or "(no summary)" for r, _ in best], cap=DISTILLER_DOC_CHARS)
    lines = []
    for (r, n_from_parent), doc in zip(best, docs):
        notes = []
        if r["better"]:
            notes.append("improved its parent")
        if n_from_parent > 1:
            notes.append(f"best of {n_from_parent} from that parent")
        suffix = f" ({', '.join(notes)})" if notes else ""
        lines.append(f"  score {r['raw']:.6f}{suffix}: {doc}")
    return lines


def build_step_summary(records: list[dict], best_raw: Optional[float], maximize: bool,
                       summaries: dict[str, dict]) -> str:
    """The step outcome the distiller reads: per-parent best rows plus grouped failures."""
    fails = [r for r in records if not r["ok"]]
    direction = "higher" if maximize else "lower"
    lines = [f"Step outcome ({len(records)} rollouts, {direction} raw score is better; "
             f"best so far: {best_raw}):"]
    lines.extend(_success_rows([r for r in records if r["ok"]], maximize))
    if fails:
        lines.append(f"  failures ({len(fails)} of {len(records)} rollouts):")
        lines.extend(summarize_failures(fails, summaries))
    return "\n".join(lines)


def distiller_instruction() -> str:
    """The guideline-update instruction followed by the guides for the failure block."""
    return "\n\n".join(load(f"memory/{name}.txt") for name in _INSTRUCTION_PARTS)


def distiller_question(guidelines: list[dict], summary: str) -> str:
    """The full distiller question for the current Guideline Memory and this step's summary."""
    return load("memory/distill_question.txt").format(
        instruction=distiller_instruction(), memory=distiller_view(guidelines), summary=summary)
