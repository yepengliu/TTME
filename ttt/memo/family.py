"""Family memory: rewrite each touched family's memory from the notes it just received."""
from __future__ import annotations

import time

from ttt.log import event
from ttt.memo.library import Library, memory_sha
from ttt.memo.llm import LLM, run_pool
from ttt.memo.text import family_question

CARRY_MAX_STEPS = 6


def touched_notes(lib: Library, step: int, carry: dict) -> dict:
    """{family id: [note, ...]} placed at `step` or carried from a recent boundary."""
    out: dict = {}
    for fid in lib.families:
        for n in lib.notes_of(fid):
            placed = n.placed_step if n.placed_step is not None else n.step
            if (placed == step or n.sig in carry) and (n.memo or "").strip():
                out.setdefault(fid, []).append(n)
    return out


def _apply(lib: Library, fids: list, res, step: int, stats: dict) -> None:
    for i, fid in enumerate(fids):
        fam = lib.families[fid]
        if i in res.skipped:
            stats["budget"] += 1
            fam.memory_error = {"step": step, "why": "budget"}
            continue
        if i in res.uncollected:
            continue
        stats["called"] += 1
        text, why = res.done.get(i, ("", res.failed.get(i)))
        if not text:
            stats["failed"] += 1
            fam.memory_error = {"step": step, "why": why}
            continue
        stats["ok"] += 1
        fam.memory, fam.memory_sha = text, memory_sha(text)
        fam.memory_step, fam.memory_rounds = step, fam.memory_rounds + 1
        fam.memory_error = None


def distill(llm: LLM, lib: Library, step: int, carry: dict, *, budget_s: float, workers: int,
            max_tokens: int) -> tuple[dict, dict]:
    """Rewrite touched family memories; returns (stats, notes carried to the next boundary)."""
    live = {s: v for s, v in carry.items() if step - int(v) <= CARRY_MAX_STEPS}
    touched = touched_notes(lib, step, live)
    fids = sorted(touched)
    stats = {"called": 0, "ok": 0, "failed": 0, "budget": 0,
             "skipped": len(lib.families) - len(fids), "touched": len(fids),
             "carry_expired": len(carry) - len(live),
             "carried_in": sum(1 for f in fids for n in touched[f] if n.sig in live)}

    def one(fid):
        fam = lib.families[fid]
        q = family_question(fam.label or fid, fam.memory or "",
                            [n.memo.strip() for n in touched[fid]])
        text, finish = llm.call(q, effort="medium", max_tokens=max_tokens)
        if finish == "length":
            return "", "truncated"
        return text.strip(), None

    res = run_pool(one, fids, workers=workers, deadline=time.monotonic() + budget_s,
                   kind="memo_family_memory")
    _apply(lib, fids, res, step, stats)
    stats["uncollected"] = len(res.uncollected)
    new_carry, missed = {}, 0
    for fid in fids:
        fam = lib.families[fid]
        err = fam.memory_error or {}
        if fam.memory_step == step or (err.get("step") == step and err.get("why") != "budget"):
            continue
        missed += 1
        for n in touched[fid]:
            new_carry[n.sig] = int(live.get(n.sig, step))
    stats.update(carried_out=len(new_carry), carried_out_families=missed)
    event("memo_family_memory", step=step, **stats)
    return stats, new_carry
