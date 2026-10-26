"""Stage 1: per parent, choose method(s) from the library and write parent-specific memos."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Optional

from ttt.config import AuxBudgets, MemoBudgets
from ttt.log import warn
from ttt.memo import text as T
from ttt.memo.llm import LLM, run_pool
from ttt.types import State


@dataclass(frozen=True)
class Stage1Settings:
    """Everything a Stage-1 call needs that is fixed for the run."""

    methods: int
    serial: bool
    maximize: bool
    context_window: int
    aux: AuxBudgets
    budgets: MemoBudgets


def fit_select(llm: LLM, st: Stage1Settings, parent: State, ranked: list, lineage, memory_ctx,
               taken: list) -> tuple[Optional[str], list]:
    """Call A's question with as many family heads as fit, lowest-ranked dropped first."""
    limit = st.context_window - st.budgets.merge_ctx_margin - st.aux.memo_max_tokens

    def build(fams):
        task = T.select_task(st.methods, st.serial, min(st.methods, len(fams)))
        return T.select_question(task, parent, st.maximize, lineage, memory_ctx, taken, fams)

    question = build(ranked)
    if llm.count(question) <= limit:
        return question, ranked
    lo, hi, best = 1, len(ranked) - 1, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if llm.count(build(ranked[:mid])) <= limit:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    if not best:
        warn("memo_select_prompt_too_long", parent=parent.id, families=len(ranked))
        return None, []
    warn("memo_listing_trimmed", parent=parent.id, shown=best, total=len(ranked))
    return build(ranked[:best]), ranked[:best]


def parse_select(reply: str, shown: list, k: int) -> Optional[list]:
    """The chosen families, in preference order, or None when the reply names none."""
    m = re.search(r"\{.*\}", reply or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    by_id = {f.fid: f for f in shown}
    if k == 1:
        ids = [obj.get("method_id")]
    else:
        ids = obj.get("method_ids")
        if isinstance(ids, str):
            ids = [ids]
        if not isinstance(ids, list):
            ids = [obj.get("method_id")]
    want = min(k, len(shown))
    out, seen = [], set()
    for x in ids:
        if not isinstance(x, str) or not x.strip():
            continue
        fam = by_id.get(x.strip())
        if fam is None:
            warn("memo_select_unknown_method", method=x.strip()[:80])
            continue
        if fam.fid not in seen:
            seen.add(fam.fid)
            out.append(fam)
        if len(out) == want:
            break
    return out or None


def select_one(llm: LLM, st: Stage1Settings, parent: State, ranked: list, lineage, memory_ctx,
               taken: list) -> Optional[list]:
    """Call A for one parent."""
    question, shown = fit_select(llm, st, parent, ranked, lineage, memory_ctx, taken)
    if question is None:
        return None
    reply, _finish = llm.call(question, effort="low", max_tokens=st.aux.memo_max_tokens)
    return parse_select(reply, shown, st.methods)


def write_one(llm: LLM, st: Stage1Settings, parent: State, fam) -> Optional[str]:
    """Call B: the parent-specific memo for one chosen family, or None."""
    reply, _finish = llm.call(T.write_question(parent, fam), effort="low",
                              max_tokens=st.aux.memo_max_tokens)
    memo = reply.strip()
    if memo.startswith("{"):
        m = re.search(r"\{.*\}", memo, re.S)
        try:
            obj = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            obj = None
        memo = ((obj.get("memo") or obj.get("direction") or "")
                if isinstance(obj, dict) else "")
        memo = memo.strip() if isinstance(memo, str) else ""
    if not memo:
        warn("memo_write_empty", parent=parent.id, family=fam.fid)
        return None
    return T.cap_memo(memo)


def _select_all(llm, st, parents, ctxs, ranked, lineages, deadline) -> dict:
    if not st.serial:
        res = run_pool(lambda i: select_one(llm, st, parents[i], ranked, lineages[i], ctxs[i], []),
                       list(range(len(parents))), workers=st.budgets.stage1_workers,
                       deadline=deadline, kind="memo_select")
        return {i: v for i, v in res.done.items() if v}
    chosen, taken = {}, []
    for i, parent in enumerate(parents):
        if time.monotonic() >= deadline:
            warn("memo_select_budget_spent", selected=len(chosen), total=len(parents))
            break
        try:
            got = select_one(llm, st, parent, ranked, lineages[i], ctxs[i], list(taken))
        except Exception as exc:
            warn("memo_select_call_failed", group=i, error=f"{type(exc).__name__}: {exc}"[:300])
            got = None
        if got:
            chosen[i] = got
            taken.extend(f.label or f.fid for f in got)
    return chosen


def run_stage1(llm: LLM, st: Stage1Settings, parents: list, ctxs: list, ranked: list,
               assigned: dict, did: dict) -> list:
    """Per parent, [(family, memo), ...] in preference order, or None when Stage 1 failed."""
    deadline = time.monotonic() + st.budgets.stage1_budget_s
    lineages = [T.lineage_block(p, st.maximize, assigned, did) for p in parents]
    chosen = _select_all(llm, st, parents, ctxs, ranked, lineages, deadline)
    jobs = [(i, fam) for i in sorted(chosen) for fam in chosen[i]]
    res = run_pool(lambda job: write_one(llm, st, parents[job[0]], job[1]), jobs,
                   workers=st.budgets.stage1_workers, deadline=deadline, kind="memo_write")
    out = []
    for i in range(len(parents)):
        got = [(fam, res.done[j]) for j, (gi, fam) in enumerate(jobs)
               if gi == i and res.done.get(j)]
        out.append(got or None)
    return out
