"""Mining calls: one note per independent change in each selected transition."""
from __future__ import annotations

import re
import time
from typing import Optional

from ttt.config import AuxBudgets, MemoBudgets
from ttt.memo.library import Note
from ttt.memo.llm import LLM, run_pool
from ttt.memo.text import mining_question

NOTE_PARTS_CAP = 3
_SPLIT = re.compile(r"(?m)^\s*={3,}\s*$")


def mine(llm: LLM, picks: list, codes: dict, window: tuple, aux: AuxBudgets,
         budgets: MemoBudgets) -> list:
    """(transition, memo, finish) for each (transition, refused) pick that got text, in order."""
    def one(pick):
        t, refused = pick
        banner = (t.child_len, window[0], window[1]) if refused else None
        q = mining_question(codes[t.pcode], codes[t.ccode], t.pscore, t.cscore, banner)
        return llm.call(q, effort="low", max_tokens=aux.memo_max_tokens)

    res = run_pool(one, picks, workers=budgets.mining_workers,
                   deadline=time.monotonic() + budgets.mining_budget_s, kind="memo_mining")
    return [(picks[i][0],) + tuple(res.done[i]) for i in range(len(picks))
            if i in res.done and res.done[i][0].strip()]


def split_note(memo: str, cap: int = NOTE_PARTS_CAP) -> list:
    """1..cap note texts from one memo; text past the cap folds into the last part."""
    parts = [p.strip() for p in _SPLIT.split(memo or "") if p.strip()]
    if len(parts) <= 1:
        return [memo]
    if len(parts) > cap:
        parts = parts[:cap - 1] + ["\n\n".join(parts[cap - 1:])]
    return parts


def expand_notes(mined: list, step: int) -> tuple[list, int]:
    """(notes, number of memos split); a split note's parts get sigs <sig>#<i>."""
    notes, n_split = [], 0
    for t, memo, _finish in mined:
        parts = split_note(memo)
        n_split += len(parts) > 1
        for i, part in enumerate(parts):
            notes.append(Note(sig=t.sig if len(parts) == 1 else "%s#%d" % (t.sig, i),
                              memo=part, step=step, rollout_step=t.step,
                              parent_id=t.parent_id, child_id=t.child_id, group=t.group,
                              arm=t.arm, root=t.root, pscore=t.pscore, cscore=t.cscore))
    return notes, n_split
