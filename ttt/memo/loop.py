"""The memo route: per-parent method directions chosen from a library mined at boundaries."""
from __future__ import annotations

import random
from dataclasses import asdict
from typing import Optional

from ttt.config import AuxBudgets, MemoConfig
from ttt.log import event, warn
from ttt.memo import boundary as B
from ttt.memo import text as T
from ttt.memo.library import Library, Note
from ttt.memo.llm import LLM
from ttt.memo.stage1 import Stage1Settings, run_stage1
from ttt.memo.transitions import Transition, from_rollout
from ttt.types import Assignment, ChildOutcome, Client, Format, Memory, State, Task

STATE_VERSION = 1


def split_rows(n: int, m: int) -> list:
    """n rows over m methods as evenly as possible, extra rows to the earlier methods."""
    return [n // m + (1 if j < n % m else 0) for j in range(m)]


def group_rows(step: int, group: int, methods: Optional[list], *, children: int, selected: int,
               control_card: str, library_sha: str) -> list:
    """The shuffled Assignments of one parent group (all control when Stage 1 failed)."""
    base = {"group": group, "library_sha": library_sha}
    control = T.control_suffix(control_card)

    def ctrl(failed):
        tags = dict(base, control_card=control_card, **({"stage1": "failed"} if failed else {}))
        return Assignment(arm="control", text=control, tags=tags)

    if not methods:
        rows = [ctrl(True) for _ in range(children)]
    else:
        rows = []
        for rank, ((fam, memo), n) in enumerate(zip(methods, split_rows(selected, len(methods)))):
            text = T.direction_suffix(fam.fid, memo)
            tags = dict(base, family=fam.fid, direction_id=T.direction_id(fam.fid),
                        method_rank=rank)
            rows += [Assignment(arm="memo", text=text, tags=dict(tags)) for _ in range(n)]
        rows += [ctrl(False) for _ in range(children - selected)]
    random.Random("memo:%d:%d" % (step, group)).shuffle(rows)
    return rows


class MemoLoop:
    """Memo-route state: library, mining pool, backlog and the current step's picks."""

    def __init__(self, cfg: MemoConfig, *, task: Task, llm: LLM, aux: AuxBudgets,
                 children_per_parent: int, context_window: int, out_dir: Optional[str]) -> None:
        self.cfg, self.task, self.llm, self.aux = cfg, task, llm, aux
        self.children, self.context_window = children_per_parent, context_window
        self.out_dir = out_dir
        self.window = (int(task.length_window[0]), int(task.length_window[1]))
        self.refused_window = self.window if cfg.include_refused else None
        self.stage1 = Stage1Settings(methods=cfg.methods_per_parent, serial=cfg.select == "serial",
                                     maximize=task.maximize, context_window=context_window,
                                     aux=aux, budgets=cfg.budgets)
        self.library = Library()
        self.seen: set = set()
        self.pending: list = []
        self.failed_hashes: list = []
        self.carry: dict = {}
        self.pool: list = []
        self.codes: dict = {}
        self.assigned: dict = {}
        self.last_boundary: Optional[int] = None
        self.health: dict = {}
        self._picks: dict = {}
        self._rows: list = []

    def plan(self, step: int, parents: list, memory: Optional[Memory]) -> Optional[list]:
        """children_per_parent Assignments per parent, or None while the library is empty."""
        self._picks, self._rows = {}, []
        ranked = self.library.ranked()
        if not ranked or not parents:
            event("memo_plan_skipped", step=step, families=len(ranked), parents=len(parents))
            return None
        sha = self.library.sha()
        ctxs = [memory.context_for(p) if memory is not None else None for p in parents]
        picks = run_stage1(self.llm, self.stage1, parents, ctxs, ranked, self.assigned,
                           self.library.did_index())
        plans = []
        for gi, methods in enumerate(picks):
            if methods is None:
                warn("memo_group_control_fallback", step=step, group=gi, parent=parents[gi].id)
            self._picks[gi] = (parents[gi].id, [f.fid for f, _ in methods] if methods else None)
            plans.append(group_rows(step, gi, methods, children=self.children,
                                    selected=self.cfg.selected_children,
                                    control_card=self.cfg.control_card, library_sha=sha))
        event("memo_plan", step=step, families=len(ranked), library_sha=sha,
              steered=sum(1 for m in picks if m), groups=len(parents),
              picks={gi: p[1] for gi, p in self._picks.items()})
        return plans

    def record(self, step: int, group: int, parent: State, assignment: Optional[Assignment],
               outcome: ChildOutcome) -> None:
        """Note one rollout: for this step's counts and, if it improved, for mining."""
        if assignment is not None and (self._picks.get(group) or (None, None))[1]:
            self._rows.append((group, assignment.arm, assignment.tags.get("family"),
                               bool(outcome.ok), bool(outcome.beats_parent)))
        got = from_rollout(step, group, parent, assignment, outcome, self.task.maximize)
        if got is not None:
            t, pcode, ccode = got
            self.codes.setdefault(t.pcode, pcode)
            self.codes.setdefault(t.ccode, ccode)
            self.pool.append(t)

    def end_of_step(self, step: int) -> None:
        """Fold this step's per-family counts into the library and remember the picks."""
        if self._picks:
            per_family, per_group = self._step_counts()
            groups: dict = {}
            for _pid, fids in self._picks.values():
                for fid in fids or []:
                    groups[fid] = groups.get(fid, 0) + 1
            self.library.add_counts(step, per_family, groups)
            for pid, fids in self._picks.values():
                if fids:
                    self.assigned[pid] = list(fids)
            event("memo_step", step=step, per_family=per_family, per_group=per_group)
        self._picks, self._rows = {}, []

    def _step_counts(self) -> tuple[dict, list]:
        per_family: dict = {}
        per_group: dict = {}
        for group, arm, fam, ok, improved in self._rows:
            fids = self._picks[group][1]
            for fid in ([fam] if arm == "memo" else fids):
                for agg in (per_family.setdefault(fid, {}), per_group.setdefault((group, fid), {})):
                    agg[f"{arm}_n"] = agg.get(f"{arm}_n", 0) + 1
                    agg[f"{arm}_valid"] = agg.get(f"{arm}_valid", 0) + ok
                    agg[f"{arm}_improved"] = agg.get(f"{arm}_improved", 0) + improved
        return per_family, [dict(v, group=g, family=f) for (g, f), v in per_group.items()]

    def boundary(self, step: int, archive_ids: frozenset) -> None:
        """Mine, merge and distill when due (steps 0, k, 2k, ...) from still-archived states."""
        if step % self.cfg.boundary_every == 0 and (self.last_boundary is None
                                                     or self.last_boundary < step):
            B.run_boundary(self, step, frozenset(archive_ids))

    def state_dict(self) -> dict:
        """JSON-serialisable state; taken after end_of_step and before the step's boundary."""
        return {"version": STATE_VERSION, "library": self.library.to_dict(),
                "seen_sigs": sorted(self.seen), "pending": [asdict(n) for n in self.pending],
                "failed_hashes": list(self.failed_hashes), "distill_carry": dict(self.carry),
                "transitions": [asdict(t) for t in self.pool], "codes": dict(self.codes),
                "assigned": {k: list(v) for k, v in self.assigned.items()},
                "last_boundary": self.last_boundary, "merge_health": dict(self.health)}

    def load_state_dict(self, state: dict) -> None:
        """Restore the state written by state_dict."""
        if state.get("version") != STATE_VERSION:
            raise ValueError("unsupported memo state version %r" % state.get("version"))
        self.library = Library.from_dict(state["library"])
        self.seen = set(state["seen_sigs"])
        self.pending = [Note(**n) for n in state["pending"]]
        self.failed_hashes = list(state["failed_hashes"])
        self.carry = {k: int(v) for k, v in state["distill_carry"].items()}
        self.pool = [Transition(**t) for t in state["transitions"]]
        self.codes = dict(state["codes"])
        self.assigned = {k: list(v) for k, v in state["assigned"].items()}
        self.last_boundary = state["last_boundary"]
        self.health = dict(state["merge_health"])
        self._picks, self._rows = {}, []


def build_memo(cfg: MemoConfig, *, task: Task, fmt: Format, client: Client, aux: AuxBudgets,
               children_per_parent: int, context_window: int,
               out_dir: Optional[str]) -> MemoLoop:
    """Construct the memo route for one run."""
    if cfg.boundary_every < 1 or cfg.methods_per_parent < 1 or cfg.notes_per_boundary < 1:
        raise ValueError("memo: boundary_every, methods_per_parent and notes_per_boundary "
                         "must be >= 1")
    if not 0 < cfg.selected_children <= children_per_parent:
        raise ValueError("memo: selected_children must be in [1, children_per_parent]")
    if cfg.selected_children < cfg.methods_per_parent:
        raise ValueError("memo: every method needs at least one selected child")
    T.control_suffix(cfg.control_card)
    llm = LLM(fmt, client, cfg.budgets.call_timeout_s)
    return MemoLoop(cfg, task=task, llm=llm, aux=aux,
                    children_per_parent=children_per_parent, context_window=context_window,
                    out_dir=out_dir)
