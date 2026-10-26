"""Improving parent->child transitions recorded between boundaries, and their selection."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

from ttt.memo.text import code_diff, diff_sig
from ttt.types import Assignment, ChildOutcome, State, raw_score

REFUSED_PER_BOUNDARY = 2


@dataclass
class Transition:
    """One improving rollout; codes are keys into the pool's code table."""

    step: int
    parent_id: str
    child_id: str
    pcode: str
    ccode: str
    pscore: Optional[float]
    cscore: Optional[float]
    group: Optional[int]
    arm: Optional[str]
    root: Optional[str]
    child_len: Optional[int] = None
    sig: Optional[str] = None


def code_key(code: str) -> str:
    """Content key of a program in the code table."""
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:24]


def _child_code(parent: State, outcome: ChildOutcome) -> Optional[str]:
    if outcome.child is not None and outcome.child.code:
        return outcome.child.code
    if not outcome.code:
        return None
    fenced = (parent.code or "").startswith("```")
    return "```python\n" + outcome.code + "\n```" if fenced else outcome.code


def from_rollout(step: int, group: int, parent: State, assignment: Optional[Assignment],
                 outcome: ChildOutcome, maximize: bool) -> Optional[tuple]:
    """(Transition, parent code, child code) for a scored child that beat its parent, else None."""
    child = outcome.child
    if not (outcome.ok and outcome.beats_parent and parent.code and child is not None):
        return None
    ccode = _child_code(parent, outcome)
    if not ccode:
        return None
    root = (parent.parents[-1].get("id") if parent.parents else None) or parent.id
    t = Transition(step=step, parent_id=parent.id, child_id=child.id,
                   pcode=code_key(parent.code), ccode=code_key(ccode),
                   pscore=raw_score(parent.value, maximize), cscore=outcome.raw_score,
                   group=group, arm=assignment.arm if assignment is not None else None,
                   root=root,
                   child_len=len(child.construction) if child.construction is not None else None)
    return t, parent.code, ccode


def channel(t: Transition, archive_ids: frozenset, window: Optional[tuple]) -> Optional[bool]:
    """False: archived child; True: refused child outside the length window; None: not minable."""
    if t.parent_id not in archive_ids:
        return None
    if t.child_id in archive_ids:
        return False
    if window is not None and t.child_len is not None and not window[0] <= t.child_len <= window[1]:
        return True
    return None


def eligible(pool: list, codes: dict, seen: set, archive_ids: frozenset,
             window: Optional[tuple]) -> tuple[list, list]:
    """(archived, refused) unseen candidates, one per diff within each (step, parent)."""
    out, refused, fresh = [], [], {}
    for t in pool:
        kind = channel(t, archive_ids, window)
        if kind is None:
            continue
        if t.sig is None:
            t.sig = diff_sig(code_diff(codes[t.pcode], codes[t.ccode])) or ""
        key = (t.step, t.parent_id)
        if not t.sig or t.sig in seen or t.sig in fresh.get(key, ()):
            continue
        fresh.setdefault(key, set()).add(t.sig)
        (refused if kind else out).append(t)
    return out, refused


def round_robin(cands: list, quota: int, step: int) -> list:
    """Newest step first; within a step, round-robin over parents, newest child first."""
    by_step: dict = {}
    for t in cands:
        by_step.setdefault(t.step, {}).setdefault(t.parent_id, []).append(t)
    picked = []
    for s in sorted(by_step, reverse=True):
        buckets = by_step[s]
        order = list(buckets)
        r = step % len(order)
        order = order[r:] + order[:r]
        depth = 0
        while len(picked) < quota:
            added = 0
            for pid in order:
                if depth < len(buckets[pid]):
                    picked.append(buckets[pid][-1 - depth])
                    added += 1
                    if len(picked) >= quota:
                        break
            if not added:
                break
            depth += 1
        if len(picked) >= quota:
            break
    return picked


def step_best(cands: list, n: int, maximize: bool) -> list:
    """The best-scoring child of each of the `n` newest steps."""
    best: dict = {}
    for t in cands:
        v = t.cscore
        if not isinstance(v, (int, float)):
            continue
        cur = best.get(t.step)
        if cur is None or (v > cur.cscore if maximize else v < cur.cscore):
            best[t.step] = t
    return [best[s] for s in sorted(best, reverse=True)[:n]]


def select(pool: list, codes: dict, seen: set, archive_ids: frozenset, *, want: int,
           step: int, reserve: int, maximize: bool, window: Optional[tuple]) -> tuple[list, dict]:
    """(transition, refused) pairs to mine: step-best reserve, refused, then round-robin fill."""
    out, refused = eligible(pool, codes, seen, archive_ids, window)
    best = step_best(out, min(reserve, want), maximize) if reserve > 0 and out else []
    best_sigs = {t.sig for t in best}
    picked_ref = round_robin(refused, min(REFUSED_PER_BOUNDARY, want), step) if refused else []
    room = max(0, want - len(best) - len(picked_ref))
    rest = round_robin([t for t in out if t.sig not in best_sigs], room, step)
    picked = [(t, False) for t in best + rest] + [(t, True) for t in picked_ref]
    coverage: dict = {}
    for t, _refused in picked:
        k = "%s/%s" % (t.step, t.group)
        coverage[k] = coverage.get(k, 0) + 1
    diag = {"eligible": len(out), "parents": len({t.parent_id for t in out}),
            "kept": len(picked), "coverage": dict(sorted(coverage.items())),
            "roots": len({t.root for t, _r in picked if t.root}),
            "refused_eligible": len(refused), "refused_kept": len(picked_ref),
            "stepbest_kept": len(best)}
    return picked, diag


def prune(pool: list, codes: dict, seen: set, archive_ids: frozenset,
          window: Optional[tuple]) -> tuple[list, dict]:
    """Drop mined transitions and those whose states left the archive (eviction is permanent)."""
    kept = []
    for t in pool:
        if channel(t, archive_ids, window) is None:
            continue
        if t.sig is None:
            t.sig = diff_sig(code_diff(codes[t.pcode], codes[t.ccode])) or ""
        if t.sig and t.sig not in seen:
            kept.append(t)
    used = {k for t in kept for k in (t.pcode, t.ccode)}
    return kept, {k: v for k, v in codes.items() if k in used}
