"""Global merge fitted to the context: a fixed ladder of request shapes, run in rounds."""
from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from ttt.log import event, warn
from ttt.memo.library import Fusions, Library, label_key, normalise_groups
from ttt.memo.llm import LLM
from ttt.memo.merge import MergeReply, merge_call
from ttt.memo.text import FULL_EXEMPLAR_CHARS, merge_question

MAX_MERGE_NOTES = 24
MIN_NOTES = 4
LAST_RESORT_NOTES = (MIN_NOTES, 3, 2, 1)
FAILED_HASHES_KEEP = 64
PLAN_TRIES = 8
MAX_CONSECUTIVE_FAILURES = 3
ROUND_ESTIMATE_MIN_S = 90.0
_SMALL = tuple(range(8, MIN_NOTES - 1, -1))
RUNGS = (
    ("R0", 2, FULL_EXEMPLAR_CHARS, (MAX_MERGE_NOTES,)),
    ("R1", 2, FULL_EXEMPLAR_CHARS, (16, 12, 8)),
    ("R2", 1, FULL_EXEMPLAR_CHARS, _SMALL),
    ("R3", 1, 160, _SMALL),
    ("R4", 0, 0, _SMALL),
)
ROTATE_KINDS = ("zero_absorption", "zero_absorption_no_room", "unknown_ids_over_cap",
                "bad_json", "no_json_stop", "collapse", "no_direction", "zero_families",
                "not_object", "malformed_reply")


@dataclass(frozen=True)
class MergeSettings:
    """Context, token and time limits of the merge rounds of one boundary."""

    ctx: int
    margin: int
    start_max_tokens: int
    max_tokens_cap: int
    drain: bool
    drain_min_notes: int
    max_rounds: int
    rounds_budget_s: float
    call_timeout_s: float


@dataclass
class Plan:
    """One request shape that fits the context."""

    rung: str
    rung_index: int
    k: int
    chars: int
    n: int
    prompt_tokens: int
    retry_tokens: int
    max_tokens: int
    parity: bool
    hash: str


@dataclass
class MergeResult:
    """What the rounds of one boundary did to the queue and the library."""

    queue: list
    rounds: list
    rebuilds: list
    new_failed: list
    stop: str
    margin: int
    shown_ok: int
    deferred: int


@dataclass
class _Loop:
    queue: list
    failed: set
    margin: int
    fusions: Fusions = field(default_factory=Fusions)
    rounds: list = field(default_factory=list)
    rebuilds: list = field(default_factory=list)
    new_failed: list = field(default_factory=list)
    attempts: dict = field(default_factory=dict)
    shown: set = field(default_factory=set)
    min_rung: int = 0
    n_cap: int = MAX_MERGE_NOTES
    mt_floor: int = 0
    consecutive: int = 0
    last_kind: Optional[str] = None
    slowest: float = 0.0
    wall: float = 0.0
    shown_ok: int = 0
    stop: Optional[str] = None


class _BudgetRefused(RuntimeError):
    """A request that would not fit the context; raised before it is sent."""


def request_hash(question: str, max_tokens: int) -> str:
    """Identity of one request: the question and its max_tokens."""
    h = hashlib.sha256(question.encode("utf-8"))
    h.update(("|max_tokens=%d" % int(max_tokens)).encode("ascii"))
    return h.hexdigest()[:16]


def _plan(name, ri, k, chars, n, t, tr, mt, parity, question) -> Plan:
    return Plan(rung=name, rung_index=ri, k=k, chars=chars, n=n, prompt_tokens=t,
                retry_tokens=tr, max_tokens=mt, parity=parity, hash=request_hash(question, mt))


def plan_round(count: Callable, fams: list, ex: dict, queue: list, s: MergeSettings,
               margin: int, *, min_rung: int = 0, n_cap: int = MAX_MERGE_NOTES,
               mt_floor: int = 0, parity: bool = False,
               ns_override: Optional[tuple] = None) -> Optional[Plan]:
    """The first rung, in ladder order, whose request fits ctx - margin; None if none does."""
    limit = s.ctx - margin
    for ri in range(max(0, min_rung), len(RUNGS)):
        name, k, chars, ns = RUNGS[ri]
        cands = sorted({min(n, n_cap, len(queue)) for n in (ns_override or ns)}, reverse=True)
        delta = None
        for n in (c for c in cands if c >= 1):
            q = merge_question(fams, ex, queue[:n], k, chars)
            t = count(q)
            if parity and ri == 0 and n_cap >= MAX_MERGE_NOTES and not mt_floor:
                if t + s.start_max_tokens <= s.ctx:
                    tr = count(merge_question(fams, ex, queue[:n], k, chars, retry=True))
                    mt = max(s.start_max_tokens, min(s.max_tokens_cap, limit - tr))
                    return _plan(name, ri, k, chars, n, t, tr, mt, True, q)
                continue
            if delta is None:
                delta = count(merge_question(fams, ex, queue[:n], k, chars, retry=True)) - t
            mt = max(s.start_max_tokens, mt_floor, min(s.max_tokens_cap, limit - t - delta))
            if t + delta + mt > limit:
                continue
            tr = count(merge_question(fams, ex, queue[:n], k, chars, retry=True))
            mt = max(s.start_max_tokens, mt_floor, min(s.max_tokens_cap, limit - tr))
            if tr + mt > limit:
                continue
            return _plan(name, ri, k, chars, n, t, tr, mt, False, q)
    return None


def _n_eff(lp: _Loop, s: MergeSettings) -> int:
    return lp.n_cap if s.drain else min(lp.n_cap, MAX_MERGE_NOTES - lp.shown_ok)


def _stop_reason(lp: _Loop, s: MergeSettings) -> Optional[str]:
    r = len(lp.rounds)
    if r >= s.max_rounds:
        return "max_rounds"
    if r == 0:
        return None
    if not s.drain and MAX_MERGE_NOTES - lp.shown_ok < MIN_NOTES:
        return "done"
    if lp.last_kind is None and len(lp.queue) < s.drain_min_notes:
        return "drain_min_notes"
    if lp.consecutive >= MAX_CONSECUTIVE_FAILURES:
        return "consecutive_failures"
    if lp.wall + max(ROUND_ESTIMATE_MIN_S, 1.5 * lp.slowest) > s.rounds_budget_s:
        return "time_budget"
    return None


def _choose(llm: LLM, fams: list, ex: dict, lp: _Loop, s: MergeSettings) -> tuple:
    """(plan or None or "repeat", rotated, last_resort), never a request that already failed."""
    rotated, plan = 0, None
    for _ in range(PLAN_TRIES):
        plan = plan_round(llm.count, fams, ex, lp.queue, s, lp.margin, min_rung=lp.min_rung,
                          n_cap=_n_eff(lp, s), mt_floor=lp.mt_floor, parity=not lp.rounds)
        if plan is None or plan.hash not in lp.failed:
            break
        if len(lp.queue) > plan.n:
            lp.queue = lp.queue[plan.n:] + lp.queue[:plan.n]
            rotated += 1
        elif plan.n > MIN_NOTES:
            lp.n_cap = max(MIN_NOTES, plan.n // 2)
        else:
            return "repeat", rotated, False
    else:
        return "repeat", rotated, False
    if plan is not None:
        return plan, rotated, False
    by_len = sorted(range(len(lp.queue)),
                    key=lambda i: (llm.client.count_tokens(lp.queue[i].memo or ""), i))
    shortest = [lp.queue[i] for i in by_len]
    alt = plan_round(llm.count, fams, ex, shortest, s, lp.margin, min_rung=len(RUNGS) - 1,
                     n_cap=_n_eff(lp, s), mt_floor=lp.mt_floor, ns_override=LAST_RESORT_NOTES)
    if alt is None or alt.hash in lp.failed:
        return None, rotated, False
    head = {x.sig for x in shortest[:alt.n]}
    lp.queue = shortest[:alt.n] + [x for x in lp.queue if x.sig not in head]
    return alt, rotated, True


def _server_count(text: str) -> Optional[int]:
    m = re.search(r"prompt contains (at least )?(\d+) input tokens", text or "")
    return int(m.group(2)) if m and not m.group(1) else None


def _ask(llm: LLM, fams: list, ex: dict, present: list, plan: Plan, lp: _Loop,
         s: MergeSettings, started: float) -> tuple[MergeReply, list]:
    calls: list = []

    def call(question):
        t = plan.retry_tokens if calls else plan.prompt_tokens
        rec = {"prompt_tokens": t, "max_tokens": plan.max_tokens, "retry_prefix": bool(calls),
               "hash": request_hash(question, plan.max_tokens)}
        calls.append(rec)
        if t + plan.max_tokens > s.ctx - (0 if plan.parity else lp.margin):
            rec["outcome"] = "budget_refused"
            raise _BudgetRefused("prompt %d + max_tokens %d" % (t, plan.max_tokens))
        t0 = time.monotonic()
        try:
            reply, finish = llm.call(question, effort="low", max_tokens=plan.max_tokens)
        except Exception as exc:
            rec.update(outcome="error", error=f"{type(exc).__name__}: {exc}"[:300],
                       server_count=_server_count(str(exc)),
                       seconds=round(time.monotonic() - t0, 1))
            raise
        rec.update(outcome="ok", finish_reason=finish, seconds=round(time.monotonic() - t0, 1))
        return reply, finish

    def can_retry():
        return time.monotonic() - started + s.call_timeout_s <= s.rounds_budget_s

    try:
        rep = merge_call(call, fams, ex, present, (plan.k, plan.chars), can_retry)
    except _BudgetRefused:
        rep = MergeReply(kind="budget_refused")
    except Exception as exc:
        warn("memo_merge_call_failed", error=f"{type(exc).__name__}: {exc}"[:300])
        rep = MergeReply(kind="call_error")
    return rep, calls


def _succeeded(lib: Library, lp: _Loop, rep: MergeReply, fams: list, present: list,
               plan: Plan, step: int, rec: dict) -> None:
    known = {label_key(c[1]) for c in fams} | set(lp.fusions.labels)
    probe = normalise_groups(rep.groups)
    placed = sum(len(g.n) for g in probe)
    attached = sum(len(g.n) for g in probe if g.n and (g.e or label_key(g.label) in known))
    rb = lib.apply_merge(rep.groups, present, step, lp.fusions)
    by_sig = {x.sig: x for x in present}
    unplaced = [by_sig[x] for x in rep.keep_pending if x in by_sig]
    lp.queue = unplaced + lp.queue[plan.n:]
    lp.shown_ok += plan.n
    lp.rebuilds.append(rb)
    lp.consecutive, lp.last_kind = 0, None
    rec.update(outcome="ok", dropped_ids=rep.dropped, placed=placed, unplaced=len(unplaced),
               attached=attached, notes_new=placed - attached, created=rb["solo_new"],
               absorbed=rb["absorbed"], fused=rb["fused"], redirected=rb["redirected"])


def _failed(lp: _Loop, rep: MergeReply, present: list, plan: Plan, calls: list,
            s: MergeSettings, rec: dict) -> None:
    for h in [plan.hash] + [c["hash"] for c in calls]:
        if h not in lp.new_failed:
            lp.new_failed.append(h)
            lp.failed.add(h)
    kind, n = rep.kind, plan.n
    if kind == "finish_length":
        lp.n_cap = max(MIN_NOTES, n // 2)
        lp.mt_floor = min(s.max_tokens_cap, int(math.ceil(1.5 * plan.max_tokens)))
    elif kind == "budget_refused":
        lp.min_rung = plan.rung_index + 1
    elif kind == "call_error":
        drift = max([abs(c["server_count"] - c["prompt_tokens"]) for c in calls
                     if c.get("server_count") is not None] or [0])
        lp.margin += 64 + drift
        lp.min_rung = plan.rung_index + 1
    elif kind in ROTATE_KINDS:
        lp.queue = lp.queue[n:] + present
        if kind == "unknown_ids_over_cap" and lp.last_kind == kind:
            lp.n_cap = max(MIN_NOTES, n // 2)
    for x in present:
        lp.attempts[x.sig] = lp.attempts.get(x.sig, 0) + 1
    if kind not in ROTATE_KINDS:
        lp.queue = ([x for x in lp.queue if lp.attempts.get(x.sig, 0) < 2]
                    + [x for x in lp.queue if lp.attempts.get(x.sig, 0) >= 2])
    lp.consecutive += 1
    lp.last_kind = kind
    rec.update(outcome=kind, dropped_ids=rep.dropped, placed=0, unplaced=n)
    warn("memo_merge_round_failed", round=rec["round"], failure=kind, rung=plan.rung, notes=n)


def _one_round(llm: LLM, lib: Library, lp: _Loop, s: MergeSettings, step: int,
               started: float) -> bool:
    t0 = time.monotonic()
    fams, ex = lib.merge_view()
    base = {"round": len(lp.rounds) + 1, "queue": len(lp.queue), "families": len(fams)}
    try:
        plan, rotated, last_resort = _choose(llm, fams, ex, lp, s)
    except Exception as exc:
        warn("memo_merge_count_failed", step=step, error=f"{type(exc).__name__}: {exc}"[:300])
        lp.stop = lp.last_kind = "count_error"
        lp.rounds.append(dict(base, outcome="count_error"))
        return False
    if plan == "repeat" or plan is None:
        lp.stop = "repeat_refused" if plan == "repeat" else "library_exceeds_context"
        lp.last_kind = lp.stop
        lp.rounds.append(dict(base, outcome=lp.stop, seconds=round(time.monotonic() - t0, 1)))
        warn("memo_merge_" + lp.stop, step=step, families=len(fams), queue=len(lp.queue))
        return False
    present = lp.queue[:plan.n]
    rep, calls = _ask(llm, fams, ex, present, plan, lp, s, started)
    dt = time.monotonic() - t0
    lp.slowest, lp.wall = max(lp.slowest, dt), lp.wall + dt
    rec = dict(base, rung=plan.rung, exemplars=plan.k, exemplar_chars=plan.chars,
               notes_shown=plan.n, max_tokens=plan.max_tokens, prompt_tokens=plan.prompt_tokens,
               retry_tokens=plan.retry_tokens, request_hash=plan.hash, rotated=rotated,
               last_resort=last_resort, calls=calls, seconds=round(dt, 1),
               finish_reason=calls[-1].get("finish_reason") if calls else None)
    lp.shown.update(x.sig for x in present)
    if rep.groups is not None:
        _succeeded(lib, lp, rep, fams, present, plan, step, rec)
    else:
        _failed(lp, rep, present, plan, calls, s, rec)
    lp.rounds.append(rec)
    return True


def merge_rounds(llm: LLM, lib: Library, queue: list, step: int, s: MergeSettings,
                 failed_hashes: list) -> MergeResult:
    """Place queued notes into the library in rounds that fit the context."""
    lp = _Loop(queue=list(queue), failed=set(failed_hashes), margin=s.margin)
    started = time.monotonic()
    while lp.queue:
        lp.stop = _stop_reason(lp, s)
        if lp.stop or not _one_round(llm, lib, lp, s, step, started):
            break
    stop = lp.stop or ("queue_empty" if not lp.queue else "done")
    deferred = sum(1 for x in lp.queue if x.sig not in lp.shown)
    event("memo_merge_rounds", step=step, rounds=len(lp.rounds), ok=len(lp.rebuilds),
          stop=stop, pending=len(lp.queue), families=len(lib.families))
    return MergeResult(queue=lp.queue, rounds=lp.rounds, rebuilds=lp.rebuilds,
                       new_failed=lp.new_failed[-FAILED_HASHES_KEEP:], stop=stop,
                       margin=lp.margin, shown_ok=lp.shown_ok, deferred=deferred)


def roll_failed(old: list, new: list) -> list:
    """Most recent failed request hashes, newest first, at most FAILED_HASHES_KEEP."""
    fresh = set(new)
    return (list(new)[::-1] + [h for h in old if h not in fresh])[:FAILED_HASHES_KEEP]


def status_of(res: MergeResult) -> str:
    """ok / degraded / failed for one boundary's rounds."""
    ok = [r for r in res.rounds if r.get("outcome") == "ok"]
    if not ok:
        return "failed"
    if len(ok) < len(res.rounds) or any(r.get("rung") != "R0" for r in ok):
        return "degraded"
    return "ok"


def update_health(prev: dict, step: int, status: str, rounds: list, families: int,
                  pending: list, last_rung: Optional[str]) -> dict:
    """Merge health after a boundary; frozen = two or more boundaries in a row placed nothing."""
    consec = int(prev.get("consecutive_failed_boundaries") or 0)
    committed = status in ("ok", "degraded")
    if status != "skipped":
        consec = 0 if committed else consec + 1
    steps = [n.step for n in pending if isinstance(n.step, int)]
    oldest = min(steps) if steps else None
    bad = [r.get("outcome") for r in rounds if r.get("outcome") != "ok"]
    return {"step": step, "status": status, "families": families, "pending": len(pending),
            "oldest_pending_step": oldest,
            "oldest_pending_age": step - oldest if oldest is not None else 0,
            "consecutive_failed_boundaries": consec,
            "last_ok_step": step if committed else prev.get("last_ok_step"),
            "last_failure": {"step": step, "kinds": bad} if bad else prev.get("last_failure"),
            "last_rung": last_rung, "degraded": status == "degraded", "frozen": consec >= 2,
            "backlog_stale": bool(oldest is not None and step - oldest > 12)}
