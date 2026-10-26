"""The search loop: sample parents, generate, evaluate, admit, update memory, commit."""
from __future__ import annotations

import dataclasses
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from ttt import log, records
from ttt.archive import Archive, CodeStore, ConstructionStore
from ttt.config import Experiment, Runtime
from ttt.llm.client import ServerError
from ttt.memory.docstring import extract_docstring
from ttt.types import (Assignment, ChildOutcome, Client, EvalResult, Evaluator, Format, Memo,
                       Memory, State, Task, raw_score, signed_value)

NO_ASSIGNMENT = Assignment(arm="none", text="")
MAX_DEAD_STEPS = 2
MAX_INFRA_SHARE = 0.5
OBSERVATION_CHARS = 2000
ERROR_CHARS = 200


@dataclass
class Rollout:
    """One child to generate: its index in the step, its parent group and its prompt."""

    index: int
    group: int
    prompt: str
    assignment: Assignment


@dataclass
class RolloutResult:
    """What generation and evaluation produced for one rollout; `infra`: not the candidate's fault."""

    index: int
    code: Optional[str]
    result: Optional[EvalResult]
    error: Optional[str]
    infra: bool = False
    code_offset: int = 0


@dataclass
class Tally:
    """Counters filled while a step's results are drained."""

    best_valid: Optional[float]
    valid: int = 0
    failed: int = 0
    infra: int = 0
    first_error: str = ""
    first_infra_error: str = ""


@dataclass
class Run:
    """Everything a step needs; fixed for the whole run."""

    exp: Experiment
    runtime: Runtime
    task: Task
    evaluator: Evaluator
    clients: Sequence[Client]
    fmt: Format
    memory: Optional[Memory]
    memo: Optional[Memo]
    archive: Archive
    paths: records.RunPaths


def better(a: float, b: Optional[float], maximize: bool) -> bool:
    """True if raw score a beats b in the task's direction (anything beats None)."""
    if b is None:
        return True
    return a > b if maximize else a < b


def run(exp: Experiment, runtime: Runtime, *, task: Task, evaluator: Evaluator,
        clients: Sequence[Client], fmt: Format, memory: Optional[Memory] = None,
        memo: Optional[Memo] = None) -> dict:
    """Run or resume the search; returns the archive's best, the best valid child and the step rows."""
    if not clients:
        raise ValueError("run needs at least one generation client")
    if (memory is None) != (exp.memory is None) or (memo is None) != (exp.memo is None):
        raise ValueError("memory/memo objects must match exp.memory/exp.memo")
    paths = records.RunPaths.of(runtime.out_dir)
    if runtime.resume:
        archive, steps = resume_state(exp, task, paths, memory, memo)
    else:
        archive, steps = fresh_state(exp, runtime, task, paths)
    ctx = Run(exp, runtime, task, evaluator, list(clients), fmt, memory, memo, archive, paths)
    if memory is not None:
        for s in archive.states:
            memory.register(s, raw_score(s.value, task.maximize))
    if runtime.resume and memo is not None and archive.step >= 0 and boundary_due(exp, archive.step):
        # The checkpoint precedes the boundary that followed it, so that boundary runs again.
        memo.boundary(archive.step, archive_ids(archive))
    best_valid = steps[-1]["best_valid"] if steps else None
    dead = 0
    for step in range(archive.step + 1, exp.search.steps):
        row, first_error = run_step(ctx, step, best_valid)
        best_valid = row.best_valid
        steps.append(dataclasses.asdict(row))
        dead = dead + 1 if row.valid == 0 else 0
        if dead >= MAX_DEAD_STEPS:
            raise RuntimeError(f"aborting at step {step}: {dead} consecutive steps without a valid "
                               f"rollout; generation or evaluation is down. Last error: {first_error}")
    return {"best": archive_best(archive, task.maximize), "best_valid": best_valid, "steps": steps}


def boundary_due(exp: Experiment, step: int) -> bool:
    """A memo boundary after step is only worth running if a later step can use it."""
    return step + 1 < exp.search.steps


def archive_ids(archive: Archive) -> frozenset[str]:
    """Ids of the states the archive holds after its last flush."""
    return frozenset(s.id for s in archive.states)


def archive_best(archive: Archive, maximize: bool) -> Optional[float]:
    """Best raw score among the archived states, initial states included."""
    values = [s.value for s in archive.states if s.value is not None]
    return raw_score(max(values), maximize) if values else None


def stores(paths: records.RunPaths) -> tuple[ConstructionStore, CodeStore]:
    """The run's construction and program stores."""
    return ConstructionStore(paths.constructions), CodeStore(paths.codes)


def fresh_state(exp: Experiment, runtime: Runtime, task: Task,
                paths: records.RunPaths) -> tuple[Archive, list[dict]]:
    """Refuse existing run state, write the manifest and seed a new archive."""
    if records.has_run_state(paths.root):
        raise FileExistsError(f"{paths.root} already holds a run; resume it or use a new directory")
    paths.root.mkdir(parents=True, exist_ok=True)
    records.write_manifest(paths, exp, runtime)
    return Archive(task, exp.search, *stores(paths)), []


def resume_state(exp: Experiment, task: Task, paths: records.RunPaths, memory: Optional[Memory],
                 memo: Optional[Memo]) -> tuple[Archive, list[dict]]:
    """Load the latest committed checkpoint and drop everything recorded after it."""
    records.check_manifest(paths, exp)
    step = records.latest_committed(paths)
    if step is None:
        return restart_state(exp, task, paths)
    removed = records.discard_after(paths, step)
    steps = records.check_logs(paths, step)
    parts = records.read_checkpoint(paths.checkpoint(step))
    construction_store, code_store = stores(paths)
    archive = Archive.from_state_dict(parts["archive"], task=task, search=exp.search,
                                      store=construction_store, codes=code_store)
    if archive.step != step:
        raise ValueError(f"checkpoint {step} holds an archive flushed at step {archive.step}")
    if memory is not None:
        memory.load_state_dict(parts["memory"])
    if memo is not None:
        memo.load_state_dict(parts["memo"])
    log.event("resume", step=step, **{f"dropped_{k}": v for k, v in removed.items()})
    return archive, steps


def restart_state(exp: Experiment, task: Task, paths: records.RunPaths) -> tuple[Archive, list[dict]]:
    """Start a resumed run over when its first step never committed."""
    if any(int(r["step"]) > 0 for r in records.read_jsonl(paths.steps)):
        raise FileNotFoundError(f"{paths.root} has step records but no committed checkpoint")
    removed = records.discard_after(paths, -1)
    log.event("resume", step=None, **{f"dropped_{k}": v for k, v in removed.items()})
    return Archive(task, exp.search, *stores(paths)), []


def run_step(ctx: Run, step: int, best_valid: Optional[float]) -> tuple[records.StepRow, str]:
    """One full step; returns its step row and the first rollout error seen."""
    t0 = time.time()
    parents = sample_parents(ctx)
    plan = ctx.memo.plan(step, parents, ctx.memory) if ctx.memo is not None else None
    rollouts = build_rollouts(ctx, parents, plan)
    tally = Tally(best_valid=best_valid)
    with open(ctx.paths.rollouts, "a", encoding="utf-8") as rows:
        def absorb(rollout: Rollout, result: RolloutResult) -> None:
            row = absorb_rollout(ctx, step, parents[rollout.group], rollout, result, tally)
            rows.write(records.json_line(row))
        generate_and_evaluate(ctx, parents, rollouts, absorb)
        rows.flush()
        os.fsync(rows.fileno())
    check_infra(step, tally, len(rollouts))
    ctx.archive.flush(step)
    if ctx.memory is not None:
        ctx.memory.end_of_step(step, tally.best_valid)
    if ctx.memo is not None:
        ctx.memo.end_of_step(step)
    row = records.StepRow(step=step, best=archive_best(ctx.archive, ctx.task.maximize),
                          best_valid=tally.best_valid, valid=tally.valid, failed=tally.failed,
                          infra=tally.infra, seconds=round(time.time() - t0, 1))
    commit(ctx, row)
    log.event("step", **dataclasses.asdict(row))
    if ctx.memo is not None and boundary_due(ctx.exp, step):
        ctx.memo.boundary(step, archive_ids(ctx.archive))
    return row, tally.first_error or tally.first_infra_error


def check_infra(step: int, tally: Tally, total: int) -> None:
    """Raise before commit when infrastructure failures dominate a step, so a resume re-runs it."""
    if not tally.infra:
        return
    log.warn("step_infra_failures", step=step, infra=tally.infra, rollouts=total,
             first_error=tally.first_infra_error)
    if tally.infra > MAX_INFRA_SHARE * total:
        raise RuntimeError(f"aborting step {step} before commit: {tally.infra} of {total} rollouts "
                           f"hit infrastructure failures; resume to re-run it. "
                           f"First error: {tally.first_infra_error}")


def sample_parents(ctx: Run) -> list[State]:
    """Select this step's parents and register them (initial states may have been refreshed)."""
    parents = ctx.archive.sample(ctx.exp.search.parents_per_step)
    if ctx.memory is not None:
        for p in parents:
            ctx.memory.register(p, raw_score(p.value, ctx.task.maximize))
    return parents


def build_rollouts(ctx: Run, parents: list[State],
                   plan: Optional[list[list[Assignment]]]) -> list[Rollout]:
    """Render one prompt per child: the parent's question plus its assignment's direction."""
    k = ctx.exp.search.children_per_parent
    if plan is not None and (len(plan) != len(parents) or any(len(g) != k for g in plan)):
        raise ValueError(f"memo plan must hold {k} assignments for each of {len(parents)} parents")
    rollouts: list[Rollout] = []
    for g, parent in enumerate(parents):
        context = ctx.memory.context_for(parent) if ctx.memory is not None else None
        question = ctx.task.question(parent, context)
        shared = ctx.fmt.render(question) if plan is None else None
        for j in range(k):
            a = plan[g][j] if plan is not None else NO_ASSIGNMENT
            prompt = shared if shared is not None else ctx.fmt.render(question + a.text)
            rollouts.append(Rollout(index=len(rollouts), group=g, prompt=prompt, assignment=a))
    return rollouts


def generate_and_evaluate(ctx: Run, parents: list[State], rollouts: list[Rollout],
                          absorb: Callable[[Rollout, RolloutResult], None]) -> None:
    """Generate every rollout, evaluate each as soon as its text is ready, absorb in rollout order."""
    # Read every step: a rescanned pool grows as late workers join.
    eval_workers = ctx.runtime.eval_concurrency or max(1, ctx.evaluator.capacity)
    with ThreadPoolExecutor(ctx.runtime.gen_concurrency) as gen_pool, \
            ThreadPoolExecutor(eval_workers) as eval_pool:
        generated = [gen_pool.submit(generate_one, ctx, r) for r in rollouts]
        pending = {}
        for fut in as_completed(generated):
            r, gen = fut.result()
            pending[r.index] = eval_pool.submit(evaluate_one, ctx, parents[r.group], r, gen)
        generated.clear()
        # Admission and tie-breaking in the archive depend on order, so drain by rollout index.
        for r in rollouts:
            absorb(r, pending.pop(r.index).result())


def generate_one(ctx: Run, r: Rollout) -> tuple[Rollout, str | RolloutResult]:
    """Complete one rollout's prompt; a failure becomes a failed result."""
    # A two-phase completion resends prompt + phase-1 text, so it must stay on the server
    # holding that prefix in its cache.
    client = ctx.clients[r.index % len(ctx.clients)]
    try:
        return r, ctx.fmt.complete(r.prompt, client)
    except Exception as e:
        # Only an unreachable or failing server is infrastructure; anything else (a prompt with
        # no token budget left, say) is the candidate's failure, or one parent could stall resumes.
        infra = isinstance(e, ServerError)
        log.warn("generation_failed", rollout=r.index, infra=infra, error=str(e)[:ERROR_CHARS])
        return r, RolloutResult(r.index, None, None, f"gen_error: {str(e)[:ERROR_CHARS]}", infra=infra)


def evaluate_one(ctx: Run, parent: State, r: Rollout, gen: str | RolloutResult) -> RolloutResult:
    """Extract, assemble and evaluate one completion; a crash becomes an infrastructure failure."""
    if isinstance(gen, RolloutResult):
        return gen
    text, code = gen, None
    try:
        code = ctx.fmt.extract_code(text)
        if code is None:
            return RolloutResult(r.index, None, None, "no_code")
        program = ctx.task.assemble(code, parent)
        result = ctx.evaluator.evaluate(ctx.task.name, program, ctx.exp.eval.timeout_s,
                                        ctx.exp.eval.cpus)
        scored = result.ok and result.raw_score is not None
        offset = 0 if scored else ctx.task.line_offset(code, parent)
    except Exception as e:
        log.warn("evaluation_crashed", rollout=r.index, error=repr(e)[:ERROR_CHARS])
        return RolloutResult(r.index, code, None, f"eval_crash: {e!r}"[:ERROR_CHARS], infra=True)
    return RolloutResult(r.index, code, result, None if scored else (result.error or "failed"),
                         infra=bool(result.infra_error), code_offset=offset)


def absorb_rollout(ctx: Run, step: int, parent: State, r: Rollout, res: RolloutResult,
                   tally: Tally) -> dict:
    """Save the rollout's program, book it as a candidate or infrastructure failure, return its row."""
    if ctx.runtime.save_code and res.code is not None:
        ctx.archive.codes.put(res.code)
    if res.infra:
        return absorb_infra(ctx, step, parent, r, res, tally)
    return absorb_candidate(ctx, step, parent, r, res, tally)


def absorb_infra(ctx: Run, step: int, parent: State, r: Rollout, res: RolloutResult,
                 tally: Tally) -> dict:
    """Count a failure the candidate did not cause; the archive, memory and memo never see it."""
    tally.infra += 1
    error = res.error or "infra_error"
    if not tally.first_infra_error:
        tally.first_infra_error = error
    doc = extract_docstring(res.code, ctx.task.entrypoint) if res.code else None
    return log_row(ctx, step, parent, r, res, child=None, admitted=False, beats_parent=False,
                   error=error, doc=doc)


def absorb_candidate(ctx: Run, step: int, parent: State, r: Rollout, res: RolloutResult,
                     tally: Tally) -> dict:
    """Admit a scored child or book a candidate failure, then update memory and memo."""
    maximize = ctx.task.maximize
    ev = res.result
    child, admitted, raw = None, False, None
    if ev is not None and ev.ok and ev.raw_score is not None:
        raw = float(ev.raw_score)
        tally.valid += 1
        child = State(id=f"s{step}-r{r.index}", timestep=step, value=signed_value(raw, maximize),
                      code=f"```python\n{res.code}\n```", construction=ev.construction,
                      observation=(ev.stdout or "")[-OBSERVATION_CHARS:])
        admitted = ctx.archive.update(child, parent)
        if better(raw, tally.best_valid, maximize):
            tally.best_valid = raw
    else:
        tally.failed += 1
        ctx.archive.record_failure(parent)
    ok = child is not None
    error = None if ok else (res.error or "failed")
    if error and not tally.first_error:
        tally.first_error = error
    outcome = ChildOutcome(
        ok=ok, raw_score=raw, admitted=admitted,
        beats_parent=ok and better(raw, raw_score(parent.value, maximize), maximize),
        docstring=extract_docstring(res.code, ctx.task.entrypoint) if res.code else None,
        code=res.code, code_offset=res.code_offset, error=error,
        stdout=(ev.stdout if ev is not None else "") or "", child=child)
    if ctx.memory is not None:
        ctx.memory.record_child(parent, outcome)
    if ctx.memo is not None:
        ctx.memo.record(step, r.group, parent, r.assignment, outcome)
    return log_row(ctx, step, parent, r, res, child=child, admitted=admitted,
                   beats_parent=outcome.beats_parent, error=error, doc=outcome.docstring)


def log_row(ctx: Run, step: int, parent: State, r: Rollout, res: RolloutResult, *,
            child: Optional[State], admitted: bool, beats_parent: bool, error: Optional[str],
            doc: Optional[str]) -> dict:
    """The rollouts.jsonl row of one rollout."""
    ev = res.result
    return records.rollout_row(
        step=step, rollout=r.index, parent_id=parent.id,
        parent_score=raw_score(parent.value, ctx.task.maximize),
        child_id=child.id if child is not None else None,
        child_score=ev.raw_score if ev is not None else None, ok=child is not None,
        admitted=admitted, beats_parent=beats_parent, error=error, doc=doc, code=res.code,
        stdout=(ev.stdout if ev is not None else "") or "", arm=r.assignment.arm,
        tags=r.assignment.tags, infra_error=res.infra)


def commit(ctx: Run, row: records.StepRow) -> None:
    """Append the step row, then write the checkpoint and its COMMITTED marker."""
    records.append_jsonl(ctx.paths.steps, [dataclasses.asdict(row)])
    parts = {"archive": ctx.archive.state_dict()}
    if ctx.memory is not None:
        parts["memory"] = ctx.memory.state_dict()
    if ctx.memo is not None:
        parts["memo"] = ctx.memo.state_dict()
    records.write_checkpoint(ctx.paths.checkpoint(row.step), parts)
