import hashlib
import json
import re
import threading
from pathlib import Path

import pytest

from ttt import loop
from ttt.archive import code_hash
from ttt.config import (Eval, Experiment, MemoConfig, MemoryConfig, Model, Runtime, Sampling,
                        Search)
from ttt.llm import build_format
from ttt.llm.client import ServerError
from ttt.llm.fake import FakeClient, completion_for, is_generation_call
from ttt.memory.docstring import extract_docstring
from ttt.tasks.common import hoist_future
from ttt.types import Assignment, EvalResult, State

MODEL = Model(name="m", format="harmony_two_phase", phase1_max_tokens=26000, date="2026-01-01")


class StubTask:
    name = "stub"
    maximize = True
    entrypoint = "run"

    def initial_state(self, rng):
        return State(id="", timestep=-1, value=0.0, code="```python\nSCORE = 0.0\n```",
                     construction=[0.0, 0.0])

    def refresh_initial(self, state, rng):
        state.construction = [round(float(rng.random()), 6), 0.0]

    def question(self, parent, memory_context):
        return f"Improve {parent.id} (value {parent.value}).{memory_context or ''}"

    def assemble(self, code, parent):
        return f"# parent {parent.id}\n{code}"

    def line_offset(self, code, parent):
        return 1

    def admissible(self, construction):
        return len(construction) <= 4

    def score(self, result):
        raise NotImplementedError


class StubEvaluator:
    capacity = 3

    def __init__(self, fail_all=False):
        self.fail_all = fail_all
        self.calls = []
        self.lock = threading.Lock()

    def evaluate(self, task, program, timeout_s, cpus):
        with self.lock:
            self.calls.append((task, timeout_s, cpus))
        m = re.search(r"SCORE = ([0-9.]+)", program)
        if self.fail_all or m is None:
            return EvalResult(ok=False, raw_score=None, construction=None, stdout="partial",
                              error="boom")
        x = float(m.group(1))
        return EvalResult(ok=True, raw_score=x, construction=[x, round(x * 7 % 1, 6)],
                          stdout=f"score {x}", error=None)


def program_for(prompt):
    if "NOCODE" in prompt:
        return None
    x = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16) / 16 ** 8
    if x < 0.3 and "DIRECTION" in prompt:
        return f'"""Try {x:.3f}."""\nSCORE = broken\n'
    return f'"""Try {x:.3f}."""\nSCORE = {x:.6f}\n'


def responder(prompt, params):
    if not is_generation_call(params):
        return "analysis assistantfinal ok"
    code = program_for(prompt)
    if code is None:
        return "I have no program."
    if "GENFAIL" in prompt:
        raise ServerError("server went away")
    return completion_for("harmony_two_phase", code)


class StubMemory:
    def __init__(self, events, fail_at=None):
        self.events, self.fail_at = events, fail_at
        self.registered, self.children, self.steps, self.outcomes = {}, 0, [], []

    def register(self, state, raw_score):
        self.registered[state.id] = raw_score
        self.events.append(("register", state.id))

    def context_for(self, parent):
        self.events.append(("context_for", parent.id))
        return f" [ctx {parent.id}]"

    def record_child(self, parent, outcome):
        self.children += 1
        if self.fail_at is not None and self.children == self.fail_at:
            raise RuntimeError("crash in the middle of a step")
        if outcome.admitted:
            self.registered[outcome.child.id] = outcome.raw_score
        self.outcomes.append(outcome)
        self.events.append(("record_child", outcome.ok, outcome.admitted))

    def end_of_step(self, step, best_raw):
        self.events.append(("memory_end", step, best_raw))
        self.steps.append(step)

    def state_dict(self):
        return {"registered": self.registered, "steps": self.steps}

    def load_state_dict(self, state):
        self.registered, self.steps = dict(state["registered"]), list(state["steps"])


class StubMemo:
    def __init__(self, events, out_dir, fail_boundary_at=None):
        self.events, self.out_dir, self.fail_boundary_at = events, Path(out_dir), fail_boundary_at
        self.boundaries, self.records = [], []

    def plan(self, step, parents, memory):
        self.events.append(("plan", step, len(parents), memory is not None))
        return [[Assignment(arm="selected" if j % 2 else "neutral", text=f" DIRECTION-{g}-{j}",
                            tags={"g": g}) for j in range(4)] for g in range(len(parents))]

    def record(self, step, group, parent, assignment, outcome):
        self.records.append((step, group, assignment.arm, outcome.ok))
        self.events.append(("memo_record", step, group))

    def end_of_step(self, step):
        self.events.append(("memo_end", step))

    def boundary(self, step, archive_ids):
        committed = (self.out_dir / "checkpoints" / f"step_{step:06d}" / "COMMITTED").exists()
        self.events.append(("boundary", step, committed))
        if step == self.fail_boundary_at:
            self.fail_boundary_at = None
            raise RuntimeError("boundary crashed")
        self.boundaries.append(step)

    def state_dict(self):
        return {"boundaries": self.boundaries}

    def load_state_dict(self, state):
        self.boundaries = list(state["boundaries"])


def experiment(steps=2, memory=False, memo=False, children=4):
    return Experiment(name="t", task="ac2", model=MODEL, sampling=Sampling(),
                      search=Search(steps=steps, parents_per_step=2, children_per_parent=children,
                                    seed=3),
                      eval=Eval(timeout_s=5.0, cpus=1),
                      memory=MemoryConfig() if memory or memo else None,
                      memo=MemoConfig(selected_children=children) if memo else None)


def go(exp, out, *, resume=False, memory=None, memo=None, evaluator=None, clients=None,
       gen_concurrency=4, task=None):
    runtime = Runtime(out_dir=str(out), vllm_urls=("http://a", "http://b"), served_model="m",
                      api_key="never-written", gen_concurrency=gen_concurrency, resume=resume)
    clients = clients or [FakeClient(responder), FakeClient(responder)]
    return loop.run(exp, runtime, task=task or StubTask(), evaluator=evaluator or StubEvaluator(),
                    clients=clients, fmt=build_format(exp.model, exp.sampling),
                    memory=memory, memo=memo)


def rows(out, name="rollouts.jsonl"):
    return [json.loads(line) for line in (Path(out) / name).read_text().splitlines()]


def layout(out):
    return sorted(str(p.relative_to(out)) for p in Path(out).rglob("*")
                  if not {"constructions", "codes"} & set(p.relative_to(out).parts))


def code_file(out, sha):
    return Path(out) / "codes" / sha[:2] / f"{sha}.py"


def test_two_step_run_layout(tmp_path):
    out = tmp_path / "run"
    result = go(experiment(), out)
    assert [s["step"] for s in result["steps"]] == [0, 1]
    assert layout(out) == [
        "checkpoints", "checkpoints/step_000000", "checkpoints/step_000000/COMMITTED",
        "checkpoints/step_000000/archive.json", "checkpoints/step_000001",
        "checkpoints/step_000001/COMMITTED", "checkpoints/step_000001/archive.json",
        "manifest.json", "rollouts.jsonl", "steps.jsonl"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert "never-written" not in (out / "manifest.json").read_text()
    assert manifest["runtime"]["vllm_urls"] == ["http://a", "http://b"]
    r = rows(out)
    assert [(x["step"], x["rollout"]) for x in r] == [(s, i) for s in (0, 1) for i in range(8)]
    steps = rows(out, "steps.jsonl")
    assert all(s["valid"] + s["failed"] == 8 and s["infra"] == 0 for s in steps)
    assert result["best"] == steps[-1]["best"] == max(x["child_score"] for x in r if x["admitted"])
    assert result["best_valid"] == max(x["child_score"] for x in r if x["ok"])
    archive = json.loads((out / "checkpoints/step_000001/archive.json").read_text())
    for s in archive["states"]:
        assert (out / "constructions" / s["construction_hash"][:2]
                / f"{s['construction_hash']}.npy").exists()
    assert {x["arm"] for x in r} == {"none"}


def test_rollouts_stay_on_one_server(tmp_path):
    a, b = FakeClient(responder), FakeClient(responder)
    go(experiment(steps=1), tmp_path / "run", clients=[a, b])
    assert len(a.calls) == len(b.calls) == 4


def test_hook_order_and_assignments(tmp_path):
    out = tmp_path / "run"
    events = []
    memory, memo = StubMemory(events), StubMemo(events, out)
    clients = [FakeClient(responder)]
    go(experiment(memo=True), out, memory=memory, memo=memo, clients=clients)
    kinds = [e[0] for e in events]
    step0 = kinds[:kinds.index("boundary") + 1]
    assert events[:2] == [("register", "init-0"), ("register", "init-1")]
    assert [k for k in step0 if k != "register"] == (
        ["plan", "context_for", "context_for"] + ["record_child", "memo_record"] * 8
        + ["memory_end", "memo_end", "boundary"])
    assert step0[2:5] == ["register", "register", "plan"], "parents are registered before planning"
    assert step0.count("register") == 4, "admitted children are registered by record_child alone"
    assert [e for e in events if e[0] == "boundary"] == [("boundary", 0, True)]
    prompts = [p for p, params in clients[0].calls if is_generation_call(params)]
    assert sum("DIRECTION-1-3" in p for p in prompts) == 2
    assert all("[ctx " in p for p in prompts)
    r = rows(out)
    assert [x["arm"] for x in r[:4]] == ["neutral", "selected", "neutral", "selected"]
    assert r[5]["tags"] == {"g": 1}
    assert set(memory.registered) >= {"init-0", "init-1"}
    admitted = [x["child_id"] for x in r if x["admitted"]]
    assert admitted and set(admitted) <= set(memory.registered)
    assert sorted((out / "checkpoints/step_000001").iterdir()) == sorted(
        out / "checkpoints/step_000001" / n for n in ("COMMITTED", "archive.json", "memo.json", "memory.json"))


def test_row_fields(tmp_path):
    out = tmp_path / "run"
    go(experiment(steps=1), out)
    row = next(x for x in rows(out) if x["ok"])
    assert set(row) == {"step", "rollout", "parent_id", "parent_score", "child_id", "child_score",
                        "ok", "admitted", "beats_parent", "error", "doc", "code_sha256",
                        "stdout_tail", "arm", "tags", "infra_error"}
    assert row["doc"].startswith("Try ") and row["error"] is None and row["infra_error"] is False
    assert row["beats_parent"] == (row["child_score"] > row["parent_score"])
    assert row["child_id"] == f"s0-r{row['rollout']}"


def test_failures_are_recorded(tmp_path):
    out = tmp_path / "run"

    class Fails(StubTask):
        def question(self, parent, memory_context):
            return "NOCODE" if parent.id == "init-0" else "GENFAIL"

    runtime = Runtime(out_dir=str(out), vllm_urls=("u",), served_model="m")
    with pytest.raises(RuntimeError, match="consecutive steps"):
        loop.run(experiment(steps=4), runtime, task=Fails(), evaluator=StubEvaluator(),
                 clients=[FakeClient(responder)], fmt=build_format(MODEL, Sampling()))
    errors = {(x["error"].split(":")[0], x["infra_error"]) for x in rows(out)}
    assert errors == {("no_code", False), ("gen_error", True)}
    assert [(s["failed"], s["infra"]) for s in rows(out, "steps.jsonl")] == [(4, 4), (4, 4)]
    archive = json.loads((out / "checkpoints/step_000000/archive.json").read_text())
    assert archive["n"] == {"init-0": 4}, "server failures are not booked against the parent"


def test_prompt_generation_failures_are_candidate_failures(tmp_path):
    class Oversized(StubTask):
        def question(self, parent, memory_context):
            text = super().question(parent, memory_context)
            return text + "X" * 120000 if parent.id == "init-1" else text

    out = tmp_path / "run"
    memory, memo = StubMemory([]), StubMemo([], out)
    go(experiment(steps=1, memo=True), out, task=Oversized(), memory=memory, memo=memo)
    assert all(x["error"].startswith("gen_error: prompt length") for x in rows(out)[4:])
    assert not any(x["infra_error"] for x in rows(out)) and rows(out, "steps.jsonl")[0]["infra"] == 0
    booked = [o for o in memory.outcomes if (o.error or "").startswith("gen_error")]
    assert len(booked) == 4 and not any(o.ok or o.code for o in booked)
    assert sum(g == 1 for _, g, _, _ in memo.records) == 4
    archive = json.loads((out / "checkpoints/step_000000/archive.json").read_text())
    assert archive["n"]["init-1"] == 4


def test_dead_steps_abort(tmp_path):
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="boom"):
        go(experiment(steps=5), out, evaluator=StubEvaluator(fail_all=True))
    assert [s["valid"] for s in rows(out, "steps.jsonl")] == [0, 0]
    assert all(x["code_sha256"] and x["stdout_tail"] == "partial" for x in rows(out))


def test_fresh_run_refuses_existing_state(tmp_path):
    out = tmp_path / "run"
    go(experiment(steps=1), out)
    with pytest.raises(FileExistsError):
        go(experiment(steps=1), out)


def test_resume_rejects_other_config(tmp_path):
    out = tmp_path / "run"
    go(experiment(steps=1), out)
    with pytest.raises(ValueError, match="config hash"):
        go(experiment(steps=3), out, resume=True)


def test_plan_shape_is_checked(tmp_path):
    out = tmp_path / "run"
    events = []

    class Short(StubMemo):
        def plan(self, step, parents, memory):
            return [[Assignment("a", "x")] for _ in parents]

    with pytest.raises(ValueError, match="memo plan"):
        go(experiment(memo=True), out, memory=StubMemory(events), memo=Short(events, out))


def _comparable(out, last=2):
    steps = [{k: v for k, v in s.items() if k != "seconds"} for s in rows(out, "steps.jsonl")]
    archive = (Path(out) / f"checkpoints/step_{last:06d}/archive.json").read_text()
    return layout(out), rows(out), steps, archive


def test_resume_after_boundary_crash_reproduces_run(tmp_path):
    ref = tmp_path / "ref"
    events = []
    go(experiment(steps=3, memo=True), ref, memory=StubMemory(events), memo=StubMemo(events, ref))
    out = tmp_path / "run"
    events = []
    with pytest.raises(RuntimeError, match="boundary crashed"):
        go(experiment(steps=3, memo=True), out, memory=StubMemory(events),
           memo=StubMemo(events, out, fail_boundary_at=0))
    assert len(rows(out, "steps.jsonl")) == 1
    memo = StubMemo(events, out)
    go(experiment(steps=3, memo=True), out, resume=True, memory=StubMemory(events), memo=memo)
    assert memo.boundaries == [0, 1]
    assert _comparable(out) == _comparable(ref)


def test_resume_mid_step_truncates_partial_rows(tmp_path):
    ref = tmp_path / "ref"
    go(experiment(steps=3, memory=True), ref, memory=StubMemory([]))
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="middle of a step"):
        go(experiment(steps=3, memory=True), out, memory=StubMemory([], fail_at=11))
    assert len(rows(out)) == 10 and len(rows(out, "steps.jsonl")) == 1
    with open(out / "rollouts.jsonl", "a") as f:
        f.write('{"step": 1, "rol')
    memory = StubMemory([])
    go(experiment(steps=3, memory=True), out, resume=True, memory=memory)
    assert memory.steps == [0, 1, 2]
    assert _comparable(out) == _comparable(ref)


def test_resume_drops_uncommitted_checkpoint(tmp_path):
    out = tmp_path / "run"
    go(experiment(steps=2), out)
    (out / "checkpoints/step_000001/COMMITTED").unlink()
    result = go(experiment(steps=2), out, resume=True)
    assert [s["step"] for s in result["steps"]] == [0, 1]
    assert len(rows(out)) == 16 and (out / "checkpoints/step_000001/COMMITTED").exists()
    for marker in (out / "checkpoints").glob("*/COMMITTED"):
        marker.unlink()
    with pytest.raises(FileNotFoundError, match="no committed checkpoint"):
        go(experiment(steps=2), out, resume=True)


class CrashingTask(StubTask):
    def assemble(self, code, parent):
        if parent.id == "init-1":
            raise ValueError("harness could not be built")
        return super().assemble(code, parent)


class InfraEvaluator(StubEvaluator):
    def __init__(self, healthy=None, crash=False):
        super().__init__()
        self.healthy, self.crash, self.seen = healthy, crash, 0

    def evaluate(self, task, program, timeout_s, cpus):
        with self.lock:
            down = self.seen >= self.healthy if self.healthy else "# parent init-1" in program
            self.seen += 1
        if down and self.crash:
            raise RuntimeError("worker exploded")
        if down:
            return EvalResult(ok=False, raw_score=None, construction=None, stdout="",
                              error="worker 503", infra_error=True)
        return super().evaluate(task, program, timeout_s, cpus)


@pytest.mark.parametrize("task, evaluator", [(CrashingTask(), StubEvaluator()),
                                             (StubTask(), InfraEvaluator(crash=True))])
def test_assemble_and_evaluator_crashes_are_contained(tmp_path, capsys, task, evaluator):
    out = tmp_path / "run"
    go(experiment(steps=1), out, task=task, evaluator=evaluator)
    r = rows(out)
    crashed = [x for x in r if x["infra_error"]]
    assert [x["rollout"] for x in crashed] == [4, 5, 6, 7]
    assert all(x["error"].startswith("eval_crash: ") and x["code_sha256"] for x in crashed)
    (step,) = rows(out, "steps.jsonl")
    assert (step["valid"] + step["failed"], step["infra"]) == (4, 4)
    assert "evaluation_crashed" in capsys.readouterr().err


@pytest.mark.parametrize("code, doc", [("x = 1\x00\n", None), ("x = " + "-" * 200000 + "1\n", None),
                                       ("x = " + "+".join(["1"] * 100000) + "\n", "Doc.")])
def test_hostile_program_text_is_contained(code, doc):
    assert extract_docstring('"""Doc."""\n' + code, "run") == doc
    future = "from __future__ import annotations\n" + code
    assert hoist_future(future) == ("", future)


def test_infra_failures_are_not_booked(tmp_path):
    out = tmp_path / "run"
    events = []
    memory, memo = StubMemory(events), StubMemo(events, out)
    go(experiment(steps=1, memo=True), out, memory=memory, memo=memo, evaluator=InfraEvaluator())
    r = rows(out)
    assert [x["infra_error"] for x in r] == [False] * 4 + [True] * 4
    assert all(x["error"] == "worker 503" and not x["ok"] for x in r[4:])
    (step,) = rows(out, "steps.jsonl")
    assert (step["valid"] + step["failed"], step["infra"]) == (4, 4)
    assert memory.children == 4 and {g for _, g, _, _ in memo.records} == {0}
    archive = json.loads((out / "checkpoints/step_000000/archive.json").read_text())
    assert archive["T"] == 4 and archive["n"] == {"init-0": 4}


@pytest.mark.parametrize("steps, healthy, infra", [(2, 8, 8), (1, 1, 7)])
def test_infra_majority_aborts_and_resume_reruns_the_step(tmp_path, steps, healthy, infra):
    ref, out, last = tmp_path / "ref", tmp_path / "run", steps - 1
    go(experiment(steps=steps), ref)
    with pytest.raises(RuntimeError, match=f"{infra} of 8 rollouts hit infrastructure failures"):
        go(experiment(steps=steps), out, evaluator=InfraEvaluator(healthy=healthy))
    assert len(loop.records.read_jsonl(out / "steps.jsonl")) == last and len(rows(out)) == 8 * steps
    assert not (out / f"checkpoints/step_{last:06d}").exists()
    go(experiment(steps=steps), out, resume=True)
    assert _comparable(out, last=last) == _comparable(ref, last=last)


def test_reported_best_is_the_archive_best(tmp_path):
    class RefuseAll(StubTask):
        def admissible(self, construction):
            return False

    out = tmp_path / "run"
    result = go(experiment(steps=2), out, task=RefuseAll())
    r = rows(out)
    best_valid = max(x["child_score"] for x in r if x["ok"])
    assert not any(x["admitted"] for x in r) and best_valid > 0.0
    assert [(s["best"], s["best_valid"]) for s in rows(out, "steps.jsonl")] == [
        (0.0, max(x["child_score"] for x in r if x["ok"] and x["step"] == 0)), (0.0, best_valid)]
    assert (result["best"], result["best_valid"]) == (0.0, best_valid)


def test_programs_are_saved_and_checkpoints_reference_them(tmp_path, monkeypatch):
    out = tmp_path / "run"
    real_write = loop.records.write_checkpoint

    def write_checked(directory, parts):
        for s in parts["archive"]["states"]:
            assert code_file(out, s["code_sha256"]).exists(), "store written before checkpoint"
        real_write(directory, parts)

    monkeypatch.setattr(loop.records, "write_checkpoint", write_checked)
    go(experiment(steps=2), out)
    for x in rows(out):
        assert code_hash(code_file(out, x["code_sha256"]).read_text()) == x["code_sha256"]
    text = (out / "checkpoints/step_000001/archive.json").read_text()
    assert "SCORE =" not in text and '"code"' not in text
    code_file(out, json.loads(text)["states"][-1]["code_sha256"]).write_text("tampered")
    with pytest.raises(ValueError, match="corrupt"):
        go(experiment(steps=2), out, resume=True)


def test_no_boundary_after_the_final_step(tmp_path):
    out = tmp_path / "run"
    events = []

    def boundaries(**kw):
        events.clear()
        go(experiment(steps=2, memo=True), out, memory=StubMemory(events),
           memo=StubMemo(events, out), **kw)
        return [e[1] for e in events if e[0] == "boundary"]

    assert boundaries() == [0]
    (out / "checkpoints/step_000001/COMMITTED").unlink()
    assert boundaries(resume=True) == [0], "the boundary after step 0 reruns; none after step 1"
    assert boundaries(resume=True) == [], "a finished run has no boundary left to run"
