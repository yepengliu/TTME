import itertools
import json
import shutil

import pytest

from ttt import loop
from ttt.config import Eval, Experiment, MemoConfig, MemoryConfig, Model, Runtime, Sampling, Search
from ttt.llm import build_format
from ttt.llm.fake import FakeClient, aux_reply_for, completion_for, is_generation_call
from ttt.memo import build_memo
from ttt.memory import build_memory
from ttt import tasks
from ttt.eval.local import LocalEvaluator

from test_memo_fakes import DEFAULTS, kind_of

MODEL = Model(name="m", format="harmony_two_phase", phase1_max_tokens=26000, date="2026-07-24")
GUIDELINES = ("<guidelines>\nGuideline 1: a run nears the time budget -> the search keeps going past it -> "
              "stop the search and return the best construction so far.\n</guidelines>")


def flat_program(counter):
    """A flat h = 0.5 program (C5 = 0.5) whose length varies so children never duplicate."""
    n = 40 + next(counter) % 40
    return f'"""Flat half, n={n}."""\n\ndef run():\n    return [0.5] * {n}, 0.5, {n}\n'


def responder(fmt_name):
    counter = itertools.count()

    def respond(prompt, params):
        if is_generation_call(params):
            return completion_for(fmt_name, flat_program(counter))
        kind = kind_of(prompt)
        if kind in DEFAULTS:
            text, finish = DEFAULTS[kind](prompt, 0, params["max_tokens"])
            return aux_reply_for(fmt_name, text), finish
        return aux_reply_for(fmt_name, GUIDELINES)
    return respond


def build(tmp_path, steps, resume=False):
    exp = Experiment(name="smoke-memory", task="erdos", model=MODEL, sampling=Sampling(),
                     search=Search(steps=steps, parents_per_step=2, children_per_parent=4, seed=3),
                     eval=Eval(timeout_s=60.0, cpus=1), memory=MemoryConfig(),
                     memo=MemoConfig(selected_children=2, boundary_every=2))
    out = tmp_path / "run"
    runtime = Runtime(out_dir=str(out), vllm_urls=("fake",), served_model="m", gen_concurrency=4,
                      resume=resume)
    task = tasks.get("erdos")
    fmt = build_format(MODEL, exp.sampling)
    client = FakeClient(responder(MODEL.format))
    memory = build_memory(exp.memory, task=task, fmt=fmt, client=client, aux=MODEL.aux)
    memo = build_memo(exp.memo, task=task, fmt=fmt, client=client, aux=MODEL.aux,
                      children_per_parent=4, context_window=exp.sampling.context_window,
                      out_dir=str(out))
    return exp, runtime, dict(task=task, evaluator=LocalEvaluator(4), clients=[client], fmt=fmt,
                              memory=memory, memo=memo), out


def test_memory_and_memo_end_to_end_with_resume(tmp_path):
    exp, runtime, parts, out = build(tmp_path, steps=4)
    loop.run(exp, runtime, **parts)
    rows = [json.loads(line) for line in (out / "rollouts.jsonl").read_text().splitlines()]
    assert len(rows) == 4 * 2 * 4
    assert any(r["beats_parent"] for r in rows if r["step"] == 0), "step 0 must give the memo something to mine"
    arms = {r["arm"] for r in rows if r["step"] >= 1}
    assert {"memo", "control"} <= arms, arms
    memo_state = json.loads((out / "checkpoints/step_000003/memo.json").read_text())
    assert memo_state["library"] and memo_state["last_boundary"] == 2
    memory_state = json.loads((out / "checkpoints/step_000003/memory.json").read_text())
    assert memory_state["guidelines"][0]["text"].startswith("a run nears the time budget -> ")
    assert (out / "memo" / "library_step_000000.json").exists()

    shutil.rmtree(out / "checkpoints" / "step_000003")
    exp, runtime, parts, _ = build(tmp_path, steps=4, resume=True)
    result = loop.run(exp, runtime, **parts)
    assert len(result["steps"]) == 4
    rows = [json.loads(line) for line in (out / "rollouts.jsonl").read_text().splitlines()]
    assert sorted({r["step"] for r in rows}) == [0, 1, 2, 3] and len(rows) == 32


def test_cli_builds_memory_and_memo(tmp_path, monkeypatch, capsys):
    import ttt.llm
    from ttt import cli, log

    def fake_vllm(url, served_model, api_key=None, timeout_s=None):
        client = FakeClient(responder(MODEL.format))
        client.base_url = url
        return client

    monkeypatch.setattr(ttt.llm, "VLLMClient", fake_vllm)
    (tmp_path / "models").mkdir()
    (tmp_path / "experiments").mkdir()
    (tmp_path / "models" / "oss.yaml").write_text(
        "format: harmony_two_phase\nphase1_max_tokens: 26000\ndate: '2026-07-24'\n")
    config = tmp_path / "experiments" / "erdos_memory_cli.yaml"
    config.write_text("task: erdos\nmodel: oss\nsearch:\n  steps: 4\n  parents_per_step: 2\n"
                      "  children_per_parent: 4\n  seed: 3\neval:\n  timeout_s: 60\n  cpus: 1\n"
                      "memory: {}\nmemo:\n  selected_children: 2\n")
    out = tmp_path / "run"
    try:
        assert cli.main(["run", "--config", str(config), "--vllm-url", "http://fake",
                         "--served-model", "m", "--out-dir", str(out), "--local-eval", "4"]) == 0
    finally:
        log.configure(None)
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["steps"] == 4
    rows = [json.loads(line) for line in (out / "rollouts.jsonl").read_text().splitlines()]
    # Erdős initial states carry no program, so the first minable boundary is step 2.
    assert {"memo", "control"} <= {r["arm"] for r in rows if r["step"] == 3}
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["config"]["memo"]["selected_children"] == 2
