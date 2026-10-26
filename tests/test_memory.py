import json

import pytest

from ttt.config import AuxBudgets, MemoryConfig
from ttt.memory import build_memory, extract_docstring
from ttt.memory.docstring import condense_docs, unfence
from ttt.memory.failures import code_excerpt, failure_signature, progress_tail, summarize_failures
from ttt.memory.guidelines import (STALE_STEPS, distiller_view, guideline_block, merge_guidelines,
                                   parse_guidelines, strip_label)
from ttt.memory.microdistill import MAX_ATTEMPTS, validate_summary
from ttt.prompts import load
from ttt.types import ChildOutcome, State

WARM = "warm start from the incumbent -> keeps the plateau -> initialise each restart from the best vector"
CAP = "timeout at large N -> inner loop never stops -> cap the inner loop at 2000 iterations"
GUIDELINES_REPLY = (
    "<guidelines>\n"
    f"Guideline 1: {WARM}\n"
    f"**Guideline 2:** {CAP}\n"
    "Looking at this -> nothing -> nothing\n"
    "</guidelines>"
)
TIMEOUT = "Evaluation timed out after 1100s."
LOOP_CODE = '"""Big loop."""\ndef propose_candidate():\n    for it in range(10**9):\n        pass\n'


class FakeTask:
    name = "toy"
    entrypoint = "propose_candidate"

    def __init__(self, maximize=False):
        self.maximize = maximize


class FakeFormat:
    name = "fake"

    def __init__(self, respond):
        self.respond = respond
        self.calls = []

    def aux_call(self, client, prompt, *, effort, max_tokens, temperature=1.0):
        self.calls.append({"prompt": prompt, "effort": effort, "max_tokens": max_tokens,
                           "temperature": temperature})
        return self.respond(prompt), "stop"


def _is_micro(prompt):
    return prompt.startswith(load("memory/microdistill_system.txt"))


def default_respond(prompt):
    if _is_micro(prompt):
        return json.dumps({"mechanism": "an unbounded loop", "evidence_quotes": ["for it in range(10**9):"],
                           "suggested_adaptation": "cap the iterations"})
    return GUIDELINES_REPLY


def make_state(sid, doc, parents=(), fenced=True):
    code = f'"""{doc}"""\nx = 1\n'
    if fenced:
        code = "```python\n" + code + "```"
    refs = [{"id": p.id, "timestep": p.timestep, "construction_hash": p.construction_hash} for p in parents]
    return State(id=sid, timestep=0, value=None, code=code, construction=[0.0],
                 parents=refs, construction_hash="h_" + sid)


def ok_outcome(raw, doc, admitted=False, beats=False, child=None):
    return ChildOutcome(ok=True, raw_score=raw, admitted=admitted, beats_parent=beats, docstring=doc,
                        code="x = 1", code_offset=50, error=None, stdout="progress", child=child)


def fail_outcome(error, code="", stdout="", offset=0):
    return ChildOutcome(ok=False, raw_score=None, admitted=False, beats_parent=False, docstring=None,
                        code=code, code_offset=offset, error=error, stdout=stdout)


def manager(respond=default_respond, guidelines=True, max_guidelines=24, maximize=False):
    fmt = FakeFormat(respond)
    mem = build_memory(MemoryConfig(guidelines=guidelines, max_guidelines=max_guidelines),
                       task=FakeTask(maximize), fmt=fmt, client=object(), aux=AuxBudgets())
    return mem, fmt


def lineage(mem):
    s0 = make_state("s0", "Start from a uniform vector.")
    s1 = make_state("s1", "Uniform start. Then refine with an LP.", parents=[s0])
    s2 = make_state("s2", "Uniform start. Then add an Adam polish.", parents=[s1, s0])
    for s, raw in ((s0, 1.52), (s1, 1.51), (s2, 1.505)):
        mem.register(s, raw)
    return s0, s1, s2


def test_extract_docstring_prefers_module_then_entrypoint():
    code = 'def helper():\n    """Helper doc."""\n\ndef propose_candidate():\n    """Main   idea\n here."""\n'
    assert extract_docstring(code, "propose_candidate") == "Main idea here."
    assert extract_docstring(code, None) == "Helper doc."
    assert extract_docstring('"""Module doc."""\n' + code, "propose_candidate") == "Module doc."
    assert extract_docstring("def (", "propose_candidate") is None
    assert extract_docstring("", None) is None
    assert unfence("```python\nx = 1\n```") == "x = 1\n"


def test_condense_docs_drops_shared_lead_sentence():
    docs = ["Works on the simplex. Uses LP.", "Works on the simplex. Uses Adam.", "Works on the simplex. Uses CMA."]
    assert condense_docs(docs, cap=2000) == ["Uses LP.", "Uses Adam.", "Uses CMA."]
    assert condense_docs(["a " * 30], cap=20) == ["a a a a a a a a a a ..."]


def test_recipe_renders_lineage_oldest_change_first():
    mem, _ = manager(guidelines=False)
    s0, s1, s2 = lineage(mem)
    expected = "\n".join([
        load("memory/recipe_heading.txt"),
        "  1.520000 -> 1.510000 (gain +1.00e-02): Then refine with an LP.",
        "  1.510000 -> 1.505000 (gain +5.00e-03): Then add an Adam polish.",
    ])
    assert mem.context_for(s2) == expected
    assert mem.context_for(s0) is None


def test_recipe_gain_sign_follows_maximize_and_unknown_scores():
    mem, _ = manager(guidelines=False, maximize=True)
    s0 = make_state("s0", "Start.")
    s1 = make_state("s1", "Better.", parents=[s0])
    mem.register(s0, None)
    mem.register(s1, 0.9)
    assert mem.context_for(s1).splitlines()[1] == "  Better."
    mem.register(s0, 0.8)
    assert "(gain +1.00e-01)" in mem.context_for(s1)


@pytest.mark.parametrize("line", [
    "Guideline 1: x y z -> m -> act now", "**Guideline 1:** x y z -> m -> act now",
    "**Guideline 1**: x y z -> m -> act now", "1. Guideline 1. x y z -> m -> act now",
    "- Guideline 1) x y z -> m -> act now", "<Guideline 1>: x y z -> m -> act now",
    "__GUIDELINE 12:__ x y z -> m -> act now", "Lesson: Guideline 1: x y z -> m -> act now",
])
def test_parse_guidelines_strips_decorated_labels(line):
    assert parse_guidelines(f"<guidelines>\n{line}\n</guidelines>") == ["x y z -> m -> act now"]
    assert parse_guidelines(line) == ["x y z -> m -> act now"]


def test_parse_guidelines_block_strict_mode_and_noise():
    assert parse_guidelines(GUIDELINES_REPLY) == [WARM, CAP]
    raw_reasoning = "We should think -> maybe -> do it\nGuideline 1: a big trigger -> mechanism -> do the action"
    assert parse_guidelines(raw_reasoning) == ["a big trigger -> mechanism -> do the action"]
    assert parse_guidelines("thinking</think>\nunlabelled cause -> mechanism -> act on it") == \
        ["unlabelled cause -> mechanism -> act on it"]
    assert parse_guidelines("") == []
    assert parse_guidelines("<guidelines>\nGuideline 1: no arrow here at all\n</guidelines>") == []
    assert parse_guidelines("<guidelines>\nGuideline 1: <trigger> -> <mechanism> -> <action>\n</guidelines>") == []
    assert parse_guidelines("<guidelines>\nunlabelled cause words -> mech -> act\n</guidelines>") == \
        ["unlabelled cause words -> mech -> act"]
    many = "\n".join(f"Guideline {i}: cause {i} words -> mech {i} -> act {i} now" for i in range(1, 16))
    assert len(parse_guidelines(many)) == 12
    assert strip_label("**Guideline 3:** text") == "text"


def test_parse_strips_echoed_view_markers():
    assert parse_guidelines("<guidelines>\n[2] [seen 3] Guideline 1: a -> b -> c\n[1] a -> b -> c\n</guidelines>") == []
    reply = ("<guidelines>\n[2] [seen 3] Guideline 1: big cause -> b mech -> c act now\n"
             "[1] plain cause -> mech -> act on it\n"
             "[seen 4]  [7] other cause -> m -> do it now\n"
             "Guideline 2: [3] [seen 1] late marker cause -> m -> act\n</guidelines>")
    assert parse_guidelines(reply) == ["big cause -> b mech -> c act now", "plain cause -> mech -> act on it",
                                       "other cause -> m -> do it now", "late marker cause -> m -> act"]
    assert parse_guidelines("[2] [seen 3] Guideline 1: big cause -> b mech -> c act now") == \
        ["big cause -> b mech -> c act now"]
    store = merge_guidelines([], [WARM], 0, max_guidelines=24)
    echoed = parse_guidelines(f"<guidelines>\n{distiller_view(store)}\n</guidelines>")
    assert echoed == [WARM]
    assert merge_guidelines(store, echoed, 1, max_guidelines=24) == [{"text": WARM, "seen": 2, "last_seen": 1}]


def test_merge_reaffirms_expires_and_caps():
    a = "warm start from the best vector -> keeps plateau -> init each restart from the incumbent"
    a2 = "warm start from the best vector -> keeps plateau -> init every restart from the incumbent"
    b = "shape mismatch -> N and N-1 arrays -> slice to the common length"
    store = merge_guidelines([], [a, b], 0, max_guidelines=24)
    store = merge_guidelines(store, [a2], 1, max_guidelines=24)
    assert store[0] == {"text": a, "seen": 2, "last_seen": 1}
    assert store[1] == {"text": b, "seen": 1, "last_seen": 0}
    assert merge_guidelines(store, [], STALE_STEPS + 1, max_guidelines=24) == [store[0]]
    c = "totally different words here -> other mechanism -> run the LP solver"
    assert [g["text"] for g in merge_guidelines(store, [c], 2, max_guidelines=2)] == [a, c]


def test_guideline_block_and_distiller_view_layouts():
    long_text = "crash -> missing helper -> do " + "x" * 300
    store = [{"text": long_text, "seen": 1, "last_seen": 0},
             {"text": "trigger words -> mechanism -> action words", "seen": 3, "last_seen": 0}]
    assert guideline_block(store).splitlines() == [
        load("memory/guideline_heading.txt"),
        "  1. [seen 3] trigger words -> mechanism -> action words",
        "  2. [seen 1] " + long_text[:240] + "…",
    ]
    assert distiller_view(store).splitlines() == [f"[1] [seen 1] {long_text}",
                                                  "[2] [seen 3] trigger words -> mechanism -> action words"]
    assert guideline_block([]) is None and distiller_view([]) == "(empty)"


def test_failure_signature_masks_variable_parts():
    a = failure_signature("Program execution failed: '(' was never closed (program.py, line 637)")
    b = failure_signature("Program execution failed: '(' was never closed (program.py, line 740)")
    assert a == b == "<S> was never closed (...)"
    assert failure_signature("") == "(no reason recorded)"
    assert failure_signature(TIMEOUT) == "Evaluation timed out after <N>s."


def test_progress_tail_and_code_excerpt():
    out = "step 1 of 10\nstep 2 of 10\nUserWarning: resource_tracker: leaked\n"
    assert progress_tail(out) == "step 1 of 10 / step 2 of 10"
    assert progress_tail("UserWarning: x\n") == ""
    code = "\n".join(f"line{i}" for i in range(1, 31))
    excerpt = code_excerpt(code, "invalid syntax (line 70)", 50)
    assert excerpt.splitlines()[0] == "    ... model lines 12-24 (failure at line 20):"
    assert code_excerpt(code, "invalid syntax (line 70)", 0).startswith("line1\nline2")


def test_summarize_failures_groups_and_discloses_truncation():
    records = [{"reason": f"Program execution failed: name 'f{i}' is not defined", "stdout": "", "code": "",
                "code_offset": 0} for i in range(3)]
    records += [{"reason": TIMEOUT, "stdout": "iter 5 of 9\n", "code": LOOP_CODE, "code_offset": 50}]
    records += [{"reason": f"error kind {chr(97 + i)} here", "stdout": "", "code": "", "code_offset": 0}
                for i in range(6)]
    lines = summarize_failures(records, {})
    assert lines[0] == "    x3 name <S> is not defined"
    assert lines[1:4] == [f"         e.g. name 'f{i}' is not defined" for i in range(3)]
    assert "         reached before the kill: iter 5 of 9" in lines
    assert "           | def propose_candidate():" in lines
    assert lines[-1] == "    (top 6 of 8 signatures shown, covering 80% of failures)"
    summ = {"Evaluation timed out after <N>s.": {"mechanism": "loop", "evidence_quotes": ["for it"],
                                                 "suggested_adaptation": "cap it"}}
    lines = summarize_failures(records, summ)
    assert "         grounded by: \"for it\"" in lines
    assert not any("reached before the kill" in ln for ln in lines)


def test_microdistill_gate_requires_verbatim_quotes():
    evidence = "for it in range(10):\n"
    good = {"mechanism": "m", "evidence_quotes": ["for it in range"], "suggested_adaptation": "a"}
    assert validate_summary(good, evidence)["evidence_quotes"] == ["for it in range"]
    assert validate_summary({**good, "evidence_quotes": ["for each it"]}, evidence) is None
    assert validate_summary({**good, "suggested_adaptation": ""}, evidence) is None
    assert validate_summary({"mechanism": 3}, evidence) is None
    assert validate_summary({"mechanism": "Unclear"}, evidence)["mechanism"] == "unclear"


def test_distill_end_to_end_with_failures_and_microdistill():
    mem, fmt = manager()
    s0, s1, s2 = lineage(mem)
    child = make_state("c", "Child idea.", parents=[s2, s1, s0])
    mem.record_child(s2, ok_outcome(1.50, "Child idea.", admitted=True, beats=True, child=child))
    mem.record_child(s2, ok_outcome(1.506, "Other idea."))
    mem.record_child(s2, fail_outcome(TIMEOUT, code=LOOP_CODE, stdout="iter 3\n", offset=50))
    mem.end_of_step(0, 1.50)
    assert [c["max_tokens"] for c in fmt.calls] == [1200, 1024]
    assert all(c["effort"] == "low" and c["temperature"] == 1.0 for c in fmt.calls)
    question = fmt.calls[1]["prompt"]
    assert question.startswith(load("memory/guideline_update.txt") + "\n\n"
                               + load("memory/crash_attribution_guide.txt"))
    assert "\n\n## CURRENT playbook\n(empty)\n\n## THIS STEP\nStep outcome (3 rollouts, lower raw score" in question
    assert "  score 1.500000 (improved its parent, best of 2 from that parent): Child idea." in question
    assert "mechanism (from a full-evidence read): an unbounded loop" in question
    assert question.endswith("\n\nEmit the updated lessons now.")
    assert mem.guidelines == [{"text": WARM, "seen": 1, "last_seen": 0}, {"text": CAP, "seen": 1, "last_seen": 0}]
    assert mem.failure_summaries["Evaluation timed out after <N>s."]["count"] == 1
    ctx = mem.context_for(child)
    assert ctx.startswith(load("memory/recipe_heading.txt"))
    assert ctx.endswith("\n\n" + load("memory/guideline_heading.txt")
                        + f"\n  1. [seen 1] {WARM}\n  2. [seen 1] {CAP}")
    mem.record_child(s2, fail_outcome(TIMEOUT, code=LOOP_CODE, stdout="iter 4\n", offset=50))
    mem.end_of_step(1, 1.50)
    assert len(fmt.calls) == 3
    assert f"## CURRENT playbook\n[1] [seen 1] {WARM}\n[2] [seen 1] {CAP}\n\n## THIS STEP\n" in fmt.calls[2]["prompt"]
    assert [g["seen"] for g in mem.guidelines] == [2, 2]


def test_rejected_microdistill_stops_after_max_attempts():
    mem, fmt = manager(respond=lambda p: '{"mechanism": "x", "evidence_quotes": ["nope"], '
                                         '"suggested_adaptation": "y"}' if _is_micro(p) else "")
    s0 = make_state("s0", "Start.")
    mem.register(s0, 1.0)
    for step in range(MAX_ATTEMPTS + 1):
        mem.record_child(s0, fail_outcome(TIMEOUT, code=LOOP_CODE, offset=50))
        mem.end_of_step(step, None)
    assert sum(_is_micro(c["prompt"]) for c in fmt.calls) == MAX_ATTEMPTS
    assert mem.failure_summaries == {}


def test_failed_llm_call_warns_and_keeps_guidelines(capsys):
    def boom(prompt):
        raise RuntimeError("server down")

    mem, fmt = manager()
    s0 = make_state("s0", "Start.")
    mem.register(s0, 1.0)
    mem.record_child(s0, ok_outcome(0.9, "Better."))
    mem.end_of_step(0, 0.9)
    before = list(mem.guidelines)
    fmt.respond = boom
    mem.record_child(s0, fail_outcome(TIMEOUT, code=LOOP_CODE, offset=50))
    mem.end_of_step(1, 0.9)
    assert mem.guidelines == before and before
    assert mem.microdistill_attempts == {"Evaluation timed out after <N>s.": 1}
    err = capsys.readouterr().err
    assert '"event": "memory_distill_failed"' in err and '"event": "memory_microdistill_failed"' in err
    assert '"event": "memory_guidelines"' in err


def test_guidelines_off_makes_no_llm_calls():
    mem, fmt = manager(guidelines=False)
    s0, s1, s2 = lineage(mem)
    mem.guidelines = [{"text": "a b c -> d e -> f g h", "seen": 1, "last_seen": 0}]
    mem.record_child(s2, fail_outcome(TIMEOUT, code=LOOP_CODE, offset=50))
    mem.record_child(s2, ok_outcome(1.0, "x"))
    mem.end_of_step(0, 1.0)
    assert fmt.calls == []
    assert mem.context_for(s2).startswith(load("memory/recipe_heading.txt"))
    assert "Guideline" not in mem.context_for(s2)


def test_state_dict_round_trip():
    mem, _ = manager()
    s0, s1, s2 = lineage(mem)
    mem.record_child(s2, fail_outcome(TIMEOUT, code=LOOP_CODE, offset=50))
    mem.end_of_step(0, 1.5)
    mem.record_child(s2, fail_outcome("boom (line 3)", code="x = (", offset=1))
    state = json.loads(json.dumps(mem.state_dict()))
    assert state["guidelines"] and "lessons" not in state
    other, _ = manager()
    other.load_state_dict(state)
    assert other.state_dict() == mem.state_dict()
    assert other.context_for(s2) == mem.context_for(s2)
    with pytest.raises(ValueError):
        other.load_state_dict({**state, "version": 1})
    with pytest.raises(ValueError):
        other.load_state_dict({"version": 2})
