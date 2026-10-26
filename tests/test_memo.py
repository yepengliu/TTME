"""Memo route: plan rows, Stage-1 selection, fallbacks, call limits, boundaries and resume."""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from test_memo_fakes import (FakeFormat, archived, events, improving, listed_ids, make_loop,
                             make_state, seed_library)
from ttt.config import MemoBudgets, MemoConfig
from ttt.memo import text as T
from ttt.memo.library import Library
from ttt.memo.llm import run_pool
from ttt.prompts import load

ROOT = Path(__file__).resolve().parents[1]

CONTROL = "\n\n" + load("cards/deepen_v2.txt").strip() + "\n"


def parents(n=2):
    return [make_state("p%d" % i, -1.5 - i / 100) for i in range(n)]


def memo_text(fid, label):
    return ("\n\n## Assigned direction\nDirection ID: %s\n\nApply %s here, concretely.\n"
            % (T.direction_id(fid), label))


def test_plan_is_none_while_library_empty():
    loop, fmt = make_loop()
    assert loop.plan(0, parents(), None) is None
    assert fmt.calls == []


def test_plan_rows_single_method():
    loop, fmt = make_loop()
    loop.library = seed_library(3)
    plans = loop.plan(3, parents(2), None)
    assert len(plans) == 2 and all(len(g) == 64 for g in plans)
    for gi, rows in enumerate(plans):
        memo = [a for a in rows if a.arm == "memo"]
        ctrl = [a for a in rows if a.arm == "control"]
        assert len(memo) == 32 and len(ctrl) == 32
        fid = memo[0].tags["family"]
        assert {a.text for a in memo} == {memo_text(fid, loop.library.families[fid].label)}
        assert {a.text for a in ctrl} == {CONTROL}
        assert memo[0].tags == {"group": gi, "library_sha": loop.library.sha(), "family": fid,
                                "direction_id": T.direction_id(fid), "method_rank": 0}
        assert ctrl[0].tags["control_card"] == "deepen_v2"
        assert [a.arm for a in rows] != ["memo"] * 32 + ["control"] * 32


def test_plan_rows_two_methods_split_the_selected_rows():
    loop, fmt = make_loop(cfg=MemoConfig(methods_per_parent=2))
    loop.library = seed_library(4)
    rows = loop.plan(1, parents(1), None)[0]
    fams = [a.tags["family"] for a in rows if a.arm == "memo"]
    assert sorted({f: fams.count(f) for f in fams}.values()) == [16, 16]
    assert len({a.text for a in rows if a.arm == "memo"}) == 2
    assert '"method_ids"' in fmt.of("select")[0]["prompt"]
    assert "Choose the 2 methods" in fmt.of("select")[0]["prompt"]


def test_none_control_card_is_an_empty_direction():
    loop, _ = make_loop(cfg=MemoConfig(control_card="none"))
    loop.library = seed_library(2)
    rows = loop.plan(1, parents(1), None)[0]
    assert {a.text for a in rows if a.arm == "control"} == {"\n\n\n"}


def test_serial_selection_shows_earlier_picks():
    loop, fmt = make_loop(cfg=MemoConfig(select="serial"))
    loop.library = seed_library(3)
    loop.plan(1, parents(3), None)
    prompts = [c["prompt"] for c in fmt.of("select")]
    assert all(p.startswith(load("memo/select_serial.txt")) for p in prompts)
    assert "ALREADY ASSIGNED TO OTHER GROUPS" not in prompts[0]
    first = loop.library.families[listed_ids(prompts[0])[0]].label
    assert "## METHODS ALREADY ASSIGNED TO OTHER GROUPS THIS STEP\n- %s" % first in prompts[1]


def test_parallel_selection_uses_the_blind_prompt():
    loop, fmt = make_loop(cfg=MemoConfig(select="parallel"))
    loop.library = seed_library(3)
    loop.plan(1, parents(3), None)
    prompts = [c["prompt"] for c in fmt.of("select")]
    assert len(prompts) == 3
    assert all(p.startswith(load("memo/select.txt") + "\n\n") for p in prompts)
    assert not any("ALREADY ASSIGNED" in p for p in prompts)


def test_failed_select_call_falls_back_to_control(capsys):
    def select(prompt, idx, max_tokens):
        if "print('p1')" in prompt:
            raise RuntimeError("server down")
        return json.dumps({"method_id": listed_ids(prompt)[0]}), "stop"

    loop, fmt = make_loop(fmt=FakeFormat(select=select))
    loop.library = seed_library(2)
    plans = loop.plan(1, parents(2), None)
    assert {a.arm for a in plans[1]} == {"control"}
    assert all(a.tags.get("stage1") == "failed" for a in plans[1])
    assert sum(a.arm == "memo" for a in plans[0]) == 32
    kinds = {e["event"] for e in events(capsys) if e["level"] == "warning"}
    assert {"memo_select_call_failed", "memo_group_control_fallback"} <= kinds


def test_unknown_method_and_empty_memo_fall_back(capsys):
    loop, _ = make_loop(fmt=FakeFormat(select=lambda p, i, m: ('{"method_id": "nope"}', "stop")))
    loop.library = seed_library(2)
    assert all(a.arm == "control" for a in loop.plan(1, parents(1), None)[0])
    assert "memo_select_unknown_method" in {e["event"] for e in events(capsys)}
    loop, _ = make_loop(fmt=FakeFormat(write=lambda p, i, m: ("", "stop")))
    loop.library = seed_library(2)
    assert all(a.arm == "control" for a in loop.plan(1, parents(1), None)[0])
    assert "memo_write_empty" in {e["event"] for e in events(capsys)}


def test_unwritten_method_rows_go_to_the_survivor():
    def write(prompt, idx, max_tokens):
        return ("", "stop") if "Method 1 label" in prompt else ("Do it.", "stop")

    loop, _ = make_loop(fmt=FakeFormat(write=write), cfg=MemoConfig(methods_per_parent=2))
    loop.library = seed_library(2)
    rows = loop.plan(1, parents(1), None)[0]
    assert {a.tags["family"] for a in rows if a.arm == "memo"} == {"fam_00"}
    assert sum(a.arm == "memo" for a in rows) == 32


def test_memo_is_capped_at_a_sentence():
    long = "This is one sentence of the memo. " * 40
    loop, _ = make_loop(fmt=FakeFormat(write=lambda p, i, m: (long, "stop")))
    loop.library = seed_library(1)
    text = next(a.text for a in loop.plan(1, parents(1), None)[0] if a.arm == "memo")
    body = text.split("\n\n")[-1].strip()
    assert len(body) <= T.MEMO_MAX_CHARS and body.endswith(".")


def test_stage1_listing_fits_the_context(capsys):
    lib = seed_library(30, words=80)
    ranked = lib.ranked()
    probe = T.select_question(load("memo/select_serial.txt"), parents(1)[0], False, None, None,
                              [], ranked[:10])
    ctx = len(probe.split()) + 4500 + 64
    loop, fmt = make_loop(context_window=ctx)
    loop.library = lib
    loop.plan(1, parents(1), None)
    prompt = fmt.of("select")[0]["prompt"]
    assert len(prompt.split()) + 4500 + 64 <= ctx
    assert set(listed_ids(prompt)) == {f.fid for f in ranked[:10]}
    assert any(e["event"] == "memo_listing_trimmed" for e in events(capsys))


def test_lineage_and_memory_blocks():
    class Mem:
        def context_for(self, parent):
            return "Lesson for %s." % parent.id

    loop, fmt = make_loop()
    loop.library = seed_library(2)
    loop.assigned = {"a1": ["fam_00"]}
    child_of = seed_library(1).variants["fam_00_v"].notes[0].child_id
    p = make_state(child_of, -1.4, parents=[{"id": "a1"}], parent_values=[-1.45])
    loop.plan(1, [p], Mem())
    prompt = fmt.of("select")[0]["prompt"]
    assert ("- 1 step(s) back: that step steered its group toward method `fam_00` (this "
            "program may or may not have received it)  score 1.4500000000 -> 1.4000000000 "
            "(better)\n    what that change actually did: Note about method 0.") in prompt
    assert "## WHAT HAS GONE WRONG BEFORE\nLesson for %s." % child_of in prompt
    assert "LOWER is better. 1.4000000000, 1.4500000000" in prompt


def test_lineage_phrases_keep_the_research_wording():
    p = make_state("c", -1.4, parents=[{"id": "a2"}, {"id": "a1"}, {"id": "a0"}],
                   parent_values=[-1.45, -1.5, -1.6])
    block = T.lineage_block(p, False, {"a2": ["fam_00", "fam_01"], "a1": ["fam_02"]}, {})
    assert block.split("\n") == [
        "- 1 step(s) back: that step split its group across methods `fam_00`, `fam_01` (this "
        "program may or may not have received one of them)  score 1.4500000000 -> 1.4000000000 "
        "(better)",
        "- 2 step(s) back: that step steered its group toward method `fam_02` (this program may "
        "or may not have received it)  score 1.5000000000 -> 1.4500000000 (better)",
        "- 3 step(s) back: no method was in use then  score 1.6000000000 -> 1.5000000000 "
        "(better)"]
    question = T.select_question("TASK", p, True, None, None, [], [])
    assert "## SCORE OF THIS PROGRAM AND ITS RECENT ANCESTORS\nHigher is better. " \
        "-1.4000000000, -1.4500000000, -1.5000000000, -1.6000000000 (most recent first)" \
        in question


def test_pool_abandons_hung_calls_on_daemon_threads(capsys):
    release = threading.Event()

    def fn(x):
        if x == "hang":
            release.wait(60)
        if x == "boom":
            raise RuntimeError("bad reply")
        return x

    t0 = time.monotonic()
    res = run_pool(fn, ["ok", "hang", "boom"], workers=3, deadline=time.monotonic() + 0.3,
                   kind="t_pool")
    try:
        assert time.monotonic() - t0 < 10
        assert res.done == {0: "ok"} and res.uncollected == [1] and list(res.failed) == [2]
        alive = [t for t in threading.enumerate() if t.name.startswith("t_pool-")]
        assert alive and all(t.daemon for t in alive)
    finally:
        release.set()
    kinds = {e["event"] for e in events(capsys)}
    assert {"t_pool_call_failed", "t_pool_budget_spent"} <= kinds


def test_a_hung_pool_call_does_not_hold_the_process_at_exit():
    script = ("import sys, time\nsys.path.insert(0, %r)\nfrom ttt.memo.llm import run_pool\n"
              "res = run_pool(lambda x: time.sleep(600), [0], workers=1,\n"
              "               deadline=time.monotonic() + 0.2, kind='t_exit')\n"
              "print(res.uncollected)\n" % str(ROOT))
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         timeout=60)
    assert out.returncode == 0 and out.stdout.strip() == "[0]"


def record_step(loop, step, ps, children=4, base=-1.5):
    for gi, p in enumerate(ps):
        for j in range(children):
            child = make_state("c%d_%d_%d" % (step, gi, j), base + 0.001 * (j + 1),
                               code="```python\nprint(%d, %d, %d)\n```" % (step, gi, j))
            loop.record(step, gi, p, None, improving(child, -(base + 0.001 * (j + 1))))


def test_boundary_grows_the_library_and_writes_snapshots(tmp_path):
    loop, fmt = make_loop(out_dir=str(tmp_path))
    record_step(loop, 0, parents(2))
    loop.end_of_step(0)
    loop.boundary(0, archived(loop))
    assert len(loop.library.families) == 8 and loop.pending == []
    assert all(f.memory.startswith("1. MECHANISM") for f in loop.library.families.values())
    assert {c["effort"] for c in fmt.of("family")} == {"medium"}
    assert {c["max_tokens"] for c in fmt.of("family")} == {8000}
    assert os.path.exists(tmp_path / "memo" / "library_step_000000.json")
    log = [json.loads(x) for x in open(tmp_path / "memo" / "merge_log.jsonl")]
    assert log[0]["status"] == "ok" and log[0]["placed"] == 8
    plans = loop.plan(1, parents(2), None)
    assert plans is not None and loop.last_boundary == 0
    loop.boundary(1, archived(loop))
    loop.boundary(0, archived(loop))
    assert len(fmt.of("mine")) == 8


def test_every_memo_call_carries_the_call_timeout():
    cfg = MemoConfig(budgets=MemoBudgets(call_timeout_s=123.0), methods_per_parent=2)
    loop, fmt = make_loop(cfg=cfg)
    record_step(loop, 0, parents(2))
    loop.end_of_step(0)
    loop.boundary(0, archived(loop))
    assert loop.plan(1, parents(2), None) is not None
    assert {c["kind"] for c in fmt.calls} == {"select", "write", "mine", "merge", "family"}
    assert {c["timeout_s"] for c in fmt.calls} == {123.0}


def test_a_repeated_boundary_replaces_its_merge_log_records(tmp_path, capsys):
    loop, _ = make_loop(out_dir=str(tmp_path))
    record_step(loop, 0, parents(2))
    loop.end_of_step(0)
    loop.boundary(0, archived(loop))
    record_step(loop, 2, parents(2), base=-1.6)
    loop.end_of_step(2)
    state, ids = json.loads(json.dumps(loop.state_dict())), archived(loop)
    loop.boundary(2, ids)
    log_path = tmp_path / "memo" / "merge_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"step": 4, "status": "later"}) + "\n" + '{"step": 6, "sta')
    resumed, fmt = make_loop(out_dir=str(tmp_path))
    resumed.load_state_dict(state)
    resumed.boundary(2, ids)
    assert fmt.of("merge")
    log = [json.loads(x) for x in open(log_path, encoding="utf-8")]
    assert [r["step"] for r in log] == [0, 2] and log[1]["status"] == "ok"
    assert not os.path.exists(str(log_path) + ".tmp")
    seen = events(capsys)
    trimmed = [(e["dropped"], e["kept"]) for e in seen if e["event"] == "memo_merge_log_trimmed"]
    assert trimmed == [(2, 1)]
    assert any(e["event"] == "memo_merge_log_bad_line" for e in seen)


def test_second_boundary_attaches_to_existing_families():
    loop, fmt = make_loop(cfg=MemoConfig(boundary_every=1))
    record_step(loop, 0, parents(2))
    loop.end_of_step(0)
    loop.boundary(0, archived(loop))
    before = set(loop.library.families)
    record_step(loop, 1, parents(2), base=-1.6)
    loop.end_of_step(1)
    loop.boundary(1, archived(loop))
    assert set(loop.library.families) == before
    assert sum(len(f.variants) for f in loop.library.families.values()) > len(before)


def test_end_of_step_counts_and_assigned_index():
    loop, _ = make_loop()
    loop.library = seed_library(2)
    ps = parents(1)
    rows = loop.plan(1, ps, None)[0]
    child = make_state("c", -1.6)
    for a in rows:
        loop.record(1, 0, ps[0], a, improving(child, 1.6) if a.arm == "memo" else
                    improving(child, 1.6, admitted=False))
    loop.end_of_step(1)
    fid = next(a.tags["family"] for a in rows if a.arm == "memo")
    stats = loop.library.families[fid].stats
    assert stats["memo_n"] == 32 and stats["control_n"] == 32
    assert stats["parent_group_count"] == 1 and loop.assigned == {"p0": [fid]}


def test_state_dict_round_trip():
    loop, _ = make_loop()
    record_step(loop, 0, parents(2))
    loop.end_of_step(0)
    loop.boundary(0, archived(loop))
    record_step(loop, 1, parents(2), base=-1.7)
    state = json.loads(json.dumps(loop.state_dict()))
    other, _ = make_loop()
    other.load_state_dict(state)
    assert other.state_dict() == state
    assert isinstance(other.library, Library) and other.last_boundary == 0


def test_interrupted_boundary_runs_again_after_resume():
    loop, _ = make_loop()
    record_step(loop, 0, parents(2))
    loop.end_of_step(0)
    ids = archived(loop)
    state = json.loads(json.dumps(loop.state_dict()))
    resumed, fmt = make_loop()
    resumed.load_state_dict(state)
    assert resumed.last_boundary is None
    assert resumed.plan(1, parents(2), None) is None and fmt.calls == []
    resumed.boundary(0, ids)
    assert resumed.last_boundary == 0 and len(fmt.of("mine")) == 8
    resumed.boundary(0, ids)
    assert len(fmt.of("mine")) == 8
    assert resumed.plan(1, parents(2), None) is not None


def test_bad_config_is_rejected():
    with pytest.raises(ValueError):
        make_loop(cfg=MemoConfig(boundary_every=0))
    with pytest.raises(FileNotFoundError):
        make_loop(cfg=MemoConfig(control_card="nope"))
