"""Merge reply guards, the context-fitted ladder, rounds, apply_merge and family memory."""
import json
import re
from dataclasses import replace

import pytest

from test_memo_fakes import (TIMEOUT, FakeClient, FakeFormat, archived, failed_outcome,
                             improving, make_loop, make_state, merge_ids, merge_place,
                             seed_library)
from ttt.memo.family import distill
from ttt.memo.library import MergeGroup, Note, cid_for
from ttt.memo.llm import LLM
from ttt.memo.merge import merge_call, parse_reply
from ttt.memo.rounds import MergeSettings, merge_rounds, plan_round, request_hash
from ttt.memo.text import merge_question
from ttt.prompts import load


def notes(n, words=40, tag="n"):
    return [Note(sig="%s%02d" % (tag, i), memo=" ".join(["note", tag, str(i)] + ["w"] * words),
                 step=0) for i in range(n)]


def settings(ctx=32768, **kw):
    base = dict(ctx=ctx, margin=64, start_max_tokens=9000, max_tokens_cap=16384, drain=True,
                drain_min_notes=8, max_rounds=8, rounds_budget_s=720.0, call_timeout_s=580.0)
    base.update(kw)
    return MergeSettings(**base)


def count(q):
    return len(q.split())


def fake_llm(fmt, client=None):
    return LLM(fmt, client or FakeClient(), TIMEOUT)


FAMS = [("fa", "A label", "Do A."), ("fb", "B label", "Do B."), ("fc", "C label", "Do C.")]


def reply(groups):
    return json.dumps({"groups": groups})


def test_ids_route_by_prefix_and_invented_ids_are_dropped():
    ns = notes(3)
    rep = parse_reply(reply([{"existing_ids": ["N0", "E0"], "label": "A", "direction": "d"},
                             {"ids": ["N1", "E9"], "label": "X", "direction": "d"}]),
                      "stop", FAMS, ns)
    assert rep.kind is None and rep.dropped == ["E9"]
    assert (rep.groups[0].e, rep.groups[0].n) == (["fa"], ["n00"])
    assert rep.keep_pending == ["n02"]
    assert {g.e[0] for g in rep.groups if g.reuse} == {"fb", "fc"}
    many = parse_reply(reply([{"ids": ["E7", "E8", "E9", "E10", "E11", "N0"], "label": "A",
                               "direction": "d"}]), "stop", FAMS, ns)
    assert many.kind == "unknown_ids_over_cap"


def test_guards_refuse_bad_partitions():
    ns = notes(2)
    collapse = reply([{"ids": ["E0", "E1", "E2", "N0", "N1"], "label": "A", "direction": "d"}])
    assert parse_reply(collapse, "stop", FAMS, ns).kind == "collapse"
    assert parse_reply("no json here", "stop", FAMS, ns).kind == "no_json_stop"
    assert parse_reply(reply([]), "length", FAMS, ns).kind == "finish_length"
    assert parse_reply("{bad json}", "stop", FAMS, ns).kind == "bad_json"
    assert parse_reply(reply([{"ids": "E0"}]), "stop", FAMS, ns).kind == "malformed_reply"
    assert parse_reply(reply(["E0"]), "stop", FAMS, ns).kind == "not_object"
    nodir = reply([{"ids": ["E0", "N0"], "label": "A"}, {"ids": ["N1"], "label": "Z"}])
    assert parse_reply(nodir, "stop", FAMS, ns).kind == "no_direction"


def test_zero_absorption_is_asked_once_more_with_the_retry_prefix():
    ns = notes(2)
    alone = reply([{"ids": ["N0"], "label": "New 0", "direction": "d"},
                   {"ids": ["N1"], "label": "New 1", "direction": "d"}])
    sent = []

    def call(q):
        sent.append(q)
        return alone, "stop"

    rep = merge_call(call, FAMS, {}, ns, (2, 320), lambda: True)
    assert rep.kind == "zero_absorption" and len(sent) == 2
    assert sent[1] == load("memo/merge_retry_prefix.txt") + sent[0]
    sent.clear()
    assert merge_call(call, FAMS, {}, ns, (2, 320), lambda: False).kind == \
        "zero_absorption_no_room"
    assert len(sent) == 1


def test_ladder_takes_r0_when_today_request_fits():
    fams, ex = seed_library(5).merge_view()
    q = notes(30)
    plan = plan_round(count, fams, ex, q, settings(), 64, parity=True)
    assert (plan.rung, plan.n, plan.parity) == ("R0", 24, True)
    assert plan.max_tokens == 16384
    assert plan.hash == request_hash(merge_question(fams, ex, q[:24], 2, 320), 16384)


def test_ladder_cuts_notes_before_exemplars_and_never_drops_a_family():
    fams, ex = seed_library(5).merge_view()
    q = notes(30)
    t0 = count(merge_question(fams, ex, q[:24], 2, 320))
    plan = plan_round(count, fams, ex, q, settings(ctx=t0 + 9000 - 1), 64, parity=True)
    assert plan.rung == "R1" and plan.n == 16
    t4 = count(merge_question(fams, ex, q[:4], 0, 0, retry=True))
    plan = plan_round(count, fams, ex, q, settings(ctx=t4 + 9000 + 64), 64, parity=True)
    assert (plan.rung, plan.n) == ("R4", 4)
    shown = merge_question(fams, ex, q[:plan.n], plan.k, plan.chars)
    assert merge_ids(shown)[0] == list(range(len(fams)))
    assert plan_round(count, fams, ex, q, settings(ctx=t4 + 9000), 64, parity=True) is None


def test_length_finishes_are_rescued_with_smaller_requests_and_never_resent():
    fmt = FakeFormat(merge=lambda p, i, m: ("", "length"))
    llm = fake_llm(fmt)
    lib = seed_library(3)
    res = merge_rounds(llm, lib, notes(30), 4, settings(), [])
    assert [r["outcome"] for r in res.rounds] == ["finish_length"] * 3
    assert [r["notes_shown"] for r in res.rounds] == [24, 12, 6]
    assert res.stop == "consecutive_failures" and len(res.queue) == 30
    sent = [request_hash(c["prompt"], c["max_tokens"]) for c in fmt.of("merge")]
    assert len(sent) == len(set(sent)) and set(sent) <= set(res.new_failed)
    assert len(lib.families) == 3


def test_failed_requests_are_not_resent_by_a_later_boundary():
    fmt = FakeFormat(merge=lambda p, i, m: ("not json", "stop"))
    llm = fake_llm(fmt)
    lib, q = seed_library(3), notes(30)
    first = merge_rounds(llm, lib, q, 4, settings(), [])
    second = merge_rounds(llm, lib, q, 6, settings(), first.new_failed)
    sent = [request_hash(c["prompt"], c["max_tokens"]) for c in fmt.of("merge")]
    assert len(sent) == len(set(sent)) == 5
    assert [r["outcome"] for r in second.rounds] == ["no_json_stop"] * 2 + ["repeat_refused"]
    assert second.rounds[0]["rotated"] >= 1


def test_http_400_moves_down_the_ladder_with_a_wider_margin():
    def merge(prompt, idx, max_tokens):
        if idx == 0:
            raise RuntimeError("400: your prompt contains 5000 input tokens")
        return merge_place(prompt, idx, max_tokens)

    lib = seed_library(3)
    res = merge_rounds(fake_llm(FakeFormat(merge=merge)), lib, notes(30), 4,
                       settings(), [])
    assert [r["outcome"] for r in res.rounds[:2]] == ["call_error", "ok"]
    assert res.rounds[1]["rung"] == "R1" and res.margin > 64
    assert res.rounds[0]["calls"][0]["server_count"] == 5000


def test_drain_places_the_backlog_and_no_drain_keeps_todays_capacity():
    llm = fake_llm(FakeFormat())
    res = merge_rounds(llm, seed_library(3), notes(60), 4, settings(), [])
    assert [r["notes_shown"] for r in res.rounds] == [24, 24, 12]
    assert res.stop == "queue_empty" and res.queue == []
    res = merge_rounds(llm, seed_library(3), notes(60), 4, settings(drain=False), [])
    assert [r["notes_shown"] for r in res.rounds] == [24]
    assert res.stop == "done" and len(res.queue) == 36


def test_apply_merge_fuses_into_the_oldest_family():
    lib = seed_library(3)
    lib.families["fam_00"].stats = {"memo_n": 3}
    lib.families["fam_02"].stats = {"memo_n": 5}
    ns = notes(2)
    rec = lib.apply_merge([MergeGroup(e=["fam_02", "fam_00"], n=["n00"], label="Merged",
                                      direction="Merged direction."),
                           MergeGroup(e=[], n=["n01"], label="Method 1 label",
                                      direction="Restated.")], ns, 7)
    assert set(lib.families) == {"fam_00", "fam_01"}
    fused = lib.families["fam_00"]
    assert (fused.label, fused.direction) == ("Merged", "Merged direction.")
    assert fused.memory == "Memory of method 0.\n\nMemory of method 2."
    assert fused.stats["memo_n"] == 8
    assert rec["partition"] == {"fam_00": ["fam_02"]} and rec["relabel_merged"] == 1
    assert lib.families["fam_01"].label == "Method 1 label"
    assert [n.placed_step for n in lib.notes_of("fam_00") if n.sig == "n00"] == [7]


@pytest.mark.parametrize("fusion_first", [True, False])
def test_a_fused_family_is_not_resurrected_by_its_label(fusion_first):
    lib = seed_library(3)
    lib.families["fam_00"].stats = {"memo_n": 3}
    lib.families["fam_02"].stats = {"memo_n": 5}
    fuse = MergeGroup(e=["fam_02", "fam_00"], n=["n00"], label="Merged",
                      direction="Merged direction.")
    by_label = MergeGroup(e=[], n=["n01"], label="Method 2 label", direction="Restated.")
    rec = lib.apply_merge([fuse, by_label] if fusion_first else [by_label, fuse], notes(2), 7)
    assert set(lib.families) == {"fam_00", "fam_01"}
    fused = lib.families["fam_00"]
    assert fused.memory == "Memory of method 0.\n\nMemory of method 2."
    assert fused.stats["memo_n"] == 8
    assert (fused.label, fused.direction) == ("Merged", "Merged direction.")
    assert sorted(n.sig for n in lib.notes_of("fam_00")) == ["n00", "n01", "s00", "s02"]
    assert {v.family for v in lib.variants.values()} == {"fam_00", "fam_01"}
    assert rec["redirected"] == int(fusion_first)


def test_a_later_round_of_the_boundary_files_a_fused_label_under_the_survivor():
    lib = seed_library(3)
    gone = cid_for("Second head")
    lib.families[gone] = replace(lib.families.pop("fam_02"), fid=gone, label="Second head")
    lib.variants["fam_02_v"].family = gone

    def merge(prompt, idx, max_tokens):
        if idx == 0:
            return reply([{"ids": ["E0", "E2", "N0"], "label": "Merged", "direction": "F."}]), \
                "stop"
        return reply([{"ids": ["E0", "N0"], "label": "Merged", "direction": "F."},
                      {"ids": ["N1"], "label": "Second head", "direction": "Again."}]), "stop"

    fmt = FakeFormat(merge=merge)
    res = merge_rounds(fake_llm(fmt), lib, notes(30), 4, settings(max_rounds=2), [])
    assert [r["outcome"] for r in res.rounds] == ["ok", "ok"]
    assert set(lib.families) == {"fam_00", "fam_01"}
    assert lib.families["fam_00"].memory == "Memory of method 0.\n\nMemory of method 2."
    filed = re.search(r"^NEW N1\nmemo: (note n \d+) ", fmt.of("merge")[1]["prompt"], re.M)
    assert any(n.memo.startswith(filed.group(1) + " ") for n in lib.notes_of("fam_00"))
    assert (res.rounds[1]["redirected"], res.rounds[1]["attached"]) == (1, 2)


def test_untouched_family_keeps_its_head_and_unplaced_variants_are_carried():
    lib = seed_library(2)
    lib.families["fam_01"].direction = "Consolidated direction."
    lib.apply_merge([MergeGroup(e=[], n=["n00"], label="New", direction="New dir.")],
                    notes(1), 3)
    assert lib.families["fam_01"].direction == "Consolidated direction."
    assert len(lib.families) == 3 and {"fam_00", "fam_01"} <= set(lib.families)
    assert lib.families["fam_00"].variants == ["fam_00_v"]


def test_family_memory_carries_notes_it_never_reached():
    lib = seed_library(2)
    for n in lib.notes_of("fam_00") + lib.notes_of("fam_01"):
        n.placed_step = 5
    fmt = FakeFormat()
    stats, carry = distill(fake_llm(fmt), lib, 5, {}, budget_s=0.0, workers=2,
                           max_tokens=8000)
    assert stats["budget"] == 2 and set(carry) == {"s00", "s01"} and fmt.of("family") == []
    stats, carry = distill(fake_llm(fmt), lib, 7, carry, budget_s=60.0, workers=2,
                           max_tokens=8000)
    assert stats["carried_in"] == 2 and stats["ok"] == 2 and carry == {}
    assert lib.families["fam_00"].memory.startswith("1. MECHANISM")
    assert lib.families["fam_00"].memory_rounds == 1
    _, carry = distill(fake_llm(fmt), lib, 13, {"s00": 5}, budget_s=0.0, workers=1,
                       max_tokens=8000)
    assert carry == {}


def test_family_memory_failure_keeps_the_old_memory():
    lib = seed_library(1)
    lib.notes_of("fam_00")[0].placed_step = 2
    fmt = FakeFormat(family=lambda p, i, m: ("cut", "length"))
    stats, carry = distill(fake_llm(fmt), lib, 2, {}, budget_s=60.0, workers=1,
                           max_tokens=8000)
    assert stats["failed"] == 1 and carry == {}
    assert lib.families["fam_00"].memory == "Memory of method 0."
    assert lib.families["fam_00"].memory_error == {"step": 2, "why": "truncated"}


def test_boundary_failure_parks_notes_and_reports_health(tmp_path):
    loop, fmt = make_loop(fmt=FakeFormat(merge=lambda p, i, m: ("", "length")),
                          out_dir=str(tmp_path))
    p = make_state("p0", -1.5)
    for j in range(3):
        child = make_state("c%d" % j, -1.5 + 0.01 * (j + 1), code="```python\nx=%d\n```" % j)
        loop.record(0, 0, p, None, improving(child, 1.5 - 0.01 * (j + 1)))
    loop.record(0, 0, p, None, failed_outcome())
    loop.end_of_step(0)
    loop.boundary(0, archived(loop))
    assert loop.library.families == {} and len(loop.pending) == 3
    assert loop.health["status"] == "failed" and loop.failed_hashes
    log = [json.loads(x) for x in open(tmp_path / "memo" / "merge_log.jsonl")]
    assert log[0]["reason"] and re.match(r"finish_length", log[0]["reason"][0])


def test_token_count_failure_keeps_the_library(capsys):
    class BrokenCount(FakeClient):
        def count_tokens(self, text):
            raise ConnectionError("tokenize endpoint down")

    lib = seed_library(2)
    res = merge_rounds(fake_llm(FakeFormat(), BrokenCount()), lib, notes(5), 2, settings(), [])
    assert res.stop == "count_error" and len(res.queue) == 5 and len(lib.families) == 2
    assert "memo_merge_count_failed" in capsys.readouterr().err
