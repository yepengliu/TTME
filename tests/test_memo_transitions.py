"""Transition recording, archive-based channels, step-best reserve, refused children, pruning."""
import random

from test_memo_fakes import FakeTask, archived, improving, make_loop, make_state
from ttt.config import MemoConfig
from ttt.memo.mining import expand_notes, mine, split_note
from ttt.memo.transitions import prune, select


def child(sid, raw, maximize=False, n=1500, code=None):
    value = raw if maximize else -raw
    return make_state(sid, value, code=code or "```python\nprint(%r)\n```" % sid,
                      construction=[0.0] * n)


def feed(loop, step, scores, maximize=False):
    """scores: {parent index: [child raw scores]}; every child beats its parent."""
    for gi, raws in scores.items():
        parent = make_state("p%d" % gi, -9.0)
        for j, raw in enumerate(raws):
            loop.record(step, gi, parent, None,
                        improving(child("c%d_%d_%d" % (step, gi, j), raw, maximize), raw))


def pick(loop, step, ids=None, want=6, reserve=2):
    picked, diag = select(loop.pool, loop.codes, loop.seen,
                          archived(loop) if ids is None else ids, want=want, step=step,
                          reserve=reserve, maximize=loop.task.maximize, window=loop.refused_window)
    return [(t.child_id, refused) for t, refused in picked], diag


def ids_of(picked):
    return [c for c, _refused in picked]


def test_step_best_reserve_takes_the_best_child_of_the_newest_steps():
    loop, _ = make_loop()
    feed(loop, 0, {0: [5.0, 3.0], 1: [4.0]})
    feed(loop, 1, {0: [2.5, 2.9], 1: [2.7]})
    got, diag = pick(loop, 1, want=4, reserve=2)
    assert ids_of(got)[:2] == ["c1_0_0", "c0_0_1"] and diag["stepbest_kept"] == 2
    assert len(set(ids_of(got))) == 4


def test_step_best_follows_the_task_direction():
    loop, _ = make_loop(task=FakeTask(maximize=True))
    feed(loop, 0, {0: [5.0, 3.0], 1: [4.0]}, maximize=True)
    got, _ = pick(loop, 0, want=2, reserve=1)
    assert ids_of(got)[0] == "c0_0_0"


def test_fill_is_round_robin_with_a_rotating_start():
    loop, _ = make_loop()
    feed(loop, 3, {0: [1.0, 1.1, 1.2], 1: [1.0, 1.1, 1.2], 2: [1.0, 1.1, 1.2]})
    got, _ = pick(loop, 1, want=5, reserve=0)
    assert ids_of(got) == ["c3_1_2", "c3_2_2", "c3_0_2", "c3_1_1", "c3_2_1"]


def test_same_diff_is_mined_once_per_parent_and_never_twice():
    loop, _ = make_loop()
    p0, p1 = make_state("p0", -9.0), make_state("p1", -9.0)
    same = "```python\nprint('same')\n```"
    for parent, sid in ((p0, "a"), (p0, "b"), (p1, "c")):
        loop.record(0, 0, parent, None, improving(child(sid, 1.0, code=same), 1.0))
    assert sorted(ids_of(pick(loop, 0, reserve=0)[0])) == ["a", "c"]
    loop.seen.update(t.sig for t in loop.pool if t.child_id == "a")
    assert ids_of(pick(loop, 0, reserve=0)[0]) == ["c"]


def test_archived_channel_needs_parent_and_child_still_archived():
    loop, _ = make_loop()
    feed(loop, 0, {0: [1.0, 1.1], 1: [1.2]})
    got, _ = pick(loop, 0, ids=archived(loop, drop={"c0_0_1"}), reserve=0)
    assert sorted(ids_of(got)) == ["c0_0_0", "c0_1_0"]
    got, _ = pick(loop, 0, ids=archived(loop, drop={"p0"}), reserve=0)
    assert ids_of(got) == ["c0_1_0"]
    pool, codes = prune(loop.pool, loop.codes, loop.seen, archived(loop, drop={"p0"}),
                        loop.refused_window)
    assert [t.child_id for t in pool] == ["c0_1_0"] and len(codes) == 2


def refused_case(loop):
    p = make_state("p0", -9.0)
    loop.record(0, 0, p, None, improving(child("short", 1.0, n=500), 1.0, admitted=False))
    loop.record(0, 0, p, None, improving(child("inside", 1.0, n=5000), 1.0, admitted=False))
    loop.record(0, 0, p, None, improving(child("ok", 2.0), 2.0))
    return frozenset({"p0", "ok"})


def test_refused_children_outside_the_window_are_mined_with_the_banner():
    loop, fmt = make_loop()
    got, diag = pick(loop, 0, ids=refused_case(loop))
    assert got == [("ok", False), ("short", True)] and diag["refused_kept"] == 1
    refused = [t for t in loop.pool if t.child_id == "short"][0]
    mined = mine(loop.llm, [(refused, True)], loop.codes, loop.window, loop.aux,
                 loop.cfg.budgets)
    assert len(mined) == 1
    assert ("ARCHIVE-REFUSED: its construction length 500 is outside the admission window "
            "[1000, 100000]") in fmt.of("mine")[0]["prompt"]
    pool, _ = prune(loop.pool, loop.codes, loop.seen, frozenset({"p0", "ok"}),
                    loop.refused_window)
    assert sorted(t.child_id for t in pool) == ["ok", "short"]


def test_refused_channel_off_and_other_windows():
    off, _ = make_loop(cfg=MemoConfig(include_refused=False))
    assert ids_of(pick(off, 0, ids=refused_case(off))[0]) == ["ok"]
    erdos, fmt = make_loop(task=FakeTask(name="erdos", length_window=(1, 1000)))
    p = make_state("p0", -9.0)
    erdos.record(0, 0, p, None, improving(child("long", 1.0, n=1500), 1.0, admitted=False))
    got, _ = pick(erdos, 0, ids=frozenset({"p0"}))
    assert got == [("long", True)]
    mine(erdos.llm, [(erdos.pool[0], True)], erdos.codes, erdos.window, erdos.aux,
         erdos.cfg.budgets)
    assert "its construction length 1500 is outside the admission window [1, 1000]" in \
        fmt.of("mine")[0]["prompt"]


def test_split_notes_get_distinct_sigs():
    assert split_note("one\n===\ntwo\n===\nthree\n===\nfour") == ["one", "two", "three\n\nfour"]
    loop, _ = make_loop()
    feed(loop, 0, {0: [1.0]})
    t = loop.pool[0]
    t.sig = "abc"
    notes, n_split = expand_notes([(t, "first\n===\nsecond", "stop")], 4)
    assert [n.sig for n in notes] == ["abc#0", "abc#1"] and n_split == 1
    assert {n.step for n in notes} == {4} and notes[0].rollout_step == 0


def test_pruning_never_changes_a_later_selection():
    rng = random.Random(7)
    full, _ = make_loop()
    ids = set()
    for step in range(8):
        scores = {gi: [rng.uniform(1, 2) for _ in range(rng.randint(0, 4))] for gi in range(3)}
        feed(full, step, scores)
        ids |= {t.parent_id for t in full.pool} | {t.child_id for t in full.pool if t.step == step}
        ids -= {x for x in sorted(ids) if rng.random() < 0.2 and x[0] == "c"}
        if step % 2:
            continue
        picked, _ = select(full.pool, full.codes, full.seen, frozenset(ids), want=5, step=step,
                           reserve=2, maximize=False, window=full.refused_window)
        full.seen.update(t.sig for t, _r in picked[::2])
        pruned, _ = make_loop()
        pruned.pool, pruned.codes = prune(full.pool, full.codes, full.seen, frozenset(ids),
                                          full.refused_window)
        pruned.seen = set(full.seen)
        later = frozenset(ids - {x for x in sorted(ids) if rng.random() < 0.3 and x[0] == "c"})
        a = select(full.pool, full.codes, full.seen, later, want=5, step=step + 2, reserve=2,
                   maximize=False, window=full.refused_window)[0]
        b = select(pruned.pool, pruned.codes, pruned.seen, later, want=5, step=step + 2,
                   reserve=2, maximize=False, window=pruned.refused_window)[0]
        assert [(t.sig, r) for t, r in a] == [(t.sig, r) for t, r in b]
