import dataclasses
import json
import os

import numpy as np
import pytest

from ttt.archive import Archive, CodeStore, ConstructionStore, code_hash, construction_hash
from ttt.archive import store as store_module
from ttt.archive.puct import rank_prior, value_scale
from ttt.config import Search
from ttt.types import State


class StubTask:
    name = "stub"
    maximize = True
    entrypoint = "run"

    def __init__(self, min_len=1, max_len=100):
        self.min_len, self.max_len = min_len, max_len
        self.refreshed = []

    def initial_state(self, rng):
        return State(id="ignored", timestep=-1, value=0.1, code="",
                     construction=[float(rng.random())] * 4)

    def refresh_initial(self, state, rng):
        self.refreshed.append(state.id)
        state.construction = [float(rng.random())] * 5
        state.value = 0.1 + state.construction[0] / 100

    def admissible(self, construction):
        return self.min_len <= len(construction) <= self.max_len


def make(tmp_path, **search):
    search.setdefault("parents_per_step", 2)
    search.setdefault("seed", 7)
    return Archive(StubTask(), Search(**search), ConstructionStore(tmp_path / "c"),
                   CodeStore(tmp_path / "code"))


def child(cid, value, construction=None, code=""):
    return State(id=cid, timestep=0, value=value, code=code,
                 construction=construction if construction is not None else [value, 1.0])


def test_prior_and_scale():
    p = rank_prior(np.array([3.0, 1.0, 2.0]))
    assert np.allclose(p, np.array([3, 1, 2]) / 6)
    assert value_scale(np.array([1.0, 4.0, 2.0]), np.array([False, True, True])) == 2.0
    assert value_scale(np.array([1.0, 1.0]), None) == 1e-6
    assert value_scale(np.array([]), None) == 1.0


def test_initial_states_seeded_and_stored(tmp_path):
    a, b = make(tmp_path / "a"), make(tmp_path / "b")
    assert [s.id for s in a.states] == ["init-0", "init-1"]
    assert [s.construction for s in a.states] == [s.construction for s in b.states]
    for s in a.states:
        assert s.construction_hash == construction_hash(s.construction)
        assert a.store.get(s.construction_hash) == s.construction


def test_unseeded_archives_differ(tmp_path):
    a, b = make(tmp_path / "a", seed=None), make(tmp_path / "b", seed=None)
    assert a.states[0].construction != b.states[0].construction


def test_scores_follow_formula(tmp_path):
    arc = make(tmp_path, parents_per_step=1)
    p = arc.states[0]
    assert arc.update(child("c1", 0.9), p)
    assert arc.update(child("c2", 0.3), p)
    arc.record_failure(p)
    states = arc.states
    vals = np.array([s.value for s in states])
    non_initial = np.array([s.id not in arc.initial_ids for s in states])
    scale = vals[non_initial].max() - vals[non_initial].min()
    prior = rank_prior(vals)
    sqrt_t = np.sqrt(1 + arc.total_visits)
    n = {"init-0": 3}
    q = {"init-0": 0.9}
    expect = []
    for i, s in enumerate(states):
        visits = n.get(s.id, 0)
        qi = q[s.id] if visits else vals[i]
        expect.append((qi + scale * prior[i] * sqrt_t / (1 + visits), vals[i], s.id))
    best = max(expect, key=lambda e: (e[0], e[1]))[2]
    assert [s.id for s in arc.sample(1)] == [best]


def test_lineage_disjoint_picks(tmp_path):
    arc = make(tmp_path, parents_per_step=2)
    root = arc.states[0]
    b = child("b", 5.0)
    assert arc.update(b, root)
    assert arc.update(child("c", 4.0), b)
    scores = {s.id: s.value for s in arc.states}
    assert scores["b"] > scores["c"] > scores["init-1"]
    picks = [s.id for s in arc.sample(3)]
    assert picks == ["init-0", "init-1"], "init-0 (Q = best child 5.0) blocks descendants b and c"
    lone = make(tmp_path / "lone", parents_per_step=1)
    top = lone.states[0]
    assert lone.update(child("b", 5.0), top)
    lone.record_failure(top)
    lone.record_failure(top)
    assert [s.id for s in lone.sample(2)] == ["b"], "b blocks its ancestor"


def test_siblings_may_both_be_picked(tmp_path):
    arc = make(tmp_path, parents_per_step=1)
    root = arc.states[0]
    assert arc.update(child("x", 5.0), root)
    assert arc.update(child("y", 4.5), root)
    assert [s.id for s in arc.sample(2)] == ["x", "y"]


def test_gates_and_stats(tmp_path):
    arc = make(tmp_path, parents_per_step=1)
    arc.task.min_len = 2
    p = arc.states[0]
    assert not arc.update(State(id="none", timestep=0, value=None, code="", construction=[1.0]), p)
    assert arc.total_visits == 0
    assert not arc.update(child("short", 0.5, construction=[0.5]), p)
    assert arc.total_visits == 1 and "short" not in arc
    assert arc.update(child("ok", 0.5, construction=[0.0, 2.0]), p)
    assert not arc.update(child("dup", 0.7, construction=[-0.0, 2.0]), p)
    assert arc.update(child("code1", 0.6, construction=[], code="A"), p)
    assert not arc.update(child("code2", 0.6, construction=[], code="A"), p)
    assert arc.total_visits == 5
    ok = next(s for s in arc.states if s.id == "ok")
    assert ok.parents == [{"id": p.id, "timestep": -1, "construction_hash": p.construction_hash}]
    assert ok.parent_values == [p.value]
    assert ok.construction_hash in arc.store


def test_failure_counts_every_ancestor(tmp_path):
    arc = make(tmp_path, parents_per_step=1)
    root = arc.states[0]
    b = child("b", 1.0)
    arc.update(b, root)
    arc.record_failure(b)
    sd = arc.state_dict()
    assert sd["n"] == {"init-0": 2, "b": 1} and sd["T"] == 2


def test_flush_keeps_topk_per_parent(tmp_path):
    arc = make(tmp_path, parents_per_step=1, topk_children=2)
    p = arc.states[0]
    for i, v in enumerate([0.2, 0.9, 0.5, 0.7]):
        arc.update(child(f"c{i}", v), p)
    arc.flush(0)
    assert [s.id for s in arc.states] == ["init-0", "c1", "c3"]
    assert arc.step == 0


def test_eviction_keeps_initial_states(tmp_path):
    arc = make(tmp_path, parents_per_step=2, archive_capacity=3, topk_children=0)
    p = arc.states[1]
    for i, v in enumerate([0.5, 0.9, 0.7]):
        arc.update(child(f"c{i}", v), p)
    arc.flush(0)
    assert [s.id for s in arc.states] == ["init-0", "init-1", "c1"]


def test_parent_refs_pin_the_refreshed_construction(tmp_path):
    arc = make(tmp_path, parents_per_step=1)
    first = arc.sample(1)[0]
    h1 = first.construction_hash
    arc.update(child("a", 0.0, [0.0, 3.0]), first)
    second = arc.sample(1)[0]
    assert second.id == first.id and second.construction_hash != h1
    arc.update(child("b", 0.0, [0.0, 4.0]), second)
    refs = {s.id: s.parents[0]["construction_hash"] for s in arc.states if s.parents}
    assert refs == {"a": h1, "b": second.construction_hash}


def test_picked_initial_states_are_refreshed(tmp_path):
    arc = make(tmp_path, parents_per_step=2)
    before = {s.id: (s.construction, s.construction_hash) for s in arc.states}
    picked = arc.sample(2)
    assert {s.id for s in picked} == {"init-0", "init-1"}
    for s in picked:
        assert s.construction != before[s.id][0]
        assert s.construction_hash == construction_hash(s.construction)
        assert arc.store.get(s.construction_hash) == s.construction
    p = arc.states[0]
    arc.update(child("c", 9.0), p)
    arc.task.refreshed.clear()
    arc.sample(1)
    assert arc.task.refreshed == []


def test_state_dict_round_trip(tmp_path):
    arc = make(tmp_path, parents_per_step=2)
    for step in range(3):
        parents = arc.sample(2)
        for i, p in enumerate(parents):
            arc.update(child(f"s{step}p{i}a", float(step + i), [step, i, 1.0]), p)
            arc.update(child(f"s{step}p{i}b", float(step - i), [step, i, 2.0]), p)
            arc.record_failure(p)
        arc.flush(step)
    sd = json.loads(json.dumps(arc.state_dict()))
    assert all("construction" not in s and "code" not in s for s in sd["states"])
    assert all(arc.codes.get(s["code_sha256"]) == "" for s in sd["states"])
    back = Archive.from_state_dict(sd, task=arc.task, search=arc.search, store=arc.store,
                                   codes=arc.codes)
    assert back.state_dict() == arc.state_dict()
    assert [s.construction for s in back.states] == [s.construction for s in arc.states]
    assert [s.id for s in back.sample(2)] == [s.id for s in arc.sample(2)]
    assert back.rng.random() == arc.rng.random()


def test_state_dict_requires_stored_constructions(tmp_path):
    arc = make(tmp_path, parents_per_step=1)
    arc.states[0].construction_hash = None
    with pytest.raises(ValueError):
        arc.state_dict()


def test_store_round_trip_and_corruption(tmp_path):
    store = ConstructionStore(tmp_path)
    h = store.put([1.0, 2.5, -3.0])
    assert h == construction_hash([1.0, 2.5, -3.0])
    assert store.path(h) == tmp_path / h[:2] / f"{h}.npy"
    assert store.put([1.0, 2.5, -3.0]) == h
    assert store.get(h) == [1.0, 2.5, -3.0]
    np.save(store.path(h), np.array([9.0]))
    with pytest.raises(ValueError):
        store.get(h)
    assert not list(tmp_path.rglob("*.tmp"))


def test_programs_are_stored_by_reference(tmp_path, monkeypatch):
    arc = make(tmp_path, parents_per_step=1)
    p = arc.states[0]
    assert arc.update(child("c", 0.5, code="```python\nx = 1\n```"), p)
    arc.flush(0)

    def no_asdict(obj):
        raise AssertionError("state_dict must not deep-copy constructions through asdict")

    monkeypatch.setattr(dataclasses, "asdict", no_asdict)
    sd = arc.state_dict()
    rec = next(s for s in sd["states"] if s["id"] == "c")
    assert rec["code_sha256"] == code_hash("```python\nx = 1\n```") and "x = 1" not in json.dumps(sd)
    arc.codes.path(rec["code_sha256"]).write_text("```python\nx = 2\n```")
    with pytest.raises(ValueError, match="corrupt"):
        Archive.from_state_dict(sd, task=arc.task, search=arc.search, store=arc.store, codes=arc.codes)


def test_code_store_round_trip_and_corruption(tmp_path):
    store = CodeStore(tmp_path)
    text = "x = 'caf\u00e9'  # lone \ud800 surrogate\n"
    h = store.put(text)
    assert h == code_hash(text) and store.path(h) == tmp_path / h[:2] / f"{h}.py"
    assert store.put(text) == h and store.get(h) == text and h in store
    store.path(h).write_bytes(b"other")
    with pytest.raises(ValueError, match="corrupt"):
        store.get(h)
    with pytest.raises(FileNotFoundError):
        store.get("ab" * 32)
    assert not list(tmp_path.rglob("*.tmp"))


def test_stores_fsync_new_files_once(tmp_path, monkeypatch):
    synced = []
    real_fsync = os.fsync
    monkeypatch.setattr(store_module.os, "fsync", lambda fd: synced.append(fd) or real_fsync(fd))
    constructions, codes = ConstructionStore(tmp_path / "c"), CodeStore(tmp_path / "p")
    constructions.put([1.0, 2.0])
    codes.put("y = 2")
    assert len(synced) == 2
    constructions.put([1.0, 2.0])
    codes.put("y = 2")
    assert len(synced) == 2, "a stored file is never rewritten"
