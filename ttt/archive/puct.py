"""PUCT archive: parent selection, child admission, per-parent top-k and capacity eviction.

score(i) = Q(i) + c * scale * P(i) * sqrt(1 + T) / (1 + n(i)), where P is a rank-linear
prior over all states, scale is the value range of the non-initial states, Q(i) is the best
direct-child value once i has been visited, n(i) counts rollouts under i and its descendants,
and T counts all rollouts.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
from typing import Any, Optional

import numpy as np

from ttt.archive.store import CodeStore, ConstructionStore, code_hash
from ttt.config import Search
from ttt.types import State, Task


def _value(state: State) -> float:
    return float(state.value) if state.value is not None else float("-inf")


def dedup_key(state: State) -> Optional[tuple[str, str]]:
    """Exact-duplicate key: the construction's float64 bytes, else the program text."""
    if state.construction:
        arr = np.asarray(state.construction, dtype=np.float64) + 0.0
        return ("construction", hashlib.sha256(arr.tobytes()).hexdigest())
    if state.code:
        return ("code", state.code)
    return None


def rank_prior(values: np.ndarray) -> np.ndarray:
    """Rank-linear prior: the best of N states gets weight N, the worst gets 1."""
    if values.size == 0:
        return np.array([])
    ranks = np.argsort(np.argsort(-values))
    weights = (len(values) - ranks).astype(np.float64)
    return weights / weights.sum()


def value_scale(values: np.ndarray, mask: Optional[np.ndarray]) -> float:
    """Value range over the masked states, floored at 1e-6; 1.0 when there are none."""
    if values.size == 0:
        return 1.0
    v = values[mask] if mask is not None else values
    return float(max(np.max(v) - np.min(v), 1e-6)) if v.size > 0 else 1.0


def _ancestor_ids(state: State) -> list[str]:
    return [state.id] + [str(p["id"]) for p in (state.parents or []) if p.get("id")]


class Archive:
    """The PUCT sampler and the buffer of admitted states it selects parents from."""

    def __init__(self, task: Task, search: Search, store: ConstructionStore, codes: CodeStore,
                 *, seed_initial: bool = True) -> None:
        self.task = task
        self.search = search
        self.store = store
        self.codes = codes
        self.rng = np.random.default_rng(search.seed)
        self.step = -1
        self._states: list[State] = []
        self._initial_ids: list[str] = []
        self._keys: dict[str, Optional[tuple[str, str]]] = {}
        self._n: dict[str, int] = {}
        self._m: dict[str, float] = {}
        self._T = 0
        if seed_initial:
            for k in range(search.parents_per_step):
                state = task.initial_state(self.rng)
                state.id = f"init-{k}"
                self._remember(state)
                self._states.append(state)
                self._initial_ids.append(state.id)

    @property
    def states(self) -> list[State]:
        """Archived states in buffer order."""
        return list(self._states)

    @property
    def initial_ids(self) -> list[str]:
        """Ids of the initial states, which are never evicted."""
        return list(self._initial_ids)

    @property
    def total_visits(self) -> int:
        """Rollouts recorded so far (T)."""
        return self._T

    def __contains__(self, state_id: str) -> bool:
        return state_id in self._keys

    def __len__(self) -> int:
        return len(self._states)

    def _remember(self, state: State) -> None:
        # Stored now, so every checkpoint that later references the state finds its files.
        if state.construction is not None:
            state.construction_hash = self.store.put(state.construction)
        self.codes.put(state.code)
        self._keys[state.id] = dedup_key(state)

    def _visit(self, parent: State) -> None:
        for aid in _ancestor_ids(parent):
            self._n[aid] = self._n.get(aid, 0) + 1
        self._T += 1

    def sample(self, n: int) -> list[State]:
        """Pick n parents by PUCT score (lineage-disjoint when n > 1) and refresh picked initial states."""
        if not self._states:
            raise RuntimeError("cannot sample from an empty archive")
        initial = set(self._initial_ids)
        values = np.array([_value(s) for s in self._states])
        non_initial = np.array([s.id not in initial for s in self._states])
        scale = value_scale(values, non_initial if non_initial.any() else None)
        prior = rank_prior(values)
        sqrt_t = np.sqrt(1.0 + self._T)
        scored = []
        for i, s in enumerate(self._states):
            visits = self._n.get(s.id, 0)
            q = self._m.get(s.id, values[i]) if visits > 0 else values[i]
            bonus = self.search.puct_c * scale * prior[i] * sqrt_t / (1.0 + visits)
            scored.append((q + bonus, values[i], s))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        picked = self._disjoint(scored, n) if n > 1 else [s for _, _, s in scored[:n]]
        for s in picked:
            if s.id in initial:
                self.task.refresh_initial(s, self.rng)
                self._remember(s)
        return picked

    def _disjoint(self, scored: list, n: int) -> list[State]:
        children = self._children_map()
        picked: list[State] = []
        blocked: set[str] = set()
        for _, _, s in scored:
            if s.id in blocked:
                continue
            picked.append(s)
            blocked |= self._lineage(s, children)
            if len(picked) >= n:
                break
        return picked

    def _children_map(self) -> dict[str, set[str]]:
        children: dict[str, set[str]] = {}
        for s in self._states:
            for p in s.parents or []:
                if p.get("id"):
                    children.setdefault(str(p["id"]), set()).add(s.id)
        return children

    def _lineage(self, state: State, children: dict[str, set[str]]) -> set[str]:
        lineage = set(_ancestor_ids(state))
        queue, seen = [state.id], {state.id}
        while queue:
            sid = queue.pop(0)
            for cid in children.get(sid, ()):
                if cid not in seen:
                    seen.add(cid)
                    lineage.add(cid)
                    queue.append(cid)
        return lineage

    def update(self, child: State, parent: State) -> bool:
        """Record a scored rollout of parent; returns True if the child was admitted."""
        if child.value is None:
            return False
        y = float(child.value)
        self._m[parent.id] = max(self._m.get(parent.id, y), y)
        self._visit(parent)
        if child.construction and not self.task.admissible(child.construction):
            return False
        key = dedup_key(child)
        if key is not None and key in {self._keys[s.id] for s in self._states}:
            return False
        child.parent_values = [parent.value] + parent.parent_values if parent.value is not None else []
        ref = {"id": parent.id, "timestep": parent.timestep}
        if parent.construction_hash:
            ref["construction_hash"] = parent.construction_hash
        child.parents = [ref] + list(parent.parents)
        self._remember(child)
        self._states.append(child)
        return True

    def record_failure(self, parent: State) -> None:
        """Count a rollout of parent that produced no scored child."""
        self._visit(parent)

    def flush(self, step: int) -> None:
        """End of step: keep the top-k children per immediate parent, then evict down to capacity."""
        k = self.search.topk_children
        if k > 0:
            by_parent: dict[str, list[State]] = {}
            roots: list[State] = []
            for s in self._states:
                pid = s.parents[0]["id"] if s.parents else None
                if pid:
                    by_parent.setdefault(pid, []).append(s)
                else:
                    roots.append(s)
            kept: list[State] = []
            for group in by_parent.values():
                group.sort(key=_value, reverse=True)
                kept.extend(group[:k])
            self._states = roots + kept
        self._evict()
        self._keys = {s.id: self._keys[s.id] for s in self._states}
        self.step = step

    def _evict(self) -> None:
        cap = self.search.archive_capacity
        if len(self._states) <= cap:
            return
        by_value = list(np.argsort([_value(s) for s in self._states])[::-1])
        initial = set(self._initial_ids)
        keep = {i for i, s in enumerate(self._states) if s.id in initial}
        for i in by_value:
            if len(keep) >= cap:
                break
            keep.add(int(i))
        self._states = [self._states[i] for i in sorted(keep)]

    def state_dict(self) -> dict[str, Any]:
        """Serializable archive state; constructions and programs are referenced by store hash."""
        return {
            "step": self.step,
            "states": [_state_record(s) for s in self._states],
            "initial_ids": list(self._initial_ids),
            "n": dict(self._n),
            "m": dict(self._m),
            "T": self._T,
            "rng": self.rng.bit_generator.state,
        }

    @classmethod
    def from_state_dict(cls, data: dict[str, Any], *, task: Task, search: Search,
                        store: ConstructionStore, codes: CodeStore) -> "Archive":
        """Rebuild an archive from state_dict(), loading constructions and programs from the stores."""
        archive = cls(task, search, store, codes, seed_initial=False)
        archive.step = int(data["step"])
        archive._states = [_state_from_record(r, store, codes) for r in data["states"]]
        archive._initial_ids = [str(i) for i in data["initial_ids"]]
        archive._keys = {s.id: dedup_key(s) for s in archive._states}
        archive._n = {str(k): int(v) for k, v in data["n"].items()}
        archive._m = {str(k): float(v) for k, v in data["m"].items()}
        archive._T = int(data["T"])
        archive.rng.bit_generator.state = data["rng"]
        missing = set(archive._initial_ids) - set(archive._keys)
        if missing:
            raise ValueError(f"archive checkpoint lost initial states {sorted(missing)}")
        return archive


_BY_REFERENCE = ("construction", "code")


def _state_record(state: State) -> dict[str, Any]:
    """Checkpoint record of a state, with store hashes in place of its construction and program."""
    has_construction = state.construction is not None
    if has_construction and not state.construction_hash:
        raise ValueError(f"state {state.id} has a construction that was never stored")
    rec = {f.name: copy.deepcopy(getattr(state, f.name)) for f in dataclasses.fields(State)
           if f.name not in _BY_REFERENCE}
    rec["has_construction"] = has_construction
    rec["code_sha256"] = code_hash(state.code)
    return rec


def _state_from_record(rec: dict[str, Any], store: ConstructionStore, codes: CodeStore) -> State:
    rec = dict(rec)
    has_construction = rec.pop("has_construction")
    code = codes.get(rec.pop("code_sha256"))
    construction = store.get(rec["construction_hash"]) if has_construction else None
    return State(construction=construction, code=code, **rec)
