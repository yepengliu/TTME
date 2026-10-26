"""The method library: families of mined notes, rebuilt from each merge reply."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from typing import Optional

from ttt.log import warn

EXEMPLARS_PER_FAMILY = 2


@dataclass
class Note:
    """One mined description of a parent-to-child change, with its provenance."""

    sig: str
    memo: str
    step: int
    rollout_step: Optional[int] = None
    parent_id: Optional[str] = None
    child_id: Optional[str] = None
    group: Optional[int] = None
    arm: Optional[str] = None
    root: Optional[str] = None
    pscore: Optional[float] = None
    cscore: Optional[float] = None
    placed_step: Optional[int] = None


@dataclass
class Variant:
    """The notes one merge group filed under a family, in that group's wording."""

    cid: str
    family: str
    label: str
    direction: Optional[str]
    created_step: int
    written_step: int
    notes: list = field(default_factory=list)


@dataclass
class Family:
    """One selectable method: its head, accumulated memory and outcome counts."""

    fid: str
    label: str = ""
    direction: Optional[str] = None
    created_step: int = 0
    written_step: int = 0
    variants: list = field(default_factory=list)
    memory: str = ""
    memory_sha: str = ""
    memory_step: Optional[int] = None
    memory_rounds: int = 0
    memory_error: Optional[dict] = None
    stats: dict = field(default_factory=dict)


@dataclass
class MergeGroup:
    """One group of a merge reply: existing family ids and new note sigs."""

    e: list
    n: list
    reuse: bool = False
    closest: str = ""
    label: str = ""
    direction: str = ""


def label_key(label: Optional[str]) -> str:
    """Whitespace- and case-normalised label used as label identity."""
    return re.sub(r"\s+", " ", label or "").strip().lower()


def cid_for(label: Optional[str]) -> str:
    """Slug plus short hash of a label, used as a family or variant id."""
    slug = re.sub(r"[^a-z0-9]+", "_", (label or "").lower()).strip("_")[:32]
    return "%s_%s" % (slug or "c", hashlib.sha1((label or "").encode("utf-8")).hexdigest()[:6])


def mint_id(base: str, taken: dict, key: str) -> str:
    """`base`, or base+N for the first id not owned by a different label key."""
    i, out = 1, base
    while out in taken and taken[out] != key:
        out = "%s%d" % (base, i)
        i += 1
    return out


def memory_sha(text: str) -> str:
    """Short fingerprint of a family memory."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def normalise_groups(groups: list) -> list:
    """Drop repeated ids (first occurrence wins) and groups left empty."""
    seen_e, seen_n, out = set(), set(), []
    for g in groups:
        e = [c for c in g.e if not (c in seen_e or seen_e.add(c))]
        n = [s for s in g.n if not (s in seen_n or seen_n.add(s))]
        if e or n:
            out.append(replace(g, e=e, n=n, reuse=bool(g.reuse) and len(e) == 1 and not n))
    return out


@dataclass
class Fusions:
    """Families fused away during one boundary: id and head label -> the absorbing family."""

    ids: dict = field(default_factory=dict)
    labels: dict = field(default_factory=dict)

    def add(self, gone: str, into: str, label: Optional[str]) -> None:
        """Record that family `gone` was fused into `into`."""
        self.ids[gone] = into
        if label_key(label):
            self.labels.setdefault(label_key(label), gone)

    def resolve(self, fid: str) -> str:
        """The family that now holds `fid`, following chains of fusions."""
        seen = set()
        while fid in self.ids and fid not in seen:
            seen.add(fid)
            fid = self.ids[fid]
        return fid


@dataclass
class _Rebuild:
    step: int
    reserved: set
    fid_key: dict
    fusions: Fusions
    families: dict = field(default_factory=dict)
    variants: dict = field(default_factory=dict)
    labelled: set = field(default_factory=set)
    fused: set = field(default_factory=set)
    tally: Counter = field(default_factory=Counter)


class Library:
    """Families and variants of the run's mined method notes."""

    def __init__(self) -> None:
        self.families: dict[str, Family] = {}
        self.variants: dict[str, Variant] = {}

    def selectable(self) -> list:
        """Families with a direction, in insertion order."""
        return [f for f in self.families.values() if (f.direction or "").strip()]

    def notes_of(self, fid: str) -> list:
        """All notes filed under a family, variant by variant."""
        fam = self.families[fid]
        return [n for cid in fam.variants for n in self.variants[cid].notes]

    def sha(self) -> str:
        """Fingerprint of everything Stage 1 can read from the library."""
        rows = sorted([f.fid, f.label, f.direction or "", f.memory_sha] for f in self.selectable())
        return hashlib.sha256(json.dumps(rows).encode("utf-8")).hexdigest()[:12]

    def did_index(self) -> dict:
        """Child state id -> the note describing the change that produced it."""
        out = {}
        for fid in self.families:
            for n in self.notes_of(fid):
                if n.child_id and n.memo:
                    out[n.child_id] = n.memo
        return out

    def ranked(self) -> list:
        """Selectable families, most recently fed first (the trim order of Stage 1)."""
        def key(f):
            notes = self.notes_of(f.fid)
            last = max((n.placed_step if n.placed_step is not None else n.step
                        for n in notes), default=f.created_step)
            return (-last, -len(notes), f.fid)
        return sorted(self.selectable(), key=key)

    def merge_view(self) -> tuple[list, dict]:
        """(entries, exemplars) shown to the merge call, oldest family first."""
        def born(fid):
            return (min(self.variants[c].created_step for c in self.families[fid].variants), fid)
        entries, exemplars = [], {}
        for fid in sorted((f for f in self.families if self.families[f].variants), key=born):
            fam = self.families[fid]
            first = self.variants[fam.variants[0]]
            text = fam.direction or first.direction
            if not text:
                continue
            entries.append((fid, fam.label or first.label, text))
            recs = sorted((n for n in self.notes_of(fid) if (n.memo or "").strip()),
                          key=lambda n: n.step or 0)
            exemplars[fid] = [n.memo for n in recs[-EXEMPLARS_PER_FAMILY:]]
        return entries, exemplars

    def add_counts(self, step: int, per_family: dict, groups: dict) -> None:
        """Fold one step's per-family outcome counts into the family stats."""
        for fid in sorted(set(per_family) | set(groups)):
            fam = self.families.get(fid)
            if fam is None:
                warn("memo_counts_unknown_family", step=step, family=fid)
                continue
            for k, v in (per_family.get(fid) or {}).items():
                fam.stats[k] = fam.stats.get(k, 0) + v
            fam.stats["parent_group_count"] = (fam.stats.get("parent_group_count", 0)
                                               + groups.get(fid, 0))
            fam.stats["last_step"] = step

    def apply_merge(self, groups: list, notes: list, step: int,
                    fusions: Optional[Fusions] = None) -> dict:
        """Rebuild the library from a merge partition, after this boundary's earlier `fusions`."""
        variants_of: dict[str, list] = {}
        for v in self.variants.values():
            variants_of.setdefault(v.family, []).append(v)
        by_label, fid_key = self._label_index(variants_of)
        rb = _Rebuild(step=step, reserved=set(self.variants), fid_key=fid_key,
                      fusions=fusions if fusions is not None else Fusions())
        note_of = {n.sig: n for n in notes}
        for g in normalise_groups(groups):
            self._place(g, rb, variants_of, by_label, note_of)
        for cid, v in self.variants.items():
            if cid not in rb.variants:
                rb.variants[cid] = v
                self._family_for(v.family, rb, [v]).variants.append(cid)
                rb.tally["carried"] += 1
        self._finish_heads(rb)
        record = self._record(rb, len(notes))
        if self.families and notes and not rb.tally["absorbed"]:
            warn("memo_merge_nothing_absorbed", step=step, notes=len(notes),
                 families=len(self.families))
        self.families, self.variants = rb.families, rb.variants
        return record

    def _label_index(self, variants_of: dict) -> tuple[dict, dict]:
        by_label: dict[str, set] = {}
        for fid, vs in variants_of.items():
            head = self.families[fid].label if fid in self.families else ""
            for k in [label_key(head)] + [label_key(v.label) for v in vs]:
                if k:
                    by_label.setdefault(k, set()).add(fid)
        unique = {k: next(iter(v)) for k, v in by_label.items() if len(v) == 1}
        fid_key = {fid: label_key((self.families[fid].label if fid in self.families else "")
                                  or (vs[0].label if vs else ""))
                   for fid, vs in variants_of.items()}
        return unique, fid_key

    def _family_born(self, fid: str, variants_of: dict) -> int:
        if fid in self.families:
            return self.families[fid].created_step
        return min((v.created_step for v in variants_of.get(fid, [])), default=0)

    def _target(self, g: MergeGroup, parent_fids: list, rb: _Rebuild,
                variants_of: dict, by_label: dict) -> str:
        if parent_fids:
            return min(parent_fids, key=lambda x: (self._family_born(x, variants_of), x))
        key = label_key(g.label)
        hit = by_label.get(key) or rb.fusions.labels.get(key)
        if hit:
            rb.tally["relabel_merged"] += 1
        else:
            hit = mint_id(cid_for(g.label or "method %s" % g.n[0][:8]), rb.fid_key, key)
        # A family fused away earlier in this boundary must not come back with its old
        # memory and counts, which the absorbing family already carries.
        fid = rb.fusions.resolve(hit)
        rb.tally["redirected"] += fid != hit
        return fid

    def _place(self, g: MergeGroup, rb: _Rebuild, variants_of: dict, by_label: dict,
               note_of: dict) -> None:
        parent_fids = [x for x in g.e if x in variants_of]
        if not parent_fids and not g.n:
            return
        parents = [v for x in parent_fids for v in variants_of[x]]
        fid = self._target(g, parent_fids, rb, variants_of, by_label)
        rb.fid_key.setdefault(fid, label_key(g.label))
        if g.n:
            if parents or fid in rb.families or fid in self.families:
                rb.tally["absorbed"] += 1
            else:
                rb.tally["solo_new"] += 1
                rb.tally["declared_closest"] += bool(g.closest)
        fam = self._family_for(fid, rb, parents)
        for x in parent_fids:
            if x != fid:
                self._fuse(fam, x, rb)
        if g.label and fid not in rb.labelled:
            fam.label, fam.direction = g.label, g.direction or None
            rb.labelled.add(fid)
        for v in parents:
            rb.variants[v.cid] = replace(v, family=fid)
            fam.variants.append(v.cid)
            rb.reserved.add(v.cid)
        rb.tally["reused"] += bool(g.reuse and parents)
        if g.n:
            self._new_variant(g, fid, fam, rb, note_of)

    def _family_for(self, fid: str, rb: _Rebuild, parents: list) -> Family:
        if fid in rb.families:
            return rb.families[fid]
        prev = self.families.get(fid)
        created = (min(v.created_step for v in parents) if parents
                   else prev.created_step if prev else rb.step)
        fam = Family(fid=fid, created_step=created, written_step=rb.step)
        if prev is not None:
            fam = replace(fam, memory=prev.memory, memory_sha=prev.memory_sha,
                          memory_step=prev.memory_step, memory_rounds=prev.memory_rounds,
                          memory_error=prev.memory_error, stats=dict(prev.stats))
        rb.families[fid] = fam
        return fam

    def _fuse(self, fam: Family, absorbed: str, rb: _Rebuild) -> None:
        """Carry an absorbed family's memory, counts and new notes into the surviving one."""
        if absorbed in rb.fused:
            return
        rb.fused.add(absorbed)
        prev = self.families.get(absorbed)
        rb.fusions.add(absorbed, fam.fid, prev.label if prev is not None else None)
        early = rb.families.pop(absorbed, None)
        if early is not None:
            # An earlier group of this reply filed notes under it by label.
            rb.labelled.discard(absorbed)
            for cid in early.variants:
                rb.variants[cid] = replace(rb.variants[cid], family=fam.fid)
                fam.variants.append(cid)
        if prev is None:
            return
        if prev.memory.strip():
            fam.memory = "\n\n".join([m for m in [fam.memory] if m.strip()] + [prev.memory])
            fam.memory_sha = memory_sha(fam.memory)
        for k, v in prev.stats.items():
            if k != "last_step" and isinstance(v, (int, float)):
                fam.stats[k] = fam.stats.get(k, 0) + v

    def _new_variant(self, g: MergeGroup, fid: str, fam: Family, rb: _Rebuild,
                     note_of: dict) -> None:
        base = cid_for(g.label or "method %s" % g.n[0][:8])
        cid, k = base, 1
        while cid in rb.variants or cid in rb.reserved:
            cid, k = "%s%d" % (base, k), k + 1
        rb.reserved.add(cid)
        notes = [replace(note_of[s], placed_step=rb.step) for s in g.n if s in note_of]
        rb.variants[cid] = Variant(cid=cid, family=fid, label=g.label or cid.replace("_", " "),
                                   direction=g.direction or None, created_step=rb.step,
                                   written_step=rb.step, notes=notes)
        fam.variants.append(cid)

    def _finish_heads(self, rb: _Rebuild) -> None:
        """Families no group relabelled keep their previous head (else the first variant's)."""
        for fid, fam in rb.families.items():
            if fid in rb.labelled:
                continue
            prev = self.families.get(fid)
            if prev is not None and (prev.direction or "").strip():
                fam.label, fam.direction = prev.label, prev.direction
            else:
                first = rb.variants[fam.variants[0]]
                fam.label, fam.direction = first.label, first.direction

    def _record(self, rb: _Rebuild, n_notes: int) -> dict:
        was = {cid: v.family for cid, v in self.variants.items()}
        partition = {}
        for fid, fam in rb.families.items():
            src = sorted({was[c] for c in fam.variants if c in was and was[c] != fid})
            if src:
                partition[fid] = src
        rec = {"families_in": len(set(was.values())), "families_out": len(rb.families),
               "variants_in": len(self.variants), "variants_out": len(rb.variants),
               "new_notes": n_notes, "fused": len(partition), "partition": partition}
        for k in ("reused", "carried", "absorbed", "solo_new", "declared_closest",
                  "relabel_merged", "redirected"):
            rec[k] = int(rb.tally[k])
        return rec

    def to_dict(self) -> dict:
        """JSON-serialisable form of the library."""
        return {"families": {k: asdict(v) for k, v in self.families.items()},
                "variants": {k: asdict(v) for k, v in self.variants.items()}}

    @classmethod
    def from_dict(cls, data: dict) -> "Library":
        """Inverse of to_dict."""
        lib = cls()
        lib.families = {k: Family(**v) for k, v in (data.get("families") or {}).items()}
        for k, v in (data.get("variants") or {}).items():
            notes = [Note(**n) for n in v.get("notes") or []]
            lib.variants[k] = Variant(**dict(v, notes=notes))
        return lib
