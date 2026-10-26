"""One global merge call: render, ask, and validate the reply against its guards."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from ttt.memo import text as T
from ttt.memo.library import MergeGroup, label_key, normalise_groups

UNKNOWN_ID_CAP = 4
_ZERO_FIRST = "zero_absorption_first"


@dataclass
class MergeReply:
    """A validated merge partition, or the failure kind that refused it."""

    groups: Optional[list] = None
    keep_pending: list = field(default_factory=list)
    dropped: list = field(default_factory=list)
    kind: Optional[str] = None


class _Malformed(ValueError):
    """The reply parsed as JSON but not into the expected types."""


def _ids(obj: dict) -> list:
    out = []
    for key in ("ids", "existing_ids", "new_ids"):
        v = obj.get(key)
        if not v:
            continue
        if not isinstance(v, list):
            raise _Malformed(key)
        out += [str(x).strip().upper() for x in v]
    return out


def _text(obj: dict, key: str) -> str:
    v = obj.get(key) or ""
    if not isinstance(v, str):
        raise _Malformed(key)
    return v.strip()


def _route(got: list, eid: dict, nid: dict) -> tuple:
    """Groups routed by id prefix, plus the ids seen and the invented ones."""
    out, sawe, sawn, dropped = [], [], [], []
    for f in got:
        if not isinstance(f, dict):
            return None, sawe, sawn, dropped
        es, ns = [], []
        for k in _ids(f):
            if k in eid:
                es.append(eid[k])
                sawe.append(k)
            elif k in nid:
                ns.append(nid[k])
                sawn.append(k)
            else:
                dropped.append(k)
        if es or ns:
            out.append(MergeGroup(e=es, n=ns, reuse=len(es) == 1 and not ns,
                                  closest=str(f.get("closest") or "").strip().upper(),
                                  label=_text(f, "label"), direction=_text(f, "direction")))
    return out, sawe, sawn, dropped


def _first_only(groups: list) -> list:
    seen_e, seen_n, out = set(), set(), []
    for g in groups:
        g.e = [c for c in g.e if not (c in seen_e or seen_e.add(c))]
        g.n = [s for s in g.n if not (s in seen_n or seen_n.add(s))]
        if g.e or g.n:
            out.append(g)
    return out


def _guards(out: list, fams: list, notes: list) -> Optional[str]:
    if not out:
        return "zero_families"
    if len(fams) >= 2 and len(out) < 2:
        return "collapse"
    if any(not g.reuse and not g.direction for g in out):
        return "no_direction"
    known = {label_key(c[1]) for c in fams}
    if fams and notes and not any(g.n and (g.e or label_key(g.label) in known)
                                  for g in normalise_groups(out)):
        return _ZERO_FIRST
    return None


def parse_reply(reply: str, finish: Optional[str], fams: list, notes: list) -> MergeReply:
    """Validate one merge reply; fams = [(fid, label, direction)], notes = Note list."""
    eid = {"E%d" % i: c[0] for i, c in enumerate(fams)}
    nid = {"N%d" % i: n.sig for i, n in enumerate(notes)}
    m = re.search(r"\{.*\}", reply or "", re.S)
    if finish == "length":
        return MergeReply(kind="finish_length")
    if not m:
        return MergeReply(kind="no_json_stop")
    try:
        obj = json.loads(m.group(0)) or {}
        got = obj.get("groups") or obj.get("families") or []
        if not isinstance(got, list):
            raise _Malformed("groups")
        out, sawe, sawn, dropped = _route(got, eid, nid)
    except json.JSONDecodeError:
        return MergeReply(kind="bad_json")
    except _Malformed:
        return MergeReply(kind="malformed_reply")
    if out is None:
        return MergeReply(kind="not_object")
    if len(dropped) > UNKNOWN_ID_CAP:
        return MergeReply(kind="unknown_ids_over_cap", dropped=dropped)
    unplaced = sorted(set(nid) - set(sawn))
    if len(set(sawe)) < len(sawe) or len(set(sawn)) < len(sawn):
        out = _first_only(out)
    listed = {eid[k] for k in sawe}
    out += [MergeGroup(e=[c[0]], n=[], reuse=True) for c in fams if c[0] not in listed]
    kind = _guards(out, fams, notes)
    if kind is not None:
        return MergeReply(kind=kind, dropped=dropped)
    return MergeReply(groups=out, keep_pending=[nid[k] for k in unplaced], dropped=dropped)


def merge_call(call: Callable[[str], tuple], fams: list, exemplars: dict, notes: list,
               shape: tuple, can_retry: Callable[[], bool]) -> MergeReply:
    """Ask for a partition; ask once more, note by note, if nothing was absorbed."""
    k, chars = shape
    reply, finish = call(T.merge_question(fams, exemplars, notes, k, chars, retry=False))
    rep = parse_reply(reply, finish, fams, notes)
    if rep.kind != _ZERO_FIRST:
        return rep
    if not can_retry():
        return MergeReply(kind="zero_absorption_no_room", dropped=rep.dropped)
    reply, finish = call(T.merge_question(fams, exemplars, notes, k, chars, retry=True))
    rep = parse_reply(reply, finish, fams, notes)
    if rep.kind == _ZERO_FIRST:
        rep.kind = "zero_absorption"
    return rep
