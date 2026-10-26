"""Prompt assembly for the memo route; instruction texts live in ttt/prompts."""
from __future__ import annotations

import difflib
import hashlib
import random
from typing import Optional

from ttt.prompts import load
from ttt.types import State, raw_score

MEMO_MAX_CHARS = 700
LISTING_HEAD_CHARS = 400
LINEAGE_DEPTH = 3
LINEAGE_NOTE_CHARS = 300
SCORE_HISTORY = 6
DIFF_MAX_LINES = 120
FULL_EXEMPLAR_CHARS = 320


def suffix(text: str) -> str:
    """The question suffix carrying one direction text."""
    return "\n\n" + text.strip() + "\n"


def control_suffix(card: str) -> str:
    """Suffix of the control card named `card`; the empty card "none" gives three newlines."""
    return suffix(load(f"cards/{card}.txt"))


def direction_id(fid: str) -> str:
    """Opaque per-family id shown in the injected direction."""
    return "D-" + hashlib.sha1(fid.encode("utf-8")).hexdigest()[:6]


def direction_suffix(fid: str, memo: str) -> str:
    """Suffix carrying one parent-specific memo."""
    body = load("memo/assigned_direction.txt").format(tag=direction_id(fid), body=memo.strip())
    return suffix(body)


def cap_memo(memo: str) -> str:
    """Cut a memo to MEMO_MAX_CHARS, at a sentence end when there is one."""
    if len(memo) <= MEMO_MAX_CHARS:
        return memo
    cut = memo.rfind(".", 0, MEMO_MAX_CHARS)
    return memo[:cut + 1] if cut > 40 else memo[:MEMO_MAX_CHARS]


def select_task(k: int, serial: bool, want: int) -> str:
    """Instruction text of call A for one (K=1) or several methods."""
    if k == 1:
        return load("memo/select_serial.txt" if serial else "memo/select.txt")
    name = "memo/select_k_serial.txt" if serial else "memo/select_k.txt"
    return load(name).replace("{k}", str(want))


def listing(fams: list, parent_id: str) -> str:
    """Family heads in a per-parent seeded order."""
    shown = sorted(fams, key=lambda f: f.fid)
    random.Random("listing:%s" % parent_id).shuffle(shown)
    return "\n".join("- id `%s`: %s" % (f.fid, " ".join((f.direction or "").split())
                                         [:LISTING_HEAD_CHARS]) for f in shown)


def _given(got: Optional[list]) -> str:
    if got and len(got) > 1:
        return load("memo/lineage_split.txt").format(
            methods=", ".join("`%s`" % g for g in got))
    if got:
        return load("memo/lineage_steered.txt").format(method=got[0])
    return load("memo/lineage_unsteered.txt")


def lineage_block(parent: State, maximize: bool, assigned: dict, did: dict) -> Optional[str]:
    """What the parent's recent ancestors were steered toward, newest first."""
    anc = list(parent.parents or [])[:LINEAGE_DEPTH]
    vals = list(parent.parent_values or [])[:LINEAGE_DEPTH]
    if not anc:
        return None
    here = raw_score(parent.value, maximize)
    rows = []
    for i, a in enumerate(anc):
        prev = raw_score(vals[i], maximize) if i < len(vals) else None
        nxt = here if i == 0 else (raw_score(vals[i - 1], maximize) if i - 1 < len(vals) else None)
        move = ""
        if prev is not None and nxt is not None:
            better = nxt > prev if maximize else nxt < prev
            move = "  score %.10f -> %.10f (%s)" % (prev, nxt, "better" if better else "no better")
        line = "- %d step(s) back: %s%s" % (i + 1, _given(assigned.get(a.get("id"))), move)
        child = parent.id if i == 0 else anc[i - 1].get("id")
        if did.get(child):
            line += "\n    what that change actually did: %s" % (
                " ".join(did[child].split())[:LINEAGE_NOTE_CHARS])
        rows.append(line)
    return "\n".join(rows)


def select_question(task: str, parent: State, maximize: bool, lineage: Optional[str],
                    memory_ctx: Optional[str], taken: list, fams: list) -> str:
    """Call A: pick method(s) for this parent from the listed families."""
    blocks = ["## PROGRAM TO IMPROVE\n%s" % (parent.code or "").strip()]
    raws = [raw_score(v, maximize) for v in
            [parent.value] + list(parent.parent_values or [])[:SCORE_HISTORY]]
    raws = [v for v in raws if v is not None]
    if raws:
        blocks.append(load("memo/score_history.txt").format(
            better="Higher" if maximize else "LOWER", scores=", ".join("%.10f" % v for v in raws)))
    if lineage:
        blocks.append("## WHAT THIS LINE OF DESCENT HAS ALREADY BEEN GIVEN\n%s" % lineage)
    if memory_ctx:
        blocks.append("## WHAT HAS GONE WRONG BEFORE\n%s" % memory_ctx)
    if taken:
        blocks.append("## METHODS ALREADY ASSIGNED TO OTHER GROUPS THIS STEP\n%s"
                      % "\n".join("- %s" % t for t in taken))
    blocks.append("## METHODS THIS RUN HAS FOUND\n%s" % listing(fams, parent.id))
    return task + "\n\n" + "\n\n".join(blocks)


def write_question(parent: State, fam) -> str:
    """Call B: write a parent-specific memo from the chosen family's memory."""
    ctx = (fam.memory or "").strip() or (fam.direction or "")
    blocks = ["## PROGRAM TO IMPROVE\n%s" % (parent.code or "").strip(),
              "## THE CHOSEN METHOD'S ACCUMULATED MEMORY\n%s\n\n%s" % (fam.label or fam.fid, ctx)]
    return load("memo/write.txt") + "\n\n" + "\n\n".join(blocks)


def code_diff(parent: str, child: str) -> str:
    """Unified parent->child diff, cut to DIFF_MAX_LINES lines."""
    d = list(difflib.unified_diff(parent.split("\n"), child.split("\n"),
                                  "parent", "child", lineterm="", n=3))
    return "\n".join(d[:DIFF_MAX_LINES])


def diff_sig(diff: str) -> Optional[str]:
    """Order-insensitive fingerprint of a diff's changed lines, or None if empty."""
    body = "\n".join(sorted(ln.rstrip() for ln in diff.split("\n")
                            if ln[:1] in "+-" and ln[:3] not in ("+++", "---")))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16] if body.strip() else None


def mining_question(pcode: str, ccode: str, pscore: Optional[float], cscore: Optional[float],
                    refused: Optional[tuple] = None) -> str:
    """Mining call: describe what an improving parent->child change did."""
    values = {"pscore": "%.10f" % (pscore or 0.0), "pcode": pcode,
              "cscore": "%.10f" % (cscore or 0.0), "ccode": ccode,
              "diff": code_diff(pcode, ccode)}
    if refused is None:
        body = load("memo/mine_pair.txt").format(**values)
    else:
        length, lo, hi = refused
        body = load("memo/mine_pair_refused.txt").format(length=length, lo=lo, hi=hi, **values)
    return load("memo/mine.txt") + "\n\n" + body


def _entry(i: int, fam: tuple, exemplars: list, chars: int) -> str:
    s = "EXISTING E%d\nlabel: %s\ndirection: %s" % (i, fam[1] or "(unlabelled)",
                                                    (fam[2] or "").strip())
    for ex in exemplars:
        ex = " ".join((ex or "").split())[:chars]
        if chars < FULL_EXEMPLAR_CHARS:
            ex = ex.rstrip()
        if ex:
            s += "\nnote already filed here: %s" % ex
    return s


def merge_question(fams: list, exemplars: dict, notes: list, k: int, chars: int,
                   retry: bool = False) -> str:
    """Global merge call over all families and some new notes, `k` exemplars each."""
    blocks = []
    if fams:
        blocks.append("## EXISTING METHODS\n\n" + "\n\n".join(
            _entry(i, c, (exemplars.get(c[0]) or [])[-k:] if k else [], chars)
            for i, c in enumerate(fams)))
    blocks.append("## NEW NOTES\n\n" + "\n\n".join(
        "NEW N%d\nmemo: %s" % (i, n.memo.strip()) for i, n in enumerate(notes)))
    task = load("memo/merge.txt")
    if retry:
        task = load("memo/merge_retry_prefix.txt") + task
    return "\n\n".join([task] + blocks)


def family_question(label: str, memory: str, notes: list) -> str:
    """Family-memory call: start or rewrite a family's memory from new notes."""
    new = "\n\n".join("### new note %d\n%s" % (i + 1, m) for i, m in enumerate(notes))
    shape = load("memo/family_memory_shape.txt")
    if not memory:
        return load("memo/family_start.txt").format(label=repr(label), shape=shape, notes=new)
    return load("memo/family_rewrite.txt").format(label=repr(label), shape=shape,
                                                  memory=memory, notes=new)
