"""Fake format, client and task shared by the memo tests (no tests here)."""
import hashlib
import json
import re
import threading
from dataclasses import dataclass
from ttt.config import AuxBudgets, MemoConfig
from ttt.memo import build_memo
from ttt.memo.library import Family, Library, Note, Variant
from ttt.types import ChildOutcome, State


TIMEOUT = 580.0


def sha8(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]


def kind_of(prompt):
    if "## NEW NOTES" in prompt:
        return "merge"
    if "Answer these four questions about THIS pair" in prompt:
        return "mine"
    if "method memory for the family" in prompt:
        return "family"
    if "## THE CHOSEN METHOD'S ACCUMULATED MEMORY" in prompt:
        return "write"
    if "## METHODS THIS RUN HAS FOUND" in prompt:
        return "select"
    return "other"


def listed_ids(prompt):
    return re.findall(r"^- id `([^`]+)`", prompt, re.M)


def select_first(prompt, idx, max_tokens):
    ids = listed_ids(prompt)
    m = re.search(r"Choose (?:the )?(\d+) ", prompt)
    if m:
        return json.dumps({"method_ids": ids[:int(m.group(1))]}), "stop"
    return json.dumps({"method_id": ids[0]}), "stop"


def write_named(prompt, idx, max_tokens):
    label = re.search(r"ACCUMULATED MEMORY\n(.*?)\n", prompt).group(1)
    return "Apply %s here, concretely." % label, "stop"


def mine_note(prompt, idx, max_tokens):
    return "The child changes the update schedule (variant %s)." % sha8(prompt), "stop"


def merge_ids(prompt):
    e = sorted({int(x) for x in re.findall(r"^EXISTING E(\d+)$", prompt, re.M)})
    n = sorted({int(x) for x in re.findall(r"^NEW N(\d+)$", prompt, re.M)})
    return e, n


def merge_place(prompt, idx, max_tokens):
    """Library-less: every note alone; otherwise every note onto one of the first three families."""
    e, n = merge_ids(prompt)
    if not e:
        groups = [{"ids": ["N%d" % k], "label": "Fresh method %d %s" % (k, sha8(prompt)),
                   "direction": "Do the fresh thing %d." % k} for k in n]
    else:
        heads = dict(re.findall(r"^EXISTING E(\d+)\nlabel: (.*)$", prompt, re.M))
        by = {}
        for k in n:
            by.setdefault(e[k % min(len(e), 3)], []).append("N%d" % k)
        groups = [{"ids": ["E%d" % j] + ns, "label": heads[str(j)],
                   "direction": "Merged direction of E%d." % j} for j, ns in by.items()]
    return json.dumps({"groups": groups}), "stop"


def family_text(prompt, idx, max_tokens):
    return "1. MECHANISM - distilled %s" % sha8(prompt), "stop"


DEFAULTS = {"select": select_first, "write": write_named, "mine": mine_note,
            "merge": merge_place, "family": family_text}


class FakeClient:
    """Word-count tokenizer; generation is never used by the memo route."""

    def generate(self, prompt, **kw):
        raise AssertionError("the memo route only uses Format.aux_call")

    def count_tokens(self, text):
        return len(text.split())


class FakeFormat:
    """Answers memo prompts with canned replies chosen by prompt kind."""

    name = "fake"

    def __init__(self, **handlers):
        self.handlers = dict(DEFAULTS, **handlers)
        self.calls = []
        self._lock = threading.Lock()

    def render(self, question):
        return question

    def complete(self, prompt, client):
        raise AssertionError("not used")

    def extract_code(self, text):
        return None

    def aux_call(self, client, prompt, *, effort, max_tokens, temperature=1.0, timeout_s=None):
        kind = kind_of(prompt)
        with self._lock:
            idx = sum(1 for c in self.calls if c["kind"] == kind)
            self.calls.append({"kind": kind, "prompt": prompt, "effort": effort,
                               "max_tokens": max_tokens, "timeout_s": timeout_s})
        return self.handlers[kind](prompt, idx, max_tokens)

    def of(self, kind):
        return [c for c in self.calls if c["kind"] == kind]


@dataclass
class FakeTask:
    name: str = "ac1"
    maximize: bool = False
    entrypoint: str = "run"
    length_window: tuple = (1000, 100000)


def make_state(sid, value, code=None, parents=None, parent_values=None, construction=None):
    return State(id=sid, timestep=0, value=value,
                 code=code if code is not None else "```python\nprint(%r)\n```" % sid,
                 construction=construction if construction is not None else [0.0] * 1500,
                 parents=list(parents or []), parent_values=list(parent_values or []))


def improving(child, raw, admitted=True, code=None):
    return ChildOutcome(ok=True, raw_score=raw, admitted=admitted, beats_parent=True,
                        docstring=None, code=code, code_offset=0, error=None, stdout="",
                        child=child)


def failed_outcome():
    return ChildOutcome(ok=False, raw_score=None, admitted=False, beats_parent=False,
                        docstring=None, code=None, code_offset=0, error="boom", stdout="")


def config(**kw):
    return MemoConfig(**kw)


def make_loop(fmt=None, cfg=None, task=None, out_dir=None, context_window=32768, aux=None):
    fmt = fmt or FakeFormat()
    loop = build_memo(cfg or MemoConfig(), task=task or FakeTask(), fmt=fmt, client=FakeClient(),
                      aux=aux or AuxBudgets(), children_per_parent=64,
                      context_window=context_window, out_dir=out_dir)
    return loop, fmt


def seed_library(n, words=6, memory=True):
    """A library of n one-variant families; family i was last fed at step i."""
    lib = Library()
    for i in range(n):
        fid, cid = "fam_%02d" % i, "fam_%02d_v" % i
        direction = " ".join(["Direction", "of", "method", str(i)] + ["word"] * words) + "."
        note = Note(sig="s%02d" % i, memo="Note about method %d." % i, step=i,
                    child_id="child_%d" % i, placed_step=i)
        lib.variants[cid] = Variant(cid=cid, family=fid, label="Method %d label" % i,
                                    direction=direction, created_step=i, written_step=i,
                                    notes=[note])
        lib.families[fid] = Family(fid=fid, label="Method %d label" % i, direction=direction,
                                   created_step=i, written_step=i, variants=[cid],
                                   memory=("Memory of method %d." % i) if memory else "",
                                   memory_sha="m%d" % i if memory else "")
    return lib


def archived(loop, drop=()):
    """Ids of every parent and child the loop has recorded, as if all were still archived."""
    ids = {t.parent_id for t in loop.pool} | {t.child_id for t in loop.pool}
    return frozenset(ids - set(drop))


def events(capsys, kind=None):
    out = []
    for line in capsys.readouterr().err.splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if kind is None or rec.get("event") == kind:
            out.append(rec)
    return out
