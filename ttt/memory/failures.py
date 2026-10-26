"""Deterministic grouping of a step's failed rollouts into masked-signature classes."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

CRASH_MAX_CLASSES = 6
CRASH_MAX_EXAMPLES = 3
TIMEOUT_MAX_TAILS = 3
TIMEOUT_TAIL_LINES = 3
TIMEOUT_TAIL_CHARS = 100
CODE_MAX_EXEMPLARS = 2
CODE_EXCERPT_CHARS = 600
CODE_TOTAL_CHARS = 6000
CODE_WINDOW_BEFORE = 8
CODE_WINDOW_AFTER = 4
SUMMARY_FIELD_CHARS = 700
SUMMARY_MAX_QUOTES = 3

_EXEC_PREFIX = "Program execution failed:"
_SIG_PATH = re.compile(r"/[\w./-]+")
_SIG_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")
_SIG_PAREN = re.compile(r"\([^)]*\)")
_SIG_NUM = re.compile(r"\d+")
_LINE_NO = re.compile(r"line (\d+)")
# Interpreter teardown output printed after a killed program's own last line.
_TEARDOWN_NOISE = re.compile(
    r"resource_tracker|warnings\.warn\(|"
    r"(?:User|Deprecation|Runtime|Resource)Warning|"
    r"Exception ignored in|atexit\._run_exitfuncs|"
    r"multiprocessing\.(?:pool|resource_tracker)", re.I)


def strip_exec_prefix(reason: str) -> str:
    """The failure message without the sandbox's 'Program execution failed:' prefix."""
    t = (reason or "").strip()
    return t[len(_EXEC_PREFIX):].strip() if t.startswith(_EXEC_PREFIX) else t


def failure_signature(reason: str) -> str:
    """Group key: the failure message with paths, quoted names, spans and digits masked."""
    t = strip_exec_prefix(reason)
    if not t:
        return "(no reason recorded)"
    t = _SIG_PATH.sub("<PATH>", t)
    t = _SIG_QUOTED.sub("<S>", t)
    t = _SIG_PAREN.sub("(...)", t)
    t = _SIG_NUM.sub("<N>", t)
    return " ".join(t.split())[:90]


def is_timeout_signature(text: str) -> bool:
    """True for a timeout reason or signature."""
    return "timed out" in (text or "").lower()


def progress_tail(stdout: str) -> str:
    """Last few non-empty lines of a killed program's own output, joined on one line."""
    lines = [ln.strip() for ln in (stdout or "").splitlines() if ln.strip()]
    lines = [ln for ln in lines if not _TEARDOWN_NOISE.search(ln)]
    if not lines:
        return ""
    return " / ".join(ln[:TIMEOUT_TAIL_CHARS] for ln in lines[-TIMEOUT_TAIL_LINES:])


def reported_line(reason: str) -> Optional[int]:
    """The assembled-file line number a failure message reports, if any."""
    m = _LINE_NO.search(reason or "")
    return int(m.group(1)) if m else None


def _collapse_blank_runs(lines: list[str]) -> list[str]:
    kept, blank = [], False
    for ln in lines:
        if not ln.strip():
            if blank:
                continue
            blank = True
        else:
            blank = False
        kept.append(ln)
    return kept


def code_excerpt(code: str, reason: str, offset: int) -> str:
    """A bounded slice of the model's program: a window at the failing line when it maps, else the head."""
    lines = _collapse_blank_runs([ln.rstrip() for ln in (code or "").splitlines()])
    if not lines:
        return ""
    n = reported_line(reason)
    # Reported lines count the harness prefix; a non-positive offset is unknown, not zero.
    if n is not None and int(offset or 0) > 0:
        target = n - int(offset)
        if 1 <= target <= len(lines):
            lo = max(0, target - 1 - CODE_WINDOW_BEFORE)
            hi = min(len(lines), target + CODE_WINDOW_AFTER)
            head = f"    ... model lines {lo + 1}-{hi} (failure at line {target}):"
            return (head + "\n" + "\n".join(lines[lo:hi]))[:CODE_EXCERPT_CHARS].rstrip()
    return "\n".join(lines)[:CODE_EXCERPT_CHARS].rstrip()


@dataclass
class _Group:
    count: int = 0
    examples: list[str] = field(default_factory=list)
    tails: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


def _add_distinct(items: list[str], item: str, cap: int) -> None:
    if item and item not in items and len(items) < cap:
        items.append(item)


def group_failures(records: list[dict]) -> dict[str, _Group]:
    """Group failure records by signature with distinct examples, timeout tails and code excerpts."""
    groups: dict[str, _Group] = {}
    for r in records:
        reason = r["reason"]
        sig = failure_signature(reason)
        g = groups.setdefault(sig, _Group())
        g.count += 1
        _add_distinct(g.examples, strip_exec_prefix(reason), CRASH_MAX_EXAMPLES)
        if is_timeout_signature(sig):
            _add_distinct(g.tails, progress_tail(r["stdout"]), TIMEOUT_MAX_TAILS)
        _add_distinct(g.sources, code_excerpt(r["code"], reason, r["code_offset"]), CODE_MAX_EXEMPLARS)
    return groups


def _summary_lines(summ: dict) -> list[str]:
    lines = ["         mechanism (from a full-evidence read): "
             f"{str(summ['mechanism'])[:SUMMARY_FIELD_CHARS]}"]
    for q in (summ.get("evidence_quotes") or [])[:SUMMARY_MAX_QUOTES]:
        lines.append(f"         grounded by: \"{str(q)[:110]}\"")
    if summ.get("suggested_adaptation"):
        lines.append("         adaptation that keeps the ambition: "
                     f"{str(summ['suggested_adaptation'])[:SUMMARY_FIELD_CHARS]}")
    return lines


def _evidence_lines(g: _Group, spent: int) -> tuple[list[str], int]:
    """Timeout tails and code excerpts of one group, within the shared code budget."""
    lines = [f"         reached before the kill: {t}" for t in g.tails]
    for s in g.sources:
        if spent + len(s) > CODE_TOTAL_CHARS:
            continue
        spent += len(s)
        lines.append("         one failing program (model's own text, before harness assembly):")
        lines.extend(f"           | {ln}" for ln in s.splitlines())
    return lines, spent


def summarize_failures(records: list[dict], summaries: dict[str, dict]) -> list[str]:
    """Render the top failure classes with counts, examples and either a cached read or raw evidence."""
    groups = group_failures(records)
    ranked = sorted(groups.items(), key=lambda kv: -kv[1].count)[:CRASH_MAX_CLASSES]
    out: list[str] = []
    spent = shown = 0
    for sig, g in ranked:
        shown += g.count
        out.append(f"    x{g.count} {sig}")
        if not (len(g.examples) == 1 and g.examples[0][:90] == sig):
            out.extend(f"         e.g. {e[:110]}" for e in g.examples)
        summ = summaries.get(sig)
        if summ and summ.get("mechanism") and summ["mechanism"] != "unclear":
            out.extend(_summary_lines(summ))
        else:
            lines, spent = _evidence_lines(g, spent)
            out.extend(lines)
    total = sum(g.count for g in groups.values())
    if len(groups) > CRASH_MAX_CLASSES and total:
        out.append(f"    (top {CRASH_MAX_CLASSES} of {len(groups)} signatures "
                   f"shown, covering {100 * shown // total}% of failures)")
    return out
