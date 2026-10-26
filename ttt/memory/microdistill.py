"""Per-signature failure reads: one call per new failure class over its full evidence, gated on verbatim quotes."""
from __future__ import annotations

import json
from typing import Callable, Optional

from ttt.log import event, warn
from ttt.memory.failures import SUMMARY_FIELD_CHARS, SUMMARY_MAX_QUOTES, failure_signature
from ttt.prompts import load

MAX_CALLS_PER_STEP = 3
MAX_MEMBERS = 2
MAX_ATTEMPTS = 3


def build_prompt(sig: str, count: int, members: list[dict]) -> str:
    """The failure-read question for one signature and up to two evidence members."""
    parts = [load("memory/microdistill_system.txt"),
             f"\n## FAILURE CLASS (x{count} this step)\nsignature: {sig}"]
    for i, m in enumerate(members, 1):
        parts.append(f"\n## ROLLOUT {i} — exact failure message\n{(m.get('reason') or '').strip()}")
        if m.get("code"):
            parts.append(f"\n## ROLLOUT {i} — full program text\n{m['code']}")
        if m.get("stdout"):
            parts.append(f"\n## ROLLOUT {i} — full captured stdout\n{m['stdout']}")
    return "\n".join(parts)


def parse_reply(text: str) -> Optional[dict]:
    """The outermost {...} span of a reply parsed as a JSON object, or None."""
    if not text:
        return None
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b <= a:
        return None
    try:
        obj = json.loads(text[a:b + 1])
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def validate_summary(obj: Optional[dict], evidence: str) -> Optional[dict]:
    """Schema and verbatim-quote gate; returns the normalized summary or None."""
    if not isinstance(obj, dict):
        return None
    mech, adapt = obj.get("mechanism"), obj.get("suggested_adaptation")
    if not isinstance(mech, str) or not isinstance(adapt, (str, type(None))):
        return None
    mech, adapt = mech.strip(), (adapt or "").strip()
    quotes = obj.get("evidence_quotes")
    if not mech:
        return None
    if mech.lower() == "unclear":
        return {"mechanism": "unclear", "evidence_quotes": [], "suggested_adaptation": ""}
    if not adapt or not isinstance(quotes, list) or not quotes:
        return None
    kept = []
    for q in quotes[:SUMMARY_MAX_QUOTES]:
        if not isinstance(q, str) or not q.strip() or q not in evidence:
            return None
        kept.append(q[:SUMMARY_FIELD_CHARS])
    return {"mechanism": mech[:SUMMARY_FIELD_CHARS], "evidence_quotes": kept,
            "suggested_adaptation": adapt[:SUMMARY_FIELD_CHARS]}


def pending_classes(records: list[dict], cache: dict, attempts: dict) -> list[tuple[str, list[dict]]]:
    """Uncached failure signatures with evidence and attempts left, largest first, at most three."""
    groups: dict[str, list[dict]] = {}
    for r in records:
        if not r["ok"] and (r["code"] or r["stdout"]):
            groups.setdefault(failure_signature(r["reason"]), []).append(r)
    pending = [(s, ms) for s, ms in groups.items()
               if s not in cache and attempts.get(s, 0) < MAX_ATTEMPTS]
    return sorted(pending, key=lambda kv: -len(kv[1]))[:MAX_CALLS_PER_STEP]


def _read_class(sig: str, members: list[dict], step: int, ask: Callable[[str], str]) -> Optional[dict]:
    """Run one failure read; returns the cache row or None when the gate rejects the reply."""
    shown = sorted(members, key=lambda m: (bool(m["code"]), bool(m["stdout"])), reverse=True)[:MAX_MEMBERS]
    evidence = "\n".join(m["code"] + "\n" + m["stdout"] + "\n" + (m["reason"] or "") for m in shown)
    summ = validate_summary(parse_reply(ask(build_prompt(sig, len(members), shown))), evidence)
    if summ is None:
        return None
    return {"sig": sig, "step": step, "count": len(members), **summ}


def run_step(records: list[dict], cache: dict, attempts: dict, step: int, ask: Callable[[str], str]) -> None:
    """Read each pending failure class once, caching accepted summaries and counting failed attempts."""
    for sig, members in pending_classes(records, cache, attempts):
        try:
            row = _read_class(sig, members, step, ask)
        except Exception as e:  # a failed memory call must never stop the run
            attempts[sig] = attempts.get(sig, 0) + 1
            warn("memory_microdistill_failed", step=step, sig=sig, attempt=attempts[sig],
                 error=repr(e)[:300])
            continue
        if row is None:
            attempts[sig] = attempts.get(sig, 0) + 1
            warn("memory_microdistill_rejected", step=step, sig=sig, attempt=attempts[sig])
            continue
        cache[sig] = row
        event("memory_microdistill", **row)
