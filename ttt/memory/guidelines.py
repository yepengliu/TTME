"""Guideline Memory: parse distiller replies, merge into a capped store, render prompt blocks."""
from __future__ import annotations

import re
from typing import Optional

from ttt.prompts import load

MAX_PARSED_GUIDELINES = 12
STALE_STEPS = 20
MATCH_THRESHOLD = 0.5
MAX_GUIDELINE_CHARS = 240

_LABEL_RE = re.compile(r"^[\s\-*•_.)0-9<]*guideline\s*(\d+)\s*[*_>]*\s*[:.)][\s*_]*", re.IGNORECASE)
_LINE_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+(.*\S)\s*$")
_PLACEHOLDER_RE = re.compile(r"<[A-Za-z][^>\n]{0,40}>")
_LEAD_RE = re.compile(r"^\s*(?:new\s+lesson|lesson|note)\s*[:\-]\s*", re.I)
# "[3]" and "[seen 2]" markers of the distiller view, echoed back by the model in front of a line.
_VIEW_MARKER_RE = re.compile(r"^(?:\s*\[\s*(?:seen\s+)?\d+\s*\])+\s*", re.I)
_BLOCK_RE = re.compile(r"<guidelines>\s*(.*?)\s*</guidelines>", re.DOTALL | re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_HEADER_PREFIXES = ("#", "playbook", "updated playbook", "guideline memory", "updated guideline memory", "here")

# Openers of leaked reasoning, harvested log lines and meta-commentary, never real guidelines.
_NOISE_PREFIXES = (
    "a1:", "a2:", "a3:", "step-start", "this step", "this candidate",
    "crash signature", "no proposal", "no improvement", "no additional",
    "no new", "current best", "reflection", "batch reflection", "many failures",
    "specifically", "so,", "so ", "output", "here", "new best", "promoted",
    "for example", "for the", "from the", "reading", "the step", "the code",
    "the best", "there's", "there is", "since ", "the candidate", "based on",
    "also", "e.g", "example", "as shown", "for instance", "in this example",
    "looking", "perhaps", "maybe", "a guideline", "a failure-avoidance",
    "the user", "bad example", "good example", "action",
    "but ", "but,", "for this", "it could", "this could",
)
_TEMPLATE_PHRASES = (
    "an invariant that reliably breaks",
    "two edits that must go together",
    "works only with <",
    "in the example",
    "example given",
    "the code crashed -> add a guard",
    "define every helper you call inside the same edit",
    "a helper was called before it was defined",
)
_INSTRUCTION_ECHO = (
    "instruction says", "instructions say", "the instruction", "an instruction",
    "the prompt", "prompt says", "as instructed", "per the instruction",
    "we are told", "we're told", "the guideline says", "the rule says",
    "the example says", "it says to", "as stated in the",
    "user specified", "user says", "user asked",
    "the example", "the action should",
    "the query", "the human", "current playbook", "the promoted",
    "in the prompt",
)


def strip_label(text: str) -> str:
    """Drop a leading 'Guideline <n>:' label, tolerating list markers, emphasis and angle brackets."""
    m = _LABEL_RE.match(text or "")
    return text[m.end():].strip() if m else (text or "").strip()


def guideline_action(text: str) -> str:
    """The clause after the last '->' of an unlabelled guideline, or the whole text without an arrow."""
    t = strip_label(str(text or ""))
    if "->" in t:
        t = t.rsplit("->", 1)[-1].strip()
    return t


def is_noise_line(text: str) -> bool:
    """True for a line that is leaked reasoning, a log line or a template echo rather than a guideline."""
    t = text.strip()
    if "->" not in t and "→" not in t:
        return True
    if _PLACEHOLDER_RE.search(t):
        return True
    low = t.lower()
    if any(p in low for p in _TEMPLATE_PHRASES) or any(p in low for p in _INSTRUCTION_ECHO):
        return True
    if low.startswith(_NOISE_PREFIXES):
        return True
    if len(strip_label(t)) < 12:
        return True
    action = guideline_action(t)
    return not action or action.strip(" .…-") == ""


def strip_view_markers(text: str) -> str:
    """Drop leading echoed '[<n>]' and '[seen <n>]' distiller-view markers, in any order."""
    return _VIEW_MARKER_RE.sub("", text or "", count=1).strip()


def _candidate(raw: str) -> tuple[str, Optional[re.Match]]:
    """One reply line without list marker, echoed view markers or 'Lesson:' lead, plus its label match."""
    m = _LINE_RE.match(raw)
    cand = _LEAD_RE.sub("", strip_view_markers((m.group(1) if m else raw).strip())).strip()
    return cand, _LABEL_RE.match(cand)


def parse_guidelines(text: str) -> list[str]:
    """Extract up to twelve unlabelled guidelines from a distiller reply, preferring its <guidelines> block."""
    if not text:
        return []
    closed_think = "</think>" in text
    if closed_think:
        text = text.split("</think>", 1)[1]
    block = _BLOCK_RE.search(text)
    if block:
        text = block.group(1)
    # Without a block or a closed think span the reply is raw reasoning: keep labelled lines only.
    strict_only = not (block or closed_think)
    out = []
    for raw in text.splitlines():
        cand, label = _candidate(raw)
        if not cand or (strict_only and not label) or cand.lower().startswith(_HEADER_PREFIXES):
            continue
        body = strip_view_markers(cand[label.end():]) if label else cand
        # Judge a labelled line in canonical form, so label decorations are never mistaken for noise.
        if is_noise_line(f"Guideline {label.group(1)}: {body}" if label else cand):
            continue
        out.append(body)
        if len(out) >= MAX_PARSED_GUIDELINES:
            break
    return out


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _match_index(text: str, guidelines: list[dict]) -> Optional[int]:
    """Index of the stored guideline with the highest token Jaccard, if it reaches the threshold."""
    t = _tokens(text)
    if not t:
        return None
    best, best_i = 0.0, None
    for i, g in enumerate(guidelines):
        gt = _tokens(g.get("text", ""))
        if not gt:
            continue
        j = len(t & gt) / len(t | gt)
        if j > best:
            best, best_i = j, i
    return best_i if best >= MATCH_THRESHOLD else None


def merge_guidelines(prev: list[dict], new_texts: list[str], step: int, *,
                     max_guidelines: int) -> list[dict]:
    """Reaffirm matching guidelines, add new ones, expire stale ones and keep the top `max_guidelines`."""
    store = [{"text": str(g["text"]).strip(), "seen": int(g.get("seen", 1)),
              "last_seen": int(g.get("last_seen", step))} for g in prev if g.get("text")]
    for text in new_texts:
        text = (text or "").strip()
        if not text:
            continue
        idx = _match_index(text, store)
        if idx is not None:
            store[idx]["seen"] += 1
            store[idx]["last_seen"] = step
        else:
            store.append({"text": text, "seen": 1, "last_seen": step})
    store = [g for g in store if step - g["last_seen"] <= STALE_STEPS]
    store.sort(key=lambda g: (g["seen"], g["last_seen"]), reverse=True)
    return store[:max_guidelines]


def render_budget(text: str) -> str:
    """Cut a guideline to the render budget, marking the cut with an ellipsis."""
    t = text or ""
    if len(t) <= MAX_GUIDELINE_CHARS:
        return t
    return t[:MAX_GUIDELINE_CHARS].rstrip() + "…"


def guideline_block(guidelines: list[dict]) -> Optional[str]:
    """Render the store, most-seen first, as the Guideline Memory block of the generation prompt, or None."""
    norm = [(str(g.get("text", "")).strip(), int(g.get("seen", 1))) for g in guidelines]
    norm = [(t, s) for t, s in norm if t]
    if not norm:
        return None
    norm.sort(key=lambda x: x[1], reverse=True)
    body = "\n".join(f"  {i}. [seen {s}] {render_budget(t)}" for i, (t, s) in enumerate(norm, 1))
    return load("memory/guideline_heading.txt") + "\n" + body


def distiller_view(guidelines: list[dict]) -> str:
    """The whole store as numbered '[i] [seen N] text' lines for the distiller, or '(empty)'."""
    if not guidelines:
        return "(empty)"
    return "\n".join(f"[{i}] [seen {int(g.get('seen', 1) or 1)}] {g['text']}"
                      for i, g in enumerate(guidelines, 1))
