"""Memory manager: lineage registry, step records, Guideline Memory and failure-read cache."""
from __future__ import annotations

import copy
from typing import Optional

from ttt.config import AuxBudgets, MemoryConfig
from ttt.log import event, warn
from ttt.memory.distill import build_step_summary, distiller_question, step_record
from ttt.memory.docstring import extract_docstring, unfence
from ttt.memory.guidelines import guideline_block, merge_guidelines, parse_guidelines
from ttt.memory.microdistill import run_step as run_microdistill
from ttt.memory.recipe import render_recipe
from ttt.types import ChildOutcome, Client, Format, State, Task

STATE_VERSION = 2
AUX_EFFORT = "low"
AUX_TEMPERATURE = 1.0
_STATE_KEYS = ("registry", "guidelines", "step_records", "failure_summaries", "microdistill_attempts")


def state_key(state: State) -> str:
    """Registry key of a state: its construction hash when known, else its id."""
    return state.construction_hash or state.id


def _ancestor_key(ref: dict) -> str:
    return str(ref.get("construction_hash") or ref.get("id"))


class MemoryManager:
    """Records rollout outcomes, renders each parent's memory block and distills guidelines each step."""

    def __init__(self, cfg: MemoryConfig, *, task: Task, fmt: Format, client: Client, aux: AuxBudgets):
        self.cfg = cfg
        self.task = task
        self.fmt = fmt
        self.client = client
        self.aux = aux
        self.registry: dict[str, dict] = {}
        self.guidelines: list[dict] = []
        self.failure_summaries: dict[str, dict] = {}
        self.microdistill_attempts: dict[str, int] = {}
        self._records: list[dict] = []

    def register(self, state: State, raw_score: Optional[float]) -> None:
        """Remember a state's docstring and raw score for lineage recipes."""
        doc = extract_docstring(unfence(state.code), self.task.entrypoint)
        self.registry[state_key(state)] = {"doc": doc, "raw": None if raw_score is None else float(raw_score)}

    def context_for(self, parent: State) -> Optional[str]:
        """The parent's memory block: lineage recipe, then the Guideline Memory."""
        chain = [state_key(parent)] + [_ancestor_key(p) for p in parent.parents or []]
        blocks = [render_recipe(chain, self.registry, self.task.maximize)]
        if self.cfg.guidelines:
            blocks.append(guideline_block(self.guidelines))
        blocks = [b for b in blocks if b]
        return "\n\n".join(blocks) if blocks else None

    def record_child(self, parent: State, outcome: ChildOutcome) -> None:
        """Keep one rollout's outcome for the step summary; register the child when admitted."""
        if outcome.admitted and outcome.child is not None:
            self.register(outcome.child, outcome.raw_score)
        if self.cfg.guidelines:
            self._records.append(step_record(state_key(parent), outcome))

    def end_of_step(self, step: int, best_raw: Optional[float]) -> None:
        """Read new failure classes, distill this step's guidelines and merge them into the Guideline Memory."""
        records, self._records = self._records, []
        if not self.cfg.guidelines:
            return
        run_microdistill(records, self.failure_summaries, self.microdistill_attempts, step,
                         lambda prompt: self._ask(prompt, self.aux.microdistill_max_tokens))
        summary = build_step_summary(records, best_raw, self.task.maximize, self.failure_summaries)
        reply = self._distill(step, distiller_question(self.guidelines, summary))
        new_texts = parse_guidelines(reply)
        self.guidelines = merge_guidelines(self.guidelines, new_texts, step,
                                           max_guidelines=self.cfg.max_guidelines)
        event("memory_guidelines", step=step, new=new_texts, guidelines=self.guidelines)

    def _ask(self, prompt: str, max_tokens: int) -> str:
        """One auxiliary call at low effort; warns when the model produced no final answer."""
        text, finish = self.fmt.aux_call(self.client, prompt, effort=AUX_EFFORT,
                                         max_tokens=max_tokens, temperature=AUX_TEMPERATURE)
        if not text:
            warn("memory_aux_empty_reply", finish=finish, prompt_chars=len(prompt))
        return text

    def _distill(self, step: int, question: str) -> str:
        """The distiller reply, or "" after a warning when the call fails."""
        try:
            return self._ask(question, self.aux.distill_max_tokens)
        except Exception as e:  # a failed memory call must never stop the run
            warn("memory_distill_failed", step=step, error=repr(e)[:300])
            return ""

    def state_dict(self) -> dict:
        """Everything needed to resume, as JSON-serialisable data."""
        return copy.deepcopy({
            "version": STATE_VERSION,
            "registry": self.registry,
            "guidelines": self.guidelines,
            "step_records": self._records,
            "failure_summaries": self.failure_summaries,
            "microdistill_attempts": self.microdistill_attempts,
        })

    def load_state_dict(self, state: dict) -> None:
        """Restore from `state_dict()` output."""
        if state.get("version") != STATE_VERSION:
            raise ValueError(f"unsupported memory state version {state.get('version')!r}")
        missing = [k for k in _STATE_KEYS if k not in state]
        if missing:
            raise ValueError(f"memory state is missing {missing}")
        state = copy.deepcopy(state)
        self.registry = {str(k): {"doc": v["doc"], "raw": v["raw"]} for k, v in state["registry"].items()}
        self.guidelines = [{"text": str(g["text"]), "seen": int(g["seen"]), "last_seen": int(g["last_seen"])}
                           for g in state["guidelines"]]
        self._records = list(state["step_records"])
        self.failure_summaries = dict(state["failure_summaries"])
        self.microdistill_attempts = {str(k): int(v) for k, v in state["microdistill_attempts"].items()}


def build_memory(cfg: MemoryConfig, *, task: Task, fmt: Format, client: Client,
                 aux: AuxBudgets) -> MemoryManager:
    """Create the memory manager for one run."""
    return MemoryManager(cfg, task=task, fmt=fmt, client=client, aux=aux)
