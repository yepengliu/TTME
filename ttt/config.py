"""Experiment configuration: typed dataclasses loaded from YAML."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, get_args, get_origin, get_type_hints

import yaml

TaskName = Literal["ac1", "ac2", "erdos"]
FormatName = Literal["harmony_two_phase", "chatml_think"]
ControlCard = Literal["deepen_v2", "deepen_v1", "none"]
SelectMode = Literal["serial", "parallel"]


@dataclass(frozen=True)
class AuxBudgets:
    """Token budgets for memory and memo calls; they differ by model format."""

    distill_max_tokens: int = 1024
    microdistill_max_tokens: int = 1200
    memo_max_tokens: int = 4500
    family_max_tokens: int = 8000
    merge_max_tokens: int = 9000


@dataclass(frozen=True)
class Model:
    """Prompt format and decoding limits of the served model."""

    name: str
    format: FormatName
    phase1_max_tokens: Optional[int] = None
    reasoning_effort: str = "high"
    date: Optional[str] = None
    aux: AuxBudgets = field(default_factory=AuxBudgets)


@dataclass(frozen=True)
class Sampling:
    """Decoding parameters for generation rollouts."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    context_window: int = 32768


@dataclass(frozen=True)
class Search:
    """PUCT search budget and selection parameters."""

    steps: int = 50
    parents_per_step: int = 8
    children_per_parent: int = 64
    puct_c: float = 1.0
    archive_capacity: int = 1000
    topk_children: int = 2
    seed: Optional[int] = None


@dataclass(frozen=True)
class Eval:
    """Sandbox limits for one candidate program."""

    timeout_s: float = 1100.0
    cpus: int = 1


@dataclass(frozen=True)
class MemoryConfig:
    """Lineage recipe (always on when memory is enabled) plus the Guideline Memory."""

    guidelines: bool = True
    max_guidelines: int = 24


@dataclass(frozen=True)
class MemoBudgets:
    """Wall-clock and concurrency limits of the memo calls."""

    stage1_budget_s: float = 480.0
    stage1_workers: int = 8
    call_timeout_s: float = 580.0
    mining_budget_s: float = 600.0
    mining_workers: int = 6
    family_budget_s: float = 180.0
    family_workers: int = 4
    merge_max_rounds: int = 8
    merge_rounds_budget_s: float = 720.0
    merge_distill_reserve_s: float = 360.0
    merge_ctx_margin: int = 64
    merge_max_tokens_cap: int = 16384
    merge_drain_min_notes: int = 8


@dataclass(frozen=True)
class MemoConfig:
    """Parent-specific method memos mined from improving transitions."""

    selected_children: int = 32
    control_card: ControlCard = "deepen_v2"
    methods_per_parent: int = 1
    select: SelectMode = "serial"
    boundary_every: int = 2
    notes_per_boundary: int = 24
    stepbest_reserve: int = 2
    include_refused: bool = True
    merge_drain: bool = True
    budgets: MemoBudgets = field(default_factory=MemoBudgets)


@dataclass(frozen=True)
class Experiment:
    """Everything that can change a run's results; hashed into the manifest."""

    name: str
    task: TaskName
    model: Model
    sampling: Sampling = field(default_factory=Sampling)
    search: Search = field(default_factory=Search)
    eval: Eval = field(default_factory=Eval)
    memory: Optional[MemoryConfig] = None
    memo: Optional[MemoConfig] = None


@dataclass(frozen=True)
class Runtime:
    """Where and how a run executes; never changes results, so not hashed."""

    out_dir: str
    vllm_urls: tuple[str, ...]
    served_model: str
    api_key: Optional[str] = None
    eval_workers_file: Optional[str] = None
    local_eval_workers: int = 0
    gen_concurrency: int = 64
    eval_concurrency: Optional[int] = None
    request_timeout_s: float = 2400.0
    resume: bool = False
    memo_url: Optional[str] = None
    save_code: bool = True


class ConfigError(ValueError):
    """Raised for unknown keys, bad types or inconsistent settings."""


def _is_dataclass_type(tp: Any) -> bool:
    return isinstance(tp, type) and dataclasses.is_dataclass(tp)


def _coerce(tp: Any, value: Any, where: str) -> Any:
    origin = get_origin(tp)
    if origin is Optional or (origin is not None and type(None) in get_args(tp)):
        if value is None:
            return None
        inner = [a for a in get_args(tp) if a is not type(None)]
        return _coerce(inner[0], value, where)
    if origin is Literal:
        if value not in get_args(tp):
            raise ConfigError(f"{where}: {value!r} is not one of {get_args(tp)}")
        return value
    if origin is tuple:
        return tuple(value)
    if _is_dataclass_type(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping, got {type(value).__name__}")
        return build(tp, value, where)
    if tp is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if tp in (int, float, str, bool) and not isinstance(value, tp):
        raise ConfigError(f"{where}: expected {tp.__name__}, got {value!r}")
    if tp is int and isinstance(value, bool):
        raise ConfigError(f"{where}: expected int, got {value!r}")
    return value


def build(cls: type, data: dict, where: str = "") -> Any:
    """Build a config dataclass from a mapping, rejecting unknown keys."""
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"{where or cls.__name__}: unknown keys {sorted(unknown)}")
    kwargs = {k: _coerce(hints[k], v, f"{where}.{k}" if where else k) for k, v in data.items()}
    try:
        return cls(**kwargs)
    except TypeError as e:
        raise ConfigError(f"{where or cls.__name__}: {e}") from e


def _at_least(where: str, value: int, low: int) -> None:
    if value < low:
        raise ConfigError(f"{where} must be >= {low}, got {value}")


def _validate_model(exp: Experiment) -> None:
    phase1 = exp.model.phase1_max_tokens
    if exp.model.format == "harmony_two_phase" and not phase1:
        raise ConfigError("harmony_two_phase needs model.phase1_max_tokens")
    if exp.model.format == "chatml_think" and phase1:
        raise ConfigError("chatml_think is single-phase; drop model.phase1_max_tokens")
    if phase1 is not None and not 0 < phase1 < exp.sampling.context_window:
        raise ConfigError("model.phase1_max_tokens must be in [1, sampling.context_window)")


def _validate_memo(exp: Experiment) -> None:
    memo = exp.memo
    if exp.memory is None:
        raise ConfigError("memo requires memory to be enabled")
    for name in ("boundary_every", "notes_per_boundary", "methods_per_parent"):
        _at_least(f"memo.{name}", getattr(memo, name), 1)
    if not 0 < memo.selected_children <= exp.search.children_per_parent:
        raise ConfigError("memo.selected_children must be in [1, children_per_parent]")
    if memo.selected_children % memo.methods_per_parent:
        raise ConfigError("memo.selected_children must divide evenly by methods_per_parent")


def validate(exp: Experiment) -> None:
    """Check value ranges and cross-field constraints that a type check cannot express."""
    for name in ("steps", "parents_per_step", "children_per_parent", "archive_capacity"):
        _at_least(f"search.{name}", getattr(exp.search, name), 1)
    _at_least("search.topk_children", exp.search.topk_children, 0)
    if not exp.eval.timeout_s > 0:
        raise ConfigError(f"eval.timeout_s must be > 0, got {exp.eval.timeout_s}")
    _at_least("eval.cpus", exp.eval.cpus, 1)
    _validate_model(exp)
    if exp.memo is not None:
        _validate_memo(exp)


def load_experiment(path: str | Path, models_dir: str | Path | None = None) -> Experiment:
    """Load an experiment YAML; `model:` may name a file in configs/models."""
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    data.setdefault("name", path.stem)
    model = data.get("model")
    if isinstance(model, str):
        mdir = Path(models_dir) if models_dir else path.parent.parent / "models"
        data["model"] = yaml.safe_load((mdir / f"{model}.yaml").read_text())
        data["model"].setdefault("name", model)
    exp = build(Experiment, data)
    validate(exp)
    return exp


def to_dict(exp: Experiment) -> dict:
    """Fully resolved config as plain data."""
    return dataclasses.asdict(exp)


def config_hash(exp: Experiment) -> str:
    """Stable sha256 of the resolved experiment config, name excluded so a renamed config still resumes."""
    data = to_dict(exp)
    data.pop("name")
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()
