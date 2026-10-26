"""On-disk run records: manifest, rollout and step logs, step checkpoints, atomic writes."""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from ttt import __version__
from ttt.archive.store import code_hash
from ttt.config import Experiment, Runtime, config_hash, to_dict

COMMITTED = "COMMITTED"
CHECKPOINT_PREFIX = "step_"
ERROR_CHARS = 200
STDOUT_TAIL_CHARS = 2000
EVENTS = "events.jsonl"


@dataclass(frozen=True)
class RunPaths:
    """File layout of one run directory."""

    root: Path

    @classmethod
    def of(cls, out_dir: str | os.PathLike) -> "RunPaths":
        """Paths for the run in out_dir."""
        return cls(Path(out_dir))

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def rollouts(self) -> Path:
        return self.root / "rollouts.jsonl"

    @property
    def steps(self) -> Path:
        return self.root / "steps.jsonl"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def constructions(self) -> Path:
        return self.root / "constructions"

    @property
    def codes(self) -> Path:
        return self.root / "codes"

    def checkpoint(self, step: int) -> Path:
        """Directory of the checkpoint taken after step."""
        return self.checkpoints / f"{CHECKPOINT_PREFIX}{step:06d}"


@dataclass
class StepRow:
    """One line of steps.jsonl: `best` is the archive's best, `best_valid` the best valid child so far."""

    step: int
    best: Optional[float]
    best_valid: Optional[float]
    valid: int
    failed: int
    infra: int
    seconds: float


def rollout_row(*, step: int, rollout: int, parent_id: str, parent_score: Optional[float],
                child_id: Optional[str], child_score: Optional[float], ok: bool,
                admitted: bool, beats_parent: bool, error: Optional[str], doc: Optional[str],
                code: Optional[str], stdout: Optional[str], arm: str, tags: dict,
                infra_error: bool) -> dict[str, Any]:
    """One line of rollouts.jsonl."""
    return {
        "step": step,
        "rollout": rollout,
        "parent_id": parent_id,
        "parent_score": parent_score,
        "child_id": child_id,
        "child_score": child_score,
        "ok": ok,
        "admitted": admitted,
        "beats_parent": beats_parent,
        "error": error[:ERROR_CHARS] if error else None,
        "doc": doc,
        "code_sha256": code_hash(code) if code is not None else None,
        "stdout_tail": (stdout or "")[-STDOUT_TAIL_CHARS:] or None,
        "arm": arm,
        "tags": dict(tags),
        "infra_error": infra_error,
    }


def atomic_write_text(path: Path, text: str) -> None:
    """Write a file via a same-directory temp file and rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def atomic_write_json(path: Path, obj: Any) -> None:
    """Write JSON atomically."""
    atomic_write_text(path, json.dumps(obj, indent=1, sort_keys=True) + "\n")


def json_line(row: dict) -> str:
    """One JSON-lines record, newline included."""
    return json.dumps(row) + "\n"


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """Append rows to a JSON-lines file and fsync it."""
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json_line(row))
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSON-lines file; a torn final line is dropped, any other bad line raises."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").split("\n")
    rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            if i == len(lines) - 1:
                break
            raise ValueError(f"{path}: bad JSON on line {i + 1}: {e}") from e
    return rows


def truncate_jsonl(path: Path, last_step: int) -> int:
    """Rewrite a step-keyed JSON-lines file without rows after last_step; returns rows dropped."""
    rows = read_jsonl(path)
    kept = [r for r in rows if int(r["step"]) <= last_step]
    if path.exists():
        atomic_write_text(path, "".join(json_line(r) for r in kept))
    return len(rows) - len(kept)


def has_run_state(root: Path) -> bool:
    """True if root holds anything besides the event log of an attempt that never started its run."""
    return root.is_dir() and any(p.name != EVENTS for p in root.iterdir())


def claim_out_dir(out_dir: str, resume: bool) -> None:
    """Create or reuse a run directory without run state, or check that a resumed one exists."""
    root = Path(out_dir)
    if resume:
        if not (root / "manifest.json").exists():
            raise FileNotFoundError(f"cannot resume: no manifest.json in {root}")
        return
    if has_run_state(root):
        raise FileExistsError(f"refusing a fresh run in non-empty {root}; pass --resume or pick a new directory")
    root.mkdir(parents=True, exist_ok=True)


def runtime_record(runtime: Runtime) -> dict[str, Any]:
    """Runtime fields that are safe to persist (never the API key)."""
    rec = dataclasses.asdict(runtime)
    rec.pop("api_key")
    rec["vllm_urls"] = list(rec["vllm_urls"])
    return rec


def write_manifest(paths: RunPaths, exp: Experiment, runtime: Runtime) -> None:
    """Write manifest.json for a fresh run."""
    atomic_write_json(paths.manifest, {
        "config": to_dict(exp),
        "config_hash": config_hash(exp),
        "version": __version__,
        "runtime": runtime_record(runtime),
    })


def check_manifest(paths: RunPaths, exp: Experiment) -> dict[str, Any]:
    """Load manifest.json and require that it was written for the same experiment config."""
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    if manifest.get("config_hash") != config_hash(exp):
        raise ValueError(f"config hash mismatch: {paths.manifest} was written for a different experiment")
    return manifest


def write_checkpoint(directory: Path, parts: dict[str, dict]) -> None:
    """Write <name>.json for each part, then the COMMITTED marker."""
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / COMMITTED
    if marker.exists():
        raise FileExistsError(f"checkpoint {directory} is already committed")
    for name, data in parts.items():
        atomic_write_json(directory / f"{name}.json", data)
    atomic_write_text(marker, "")


def read_checkpoint(directory: Path) -> dict[str, dict]:
    """Load every part of a committed checkpoint."""
    if not (directory / COMMITTED).exists():
        raise FileNotFoundError(f"checkpoint {directory} is not committed")
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.json"))}


def checkpoint_steps(paths: RunPaths) -> dict[int, Path]:
    """All checkpoint directories by step, committed or not."""
    if not paths.checkpoints.exists():
        return {}
    out = {}
    for d in paths.checkpoints.iterdir():
        if d.is_dir() and d.name.startswith(CHECKPOINT_PREFIX):
            out[int(d.name[len(CHECKPOINT_PREFIX):])] = d
    return out


def latest_committed(paths: RunPaths) -> Optional[int]:
    """Highest step whose checkpoint carries the COMMITTED marker."""
    done = [s for s, d in checkpoint_steps(paths).items() if (d / COMMITTED).exists()]
    return max(done) if done else None


def discard_after(paths: RunPaths, step: int) -> dict[str, int]:
    """Drop log rows and checkpoint directories beyond step; returns what was removed."""
    removed_dirs = 0
    for s, d in checkpoint_steps(paths).items():
        if s > step:
            shutil.rmtree(d)
            removed_dirs += 1
    return {"rollouts": truncate_jsonl(paths.rollouts, step),
            "steps": truncate_jsonl(paths.steps, step), "checkpoints": removed_dirs}


def check_logs(paths: RunPaths, step: int) -> list[dict]:
    """Require steps.jsonl to end at step and rollouts.jsonl to hold valid+failed+infra rows per step."""
    steps = read_jsonl(paths.steps)
    if not steps or int(steps[-1]["step"]) != step:
        raise ValueError(f"steps.jsonl ends at {steps[-1]['step'] if steps else None}, checkpoint is {step}")
    counts: dict[int, int] = {}
    for row in read_jsonl(paths.rollouts):
        counts[int(row["step"])] = counts.get(int(row["step"]), 0) + 1
    for row in steps:
        expected = int(row["valid"]) + int(row["failed"]) + int(row["infra"])
        if counts.pop(int(row["step"]), 0) != expected:
            raise ValueError(f"rollouts.jsonl does not hold {expected} rows for step {row['step']}")
    if counts:
        raise ValueError(f"rollouts.jsonl has rows for steps without a step record: {sorted(counts)}")
    return steps
