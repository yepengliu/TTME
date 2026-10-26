"""Score a sandboxed program's result envelope in a fresh interpreter: `python -m ttt.eval.score TASK FILE`."""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import ttt
from ttt import tasks
from ttt.eval.sandbox import child_env, remove_tree
from ttt.log import warn
from ttt.tasks.common import Returned
from ttt.types import EvalResult

SCORE_TIMEOUT_S = 150.0
SCORE_ERROR_TAIL_CHARS = 500


def failed(error: str, stdout: str = "", infra_error: bool = False) -> EvalResult:
    """An unsuccessful evaluation carrying `error`; `infra_error` when the candidate is not to blame."""
    return EvalResult(ok=False, raw_score=None, construction=None, stdout=stdout, error=error,
                      infra_error=infra_error)


def result_to_dict(result: EvalResult) -> dict:
    """EvalResult as plain JSON data."""
    return dataclasses.asdict(result)


def result_from_dict(data: Any) -> EvalResult:
    """Rebuild an EvalResult received as JSON; raises ValueError on a malformed record."""
    if not isinstance(data, dict):
        raise ValueError("eval result must be a JSON object")
    ok, raw, construction = data.get("ok"), data.get("raw_score"), data.get("construction")
    stdout, error = data.get("stdout", ""), data.get("error")
    if not isinstance(ok, bool) or not isinstance(stdout, str):
        raise ValueError("eval result needs a boolean 'ok' and a string 'stdout'")
    if raw is not None and (isinstance(raw, bool) or not isinstance(raw, (int, float))):
        raise ValueError("eval result 'raw_score' must be a number or null")
    if construction is not None and not isinstance(construction, list):
        raise ValueError("eval result 'construction' must be a list or null")
    if error is not None and not isinstance(error, str):
        raise ValueError("eval result 'error' must be a string or null")
    infra = data.get("infra_error", False)
    if not isinstance(infra, bool):
        raise ValueError("eval result 'infra_error' must be a boolean")
    return EvalResult(ok=ok, raw_score=None if raw is None else float(raw),
                      construction=construction, stdout=stdout, error=error, infra_error=infra)


def decode_result(text: str) -> Returned:
    """Parse the sandbox's {"value", "signature"} envelope; raises ValueError when malformed."""
    data = json.loads(text)
    signature = data.get("signature") if isinstance(data, dict) else None
    if not isinstance(signature, dict) or "value" not in data:
        raise ValueError("result envelope needs 'value' and 'signature'")
    type_name, is_list, plain = (signature.get(k) for k in ("type_name", "is_list", "plain_numbers"))
    if not isinstance(type_name, str) or not isinstance(is_list, bool) or not isinstance(plain, bool):
        raise ValueError("result signature needs type_name, is_list and plain_numbers")
    return Returned(value=data["value"], type_name=type_name, is_list=is_list, plain_numbers=plain)


def package_root() -> str:
    """Directory that contains the `ttt` package, for the scorer's PYTHONPATH."""
    return str(Path(ttt.__file__).resolve().parent.parent)


def _run_scorer(task: str, result_json: str, cpus: int, timeout_s: float) -> subprocess.CompletedProcess:
    """Write the envelope to a scratch directory and run the scorer process on it."""
    root = tempfile.mkdtemp(prefix="ttt_score_")
    try:
        path = os.path.join(root, "result.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(result_json)
        env = child_env(root, cpus, sys.executable, {"PYTHONPATH": package_root()})
        return subprocess.run([sys.executable, "-m", "ttt.eval.score", task, path],
                              cwd=root, env=env, stdin=subprocess.DEVNULL, capture_output=True,
                              encoding="utf-8", errors="replace", timeout=timeout_s)
    finally:
        remove_tree(root)


def score_result(task: str, result_json: str, cpus: int = 1,
                 timeout_s: float = SCORE_TIMEOUT_S) -> EvalResult:
    """Score a sandbox result envelope in a separate `python -m ttt.eval.score` process."""
    # A fresh interpreter, so the candidate cannot have patched the verifier in-process; files it
    # changed on disk are not covered by this.
    try:
        proc = _run_scorer(task, result_json, cpus, timeout_s)
    except subprocess.TimeoutExpired:
        return failed(f"Scoring timed out after {timeout_s:g}s.")
    except OSError as e:
        warn("eval_scorer_start_failed", task=task, error=str(e)[:200])
        return failed(f"eval_sandbox_error: cannot start the scorer: {e}", infra_error=True)
    if proc.returncode != 0:
        return failed(f"Scoring failed: {proc.stderr.strip()[-SCORE_ERROR_TAIL_CHARS:]}")
    return result_from_dict(json.loads(proc.stdout))


def main(argv: Optional[list[str]] = None) -> int:
    """Scorer process: print the task's EvalResult for one result envelope file as a JSON object."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        sys.stderr.write("usage: python -m ttt.eval.score TASK RESULT_JSON\n")
        return 2
    with open(args[1], encoding="utf-8") as f:
        returned = decode_result(f.read())
    sys.stdout.write(json.dumps(result_to_dict(tasks.get(args[0]).score(returned))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
