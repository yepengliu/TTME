#!/usr/bin/env python3
"""Recompute the paper's stored constructions without running candidate programs.

The ThetaEvolve scoring formulas below are adapted from
https://github.com/ypwang61/ThetaEvolve/tree/main/examples
under Apache-2.0; see NOTICE and LICENSES/Apache-2.0.txt.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ttt.tasks.ac import AC1, AC2, sequence_value
from ttt.tasks.erdos import ERDOS, stored_c5


def det_bareiss(matrix: list[list[int]]) -> int:
    """Return an exact integer determinant, including the row-swap sign."""
    a = [row.copy() for row in matrix]
    n = len(a)
    sign, previous = 1, 1
    for k in range(n - 1):
        if a[k][k] == 0:
            pivot = next((i for i in range(k + 1, n) if a[i][k] != 0), None)
            if pivot is None:
                return 0
            a[k], a[pivot] = a[pivot], a[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                numerator = a[i][j] * a[k][k] - a[i][k] * a[k][j]
                quotient, remainder = divmod(numerator, previous)
                if remainder:
                    raise ValueError("Non-exact division in integer determinant")
                a[i][j] = quotient
            a[i][k] = 0
        previous = a[k][k]
    return sign * a[-1][-1] if n else 1


def theta_score(task: str, values: np.ndarray) -> tuple[float, int | None]:
    """ThetaEvolve objectives: AC2 ratio, raw AC3 bound, normalized determinant."""
    if task == "hadamard":
        if values.shape != (29, 29) or not np.all(np.isin(values, [-1, 1])):
            raise ValueError("Hadamard requires a 29 by 29 matrix with entries +/-1")
        determinant = abs(det_bareiss(values.astype(int).tolist()))
        return determinant / ((2 ** 28) * (7 ** 12) * 342), determinant
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError("Autocorrelation requires a finite, nonempty vector")
    if np.max(np.abs(values)) > 1e10:
        raise ValueError("Heights exceed the evaluator's magnitude limit")
    if task == "ac2" and np.any(values < -1e-10):
        raise ValueError("AC2 heights must be nonnegative")
    convolution = np.convolve(values, values)
    peak = float(np.max(np.abs(convolution)))
    if peak > 1e10:
        raise ValueError("Convolution exceeds the evaluator's magnitude limit")
    if task == "ac3":
        total = float(np.sum(values))
        if total ** 2 < 1e-10:
            raise ValueError("AC3 heights have near-zero total mass")
        return abs(2 * values.size * peak / total ** 2), None
    if task == "ac2":
        y = np.concatenate(([0.0], convolution, [0.0]))
        widths = np.diff(np.linspace(-0.5, 0.5, convolution.size + 2))
        l2_squared = 0.0
        for left, right, width in zip(y[:-1], y[1:], widths):
            l2_squared += (width / 3.0) * (left * left + left * right + right * right)
        l1 = float(np.sum(np.abs(convolution))) / (convolution.size + 1)
        if l1 <= 1e-10 or peak <= 1e-10 or l2_squared < 1e-20:
            raise ValueError("AC2 convolution has near-zero norm")
        return float(l2_squared / (l1 * peak)), None
    raise ValueError(f"Unknown ThetaEvolve task: {task}")


def verify_entry(entry: dict, root: Path = ROOT) -> float:
    path = root / entry["path"]
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != entry["sha256"]:
        raise ValueError("Construction file SHA-256 mismatch")
    data = json.loads(contents)
    values = np.asarray(data[entry["array_key"]], dtype=float)
    if list(values.shape) != entry["shape"] or not np.all(np.isfinite(values)):
        raise ValueError("Construction shape or finite-value check failed")
    if entry["task"] == "hadamard" and not np.all(np.isin(values, [-1, 1])):
        raise ValueError("Hadamard entries must be +/-1 before integer conversion")
    raw = values.astype(entry["raw_dtype"]).tobytes(order="C")
    if hashlib.sha256(raw).hexdigest() != entry["raw_sha256"]:
        raise ValueError("Construction values SHA-256 mismatch")
    determinant = None
    if entry["framework"] == "TTT-Discover":
        task = {"ac1": AC1, "ac2": AC2, "erdos": ERDOS}[entry["task"]]
        if values.ndim != 1 or not task.admissible(values):
            raise ValueError("Construction is outside the archive's length limits")
        if entry["task"] == "erdos":
            score = stored_c5(values.tolist())
        else:
            score = sequence_value(entry["task"], values.tolist())
    elif entry["framework"] == "ThetaEvolve":
        score, determinant = theta_score(entry["task"], values)
    else:
        raise ValueError("Unknown evaluation framework")
    if score is None or not math.isfinite(score):
        raise ValueError("Construction failed evaluation")
    if not math.isclose(score, entry["score"], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"Recomputed score {score:.16g} differs from stored score")
    if f"{score:.6f}" != entry["paper_score"]:
        raise ValueError(f"Recomputed score {score:.6f} differs from the paper")
    if determinant is not None and str(determinant) != entry["absolute_determinant"]:
        raise ValueError("Exact determinant differs from the stored determinant")
    if not 0 <= entry["discovery_step"] < entry["search_steps"]:
        raise ValueError("Discovery step is outside the reported search budget")
    if "program" in entry:
        program = (root / entry["program"]).read_bytes()
        if hashlib.sha256(program).hexdigest() != entry["program_sha256"]:
            raise ValueError("Generator snapshot SHA-256 mismatch")
    return float(score)


def main() -> int:
    manifest = json.loads((ROOT / "results/manifest.json").read_text())
    failures = 0
    for entry in manifest["constructions"]:
        label = f"{entry['framework']} / {entry['model']} / {entry['task']}"
        try:
            score = verify_entry(entry)
            print(f"PASS {label}: {score:.15f} (paper: {entry['paper_score']})")
        except (ValueError, KeyError, OSError, TypeError) as error:
            failures += 1
            print(f"FAIL {label}: {error}", file=sys.stderr)
    total = len(manifest["constructions"])
    print(f"{total - failures}/{total} constructions verified")
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
