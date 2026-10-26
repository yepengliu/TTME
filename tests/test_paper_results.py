"""Release checks tied to the paper's main TTME rows."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_results", ROOT / "scripts/verify_results.py")
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)
ENTRIES = json.loads((ROOT / "results/manifest.json").read_text())["constructions"]


def test_only_reported_main_constructions_are_released():
    expected = {
        ("gpt-oss-120b", "ac1", "1.502745"),
        ("gpt-oss-120b", "ac2", "0.960784"),
        ("gpt-oss-120b", "erdos", "0.380862"),
        ("Qwen3-8B", "ac1", "1.504329"),
        ("Qwen3-8B", "ac2", "0.949248"),
        ("Qwen3-8B", "erdos", "0.380927"),
        ("Distill-Qwen3-8B", "ac2", "0.948148"),
        ("Distill-Qwen3-8B", "ac3", "1.464224"),
        ("Distill-Qwen3-8B", "hadamard", "0.579938"),
    }
    assert len(ENTRIES) == len(expected)
    assert {(e["model"], e["task"], e["paper_score"]) for e in ENTRIES} == expected
    expected_files = {ROOT / e["path"] for e in ENTRIES}
    expected_files |= {ROOT / e["program"] for e in ENTRIES if "program" in e}
    expected_files |= {ROOT / "results/README.md", ROOT / "results/manifest.json"}
    actual_files = {p for p in (ROOT / "results").rglob("*") if p.is_file()}
    assert actual_files == expected_files


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: f"{e['model']}-{e['task']}")
def test_paper_construction_recomputes(entry):
    VERIFIER.verify_entry(entry)


def test_changed_construction_fails_hash_check(tmp_path):
    entry = ENTRIES[3]
    target = tmp_path / entry["path"]
    target.parent.mkdir(parents=True)
    data = json.loads((ROOT / entry["path"]).read_text())
    data["sequence"][0] += 0.01
    target.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        VERIFIER.verify_entry(entry, tmp_path)


def test_unreported_score_is_rejected():
    entry = copy.deepcopy(ENTRIES[3])
    entry["paper_score"] = "1.500000"
    with pytest.raises(ValueError, match="differs from the paper"):
        VERIFIER.verify_entry(entry)


def test_out_of_budget_construction_is_rejected():
    entry = copy.deepcopy(ENTRIES[3])
    entry["discovery_step"] = entry["search_steps"]
    with pytest.raises(ValueError, match="outside the reported search budget"):
        VERIFIER.verify_entry(entry)


def test_exact_determinant_handles_pivoting_and_singular_matrices():
    assert VERIFIER.det_bareiss([[0, 2], [3, 4]]) == -6
    assert VERIFIER.det_bareiss([[1, 2], [2, 4]]) == 0
    assert VERIFIER.det_bareiss([[1, 1, 1], [1, -1, 1], [1, 1, -1]]) == 4
