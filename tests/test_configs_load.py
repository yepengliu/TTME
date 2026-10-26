"""Every shipped experiment config loads and validates, and each task has the full set of arms."""
from pathlib import Path

import pytest

from ttt.config import load_experiment, to_dict, validate
from ttt.tasks import TASK_NAMES

EXPERIMENTS = Path(__file__).resolve().parents[1] / "configs" / "experiments"
CONFIGS = sorted(EXPERIMENTS.glob("*.yaml"))

# arm -> (model config, memory on, guidelines on, memo on)
ARMS = {
    "baseline": ("gpt-oss-120b", False, False, False),
    "memory": ("gpt-oss-120b", True, True, True),
    "memory_no_guidelines": ("gpt-oss-120b", True, False, True),
    "memory_no_memo": ("gpt-oss-120b", True, True, False),
    "memory_recipe_only": ("gpt-oss-120b", True, False, False),
    "baseline_8b": ("8b-chatml", False, False, False),
    "memory_8b": ("8b-chatml", True, True, True),
}


def _split(path: Path) -> tuple[str, str]:
    task = next(t for t in TASK_NAMES if path.stem.startswith(f"{t}_"))
    return task, path.stem[len(task) + 1:]


def test_each_task_has_every_arm():
    assert {p.stem for p in CONFIGS} == {f"{t}_{arm}" for t in TASK_NAMES for arm in ARMS}


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_loads_as_its_arm(path):
    exp = load_experiment(path)
    validate(exp)
    task, arm = _split(path)
    model, memory, guidelines, memo = ARMS[arm]
    assert (exp.name, exp.task, exp.model.name) == (path.stem, task, model)
    assert (exp.memory is not None, exp.memo is not None) == (memory, memo)
    if memory:
        assert exp.memory.guidelines == guidelines


@pytest.mark.parametrize("task", TASK_NAMES)
def test_arms_of_a_task_share_search_and_eval(task):
    exps = [load_experiment(p) for p in CONFIGS if _split(p)[0] == task]
    assert len({(e.search, e.eval, e.sampling) for e in exps}) == 1


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_memo_configs_use_deepen_v2(path):
    exp = load_experiment(path)
    if exp.memo is not None:
        assert exp.memo.control_card == "deepen_v2"


@pytest.mark.parametrize("task", TASK_NAMES)
def test_no_guidelines_differs_from_memory_only_by_guidelines(task):
    full = to_dict(load_experiment(EXPERIMENTS / f"{task}_memory.yaml"))
    ablated = to_dict(load_experiment(EXPERIMENTS / f"{task}_memory_no_guidelines.yaml"))
    assert (full["memory"].pop("guidelines"), ablated["memory"].pop("guidelines")) == (True, False)
    assert (full.pop("name"), ablated.pop("name")) == (f"{task}_memory", f"{task}_memory_no_guidelines")
    assert full == ablated
