import pytest

from ttt.config import ConfigError, Experiment, build, config_hash, validate

MODEL = {"name": "m", "format": "harmony_two_phase", "phase1_max_tokens": 26000}


def test_unknown_key_rejected():
    with pytest.raises(ConfigError):
        build(Experiment, {"name": "x", "task": "ac1", "model": MODEL, "serch": {}})


def test_memo_requires_memory():
    exp = build(Experiment, {"name": "x", "task": "ac2", "model": MODEL, "memo": {}})
    with pytest.raises(ConfigError):
        validate(exp)


def test_hash_is_stable_and_sensitive():
    a = build(Experiment, {"name": "x", "task": "ac2", "model": MODEL, "memory": {}, "memo": {}})
    b = build(Experiment, {"name": "x", "task": "ac2", "model": MODEL, "memory": {}, "memo": {"select": "parallel"}})
    renamed = build(Experiment, {"name": "y", "task": "ac2", "model": MODEL, "memory": {}, "memo": {}})
    validate(a)
    assert config_hash(a) == config_hash(a) == config_hash(renamed)
    assert config_hash(a) != config_hash(b)


def test_bad_literal_rejected():
    with pytest.raises(ConfigError):
        build(Experiment, {"name": "x", "task": "ac3", "model": MODEL})


@pytest.mark.parametrize("override", [
    {"memo": {"boundary_every": 0}},
    {"memo": {"notes_per_boundary": 0}},
    {"memo": {"methods_per_parent": 0}},
    {"memo": {"selected_children": 0}},
    {"model": dict(MODEL, phase1_max_tokens=32768)},
    {"sampling": {"context_window": 26000}},
    {"eval": {"timeout_s": 0}},
    {"eval": {"timeout_s": -5.0}},
    {"eval": {"cpus": 0}},
    {"search": {"steps": 0}},
    {"search": {"parents_per_step": 0}},
    {"search": {"children_per_parent": 0}},
    {"search": {"archive_capacity": 0}},
    {"search": {"topk_children": -1}},
])
def test_validate_rejects_bad_values(override):
    exp = build(Experiment, {"name": "x", "task": "ac2", "model": MODEL, "memory": {}, "memo": {},
                             **override})
    with pytest.raises(ConfigError):
        validate(exp)


def test_validate_accepts_defaults():
    validate(build(Experiment, {"name": "x", "task": "ac2", "model": MODEL, "memory": {}, "memo": {}}))
    validate(build(Experiment, {"name": "x", "task": "erdos", "search": {"topk_children": 0},
                                "model": {"name": "m", "format": "chatml_think"}}))
