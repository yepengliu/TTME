import hashlib
import json

import pytest

from ttt import records
from ttt.config import Experiment, Model, Runtime
from ttt.records import RunPaths

EXP = Experiment(name="x", task="ac1", model=Model(name="m", format="chatml_think"))


def test_rollout_row_truncates_and_hashes():
    row = records.rollout_row(step=1, rollout=2, parent_id="p", parent_score=0.5, child_id=None,
                              child_score=None, ok=False, admitted=False, beats_parent=False,
                              error="e" * 500, doc=None, code="x = 1", stdout="s" * 3000,
                              arm="none", tags={"k": 1}, infra_error=True)
    assert len(row["error"]) == 200 and len(row["stdout_tail"]) == 2000
    assert row["code_sha256"] == hashlib.sha256(b"x = 1").hexdigest() and row["infra_error"] is True
    empty = records.rollout_row(step=0, rollout=0, parent_id="p", parent_score=None,
                                child_id="c", child_score=1.0, ok=True, admitted=True,
                                beats_parent=True, error=None, doc="d", code=None, stdout="",
                                arm="a", tags={}, infra_error=False)
    assert empty["error"] is None and empty["stdout_tail"] is None and empty["code_sha256"] is None
    odd = records.rollout_row(step=0, rollout=0, parent_id="p", parent_score=None, child_id=None,
                              child_score=None, ok=False, admitted=False, beats_parent=False,
                              error="x", doc=None, code="s = '\ud800'", stdout="", arm="a",
                              tags={}, infra_error=False)
    assert len(odd["code_sha256"]) == 64


def test_truncate_drops_later_steps_and_torn_tail(tmp_path):
    path = tmp_path / "r.jsonl"
    records.append_jsonl(path, [{"step": 0}, {"step": 1}, {"step": 2}])
    with open(path, "a") as f:
        f.write('{"step": 2, "x"')
    assert records.truncate_jsonl(path, 1) == 1
    assert records.read_jsonl(path) == [{"step": 0}, {"step": 1}]
    path.write_text('{"step": 0}\nnot json\n{"step": 1}\n')
    with pytest.raises(ValueError):
        records.read_jsonl(path)


def test_claim_out_dir(tmp_path):
    out = tmp_path / "run"
    records.claim_out_dir(str(out), resume=False)
    assert out.is_dir()
    (out / "events.jsonl").write_text('{"event": "waiting_for_server"}\n')
    assert not records.has_run_state(out)
    records.claim_out_dir(str(out), resume=False)
    (out / "something").write_text("x")
    assert records.has_run_state(out)
    with pytest.raises(FileExistsError):
        records.claim_out_dir(str(out), resume=False)
    with pytest.raises(FileNotFoundError):
        records.claim_out_dir(str(out), resume=True)


def test_manifest_has_no_secret_and_checks_hash(tmp_path):
    paths = RunPaths.of(tmp_path)
    runtime = Runtime(out_dir=str(tmp_path), vllm_urls=("http://h:1",), served_model="m",
                      api_key="TOPSECRET")
    records.write_manifest(paths, EXP, runtime)
    text = paths.manifest.read_text()
    assert "TOPSECRET" not in text and "api_key" not in text
    manifest = json.loads(text)
    assert manifest["config"]["task"] == "ac1" and manifest["version"]
    assert records.check_manifest(paths, EXP)["config_hash"] == manifest["config_hash"]
    assert records.check_manifest(paths, Experiment(name="renamed", task="ac1", model=EXP.model))
    with pytest.raises(ValueError):
        records.check_manifest(paths, Experiment(name="x", task="ac2", model=EXP.model))


def test_checkpoints_and_discard(tmp_path):
    paths = RunPaths.of(tmp_path)
    records.write_checkpoint(paths.checkpoint(0), {"archive": {"a": 0}})
    records.write_checkpoint(paths.checkpoint(1), {"archive": {"a": 1}, "memo": {"m": 1}})
    paths.checkpoint(2).mkdir(parents=True)
    (paths.checkpoint(2) / "archive.json").write_text("{}")
    assert records.latest_committed(paths) == 1
    assert records.read_checkpoint(paths.checkpoint(1)) == {"archive": {"a": 1}, "memo": {"m": 1}}
    with pytest.raises(FileNotFoundError):
        records.read_checkpoint(paths.checkpoint(2))
    with pytest.raises(FileExistsError):
        records.write_checkpoint(paths.checkpoint(1), {"archive": {}})
    records.append_jsonl(paths.steps, [{"step": s, "valid": 1, "failed": 0, "infra": 1}
                                       for s in range(3)])
    records.append_jsonl(paths.rollouts, [{"step": s} for s in (0, 0, 1, 1, 2)])
    assert records.discard_after(paths, 1) == {"rollouts": 1, "steps": 1, "checkpoints": 1}
    assert not paths.checkpoint(2).exists()
    assert [s["step"] for s in records.check_logs(paths, 1)] == [0, 1]


def test_check_logs_rejects_mismatch(tmp_path):
    paths = RunPaths.of(tmp_path)
    records.append_jsonl(paths.steps, [{"step": 0, "valid": 1, "failed": 1, "infra": 1}])
    records.append_jsonl(paths.rollouts, [{"step": 0}, {"step": 0}])
    with pytest.raises(ValueError, match="3 rows"):
        records.check_logs(paths, 0)
    with pytest.raises(ValueError, match="ends at"):
        records.check_logs(paths, 1)


def test_run_state_ignores_a_lone_event_log(tmp_path):
    out = tmp_path / "run"
    assert not records.has_run_state(out)
    out.mkdir()
    (out / "events.jsonl").write_text("{}\n")
    assert not records.has_run_state(out)
    for name in ("manifest.json", "steps.jsonl", "rollouts.jsonl", "checkpoints"):
        other = tmp_path / name.split(".")[0]
        other.mkdir()
        (other / "events.jsonl").write_text("{}\n")
        (other / name).write_text("")
        assert records.has_run_state(other), name
