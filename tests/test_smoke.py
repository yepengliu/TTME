import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from ttt import cli, log, loop
from ttt.archive import ConstructionStore
from ttt.config import Eval, Experiment, Model, Runtime, Sampling, Search
from ttt.llm import build_format
from ttt.llm.fake import FakeClient, harmony_completion, program_responder

tasks = pytest.importorskip("ttt.tasks", reason="ttt.tasks is not available yet")
local = pytest.importorskip("ttt.eval.local", reason="ttt.eval.local is not available yet")

PROGRAMS = {
    "ac1": '"""Flat start."""\n\ndef propose_candidate():\n    return [1.0] * 1200\n',
    "ac2": '"""Flat start."""\n\ndef construct_function():\n    return [1.0] * 1200\n',
    "erdos": '"""Half everywhere."""\n\ndef run():\n    n = 50\n    return [0.5] * n, 0.5, n\n',
}
MODEL = Model(name="m", format="harmony_two_phase", phase1_max_tokens=26000, date="2026-07-24")


@pytest.mark.parametrize("task_name", sorted(PROGRAMS))
def test_two_steps_end_to_end(tmp_path, task_name):
    task = tasks.get(task_name)
    exp = Experiment(name=f"smoke-{task_name}", task=task_name, model=MODEL, sampling=Sampling(),
                     search=Search(steps=2, parents_per_step=2, children_per_parent=2, seed=0),
                     eval=Eval(timeout_s=60.0, cpus=1))
    out = tmp_path / "run"
    runtime = Runtime(out_dir=str(out), vllm_urls=("fake",), served_model="m", gen_concurrency=4)
    client = FakeClient(program_responder("harmony_two_phase", PROGRAMS[task_name]))
    result = loop.run(exp, runtime, task=task, evaluator=local.LocalEvaluator(2), clients=[client],
                      fmt=build_format(MODEL, exp.sampling))
    rows = [json.loads(line) for line in (out / "rollouts.jsonl").read_text().splitlines()]
    assert len(rows) == 8
    assert all(r["ok"] for r in rows), [r["error"] for r in rows if not r["ok"]]
    assert all(r["doc"] in ("Flat start.", "Half everywhere.") for r in rows)
    assert (out / "checkpoints/step_000001/COMMITTED").exists()
    archive = json.loads((out / "checkpoints/step_000001/archive.json").read_text())
    child = next(s for s in archive["states"] if s["id"] == rows[0]["child_id"])
    construction = ConstructionStore(out / "constructions").get(child["construction_hash"])
    assert tasks.score_sequence(task_name, construction) == pytest.approx(result["best"], rel=1e-9)


class _VLLMStub(BaseHTTPRequestHandler):
    program = PROGRAMS["erdos"]
    paths: list = []

    def log_message(self, *args):
        pass

    def _json(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.paths.append((self.path, self.headers.get("Authorization")))
        self._json({})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.paths.append((self.path, self.headers.get("Authorization")))
        if self.path == "/tokenize":
            self._json({"count": len(body["prompt"]) // 4})
        else:
            self._json({"choices": [{"text": harmony_completion(self.program), "finish_reason": "stop"}]})


@pytest.fixture
def vllm_stub():
    _VLLMStub.paths = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _VLLMStub)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()
    log.configure(None)


def write_config(tmp_path, name, steps=1, parents=2):
    (tmp_path / "models").mkdir(exist_ok=True)
    (tmp_path / "experiments").mkdir(exist_ok=True)
    (tmp_path / "models" / "oss.yaml").write_text(
        "format: harmony_two_phase\nphase1_max_tokens: 26000\ndate: '2026-07-24'\n")
    config = tmp_path / "experiments" / f"{name}.yaml"
    config.write_text(f"task: erdos\nmodel: oss\nsearch:\n  steps: {steps}\n  parents_per_step: "
                      f"{parents}\n  children_per_parent: 2\n  seed: 1\neval:\n  timeout_s: 60\n"
                      "  cpus: 1\n")
    return config


def test_cli_run_and_resume(tmp_path, vllm_stub, monkeypatch, capsys):
    config = write_config(tmp_path, "erdos_cli")
    monkeypatch.setenv("STUB_KEY", "k3y")
    out = tmp_path / "run"
    argv = ["run", "--config", str(config), "--vllm-url", vllm_stub, "--vllm-url", vllm_stub + "/",
            "--served-model", "m", "--out-dir", str(out), "--local-eval", "2",
            "--api-key-env", "STUB_KEY"]
    assert cli.main(argv) == 0
    final = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert final["experiment"] == "erdos_cli" and final["steps"] == 1
    assert final["best"] == pytest.approx(0.5) and final["best_valid"] == pytest.approx(0.5)
    assert {auth for _, auth in _VLLMStub.paths} == {"Bearer k3y"}
    events = [json.loads(line)["event"] for line in (out / "events.jsonl").read_text().splitlines()]
    assert events[0] == "start" and "step" in events
    assert "k3y" not in (out / "manifest.json").read_text() + (out / "events.jsonl").read_text()
    with pytest.raises(FileExistsError):
        cli.main(argv)
    assert cli.main(argv + ["--resume"]) == 0
    assert json.loads(capsys.readouterr().out.strip())["steps"] == 1


def test_cli_retries_after_a_failed_server_wait(tmp_path, vllm_stub, monkeypatch, capsys):
    config = write_config(tmp_path, "erdos_retry", parents=1)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        dead = f"http://127.0.0.1:{sock.getsockname()[1]}"
    out = tmp_path / "run"
    argv = ["run", "--config", str(config), "--served-model", "m", "--out-dir", str(out),
            "--local-eval", "1", "--no-save-code"]
    monkeypatch.setattr(cli, "READY_TIMEOUT_S", 0.0)
    with pytest.raises(RuntimeError, match="not ready"):
        cli.main(argv + ["--vllm-url", dead])
    assert [p.name for p in out.iterdir()] == ["events.jsonl"]
    assert cli.main(argv + ["--vllm-url", vllm_stub]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["steps"] == 1
    assert json.loads((out / "manifest.json").read_text())["runtime"]["save_code"] is False
    lines = (out / "rollouts.jsonl").read_text().splitlines()
    shas = [json.loads(line)["code_sha256"] for line in lines]
    assert shas and not any((out / "codes" / h[:2] / f"{h}.py").exists() for h in shas)
    archive = json.loads((out / "checkpoints/step_000000/archive.json").read_text())
    assert all((out / "codes" / s["code_sha256"][:2] / f"{s['code_sha256']}.py").exists()
               for s in archive["states"]), "archived programs are stored even with --no-save-code"


def test_cli_passes_the_pool_path_and_resizes_each_step(tmp_path, vllm_stub, monkeypatch, capsys):
    import ttt.eval.remote as remote

    paths, sizes = [], []

    class GrowingPool:
        def __init__(self, path):
            paths.append(path)
            self.inner, self.calls, self.lock = local.LocalEvaluator(2), 0, threading.Lock()

        @property
        def capacity(self):
            with self.lock:
                return 1 + self.calls

        def evaluate(self, *args):
            with self.lock:
                self.calls += 1
            return self.inner.evaluate(*args)

    def executor(n):
        sizes.append(n)
        return real_executor(n)

    real_executor = loop.ThreadPoolExecutor
    monkeypatch.setattr(remote, "RemoteEvaluator", GrowingPool)
    monkeypatch.setattr(loop, "ThreadPoolExecutor", executor)
    pool = tmp_path / "pool" / "endpoints"
    assert cli.main(["run", "--config", str(write_config(tmp_path, "erdos_pool", steps=2, parents=1)),
                     "--vllm-url", vllm_stub, "--served-model", "m", "--out-dir", str(tmp_path / "run"),
                     "--eval-workers", str(pool), "--gen-concurrency", "8"]) == 0
    assert paths == [str(pool)]
    assert sizes == [8, 1, 8, 3], "the eval pool is sized from the capacity of each step"
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["steps"] == 2


def test_cli_score(capsys):
    seq = Path(__file__).resolve().parents[1] / "results" / "erdos" / "sequence.json"
    if not seq.exists():
        pytest.skip("results/erdos/sequence.json is not in this checkout")
    assert cli.main(["score", "erdos", str(seq)]) == 0
    stored = json.loads(seq.read_text())["score"]
    assert float(capsys.readouterr().out) == pytest.approx(stored, rel=1e-12)
