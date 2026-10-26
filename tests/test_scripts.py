"""Cluster and local launch scripts, run with stand-ins for sbatch, vllm and `python -m ttt`."""
import os
import shutil
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
pytestmark = pytest.mark.skipif(
    not Path("/proc/net/tcp").exists() or not all(shutil.which(t) for t in ("bash", "curl", "ps")),
    reason="needs Linux /proc, bash, curl and ps")

FAKE_VLLM = """#!{python}
# Stand-in for `vllm serve`: answers /health on --port, binding with SO_REUSEPORT as vLLM does.
import os, socket, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1])
record = os.environ["FAKE_RECORD"]
with open(os.path.join(record, f"{{port}}.env"), "w") as f:
    f.write(os.environ.get("CUDA_VISIBLE_DEVICES", "") + "\\n" + os.environ.get("VLLM_API_KEY", "") + "\\n"
            + str(os.getpid()) + "\\n")
if os.environ.get("FAKE_VLLM_FAIL"):
    sys.exit(1)
print(f"fake vllm serving on {{port}}", flush=True)
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()
    def log_message(self, *a):
        pass
class Server(HTTPServer):
    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        super().server_bind()
Server(("0.0.0.0", port), Handler).serve_forever()
"""

# Stand-in for the interpreter: `-m ttt ...` records its arguments, anything else runs for real.
FAKE_PYTHON = """#!/bin/bash
if [[ $1 == -m ]]; then
  printf '%s\\n' "$@" > "$FAKE_RECORD/argv"
  printf '%s\\n' "${{TTT_API_KEY:-}}" > "$FAKE_RECORD/api_key"
  if [[ -n ${{FAKE_ENDPOINT_ARG:-}} ]]; then
    for ((i = 1; i <= $#; i++)); do
      if [[ ${{!i}} == --endpoint-file ]]; then j=$((i + 1)); echo '{{}}' > "${{!j}}"; fi
    done
  fi
  [[ -z ${{FAKE_SLEEP:-}} ]] || exec sleep "$FAKE_SLEEP"
  exit "${{FAKE_RC:-0}}"
fi
exec {python} "$@"
"""

FAKE_SBATCH = """#!/bin/bash
# Stand-in for sbatch: records each call's arguments and prints the next job id.
n=$(( $(ls "$FAKE_RECORD"/sbatch_* 2> /dev/null | wc -l) + 1 ))
printf '%s\\n' "$@" > "$FAKE_RECORD/sbatch_$n"
echo $(( 1000 + n ))
"""


def _executable(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def env(tmp_path):
    """Clean environment (no Slurm job variables leak in) with the stand-ins installed."""
    record = tmp_path / "record"
    record.mkdir()
    e = {k: v for k, v in os.environ.items()
         if not k.startswith(("SLURM_", "SBATCH_")) and k != "CUDA_VISIBLE_DEVICES"}
    e.update(FAKE_RECORD=str(record), POLL_S="0.2", TTT_SCRIPTS=str(SCRIPTS),
             VLLM=str(_executable(tmp_path / "vllm", FAKE_VLLM.format(python=sys.executable))),
             PYTHON=str(_executable(tmp_path / "python", FAKE_PYTHON.format(python=sys.executable))),
             SBATCH=str(_executable(tmp_path / "sbatch", FAKE_SBATCH)))
    return e


@pytest.fixture
def run_dir(tmp_path):
    rd = tmp_path / "run_dir"
    (rd / "servers").mkdir(parents=True)
    (rd / "logs").mkdir()
    (rd / "api_key").write_text("k" * 48 + "\n")
    return rd


def _wait_for(predicate, timeout_s=20.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False


def _bash(script, env, *args):
    return subprocess.run(["bash", str(script), *args], env=env, capture_output=True, text=True,
                          timeout=60)


def _listener(port: int = 0) -> subprocess.Popen:
    code = ("import socket, sys, time\n"
            "s = socket.socket()\n"
            "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)\n"
            f"s.bind(('127.0.0.1', {port})); s.listen()\n"
            "print(s.getsockname()[1], flush=True)\n"
            "time.sleep(60)\n")
    return subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)


def test_owns_port_rejects_a_shared_port():
    lib = SCRIPTS / "lib" / "serve.sh"
    first = _listener()
    stranger = subprocess.Popen(["sleep", "60"])
    second = None
    try:
        port = int(first.stdout.readline())
        check = f'source "{lib}"; owns_port {{pid}} {port}'
        assert subprocess.run(["bash", "-c", check.format(pid=first.pid)]).returncode == 0
        assert subprocess.run(["bash", "-c", check.format(pid=stranger.pid)]).returncode != 0
        second = _listener(port)
        assert int(second.stdout.readline()) == port
        assert subprocess.run(["bash", "-c", check.format(pid=first.pid)]).returncode != 0
        ports = subprocess.run(["bash", "-c", f'source "{lib}"; free_ports 3'], capture_output=True,
                               text=True, check=True).stdout.split()
        assert len(set(ports)) == 3
    finally:
        for p in (first, stranger, second):
            if p:
                p.kill()
                p.wait()


@pytest.mark.parametrize("stop", ["driver_done", "sigterm"])
def test_vllm_servers_split_slurm_gpus_publish_urls_and_stop(env, run_dir, stop):
    (run_dir / "driver.alive").touch()
    env.update(RUN_DIR=str(run_dir), MODEL_PATH="m", N_SERVERS="2", CUDA_VISIBLE_DEVICES="5,7",
               SLURM_JOB_ID="77")
    proc = subprocess.Popen(["bash", str(SCRIPTS / "slurm" / "vllm_servers.sbatch")], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert _wait_for(lambda: len(list((run_dir / "servers").glob("*.url"))) == 2), proc
        record = Path(env["FAKE_RECORD"])
        devices, server_pids = set(), []
        for url_file in (run_dir / "servers").glob("77_*.url"):
            port = url_file.read_text().strip().rsplit(":", 1)[1]
            cuda, key, pid = (record / f"{port}.env").read_text().splitlines()
            devices.add(cuda)
            server_pids.append(int(pid))
            assert key == "k" * 48
        assert devices == {"5", "7"}
        if stop == "driver_done":
            (run_dir / "driver.done").touch()
        else:
            proc.terminate()
        out, err = proc.communicate(timeout=20)
    finally:
        proc.kill()
    if stop == "driver_done":
        assert proc.returncode == 0, err
        assert not (run_dir / "servers" / "FAILED").exists()
    else:
        assert proc.returncode == 143
        assert "SIGTERM" in (run_dir / "servers" / "FAILED").read_text()
    assert not list((run_dir / "servers").glob("*.url"))
    for pid in server_pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


@pytest.mark.parametrize("case", ["server_exits", "too_few_gpus", "no_driver", "silent_driver"])
def test_vllm_servers_write_failed(env, run_dir, case):
    env.update(RUN_DIR=str(run_dir), MODEL_PATH="m", N_SERVERS="2", CUDA_VISIBLE_DEVICES="0,1",
               DRIVER_START_WAIT_S="1")
    if case != "no_driver":
        (run_dir / "driver.alive").touch()
    if case == "server_exits":
        env["FAKE_VLLM_FAIL"] = "1"
    elif case == "too_few_gpus":
        env["GPUS_PER_SERVER"] = "2"
    elif case == "silent_driver":
        old = time.time() - 3600
        os.utime(run_dir / "driver.alive", (old, old))
    r = _bash(SCRIPTS / "slurm" / "vllm_servers.sbatch", env)
    assert r.returncode != 0
    failed = (run_dir / "servers" / "FAILED").read_text()
    expected = {"server_exits": "exited during startup", "too_few_gpus": "= 4 GPUs",
                "no_driver": "did not start", "silent_driver": "has not heartbeated"}[case]
    assert expected in failed, failed
    assert not list((run_dir / "servers").glob("*.url"))


@pytest.mark.parametrize("resume", ["0", "1"])
def test_driver_runs_on_the_endpoint_directory(env, run_dir, tmp_path, resume):
    for i, port in enumerate((9001, 9002)):
        (run_dir / "servers" / f"5_{i}.url").write_text(f"http://h:{port}\n")
    env.update(CONFIG="c.yaml", RUN_DIR=str(run_dir), POOL_DIR=str(tmp_path / "pool"), N_SERVERS="2",
               RESUME=resume, STOP_POOL_ON_EXIT="1")
    (tmp_path / "pool").mkdir()
    r = _bash(SCRIPTS / "slurm" / "driver.sbatch", env)
    assert r.returncode == 0, r.stderr
    argv = (Path(env["FAKE_RECORD"]) / "argv").read_text().splitlines()
    assert argv[argv.index("--eval-workers") + 1] == str(tmp_path / "pool" / "endpoints")
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "--vllm-url"] == ["http://h:9001", "http://h:9002"]
    assert ("--resume" in argv) == (resume == "1")
    assert (Path(env["FAKE_RECORD"]) / "api_key").read_text().strip() == "k" * 48
    assert (run_dir / "driver.done").exists() and (run_dir / "driver.alive").exists()
    assert (tmp_path / "pool" / "STOP").exists()


@pytest.mark.parametrize("when", ["before", "during"])
def test_driver_gives_up_on_failed(env, run_dir, tmp_path, when):
    (run_dir / "servers" / "5_0.url").write_text("http://h:9001\n")
    env.update(CONFIG="c.yaml", RUN_DIR=str(run_dir), POOL_DIR=str(tmp_path), N_SERVERS="1",
               FAKE_SLEEP="30")
    if when == "before":
        (run_dir / "servers" / "FAILED").write_text("vLLM job 5: boom\n")
        r = _bash(SCRIPTS / "slurm" / "driver.sbatch", env)
    else:
        proc = subprocess.Popen(["bash", str(SCRIPTS / "slurm" / "driver.sbatch")], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert _wait_for(lambda: (Path(env["FAKE_RECORD"]) / "argv").exists())
        (run_dir / "servers" / "FAILED").write_text("vLLM job 5: boom\n")
        out, err = proc.communicate(timeout=15)
        r = subprocess.CompletedProcess(proc.args, proc.returncode, out, err)
    assert r.returncode == 1
    assert "giving up: vLLM job 5: boom" in r.stderr
    assert (Path(env["FAKE_RECORD"]) / "argv").exists() == (when == "during")
    assert (run_dir / "driver.done").exists()


def test_eval_worker_exit_status(env, tmp_path):
    pool = tmp_path / "pool"
    (pool / "endpoints").mkdir(parents=True)
    env.update(POOL_DIR=str(pool), SLURM_JOB_ID="42", SLURM_CPUS_PER_TASK="2")
    crashed = _bash(SCRIPTS / "slurm" / "eval_worker.sbatch", dict(env, FAKE_RC="3"))
    assert crashed.returncode == 3, crashed.stderr
    proc = subprocess.Popen(["bash", str(SCRIPTS / "slurm" / "eval_worker.sbatch")],
                            env=dict(env, FAKE_SLEEP="30", FAKE_ENDPOINT_ARG="1"),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert _wait_for(lambda: (pool / "endpoints" / "42.json").exists())
    (pool / "STOP").touch()
    proc.communicate(timeout=15)
    assert proc.returncode == 0
    assert not (pool / "endpoints" / "42.json").exists()


def test_submit_pool_and_run(env, tmp_path):
    pool = tmp_path / "a" / "b" / "pool"
    r = _bash(SCRIPTS / "slurm" / "submit_pool.sh", env, "4", str(pool))
    assert r.returncode == 0, r.stderr
    assert stat.S_IMODE(pool.stat().st_mode) == 0o700
    assert (pool / "size").read_text().strip() == "4" and (pool / "jobid").read_text().strip() == "1001"
    assert "already holds a pool" in _bash(SCRIPTS / "slurm" / "submit_pool.sh", env, "4", str(pool)).stderr
    record = Path(env["FAKE_RECORD"])
    shutil.rmtree(record)
    record.mkdir()

    config = tmp_path / "exp.yaml"
    config.write_text("task: ac1\n")
    run_dir = tmp_path / "runs" / "r1"
    env.update(CONFIG=str(config), RUN_DIR=str(run_dir), POOL_DIR=str(pool), MODEL_PATH="m")
    submit = SCRIPTS / "slurm" / "submit_run.sh"
    (pool / "endpoints" / "1.json").write_text("{}")
    r = _bash(submit, env)
    assert r.returncode != 0 and "only 1 of 2" in r.stderr  # default MIN_WORKERS: half of 4
    (pool / "endpoints" / "2.json").write_text("{}")
    assert "holds no run" in _bash(submit, dict(env, RESUME="1")).stderr
    assert "commas" in _bash(submit, dict(env, MODEL_PATH="a,b")).stderr
    r = _bash(submit, env)
    assert r.returncode == 0, r.stderr
    assert stat.S_IMODE(run_dir.stat().st_mode) == 0o700
    driver = (record / "sbatch_1").read_text().splitlines()
    vllm = (record / "sbatch_2").read_text().splitlines()
    assert driver[-1].endswith("driver.sbatch") and vllm[-1].endswith("vllm_servers.sbatch")
    assert "--dependency=after:1001" in vllm and "--gpus=4" in vllm
    driver_export = next(a for a in driver if a.startswith("--export="))
    assert "RESUME=0" in driver_export and f"POOL_DIR={pool}" in driver_export
    assert f"TTT_SCRIPTS={SCRIPTS}" in next(a for a in vllm if a.startswith("--export="))

    (run_dir / "run").mkdir()
    (run_dir / "run" / "manifest.json").write_text("{}")
    (run_dir / "driver.done").touch()
    assert "RESUME=1 to continue" in _bash(submit, env).stderr
    assert _bash(submit, dict(env, RESUME="1")).returncode == 0
    assert not (run_dir / "driver.done").exists()


def test_run_local(env, tmp_path):
    out = tmp_path / "out"
    r = _bash(SCRIPTS / "local" / "run_local.sh", dict(env, MODEL_PATH="m"), "c.yaml", str(out), "3", "--extra")
    assert r.returncode == 0, r.stderr
    record = Path(env["FAKE_RECORD"])
    argv = record.joinpath("argv").read_text().splitlines()
    port = argv[argv.index("--vllm-url") + 1].rsplit(":", 1)[1]
    key = record.joinpath("api_key").read_text().strip()
    assert len(key) == 48 and record.joinpath(f"{port}.env").read_text().splitlines()[1] == key
    assert argv[argv.index("--local-eval") + 1] == "3" and argv[-1] == "--extra"
    assert argv[argv.index("--api-key-env") + 1] == "TTT_API_KEY"
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        busy = s.getsockname()[1]
        r = _bash(SCRIPTS / "local" / "run_local.sh", dict(env, MODEL_PATH="m", PORT=str(busy)),
                  "c.yaml", str(out))
    assert r.returncode != 0 and "already in use" in r.stderr


@pytest.mark.parametrize("vllm_log", ["default", "custom"])
def test_run_local_creates_missing_parent_directories(env, tmp_path, vllm_log):
    """As in the README example from a fresh checkout: neither OUT_DIR's parents nor VLLM_LOG's exist."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    out = "runs/nested/erdos_baseline"
    e = dict(env, MODEL_PATH="m")
    if vllm_log == "custom":
        log = tmp_path / "logs" / "a" / "b" / "server.log"
        e["VLLM_LOG"] = str(log)
    else:
        log = checkout / f"{out}.vllm.log"
    r = subprocess.run(["bash", str(SCRIPTS / "local" / "run_local.sh"), "c.yaml", out, "2"], cwd=checkout,
                       env=e, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    argv = (Path(env["FAKE_RECORD"]) / "argv").read_text().splitlines()
    assert argv[argv.index("--out-dir") + 1] == out
    assert (checkout / "runs" / "nested").is_dir()
    port = argv[argv.index("--vllm-url") + 1].rsplit(":", 1)[1]
    assert log.read_text() == f"fake vllm serving on {port}\n"
