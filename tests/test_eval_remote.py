import json
import os
import socket
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import requests

from ttt import tasks
from ttt.eval import remote, worker
from ttt.eval.remote import Endpoint, RemoteEvaluator, load_endpoints
from ttt.types import State

AC1_CODE = "def propose_candidate():\n    print('remote')\n    return [1.0] * 1100\n"


def ac1_program():
    parent = State(id="p", timestep=0, value=None, code="", construction=[0.5] * 1000)
    return tasks.get("ac1").assemble(AC1_CODE, parent)


@pytest.fixture
def server():
    srv = worker.make_server("127.0.0.1", 0, capacity=2, token="secret-token")
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv, f"127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def closed_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_round_trip_through_a_real_worker(server):
    _, addr = server
    health = requests.get(f"http://{addr}/health", timeout=5).json()
    assert health == {"ok": True, "capacity": 2, "active": 0}
    ev = RemoteEvaluator([Endpoint(addr=addr, token="secret-token", capacity=2)])
    assert ev.capacity == 2
    result = ev.evaluate("ac1", ac1_program(), 30, 1)
    assert result.ok and result.raw_score == 2.0 and result.stdout == "remote\n"
    assert result.construction == [1.0] * 1100


def test_bad_token_is_rejected(server):
    _, addr = server
    reply = requests.post(f"http://{addr}/eval", timeout=5, headers={"X-Eval-Token": "nope"},
                          json={"task": "ac1", "program": "", "timeout_s": 5, "cpus": 1})
    assert reply.status_code == 401
    ev = RemoteEvaluator([f"{addr}#wrong"])
    result = ev.evaluate("ac1", ac1_program(), 30, 1)
    assert not result.ok and "unauthorized" in result.error
    assert ev.status()[0]["failures"] == 1


def test_bad_request_is_rejected(server):
    _, addr = server
    for body in ({"task": "ac9", "program": "", "timeout_s": 5, "cpus": 1},
                 {"task": "ac1", "program": "", "timeout_s": 0, "cpus": 1},
                 {"task": "ac1", "program": 3, "timeout_s": 5, "cpus": 1}):
        reply = requests.post(f"http://{addr}/eval", json=body, timeout=5,
                              headers={"X-Eval-Token": "secret-token"})
        assert reply.status_code == 400 and reply.json()["error"].startswith("bad request")


def test_dead_endpoint_is_skipped(server):
    _, addr = server
    dead = Endpoint(addr=f"127.0.0.1:{closed_port()}", token="x", capacity=4)
    ev = RemoteEvaluator([dead, Endpoint(addr=addr, token="secret-token", capacity=2)])
    result = ev.evaluate("ac1", ac1_program(), 30, 1)
    assert result.ok and result.raw_score == 2.0
    status = ev.status()
    assert status[0]["failures"] == 1 and status[0]["down_for_s"] > 0
    assert [s["free"] for s in status] == [4, 2]


def test_capacity_probed_for_literal_endpoints(server):
    _, addr = server
    assert RemoteEvaluator([f"{addr}#secret-token"]).capacity == 2
    assert RemoteEvaluator([f"127.0.0.1:{closed_port()}#t"], connect_timeout_s=1).capacity == 1


def test_worker_main_serves_from_its_endpoint_file(tmp_path):
    endpoint = tmp_path / "pool" / "w0.json"
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    proc = subprocess.Popen([sys.executable, "-m", "ttt.eval.worker", "--capacity", "2",
                             "--endpoint-file", str(endpoint)], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 20
        while not endpoint.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert endpoint.exists() and stat.S_IMODE(endpoint.stat().st_mode) == 0o600
        info = json.loads(endpoint.read_text())
        assert info["addr"].startswith("127.0.0.1:") and info["pid"] == proc.pid
        ev = RemoteEvaluator(endpoint.parent)
        assert ev.capacity == 2
        result = ev.evaluate("ac1", ac1_program(), 30, 1)
        assert result.ok and result.raw_score == 2.0
    finally:
        proc.terminate()
        proc.wait(10)


def test_endpoint_file_and_endpoint_list(tmp_path):
    path = tmp_path / "workers" / "w1.endpoint"
    worker.write_endpoint_file(str(path), "node7:4242", "tok", 16)
    data = json.loads(path.read_text())
    assert data == {"addr": "node7:4242", "token": "tok", "capacity": 16, "pid": os.getpid()}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert [p.name for p in path.parent.iterdir()] == ["w1.endpoint"]
    listing = tmp_path / "endpoints.txt"
    listing.write_text("# pool\nworkers/w1.endpoint\n\nnode8:99#abc\nworkers/missing.endpoint\n")
    eps = load_endpoints(listing)
    assert [(e.addr, e.token, e.capacity) for e in eps] == [("node7:4242", "tok", 16),
                                                            ("node8:99", "abc", None)]
    assert eps[0].src == str(path)
    (tmp_path / "empty.txt").write_text("# nothing\n")
    with pytest.raises(ValueError):
        load_endpoints(tmp_path / "empty.txt")
    pool = tmp_path / "pool"
    worker.write_endpoint_file(str(pool / "7.json"), "node7:1", "tok", 4)
    (pool / ".endpoint-x.tmp").write_text("{partial")
    (pool / "notes.txt").write_text("not an endpoint")
    (pool / "8.json").write_text("{partial")
    assert [(e.addr, e.src) for e in load_endpoints(pool)] == [("node7:1", str(pool / "7.json"))]


def test_advertised_addr():
    assert worker.advertised_addr("0.0.0.0", 5) == f"{socket.gethostname()}:5"
    assert worker.advertised_addr("127.0.0.1", 5) == "127.0.0.1:5"


class Reply:
    def __init__(self, body, status=200):
        self.status_code = status
        self._body = body
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        if isinstance(self._body, str):
            raise ValueError("not JSON")
        return self._body


def ok_body(addr):
    return {"ok": True, "raw_score": 1.0, "construction": [1.0], "stdout": addr, "error": None}


class FakePool:
    """Stand-in for requests.post with per-address behaviour; records every call."""

    def __init__(self, dead=(), timeouts=(), slow=None, replies=None, delay=None):
        self.dead, self.timeouts, self.slow = set(dead), set(timeouts), slow or {}
        self.replies, self.delay = replies or {}, delay or {}
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, url, json=None, timeout=None, headers=None):
        addr = url.split("//", 1)[1].split("/", 1)[0]
        with self.lock:
            self.calls.append((addr, timeout))
        if addr in self.dead:
            raise requests.ConnectionError(f"refused {addr}")
        if addr in self.timeouts:
            raise requests.ReadTimeout(f"read timeout {addr}")
        if addr in self.replies:
            return self.replies[addr]
        if addr in self.slow:
            self.slow[addr].wait(10)
        time.sleep(self.delay.get(addr, 0.0))
        return Reply(ok_body(addr))

    def hits(self, addr):
        with self.lock:
            return sum(1 for a, _ in self.calls if a == addr)


def endpoints(n, cap=4):
    return [Endpoint(addr=f"w{i}:1", capacity=cap) for i in range(n)]


def test_dead_worker_hit_once_then_skipped(monkeypatch):
    fake = FakePool(dead={"w0:1"})
    monkeypatch.setattr(remote.requests, "post", fake)
    ev = RemoteEvaluator(endpoints(4))
    for _ in range(40):
        assert ev.evaluate("ac1", "x", 10, 1).stdout in ("w1:1", "w2:1", "w3:1")
    assert fake.hits("w0:1") == 1
    assert [s["free"] for s in ev.status()] == [4, 4, 4, 4]


def test_whole_pool_dead_opens_the_circuits_and_fails_fast(monkeypatch):
    fake = FakePool(dead={"w0:1", "w1:1"})
    monkeypatch.setattr(remote.requests, "post", fake)
    ev = RemoteEvaluator(endpoints(2))
    for _ in range(12):
        result = ev.evaluate("ac1", "x", 10, 1)
        assert not result.ok and result.infra_error
        assert result.error.startswith("eval_transport_error:")
    status = ev.status()
    assert [s["failures"] for s in status] == [remote.BREAKER_FAILURES] * 2
    assert len(fake.calls) == 2 * remote.BREAKER_FAILURES
    assert all(0 < s["down_for_s"] <= remote.DOWN_MAX_S for s in status)
    assert [s["free"] for s in status] == [4, 4]


def test_cooldown_expiry_probes_and_marks_up(monkeypatch):
    monkeypatch.setattr(remote, "DOWN_BASE_S", 0.2)
    fake = FakePool(dead={"w0:1"})
    monkeypatch.setattr(remote.requests, "post", fake)
    ev = RemoteEvaluator(endpoints(2))
    ev.evaluate("ac1", "x", 10, 1)
    assert ev.status()[0]["failures"] == 1
    fake.dead.clear()
    time.sleep(0.3)
    for _ in range(4):
        ev.evaluate("ac1", "x", 10, 1)
    assert ev.status()[0]["failures"] == 0 and fake.hits("w0:1") >= 2


def test_timeouts_are_capped_and_do_not_mark_down(monkeypatch):
    fake = FakePool(timeouts={"w0:1", "w1:1", "w2:1"})
    monkeypatch.setattr(remote.requests, "post", fake)
    ev = RemoteEvaluator(endpoints(4))
    result = ev.evaluate("ac1", "x", 10, 1)
    assert not result.ok and result.infra_error and "timeout" in result.error
    assert len(fake.calls) == remote.MAX_TIMEOUTS
    assert [s["failures"] for s in ev.status()] == [0, 0, 0, 0]


def test_connect_timeout_is_short_and_read_timeout_has_margin(monkeypatch):
    fake = FakePool()
    monkeypatch.setattr(remote.requests, "post", fake)
    RemoteEvaluator(endpoints(1)).evaluate("ac1", "x", 1100, 1)
    assert fake.calls[0][1] == (15.0, 1300.0)


def test_candidate_failures_are_not_rerun_but_worker_errors_are(monkeypatch):
    calls = []
    replies = [Reply({"ok": False, "raw_score": None, "construction": None, "stdout": "s",
                      "error": "Invalid solution."}),
               Reply({"error": "worker_error: disk full"}, status=500),
               Reply(ok_body("w2:1"))]

    def post(url, json=None, timeout=None, headers=None):
        calls.append(url)
        return replies[len(calls) - 1]

    monkeypatch.setattr(remote.requests, "post", post)
    ev = RemoteEvaluator(endpoints(3))
    first = ev.evaluate("ac1", "x", 10, 1)
    assert not first.ok and first.error == "Invalid solution." and first.stdout == "s"
    assert not first.infra_error
    second = ev.evaluate("ac1", "x", 10, 1)
    assert second.ok and second.stdout == "w2:1" and len(calls) == 3
    assert [s["failures"] for s in ev.status()] == [0, 1, 0]


def run_concurrently(ev, n):
    results = []
    threads = [threading.Thread(target=lambda: results.append(ev.evaluate("ac1", "x", 10, 1)))
               for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    return results


@pytest.mark.parametrize("bad", [Reply({"error": "boom"}, status=500), Reply("<html>bad gateway", 502),
                                 Reply("not json"), Reply({"ok": "maybe"}),
                                 Reply({"error": "eval_sandbox_error: no tmp"}, status=503)],
                         ids=["500", "502-html", "non-json", "malformed", "503"])
def test_fast_failing_worker_does_not_absorb_the_load(monkeypatch, bad):
    fake = FakePool(replies={"w0:1": bad}, delay={"w1:1": 0.02})
    monkeypatch.setattr(remote.requests, "post", fake)
    ev = RemoteEvaluator(endpoints(2, cap=4))
    results = run_concurrently(ev, 80)
    assert len(results) == 80 and all(r.ok and r.stdout == "w1:1" for r in results)
    # The first wave can reach w0 before it is marked down; after that it gets no traffic.
    assert fake.hits("w0:1") <= 4 and ev.status()[0]["failures"] == 1
    assert [s["free"] for s in ev.status()] == [4, 4]


def test_every_worker_failing_is_an_infrastructure_error(monkeypatch):
    fake = FakePool(replies={a: Reply({"error": "boom"}, status=500) for a in ("w0:1", "w1:1")})
    monkeypatch.setattr(remote.requests, "post", fake)
    result = RemoteEvaluator(endpoints(2)).evaluate("ac1", "x", 10, 1)
    assert not result.ok and result.infra_error and "HTTP 500: boom" in result.error
    assert fake.hits("w0:1") == fake.hits("w1:1") == 1


def test_failovers_per_request_are_capped(monkeypatch):
    fake = FakePool(dead={f"w{i}:1" for i in range(6)})
    monkeypatch.setattr(remote.requests, "post", fake)
    result = RemoteEvaluator(endpoints(6)).evaluate("ac1", "x", 10, 1)
    assert result.infra_error and len(fake.calls) == remote.MAX_FAILOVERS


def test_rejected_request_is_infra_but_not_retried(monkeypatch):
    fake = FakePool(replies={"w0:1": Reply({"error": "bad request: task must be one of"}, status=400)})
    monkeypatch.setattr(remote.requests, "post", fake)
    ev = RemoteEvaluator(endpoints(2))
    result = ev.evaluate("ac1", "x", 10, 1)
    assert result.infra_error and result.error.startswith("eval_request_rejected: bad request")
    assert len(fake.calls) == 1 and [s["failures"] for s in ev.status()] == [0, 0]


def test_circuit_opens_after_repeated_failures_and_trials_recover_it(monkeypatch):
    monkeypatch.setattr(remote, "DOWN_BASE_S", 0.1)
    fake = FakePool(replies={"w0:1": Reply({"error": "boom"}, status=500)})
    monkeypatch.setattr(remote.requests, "post", fake)
    ev = RemoteEvaluator(endpoints(1))
    for n in range(1, remote.BREAKER_FAILURES + 1):
        assert ev.evaluate("ac1", "x", 10, 1).infra_error
        assert ev.status()[0]["failures"] == n
    assert ev.evaluate("ac1", "x", 10, 1).infra_error
    assert fake.hits("w0:1") == remote.BREAKER_FAILURES
    time.sleep(ev.status()[0]["down_for_s"] + 0.05)
    assert ev.evaluate("ac1", "x", 10, 1).infra_error
    assert fake.hits("w0:1") == remote.BREAKER_FAILURES + 1
    fake.replies.clear()
    time.sleep(ev.status()[0]["down_for_s"] + 0.05)
    assert ev.evaluate("ac1", "x", 10, 1).ok and ev.status()[0]["failures"] == 0


def test_recovering_worker_gets_one_trial_at_a_time(monkeypatch):
    monkeypatch.setattr(remote, "DOWN_BASE_S", 0.05)
    release = threading.Event()
    fake = FakePool(replies={"w0:1": Reply({"error": "boom"}, status=500)})
    monkeypatch.setattr(remote.requests, "post", fake)
    ev = RemoteEvaluator(endpoints(2, cap=4))
    ev.evaluate("ac1", "x", 10, 1)
    assert ev.status()[0]["failures"] == 1
    fake.replies.clear()
    fake.slow = {"w0:1": release, "w1:1": release}
    time.sleep(0.1)
    threads = [threading.Thread(target=ev.evaluate, args=("ac1", "x", 10, 1)) for _ in range(6)]
    for t in threads:
        t.start()
    time.sleep(0.3)
    assert [s["free"] for s in ev.status()] == [3, 0]
    release.set()
    for t in threads:
        t.join(5)
    assert ev.status()[0]["failures"] == 0 and [s["free"] for s in ev.status()] == [4, 4]


def test_waits_for_the_first_slot_on_any_endpoint(monkeypatch):
    release = {f"w{i}:1": threading.Event() for i in range(3)}
    fake = FakePool(slow=release)
    monkeypatch.setattr(remote.requests, "post", fake)
    ev = RemoteEvaluator(endpoints(3, cap=1))
    busy = [threading.Thread(target=ev.evaluate, args=("ac1", "x", 10, 1)) for _ in range(3)]
    for t in busy:
        t.start()
    time.sleep(0.2)
    assert [s["free"] for s in ev.status()] == [0, 0, 0]
    out = []
    waiter = threading.Thread(target=lambda: out.append(ev.evaluate("ac1", "x", 10, 1)))
    waiter.start()
    time.sleep(0.2)
    assert not out
    fake.slow = {"w0:1": release["w0:1"], "w1:1": release["w1:1"]}
    release["w2:1"].set()
    waiter.join(5)
    assert out and out[0].stdout == "w2:1"
    for event in release.values():
        event.set()
    for t in busy:
        t.join(5)
    assert [s["free"] for s in ev.status()] == [1, 1, 1]


def test_moved_endpoint_file_is_adopted(monkeypatch, tmp_path):
    src = tmp_path / "w.endpoint"
    worker.write_endpoint_file(str(src), "old:1", "t", 2)
    fake = FakePool(dead={"old:1"})
    monkeypatch.setattr(remote.requests, "post", fake)
    ev = RemoteEvaluator([remote.read_endpoint_file(src), Endpoint(addr="w1:1", capacity=2)])
    worker.write_endpoint_file(str(src), "new:1", "t2", 2)
    assert ev.evaluate("ac1", "x", 10, 1).stdout == "w1:1"
    assert ev.status()[0]["addr"] == "new:1" and ev.status()[0]["failures"] == 0
    ev.evaluate("ac1", "x", 10, 1)
    ev.evaluate("ac1", "x", 10, 1)
    assert fake.hits("new:1") >= 1


def test_concurrent_load_does_not_leak_slots(monkeypatch):
    fake = FakePool(dead={"w1:1"})
    monkeypatch.setattr(remote.requests, "post", fake)
    ev = RemoteEvaluator(endpoints(4, cap=3))
    results = []
    threads = [threading.Thread(target=lambda: results.append(ev.evaluate("ac1", "x", 10, 1)))
               for _ in range(60)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert len(results) == 60 and all(r.stdout in ("w0:1", "w2:1", "w3:1") for r in results)
    assert [s["free"] for s in ev.status()] == [3, 3, 3, 3]
    assert fake.hits("w1:1") == 1


def test_directory_pool_adds_late_workers_and_keeps_vanished_ones(monkeypatch, tmp_path):
    monkeypatch.setattr(remote, "SCAN_MIN_GAP_S", 0.05)
    monkeypatch.setattr(remote, "WAIT_S", 0.05)
    pool = tmp_path / "endpoints"
    worker.write_endpoint_file(str(pool / "1.json"), "w0:1", "t", 1)
    release = threading.Event()
    fake = FakePool(slow={"w0:1": release})
    monkeypatch.setattr(remote.requests, "post", fake)
    ev = RemoteEvaluator(pool)
    assert ev.capacity == 1
    busy = threading.Thread(target=ev.evaluate, args=("ac1", "x", 10, 1))
    busy.start()
    out = []
    waiter = threading.Thread(target=lambda: out.append(ev.evaluate("ac1", "x", 10, 1)))
    waiter.start()
    time.sleep(0.3)
    assert not out and ev.status()[0]["free"] == 0
    worker.write_endpoint_file(str(pool / "2.json"), "w1:1", "t", 2)
    waiter.join(5)
    assert out and out[0].stdout == "w1:1" and ev.capacity == 3
    (pool / "1.json").unlink()
    release.set()
    busy.join(5)
    monkeypatch.setattr(remote, "SCAN_EVERY_S", 0.0)
    for _ in range(6):
        assert ev.evaluate("ac1", "x", 10, 1).ok
    assert fake.hits("w0:1") >= 3 and [s["addr"] for s in ev.status()] == ["w0:1", "w1:1"]


def test_list_pool_warns_once_about_missing_files_and_adopts_them(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(remote, "SCAN_EVERY_S", 0.0)
    monkeypatch.setattr(remote.requests, "post", FakePool())
    listing = tmp_path / "endpoints.list"
    listing.write_text("w/a.json\nw/b.json\n")
    worker.write_endpoint_file(str(tmp_path / "w" / "a.json"), "w0:1", "t", 2)
    ev = RemoteEvaluator(listing)
    for _ in range(3):
        assert ev.evaluate("ac1", "x", 10, 1).ok
    assert ev.capacity == 2
    assert capsys.readouterr().err.count('"eval_endpoint_missing"') == 1
    worker.write_endpoint_file(str(tmp_path / "w" / "b.json"), "w1:1", "t", 3)
    assert ev.evaluate("ac1", "x", 10, 1).ok
    assert ev.capacity == 5 and [s["addr"] for s in ev.status()] == ["w0:1", "w1:1"]


def test_worker_with_a_broken_sandbox_answers_503_and_is_failed_over(server):
    _, good = server
    broken = worker.make_server("127.0.0.1", 0, capacity=2, token="t2")
    broken.worker_state.python = "/nonexistent/python"
    threading.Thread(target=broken.serve_forever, daemon=True).start()
    try:
        bad = f"127.0.0.1:{broken.server_address[1]}"
        reply = requests.post(f"http://{bad}/eval", timeout=30, headers={"X-Eval-Token": "t2"},
                              json={"task": "ac1", "program": ac1_program(), "timeout_s": 30, "cpus": 1})
        assert reply.status_code == 503 and reply.json()["error"].startswith("eval_sandbox_error:")
        ev = RemoteEvaluator([Endpoint(addr=bad, token="t2", capacity=2),
                              Endpoint(addr=good, token="secret-token", capacity=2)])
        result = ev.evaluate("ac1", ac1_program(), 30, 1)
        assert result.ok and result.raw_score == 2.0
        assert [s["failures"] for s in ev.status()] == [1, 0]
    finally:
        broken.shutdown()
        broken.server_close()
