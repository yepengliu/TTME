"""HTTP evaluation worker: POST /eval runs and scores one program, GET /health reports load."""
from __future__ import annotations

import argparse
import hmac
import json
import os
import secrets
import socket
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from ttt import tasks
from ttt.eval.local import evaluate_program
from ttt.eval.sandbox import resolve_python
from ttt.eval.score import result_to_dict
from ttt.log import event, warn

MAX_REQUEST_BYTES = 256 * 1024 * 1024
TOKEN_HEADER = "X-Eval-Token"


@dataclass
class WorkerState:
    """Shared state of one worker: credentials, interpreter, slot semaphore and in-flight count."""

    capacity: int
    token: str
    python: str
    slots: threading.Semaphore
    active: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


def authorized(sent: str, token: str) -> bool:
    """Constant-time token check."""
    return hmac.compare_digest(sent.encode("utf-8"), token.encode("utf-8"))


def parse_request(body: bytes) -> dict:
    """Validate an /eval request body; raises ValueError naming the bad field."""
    req = json.loads(body)
    if not isinstance(req, dict):
        raise ValueError("body must be a JSON object")
    if req.get("task") not in tasks.TASK_NAMES:
        raise ValueError(f"task must be one of {tasks.TASK_NAMES}")
    if not isinstance(req.get("program"), str):
        raise ValueError("program must be a string")
    timeout_s, cpus = req.get("timeout_s"), req.get("cpus")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
        raise ValueError("timeout_s must be a positive number")
    if isinstance(cpus, bool) or not isinstance(cpus, int) or cpus < 1:
        raise ValueError("cpus must be a positive integer")
    return {"task": req["task"], "program": req["program"], "timeout_s": float(timeout_s),
            "cpus": cpus}


def run_request(state: WorkerState, req: dict) -> tuple[int, dict]:
    """Evaluate one validated request inside a slot; returns (HTTP status, JSON body)."""
    with state.slots:
        with state.lock:
            state.active += 1
        try:
            result = evaluate_program(req["task"], req["program"], req["timeout_s"], req["cpus"],
                                      state.python)
        except Exception as e:  # keep serving; the client fails this request over to another worker
            warn("eval_worker_error", task=req["task"], error=repr(e)[:500])
            return 500, {"error": f"worker_error: {str(e)[:200]}"}
        finally:
            with state.lock:
                state.active -= 1
    if result.infra_error:
        # This worker could not evaluate at all: 503 makes the client rest it and retry elsewhere.
        return 503, {"error": result.error}
    return 200, result_to_dict(result)


class EvalHandler(BaseHTTPRequestHandler):
    """Request handler; the server object carries the WorkerState as `worker_state`."""

    def log_message(self, format: str, *args) -> None:
        return None

    def log_error(self, format: str, *args) -> None:
        warn("eval_worker_http_error", message=format % args)

    def _reply(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError) as e:
            warn("eval_worker_reply_dropped", status=status, error=str(e))

    def do_GET(self) -> None:
        state = self.server.worker_state
        if self.path != "/health":
            self._reply(404, {"error": "not found"})
            return
        with state.lock:
            active = state.active
        self._reply(200, {"ok": True, "capacity": state.capacity, "active": active})

    def do_POST(self) -> None:
        state = self.server.worker_state
        if self.path != "/eval":
            self._reply(404, {"error": "not found"})
            return
        if not authorized(self.headers.get(TOKEN_HEADER, ""), state.token):
            self._reply(401, {"error": "unauthorized"})
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
            if not 0 <= length <= MAX_REQUEST_BYTES:
                raise ValueError(f"Content-Length must be within [0, {MAX_REQUEST_BYTES}]")
            req = parse_request(self.rfile.read(length))
        except ValueError as e:
            self._reply(400, {"error": f"bad request: {str(e)[:200]}"})
            return
        self._reply(*run_request(state, req))


def make_server(host: str, port: int, capacity: int, token: str,
                python: Optional[str] = None) -> ThreadingHTTPServer:
    """Bind an evaluation server (port 0 picks a free one); call serve_forever() to start it."""
    if capacity < 1:
        raise ValueError("capacity must be >= 1")
    server = ThreadingHTTPServer((host, port), EvalHandler)
    server.daemon_threads = True
    server.worker_state = WorkerState(capacity=capacity, token=token, python=resolve_python(python),
                                      slots=threading.Semaphore(capacity))
    return server


def advertised_addr(host: str, port: int) -> str:
    """host:port that clients should dial; a wildcard bind advertises this machine's hostname."""
    return f"{socket.gethostname() if host in ('', '0.0.0.0', '::') else host}:{port}"


def write_endpoint_file(path: str, addr: str, token: str, capacity: int) -> None:
    """Atomically write {addr, token, capacity, pid}; mode 0600 because it holds the token."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, mode=0o700, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".endpoint-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"addr": addr, "token": token, "capacity": capacity, "pid": os.getpid()}, f)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ttt worker", description="Serve sandboxed evaluations over HTTP.")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address; use 0.0.0.0 when drivers run on other machines")
    ap.add_argument("--port", type=int, default=0, help="0 picks a free port")
    ap.add_argument("--capacity", type=int, required=True, help="concurrent evaluations")
    ap.add_argument("--endpoint-file", required=True,
                    help="where to write {addr, token, capacity, pid} for drivers (mode 0600)")
    ap.add_argument("--python", default=None, help="interpreter for candidate programs")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    """`ttt worker`: serve evaluations until interrupted."""
    ap = _parser()
    args = ap.parse_args(argv)
    if args.capacity < 1:
        ap.error("--capacity must be >= 1")
    token = secrets.token_urlsafe(24)
    server = make_server(args.host, args.port, args.capacity, token, args.python)
    addr = advertised_addr(args.host, server.server_address[1])
    write_endpoint_file(args.endpoint_file, addr, token, args.capacity)
    event("eval_worker_started", addr=addr, capacity=args.capacity, pid=os.getpid(),
          endpoint_file=args.endpoint_file, python=server.worker_state.python)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        event("eval_worker_stopped", addr=addr)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
