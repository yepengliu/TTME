"""Client for a pool of HTTP evaluation workers: slots, failover, circuit breaking and pool rescans."""
from __future__ import annotations

import dataclasses
import itertools
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import requests

from ttt.eval.score import failed, result_from_dict
from ttt.log import event, warn
from ttt.types import EvalResult

HTTP_MARGIN_S = 200.0
MAX_TIMEOUTS = 2
# How many workers one request may mark down, so a request that breaks every worker cannot drain the pool.
MAX_FAILOVERS = 3
BUDGET_FACTOR = 3.0
DOWN_BASE_S = 30.0
DOWN_MAX_S = 600.0
BREAKER_FAILURES = 3
WAIT_S = 5.0
SCAN_EVERY_S = 60.0
SCAN_MIN_GAP_S = 5.0
TOKEN_HEADER = "X-Eval-Token"
_LITERAL = re.compile(r"[^\s/#]+:\d+(?:#\S+)?")

Problems = list[tuple[str, str]]


@dataclass
class Endpoint:
    """One worker: address, token and declared capacity; `src` is its endpoint file, if any."""

    addr: str
    token: Optional[str] = None
    capacity: Optional[int] = None
    src: Optional[str] = None


def parse_endpoint(text: str) -> Endpoint:
    """Parse a literal `host:port` or `host:port#token` endpoint."""
    text = text.strip()
    if not _LITERAL.fullmatch(text):
        raise ValueError(f"not a host:port[#token] endpoint: {text!r}")
    addr, _, token = text.partition("#")
    return Endpoint(addr=addr, token=token or None)


def read_endpoint_file(path: Union[str, Path]) -> Endpoint:
    """Read a worker's endpoint file {addr, token, capacity, pid}."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    addr = data.get("addr") if isinstance(data, dict) else None
    if not isinstance(addr, str) or not addr:
        raise ValueError(f"endpoint file has no address: {path}")
    capacity = data.get("capacity")
    if capacity is not None and (isinstance(capacity, bool) or not isinstance(capacity, int)
                                 or capacity < 1):
        raise ValueError(f"endpoint file has a bad capacity: {path}")
    return Endpoint(addr=addr, token=data.get("token") or None, capacity=capacity, src=str(path))


def _scan_directory(root: Path) -> tuple[list[Endpoint], Problems]:
    endpoints, problems = [], []
    for entry in sorted(root.iterdir()):
        # Workers publish `<id>.json`; hidden names are their in-progress atomic writes.
        if entry.name.startswith(".") or entry.suffix != ".json" or not entry.is_file():
            continue
        try:
            endpoints.append(read_endpoint_file(entry))
        except (OSError, ValueError) as e:
            problems.append((str(entry), str(e)[:200]))
    return endpoints, problems


def _scan_list(listing: Path) -> tuple[list[Endpoint], Problems]:
    endpoints, problems = [], []
    for raw in listing.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line) if os.path.isabs(line) else listing.parent / line
        if candidate.is_file():
            try:
                endpoints.append(read_endpoint_file(candidate))
            except (OSError, ValueError) as e:
                problems.append((str(candidate), str(e)[:200]))
        elif _LITERAL.fullmatch(line):
            endpoints.append(parse_endpoint(line))
        else:
            problems.append((str(candidate), "no such endpoint file (yet)"))
    return endpoints, problems


def scan_pool(path: Union[str, os.PathLike]) -> tuple[list[Endpoint], Problems]:
    """Endpoints a pool names now (directory of `*.json` endpoint files, or list file), plus unusable ones."""
    root = Path(path).resolve()
    return _scan_directory(root) if root.is_dir() else _scan_list(root)


def load_endpoints(path: Union[str, os.PathLike]) -> list[Endpoint]:
    """One-off snapshot of a pool path, warning about unusable entries; RemoteEvaluator(path) also rescans."""
    endpoints, problems = scan_pool(path)
    for where, problem in problems:
        warn("eval_endpoint_missing", path=where, problem=problem, pool=str(path))
    if not endpoints:
        raise ValueError(f"no eval endpoints resolved from {path}")
    return endpoints


def probe_capacity(endpoint: Endpoint, timeout_s: float) -> int:
    """Capacity a worker reports on /health; 1 (with a warning) when it cannot be read."""
    try:
        reply = requests.get(f"http://{endpoint.addr}/health", timeout=timeout_s)
        capacity = reply.json().get("capacity")
    except (requests.RequestException, ValueError, AttributeError) as e:
        warn("eval_endpoint_capacity_unknown", addr=endpoint.addr, error=str(e)[:200])
        return 1
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        warn("eval_endpoint_capacity_unknown", addr=endpoint.addr, error=f"capacity={capacity!r}")
        return 1
    return capacity


@dataclass
class _Member:
    """Slots and health of one endpoint; guarded by the evaluator's condition variable."""

    ep: Endpoint
    cap: int
    free: int
    fails: int = 0
    down_until: float = 0.0
    probing: bool = False


class RemoteEvaluator:
    """Evaluator over HTTP workers: any free slot, failover on infrastructure errors, circuit breaking."""

    def __init__(self, endpoints: Union[str, os.PathLike, Sequence[Union[Endpoint, str]]],
                 connect_timeout_s: float = 15.0):
        """`endpoints`: a list, or a pool path (directory or list file) rescanned so late workers join."""
        self._source: Optional[str] = None
        self._unusable: set[str] = set()
        if isinstance(endpoints, (str, os.PathLike)):
            self._source = os.fspath(endpoints)
            eps, problems = scan_pool(self._source)
            self._note_problems(problems)
        else:
            eps = [dataclasses.replace(e) if isinstance(e, Endpoint) else parse_endpoint(e)
                   for e in endpoints]
        if not eps:
            raise ValueError(f"RemoteEvaluator needs at least one endpoint (source: {self._source})")
        self._connect_timeout_s = connect_timeout_s
        caps = [e.capacity or probe_capacity(e, connect_timeout_s) for e in eps]
        self._members = [_Member(e, cap, cap) for e, cap in zip(eps, caps)]
        self._cv = threading.Condition()
        self._counter = itertools.count()
        self._scanning = False
        self._last_scan = time.monotonic()

    @property
    def capacity(self) -> int:
        """Total declared slots of every endpoint known so far; grows as rescans add workers."""
        with self._cv:
            return sum(m.cap for m in self._members)

    def status(self) -> list[dict]:
        """Per-endpoint snapshot: address, capacity, free slots, consecutive failures, cooldown left."""
        with self._cv:
            now = time.monotonic()
            return [{"addr": m.ep.addr, "capacity": m.cap, "free": m.free, "failures": m.fails,
                     "down_for_s": max(0.0, m.down_until - now)} for m in self._members]

    def evaluate(self, task: str, program: str, timeout_s: float, cpus: int) -> EvalResult:
        """Run one program on some worker; every failure not caused by the candidate has infra_error set."""
        payload = {"task": task, "program": program, "timeout_s": timeout_s, "cpus": cpus}
        http_timeout = timeout_s + HTTP_MARGIN_S
        request_timeout = (min(self._connect_timeout_s, http_timeout), http_timeout)
        with self._cv:
            start = next(self._counter)
        tried: set[int] = set()
        last_err, deadline, n_timeouts, n_failed = "no eval endpoint available", None, 0, 0
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                last_err = f"eval budget exhausted after {len(tried)} attempt(s); last: {last_err}"
                break
            i, trial, waited = self._acquire(start, tried)
            if i is None:
                break
            # Waiting for a slot is not a failure: only time spent attempting uses the budget.
            deadline = (time.monotonic() + BUDGET_FACTOR * http_timeout if deadline is None
                        else deadline + waited)
            tried.add(i)
            try:
                kind, out = self._attempt(i, trial, payload, request_timeout)
            finally:
                self._release(i, trial)
            if kind == "done":
                return out
            last_err = out
            if kind == "timeout":
                n_timeouts += 1
                warn("eval_timeout", error=out, attempt=n_timeouts, max_attempts=MAX_TIMEOUTS)
                if n_timeouts >= MAX_TIMEOUTS:
                    break
            else:
                n_failed += 1
                warn("eval_endpoint_error", error=out, attempt=n_failed, max_attempts=MAX_FAILOVERS)
                if n_failed >= MAX_FAILOVERS:
                    break
        return failed(f"eval_transport_error: {last_err}", infra_error=True)

    def _attempt(self, i: int, trial: bool, payload: dict,
                 timeout: tuple[float, float]) -> tuple[str, Any]:
        """One POST: ("done", result), ("timeout", error) or ("failed", error) with the endpoint marked down."""
        with self._cv:
            sent = (self._members[i].ep.addr, self._members[i].ep.token)
        addr, token = sent
        headers = {TOKEN_HEADER: token} if token else {}
        try:
            reply = requests.post(f"http://{addr}/eval", json=payload, timeout=timeout,
                                  headers=headers)
        except requests.ConnectionError as e:  # before Timeout: a connect timeout is a dead worker
            return self._failover(i, trial, sent, type(e).__name__, str(e))
        except requests.Timeout as e:
            return "timeout", f"timeout after {timeout[1]:.0f}s @ {addr}: {str(e)[:120]}"
        except requests.RequestException as e:
            return self._failover(i, trial, sent, type(e).__name__, str(e))
        if reply.status_code == 400:
            # The worker validated and refused the request itself; every worker would do the same.
            return "done", failed(f"eval_request_rejected: {_reply_text(reply)}", infra_error=True)
        if reply.status_code != 200:
            return self._failover(i, trial, sent, f"HTTP {reply.status_code}", _reply_text(reply))
        try:
            result = result_from_dict(reply.json())
        except ValueError as e:
            return self._failover(i, trial, sent, "malformed reply", str(e))
        if result.infra_error:
            return self._failover(i, trial, sent, "worker infrastructure error", result.error or "")
        self._mark_up(i)
        return "done", result

    def _failover(self, i: int, trial: bool, sent: tuple[str, Optional[str]], reason: str,
                  detail: str) -> tuple[str, str]:
        self._mark_down(i, reason, trial, sent)
        self._refresh(i)
        return "failed", f"{sent[0]}: {reason}: {detail[:160]}"

    def _acquire(self, start: int, tried: set[int]) -> tuple[Optional[int], bool, float]:
        """Take a slot: (endpoint index or None, whether it is a trial, seconds waited)."""
        began = time.monotonic()
        while True:
            self._scan_if_due(SCAN_EVERY_S)
            with self._cv:
                pick, waitable = self._pick(start, tried)
                if pick is not None:
                    return pick, self._take(pick), time.monotonic() - began
                if waitable and not self._scan_due(SCAN_MIN_GAP_S):
                    self._cv.wait(WAIT_S)
                    continue
            # No slot anywhere: workers may have joined since the last scan.
            if not self._scan_if_due(SCAN_MIN_GAP_S) and not waitable:
                return None, False, time.monotonic() - began

    def _pick(self, start: int, tried: set[int]) -> tuple[Optional[int], bool]:
        """Endpoint to try next, and whether waiting could still help; the caller holds the lock."""
        now, n = time.monotonic(), len(self._members)
        untried = [i for i in ((start + k) % n for k in range(n)) if i not in tried]
        ready = [i for i in untried if self._members[i].down_until <= now]
        closed = [i for i in untried if self._members[i].fails < BREAKER_FAILURES]
        pick = next((i for i in ready if self._can_take(i)), None)
        if pick is None and not ready:
            # All untried endpoints cool down: probe the one recovering first, unless its circuit is open.
            by_expiry = sorted(closed, key=lambda j: self._members[j].down_until)
            pick = next((i for i in by_expiry if self._can_take(i)), None)
        return pick, bool(ready or closed)

    def _can_take(self, i: int) -> bool:
        m = self._members[i]
        # After a failure an endpoint takes one trial request at a time until a success closes it.
        return m.free > 0 and not (m.fails and m.probing)

    def _take(self, i: int) -> bool:
        m = self._members[i]
        m.free -= 1
        trial = m.fails > 0
        m.probing = m.probing or trial
        return trial

    def _release(self, i: int, trial: bool) -> None:
        with self._cv:
            m = self._members[i]
            m.free += 1
            if trial:
                m.probing = False
            self._cv.notify_all()

    def _mark_down(self, i: int, reason: str, trial: bool, sent: tuple[str, Optional[str]]) -> None:
        """Count a failure: the first one or a failed trial doubles the cooldown; others are stale reports."""
        with self._cv:
            m = self._members[i]
            # Stale: sent before an earlier failure, or to an address/token the worker has moved away from.
            if (m.fails and not trial) or sent != (m.ep.addr, m.ep.token):
                return
            now = time.monotonic()
            m.fails += 1
            cooldown = min(DOWN_BASE_S * 2 ** min(m.fails - 1, 30), DOWN_MAX_S)
            m.down_until = now + cooldown
            fields = dict(addr=m.ep.addr, reason=reason, failures=m.fails, cooldown_s=cooldown,
                          down=sum(1 for x in self._members if x.fails), total=len(self._members))
        if fields["failures"] == 1:
            warn("eval_endpoint_down", **fields)
        elif fields["failures"] == BREAKER_FAILURES:
            warn("eval_endpoint_circuit_open", **fields)

    def _mark_up(self, i: int) -> None:
        with self._cv:
            m = self._members[i]
            if not m.fails:
                return
            m.fails, m.down_until = 0, 0.0
            n_down, addr = sum(1 for x in self._members if x.fails), m.ep.addr
            self._cv.notify_all()
        event("eval_endpoint_up", addr=addr, down=n_down, total=len(self._members))

    def _refresh(self, i: int) -> bool:
        """Re-read a worker's endpoint file; True when its address or token changed."""
        with self._cv:
            src = self._members[i].ep.src
        if not src:
            return False
        try:
            fresh = read_endpoint_file(src)
        except (OSError, ValueError) as e:
            warn("eval_endpoint_file_unreadable", path=src, error=str(e)[:200])
            return False
        return self._adopt(i, fresh)

    def _adopt(self, i: int, fresh: Endpoint) -> bool:
        """Point endpoint i at a re-read address/token; a changed one is a restarted, healthy worker."""
        with self._cv:
            m = self._members[i]
            if (fresh.addr, fresh.token) == (m.ep.addr, m.ep.token):
                return False
            old, m.ep.addr, m.ep.token = m.ep.addr, fresh.addr, fresh.token
            m.fails, m.down_until = 0, 0.0
            self._cv.notify_all()
        event("eval_endpoint_moved", old=old, new=fresh.addr)
        return True

    def _scan_due(self, min_gap_s: float) -> bool:
        """Whether a pool rescan may start now; the caller holds the lock."""
        return (self._source is not None and not self._scanning
                and time.monotonic() - self._last_scan >= min_gap_s)

    def _scan_if_due(self, min_gap_s: float) -> bool:
        """Rescan the pool source when due; True when that added or moved an endpoint."""
        with self._cv:
            if not self._scan_due(min_gap_s):
                return False
            self._scanning, self._last_scan = True, time.monotonic()
        try:
            return self._scan()
        finally:
            with self._cv:
                self._scanning = False

    def _scan(self) -> bool:
        """Re-read the pool source: adopt moved workers, add new ones; True when anything changed."""
        try:
            found, problems = scan_pool(self._source)
        except OSError as e:
            warn("eval_pool_unreadable", pool=self._source, error=str(e)[:200])
            return False
        self._note_problems(problems)
        with self._cv:
            index = {m.ep.src or m.ep.addr: i for i, m in enumerate(self._members)}
        moved, new = False, []
        for e in found:
            if (e.src or e.addr) in index:
                moved = self._adopt(index[e.src or e.addr], e) or moved
            else:
                new.append(e)
        caps = [e.capacity or probe_capacity(e, self._connect_timeout_s) for e in new]
        with self._cv:
            self._members.extend(_Member(e, cap, cap) for e, cap in zip(new, caps))
            total = sum(m.cap for m in self._members)
            self._cv.notify_all()
        for e, cap in zip(new, caps):
            event("eval_endpoint_added", addr=e.addr, capacity=cap, total_capacity=total)
        return bool(new) or moved

    def _note_problems(self, problems: Problems) -> None:
        """Warn once about each pool entry that cannot be used yet; scans happen one at a time."""
        for where, problem in problems:
            if where not in self._unusable:
                warn("eval_endpoint_missing", path=where, problem=problem, pool=self._source)
        self._unusable = {where for where, _ in problems}


def _reply_text(reply: Any) -> str:
    try:
        body = reply.json()
    except ValueError:
        return str(getattr(reply, "text", ""))[:200]
    if isinstance(body, dict) and isinstance(body.get("error"), str):
        return body["error"][:200]
    return str(body)[:200]
