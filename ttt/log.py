"""Structured logging: one JSON object per line on stderr and in events.jsonl."""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

_lock = threading.Lock()
_sink: Optional[Path] = None


def configure(out_dir: Optional[str]) -> None:
    """Also append events to <out_dir>/events.jsonl."""
    global _sink
    _sink = Path(out_dir) / "events.jsonl" if out_dir else None


def event(kind: str, level: str = "info", **fields: Any) -> None:
    """Write one structured event."""
    rec = {"t": round(time.time(), 3), "level": level, "event": kind, **fields}
    line = json.dumps(rec, default=str)
    with _lock:
        print(line, file=sys.stderr, flush=True)
        if _sink is not None:
            with open(_sink, "a") as f:
                f.write(line + "\n")


def warn(kind: str, **fields: Any) -> None:
    """Write a warning event; use for every degraded path instead of passing silently."""
    event(kind, level="warning", **fields)
