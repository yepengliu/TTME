"""Content-addressed, write-once stores for constructions and program texts."""
from __future__ import annotations

import hashlib
import io
import os
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

_DTYPE = np.dtype("<f8")


def _as_array(seq: Sequence[float]) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(seq, dtype=_DTYPE).reshape(-1))


def _code_bytes(text: str) -> bytes:
    # Model text can carry lone surrogates; they must hash and store rather than raise.
    return text.encode("utf-8", "surrogatepass")


def construction_hash(seq: Sequence[float]) -> str:
    """Return the sha256 of a construction's little-endian float64 bytes."""
    return hashlib.sha256(_as_array(seq).tobytes()).hexdigest()


def code_hash(text: str) -> str:
    """Return the sha256 of a program text's UTF-8 bytes."""
    return hashlib.sha256(_code_bytes(text)).hexdigest()


def write_once(path: Path, data: bytes) -> None:
    """Write data to path through an fsynced same-directory temp file; no-op if path exists."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


class ConstructionStore:
    """Constructions saved as <root>/<h[:2]>/<h>.npy, written once and verified on read."""

    def __init__(self, root: str | os.PathLike) -> None:
        self.root = Path(root)

    def path(self, digest: str) -> Path:
        """Return the file that holds the construction with this hash."""
        return self.root / digest[:2] / f"{digest}.npy"

    def put(self, seq: Sequence[float]) -> str:
        """Store a construction durably (no-op if present) and return its hash."""
        arr = _as_array(seq)
        digest = hashlib.sha256(arr.tobytes()).hexdigest()
        path = self.path(digest)
        if not path.exists():
            buf = io.BytesIO()
            np.save(buf, arr, allow_pickle=False)
            write_once(path, buf.getvalue())
        return digest

    def get(self, digest: str) -> list[float]:
        """Load a construction by hash; raises if missing or if its bytes do not match the hash."""
        arr = np.load(self.path(digest), allow_pickle=False)
        actual = hashlib.sha256(_as_array(arr).tobytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"construction {digest} is corrupt (hashes to {actual})")
        return arr.tolist()

    def __contains__(self, digest: str) -> bool:
        return self.path(digest).exists()


class CodeStore:
    """Program texts saved as <root>/<h[:2]>/<h>.py, written once and verified on read."""

    def __init__(self, root: str | os.PathLike) -> None:
        self.root = Path(root)

    def path(self, digest: str) -> Path:
        """Return the file that holds the program with this hash."""
        return self.root / digest[:2] / f"{digest}.py"

    def put(self, text: str) -> str:
        """Store a program text durably (no-op if present) and return its hash."""
        data = _code_bytes(text)
        digest = hashlib.sha256(data).hexdigest()
        write_once(self.path(digest), data)
        return digest

    def get(self, digest: str) -> str:
        """Load a program by hash; raises if missing or if its bytes do not match the hash."""
        data = self.path(digest).read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != digest:
            raise ValueError(f"program {digest} is corrupt (hashes to {actual})")
        return data.decode("utf-8", "surrogatepass")

    def __contains__(self, digest: str) -> bool:
        return self.path(digest).exists()
