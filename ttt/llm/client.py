"""HTTP client for one vLLM OpenAI-compatible server."""
from __future__ import annotations

import time
from typing import Any, Optional

import requests


class ServerError(RuntimeError):
    """The server could not be reached or did not answer with a usable reply."""


class VLLMClient:
    """Completions, tokenization and health checks against one served model."""

    def __init__(self, base_url: str, served_model: str, api_key: Optional[str] = None,
                 timeout_s: float = 2400.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.served_model = served_model
        self.timeout_s = float(timeout_s)
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def __repr__(self) -> str:
        return f"VLLMClient({self.base_url!r}, {self.served_model!r})"

    def _post(self, path: str, body: dict[str, Any],
              timeout_s: Optional[float] = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        timeout = self.timeout_s if timeout_s is None else float(timeout_s)
        try:
            r = requests.post(url, json=body, headers=self._headers, timeout=timeout)
        except requests.RequestException as e:
            raise ServerError(f"POST {url} failed: {e}") from e
        if r.status_code != 200:
            raise ServerError(f"POST {url} returned {r.status_code}: {r.text[:500]}")
        try:
            return r.json()
        except ValueError as e:
            raise ServerError(f"POST {url} returned a reply that is not JSON: {r.text[:200]}") from e

    def _field(self, reply: dict[str, Any], *keys: Any) -> Any:
        """reply[k1][k2]...; a reply without that field is a server failure."""
        try:
            for key in keys:
                reply = reply[key]
        except (KeyError, IndexError, TypeError) as e:
            raise ServerError(f"{self.base_url} reply lacks {list(keys)}: {str(reply)[:200]}") from e
        return reply

    def generate(self, prompt: str, *, max_tokens: int, temperature: float = 1.0,
                 top_p: float = 1.0, top_k: int = -1, stop: Optional[list[str]] = None,
                 extra: Optional[dict] = None,
                 timeout_s: Optional[float] = None) -> tuple[str, str]:
        """Complete a raw prompt; returns (text, finish_reason); timeout_s defaults to the client's."""
        body: dict[str, Any] = {
            "model": self.served_model,
            "prompt": prompt,
            "max_tokens": int(max_tokens),
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }
        if stop:
            body["stop"] = list(stop)
        if extra:
            body.update(extra)
        choice = self._field(self._post("/v1/completions", body, timeout_s), "choices", 0)
        if not isinstance(choice, dict):
            raise ServerError(f"{self.base_url} completion choice is not an object: {str(choice)[:200]}")
        return choice.get("text") or "", choice.get("finish_reason") or ""

    def count_tokens(self, text: str) -> int:
        """Number of tokens the server's tokenizer produces for text, without special tokens."""
        body = {"model": self.served_model, "prompt": text, "add_special_tokens": False}
        return int(self._field(self._post("/tokenize", body), "count"))

    def wait_ready(self, timeout_s: float = 2400.0, interval_s: float = 2.0) -> None:
        """Poll /health until it answers 200; raises RuntimeError after timeout_s."""
        deadline = time.monotonic() + timeout_s
        last = "no response"
        while True:
            try:
                r = requests.get(f"{self.base_url}/health", headers=self._headers, timeout=5)
                if r.status_code == 200:
                    return
                last = f"HTTP {r.status_code}"
            except requests.RequestException as e:
                last = str(e)[:200]
            if time.monotonic() >= deadline:
                raise RuntimeError(f"{self.base_url} not ready after {timeout_s:.0f}s: {last}")
            time.sleep(interval_s)
