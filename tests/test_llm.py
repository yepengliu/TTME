import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ttt.config import Model, Sampling
from ttt.llm import ServerError, VLLMClient, build_format
from ttt.llm.fake import (FakeClient, aux_reply_for, chatml_completion, completion_for,
                          harmony_completion, program_responder)
from ttt.llm.formats import (FINAL_MARKER, FINAL_OPEN, PHASE2_PREFILL, ChatmlThink,
                             HarmonyTwoPhase, extract_harmony_code, last_python_block)

HARMONY = Model(name="m", format="harmony_two_phase", phase1_max_tokens=30, date="2026-07-24")
CHATML = Model(name="m", format="chatml_think")
KEEP = {"skip_special_tokens": False, "spaces_between_special_tokens": False}


def scripted(*replies):
    queue = list(replies)
    return FakeClient(lambda prompt, params: queue.pop(0))


def harmony(context_window=200):
    return build_format(HARMONY, Sampling(context_window=context_window, temperature=0.7, top_p=0.9, top_k=5))


def test_harmony_render_bytes():
    got = harmony().render("Q")
    assert got == (
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
        "Knowledge cutoff: 2024-06\nCurrent date: 2026-07-24\n\nReasoning: high\n\n"
        "# Valid channels: analysis, commentary, final. Channel must be included for every message."
        "<|end|><|start|>user<|message|>Q<|end|><|start|>assistant")


def test_phase2_prefill_bytes():
    assert PHASE2_PREFILL == ("\n\n... okay, I am out of thinking tokens. "
                              "I need to send my final message now.")


def test_chatml_render_bytes():
    got = build_format(CHATML, Sampling()).render("Q")
    assert got == "<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n<think>\n"


def test_build_format_dispatch():
    assert isinstance(harmony(), HarmonyTwoPhase)
    assert isinstance(build_format(CHATML, Sampling()), ChatmlThink)
    with pytest.raises(ValueError):
        build_format(Model(name="m", format="harmony_two_phase"), Sampling())


def test_phase_one_finishes():
    client = scripted(("done", "stop"))
    assert harmony().complete("P" * 40, client) == "done"
    (prompt, params), = client.calls
    assert prompt == "P" * 40
    assert params == {"max_tokens": 20, "temperature": 0.7, "top_p": 0.9, "top_k": 5,
                      "stop": ["<|return|>"], "extra": KEEP, "timeout_s": None}


def test_prompt_over_phase_one_budget():
    with pytest.raises(ValueError):
        harmony().complete("P" * 120, scripted())


def test_truncated_in_final_channel_continues_without_prefill():
    text1 = "a<|channel|>final<|message|>" + "x" * 12
    client = scripted((text1, "length"), ("rest", "stop"))
    assert harmony().complete("P" * 40, client) == text1 + "rest"
    prompt2, params2 = client.calls[1]
    assert prompt2 == "P" * 40 + text1
    assert params2["max_tokens"] == 200 - 10 - len(text1) // 4 - 50


def test_truncated_in_final_channel_without_budget():
    text1 = "a<|channel|>final<|message|>" + "x" * 400
    client = scripted((text1, "length"))
    assert harmony(context_window=100).complete("P" * 40, client) == text1
    assert len(client.calls) == 1


@pytest.mark.parametrize("text1, prefill", [
    ("thinking<|end|>", PHASE2_PREFILL + FINAL_OPEN),
    ("thinking more", PHASE2_PREFILL + FINAL_MARKER),
])
def test_truncated_analysis_forces_final(text1, prefill):
    client = scripted((text1, "length"), ("final", "stop"))
    assert harmony(context_window=300).complete("P" * 40, client) == text1 + prefill + "final"
    prompt2, params2 = client.calls[1]
    assert prompt2 == "P" * 40 + text1 + prefill
    assert params2["max_tokens"] == 300 - 10 - len(text1) // 4 - len(prefill) // 4 - 50
    assert params2["stop"] == ["<|return|>"] and params2["extra"] == KEEP


def test_truncated_analysis_without_budget_returns_prefill():
    client = scripted(("thinking", "length"))
    out = harmony(context_window=70).complete("P" * 40, client)
    assert out == "thinking" + PHASE2_PREFILL + FINAL_MARKER
    assert len(client.calls) == 1


def test_chatml_single_call():
    fmt = build_format(CHATML, Sampling(context_window=100))
    client = scripted(("t</think>```python\nx=1\n```", "stop"))
    assert fmt.complete("P" * 40, client).startswith("t</think>")
    (_, params), = client.calls
    assert params == {"max_tokens": 40, "temperature": 1.0, "top_p": 1.0, "top_k": -1,
                      "stop": ["<|im_end|>"], "extra": KEEP, "timeout_s": None}
    with pytest.raises(ValueError):
        fmt.complete("P" * 400, client)


def test_harmony_extraction():
    text = ("<|channel|>analysis<|message|>try ```python\nold = 1\n```<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Here:\n```python\nnew = 2\n```")
    assert extract_harmony_code(text) == "new = 2"
    assert extract_harmony_code(text + "<|return|>```python\nlater = 3\n```") == "new = 2"
    assert extract_harmony_code(text + "<|return|>x<|return|>") is None
    assert extract_harmony_code("```python\nopen = 1") == "open = 1"
    assert extract_harmony_code("```python\n\n```") is None
    assert extract_harmony_code("no code") is None
    assert extract_harmony_code(harmony_completion("p = 1")) == "p = 1"


def test_chatml_extraction():
    fmt = build_format(CHATML, Sampling())
    text = "<think>\nx ```python\na=1\n```\n</think>\n```python\nb=2\n```"
    assert fmt.extract_code(text) == "b=2"
    assert fmt.extract_code("t <|return|> t <|return|>\n```python\nc=3\n```") == "c=3"
    assert fmt.extract_code(chatml_completion("q = 1")) == "q = 1"
    assert last_python_block("```py\nx\n```") is None


def test_aux_call_harmony():
    client = scripted(("analysis plan assistantfinal  ANSWER\n", "stop"))
    text, finish = harmony().aux_call(client, "QUESTION", effort="low", max_tokens=99,
                                      temperature=0.0, timeout_s=12.5)
    assert (text, finish) == ("ANSWER", "stop")
    (prompt, params), = client.calls
    assert prompt == HarmonyTwoPhase(HARMONY, Sampling()).render("QUESTION").replace(
        "Reasoning: high", "Reasoning: low")
    assert params == {"max_tokens": 99, "temperature": 0.0, "top_p": 1.0, "top_k": -1,
                      "stop": ["<|return|>"], "extra": None, "timeout_s": 12.5}
    assert harmony().aux_call(scripted(("no marker", "length")), "Q", effort="low",
                              max_tokens=5) == ("", "length")


def test_aux_call_chatml():
    fmt = build_format(CHATML, Sampling())
    client = scripted(("a</think>b</think>  ANSWER ", "stop"))
    assert fmt.aux_call(client, "QUESTION", effort="low", max_tokens=7,
                        timeout_s=3.0) == ("ANSWER", "stop")
    (prompt, params), = client.calls
    assert prompt == fmt.render("QUESTION")
    assert params["stop"] == ["<|im_end|>"] and params["extra"] is None
    assert params["timeout_s"] == 3.0
    assert fmt.aux_call(scripted("unclosed"), "Q", effort="low", max_tokens=5) == ("", "stop")


def test_fake_helpers_round_trip():
    roomy = Model(name="m", format="harmony_two_phase", phase1_max_tokens=26000)
    for name, model in (("harmony_two_phase", roomy), ("chatml_think", CHATML)):
        fmt = build_format(model, Sampling())
        client = FakeClient(program_responder(name, "z = 3", aux_answer="NOTE"))
        assert fmt.extract_code(fmt.complete(fmt.render("q"), client)) == "z = 3"
        assert fmt.aux_call(client, "q", effort="low", max_tokens=10)[0] == "NOTE"
        assert fmt.extract_code(completion_for(name, "y = 1")) == "y = 1"
        assert aux_reply_for(name, "A").endswith("A")
    assert FakeClient(lambda p, q: "").count_tokens("abcdefgh") == 2


class _Stub(BaseHTTPRequestHandler):
    seen: list = []
    health_codes: list = []

    def log_message(self, *args):
        pass

    def _reply(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        code = self.health_codes.pop(0) if self.health_codes else 200
        self.seen.append(("GET", self.path, dict(self.headers), None))
        self._reply(code, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.seen.append(("POST", self.path, dict(self.headers), body))
        if body.get("prompt") == "bad":
            self._reply(400, {"error": "prompt too long"})
        elif body.get("prompt") in ("no-choices", "not-json", "slow"):
            if body["prompt"] == "slow":
                time.sleep(1.0)
            payload = b"<html>oops</html>" if body["prompt"] == "not-json" else b'{"choices": []}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif self.path == "/tokenize":
            self._reply(200, {"count": 7, "tokens": [1] * 7})
        else:
            self._reply(200, {"choices": [{"text": "hi", "finish_reason": "length"}]})


@pytest.fixture
def server():
    _Stub.seen, _Stub.health_codes = [], []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def test_client_request_bodies(server):
    client = VLLMClient(server + "/", "served", api_key="sekret", timeout_s=5)
    out = client.generate("hello", max_tokens=12, temperature=0.5, top_p=0.8, top_k=4,
                          stop=["<|return|>"], extra={"skip_special_tokens": False})
    assert out == ("hi", "length")
    assert client.count_tokens("hello") == 7
    (_, path1, headers1, body1), (_, path2, _, body2) = _Stub.seen
    assert path1 == "/v1/completions" and headers1["Authorization"] == "Bearer sekret"
    assert body1 == {"model": "served", "prompt": "hello", "max_tokens": 12, "temperature": 0.5,
                     "top_p": 0.8, "top_k": 4, "stop": ["<|return|>"], "skip_special_tokens": False}
    assert path2 == "/tokenize"
    assert body2 == {"model": "served", "prompt": "hello", "add_special_tokens": False}
    assert "sekret" not in repr(client)


def test_client_without_key_and_errors(server):
    client = VLLMClient(server, "served", timeout_s=5)
    client.generate("x", max_tokens=1)
    assert "Authorization" not in _Stub.seen[-1][2]
    assert "stop" not in _Stub.seen[-1][3]
    with pytest.raises(RuntimeError, match="400.*prompt too long"):
        client.generate("bad", max_tokens=1)


def test_wait_ready(server):
    _Stub.health_codes = [503, 503]
    VLLMClient(server, "served").wait_ready(timeout_s=5, interval_s=0.01)
    assert [s[1] for s in _Stub.seen] == ["/health"] * 3
    _Stub.health_codes = [503] * 1000
    with pytest.raises(RuntimeError, match="not ready"):
        VLLMClient(server, "served").wait_ready(timeout_s=0.2, interval_s=0.01)


def test_client_per_call_timeout(server, monkeypatch):
    import ttt.llm.client as client_module

    seen = []
    real_post = client_module.requests.post

    def post(url, **kwargs):
        seen.append(kwargs["timeout"])
        return real_post(url, **kwargs)

    monkeypatch.setattr(client_module.requests, "post", post)
    client = VLLMClient(server, "served", timeout_s=7)
    client.generate("x", max_tokens=1)
    client.generate("x", max_tokens=1, timeout_s=2.5)
    client.count_tokens("x")
    assert seen == [7.0, 2.5, 7.0]


def test_client_failures_are_server_errors(server):
    client = VLLMClient(server, "served", timeout_s=5)
    for prompt, match in (("bad", "400"), ("no-choices", "lacks"), ("not-json", "not JSON")):
        with pytest.raises(ServerError, match=match):
            client.generate(prompt, max_tokens=1)
    with pytest.raises(ServerError, match="failed"):
        client.generate("slow", max_tokens=1, timeout_s=0.2)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        dead = f"http://127.0.0.1:{sock.getsockname()[1]}"
    with pytest.raises(ServerError, match="failed"):
        VLLMClient(dead, "served", timeout_s=5).generate("x", max_tokens=1)
    with pytest.raises(ServerError, match="failed"):
        VLLMClient(dead, "served", timeout_s=5).count_tokens("x")


def test_prompt_budget_errors_are_not_server_errors():
    with pytest.raises(ValueError) as info:
        harmony().complete("P" * 120, scripted())
    assert not isinstance(info.value, ServerError)
