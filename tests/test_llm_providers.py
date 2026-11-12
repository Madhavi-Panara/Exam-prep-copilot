"""Exercise the OpenAI-compatible provider against a tiny local fake server (no network, no API key)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from backend.app import config
from backend.app.errors import LLMError
from backend.app.llm import LLMClient


class _Handler(BaseHTTPRequestHandler):
    replies = []   # list of (status, body) consumed in order
    requests = []  # captured (path, auth header, json body)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append((self.path, self.headers.get("Authorization"), body))
        status, payload = type(self).replies.pop(0)
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def ok(text):
    return 200, {"choices": [{"message": {"content": text}}]}


@pytest.fixture
def server(monkeypatch):
    _Handler.replies, _Handler.requests = [], []
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(config, "LLM_BASE_URL", f"http://127.0.0.1:{httpd.server_port}/v1")
    monkeypatch.setattr(config, "LLM_API_KEY", "secret-key")
    monkeypatch.setattr(config, "MODEL_NAME", "test-model")
    monkeypatch.setattr("time.sleep", lambda s: None)
    yield _Handler
    httpd.shutdown()


def test_request_shape_and_reply(server):
    server.replies = [ok("  hello  ")]
    out = LLMClient().complete("be brief", "hi", max_tokens=50)
    assert out == "hello"
    path, auth, body = server.requests[0]
    assert path == "/v1/chat/completions" and auth == "Bearer secret-key"
    assert body["model"] == "test-model" and body["max_tokens"] == 50
    assert body["messages"][0] == {"role": "system", "content": "be brief"}
    assert body["messages"][1] == {"role": "user", "content": "hi"}


def test_full_message_history_is_passed_through(server):
    server.replies = [ok("ok")]
    history = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}]
    LLMClient().complete("sys", history)
    assert [m["role"] for m in server.requests[0][2]["messages"]] == ["system", "user", "assistant", "user"]


def test_complete_json_works_through_provider(server):
    server.replies = [ok('```json\n{"a": 1}\n```')]
    assert LLMClient().complete_json("sys", "give json") == {"a": 1}


def test_rate_limit_is_retried_then_succeeds(server):
    server.replies = [(429, {"error": "slow down"}), ok("fine")]
    assert LLMClient().complete("s", "u") == "fine"
    assert len(server.requests) == 2


def test_persistent_rate_limit_gives_friendly_error(server):
    server.replies = [(429, {})] * 3
    with pytest.raises(LLMError, match="rate limit"):
        LLMClient().complete("s", "u")


def test_http_error_and_bad_payload(server):
    server.replies = [(401, {"error": "bad key"})]
    with pytest.raises(LLMError, match="401"):
        LLMClient().complete("s", "u")
    server.replies = [(200, {"unexpected": True})]
    with pytest.raises(LLMError, match="unexpected"):
        LLMClient().complete("s", "u")


def test_gemini_gets_extra_token_headroom(server, monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    server.replies = [ok("x")]
    LLMClient().complete("s", "u", max_tokens=100)
    assert server.requests[0][2]["max_tokens"] == 200


def test_missing_key_and_unknown_provider(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    monkeypatch.setattr(config, "LLM_BASE_URL", "http://x/v1")
    assert LLMClient().configured is False
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        LLMClient().complete("s", "u")
    monkeypatch.setattr(config, "LLM_PROVIDER", "ollama")
    assert LLMClient().configured is True  # local models need no key
    monkeypatch.setattr(config, "LLM_PROVIDER", "nonsense")
    with pytest.raises(LLMError, match="Unknown LLM_PROVIDER"):
        LLMClient().complete("s", "u")
