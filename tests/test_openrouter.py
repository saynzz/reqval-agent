"""Провайдер OpenRouter: запрос строго по JSON-схеме, разбор ответа, стоимость, понятные ошибки.

Сеть не нужна: поднимается локальный HTTP-сервер, который отвечает как OpenRouter.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from reqval.llm import LLMError, OpenRouterProvider, make_provider, strict_json_schema
from reqval.models import LLMReqBatch


class FakeOpenRouter(BaseHTTPRequestHandler):
    replies: list[tuple[int, dict]] = []
    requests: list[dict] = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOpenRouter.requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        code, payload = FakeOpenRouter.replies.pop(0)
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def server(monkeypatch):
    for var in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "*")
    srv = HTTPServer(("127.0.0.1", 0), FakeOpenRouter)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    FakeOpenRouter.replies, FakeOpenRouter.requests = [], []
    monkeypatch.setenv("OPENROUTER_BASE_URL", f"http://127.0.0.1:{srv.server_port}/api/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemini-test")
    monkeypatch.setenv("LLM_MIN_INTERVAL", "0")
    yield srv
    srv.shutdown()


def _ok(content: dict, cost: float = 0.0123) -> tuple[int, dict]:
    return 200, {"choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)},
                              "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "cost": cost}}


BATCH = {"assessments": [{"req_id": "R-01", "issues": [{
    "criterion": "C7", "severity": "критичный", "title": "Нет порога измерения", "quote": "в течение минуты",
    "explanation": "e", "recommendation": "r"}]}]}


def test_request_uses_strict_schema_and_parses_reply(server):
    FakeOpenRouter.replies = [_ok(BATCH)]
    p = make_provider("openrouter", None)
    res = p.generate("sys", "user", LLMReqBatch, label="t")
    assert res.assessments[0].issues[0].criterion == "C7"

    req = FakeOpenRouter.requests[0]
    assert req["path"] == "/api/v1/chat/completions" and req["auth"] == "Bearer sk-or-test"
    body = req["body"]
    assert body["model"] == "google/gemini-test" and body["temperature"] == 0
    rf = body["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
    assert '"C9"' in json.dumps(rf["json_schema"]["schema"]), "закрытый список критериев уходит в схему"
    assert body["provider"]["require_parameters"] is True

    assert p.usage.calls == 1 and p.usage.input_tokens == 1000 and p.usage.cost_usd == pytest.approx(0.0123)


def test_invalid_json_is_retried_then_accepted(server):
    bad = (200, {"choices": [{"message": {"content": '{"assessments": [{"req_id": "R-01", "issues": [{"criterion": "C10"'}}],
                 "usage": {"cost": 0.001}})
    FakeOpenRouter.replies = [bad, _ok(BATCH, cost=0.002)]
    p = OpenRouterProvider()
    assert p.generate("sys", "user", LLMReqBatch).assessments[0].req_id == "R-01"
    assert p.usage.retries == 1 and p.usage.cost_usd == pytest.approx(0.003)
    assert "не прошёл валидацию" in FakeOpenRouter.requests[1]["body"]["messages"][1]["content"]


def test_rate_limit_is_retried(server, monkeypatch):
    monkeypatch.setattr("reqval.llm.time.sleep", lambda s: None)
    FakeOpenRouter.replies = [(429, {"error": {"code": 429, "message": "rate limited"}}), _ok(BATCH)]
    p = OpenRouterProvider()
    assert p.generate("sys", "user", LLMReqBatch).assessments
    assert p.usage.retries == 1


def test_no_balance_fails_fast_with_reason(server):
    FakeOpenRouter.replies = [(402, {"error": {"code": 402, "message": "Insufficient credits"}})]
    with pytest.raises(LLMError, match="402.*Insufficient credits"):
        OpenRouterProvider().generate("sys", "user", LLMReqBatch)
    assert len(FakeOpenRouter.requests) == 1, "оплата не пройдёт от повтора — не тратим время"


def test_config_errors_are_explicit(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        OpenRouterProvider("google/x")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    with pytest.raises(LLMError, match="OPENROUTER_MODEL"):
        OpenRouterProvider()


def test_strict_schema_shape():
    s = strict_json_schema(LLMReqBatch)
    txt = json.dumps(s)
    assert "$ref" not in txt and "$defs" not in txt
    issue = s["properties"]["assessments"]["items"]["properties"]["issues"]["items"]
    assert issue["additionalProperties"] is False and set(issue["required"]) == set(issue["properties"])
