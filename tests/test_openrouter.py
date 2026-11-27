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
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY.*openrouter.ai/settings/keys.*--provider gemini"):
        OpenRouterProvider("google/x")


def test_defaults_are_the_submission_model(monkeypatch):
    """По умолчанию — модель, на которой сделан сдаваемый отчёт: OpenRouter + Claude Sonnet 5.5."""
    from reqval.llm import GeminiProvider, make_provider
    for var in ("LLM_PROVIDER", "OPENROUTER_MODEL", "GEMINI_MODEL", "GEMINI_FALLBACK_MODEL", "LLM_FALLBACK_MODEL",
                "OPENROUTER_MODE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    p = make_provider(None, None)
    assert isinstance(p, OpenRouterProvider) and p.model == "anthropic/claude-sonnet-5.5"
    assert p.mode == 2, "Claude: строгая схема без seed и temperature"
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    g = GeminiProvider()
    assert g.model == "gemini-3.8-flash" and g.fallback_model == "gemini-3.5-flash-lite"
    assert GeminiProvider("gemini-3.5-flash-lite").fallback_model is None


def test_strict_schema_shape():
    s = strict_json_schema(LLMReqBatch)
    txt = json.dumps(s)
    assert "$ref" not in txt and "$defs" not in txt
    issue = s["properties"]["assessments"]["items"]["properties"]["issues"]["items"]
    assert issue["additionalProperties"] is False and set(issue["required"]) == set(issue["properties"])


# --------------------------------------------------------------------------- обрезанный ответ / диагностика

def test_truncated_answer_fails_fast_with_advice_and_dump(server, tmp_path):
    cut = (200, {"choices": [{"message": {"content": '{"assessments": [{"req_id": "R-01", "issues": [{"crit'},
                              "finish_reason": "length"}], "usage": {"completion_tokens": 32768}})
    FakeOpenRouter.replies = [cut]
    p = OpenRouterProvider()
    p.debug_dir = str(tmp_path / "dbg")
    with pytest.raises(LLMError, match="обрезан по лимиту длины.*LLM_MAX_OUTPUT_TOKENS"):
        p.generate("sys", "user", LLMReqBatch, label="проверка требований 1/11")
    assert len(FakeOpenRouter.requests) == 1, "повтор с теми же параметрами бессмыслен — не тратим деньги"
    dumps = list((tmp_path / "dbg").iterdir())
    assert len(dumps) == 1 and '"crit' in dumps[0].read_text(encoding="utf-8")
    assert "finish_reason: length" in dumps[0].read_text(encoding="utf-8")


def test_schema_error_is_explained_in_log(server, tmp_path, caplog):
    wrong = _ok({"assessments": [{"req_id": "R-01", "issues": [{"criterion": "C8", "severity": "важный",
                 "title": "t", "quote": "q", "explanation": "e", "recommendation": "r"}]}]})
    FakeOpenRouter.replies = [wrong, _ok(BATCH)]
    p = OpenRouterProvider()
    p.debug_dir = str(tmp_path / "dbg")
    with caplog.at_level("WARNING", logger="reqval"):
        p.generate("sys", "user", LLMReqBatch, label="t")
    msg = caplog.text
    assert "severity" in msg and "finish=stop" in msg, "в логе видно, какое поле не прошло проверку"
    assert len(list((tmp_path / "dbg").iterdir())) == 1


def test_output_limit_and_reasoning_are_sent(server, monkeypatch):
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "12000")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "low")
    FakeOpenRouter.replies = [_ok(BATCH)]
    OpenRouterProvider().generate("sys", "user", LLMReqBatch)
    body = FakeOpenRouter.requests[0]["body"]
    assert body["max_tokens"] == 12000 and body["reasoning"] == {"effort": "low"}


def test_reasoning_not_sent_by_default(server, monkeypatch):
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)
    FakeOpenRouter.replies = [_ok(BATCH)]
    OpenRouterProvider().generate("sys", "user", LLMReqBatch)
    assert "reasoning" not in FakeOpenRouter.requests[0]["body"]


def test_gemini_max_tokens_is_reported(monkeypatch):
    pytest.importorskip("google.genai")
    from types import SimpleNamespace

    from reqval.llm import GeminiProvider
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    p = GeminiProvider("gemini-test")
    resp = SimpleNamespace(text="", usage_metadata=None,
                           candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="MAX_TOKENS"))])
    calls = []
    monkeypatch.setattr(p.client.models, "generate_content", lambda **kw: calls.append(kw) or resp)
    with pytest.raises(LLMError, match="обрезан по лимиту длины"):
        p.generate("sys", "user", LLMReqBatch)
    assert len(calls) == 1 and calls[0]["config"].max_output_tokens == 32768


# --------------------------------------------------------------------------- лестница совместимости (Claude и т.п.)

NO_ENDPOINTS = (404, {"error": {"code": 404, "message": "No endpoints found that can handle the requested "
                                                       "parameters. To learn more about provider routing..."}})


def test_no_endpoints_drops_seed_and_remembers(server):
    """Провайдер модели не принимает seed — агент убирает только его, строгая схема остаётся."""
    FakeOpenRouter.replies = [NO_ENDPOINTS, _ok(BATCH), _ok(BATCH)]
    p = OpenRouterProvider("vendor/model-without-seed")
    assert p.generate("sys", "user", LLMReqBatch).assessments
    first, second = (r["body"] for r in FakeOpenRouter.requests[:2])
    assert "seed" in first and "seed" not in second
    assert second["response_format"]["type"] == "json_schema", "строгая схема сохранена"
    assert p.usage.retries == 0, "переход на другой режим — не повтор и не ошибка"
    p.generate("sys", "user", LLMReqBatch)
    assert "seed" not in FakeOpenRouter.requests[2]["body"], "режим запомнен до конца прогона"
    assert "без seed" in p.describe()


def test_ladder_reaches_prompt_schema_mode(server, monkeypatch):
    monkeypatch.setenv("LLM_REASONING_EFFORT", "low")
    FakeOpenRouter.replies = [NO_ENDPOINTS] * 5 + [_ok(BATCH)]
    p = OpenRouterProvider("vendor/old-model")
    assert p.generate("sys", "user", LLMReqBatch).assessments
    bodies = [r["body"] for r in FakeOpenRouter.requests]
    assert "seed" in bodies[0] and "temperature" in bodies[0]
    assert "seed" not in bodies[1] and "temperature" in bodies[1]
    assert "temperature" not in bodies[2] and "reasoning" in bodies[2] and "response_format" in bodies[2]
    assert "reasoning" not in bodies[3] and "response_format" in bodies[3]
    assert "tools" in bodies[4] and "response_format" not in bodies[4], "перед промптом пробуется tool calling"
    last = bodies[5]
    assert "response_format" not in last and "provider" not in last
    assert '"C9"' in last["messages"][0]["content"], "схема со справочником критериев ушла в промпт"
    assert "в промпте" in p.describe()


def test_ladder_gives_up_with_original_error(server):
    FakeOpenRouter.replies = [NO_ENDPOINTS] * 6
    with pytest.raises(LLMError, match="404.*No endpoints"):
        OpenRouterProvider("vendor/x").generate("sys", "user", LLMReqBatch)
    assert len(FakeOpenRouter.requests) == 6


def test_other_404_is_not_a_ladder_case(server):
    FakeOpenRouter.replies = [(404, {"error": {"code": 404, "message": "Model vendor/typo does not exist"}})]
    with pytest.raises(LLMError, match="does not exist"):
        OpenRouterProvider("vendor/typo").generate("sys", "user", LLMReqBatch)
    assert len(FakeOpenRouter.requests) == 1, "опечатка в id модели — сразу понятная ошибка, без перебора"


# --------------------------------------------------------------------------- схема и ответы без title

from pydantic import BaseModel  # noqa: E402

from reqval import models as M  # noqa: E402


def _nested_models(model: type[BaseModel]):
    yield model
    for f in model.model_fields.values():
        for arg in getattr(f.annotation, "__args__", ()) or ():
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                yield from _nested_models(arg)


@pytest.mark.parametrize("model", [M.LLMExtraction, M.LLMReqBatch, M.LLMSetResult, M.LLMVerdict])
def test_strict_schema_keeps_every_field(model):
    """Регрессия: поле с именем «title» удалялось вместе со служебным ключом JSON-схемы, и в strict-режиме
    модель не могла его вернуть, а агент отбрасывал каждый ответ как невалидный."""
    schema = strict_json_schema(model)

    def objects(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                yield node
            for v in node.values():
                yield from objects(v)
        elif isinstance(node, list):
            for v in node:
                yield from objects(v)

    schema_fields = {frozenset(o["properties"]) for o in objects(schema)}
    for m in _nested_models(model):
        assert frozenset(m.model_fields) in schema_fields, f"{m.__name__}: не все поля попали в схему"
    for o in objects(schema):
        assert set(o["required"]) == set(o["properties"]) and o["additionalProperties"] is False
        assert all("default" not in p for p in o["properties"].values())


CLAUDE_STYLE = {"assessments": [{"req_id": "R-01", "issues": [
    {"criterion": "C3", "severity": "значительный", "quote": "в момент покупки",
     "explanation": "«В момент покупки» не определено: мгновенно, в течение минуты или при проведении операции.",
     "recommendation": "Задать измеримый срок."},
    {"criterion": "C7", "severity": "значительный", "quote": "в момент покупки", "title": "",
     "explanation": "Цель не содержит измеримого критерия достижения (срок, доля клиентов, канал), поэтому "
                    "нельзя объективно проверить её выполнение.", "recommendation": "r"}]}]}


def test_answer_without_title_is_accepted_first_time(server):
    FakeOpenRouter.replies = [_ok(CLAUDE_STYLE)]
    p = OpenRouterProvider("anthropic/claude-test")
    res = p.generate("sys", "user", LLMReqBatch)
    assert p.usage.retries == 0 and len(FakeOpenRouter.requests) == 1
    t1, t2 = (i.title for i in res.assessments[0].issues)
    assert t1 == "«В момент покупки» не определено"
    assert 10 < len(t2) <= 81 and t2.startswith("Цель не содержит измеримого критерия")


def test_set_issue_without_title():
    s = M.LLMSetIssue(criterion="S2", severity="критичный", explanation="Ф-2 противоречит критерию №2. Подробнее…",
                      recommendation="r", evidence=[])
    assert s.title == "Ф-2 противоречит критерию №2"


# --------------------------------------------------------------------------- tool calling (Claude)

def _tool_reply(args: dict, as_string: bool = True, cost: float = 0.02) -> tuple[int, dict]:
    a = json.dumps(args, ensure_ascii=False) if as_string else args
    return 200, {"choices": [{"message": {"content": None, "tool_calls": [{"id": "call_1", "type": "function",
                 "function": {"name": "submit_review", "arguments": a}}]}, "finish_reason": "tool_calls"}],
                 "usage": {"prompt_tokens": 900, "completion_tokens": 300, "cost": cost}}


def test_claude_starts_with_strict_schema_without_temperature(server, monkeypatch):
    """Claude 5.x думает всегда: temperature не задаётся, а строгая схема и effort доступны."""
    monkeypatch.setenv("LLM_REASONING_EFFORT", "low")
    monkeypatch.delenv("OPENROUTER_MODE", raising=False)
    FakeOpenRouter.replies = [_ok(BATCH)]
    p = OpenRouterProvider("anthropic/claude-sonnet-5.5")
    assert p.generate("sys", "user", LLMReqBatch).assessments[0].issues[0].criterion == "C7"
    assert len(FakeOpenRouter.requests) == 1, "для Claude — сразу рабочий режим, без 404"
    body = FakeOpenRouter.requests[0]["body"]
    assert body["response_format"]["json_schema"]["strict"] is True
    assert "temperature" not in body and "seed" not in body
    assert body["reasoning"] == {"effort": "low"} and body["provider"]["require_parameters"] is True
    d = p.describe()
    assert "temperature не задаётся" in d and "reasoning=low" in d


def test_claude_path_down_to_tool_calling(server, monkeypatch):
    monkeypatch.delenv("OPENROUTER_MODE", raising=False)
    FakeOpenRouter.replies = [NO_ENDPOINTS, NO_ENDPOINTS, _tool_reply(BATCH)]
    p = OpenRouterProvider("anthropic/claude-sonnet-5.5")
    assert p.generate("sys", "user", LLMReqBatch).assessments
    body = FakeOpenRouter.requests[2]["body"]
    tool = body["tools"][0]["function"]
    assert body["tool_choice"] == {"type": "function", "function": {"name": "submit_review"}}
    assert "title" in json.dumps(tool["parameters"]) and '"C9"' in json.dumps(tool["parameters"])
    assert not {"response_format", "seed", "temperature", "reasoning"} & set(body)
    assert "tool calling" in p.describe() and p.usage.cost_usd == pytest.approx(0.02)


def test_tool_arguments_as_object(server, monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODE", "tool")
    FakeOpenRouter.replies = [_tool_reply(BATCH, as_string=False)]
    assert OpenRouterProvider("anthropic/claude-x").generate("sys", "user", LLMReqBatch).assessments


def test_tool_answer_is_validated_too(server, monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODE", "tool")
    bad = {"assessments": [{"req_id": "R-01", "issues": [{**BATCH["assessments"][0]["issues"][0], "criterion": "C10"}]}]}
    FakeOpenRouter.replies = [_tool_reply(bad), _tool_reply(BATCH)]
    p = OpenRouterProvider("anthropic/claude-x")
    assert p.generate("sys", "user", LLMReqBatch).assessments[0].issues[0].criterion == "C7"
    assert p.usage.retries == 1


def test_tool_mode_unavailable_falls_to_prompt(server, monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODE", "tool")
    FakeOpenRouter.replies = [NO_ENDPOINTS, _ok(BATCH)]
    p = OpenRouterProvider("anthropic/claude-x")
    assert p.generate("sys", "user", LLMReqBatch).assessments
    assert "tools" not in FakeOpenRouter.requests[1]["body"] and "в промпте" in p.describe()


@pytest.mark.parametrize("mode, key", [("json_schema", "response_format"), ("tool", "tools")])
def test_mode_can_be_forced(server, monkeypatch, mode, key):
    monkeypatch.setenv("OPENROUTER_MODE", mode)
    FakeOpenRouter.replies = [_tool_reply(BATCH) if mode == "tool" else _ok(BATCH)]
    OpenRouterProvider("anthropic/claude-x" if mode == "json_schema" else "google/gemini-x").generate(
        "sys", "user", LLMReqBatch)
    assert key in FakeOpenRouter.requests[0]["body"]
