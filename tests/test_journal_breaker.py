"""Задачи 12 и А.

12. Журнал всех обращений к модели (report.llm.jsonl) — доказательство, что отчёт сделан агентом (ТЗ §2),
    и кэш ответов для отладки (по умолчанию выключен, в паспорте отмечается).
А.  Запасная модель при недоступности основной и предохранитель: если модель не отвечает несколько шагов подряд,
    прогон не висит по несколько минут на каждом шаге, а сразу пишет отчёт «ПРОВЕРКА НЕ ЗАВЕРШЕНА».

Фиктивный провайдер подменяет только сетевой вызов _raw — повторы, журнал, кэш и предохранитель настоящие.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from reqval import cli
from reqval.llm import BaseProvider, CircuitOpen, LLMError, SchemaError, Unavailable
from reqval.loader import load_document
from reqval.models import LLMExtraction, LLMReqBatch, LLMSetResult, LLMVerdict
from reqval.pipeline import INCOMPLETE, Pipeline
from reqval.report import build_markdown

DOC = """# ТЗ

## Функциональные требования
* **Ф-1.** Пользователь задаёт порог от 1 000 до 1 000 000 ₸ включительно.
* **Ф-2.** Месяц определяется по дате совершения покупки в Алматы.
"""


class Overloaded(Exception):
    code = 503

    def __str__(self):
        return "503 UNAVAILABLE: model is overloaded"


def answer(schema, prompt) -> str:
    if schema is LLMExtraction:
        return json.dumps({"requirements": [
            {"doc_id": "Ф-1", "kind": "ф", "summary": "порог", "line_from": 4, "line_to": 4,
             "quote": "Ф-1. Пользователь задаёт порог от 1 000 до 1 000 000 ₸ включительно."},
            {"doc_id": "Ф-2", "kind": "ф", "summary": "месяц", "line_from": 5, "line_to": 5,
             "quote": "Ф-2. Месяц определяется по дате совершения покупки в Алматы."}]}, ensure_ascii=False)
    if schema is LLMReqBatch:
        return json.dumps({"assessments": [
            {"req_id": "R-01", "issues": [{"criterion": "C4", "severity": "значительный", "title": "Нет валюты",
                                           "quote": "от 1 000 до 1 000 000", "explanation": "e", "recommendation": "r"}]},
            {"req_id": "R-02", "issues": []}]}, ensure_ascii=False)
    if schema is LLMSetResult:
        return json.dumps({"issues": []})
    return json.dumps({"development_comment": "d", "testing_comment": "t",
                       "priorities": [{"finding_ids": ["D-001"], "action": "Указать валюту"}]}, ensure_ascii=False)


class RawFake(BaseProvider):
    """Сценарий по моделям: {"model": ["ok" | "503" | "bad", ...]}; когда сценарий кончился — "ok"."""

    name = "fake"

    def __init__(self, model="primary", script=None, fallback=None):
        super().__init__()
        self.model, self.fallback_model = model, fallback
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.raw_calls: list[tuple[str, str]] = []
        self.max_retries = 3

    def describe(self):
        return "fake params"

    def _raw(self, system, prompt, schema):
        self.raw_calls.append((self.model, schema.__name__))
        step = (self.script.get(self.model) or ["ok"]).pop(0) if self.script.get(self.model) else "ok"
        if step == "503":
            raise Overloaded()
        if step == "bad":
            return '{"assessments": [{"req_id": "R-01", "issues": [{"criterion": "C10"}]}]}', 10, 5
        self.last_finish = "stop"
        return answer(schema, prompt), 100, 20


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("reqval.llm.time.sleep", lambda s: None)


@pytest.fixture
def doc(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text(DOC, encoding="utf-8")
    return load_document(str(p))


def _journal(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


# --------------------------------------------------------------------------- 12. журнал

def test_every_call_is_journaled(doc, tmp_path):
    p = RawFake(script={"primary": ["ok", "bad", "ok"]})
    p.journal_path = str(tmp_path / "r.llm.jsonl")
    res = Pipeline(doc, p).run()
    recs = _journal(p.journal_path)
    assert len(recs) == len(p.raw_calls) == p.usage.journal_records
    assert [r["outcome"] for r in recs].count("schema_error") == 1
    bad = next(r for r in recs if r["outcome"] == "schema_error")
    assert bad["step"].startswith("проверка требований") and bad["attempt"] == 1 and "C10" in bad["response"]
    for r in recs:
        assert {"ts", "step", "model", "params", "prompt_sha256", "system_sha256"} <= set(r)
        assert len(r["prompt_sha256"]) == 64
    oks = [r for r in recs if r["outcome"] == "ok"]
    assert {r["step"] for r in oks} == {"нарезка", "проверка требований 1/1", "проверка набора", "итог"}
    assert all(r["response"] and r["input_tokens"] == 100 for r in oks)

    md = build_markdown(res)
    digest = hashlib.sha256(Path(p.journal_path).read_bytes()).hexdigest()
    row = next(l for l in md.splitlines() if l.startswith("| Журнал обращений к модели"))
    assert "r.llm.jsonl" in row and digest[:16] in row and f"{len(recs)} записей" in row


def test_api_key_never_in_journal(tmp_path, monkeypatch):
    import threading
    from http.server import HTTPServer

    from test_openrouter import BATCH, FakeOpenRouter, _ok

    from reqval.llm import OpenRouterProvider
    for var in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "*")
    srv = HTTPServer(("127.0.0.1", 0), FakeOpenRouter)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("OPENROUTER_BASE_URL", f"http://127.0.0.1:{srv.server_port}/api/v1")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-secret-123")
        FakeOpenRouter.replies, FakeOpenRouter.requests = [_ok(BATCH)], []
        p = OpenRouterProvider("google/x")
        p.journal_path = str(tmp_path / "j.jsonl")
        p.generate("sys", "user", LLMReqBatch, label="t")
        text = Path(p.journal_path).read_text(encoding="utf-8")
        assert FakeOpenRouter.requests[0]["auth"] == "Bearer sk-or-secret-123", "ключ ушёл в запрос"
        assert "sk-or-secret-123" not in text and '"outcome": "ok"' in text, "но не в журнал"
    finally:
        srv.shutdown()


def test_cache_replays_without_calls(doc, tmp_path):
    first = RawFake()
    first.cache_dir = str(tmp_path / "cache")
    r1 = Pipeline(doc, first).run()
    second = RawFake()
    second.cache_dir = str(tmp_path / "cache")
    second.journal_path = str(tmp_path / "j.jsonl")
    r2 = Pipeline(doc, second).run()
    assert second.raw_calls == [] and second.usage.cached == len(first.raw_calls)
    assert [f.title for f in r1.unique_findings] == [f.title for f in r2.unique_findings]
    assert {r["outcome"] for r in _journal(second.journal_path)} == {"cache"}
    assert "ответов из кэша" in build_markdown(r2)


def test_cache_misses_when_document_changes(doc, tmp_path):
    p = RawFake()
    p.cache_dir = str(tmp_path / "cache")
    Pipeline(doc, p).run()
    other = tmp_path / "other.md"
    other.write_text(DOC.replace("1 000 000", "2 000 000"), encoding="utf-8")
    p2 = RawFake()
    p2.cache_dir = str(tmp_path / "cache")
    Pipeline(load_document(str(other)), p2).run()
    assert p2.raw_calls, "другой документ — другие промпты — ответы из кэша не подходят"


def test_cli_writes_journal_next_to_report(doc, tmp_path, monkeypatch):
    fake = RawFake()
    monkeypatch.setattr(cli, "make_provider", lambda kind, model: fake)
    out = tmp_path / "out" / "report.md"
    assert cli.main([doc.path, str(out)]) == 0
    j = out.with_name("report.llm.jsonl")
    assert j.exists() and len(_journal(j)) == len(fake.raw_calls)
    assert json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))["run"]["journal"] == "report.llm.jsonl"
    # повторный прогон перезаписывает журнал, а не дописывает
    fake2 = RawFake()
    monkeypatch.setattr(cli, "make_provider", lambda kind, model: fake2)
    assert cli.main([doc.path, str(out)]) == 0
    assert len(_journal(j)) == len(fake2.raw_calls)


def test_cli_cache_flag(doc, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "make_provider", lambda kind, model: RawFake())
    assert cli.main([doc.path, str(tmp_path / "r.md"), "--cache"]) == 0
    assert any((tmp_path / ".reqval_cache").iterdir())


# --------------------------------------------------------------------------- А. запасная модель и предохранитель

def test_fallback_model_takes_over(doc, tmp_path):
    p = RawFake(script={"primary": ["503"] * 3}, fallback="backup")
    p.journal_path = str(tmp_path / "j.jsonl")
    res = Pipeline(doc, p).run()
    assert res.complete and res.unchecked == []
    assert p.usage.steps["нарезка"] == "backup" and p.model == "backup"
    assert [m for m, _ in p.raw_calls[:3]] == ["primary"] * 3 and p.raw_calls[3][0] == "backup"
    assert any(r["outcome"] == "fallback" and r["next_model"] == "backup" for r in _journal(p.journal_path))
    model_row = next(l for l in build_markdown(res).splitlines() if l.startswith("| Модель |"))
    assert "primary" in model_row and "запасная модель backup" in model_row and "нарезка" in model_row


def test_breaker_stops_waiting(doc):
    p = RawFake(script={"primary": ["503"] * 100})
    res = Pipeline(doc, p).run()
    assert len(p.raw_calls) == 2 * p.max_retries, "нарезка и первый пакет — по 3 попытки, дальше ни одной"
    assert p.tripped and set(p.usage.skipped) >= {"проверка набора"}
    assert res.verdict.status == INCOMPLETE and not res.complete
    assert "Шаги, пропущенные предохранителем" in build_markdown(res)


def test_breaker_raises_circuit_open_without_calls():
    p = RawFake(script={"primary": ["503"] * 100})
    for _ in range(2):
        with pytest.raises(Unavailable):
            p.generate("s", "u", LLMVerdict, label="x")
    n = len(p.raw_calls)
    with pytest.raises(CircuitOpen):
        p.generate("s", "u", LLMVerdict, label="y")
    assert len(p.raw_calls) == n


def test_bad_answers_do_not_trip_breaker():
    p = RawFake(script={"primary": ["bad"] * 6})
    for _ in range(2):
        with pytest.raises(SchemaError):
            p.generate("s", "u", LLMReqBatch, label="x")
    assert not p.tripped, "модель отвечает (пусть и плохо) — это не недоступность"
    assert p.generate("s", "u", LLMVerdict, label="z").development_comment == "d"


def test_success_resets_breaker():
    p = RawFake(script={"primary": ["503"] * 3 + ["ok"] + ["503"] * 3})
    with pytest.raises(Unavailable):
        p.generate("s", "u", LLMVerdict, label="a")
    p.generate("s", "u", LLMVerdict, label="b")
    with pytest.raises(Unavailable):
        p.generate("s", "u", LLMVerdict, label="c")
    assert not p.tripped


def test_fallback_also_down_trips_breaker(doc):
    p = RawFake(script={"primary": ["503"] * 50, "backup": ["503"] * 50}, fallback="backup")
    res = Pipeline(doc, p).run()
    assert p.tripped and res.verdict.status == INCOMPLETE
    assert len(p.raw_calls) <= 3 * p.max_retries


def test_openrouter_fallback_recomputes_mode(monkeypatch):
    from reqval.llm import OpenRouterProvider
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("OPENROUTER_MODE", raising=False)
    monkeypatch.setenv("OPENROUTER_FALLBACK_MODEL", "anthropic/claude-x")
    p = OpenRouterProvider("google/gemini-x")
    assert p.mode == 0 and p.fallback_model == "anthropic/claude-x"
    p._switch_model(p.fallback_model)
    assert p.mode == 2, "для Claude сразу режим без seed и temperature"
