"""Задача Б: параллельные запросы к модели.

Главное требование — параллельный прогон даёт тот же отчёт, что и последовательный: те же дефекты в том же
порядке с теми же номерами, те же заметки. Плюс: быстрее, общая пауза между запросами соблюдается, счётчики,
журнал и лестница режимов OpenRouter не ломаются при одновременных ответах.
"""
from __future__ import annotations

import json
import random
import re
import threading
import time

import pytest

from reqval import cli
from reqval.llm import BaseProvider
from reqval.loader import load_document
from reqval.models import LLMExtraction, LLMReqBatch, LLMSetResult
from reqval.pipeline import INCOMPLETE, Pipeline

N_REQ = 20
DOC = "# ТЗ\n\n## Функциональные требования\n" + "".join(
    f"* **Ф-{i}.** Система обрабатывает операцию типа {i} и сохраняет результат номер {i} в журнале.\n"
    for i in range(1, N_REQ + 1))


class SlowFake(BaseProvider):
    """Отвечает с задержкой; задержка зависит от промпта — пакеты завершаются в «случайном» порядке."""

    name = "fake"

    def __init__(self, doc, concurrency=1, delay=0.05, jitter=True, fail=False):
        super().__init__()
        self.model, self.doc, self.delay, self.jitter, self.fail = "slow", doc, delay, jitter, fail
        self.concurrency = concurrency
        self.max_retries = 2
        self.starts: list[float] = []
        self._m = threading.Lock()

    def describe(self):
        return "fake"

    def _raw(self, system, prompt, schema):
        with self._m:
            self.starts.append(time.monotonic())
        d = self.delay * (random.Random(prompt).random() * 2 if self.jitter else 1)
        time.sleep(d)
        if self.fail:
            e = RuntimeError("503 UNAVAILABLE")
            e.code = 503
            raise e
        self.last_finish = "stop"
        if schema is LLMExtraction:
            reqs = [{"doc_id": f"Ф-{i}", "kind": "ф", "summary": f"операция {i}", "line_from": l.no, "line_to": l.no,
                     "quote": l.text.lstrip("* ")}
                    for i, l in ((int(re.search(r"Ф-(\d+)", l.text).group(1)), l)
                                 for l in self.doc.lines if l.text.startswith("* Ф-"))]
            return json.dumps({"requirements": reqs}, ensure_ascii=False), 10, 10
        if schema is LLMReqBatch:
            part = prompt.split("ПАКЕТ НА ПРОВЕРКУ")[1].split("Формальные подсказки")[0]
            ids = re.findall(r"^(R-\d+) ", part, re.M)
            out = [{"req_id": rid, "issues": [{"criterion": "C4", "severity": "значительный",
                                               "title": f"Нет обработки ошибок {rid}",
                                               "quote": f"обрабатывает операцию типа {int(rid[2:])} ",
                                               "explanation": "e", "recommendation": "r"}]} for rid in ids]
            # одна выдуманная цитата на пакет — для проверки порядка отброшенных
            out[0]["issues"].append({"criterion": "C3", "severity": "незначительный", "title": f"Выдумка {ids[0]}",
                                     "quote": "этого текста в документе нет вообще", "explanation": "e",
                                     "recommendation": "r"})
            return json.dumps({"assessments": out}, ensure_ascii=False), 10, 10
        if schema is LLMSetResult:
            return json.dumps({"issues": [{"criterion": "S6", "severity": "незначительный", "title": "Нет приоритетов",
                                           "explanation": "e", "recommendation": "r",
                                           "evidence": [{"req_id": "R-01", "quote": "обрабатывает операцию типа 1 "}]}]},
                              ensure_ascii=False), 10, 10
        return json.dumps({"development_comment": "d", "testing_comment": "t", "priorities": []}), 10, 10


@pytest.fixture
def doc(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text(DOC, encoding="utf-8")
    return load_document(str(p))


def _snapshot(res):
    return ([(f.fid, f.scope, f.criterion, f.title, f.req_ids, [(l.line_from, l.line_to) for _, l in f.evidence])
             for f in res.findings],
            [(j.req_id, j.title) for j in res.rejected], res.unchecked, res.verdict.status)


def test_parallel_report_equals_sequential(doc, tmp_path):
    seq = SlowFake(doc, concurrency=1)
    seq.journal_path = str(tmp_path / "seq.jsonl")
    par = SlowFake(doc, concurrency=4)
    par.journal_path = str(tmp_path / "par.jsonl")
    r1 = Pipeline(doc, seq, batch_size=3).run()
    r2 = Pipeline(doc, par, batch_size=3).run()
    assert _snapshot(r1) == _snapshot(r2)
    assert r1.notes == r2.notes
    assert seq.usage.calls == par.usage.calls == 1 + 7 + 1 + 1
    assert par.usage.journal_records == seq.usage.journal_records == par.usage.calls
    assert len((tmp_path / "par.jsonl").read_text(encoding="utf-8").splitlines()) == par.usage.calls
    assert [j.req_id for j in r2.rejected] == [f"R-{i:02d}" for i in (1, 4, 7, 10, 13, 16, 19)]


def test_parallel_is_faster(doc):
    t = time.monotonic()
    Pipeline(doc, SlowFake(doc, concurrency=1, delay=0.15, jitter=False), batch_size=3).run()
    seq = time.monotonic() - t
    t = time.monotonic()
    Pipeline(doc, SlowFake(doc, concurrency=4, delay=0.15, jitter=False), batch_size=3).run()
    par = time.monotonic() - t
    # последовательно 10 × 0.15 = 1.5 c; параллельно ≈ нарезка + 2 волны + итог = 0.6 c
    assert par < seq * 0.6, (seq, par)


def test_min_interval_is_shared_between_threads(doc, monkeypatch):
    monkeypatch.setenv("LLM_MIN_INTERVAL", "0.08")
    p = SlowFake(doc, concurrency=4, delay=0.01, jitter=False)
    Pipeline(doc, p, batch_size=3).run()
    gaps = [b - a for a, b in zip(p.starts, p.starts[1:])]
    assert min(gaps) >= 0.075, gaps


def test_outage_with_parallel_requests(doc, monkeypatch):
    monkeypatch.setattr("reqval.llm.time.sleep", lambda s: None)
    p = SlowFake(doc, concurrency=4, delay=0, fail=True)
    res = Pipeline(doc, p, batch_size=3).run()
    assert p.tripped and res.verdict.status == INCOMPLETE
    assert len(p.starts) <= p.max_retries * (1 + 4), "после срабатывания предохранителя новые запросы не уходят"


def test_openrouter_ladder_has_no_race(monkeypatch):
    from reqval.llm import OpenRouterProvider
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("OPENROUTER_MODE", raising=False)
    p = OpenRouterProvider("google/gemini-x")
    assert p.mode == 0
    barrier = threading.Barrier(4)

    def hit():
        barrier.wait()
        p._advance_mode(0)          # четыре одновременных 404 на ступени 0

    ts = [threading.Thread(target=hit) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert p.mode == 1, "одна ступень, а не четыре"


def test_default_concurrency(monkeypatch):
    from reqval.llm import GeminiProvider, OpenRouterProvider
    monkeypatch.delenv("LLM_CONCURRENCY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    assert OpenRouterProvider("google/x").concurrency == 4
    assert GeminiProvider("gemini-x").concurrency == 1, "бесплатный Gemini — по одному запросу (лимит RPM)"
    monkeypatch.setenv("LLM_CONCURRENCY", "3")
    assert GeminiProvider("gemini-x").concurrency == 3


def test_cli_flag(doc, tmp_path, monkeypatch):
    p = SlowFake(doc, delay=0)
    monkeypatch.setattr(cli, "make_provider", lambda kind, model: p)
    assert cli.main([doc.path, str(tmp_path / "r.md"), "--concurrency", "5"]) == 0
    assert p.concurrency == 5
    assert "параллельно до 5 запросов" in (tmp_path / "r.md").read_text(encoding="utf-8")
