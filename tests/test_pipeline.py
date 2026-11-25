"""Офлайн-тесты конвейера с фиктивной моделью (без сети и ключей).

Запуск: python -m pytest -q
Документ берётся из REQVAL_TEST_DOC; если переменная не задана — встроенный SAMPLE.
Прогон на документе организаторов: REQVAL_TEST_DOC=docs/requirements.md python -m pytest -q
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from reqval.anchor import Anchorer
from reqval.llm import BaseProvider
from reqval.loader import load_document
from reqval.models import (LLMEvidence, LLMExtraction, LLMIssue, LLMPriority, LLMReqAssessment, LLMReqBatch,
                           LLMRequirement, LLMSetIssue, LLMSetResult, LLMVerdict)
from reqval.pipeline import Pipeline
from reqval.report import build_markdown

DOC = os.getenv("REQVAL_TEST_DOC", str(Path(__file__).parent / "data" / "sample.md"))

SAMPLE = """# Тестовое ТЗ

## Функциональные требования
* **Ф-1.** Пользователь задаёт порог от 1 000 до 1 000 000 ₸ включительно.
* **Ф-2.** Система быстро отправляет push при достижении порога.
* **Ф-3.** Push отправляется не чаще одного раза в месяц.

## Нефункциональные требования
Сервис передаёт push в течение минуты.

## Требования к тестированию
заполняются тестировщиками
"""


@pytest.fixture(scope="module")
def doc(tmp_path_factory):
    if os.path.exists(DOC):
        return load_document(DOC)
    p = tmp_path_factory.mktemp("d") / "sample.md"
    p.write_text(SAMPLE, encoding="utf-8")
    return load_document(str(p))


class FakeProvider(BaseProvider):
    """Имитирует ответы модели: часть цитат верные, часть — выдуманные (должны отсеяться)."""

    name = "fake"
    model = "fake-model"

    def __init__(self, doc):
        super().__init__()
        self.doc = doc

    def generate(self, system, prompt, schema, label=""):
        self.usage.calls += 1
        if schema is LLMExtraction:
            reqs = []
            for ln in self.doc.lines:
                if ln.text.startswith("* Ф-") or ln.text.startswith("Ф-") or "в течение" in ln.text:
                    reqs.append(LLMRequirement(doc_id=ln.text.split(".")[0].lstrip("* ") if "Ф-" in ln.text else "—",
                                               kind="функциональное", summary=ln.text[:40], quote=ln.text.lstrip("* "),
                                               line_from=ln.no, line_to=ln.no))
            reqs.append(LLMRequirement(doc_id="—", kind="x", summary="галлюцинация",
                                       quote="Этого текста нет в документе совсем никак", line_from=1, line_to=1))
            return LLMExtraction(requirements=reqs)
        if schema is LLMReqBatch:
            out = []
            for line in prompt.split("ПАКЕТ НА ПРОВЕРКУ")[1].splitlines():
                if line.startswith("R-"):
                    rid = line.split()[0]
                    quote = line.split("«", 1)[1].rsplit("»", 1)[0]
                    words = " ".join(quote.split()[:5])
                    out.append(LLMReqAssessment(req_id=rid, issues=[LLMIssue(
                        criterion="C7", severity="значительный", title="Тестовый дефект", quote=words,
                        explanation="пояснение", recommendation="исправить")]))
            return LLMReqBatch(assessments=out)
        if schema is LLMSetResult:
            fs = [ln for ln in self.doc.lines if "Ф-" in ln.text][:2]
            return LLMSetResult(issues=[
                LLMSetIssue(criterion="S2", severity="критичный", title="Противоречие", explanation="e",
                            recommendation="r", evidence=[LLMEvidence(req_id="R-01", quote=f.text.lstrip("* ")[:50]) for f in fs]),
                LLMSetIssue(criterion="S1", severity="значительный", title="Выдуманный", explanation="e",
                            recommendation="r", evidence=[LLMEvidence(req_id="—", quote="несуществующая цитата про марсоход")]),
            ])
        if schema is LLMVerdict:
            return LLMVerdict(development_comment="нельзя", testing_comment="нельзя",
                              priorities=[LLMPriority(finding_ids=["D-001", "D-999"], action="Исправить")])
        raise AssertionError(schema)


def test_anchor_exact_and_fuzzy(doc):
    a = Anchorer(doc)
    ln = next(l for l in doc.lines if "Ф-1" in l.text)
    assert a.locate(ln.text[5:60]).line_from == ln.no
    assert a.locate("совершенно выдуманная фраза которой нет в тексте") is None


def test_pipeline_fake(doc):
    res = Pipeline(doc, FakeProvider(doc), batch_size=2).run()
    assert res.requirements, "требования выделены"
    assert all(r.summary != "галлюцинация" for r in res.requirements), "выдуманная цитата отброшена"
    titles = [f.title for f in res.findings]
    assert "Противоречие" in titles
    assert "Выдуманный" not in titles, "дефект без подтверждённой цитаты отброшен"
    for f in res.findings:
        assert f.evidence and all(loc.line_from >= 1 for _, loc in f.evidence)
    assert res.verdict.status == "НЕ ГОТОВ"
    assert all("D-999" not in ids for ids, _ in res.verdict.priorities)
    md = build_markdown(res)
    for section in ("Паспорт прогона", "нарезал требования", "Проверка каждого требования",
                    "Проверка всего набора", "Сводка", "Итог"):
        assert section in md


def test_offline(doc):
    res = Pipeline(doc, None).run()
    assert res.requirements and res.mode.startswith("offline")
    assert any(f.source == "правило" for f in res.findings)
