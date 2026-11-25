"""Задача 9: короткая цитата-доказательство не «подтверждается» первым попавшимся словом.

Цитата сначала ищется в строках своего требования. Короче MIN_EVIDENCE_CHARS символов — только там:
«push», «1 000», «GET» встречаются в документе много раз.
"""
from __future__ import annotations

import pytest

from reqval.anchor import MIN_EVIDENCE_CHARS, Anchorer
from reqval.llm import BaseProvider
from reqval.loader import load_document
from reqval.models import (LLMEvidence, LLMExtraction, LLMIssue, LLMReqAssessment, LLMReqBatch, LLMRequirement,
                           LLMSetIssue, LLMSetResult, LLMVerdict)
from reqval.pipeline import Pipeline

DOC = """# ТЗ

## Архитектура
Карточный процессинг → учёт расходов → сервис push.

## Функциональные требования
* **Ф-1.** Пользователь задаёт порог от 1 000 до 1 000 000 ₸ включительно.
* **Ф-3.** Клиент получает push при достижении порога, максимум один push за месяц.

## Нефункциональные требования
Сервис передаёт push на отправку в течение минуты после получения события.
"""


@pytest.fixture
def doc(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text(DOC, encoding="utf-8")
    return load_document(str(p))


def line(doc, text):
    return next(l.no for l in doc.lines if text in l.text)


def test_short_quote_resolves_inside_own_requirement(doc):
    a = Anchorer(doc)
    f3 = line(doc, "Ф-3")
    from reqval.models import Location
    within = Location(f3, f3, None, "")
    loc = a.locate("push", within=within, min_global=MIN_EVIDENCE_CHARS)
    assert loc and loc.line_from == f3, "«push» есть в строке 4 раньше, но замечание — к Ф-3"


def test_short_quote_outside_requirement_is_rejected(doc):
    a = Anchorer(doc)
    from reqval.models import Location
    f1 = line(doc, "Ф-1")
    assert a.locate("push", within=Location(f1, f1, None, ""), min_global=MIN_EVIDENCE_CHARS) is None
    assert a.locate("push", min_global=MIN_EVIDENCE_CHARS) is None, "без требования короткая цитата не принимается"
    assert a.locate("push") is not None, "у нарезки (min_global=0) прежнее поведение"


def test_long_quote_may_come_from_other_place(doc):
    a = Anchorer(doc)
    from reqval.models import Location
    f1 = line(doc, "Ф-1")
    loc = a.locate("в течение минуты после получения события", within=Location(f1, f1, None, ""),
                   min_global=MIN_EVIDENCE_CHARS)
    assert loc and loc.line_from == line(doc, "в течение минуты"), "противоречие с другим местом — допустимо"


class Model(BaseProvider):
    name, model = "fake", "evidence"

    def __init__(self, doc):
        super().__init__()
        self.doc = doc

    def generate(self, system, prompt, schema, label=""):
        d = self.doc
        if schema is LLMExtraction:
            return LLMExtraction(requirements=[
                LLMRequirement(doc_id=t, kind="ф", summary=t, quote=q, line_from=line(d, t), line_to=line(d, t))
                for t, q in (("Ф-1", "Ф-1. Пользователь задаёт порог от 1 000 до 1 000 000 ₸ включительно."),
                             ("Ф-3", "Ф-3. Клиент получает push при достижении порога, максимум один push за месяц."))])
        if schema is LLMReqBatch:
            iss = lambda q, t: LLMIssue(criterion="C3", severity="значительный", title=t, quote=q, explanation="e",
                                        recommendation="r")
            return LLMReqBatch(assessments=[
                LLMReqAssessment(req_id="R-01", issues=[iss("1 000", "Нет валюты"), iss("push", "Чужое слово")]),
                LLMReqAssessment(req_id="R-02", issues=[iss("push", "Неясно, какой push")])])
        if schema is LLMSetResult:
            return LLMSetResult(issues=[LLMSetIssue(
                criterion="S4", severity="незначительный", title="Термин push не определён", explanation="e",
                recommendation="r", evidence=[LLMEvidence(req_id="—", quote="push"),
                                              LLMEvidence(req_id="R-02", quote="push")])])
        return LLMVerdict(development_comment="d", testing_comment="t", priorities=[])


def test_pipeline_applies_rule(doc):
    res = Pipeline(doc, Model(doc)).run()
    titles = {f.title: f for f in res.findings}
    assert titles["Нет валюты"].evidence[0][1].line_from == line(doc, "Ф-1")
    assert titles["Неясно, какой push"].evidence[0][1].line_from == line(doc, "Ф-3")
    assert "Чужое слово" not in titles, "«push» нет в Ф-1 — замечание к Ф-1 с такой цитатой не подтверждено"
    assert any(j.title == "Чужое слово" for j in res.rejected)
    s = titles["Термин push не определён"]
    assert [loc.line_from for _, loc in s.evidence] == [line(doc, "Ф-3")], "цитата без требования исключена"
