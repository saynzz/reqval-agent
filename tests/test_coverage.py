"""Проверка покрытия нарезки.

Слабая модель (Gemini 3.1 Flash-Lite) при нарезке пропустила целые разделы: тело запроса («Content-Type:
application/json», «Тело не используется»), таблицу полей ответа (где поля — string), примеры JSON. То, что не
попало в нарезку, дальше не проверяется — поэтому она не нашла «Content-Type при пустом теле» и «integer vs string».
Агент добирает такие строки сам. Образцы — реальные нарезки двух моделей (tests/data/run_slicings.json).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from reqval import prompts
from reqval.llm import BaseProvider
from reqval.loader import load_document
from reqval.models import (LLMExtraction, LLMIssue, LLMReqBatch, LLMRequirement, LLMSetResult, LLMVerdict,
                           Location, Requirement)
from reqval.pipeline import COVERAGE_KIND, Pipeline

ROOT = Path(__file__).resolve().parents[1]
SLICINGS = json.loads((ROOT / "tests" / "data" / "run_slicings.json").read_text(encoding="utf-8"))["slicings"]


@pytest.fixture(scope="module")
def doc():
    return load_document(str(ROOT / "docs" / "requirements.md"))


def _reqs(name):
    return [Requirement(r["id"], r["doc_id"], r["summary"], r["quote"], r["kind"], Location(**r["location"]))
            for r in SLICINGS[name]]


def _added(doc, name):
    out = Pipeline(doc, None)._ensure_coverage(_reqs(name))
    return out, [r for r in out if r.kind == COVERAGE_KIND]


def test_weak_model_gaps_are_filled(doc):
    out, added = _added(doc, "gemini-3.1-flash-lite")
    quotes = [r.quote for r in added]
    for must in ("Content-Type: application/json .", "Тело не используется.",
                 "amount | string | При успехе | Порог в тенге | limit_kzt",
                 "enabled | string | «0» или «1» | Признак включения | enabled"):
        assert must in quotes, must
    assert len(out) == len(SLICINGS["gemini-3.1-flash-lite"]) + len(added)
    assert [r.rid for r in out] == [f"R-{i:02d}" for i in range(1, len(out) + 1)], "перенумерованы по порядку документа"
    lines = [r.location.line_from for r in out]
    assert lines == sorted(lines)


def test_strong_model_gets_almost_nothing(doc):
    _, added = _added(doc, "claude-sonnet-5.5")
    assert [r.quote for r in added] == ['{"result":"ERROR","message":"Проверьте сумму"}'], \
        "у Claude добирается только пропущенный пример ERROR внутри ответа 200"


def test_metadata_and_placeholders_are_not_requirements(doc):
    for name in SLICINGS:
        _, added = _added(doc, name)
        for r in added:
            assert not r.quote.startswith(("Author", "Developer", "Gitlab", "Framework", "Swagger", "Поле |"))
            assert "заполняются тестировщиками" not in r.quote
            assert not r.quote.startswith("Дочерний документ части A"), "вводный текст части документа"


class Partial(BaseProvider):
    """Модель, которая нарезала только Ф-1 и Ф-2, а проверяет всё, что ей прислали."""

    name, model = "fake", "partial"

    def __init__(self, doc):
        super().__init__()
        self.doc, self.checked = doc, []

    def generate(self, system, prompt, schema, label=""):
        if schema is LLMExtraction:
            return LLMExtraction(requirements=[
                LLMRequirement(doc_id=t, kind="ф", summary=t, line_from=l.no, line_to=l.no, quote=l.text.lstrip("* "))
                for l in self.doc.lines for t in ("Ф-1", "Ф-2") if l.text.lstrip("* ").startswith(t + ".")])
        if schema is LLMReqBatch:
            import re
            ids = re.findall(r"^(R-\d+) ", prompt.split("ПАКЕТ НА ПРОВЕРКУ")[1], re.M)
            self.checked += ids
            return LLMReqBatch(assessments=[{"req_id": i, "issues": []} for i in ids])
        if schema is LLMSetResult:
            return LLMSetResult(issues=[])
        return LLMVerdict(development_comment="d", testing_comment="t", priorities=[])


def test_added_requirements_are_checked_and_reported(doc):
    m = Partial(doc)
    res = Pipeline(doc, m).run()
    added = [r for r in res.requirements if r.kind == COVERAGE_KIND]
    assert len(added) > 30 and res.unchecked == []
    assert set(m.checked) == {r.rid for r in res.requirements}, "добавленные строки тоже проверены моделью"
    assert any("модель пропустила" in n for n in res.notes)
    from reqval.report import build_markdown
    table = build_markdown(res).split("## 3.")[1].split("## 4.")[0]
    assert COVERAGE_KIND in table


def test_coverage_can_be_disabled(doc):
    res = Pipeline(doc, Partial(doc), coverage=False).run()
    assert len(res.requirements) == 2


def test_prompts_demand_concrete_titles_and_cross_checks():
    for text in (prompts.CHECK_SYSTEM, prompts.SET_SYSTEM):
        assert "title — конкретное название" in text and "Противоречие бизнес-логике" in text
    assert "Content-Type" in prompts.CHECK_SYSTEM and "частичные возвраты" in prompts.CHECK_SYSTEM
    assert "ограничения частоты" in prompts.SET_SYSTEM
    assert "не общие слова" in LLMIssue.model_fields["title"].description
