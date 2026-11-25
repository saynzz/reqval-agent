"""Задачи 3–4: ID требований и коды критериев из ответа модели не теряются и не ломают отчёт.

Баги, которые здесь закрыты:
- модель ответила «R-1» вместо «R-01» → требование уходило в «не проверено», дефекты терялись;
- модель написала «С8» (кириллическая С) или «C10» → дефект был в отчёте, а в матрице у требования стояли одни ✔.
"""
from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from reqval.criteria import REQ_CRITERIA, normalize_code, normalize_rid
from reqval.llm import BaseProvider
from reqval.loader import load_document
from reqval.models import (LLMEvidence, LLMExtraction, LLMIssue, LLMPriority, LLMReqAssessment, LLMReqBatch,
                           LLMRequirement, LLMSetIssue, LLMSetResult, LLMVerdict)
from reqval.pipeline import Pipeline
from reqval.report import build_markdown


@pytest.mark.parametrize("raw, expected", [
    ("R-01", "R-01"), ("R-1", "R-01"), ("R01", "R-01"), ("r-001", "R-01"), ("R 7", "R-07"),
    ("[R-12]", "R-12"), ("R-03 (Ф-2)", "R-03"), ("R‑05", "R-05"),        # неразрывный дефис
    ("Р-02", "R-02"), ("р-2", "R-02"),                                      # кириллическая Р
    ("R-100", "R-100"), ("Ф-2", None), ("—", None), ("", None),
])
def test_normalize_rid(raw, expected):
    assert normalize_rid(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("C8", "C8"), ("c8", "C8"), ("С8", "C8"), ("с3", "C3"), ("C-08", "C8"), ("C 7", "C7"),
    ("C8 Корректность", "C8"), ("S2", "S2"), ("s-02", "S2"), ("Ѕ1", "S1"),
])
def test_normalize_code(raw, expected):
    assert normalize_code(raw) == expected


def _issue(criterion):
    return LLMIssue(criterion=criterion, severity="значительный", title="t", quote="q", explanation="e",
                    recommendation="r")


def test_schema_accepts_lookalikes_and_rejects_unknown_codes():
    assert _issue("С8").criterion == "C8"
    for bad in ("C10", "C0", "S2", "X1", "Корректность"):      # S2 — код набора, не требования
        with pytest.raises(ValidationError):
            _issue(bad)
    with pytest.raises(ValidationError):
        LLMSetIssue(criterion="C3", severity="значительный", title="t", explanation="e", recommendation="r",
                    evidence=[])


def test_gemini_schema_has_closed_enum():
    genai = pytest.importorskip("google.genai")
    from google.genai import _transformers as t
    schema = t.t_schema(genai.Client(api_key="test")._api_client, LLMReqBatch).model_dump(exclude_none=True)
    txt = str(schema)
    assert all(code in txt for code in REQ_CRITERIA) and "enum" in txt


# --------------------------------------------------------------------------- интеграция с конвейером

DOC = """# ТЗ

## Функциональные требования
* **Ф-1.** Пользователь задаёт порог от 1 000 до 1 000 000 ₸ включительно.
* **Ф-2.** Месяц определяется по дате совершения покупки в Алматы.
* **Ф-3.** На карту приходится максимум один push за календарный месяц.

## Нефункциональные требования
Сервис передаёт push на отправку в течение минуты после получения события.
"""


class MessyModel(BaseProvider):
    """Отвечает так, как реально отвечают LLM: ID без ведущего нуля, кириллица, номер из документа."""

    name, model = "fake", "messy"

    def __init__(self, doc):
        super().__init__()
        self.doc = doc

    def generate(self, system, prompt, schema, label=""):
        self.usage.calls += 1
        if schema is LLMExtraction:
            reqs = [LLMRequirement(doc_id=ln.text.split(".")[0].lstrip("* ") if "Ф-" in ln.text else "—",
                                   kind="ф", summary=ln.text[:30], quote=ln.text.lstrip("* "),
                                   line_from=ln.no, line_to=ln.no)
                    for ln in self.doc.lines if ln.text.startswith("* Ф-") or "в течение" in ln.text]
            return LLMExtraction(requirements=reqs)
        if schema is LLMReqBatch:
            return LLMReqBatch(assessments=[
                LLMReqAssessment(req_id="R-1", issues=[_issue_q("C4", "от 1 000 до 1 000 000")]),
                LLMReqAssessment(req_id="Р-02", issues=[_issue_q("С8", "по дате совершения покупки")]),  # кириллица
                LLMReqAssessment(req_id="Ф-3", issues=[_issue_q("c5", "максимум один push")]),        # номер документа
                LLMReqAssessment(req_id="R04", issues=[_issue_q("C-07", "в течение минуты")]),
                LLMReqAssessment(req_id="R-77", issues=[_issue_q("C1", "в течение минуты")]),          # такого нет
            ])
        if schema is LLMSetResult:
            return LLMSetResult(issues=[LLMSetIssue(
                criterion="Ѕ2", severity="критичный", title="Ф-2 против Ф-3", explanation="e", recommendation="r",
                evidence=[LLMEvidence(req_id="Р-2", quote="по дате совершения покупки"),
                          LLMEvidence(req_id="R3", quote="максимум один push за календарный месяц")])])
        if schema is LLMVerdict:
            return LLMVerdict(development_comment="d", testing_comment="t",
                              priorities=[LLMPriority(finding_ids=["D-001"], action="a")])
        raise AssertionError(schema)


def _issue_q(code, quote):
    return LLMIssue(criterion=code, severity="значительный", title=f"дефект {code}", quote=quote,
                    explanation="e", recommendation="r")


@pytest.fixture
def result(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text(DOC, encoding="utf-8")
    doc = load_document(str(p))
    return Pipeline(doc, MessyModel(doc), batch_size=10).run()


def test_messy_ids_are_resolved(result):
    assert [r.rid for r in result.requirements] == ["R-01", "R-02", "R-03", "R-04"]
    assert result.unchecked == [], "все требования получили оценку, несмотря на «R-1», «Р-02», «Ф-3»"
    per_req = {f.req_ids[0]: f.criterion for f in result.findings if f.scope == "requirement"}
    assert per_req == {"R-01": "C4", "R-02": "C8", "R-03": "C5", "R-04": "C7"}
    assert any("R-77" in n for n in result.notes), "оценка несуществующего требования не молча проглочена"


def test_set_issue_links_requirements(result):
    s = [f for f in result.findings if f.scope == "set"]
    assert len(s) == 1 and s[0].criterion == "S2" and s[0].req_ids == ["R-02", "R-03"]


def test_matrix_matches_findings(result):
    """Инвариант отчёта: число ✖ в строке матрицы = число дефектов требования, все коды из справочника."""
    md = build_markdown(result)
    for r in result.requirements:
        own = [f for f in result.findings if f.scope == "requirement" and r.rid in f.req_ids]
        assert all(f.criterion in REQ_CRITERIA for f in own)
        row = next(line for line in md.splitlines() if line.startswith(f"| {r.rid} ("))
        cells = [c.strip() for c in row.strip("|").split("|")][1:-1]
        marks = sum(int(m.group(1) or 1) for c in cells for m in [re.match(r"✖(\d*)", c)] if m)
        assert marks == len(own) == int(row.strip("|").split("|")[-1])
