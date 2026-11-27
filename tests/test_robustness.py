"""Задачи 5–6.

5. Дефект, цитаты которого нет в документе (галлюцинация модели), не попадает в отчёт; он виден в приложении,
   а «Методика» описывает то, что агент делает на самом деле.
6. Ошибка модели на нарезке (или пустая нарезка) не роняет прогон: требования режутся правилами, отчёт пишется.
   Если часть проверки не выполнена, агент не имеет права сказать «ГОТОВ».
"""
from __future__ import annotations

from pathlib import Path

import pytest

from reqval import cli
from reqval.llm import BaseProvider, LLMError
from reqval.loader import load_document
from reqval.models import (LLMEvidence, LLMExtraction, LLMIssue, LLMReqAssessment, LLMReqBatch, LLMRequirement,
                           LLMSetIssue, LLMSetResult, LLMVerdict)
from reqval.pipeline import INCOMPLETE, NOT_READY, READY, Pipeline
from reqval.report import build_markdown

ROOT = Path(__file__).resolve().parents[1]

DOC = """# ТЗ

Вводный текст: документ не вычитывался.

## Функциональные требования
* **Ф-1.** Пользователь задаёт порог от 1 000 до 1 000 000 ₸ включительно.
* **Ф-2.** Месяц определяется по дате совершения покупки в Алматы.

## Нефункциональные требования
Сервис быстро передаёт push на отправку в течение минуты после получения события.
"""


@pytest.fixture
def doc(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text(DOC, encoding="utf-8")
    return load_document(str(p))


def _extraction(doc) -> LLMExtraction:
    return LLMExtraction(requirements=[
        LLMRequirement(doc_id=ln.text.split(".")[0].lstrip("* ") if "Ф-" in ln.text else "—", kind="ф",
                       summary=ln.text[:30], quote=ln.text.lstrip("* "), line_from=ln.no, line_to=ln.no)
        for ln in doc.lines if ln.text.startswith("* Ф-") or "в течение" in ln.text])


def _issue(code, quote, title, severity="значительный"):
    return LLMIssue(criterion=code, severity=severity, title=title, quote=quote, explanation="e", recommendation="r")


class Scripted(BaseProvider):
    """Фиктивная модель: на каждый шаг — заданный ответ или исключение."""

    name, model = "fake", "scripted"

    def __init__(self, doc, extraction=None, batch=None, set_result=None, fail=()):
        super().__init__()
        self.doc, self.fail = doc, set(fail)
        self.extraction = extraction if extraction is not None else _extraction(doc)
        self.batch = batch if batch is not None else LLMReqBatch(assessments=[
            LLMReqAssessment(req_id=f"R-0{i}", issues=[]) for i in range(1, 4)])
        self.set_result = set_result if set_result is not None else LLMSetResult(issues=[])

    def generate(self, system, prompt, schema, label=""):
        self.usage.calls += 1
        step = {LLMExtraction: "extract", LLMReqBatch: "check", LLMSetResult: "set", LLMVerdict: "verdict"}[schema]
        if step in self.fail or "all" in self.fail:
            raise LLMError(f"[{label}] 503 UNAVAILABLE")
        return {"extract": lambda: self.extraction, "check": lambda: self.batch, "set": lambda: self.set_result,
                "verdict": lambda: LLMVerdict(development_comment="d", testing_comment="t", priorities=[])}[step]()


# --------------------------------------------------------------------------- задача 5

def test_hallucinated_requirement_issue_is_dropped(doc):
    batch = LLMReqBatch(assessments=[
        LLMReqAssessment(req_id="R-01", issues=[
            _issue("C4", "от 1 000 до 1 000 000", "Реальный дефект"),
            _issue("C7", "система должна отвечать за 200 мс", "Выдуманный дефект")]),
        LLMReqAssessment(req_id="R-02", issues=[]), LLMReqAssessment(req_id="R-03", issues=[])])
    res = Pipeline(doc, Scripted(doc, batch=batch)).run()

    titles = [f.title for f in res.findings]
    assert "Реальный дефект" in titles and "Выдуманный дефект" not in titles
    assert [(j.req_id, j.title) for j in res.rejected] == [("R-01", "Выдуманный дефект")]
    for f in res.findings:   # всё, что попало в отчёт, найдено в тексте
        for quote, loc in f.evidence:
            assert loc.line_from >= 1

    md = build_markdown(res)
    assert "Выдуманный дефект" in md.split("### Отброшенные замечания модели")[1], "виден в приложении"
    assert "отброшено замечаний: 1" in md
    assert res.complete and res.verdict.status != INCOMPLETE, "отброшенная галлюцинация — не пробел в проверке"


def test_hallucinated_set_issue_is_dropped(doc):
    sr = LLMSetResult(issues=[
        LLMSetIssue(criterion="S2", severity="критичный", title="Выдуманное противоречие", explanation="e",
                    recommendation="r", evidence=[LLMEvidence(req_id="R-01", quote="лимит 5 000 ₸ в сутки"),
                                                  LLMEvidence(req_id="R-02", quote="по московскому времени")])])
    res = Pipeline(doc, Scripted(doc, set_result=sr)).run()
    assert not [f for f in res.findings if f.scope == "set"]
    assert res.rejected and res.rejected[0].scope == "set"
    assert res.verdict.status != NOT_READY, "выдуманный критичный дефект не должен влиять на итог"


def test_methodology_matches_mode(doc):
    md_llm = build_markdown(Pipeline(doc, Scripted(doc)).run())
    assert "модель проверяет каждое требование" in md_llm and "без подтверждённой цитаты в отчёт не попадает" in md_llm
    md_off = build_markdown(Pipeline(doc, None).run())
    method = md_off.split("## 2. Методика")[1].split("##")[0]
    assert "модель не использовалась" in method and "проверяет каждое требование" not in method


# --------------------------------------------------------------------------- задача 6

def test_extraction_error_falls_back_to_rules(doc):
    res = Pipeline(doc, Scripted(doc, fail={"extract"})).run()
    assert res.requirements, "требования выделены правилами"
    assert res.extraction.startswith("правила") and "нарезка" in res.failed_steps
    assert not res.complete and res.verdict.status == INCOMPLETE
    md = build_markdown(res)
    assert "Проверка выполнена не полностью" in md and "нарезка требований: правила (ошибка LLM)" in md


def test_empty_extraction_falls_back_to_rules(doc):
    bogus = LLMExtraction(requirements=[LLMRequirement(doc_id="—", kind="x", summary="s", line_from=1, line_to=1,
                                                       quote="Этого текста в документе нет совсем")])
    res = Pipeline(doc, Scripted(doc, extraction=bogus)).run()
    assert res.requirements and res.extraction.startswith("правила")


def test_total_outage_never_says_ready(doc):
    res = Pipeline(doc, Scripted(doc, fail={"all"})).run()
    assert res.verdict.status == INCOMPLETE and res.verdict.status != READY
    assert set(res.unchecked) == {r.rid for r in res.requirements}
    vague = [f for f in res.findings if f.source == "правило" and f.scope == "requirement"]
    assert vague, "для непроверенных моделью требований остаются формальные находки («быстро», «в течение минуты»)"
    assert "часть проверки не выполнена" in res.verdict.development


def test_critical_defect_wins_over_incomplete(doc):
    batch = LLMReqBatch(assessments=[LLMReqAssessment(req_id="R-02", issues=[
        _issue("C8", "по дате совершения покупки", "Противоречие", severity="критичный")])])
    res = Pipeline(doc, Scripted(doc, batch=batch, fail={"set"})).run()
    assert not res.complete and res.verdict.status == NOT_READY


def test_cli_writes_report_and_exits_3_when_incomplete(doc, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "make_provider", lambda kind, model: Scripted(doc, fail={"all"}))
    out = tmp_path / "r.md"
    assert cli.main([doc.path, str(out)]) == 3
    assert out.exists() and INCOMPLETE in out.read_text(encoding="utf-8")


def test_offline_skips_preamble(doc):
    res = Pipeline(doc, None).run()
    assert all("Вводный текст" not in r.quote for r in res.requirements)


def test_offline_real_document_has_no_preamble():
    real = ROOT / "docs" / "requirements.md"
    if not real.exists():
        pytest.skip("нет docs/requirements.md")
    res = Pipeline(load_document(str(real)), None).run()
    assert all("Комплект из двух" not in r.quote for r in res.requirements)


def test_offline_never_claims_readiness():
    """Без модели проверены только формальные признаки — «готов» / «готов с оговорками» был бы необоснованным."""
    from reqval.pipeline import FORMAL_ONLY, READY, READY_WITH_REMARKS
    real = ROOT / "docs" / "requirements.md"
    res = Pipeline(load_document(str(real)), None).run()
    assert res.verdict.status == FORMAL_ONLY and res.verdict.status not in (READY, READY_WITH_REMARKS)
    assert "без проверки моделью сделать нельзя" in res.verdict.development
    md = build_markdown(res)
    assert f"**Решение: {FORMAL_ONLY}**" in md
