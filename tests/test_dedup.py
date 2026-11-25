"""Задачи 7–8.

7. Одно противоречие «Ф-2 ↔ критерий приёмки №2» модель находит трижды: C8 у Ф-2, C8 у критерия и S2 в наборе
   (раньше ещё и дважды в двух проходах по набору). В сводке и итоге оно должно считаться один раз, а проход
   по набору — делать один запрос к модели, а не два.
8. Требование, которое модель не оценила, в матрице показывает «—», а не ✔.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from reqval.llm import BaseProvider
from reqval.loader import load_document
from reqval.models import (LLMEvidence, LLMExtraction, LLMIssue, LLMReqAssessment, LLMReqBatch, LLMRequirement,
                           LLMSetIssue, LLMSetResult, LLMVerdict)
from reqval.pipeline import Pipeline
from reqval.report import build_markdown

ROOT = Path(__file__).resolve().parents[1]

F2 = "Месяц определяется по дате совершения покупки в Алматы."
AC2 = "Покупка 31 января в 23:58 на 4 000 ₸ | Провести 1 февраля в 00:03 по Алматы | Февральский счётчик вырос на 4 000 ₸"
NFR = "Сервис передаёт push на отправку в течение минуты после получения события."


@pytest.fixture(scope="module")
def doc():
    p = ROOT / "docs" / "requirements.md"
    if not p.exists():
        pytest.skip("нет docs/requirements.md")
    return load_document(str(p))


def line_of(doc, text):
    return next(l.no for l in doc.lines if text[:30] in l.text)


def _iss(code, quote, title, expl, severity="значительный"):
    return LLMIssue(criterion=code, severity=severity, title=title, quote=quote, explanation=expl, recommendation="r")


def _set(code, title, expl, quotes, severity="критичный"):
    return LLMSetIssue(criterion=code, severity=severity, title=title, explanation=expl, recommendation="r",
                       evidence=[LLMEvidence(req_id=rid, quote=q) for rid, q in quotes])


CONTRA = "Ф-2 считает месяц по дате совершения покупки, а критерий приёмки относит покупку 31 января к февралю"


class Model(BaseProvider):
    name, model = "fake", "dedup"

    def __init__(self, doc, skip_nfr=False):
        super().__init__()
        self.doc, self.skip_nfr, self.set_calls = doc, skip_nfr, 0

    def generate(self, system, prompt, schema, label=""):
        self.usage.calls += 1
        if schema is LLMExtraction:
            return LLMExtraction(requirements=[
                LLMRequirement(doc_id="Ф-2", kind="ф", summary="месяц", quote=F2, line_from=line_of(self.doc, F2), line_to=line_of(self.doc, F2)),
                LLMRequirement(doc_id="—", kind="ак", summary="АК-2", quote=AC2, line_from=line_of(self.doc, AC2), line_to=line_of(self.doc, AC2)),
                LLMRequirement(doc_id="—", kind="нфт", summary="нфт", quote=NFR, line_from=line_of(self.doc, NFR), line_to=line_of(self.doc, NFR))])
        if schema is LLMReqBatch:
            a = [
                # R-01 = Ф-2: противоречие с R-03 (критерий приёмки) + НЕЗАВИСИМЫЙ дефект полноты на той же строке
                LLMReqAssessment(req_id="R-01", issues=[
                    _iss("C8", "по дате совершения покупки", "Противоречит критерию приёмки",
                         "Правило противоречит критерию приёмки R-03: покупка 31 января попадает в февраль.", "критичный"),
                    _iss("C4", "по дате совершения покупки в Алматы", "Не определён часовой пояс сервера",
                         "Не сказано, как хранится время операции процессинга: UTC или локальное."),
                    _iss("C4", "по дате совершения покупки в Алматы", "Не определён часовой пояс сервера процессинга",
                         "Не сказано, как хранится время операции процессинга — UTC или локальное время.")]),
                LLMReqAssessment(req_id="R-03", issues=[
                    _iss("C8", "Февральский счётчик вырос на 4 000", "Противоречит Ф-2",
                         "Ожидаемый результат противоречит R-01: по Ф-2 покупка 31 января относится к январю.", "критичный")]),
            ]
            if not self.skip_nfr:
                a.append(LLMReqAssessment(req_id="R-02", issues=[
                    _iss("C7", "в течение минуты", "Нет условий измерения", "Не указан перцентиль и нагрузка.")]))
            return LLMReqBatch(assessments=a)
        if schema is LLMSetResult:
            self.set_calls += 1
            return LLMSetResult(issues=[
                _set("S2", "Ф-2 противоречит критерию приёмки №2", CONTRA,
                     [("R-01", "по дате совершения покупки"), ("R-03", "Февральский счётчик вырос на 4 000")]),
                _set("S2", "Противоречие Ф-2 и критерия приёмки №2 о границе месяца", CONTRA + ".",
                     [("R-01", "Месяц определяется по дате совершения"), ("R-03", "Провести 1 февраля в 00:03")]),
                _set("S1", "Не описан возврат покупки в следующем месяце",
                     "Неясно, может ли возврат в марте за февральскую покупку повторно вызвать push.",
                     [("R-01", "по дате совершения покупки в Алматы")], severity="значительный"),
            ])
        if schema is LLMVerdict:
            return LLMVerdict(development_comment="d", testing_comment="t", priorities=[])
        raise AssertionError(schema)


@pytest.fixture(scope="module")
def run(doc):
    m = Model(doc)
    return m, Pipeline(doc, m, batch_size=10, coverage=False).run()


def test_set_check_is_one_request(run):
    model, _ = run
    assert model.set_calls == 1


def test_contradiction_counted_once(run):
    _, res = run
    s2 = [f for f in res.unique_findings if f.criterion == "S2"]
    assert len(s2) == 1, "два одинаковых S2 склеены"
    lines = {loc.line_from for _, loc in s2[0].evidence}
    assert len(lines) >= 2 and s2[0].req_ids[:2] == ["R-01", "R-03"]
    c8 = [f for f in res.findings if f.criterion == "C8"]
    assert len(c8) == 2 and all(f.dup_of is s2[0] for f in c8), "C8 у обеих сторон — повторы S2"
    crit = [f for f in res.unique_findings if f.severity == "критичный"]
    assert crit == s2, "в списке критичных противоречие один раз"


def test_distinct_issues_are_not_merged(run):
    _, res = run
    c4 = [f for f in res.findings if f.criterion == "C4"]
    assert len(c4) == 1, "два почти одинаковых C4 у одного требования склеены"
    assert c4[0].dup_of is None, "C4 про часовой пояс — не то же, что S1 про возвраты, хотя строка та же"
    assert any(f.criterion == "S1" for f in res.unique_findings)


def test_numbering_and_report(run):
    _, res = run
    ids = [f.fid for f in res.unique_findings]
    assert ids == [f"D-{i:03d}" for i in range(1, len(ids) + 1)], "номера без дыр"
    md = build_markdown(res)
    s2 = next(f for f in res.unique_findings if f.criterion == "S2")
    assert md.count(f"тот же дефект описан в проверке набора: **{s2.fid}**") == 2
    assert re.search(r"\| Повторов дефектов набора .* \| 2 \|", md)
    assert f"| Всего дефектов | {len(res.unique_findings)} |" in md
    row = next(l for l in md.splitlines() if l.startswith("| R-01 ("))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[1 + 7] == "✖" and cells[1 + 3] == "✖", "в матрице у Ф-2 по-прежнему видны C8 и C4"


def test_unchecked_requirement_shows_dash(doc):
    res = Pipeline(doc, Model(doc, skip_nfr=True), batch_size=10, coverage=False).run()
    assert res.unchecked == ["R-02"]
    row = next(l for l in build_markdown(res).splitlines() if l.startswith("| R-02 ("))
    cells = [c.strip() for c in row.strip("|").split("|")][1:-1]
    assert "✔" not in cells, "непроверенное требование не может выглядеть как «нарушений нет»"
    assert cells.count("✖") == 1 and cells[6] == "✖", "формальная находка C7 («в течение минуты») видна"
    assert set(cells) == {"—", "✖"}


# --------------------------------------------------------------------------- 7б: регрессия на реальном прогоне

import json  # noqa: E402

from reqval.dedup import dedupe  # noqa: E402
from reqval.models import Finding, Location  # noqa: E402

RUN = ROOT / "tests" / "data" / "run_findings.json"


def _load_run():
    d = json.loads(RUN.read_text(encoding="utf-8"))
    return [Finding(x["id"], x["scope"], x["criterion"], x["severity"], x["title"], x["explanation"],
                    x["recommendation"], list(x["requirements"]),
                    [(e["quote"], Location(**e["location"])) for e in x["evidence"]], x["source"])
            for x in d["findings"]]


def test_real_run_api_contradictions_counted_once():
    """Прогон Claude Sonnet 5.5: «200+ERROR или 400» шло трижды, «403 или 404» — дважды, с соседних строк API."""
    out, merged, linked = dedupe(_load_run())
    links = {f.fid: f.dup_of.fid for f in out if f.dup_of is not None}
    assert links == {"D-042": "D-068", "D-049": "D-068", "D-056": "D-068", "D-059": "D-067"}
    crit = sorted(f.fid for f in out if f.dup_of is None and f.severity == "критичный")
    assert crit == ["D-025", "D-064", "D-065", "D-066", "D-067", "D-068"]
    d068 = next(f for f in out if f.fid == "D-068")
    assert {"R-35", "R-44", "R-50"} <= set(d068.req_ids), "дефект набора перечисляет все затронутые требования"


def test_real_run_different_contradictions_stay_apart():
    out, _, _ = dedupe(_load_run())
    by = {f.fid: f for f in out}
    for fid in ("D-064", "D-065", "D-066", "D-067", "D-068", "D-077"):    # разные противоречия набора
        assert by[fid].dup_of is None
    for fid in ("D-046", "D-047", "D-052", "D-013"):   # соседние, но другие дефекты (типы полей, тело 403 и т.п.)
        assert by[fid].dup_of is None, fid
