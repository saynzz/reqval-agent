"""Задача 11: «Что исправить в первую очередь» (ТЗ §6, «Итог») не бывает пустым без причины.

Раньше запасной список строился только из критичных дефектов: в офлайн-режиме, при сбое модели на шаге «итог»
или когда модель вернула пустые приоритеты, раздел оставался пустым.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from reqval.llm import LLMError
from reqval.loader import load_document
from reqval.models import Finding, LLMPriority, LLMVerdict, Location
from reqval.pipeline import INCOMPLETE, Pipeline, RunResult, Verdict
from reqval.report import build_markdown

ROOT = Path(__file__).resolve().parents[1]


def F(fid, sev, crit, title="t", rec="Исправить"):
    return Finding(fid, "requirement", crit, sev, title, "e", rec, ["R-01"], [("q", Location(1, 1, None, ""))])


def test_order_severity_then_tz_group():
    fs = [F("D-001", "значительный", "C4", "неполное"), F("D-002", "незначительный", "C8", "мелочь"),
          F("D-003", "значительный", "C8", "противоречие"), F("D-004", "критичный", "C3", "критичное"),
          F("D-005", "значительный", "C7", "непроверяемое"), F("D-006", "значительный", "C1", "прочее")]
    pr = Pipeline.fallback_priorities(fs)
    assert [ids[0] for ids, _ in pr] == ["D-004", "D-003", "D-005", "D-001", "D-006"]
    assert all("Исправить" in a for _, a in pr), "в пункте — что сделать, а не только название дефекта"
    assert len(Pipeline.fallback_priorities([F(f"D-{i:03d}", "значительный", "C4") for i in range(20)])) == 8


def test_offline_real_document_has_priorities():
    res = Pipeline(load_document(str(ROOT / "docs" / "requirements.md")), None).run()
    ids = {f.fid for f in res.unique_findings}
    assert res.verdict.priorities and all(set(i) <= ids for i, _ in res.verdict.priorities)
    section = build_markdown(res).split("### Что исправить в первую очередь")[1].split("##")[0]
    assert "1. " in section


class VerdictModel:
    """Провайдер, у которого всё, кроме итога, уже сделано: проверяем только шаг verdict."""

    def __init__(self, mode):
        self.mode = mode

    def generate(self, system, prompt, schema, label=""):
        if self.mode == "fail":
            raise LLMError("[итог] 503")
        if self.mode == "empty":
            return LLMVerdict(development_comment="d", testing_comment="t", priorities=[])
        return LLMVerdict(development_comment="d", testing_comment="t",
                          priorities=[LLMPriority(finding_ids=["D-999"], action="")])


@pytest.mark.parametrize("mode", ["fail", "empty", "garbage"])
def test_verdict_falls_back(mode, tmp_path):
    p = tmp_path / "d.md"
    p.write_text("# ТЗ\n\n## Р\nтекст требования достаточно длинный\n", encoding="utf-8")
    pipe = Pipeline(load_document(str(p)), None)
    pipe.llm = VerdictModel(mode)
    fs = [F("D-001", "критичный", "C8", "Противоречие Ф-2 и критерия")]
    v = pipe.verdict([], fs, [])
    assert v.priorities == [(["D-001"], "Противоречие Ф-2 и критерия. Исправить")]


def _res(findings, complete=True):
    doc = load_document(str(ROOT / "docs" / "requirements.md"))
    now = datetime.now().astimezone()
    r = RunResult(doc, [], findings, Verdict(INCOMPLETE if not complete else "ГОТОВ", "d", "t", []), now, now,
                  None, "LLM", failed_steps=[] if complete else ["итог набора"])
    return build_markdown(r).split("### Что исправить в первую очередь")[1].split("##")[0]


def test_empty_priorities_are_explained():
    assert "Исправлять нечего" in _res([])
    assert "незначительные замечания" in _res([F("D-001", "незначительный", "C9")])
    assert "Сначала повторите прогон" in _res([], complete=False)
