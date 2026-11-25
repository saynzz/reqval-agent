"""Задача 10: сводка по формулировке ТЗ §6 — непроверяемые, противоречивые, неполные формулировки.

Проверяется на дефектах реального прогона (tests/data/run_findings.json): группы считаются без повторов,
каждый дефект попадает в свою группу, отчёт и JSON согласованы.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pytest

from reqval.criteria import TZ_GROUPS
from reqval.dedup import dedupe
from reqval.loader import load_document
from reqval.pipeline import RunResult, Verdict
from reqval.report import build_json, build_markdown, tz_summary

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def res():
    import test_dedup
    findings, _, _ = dedupe(test_dedup._load_run())
    doc = load_document(str(ROOT / "docs" / "requirements.md"))
    now = datetime.now().astimezone()
    return RunResult(doc, [], findings, Verdict("НЕ ГОТОВ", "d", "t", []), now, now, None, "LLM")


def test_groups_follow_tz_and_skip_duplicates(res):
    g = tz_summary(res.unique_findings)
    contra = [f.fid for f in g["contradictory"]]
    assert {"D-064", "D-065", "D-066", "D-067", "D-068"} <= set(contra)
    assert not {"D-042", "D-049", "D-056", "D-059"} & set(contra), "повторы дефектов набора не считаются"
    from reqval.criteria import SEVERITY_ORDER
    keys = [(SEVERITY_ORDER[f.severity], f.fid) for f in g["contradictory"]]
    assert keys == sorted(keys), "сначала критичные, внутри уровня — по номеру"
    assert [f.fid for f in g["contradictory"][:5]] == ["D-064", "D-065", "D-066", "D-067", "D-068"]
    for key, (_, codes) in TZ_GROUPS.items():
        assert all(f.criterion in codes for f in g[key])
    every = [f for f in res.unique_findings if f.criterion in ("C7", "S5", "C8", "S2", "C4", "S1")]
    assert sum(len(v) for v in g.values()) == len(every)


def test_summary_section_in_report(res):
    md = build_markdown(res)
    summary = md.split("## 6. Сводка")[1].split("## 7. Итог")[0]
    for title, _ in TZ_GROUPS.values():
        assert f"**{title}**" in summary and f"| {title} |" in summary
    g = tz_summary(res.unique_findings)
    row = re.search(r"\| Противоречивые формулировки \| C8, S2 \| (\d+) \|", summary)
    assert row and int(row.group(1)) == len(g["contradictory"])
    assert "D-065" in summary and "D-042" not in summary
    # незначительные не раздувают сводку: только числом
    minor_ids = [f.fid for v in g.values() for f in v if f.severity == "незначительный"]
    assert all(f"**{fid}**" not in summary for fid in minor_ids)


def test_summary_in_json(res):
    j = build_json(res)["summary"]
    assert j["findings"] == len(res.unique_findings)
    assert set(j["tz_groups"]) == set(TZ_GROUPS)
    assert "D-068" in j["tz_groups"]["contradictory"]["findings"]
    json.dumps(j, ensure_ascii=False)
