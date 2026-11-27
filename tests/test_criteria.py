"""Справочник критериев покрывает все пункты, указанные в ТЗ §5:
ISO/IEC/IEEE 29148:2018 п. 5.2.5 (9 свойств требования), п. 5.2.6 (5 свойств набора), IEEE Std 830-1998 п. 4.3
(8 свойств SRS, 4.3.1–4.3.8). Если кто-то удалит или переименует критерий, тест покажет, какой пункт выпал.
"""
from __future__ import annotations

import pytest

from reqval.criteria import REQ_CRITERIA, SET_CRITERIA

ISO_5_2_5 = ["Necessary", "Appropriate", "Unambiguous", "Complete", "Singular", "Feasible", "Verifiable",
             "Correct", "Conforming"]
ISO_5_2_6 = ["Complete", "Consistent", "Feasible", "Comprehensible", "Able to be validated"]
IEEE_4_3 = {"4.3.1": "Correct", "4.3.2": "Unambiguous", "4.3.3": "Complete", "4.3.4": "Consistent",
            "4.3.5": "Ranked", "4.3.6": "Verifiable", "4.3.7": "Modifiable", "4.3.8": "Traceable"}


def _sources(catalog):
    return " ; ".join(src for _, src, _ in catalog.values())


@pytest.mark.parametrize("prop", ISO_5_2_5)
def test_iso_5_2_5_individual(prop):
    assert f"29148 5.2.5 {prop}" in _sources(REQ_CRITERIA)


@pytest.mark.parametrize("prop", ISO_5_2_6)
def test_iso_5_2_6_set(prop):
    assert f"29148 5.2.6 {prop}" in _sources(SET_CRITERIA)


@pytest.mark.parametrize("clause, prop", IEEE_4_3.items())
def test_ieee_830_4_3(clause, prop):
    src = _sources(REQ_CRITERIA) + _sources(SET_CRITERIA)
    assert f"830 {clause}" in src or f"830 4.3 {prop}" in src or f"830 4.3.5 {prop}" in src, f"{clause} {prop}"


def test_individual_criteria_map_one_to_one_to_5_2_5():
    first = [src.split(";")[0].strip() for _, src, _ in REQ_CRITERIA.values()]
    assert first == [f"29148 5.2.5 {p}" for p in ISO_5_2_5], "C1–C9 — ровно девять свойств п. 5.2.5, по порядку"


def test_no_misattributed_clauses():
    src = _sources(REQ_CRITERIA) + _sources(SET_CRITERIA)
    assert "830 4.3 (" not in src, "в п. 4.3 стандарта 830 нет свойства «без проектных решений»"
