"""Дедупликация дефектов.

Модель находит одно и то же несколько раз: противоречие «Ф-2 ↔ критерий приёмки №2» приходит как C8 у Ф-2,
как C8 у критерия и как S2 на уровне набора. В отчёте это выглядело бы как три дефекта, сводка и число
критичных завышались бы. Правила:

1. Внутри одной области (два дефекта набора или два дефекта одного требования) — одинаковые замечания
   склеиваются: цитаты объединяются, остаётся более высокая критичность.
2. Замечание по требованию, которое повторяет дефект набора, не удаляется: требование действительно
   нарушает критерий, и это видно в матрице. Оно помечается как дубль (dup_of) и в сводке, итоге и
   нумерации считается один раз — как дефект набора.

«Одинаковые» = родственные критерии + пересекающиеся места в документе + похожий текст замечания.
"""
from __future__ import annotations

import re

from rapidfuzz import fuzz

from .anchor import normalize
from .criteria import SEVERITY_ORDER
from .models import Finding, Location

# Критерий требования ↔ свойство набора, которые описывают один и тот же вид дефекта.
FAMILY = {
    "C8": "contradiction", "S2": "contradiction",
    "C4": "incomplete", "S1": "incomplete",
    "C3": "ambiguous", "S4": "ambiguous",
    "C6": "feasible", "S3": "feasible",
    "C9": "trace", "S8": "trace", "S7": "trace",
    "C7": "verifiable", "S5": "verifiable",
}

SAME_SCOPE_TEXT = 70     # похожесть текста для склейки внутри области
CROSS_SCOPE_TEXT = 45    # для связи «требование ↔ набор» (тексты пишутся с разной точки зрения)
QUOTE_SAME = 90          # совпадение цитат
# Противоречие часто описывают с соседних строк (заголовок статуса, пример ответа, строка таблицы) —
# места не пересекаются, но дефект тот же. Пороги подобраны на реальном прогоне (tests/data/run_findings.json):
# у настоящих повторов похожесть 51–63 при расстоянии 1–13 строк, у разных противоречий того же раздела — не выше 46.
NEAR_LINES = 15
NEAR_TEXT = 50
MENTION_LINES = 30       # замечание называет ≥2 требований дефекта набора и стоит не дальше 30 строк


def family(code: str) -> str:
    return FAMILY.get(code, code)


def _overlap(a: Location, b: Location) -> bool:
    return not (a.line_to < b.line_from or b.line_to < a.line_from)


def _locs_overlap(a: Finding, b: Finding) -> bool:
    return any(_overlap(la, lb) for _, la in a.evidence for _, lb in b.evidence)


def _text(f: Finding) -> str:
    return normalize(f"{f.title}. {f.explanation}")


def _text_sim(a: Finding, b: Finding) -> float:
    return fuzz.token_set_ratio(_text(a), _text(b))


def _line_distance(a: Finding, b: Finding) -> int:
    return min(max(0, max(la.line_from, lb.line_from) - min(la.line_to, lb.line_to))
               for _, la in a.evidence for _, lb in b.evidence)


def _quote_sim(a: Finding, b: Finding) -> float:
    return max((fuzz.ratio(normalize(qa), normalize(qb)) for qa, _ in a.evidence for qb, _ in b.evidence), default=0)


def same_issue(a: Finding, b: Finding) -> bool:
    """Два замечания одной области описывают один дефект."""
    if family(a.criterion) != family(b.criterion) or not _locs_overlap(a, b):
        return False
    if a.scope == "requirement" and a.req_ids != b.req_ids:
        return False
    return _text_sim(a, b) >= SAME_SCOPE_TEXT or (a.criterion == b.criterion and _quote_sim(a, b) >= QUOTE_SAME)


def repeats_set_issue(req_f: Finding, set_f: Finding) -> bool:
    """Замечание по требованию повторяет дефект набора."""
    if family(req_f.criterion) != family(set_f.criterion) or not req_f.req_ids:
        return False
    sim = _text_sim(req_f, set_f)
    mentioned = set(re.findall(r"R-\d{2,}", f"{req_f.title} {req_f.explanation}"))
    if req_f.req_ids[0] in set_f.req_ids and _locs_overlap(req_f, set_f):
        if family(req_f.criterion) == "contradiction":
            # По промпту модель называет в C8 вторую сторону противоречия (R-xx) — та же пара, что в S2, → дубль.
            if (set(set_f.req_ids) - set(req_f.req_ids)) & mentioned:
                return True
        return sim >= CROSS_SCOPE_TEXT
    if family(req_f.criterion) == "contradiction":
        dist = _line_distance(req_f, set_f)
        if len(mentioned & set(set_f.req_ids)) >= 2 and dist <= MENTION_LINES:
            return True
        if dist <= NEAR_LINES and sim >= NEAR_TEXT:
            return True
    return False


def _merge_into(keep: Finding, other: Finding) -> None:
    if SEVERITY_ORDER.get(other.severity, 3) < SEVERITY_ORDER.get(keep.severity, 3):
        keep.severity = other.severity
    seen = {(q, l.line_from, l.line_to) for q, l in keep.evidence}
    for q, l in other.evidence:
        if (q, l.line_from, l.line_to) not in seen:
            keep.evidence.append((q, l))
    for rid in other.req_ids:
        if rid not in keep.req_ids:
            keep.req_ids.append(rid)
    if len(other.explanation) > len(keep.explanation):
        keep.explanation = other.explanation
    if len(other.recommendation) > len(keep.recommendation):
        keep.recommendation = other.recommendation


def dedupe(findings: list[Finding]) -> tuple[list[Finding], int, int]:
    """Возвращает (findings без склеенных дублей, сколько склеено, сколько помечено как дубль набора)."""
    out: list[Finding] = []
    merged = 0
    for f in findings:
        twin = next((k for k in out if k.scope == f.scope and k.dup_of is None and same_issue(k, f)), None)
        if twin is not None and f.scope in ("requirement", "set"):
            _merge_into(twin, f)
            merged += 1
        else:
            out.append(f)

    sets = [f for f in out if f.scope == "set"]
    linked = 0
    for f in out:
        if f.scope != "requirement" or f.source != "LLM":
            continue
        target = next((s for s in sets if repeats_set_issue(f, s)), None)
        if target is not None:
            f.dup_of = target
            for rid in f.req_ids:            # дефект набора перечисляет все затронутые требования
                if rid not in target.req_ids:
                    target.req_ids.append(rid)
            if SEVERITY_ORDER.get(f.severity, 3) < SEVERITY_ORDER.get(target.severity, 3):
                target.severity = f.severity       # критичность дефекта набора — не ниже, чем у его частей
            linked += 1
    return out, merged, linked
