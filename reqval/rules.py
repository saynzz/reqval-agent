"""Детерминированные (формальные) проверки без LLM.

Они воспроизводимы, дают подсказки модели и работают в офлайн-режиме.
Основа: ISO/IEC/IEEE 29148:2018 п. 5.2.7 (неоднозначные и неограниченные
термины, заглушки) и п. 5.2.8 (атрибуты: идентификатор, версия).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Document, Location, Requirement

# Размытые, неограниченные термины (29148 5.2.7). Ищутся как начало слова.
VAGUE_TERMS = [
    "быстр", "оперативн", "своевременн", "удобн", "интуитивн", "понятн",
    "эффективн", "оптимальн", "надежн", "надёжн", "гибк", "современн", "дружелюбн",
    "достаточн", "значительн", "максимально", "минимально", "как можно", "по возможности",
    "при необходимости", "в случае необходимости", "как правило", "обычно", "как минимум",
    "и т.д", "и т. д", "и т.п", "и т. п", "и др", "и/или", "примерно", "около ", "порядка ",
    "некотор", "различн", "соответствующ", "подходящ", "нормальн", "должным образом",
]

PLACEHOLDER_RE = re.compile(
    r"(заполня[ею]тся|не заполнен|tbd|todo|tbc|уточнить|уточняется|будет определ|\?\?\?|xxx|заглушк)",
    re.IGNORECASE,
)
UNCHECKED_RE = re.compile(r"Да\s*/\s*Нет")
DASH_VALUE_RE = re.compile(r"(?:^|\|\s*|:\s*)[—–-]\s*(?:\||\.|$)")
TIME_RE = re.compile(r"(\d+\s*(мс|сек|секунд|минут|мин|час)|в течение (минуты|секунды|часа|дня|суток))", re.IGNORECASE)
PERCENTILE_RE = re.compile(r"(\d+\s*%|перцентил|p9\d|в\s+\d+\s*%|доля|не менее чем в)", re.IGNORECASE)


@dataclass
class RuleHit:
    rule: str           # код правила
    criterion: str      # код критерия
    severity: str
    title: str
    explanation: str
    recommendation: str
    quote: str
    location: Location
    req_id: str | None = None


def _line_loc(doc: Document, no: int) -> Location:
    ln = doc.line(no)
    return Location(no, no, ln.page if ln else None, ln.section if ln else "")


def check_document(doc: Document) -> list[RuleHit]:
    """Проверки уровня документа: заглушки, пустые поля, неотмеченный чек-лист, пустое согласование."""
    hits: list[RuleHit] = []
    unchecked_lines: list[int] = []
    for ln in doc.lines:
        t = ln.text
        if not t.strip() or ln.is_heading:
            continue
        if PLACEHOLDER_RE.search(t):
            hits.append(RuleHit("R1", "S1", "значительный", "Заглушка / незаполненный раздел",
                                "Раздел содержит заглушку вместо содержания — набор требований неполон (29148 5.2.6, 5.2.7: недопустимы TBD).",
                                "Заполнить раздел или явно указать, что он не применяется, с обоснованием.",
                                t, _line_loc(doc, ln.no)))
        if UNCHECKED_RE.search(t):
            unchecked_lines.append(ln.no)
        cells = [c.strip() for c in t.split("|")] if "|" in t else []
        if cells and len(cells) >= 2 and all(not c for c in cells):
            hits.append(RuleHit("R2", "S7", "значительный", "Пустая строка таблицы",
                                "Строка таблицы не заполнена (например, блок согласования без ФИО и подписей).",
                                "Заполнить таблицу или удалить строку.", t or "(пустая строка таблицы)", _line_loc(doc, ln.no)))
        elif cells and len(cells) >= 2 and any(c in ("—", "–", "-") for c in cells[1:]):
            hits.append(RuleHit("R3", "S1", "незначительный", "Значение не указано («—»)",
                                "В поле стоит прочерк — значение не определено.",
                                "Указать значение или пометить поле как неприменимое с причиной.", t, _line_loc(doc, ln.no)))
        elif re.search(r":\s*[—–]\s*(\.|$)", t):
            hits.append(RuleHit("R3", "S1", "незначительный", "Значение не указано («—»)",
                                "В тексте значение заменено прочерком.",
                                "Указать значение.", t, _line_loc(doc, ln.no)))
    if unchecked_lines:
        lf, lt = unchecked_lines[0], unchecked_lines[-1]
        first = doc.line(lf)
        hits.append(RuleHit("R4", "S7", "значительный", "Чек-лист ревью не заполнен",
                            f"В {len(unchecked_lines)} строках чек-листа оставлены варианты «Да / Нет» без выбора — ревью требований формально не проведено.",
                            "Провести ревью и проставить отметки по каждому пункту.",
                            doc.line(lf).text, Location(lf, lt, first.page if first else None, first.section if first else "")))
    return hits


def check_requirement(req: Requirement) -> list[RuleHit]:
    hits: list[RuleHit] = []
    low = req.quote.lower()
    found = sorted({t.strip() for t in VAGUE_TERMS if re.search(r"(?<![а-яa-z])" + re.escape(t), low)})
    if found:
        hits.append(RuleHit("R5", "C3", "незначительный", "Размытые термины",
                            f"Найдены неоднозначные/неограниченные слова: {', '.join('«'+f+'»' for f in found)} (29148 5.2.7).",
                            "Заменить измеримыми значениями или определить термин.", req.quote, req.location, req.rid))
    if TIME_RE.search(req.quote) and not PERCENTILE_RE.search(req.quote):
        hits.append(RuleHit("R6", "C7", "значительный", "Временная характеристика без условий измерения",
                            "Указано время, но нет точки отсчёта/окончания измерения, доли событий (перцентиля) и условий нагрузки — объективно проверить нельзя.",
                            "Указать: от какого события до какого, для какой доли событий (например, 95%) и при какой нагрузке.",
                            TIME_RE.search(req.quote).group(0), req.location, req.rid))
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", req.quote.strip()) if len(s) > 15]
    if len(sentences) >= 3:
        hits.append(RuleHit("R7", "C5", "незначительный", "Несколько утверждений в одном требовании",
                            f"Требование содержит {len(sentences)} отдельных утверждения — вероятно, это несколько требований.",
                            "Разделить на атомарные требования с собственными идентификаторами.", req.quote, req.location, req.rid))
    return hits


def check_ids(reqs: list[Requirement]) -> RuleHit | None:
    no_id = [r for r in reqs if r.doc_id.strip() in ("", "—", "-")]
    if not no_id:
        return None
    first = no_id[0]
    return RuleHit("R8", "S8", "значительный", "Требования без идентификаторов",
                   f"{len(no_id)} из {len(reqs)} требований не имеют идентификатора в документе "
                   f"({', '.join(r.rid for r in no_id[:12])}{'…' if len(no_id) > 12 else ''}); на них нельзя сослаться из тестов и изменений (29148 5.2.8, 830 4.3.8).",
                   "Присвоить уникальные идентификаторы всем требованиям, включая НФТ, требования к данным, интеграции и API.",
                   first.quote, first.location, None)
