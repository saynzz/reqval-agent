"""Модели данных агента: документ, требования, дефекты."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .criteria import normalize_code

# ---------------------------------------------------------------------------
# Документ
# ---------------------------------------------------------------------------


@dataclass
class Line:
    no: int                 # сквозной номер строки (1..N)
    text: str               # текст строки (очищенный)
    page: Optional[int]     # номер страницы (для PDF/DOCX→PDF), иначе None
    section: str            # путь раздела: «Часть A › Функциональные требования»
    is_heading: bool = False


@dataclass
class Document:
    name: str               # название документа (заголовок)
    path: str
    fmt: str                # md / pdf / docx / txt
    sha256: str
    lines: list[Line] = field(default_factory=list)

    def numbered_text(self) -> str:
        """Текст с номерами строк — так модель видит документ и ссылается на места."""
        out = []
        for ln in self.lines:
            if not ln.text.strip():
                continue
            out.append(f"L{ln.no:03d}| {ln.text}")
        return "\n".join(out)

    def line(self, no: int) -> Optional[Line]:
        if 1 <= no <= len(self.lines):
            return self.lines[no - 1]
        return None


@dataclass
class Location:
    line_from: int
    line_to: int
    page: Optional[int]
    section: str

    def human(self) -> str:
        if self.line_from == self.line_to:
            rng = f"строка {self.line_from}"
        else:
            rng = f"строки {self.line_from}–{self.line_to}"
        parts = [self.section] if self.section else []
        if self.page:
            parts.append(f"с. {self.page}")
        parts.append(rng)
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# Требования и дефекты (внутреннее представление)
# ---------------------------------------------------------------------------

Severity = Literal["критичный", "значительный", "незначительный"]


def title_from(explanation: str, limit: int = 80) -> str:
    """Короткое название дефекта из первого предложения объяснения — если модель его не дала.
    Без строгой схемы (например, Claude через OpenRouter) модели часто опускают «второстепенное» поле title,
    и отбрасывать из-за этого весь содержательный ответ нельзя."""
    first = re.split(r"(?<=[.!?:;])\s|\s[—–]\s", explanation.strip(), maxsplit=1)[0].rstrip(".:;")
    return first if len(first) <= limit else first[:limit].rsplit(" ", 1)[0] + "…"

# Коды критериев — закрытый список (enum в JSON-схеме). Код вне справочника выпадал бы из матрицы
# соответствия, и отчёт противоречил бы сам себе.
ReqCriterion = Literal["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
SetCriterion = Literal["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]


@dataclass
class Requirement:
    rid: str                # идентификатор в отчёте: R-01
    doc_id: str             # идентификатор в документе (Ф-1) или «—»
    summary: str            # краткая формулировка
    quote: str              # дословная цитата
    kind: str               # тип: функциональное / нефункциональное / ...
    location: Location
    anchored: bool = True   # цитата найдена в тексте документа


@dataclass
class Finding:
    fid: str                # D-001
    scope: Literal["requirement", "set", "rule"]
    criterion: str          # код критерия (C3, S2 ...)
    severity: str
    title: str
    explanation: str
    recommendation: str
    req_ids: list[str]
    evidence: list[tuple[str, Location]]   # (цитата, место) — подтверждённые в документе
    source: str = "LLM"     # LLM / правило
    dup_of: Optional["Finding"] = None   # повторяет дефект набора: в сводке считается один раз (см. dedup.py)


# ---------------------------------------------------------------------------
# Схемы ответов модели (structured output)
# ---------------------------------------------------------------------------


class LLMRequirement(BaseModel):
    doc_id: str = Field(description="Идентификатор в документе (например «Ф-1», «Часть B, Логика работы метода, п. 2») или «—», если номера нет")
    kind: str = Field(description="Тип: бизнес-цель, UI, функциональное, нефункциональное, данные, интеграция, критерий приемки, API")
    summary: str = Field(description="Краткая формулировка требования своими словами, до 20 слов")
    quote: str = Field(description="ДОСЛОВНАЯ цитата требования из документа без изменений")
    line_from: int
    line_to: int


class LLMExtraction(BaseModel):
    requirements: list[LLMRequirement]


class LLMIssue(BaseModel):
    criterion: ReqCriterion = Field(description="Код критерия отдельного требования C1–C9")
    severity: Literal["критичный", "значительный", "незначительный"]
    title: str = Field(default="", description="Конкретное название дефекта: что и с чем не так, с номерами "
                       "(например «Ф-2 противоречит критерию приёмки №2 о границе месяца»), не общие слова")
    quote: str = Field(description="ДОСЛОВНЫЙ фрагмент документа, к которому относится дефект")
    explanation: str = Field(description="В чём дефект и почему это нарушение критерия")
    recommendation: str = Field(description="Как переформулировать/исправить")

    _norm_criterion = field_validator("criterion", mode="before")(normalize_code)

    @model_validator(mode="after")
    def _fill_title(self):
        if not self.title.strip():
            self.title = title_from(self.explanation) or "Дефект"
        return self


class LLMReqAssessment(BaseModel):
    req_id: str
    issues: list[LLMIssue]


class LLMReqBatch(BaseModel):
    assessments: list[LLMReqAssessment]


class LLMEvidence(BaseModel):
    req_id: str = Field(description="ID требования из списка (R-xx) или «—» для места вне требований")
    quote: str = Field(description="ДОСЛОВНЫЙ фрагмент документа")


class LLMSetIssue(BaseModel):
    criterion: SetCriterion = Field(description="Код критерия набора S1–S8")
    severity: Literal["критичный", "значительный", "незначительный"]
    title: str = Field(default="", description="Конкретное название дефекта: что и с чем не так, с номерами требований, не общие слова")
    explanation: str
    recommendation: str
    evidence: list[LLMEvidence]

    _norm_criterion = field_validator("criterion", mode="before")(normalize_code)

    @model_validator(mode="after")
    def _fill_title(self):
        if not self.title.strip():
            self.title = title_from(self.explanation) or "Дефект набора"
        return self


class LLMSetResult(BaseModel):
    issues: list[LLMSetIssue]


class LLMPriority(BaseModel):
    finding_ids: list[str]
    action: str = Field(description="Что исправить, конкретно")


class LLMVerdict(BaseModel):
    development_comment: str = Field(description="Можно ли начинать разработку и почему, 2-4 предложения со ссылками на D-xxx")
    testing_comment: str = Field(description="Можно ли начинать тестирование/писать тест-кейсы и почему, 2-4 предложения со ссылками на D-xxx")
    priorities: list[LLMPriority] = Field(description="Что исправить в первую очередь, по убыванию важности, 3-8 пунктов")
