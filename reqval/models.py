"""Модели данных агента: документ, требования, дефекты."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic import BaseModel, Field

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
    criterion: str = Field(description="Код критерия, например C3")
    severity: Literal["критичный", "значительный", "незначительный"]
    title: str = Field(description="Короткое название дефекта")
    quote: str = Field(description="ДОСЛОВНЫЙ фрагмент документа, к которому относится дефект")
    explanation: str = Field(description="В чём дефект и почему это нарушение критерия")
    recommendation: str = Field(description="Как переформулировать/исправить")


class LLMReqAssessment(BaseModel):
    req_id: str
    issues: list[LLMIssue]


class LLMReqBatch(BaseModel):
    assessments: list[LLMReqAssessment]


class LLMEvidence(BaseModel):
    req_id: str = Field(description="ID требования из списка (R-xx) или «—» для места вне требований")
    quote: str = Field(description="ДОСЛОВНЫЙ фрагмент документа")


class LLMSetIssue(BaseModel):
    criterion: str
    severity: Literal["критичный", "значительный", "незначительный"]
    title: str
    explanation: str
    recommendation: str
    evidence: list[LLMEvidence]


class LLMSetResult(BaseModel):
    issues: list[LLMSetIssue]


class LLMPriority(BaseModel):
    finding_ids: list[str]
    action: str = Field(description="Что исправить, конкретно")


class LLMVerdict(BaseModel):
    development_comment: str = Field(description="Можно ли начинать разработку и почему, 2-4 предложения со ссылками на D-xxx")
    testing_comment: str = Field(description="Можно ли начинать тестирование/писать тест-кейсы и почему, 2-4 предложения со ссылками на D-xxx")
    priorities: list[LLMPriority] = Field(description="Что исправить в первую очередь, по убыванию важности, 3-8 пунктов")
