"""Привязка цитат к тексту документа.

Модель может ошибиться в цитате или номере строки. Каждая цитата, которую
агент кладёт в отчёт, проверяется здесь по исходному тексту. Если найти её
нельзя, дефект без подтверждённой цитаты в отчёт не попадёт.
"""
from __future__ import annotations

import re
from typing import Optional

from rapidfuzz import fuzz

from .models import Document, Location

_CHAR_MAP = {
    "ё": "е", "«": '"', "»": '"', "“": '"', "”": '"', "„": '"', "'": '"', "’": '"',
    "–": "-", "—": "-", "−": "-", " ": " ", "­": "", "*": "", "`": "", "|": " ",
    "₸": " тг ", "→": " -> ",
}

FUZZY_THRESHOLD = 88
# Цитата-доказательство короче этого (после нормализации) ищется только внутри строк своего требования:
# «push», «1 000», «GET» встречаются в документе много раз, и первое попавшееся совпадение ничего не доказывает.
MIN_EVIDENCE_CHARS = 16


def _norm_char(c: str) -> str:
    c = c.lower()
    return _CHAR_MAP.get(c, c)


def normalize(s: str) -> str:
    s = "".join(_norm_char(c) for c in s)
    return re.sub(r"\s+", " ", s).strip()


class Anchorer:
    def __init__(self, doc: Document):
        self.doc = doc
        chars: list[str] = []
        owner: list[int] = []
        for ln in doc.lines:
            if not ln.text.strip():
                continue
            for c in ln.text:
                for nc in _norm_char(c):
                    if nc.isspace():
                        if chars and chars[-1] == " ":
                            continue
                        nc = " "
                    chars.append(nc)
                    owner.append(ln.no)
            if chars and chars[-1] != " ":
                chars.append(" ")
                owner.append(ln.no)
        self.stream = "".join(chars)
        self.owner = owner

    # ------------------------------------------------------------------
    def _span_to_loc(self, start: int, end: int) -> Location:
        end = max(start, min(end, len(self.owner)) - 1)
        lf, lt = self.owner[start], self.owner[end]
        first = self.doc.line(lf)
        return Location(line_from=lf, line_to=lt, page=first.page if first else None,
                        section=first.section if first else "")

    def _span_of_lines(self, lf: int, lt: int) -> Optional[tuple[int, int]]:
        """Диапазон символов потока, принадлежащий строкам lf..lt."""
        idx = [i for i, o in enumerate(self.owner) if lf <= o <= lt]
        return (idx[0], idx[-1] + 1) if idx else None

    def _find_one(self, q: str, hint: Optional[int], span: Optional[tuple[int, int]] = None) -> Optional[tuple[int, int]]:
        if len(q) < 3:
            return None
        lo, hi = span if span else (0, len(self.stream))
        hits = [m.start() for m in re.finditer(re.escape(q), self.stream[lo:hi])]
        if hits:
            hits = [lo + h for h in hits]
            if hint is not None and len(hits) > 1:
                hits.sort(key=lambda h: abs(self.owner[h] - hint))
            return hits[0], hits[0] + len(q)
        if len(q) < 12:
            return None
        al = fuzz.partial_ratio_alignment(q, self.stream[lo:hi], score_cutoff=FUZZY_THRESHOLD)
        if al is None:
            return None
        return lo + al.dest_start, lo + al.dest_end

    def locate(self, quote: str, hint_line: Optional[int] = None, within: Optional[Location] = None,
               min_global: int = 0) -> Optional[Location]:
        """Ищет цитату в документе. Возвращает место или None.

        within — сначала искать внутри этих строк (строки требования, к которому относится замечание);
        min_global — вне within принимать только части цитаты не короче стольких символов."""
        if not quote:
            return None
        quote = re.sub(r"L\d{2,4}\|\s?", " ", quote)
        parts = [normalize(p) for p in re.split(r"\.\.\.|…|\[\.\.\.\]", quote)]
        parts = [p.strip(' ."') for p in parts if len(p.strip(' ."')) >= 3]
        if not parts:
            return None
        inner = self._span_of_lines(within.line_from, within.line_to) if within else None
        spans = []
        for p in parts:
            sp = self._find_one(p, hint_line, inner) if inner else None
            if sp is None and len(p) >= min_global:
                sp = self._find_one(p, hint_line)
            if sp is None:
                return None
            spans.append(sp)
        start = min(s for s, _ in spans)
        end = max(e for _, e in spans)
        return self._span_to_loc(start, end)

    def text_of(self, loc: Location) -> str:
        return " ".join(self.doc.lines[i - 1].text for i in range(loc.line_from, loc.line_to + 1)
                        if self.doc.lines[i - 1].text)
