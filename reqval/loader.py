"""Чтение документа требований: PDF, DOCX, Markdown, TXT.

Результат — список строк со сквозными номерами, страницей и путём раздела.
По этим номерам агент ссылается на конкретные места документа.
"""
from __future__ import annotations

import hashlib
import re
import statistics
from pathlib import Path

from .models import Document, Line

SECTION_SEP = " › "


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class _SectionTracker:
    """Ведёт стек заголовков и отдаёт текущий путь раздела."""

    def __init__(self) -> None:
        self.stack: list[tuple[int, str]] = []
        self.title: str | None = None

    def push(self, level: int, text: str) -> None:
        if self.title is None and level == 1:
            self.title = text
            return
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, text))

    @property
    def path(self) -> str:
        return SECTION_SEP.join(t for _, t in self.stack)


def _clean_inline(s: str) -> str:
    s = s.replace("**", "").replace("__", "")
    s = s.replace("`", "")
    s = s.replace(" ", " ")
    return re.sub(r"[ \t]+", " ", s).strip()


# ---------------------------------------------------------------------------
# Markdown / TXT
# ---------------------------------------------------------------------------

def _load_markdown(text: str) -> tuple[list[Line], str | None]:
    tr = _SectionTracker()
    lines: list[Line] = []
    in_code = False
    for raw in text.splitlines():
        s = raw.rstrip()
        heading = False
        if s.strip().startswith("```"):
            in_code = not in_code
            s = ""  # ограждения кода не несут смысла
        elif not in_code:
            m = re.match(r"^(#{1,6})\s+(.*)$", s)
            if m:
                level = len(m.group(1))
                s = _clean_inline(m.group(2))
                tr.push(level, s)
                heading = True
            elif re.match(r"^\s*\|?\s*:?-{3,}", s):
                s = ""  # разделитель таблицы
            elif s.strip().startswith("|"):
                cells = [_clean_inline(c) for c in s.strip().strip("|").split("|")]
                s = " | ".join(cells)
            elif s.strip() == "---":
                s = ""
            else:
                s = _clean_inline(s)
        lines.append(Line(no=len(lines) + 1, text=s, page=None, section=tr.path, is_heading=heading))
    return lines, tr.title


def _load_txt(text: str) -> tuple[list[Line], str | None]:
    lines = [Line(no=i + 1, text=_clean_inline(t), page=None, section="") for i, t in enumerate(text.splitlines())]
    title = next((l.text for l in lines if l.text), None)
    return lines, title


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

_LIST_START = re.compile(r"^(•|·|-|–|—|\*|\d{1,2}[.)]\s|\d{3}\.|[A-ZА-Я]{1,4}-\d|Ф-\d)")


def _join_wrapped(parts: list[str]) -> str:
    """Склейка строк, перенесённых вёрсткой: «бизнес-» + «процесса» → «бизнес-процесса», иначе через пробел."""
    out = ""
    for p in (x.strip() for x in parts):
        if not p:
            continue
        if out.endswith("-") and not out.endswith(" -") and len(out) > 1 and out[-2].isalpha():
            out += p
        else:
            out = f"{out} {p}" if out else p
    return out


COLUMN_GAP_EM = 1.0   # разрыв между символами строки шире 1 em — граница колонок (обычный пробел ≈ 0.25–0.6 em)


def _columns(chars: list[dict]) -> list[str]:
    """Делит строку на колонки по широким разрывам. Одна колонка — обычная строка текста."""
    chars = sorted(chars, key=lambda c: c["x0"])
    cols: list[list[dict]] = [[chars[0]]]
    for prev, c in zip(chars, chars[1:]):
        if c["x0"] - prev["x1"] > COLUMN_GAP_EM * max(prev["size"], c["size"]):
            cols.append([c])
        else:
            cols[-1].append(c)
    if len(cols) == 1:
        return [""]

    def text(cs: list[dict]) -> str:
        out = cs[0]["text"]
        for a, b in zip(cs, cs[1:]):
            out += (" " if b["x0"] - a["x1"] > 0.15 * b["size"] else "") + b["text"]
        return _clean_inline(out)

    return [text(cs) for cs in cols]


def _in_box(tl: dict, box: tuple[float, float, float, float]) -> bool:
    x0, top, x1, bottom = box
    cy = (tl["top"] + tl["bottom"]) / 2
    return top - 1 <= cy <= bottom + 1 and tl["x1"] > x0 and tl["x0"] < x1


def _load_pdf(path: Path) -> tuple[list[Line], str | None]:
    """PDF с текстовым слоем.

    - Таблицы (pdfplumber.find_tables) читаются как таблицы: строка таблицы → одна строка документа с ячейками
      через « | », как в Markdown. Без этого ячейки склеиваются в сплошной текст, длинная ячейка рвётся на
      несколько строк, а шапка таблицы принимается за заголовок раздела.
    - Строки абзаца, перенесённые вёрсткой, склеиваются обратно: требование «Ф-1» — одна строка, а не три.
    - Заголовки определяются по шрифту (крупнее или жирнее основного текста), таблицы в этом не участвуют.
    """
    import pdfplumber

    from collections import Counter, defaultdict

    def font(c: dict) -> str:
        return c.get("fontname", "").split("+")[-1]

    raw: list[dict] = []  # {text, page, style, uniform, nchars, top, bottom, table}
    with pdfplumber.open(str(path)) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            try:
                tables = page.find_tables()
            except Exception:  # noqa: BLE001 — битая разметка таблицы не должна ронять чтение
                tables = []
            boxes = [t.bbox for t in tables]
            items: list[dict] = []
            for t, box in zip(tables, boxes):
                seen_text = False
                for row in t.extract():
                    cells = [_clean_inline(_join_wrapped((c or "").split("\n"))) for c in row]
                    if not any(cells) and not seen_text:
                        continue   # пустые строки в начале таблицы — артефакт разрыва таблицы между страницами
                    seen_text = seen_text or any(cells)
                    items.append({"text": " | ".join(cells), "page": pno, "style": None, "uniform": False,
                                  "nchars": 0, "top": box[1], "bottom": box[1], "table": True})
                # строки одной таблицы — подряд, в порядке строк
            for tl in page.extract_text_lines(layout=False, strip=True):
                if any(_in_box(tl, box) for box in boxes):
                    continue
                chars = [c for c in tl.get("chars", []) if c.get("text", "").strip()]
                if not chars:
                    continue
                fc = Counter((font(c), round(c["size"], 1)) for c in chars)
                style, n = fc.most_common(1)[0]
                cols = _columns(chars)
                if len(cols) > 1:   # строка таблицы без рамок: колонки разделены широким разрывом
                    items.append({"text": " | ".join(cols), "page": pno, "style": style, "uniform": False,
                                  "nchars": len(chars), "top": tl["top"], "bottom": tl["bottom"], "table": True})
                    continue
                items.append({"text": _clean_inline(tl["text"]), "page": pno, "style": style,
                              "uniform": n / len(chars) >= 0.95, "nchars": len(chars),
                              "top": tl["top"], "bottom": tl["bottom"], "table": False})
            items.sort(key=lambda r: r["top"])   # sort устойчив: строки таблицы сохраняют порядок
            raw += items

    if not any(r["text"] for r in raw):
        raise ValueError("В PDF не найден текстовый слой (возможно, это скан). Нужен PDF с текстом.")

    # Основной стиль — тот, которым набрано больше всего символов. Стиль заголовка — отличается от
    # основного, применяется только к коротким однородным строкам без завершающей пунктуации.
    text_rows = [r for r in raw if not r["table"]]
    chars_by_style: Counter = Counter()
    lines_by_style: dict = defaultdict(list)
    for r in text_rows:
        chars_by_style[r["style"]] += r["nchars"]
        lines_by_style[r["style"]].append(r)
    heading_styles = []
    if chars_by_style:
        body_style = chars_by_style.most_common(1)[0][0]
        body_size = body_style[1]
        for st, rs in lines_by_style.items():
            if st == body_style or st[1] < body_size * 0.95:
                continue
            name = st[0].lower()
            emph = st[1] > body_size * 1.05 or "bold" in name or "italic" in name or "oblique" in name
            share = sum(1 for r in rs if r["uniform"] and len(r["text"]) < 90) / len(rs)
            if emph and share >= 0.8 and len(rs) <= max(3, len(text_rows) * 0.3):
                heading_styles.append(st)

    def rank(st: tuple) -> tuple:
        name = st[0].lower()
        return (-st[1], 0 if "bold" in name else 1)

    level_of = {st: i + 1 for i, st in enumerate(sorted(heading_styles, key=rank))}
    for r in raw:
        r["level"] = (level_of.get(r["style"]) if not r["table"] and r["uniform"] and len(r["text"]) < 90 else None)

    # Склейка строк абзаца, перенесённых вёрсткой: тот же шрифт, та же страница, небольшой зазор,
    # следующая строка не начинает новый пункт списка.
    merged: list[dict] = []
    for r in raw:
        prev = merged[-1] if merged else None
        if (prev and not r["table"] and not prev["table"] and not r["level"] and not prev["level"]
                and r["page"] == prev["page"] and r["style"] == prev["style"]
                and r["top"] - prev["bottom"] < 0.6 * (r["style"][1] if r["style"] else 10)
                and not _LIST_START.match(r["text"])):
            prev["text"] = _join_wrapped([prev["text"], r["text"]])
            prev["bottom"] = r["bottom"]
            continue
        merged.append(dict(r))

    tr = _SectionTracker()
    lines: list[Line] = []
    for r in merged:
        if r["level"]:
            tr.push(r["level"], r["text"])
        lines.append(Line(no=len(lines) + 1, text=r["text"], page=r["page"], section=tr.path,
                          is_heading=bool(r["level"])))
    return lines, tr.title


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------

def _load_docx(path: Path) -> tuple[list[Line], str | None]:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(str(path))
    tr = _SectionTracker()
    lines: list[Line] = []

    def add(text: str, heading: bool = False) -> None:
        lines.append(Line(no=len(lines) + 1, text=text, page=None, section=tr.path, is_heading=heading))

    for child in d.element.body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            p = Paragraph(child, d)
            text = _clean_inline(p.text)
            style = (p.style.name or "") if p.style is not None else ""
            m = re.match(r"(?:Heading|Заголовок)\s*(\d)", style)
            if style == "Title":
                tr.push(1, text)
                add(text, True)
            elif m and text:
                tr.push(int(m.group(1)), text)
                add(text, True)
            else:
                add(text)
        elif tag == "tbl":
            t = Table(child, d)
            for row in t.rows:
                cells = []
                for c in row.cells:
                    ct = _clean_inline(c.text)
                    if not cells or cells[-1] != ct:  # объединённые ячейки
                        cells.append(ct)
                add(" | ".join(cells))
    return lines, tr.title


# ---------------------------------------------------------------------------

def load_document(path: str) -> Document:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    ext = p.suffix.lower()
    if ext == ".pdf":
        lines, title, fmt = *_load_pdf(p), "pdf"
    elif ext == ".docx":
        lines, title, fmt = *_load_docx(p), "docx"
    elif ext in (".md", ".markdown"):
        lines, title, fmt = *_load_markdown(p.read_text(encoding="utf-8")), "md"
    elif ext in (".txt", ""):
        lines, title, fmt = *_load_txt(p.read_text(encoding="utf-8")), "txt"
    else:
        raise ValueError(f"Неподдерживаемый формат: {ext}. Поддерживаются .pdf, .docx, .md, .txt")

    if not any(l.text.strip() for l in lines):
        raise ValueError("Документ пустой: текст не извлечён")
    return Document(name=title or p.stem, path=str(p), fmt=fmt, sha256=_sha256(p), lines=lines)
