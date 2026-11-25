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

def _load_pdf(path: Path) -> tuple[list[Line], str | None]:
    import pdfplumber

    from collections import Counter, defaultdict

    def font(c: dict) -> str:
        return c.get("fontname", "").split("+")[-1]

    raw: list[dict] = []  # {text, page, style, uniform}
    with pdfplumber.open(str(path)) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            for tl in page.extract_text_lines(layout=False, strip=True):
                chars = [c for c in tl.get("chars", []) if c.get("text", "").strip()]
                if not chars:
                    continue
                fc = Counter((font(c), round(c["size"], 1)) for c in chars)
                style, n = fc.most_common(1)[0]
                raw.append({"text": _clean_inline(tl["text"]), "page": pno, "style": style,
                            "uniform": n / len(chars) >= 0.95, "nchars": len(chars)})

    if not raw:
        raise ValueError("В PDF не найден текстовый слой (возможно, это скан). Нужен PDF с текстом.")

    # Основной стиль — тот, которым набрано больше всего символов. Стиль заголовка — отличается от
    # основного, применяется только к коротким однородным строкам без завершающей пунктуации.
    chars_by_style: Counter = Counter()
    lines_by_style: dict = defaultdict(list)
    for r in raw:
        chars_by_style[r["style"]] += r["nchars"]
        lines_by_style[r["style"]].append(r)
    body_style = chars_by_style.most_common(1)[0][0]
    body_size = body_style[1]

    heading_styles = []
    for st, rs in lines_by_style.items():
        if st == body_style or st[1] < body_size * 0.95:
            continue
        name = st[0].lower()
        emph = st[1] > body_size * 1.05 or "bold" in name or "italic" in name or "oblique" in name
        share = sum(1 for r in rs if r["uniform"] and len(r["text"]) < 90) / len(rs)
        if emph and share >= 0.8 and len(rs) <= max(3, len(raw) * 0.3):
            heading_styles.append(st)

    def rank(st: tuple) -> tuple:
        name = st[0].lower()
        return (-st[1], 0 if "bold" in name else 1)

    level_of = {st: i + 1 for i, st in enumerate(sorted(heading_styles, key=rank))}

    tr = _SectionTracker()
    lines: list[Line] = []
    for r in raw:
        level = level_of.get(r["style"]) if r["uniform"] and len(r["text"]) < 90 else None
        if level:
            tr.push(level, r["text"])
        lines.append(Line(no=len(lines) + 1, text=r["text"], page=r["page"], section=tr.path, is_heading=bool(level)))
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
