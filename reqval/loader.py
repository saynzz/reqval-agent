"""Чтение документа требований: PDF, DOCX, Markdown, TXT.

Результат — список строк со сквозными номерами, страницей и путём раздела.
По этим номерам агент ссылается на конкретные места документа.
"""
from __future__ import annotations

import hashlib
import logging
import re
import statistics
import zipfile
from pathlib import Path

from .models import Document, Line

log = logging.getLogger("reqval")


class DocumentError(ValueError):
    """Файл нельзя прочитать как документ требований. Сообщение — для человека, на русском."""

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

# ---------------------------------------------------------------------------
# Определение формата и кодировки, защита от повреждённых файлов
# ---------------------------------------------------------------------------

SUPPORTED = {".pdf": "pdf", ".docx": "docx", ".md": "md", ".markdown": "md", ".txt": "txt", "": "txt"}
FORMAT_NAMES = {"pdf": "PDF", "docx": "DOCX", "md": "Markdown", "txt": "текст"}
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f]")
MAX_CONTROL_SHARE = 0.01   # больше 1 % управляющих символов — это не текст


def sniff_format(head: bytes) -> str | None:
    """Формат по содержимому: pdf / docx / doc / zip / None (похоже на текст)."""
    if head.lstrip()[:5] == b"%PDF-":
        return "pdf"
    if head[:4] == b"PK\x03\x04":
        return "zip"           # DOCX — это zip; что внутри, проверяется отдельно
    if head[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        return "doc"           # старый Word (OLE)
    if head[:5] == b"{\\rtf":
        return "rtf"
    return None


def _is_docx(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            return "word/document.xml" in z.namelist()
    except (zipfile.BadZipFile, OSError):
        return False


def decode_text(data: bytes) -> tuple[str, str]:
    """Байты → текст. BOM убирается; если не UTF-8 — пробуется Windows-1251 (частый случай для русских файлов
    из Windows). Возвращает (текст, кодировка)."""
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", errors="strict"), "utf-8 (с BOM)"
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16"), "utf-16"
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    text = data.decode("cp1251", errors="strict") if _looks_cp1251(data) else None
    if text is None:
        raise DocumentError("Не удалось определить кодировку файла: это не UTF-8 и не Windows-1251. "
                            "Сохраните документ в UTF-8.")
    return text, "windows-1251"


def _looks_cp1251(data: bytes) -> bool:
    try:
        text = data.decode("cp1251")
    except UnicodeDecodeError:
        return False
    # Любые байты 0x80–0xFF «декодируются» в cp1251, поэтому проверяем, что получился именно русский текст:
    # кириллица — хотя бы половина букв, и почти вся она из обычного русского алфавита (а не Ѓ, Ќ, Џ, ђ…).
    letters = [c for c in text if c.isalpha()]
    cyr = [c for c in letters if "\u0400" <= c <= "\u04ff"]
    russian = sum(1 for c in cyr if "а" <= c.lower() <= "я" or c in "ёЁ")
    return bool(cyr) and len(cyr) / len(letters) >= 0.5 and russian / len(cyr) >= 0.9


def _check_is_text(text: str) -> None:
    sample = text[:200_000]
    if not sample.strip():
        return
    bad = len(CONTROL_RE.findall(sample))
    if "\x00" in sample or bad / len(sample) > MAX_CONTROL_SHARE:
        raise DocumentError("Файл не похож на текстовый документ: внутри двоичные данные. "
                            "Проверьте, что передан нужный файл (PDF, DOCX, Markdown или TXT).")


def _reason(e: Exception) -> str:
    msg = str(e).strip().splitlines()[0][:160] if str(e).strip() else ""
    return f"{type(e).__name__}: {msg}" if msg else type(e).__name__


def load_document(path: str) -> Document:
    """Чтение документа. Любая проблема с файлом — DocumentError (ValueError) с понятным сообщением:
    CLI печатает его и выходит с кодом 2, без трассировки Python."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    if p.is_dir():
        raise DocumentError(f"Указана папка, а не файл: {path}")
    try:
        with open(p, "rb") as f:
            head = f.read(4096)
    except OSError as e:
        raise DocumentError(f"Файл не удаётся открыть: {_reason(e)}") from e
    if not head:
        raise DocumentError("Файл пустой (0 байт)")

    notes: list[str] = []
    ext = p.suffix.lower()
    by_ext = SUPPORTED.get(ext)
    sniffed = sniff_format(head)
    if sniffed == "zip":
        sniffed = "docx" if _is_docx(p) else "zip"
    if sniffed == "doc":
        raise DocumentError("Это файл старого формата Word (.doc). Сохраните его как .docx или PDF.")
    if sniffed == "rtf":
        raise DocumentError("Это файл RTF. Сохраните его как .docx, PDF или Markdown.")
    if sniffed == "zip":
        raise DocumentError("Это zip-архив, а не документ DOCX" + (" (файл DOCX повреждён?)" if ext == ".docx" else "")
                            + ". Нужен PDF, DOCX, Markdown или TXT.")
    if by_ext is None and sniffed is None:
        raise DocumentError(f"Неподдерживаемый формат: {ext}. Поддерживаются .pdf, .docx, .md, .txt")

    fmt = sniffed or by_ext
    if sniffed is None and by_ext in ("pdf", "docx"):
        raise DocumentError(f"Файл с расширением {ext} не является {FORMAT_NAMES[by_ext]}: "
                            "содержимое повреждено или это другой формат.")
    if by_ext and sniffed and by_ext != sniffed:
        msg = (f"расширение {ext or '(нет)'}, а по содержимому это {FORMAT_NAMES[sniffed]} — "
               f"файл прочитан как {FORMAT_NAMES[sniffed]}")
        log.warning("Документ: %s", msg)
        notes.append(msg)

    try:
        if fmt == "pdf":
            lines, title = _load_pdf(p)
        elif fmt == "docx":
            lines, title = _load_docx(p)
        else:
            text, enc = decode_text(p.read_bytes())
            if not enc.startswith("utf-8"):        # BOM убирается молча: он безвреден
                log.warning("Документ: кодировка %s", enc)
                notes.append(f"кодировка {enc} — прочитано с перекодировкой в UTF-8")
            _check_is_text(text)
            lines, title = _load_markdown(text) if fmt == "md" else _load_txt(text)
    except DocumentError:
        raise
    except ValueError as e:      # понятные сообщения загрузчиков (скан PDF и т.п.)
        if type(e) is ValueError:
            raise DocumentError(str(e)) from e
        raise DocumentError(f"Файл {FORMAT_NAMES[fmt]} не удаётся прочитать: {_reason(e)}") from e
    except Exception as e:  # noqa: BLE001 — повреждённый PDF/DOCX: pdfminer, zipfile, lxml бросают свои исключения
        hint = {"pdf": "PDF повреждён, обрезан или защищён паролем",
                "docx": "DOCX повреждён или сохранён не полностью"}.get(fmt, "файл повреждён")
        if "password" in type(e).__name__.lower() or "encrypt" in str(e).lower():
            hint = "PDF защищён паролем — снимите защиту или сохраните копию без пароля"
        raise DocumentError(f"Не удалось прочитать документ: {hint} ({_reason(e)}).") from e

    if not any(l.text.strip() for l in lines):
        raise DocumentError("Документ пустой: текст не извлечён")
    return Document(name=title or p.stem, path=str(p), fmt=fmt, sha256=_sha256(p), lines=lines, notes=notes)
