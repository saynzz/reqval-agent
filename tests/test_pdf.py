"""Задача 13: PDF — основной формат входа по ТЗ (§3).

Образцы — документ организаторов, сконвертированный в PDF двумя способами:
- requirements_tables.pdf — таблицы с рамками (как из Word / Google Docs);
- requirements_borderless.pdf — таблицы без рамок (LibreOffice из DOCX без стиля таблицы).
Раньше таблицы склеивались в сплошной текст, длинные ячейки рвались, шапка таблицы считалась заголовком,
а абзацы разбивались по строкам вёрстки: 88 «требований» и 6 формальных дефектов вместо 49 и 13.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("pdfplumber")

from reqval.anchor import Anchorer  # noqa: E402
from reqval.loader import _columns, _join_wrapped, load_document  # noqa: E402
from reqval.pipeline import Pipeline  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
MD = Path(__file__).resolve().parents[1] / "docs" / "requirements.md"


@pytest.fixture(scope="module")
def pdf():
    return load_document(str(DATA / "requirements_tables.pdf"))


def texts(doc):
    return [l.text for l in doc.lines]


def test_tables_are_rows_with_cells(pdf):
    t = texts(pdf)
    assert "Поле | Значение" in t
    assert "Business owner / Product owner / Аналитик | Учебные роли розничного бизнеса, владельца мобильного " \
           "банка и аналитика команды" in t, "длинная ячейка — одна строка, не две"
    assert "amount | integer | 32 bit | Да | 1 000–1 000 000 | Порог в тенге | limit_kzt" in t
    assert any(x.startswith("Покупка 31 января в 23:58 на 4 000 ₸ | Провести 1 февраля") for x in t)
    heads = [l.text for l in pdf.lines if l.is_heading]
    assert "Поле Значение" not in heads and "Поле | Значение" not in heads, "шапка таблицы — не заголовок раздела"


def test_paragraphs_are_not_split_by_layout(pdf):
    f1 = [x for x in texts(pdf) if x.startswith("Ф-1.")]
    assert f1 == ["Ф-1. Держатель активной карты KZT задаёт целый порог от 1 000 до 1 000 000 ₸ включительно. "
                  "По умолчанию уведомление выключено."]
    assert sum(1 for x in texts(pdf) if x.startswith("Ф-")) == 4, "пункты списка не слиты между собой"
    assert any("схема полного бизнес-процесса" in x for x in texts(pdf)), "перенос с дефисом склеен без пробела"


def test_page_split_artifacts_removed_real_empty_row_kept(pdf):
    t = texts(pdf)
    i = t.index("Подразделение | ФИО ответственного | Подпись")
    assert t[i + 1].replace("|", "").strip() == "", "настоящая пустая строка блока согласующих осталась"
    assert t[i - 1] != t[i + 1], "пустые строки-артефакты разрыва страницы перед шапкой убраны"


def test_sections_match_markdown(pdf):
    md = load_document(str(MD))
    sec = lambda d, s: next(l.section for l in d.lines if l.text.lstrip("* ").startswith(s))
    for start in ("Ф-2.", "Сервис передаёт push", "REST type | GET"):
        assert sec(pdf, start) == sec(md, start)


def test_offline_result_matches_markdown(pdf):
    md = Pipeline(load_document(str(MD)), None).run()
    pd = Pipeline(pdf, None).run()
    assert len(pd.requirements) == len(md.requirements)
    assert Counter(f.title for f in pd.findings) == Counter(f.title for f in md.findings)


def test_quotes_anchor_with_page(pdf):
    loc = Anchorer(pdf).locate("Месяц определяется по дате совершения покупки в Алматы")
    assert loc and loc.page == 1 and "Функциональные требования" in loc.section
    assert "с. 1" in loc.human()


def test_borderless_tables_are_split_into_columns():
    doc = load_document(str(DATA / "requirements_borderless.pdf"))
    t = texts(doc)
    assert "Наименование задачи | Месячный порог расходов по карте" in t
    assert not any(x.startswith("Поле Значение Наименование") for x in t), "таблица не слита в одну строку"
    rules = Counter(f.title for f in Pipeline(doc, None).run().findings)
    assert rules["Чек-лист ревью не заполнен"] == 1 and rules["Значение не указано («—»)"] == 4


@pytest.mark.parametrize("parts, out", [
    (["бизнес-", "процесса"], "бизнес-процесса"),
    (["Карты → Расходы", "→ Уведомления"], "Карты → Расходы → Уведомления"),
    (["сумма —", "порог"], "сумма — порог"),
    (["Dev / Prod -", "x"], "Dev / Prod - x"),
    (["", "a", ""], "a"),
])
def test_join_wrapped(parts, out):
    assert _join_wrapped(parts) == out


def _chars(spec):
    """spec: [(x0, text)] — символ шириной 5 pt, кегль 10."""
    return [{"x0": x, "x1": x + 5, "text": ch, "size": 10} for x, ch in spec]


def test_columns_split_only_on_wide_gaps():
    word = lambda x, w: [(x + i * 5, ch) for i, ch in enumerate(w)]
    normal = _chars(word(0, "ab") + word(13, "cd"))        # разрыв 3 pt — пробел
    assert _columns(normal) == [""]
    table = _chars(word(0, "ab") + word(40, "cd"))         # разрыв 30 pt — граница колонок
    assert _columns(table) == ["ab", "cd"]


def test_scanned_pdf_message(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    w = pypdf.PdfWriter()
    w.add_blank_page(width=200, height=200)
    p = tmp_path / "scan.pdf"
    with open(p, "wb") as f:
        w.write(f)
    with pytest.raises(ValueError, match="текстовый слой"):
        load_document(str(p))
