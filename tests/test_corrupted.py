"""Повреждённые и «не те» входные файлы.

Любая проблема с файлом — понятное сообщение на русском и код выхода 2, без трассировки Python (README обещает
именно это жюри). Файлы, которые можно прочитать (не та кодировка, не то расширение), читаются с пометкой
в логе и в журнале прогона отчёта.
"""
from __future__ import annotations

import logging
import re
import zipfile
from pathlib import Path

import pytest

from reqval import cli
from reqval.loader import DocumentError, decode_text, load_document, sniff_format

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "tests" / "data" / "requirements_tables.pdf"
MD_TEXT = "# ТЗ\n\n## Требования\n* **Ф-1.** Порог от 1 000 до 1 000 000 тг включительно.\n"


def _docx(path: Path) -> Path:
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_heading("ТЗ", 1)
    d.add_paragraph("Ф-1. Порог от 1 000 до 1 000 000 тг включительно.")
    d.save(str(path))
    return path


def make(tmp: Path, kind: str) -> Path:
    if kind == "truncated.pdf":
        (tmp / kind).write_bytes(PDF.read_bytes()[:3000])
    elif kind == "fake.pdf":
        (tmp / kind).write_bytes(b"not a pdf at all")
    elif kind == "broken.docx":
        (tmp / kind).write_bytes(b"PK\x03\x04garbage")
    elif kind == "archive.docx":
        with zipfile.ZipFile(tmp / kind, "w") as z:
            z.writestr("a.txt", "x")
    elif kind == "old.doc":
        (tmp / kind).write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 200)
    elif kind == "binary.txt":
        (tmp / kind).write_bytes(b"\x00\x01\x02" * 100)
    elif kind == "latin.md":
        (tmp / kind).write_bytes("Résumé ».".encode("latin-1") + b"\x81\x8d\x8f\x90\x9d" * 20)
    elif kind == "empty.md":
        (tmp / kind).write_bytes(b"")
    elif kind == "spaces.txt":
        (tmp / kind).write_text("   \n\n  ")
    elif kind == "folder.md":
        (tmp / kind).mkdir()
    elif kind == "doc.rtf":
        (tmp / kind).write_bytes(b"{\\rtf1\\ansi hello}")
    elif kind == "notes.xlsx":
        (tmp / kind).write_text("x")
    return tmp / kind


BAD = [
    ("truncated.pdf", "PDF повреждён, обрезан"),
    ("fake.pdf", "не является PDF"),
    ("broken.docx", "zip-архив, а не документ DOCX"),
    ("archive.docx", "zip-архив, а не документ DOCX"),
    ("old.doc", "старого формата Word (.doc)"),
    ("binary.txt", "не похож на текстовый документ"),
    ("latin.md", "Не удалось определить кодировку"),
    ("empty.md", "Файл пустой (0 байт)"),
    ("spaces.txt", "Документ пустой"),
    ("folder.md", "Указана папка, а не файл"),
    ("doc.rtf", "файл RTF"),
    ("notes.xlsx", "Неподдерживаемый формат"),
]


@pytest.mark.parametrize("kind, message", BAD, ids=[k for k, _ in BAD])
def test_bad_file_gives_clear_error(tmp_path, kind, message):
    path = make(tmp_path, kind)
    with pytest.raises((DocumentError, FileNotFoundError), match=re.escape(message)):
        load_document(str(path))


@pytest.mark.parametrize("kind, message", BAD[:4] + BAD[5:6] + BAD[9:10], ids=[k for k, _ in BAD[:4] + BAD[5:6] + BAD[9:10]])
def test_cli_exit_code_2_without_traceback(tmp_path, kind, message, caplog, capsys):
    path = make(tmp_path, kind)
    out = tmp_path / "out" / "r.md"
    with caplog.at_level(logging.ERROR, logger="reqval"):
        assert cli.main([str(path), str(out), "--provider", "offline"]) == 2
    assert message in caplog.text and "Traceback" not in caplog.text + capsys.readouterr().err
    assert not out.exists()


def test_missing_file(tmp_path):
    assert cli.main([str(tmp_path / "nope.md"), str(tmp_path / "r.md"), "--provider", "offline"]) == 2


def test_cp1251_is_read_and_noted(tmp_path):
    p = tmp_path / "win.md"
    p.write_bytes(MD_TEXT.encode("cp1251"))
    d = load_document(str(p))
    assert d.name == "ТЗ" and any("Ф-1. Порог" in l.text for l in d.lines)
    assert d.notes == ["кодировка windows-1251 — прочитано с перекодировкой в UTF-8"]
    from reqval.pipeline import Pipeline
    from reqval.report import build_markdown
    assert "Чтение документа: кодировка windows-1251" in build_markdown(Pipeline(d, None).run())


def test_bom_is_stripped_silently(tmp_path):
    p = tmp_path / "bom.md"
    p.write_bytes(b"\xef\xbb\xbf" + MD_TEXT.encode("utf-8"))
    d = load_document(str(p))
    assert d.name == "ТЗ" and d.lines[0].is_heading, "заголовок «# ТЗ» распознан, BOM не мешает"
    assert d.notes == []


def test_utf16(tmp_path):
    p = tmp_path / "u16.txt"
    p.write_bytes(MD_TEXT.encode("utf-16"))
    assert any("Ф-1" in l.text for l in load_document(str(p)).lines)


def test_docx_with_pdf_extension(tmp_path):
    d = load_document(str(_docx(tmp_path / "req.pdf")))
    assert d.fmt == "docx" and any("Ф-1" in l.text for l in d.lines)
    assert "по содержимому это DOCX" in d.notes[0]


def test_pdf_with_docx_extension(tmp_path):
    p = tmp_path / "req.docx"
    p.write_bytes(PDF.read_bytes())
    d = load_document(str(p))
    assert d.fmt == "pdf" and "по содержимому это PDF" in d.notes[0]


def test_text_file_without_extension_or_wrong_text_ext(tmp_path):
    p = tmp_path / "README"
    p.write_text(MD_TEXT, encoding="utf-8")
    assert load_document(str(p)).fmt == "txt"


def test_unwritable_output(tmp_path, monkeypatch, caplog):
    src = tmp_path / "d.md"
    src.write_text(MD_TEXT, encoding="utf-8")

    def deny(*a, **k):
        raise PermissionError(13, "Permission denied", str(tmp_path / "ro" / "r.md"))
    monkeypatch.setattr(cli, "write_report", deny)
    with caplog.at_level(logging.ERROR, logger="reqval"):
        assert cli.main([str(src), str(tmp_path / "ro" / "r.md"), "--provider", "offline"]) == 2
    assert "Нет прав на запись" in caplog.text


@pytest.mark.parametrize("head, fmt", [(b"%PDF-1.7", "pdf"), (b"  %PDF-1.4", "pdf"), (b"PK\x03\x04", "zip"),
                                       (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "doc"), (b"{\\rtf1", "rtf"),
                                       (b"# title", None)])
def test_sniff(head, fmt):
    assert sniff_format(head) == fmt


def test_decode_rejects_non_cyrillic_garbage():
    with pytest.raises(DocumentError, match="кодировку"):
        decode_text(b"\x81\x8d\x8f\x90\x9d" * 10 + b"abc\xff\xfe\xfd")
    assert decode_text("Привет".encode("cp1251")) == ("Привет", "windows-1251")
