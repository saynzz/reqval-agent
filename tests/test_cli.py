"""Задача 2: команда запуска из README работает «как есть» (ТЗ §7: запуск одной командой)."""
from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

import pytest

from reqval.cli import main

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

TZ_SECTIONS = ("Паспорт прогона", "Как агент нарезал требования", "Проверка каждого требования",
               "Проверка всего набора требований", "Сводка", "Итог")


def _readme_run_command() -> list[str]:
    text = README.read_text(encoding="utf-8")
    m = re.search(r"^python -m reqval (\S+) (\S+)\s*$", text, re.M)
    assert m, "в README нет команды вида `python -m reqval <требования> <отчёт>`"
    return shlex.split(m.group(0))[3:]


def test_readme_input_file_exists():
    inp, out = _readme_run_command()
    assert (ROOT / inp).is_file(), f"README ссылается на {inp}, но такого файла в репозитории нет"
    assert out.endswith((".md", ".html"))


def test_documented_command_runs_offline(tmp_path, monkeypatch):
    inp, _ = _readme_run_command()
    monkeypatch.chdir(ROOT)
    out = tmp_path / "report.md"
    assert main([inp, str(out), "--provider", "offline"]) == 0
    md = out.read_text(encoding="utf-8")
    for s in TZ_SECTIONS:
        assert s in md, f"в отчёте нет раздела «{s}»"
    data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert data["requirements"] and data["run"]["agent_version"]


def test_missing_input_gives_clear_error(tmp_path):
    assert main([str(tmp_path / "nope.md"), str(tmp_path / "r.md"), "--provider", "offline"]) == 2
    assert not (tmp_path / "r.md").exists()


@pytest.mark.parametrize("ext", [".html"])
def test_html_output(tmp_path, monkeypatch, ext):
    inp, _ = _readme_run_command()
    monkeypatch.chdir(ROOT)
    out = tmp_path / f"report{ext}"
    assert main([inp, str(out), "--provider", "offline", "--no-json"]) == 0
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert not out.with_suffix(".json").exists()
