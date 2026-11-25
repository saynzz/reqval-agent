"""Задача 1: версия агента в паспорте прогона берётся из git и честно помечает проблемы (ТЗ §6)."""
from __future__ import annotations

import shutil
import subprocess

import pytest

from reqval.version import git_info

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git не установлен")


def _git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t", "HOME": str(cwd), "PATH": "/usr/bin:/bin:/usr/local/bin"})


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_VERSION", raising=False)
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", "a.py")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_clean_commit_and_tag(repo):
    _git(repo, "tag", "v1.0")
    vi = git_info(repo)
    assert vi.from_git and not vi.dirty and vi.warning is None
    assert vi.label.startswith("v1.0 (commit ")
    assert len(vi.commit) == 40 and vi.commit[:12] in vi.label


def test_dirty_working_copy_is_flagged(repo):
    (repo / "a.py").write_text("x = 2\n")
    vi = git_info(repo)
    assert vi.dirty and "-dirty" in vi.describe
    assert "незакоммиченные" in vi.label and vi.warning


def test_not_a_git_repo(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_VERSION", raising=False)
    vi = git_info(tmp_path)
    assert not vi.from_git and vi.commit is None
    assert "не из git" in vi.label and vi.warning


def test_parent_repo_is_not_used(repo):
    # Агент лежит в подпапке чужого репозитория (например, скачан архивом в чей-то проект) —
    # версию родителя брать нельзя.
    sub = repo / "reqval-agent"
    sub.mkdir()
    assert not git_info(sub).from_git


def test_repo_without_commits(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_VERSION", raising=False)
    _git(tmp_path, "init", "-q")
    vi = git_info(tmp_path)
    assert not vi.from_git and "нет коммитов" in vi.label


def test_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_VERSION", "v9.9-ci")
    vi = git_info(tmp_path)
    assert vi.label == "v9.9-ci" and vi.warning is None
