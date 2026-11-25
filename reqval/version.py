"""Версия агента для паспорта прогона.

ТЗ (§6) требует указать в паспорте версию агента — коммит или тег. Версия берётся из git:
`git describe --tags --always --dirty` + полный хэш коммита. Если git недоступен или в рабочей копии
есть незакоммиченные изменения, это явно помечается в паспорте и в логе: отчёт, сделанный не из
зафиксированного коммита, нельзя воспроизвести по ссылке на репозиторий.
"""
from __future__ import annotations

import functools
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

__version__ = "0.2.0"

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class VersionInfo:
    label: str              # как показывать в паспорте
    commit: str | None      # полный хэш коммита или None
    describe: str | None    # вывод git describe (тег/коммит, суффикс -dirty)
    dirty: bool             # есть незакоммиченные изменения в отслеживаемых файлах
    from_git: bool          # версия получена из git

    @property
    def warning(self) -> str | None:
        if not self.from_git:
            return ("версия агента не из git: коммит неизвестен. Запускайте агента из клона репозитория "
                    "(git clone …) или задайте AGENT_VERSION.")
        if self.dirty:
            return ("в рабочей копии есть незакоммиченные изменения: отчёт нельзя воспроизвести по коммиту. "
                    "Закоммитьте изменения перед финальным прогоном.")
        return None


def _git(root: Path, *args: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def git_info(root: Path | str | None = None) -> VersionInfo:
    """Версия по git-репозиторию в каталоге root (по умолчанию — корень проекта)."""
    env = os.getenv("AGENT_VERSION")
    if env:
        return VersionInfo(label=env, commit=None, describe=None, dirty=False, from_git=True)
    root = Path(root) if root else REPO_ROOT
    # Каталог должен сам быть корнем репозитория: иначе git возьмёт версию чужого родительского репо.
    top = _git(root, "rev-parse", "--show-toplevel")
    if not top or Path(top).resolve() != root.resolve():
        return VersionInfo(label=f"v{__version__} (⚠ не из git-репозитория, коммит неизвестен)",
                           commit=None, describe=None, dirty=False, from_git=False)
    commit = _git(root, "rev-parse", "HEAD")
    desc = _git(root, "describe", "--tags", "--always", "--dirty")
    if not commit or not desc:   # репозиторий без единого коммита
        return VersionInfo(label=f"v{__version__} (⚠ в репозитории нет коммитов)",
                           commit=None, describe=None, dirty=False, from_git=False)
    dirty = desc.endswith("-dirty")
    label = f"{desc} (commit {commit[:12]})"
    if dirty:
        label += " ⚠ есть незакоммиченные изменения"
    return VersionInfo(label=label, commit=commit, describe=desc, dirty=dirty, from_git=True)


@functools.lru_cache(maxsize=1)
def version_info() -> VersionInfo:
    return git_info()


def agent_version() -> str:
    return version_info().label
