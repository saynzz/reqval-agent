"""Точка входа: python -m reqval <файл требований> <файл отчёта>."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from .llm import LLMError, make_provider
from .loader import load_document
from .pipeline import Pipeline
from .report import write_report
from .version import version_info


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="reqval", description="AI-агент статической проверки требований "
                                 "(ISO/IEC/IEEE 29148:2018, IEEE 830-1998)")
    ap.add_argument("input", help="Путь к документу требований (.pdf, .docx, .md, .txt)")
    ap.add_argument("output", help="Путь к файлу отчёта (.md или .html); рядом пишется .json")
    ap.add_argument("--provider", default=None, help="gemini (по умолчанию) | openrouter | openai | offline")
    ap.add_argument("--model", default=None, help="Модель (по умолчанию GEMINI_MODEL из .env, иначе gemini-2.5-flash)")
    ap.add_argument("--batch-size", type=int, default=6, help="Требований в одном запросе проверки (по умолчанию 6)")
    ap.add_argument("--no-json", action="store_true", help="Не сохранять JSON-копию отчёта")
    ap.add_argument("--concurrency", type=int, default=None, metavar="N",
                    help="Параллельных запросов к модели (по умолчанию: Gemini — 1 из-за лимита бесплатного тарифа, "
                         "OpenRouter — 4; или LLM_CONCURRENCY в .env)")
    ap.add_argument("--cache", nargs="?", const=".reqval_cache", default=None, metavar="DIR",
                    help="Брать повторяющиеся ответы модели из кэша (для отладки; по умолчанию выключено, "
                         "в паспорте отмечается). Каталог по умолчанию: .reqval_cache")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "httpcore2", "google_genai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    log = logging.getLogger("reqval")

    vi = version_info()
    log.info("Версия агента: %s", vi.label)
    if vi.warning:
        log.warning("Паспорт прогона: %s", vi.warning)

    try:
        doc = load_document(a.input)
        provider = make_provider(a.provider, a.model)
        if provider is not None and a.concurrency:
            provider.concurrency = max(1, a.concurrency)
        if provider is not None:
            out = Path(a.output).resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            if not provider.debug_dir:
                # Неудачные сырые ответы модели — рядом с отчётом, чтобы причину можно было прочитать.
                provider.debug_dir = str(out.parent / "llm_debug")
            # Журнал всех обращений к модели — рядом с отчётом: report.md → report.llm.jsonl
            provider.journal_path = str(out.with_suffix(".llm.jsonl"))
            Path(provider.journal_path).write_text("", encoding="utf-8")
            if a.cache:
                provider.cache_dir = a.cache
                log.warning("Кэш ответов модели включён (%s): для финального прогона запускайте без --cache", a.cache)
        if provider:
            log.info("Модель: %s (%s)", provider.model, provider.name)
        else:
            log.warning("Режим offline: только формальные правила, без LLM")
        res = Pipeline(doc, provider, batch_size=a.batch_size).run()
        paths = write_report(res, a.output, with_json=not a.no_json)
    except (LLMError, FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        return 2
    for p in paths:
        log.info("Отчёт: %s", p)
    if provider is not None and provider.journal_path:
        log.info("Журнал обращений к модели: %s (%d записей)", provider.journal_path, provider.usage.journal_records)
    log.info("Готово: требований %d, дефектов %d, итог: %s", len(res.requirements), len(res.unique_findings), res.verdict.status)
    if provider is not None and provider.usage.cost_usd:
        log.info("Стоимость прогона: $%.4f", provider.usage.cost_usd)
    if not res.complete:
        log.warning("Проверка выполнена не полностью (%s) — отчёт записан, но перед сдачей повторите прогон",
                    ", ".join(res.failed_steps + ([f"не оценены: {', '.join(res.unchecked)}"] if res.unchecked else [])))
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
