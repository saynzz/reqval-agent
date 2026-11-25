"""Подключение к LLM. Основной провайдер — Google Gemini (google-genai).

Дополнительно: любой OpenAI-совместимый API (OpenAI, Ollama, vLLM, LM Studio).
Все ответы запрашиваются в виде JSON по pydantic-схеме, temperature = 0.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from typing import TypeVar

from pydantic import BaseModel, ValidationError

log = logging.getLogger("reqval")
T = TypeVar("T", bound=BaseModel)


@dataclass
class Usage:
    calls: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    cost_usd: float = 0.0          # если провайдер сообщает стоимость (OpenRouter)
    cached: int = 0                # ответов взято из кэша (--cache)
    journal_records: int = 0       # записей в журнале обращений
    steps: dict = field(default_factory=dict)      # шаг → модель, которая на нём ответила
    skipped: list = field(default_factory=list)    # шаги, пропущенные предохранителем
    errors: list[str] = field(default_factory=list)


class LLMError(RuntimeError):
    pass


class TruncatedError(LLMError):
    """Ответ модели обрезан по лимиту длины. Повтор с теми же параметрами даст то же самое — не повторяем."""


class SchemaError(LLMError):
    """Модель отвечает, но ответы раз за разом не проходят проверку схемы."""


class Unavailable(LLMError):
    """Модель недоступна: временные ошибки (429/5xx/таймаут) не прошли за все попытки."""


class CircuitOpen(LLMError):
    """Предохранитель сработал: модель недоступна несколько шагов подряд, обращения пропускаются."""


TRUNCATION_ADVICE = ("Увеличьте LLM_MAX_OUTPUT_TOKENS в .env, уменьшите --batch-size или задайте "
                     "LLM_REASONING_EFFORT=low (рассуждения модели расходуют тот же лимит, что и ответ).")


def _safe_name(s: str) -> str:
    return re.sub(r"[^\w-]+", "_", s, flags=re.U).strip("_")[:60] or "call"


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _short_error(e: Exception) -> str:
    """Первая ошибка валидации одной строкой: где и что не так."""
    if isinstance(e, ValidationError):
        errs = e.errors()
        first = errs[0] if errs else {}
        loc = ".".join(str(x) for x in first.get("loc", ()))
        return f"{len(errs)} ошибок, первая: {loc or '—'}: {first.get('msg', '')}"[:200]
    return str(e)[:200]


def _extract_json(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    if not text.startswith(("{", "[")):
        i = text.find("{")
        if i >= 0:
            text = text[i:]
    return text


class BaseProvider:
    name = "base"
    model = ""

    def __init__(self) -> None:
        self.usage = Usage()
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
        self.max_retries = int(os.getenv("LLM_MAX_RETRIES", "6"))
        self.min_interval = float(os.getenv("LLM_MIN_INTERVAL", "0"))
        self.max_output_tokens = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "32768"))
        self.reasoning_effort = (os.getenv("LLM_REASONING_EFFORT") or "").strip().lower() or None
        self.debug_dir = os.getenv("LLM_DEBUG_DIR") or None
        self.journal_path: str | None = None  # журнал обращений (JSONL), путь задаёт CLI
        self.cache_dir: str | None = None     # кэш ответов (--cache), по умолчанию выключен
        self.fallback_model = (os.getenv("LLM_FALLBACK_MODEL") or "").strip() or None
        self.breaker_after = int(os.getenv("LLM_BREAKER_AFTER", "2"))  # шагов подряд с недоступной моделью
        self.primary_model = ""
        self.tripped: str | None = None       # причина срабатывания предохранителя
        self._failed_in_row = 0
        self.last_finish: str | None = None   # finish_reason последнего ответа (заполняет _raw)
        self._last_call = 0.0

    def describe(self) -> str:
        """Параметры запросов для паспорта прогона."""
        extra = f", reasoning={self.reasoning_effort}" if self.reasoning_effort else ""
        return (f"temperature={self.temperature}, seed=42{extra}, max_tokens={self.max_output_tokens}; "
                "формат ответа: JSON-схема")

    def _raw(self, system: str, prompt: str, schema: type[BaseModel]) -> tuple[str, int, int]:
        raise NotImplementedError

    def _dump(self, label: str, attempt: int, text: str, error: str) -> str | None:
        """Сохраняет неудачный сырой ответ модели, чтобы причину можно было прочитать глазами."""
        if not self.debug_dir:
            return None
        try:
            os.makedirs(self.debug_dir, exist_ok=True)
            path = os.path.join(self.debug_dir, f"{time.strftime('%H%M%S')}-{_safe_name(label)}-{attempt}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"model: {self.model}\nfinish_reason: {self.last_finish}\nerror: {error}\n\n{text}")
            return path
        except OSError:
            return None

    @staticmethod
    def _retryable(exc: Exception) -> tuple[bool, float | None]:
        if isinstance(exc, TruncatedError):
            return False, None
        s = str(exc)
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if code in (429, 500, 502, 503, 504) or re.search(r"\b(429|500|502|503|504)\b|RESOURCE_EXHAUSTED|UNAVAILABLE|timed? ?out|deadline", s, re.I):
            m = re.search(r"retry(?:Delay|_delay)?['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", s, re.I)
            return True, float(m.group(1)) if m else None
        return False, None

    # ------------------------------------------------------------------ журнал и кэш
    def _journal(self, **rec) -> None:
        """Одна строка JSONL на каждое обращение к модели: доказательство, что отчёт сделан агентом,
        и материал для разбора сбоев. Ключ API в журнал не попадает."""
        if not self.journal_path:
            return
        rec = {"ts": datetime.now().astimezone().isoformat(timespec="seconds"), "provider": self.name,
               "model": self.model, "params": self.describe(), **rec}
        try:
            with open(self.journal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.usage.journal_records += 1
        except OSError as e:
            log.warning("Журнал обращений не записан: %s", e)

    def _cache_key(self, system: str, prompt: str, schema: type[BaseModel]) -> str:
        raw = json.dumps([self.name, self.model, self.describe(), schema.__name__, system, prompt], ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cache_get(self, key: str) -> str | None:
        if not self.cache_dir:
            return None
        path = os.path.join(self.cache_dir, key + ".json")
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)["response"]
        except (OSError, ValueError, KeyError):
            return None

    def _cache_put(self, key: str, text: str, label: str) -> None:
        if not self.cache_dir:
            return
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(os.path.join(self.cache_dir, key + ".json"), "w", encoding="utf-8") as f:
                json.dump({"model": self.model, "step": label, "response": text}, f, ensure_ascii=False)
        except OSError:
            pass

    def _switch_model(self, model: str) -> None:
        """Переход на запасную модель. Провайдеры с режимами (OpenRouter) пересчитывают их здесь."""
        self.model = model

    # ------------------------------------------------------------------ обращение к модели
    def generate(self, system: str, prompt: str, schema: type[T], label: str = "") -> T:
        """Один шаг конвейера: кэш → попытки → (при недоступности) запасная модель → предохранитель."""
        if not self.primary_model:
            self.primary_model = self.model
        sys_h, prm_h = _sha(system), _sha(prompt)
        if self.tripped:
            self.usage.skipped.append(label)
            self._journal(step=label, outcome="skipped", error=self.tripped, system_sha256=sys_h, prompt_sha256=prm_h)
            raise CircuitOpen(f"[{label}] шаг пропущен: модель недоступна ({self.tripped})")

        key = self._cache_key(system, prompt, schema)
        cached = self._cache_get(key)
        if cached is not None:
            try:
                res = schema.model_validate_json(_extract_json(cached))
                self.usage.cached += 1
                self.usage.steps[label] = self.model
                self._journal(step=label, outcome="cache", system_sha256=sys_h, prompt_sha256=prm_h, response=cached)
                return res
            except (ValidationError, json.JSONDecodeError):
                pass

        try:
            res, text = self._attempts(system, prompt, schema, label, sys_h, prm_h)
        except Unavailable as e:
            if self.fallback_model and self.model != self.fallback_model:
                log.warning("[%s] модель %s недоступна — переключаюсь на запасную %s", label, self.model,
                            self.fallback_model)
                self._journal(step=label, outcome="fallback", error=str(e)[:300], next_model=self.fallback_model)
                self._switch_model(self.fallback_model)
                try:
                    res, text = self._attempts(system, prompt, schema, label, sys_h, prm_h)
                except LLMError as e2:
                    self._step_failed(label, e2)
                    raise
            else:
                self._step_failed(label, e)
                raise
        except LLMError as e:
            self._step_failed(label, e)
            raise
        self._failed_in_row = 0
        self.usage.steps[label] = self.model
        self._cache_put(key, text, label)
        return res

    def _step_failed(self, label: str, e: LLMError) -> None:
        """Предохранитель: если модель недоступна несколько шагов подряд, дальше не ждём на каждом шаге
        по несколько минут повторов, а сразу пишем отчёт с итогом «ПРОВЕРКА НЕ ЗАВЕРШЕНА».
        Плохие ответы (схема, обрезка) — не недоступность: модель отвечает, предохранитель их не считает."""
        self.usage.errors.append(f"{label}: {e}")
        if isinstance(e, (SchemaError, TruncatedError)):
            return
        self._failed_in_row += 1
        if self._failed_in_row >= self.breaker_after and not self.tripped:
            self.tripped = f"{self._failed_in_row} шага подряд без ответа; последняя ошибка: {str(e)[:160]}"
            log.error("Предохранитель: модель %s недоступна %d шага подряд — остальные шаги пропускаются, "
                      "отчёт будет помечен «ПРОВЕРКА НЕ ЗАВЕРШЕНА»", self.model, self._failed_in_row)

    def _attempts(self, system: str, prompt: str, schema: type[T], label: str, sys_h: str, prm_h: str) -> tuple[T, str]:
        last_err: Exception | None = None
        note = ""
        transient = False
        for attempt in range(1, self.max_retries + 1):
            wait = self.min_interval - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            t0 = time.time()
            cost0 = self.usage.cost_usd
            base = dict(step=label, attempt=attempt, system_sha256=sys_h, prompt_sha256=prm_h,
                        prompt_chars=len(prompt + note))
            try:
                self._last_call = time.time()
                self.last_finish = None
                text, tin, tout = self._raw(system, prompt + note, schema)
            except Exception as e:  # noqa: BLE001
                dt = time.time() - t0
                self.usage.seconds += dt
                ok, delay = self._retryable(e)
                self._journal(**base, outcome="error", seconds=round(dt, 2), error=str(e)[:500],
                              cost_usd=round(self.usage.cost_usd - cost0, 6))
                last_err = e
                if not ok:
                    if isinstance(e, LLMError):
                        raise type(e)(f"[{label}] ошибка LLM: {e}") from e
                    raise LLMError(f"[{label}] ошибка LLM: {e}") from e
                transient = True
                self.usage.retries += 1
                if attempt == self.max_retries:
                    break
                sleep = delay if delay else min(60, 5 * 2 ** (attempt - 1))
                log.warning("[%s] временная ошибка LLM (%s), ждём %.0f c, попытка %d/%d",
                            label, str(e)[:120], sleep, attempt, self.max_retries)
                time.sleep(sleep)
                continue

            dt = time.time() - t0
            self.usage.calls += 1
            self.usage.input_tokens += tin
            self.usage.output_tokens += tout
            self.usage.seconds += dt
            rec = dict(**base, seconds=round(dt, 2), finish=self.last_finish, input_tokens=tin, output_tokens=tout,
                       cost_usd=round(self.usage.cost_usd - cost0, 6), response=text)
            log.debug("[%s] ответ за %.1f c: вход %d, выход %d токенов, finish=%s", label, dt, tin, tout,
                      self.last_finish)
            try:
                res = schema.model_validate_json(_extract_json(text))
            except (ValidationError, json.JSONDecodeError) as ve:
                err = _short_error(ve)
                self._journal(**rec, outcome="schema_error", error=err)
                dump = self._dump(label, attempt, text, err)
                if (self.last_finish or "").lower() in ("length", "max_tokens"):
                    raise TruncatedError(
                        f"[{label}] ответ модели обрезан по лимиту длины (finish_reason={self.last_finish}, "
                        f"выход {tout} токенов, лимит {self.max_output_tokens}). {TRUNCATION_ADVICE}"
                        + (f" Сырой ответ: {dump}" if dump else ""))
                last_err = ve
                transient = False
                self.usage.retries += 1
                note = ("\n\nПредыдущий ответ не прошёл валидацию JSON-схемы: "
                        f"{str(ve)[:400]}. Верни строго валидный JSON по схеме.")
                log.warning("[%s] ответ не прошёл проверку схемы (finish=%s, выход %d токенов): %s — повтор %d/%d%s",
                            label, self.last_finish, tout, err, attempt, self.max_retries,
                            f"; сырой ответ: {dump}" if dump else "")
                continue
            self._journal(**rec, outcome="ok")
            return res, text
        if transient:
            raise Unavailable(f"[{label}] модель недоступна после {self.max_retries} попыток: {last_err}")
        raise SchemaError(f"[{label}] не удалось получить ответ по схеме за {self.max_retries} попыток: {last_err}")


class GeminiProvider(BaseProvider):
    name = "Google Gemini API"

    def __init__(self, model: str | None = None) -> None:
        super().__init__()
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key or key.startswith("your_"):
            raise LLMError("Не задан GEMINI_API_KEY. Скопируйте .env.example в .env и укажите ключ "
                           "(https://aistudio.google.com/apikey) или запустите с --provider offline.")
        from google import genai

        self.client = genai.Client(api_key=key)
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.fallback_model = (os.getenv("GEMINI_FALLBACK_MODEL") or "").strip() or self.fallback_model

    def _raw(self, system, prompt, schema):
        from google.genai import types

        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            seed=42,
            response_mime_type="application/json",
            response_schema=schema,
            max_output_tokens=self.max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=self.reasoning_effort) if self.reasoning_effort else None,
        )
        resp = self.client.models.generate_content(model=self.model, contents=prompt, config=cfg)
        if resp.candidates:
            fr = getattr(resp.candidates[0], "finish_reason", None)
            self.last_finish = getattr(fr, "name", None) or (str(fr) if fr else None)
        um = getattr(resp, "usage_metadata", None)
        tin = (getattr(um, "prompt_token_count", 0) or 0) if um else 0
        tout = (getattr(um, "candidates_token_count", 0) or 0) if um else 0
        if not resp.text:
            if (self.last_finish or "").upper() == "MAX_TOKENS":
                raise TruncatedError(f"ответ модели обрезан по лимиту длины до начала JSON. {TRUNCATION_ADVICE}")
            raise LLMError(f"пустой ответ модели (finish_reason={getattr(resp.candidates[0], 'finish_reason', '?') if resp.candidates else '?'})")
        return resp.text, tin, tout


class OpenAICompatProvider(BaseProvider):
    """OpenAI / Ollama / vLLM / LM Studio через /chat/completions."""

    name = "OpenAI-compatible API"

    def __init__(self, model: str | None = None) -> None:
        super().__init__()
        self.base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.key = os.getenv("OPENAI_API_KEY", "")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if "api.openai.com" in self.base and not self.key:
            raise LLMError("Не задан OPENAI_API_KEY")

    def _raw(self, system, prompt, schema):
        schema_txt = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "seed": 42,
            "max_tokens": self.max_output_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system + "\n\nОтвечай только JSON по схеме:\n" + schema_txt},
                {"role": "user", "content": prompt},
            ],
        }
        req = urllib.request.Request(self.base + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {self.key}"} if self.key else {})})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.loads(r.read())
        u = data.get("usage", {})
        self.last_finish = data["choices"][0].get("finish_reason")
        return data["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def strict_json_schema(schema: type[BaseModel]) -> dict:
    """JSON-схема pydantic-модели в виде для strict structured outputs (OpenAI / OpenRouter):
    без $ref (определения встроены), additionalProperties=false, все поля обязательны."""
    root = schema.model_json_schema()
    defs = root.pop("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(dict(defs[node["$ref"].split("/")[-1]]))
            out = {}
            for k, v in node.items():
                if k == "title":            # служебный ключ JSON-схемы, а не поле модели
                    continue
                if k == "properties":       # имена полей сохраняем все, включая поле с именем «title»
                    out[k] = {name: walk(sub) for name, sub in v.items()}
                else:
                    out[k] = walk(v)
            if out.get("type") == "object" and "properties" in out:
                out["additionalProperties"] = False
                out["required"] = list(out["properties"])
                for sub in out["properties"].values():   # strict-режим не принимает default
                    if isinstance(sub, dict):
                        sub.pop("default", None)
            return out
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(root)


class OpenRouterProvider(OpenAICompatProvider):
    """OpenRouter (https://openrouter.ai): любые модели, в т.ч. Gemini, через OpenAI-совместимый API.

    В отличие от общего OpenAI-совместимого провайдера ответ запрашивается строго по JSON-схеме
    (response_format=json_schema, strict), а запрос направляется только к хостингам, которые эту схему
    поддерживают (provider.require_parameters). Стоимость прогона берётся из ответа OpenRouter."""

    name = "OpenRouter"

    def __init__(self, model: str | None = None) -> None:
        BaseProvider.__init__(self)
        self.base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        self.key = os.getenv("OPENROUTER_API_KEY", "")
        self.model = model or os.getenv("OPENROUTER_MODEL", "")
        if not self.key or self.key.startswith("your_"):
            raise LLMError("Не задан OPENROUTER_API_KEY (https://openrouter.ai/settings/keys)")
        if not self.model:
            raise LLMError("Не задан OPENROUTER_MODEL — точный id модели со страницы openrouter.ai, "
                           "например google/gemini-3.1-pro-preview")

    # Лестница совместимости. Не у каждой модели на OpenRouter есть провайдер, поддерживающий ВСЕ параметры
    # запроса (например, Anthropic не принимает seed). Тогда OpenRouter отвечает 404 «No endpoints found that
    # can handle the requested parameters», и агент спускается на ступень ниже, запоминая её до конца прогона.
    MODES = [
        {"name": "json_schema strict, seed=42", "schema": True, "seed": True, "temp": True, "reasoning": True,
         "require": True},
        {"name": "json_schema strict (без seed)", "schema": True, "seed": False, "temp": True, "reasoning": True,
         "require": True},
        # Модели с постоянно включённым мышлением (например, Claude 5.x) не принимают temperature:
        # строгая схема и глубина рассуждений (effort) при этом доступны.
        {"name": "json_schema strict (без seed и temperature — модель с постоянным мышлением)", "schema": True,
         "seed": False, "temp": False, "reasoning": True, "require": True},
        {"name": "json_schema strict (без seed, temperature и настройки рассуждений)", "schema": True,
         "seed": False, "temp": False, "reasoning": False, "require": True},
        # Вызов инструмента: схема — параметры функции, модель обязана её вызвать (tool_choice).
        {"name": "вызов инструмента по JSON-схеме (tool calling)", "schema": False, "tool": True, "seed": False,
         "temp": False, "reasoning": False, "require": True},
        {"name": "JSON-схема в промпте + проверка ответа агентом", "schema": False, "seed": False, "temp": False,
         "reasoning": False, "require": False},
    ]
    TOOL_NAME = "submit_review"

    @classmethod
    def _start_mode(cls, model: str) -> int:
        """С какой ступени начинать. OPENROUTER_MODE задаёт её явно. Для Claude — сразу «без seed и temperature»:
        у моделей Anthropic с постоянным мышлением temperature не задаётся, и первые ступени дали бы 404."""
        forced = (os.getenv("OPENROUTER_MODE") or "").strip().lower()
        names = {"json_schema": 0, "json_schema_noseed": 1, "json_schema_notemp": 2, "tool": 4, "prompt": 5}
        if forced in names:
            return names[forced]
        return 2 if model.lower().startswith("anthropic/") else 0

    def __init__(self, model: str | None = None) -> None:
        BaseProvider.__init__(self)
        self.base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        self.key = os.getenv("OPENROUTER_API_KEY", "")
        self.model = model or os.getenv("OPENROUTER_MODEL", "")
        self.mode = self._start_mode(self.model)
        self.fallback_model = (os.getenv("OPENROUTER_FALLBACK_MODEL") or "").strip() or self.fallback_model
        if not self.key or self.key.startswith("your_"):
            raise LLMError("Не задан OPENROUTER_API_KEY (https://openrouter.ai/settings/keys)")
        if not self.model:
            raise LLMError("Не задан OPENROUTER_MODEL — точный id модели со страницы openrouter.ai, "
                           "например google/gemini-3.1-pro-preview")

    def _switch_model(self, model: str) -> None:
        self.model = model
        self.mode = self._start_mode(model)

    def describe(self) -> str:
        m = self.MODES[self.mode]
        temp = (f"temperature={self.temperature}" if m["temp"] else
                "temperature не задаётся (модель не поддерживает) — ответы между прогонами могут различаться")
        extra = f", reasoning={self.reasoning_effort}" if self.reasoning_effort and m["reasoning"] else ""
        return f"{temp}{extra}, max_tokens={self.max_output_tokens}; формат ответа: {m['name']}"

    def _body(self, system: str, prompt: str, schema: type[BaseModel]) -> dict:
        m = self.MODES[self.mode]
        if m["schema"]:
            sys_text = system
        elif m.get("tool"):
            sys_text = system + f"\n\nВерни результат ТОЛЬКО вызовом инструмента {self.TOOL_NAME}."
        else:
            sys_text = (system + "\n\nОтвечай ТОЛЬКО одним JSON-объектом, без пояснений и markdown, строго по схеме:\n"
                        + json.dumps(strict_json_schema(schema), ensure_ascii=False))
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": sys_text}, {"role": "user", "content": prompt}],
            "max_tokens": self.max_output_tokens,
            "usage": {"include": True},
        }
        if m["schema"]:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": schema.__name__, "strict": True, "schema": strict_json_schema(schema)}}
        if m.get("tool"):
            body["tools"] = [{"type": "function", "function": {
                "name": self.TOOL_NAME, "description": f"Результат шага ревью требований ({schema.__name__})",
                "parameters": strict_json_schema(schema)}}]
            body["tool_choice"] = {"type": "function", "function": {"name": self.TOOL_NAME}}
        if m["seed"]:
            body["seed"] = 42
        if m["temp"]:
            body["temperature"] = self.temperature
        if m["reasoning"] and self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        if m["require"]:
            body["provider"] = {"require_parameters": True}
        return body

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(self.base + "/chat/completions", data=json.dumps(body).encode(), headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.key}",
            "X-Title": "reqval requirements review agent"})  # только ASCII: заголовки HTTP в latin-1
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:   # тело ошибки OpenRouter объясняет причину (ключ, баланс, модель)
            detail = e.read().decode("utf-8", "replace")[:500]
            err = LLMError(f"HTTP {e.code} от OpenRouter: {detail}")
            err.code = e.code
            err.detail = detail
            raise err from e

    @staticmethod
    def _no_endpoints(e: LLMError) -> bool:
        return getattr(e, "code", None) == 404 and "no endpoints found" in (getattr(e, "detail", "") or "").lower()

    def _raw(self, system, prompt, schema):
        while True:
            try:
                data = self._post(self._body(system, prompt, schema))
                break
            except LLMError as e:
                if not self._no_endpoints(e) or self.mode + 1 >= len(self.MODES):
                    raise
                self.mode += 1
                log.warning("OpenRouter: для %s нет провайдера с такими параметрами — режим «%s»",
                            self.model, self.MODES[self.mode]["name"])
        if "error" in data:
            err = LLMError(f"OpenRouter: {data['error']}")
            err.code = (data["error"] or {}).get("code") if isinstance(data["error"], dict) else None
            raise err
        u = data.get("usage") or {}
        self.usage.cost_usd += float(u.get("cost") or 0)
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        for call in msg.get("tool_calls") or []:          # режим tool calling: ответ — аргументы вызова
            fn = call.get("function") or {}
            if fn.get("name") == self.TOOL_NAME:
                args = fn.get("arguments")
                content = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
                break
        self.last_finish = data["choices"][0].get("finish_reason")
        if not content and (self.last_finish or "").lower() == "length":
            raise TruncatedError(f"ответ модели обрезан по лимиту длины до начала JSON. {TRUNCATION_ADVICE}")
        if not content:
            raise LLMError(f"пустой ответ модели (finish_reason={data['choices'][0].get('finish_reason')})")
        return content, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def make_provider(kind: str, model: str | None) -> BaseProvider | None:
    kind = (kind or os.getenv("LLM_PROVIDER", "gemini")).lower()
    if kind == "offline":
        return None
    if kind == "gemini":
        return GeminiProvider(model)
    if kind == "openrouter":
        return OpenRouterProvider(model)
    if kind in ("openai", "ollama", "openai-compatible"):
        return OpenAICompatProvider(model)
    raise LLMError(f"Неизвестный провайдер: {kind}")
