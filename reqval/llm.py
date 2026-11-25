"""Подключение к LLM. Основной провайдер — Google Gemini (google-genai).

Дополнительно: любой OpenAI-совместимый API (OpenAI, Ollama, vLLM, LM Studio).
Все ответы запрашиваются в виде JSON по pydantic-схеме, temperature = 0.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field
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
    errors: list[str] = field(default_factory=list)


class LLMError(RuntimeError):
    pass


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
        self._last_call = 0.0

    def _raw(self, system: str, prompt: str, schema: type[BaseModel]) -> tuple[str, int, int]:
        raise NotImplementedError

    @staticmethod
    def _retryable(exc: Exception) -> tuple[bool, float | None]:
        s = str(exc)
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if code in (429, 500, 502, 503, 504) or re.search(r"\b(429|500|502|503|504)\b|RESOURCE_EXHAUSTED|UNAVAILABLE|timed? ?out|deadline", s, re.I):
            m = re.search(r"retry(?:Delay|_delay)?['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", s, re.I)
            return True, float(m.group(1)) if m else None
        return False, None

    def generate(self, system: str, prompt: str, schema: type[T], label: str = "") -> T:
        last_err: Exception | None = None
        note = ""
        for attempt in range(1, self.max_retries + 1):
            wait = self.min_interval - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            t0 = time.time()
            try:
                self._last_call = time.time()
                text, tin, tout = self._raw(system, prompt + note, schema)
                self.usage.calls += 1
                self.usage.input_tokens += tin
                self.usage.output_tokens += tout
                self.usage.seconds += time.time() - t0
                try:
                    return schema.model_validate_json(_extract_json(text))
                except (ValidationError, json.JSONDecodeError) as ve:
                    last_err = ve
                    self.usage.retries += 1
                    note = ("\n\nПредыдущий ответ не прошёл валидацию JSON-схемы: "
                            f"{str(ve)[:400]}. Верни строго валидный JSON по схеме.")
                    log.warning("[%s] невалидный JSON, повтор %d", label, attempt)
                    continue
            except Exception as e:  # noqa: BLE001
                self.usage.seconds += time.time() - t0
                ok, delay = self._retryable(e)
                last_err = e
                if not ok:
                    raise LLMError(f"[{label}] ошибка LLM: {e}") from e
                self.usage.retries += 1
                sleep = delay if delay else min(60, 5 * 2 ** (attempt - 1))
                log.warning("[%s] временная ошибка LLM (%s), ждём %.0f c, попытка %d/%d",
                            label, str(e)[:120], sleep, attempt, self.max_retries)
                time.sleep(sleep)
        self.usage.errors.append(f"{label}: {last_err}")
        raise LLMError(f"[{label}] не удалось получить ответ LLM: {last_err}")


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

    def _raw(self, system, prompt, schema):
        from google.genai import types

        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            seed=42,
            response_mime_type="application/json",
            response_schema=schema,
        )
        resp = self.client.models.generate_content(model=self.model, contents=prompt, config=cfg)
        um = getattr(resp, "usage_metadata", None)
        tin = (getattr(um, "prompt_token_count", 0) or 0) if um else 0
        tout = (getattr(um, "candidates_token_count", 0) or 0) if um else 0
        if not resp.text:
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
        return data["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def make_provider(kind: str, model: str | None) -> BaseProvider | None:
    kind = (kind or os.getenv("LLM_PROVIDER", "gemini")).lower()
    if kind == "offline":
        return None
    if kind == "gemini":
        return GeminiProvider(model)
    if kind in ("openai", "ollama", "openai-compatible"):
        return OpenAICompatProvider(model)
    raise LLMError(f"Неизвестный провайдер: {kind}")
