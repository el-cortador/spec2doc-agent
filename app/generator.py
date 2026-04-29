from __future__ import annotations

import json
import logging
import os

import requests

from app.config import MODEL, SYSTEM_PROMPT_PATH

logger = logging.getLogger(__name__)

# Читается при загрузке модуля; importlib.reload() подхватывает новое значение env
BACKEND = os.environ.get("BACKEND", "ollama").lower()

_BACKENDS: dict[str, str] = {
    "ollama":   "http://localhost:11434/api/chat",
    "llamacpp": "http://localhost:8080/v1/chat/completions",
}

if BACKEND not in _BACKENDS:
    raise ValueError(f"Неизвестный бэкенд «{BACKEND}». Допустимы: {list(_BACKENDS)}")

LLM_URL = _BACKENDS[BACKEND]


class LLMConnectionError(Exception):
    pass


class ModelNotFoundError(Exception):
    pass


def _load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def _build_messages(extracted_text: str) -> list[dict]:
    user_message = (
        "/no_think\n\n"
        "Обработай следующую постановку аналитика и сформируй черновик технической документации "
        "согласно инструкции.\n\n"
        f"<ПОСТАНОВКА>\n{extracted_text}\n</ПОСТАНОВКА>"
    )
    return [
        {"role": "system", "content": _load_system_prompt()},
        {"role": "user",   "content": user_message},
    ]


def _call_ollama(messages: list[dict]) -> str:
    payload = {
        "model":   MODEL,
        "stream":  True,
        "options": {"temperature": 0.2, "num_predict": 4096},
        "messages": messages,
    }
    response = requests.post(LLM_URL, json=payload, stream=True, timeout=(10, None))

    if response.status_code == 404:
        raise ModelNotFoundError(f"Модель {MODEL} не найдена. Выполните: ollama pull {MODEL}")
    response.raise_for_status()

    chunks: list[str] = []
    for line in response.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        content = chunk.get("message", {}).get("content", "")
        if content:
            chunks.append(content)
        if chunk.get("done"):
            break

    logger.debug("[generator] ollama chunks=%d", len(chunks))
    return "".join(chunks)


def _call_llamacpp(messages: list[dict]) -> str:
    payload = {
        "model":       MODEL,
        "stream":      True,
        "temperature": 0.2,
        "max_tokens":  4096,
        "messages":    messages,
    }
    response = requests.post(LLM_URL, json=payload, stream=True, timeout=(10, None))

    if response.status_code == 404:
        raise ModelNotFoundError(
            "Модель не загружена в llama-server. "
            "Укажите модель при запуске: llama-server -m qwen3:4b.gguf"
        )
    response.raise_for_status()

    chunks: list[str] = []
    for line in response.iter_lines():
        if not line:
            continue
        text = line.decode("utf-8") if isinstance(line, bytes) else line
        if not text.startswith("data:"):
            continue
        payload_str = text[len("data:"):].strip()
        if payload_str == "[DONE]":
            break
        chunk = json.loads(payload_str)
        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
        if content:
            chunks.append(content)

    logger.debug("[generator] llamacpp chunks=%d", len(chunks))
    return "".join(chunks)


def generate_draft(extracted_text: str) -> str:
    messages = _build_messages(extracted_text)
    logger.info("[generator] backend=%s model=%s", BACKEND, MODEL)
    try:
        if BACKEND == "llamacpp":
            return _call_llamacpp(messages)
        return _call_ollama(messages)
    except requests.exceptions.ConnectionError:
        hints = {
            "ollama":   "Выполните: ollama serve",
            "llamacpp": "Выполните: llama-server -m qwen3:4b.gguf --port 8080",
        }
        raise LLMConnectionError(f"Бэкенд «{BACKEND}» недоступен. {hints[BACKEND]}")
