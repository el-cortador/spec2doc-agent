"""
Модуль генерации черновика. Поддерживает два бэкенда:
  - ollama    (по умолчанию): http://localhost:11434/api/chat
  - llamacpp              : http://localhost:8080/v1/chat/completions

Выбор бэкенда задаётся переменной окружения:
  BACKEND=llamacpp   python app.py
  BACKEND=ollama     python app.py   (или просто python app.py)
"""
import json
import os
import requests
from pathlib import Path

# ── Конфигурация ──────────────────────────────────────────────────────────────

BACKEND = os.environ.get("BACKEND", "ollama").lower()

_BACKENDS = {
    "ollama":   "http://localhost:11434/api/chat",
    "llamacpp": "http://localhost:8080/v1/chat/completions",
}

if BACKEND not in _BACKENDS:
    raise ValueError(f"Неизвестный бэкенд «{BACKEND}». Допустимы: {list(_BACKENDS)}")

LLM_URL = _BACKENDS[BACKEND]
MODEL   = "qwen3.5:0.8b"  # для ollama; llamacpp использует модель, загруженную при старте

_SKILL_PATH = Path(__file__).parent.parent / "prompts" / "system_prompt.md"

# ── Исключения ────────────────────────────────────────────────────────────────

class LLMConnectionError(Exception):
    pass


class ModelNotFoundError(Exception):
    pass


# ── Внутренние функции ────────────────────────────────────────────────────────

def _load_system_prompt() -> str:
    return _SKILL_PATH.read_text(encoding="utf-8")


def _build_messages(extracted_text: str) -> list[dict]:
    user_message = (
        "/no_think\n\n"
        "Обработай следующую постановку аналитика и сформируй черновик технической документации "
        "согласно инструкции.\n\n"
        "<ПОСТАНОВКА>\n"
        f"{extracted_text}\n"
        "</ПОСТАНОВКА>"
    )
    return [
        {"role": "system", "content": _load_system_prompt()},
        {"role": "user",   "content": user_message},
    ]


def _call_ollama(messages: list[dict]) -> str:
    payload = {
        "model": MODEL,
        "stream": True,
        "options": {"temperature": 0.2, "num_predict": 4096},
        "messages": messages,
    }
    response = requests.post(LLM_URL, json=payload, stream=True, timeout=(10, None))

    if response.status_code == 404:
        raise ModelNotFoundError(
            f"Модель {MODEL} не найдена. Выполните: ollama pull {MODEL}"
        )
    response.raise_for_status()

    chunks = []
    for line in response.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        content = chunk.get("message", {}).get("content", "")
        if content:
            chunks.append(content)
        if chunk.get("done"):
            break
    return "".join(chunks)


def _call_llamacpp(messages: list[dict]) -> str:
    payload = {
        "model": MODEL,      # сервер игнорирует, но поле обязательно по спецификации
        "stream": True,
        "temperature": 0.2,
        "max_tokens": 4096,
        "messages": messages,
    }
    response = requests.post(LLM_URL, json=payload, stream=True, timeout=(10, None))

    if response.status_code == 404:
        raise ModelNotFoundError(
            "Модель не загружена в llama.cpp server. "
            "Укажите модель при запуске: llama-server -m qwen3.5:0.8b.gguf"
        )
    response.raise_for_status()

    # SSE-формат: каждая строка — "data: {...}" или "data: [DONE]"
    chunks = []
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
    return "".join(chunks)


# ── Публичный API ─────────────────────────────────────────────────────────────

def generate_draft(extracted_text: str) -> str:
    messages = _build_messages(extracted_text)
    try:
        if BACKEND == "llamacpp":
            return _call_llamacpp(messages)
        return _call_ollama(messages)
    except requests.exceptions.ConnectionError:
        hints = {
            "ollama":   "Выполните: ollama serve",
            "llamacpp": "Выполните: llama-server -m qwen3.5:0.8b.gguf --port 8080",
        }
        raise LLMConnectionError(f"Бэкенд «{BACKEND}» недоступен. {hints[BACKEND]}")
