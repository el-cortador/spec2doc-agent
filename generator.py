import requests
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:4b"

_SKILL_PATH = Path(__file__).parent / "skill-description.md"


class OllamaConnectionError(Exception):
    pass


class ModelNotFoundError(Exception):
    pass


def _load_system_prompt() -> str:
    return _SKILL_PATH.read_text(encoding="utf-8")


def generate_draft(extracted_text: str) -> str:
    system_prompt = _load_system_prompt()

    user_message = (
        "/no_think\n\n"
        "Обработай следующую постановку аналитика и сформируй черновик технической документации "
        "согласно инструкции.\n\n"
        "<ПОСТАНОВКА>\n"
        f"{extracted_text}\n"
        "</ПОСТАНОВКА>"
    )

    payload = {
        "model": MODEL,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 4096,
        },
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    except requests.exceptions.ConnectionError:
        raise OllamaConnectionError(
            "Ollama не запущена. Выполните: ollama serve"
        )

    if response.status_code == 404:
        raise ModelNotFoundError(
            f"Модель {MODEL} не найдена. Выполните: ollama pull {MODEL}"
        )

    response.raise_for_status()

    return response.json()["message"]["content"]
