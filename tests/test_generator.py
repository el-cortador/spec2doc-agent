"""
Тесты модуля core/generator.py.
Ollama не поднимается — все HTTP-вызовы мокируются.
"""
import pytest
import requests as req_lib
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.generator import generate_draft, OllamaConnectionError, ModelNotFoundError


def _mock_response(status_code: int, json_body: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body
    resp.raise_for_status = MagicMock()
    return resp


# ── Успешная генерация ────────────────────────────────────────────────────────

def test_generate_draft_returns_content(tmp_path):
    fake_reply = {"message": {"content": "## Черновик\n\n### Описание\nТекст"}}

    with patch("core.generator.requests.post", return_value=_mock_response(200, fake_reply)):
        result = generate_draft("Хочу фичу")

    assert result == "## Черновик\n\n### Описание\nТекст"


def test_generate_draft_sends_no_think_directive(tmp_path):
    fake_reply = {"message": {"content": "ok"}}

    with patch("core.generator.requests.post", return_value=_mock_response(200, fake_reply)) as mock_post:
        generate_draft("Постановка")

    payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args.args[1]
    user_content = payload["messages"][1]["content"]
    assert user_content.startswith("/no_think")


def test_generate_draft_passes_system_prompt(tmp_path):
    fake_reply = {"message": {"content": "ok"}}

    with patch("core.generator.requests.post", return_value=_mock_response(200, fake_reply)) as mock_post:
        generate_draft("Постановка")

    payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args.args[1]
    system_content = payload["messages"][0]["content"]
    # Системный промпт не должен быть пустым
    assert len(system_content) > 100


def test_generate_draft_uses_correct_model():
    fake_reply = {"message": {"content": "ok"}}

    with patch("core.generator.requests.post", return_value=_mock_response(200, fake_reply)) as mock_post:
        generate_draft("Постановка")

    payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args.args[1]
    assert payload["model"] == "qwen3:4b"


def test_generate_draft_wraps_text_in_tags():
    fake_reply = {"message": {"content": "ok"}}

    with patch("core.generator.requests.post", return_value=_mock_response(200, fake_reply)) as mock_post:
        generate_draft("Моя постановка")

    payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args.args[1]
    user_content = payload["messages"][1]["content"]
    assert "<ПОСТАНОВКА>" in user_content
    assert "Моя постановка" in user_content
    assert "</ПОСТАНОВКА>" in user_content


# ── Ошибки подключения ────────────────────────────────────────────────────────

def test_ollama_connection_error():
    with patch("core.generator.requests.post", side_effect=req_lib.exceptions.ConnectionError):
        with pytest.raises(OllamaConnectionError, match="ollama serve"):
            generate_draft("текст")


def test_model_not_found_error():
    with patch("core.generator.requests.post", return_value=_mock_response(404, {})):
        with pytest.raises(ModelNotFoundError, match="ollama pull"):
            generate_draft("текст")
