"""
Тесты модуля core/generator.py.
Покрывают оба бэкенда: ollama и llamacpp.
Ollama/llama.cpp не поднимаются — все HTTP-вызовы мокируются.
"""
import json
import pytest
import requests as req_lib
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Вспомогательные фабрики ───────────────────────────────────────────────────

def _ollama_stream_response(chunks: list[dict]) -> MagicMock:
    """Мок Ollama: iter_lines() → JSON-строки чанков."""
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.iter_lines.return_value = [json.dumps(c).encode() for c in chunks]
    return resp


def _llamacpp_stream_response(contents: list[str]) -> MagicMock:
    """Мок llama.cpp: iter_lines() → SSE-строки data: {...} + data: [DONE]."""
    lines = []
    for content in contents:
        chunk = {"choices": [{"delta": {"content": content}}]}
        lines.append(f"data: {json.dumps(chunk)}".encode())
    lines.append(b"data: [DONE]")

    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.iter_lines.return_value = lines
    return resp


def _error_response(status_code: int) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    resp.iter_lines.return_value = []
    return resp


# ── Тесты бэкенда: Ollama ─────────────────────────────────────────────────────

class TestOllamaBackend:

    @pytest.fixture(autouse=True)
    def use_ollama(self, monkeypatch):
        monkeypatch.setenv("BACKEND", "ollama")
        # Перезагружаем модуль чтобы подхватить новый env
        import importlib
        import core.generator as gen
        importlib.reload(gen)
        self.gen = gen

    def test_returns_content(self):
        chunks = [
            {"message": {"content": "## Черновик\nТекст"}, "done": False},
            {"message": {"content": ""}, "done": True},
        ]
        with patch.object(self.gen.requests, "post", return_value=_ollama_stream_response(chunks)):
            result = self.gen.generate_draft("постановка")
        assert result == "## Черновик\nТекст"

    def test_joins_multiple_chunks(self):
        chunks = [
            {"message": {"content": "часть1 "}, "done": False},
            {"message": {"content": "часть2"}, "done": False},
            {"message": {"content": ""}, "done": True},
        ]
        with patch.object(self.gen.requests, "post", return_value=_ollama_stream_response(chunks)):
            result = self.gen.generate_draft("постановка")
        assert result == "часть1 часть2"

    def test_sends_no_think_directive(self):
        chunks = [{"message": {"content": "ok"}, "done": True}]
        with patch.object(self.gen.requests, "post", return_value=_ollama_stream_response(chunks)) as mock_post:
            self.gen.generate_draft("постановка")
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args.args[1]
        assert payload["messages"][1]["content"].startswith("/no_think")

    def test_passes_system_prompt(self):
        chunks = [{"message": {"content": "ok"}, "done": True}]
        with patch.object(self.gen.requests, "post", return_value=_ollama_stream_response(chunks)) as mock_post:
            self.gen.generate_draft("постановка")
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args.args[1]
        assert len(payload["messages"][0]["content"]) > 100

    def test_uses_streaming(self):
        chunks = [{"message": {"content": "ok"}, "done": True}]
        with patch.object(self.gen.requests, "post", return_value=_ollama_stream_response(chunks)) as mock_post:
            self.gen.generate_draft("постановка")
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args.args[1]
        assert payload["stream"] is True

    def test_wraps_text_in_tags(self):
        chunks = [{"message": {"content": "ok"}, "done": True}]
        with patch.object(self.gen.requests, "post", return_value=_ollama_stream_response(chunks)) as mock_post:
            self.gen.generate_draft("моя постановка")
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args.args[1]
        user = payload["messages"][1]["content"]
        assert "<ПОСТАНОВКА>" in user and "моя постановка" in user and "</ПОСТАНОВКА>" in user

    def test_uses_correct_model(self):
        chunks = [{"message": {"content": "ok"}, "done": True}]
        with patch.object(self.gen.requests, "post", return_value=_ollama_stream_response(chunks)) as mock_post:
            self.gen.generate_draft("постановка")
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args.args[1]
        assert payload["model"] == "qwen3.5:0.8b"

    def test_connection_error(self):
        with patch.object(self.gen.requests, "post", side_effect=req_lib.exceptions.ConnectionError):
            with pytest.raises(self.gen.LLMConnectionError, match="ollama serve"):
                self.gen.generate_draft("текст")

    def test_model_not_found(self):
        with patch.object(self.gen.requests, "post", return_value=_error_response(404)):
            with pytest.raises(self.gen.ModelNotFoundError, match="ollama pull"):
                self.gen.generate_draft("текст")


# ── Тесты бэкенда: llama.cpp ─────────────────────────────────────────────────

class TestLlamaCppBackend:

    @pytest.fixture(autouse=True)
    def use_llamacpp(self, monkeypatch):
        monkeypatch.setenv("BACKEND", "llamacpp")
        import importlib
        import core.generator as gen
        importlib.reload(gen)
        self.gen = gen

    def test_returns_content(self):
        with patch.object(self.gen.requests, "post", return_value=_llamacpp_stream_response(["## Черновик\nТекст"])):
            result = self.gen.generate_draft("постановка")
        assert result == "## Черновик\nТекст"

    def test_joins_multiple_chunks(self):
        with patch.object(self.gen.requests, "post", return_value=_llamacpp_stream_response(["часть1 ", "часть2"])):
            result = self.gen.generate_draft("постановка")
        assert result == "часть1 часть2"

    def test_sends_no_think_directive(self):
        with patch.object(self.gen.requests, "post", return_value=_llamacpp_stream_response(["ok"])) as mock_post:
            self.gen.generate_draft("постановка")
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args.args[1]
        assert payload["messages"][1]["content"].startswith("/no_think")

    def test_uses_streaming(self):
        with patch.object(self.gen.requests, "post", return_value=_llamacpp_stream_response(["ok"])) as mock_post:
            self.gen.generate_draft("постановка")
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args.args[1]
        assert payload["stream"] is True

    def test_uses_max_tokens_not_num_predict(self):
        with patch.object(self.gen.requests, "post", return_value=_llamacpp_stream_response(["ok"])) as mock_post:
            self.gen.generate_draft("постановка")
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args.args[1]
        assert "max_tokens" in payload
        assert "options" not in payload

    def test_parses_sse_done_sentinel(self):
        """Генерация останавливается на data: [DONE], не падает."""
        with patch.object(self.gen.requests, "post", return_value=_llamacpp_stream_response(["текст"])):
            result = self.gen.generate_draft("постановка")
        assert result == "текст"

    def test_connection_error(self):
        with patch.object(self.gen.requests, "post", side_effect=req_lib.exceptions.ConnectionError):
            with pytest.raises(self.gen.LLMConnectionError, match="llama-server"):
                self.gen.generate_draft("текст")

    def test_model_not_found(self):
        with patch.object(self.gen.requests, "post", return_value=_error_response(404)):
            with pytest.raises(self.gen.ModelNotFoundError, match="llama-server"):
                self.gen.generate_draft("текст")
