"""
Тесты Flask-приложения (app.py).
Используется тестовый клиент Flask; Ollama и файловые операции мокируются.
"""
import io
import json
import pytest
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import app as app_module
from app import app, jobs


@pytest.fixture(autouse=True)
def clear_jobs():
    """Очищаем словарь задач перед каждым тестом."""
    jobs.clear()
    yield
    jobs.clear()


@pytest.fixture
def client(tmp_path):
    app_module.UPLOAD_FOLDER = tmp_path / "uploads"
    app_module.OUTPUT_FOLDER = tmp_path / "output"
    app_module.UPLOAD_FOLDER.mkdir()
    app_module.OUTPUT_FOLDER.mkdir()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _pdf_bytes() -> bytes:
    """Минимальный валидный PDF."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj\n"
        b"xref\n0 4\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"trailer<</Size 4/Root 1 0 R>>\n"
        b"startxref\n190\n%%EOF"
    )


def _docx_bytes() -> bytes:
    import io
    from docx import Document as DocxDocument
    doc = DocxDocument()
    doc.add_paragraph("Тестовая постановка")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ── GET / ─────────────────────────────────────────────────────────────────────

def test_index_returns_200(client):
    resp = client.get("/")
    assert resp.status_code == 200


# ── GET /health ───────────────────────────────────────────────────────────────

def test_health_ollama_ready(client):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"models": [{"name": "qwen3:4b"}]}
    with patch("app.requests.get", return_value=mock_resp):
        resp = client.get("/health")
    data = resp.get_json()
    assert data["ollama"] is True
    assert data["model_ready"] is True


def test_health_ollama_running_model_missing(client):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"models": [{"name": "llama3:8b"}]}
    with patch("app.requests.get", return_value=mock_resp):
        resp = client.get("/health")
    data = resp.get_json()
    assert data["ollama"] is True
    assert data["model_ready"] is False


def test_health_ollama_unavailable(client):
    import requests as req_lib
    with patch("app.requests.get", side_effect=req_lib.exceptions.ConnectionError):
        resp = client.get("/health")
    data = resp.get_json()
    assert data["ollama"] is False
    assert data["model_ready"] is False


# ── POST /upload ──────────────────────────────────────────────────────────────

def test_upload_no_files_returns_400(client):
    resp = client.post("/upload", data={})
    assert resp.status_code == 400


def test_upload_valid_pdf(client):
    data = {"files[]": (io.BytesIO(_pdf_bytes()), "spec.pdf")}
    resp = client.post("/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.get_json()
    assert len(body["jobs"]) == 1
    assert body["jobs"][0]["filename"] == "spec.pdf"
    assert len(body["errors"]) == 0


def test_upload_valid_docx(client):
    data = {"files[]": (io.BytesIO(_docx_bytes()), "story.docx")}
    resp = client.post("/upload", data=data, content_type="multipart/form-data")
    body = resp.get_json()
    assert len(body["jobs"]) == 1


def test_upload_invalid_extension(client):
    data = {"files[]": (io.BytesIO(b"hello"), "notes.txt")}
    resp = client.post("/upload", data=data, content_type="multipart/form-data")
    body = resp.get_json()
    assert len(body["jobs"]) == 0
    assert len(body["errors"]) == 1
    assert "Неподдерживаемый формат" in body["errors"][0]["error"]


def test_upload_file_too_large(client):
    big = io.BytesIO(b"x" * (21 * 1024 * 1024))
    data = {"files[]": (big, "huge.pdf")}
    resp = client.post("/upload", data=data, content_type="multipart/form-data")
    body = resp.get_json()
    assert len(body["errors"]) == 1
    assert "20 МБ" in body["errors"][0]["error"]


def test_upload_mixed_valid_and_invalid(client):
    from werkzeug.datastructures import MultiDict
    data = MultiDict([
        ("files[]", (io.BytesIO(_docx_bytes()), "good.docx")),
        ("files[]", (io.BytesIO(b"x"), "bad.xlsx")),
    ])
    resp = client.post("/upload", data=data, content_type="multipart/form-data")
    body = resp.get_json()
    assert len(body["jobs"]) == 1
    assert len(body["errors"]) == 1


def test_upload_saves_file_to_disk(client):
    data = {"files[]": (io.BytesIO(_docx_bytes()), "saved.docx")}
    client.post("/upload", data=data, content_type="multipart/form-data")
    saved = list(app_module.UPLOAD_FOLDER.iterdir())
    assert len(saved) == 1
    assert saved[0].suffix == ".docx"


def test_upload_creates_job_with_queued_status(client):
    data = {"files[]": (io.BytesIO(_docx_bytes()), "task.docx")}
    resp = client.post("/upload", data=data, content_type="multipart/form-data")
    job_id = resp.get_json()["jobs"][0]["job_id"]
    assert jobs[job_id]["status"] == "queued"


# ── POST /process ─────────────────────────────────────────────────────────────

def test_process_no_job_ids_returns_400(client):
    resp = client.post("/process", json={})
    assert resp.status_code == 400


def test_process_starts_worker(client):
    # Создаём задачу вручную
    jobs["test-id"] = {
        "job_id": "test-id", "filename": "f.docx", "stem": "f",
        "path": "/fake", "status": "queued", "error": None,
        "result": None, "output_path": None,
    }
    with patch("app._process_job"):  # не запускаем реальную обработку
        resp = client.post("/process", json={"job_ids": ["test-id"]})
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "started"


# ── GET /status ───────────────────────────────────────────────────────────────

def test_status_unknown_job_returns_404(client):
    resp = client.get("/status?job_id=nonexistent")
    assert resp.status_code == 404


def test_status_returns_correct_fields(client):
    jobs["abc"] = {
        "job_id": "abc", "filename": "f.pdf", "stem": "f",
        "path": "/p", "status": "processing", "error": None,
        "result": None, "output_path": None,
    }
    resp = client.get("/status?job_id=abc")
    data = resp.get_json()
    assert data["status"] == "processing"
    assert data["filename"] == "f.pdf"


# ── GET /preview ──────────────────────────────────────────────────────────────

def test_preview_not_ready_returns_400(client):
    jobs["xyz"] = {
        "job_id": "xyz", "filename": "f.pdf", "stem": "f",
        "path": "/p", "status": "processing", "error": None,
        "result": None, "output_path": None,
    }
    resp = client.get("/preview?job_id=xyz")
    assert resp.status_code == 400


def test_preview_done_returns_content(client):
    jobs["done-id"] = {
        "job_id": "done-id", "filename": "spec.pdf", "stem": "spec",
        "path": "/p", "status": "done", "error": None,
        "result": "## Черновик\nТекст",
        "output_path": "output/spec_черновик доки.md",
    }
    resp = client.get("/preview?job_id=done-id")
    data = resp.get_json()
    assert data["content"] == "## Черновик\nТекст"
    assert "output_path" in data


def test_preview_unknown_job_returns_404(client):
    resp = client.get("/preview?job_id=ghost")
    assert resp.status_code == 404


# ── save_draft ────────────────────────────────────────────────────────────────

def test_save_draft_creates_file(client, tmp_path):
    app_module.OUTPUT_FOLDER = tmp_path
    path = app_module.save_draft("feature_auth", "## Черновик\nТекст")
    assert path.exists()
    assert path.name == "feature_auth_черновик доки.md"
    assert "## Черновик" in path.read_text(encoding="utf-8")


def test_save_draft_overwrites_existing(client, tmp_path):
    app_module.OUTPUT_FOLDER = tmp_path
    app_module.save_draft("doc", "Старый текст")
    app_module.save_draft("doc", "Новый текст")
    result = (tmp_path / "doc_черновик доки.md").read_text(encoding="utf-8")
    assert result == "Новый текст"
