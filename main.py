from __future__ import annotations

import logging
import uuid
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template, request

from app.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE, MODEL, OLLAMA_HEALTH_URL
from app.config import UPLOAD_FOLDER as _DEFAULT_UPLOAD
from app.config import OUTPUT_FOLDER as _DEFAULT_OUTPUT
from app.jobs import Job, avg_duration, create_job, enqueue, jobs, save_draft
import app.jobs as _jobs_module

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Позволяет тестам подменить папки на уровне модуля
UPLOAD_FOLDER: Path = _DEFAULT_UPLOAD
OUTPUT_FOLDER: Path = _DEFAULT_OUTPUT

UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)


# ── Маршруты ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files[]")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "Файлы не переданы"}), 400

    result_jobs: list[dict] = []
    result_errors: list[dict] = []

    for file in files:
        if not file.filename:
            continue

        ext = Path(file.filename).suffix.lower()

        if ext not in ALLOWED_EXTENSIONS:
            result_errors.append({
                "filename": file.filename,
                "error": f"Неподдерживаемый формат «{ext}». Допустимы: .pdf, .docx",
            })
            continue

        content = file.read()

        if len(content) > MAX_FILE_SIZE:
            result_errors.append({
                "filename": file.filename,
                "error": "Файл превышает допустимый размер 20 МБ",
            })
            continue

        job_id = str(uuid.uuid4())
        save_path = UPLOAD_FOLDER / f"{job_id}{ext}"
        save_path.write_bytes(content)

        create_job(
            job_id=job_id,
            filename=file.filename,
            stem=Path(file.filename).stem,
            path=str(save_path),
        )
        result_jobs.append({"job_id": job_id, "filename": file.filename})
        logger.info("[routes] upload filename=%s job_id=%s", file.filename, job_id)

    return jsonify({"jobs": result_jobs, "errors": result_errors})


@app.route("/process", methods=["POST"])
def process():
    data = request.get_json(silent=True) or {}
    job_ids: list[str] = data.get("job_ids", [])

    if not job_ids:
        return jsonify({"error": "job_ids не переданы"}), 400

    enqueue(job_ids)
    return jsonify({"status": "started"})


@app.route("/status")
def status():
    job_id = request.args.get("job_id")
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Задача не найдена"}), 404

    return jsonify({
        "job_id":       job.job_id,
        "filename":     job.filename,
        "status":       job.status,
        "error":        job.error,
        "started_at":   job.started_at,
        "avg_duration": avg_duration(),
    })


@app.route("/preview")
def preview():
    job_id = request.args.get("job_id")
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Задача не найдена"}), 404
    if job.status != "done":
        return jsonify({"error": "not ready"}), 400

    return jsonify({
        "job_id":      job.job_id,
        "filename":    job.filename,
        "output_path": job.output_path,
        "content":     job.result,
    })


@app.route("/health")
def health():
    try:
        res = requests.get(OLLAMA_HEALTH_URL, timeout=3)
        tags = res.json()
        model_ready = any(
            m.get("name", "").startswith(MODEL)
            for m in tags.get("models", [])
        )
        return jsonify({"ollama": True, "model_ready": model_ready})
    except Exception:
        return jsonify({"ollama": False, "model_ready": False})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
