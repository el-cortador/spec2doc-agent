import uuid
import queue
import threading
import requests
from pathlib import Path
from flask import Flask, request, jsonify, render_template

from core.parser import extract_text, ParserError
from core.generator import generate_draft, OllamaConnectionError, ModelNotFoundError

app = Flask(__name__)

UPLOAD_FOLDER = Path("uploads")
OUTPUT_FOLDER = Path("output")
ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 МБ

UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

# Хранилище состояния задач в памяти: job_id → dict
jobs: dict[str, dict] = {}

# Очередь задач для фонового воркера
job_queue: queue.Queue = queue.Queue()
worker_thread: threading.Thread | None = None
worker_lock = threading.Lock()


# ── Фоновый воркер ────────────────────────────────────────────────────────────

def _worker():
    while True:
        job_id = job_queue.get()
        if job_id is None:
            break
        _process_job(job_id)
        job_queue.task_done()


def _process_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return

    job["status"] = "processing"

    try:
        text = extract_text(job["path"])
        draft = generate_draft(text)
        output_path = save_draft(job["stem"], draft)
        job["result"] = draft
        job["output_path"] = str(output_path)
        job["status"] = "done"
    except (ParserError, OllamaConnectionError, ModelNotFoundError) as e:
        job["status"] = "error"
        job["error"] = str(e)
    except Exception as e:
        job["status"] = "error"
        job["error"] = f"Неожиданная ошибка: {e}"


def _ensure_worker_running():
    global worker_thread
    with worker_lock:
        if worker_thread is None or not worker_thread.is_alive():
            worker_thread = threading.Thread(target=_worker, daemon=True)
            worker_thread.start()


# ── Маршруты ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files[]")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "Файлы не переданы"}), 400

    result_jobs = []
    result_errors = []

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

        jobs[job_id] = {
            "job_id": job_id,
            "filename": file.filename,
            "stem": Path(file.filename).stem,
            "path": str(save_path),
            "status": "queued",
            "error": None,
            "result": None,
            "output_path": None,
        }

        result_jobs.append({"job_id": job_id, "filename": file.filename})

    return jsonify({"jobs": result_jobs, "errors": result_errors})


@app.route("/process", methods=["POST"])
def process():
    data = request.get_json(silent=True) or {}
    job_ids = data.get("job_ids", [])

    if not job_ids:
        return jsonify({"error": "job_ids не переданы"}), 400

    for job_id in job_ids:
        if job_id in jobs:
            job_queue.put(job_id)

    _ensure_worker_running()
    return jsonify({"status": "started"})


@app.route("/status")
def status():
    job_id = request.args.get("job_id")
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Задача не найдена"}), 404

    return jsonify({
        "job_id": job["job_id"],
        "filename": job["filename"],
        "status": job["status"],
        "error": job["error"],
    })


@app.route("/preview")
def preview():
    job_id = request.args.get("job_id")
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Задача не найдена"}), 404
    if job["status"] != "done":
        return jsonify({"error": "not ready"}), 400

    return jsonify({
        "job_id": job["job_id"],
        "filename": job["filename"],
        "output_path": job["output_path"],
        "content": job["result"],
    })


def save_draft(stem: str, content: str) -> Path:
    """Сохраняет черновик в output/{stem}_черновик доки.md и возвращает путь."""
    output_path = OUTPUT_FOLDER / f"{stem}_черновик доки.md"
    output_path.write_text(content, encoding="utf-8")
    return output_path


@app.route("/health")
def health():
    try:
        res = requests.get("http://localhost:11434/api/tags", timeout=3)
        tags = res.json()
        model_ready = any(
            m.get("name", "").startswith("qwen3:4b")
            for m in tags.get("models", [])
        )
        return jsonify({"ollama": True, "model_ready": model_ready})
    except Exception:
        return jsonify({"ollama": False, "model_ready": False})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
