import uuid
import requests
from pathlib import Path
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

UPLOAD_FOLDER = Path("uploads")
OUTPUT_FOLDER = Path("output")
ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 МБ

UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

# Хранилище состояния задач в памяти: job_id → dict
jobs: dict[str, dict] = {}


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
        }

        result_jobs.append({"job_id": job_id, "filename": file.filename})

    return jsonify({"jobs": result_jobs, "errors": result_errors})


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
