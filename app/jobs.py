from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.config import MAX_DURATION_SAMPLES, OUTPUT_FOLDER as _DEFAULT_OUTPUT
from app.parser import ParserError, extract_text
from app.generator import LLMConnectionError, ModelNotFoundError, generate_draft

logger = logging.getLogger(__name__)

# Позволяет тестам подменить папку вывода на уровне модуля
OUTPUT_FOLDER: Path = _DEFAULT_OUTPUT


@dataclass
class Job:
    job_id: str
    filename: str
    stem: str
    path: str
    status: str = "queued"
    error: str | None = None
    result: str | None = None
    output_path: str | None = None
    started_at: float | None = None


# ── Хранилище ─────────────────────────────────────────────────────────────────

jobs: dict[str, Job] = {}

_completed_durations: list[float] = []
_job_queue: queue.Queue[str] = queue.Queue()
_worker_thread: threading.Thread | None = None
_worker_lock = threading.Lock()


# ── Публичный API ─────────────────────────────────────────────────────────────

def avg_duration() -> float | None:
    if not _completed_durations:
        return None
    return sum(_completed_durations) / len(_completed_durations)


def create_job(job_id: str, filename: str, stem: str, path: str) -> Job:
    job = Job(job_id=job_id, filename=filename, stem=stem, path=path)
    jobs[job_id] = job
    return job


def enqueue(job_ids: list[str]) -> None:
    for job_id in job_ids:
        if job_id in jobs:
            _job_queue.put(job_id)
    _ensure_worker_running()


def save_draft(stem: str, content: str) -> Path:
    output_path = OUTPUT_FOLDER / f"{stem}_черновик доки.md"
    output_path.write_text(content, encoding="utf-8")
    return output_path


# ── Воркер ────────────────────────────────────────────────────────────────────

def _worker() -> None:
    while True:
        job_id = _job_queue.get()
        if job_id is None:
            break
        _process_job(job_id)
        _job_queue.task_done()


def _process_job(job_id: str) -> None:
    job = jobs.get(job_id)
    if not job:
        return

    job.started_at = time.time()
    job.status = "processing"
    logger.info("[jobs] processing job_id=%s filename=%s", job_id, job.filename)

    try:
        text = extract_text(job.path)
        draft = generate_draft(text)
        out = save_draft(job.stem, draft)
        job.result = draft
        job.output_path = str(out)
        job.status = "done"

        duration = time.time() - job.started_at
        _completed_durations.append(duration)
        if len(_completed_durations) > MAX_DURATION_SAMPLES:
            _completed_durations.pop(0)

        logger.info("[jobs] done job_id=%s duration=%.1fs", job_id, duration)

    except (ParserError, LLMConnectionError, ModelNotFoundError) as e:
        job.status = "error"
        job.error = str(e)
        logger.warning("[jobs] error job_id=%s error=%s", job_id, e)
    except Exception as e:
        job.status = "error"
        job.error = f"Неожиданная ошибка: {e}"
        logger.exception("[jobs] unexpected job_id=%s", job_id)


def _ensure_worker_running() -> None:
    global _worker_thread
    with _worker_lock:
        if _worker_thread is None or not _worker_thread.is_alive():
            _worker_thread = threading.Thread(target=_worker, daemon=True)
            _worker_thread.start()
