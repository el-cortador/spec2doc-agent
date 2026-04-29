from __future__ import annotations

from pathlib import Path

UPLOAD_FOLDER = Path("uploads")
OUTPUT_FOLDER = Path("output")
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx"})
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 МБ
MAX_DURATION_SAMPLES = 5

MODEL = "qwen3:4b"
OLLAMA_HEALTH_URL = "http://localhost:11434/api/tags"

SYSTEM_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "system_prompt.md"
