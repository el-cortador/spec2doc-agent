import sys
import time
import logging
import subprocess

import requests


OLLAMA_URL = "http://localhost:11434/api/tags"
OLLAMA_STARTUP_TIMEOUT = 30  # секунд
STATUS_LOG_INTERVAL = 180    # секунд между записями /status в лог


class _StatusRateLimit(logging.Filter):
    """Пропускает лог-записи GET /status не чаще одного раза в STATUS_LOG_INTERVAL секунд."""

    def __init__(self):
        super().__init__()
        self._last = 0.0

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "GET /status" not in msg:
            return True
        now = time.monotonic()
        if now - self._last < STATUS_LOG_INTERVAL:
            return False
        self._last = now
        return True


def _ollama_running() -> bool:
    try:
        requests.get(OLLAMA_URL, timeout=2)
        return True
    except Exception:
        return False


def _wait_for_ollama() -> bool:
    for _ in range(OLLAMA_STARTUP_TIMEOUT):
        if _ollama_running():
            return True
        time.sleep(1)
    return False


def main():
    ollama_proc = None

    if _ollama_running():
        print("[run] Ollama уже запущена.")
    else:
        print("[run] Запускаем ollama serve...")
        ollama_proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not _wait_for_ollama():
            print("[run] Ошибка: Ollama не ответила за 30 секунд. Убедитесь, что Ollama установлена.")
            ollama_proc.terminate()
            sys.exit(1)
        print("[run] Ollama готова.")

    print("[run] Запускаем приложение на http://localhost:5000")
    print("[run] Для остановки нажмите Ctrl+C\n")

    logging.getLogger("werkzeug").addFilter(_StatusRateLimit())
    logging.getLogger("pdfminer").setLevel(logging.ERROR)

    try:
        from main import app
        app.run(host="127.0.0.1", port=5000, debug=False)
    except KeyboardInterrupt:
        pass
    finally:
        if ollama_proc is not None:
            print("\n[run] Останавливаем Ollama...")
            ollama_proc.terminate()
            ollama_proc.wait()


if __name__ == "__main__":
    main()
