from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path

from utils.paths import ensure_dir, project_root


_console_logger = logging.getLogger("rag_demo")
if not _console_logger.handlers:
    _console_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    _console_logger.addHandler(handler)
    _console_logger.propagate = False


def info(msg: str) -> None:
    _console_logger.info(msg)


def warning(msg: str) -> None:
    _console_logger.warning(msg)


def error(msg: str) -> None:
    _console_logger.error(msg)


def _logs_dir() -> Path:
    root = project_root()
    d = root / "logs"
    ensure_dir(d)
    return d


def log_step_data(step_name: str, data_dict: dict, mode: str, run_timestamp: str) -> None:
    if mode != "test":
        return
    _logs_dir()
    path = _logs_dir() / f"debug_{run_timestamp}.jsonl"
    record = {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "step": step_name,
        "data": data_dict,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


_service_lock = threading.Lock()


def log_service_summary(query: str, top_k: int, latency_ms: float) -> None:
    _logs_dir()
    path = _logs_dir() / "service.log"
    line = (
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f'query="{query}" top_k={top_k} latency={int(latency_ms)}ms'
    )
    with _service_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

