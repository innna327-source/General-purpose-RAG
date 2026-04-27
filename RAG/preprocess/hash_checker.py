from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

from config.settings import SETTINGS
from utils.hash_utils import sha256_file
from utils.paths import ensure_dir


def _load_db() -> dict:
    path = SETTINGS.hash_db_path
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def is_processed(file_path: str) -> Tuple[bool, str]:
    p = Path(file_path)
    file_hash = sha256_file(p)
    db = _load_db()
    return file_hash in db, file_hash


def record_hash(file_path: str, file_hash: str, metadata_dict: dict) -> None:
    # ensure data/ exists
    ensure_dir(SETTINGS.root / "data")
    db_path = SETTINGS.hash_db_path
    db = _load_db()
    if file_hash in db:
        return
    db[file_hash] = metadata_dict
    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

