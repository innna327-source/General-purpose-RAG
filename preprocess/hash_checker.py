from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

from config.settings import SETTINGS
from utils.hash_utils import sha256_file
from utils.index_paths import index_paths
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


def record_version(file_name: str, file_hash: str, metadata_dict: dict) -> None:
    """将本次构建的版本信息追加到 version_log.json，同一 hash 已存在则跳过（幂等）。"""
    ensure_dir(SETTINGS.root / "data")
    log_path = SETTINGS.version_log_path
    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                log = json.load(f) or {}
        except Exception:
            log = {}
    else:
        log = {}

    versions = log.get(file_name, [])
    if any(v["hash"] == file_hash for v in versions):
        return

    entry = {
        "hash": file_hash,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **metadata_dict,
    }
    versions.append(entry)
    log[file_name] = versions
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def get_version_history(file_name: str) -> list[dict]:
    """返回指定文件的版本历史列表（时间升序），文件不存在或无记录时返回 []。"""
    log_path = SETTINGS.version_log_path
    if not log_path.exists():
        return []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            log = json.load(f) or {}
        return log.get(file_name, [])
    except Exception:
        return []


def purge_old_versions(file_name: str) -> None:
    """删除超过保留天数的旧版本索引文件，始终保留最新版本（无论多旧）。"""
    log_path = SETTINGS.version_log_path
    if not log_path.exists():
        return
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            log = json.load(f) or {}
    except Exception:
        return

    versions = log.get(file_name, [])
    if len(versions) <= 1:
        return

    cutoff = datetime.now() - timedelta(days=SETTINGS.version_retention_days)
    latest = versions[-1]
    to_delete = []
    to_keep = []

    for v in versions[:-1]:
        ts = datetime.strptime(v["timestamp"], "%Y-%m-%d %H:%M:%S")
        if ts < cutoff:
            to_delete.append(v)
        else:
            to_keep.append(v)

    if not to_delete:
        return

    db_path = SETTINGS.hash_db_path
    for v in to_delete:
        h = v["hash"]
        for p in index_paths(h):
            if p.exists():
                p.unlink()
        if db_path.exists():
            try:
                with open(db_path, "r", encoding="utf-8") as f:
                    db = json.load(f) or {}
                if h in db:
                    del db[h]
                    with open(db_path, "w", encoding="utf-8") as f:
                        json.dump(db, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

    log[file_name] = to_keep + [latest]
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

