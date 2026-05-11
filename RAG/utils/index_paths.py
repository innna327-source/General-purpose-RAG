"""
索引文件路径管理：统一 hash 对应的 4 个索引/图谱文件路径构造。

消除 main.py、hash_checker.py 中的路径重复。
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from config.settings import SETTINGS


def index_paths(file_hash: str) -> List[Path]:
    """返回 file_hash 对应的全部索引/图谱文件路径（固定顺序）。"""
    return [
        SETTINGS.index_dir / f"{file_hash}.faiss",
        SETTINGS.index_dir / f"{file_hash}.bm25.pkl",
        SETTINGS.index_dir / f"{file_hash}.chunks.jsonl",
        SETTINGS.graph_dir / f"{file_hash}_semantic_graph.json",
    ]


def all_index_files_exist(file_hash: str) -> bool:
    """检查 4 个索引文件是否全部存在。"""
    return all(p.exists() for p in index_paths(file_hash))
