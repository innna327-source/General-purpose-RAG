"""
测试查询加载工具：统一读取 tests/test_queries.json，处理 str / dict 兼容格式。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List


def load_test_queries(queries_path: Path | None = None) -> List[dict]:
    """加载测试查询，自动把纯字符串列表转为 [{"query": ...}] 格式。"""
    if queries_path is None:
        from config.settings import SETTINGS
        queries_path = SETTINGS.root / "tests" / "test_queries.json"

    if not queries_path.exists():
        return []

    with open(queries_path, "r", encoding="utf-8") as f:
        data = json.load(f) or []

    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
        return [{"query": q} for q in data]
    return data
