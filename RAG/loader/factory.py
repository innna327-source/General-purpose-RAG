"""
Loader 工厂：根据文件扩展名返回对应的 Loader 实例。

新增文件类型只需在此处注册，调用方不感知具体实现。
"""

from __future__ import annotations

from pathlib import Path

from loader.base_loader import BaseLoader
from loader.pdf_loader import PDFLoader


def get_loader(file_path: Path) -> BaseLoader:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return PDFLoader()
    raise ValueError(f"不支持的文件类型：{suffix}，目前支持：.pdf")
